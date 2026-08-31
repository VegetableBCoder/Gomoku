"""标准 AlphaZero 风格 MCTS 搜索（PUCT），用于外部 PolicyValueNet 模型落子。

搜索配置来自模型仓库的 fast50_search.json（HUBUGUII/Gomoku-MultiSize-PolicyValue）:
  simulations, c_puct, candidate_width, candidate_radius, dirichlet_alpha,
  noise_fraction, first_move_center, force_win, force_block, fallback_all_legal

本模块是标准公开算法实现（AlphaZero），网络评估用外部的 PolicyValueNet。
特征规则（README）：[B,4,H,W] = 当前方棋子 / 对方棋子 / 最后一步 one-hot / 已落子数÷(H×W)。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from gomoku.board import Board

DEFAULT_CFG = {
    "simulations": 200,
    "c_puct": 1.0,
    "candidate_width": 16,
    "candidate_radius": 2,
    "dirichlet_alpha": 0.15,
    "noise_fraction": 0.25,
    "first_move_center": True,
    "force_win": True,
    "force_block": True,
    "fallback_all_legal": True,
}


def load_config(path=None) -> dict:
    cfg = dict(DEFAULT_CFG)
    if path and Path(path).exists():
        cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg


def candidates(board: Board, radius: int, fallback_all_legal: bool) -> list[tuple[int, int]]:
    size = board.size
    stones = [(r, c) for r in range(size) for c in range(size)
              if board.get(r, c) != Board.EMPTY]
    if not stones:
        return [(size // 2, size // 2)]
    cand = set()
    for r, c in stones:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if board.is_empty(rr, cc):
                    cand.add((rr, cc))
    if not cand and fallback_all_legal:
        return [(r, c) for r in range(size) for c in range(size)
                if board.is_empty(r, c)]
    return sorted(cand)


def ext_features(board: Board, color: int) -> torch.Tensor:
    n = board.size
    x = torch.zeros(1, 4, n, n, dtype=torch.float32)
    opp = 3 - color
    for r, c, s in board.stones():
        x[0, 0 if s == color else 1, r, c] = 1.0
    if board.history:
        lm = board.history[-1]
        x[0, 2, lm[0], lm[1]] = 1.0
    x[0, 3] = len(board.history) / (n * n)
    return x


def eval_ext(net, board: Board, color: int, device):
    """外部模型评估：返回 (policy_logits[0], 标量value)。"""
    x = ext_features(board, color).to(device)
    with torch.no_grad():
        logits, v = net(x)
    return logits[0].float(), v.item()


def eval_our(net, board: Board, color: int, device):
    """我们模型(GomokuNet, 2通道, value 3分类[win,loss,draw])评估。返回 (logits, 标量value)。"""
    n = board.size
    x = torch.zeros(1, 2, n, n, dtype=torch.float32, device=device)
    for r, c, s in board.stones():
        x[0, 0 if s == color else 1, r, c] = 1.0
    with torch.no_grad():
        logits, v = net(x)
    logits = logits[0].float()
    vp = F.softmax(v, dim=1)[0]           # (3,) = [win, loss, draw]
    return logits, (vp[0] - vp[1]).item()  # 当前方视角胜率 - 负率


def _priors(net, board, color, device, cand, width, eval_fn) -> dict:
    """返回 {action: prior_prob}，按网络 policy 在前 width 个候选上 softmax。"""
    n = board.size
    logits, _ = eval_fn(net, board, color, device)
    vals = torch.tensor([logits[r * n + c].item() for r, c in cand],
                        dtype=torch.float32, device=device)
    p = F.softmax(vals, dim=0)
    order = torch.argsort(p, descending=True)[:width]
    return {cand[i.item()]: p[i].item() for i in order}


def _value(net, board, color, device, eval_fn) -> float:
    _, value = eval_fn(net, board, color, device)
    return value


def find_win(board: Board, color: int, radius: int) -> tuple[int, int] | None:
    """返回使 color 立即连五的空位，无则 None。"""
    for r, c in candidates(board, radius, True):
        if board.place(r, c, color):
            won = board.winner() == color
            board.undo()
            if won:
                return (r, c)
    return None


class _Node:
    __slots__ = ("prior", "visit", "w", "children")

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visit = 0
        self.w = 0.0
        self.children = {}


def _select(node: _Node, c_puct: float):
    best = None
    bestv = -1e18
    for a, ch in node.children.items():
        # 节点存"自身(该节点玩家)视角"的 W；对父节点来说走 a 的价值 = -子Q
        q = -(ch.w / ch.visit) if ch.visit else 0.0
        u = c_puct * ch.prior * math.sqrt(node.visit) / (1 + ch.visit)
        v = q + u
        if v > bestv:
            bestv, best = v, (a, ch)
    return best


def _simulate(root, net, board, color, device, cfg, eval_fn):
    node, b, t = root, board.clone(), color
    path = [node]
    while node.children:
        a, child = _select(node, cfg["c_puct"])
        b.place(a[0], a[1], t)
        t = 3 - t
        node = child
        path.append(node)
        if b.winner():
            break
    if b.winner():
        v = -1.0  # node 是最后落子后的子节点，视角玩家为落子方的对手(败方)
    elif b.is_full():
        v = 0.0
    else:
        cand = candidates(b, cfg["candidate_radius"], cfg["fallback_all_legal"])
        priors = _priors(net, b, t, device, cand, cfg["candidate_width"], eval_fn)
        for a, p in priors.items():
            node.children[a] = _Node(p)
        v = _value(net, b, t, device, eval_fn)
    for nd in reversed(path):       # 每个节点存"该节点玩家视角"的累积值
        nd.visit += 1
        nd.w += v
        v = -v


def mcts_move(net, board: Board, color: int, device, cfg: dict,
              eval_fn=eval_ext) -> tuple[int, int]:
    """用 MCTS 搜索返回落子坐标。eval_fn: (net, board, color, device) -> (logits, scalar_value)。"""
    size = board.size
    radius = int(cfg["candidate_radius"])

    if cfg.get("first_move_center") and len(board.history) == 0:
        return (size // 2, size // 2)
    if cfg.get("force_win"):
        m = find_win(board, color, radius)
        if m:
            return m
    if cfg.get("force_block"):
        m = find_win(board, 3 - color, radius)
        if m:
            return m

    root = _Node()
    cand0 = candidates(board, radius, cfg["fallback_all_legal"])
    priors = _priors(net, board, color, device, cand0, cfg["candidate_width"], eval_fn)

    # root Dirichlet 噪声
    alpha = float(cfg["dirichlet_alpha"])
    frac = float(cfg["noise_fraction"])
    if alpha > 0 and frac > 0 and priors:
        keys = list(priors.keys())
        alphas = torch.full((len(keys),), alpha)
        noise = torch.distributions.Dirichlet(alphas).sample()
        for i, k in enumerate(keys):
            priors[k] = priors[k] * (1 - frac) + frac * noise[i].item()
    for a, p in priors.items():
        root.children[a] = _Node(p)

    for _ in range(int(cfg["simulations"])):
        _simulate(root, net, board, color, device, cfg, eval_fn)

    return max(root.children.items(), key=lambda kv: kv[1].visit)[0]
