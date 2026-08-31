"""生成均衡的随机开局集：对称去重 + 用外部模型 value head 筛黑方胜率接近 0 的开局。

用于"温度0 + 随机开局"的棋力评估：确保开局不偏袒黑/白，对局结果才反映棋力而非开局。

输出文件每行一个开局：交替落子坐标串，如 "7,7 8,8 7,8 8,7 9,9 9,10"（黑/白轮流，各 K 手）。

用法:
    uv run python scripts/gen_openings.py \
        --out scripts/openings_1000.txt --n 1000 --k 3 --candidates 8000 \
        --ext F:\\FTP\\dataset\\models\\best.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from gomoku.board import Board
from ext_mcts import ext_features

N_BOARD = 15


# ---------- 随机开局生成 ----------
def gen_opening(rng: random.Random, k: int, region: int):
    """随机落子 k 黑 + k 白（白方先下? 不，黑先）。落在中心 region×region 区域，保证合法无立即胜负。"""
    lo, hi = (N_BOARD - region) // 2, (N_BOARD + region) // 2
    cells = [(r, c) for r in range(lo, hi) for c in range(lo, hi)]
    rng.shuffle(cells)
    b = Board(N_BOARD)
    moves = []
    for i in range(2 * k):
        color = 1 if i % 2 == 0 else 2
        placed = False
        for r, c in cells:
            if b.is_empty(r, c):
                b.place(r, c, color)
                if b.winner() == 0:
                    moves.append((r, c))
                    placed = True
                    break
                b.undo()
        if not placed:
            return None
    return b, moves


# ---------- 对称规范化去重 ----------
def transforms(r, c):
    return [(r, c), (c, N_BOARD - 1 - r), (N_BOARD - 1 - r, N_BOARD - 1 - c),
            (N_BOARD - 1 - c, r), (r, N_BOARD - 1 - c),
            (N_BOARD - 1 - c, N_BOARD - 1 - r), (N_BOARD - 1 - r, c),
            (c, r)]


def canonical_key(b: Board) -> str:
    """把棋盘状态用 8 种对称变换编码，取字典序最小者作去重键。"""
    blacks = sorted((r, c) for r, c, s in b.stones() if s == Board.BLACK)
    whites = sorted((r, c) for r, c, s in b.stones() if s == Board.WHITE)
    best = None
    for t in range(8):
        tb = sorted(transforms(r, c)[t] for r, c in blacks)
        tw = sorted(transforms(r, c)[t] for r, c in whites)
        key = ";".join(f"{r},{c}" for r, c in tb + tw)
        if best is None or key < best:
            best = key
    return best


def load_ext(best_pt: Path, model_dir: Path, device):
    spec = importlib.util.spec_from_file_location("ext_pvn", model_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    payload = torch.load(best_pt, map_location="cpu", weights_only=False)
    net = mod.PolicyValueNet(**payload["model_architecture"])
    net.load_state_dict(payload["model_state"])
    net.eval().to(device)
    return net


def value_black(net, b: Board, device) -> float:
    """轮到黑走时黑方的 value（正=黑优，负=白优）。"""
    x = ext_features(b, Board.BLACK).to(device)
    with torch.no_grad():
        _, v = net(x)
    return v.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=3, help="每方手数（共 2k 子）")
    ap.add_argument("--candidates", type=int, default=8000, help="生成多少候选再筛选")
    ap.add_argument("--region", type=int, default=9, help="落子限定在中心 region×region 区域")
    ap.add_argument("--ext", default=r"F:\FTP\dataset\models\best.pt")
    ap.add_argument("--ext-model-dir", default=r"F:\FTP\dataset\models")
    ap.add_argument("--seed", type=int, default=20240815)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(f"加载外部评估模型 {args.ext} -> {dev}", flush=True)
    ext = load_ext(Path(args.ext), Path(args.ext_model_dir), dev)

    rng = random.Random(args.seed)
    seen = {}
    made = 0
    while made < args.candidates:
        g = gen_opening(rng, args.k, args.region)
        if g is None:
            continue
        b, moves = g
        key = canonical_key(b)
        if key not in seen:
            seen[key] = moves
            made += 1
    print(f"候选去重后 {len(seen)} 个开局", flush=True)

    # 用 value 评估黑方胜率
    scored = []
    for i, moves in enumerate(seen.values()):
        b = Board(N_BOARD)
        for j, (r, c) in enumerate(moves):
            b.place(r, c, 1 if j % 2 == 0 else 2)
        v = value_black(ext, b, dev)
        scored.append((v, moves))
        if (i + 1) % 2000 == 0:
            print(f"  已评估 {i + 1}", flush=True)

    # 按 |value| 升序取最均衡的 n 个，尽量黑白各半
    scored.sort(key=lambda x: abs(x[0]))
    black_fav = [x for x in scored if x[0] > 0]
    white_fav = [x for x in scored if x[0] <= 0]
    chosen = []
    bf = wf = 0
    for v, moves in scored:
        if len(chosen) >= args.n:
            break
        if v > 0 and bf < args.n // 2:
            chosen.append((v, moves)); bf += 1
        elif v <= 0 and wf < args.n - args.n // 2:
            chosen.append((v, moves)); wf += 1
    if len(chosen) < args.n:
        # 若某方不足，补足
        for v, moves in scored:
            if len(chosen) >= args.n:
                break
            if (v, moves) not in chosen:
                chosen.append((v, moves))

    chosen.sort(key=lambda x: x[0])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for _, moves in chosen:
            f.write(" ".join(f"{r},{c}" for r, c in moves) + "\n")

    vals = [v for v, _ in chosen]
    fav_b = sum(1 for v in vals if v > 0)
    print(f"\n写入 {len(chosen)} 个开局 -> {args.out}")
    print(f"黑优 {fav_b} / 白优 {len(chosen) - fav_b}")
    print(f"黑方 value 范围 [{min(vals):+.3f}, {max(vals):+.3f}], "
          f"均值 {sum(vals) / len(vals):+.3f}, 中位 {sorted(vals)[len(vals)//2]:+.3f}")


if __name__ == "__main__":
    main()
