"""round-robin 模型对弈排名：让一个目录下的所有 checkpoint 两两对战，按胜率排出棋力最强。

用法:
    uv run python scripts/tournament.py --dir runs/probe3060_12x192 --games 20
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from gomoku.board import Board
from training.model import GomokuNet

RADIUS = 2


def shard_of(path: Path) -> int:
    return int(path.stem.split("_shard")[-1])


def _candidates(board: Board) -> list[tuple[int, int]]:
    size = board.size
    stones = [(r, c) for r in range(size) for c in range(size)
              if board.get(r, c) != Board.EMPTY]
    if not stones:
        return [(size // 2, size // 2)]
    cand = set()
    for r, c in stones:
        for dr in range(-RADIUS, RADIUS + 1):
            for dc in range(-RADIUS, RADIUS + 1):
                rr, cc = r + dr, c + dc
                if board.is_empty(rr, cc):
                    cand.add((rr, cc))
    return sorted(cand)


def choose(net: torch.nn.Module, board: Board, color: int, device: torch.device,
           temperature: float = 0.0):
    n = board.size
    x = torch.zeros(1, 2, n, n, dtype=torch.float32, device=device)
    for r, c, s in board.stones():
        x[0, 0 if s == color else 1, r, c] = 1.0
    with torch.no_grad():
        logits, _ = net(x)
    logits = logits[0].float()
    mask = torch.full((n * n,), float("-inf"), dtype=logits.dtype, device=device)
    for r, c in _candidates(board):
        mask[r * n + c] = logits[r * n + c]
    if temperature > 0:
        probs = F.softmax(mask / max(temperature, 1e-3), dim=0)
        idx = torch.multinomial(probs, 1).item()
    else:
        best = mask.argmax().item()
        mx = mask[best].item()
        tied = (mask == mx).nonzero(as_tuple=False).flatten().tolist()
        hw = (n - 1) / 2
        idx = min(tied, key=lambda i: (abs(i // n - hw) ** 2 + abs(i % n - hw) ** 2))
    return divmod(idx, n)


def play(net1, net2, size, seed, first, device, temperature: float = 0.0):
    """first: 1=net1 执黑先手, 2=net2 执黑先手。返回 0=平 1=net1胜 2=net2胜。"""
    torch.manual_seed(seed)   # 采样确定性：temperature>0 时不同局走法不同
    board = Board(size)
    turn = Board.BLACK
    while not board.is_full():
        if turn == Board.BLACK:
            cur = net1 if first == 1 else net2
        else:
            cur = net2 if first == 1 else net1
        mv = choose(cur, board, turn, device, temperature)
        if mv is None:
            break
        board.place(*mv, turn)
        w = board.winner()
        if w:
            return w, None
        turn = 3 - turn
    return 0, None


def load_model(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args") or {}
    net = GomokuNet(board=15, in_channels=2,
                    channels=int(args.get("channels", 128)),
                    blocks=int(args.get("blocks", 8)),
                    policy_out=int(args.get("policy_out", 225)), value_out=3)
    net.load_state_dict(ck["model"])
    net.eval().to(device)
    return net


def _run_batch(tasks, model_paths, size, device, temperature):
    """在单个进程内打一组对局。tasks: list[(s1,s2,first,seed)]。
    返回 {shard: [wins, draws, losses]}。多进程 worker 入口，须为模块级函数。"""
    dev = torch.device(device)
    cache: dict[int, torch.nn.Module] = {}
    stats: dict[int, list[int]] = {}
    for s1, s2, first, seed in tasks:
        m1 = cache.get(s1)
        if m1 is None:
            m1 = load_model(Path(model_paths[s1]), dev)
            cache[s1] = m1
        m2 = cache.get(s2)
        if m2 is None:
            m2 = load_model(Path(model_paths[s2]), dev)
            cache[s2] = m2
        w, _ = play(m1, m2, size, seed, first, dev, temperature)
        st1 = stats.setdefault(s1, [0, 0, 0])
        st2 = stats.setdefault(s2, [0, 0, 0])
        if w == 0:
            st1[1] += 1; st2[1] += 1
        elif (w == Board.BLACK) == (first == 1):
            st1[0] += 1; st2[2] += 1
        else:
            st1[2] += 1; st2[0] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--games", type=int, default=10, help="每对模型的局数（需为偶数，黑白各半）")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--jobs", type=int, default=1,
                    help="并行 worker 进程数（>1 用多进程并行跑对局）")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="选点采样温度（>0 按 softmax 概率选点使对局多样；0=贪心确定性）")
    args = ap.parse_args()

    paths = sorted(Path(args.dir).glob("ckpt_ep0_shard*.pt"), key=shard_of)
    if len(paths) < 2:
        raise SystemExit(f"至少需要 2 个 ckpt，找到 {len(paths)}")
    shards = [shard_of(p) for p in paths]
    model_paths = {shard_of(p): str(p) for p in paths}
    n = len(shards)
    print(f"{n} 个模型, device={args.device}, jobs={args.jobs}", flush=True)

    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = shards[i], shards[j]
            for g in range(args.games):
                first = 1 if g % 2 == 0 else 2
                tasks.append((s1, s2, first, args.seed * 10000 + len(tasks)))
    total_games = len(tasks)

    if args.jobs <= 1:
        stats = _run_batch(tasks, model_paths, args.size, args.device, args.temperature)
    else:
        # 按先手/后手分工：先手局排前、后手局排后；jobs=2 时进程0=全先手、进程1=全后手
        ordered = sorted(tasks, key=lambda t: t[2])
        chunks = [ordered[k * total_games // args.jobs:(k + 1) * total_games // args.jobs]
                  for k in range(args.jobs)]
        ctx = multiprocessing.get_context("spawn")
        print(f"并行 jobs={args.jobs}（先手局 {sum(1 for t in tasks if t[2] == 1)}"
              f" + 后手局 {sum(1 for t in tasks if t[2] == 2)}）...", flush=True)
        with ctx.Pool(args.jobs) as pool:
            results = pool.starmap(
                _run_batch,
                [(chunk, model_paths, args.size, args.device, args.temperature)
                 for chunk in chunks])
        stats = {}
        for r in results:
            for s, arr in r.items():
                st = stats.setdefault(s, [0, 0, 0])
                st[0] += arr[0]; st[1] += arr[1]; st[2] += arr[2]

    rows = []
    for s in shards:
        w, d, l = stats.get(s, [0, 0, 0])
        p = w + d + l
        wr = (w + 0.5 * d) / p if p else 0.0
        rows.append((s, w, d, l, wr))
    rows.sort(key=lambda r: -r[4])
    print(f"\n== round-robin 排名 ({args.games} 局/对, 共 {total_games} 局) ==")
    print(f"{'排名':<4}{'shard':<8}{'胜':>5}{'平':>4}{'负':>5}{'胜率':>8}")
    for rank, (s, w, d, l, wr) in enumerate(rows, 1):
        print(f"{rank:<4}{s:<8}{w:>5}{d:>4}{l:>5}{wr:>8.1%}")


if __name__ == "__main__":
    main()
