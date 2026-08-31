"""阶段1筛选：把我们的所有 ckpt 与外部基准模型(纯策略, 无搜索)对战，按胜率排序。

支持外部 seed 文件：先手和后手共用同一批 seed（每模型 先手 N 局 + 后手 N 局，
各自按序号取 seed[g]），保证不同温度/不同运行在相同局面序列下对比。

报告含 先手胜率 / 后手胜率 / 综合胜率。

用法:
    uv run python scripts/benchmark_vs_ext.py \
        --dir runs/probe3060_12x192 \
        --ext F:\\FTP\\dataset\\models\\best.pt \
        --games 1000 --seed-file scripts/seeds_500.txt --jobs 2 --temperature 0.5
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from gomoku.board import Board
from training.model import GomokuNet
from ext_mcts import candidates, ext_features

RADIUS = 2


def shard_of(path: Path) -> int:
    return int(path.stem.split("_shard")[-1])


def load_seeds(path: Path) -> list[int]:
    return [int(line.strip()) for line in Path(path).read_text().splitlines() if line.strip()]


def load_our(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args") or {}
    net = GomokuNet(board=15, in_channels=2,
                    channels=int(args.get("channels", 128)),
                    blocks=int(args.get("blocks", 8)),
                    policy_out=int(args.get("policy_out", 225)), value_out=3)
    net.load_state_dict(ck["model"])
    net.eval().to(device)
    return net


def load_ext(best_pt: Path, model_py: Path, device):
    spec = importlib.util.spec_from_file_location("ext_pvn", model_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    payload = torch.load(best_pt, map_location="cpu", weights_only=False)
    net = mod.PolicyValueNet(**payload["model_architecture"])
    net.load_state_dict(payload["model_state"])
    net.eval().to(device)
    return net


def _sample(logits, board, device, temperature):
    n = board.size
    mask = torch.full((n * n,), float("-inf"), dtype=logits.dtype, device=device)
    for r, c in candidates(board, RADIUS, True):
        mask[r * n + c] = logits[r * n + c]
    if temperature > 0:
        p = F.softmax(mask / max(temperature, 1e-3), dim=0)
        idx = torch.multinomial(p, 1).item()
    else:
        idx = mask.argmax().item()
    return divmod(idx, n)


def our_choose(net, board, color, device, temperature):
    n = board.size
    x = torch.zeros(1, 2, n, n, dtype=torch.float32, device=device)
    for r, c, s in board.stones():
        x[0, 0 if s == color else 1, r, c] = 1.0
    with torch.no_grad():
        logits, _ = net(x)
    return _sample(logits[0].float(), board, device, temperature)


def ext_choose(net, board, color, device, temperature):
    x = ext_features(board, color).to(device)
    with torch.no_grad():
        logits, _ = net(x)
    return _sample(logits[0].float(), board, device, temperature)


def _play(our_net, ext_net, size, seed, first_our, device, temperature):
    torch.manual_seed(seed)
    b = Board(size)
    turn = Board.BLACK
    while not b.is_full():
        our_turn = (turn == Board.BLACK) == first_our
        if our_turn:
            mv = our_choose(our_net, b, turn, device, temperature)
        else:
            mv = ext_choose(ext_net, b, turn, device, temperature)
        b.place(*mv, turn)
        w = b.winner()
        if w:
            return (1 if (w == Board.BLACK) == first_our else 2), None
        turn = 3 - turn
    return 0, None


def _run_shards(tasks, ext_path, model_dir, size, temperature, device, seed_list, g_first):
    """进程内处理一组 (shard, path, first) 任务：各以固定 first 与外部打 g_first 局。
    先手/后手共用 seed_list[g]。返回 [(shard, first, w, d, l)]。"""
    dev = torch.device(device)
    ext = load_ext(Path(ext_path), Path(model_dir) / "model.py", dev)
    out = []
    for shard, path, first in tasks:
        our = load_our(Path(path), dev)
        w = d = l = 0
        for g in range(g_first):
            first_our = (first == 1)
            r, _ = _play(our, ext, size, seed_list[g], first_our, dev, temperature)
            if r == 0:
                d += 1
            elif r == 1:
                w += 1
            else:
                l += 1
        out.append((shard, first, w, d, l))
        del our
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="我们的 ckpt 目录")
    ap.add_argument("--ext", required=True, help="外部 best.pt")
    ap.add_argument("--ext-model-dir", default=r"F:\FTP\dataset\models")
    ap.add_argument("--games", type=int, default=1000, help="每模型每轮局数(需偶数, 先手/后手各半)")
    ap.add_argument("--seed-file", required=True, help="每行一个整数的 seed 文件")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_list = load_seeds(Path(args.seed_file))
    g_first = args.games // 2
    if len(seed_list) < g_first:
        raise SystemExit(f"seed 文件只有 {len(seed_list)} 个，但每方需要 {g_first} 个")

    paths = sorted(Path(args.dir).glob("ckpt_ep0_shard*.pt"), key=shard_of)
    if not paths:
        raise SystemExit(f"目录下没有 ckpt: {args.dir}")
    models = [(shard_of(p), str(p)) for p in paths]
    tasks = [(s, path, f) for s, path in models for f in (1, 2)]
    print(f"{len(models)} 个模型 × 每轮(先手{g_first} + 后手{g_first})局 = "
          f"{len(models) * args.games} 局, temperature={args.temperature}, "
          f"jobs={args.jobs}, device={args.device}", flush=True)

    if args.jobs <= 1:
        results = [_run_shards(tasks, args.ext, args.ext_model_dir, args.size,
                               args.temperature, args.device, seed_list, g_first)]
    else:
        # 按先手/后手分工：先手任务在前、后手任务在后；jobs=2 时进程0=全先手、进程1=全后手
        ordered = sorted(tasks, key=lambda t: t[2])
        n = len(ordered)
        chunks = [ordered[k * n // args.jobs:(k + 1) * n // args.jobs] for k in range(args.jobs)]
        chunks = [c for c in chunks if c]
        ctx = multiprocessing.get_context("spawn")
        print(f"并行 {len(chunks)} 个 worker（jobs={args.jobs}: 先手/后手分工）...", flush=True)
        with ctx.Pool(len(chunks)) as pool:
            results = pool.starmap(
                _run_shards,
                [(chunk, args.ext, args.ext_model_dir, args.size, args.temperature,
                  args.device, seed_list, g_first) for chunk in chunks])

    # 汇总：先手/后手分开，再算综合
    agg: dict[int, dict[int, list[int]]] = {}
    for r in results:
        for shard, first, w, d, l in r:
            e = agg.setdefault(shard, {1: [0, 0, 0], 2: [0, 0, 0]})[first]
            e[0] += w; e[1] += d; e[2] += l

    rows = []
    for s, _ in models:
        f = agg[s][1]; b = agg[s][2]
        fw = (f[0] + 0.5 * f[1]) / g_first
        bw = (b[0] + 0.5 * b[1]) / g_first
        tot = (f[0] + b[0] + 0.5 * (f[1] + b[1])) / args.games
        rows.append((s, fw, bw, tot))
    rows.sort(key=lambda x: -x[3])

    print(f"\n== 对基准({Path(args.ext).name})温度 {args.temperature} 报告 "
          f"(每模型 先手{g_first} + 后手{g_first}) ==")
    print(f"{'排名':<4}{'shard':<7}{'先手':>8}{'后手':>8}{'综合':>8}")
    for rank, (s, fw, bw, tot) in enumerate(rows, 1):
        print(f"{rank:<4}{s:<7}{fw:>7.1%}{bw:>8.1%}{tot:>8.1%}")


if __name__ == "__main__":
    main()
