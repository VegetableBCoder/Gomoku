"""开局评测：从均衡随机开局续下，温度 0(贪心)，各模型执黑/执白 VS 外部基准模型。

每个模型对每个开局执黑打一次 + 执白打一次。用 scripts/openings_1000.txt（均衡开局）。
报告含 先手(执黑)胜率 / 后手(执白)胜率 / 综合胜率。

用法:
    uv run python scripts/eval_openings.py \
        --dir runs/probe3060_12x192 \
        --ext F:\\FTP\\dataset\\models\\best.pt \
        --openings scripts/openings_1000.txt --jobs 2 --temperature 0
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


def load_openings(path: Path) -> list[list[tuple[int, int]]]:
    """每行: 交替落子坐标串。返回 开局落子序列列表（黑1白2交替）。"""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append([tuple(map(int, tok.split(","))) for tok in line.split()])
    return out


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


def _play(opening, our, ext, size, first_our, device, temperature):
    """从开局续下。first_our: 1=我们执黑, 2=我们执白。返回 0平/1我们胜/2外部胜。"""
    b = Board(size)
    for i, (r, c) in enumerate(opening):
        b.place(r, c, 1 if i % 2 == 0 else 2)
    turn = Board.BLACK
    while not b.is_full():
        our_turn = (turn == first_our)
        if our_turn:
            mv = our_choose(our, b, turn, device, temperature)
        else:
            mv = ext_choose(ext, b, turn, device, temperature)
        b.place(*mv, turn)
        w = b.winner()
        if w:
            return (1 if w == first_our else 2), None
        turn = 3 - turn
    return 0, None


def _run_tasks(tasks, ext_path, model_dir, size, temperature, device, openings):
    """进程内处理一组 (shard, path, first)：对每个开局打 1 局（我们执 first）。
    返回 [(shard, first, w, d, l)]。"""
    dev = torch.device(device)
    ext = load_ext(Path(ext_path), Path(model_dir) / "model.py", dev)
    out = []
    for shard, path, first in tasks:
        our = load_our(Path(path), dev)
        w = d = l = 0
        for opening in openings:
            r, _ = _play(opening, our, ext, size, first, dev, temperature)
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
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--ext-model-dir", default=r"F:\FTP\dataset\models")
    ap.add_argument("--openings", required=True, help="开局文件")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    openings = load_openings(Path(args.openings))
    if not openings:
        raise SystemExit("开局文件为空")
    paths = sorted(Path(args.dir).glob("ckpt_ep0_shard*.pt"), key=shard_of)
    if not paths:
        raise SystemExit(f"目录下没有 ckpt: {args.dir}")
    models = [(shard_of(p), str(p)) for p in paths]

    # 每个模型生成 执黑(first=1) + 执白(first=2) 两组任务
    tasks = [(s, path, f) for s, path in models for f in (1, 2)]
    per = len(openings)
    print(f"{len(models)} 个模型 × (执黑{per} + 执白{per}) 局 = {len(models) * per * 2} 局, "
          f"temperature={args.temperature}, jobs={args.jobs}, device={args.device}", flush=True)

    if args.jobs <= 1:
        results = [_run_tasks(tasks, args.ext, args.ext_model_dir, args.size,
                              args.temperature, args.device, openings)]
    else:
        ordered = sorted(tasks, key=lambda t: t[2])  # 先黑后白，jobs=2 时进程0=全执黑、进程1=全执白
        n = len(ordered)
        chunks = [ordered[k * n // args.jobs:(k + 1) * n // args.jobs] for k in range(args.jobs)]
        chunks = [c for c in chunks if c]
        ctx = multiprocessing.get_context("spawn")
        print(f"并行 {len(chunks)} 个 worker（jobs={args.jobs}: 执黑/执白分工）...", flush=True)
        with ctx.Pool(len(chunks)) as pool:
            results = pool.starmap(
                _run_tasks,
                [(chunk, args.ext, args.ext_model_dir, args.size, args.temperature,
                  args.device, openings) for chunk in chunks])

    agg: dict[int, dict[int, list[int]]] = {}
    for r in results:
        for shard, first, w, d, l in r:
            e = agg.setdefault(shard, {1: [0, 0, 0], 2: [0, 0, 0]})[first]
            e[0] += w; e[1] += d; e[2] += l

    rows = []
    for s, _ in models:
        bl = agg[s][1]; wh = agg[s][2]
        bw = (bl[0] + 0.5 * bl[1]) / per
        ww = (wh[0] + 0.5 * wh[1]) / per
        tot = (bl[0] + wh[0] + 0.5 * (bl[1] + wh[1])) / (2 * per)
        rows.append((s, bw, ww, tot))
    rows.sort(key=lambda x: -x[3])

    print(f"\n== 开局评测报告（{Path(args.openings).name}, 温度 {args.temperature}, "
          f"{per} 开局 × 执黑+执白）==")
    print(f"{'排名':<4}{'shard':<7}{'执黑':>8}{'执白':>8}{'综合':>8}")
    for rank, (s, bw, ww, tot) in enumerate(rows, 1):
        print(f"{rank:<4}{s:<7}{bw:>7.1%}{ww:>8.1%}{tot:>8.1%}")


if __name__ == "__main__":
    main()
