"""我们模型(GomokuNet, 2通道) vs 外部 AlphaZero 五子棋模型(PolicyValueNet, 4通道) 对局。

外部模型特征规则(来自其 README):
    输入 [B, 4, H, W]:
      通道0 = 当前行动方棋子
      通道1 = 对方棋子
      通道2 = 最后一步 one-hot
      通道3 = 全盘填充 已落子数/(H×W)

用法:
    uv run python scripts/vs_external.py \
        --our runs/probe3060_12x192/ckpt_ep0_shard899.pt \
        --ext F:\\FTP\\dataset\\models\\best.pt \
        --games 20
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 scripts/ 内模块可 import

import torch
import torch.nn.functional as F

from gomoku.board import Board
from training.model import GomokuNet
from ext_mcts import eval_our, load_config, mcts_move

RADIUS = 2


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


def load_external(best_pt: Path, model_py: Path, device):
    spec = importlib.util.spec_from_file_location("ext_pvn", model_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    payload = torch.load(best_pt, map_location="cpu", weights_only=False)
    net = mod.PolicyValueNet(**payload["model_architecture"])
    net.load_state_dict(payload["model_state"])
    net.eval().to(device)
    return net


def candidates(board: Board) -> list[tuple[int, int]]:
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


def sample(logits, board, device, temperature):
    n = board.size
    mask = torch.full((n * n,), float("-inf"), dtype=logits.dtype, device=device)
    for r, c in candidates(board):
        mask[r * n + c] = logits[r * n + c]
    if temperature > 0:
        probs = F.softmax(mask / max(temperature, 1e-3), dim=0)
        idx = torch.multinomial(probs, 1).item()
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
    return sample(logits[0].float(), board, device, temperature)


def ext_choose(net, board, color, device, cfg):
    """外部模型用 MCTS 搜索落子。"""
    return mcts_move(net, board, color, device, cfg)


def play(our_net, ext_net, size, seed, first_our, device, temperature, ext_cfg,
         our_mcts=False, our_cfg=None):
    """first_our=True: 我们执黑先手。返回 0=平 1=我们胜 2=外部胜。"""
    torch.manual_seed(seed)
    board = Board(size)
    turn = Board.BLACK
    while not board.is_full():
        our_turn = (turn == Board.BLACK) == first_our
        if our_turn:
            if our_mcts:
                mv = mcts_move(our_net, board, turn, device, our_cfg, eval_our)
            else:
                mv = our_choose(our_net, board, turn, device, temperature)
        else:
            mv = ext_choose(ext_net, board, turn, device, ext_cfg)
        board.place(*mv, turn)
        w = board.winner()
        if w:
            black_is_our = first_our
            return (1 if (w == Board.BLACK) == black_is_our else 2), None
        turn = 3 - turn
    return 0, None


def _run_worker(item, ext_path, ext_model_dir, size, temperature, device, cfg,
                games_per, base_seed, our_mcts, our_cfg):
    """多进程 worker：处理一个 (our_path, first, seed_offset) 任务，跑 games_per 局。返回 (shard,first,w,d,l)。"""
    dev = torch.device(device)
    our_path, first, seed_offset = item
    our = load_model(Path(our_path), dev)
    ext = load_external(Path(ext_path), Path(ext_model_dir) / "model.py", dev)
    shard = int(Path(our_path).stem.split("_shard")[-1])
    w = d = l = 0
    for g in range(games_per):
        seed = base_seed * 10000 + shard * 10000 + (1 if first else 2) * 5000 + seed_offset + g
        r, _ = play(our, ext, size, seed, first, dev, temperature, cfg, our_mcts, our_cfg)
        if r == 0:
            d += 1
        elif r == 1:
            w += 1
        else:
            l += 1
    return (shard, first, w, d, l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", required=True, help="我们的 ckpt(.pt)，可传多个")
    ap.add_argument("--ext", required=True, help="外部 best.pt")
    ap.add_argument("--ext-model-dir", default=r"F:\FTP\dataset\models",
                    help="外部 model.py 所在目录")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--games", type=int, default=200, help="每模型局数(需偶数,黑白各半)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="双方选点采样温度(>0 采样, 0 贪心)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--jobs", type=int, default=None,
                    help="并行进程数(默认=模型数×2，即先手/后手各一进程)")
    ap.add_argument("--ext-config", default=r"F:\FTP\dataset\models\fast50_search.json",
                    help="外部 MCTS 配置 json（默认 fast50_search.json）")
    ap.add_argument("--sims", type=int, default=None, help="覆盖外部 MCTS simulations")
    ap.add_argument("--our-mcts", action="store_true", help="我们侧也用 MCTS 搜索")
    ap.add_argument("--our-sims", type=int, default=None, help="我们侧 MCTS simulations")
    ap.add_argument("--split", type=int, default=1,
                    help="每执色拆成几份(用于增加并行进程数)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg = load_config(args.ext_config)
    if args.sims is not None:
        cfg["simulations"] = args.sims
    our_cfg = dict(cfg)
    if args.our_sims is not None:
        our_cfg["simulations"] = args.our_sims
    print(f"我们的模型 {len(args.ours)} 个, 外部 {args.ext} -> {dev}", flush=True)
    print(f"我们侧: {'MCTS sims=' + str(our_cfg['simulations']) if args.our_mcts else '纯策略'}"
          f"   外部: MCTS sims={cfg['simulations']}, c_puct={cfg['c_puct']}, "
          f"candidate_width={cfg['candidate_width']}, radius={cfg['candidate_radius']}", flush=True)

    games_first = args.games // 2
    if games_first % args.split != 0:
        raise SystemExit(f"games/2={games_first} 需能被 --split {args.split} 整除")
    gps = games_first // args.split  # 每份局数
    tasks = [(p, f, s * gps) for p in args.ours for f in (True, False)
             for s in range(args.split)]  # (模型, 执色, seed_offset)
    jobs = args.jobs or len(tasks)
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    print(f"{len(tasks)} 个任务(模型×执黑/执白×split={args.split}), jobs={jobs}, "
          f"每任务 {gps} 局...", flush=True)
    with ctx.Pool(jobs) as pool:
        results = pool.starmap(
            _run_worker,
            [(item, args.ext, args.ext_model_dir, args.size, args.temperature,
              args.device, cfg, gps, args.seed, args.our_mcts, our_cfg)
             for item in tasks])
    dt = time.time() - t0

    agg: dict[int, dict[bool, list[int]]] = {}
    for shard, first, w, d, l in results:
        e = agg.setdefault(shard, {True: [0, 0, 0], False: [0, 0, 0]})[first]
        e[0] += w; e[1] += d; e[2] += l

    rows = []
    for shard, e in agg.items():
        b = e[True]; w = e[False]
        bw = (b[0] + 0.5 * b[1]) / games_first
        ww = (w[0] + 0.5 * w[1]) / games_first
        tot = (b[0] + w[0] + 0.5 * (b[1] + w[1])) / args.games
        rows.append((shard, bw, ww, tot))
    rows.sort(key=lambda x: -x[3])

    print(f"\n== 我们({'MCTS sims=' + str(our_cfg['simulations']) if args.our_mcts else '纯策略'}) "
          f"vs 外部 MCTS(sims={cfg['simulations']}) "
          f"({len(agg)} 模型 × 先手+后手各 {games_first} 局, temp={args.temperature}) ==")
    print(f"{'shard':<7}{'执黑':>8}{'执白':>8}{'综合':>8}")
    for shard, bw, ww, tot in rows:
        print(f"{shard:<7}{bw:>7.1%}{ww:>8.1%}{tot:>8.1%}")
    total_games = len(results) * gps
    print(f"总耗时 {dt:.1f}s = {dt / 60:.1f} 分钟, jobs={jobs}, "
          f"约 {dt / total_games:.2f}s/局 (全局吞吐)", flush=True)


if __name__ == "__main__":
    main()
