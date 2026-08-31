"""监督训练 五子棋 policy+value 双头模型。

数据目录默认读项目根目录 .env 的 GOBANG_PROCESSED_DIR（见 .env.example），
也可用 --data 显式覆盖。

用法(冒烟, 8x128, 5 个分片):
    uv run python -m training.train --limit-shards 5 --epochs 1 --batch-size 1024 --out runs/smoke

用法(全量):
    uv run python -m training.train --epochs 2 --batch-size 1024 --amp --out runs/full
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gomoku.config import processed_dir
from training.model import GomokuNet
from training.loader_katago import KatagoShards


def soft_ce(logits, target):
    """软标签交叉熵: -sum(t * log p)"""
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def top1(logits, target):
    return (logits.argmax(1) == target.argmax(1)).float().mean().item()


def evaluate(model, valset, device, bs, amp):
    model.eval()
    pl = pa = va = cnt = 0.0
    with torch.no_grad():
        for f in valset.files:
            board, policy, value = KatagoShards.load_shard(f)
            for i in range(0, len(board), bs):
                x = torch.from_numpy(board[i:i + bs]).to(device)
                yp = torch.from_numpy(policy[i:i + bs]).to(device)
                yv = torch.from_numpy(value[i:i + bs]).to(device)
                with torch.autocast("cuda", enabled=amp):
                    p, v = model(x)
                pl += soft_ce(p, yp).item() * len(x)
                pa += top1(p, yp) * len(x)
                va += top1(v, yv) * len(x)
                cnt += len(x)
    model.train()
    return pl / cnt, pa / cnt, va / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(processed_dir()),
                    help="训练数据根目录（默认读 .env 的 GOBANG_PROCESSED_DIR）")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--limit-shards", type=int, default=0, help="冒烟时只取前 N 个 train 分片")
    ap.add_argument("--val-shards", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=100, help="每 N 个分片评估+存一次 checkpoint")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 runs/smoke；--resume 未给时沿用 ckpt 里的 out）")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="从 checkpoint 断点续训：新格式恢复 权重+优化器+scheduler+scaler 并从断点 shard 继续；"
                         "旧格式（无 trainer 字段）仅恢复权重、优化器/scheduler 从头")
    args = ap.parse_args()

    # --resume: 覆盖结构与数据相关参数，保证网络结构 / shuffle 顺序 / lr schedule 与原始训练一致
    _resume_core = ("blocks", "channels", "epochs", "batch_size",
                    "limit_shards", "val_shards", "data", "seed", "device", "amp")
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = resume_ckpt.get("args") or {}
        for k in _resume_core:
            if k in saved:
                setattr(args, k, saved[k])
        if not args.out:
            args.out = saved.get("out", "runs/smoke")
    if not args.out:
        args.out = "runs/smoke"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    trainset = KatagoShards(args.data, "train", limit=args.limit_shards, seed=args.seed)
    valset = KatagoShards(args.data, "val", limit=args.val_shards, seed=args.seed + 1)
    if not trainset.files:
        raise SystemExit(f"train 分片为空: {args.data}/train/data*.npz")
    print(f"train 分片: {len(trainset)}   val 分片: {len(valset)}")

    model = GomokuNet(blocks=args.blocks, channels=args.channels).to(device)
    print(f"model = {args.blocks}x{args.channels}, params = {model.num_params()/1e6:.2f}M")
    print(f"device = {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # T_max 按总步数估算(每分片约 25k 样本), 让 cosine 在整段训练上正确衰减
    total_steps = max(1, args.epochs * len(trainset) * 25000 // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch, start_shard, step = 0, 0, 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        trainer = resume_ckpt.get("trainer")
        if trainer:
            opt.load_state_dict(trainer["optimizer"])
            sched.load_state_dict(trainer["scheduler"])
            if "scaler" in trainer:
                scaler.load_state_dict(trainer["scaler"])
            start_epoch = int(trainer["epoch"])
            start_shard = int(trainer["shard"]) + 1
            step = int(trainer["step"])
            print(f"恢复训练: epoch {start_epoch}, 从 shard {start_shard} 起, step {step}",
                  flush=True)
        else:
            print("警告: checkpoint 为旧格式（无优化器/scheduler 状态），仅恢复模型权重，"
                  "lr 从初始值重新开始", flush=True)
        if start_epoch >= args.epochs:
            raise SystemExit(f"checkpoint 已训练完所有 epoch ({start_epoch} >= {args.epochs})，无需续训")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    bs = args.batch_size
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        for si, f in enumerate(trainset.shuffled_files()):
            if epoch == start_epoch and si < start_shard:
                continue
            t0 = time.time()
            board, policy, value = KatagoShards.load_shard(f)
            n = len(board)
            for i in range(0, n, bs):
                x = torch.from_numpy(board[i:i + bs]).to(device)
                yp = torch.from_numpy(policy[i:i + bs]).to(device)
                yv = torch.from_numpy(value[i:i + bs]).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=args.amp):
                    p, v = model(x)
                    lp = soft_ce(p, yp)
                    lv = soft_ce(v, yv)
                    loss = lp + args.value_weight * lv
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step % args.log_every == 0:
                    print(f"ep{epoch} shard{si}/{len(trainset)} step{step} "
                          f"loss={loss.item():.4f} pl={lp.item():.4f} vl={lv.item():.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} {n/(time.time()-t0):.0f}/s", flush=True)
            if (si + 1) % args.save_every == 0 or si == len(trainset) - 1:
                pl, pa, va = evaluate(model, valset, device, bs, args.amp)
                print(f"  [eval @shard{si}] val: pl={pl:.4f} policy_top1={pa:.4f} "
                      f"value_top1={va:.4f}", flush=True)
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "trainer": {
                        "epoch": epoch,
                        "shard": si,
                        "step": step,
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                }, out / f"ckpt_ep{epoch}_shard{si}.pt")

    elapsed = time.time() - t_start
    print(f"完成: {elapsed/60:.1f} min, checkpoint -> {out}")


if __name__ == "__main__":
    main()
