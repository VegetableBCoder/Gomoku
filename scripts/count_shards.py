"""统计训练/验证分片（processed/{train,val}/data*.npz）的真实样本数，并与 train.py 的 25k/分片估算对比。

用法:  uv run python scripts/count_shards.py [--batch 768]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gomoku.config import processed_dir


def count_split(root: Path, split: str, batch: int):
    files = sorted((root / split).glob("data*.npz"))
    counts = []
    for f in files:
        with np.load(f) as d:
            counts.append(int(d["board"].shape[0]))
    counts = np.array(counts)
    total = int(counts.sum())
    print(f"== {split}: {len(files)} shards, total {total:,} samples ==")
    if len(counts):
        print(f"   min {counts.min()} | max {counts.max()} | "
              f"mean {counts.mean():.1f} | median {int(np.median(counts))}")
        print(f"   <20k: {(counts < 20000).sum()} | 20~30k: "
              f"{((counts >= 20000) & (counts <= 30000)).sum()} | >30k: {(counts > 30000).sum()}")
        est = len(files) * 25000
        print(f"   batch {batch}: 真实步数 {total // batch} vs 代码估算 {est // batch} "
              f"(误差 {(total / est - 1) * 100:+.1f}%)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--data", default=str(processed_dir()))
    args = ap.parse_args()
    root = Path(args.data)
    train_total = count_split(root, "train", args.batch)
    count_split(root, "val", args.batch)
    print(f"\n每 epoch 真实训练样本: {train_total:,}")


if __name__ == "__main__":
    main()
