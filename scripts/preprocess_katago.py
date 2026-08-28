#!/usr/bin/env python3
"""把原始 KataGomo npz 分片裁剪成训练用的精简格式。

输入 (raw):  /home/kita-ikuyo/dataset/fs15x_label28b/{train,val}/data*.npz
输出 (out):  /home/kita-ikuyo/dataset/processed/{train,val}/data*.npz

每个输出 npz 只保留 3 个数组:
    board  (N, 2, 15, 15) uint8   己方/对方棋子 (已 unpack，省去训练时 unpackbits)
    policy (N, 225)      int16    策略原始计数 (未归一化，pass 已裁掉)
    value  (N, 3)        float16  胜率/负率/和棋率

用法:
    uv pip install numpy                # 装到 .venv
    .venv/bin/python preprocess_katago.py --workers 8
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def process_shard(task):
    src, dst = task
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return src.name, "skip"
    try:
        data = np.load(src)
        n = data["binaryInputNCHWPacked"].shape[0]
        # 每通道 29 字节=232 位, 棋盘只占前 225 位, 必须裁掉 7 位 padding 再 reshape
        board = np.unpackbits(data["binaryInputNCHWPacked"], axis=2)[:, 1:3, :225].reshape(n, 2, 15, 15)
        policy = data["policyTargetsNCMove"][:, 0, :225].astype(np.int16)
        value = data["globalTargetsNC"][:, :3].astype(np.float16)
        np.savez_compressed(dst, board=board, policy=policy, value=value)
        return src.name, "ok"
    except Exception as e:  # noqa: BLE001
        return src.name, f"ERR: {e}"


def main():
    ap = argparse.ArgumentParser(description="裁剪 KataGomo npz -> 训练格式")
    ap.add_argument("--raw", default="/home/kita-ikuyo/dataset/fs15x_label28b")
    ap.add_argument("--out", default="/home/kita-ikuyo/dataset/processed")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    srcs = sorted(raw.rglob("data*.npz"))
    if not srcs:
        print(f"没找到数据: {raw}/data*.npz")
        sys.exit(1)
    print(f"找到 {len(srcs)} 个分片, 输出到 {out}")

    tasks = []
    for s in srcs:
        rel = s.relative_to(raw)  # 例如 train/data0.npz
        tasks.append((s, out / rel))

    done = ok = skip = errs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_shard, t) for t in tasks]
        for fut in as_completed(futs):
            name, status = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                errs += 1
                print(f"  ! {status}  {name}", flush=True)
            if done % 200 == 0 or done == len(tasks):
                print(f"      {done}/{len(tasks)}  ok={ok} skip={skip} err={errs}", flush=True)

    print(f"完成: ok={ok} skip={skip} err={errs}")
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
