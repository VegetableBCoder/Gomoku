#!/usr/bin/env python3
"""下载 ModelScope 数据集 sigmoid/katago-gomoku-distill-2025.5 的 fs15x 子集。

用法:
    python3 download_ms_dataset.py
    python3 download_ms_dataset.py --target /home/kita-ikuyo/dataset --subset fs15x_label28b
    python3 download_ms_dataset.py --no-proxy        # 国内直连，不走代理

说明:
    - 只下载 fs15x_label28b (无禁手 15x15, ~11.9GB, 5432 个文件)
    - 已存在且大小一致的文件自动跳过，可断点续传
    - 默认使用环境变量里的代理 (http_proxy/https_proxy)；国内访问 ModelScope 通常直连更快，
      可加 --no-proxy
"""
import argparse
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://modelscope.cn/api/v1/datasets/sigmoid/katago-gomoku-distill-2025.5"


def make_opener(no_proxy: bool):
    if no_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def list_files(subset: str):
    files = []
    page = 1
    while True:
        url = f"{BASE}/repo/tree?Revision=master&Recursive=true&PageSize=200&PageNumber={page}"
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        batch = data["Data"]["Files"]
        if not batch:
            break
        for f in batch:
            if f.get("Type") == "blob" and f.get("Path", "").startswith(subset + "/"):
                files.append((f["Path"], int(f.get("Size") or 0)))
        if len(batch) < 200:
            break
        page += 1
    return files


def download_one(task, target_dir: Path, opener):
    path, size = task
    dst = target_dir / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == size:
        return path, "skip"
    tmp = dst.with_suffix(dst.suffix + ".part")
    url = f"{BASE}/repo?Revision=master&FilePath=" + urllib.parse.quote(path, safe="")
    for attempt in range(3):
        try:
            with opener.open(url, timeout=120) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            if tmp.stat().st_size != size:
                raise RuntimeError(f"size mismatch: {tmp.stat().st_size} != {size}")
            tmp.replace(dst)
            return path, "ok"
        except Exception as e:
            if attempt == 2:
                return path, f"ERR: {e}"
            time.sleep(2 * (attempt + 1))
    return path, "ERR"


def main():
    ap = argparse.ArgumentParser(description="下载 katago-gomoku-distill fs15x 子集")
    ap.add_argument("--target", default="/home/kita-ikuyo/dataset")
    ap.add_argument("--subset", default="fs15x_label28b")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-proxy", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] 枚举 {args.subset} 文件列表 ...")
    files = list_files(args.subset)
    total = sum(s for _, s in files)
    print(f"      共 {len(files)} 个文件, {total/1e9:.1f} GB")

    opener = make_opener(args.no_proxy)
    print(f"[2/2] 并发下载 (workers={args.workers}, 已存在则跳过) ...")
    done = errs = ok = skipped = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, t, target, opener) for t in files]
        for fut in as_completed(futs):
            path, status = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                errs += 1
                print(f"  ! {status}  {path}", flush=True)
            if done % 200 == 0 or done == len(files):
                el = time.time() - t0
                print(f"     {done}/{len(files)}  ok={ok} skip={skipped} err={errs}  "
                      f"{el:.0f}s", flush=True)

    print(f"完成: ok={ok} skip={skipped} err={errs}  总耗时 {time.time()-t0:.0f}s")
    print(f"数据目录: {target / args.subset}")
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
