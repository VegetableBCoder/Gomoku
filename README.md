# 瓜皮五子棋 2.0

## 分支lagacy

- 2017年刚学PYTHON的时候写的垃圾实现
- 别问为啥是19*19的围棋棋盘 问就是不会围棋 也不会五子棋

## 分支 main

纯逻辑引擎 + 贪心/神经网络棋手 + KataGo 蒸馏数据监督训练（自对弈 RL 规划中）。

- 规则：自由规则（无禁手），长连（≥5）即胜；黑先手优势显著
- 网络：8×128 残差 CNN 双头（policy+value，约 2.38M 参数）
- 数据：ModelScope katago-gomoku-distill 2025.5 fs15x（约 6300 万局面，73M 老师蒸馏软标签）

## 环境与安装

要求：uv ≥ 0.12、Python ≥ 3.10（推荐 3.14）。

```bash
# 1. 配置镜像源（可选，国内建议国内镜像源）

# 2. 安装依赖（核心 + GUI）
uv sync

# 3. GUI 需要系统 tkinter（Debian/Ubuntu需要 windows似乎不需要）
sudo apt install -y python3-tk
```

## 环境配置（.env）

项目内本地绝对路径都通过项目根目录的 `.env` 。
复制模板并按需修改：

```bash
cp .env.example .env
# 默认值即以项目根为基准的相对路径，多数情况下无需修改
```

| 变量 | 默认值 | 用途 |
|---|---|---|
| `GOBANG_DATA_ROOT` | `./data` | 原始数据下载根目录（download 脚本 --target） |
| `GOBANG_RAW_DIR` | `./data/fs15x_label28b` | 原始 ModelScope 数据子集（preprocess --raw） |
| `GOBANG_PROCESSED_DIR` | `./data/processed` | 预处理后的训练数据（train.py --data 默认） |
| `GOBANG_RUNS_DIR` | `./runs` | 训练输出目录 |
| `GOBANG_MODEL_DIR` | `./models` | 模型 checkpoint 存放目录 |

优先级：进程环境变量 > `.env` > 默认值

## 目录结构

```
gomoku/                核心包（纯逻辑，无 GUI）
  board.py             15×15 棋盘引擎：落子/撤销/胜负/合法位
  patterns.py          棋型评估：活四/冲四/活三/眠三/活二 + 威胁阶梯
  point15.py           旧 point.py 的 15×15 移植版
  config.py            .env 配置加载（无额外依赖）
  players/
    greedy.py          贪心棋手（威胁阶梯 + 1-ply 前瞻）
    random.py          随机棋手（基线）
    point15_player.py  point15 包装棋手
    nn_player.py       神经网络棋手（加载 .pt checkpoint，policy 选点）
training/
  model.py             8×128 残差 CNN（可配 blocks/channels）
  loader_katago.py     KataGo 蒸馏数据流式加载
  train.py             监督训练 CLI（每 N 分片自动 eval + 存 ckpt）
scripts/
  download_ms_dataset.py   ModelScope 数据下载
  preprocess_katago.py     原始 npz → 精简训练格式
  gpu_monitor.sh           远程 GPU 功率/温度落盘监控
gui.py                 人机 GUI（自绘棋盘 + legacy PNG 棋子）
evaluate.py            棋手对战评估 CLI
models/                模型 checkpoint（gitignore）
runs/                  训练输出/日志（gitignore）
data/                  数据（下载/预处理产物，gitignore）
tests/                 引擎/棋手单测
legacy/                旧版 tkinter GUI + point.py（保留）
```

## 数据准备（从零到可训练）

```bash
# 1. 下载原始数据（ModelScope fs15x_label28b，约 11.9GB，可断点续传）
uv run python scripts/download_ms_dataset.py

# 2. 预处理成训练格式（npz 裁剪 + unpack，多进程）
uv run python scripts/preprocess_katago.py --workers 8
```

处理完的 `data/processed` 里：train 2514 分片, 数据源作者声称50M数据；val 200 分片。
每个 npz：`board (N,2,15,15) uint8`、`policy (N,225) int16 计数`、`value (N,3) float16`。

## 快速上手

```bash
# 人机对战（电脑 = 贪心By Deepseek）
uv run python gui.py
# 人机对战 (电脑 = point15 By 2017年刚开始写代码的自己)
uv run python gui.py --player point15

# 人机对战（电脑 = 神经网络，--model 指向任意 checkpoint）
uv run python gui.py --model models/ckpt_ep0_shard59.pt
#   可选: --device cuda/cpu  --temperature 0.3(采样) 0(贪心,默认)

# 对战评估
uv run python evaluate.py --p1 greedy --p2 random --games 100
uv run python evaluate.py --p1 nn --p2 greedy --ckpt models/ckpt_ep0_shard59.pt --games 20

# 单测
uv run python -m tests.test_board
```

## 训练

```bash
# 冒烟（N 个分片，几分钟）
uv run python -m training.train --limit-shards 5 --epochs 1 --batch-size 1024 --out runs/smoke

# 全量（推荐 GPU，2 个 epoch + AMP）
uv run python -m training.train --epochs 2 --batch-size 1024 --device cuda --amp --out runs/full
```

数据目录默认读 `.env` 的 `GOBANG_PROCESSED_DIR`，也可 `--data` 显式覆盖。
每 `--save-every`（默认 100）个分片自动 eval + 存 `ckpt_ep{epoch}_shard{si}.pt`
（内含 `model` state_dict + `args` 训练参数），同时打印 `val policy_top1`。

### GPU 兼容性（作者运行过的环境配置）

| GPU | torch 版本 | 说明 |
|---|---|---|
| GTX 1060（Pascal sm_61） | **2.12.1+cu126**（Python 3.14 有 cp314 wheel，实测 CUDA 可用） | cu130 不支持 Pascal；Pascal FP16 无加速，**不要 `--amp`** |
| RTX 3060（Ampere sm_86） | 2.12.1+cu126 或 2.13+cu130 均可 | 建议 `--amp`（FP16 张量核） |

在已有 uv 镜像源配置基础上追加 PyTorch 官方索引装 cu126：

```bash
uv pip install "torch==2.12.1+cu126" "torchvision==0.27.1+cu126" \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match
```

### 实测吞吐（8×128，batch 1024）

| 环境 | 吞吐 | 1 epoch（63.6M） |
|---|---|---|
| CPU | ~80 样本/s | ~9 天 |
| GTX 1060 FP32 | ~900 样本/s | ~20 小时 |
| RTX 3060 + AMP | ~2500–3300 样本/s（估） | ~6–8 小时 |

参考进度：60 分片（151 万局面）→ `policy_top1=0.510`，`value_top1=0.659`，对战贪心 12:0 全胜。

## 远程监控 GPU

```bash
# 落盘 CSV（每 10s 记 功率/温度/占用/显存），远程 tail 即可
nohup bash scripts/gpu_monitor.sh runs/gpu_monitor.csv > /dev/null 2>&1 &
```

实时看：`ssh 主机 "watch -n1 nvidia-smi"`、`gpustat`、`nvtop`（在项目目录下
`tail -f runs/gpu_monitor.csv` 即可）。无公网 IP 时用 Tailscale 组网。

## 训练优化清单

优化方案的逐条评审、待实施清单、12×192 冒烟决策门见
[`docs/training-optimization.md`](docs/training-optimization.md)。

## 下一步

- 全量监督训练（63.6M × 2 epoch）→ 更强的策略/价值头
- 自对弈 RL（6×96 轻量模型用于提速）
- 贪心/point15 已确认弱于当前 NN 模型，后续以 NN 对局为准

## 其他

### 作者运系统环境

* Ubuntu 26.04 + 7700hq + m1060 + Cuda 13 
* Win 10 + R7 5800H + m3060 + Cuda 13
 

