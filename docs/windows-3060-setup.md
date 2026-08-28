# RTX 3060 (Windows) 训练准备笔记【临时】

> 用途：下班回家在 3060 Windows 机器上搭环境、跑 12×192 冒烟。
> 本文件是临时笔记，用完可删；已确认不影响其他文件。

---

## 0. 已验证的事实（写死，不用再查）

- ✅ `torch 2.12.1+cu126` 的 cp314 **win_amd64** wheel 存在（官方索引确认）
- ✅ `torchvision 0.27.1+cu126` 的 cp314 **win_amd64** wheel 存在
- ✅ 3060 (sm_86)：AMP(FP16) 有效，**必须开 --amp**；6GB 显存跑 batch 512 预计 <5GB
- ⚠️ 1060 那套 cu130 不支持 Pascal 的限制与 3060 无关，cu126/cu130 都行

## 1. 环境安装（按顺序）

```powershell
# 1) 装 Python 3.14（**不要 3.14.1**，torchvision 明确排除 3.14.1）
#    python.org 安装时勾选 Add to PATH；或让 uv 管理：
uv python install 3.14.4
uv python pin 3.14.4

# 2) 装 uv（二选一）
winget install astral-sh.uv
#  或: irm https://astral.sh/uv/install.ps1 | iex      # 装完重开终端

# 3) 拿到项目代码（git clone / 压缩包）
#    ⚠️ 不要带着 Linux 的 .venv 拷过来，Windows 重新生成

# 4) 项目根目录装依赖（自动走 uv.toml 的清华源）
cd D:\gobang
uv sync

# 5) 装 torch cu126（Windows wheel 已确认存在）
uv pip install "torch==2.12.1+cu126" "torchvision==0.27.1+cu126" ^
  --extra-index-url https://download.pytorch.org/whl/cu126 ^
  --index-strategy unsafe-best-match
```

**数据**（6.6GB，一次性）：

```powershell
# 从 Linux 机器拷过来（Windows 10+ 自带 scp），然后改 .env
scp -r user@linux主机:/home/kita-ikuyo/dataset/processed D:/dataset/
```

**配置 .env**（项目根目录: copy .env.example .env，改成 Windows 路径）：

```text
GOBANG_PROCESSED_DIR=D:/dataset/processed
```

**自检**：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望: 2.12.1+cu126 / 12.6 / True
```

## 2. 冒烟 12×192

前台跑（先看 2 分钟速度）：

```powershell
cd D:\gobang
uv run python -m training.train --blocks 12 --channels 192 --limit-shards 2 --epochs 1 --batch-size 512 --device cuda --amp --out runs/probe3060_12x192
```

另一终端盯显存：

```powershell
nvidia-smi -l 2
# 或只看功率/温度/显存:
nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -l 2
```

后台版（关窗口也能跑）：

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
$py = ".\.venv\Scripts\python.exe"
$trainArgs = @('-m','training.train','--blocks','12','--channels','192','--limit-shards','2','--epochs','1','--batch-size','512','--device','cuda','--amp','--out','runs/probe3060_12x192')
$p = Start-Process -FilePath $py -ArgumentList $trainArgs -RedirectStandardOutput 'runs\probe.log' -RedirectStandardError 'runs\probe.err' -WindowStyle Hidden -PassThru
$p.Id            # 记下进程号
Get-Content runs\probe.log -Wait -Tail 20   # 实时看日志
```

**冒烟判定**（对照 docs/training-optimization.md §3 的表填）：

| 指标 | 达标线 | 不达标动作 |
|---|---|---|
| 显存峰值 | <5.5GB | 降 batch 512→256 |
| 吞吐（日志尾部 xxx/s） | ≥600/s | 回退 8×160（~3.7M） |
| 1 epoch 外推 | 63.6M / 吞吐 | >30h 建议回退 |

## 3. Windows 专属注意事项（容易翻车的点）

1. **禁止睡眠/休眠（头号坑）**：训练时屏幕能锁，但系统睡眠会掐 GPU。
   管理员 PowerShell:
   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   # 或 设置 → 电源和睡眠 → 从不
   ```
2. **Linux 的 bash 脚本在 Windows 原生跑不了**：nohup、gpu_monitor.sh、rsync 都是类 Unix 的。后台用 §2 的 Start-Process，监控用 nvidia-smi -l 2；要 rsync/WSL 工作流见 §4。
3. **杀毒软件**：把项目目录和 D:/dataset 加排除项，否则 npz 读取会莫名变慢/被拦截。
4. **日志中文乱码**：PowerShell 里先 chcp 65001 再跑。
5. **TDR（备用）**：极少见，但若报 CUDA error: device timeout，注册表 TdrDelay=10 并重启。
6. **checkpoint 跨平台**：Linux 上训的 models/ckpt_ep0_shard59.pt 在 Windows 能直接 torch.load(weights_only=False) 加载；但任何结构改动（policy 头等）后旧 ckpt 不兼容。

## 4. 备选：想用 bash 工作流就上 WSL2

```powershell
wsl --install -d Ubuntu        # Windows 驱动已支持 WSL CUDA，无需装 Linux 侧驱动
```

- WSL 里装 uv → uv sync → 跑 Linux 全套命令（nohup / gpu_monitor.sh / rsync 全兼容）
- nvidia-smi 在 WSL 里可用，但功率/温度读数不如 Windows 本生命令全
- 数据互通：/mnt/d/dataset（Windows 的 D 盘）

## 5. 回家速查流程

1. 装 Python 3.14.4 + uv → 拷代码 → uv sync → 装 cu126
2. 拷数据 → 改 .env → 自检 cuda.is_available() = True
3. 跑 §2 冒烟 → 显存/速度达标就继续全量（去掉 --limit-shards，epochs 1）
4. 结果填 docs/training-optimization.md 的表格，异常贴 runs/probe.err
5. 本笔记用完可删

