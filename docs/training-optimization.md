# 训练优化清单（评审记录）

> 适用环境：RTX 3060 6GB + FP16（主力）/ GTX 1060 6GB FP32（备选）
> 任务阶段：**监督蒸馏**（KataGo 73M 老师软标签，63.6M 局面）；自对弈 RL 另记

---

## 1. 评审结论总表

| 建议 | 判定 | 一句话理由 |
|---|---|---|
| 软标签 loss 用 log_softmax+KL | ✅ 已实现 | `train.py` 的 `soft_ce = -(t·log_softmax(p)).mean()` 即软标签 CE（=KL+常数） |
| 严禁硬标签 CrossEntropyLoss | ✅ 本就未用 | 不踩坑 |
| np.load mmap_mode | ❌ 不适用 | 数据是**压缩 npz**，mmap 无效；单分片 2.8MB、28s 才读一次，磁盘占比 <2% |
| DataLoader num_workers / prefetch | ❌ 低优先 | 手写循环、GPU 计算是瓶颈；改 DataLoader 收益小 |
| **pin_memory + non_blocking** | ✅ P0 采纳 | CPU→GPU 传输优化，几行代码 |
| **D4 在线增强（8 变换）** | ✅ P0 采纳 | 自由规则完全对称，免费 8× 数据；board 与 policy 必须同步变换，value 不变 |
| 标签平滑 ε=0.1 | ⚠️ 实验 0.03 | 标签本就是老师软分布，0.1 会稀释信息 |
| **梯度裁剪 max_norm=1.0** | ✅ P0 采纳 | 大模型+大 batch 廉价保险；AMP 下 unscale→clip→step |
| **EMA decay=0.999** | ✅ P1 采纳 | 尾段震荡时推理更稳 |
| SGD+Nesterov lr=0.08 | ❌ 保持 AdamW | 监督蒸馏下 AdamW 更稳，SGD 切换只是信仰 |
| **warmup ~2000 步** | ✅ P1 采纳 | 整段 cosine 起步太猛，加线性预热 |
| 每 epoch 余弦重启 | ⚠️ 可选实验 | 当前整段 cosine 已够 |
| 固定验证集 + top1/top3 | ✅ P0 补 top-3 | 每 100 分片 eval 已有，补 top-3 一行 |
| 早停 | ❌ 不必要 | 固定数据单轮跑完，63M 不容易过拟合 |
| AMP GradScaler | ✅ 已实现 | `torch.amp.GradScaler("cuda")` + autocast；3060 必开，1060 别开 |
| 推理温度 T=0.8~1.0 | ⚠️ 仅娱乐 | GUI `--temperature` 已支持；评估/对战用 T=0；T≈1+Dirichlet 留给自对弈数据生成 |
| Policy 头加隐藏层 Conv1×1(192→32)+ReLU+1×1 | ✅ P1 采纳 | 成本≈0，特征组合后再决策 |
| Value 头改 scalar+Tanh+MSE | ❌ 阶段错位 | 监督标签是 3 类软概率，3 类 logits+软 CE 精确配对；**自对弈 RL 阶段再换** |
| 输入通道 2→3 | ❌ 不必要 | 己方/对方相对编码已隐含轮到谁；自由规则颜色对称 |
| **12 残差块 × 192 通道** | ✅ 本轮冒烟验证 | ~8M 参数，见 §3 决策门 |

---

## 2. 执行清单（按优先级）

### P0（低风险，先做）
- [ ] D4 增强：每分片随机取 8 变换之一（numpy），board 与 policy（225→15×15→变换→225）同步，value 不动
- [ ] GPU 传输：`torch.from_numpy(x).pin_memory()` + `.to(device, non_blocking=True)`
- [ ] 评估补 policy Top-3
- [ ] 梯度裁剪：`scaler.unscale_(opt)` → `clip_grad_norm_(model.parameters(), 1.0)` → `scaler.step(opt)`

### P1（推荐）
- [ ] EMA（decay=0.999）：影子权重，eval/推理用 EMA
- [ ] warmup：线性预热 ~2000 步再进 cosine
- [ ] Policy 头：`Conv1×1(C→32) + ReLU + Conv1×1(32→1)`（结构变更 → 旧 ckpt 不兼容）

### P2（实验项）
- [ ] 标签平滑 ε=0.03（仅 policy）
- [ ] 8×160 vs 10×128 A/B（若 12×192 冒烟不达标）
- [ ] 每 epoch 余弦重启对比

---

## 3. 模型参数量决策（12×192 冒烟验证）

| 方案 | 参数量 | 1 epoch 预估（3060+AMP） | 决策 |
|---|---|---|---|
| 8×128（当前） | 2.38M | ~5–8h | 基线 |
| **12×192（本轮试）** | **~8.0M** | **12–30h** | 显存不炸 & ≥600/s 才保留 |
| 8×160 | ~3.7M | ~6–10h | 回退首选 |
| 10×128 | ~3.0M | ~8–12h | 回退备选 |

- 显存预算（6GB）：fp16 权重+优化器 ~100MB，激活 ~1.2GB（batch 512），预计 <5GB，应能装下
- OOM 时：batch 512→256 优先；仍炸则回退 8×160
- 1060 上 12×192 无意义：fp32 ~360/s ≈ 49h/epoch，**锁定 3060**

### 冒烟命令（3060）

```bash
uv run python -m training.train \
  --blocks 12 --channels 192 --limit-shards 2 --epochs 1 \
  --batch-size 512 --device cuda --amp --out runs/probe3060_12x192
```

另开终端盯显存：`watch -n1 nvidia-smi`

### 冒烟结果记录表（跑完填）

| 项目 | 值 |
|---|---|
| 日期 / GPU | |
| blocks × channels | 12 × 192 |
| batch | 512（AMP fp16） |
| 显存峰值（MiB） | |
| 吞吐（日志尾部 xxx/s） | |
| 1 epoch 外推 | 63.6M / 吞吐 |
| val policy_top1（2 shard 仅供参考） | |
| 结论（保留/回退 8×160/降 batch） | |

---

## 4. 重要提醒

- **结构变更即 ckpt 不兼容**：policy 头、输入通道、value 头的任何改动都会让旧 checkpoint 无法加载（state_dict 不匹配）。当前 `models/ckpt_ep0_shard59.pt` 对应"旧结构"。
- 每次结构改动后，先跑 **60-shard 探针**（~30 分钟）对比 `policy_top1` 和对贪心胜率，再决定是否全量。
- 3060 换机注意：`uv sync` 装好依赖、`.env` 指到 3060 上的数据目录、数据先 rsync 过去。
