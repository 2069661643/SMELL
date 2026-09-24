# Ailog 260924-103741 — SMELL-v3: src/ 简化框架落地 + 16k 冒烟通过 + 位置嵌入外推发现

## 概述

按 `260923-210703` 计划，用 3 个子智能体并行搭建 `src/` 简化框架（数据管线 / 模型与训练 / 评测与联邦），并补上关键缺口 `position_embed`（OPT 位置表 2050 行，16k 序列必须扩展，此前 run_fed 因 gather 越界崩溃）。串行 FedAvg + ZOO + LoRA 在 16k 上端到端冒烟通过（1 client、4 样本、1 round，5.5s，峰值显存未溢出 8GB）。评测侧实现了全文与 answer-only 两种 PPL。**同时发现重要研究问题**：OPT 的学习式绝对位置嵌入零样本无法外推——PPL 随长度从 2048 的 20.2 恶化到 4096 的 113.6 / 8192 的 356.8 / 16384 的 588.2；Jenga 的 `×i` 复制缩放最差（16384 时 5791）；插值稍好（536.9）。后续必须先解决"位置适配"（可训练位置嵌入或 RoPE retrofit）再做正式 16k 实验。

## 变更列表

| 时间 | 操作 | 文件 | 说明 |
|---|---|---|---|
| 21:09 | 新增 | `src/models/modeling_opt_smell.py` (1425 行) | 从 Jenga `modeling_opt.py` 原样复制 + `SMELL 3 ... COPY` 头（后续接 TokenSelector/CATV） |
| 21:12 | 新增 | `src/models/token_selector.py` (114 行) | `BaseTokenSelector` / `JengaSparseSelector`（块级 topk）/ `CATVSelector`（占位） |
| 21:13 | 新增 | `src/train/lora.py` (47 行) | LoRA 工厂 + 可训练参数字典一致性断言 |
| 21:13 | 新增 | `src/train/zoo.py` (149 行) | 中心差分前向梯度（按维度无偏缩放）+ FedAvg 聚合工具；CPU 自测 PASS |
| 21:13 | 新增 | `src/fed/serial_fedavg.py` (177 行) | ClientRunner / ServerAggregator（samples/equal 加权、一致性断言、JSONL 指标） |
| 21:13 | 新增 | `src/fed/run_fed.py` (279 行) | 串行 FedAvg CLI（`--alpha/--catv/--local-steps/--rounds/--weight/--gpu` 等） |
| 21:13 | 新增 | `src/eval/ppl.py` (187 行) | 全文/answer-only PPL（token 加权 + per-sample），logits 分块避免 16k×50272 |
| 21:13 | 新增 | `src/bench/memory.py` (187 行) | 16k 显存矩阵（sparse/dense × LoRA targets × ckpt） |
| 21:13 | 新增 | `src/data/build_discovery_16k.py` (434 行) | Discovery per-class Dirichlet 16k 构建（方案 A，LongICLBench 模板，uint16/int64/int32） |
| 21:13 | 新增 | `src/data/check_partition.py` (214 行) | 不变量/泄漏/JS 散度检查；mini 数据 PASSED（0 errors） |
| 21:25 | 新增/修改 | `src/models/position_embed.py` + `ppl.py`/`run_fed.py`/`memory.py` 接线 | 三种模式：`jenga_dup_scaled` / `duplicate` / `interpolate`；`ensure_positions()` |
| 10:35 | 新增 | `src/eval/ppl.py --pos-mode` | 暴露位置模式便于 P2 A/B |
| 10:45 | 验证 | 冒烟与探针日志 | `temp/logs/smoke_260924-103441/`（gitignored） |

## 验证结果

**静态/单测**：10 个新文件 `py_compile` 全过；`zoo.py`、`serial_fedavg.py` CPU 自测 PASS；mini 数据 `check_partition` PASSED（seq_len=16384 %64、spans 解码=label、三池互斥、复用=0、可复现）。

**串行 FedAvg 冒烟**（mini 数据：1 client / 4 train / 16k / 1 round / local_steps=1 / zo_dirs=2，LoRA r=8 qkvo=1,572,864 参数）：
```
[positions] mode=jenga_dup_scaled capacity=2048 -> seq_len=16384 (extended to 16384)
[fed] round 0/0 loss=7.5 delta_norm=4.4283e+04 cos=None time=5.5s
metrics: communication_bytes=3145728, round_seconds=5.49, delta_norm_mean=44282.6
```

**PPL 冒烟 + 长度曲线**（base OPT-350M，无 adapter，sparse=1.0，1 样本）：
| seq_len | 位置模式 | full-text token PPL |
|---|---|---|
| 1024 | 未扩展 | 30.5 |
| 2048 | 未扩展 | **20.2** |
| 4096 | duplicate | 113.6 |
| 8192 | duplicate | 356.8 |
| 16384 | duplicate | 588.2 |
| 16384 | interpolate | 536.9 |
| 16384 | jenga_dup_scaled | **5791.2** |

## 设计决策 / 发现

- **位置扩展必须且已实现**：Jenga 原版脚本用 `×i` 缩放的复制方案在零样本评测中灾难性退化（5791 vs 522），推测其正式实验依赖全量微调让位置嵌入参与适配（其 LoRA 路径不训位置嵌入，故零样本不可用）。
- **学习式绝对位置没有零样本外推能力**：一旦位置超过 2048，PPL 立即从 20 涨到 100+，且随长度继续恶化。这与 RoPE 模型（可线性插值/NTK 外推）根本不同。
- **影响**：正式 16k 实验前必须决定位置适配方案：(a) 将扩展位置嵌入纳入可训练参数（+16.8M 参数，会显著恶化 ZOO 的 SNR=√(M/d)）；(b) 加入短程长上下文适配阶段（先用 LM 目标适配位置嵌入/低秩位置增量）；(c) RoPE retrofit（改动最大，但外推最优，且与 FA2 兼容）。见待办。
- **随机 predictor 警告**：Jenga 稀疏路径的 predictor 对 OPT-350M 无预训练权重（全随机），sparse=0.4 的 PPL 不能作为质量依据；本次仅验证机制。CATV 接入后替换选择器。
- **ZOO 步长**：1 步 lr=1e-3 时 `‖Δ‖≈4.4e4`，量级偏大（估计器按维度无偏缩放），正式实验需重调 lr/eps/directions。

## 待办 / 风险

- **P2 位置适配决策**（阻塞正式实验）：可训练位置嵌入 vs 适配阶段 vs RoPE retrofit；建议先做 4096/8192/16384 的小规模 LM 适配对照（base 冻结，仅位置/LoRA 可训），用 G-PPL 判定。
- 全量数据构建（30 clients × 100 + local 16 + G-PPL 500）尚未运行；mini 单次 29.7s、token 缓存命中率低，预估全量 ~1-1.5h（可在服务器执行）。
- CATV 选择器仍是占位；`--catv on` 会明确报错退出。
- 云端 4 并发消融脚本尚未编写；服务器环境未就绪。
