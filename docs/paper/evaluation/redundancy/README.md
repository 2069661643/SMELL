# 长上下文 token 冗余统计（引言 teaser 数据包）

> 用途：支撑 01 引言 teaser 图「长上下文 token 冗余」与 §4.2 块稀疏（保留预算 ρ）的设计动机。
> 生成：`scripts/collect_redundancy.py`（本次运行 2026-10-14，SMELL-v3 @ `df6d5e2`，AutoDL 1×RTX 4090，纯前向无训练）。
> 约定：**禁止手改本目录数字**；所有数字由脚本从原始权重/数据重算。

## 1. 关键结论（cross-layer mean ± std，32 样本 × 12 层）

| 指标 | 4K | 8K | 16K |
|---|---|---|---|
| **SR(θ=0.3)**：块分 < 0.3×max 的占比 | 61.6% ± 17.7% | 78.6% ± 10.9% | **85.9% ± 7.7%** |
| SR(θ=0.1) | 18.5% ± 14.4% | 40.5% ± 17.1% | 46.0% ± 15.4% |
| SR(θ=0.5) | 88.4% ± 7.5% | 93.7% ± 4.2% | 96.0% ± 2.7% |
| **CR(ρ=0.4)**：top-40% 块覆盖的正重要性占比 | 64.9% ± 7.0% | 77.6% ± 9.0% | **77.4% ± 8.0%** |
| CR(ρ=0.2) | 38.6% ± 6.0% | 50.5% ± 8.1% | 52.6% ± 7.8% |
| CR(ρ=0.1) | 22.7% ± 4.3% | 29.6% ± 5.7% | 32.5% ± 5.8% |

- 16K 上 **SR(0.3)=85.9% > 50%**、「SR 随长度上升（16K ≥ 8K ≥ 4K）」两条期望方向均成立；
  CR(0.4) 在 16K 达 77%，说明保留 40% 块即可覆盖大部分正重要性。
- 观察到的数值细节（非主张）：max 块分随长度增长（4K≈420 → 8K≈850 → 16K≈1950），
  层间 std 较大（尤其是 SR(0.1)），故图中误差棒为**跨层 std**；逐层数值见 CSV。

## 2. 统计口径（对应需求表 S1–S6）

| 项 | 本次实现 |
|---|---|
| 模型 | OPT-350M + 位置适配 **pos_only**（`checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt`） |
| 前向 | `attn=sdpa_prune`、`dtype=fp32`、**`sparse=1.0`**（保留全部块 ⇒ 稠密等价；见断言 A1）、无掩码 |
| 打分器 | block-importance predictor = `step5_a01a03_clients_causal`（144 张量，缺 bias 零填充）；层 11–22、pool=64；原始分 `sum(dim=-2)`，**不额外归一** |
| 数据 | `dataset_v3/discovery_16k/a03/global_test_input_ids.npy`，seed=0 抽 32 条（清单 `data/samples.csv`），**前缀截断**到 L ∈ {4K, 8K, 16K} |
| 指标 A | CR(ρ,L,layer) = top-k 正分质量 / 全块正分质量，k = floor(ρ·n_blk)（与模型选块规则一致，ρ ∈ {0.1,0.2,0.4,0.6,0.8}） |
| 指标 B | SR(θ,L,layer) = #{score < max(score)·θ} / n_blk（θ ∈ {0.1,0.3,0.5}） |
| 聚合 | 先按样本均值，再跨 12 层求 mean/std；`data/raw_metrics.csv` 保留逐样本×逐层明细 |

**与需求表的一处偏差（需知晓）**：需求写「位置适配后的 OPT-350M（C 产物）」；C=pos_lora 的权重不在本机，
且本仓**全部主实验**（TD-3/k24 系列、predictor step5 训练）均使用 **B=pos_only**。为保证"与主实验一致"
本次用 pos_only；如确需 pos_lora 口径，需要从云端取回 `checkpoints/posemb_step1/a01_pos_lora_500step/` 后重跑
（脚本 `--pos-checkpoint` 一键切换）。

## 3. 复现

```bash
PY=~/miniconda3/envs/SMELL/bin/python
$PY docs/paper/evaluation/redundancy/scripts/collect_redundancy.py --gpu 0
# 冒烟：--n-samples 2 --lengths 4096 --out-dir /tmp/opencode/red_smoke
```

- 环境：AutoDL 1×RTX 4090；运行 ~4 分钟（与另一个训练任务并行时的实测；空卡更快）。
- 确定性：seed 固定（样本清单固定）、无 dropout/no_grad、脚本重跑输出一致（PDF 为确定性写出）。
- 输入校验（`summary.json → hashes`）：

| 输入 | sha256 前缀 | 与本仓清单一致 |
|---|---|---|
| `pos_embed.pt` | `414394f8b71971ec` | ✅（`docs/weight-manifest.md`） |
| `predictor.pth`（step5） | `1fabeab15e81b378` | ✅ |
| `pruned_config.pth` | `38e94fba64899d18` | ✅ |
| `global_test_input_ids.npy` | `b20f9304f39ea8da` | ✅ |

- 输出校验（本 commit 生成，sha256 前缀）：`coverage_curve.csv e23a94cd…`、`sparsity_ratio.csv 82971bf1…`、
  `raw_metrics.csv 822c4c4a…`、`samples.csv 575542a2…`、`fig0_redundancy_preview.pdf 5d5136ab…`、
  `fig0_redundancy_preview.png be51f60c…`、`summary.json a9b20837…`。

## 4. 对账断言（脚本内置，全部通过）

1. **A1 稠密等价**：同一 4K 样本上 `sdpa_prune(sparse=1.0)` 与 `sdpa`（dense）的 NLL **逐位相等**
   （`|ΔNLL|=0.0`）⇒ 打分前向不改变模型数值。
2. **A2 结构**：每次前向捕获层集合恰为 {11..22}；`n_blk = L/64`（64/128/256）；分数全有限。
3. **A3 范围**：所有 CR/SR 的 (逐样本均值) ∈ [0,1]。
4. **A4 计数**：32 样本 × 3 长度 × 12 层 × (5 ρ + 3 θ) = 9216 条 raw 记录（`raw_metrics.csv` 9216 行 + 表头）。

## 5. 文件

```
redundancy/
├── README.md                     # 本文件
├── summary.json                  # 关键数字 + 输入 hash + 稠密等价结果
├── data/
│   ├── samples.csv               # 样本清单（32 个 global_test 下标）
│   ├── coverage_curve.csv        # length,layer,rho,mean,std,n
│   ├── sparsity_ratio.csv        # length,layer,theta,mean,std,n
│   └── raw_metrics.csv           # metric,length,sample_index,layer,param,value（逐样本×逐层）
├── figures/fig0_redundancy_preview.{pdf,png}
│                                 # (a) CR(ρ) 曲线（4K/8K/16K）；(b) SR(θ) vs 长度（θ=0.3 主）
└── scripts/collect_redundancy.py # 唯一入口（自动定位仓库；可 --repo 覆盖）
```

## 6. 说明与限制

- 误差棒 = **跨层 std**（需求 S6 指定）；若论文想用跨样本 std，可用 `raw_metrics.csv` 自行重聚合。
- block score 为 predictor 输出沿 query 维求和（与 Jenga/本仓选块逻辑一致）；CR 的"正重要性"= `max(score,0)`。
- 前缀截断是简单前缀（不重排 demo），统计的是"块重要性分布"随长度变化，与具体 task 答案无关。
- 运行时的 GPU 并行任务（BERT）不影响数值（脚本无随机性；稠密等价断言亦通过）。
