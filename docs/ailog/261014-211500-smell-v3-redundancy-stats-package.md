# Ailog 261014-211500 — Phase N: 引言冗余统计数据包（token redundancy teaser）

> 需求：`reports/opt/requests/01-intro-figure-delivery-v0-zh.md`（论文侧，01 引言 teaser「长上下文 token 冗余」，
> 同时支撑 §4.2 块稀疏 ρ 的动机）。用户要求与正在跑的 BERT 实验并行完成并交付。

## 1. 交付物

`docs/paper/evaluation/redundancy/`（按需求表路径直接落盘，git 跟踪）：

| 文件 | 内容 |
|---|---|
| `scripts/collect_redundancy.py` | 唯一入口：自动定位仓库；纯前向；固定 seed；内含对账断言 |
| `data/samples.csv` | 32 个 global_test 样本下标（seed=0 无放回抽取） |
| `data/coverage_curve.csv` | length,layer,rho,mean,std,n（CR） |
| `data/sparsity_ratio.csv` | length,layer,theta,mean,std,n（SR） |
| `data/raw_metrics.csv` | 逐样本×逐层明细（9216 行） |
| `figures/fig0_redundancy_preview.{pdf,png}` | (a) CR(ρ) 曲线 4K/8K/16K；(b) SR(θ) vs 长度（θ=0.3 主） |
| `README.md` / `summary.json` | 口径、复现、hash、关键数字 |

## 2. 关键结果（cross-layer mean ± std）

| 指标 | 4K | 8K | 16K |
|---|---|---|---|
| SR(θ=0.3) | 61.6% ± 17.7% | 78.6% ± 10.9% | **85.9% ± 7.7%** |
| CR(ρ=0.4) | 64.9% ± 7.0% | 77.6% ± 9.0% | **77.4% ± 8.0%** |

- A2 期望方向全部成立（16K SR>50%；SR 随长度单调上升）；CR(0.4) 8K 后趋平。
- 附加反例防护：`sparse=1.0`（保留全部块）与 dense 前向 NLL **逐位相等**（|ΔNLL|=0）——承诺"稠密统计"。

## 3. 实现与口径

- 模型：OPT-350M + **pos_only** 位置表 + step5 predictor；层 11–22、pool=64；`attn=sdpa_prune, dtype=fp32, sparse=1.0`、无掩码。
- 打分：monkeypatch `OptSdpaPruneAttention._prune_block_scores` dump 原始块分（`sum(dim=-2)`，不归一）；
- 指标：CR(ρ)=top-k 正分质量/全正分质量（k=floor(ρ·n_blk)，与选块规则一致）；SR(θ)=#{score<max·θ}/n_blk；
- 数据：a03 global_test 500 → seed=0 抽 32 条（清单固定）→ 前缀截断 4K/8K/16K；
- 输入 hash 与 `docs/weight-manifest.md` 一致（pos_embed 414394f8 / step5 1fabeab1 / pruned_config 38e94fba）。
- **与需求表的偏差（已在 README/回填注明）**：需求写"C 产物（pos_lora）"，实际用 B=pos_only——C 权重不在本机，
  且全部主实验 + step5 predictor 均为 pos_only 口径；如需 C 需云端取回后重跑（脚本支持 `--pos-checkpoint` 切换）。

## 4. 过程记录

| 时间 | 操作 | 结果 |
|---|---|---|
| 20:45 | 首跑（32×3 长度） | 聚合段断言误伤：CR 浮点累加 1.0000000029 > 1 → 中止（前向全部完成但未落盘） |
| 20:48 | 修复 | CR 输出 clamp 到 [0,1]；断言加 1e-9 容差；raw CSV 提前落盘（防崩溃重跑） |
| 20:50 | 重跑 | ~4 min 完成（与 BERT 训练/评测任务并行；显存无冲突） |
| 21:10 | 交付 | README + 回填需求表 §六（F1–F3）+ commit |

## 5. 回填（需求表 §六）

- F1：可复用 `_prune_block_scores` dump（同 probe_selection_drift 手法）；全量实测 ~4 分钟。
- F2：无调整；仅模型口径由 C 改 B（原因如上），样本/截断按需原样。
- F3：SR(0.3)16K=85.9%、CR(0.4)16K=77.4%，趋势成立。
