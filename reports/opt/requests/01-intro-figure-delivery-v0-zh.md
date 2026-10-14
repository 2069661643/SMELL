# 01 引言图交付表 v0（面向实验侧 Agent）

> 需求方：论文侧（ICDE「SMELL」）。用途：为引言 teaser 图「长上下文 token 冗余统计」产出**统计数据包**；该图同时强化 §4.2 块稀疏（保留预算 ρ）的设计动机。
> 背景：原 TODO 起自 Notion 设计稿「图 1：一半以上的 token 是冗余的」；现选定方案 (a)：由实验侧用本项目自己的模型与数据产生统计（不用 Jenga 的现成数字）。
> 交付形式：仿 `docs/paper/evaluation/CATV/opt/` 的结果包（`data/` + `figures/` + `README` + 生成脚本，**禁止手改数字**）；完成后将数据包拷入 `docs/paper/evaluation/redundancy/`。
> 成本预估：单卡 4090，纯前向统计、无训练，预计 < 1 小时。

## 一、统计口径（硬性）

| # | 项 | 定义 |
|---|---|---|
| S1 | 模型 | 与主实验一致：位置适配后的 OPT-350M（C 产物）；前向为**稠密**（`sparse=1.0`），不得带掩码 |
| S2 | 打分器 | 现有 block-importance 预测器（与 `--predictor step5_a01a03_clients_causal` 同一权重）；对层 11–22 逐层打分，block/pool = 64；取原始打分、不额外归一 |
| S3 | 数据 | Discovery ICL 16K 生成式样本；从全局评测集（500 条）固定抽取 **≥32 条**（记录样本 ID 清单，seed 固定）；**前缀截断**到 L ∈ {4K, 8K, 16K}（2K 为可选项） |
| S4 | 指标 A（主） | 覆盖率 CR(ρ, L, layer)：按分数降序保留 top-ρ 比例的块后，其正重要性之和 ÷ 全部块正重要性之和；ρ ∈ {0.1, 0.2, 0.4, 0.6, 0.8} |
| S5 | 指标 B（辅） | 稀疏比例 SR(θ, L, layer)：打分低于 max(score) × θ 的块占比；θ ∈ {0.1, 0.3, 0.5}（θ=0.3 对齐 Jenga 口径） |
| S6 | 聚合 | 先按样本平均、再给跨层均值 ± std；CSV 保留逐层明细（`layer` 列）；再保留逐样本值（供误差棒） |

## 二、交付物

```
docs/paper/evaluation/redundancy/
├── README.md                  # 口径、复现命令、对账断言、checkpoint/commit/数据文件 hash
├── data/
│   ├── coverage_curve.csv     # length,layer,rho,mean,std,n（另建议 raw_coverage.csv 逐样本）
│   └── sparsity_ratio.csv     # length,layer,theta,mean,std,n
├── figures/fig0_redundancy_preview.{pdf,png}
│                              # 预览双面板：(a) CR(ρ) 曲线（4K/8K/16K）；(b) SR(0.3) vs 长度。样式不限
└── scripts（或注明 SMELL-v3 侧脚本路径与复跑命令）
```

## 三、验收

- A1. **数字如实**：不调参、不挑样本凑主张；若不等预期方向，如实交付并在 README 说明。
- A2. 期望方向（供参考，非硬性）：16K 上 SR(0.3) > 50%（支持「一半以上冗余」主张）、且 CR(0.4) 较高；SR/1−CR 随长度上升（16K ≥ 8K ≥ 4K）。
- A3. **确定性**：固定 seed 与样本清单；脚本重跑一致；内置对账断言（输出 N/N PASS），沿用「禁手改数字」约定。
- A4. 附 ailog（按现行约定），并回填本表第六节。

## 四、禁止事项

- 不使用带掩码（ρ=0.4）的前向做统计；不引入任何训练。
- 本数据包不含 k 轮转、lr 等内部口径；内部信息按惯例只进 ailog。

## 五、参考

- Jenga ATC'25 同型统计（仅作对照，**不要把其数字写进本包**）：Llama2「注意力分数 < 0.3×max」的比例 4K = 38.6% → 16K = 69.6%；见 `docs/zh/Jenga/zh.tex` 表 3。
- 论文侧使用位置：`01-introduction-v1.tex:10`（TODO 占位）；最终图样式（Times、无标题、白底黑框图例）由论文侧重绘。

## 六、回填（实验侧）

- F1. S2 的打分 dump 是否可复用现有代码路径？预计耗时？
  答：可复用。打分即模型内部 `OptSdpaPruneAttention._prune_block_scores`（predictor → `sum(dim=-2)`，与 Jenga 选块同口径）；本次用 monkeypatch 该函数 dump 层 11–22 的原始分（与 `temp/probe_selection_drift.py` 同款手法；亦可走 `config.vote_callback` 路径）。全量（32 样本 × {4K,8K,16K}）在 4090 上实测 **~4 分钟**（与另一训练任务并行时；脚本见 `docs/paper/evaluation/redundancy/scripts/collect_redundancy.py`）。
- F2. 样本子集/截断方式是否有调整？
  答：无调整。seed=0 从全局 500 条无放回抽 32 条（清单 `data/samples.csv`），同一批样本做前缀截断 4K/8K/16K（2K 可选项未做）。**一处需知晓**：模型用 **B=pos_only**（`a01_pos_only_500step`）而非需求写的 C=pos_lora——C 的权重不在本机，且本仓全部主实验与该 predictor（step5）均为 pos_only 口径；如确要 C，需从云端取回 checkpoint 后一键重跑（`--pos-checkpoint`）。
- F3. 结果摘要：16K 的 SR(0.3) = ？CR(0.4) = ？长度趋势是否成立？
  答：**SR(0.3)=85.9% ± 7.7%**（4K 61.6% → 8K 78.6% → 16K 85.9%，单调上升 ✓）；**CR(0.4)=77.4% ± 8.0%**（4K 64.9% → 8K 77.6% → 16K 77.4%，8K 后趋平）。A2 两条期望方向均成立（16K SR>50%、SR 随长度上升）。附加保障：`sparse=1.0` 前向与 dense 的 NLL **逐位相等**（|ΔNLL|=0）。数据包在 `docs/paper/evaluation/redundancy/`（README + data/*.csv + figures/fig0_redundancy_preview.{pdf,png} + summary.json）。
