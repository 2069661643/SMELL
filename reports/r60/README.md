# r60 results package (BERT third segment r60–r89)

由 `reports/r60/make_report.py` 从 `logs/fed/`、`checkpoints/` 自动生成；纯 CPU、确定性；**不含图片**。
禁止手改数据；改脚本后重跑。原始 `logs/`、`temp/` 不进 git。

## 范围

- 主链：BERT 4 臂（a01/a03 × CATV off/on），**r60→r89**（`--round-offset 60 --rounds 30`），
  **lr=4.6667e-4（1.4e-3/3）**，L8/D8、clip 1.0、cycle、eval 500、`catv-normalize off`；
  起点为 cont30 的 `adapter_round059`。
- 技术线（a03）：dense VAR（`sparse=1.0`）、3-on（VAR+CATV client+VQ 0.5 per-result）、VQ 0.5 per-result，
  同为 r60→r89、lr=4.6667e-4。
- eval on/off：4 臂的 on 臂与 combo 提供去掩码（eval-off）评测；off 臂无掩码，on≡off。
- var-only（VAR Method A，CATV/VQ off）：**只有 r30 存档（端点 r29），没有 r60 续训**，单列并以 note 标注。

## 文件索引

| 文件 | 行数 | 作用 |
|---|---:|---|
| `data/bert_r60_rounds.csv` | 120 | 4 臂 × r60–r89 逐轮 NLL/acc/训练统计（on-mask 评测） |
| `data/bert_r60_endpoints.csv` | 4 | r59/r60/r89 端点、段内下降、acc、墙钟、adapter sha256 |
| `data/bert_r60_paired.csv` | 4 | r59→r89 逐样本配对 |
| `data/bert_r60_evaloff_rounds.csv` | 60 | on 臂逐轮 带掩码/去掩码 对比 |
| `data/bert_r60_evaloff_endpoints.csv` | 4 | r59/r89 的 on/off 端点 |
| `data/bert_r60_evaloff_paired.csv` | 6 | mask 效应、权重效应、总效应（r89） |
| `data/tech_r60_rounds.csv` | 90 | 三条技术线 × r60–r89 曲线 |
| `data/tech_r60_endpoints.csv` | 3 | 三条技术线 r59/r89、%/h、sha |
| `data/tech_r60_paired.csv` | 6 | 各线 r59→r89 + r89 跨线配对 |
| `data/combo_r60_evaloff_rounds.csv` | 30 | combo 逐轮 带掩码/去掩码 |
| `data/combo_r60_evaloff_endpoints.csv` | 2 | combo r59/r89 的 on/off 端点 |
| `data/combo_r60_evaloff_paired.csv` | 3 | combo mask/权重/总效应（r89） |
| `data/evaloff_history_r0_r89.csv` | 270 | 4 臂 on 臂 + combo 的全历史 eval on/off（r0-r89） |
| `data/evaloff_history_endpoints.csv` | 9 | 三条线 × r29/r59/r89 的 on/off/mask/权重代价 |
| `data/evaloff_history_paired.csv` | 9 | 各段端点的 mask 效应配对 |
| `data/history_r0_r89_rounds.csv` | 630 | 7 条线 × 全历史（r0-r29 / r30-r59 / r60-r89）逐轮 |
| `data/history_r0_r89_endpoints.csv` | 7 | 各线 r29/r59/r89 端点与两段降幅 |
| `data/provenance.csv` | 8 | 本包收录 run 的原始路径/状态 |
| `data/excluded_runs.csv` | 8 | 不收录的 partial/旧段/旧语义 run |
| `data/var_only_r30_rounds.csv` | 30 | VAR-only（Method A）r0–r29 曲线（无 r60 续训） |
| `data/var_only_r30_endpoints.csv` | 1 | VAR-only r29 端点 + note |

## 主要端点（r89）

| 线 | r89 NLL | acc |
|---|---:|---:|
| dense VAR（sparse=1.0） | 1.3096 | .520 |
| 4 臂 a03_on | 1.4610 | .488 |
| 3-on（VAR+CATV+VQ pr） | 1.4598 | .480 |
| 4 臂 a01_on | 1.4659 | .490 |
| 4 臂 a01_off | 1.4703 | .502 |
| 4 臂 a03_off | 1.4724 | .490 |
| VQ 0.5 per-result | 1.4780 | .494 |

## eval-off 分解（r89，同一 adapter 去掩码）

| α | on-mask | off-mask | mask 效应 | on 臂权重代价（off − off臂） | 净 on-vs-off |
|---|---:|---:|---:|---:|---:|
| a01 | 1.4659 | 1.5026 | -0.0367 | +0.0324 | -0.0044 |
| a03 | 1.4610 | 1.5015 | -0.0405 | +0.0291 | -0.0114 |


- mask 效应 = on(同一 adapter) − off(同一 adapter)；权重代价 = off(该 on 臂) − off 臂 r89。
- combo r89：on 1.4598 / off 1.4979（mask -0.0381）。

## 口径与 caveat

- r89 评测为 **on-mask**（CATV on 臂/combo）；跨臂比较用同条件端点，不要混用 eval-off 数字。
- CATV 共识掩码在 r60 重置（跨 run 不继承），与之前每段续训一致。
- dense VAR 用 `sparse=1.0`（不剪枝的 dense fallback），与 sparse 0.4 线不是同一推理成本。
- 配对检验：逐样本 NLL，双尾；p 为正态近似（与历史 ailog 一致）。
- 对账：19/19 断言 PASS（4 臂 r89、3 条技术线 r89、eval-off 端点与配对存在性）。
