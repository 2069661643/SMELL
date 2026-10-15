# Ailog 261015-213000 — SMELL-v3 DistilBERT：修复版 r30 四臂消融完成 —— CATV 显著有效（a01/a03）

> 承接 `261014-143500`（P0a valid-mean + block0 修复与冒烟）。修复版四臂链 `temp/run_bert_da_fixed_chain_bg.sh`
> 于 2026-10-14 14:51 → 2026-10-15 20:02 完成（30 轮/臂，eval 500，cycle，L8D8/lr=1.4e-3）。
> 数据：a01/a03；sparse=0.4 剪枝路径；`keep_block0=True`（修复生效，config 落盘核验）。

## 一、四臂结果（own-path：各自 round-0 eval 为基线；paired 检验 n=500）

| 臂 | r0 NLL | r29 NLL | ΔNLL% | ERR↓ | ΔAcc% | clipC/轮 | s/轮 | **%/h** |
|---|---|---|---|---|---|---|---|---|
| r30_a01_catv_off | 1.6910 | 1.5836 | +6.35% | +7.6% | +10.4% | 205 | 790 | +0.9 |
| **r30_a01_catv_on** | 1.7527 | **1.5563** | **+11.20%** | +8.4% | +12.3% | 220 | 776 | **+1.6** |
| r30_a03_catv_off | 1.6906 | 1.5887 | +6.03% | +7.0% | +9.4% | 189 | 772 | +0.9 |
| **r30_a03_catv_on** | 1.7511 | **1.5484** | **+11.57%** | +7.5% | +10.7% | 234 | 778 | **+1.7** |

**配对检验（per-sample NLL）**
- 各臂 r0 vs r29：全部显著（ΔNLL −0.10 ~ −0.20，t≈−9.5~−11.4，p 1e-21 ~ 1e-30）；
- **CATV on vs off @r29**：a01 **ΔNLL=−0.0272（t=−2.16，p=0.030）**；a03 **ΔNLL=−0.0403（t=−3.02，p=0.0025）** ⇒ 修复后 CATV 在两个 α 上均显著为正，且 %/h 亦更高（+1.6/+1.7 vs +0.9）。

## 二、缺陷版 vs 修复版（同协议，30 轮）

| 臂 | 缺陷版 ΔNLL%（own） | 修复版 ΔNLL%（own） | 绝对终值（缺陷→修复） |
|---|---|---|---|
| a01_off | +8.29% | +6.35% | 1.6563 → **1.5836** |
| a01_on | +1.72%（起点被假象注水：1.5424） | +11.20%（起点 1.7527） | 1.5159 → 1.5563* |
| a03_off | +11.86% | +6.03% | 1.5898 → **1.5887** |
| a03_on | 空（a03_on 停机） | +11.57% | — |

- 缺陷 on 臂的「低起点/低增益」= 冻结前缀掩码顺带保 block0 的假象收益；修复后 on 臂起点不再被注水（1.75 vs 1.69），但**终点反超 off**（1.5563/1.5484 vs 1.5836/1.5887）。
- \*a01_on 修复终值略高于缺陷版（1.556 vs 1.516）：缺陷版的绝对水平受"前缀掩码在 eval 中始终保留头部 51 块"的强先验加持，不可与修复版直接比。
- **结论**：修复后的 sparse 基线（block0 保 [CLS]）整体更强（a01_off 1.5836 vs 缺陷 1.6563；a03_off 基本持平 1.5887 vs 1.5898），CATV 的真实增益得以干净测出且显著。

## 三、归档（USE）

`checkpoints/distilbert_ablation_r30_fixed/`（每臂含 `adapter_round029/`、`metrics.jsonl`、`config.json`、`SHA256SUMS`；config 核验 `keep_block0=True`、`lr=0.0014`、`L=8`、`D=8`、`rounds=30`）：

| 臂 | adapter sha256 |
|---|---|
| r30_a01_catv_off | `6b83f8fa104cdbcbd00c977eb5529a1d53cea0038afcaabb234cfa2a5485f15b` |
| r30_a01_catv_on | `ed72c112464e0dec0b7f611b4749911248dfb1a0c7efc1aea91801e51c058a2a` |
| r30_a03_catv_off | `ab80899cd03f5c5f4e1469070fb764cc3e0b566af99720a09a2d16cdd7436741` |
| r30_a03_catv_on | `8684622738f960cf13a2678141d8b9cb891b60b796acaee8e96fa3446fd52a20` |

报告：`temp/logs/bert_r30_fixed_report.txt`（含逐轮曲线与配对检验）；逐轮 per-sample：`logs/fed/bert_da_fixed/<arm>/<tag>/eval_roundNNN_persample.json`。

## 四、口径与 caveat

- on 臂 **r0 eval 即带 round-0 掩码**（训练侧第 0 轮无掩码）；off 臂无掩码 ⇒ r0 起点差（−0.06 nats）是掩码的净效应，不是不公平；跨臂比较用 **r29 同轮配对**为准。
- 掩码为真·分数驱动（逐轮微漂移、预算 51/51；修复验证见 `261014-143500` §2.2）。
- clip 依赖度高（189–234 client-轮 被裁），lr=1.4e-3 为重尾下的上限档；论文可备注。
- 缺陷臂（`checkpoints/distilbert_ablation_r30_defect/`）仅存档、禁用于论文（D1/D4）。

## 五、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10-15 20:02 | 完成 | `logs/fed/bert_da_fixed/*` | 修复版 r30 四臂 30/30 |
| 10-15 20:1x | 归档 | `checkpoints/distilbert_ablation_r30_fixed/` | 四臂 adapter + SHA256SUMS |
| 10-15 21:30 | 新增 | 本 ailog | 结果表 + 配对检验 + 缺陷/修复对照 |
| 10-15 21:3x | 修改 | `docs/weight-manifest.md` §6 | 修复版归档 sha256 + 更新运行口径 |
