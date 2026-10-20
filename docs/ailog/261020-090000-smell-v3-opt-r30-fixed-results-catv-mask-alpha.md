# Ailog 261020-090000 — SMELL-v3 OPT-350M：修复版（cycle 语料轮转）r30 四臂完成 —— full 不变、answer 大幅改善；CATV 为"掩码主导 + α 依赖"

> 承接用户 10/15 指令（"重测 opt350m，本次测修复版能有多好，依旧 4 项消融 r30"；口径：统一 a01 旧正式档 + `--doc-order cycle` + eval-every 5）。
> 链：`temp/run_opt_r30_fixed_chain_bg.sh`，2026-10-15 21:34 → 10/20 02:21；fresh 基线先测；每臂 30 轮、~24h。
> 归档：`checkpoints/opt_ablation_r30_fixed/`（4 臂 adapter_round029 + metrics + config + SHA256SUMS）。
> 报告：`temp/logs/opt_r30_fixed_report.txt`、`temp/logs/opt_r30_evaloff_report.txt`。

## 一、配置（统一、与旧 a01 档一致）

k=24 全层、D=32、L=2、c=30、r=30、LoRA r=1/α=2、lr=0.30、delta-clip=13.34、fp32+sdpa_prune+sparse 0.4、
step5 predictor + pos_only；**唯一协议修复 = `--doc-order cycle`**（旧长跑每 client 固定前 2 篇）。
评测：每 5 轮（r4/9/14/19/24/29）；fresh 基线（无 adapter）**33.6653 / 1740.63**（a01/a03 相同——同 500 条 test 行 + 同 base 模型，属预期，复测与 10-14 旧测量逐位一致）。

## 二、四臂结果

| 臂 | full r4→r29 | own% | vs fresh% | answer r29 | answer own%* | train_h | %/h |
|---|---|---|---|---|---|---|---|
| r30_a01_catv_off | 32.848 → **31.533** | +4.00 | **+6.33** | 2436.1 | −16.8 | 23.90 | +0.17 |
| r30_a01_catv_on | 33.210 → 31.635 | +4.74 | +6.03 | 2474.4 | −23.3 | 24.06 | +0.20 |
| r30_a03_catv_off | 32.869 → **31.457** | +4.30 | **+6.56** | 2448.8 | −19.5 | 24.65 | +0.17 |
| r30_a03_catv_on | 33.238 → 31.696 | +4.64 | +5.85 | **2291.7** | −9.7 | 24.04 | +0.19 |

\*answer 的 own% = (r4−r29)/r4；负 = answer PPL 随训练变差（已知退化），CATV 臂退化最小。

**与未修复版（a01）同轮对比**：
| 臂 | full r29 修复/旧 | answer r29 修复/旧 | Δfull | Δanswer |
|---|---|---|---|---|
| off | 31.533 / 31.467 | **2436 / 2907** | +0.07 | **−471（−16%）** |
| on | 31.635 / 31.696 | **2474 / 2818** | −0.06 | **−344（−12%）** |

配对（r29，n=500）：old vs fixed 的 **answer** ΔNLL = −0.1928（off）/ −0.1150（on），p≈0 / 2.5e-12；**full** 仅 ±0.002。

## 三、CATV 效应分解（deploy = 带掩码推理；evaloff = 同 adapter 无掩码）

| α | 对比 | full ΔNLL | answer ΔNLL |
|---|---|---|---|
| a01 | **权重**（off vs on-evaloff） | −0.0049（p≈0，更好） | +0.0532（p=2.8e-4，更差） |
| a01 | **掩码**（on-evaloff vs on-evalon） | +0.0082（p≈0，更差） | −0.0270（p=0.015，更好） |
| a01 | **部署净**（off vs on-evalon） | +0.0032（略差） | +0.0262（**p=0.14 n.s.**） |
| a03 | **权重** | −0.0013（p=4e-16，更好） | +0.0089（p=0.55 n.s.） |
| a03 | **掩码** | +0.0088（p≈0，更差） | **−0.1021（p=2.7e-15，更好）** |
| a03 | **部署净** | +0.0076（略差） | **−0.0932（p=5.4e-8，更好）** |

## 四、结论（论文口径）

1. **cycle 修复的收益集中在 answer**：同轮 full 不变（±0.004 nats），answer 显著改善（a01 −0.19/−0.12 nats，p≈0；终值 −12%~−16%）。旧"固定 60 篇"缺陷主要伤害 answer 侧。
2. **CATV 是"推理掩码效应"，且 α 依赖**：
   - 掩码本身：full +0.008~0.009（更差，显著）、answer −0.027/−0.102（更好，显著）；
   - 训练权重：full −0.001~−0.005（更好）、answer 中性~更差；
   - **部署净效应：a03 answer −0.093（p=5e-8，稳健）；a01 +0.026（n.s.）** ⇒ 修复协议下 CATV 只在 α=0.3 有净收益，且收益全部来自推理期掩码。
3. **旧结论需改写**：旧协议 a01 的 CATV answer −0.0516（p=0.0048）在 cycle 下不复现（+0.026 n.s.）——旧收益与固定语料过拟合的抵消有关（协议副产物）。
4. full 侧：所有 on 臂部署口径 full 略差（+0.003~+0.008 nats，显著但量级小），与旧报告"mask-effect full +0.30~0.41"方向一致。

## 五、归档（USE）与 sha256

`checkpoints/opt_ablation_r30_fixed/<arm>/`（adapter_round029 + metrics.jsonl + config.json + SHA256SUMS）：

| 臂 | adapter sha256 |
|---|---|
| r30_a01_catv_off | `792f66131ad46483582b7f13103a77f6396524cf3cf73a51450d86308093d9e5` |
| r30_a01_catv_on | `32e46b70adf1a0eb5c84d160c860fde09a3c030d61bf5b8cc5b059ea7dc7eb07` |
| r30_a03_catv_off | `e1d4acda62440c2cf529273cff418df77b5ca2a6d4e1d28ff94e420a64923b7a` |
| r30_a03_catv_on | `b3cc1cc5f875d93e9487986926c054c1c76cd2a7df7754df947fa8cc966b0c23` |

## 六、Caveat

- 旧 a03 历史跑（r=8/α=16、lr=0.10、15 轮续跑链）不可与本链直接比；本链 a03 用统一 a01 档。
- eval 每 5 轮（旧口径），"own%" 起点是 r4（训练 5 轮后），非训练前；跨协议对比用 fresh 基线。
- fresh 基线 a01/a03 相同属构造性等价（同 test 行、无 adapter）。
- 并行评测（10/18 11:49–10/19 02:57 的 evaloff）与训练共卡，arm3/4 单轮略慢（53–56 min）。

## 七、变更列表

| 时间 | 操作 | 文件/产物 | 影响 |
|---|---|---|---|
| 10-15 21:34 | 启动 | `temp/run_opt_r30_fixed_chain_bg.sh` | fresh 基线 + 4 臂 |
| 10-18 11:49 | 并行 | `temp/run_opt_evaloff_bg.sh` | on 臂 6 点无掩码评测（a01 13:08、a03 10-20 02:57） |
| 10-20 02:21 | 完成 | `logs/fed/opt_r30_fixed/*` | 4 臂 30/30 |
| 10-20 08:5x | 归档 | `checkpoints/opt_ablation_r30_fixed/` | 4 臂 + SHA256SUMS |
| 10-20 08:5x | 新增 | 本 ailog | 结果 + 分解 + 结论 |
| 10-20 08:5x | 修改 | `docs/weight-manifest.md` §7 | OPT 修复版归档 sha256 + 口径 |
