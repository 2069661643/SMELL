# Ailog 261024-112000 — 补记：FwdLLM 稠密（sparse=1.0）VAR r30 已完成（此前漏文档）

> 该 run 于 10/22 21:48:46 → 10/23 10:25:32 完成并归档，但**没有 ailog、weight-manifest 也没收录**（本次补记）。
> 归档：`checkpoints/distilbert_fwdllm_a03_var_r30/`；adapter sha `8f9435fa…`。

## 一、配置

| 项 | 值 |
|---|---|
| 场景 | FwdLLM 原教旨：`--sparse 1.0`（`sdpa_prune` 走 dense fallback，全部块保留） |
| 模型/数据 | DistilBERT、a03、16k、LoRA r1/α2、fp32、cycle、eval 500 |
| VAR | Method A（raw var、D=4..16、no patience）：T0=0.5872309208、T10=1.477083、T20=3.225293 |
| VQ / CATV | 都 off（`vq_rate=1.0`、`catv=off`） |
| 其他 | L8、D8、lr 1.4e-3、clip 1.0、r30 |

## 二、结果

| 指标 | 值 |
|---|---|
| r0 | **1.592325 / acc 0.454** |
| r29 | **1.493650 / acc 0.472** |
| own 降幅 | **−6.20%** |
| 轮速 | **1346s/轮**（sparse 0.4 为 849s，1.59×） |
| 总时长 | 11.39h 训练（21:48→10:25 墙钟） |
| VAR | mean D **9.108**、samples **72.87**；hit_low 32/64、hit_high 16/64 |

## 三、与 sparse 0.4 对照

| 臂 | r0 NLL/acc | r29 NLL/acc | own | s/轮 |
|---|---|---|---|---|
| **dense（sparse=1.0）+ VAR** | 1.592325/.454 | **1.493650/.472** | −6.20% | 1346 |
| sparse 0.4 + VAR（Method A） | 1.690642/.426 | 1.584451/.464 | −6.28% | 849 |
| sparse 0.4、VQ off 基线（无 VAR） | 1.690632/.426 | 1.588699/.466 | −6.03% | 772 |

- dense vs sparse VAR @r29：**−0.090801 nats（−5.73%）、acc +0.8pt**；
- dense vs sparse 无 VAR 基线：−0.095048。
- **caveat**：dense 的 r0 起点低 0.098 nats（注意力全保留），跨稀疏度**不能直接比终点绝对值**；own% 接近（6.20% vs 6.28%）。
- dense 下 VAR 阈值尺度更大（T10 1.477 vs 0.612、T20 3.225 vs 2.210），说明 raw var 绝对值依赖稀疏度/精度，**阈值不可跨配置复用**。

## 四、未做

- dense 续训 r30→r59（原“插入 B”计划，随 A/B/C 队列停机**未启动**）；
- dense + VQ、dense + CATV 组合均未测；
- eval-off 不适用：CATV off，评测本身就是无掩码口径。

## 五、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10/23 10:25 | 完成 | `logs/fed/bert_fwdllm_a03_var_r30/` | 30/30 + 归档 |
| 10/24 11:2x | 补档 | `docs/weight-manifest.md` §11 | 配置/结果/sha/未测项 |
| 10/24 11:2x | 补档 | 本 ailog | 漏文档补记 |
| 10/24 11:2x | 补拷 | 归档内 var_thresholds/probe | 与其它 VAR 归档一致 |
