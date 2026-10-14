# Ailog 261014-143500 — SMELL-v3 DistilBERT：CATV 投票 padding 修复落地 + block0 强保 + 复现/冒烟全过

> 承接 handoff `261014-111800`（v4 规格：D1–D5）。本文档 = 实现记录 + 验收证据（§4.3 全项）+ 归档状态 + 后续。
> 修复范围：§4.1 P0a（逐块有效均值/NaN 载荷）+ §4.1.3（预算 51/51 不变）+ §4.2（block0 强制保留，BERT 特供）。
> **未触碰运行中任务**（a03_catv_on 缺陷臂仍在跑，D4 停机由用户处理）。

## 一、实现（代码级）

| 文件 | 改动 | 规格 |
|---|---|---|
| `src/models/modeling_distilbert_smell.py` | ① 投票载荷：pad 块置 `NaN` 后上报（topk/共识仍用 `finfo.min` 占位）；② `keep_block0`（默认 True）在 topk+共识掩码之后强制保留 block0（挤掉已入选最低分者，block0 自身不可丢）；③ `last_prune_stats` 增 `block0_kept/block0_forced`；④ `patch_distilbert_sdpa_prune` 显式写 `config.keep_block0` | §4.1.1、§4.2 |
| `src/fed/serial_fedavg.py` | `_VoteAccumulator` 改逐块有效均值：`nansum` + 逐块 valid 计数向量；零覆盖 → 0 + WARN；无 NaN 输入与旧实现逐位等价 | §4.1.1、§4.1.3 |
| `src/fed/run_fed.py` | run `config.json` 增 `keep_block0` 字段（复现） | §4.2 |

P0b 覆盖资格按 v4 决定取消（数据：最坏块有效样本 83/240 ⇒ 均值可用）；P0c 预算退化为 `k=floor(0.2×256)=51` 不变；P1 核查为已实现（不改）。

## 二、验收证据（§4.3 全项）

### 2.1 单测（CPU，全过）
- `serial_fedavg.py`：**6/6 PASS**（新增「vote valid mean per block」：`[[1,2,NaN],[3,NaN,NaN]]+[[5,4,NaN]] → [3,3,0]`；「vote finite equivalence」：无 NaN 与旧标量均值逐位等价）。
- `modeling_distilbert_smell.py`：**19/19 PASS**（新增「vote payload NaN on pad blocks」：短文档 block1 全 pad → NaN、block0 有限；「block0 forced under -inf mask」；「block0 not forced when disabled」）。
- `py_compile` 3 文件 OK。

### 2.2 复现测试（`temp/bert_catv_fix_repro.py`，真实 a01 round-0 240 篇，16k）
| 项 | 结果 |
|---|---|
| old-style vs **归档掩码**（`r30_a01_catv_on/consensus_mask_round0.pt`） | 三层 anchor_in/out **Jaccard = 1.000**（假象机制完全复现） |
| new-style vs old-style | anchor_in J = 0.010/0.000/0.000；新锚点分数驱动（如 layer3 `[85..]`、layer4 `[107..]`），不再恒为前缀 |
| block0 强保 | 三层（2/3/4）30/30 forwards `block0_kept=True` |
| 有效覆盖（逐块 valid 计数） | 三层 min=85 / max=240 / 零覆盖块 = 0（与 handoff 预测 83/240 一致） |
| 产物 | `temp/bert_catv_fix_repro.json`；日志 `temp/logs/bert_fix_repro.log` |

### 2.3 端到端冒烟（`temp/bert_fix_smoke.sh`，2 clients × 2 轮，catv on，16k）
- votes 非冻结：r0 `+inf` 至 173、r1 至 255，两轮掩码不同（`ROUND0_EQ_ROUND1=False`）；`ir_mean` 0.873 → 0.938；
- 零覆盖兜底 WARN 生效（小样本 16 篇下尾部块 0 + WARN，未污染排序）；eval 2/2 ok（acc 0.594/0.562，n=32）；
- 结论 `SMOKE_PASS: True`；哨标 `temp/fix_smoke/a01/consensus_mask_round{0,1}.pt`。

## 三、缺陷臂归档（D1/D4：仅存档，禁用于论文）

`checkpoints/distilbert_ablation_r30_defect/`（含缺陷 `README.md` + 每臂 `SHA256SUMS`）；**a03_on 由用户停机且无产物 → 归档定格 3/4 臂**：

| 臂 | own ΔNLL%（30 轮，缺陷口径） | adapter sha256（前缀） |
|---|---|---|
| r30_a01_catv_off | +8.29% | `b43978ee…` |
| r30_a01_catv_on | +1.72%（含前缀假象注水） | `d3be4e4e…` |
| r30_a03_catv_off | +11.86% | `5160c96d…` |
| r30_a03_catv_on | **空产物**：14:21:58 启动后被用户停机（D4；rc 记账为 0，无 metrics/无 adapter） | — |

`docs/weight-manifest.md` 已新增 §6（BERT 线产物 sha256 + 即用运行口径）。

## 四、后续

1. **修复版 4 臂 r30**（α×CATV，L8D8/lr=1.4e-3/cycle/eval500）：脚本 `temp/run_bert_da_fixed_chain_bg.sh`（含逐臂自动归档 `checkpoints/distilbert_ablation_r30_fixed/` + 自动报告）。
   - 14:36 起：挂「用户 OPT 补评任务（run_d32_r7r8）结束 → 自动启动本链」衔接器（避免 GPU 争用）。
2. 修复后的预期：on 臂起点不再有前缀假象注水（缺陷 on 臂 r0 NLL 1.5424 vs off 臂 1.8060 的差将收缩），CATV 对照变干净。

## 五、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 14:20 | 修改 | `src/models/modeling_distilbert_smell.py` | vote NaN 载荷 + block0 强保 + 统计 + 单测（19 项） |
| 14:20 | 修改 | `src/fed/serial_fedavg.py` | `_VoteAccumulator` 有效均值 + 单测（6 项） |
| 14:21 | 修改 | `src/fed/run_fed.py` | config.json 记录 keep_block0 |
| 14:23 | 归档 | `checkpoints/distilbert_ablation_r30_defect/` | 缺陷臂 3/4 已归档（a03_on 跑完补） |
| 14:30 | 修改 | `docs/weight-manifest.md` | 新增 §6 BERT 线产物与口径 |
| 14:35 | 新增 | `temp/bert_catv_fix_repro.py`、`temp/bert_fix_smoke.sh`、`temp/run_bert_da_fixed_chain_bg.sh`、`temp/archive_bert_r30_defect.sh` | 复现/冒烟/修复版链/归档脚本（temp/ 不入 git） |
