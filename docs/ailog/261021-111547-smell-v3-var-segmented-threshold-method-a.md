# Ailog 261021-111547 — SMELL-v3：VAR per-sample **Method A 分段阈值**（min D=4）交付方案

> **本文件由 fork 分支交付，只写方案，不改代码。** 当前 working tree 里仍有一版
> per-sample 实现（D=2..16、停止准则仍是旧的 `var_norm`），**不能直接用于本方案**；
> 实现 agent 需按本文修改。
>
> 承接：`261021-095531`（per-sample 拍板）、`261020-162000`（VQ per-result）。
> 本文修订两个关键点：
> 1. **停止统计量改用 raw var**（FwdLLM half-mean var），不再使用两种失败的 `var_norm`；
> 2. **D 低点从 2 上调到 4**，并用 **Method A：每 10 轮轻量探针重选 T_r** 处理 raw var 漂移。

## 一、结论先行

| 项 | 决定 |
|---|---|
| var 统计 | **逐 sample** 的 FwdLLM half-mean **raw var** |
| 停止准则 | `raw_var <= T_r`；不使用 `var_norm` |
| D 范围 | **D=4,6,8,10,12,14,16**（低点 4，高点 16） |
| patience | **无**（第一次满足即停） |
| L | 固定 **8** |
| 目标 | **mean D=8**（mean total samples=64，与 L8D8 基线对齐） |
| 更新 schedule | 逐 sample 采样 → 立即更新 → 下一个 sample |
| 阈值 | **Method A**：r0 全量标定 T0；r10/r20 各跑 8-client 探针重选 T10/T20；分段使用 |
| 对照 | BERT a03、CATV off、VQ off；基线 r29 **1.588699 / acc 0.466** |

## 二、为什么不用两种 `var_norm`

### 2.1 fork 公式：恒 ~1

```text
var_norm = mean((a-b)^2) / (mean(a^2) + mean(b^2) + eps)
```

BERT 16k、单 sample、D=2..16 实测：

| D | raw var | fork `var_norm` |
|---|---|---|
| 2 | 0.007466 | 1.0065 |
| 4 | 0.013926 | 0.9981 |
| 8 | 0.004120 | 1.0012 |
| 16 | 0.004741 | 1.0015 |

分母是**半均值向量**的能量，和分子同量级；独立噪声下 `E||a-b||² ≈ E||a||²+E||b||²`，比值恒 ~1。它不随 D 下降，无法选阈值。

### 2.2 energy 归一化：≈2/D，和 sample 无关

```text
var_norm_E = raw_var / mean_t ||c_t||^2
```

同一组 16k 数据：

| D | raw var | `raw/E` |
|---|---|---|
| 2 | 0.007466 | 1.0065 |
| 4 | 0.013926 | 0.4987 |
| 8 | 0.004120 | 0.2506 |
| 16 | 0.004741 | 0.1253 |

它几乎就是 `2/D`：sample 自己的噪声能量被约掉，固定 D 下几乎没有 sample 区分度。**它能选 D，但不能选“难 sample”**，因此不作为停止准则。

### 2.3 结论

- **raw var** 保留 sample 区分度（依赖 `trΣ`），但绝对尺度会随训练漂移；
- 两种归一化都把这个区分度消掉了；
- 因此用 **raw var + 分段阈值**：用 r0/r10/r20 的探针重新对齐尺度，而不是做 per-sample 归一化。

## 三、raw var 漂移有多严重（来自已完成的 pool run）

固定阈值 `T0=0.16417361795902252`（r0 标定 mean total samples=64）的正式 pool run：

| round | mean total samples | 相对目标 64 | hit 256 clients |
|---|---|---|---|
| r0 | 60.5 | ~1.0× | 3/30 |
| r5 | 40.0 | 0.63× | 1/30 |
| r10 | 57.9 | 0.90× | 1/30 |
| r15 | 87.5 | 1.37× | 2/30 |
| r20 | 105.9 | 1.65× | 3/30 |
| r25 | 139.5 | 2.18× | 4/30 |
| r29 | 160.5 | 2.51× | 7/30 |

- **r0–r10 基本可用；r10 后明显上漂**，r29 平均样本量约 2.5×。
- 用 r0 的 30 条 history 反推“保持 mean 64 所需阈值”：
  - r10 ≈ 0.138（0.84×）
  - r20 ≈ 0.347（2.11×）
  - r29 ≈ 0.667（4.06×）
  - 这是粗略缩放估计；早期 r5/r10 非单调，说明样本/数据构成有差异。
- tail 很重：hit 256 的 client，其当轮 raw var 与 r0 同 client n=256 的比值中位数约 **14×**，p90 约 **10^4**。少数 client 会把 cap 打满；单纯调一个全局阈值只能追平均，追不上尾部。

> 注意：以上是 **pool 级、cap 256** 的数据，不是 per-sample D=4..16；per-sample 的漂移需要 r10/r20 探针实测。但方向明确：**固定阈值不能跑完 r30**。

## 四、Method A：每 10 轮探针 + 分段阈值

### 4.1 时间线

| 段 | 轮次 | 阈值来源 |
|---|---|---|
| 段 0 | r0–r9 | r0 全量标定 T0 |
| 段 1 | r10–r19 | r10 探针重选 T10 |
| 段 2 | r20–r29 | r20 探针重选 T20 |

### 4.2 r0 全量标定

- BERT a03、CATV off、VQ off、L=8、LoRA r=1、16k。
- 30 clients × 8 samples = **240 条 per-sample history**。
- 每个 sample 跑到 D=16，记录 `D=4,6,…,16` 的 **raw var**。
- 选 `T0`：对候选阈值 T，停止 D = 第一个满足 `raw<=T` 的偶数 D（4..16），无则 16；
  计算 240 条 mean D；取 **mean D ≤ 8 且最接近 8** 的 T。
- 记录：D 分布、hit_low(D=4)、hit_high(D=16)、raw var 分位数、阈值 bracket。

成本：30×8×16×2 = 7,680 forwards ≈ **26–30 min**（16k，4090）。

### 4.3 r10 / r20 轻量探针

- 在当前 global model 上，固定取 **8 clients × 8 samples**（建议 `client_00..client_07`，跨段一致，便于比较）。
- 当前 round 的数据游标（`start=round*local_steps`，cycle）；**不更新模型**，只采样。
- 每个 sample 跑到 D=16，记录 D=4..16 的 raw var；64 条 history。
- 选 `T10` / `T20`，规则同 r0（mean D ≤8 且最接近 8）。
- 探针结果写 `var_probe_round010.jsonl` / `var_probe_round020.jsonl`；阈值写 `var_thresholds.json`
  （字段：round、T、mean D、median D、hit_low、hit_high、bracket）。
- 探针完成后，后续 10 轮使用新 T；模型状态不变。

成本：8×8×16×2 = 2,048 forwards ≈ **7 min/次**；r10+r20 约 **14 min**。

### 4.4 阈值选择细节

- 候选 T 取探针中出现的所有 raw var 值（或加 bracket 中点）。
- 停止 D 只在偶数 D 检查；D=4 是低点。
- 选 mean D **≤8 且最接近 8** 的阈值；若多个，取较小阈值（更保守、D 更大）。
- 若探针的 mean D 在最低/最高 T 下都 >8（说明 D=4..16 的分布无法达到），
  记录 `unreachable`，保留上一段 T，并在报告里显式标注。
- 报告 bracket：`T_low -> mean>8`、`T_high -> mean<=8`。

## 五、per-sample 算法（min D=4）

对每个 local sample（共 L=8，逐 sample 顺序）：

```text
pool = []
for D = 4, 6, 8, 10, 12, 14, 16:
    采 2 个方向（D=4 时先采 4 个），加入 pool
    a = mean(pool[:D/2])
    b = mean(pool[D/2:])
    raw = mean((a-b)^2) / 2          # FwdLLM var，逐元素 mean
    if raw <= T_r:
        break
    if D == 16:
        break                         # hit_high
g = mean(pool)                        # D 个方向的平均梯度
update(model, g)                      # 立即更新
record D, raw, hit_low(D=4), hit_high(D=16)
```

- 每个 sample 的 pool 只含自己的方向，不跨 sample。
- 逐 sample 更新，与基线 L8D8 的顺序更新一致。
- **D=4 是 2 vs 2**：比 D=2 的 1 vs 1 稳一些，但仍然噪声大；no patience。
- 如果 hit_low(D=4) 过多，回来看 min D=6 / patience=2，不在本次先改。

## 六、CLI / metrics

主接口保持两个：

```bash
--var-control {on,off}     # 默认 off
--var-threshold FLOAT      # 默认 = r0 标定 T0；r10/r20 由探针更新
```

- `--var-diagnose`：r0 全量标定用；正式训练不暴露。
- `--var-segment-rounds 10`（或硬编码 10）：Method A 的分段长度；只作为内部/可选参数。
- 每 sample 记录：`D`、`raw_var`、`hit_low`、`hit_high`、当前 `T_r`。
- 每轮聚合：`D_mean/min/max`、`total_samples_mean`（目标 64）、
  `hit_low_samples`、`hit_high_samples`、`raw_var_mean`、`delta_norm`、`clipped`、
  当前 `var_threshold`。
- 探针文件：`var_probe_round010.jsonl`、`var_probe_round020.jsonl`、`var_thresholds.json`。
- **不要**把 `var_norm` 放进停止路径；如保留，仅作对照列。

## 七、正式实验

- BERT a03、CATV off、VQ off、L=8、LoRA r=1、lr=1.4e-3、clip=1.0、cycle、r30、eval 500。
- r0 全量标定 T0；r10/r20 探针 T10/T20；段内固定。
- 基线：a03 off L8D8，r29 **NLL 1.588699 / acc 0.466**。
- 报告：r29 NLL/acc、r0→r29 曲线、逐样本配对、`D_mean`、total samples、hit_low/high、
  D 分布、raw var 均值、delta norm/clipped、每段 T 和探针 bracket。

## 八、风险 / 待确认

1. **探针只有 8 clients × 8 samples**：T_r 有噪声；如果 r10 探针的 bracket 很宽或 mean D 跳变 >2，
   下次把探针扩到 12–16 clients，或把段长缩到 5 轮。
2. **段内漂移**：每 10 轮调一次仍可能在段末偏；报告段末实际 D_mean，若 >10 考虑缩短段长。
3. **D=4 仍噪声**：2 vs 2 的单次判断可能早停/晚停；hit_low 是核心诊断。
4. **raw var 绝对尺度**：分段阈值是经验修正，不是理论上界；换模型/层数/精度需重标。
5. **CATV/VQ**：本次均 off；若组合，CATV 的 vote 权重和 VQ 的 C 会随 per-sample D 变化，需重做。
6. **探针不更新模型**：必须在 eval 模式、用 `zo_grad` 的参数还原保证 global state 不变；
   探针本身不得触发 CATV vote 回调。

## 九、实现 agent 需要做的事（本 fork 不执行）

1. `ClientRunner`：把 per-sample 路径的低点从 2 改成 4，停止准则改为 **raw var**；
   `D=4,6,…,16`，逐 sample 立即更新；记录 D/raw/hit_low/hit_high。
2. `run_fed`：r0 全量 diagnose（30×8×D=4..16）；r10/r20 在 round 开始前跑 8-client 探针；
   探针选 T_r，更新当前段阈值；写 probe JSONL / thresholds JSON。
3. 阈值选择脚本：raw var，mean D≤8 最接近 8；报告 bracket/D 分布/hit。
4. Metrics：D_mean/min/max、total_samples_mean、hit_low/high、raw_var_mean、当前 T。
5. 不要使用 fork `var_norm` 或 energy `var_norm` 作为停止准则；可保留作对照。

## 十、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 11:15 | 新增 | 本 ailog | Method A 分段阈值 + min D=4 的交付方案；不改代码 |
| 11:15 | 修订 | 方案 | 停止统计量 raw var；放弃两种 var_norm；D 低点 2→4 |

## 十一、实现落地（261021-1155，用户选 A）

> 本节由实现 agent 追加。用户明确选择 **A：Method A 原样（min D=4、no patience、raw var）**。

- 代码已改为 per-sample raw var：`D=4..16`，逐 sample 立即更新；停止条件 `raw_var <= T_r`。
- r0 T0 复用 240 条 per-sample history 的 raw 曲线（D=4..16）：
  **T0 = 0.587230920791626**；mean D=7.967、median 4、
  hit_low(D=4)=123/240、hit_high(D=16)=59/240。
- Method A 分段实现：r10/r20 各 8 clients × 8 samples 探针（不更新模型），
  选 T10/T20；写 `var_probe_roundNNN.jsonl` + `var_thresholds.json`。
- 正式 run：BERT a03、CATV off、VQ off、L=8、r30、eval 500；
  `logs/fed/bert_var_methoda_a03_r30/`，归档 `checkpoints/distilbert_var_methoda_a03_r30/`。
- 注意：此前 `261021-095531` 里追加的 energy-norm 修正已被本方案取代；energy norm 只保留为对照记录。
