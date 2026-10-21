# Ailog 261021-095531 — SMELL-v3：VAR 控制讨论、schedule 混杂与 **per-sample VAR 拍板方案**

> 承接 `261020-162000`（VQ per-result）与 261021 的 VAR 标定/实验。本文记录：
> ① 当前 pool 级 VAR 为什么没有带来收益；② 讨论过的备选方案（保留意见）；
> ③ **用户当前拍板、供其他 agent 执行的方案**：逐 sample 测 var，mean D=8，low=2，high=16，no patience。

## 一、已有结果（截至 261021 09:55）

### 1.1 VQ per-result（已完成，USE）

- BERT a03、CATV off、VQ rate=0.5、bf16、per-result `c_dim=rate*d`、r30。
- r29 NLL **1.587080** / acc **0.466**；同轮无 VQ a03 off = 1.588699 / 0.466；
  逐样本配对 ΔNLL = **−0.001618**（t=−2.416，p=0.0157，n=500）。
- `vq_cos_mean` 全程 0.9945；单条结果 2× 字节压缩，总量 32× 一条 delta。
- 归档：`checkpoints/distilbert_vq050_bf16_perresult_a03_r30/`；详见 `docs/weight-manifest.md` §8。

### 1.2 旧 pool 级 VAR 标定（已做，但不是本次拍板方案）

- 结构：固定模型，轮询 L=8 个样本；每个 pass 每样本 1 方向；pool 8..256；
  half-mean var；目标 mean total samples=64。
- 选中阈值 **0.16417361795902252**；标定 mean total=64、median=32、min=8、max=256；
  hit 8=2/30、hit 256=3/30。
- var 重尾：n=8 时 mean 586 / median 0.563 / max 17,583；n=64 mean 171 / median 0.102；
  n=256 mean 34.6 / median 0.0275。
- cap 128 的离线重算（raw var）：T=0.1087283，mean 64、median 60、hit8=2、hit128=6。
- cap 192：T=0.1363915，mean 64、median 40、hit192=3。

### 1.3 旧 pool 级 VAR ON r30（诊断用，不是最终方案）

- 配置：BERT a03、CATV off、VQ off、L=8、threshold=0.16417、r30。
- 截至 09:55：r27 训练完成，r26 已 eval；r27–r29 仍在跑。
- r25：NLL **1.608264** / acc 0.460 vs 基线 1.601956 / 0.464；ΔNLL **+0.006308**，
  t=5.08，p=3.9e-7。
- r26：NLL 1.604096 / acc 0.464 vs 基线 1.599048 / 0.466。
- 实际平均 total samples = **84.7**（目标 64）；r26 167.5、r27 148.3；
  27 轮共 810 client-轮，hit 8=51、hit 256=72。
- 单轮均值 1015.8s；最近 27–33 min/轮（基线 ~13 min）。
- 结论：该 run 仅作诊断；其 NLL 小幅退化不能直接归因于 var（见 §二）。

## 二、为什么 pool 级 VAR 没有收益（讨论结论）

1. **schedule 混杂（最大嫌疑）**：旧实现是“固定初始模型轮询 8 个样本 → 达到 var → 每个样本用跨 pass 平均梯度 → 连续更新 8 次”。基线是“每个 sample 采样 D=8 → 立即更新 → 下一个 sample”。后几个 sample 的梯度是旧模型的；增加 pass 只降方差，不消除 stale gradient。两者不是只差一个 var 开关。
2. **var 不是 update error**：pool 级 var 在 n=8 时是 batch 1–4 vs 5–8（数据异质）；n≥16 才是 early vs late passes。var 下降不等于梯度偏差下降；有限差分偏差、LoRA 子空间、local update 偏差不会被更多方向样本消除。
3. **额外样本集中在外点**：r0/r20/r25/r26 被 `delta-clip` 的 client 全部是 256 samples；`corr(samples, delta_norm)` 0.30–0.77；72/810 client-轮 hit 256。平均 84 是“少数 256 + 多数 8–32”的混合，不是均匀提升。
4. **有效步长/阈值漂移**：多方向平均会压低噪声梯度 norm，`delta-clip` 只向下裁、不 rescale up；阈值 0.164 是 r0 的 mean=64，训练中 var 上移，实际 mean 84.7、r26 167，controller 已在偏离设计点工作。

## 三、备选方案（保留意见，不采用）

### 3.1 静态 per-client D allocation（本文作者建议，保留）

- 不改 local schedule：仍逐 sample 采样 → 立即更新；L=8。
- 用 round-0 的 pooled var(D) 曲线离线给每个 client 分配固定 `D_i`（`var_norm_i(D_i)<=T`，cap 16），
  选 T 使 mean `D_i`=8；`D_i` 固定用于 r30。
- 优点：完全保留基线 schedule，不引入 stale gradient；CATV/VQ 路径不变（本次都 off）；均值 64，与基线同预算。
- 缺点：静态分配，不随训练漂移；round 内不重采样；VQ 若开启需 per-client C。
- 结论：作为保留方案；本次用户选择 per-sample VAR。

### 3.2 同 schedule 的 VAR OFF / shuffled-D 对照

- 同“固定模型轮询 + 采样后更新”的 VAR OFF，或把 `D_i` 随机打乱分配给 client；
  用于隔离 schedule 与分配效应。计算成本各 +7h；本次不排入。

### 3.3 归一化（已拍板启用）

- **归一化启用**：`var_norm = mean((a-b)^2) / (mean(a^2)+mean(b^2)+eps)`；
  无量纲、随 D 增大下降，比绝对 var 更不容易随训练尺度漂移。
- per-sample 标定**用 var_norm 选阈值**；raw var 只作为对照记录，不作为默认停止准则。

## 四、**用户拍板方案（其他 agent 以此为准）**

### 4.1 核心决定

| 项 | 决定 |
|---|---|
| var 统计 | **逐 sample** 计算；每个 sample 的 pool 只含该 sample 自己的方向贡献 |
| 归一化 | **启用**：`var_norm = mean((a-b)^2)/(mean(a^2)+mean(b^2)+eps)`；停止准则用 var_norm，raw var 仅记录对照 |
| 目标 | **mean D = 8**（每个 sample 平均 8 个方向；L=8 → mean total samples=64） |
| D 低点 | **2** |
| D 高点 | **16**（L=8 → 单 client 单轮最多 128 samples） |
| patience | **先不考虑**（单次 var <= threshold 即停） |
| 数据异质 | **在本设计下不影响**：不同 sample 的贡献不进同一个 var pool |
| 更新 schedule | **逐 sample 采样 → 立即更新 → 下一个 sample**；与基线顺序更新一致，可行 |
| 基线 | 不改；`--var-control off` 时仍为固定 D=8 的原路径 |

> **实现责任**：本文只记录用户拍板的方案；代码修改由后续实现 agent 执行。
> 当前仓库代码仍是 pool 级 VAR（`_run_zoo_var_pool`），**不能直接用于本 per-sample 实验**。

### 4.2 逐 sample 流程

对 client 的每个 local sample（共 L=8）：

1. 从 `D=2` 开始，在当前模型上采 2 个方向，算 per-sample `var_norm`；
2. 若 `var_norm <= threshold`，用这 D 个方向的平均梯度**立即更新**，进入下一个 sample；
3. 否则每次增加方向（只在偶数 D 检查：2,4,6,…,16），直到 `var_norm <= threshold` 或 `D=16`；
4. `D=16` 仍未达标则强制接受，记录 `hit_high`；`D=2` 即停记录 `hit_low`。

- 每个 sample 的 var 只用该 sample 的贡献，所以数据异质性不会进入 var。
- 每个 sample 更新后模型状态变化，下一个 sample 在新模型上采样；与基线一致。

### 4.3 标定（per-sample diagnose）

- BERT a03、CATV off、VQ off、L=8、LoRA r=1、16k。
- 对每个 sample 跑到 `D=16`，记录 `var_raw(D)` 与 `var_norm(D)`（D=2,4,…,16）；
  30 clients × 8 samples = **240 条 per-sample history**。
- **用 var_norm 选 threshold**，使 **mean stopping D = 8**（即 mean total samples=64）；
  同时记录 hit_low(D=2)、hit_high(D=16)、D 分布、var 分位数。
- 预计成本：30 clients × 8 samples × 16 dirs × 2 forwards ≈ 7,680 forwards，
  约 **26 min**（cap 128）。

### 4.4 CLI / metrics

```bash
--var-control {on,off}   # 默认 off
--var-threshold FLOAT    # 默认值 = 本次 per-sample 标定结果
```

- `--var-diagnose` 仅标定用；正式训练不暴露。
- 每 sample 记录：`D_used`、`var_final`、`hit_low`、`hit_high`；
- 每轮聚合：`D_mean/min/max`、`total_samples_mean`、`hit_low/high` 计数、`var_final`、
  delta norm、clipped。
- 必须保留 `hit_low/hit_high` 计数，便于极端值多的分析。

### 4.5 实验

- 正式：BERT a03、CATV off、VQ off、L=8、r30、eval 500、threshold=标定值。
- 对照：无 VQ/无 VAR 的 a03 off 基线（L8D8，r29 1.588699/0.466）。
- 因为 local schedule 与基线一致，CATV/VQ 模块不需要重测；本次两者均 off。
- 若以后 VAR 与 CATV/VQ 组合：
  - CATV：D 逐 sample 变化，投票的 forward 数/权重会变，需单独验证；
  - VQ：`C=L*D` 会变成 per-sample/per-client 可变，需重做 C 与聚合口径。

## 五、风险 / 待确认（开工前已知）

1. **D=2 的 var 噪声极大（用户已知悉）**：只有 1 vs 1（前一半/后一半各 1 个方向），
   单次判断容易撞低点或高点；no patience 会放大这一点。本次先按拍板跑标定，
   **看 hit_low(D=2) 多不多**再决定；若 hit_low 过多或 mean=8 不可达，再回来讨论
   min D=4 / patience=2。
2. **阈值跨轮漂移**：per-sample 标定在 round-0 做；训练后段 var 可能上移。
   **已启用 var_norm**；仍建议在 r0/r10 做小样本复核；若漂移明显，再加每 N 轮重标定。
3. **当前代码仍是 pool 级**：`ClientRunner._run_zoo_var_pool` 是“固定模型轮询 + 采样后更新”，
   **不是本次拍板方案**；实现时必须改成 per-sample 顺序更新。`zo_grad` 与
   `half_mean_var` 可复用。旧 pool VAR ON 结果只作诊断，不能当最终 VAR 结果。
4. **当前 pool VAR ON r30 尚未结束**（r27–r29）；其 r29 结果仅用于解释 schedule 问题，
   不作为 per-sample 方案的基线。

## 六、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 09:55 | 新增 | 本 ailog | 记录讨论、备选方案、per-sample 拍板方案 |
| 10:00 | 修订拍板 | 本 ailog | **归一化启用**（var_norm 选阈值）；D=2 噪声用户已知悉，先看 hit_low；代码由实现 agent 改 |
| 09:55 | 待办 | `src/fed/serial_fedavg.py` 等 | 实现 per-sample VAR（当前仍是 pool 级；本 agent 不执行） |
| 09:55 | 待办 | BERT a03 标定 | per-sample var(D)，目标 mean D=8，cap 16 |

## 七、下一步（串行）

1. 等当前 pool VAR ON r30 结束（仅诊断，r29 归档）；
2. 实现 per-sample VAR：L=8、D=2..16、逐 sample 更新、no patience；
3. 跑 per-sample 标定（~26 min），**用 var_norm** 选 threshold 使 mean D=8；
4. 跑 BERT a03、CATV off、VQ off、VAR ON r30；
5. 与 a03 off 基线（1.588699/0.466）做 r29 配对；报告 D 分布、hit_low/hit_high、NLL/acc。

## 八、261021-1143 实测修正（per-sample 打点，已执行）

> **已被 `261021-111547` Method A 取代**：用户最终选 A（raw var、min D=4、no patience）；
> 本节的 energy norm 只作历史对照，不是当前停止准则。
> 本节由实现 agent 追加；不改上文拍板记录，只记录实测结果与必要修正。

- per-sample 代码已实现：`ClientRunner._run_zoo_var_per_sample`，L=8、D=2..16、逐 sample 立即更新、no patience。
- 标定：30 clients × 8 samples = **240 条 per-sample history**，`temp/var_diag_bert_a03_persample/`，11:43 完成。
- **fork `var_norm` 实测不可用**：`mean((a-b)^2)/(mean(a^2)+mean(b^2))` 在 D=2..16 恒为 **0.9996–1.0001**（median 也 ~1.0），没有随 D 下降，无法卡 mean D=8。
- **改用 energy 归一化**：`var_norm = raw_var / mean_t ||c_t||^2`，实测严格 ~2/D：
  D=2 1.0003 / 4 0.5000 / 6 0.3332 / 8 0.2499 / 10 0.2001 / 12 0.1667 / 14 0.1429 / 16 0.1250。
- **选中阈值 T=0.2537100843**：mean D=**8.000**、median D=8、hit_low(D=2)=0、hit_high(D=16)=0。
- raw var 重尾且 D=2→4 会反弹（median 0.889→0.530，max 到 9.2e8），不适合作为停止准则；mean-norm 归一化恒 ~2.0，也不可用。
- CLI：`--var-control {on,off}` 默认 off；`--var-threshold` 默认 **0.2537100843**（energy）；`--var-diagnose` 仅标定用。
- 正式 run 于 11:44:16 启动：BERT a03、CATV off、VQ off、L=8、r30、eval 500；`logs/fed/bert_var_persample_a03_r30/`；归档 `checkpoints/distilbert_var_persample_a03_r30/`。

### 变更列表（追加）

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 11:43 | 完成 | per-sample diagnose | 240 histories；fork norm 恒 ~1；energy norm ~2/D |
| 11:44 | 修改 | `src/fed/serial_fedavg.py` | 停止准则改为 energy norm；逐 sample 更新 |
| 11:44 | 修改 | `src/fed/run_fed.py` | 默认阈值 0.2537100843；D/hit 指标 |
| 11:44 | 启动 | `bert_var_persample_a03` | per-sample VAR ON r30 |
