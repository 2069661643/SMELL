# Ailog 261014-111800 — SMELL-v3 DistilBERT：CATV 投票 padding 假象（掩码恒为 {0..50}/{205..255}）确认 + 修复交接

> 交接对象：修复 agent（用户指派）。本文档 = 缺陷确认 + 复现证据 + 修复规格 + 当前实验状态/注意事项。
> 触发问题：BERT CATV 掩码 30 轮完全静止（±inf 锚 Jaccard=1.000），锚恒为 blocks 0–50、反锚 205–255。
> 结论：**不是模型共识，是投票聚合里 padding 占位符（`float32 min = -3.4e38`）泄漏导致的排序假象**。

## 1. 缺陷机理（代码级，已逐条实证）

1. `src/models/modeling_distilbert_smell.py::DistilBertSdpaPruneAttention.forward`：
   ```python
   scores = scores.float().sum(dim=-2)
   block_keep = keep_mask.view(bsz, n_blocks, pool).amax(dim=-1)
   scores = scores.masked_fill(block_keep < 0.5, torch.finfo(scores.dtype).min)  # ← pad 块 = -3.4e38
   vote_callback(self.layer_idx, scores.detach())                               # ← 占位符随分数进入投票
   ```
   即 **pad 排除先于投票**，被 pad 的块以 −3.4e38 参与聚合。
2. `src/fed/serial_fedavg.py::_VoteAccumulator`：对每层累计 `values.sum(dim=0)` 与批次数，最后取**均值**。
   −3.4e38 主导均值：某块在任一篇里被 pad 一次 → 均值 ≈ −1.4e37；≥2–3 次 → float32 **溢出成 −inf**。
   有效分（~1e3–1e4）与占位符相差 34 个数量级，**真实评分被完全淹没**。
3. 服务器端 `compute_consensus_mask`（`src/models/token_selector.py`）：
   `argsort(descending=True, stable=True)` → 顺序为 **[有效块按分数降序] → [−1.4e37 组：并列按索引升序] → [−inf 组：并列按索引升序]**；
   - 取前 51（锚）= 永不 pad 的**前缀**（≤51 块）+ 并列组最小索引补齐 → **恒等于 {0..50}**（与分数无关）；
   - 取后 51（反锚）= 最负组里索引最大的 51 个 → **恒等于 {205..255}**。
4. 触发条件 = **每轮投票的 240 篇（30 client × 8 篇）里存在至少 1 篇 < 51 块（3264 token）**。
   逐轮实测（cycle 采样，a01）：每轮最短文 = 1029–3238 token（16–51 块），**30/30 轮满足** → 掩码 30 轮 J=1.000。
   （若某轮全部 >3264 token，锚会变成"前缀内按分数取 51" → 可能漂移；本 run 未出现。）

## 2. 复现证据（可照抄核验）

- **精确复现**：用同轮真实 240 篇重算投票 + 同款排序 → r0、r1、三层**锚 51/51、反锚 51/51 完全一致**（与 `logs/fed/bert_da/r30_a01_catv_on/a01/consensus_mask_round{0,1}.pt` 对比）。
- **占位符污染实测**（24 篇模拟，layer2 逐块）：
  `block25 pad=0 avg=7.2e3 | block26 pad=0 avg=6.8e3 | block30 pad=1 avg=−1.4e37 | block255 pad=14 avg=−inf`
- **对照 OPT**：`logs/fed/a01_zoo_k24_d32_catv_lr0.30_r30` 的掩码**缓慢漂移**（+inf J=0.82–1.0、−inf J=0.79–0.96）——OPT 剪枝注意力的投票路径无 pad 补丁（其数据/实现无 pad 占位）→ 真·分数驱动。
- **效应侧证据**（r0，同权重；free=1.8060/0.3860）：
  | 评测选择 | NLL | acc |
  |---|---|---|
  | 只 +inf 锚（=保前 51 块） | **1.5367** | 0.4840（≈全部收益） |
  | 只 −inf 反锚 | 1.7203 | 0.4060 |
  | 只 +inf block0（[CLS] 块） | 1.6917 | 0.4240（占 42%） |
  | 完整掩码 | 1.5424 | 0.4740 |
- free 选择实测：layer4 在 6 个样本里 5 次丢掉 block0（[CLS] 块）→ 单保 block0 即回收 42% 收益。

## 3. 修复规格（建议，~10 行；**不得影响 OPT 路径**）

1. `modeling_distilbert_smell.py`（BERT 投票处）：给 callback 传**独立副本**，pad 位置置 NaN（topk 仍用 finfo.min 的 `scores`）：
   ```python
   vote_scores = scores.masked_fill(block_keep < 0.5, float("nan"))
   vote_callback(self.layer_idx, vote_scores.detach())
   ```
2. `serial_fedavg.py::_VoteAccumulator`：改为逐块"有效项均值"（NaN-aware）：
   - `__call__`：`valid = ~torch.isnan(values)`；`sums += torch.nansum(values, dim=0)`；`counts += valid.sum(dim=0)`（counts 改为逐块向量；某块全 pad → count=0）；
   - `average()`：`sums / counts.clamp(min=1)`；
   - OPT 路径无 NaN → 行为与旧版完全一致（等价性可加断言/自测）。
3. 备选（更显式但要改签名）：callback 传 `(scores, valid_mask)`；因 OPT 共用该回调，**不推荐**。
4. 自测建议：固定构造（2 篇文档：一篇 16k、一篇 1.7k）跑一个 round 的投票，断言掩码不再恒为 {0..50}/{205..255}，且逐样本 top-102 的锚行为符合分数。

## 4. 影响与论文口径（重要）
- 本 run 的 "CATV" 实质 = **固定头部前缀先验（{0..50}）+ [CLS] 保活**，**不是 client-voted consensus**；
- a01 的结论（同协议训练效应小正、free 口径迁移被截断）在"前缀先验"解释下依然成立，但**叙述必须改**（不能称 consensus），修复后需补一臂 "fixed-CATV" 对照；
- OPT 侧的 CATV 结论不受影响（其掩码是真分数驱动）。

## 5. 当前实验状态（修复 agent 勿动）
- `temp/run_bert_d_a_chain_bg.sh`（pid 文件 `temp/logs/last_bert_da.pid`）正在跑 **a03 双臂**：
  - `r30_a03_catv_off` 06:46 起；之后 `r30_a03_catv_on`；预计 10-14 ~22:10 出报告 `temp/logs/bert_r30_report.txt`。
  - **它们用的仍是含缺陷实现**（保持与 a01 双臂同口径）→ 修复后不要重跑这两臂，另排 "fixed-CATV"。
- a01 双臂结果（同口径 2×2 与轨迹）见 `docs/ailog/261012-183000-...` 与本文件 §2；原始：`logs/fed/bert_da/r30_a01_catv_{off,on}/`、交叉评测 `temp/eval_{off,on}{00,09,19,29}_*.json`、`temp/mask_*_only.pt`。

## 6. 变更列表
| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10-14 11:18 | 新增 | 本 ailog | 缺陷确认（含复现证据）+ 修复规格 + 状态交接 |
| （未改动任何代码） | — | — | 修复由后续 agent 执行 |
