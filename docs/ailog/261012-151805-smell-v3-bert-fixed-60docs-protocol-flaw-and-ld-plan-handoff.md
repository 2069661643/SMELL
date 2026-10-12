# Ailog 261012-151805 — SMELL-v3 DistilBERT：OPT 协议「固定 60 篇语料」缺陷确认 + 轮转接口与 L×D/rank/r30 执行计划（交接）

> 交接对象：下一个 Agent（用户会另行启动）。
> 用户立场（锁定）：**OPT 不重做**；轮转（逐轮换文档）是正确方向；怀疑当前 BERT 上观察到的
> 「L 效应」实为**语料太少/太重复**所致，而非 L 本身；论文层面另行解释。
> 本文档 = 错误确认 + 证据 + 接口设计 + 假设判定实验 + 完整执行计划 + 现有产物清单。
> 当前机器状态：**无任务在跑，GPU 空闲**（2026-10-12 15:18 核实）。

---

## 1. 缺陷确认：OPT 长跑的"训练语料"其实是每 client 固定的前 L 篇

### 1.1 机制（代码级）
- `src/fed/run_fed.py`：每轮对每个 client **新建** `iter_batches` / `iter_classification_batches`，
  从分片第 0 行**顺序**取；`ClientRunner.run()` 每轮只 `next()` `local_steps` 条（batch=1）。
  ⇒ 每个 client 全程只用其 `train_*.npy` 的**前 L 行**；`--max-train-samples 0`（默认）不限量但**不推进游标**。
- ZOO 与 BP 共用该取数路径；`local_test`（每人 16 篇）与其余 98 篇（L=2 时）从未参与训练。

### 1.2 实测证据（三个代表 run 的 config.json 全部一致）
| run | local_steps | max_clients | max_train_samples | 每轮样本 | 轮数 |
|---|---|---|---|---|---|
| `logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip`（首个正结果） | 2 | 30 | 0 | 30×2=**60** | 30 |
| `logs/fed/a03_zoo_k24_d32_lr0.10_cont15c` | 2 | 30 | 0 | 60 | 15 |
| `logs/fed/a01_zoo_k24_d32_catv_lr0.30_r30` | 2 | 30 | 0 | 60 | 30 |

⇒ **30 轮训练暴露 = 1800 次 doc-views，全部落在同一批 60 篇上（每篇 30 遍）**；每 tag 3000 篇（a01 唯一 2899）中仅 60 篇被使用。

### 1.3 数据瑕疵（顺带核验，α=0.1 特有）
逐行内容哈希（2026-10-12 核实）：
| tag | train 行数 | 重复内容行 | 每 client 唯一文档 min/median | fallback_reuse 总数 |
|---|---|---|---|---|
| a01 (α=0.1) | 3000 | **101** | **client_17 仅 10 篇**（存储 100 行中 90 行重复） | 133 |
| a03 (α=0.3) | 3000 | 1 | 99 / 100 | 10 |

原因：`ClientPoolStream`（`src/data/build_discovery_16k.py:65-81`）在池子耗尽后**有放回**补抽；
α=0.1 的 per-class Dirichlet 尾巴把 client_17 的池压到 10 行。处置（用户定）：**不改数据**，论文/附录注明。

### 1.4 影响定性
- 臂间对比（CATV on/off、α、D、L）**仍内部有效**（各臂同一底料）；
- 绝对下降幅度被 60 篇封顶；部分"下降"可能是**对固定 mini-batch 的适应**；
- 分类任务（1 label / 16k tokens）比 OPT 的 LM（16k token 损失）对该缺陷敏感得多——
  BERT 上「L=2 平 / L=8 降」与「重复语料」高度纠缠（见 §4）。

---

## 2. 已完成的 BERT 前置工作（可复用）

### 2.1 分支与提交
- 工作分支 **`exp/distilbert-16k`**；已合并 OPT 主线（commit `8db77eb`，零冲突，selftest 全过）。
- 本轮新增代码已随本 ailog 一起提交（见变更列表）；`temp/` 下运行脚本不随 git（本机在，路径见 §6）。

### 2.2 数据（`dataset_v3/arxiv_16k/`，均 check_partition PASSED）
- a01（α=0.1）/ a03（α=0.3）：30 clients × (100 train + 16 local)、seq 16384 头尾截断 75/25、
  global_test 500（取自 test split）、global_train 3832 / global_val 426（15% holdout）、warmup 512（与 client/global 互斥）。
- 数据隔离：warmup 排除集 7605（clients 2899+448、global 4258），可用池 20783；a01/a03 的 global 池逐行一致。

### 2.3 权重/检查点（权重隔离：distilbert_* 前缀，与 OPT 产物零混用）
- `checkpoints/distilbert_step1/a01_pos_head_1000step/{pos_embed.pt,head.pt}`（+`best/`）：
  16k global_test acc **0.4540** / NLL 1.5947 / macro-F1 0.3511（final 保留；best 备选 0.4420/1.4937/0.3334）；
  2k（truncate 2048）acc 0.398 / NLL 1.818 无退化。
- `checkpoints/predictor_bert/a01_warmup_400step/{predictor.pth,pruned_config.pth}`：36 张量（6 层×6），
  train loss 0.0169；剪枝层=2/3/4（0,1 不剪、5 豁免）。

### 2.4 代码（均在 `src/`，本轮提交）
- `src/models/modeling_distilbert_smell.py`：`DistilBertSdpaAttention` / **`DistilBertSdpaPruneAttention`**（双向 token 子集化、
  padding 安全、CATV votes+consensus_mask）、`patch_distilbert_sdpa_prune`、`block_attn_pool_bidir`、
  `load_distilbert_classifier(head/predictor/pruned_config/sparse)`；selftest 16 项 PASS。
- `src/train/train_predictor_bert.py`：双向目标 predictor 训练器（36 张量、局部 enable_grad）。
- `src/fed/run_fed.py`：`--arch {opt,distilbert}` 分派、`--head-checkpoint/--num-labels`、
  分类 batch（ids/mask/labels）、eval 走 `classify.py`（`--eval-max-samples/--eval-batch-size`）。
- `src/eval/classify.py`：predictor/pruned-config/CATV 掩码/逐样本输出/`--sparse`。
- `src/fed/serial_fedavg.py`：`task=classification` 的 CE 损失路径。
- `src/train/lora.py`：`task_type` 参数；`src/train/zoo.py`：`layer.N.`（DistilBERT）兼容。
- `scripts/diag_cos_bert.py`：cos 诊断（`--attn/--ls/--ds/--blocks`）。

### 2.5 关键实验数据（原始 log 均在 `logs/fed/bert_*` 与 `temp/logs/bert_*`）
**BP 参考系（a01、dense、c30、10 轮、250 样本评测；基线 acc 0.4880 / NLL 1.5317）**
| 配置 | ΔNLL% | ΔAcc | 秒/轮 | δ/轮 |
|---|---|---|---|---|
| 集中（global_train 3832 篇 200 步） | **+11.5%** | +6.0 | 268 | 1.52* |
| **FL lr3e-4 ls8** | **+1.50%**（单调） | **+4.4** | 127 | 0.223 |
| FL lr1e-4 ls8 | +0.54% | +1.6 | 247 | 0.076 |
| FL lr1e-4/3e-4/1e-3 ls2 | +0.06% / −1.11% / −11.6% | ≈0 | 61/60/27 | 0.028/0.085/0.274 |
| 单 client 100 步 | −16.4% | −6.0 | 46 | 2.08* |
（*=被 clip 1.0；详见 `temp/logs/bert_bpref_261012-113658/`、`logs/fed/bert_bp_ref/`、`logs/fed/bert_central_ctrl/`）

**剪枝推理惩罚（2×2 控制，同 adapter 换推理路径）**：dense→sparse 从 +0.21 nats 缩到 +0.11 nats；
sparse adapter 特化到剪枝路径（dense 推理反而 1.93 vs 剪枝 1.59）。原始：`temp/ctrl_*.json`。

**cos（fp32、sparse、L2）**：c=1 ratio≈1.1；**c↑ 不涨**（α=0.1 标签正交；OPT 的 LM 同规格涨到 0.093）；
full6 D22/D64 @c30 = 0.050 / 0.061；k3 同量级；dense 对照与 sparse 无差（排除剪枝噪声假说）。
原始：`temp/diag_cos_bert_a01.json`、`temp/diag_cos_bert_a01_dense.json`。

---

## 3. 接口设计：`--doc-order {fixed,cycle}`（用户方案：是否允许轮转）

- **默认 `fixed`**（= 现状，保持与 OPT/discovery 老协议可比，legacy 不动）；
  **BERT 后续所有训练/消融一律显式 `cycle`**。
- 语义：
  - `fixed`：每轮从第 0 行取 L 条（现状）；
  - `cycle`：`start = (round_idx * local_steps) % num_samples`，从 `start` 起取 L 条、**循环卷绕**；
    30 轮 × L8 = 240 次抽取覆盖分片 100 篇 ≈ 2.4 遍。
- 实现点（下个 Agent 落地，约 30 行）：
  - `src/fed/run_fed.py`：新增 `--doc-order {fixed,cycle}`（默认 fixed）；
    分类路径把 `start=(round_idx*args.local_steps) % num_samples` 传给
    `iter_classification_batches(..., start=start)`；OPT 路径 `iter_batches(..., start=)` 同签名但默认 0；
  - `iter_classification_batches` / `iter_batches`：起始行 + 卷绕（`(start + k) % num_samples`）；
  - `run_config` 与每条 round 记录 `doc_order`、`round_start`（可复现）；
  - 注明：`cycle` 是 BERT 分支的协议改进，与 OPT 老结果不可直接混谈（论文另行解释）。
- 数据固有情形：a01 `client_17` 只有 10 篇唯一文档（§1.3），cycle 也无效，报告注明即可。

---

## 4. 待验证假设与判定实验（用户核心关切：不是 L，而是语料）

- **H1（L 效应）**：每轮样本数 L 本身提升聚合更新质量（ls8 单调 vs ls2 不降）。
- **H2（语料效应）**：真正瓶颈是**累计唯一语料/多样性**；L 只是"每轮语料量"的代理。
  （ls2 固定=全程仅 2 篇/人；ls8 固定=8 篇/人，都不是语料增长的真正形态。）

判定实验（并入主表，成本很低）：
1. **fixed vs cycle @ (L8,D8)**：同 L、同 D，仅换轮次间文档复用 → 分离"轮次重复"效应；
2. **等语料对照**：`L8 × 5 轮`（40 篇唯一/人）vs `L2 × 20 轮`（40 篇唯一/人，成本≈1/4）→
   H2 成立则两者接近（L 非根因）；H1 成立则 L8 显著更好；
3. 主网格 iso-M 对角（L8D8↔L16D4、L8D16↔L16D8）：同 M 下 L/D 的分离。
- 预判（若 H2 成立）：cycle 后 L2 长轮数也下降 ⇒ r30 可用更便宜的 L2（成本省 4×）；
  若 H1 成立：r30 必须 L≥8，预算见 §5。
- 注意：cycle 之后"L=8 × 6 轮"= 48 篇/人；要复现 BP 的下降幅度可能需要更多轮（轮数×L=总语料），
  轮数预算需在看中间表时和用户确认（用户已定 r30，若语料是杠杆，r30 就是既定语料量）。

---

## 5. 执行清单（下一个 Agent 按序启动；用户会另行拍板预算）

**P0 接口 + smoke（~15 min）**
1. 实现 §3 的 `--doc-order`；`py_compile` + 2-client/1-round smoke（验证 offset 单调、卷绕、metrics 记录）；
2. 两个 ZOO 探针（δ 锚定）：`--trainer zoo --local-steps 8/16 --max-clients 1 --max-train-samples 8/16 --lr 4e-6`
   → `lr0(L) = 4e-6 × 0.223 / δ_probe(L)`（0.223 = BP 参考 δ/轮；clamp [1e-4, 1.0]）；
   探针命令模板见 `temp/run_bert_chain_bg.sh` 的 probe 段。

**P1 L×D 主表（r=1，c30，6 轮，评测 250@bs8，`--attn sdpa_prune --sparse 0.4 --catv off --doc-order cycle`）**
| 格 | 秒/轮 | 6 轮 | 备注 |
|---|---|---|---|
| L8·D4 | ~7.6 min | ~46 min | 便宜锚点 |
| L8·D8 | ~14 min | ~1.4h | **iso-M ↔ L16·D4** |
| L16·D4 | ~14 min | ~1.4h | iso-M 对照 |
| L16·D8 | ~27 min | ~2.7h | **iso-M ↔ L8·D16** |
| L8·D16 | ~27 min | ~2.7h | iso-M 对照 |
| L16·D16 | ~52 min | ~5.2h | 收角（趋势好再补） |
| **对照 A**：L8·D8 `--doc-order fixed` | ~14 min | ~1.4h | H2 判定 ① |
| **对照 B1**：L8·D4 ×5 轮 | — | ~38 min | H2 判定 ② |
| **对照 B2**：L2·D4 ×20 轮 | ~2.8 min | ~56 min | 同 40 篇唯一语料 |
| dense 锚点：L8·D8 `--attn sdpa` | ~14 min | ~1.4h | 剥离剪枝惩罚 |
- P1 合计 ≈ **12h**（不含收角）；顺序：L8D4 → L8D8 → 对照A → L16D4 → L16D8 → L8D16 → 对照B → 锚点 →（收角）。
- 每格记录：ΔNLL% / ΔAcc / ΔAcc% / ERR↓ / %/h（train-only 与含评测）/ δ / 秒/轮 / 唯一语料数（cycle 下 = min(轮数×L, 100)）。
- **中间结果一出来先给用户**：fixed vs cycle、等语料对照，回答"H1 or H2"。

**P2 rank 表（P1 之后）**：在胜出格上 r∈{2,4}（`--lora-alpha 2r`，保持 alpha/r=2；r=1 复用主表格），6 轮，~1.4-2.7h/格。

**P3 r30 四臂（α∈{a01,a03} × CATV on/off，c30，评测 500，`--doc-order cycle`）**
- 配置取 P1/P2 胜者；预算（成本模型 0.2s/前向）：L8D8 ≈ 6.5h/臂 → **26h**；L16D8 ≈ 13h/臂 → 52h；L2D8 ≈ 1.6h/臂 → 6.5h（若 H2 成立）。
- CATV 前置已就绪（votes/consensus_mask/mask 注入评测，均在本轮代码内并有 smoke 记录）。

**预算汇总**：P0 15min + P1 ~12h + P2 ~4h + P3 6.5–52h（取决于 H1/H2 与用户选择）。

---

## 6. 现有脚本与产物索引（本机）

- 数据：`temp/run_bert_data_bg.sh`（已完成，a01/a03 全套 + warmup + check）
- Step1：`temp/run_bert_step1_bg.sh`（已完成）→ `temp/logs/bert_step1_261012-095601/`
- 链（predictor→BP 先导→cos，已完成）：`temp/run_bert_chain_bg.sh` → `temp/logs/bert_chain_261012-100413/`
- BP 参照：`temp/run_bert_bpref_bg.sh` → `temp/logs/bert_bpref_261012-113658/`
- cos：`scripts/diag_cos_bert.py`；原始 `temp/diag_cos_bert_a01{,_dense}.json`
- 旧扫描脚本（**将被 L×D 表替换，勿直接用**）：`temp/run_bert_scan_bg.sh`（D8/16/32/64 + lr）、`temp/run_bert_scan_r_bg.sh`（r2/r4）
- 控制实验原始：`temp/ctrl_*.json`、`temp/eval_baseline_250.json`、`temp/eval_step1_16k_{final,best}.json`
- 集中对照：`logs/fed/bert_central_ctrl/`（数据用软链 `temp/bert_central/`）
- 尚无：`--doc-order` 实现、L×D 表脚本、探针脚本、r30 脚本（下个 Agent 写）。

---

## 7. 风险与注意事项
- **不要重做 OPT**（用户明确）；不动 `third_party/`；`--doc-order` 默认 `fixed`，旧行为零变化。
- 评测口径：表中 250 样本（快）；r30 用 500 样本 + 逐样本 JSON（配对检验）。
- 剪枝推理惩罚是常数项（起始 +0.19 nats，随训练修复），臂间比较自洽；绝对数需注明。
- cos 仅辅助（用户定调）：主判据 = 相对下降 + %/h。
- a01 `client_17`（10 篇）与 101/3000 重复行：不改数据，注明。
- ZOO 的 L 与 D 成本耦合（L×D×2 前向）；同 cLD 下优先 L（数据暴露），除非 P1 判定 H2。

## 8. 变更列表
| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 15:18 | 新增 | `docs/ailog/261012-151805-smell-v3-bert-fixed-60docs-protocol-flaw-and-ld-plan-handoff.md` | 本交接 |
| 15:20 | 提交 | `src/models/modeling_distilbert_smell.py`、`src/train/train_predictor_bert.py`、`src/fed/run_fed.py`、`src/fed/serial_fedavg.py`、`src/eval/classify.py`、`src/train/lora.py`、`src/train/zoo.py`、`scripts/diag_cos_bert.py` | BERT 流水线代码（本轮）|

## 9. 待办（下个 Agent）
- [ ] `--doc-order {fixed,cycle}` 实现 + smoke
- [ ] P0 探针（ls8/ls16，δ 锚 0.223）
- [ ] P1 L×D 主表（含 fixed 对照/等语料对照/dense 锚点）→ **先报 H1/H2 判定**
- [ ] P2 rank 表 r∈{2,4}
- [ ] P3 r30 四臂（α×CATV，cycle，500 样本）
- [ ] 每阶段向用户汇报（%/h + 相对下降 + δ）
