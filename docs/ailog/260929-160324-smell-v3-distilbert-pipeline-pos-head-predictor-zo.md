# Ailog 260929-160324 — SMELL-v3: DistilBERT pipeline 定稿（pos+head → predictor → BP/cos 门控 → ZO 消融）

> 承接 `260929-095542`（框架 + 前向 smoke）与用户讨论。本 ailog 锁定 DistilBERT 16k 主线训练管线与评测口径，
> 随后按此开工。数据集 = `ccdv/arxiv-classification`（no_ref，11 类），seq 16384、头尾截断 75/25 + padding。

## 一、用户决策（锁定）

| # | 决策 | 内容 |
|---|---|---|
| D1 | 训练顺序 | ① 联合训练 pos + head（encoder 冻结）→ ② 训练 predictor → ③ BP 先导 + ZOO cos 打点门控 → ④ 4 点 ZO 消融 |
| D2 | Step 1 目标函数 | **A：`CE([CLS]) + λ·MLM(15% mask)`**，同一 batch 双目标（MLM 头冻结，用 mask 位 gather 控制显存）；备选 C（先 MLM 后 CE） |
| D3 | 分类头 | **完整头**：`pre_classifier(768→768) + classifier(768→11)` = 0.60M；FL 中冻结，只传 LoRA |
| D4 | Step 4 四点 | **α∈{0.1,0.3} × CATV off/on**（原 SMELL 口径） |

## 二、四步计划与门控

### Step 1 — 联合训练 pos + head（冻结 encoder）
- **数据**：用构建时预留的 `global_demo_idx` 池（train 的 15%，a01≈4.2k 文档），再切 `global_val`（默认 10%）早停用；
  **不得用 client 分片**（防 FL 评测泄漏）。需要构建器新增 `global_train_*` / `global_val_*` 输出。
- **可训练**：pos 表 12.58M（lr 1e-4）+ 头 0.60M（lr 1e-3）；encoder/MLM 头冻结；fp32 + SDPA + 可选 grad-ckpt。
- **产物**：`checkpoints/distilbert_step1/pos_embed.pt`（`{"position_embeddings.weight": ...}`）+ `head.pt`（4 个张量）。
- **门控**：16k global acc ≫ 1/11；2k 无回归；val 不早停衰退。
- **成本**：~1k step，4090 fp32 约 40–60min，显存 ~5–6GB。

### Step 2 — predictor（只依赖 pos）
- 新写 `train_predictor_bert.py`：非 causal `block_attn_pool`（双向正确目标）+ 分块 pooling + SDPA，只训 36 张量 ≈15.7M。
- 数据 = warmup 16k；产物 `predictor.pth` + `pruned_config.pth`（推理层径：layers 2–4 剪，末层豁免）。
- **门控**：sparse=0.4 acc 回退 <1–2pt。
- **成本**：<1GB 显存，本地 8GB 可跑，400–800 step ≈10–20min。

### Step 3 — BP 先导 + ZOO cos 打点（Go/No-Go）
- BP 先导：LoRA r=1 全层 × {dense, sparse} × c∈{3,10,30}，给 ZO 上限与 lr/δ 参考。
- cos 打点：移植 `diag_cos_grid.py` → `scripts/diag_cos_bert.py`；测 {全层 r1, k4 轮转} × D∈{8,22,64} × L2,c30 的 cos（vs 同 batch BP 梯度）。
- **门控**：cos ≥ 0.05 才进 Step 4；否则判 16k ZO 不可行、转 BP 主线。

### Step 4 — ZO 消融 α×CATV（4 点）
- 前置：① DistilBERT SDPA 剪枝路径补 CATV 投票 + consensus_mask（现仅 `OptFlashAttention2` 有）；② a03 数据构建。
- 成本：fp32 16k 单轮 ≈45–90min（D22,c30,L2 串行），4 点建议双机（AutoDL 4090 / amax GPU2）分摊。

## 三、本 session 开工范围（本机 8GB，只做实现 + smoke）

1. 本 ailog；
2. `build_arxiv_16k.py` 增 `global_train_*` / `global_val_*` 输出（Step 1 数据）；
3. 新 `src/train/train_pos_head_bert.py`（A 口径，含 `--mlm-weight 0` 退化为 B、`--freeze-pos/--freeze-head` 兼容 C）；
4. 小规模 smoke（a01mini / truncate，验证 loss 下降 + 产物可加载）+ 全部 py_compile；
5. 提交推送 `exp/distilbert-16k`。

## 四、风险

- 32× 位置外推（512→16k）+ 4.2k 文档：MLM 辅助是主要对冲；若 val 仍崩，退 C 或重初始化（sinusoidal/ALiBi）。
- `global_demo` 池语义：原为 Discovery ICL 预留，arxiv 中未使用，复用安全（构建器会在 meta 中标注）。
- Step 2 的 predictor 必须用 Step 1 定稿的 pos 表重训一次；FL 期间 pos 冻结。
- Step 4 的 CATV 在 SDPA 剪枝路径缺失，是额外开发项（+0.5~1d），需先于消融完成。

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 16:03 | 新增 | `docs/ailog/260929-160324-...distilbert-pipeline-pos-head-predictor-zo.md` | 管线定稿（D1–D4 + 门控） |
| 16:08 | 修改 | `src/data/build_arxiv_16k.py` | +78 行：`global_train_*`/`global_val_*` 输出（`--global-val-frac 0.1`、`--no-global-pool`），meta 增 `global_pool` 与标签直方图 |
| 16:12 | 新增 | `src/train/train_pos_head_bert.py` | Step 1 trainer：单遍 CE+MLM（A）；`--mlm-weight 0`=B；`--freeze-pos/--freeze-head`=C；产物 pos_embed.pt + head.pt |
| 16:15 | 修复 | `src/train/train_pos_head_bert.py` | 4.45 的 `DistilBertForMaskedLM` 为扁平 MLM 头（无 `model.cls`），改为 `vocab_transform/activation/vocab_layer_norm/vocab_projector` |

## 本 session 验证（WSL 8GB，a01mini / truncate 512）

| 项 | 结果 |
|---|---|
| `py_compile` | PASS（trainer + builder） |
| A 模式 smoke（4 step，bs2） | loss 3.10（ce 2.37 + 0.1·mlm 7.34），grad_norm 2.74，eval acc 0.2（随机头 + 5 样本，链路通） |
| C 段1（`--freeze-head --mlm-weight 1.0`，pos-only） | trainable 1,572,864，2 step 正常 |
| C 段2（`--freeze-pos --mlm-weight 0`，head-only） | trainable 599,051，2 step 正常 |
| 产物加载 | `pos_embed.pt` key=`position_embeddings.weight` shape (2048,768)；`head.pt` 4 张量形状正确 |
| mini 数据（a01mini，由子智能体重建） | global_train (40,2048)/global_val (5,2048)，`check_partition` 仍 PASSED |

> 注：16k 全序列的 Step 1 训练与显存实测留待 AutoDL/amax（本机按约定只做实现 + 短序列 smoke）。

## 待办

- [x] Step 1 实现 + 本机短序列 smoke + 推送
- [ ] M1 全量：a01/a03 数据（含 global pool）在 AutoDL/amax 构建
- [ ] Step 1 正式训练（4090，16k fp32）→ 过门控（16k acc、2k 无回归）
- [ ] Step 2 predictor 训练器 + 训练
- [ ] Step 3 `diag_cos_bert.py` + BP 先导 + cos 门控
- [ ] Step 4 CATV 投票移植 + a03 + α×CATV 四点 ZO
