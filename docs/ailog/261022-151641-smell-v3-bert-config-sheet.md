# Ailog 261022-151641 — SMELL-v3: BERT 论文用 config 表（BERT_CONFIG.md + JSON）

> 论文端询问：BERT 能否补一份 config（序列长度、LoRA、local steps、lr、数据构造）？
> 本文件记录新增的 paper-facing config 表；不改任何实验结果/CSV。

## 一、新增产物

| 文件 | 内容 |
|---|---|
| `reports/bert_and_other/BERT_CONFIG.md` | 人读版：Model / Sequence & sparse attention / LoRA / Federated / Data construction / Evaluation / Variants / Notes |
| `reports/bert_and_other/bert_config.json` | 同内容机器可读版 |

两者由 `reports/bert_and_other/make_report.py` 的 `bert_config_package()` 从实际
`logs/fed/bert_da_fixed/r30_a03_catv_off/a03/config.json`、`third_party/Jenga/checkpoints/distilbert-base-uncased/config.json`
和 `dataset_v3/arxiv_16k/{a01,a03}/meta.json` / `warmup_meta.json` 生成。

## 二、核心值（r30 四臂正式口径）

- Model：DistilBERT-base-uncased，6 层 / 12 heads / hidden 768 / FFN 3072 / vocab 30522 / 11 类；
  16k 位置表来自 `distilbert_step1/a01_pos_head_1000step/pos_embed.pt`；分类头冻结。
- Sequence：**16,384**；head_tail 截断 **75% head + 25% tail**（12,288 + 4,096）；右 padding、pad_id=0、uint16 存储。
- Attention：`sdpa_prune`、`sparse=0.4`、256 blocks × 64 tokens、`keep_block0=True`；
  层 0/1 全保留、层 2/3/4 保留 top-40% query blocks、层 5 豁免。
- LoRA：`r=1`、`alpha=2`（scaling=2）、dropout=0、bias=none、task=SEQ_CLS；
  targets `q_lin/k_lin/v_lin/out_lin`；仅 LoRA 可训练，**36,864** 参数（6×4×(768+768)）。
- Federated：`trainer=zoo`、**30 clients**、**local_steps=8**（batch=1，逐 sample 立即更新）、
  **rounds=30**、**lr=1.4e-3**、ZOO layers 0–5、D=8、eps=1e-3、**delta-clip=1.0**、
  sample-weighted FedAvg（每 client 100 篇 → 等权）、`doc_order=cycle`、seed=0、fp32。
- Evaluation：global_test **500** 样本、batch 8、每轮 eval；指标 NLL / accuracy / macro-F1。
- Data construction：`ccdv/arxiv-classification [no_ref]`，11 类；per-class Dirichlet：
  a01 α=0.1 / a03 α=0.3，seed=42；30 clients × (100 train + 16 local-test)；
  global_test 500（test split 随机 500，seed [42,303]）；global_train 3832 / global_val 426（Step1）；
  warmup 512（predictor，与 client/global 互斥）；global_demo_frac=0.15；
  train/test split 28388/2500；excluded/available 7605/20783。
- 数据 quirk：a01 总 fallback reuse=133（client_17 仅 10 篇唯一文档）；a03=10（client_29）；
  a01/a03 的 global 池逐行一致，α 只改变 client 分区。
- 变体：CATV on r=0.2/s=0.4、anchors 51/51；VQ per-result 0.5 bf16 C=64/c_dim=18432；
  VAR Method A T0=0.587230920791626、D=4..16；combo CATV(client)+VAR+VQ 0.5，mean D=9.003、mean C=72.02。

## 三、注意

- 16k 必须用学习式位置表；基础 DistilBERT config 仍写 `max_position_embeddings=512`，不要误读。
- BERT 的 NLL/acc 与 OPT 的 token PPL 不可混表。
- 四臂 CATV 用 `catv_normalize=off`；组合系统用 `client`，两者不要混用。

## 四、验证

- `make_report.py` 仍 **99/99 PASS**（只新增 config 生成，不改结果 CSV）。
- `BERT_CONFIG.md` / `bert_config.json` 由实际 config/meta 生成，避免手工抄错。

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 15:15 | 新增 | `reports/bert_and_other/BERT_CONFIG.md` | 论文用 BERT config 人读版 |
| 15:15 | 新增 | `reports/bert_and_other/bert_config.json` | 机器可读版 |
| 15:15 | 修改 | `reports/bert_and_other/make_report.py` | `bert_config_package()` + README 索引 |
| 15:16 | 修改 | `AGENTS.md` | 报告包描述 + 必读 ailog |
| 15:16 | 新增 | 本 ailog | config 交付记录 |
