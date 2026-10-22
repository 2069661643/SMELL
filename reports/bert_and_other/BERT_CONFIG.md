# BERT experiment configuration (paper-facing)

由 `reports/bert_and_other/make_report.py` 从实际 `config.json` 与 `dataset_v3/arxiv_16k/*/meta.json` 生成；
对应 BERT r30 四臂正式口径。机器可读版见 `bert_config.json`。

## 1. Model

| item | value |
|---|---|
| model | distilbert-base-uncased |
| task | 11-class arXiv sequence classification |
| layers / heads / hidden / FFN | 6 / 12 / 768 / 3072 |
| vocab / labels | 30522 / 11 |
| position adaptation | distilbert_step1/a01_pos_head_1000step/pos_embed.pt (16k) |
| classification head | distilbert_step1/a01_pos_head_1000step/head.pt (frozen in FL) |
| predictor | predictor_bert/a01_warmup_400step/predictor.pth |
| pruned config | predictor_bert/a01_warmup_400step/pruned_config.pth |
| trainable params | 36,864 (LoRA only) |

## 2. Sequence and sparse attention

| item | value |
|---|---|
| sequence length | 16,384 |
| truncation | head_tail: 75% head + 25% tail |
| head / tail tokens | 12,288 / 4,096 |
| padding | right, pad_token_id=0 |
| storage dtype | uint16 |
| attention backend | sdpa_prune |
| sparse keep ratio s | 0.4 |
| blocks / block size | 256 / 64 |
| pruned layers | 2,3,4 (top-40% query blocks); 0,1 full; 5 exempt |
| keep block0 | True |
| compute dtype | fp32 |

## 3. LoRA

| item | value |
|---|---|
| rank r | 1 |
| alpha | 2.0 |
| scaling alpha/r | 2.0 |
| dropout | 0.0 |
| bias | none |
| task type | SEQ_CLS |
| target modules | q_lin, k_lin, v_lin, out_lin |
| classification head | frozen (not federated) |
| trainable params | 36,864 |
| parameter formula | 6 layers x 4 modules x (768x1 + 1x768) = 36,864 |

## 4. Federated optimization

| item | value |
|---|---|
| trainer | zoo |
| clients | 30 |
| local steps (samples/client/round) | 8 |
| local batch size | 1 |
| rounds | 30 |
| learning rate | 0.0014 |
| ZOO subspace | layers |
| ZOO layers | 0-5 |
| ZOO directions D | 8 |
| ZOO epsilon | 0.001 |
| client update | ZOO central-difference; immediate update per sample; no optimizer |
| server update | FedAvg; sample-weighted (all clients have 100 docs -> equal); per-client delta L2-clipped to 1.0 before aggregation |
| delta clip (L2) | 1.0 |
| weighting | samples |
| doc order | cycle |
| seed | 0 |
| forwards per round | 30 clients x 8 samples x 8 directions x 2 = 3840 forwards |

## 5. Data construction

| item | value |
|---|---|
| source | ccdv/arxiv-classification [no_ref] |
| classes | math.AC, cs.CV, cs.AI, cs.SY, math.GR, cs.CE, cs.PL, cs.IT, cs.DS, cs.NE, math.ST |
| data root | dataset_v3/arxiv_16k/{a01,a03} |
| sequence length | 16,384 |
| truncation | head_tail, head_frac=0.75 |
| partition | per-class Dirichlet |
| alpha | a01=0.1, a03=0.3 |
| partition seed | 42 |
| clients | 30 |
| train / local-test per client | 100 / 16 |
| global test | 500 — random test rows (seed [42,303]); all FL evaluation |
| global train / val | 3832 / 426 — Step1 position/head train / val |
| warmup | 512 — predictor training (400 steps), disjoint |
| global_demo_frac | 0.15 |
| global_val_frac | 0.1 |
| train / test split size | 28388 / 2500 |
| client source / held-out | 24130 / 4258 |
| excluded / available pool | 7605 / 20783 |
| doc order | cycle: start = (round * local_steps) % 100; 30 rounds x 8 docs/client = 240 doc-views (~2.4 passes over 100 docs/client) |

| tag | train rows | mean length | truncated at 16k | fallback reuse |
|---|---:|---:|---:|---:|
| a01 | 3000 | 12047.4 | 1034 | 133 |
| a03 | 3000 | 12012.5 | 1032 | 10 |

- Global split: global_test / global_train / global_val rows are shared across a01/a03; only the client partition differs.
- `warmup`: 512 samples, disjoint from client/global pools.
- Data quirks:
  - a01: total_client_reuse=133; client_17 has only 10 unique docs (90 duplicate storage rows)
  - a03: total_client_reuse=10 (client_29)
  - a01/a03 global_train/global_val/global_test rows are row-identical; alpha only changes client partition

## 6. Evaluation

| item | value |
|---|---|
| split | global_test |
| samples | 500 |
| batch size | 8 |
| cadence | every 1 round(s) |
| metrics | NLL (mean per-sample CE), accuracy, macro-F1 |

## 7. Variants

| variant | key settings |
|---|---|
| CATV on (four-arm) | r=0.2, s=0.4, anchors 51/51, normalize=off |
| VQ per-result | rate=0.5, bf16, C=64, c_dim=18432, total=32.0x one delta |
| VAR Method A | control=on, T0=0.587230920791626, 4..16, even-D checks, no patience, segments=10 rounds, probe=8 clients, mean D=8.791 |
| Combo | CATV=on (client), VAR=on, VQ=0.5, mean D=9.003, mean C=72.02 |

## 8. Notes

- Sequence length 16k requires the learned position table (pos_embed.pt); the base DistilBERT config still says 512.
- Only LoRA parameters are trainable; the Step1 classification head is frozen and not federated.
- The four-arm CATV runs use raw summed votes (catv_normalize=off); the system combo uses per-client L1 normalization (client).
- Do not compare BERT NLL/accuracy with OPT token PPL; see opt_* CSVs for OPT.
