# Table 3 - Sparse-path quality

## 3a. Predictor/selection quality at 16k (n=4; small-n support row, not paired; evaluated without the position checkpoint, so not comparable to the 500-sample rows)

| selection | full PPL | answer PPL | n |
|---|---|---|---|
| step5 predictor (main) | 33.285 | 1056.4 | 4 |
| random predictor | 31.900 | 1761.7 | 4 |
| dense (no prune) | 26.208 | 584.7 | 4 |

## 3b. Dense vs sparse on the same adapters (paired per-sample, n=500)

| variant | r0 full | r29 full | full ratio | p(full) | r0 ans | r29 ans | ans ratio | p(ans) |
|---|---|---|---|---|---|---|---|---|
| dense (sparse=1.0) | 26.467 | 26.062 | 0.9847 | <1e-300 | 3594.0 | 3154.7 | 0.8709 | 2.6e-103 |
| sparse (sparse=0.4, TD-3) | 33.6538 | 33.2851 | 0.9890 | <1e-300 | 1743.8 | 1855.0 | 1.0671 | 8.4e-68 |

Environment: OPT-350M + pos_only table, TD-3 adapters (LoRA r=1, a03, trained on the sparse rho=0.4 path), fp32 sdpa_prune, 500-sample global test; dense = the same adapters re-evaluated with sparse=1.0 (no pruning).
