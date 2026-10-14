# Table 7 - D comparison (unified 8-round window, full G-PPL)

Shared pre-training baseline = base model + pos_only table with identity LoRA (PEFT B=0), same 500-sample protocol; measured once (full 33.6653 / answer 1740.6), applies to all arms. All rows use the 8-round window (r7 = after round 8; eval_every=1). D16/D22 from the 8-round pilots; D32 endpoint is a post-hoc eval of the 30-round run's adapter_round007 (the in-run eval grid is every 5 rounds). D16 own r0 33.5332 (-0.8189); D22 own r0 33.5211 (-0.9875); D32 own r0 33.5296 (-1.0225). D32 extended points (not in the unified window): r8 32.4151, r9 32.3408.

| D | window | baseline (pre-r0, shared) | endpoint full PPL | delta | delta % |
|---|---|---|---|---|---|
| 16 | pre-r0->r7 | 33.6653 | 32.7144 | -0.9510 | -2.82 |
| 22 | pre-r0->r7 | 33.6653 | 32.5335 | -1.1318 | -3.36 |
| 32 | pre-r0->r7 | 33.6653 | 32.5071 | -1.1582 | -3.44 |
