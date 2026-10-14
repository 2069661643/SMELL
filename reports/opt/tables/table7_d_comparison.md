# Table 7 - D comparison (unified 8-round window, full G-PPL)

All rows use the 8-round window (r0 = after round 0, r7 = after round 8; eval_every=1). D16/D22 from the 8-round pilots; D32 endpoint is a post-hoc eval of the 30-round run's adapter_round007 (the in-run eval grid is every 5 rounds). Each row's r0 is that arm's own post-round-0 state (fixed order, first 2 samples/client per round). D32 extended points (not in the unified window): r8 32.4151, r9 32.3408.

| D | window | r0 full PPL | endpoint full PPL | delta | delta % |
|---|---|---|---|---|---|
| 16 | r0->r7 | 33.5332 | 32.7144 | -0.8189 | -2.44 |
| 22 | r0->r7 | 33.5211 | 32.5335 | -0.9875 | -2.95 |
| 32 | r0->r7 | 33.5296 | 32.5071 | -1.0225 | -3.05 |
