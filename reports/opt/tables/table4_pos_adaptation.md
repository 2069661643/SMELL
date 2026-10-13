# Table 4 - Position adaptation (a01 warmup pool)

| arm | mode | 16k G-PPL | 2k G-PPL | judgement |
|---|---|---|---|---|
| no adaptation | - | 398.77 | 40.01 | reference |
| pos_only (B) | learned offset (interp init) | 31.42 | 32.49 | 16k G-PPL < 50 PASS; no 2k regression |
| pos_lora (C) | pos_only + LoRA adapter | 18.14 | 18.13 | selected; best 16k and 2k |
