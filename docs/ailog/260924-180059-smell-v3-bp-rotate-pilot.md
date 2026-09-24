# Ailog 260924-180059 — SMELL-v3: BP-rotate 多轮先导（秩轮换可行性判断）

## 概述

在 `exp/rank-rotation-lora` 分支上，用**可信梯度（BP）**做秩轮换的多轮先导，判断「LoRA 8 秩轮流（每轮 1 秩）」是否可行。为避免 warmup 基线过高掩盖趋势，**从 base 起训**（无 `--adapter-init`，仅 `--pos-checkpoint`）。结论：**机制可行但 k=1 明显偏慢**——同预算（r24/l4/c5）下 BP-rotate 的 loss/G-PPL 约为 BP-all 的 4×（G-PPL 106 vs 25），需更大 k 或更多轮才能补齐。

## 设置（16k, c=5, local=4, lr=1e-4, rounds=24, eval-every=8, catv off, 从 base 起训）

| run | rank-mode | loss r0/r4/r8/r12/r16/r20/r23 | δ 范围 | eval G-PPL r7/15/23 |
|---|---|---|---|---|
| bp_all_base | all | 5.557 / 4.368 / 3.653 / 3.430 / 3.347 / 3.289 / **3.256** | 0.29–0.44 | 36.18 / 26.96 / **24.93** |
| bp_rotate_k1_base | rotate k=1 | 5.601 / 5.518 / 5.417 / 5.279 / 5.121 / 4.962 / **4.829** | 0.12–0.16 | 209.31 / 149.00 / **106.20** |

- `bp_rotate_k1` 的 `active_ranks` = [0]→[1]→…→[7]→[0]…，**轮换正确**（r8 回到 [0]）；loss 与 G-PPL 随轮数**单调改善、无发散/NaN/OOM**。
- 产物齐全：`logs/fed/bp_rotate_pilot/<run>/a01/{config.json,metrics.jsonl,adapter_round007/015/023,eval_round007/015/023.json}`。

## 结论

1. **可行性**：秩轮换机制正确且稳定，从 base 起训能正常收敛（loss/G-PPL 单调下降）。
2. **效率**：k=1/c=5/r24 下比全秩慢约 **4×**（G-PPL 106 vs 25）——原因是每轮仅更新 1/8 参数、每秩 8 轮才轮到一次，覆盖太稀疏。
3. **判断**：作为主线「ZOO 降维稳梯度」的载体，秩轮换**机制可用但 k=1 不划算**；若要采用应改为 **k≥2/4（每轮多秩）**或**显著加长轮数**，并配合 SVD 重整以均衡各秩贡献（未做）。

## 云端环境

host `amax`（A40×4），仅 GPU2（GPU0/1/3 被他人占用）；`smell-v2`（torch 2.1.2+cu118 / FA 2.4.2 / transformers 4.45.2 / peft 0.19.1）。

## 待办

- 先导结论：**BP-rotate k=1 偏慢**。下一步按用户选择 B（ZOO 方向质量），或先试 `k=2/4` 看能否补齐差距。
- 未做 SVD 重整 ⇒ 秩基依赖 warmup/随机 (A,B) 参数基，各秩贡献不均（δ 0.12→0.16 波动）。
- 分支 `exp/rank-rotation-lora`，便于回滚 main。
