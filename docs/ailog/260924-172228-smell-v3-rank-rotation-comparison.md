# Ailog 260924-172228 — SMELL-v3: LoRA 秩轮换实现后的 16k 对照（exp/rank-rotation-lora）

## 概述

在实验分支 `exp/rank-rotation-lora` 上实现「LoRA 8 秩轮流（每轮激活 rank_k 个秩，ZOO/BP 通用）」并在 16k 做首轮对照：`bp_all`（参照）、`zoo_rotate_k1`（每轮轮换 1 秩）、`zoo_all`（不轮换）。**机制验证全部通过**（`active_ranks` 0→7 完美轮换、非激活逐位冻结、ZOO dim=196608、无 NaN/OOM），但**短轮次下三者 G-PPL 无显著差异**（18.52/18.52/18.61，warmup 基线 18.14），ZOO 的 `cos≈0` 问题未因秩掩码改变。同时修复了 `--eval-every` 漏传 `--pos-checkpoint` 导致 G-PPL 失真的 bug。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 16:51 | 新增 | `src/train/zoo.py` | — | `build_active_index` + `flatten/apply/restore/zo_grad` 的 `index`（None 逐位兼容）；selftest +4 项 |
| 16:51 | 新增 | `src/train/lora.py` | — | `num_lora_ranks`、`active_ranks_for`（rotate/lock/all） |
| 16:51 | 新增 | `src/fed/serial_fedavg.py` | — | `ClientRunner` 接 `active_index/active_ranks`；BP step 后快照-还原非激活 |
| 16:51 | 新增 | `src/fed/run_fed.py` | — | `--rank-mode/--rank-k`，每轮秩调度，通信量按激活子空间计 |
| 17:07 | 修改 | `src/fed/run_fed.py` | `run_global_eval` | 评测补 `--pos-checkpoint`（否则位置表错位致 G-PPL 失真） |
| 17:08-17:20 | 运行 | `logs/fed/rankrot_cmp/*`（gitignore） | — | 3 个 16k 对照 run |

## 对照结果（16k, c=3, catv off, pos_lora adapter, eval=global 500）

| run | trainer | rank | lr | rounds/local | 末轮 δ | 最终 G-PPL |
|---|---|---|---|---|---|---|
| bp_all | bp | all | 1e-4 | 3/4 | 0.292 | **18.516** |
| zoo_rotate_k1 | zoo | rotate k=1 | 2e-7 | 8/1 | 0.075 | **18.524** |
| zoo_all | zoo | all | 1e-7 | 3/1 | 0.265 | **18.608** |
| （参照）warmup base | — | — | — | — | — | 18.14 |

- `zoo_rotate_k1` 的 `active_ranks` = [0]→[1]→…→[7]，逐轮正确；`delta_norm` 0.05–0.10（单秩，明显小于 all 的 0.40→0.27），全程稳定无 NaN。
- 三者 loss 均在 ~3.0 附近徘徊（ZOO 基本不动）；`cos(ZOO,BP)≈0`（±1e-4），与 v2/Step3 一致。
- 注：`answer_ppl_token`（998–1259）偏高，非本次评价重点，未纳入判据。

## 结论

1. **实现正确**：秩轮换/冻结机制按预期工作，ZOO 的 `dim` 由 1.57M 降到 196,608（√dim 1254→443），可用 lr 相应放大（全秩 1e-7 → 激活维 ~2e-7）。
2. **短轮次无质量差异**：warmup 基线（18.14）已很好，3–8 轮微调无法在 G-PPL 上区分；且三者都略高于基线，说明这几轮更新未带来净收益（甚至轻微扰动）。
3. **秩掩码未解决 ZOO 的本质问题**：`cos≈0` 依旧——ZOO 方向与有效方向近乎正交。秩轮换降低的是**方差/通信**（稳定性提升），不是**方向质量/收敛速度**。

## 云端环境

host `amax`（A40×4），仅 GPU2（GPU0/1/3 被他人占用）；`smell-v2`（torch 2.1.2+cu118 / FA 2.4.2 / transformers 4.45.2 / peft 0.19.1）。

## 待办 / 风险

- 若要真正比较 `full-r8` vs `rotate-8`，需给训练留出**可测空间**：更多轮次 + 更大 lr，或**从无 warmup adapter 的 base 起训**，或换更敏感的质量指标；当前 warmup 基线太高使 G-PPL 不敏感。
- 未做 SVD 重整 ⇒ 秩基依赖 warmup 的 (A,B) 参数基，各秩贡献不均（δ 波动 0.052–0.097）；若要均衡轮换需先重整。
- ZOO `cos≈0` 是主线更根本的阻塞，建议 Step4 前先解决 ZOO 方向质量（更大 directions / 更小 eps / 或改用 SNP/MeZO 变体），而非仅调 rank。
- 分支 `exp/rank-rotation-lora` 便于回滚到 `main`。
