# Ailog 260925-112456 — SMELL-v3: 想法2 低秩再参数化框架 + ZO 调研结论（含想法3 改进）

## 概述

按规划，把「想法2：把 LoRA 权重重组为矩阵再在其上做低秩/LoRA」落成一个**独立原型框架**（基底为 r=1 LoRA），并把 dominant-layer ZO / MeZO-BCD / AGZO-ZO-Act 的调研结论与**风险**归档。原型已 smoke 通过（k=4 时有效可训练维 1536=96·k²），但存在 PEFT 持久化与 bf16 量化两处关键风险，接入 run_fed 尚需若干待办。

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 11:16 | 新增 | `src/train/lowrank_reparam.py` | 「想法2」原型：`attach_lowrank_reparam(model,k,seed)` 等 + 9 项自测 PASS |
| 11:16 | 新增(临时) | `temp/smoke_lowrank_reparam.py`、`temp/run_lowrank_smoke_bg.sh` | smoke（gitignore，不入库） |

## 设计

- 对 96 个模块（24 层 × q/k/v/out）各挂子模块 `_lowrank`：
  - `U(out×k)`、`V(in×k)`：固定高斯 QR **列正交**，`register_buffer`（不训练）；
  - `Z(k×k)`：**唯一可训练**（fp32 `Parameter`）；
  - A/B（r=1 LoRA）`requires_grad_(False)` 冻结；
  - forward hook：`output += (x @ V) @ Zᵀ @ Uᵀ`（fp32 计算），**不改 PEFT forward/state_dict**。
- 有效可训练维数 = `96·k²`：k=1/2/4/8 → **96 / 384 / 1536 / 6144**。
- 对应 SubZero / LoRA-XS：U/V≈固定子空间基，Z≈低维核心；差异是 U/V 用随机正交（非预训练 ΔW 的 SVD），且保留冻结的 r=1 LoRA 基底。

## Smoke 证据（OPT-350M + r=1 LoRA，GPU3）

- k=4：`flatten_trainable` dim=**1536=96·16**；`U=(1024,4) V=(1024,4) Z=(4,4)`；512 前向 loss 有限；探针 `out_delta ≡ x@V@Zᵀ@Uᵀ`（abs_err 2.4e-7）；`zo_grad(index=None, 2 dirs)` norm 有限且参数逐位恢复。

## 风险（重要，须在接入前解决）

1. **PEFT 持久化断裂**：`save_pretrained` **不含 U/V/Z**；hook 不可序列化 → `--adapter-init`/评测加载后**必须重挂并载入 Z**。
2. **bf16+flash 量化噪声**：loss 的 ULP≈1.6e-2，2-方向 ZO 的有限差分被量化噪声主导（梯度 norm 在 0↔1.2e4 间跳）→ smoke 只能用 fp32 eager 验证；生产需 fp32 计算损失或更大 eps / fp32 主权重。
3. hook 与 `deepcopy`/`reset_lora_parameters`/grad-ckpt 叠加需验证；Z 为 fp32 而 U/V 随权重 dtype。
4. 随机正交基 ≠ 任务子空间：若想更优，应改**激活引导子空间**（见下）。

## 调研结论（ZO 方向质量）

- **Dominant-Layer ZO**（[2606.05516](https://arxiv.org/abs/2606.05516)）：单 decoding 层即可匹配全模型 ZO；**选层靠纯推理的 activation-outlier 层**（LLaMA2→layer1、Qwen3→layer6）；但仅 2 个 7–8B 模型、1000 样本、短上下文，**未测 OPT**，且作者自证「敏感度大≠好层」。→ 不足以支撑「按模块重要性采样」。
- **MeZO-BCD / 子空间对齐**（[2501.19099](https://arxiv.org/abs/2501.19099)）：理论上同 stable-rank 子空间**期望对齐相同**（层重要性先验近似均匀），自适应采样实测无增益。→ 与我们实测「模块 δ/净变化近似均匀」一致。
- **AGZO / ZO-Act**（[2601.17261](https://arxiv.org/abs/2601.17261) / [2607.01125](https://arxiv.org/abs/2607.01125)）：用**层内激活的右奇异子空间**做扰动，理论证明 `cos` **严格高于各向同性**，OPT-13B 上有效、r=1 最优。

## 想法3 的潜在改进（写入待办）

- **AGZO/ZO-Act 激活引导低秩子空间**：把「想法2 的随机正交 U/V」换成**由输入激活 SVD 得到的子空间**（或把「想法3 的坐标块」换成激活子空间块），可在同 `d_eff` 下进一步抬 `cos`。列为想法3 的首选改进方向。

## 待办

- 接入 run_fed：`--lowrank-k/--lowrank-seed`（`.cuda()` 前 attach）、Z 持久化+重挂、ppl 评测子进程 attach+load、按低维重调 ZO lr/eps。
- 想法3 正式实验（块坐标/层轮转 + 激活子空间）参数表见后续聊天。
