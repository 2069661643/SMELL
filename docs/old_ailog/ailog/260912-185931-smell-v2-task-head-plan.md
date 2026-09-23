# SMELL-v2-debug — 高-M 极限实验收尾 + 任务头方案

## 概述

`a0.1_sample1_len4k_datagoe`（samples=1 / epochs=1024 / M=3072 / LoRA r=1）跑 4 轮（~6h13m）后停止并归档。**即使 M=3072（cos≈5.4%），PPL 仍平躺**（baseline 45.038 → r0 45.295 → r1 45.327 → r2 45.009 → r3 44.996），再次确认单靠增大 M 无效。据此产出「仅训练任务头」方案。

## 结果（高-M 极限实验）

| 点 | PPL | vs baseline |
|---|---|---|
| baseline | 45.0380 | — |
| round 0 | 45.2947 | +0.257 |
| round 1 | 45.3268 | +0.289 |
| round 2 | 45.0089 | −0.029 |
| round 3 | 44.9955 | −0.043 |

- 每轮 1 个 data_id × 3072 batch ≈ **84 min/轮**；无重采样（M 大，var 直接达标）。
- 结论：M 从 48→3072（8×），cos 从 0.7%→5.4%，**PPL 无任何下降趋势** → 瓶颈不是 M，而是 M/d 量级（d=1.05M）。

## 归档

`log/smell-v2-debug-len4k_20260912_123453.log` + `checkpoint_20260912_123458_r1..r4.h5` + `consensus_history_20260912_123458.h5`
→ `log/a0.1_sample1_len4k_datagoe/`、`checkpoints/a0.1_sample1_len4k_datagoe/`。

## 产出方案

`docs/SMELL/phase5_task_head_plan.md`：仅训练任务头（FwdLLM 式）+ Adapter(δx) / soft-prompt / LoRA 的 trainable 参数估算与 SNR 对比：

- **A1 线性分类头**：d=**114,716**，cos(M=3072)=**16.4%**（最贴近 FwdLLM，最小 d）。
- **D3 lm_head LoRA r=1**：d=**36,096**，cos=**29.2%**（保持 causal-LM，改动最小）。
- **C1 soft prompt L=8**：d=**32,768**，cos=**30.6%**（保持 causal-LM，最佳 SNR）。
- 现 LoRA r=1 qkvo：d=1,048,576，cos=5.4%（对照）。

推荐：先用 A1 验证「小 d 下前向梯度可收敛」；若要保 PPL 口径用 D3/C1。

## 待办

- [ ] 选定方案后实现（注意 R1：`var_control` 的 `layer_id_for_check` 依赖 q_proj，换模块需替换；R2：server/client 参数枚举顺序一致性）。
- [ ] 归档目录 `a0.1_sample1_len4k_datagoe` 仅含 4 轮（未跑满），结论以 PPL 平躺为准。
