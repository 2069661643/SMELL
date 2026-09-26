# Ailog 260926-183811 — SMELL-v3: Step A 确认 bf16 量化根因；TB/TC 待执行（稀疏留到 TD）

## 概述

**Step A 证实**：bf16 损失量化就是 ZOO 差分失效的根因；**fp32 前向恢复了有限差分信号与 cos**。下一步 **TB（16k fp32 + SDPA 注意力，dense）**、**TC（fp32 下 cos 打点）**；**稀疏留到 TD** 再接入（先隔离精度变量）。

## Step A 结果（`temp/diag_bf16_swallow.py` 扩展 `--dtype/--flash`；seq=2048，adapter=A1 round299）

**损失 ε 扫描（fp32 vs bf16）**
| ε | dL_meas(fp32) | dL_pred(2ε·g·v) | bf16 |
|---|---|---|---|
| 1e-4 | +9.54e-7 | +5.57e-7 | ~0 |
| 1e-3 | +2.38e-7 | +6.66e-7 | 0 / −0.0156 |
| 1e-2 | +1.478e-5 | +1.442e-5 | +0.0156 |
| 1e-1 | +6.48e-4 | +6.25e-4 | 0 |

⇒ fp32 下 `dL_meas` 与真方向导数**同号同量级**；bf16 恒为量化台阶（0.0156）且与真值无关。**根因确认。**

**cos（layer0, seq=2048）**
| D | 理论 √(D/d) | fp32 | bf16 |
|---|---|---|---|
| 8 | 0.031 | 0.017 | 0.015 |
| 32 | 0.063 | **0.086** | 0.013 |

⇒ fp32 的 cos 达理论量级 ⇒ ZOO **可救**，前提是高精度损失。

**限制**：seq=4096 fp32 eager OOM（共享卡仅 ~28.6GB 可用；独占 A40 应可）；fp32 全模型显存 ~bf16 的 6×。

## 决策

- **稀疏接入点 = TD**（不在 TB/TC 带，先隔离精度变量）；TD-1 以「top-k 块掩码 → SDPA」形式在 fp32 重接稀疏（Jenga 稀疏绑 FA2/bf16，须改写），TD-2 稀疏下复验 cos，TD-3 生产。
- TB/TC **dense、fp32**。

## TB/TC 计划（待执行，子智能体）

- **TB**：`src/models/modeling_opt_smell.py` 新增 `OptSdpaAttention`（`F.scaled_dot_product_attention(is_causal=True)`，mem-efficient 后端，fp32、O(seq)），注册进 `OPT_ATTENTION_CLASSES`，`config.attn_implementation="sdpa"`；验证 **16k fp32** 前向显存/时间；若 OOM 如实上报（不硬撑）。
- **TC**：`diag_cos_grid` 加 `--dtype fp32`（+ SDPA），fp32 跑单层/全维 `cos(c,L,D)`；判据 cos ≥0.1；OOM 则降 seq 并上报。

## 环境
host `amax`（A40×4）。GPU0/3 满、GPU1 95% util、**GPU2 29GB 空闲** ⇒ TB/TC 用 GPU2。`smell-v2` 栈。
