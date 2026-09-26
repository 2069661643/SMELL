# Ailog 260926-182711 — SMELL-v3: bf16 损失量化确认 + Step A(fp32 短序列) 进行中；B/C/D 待办

## 概述

Phase 0 与大 D 验证**共同锁定根因**：ZOO 有限差分被 **bf16 损失量化台阶（0.0156）** 完全吞没（真信号 ~1e-6），与维度/D/L/lr 无关；**加大 D 无效**。下一步按计划先做 **Step A（短序列 fp32 下验证损失恢复可分辨）**，B/C/D 见待办。

## 结果

### Phase 0（`temp/diag_bf16_swallow.py`，A1 训好的 BP r=1 adapter）
| 观测 | 值 |
|---|---|
| `‖lora_fp32‖/‖base‖` (q_proj, layer0) | 4.93e-2（**未被模块吞**） |
| `fraction(qmod_out==base)` | 0.0005 |
| `loss0` | 2.906250 |
| `dL_meas`(ε=1e-3/1e-2/1e-1) | **0 或 −0.015625**（量化到 1/64） |
| `dL_pred`(2ε·g·v) | 1e-8 ~ 3e-4 |

⇒ **损失量化 0.0156 ≫ ZOO 信号 1e-6**；根因是「损失精度」，不是模块吞增量。

### 大 D 验证（`temp/diag_largeD.json`，layer0, L=1, fp32? 否—bf16）
| D | c | cos_meas | cos_formula |
|---|---|---|---|
| 8 | 1 | 0.0034 | 0.031 |
| 128 | 3 | -0.0153 | 0.217 |
| **1024** | 1 | 0.0107 | 0.354 |
| **1024** | 3 | -0.0007 | 0.612 |

⇒ D=1024 仍 ≈0；**大采样无效**。

## 计划

### Step A（**进行中**，子智能体交付）
`seq=2048 + fp32 + eager`（允许丢稀疏）重跑损失量化/差分：
- 测 `dL(ε)` 是否 > fp32 ULP(~3e-7)、是否与 `dL_pred` 相关；
- 若恢复可分辨（并 cos 抬升）⇒ 精度假设成立。
- fp64 可另在短序列做金标准（需修 `finfo(fp64).min` mask 溢出）。

## 待办（TODO）

### TB. fp32 16k 前向（子智能体）
- 在 `src/models/modeling_opt_smell.py` 新增 `OptSdpaAttention`（`F.scaled_dot_product_attention(is_causal=True)`，**mem-efficient 后端**，fp32、O(seq)），注册进 `OPT_ATTENTION_CLASSES`，`config.attn_implementation="sdpa"`。
- 验证 **16k fp32** 前向显存/时间；确认 LoRA 扰动在 loss 可见。
- 退路：SDPA 16k 不可行 → 诊断降 seq≤4096。

### TC. fp32 下重测 cos 打点
- `diag_cos_grid` 加 `--dtype fp32 --attn sdpa`，单层/全维跑 `cos(c,L,D)`。
- 判据：fp32 下 cos 是否 ≫0（≥0.1）。

### TD. 决策
- cos 显著 → 把 fp32 前向接入 `run_fed` 的 ZOO（`--dtype fp32 --attn sdpa`），重调 lr/eps，重跑单层/前4层长程。
- cos 仍≈0 → 非精度问题 → 转 AGZO/ZO-Act 或 BP-based FL 主线。

## 风险
- dense SDPA 丢 Jenga 稀疏：诊断可接受，主线需把稀疏逻辑整合到 fp32 路径。
- fp32 使显存/时间 ~2×；16k 可行性待 TB 实测。
- fp64 全序列不可行（mask 溢出 + O(seq²)）。

## 环境
host `amax`（A40×4）；GPU2 跑诊断。`smell-v2` 栈。
