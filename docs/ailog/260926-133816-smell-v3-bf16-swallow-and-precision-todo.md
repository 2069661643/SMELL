# Ailog 260926-133816 — SMELL-v3: bf16 吞增量 / FA 阻挡精度重测 + 精度修复 TODO

## 概述

块级 cos 打点（`temp/diag_cos_grid.py`，见 `temp/diag_cos_grid.json`）实测**所有配置 cos≈0**、且 `√(cLD/d_eff)` 公式高估 15–100×。追根到底是**经典 bf16 吞增量**问题（old_ailog `260914-103130` 已记录同款），而非维度。尝试 FP64 重测被 **FlashAttention 只支持 fp16/bf16** 挡住（关 FA 走 eager → 16k O(seq²) OOM；fp64 还撞 mask 溢出）。v2 的 `_promote_lm_head_fp32` 修正**不能直接套用**（我们 LoRA 在 q/k/v/out，非 head）。以下为结论与待办。

## 打点结果（`temp/diag_cos_grid.json` 摘要）

| block | L | D | c | cos_meas | cos_formula | ratio |
|---|---|---|---|---|---|---|
| layer0 | 2 | 32 | 30 | 0.0318 | 0.484 | 0.07 |
| layer0 | 1 | 8 | 30 | 0.0075 | 0.171 | 0.04 |
| full | 2 | 32 | 30 | -0.0001 | 0.099 | -0.00 |
| full | 1 | 8 | 30 | -0.0036 | 0.035 | -0.10 |

⇒ 真实 cos ≤0.032；加 c/L/D 均不涨；公式作废。

## 根因：bf16 吞增量（经典问题）
- 有限差分信号 `ΔL = 2·eps·(g·v)`；layer0 `‖g‖≈0.02` ⇒ `g·v≈2.2e-4` ⇒ **`ΔL≈4.4e-7`**。
- bf16 loss（≈3.2）ULP ≈ **0.016** ⇒ 信号小 4 个数量级 ⇒ **ZOO 差分全是量化噪声**。
- old_ailog `260914-103130`：`base + lora_delta` 输出 cast 到 bf16 时，`‖ΔW‖/‖W‖≲3.9e-3` 的增量被整体舍掉；autograd 因 straight-through 仍给非零梯度 ⇒「梯度非零但 loss 不响应」。并明确留待办：**r=8 q_proj 是否也被吞需单独量**。

## FA 为何阻挡精度重测
- FA2 内核**只支持 fp16/bf16** ⇒ 高精度必须关 FA → eager attention 建 `[b,heads,seq,seq]`（O(seq²)）；实测 **fp32 seq=4096 即 OOM**，16k 必 OOM。
- fp64 还撞 `torch.full((t,t), finfo(fp64).min)` 溢出（mask 构造）。

## v2 解法可套用性
- `_promote_lm_head_fp32` 只针对 v2 的 lm_head-LoRA，**不适用**我们的 q/k/v/out LoRA。
- **可复用**：① v2 的「bf16 吞增量指纹」诊断（`‖full−(base+lora_fp32)‖`、不同取值个数）；② v2 结论：**问题来自 base 权重是 bf16，只改 FD dtype 无效** ⇒ 必须让 LoRA 相关前向链路整体 fp32/fp64。

## 待办（TODO）

### T3. 精度修复与验证（高优先）
- **A（先做，~10min）**：对 r=1 `q_proj`（及 out_proj）跑 v2 同款 **bf16 吞增量指纹**，确认是否被吞。
- **B（中，~1h）**：`seq≤2048` + **fp32 + eager** 重测 cos（eager 在 2048 可容纳）——回答「高精度下 cos 是否显著抬升」。
- **C（治本，工作量大）**：把注意力从 FA 换成 **`torch.nn.functional.scaled_dot_product_attention` 的 mem-efficient 后端**（支持 fp32、O(seq) 内存），实现 **16k fp32** 前向；fp64 另需修 mask 溢出。
- 若 A/B 证实「高精度 cos 显著抬升」，则 v3 ZOO 路线需**整体改高精度前向**才有意义。

### T1（既有）：AGZO/ZO-Act 激活引导子空间原型；T2：想法2 低秩再参数化接入。

## 环境
host `amax`（A40×4）。GPU2 曾跑诊断（已停）。`smell-v2` 栈。当前无我们的任务在跑。
