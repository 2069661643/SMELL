# Ailog 260927-130020 — SMELL-v3: thresh 口径确认 + token-prune(gather) 原型 + TD-2 B1/先导BP 启动

## 概述

查清 Jenga `thresh` 口径并**修正 TD-1 的标注误差**；实现 **token-prune（gather 式）原型**并 benchmark（结论：**gather 不划算，dense fp32 SDPA 在 16k 已 BP 可行**）。**TD-2 B1（稀疏 cos）与先导 BP（BP 层轮转）已串行启动**；TD-2 因 4096 OOM 自动降级到 **seq=2048**。

## A. `thresh` 口径（确认）

- `get_opt_qk` 把 `thresh` 存为 `config.sparse`（`third_party/Jenga/src/jenga/utils/config_utils.py:64-66`）。
- 消费于 `JengaForMemoryTest/Jenga/src/jenga/models/modeling_opt.py:300-306`：`q_len_now=int(sum_q.size(1)*config.sparse)` → `torch.topk(sum_q, q_len_now, largest=True, dim=-1)`（`:313-317` 展开 `idx*64+base` 到 token）。
- **`thresh=0.4` = 保留 top-40% 的「query 块」**（块级选、token 级剪）：`n_blocks=256` ⇒ 保留 **102 块 ×64 = 6528/16384 token（~39.8%），剪 60%**。
- 作用范围：仅 `layer_idx != last` 且 `layer_idx >= num_layers//2 - 1`（**上半层**；前半层全留）。
- **修正**：TD-1 的 `OptSdpaAttention` 稀疏是「每 query 块 top-k 个 **KV 块**」，与 Jenga 的「全局 top-k **query 块**」**轴不同**；`modeling_opt_smell.py:450-453` 注释「Jenga 同口径」不成立，需修正。

## B. token-prune（gather）原型 + benchmark

- 新类 `OptSdpaSparseGatherAttention`（`src/models/modeling_opt_smell.py:585-711`，开关 `config.sparse_gather`，另注册 `"sdpa_gather"`）：块分（predictor→q·k 退化）、每 query 块 causal 内 top-k、`torch.gather` 收 KV、`pool×(k·pool)` 小块 SDPA、`checkpoint(use_reentrant=False)` 包 chunk；非法/past/cross → dense。`py_compile OK`。
- benchmark（`temp/bench_sparse_gather.py`，1 层、bsz=1、fp32）：**16k 单层 fwd+bwd**：dense **0.54s**、gather **3.0s（~5.6×慢）**、L×L 掩码 OOM。正确性：`thresh=1` vs dense **逐位一致**。
- **结论**：dense SDPA mem-efficient 本身已 **O(L)** 且快 ~5×；gather 的逐块循环+checkpoint 使其**不划算**。**16k BP 直接用 dense fp32 SDPA 即可（~0.54s/层 ⇒ ~13s/样本）**；要省算力应走 fused block-sparse（Jenga `ops/flash_block.py` triton）或 torch≥2.5 `flex_attention`，而非 gather。
- **指导**：后续 BP 用 dense fp32 SDPA；gather 仅在必须「物理剪 token」时作为正确性参照。

## C. 弱推定（写入，明确标注）

- 已验证：fp32 SDPA 在 **16k 单层** forward 2.9s、BP fwd+bwd ~0.54s/层；TD-2 稀疏 cos 因 L×L 掩码在 16k/4096 OOM，降级到 seq=2048。
- **弱推定**：**稀疏对 cos 的影响随 seq 变化不大 ⇒ 以 seq=2048 的稀疏 cos 近似外推到 16k**（未验证，标注为弱推定，待后续 fused block-sparse 或 FlexAttention 复验）。

## D. 启动状态（串行脚本 `temp/run_td2b1_then_bplayerrot_bg.sh`）

- **先导BP 预检通过**：config 全对（`zo_layer_rotate=True, count=0(全24层), local_steps=4, rounds=150, lr=1e-4, max_clients=30, pos_only, 无 adapter`），启动 30s 无错后 kill，GPU2 回落基线、无孤儿。
- **联合已启动**：Stage1 **TD-2 B1** 运行中；**seq=4096 OOM → 自动降级 seq=2048**（`temp/diag_cos_sparse_s2048.json`）；ETA ~27min。Stage2 随后 **先导BP（150 轮，~5h）**。
- 日志 `temp/logs/td2b1_pilotbp_260927-125457/driver.log`；`last_td2b1_pilotbp_dir.txt/.pid`。

## 变更
- `src/models/modeling_opt_smell.py`：TD-1 稀疏掩码 + `OptSdpaSparseGatherAttention`（本提交）。

## 环境
host `amax`（A40×4）。GPU2 跑联合任务。`smell-v2`（torch 2.1.2+cu118 / FA 2.4.2）。
