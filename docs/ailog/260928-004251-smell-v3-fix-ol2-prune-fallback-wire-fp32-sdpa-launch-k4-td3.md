# Ailog 260928-004251 — Phase N: 修复 sdpa_prune 末层 O(L²) 回退 + run_fed 接入 fp32/SDPA + 启动 k=4 TD-2'/TD-3

> 承接 `260927-200324`（SIGKILL 根因 + 未实施修复清单）。本 session **实施修复 1/3 + 接线**，并启动 k=4 稀疏 ZOO 队列。
> 决策依据：k-scan 显示收敛随覆盖单调（k=1 −0.038 / k=4 −0.169 / k=12 −0.280，同轮 0–32）⇒ 无可用中间粒度，但用户选定 **k=4 rotate4 sparsity ZOO（L2 D22, c30, cos≈0.10）作为 TD-3**，先由 TD-2' 门控。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 00:35 | 修复末层 O(L²) 回退 | `src/models/modeling_opt_smell.py` | `OptSdpaAttention.forward`（掩码门控）、`OptSdpaPruneAttention.__init__`（`_enable_sparse_mask=False`） | 剪枝路径末层/回退改走纯 causal SDPA，不再建 dense `(1,L,L)` fp32 掩码 |
| 00:36 | 新增 `--attn/--dtype` 接线 | `src/fed/run_fed.py` | `parse_args`（`--attn`/`--dtype`）、`main` 建模处（L411-426） | 可跑 fp32+SDPA/`sdpa_prune` 的 ZOO（原先硬编码 bf16+FA2） |
| 00:40 | 诊断支持多层块 | `temp/diag_cos_grid.py` | `--block-layers`、`blocks` 构造 | TD-2' 可测 k=4（4 层）子空间 d_eff=32768 |
| 00:42 | 新增本 ailog | `docs/ailog/260928-004251-...md` | — | 记录修复 + 接线 + 队列启动 |

## 变更 1：末层 O(L²) 回退修复

- 根因（`260927-200324` §二）：`OptSdpaPruneAttention._prune_ok` 豁免末层 → 回退 `OptSdpaAttention.forward` → 建 dense `(1,L,L)` fp32 加性掩码，torch 2.1.2 fp32 无 mem-efficient 加性掩码核 ⇒ math 核每头 `(h,L,L)`，16k≈16 GiB。
- 修复：`OptSdpaAttention.forward` 仅在 `getattr(self, "_enable_sparse_mask", True)` 时建掩码；`OptSdpaPruneAttention.__init__` 置 `self._enable_sparse_mask = False`。故剪枝类回退时 `is_causal=True, attn_mask=None`，走 mem-efficient O(L)。
- 保留 `OptSdpaAttention`（`--attn sdpa`）原掩码行为不变（本次不扩大改动面）。

## 变更 2：`run_fed` fp32/SDPA 接线

- `--attn {flash,eager,sdpa,sdpa_prune,sdpa_gather}`（默认 flash，旧行为不变）、`--dtype {bf16,fp32}`（默认 bf16）。
- 建模：`get_opt_qk(flash_attention=(attn=="flash"))`，非 flash 时 `config.attn_implementation=attn`；`from_pretrained(torch_dtype=model_dtype)`。
- **注**：`--eval-every` 的 G-PPL 子进程 `src/eval/ppl.py` 仍用 Jenga 原生 bf16+FA2 稀疏（部署语义），未随训练 dtype 改；后续如需一致再议。

## 冒烟验证（`temp/smoke_k4_sdpa_prune.json`）

- 命令：`diag_cos_grid.py --gpu 2 --attn sdpa_prune --dtype fp32 --seq 16384 --block-layers 20-23 --blocks k4 --n-clients 1 --ls 1 --ds 1 --cs 1`。
- 结果：**EXIT=0，无 OOM**；`block k4: d_eff=32768`；8.6s；**`peak_alloc_gb=18.26 / reserved=19.05`**。⇒ 修复生效，16k k4 稀疏 fp32 可跑（GPU2 剩余 ~29 GB 容纳）。

## 启动队列（`temp/run_k4_td2_td3_bg.sh`，pid 672437，仅 GPU2）

1. **TD-2'**：k4（layer20-23）、`sdpa_prune` fp32、seq=16384、L2 D22、c=1..30 ⇒ `temp/diag_cos_sparse_td2_k4.json`。
2. **门控**：`cos(k4,L2,D22,c30) >= 0.05` 才继续（dense 预测≈0.10，稀疏若 ≥ 半即算保真）。
3. **lr 预检**：1 client/1 sample、探针 `lr=4e-6` ⇒ 读 `delta_norm`，按 `lr = 4e-6 × 0.38 / delta_norm` 校准（对齐 BP k=4 实测 δ≈0.38–0.42）。
4. **TD-3**：`run_fed --trainer zoo --dtype fp32 --attn sdpa_prune --sparse 0.4 --zo-subspace layers --zo-layer-rotate --zo-layer-group 4 --max-clients 30 --local-steps 2 --zo-directions 22 --zo-eps 1e-3 --rounds 30 --eval-every 1`（**每轮 eval + save**）⇒ `logs/fed/zoo_k4_sparse/a01/`。

- 日志入口：`temp/logs/last_k4_td2_td3_dir.txt`、`.pid`；读 `metrics.jsonl`（driver.log 有块缓冲）。
- ETA：TD-2' ~1.5h；若过门控，TD-3 约 1.5–2.2h/轮 × 30 轮 ≈ 2–3 天。

## 待办

- [ ] TD-2' 完成后读 k4 稀疏 cos，与 dense `0.5√(cLD/d_eff)` 对比（判稀疏是否保方向）。
- [ ] 若 cos 过关：监控 TD-3 的 G-PPL 是否随轮下降（`eval_round*.json`），并看 lr 校准后的 `delta_norm`、是否 NaN。
- [ ] 若 cos 不过关：停机，回到融合 block-sparse（Jenga `flash_block.py`）或 `flex_attention` 方案。
- [ ] 决定修复 2（`OptSdpaAttention` 稀疏掩码改显式开关）是否需要（当前仅 prune 路径可用）。

## 环境

host `amax`（A40×4，仅 GPU2）；`smell-v2`（torch 2.1.2+cu118 / FA 2.4.2）。GPU2 邻位 sys21021 占 ~15 GB。

## 追加：train_loss 口径差异根因（09:40 调查）

现象：TD-3（ZOO, fp32+sdpa_prune）`train_loss≈5.33`，而 BP k-scan（bf16+FA2）≈3.61。子智能体 + 实测结论：**不是记录 bug，而是训练前向（注意力实现）不同**。

隔离实验（`run_fed` 1 client/1 sample；探针 `temp/probe16_*`、`temp/probeG_*`）：

| 配置 | 16k train_loss | 2048 train_loss |
|---|---|---|
| sdpa_prune，无 prune（sparse=1.0） | 3.303 | 3.383 |
| flash bf16 sparse=0.4（Jenga 原生） | 3.516 | 3.578 |
| sdpa_prune fp32 sparse=0.4，**随机 predictor** | 3.538 | — |
| sdpa_prune fp32 sparse=0.4，**训练后 predictor** | **5.371** | 3.582 |
| 同上 + 匹配 pos_lora+adapter | 4.261 | — |

- **主因**：加载训练后 predictor 后 16k loss 3.54→5.37（+1.83）；2048 无此问题 ⇒ **长度相关**。
- **机制**（`temp/probe_pred_sel.py`，8192，layer5/20）：训练后 predictor 的 `sum(dim=-2)` 分数 top-k **只选最后 ~40% 块**（n_blk=128,k=51：idx min≈69, med=102, max=127，first-half 占比 **0.00**；随机基线 0.47）。物理子集化丢掉前 60% token ⇒ 早期上下文全失，LM loss 暴涨。选块逻辑与 Jenga 原生一致（`modeling_opt.py:296-317`）⇒ 是 **predictor 训练目标/选块方向问题**，非 prune 实现 bug。
- 次因：predictor 训练用 `a01_pos_lora_500step`+adapter，TD-3 用 `a01_pos_only_500step`（无 adapter）⇒ 再 +0.9（5.37→4.26）。
- **影响**：TD-3 一直在**被污染的 16k 目标**上训练；G-PPL 平在 ~32 **不能**作为 ZOO 成败的干净判据。TD-2' 的 cos=0.197 亦在该前向下测得。
- **下一步候选**：① 停机 TD-3；② 重训/修正 predictor（或改用不物理丢 token 的 block-sparse）；③ 若只为验证 ZOO 可学，先用「随机 predictor / sparse=1.0 dense causal / Jenga 原生」重测 cos 与短程 G-PPL。
