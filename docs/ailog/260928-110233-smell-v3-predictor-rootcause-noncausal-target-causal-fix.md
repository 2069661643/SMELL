# Ailog 260928-110233 — Phase N: predictor 选块病根（非 causal target）+ causal 目标修复

> 承接 `260928-004251`（train_loss 口径差异）。本 session 定位：**训练后 predictor 的 top-k 只选尾部块，根因是 Jenga 训练目标 `block_attn_pool` 非 causal**；在 OPT（post-norm、未归一化残差）上退化为位置/范数驱动。已改 `train_predictor.py` 加 causal mask 并重训（step4）。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 10:20 | 核查链路（只读） | `src/models/modeling_opt_smell.py`、`jenga/models/predictor.py`、`modeling_opt_train_predictor.py`、`ops/flash_block.py` | — | 确认 predictor 确实被调用、轴正确、bias 为 0、target 非 causal 与 Jenga 一致 |
| 10:24 | 重训 predictor（pos_only，非 causal target） | `checkpoints/predictor/step3_a01_pos_only/` | — | **未修复**：仍选尾部，16k loss 5.08 |
| 10:50 | 定位：真·causal 注意力选前段 | `temp/probe_oracle.py` | — | predictor≈非causal oracle(corr 0.97, 尾部)，causal oracle=前段(frac 0.90, corr −0.35) |
| 10:58 | 验证修复方向 | `temp/probe_keep.py` | — | keep-early40% loss=**3.53** vs keep-late40%=**5.08** |
| 11:00 | 修复：target 加 causal mask | `src/train/train_predictor.py` | `block_attn_pool_fixed`（L64-84） | 训练目标改为 causal，topk 将选前段（待验证） |
| 11:02 | 重训（causal target） | `temp/run_predictor_step4_causal_bg.sh` → `checkpoints/predictor/step4_a01_pos_only_causal/` | — | ETA ~34min |

## 一、链路核查（全部通过）

- 打分**确实走 predictor**：TD-3 `config.json` `predictor_loaded_tensors=144`；`_prune_block_scores` 在 16k（n_blocks=256）形状校验通过，`scored_by="predictor"`；无静默 fallback。
- **轴正确**：predictor 输出与 `block_attn_pool` 目标均为 `[b, query_block, key_block]`；`sum(dim=-2)`+topk 选 **key 块**（Jenga 注释 "query block" 是命名误导）。无转置 bug。
- **bias 为 0**：144 个 predictor bias `mean_norm=0.00000`，零化后 loss/选块不变 ⇒ 训练 `bias=False`/推理 `bias=True` 的不匹配**非病因**。
- Jenga 原始 `ops/flash_block.py` 的 `block_attn_pool` 也**无 causal mask**（Triton kernel 对所有 (row,col) 块计算），与我们的 monkeypatch 一致 ⇒ 无实现差异。

## 二、病根：非 causal target + OPT 未归一化残差 → 位置捷径

- `temp/probe_corr.py`（8192，layer5/20）：predictor 块分与**块序号**相关 `0.94/0.88`、与**块 L2 范数**相关 `0.84`，与「块均值范数」相关 0.00；对输入做 LayerNorm 后选块**不变** ⇒ 位置信息编码在残差**方向**里，非单纯幅度。
- Jenga 的 Llama 在 self_attn **前**做 `input_layernorm`（`modeling_llama.py:806-809`），predictor 拿归一化输入；OPT `do_layer_norm_before=false`（`modeling_opt.py:469-484`）⇒ predictor 拿**原始残差**，而 OPT 学习式绝对位置嵌入范数沿位置递增（base 表 Pearson 0.906）。
- 训练目标 `amax ReLU(QK)` 非 causal、无归一化 ⇒ 对 OPT 退化为「越靠后越大」，predictor 忠实学到它。

## 三、oracle 对照（决定性）

`temp/probe_oracle.py`（8192，layer20，k=51）：

| 打分 | 选中块 min/med/max | first-half 占比 | corr(pos) |
|---|---|---|---|
| predictor | 73/102/127 | 0.00 | +0.88 |
| oracle **非 causal** | 71/102/127 | 0.00 | +0.94 |
| oracle **causal** | 0/39/74 | **0.90** | **−0.50** |

- `corr(predictor, oracle_noncausal)=0.972`（faithful）；`corr(predictor, oracle_causal)=−0.350`（**方向相反**）。
- **真·causal 注意力集中在前段**，而非 causal 目标集中在尾部。

## 四、修复验证 + 实施

- `temp/probe_keep.py`（16k）：强制 keep-early40% → loss **3.53**（≈随机/flash）；keep-late40% → loss **5.08**（= 训练后 predictor）。⇒ 只要选块与 causal 对齐，loss 恢复正常。
- 修复：`src/train/train_predictor.py` 的 `block_attn_pool_fixed` 对 `key_pos > query_pos` 做 causal mask（`attn * (k_idx <= q_idx)`）。`py_compile` OK。
- 重训：`temp/run_predictor_step4_causal_bg.sh`，pid 1200557，GPU2，11:02 起，ETA ~11:37 ⇒ `checkpoints/predictor/step4_a01_pos_only_causal/`。

## 待办

- [ ] step4（causal）训完：`temp/probe_bias.py` 测 16k 选块是否转为前段、loss 是否回到 ~3.5；`temp/probe_oracle.py` 复核与 causal oracle 的一致性。
- [ ] 若达标：以 `step4_a01_pos_only_causal/predictor.pth` 重跑 TD-2'（k4 稀疏 cos）+ TD-3（k4 sparsity ZOO）。
- [ ] 判定该 dev 是否要回灌到 `OptSdpaAttention`（`--attn sdpa` 的块稀疏掩码同病）。
- [ ] 记录：Jenga 对 Llama（pre-norm）可用、对 OPT（post-norm）需 causal target，属 OPT 适配性问题。

## 五、评估结果（step4 causal predictor，已验证 ✅）

| 指标 | step2/step3（非 causal） | **step4（causal）** |
|---|---|---|
| 16k train_loss（`probe_bias 16384`） | 5.37 / 5.08 | **3.4394** |
| 选块 first-half 占比（layer5/20） | 0.00 / 0.00（全尾） | **0.59 / 0.52**（跨全序列） |
| corr(pred, oracle_causal)（8192） | −0.350 | **+0.412** |
| corr(pred, oracle_noncausal)（8192） | +0.972 | −0.242 |

- 修复生效：16k loss 回到 ~3.44（≈随机 3.54 / flash 3.52 / 不剪 3.30），选块不再集中尾部，与 causal 方向正相关。
- 复现命令：
  - `PRED/PCS=checkpoints/predictor/step4_a01_pos_only_causal/* python temp/probe_bias.py 16384`
  - `PRED/PCS=... python temp/probe_oracle.py 8192 20`
- 结论：**Jenga 非 causal 目标对 OPT（post-norm 未归一化残差）不适配；加 causal mask 后 predictor 才选出 causal 需要的块。**

## 六、影响 / 待办

- [ ] 用 `step4_a01_pos_only_causal/predictor.pth` 重跑 **TD-2'**（k4 稀疏 cos）→ 过门控后重跑 **TD-3**（k4 sparsity ZOO, rotate4, L2D22, c30, 每轮 eval）。（按用户要求：长跑暂不启动）
- [ ] 判定 `OptSdpaAttention._build_sparse_attn_mask`（`--attn sdpa` 路径）是否同病（它用真实 q/k 的块掩码，非 predictor，影响面不同）。
- [ ] 是否把 causal-target 修复固化进 `train_predictor.py`（当前已改）并补 `AGENTS.md`/handoff 的「Jenga 对 OPT 的 predictor 需 causal target」条目。

## 环境

host `amax`（A40×4，仅 GPU2）；`smell-v2`（torch 2.1.2+cu118 / FA 2.4.2）。TD-3 已停（`logs/fed/zoo_k4_sparse/a01/` 保有 adapter_round000–004）。step4 predictor 已就绪于 `checkpoints/predictor/step4_a01_pos_only_causal/`。
