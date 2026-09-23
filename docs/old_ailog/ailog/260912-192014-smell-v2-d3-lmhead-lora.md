# SMELL Phase 5 — D3：只训 lm_head LoRA（CLI 可开关 + 可调秩）

## 概述

依据 `docs/SMELL/phase5_task_head_plan.md`，实现方案 **D3**：冻结整个 Llama2-7B，只训 **lm_head 的低秩增量**（`W_head + (α/r)·B·A`）。d 从 LoRA r=1 的 1.05M 降到 **144,384（r=4）**，`cos θ=√(M/d)` 相应提升（M=3072 时 ~14.6%）。

## CLI（新增）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lora_lm_head` | off | 开启 D3：只给 `lm_head` 加 LoRA，忽略 `--lora_r/--lora_target_modules` |
| `--lora_lm_head_r` | 4 | lm_head LoRA 秩 |
| `--lora_lm_head_alpha` | 0 | alpha；<=0 → `2*r` |

## 代码改动（commit `c991220`）

| 文件 | 改动 |
|---|---|
| `initializer.py` | 新增 3 个 CLI 参数；`_build_lora_config` 支持 lm_head 分支（`target_modules=["lm_head"]`） |
| `run_lm_exps/smell_main.py` | Server `LoRAOnlyModel` 支持 lm_head 分支（`lora_A_lm_head [r,4096]` / `lora_B_lm_head [32000,r]`，顺序 A→B 与 Client 一致） |
| `training/fed_trainer_transformer.py` | `_translate_lora_param_names` 增加 `lora_{A,B}_lm_head → base_model.model.lm_head.lora_{A,B}.default.weight` |
| `forward_training/tc_transformer_trainer_distribute.py` | `var_control` 检查目标在 D3 下改为 `lm_head.lora_A`（原为 `layers.14.q_proj.lora_A`） |

**Client 实测**（`create_model`，`--lora_lm_head_r 4`）：`trainable=144,384`，参数名 `base_model.model.lm_head.lora_A/B.default.weight`，形状 `(4,4096)`/`(32000,4)` ✓

## 新增脚本

`script/RUNME-v2-len4k-d3-lmhead.sh`：D3 + 其余同上次极限 M（samples=1 / epochs=1024 / max_resample=50 / sparse 0.4 / c3 / alpha 0.1 / len4k）。

## 待办 / 风险

- [ ] Server `LoRAOnlyModel` 与 Client 参数数量/顺序一致性（冒烟确认 aggregate 正常）。
- [ ] eval 脚本（`eval_ppl.py`/`eval_train_vs_test.py`）若用于 D3 checkpoint，需同样传 `--lora_lm_head --lora_lm_head_r 4`。
- [ ] r=4 容量仍受限（秩 4）；若 PPL 有下降趋势再调大。
