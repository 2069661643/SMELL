# SMELL-v2-debug — 发现 B 混杂、撤回 B（P1: A3-only）、加 baseline eval

## 概述

上一轮 A3+B 运行中，用户敏锐指出：**归一化 v 后 var 骤降 → MORE_V 重采样被静默关闭 → 方向变噪声**。核实属实，B 引入混杂（A3 与 B 副作用无法分离），故按 P1 撤回 B，跑纯 A3（r=1），并在训练开始前加一次 baseline eval。

## B 混杂的证据

| 指标 | B 之前（samples=20, r=8） | A3+B（r=1, 归一化） |
|---|---|---|
| `var` | ~1–2（阈值 1，反复触发） | median 1.95e-8, max 1.52e-6（全 < 1） |
| `calculate more v` | ~87/轮 | **0** |
| `jvp` mean\|·\| | ~2.9 | 0.076 |
| `grad_norm` | median ~1341 | median 0.081 |

**机制**：`var` 是绝对量且刻度依赖扰动幅度。B 把每张量 `‖v‖` 从 ~64 压到 1，`var` 随之压到 ~1e-8，使绝对阈值 `var <= var_threthod(=1)`（`FedSgdAggregator.py:247`）恒成立 → MORE_V 重采样静默失效。

**判定**：估计仍无偏（非数学无效），但方差大、更噪声；且与 A3 效果混杂，无法干净归因。

## 代码改动（commit `7b4199b`）

1. **撤回 B**（`tc_transformer_trainer_distribute.py`）：归一化 v 的 for 循环改为 `SMELL 4 ... COMMENTED`（A3-only，重采样恢复）。
2. **新增 baseline eval**（`FedSgdClientManager.py`）：
   - 抽 `_maybe_eval_ppl(round_label)` helper，baseline 与每轮共用。
   - `handle_message_init` 在 `update_dataset` 后、round 0 训练前调用 `_maybe_eval_ppl("baseline")`。

## 归档

samples=20 run（`..._20260912_011704.log` + `checkpoint_...011708_r1..r25` + `global_lora_final` + `consensus_history`）→ `log/a0.1_sample20_len4k_datagoe/`、`checkpoints/a0.1_sample20_len4k_datagoe/`。
（A3+B 的 `..._094408.log` 不归档，按要求废弃。）

## 新 run（A3-only）

`--lora_r 1 --lora_alpha 2 --samples_per_round 100 --epochs 2`，其余不变（sparse 0.4 / c3 / alpha 0.1 / len4k / max_resample 200 / eval_ppl 50）。

**冒烟验证**：
- `trainable=1,048,576`（d: 8.4M→1.05M）✓
- baseline eval：`ppl=45.0380 eval_loss=3.8075 n=50` ✓
- B 已撤：jvp 恢复 ~±0.1–5.3，var 恢复 ~0.36–1.56，`calculate more v` 重新出现 ✓

## 待办

- [ ] 观察 A3-only 是否比 A3+B / samples=20 有更好的 `[EVAL-PPL]` 趋势（B 撤回后 d 仍 8× 小、重采样恢复）。
- [ ] 若仍平躺，再考虑 A4（只 q_proj, d=262K）或方案 D（结构化低秩扰动）。
- [ ] `var_threthod` 尺度相关是设计缺陷，后续若再用 B 类改动需改相对阈值。
