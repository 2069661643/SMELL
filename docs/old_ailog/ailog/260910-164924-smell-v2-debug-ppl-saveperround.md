# SMELL-v2-debug — PPL 打印 + save_per_round + samples_per_round=0 全量

## 概述

本次任务（延续 **SMELL-v2-debug**）三项改动：
1. **回退 forward-mode AD**：恢复旧版中心差分 jvp（forward-mode AD 被 flash_attn 的 functorch 限制阻塞，reverse-mode 又违背「前向梯度省显存」初衷），并在**客户端**加入 PPL 计算打印。
2. **`--save_per_round <N>`**：每 N 轮保存一次带时间戳的 LoRA checkpoint，保留最后一次 `global_lora_final.h5`。
3. **`--samples_per_round 0` = 全量**（默认改为 0），checkpoint 目录默认改到**仓库根目录 `checkpoints/`**。

## 文件操作

### 修改 (6)

| 文件 | 改动 |
|---|---|
| `forward_training/utils/fwdgrad_utils.py` | `calculate_jvp_causal_lm` 回退为中心差分（移除 reverse-mode/autograd.grad、`torch.func.jvp` import） |
| `forward_training/tc_transformer_trainer_distribute.py` | 恢复 `functional_call` 的 `f` partial；新增 `_round_nll/_round_tokens` 累计 + `get_round_ppl()`/`reset_round_ppl()`；每 batch 打印 `PPL=exp(loss)` |
| `FedML/.../FedSgdClientManager.py` | round 结束（发 model）打印 `[SMOKE-PPL] Round X avg PPL = ...` 并清零 |
| `FedML/.../FedSgdServerManager.py` | 新增 `_save_lora_checkpoint(filename)`；`__init__` 记录 `task_start_time`；round 推进时按 `save_per_round` 保存 `checkpoint_<ts>_r<round>.h5`；最后一轮仍存 `global_lora_final.h5` |
| `experiments/.../initializer.py` | 新增 `--save_per_round`（默认 0）；`--samples_per_round` 默认 `None→0` |
| `experiments/.../run_lm_exps/smell_main.py` | 新增 `_REPO_ROOT`；`--checkpoint_dir` 默认 `_PACKAGE_ROOT/checkpoints/phase3` → 仓库根 `checkpoints/` |

## 测试结果（r2，均通过）

1. **PPL 打印**：`[SMOKE-PPL] Round 0 avg PPL = 28.4~29.9`、`Round 1 avg PPL = 62.3~63.8`（PPL 上升，符合前向梯度不收敛预期）；修复了 `labels[1:]`→`labels[..., 1:]` 切片 bug（否则 PPL 恒 inf）。
2. **save_per_round**：`--save_per_round 1` 产出 `checkpoint_<ts>_r1.h5`、`checkpoint_<ts>_r2.h5` + `global_lora_final.h5`，均保存到仓库根 `checkpoints/`。
3. **samples_per_round=0 全量**：`len(train_local_list[0])=21153`、`data_id now=1→2`（多 data_id 遍历，全量生效）。

## 结论 / 待办

- [ ] forward-mode AD（真前向梯度）仍待解决 flash_attn functorch 限制（eager/sdpa attention 或 FlashAttnFunc 补 forward-mode）。
- [ ] PPL 每 round 打印多次（MORE_V 重采样期间反复发 model）；若要「每轮最终一次」需把打印挪到 var 达标进入下一轮时。
- [ ] checkpoint 加载是 index-based（不校验参数名），建议后续改 name-based 匹配增强健壮性。
