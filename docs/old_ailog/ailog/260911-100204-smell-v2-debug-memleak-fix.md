# SMELL-v2-debug — 显存泄漏定位与修复 + log 训练时间戳

## 概述

上次长跑（`samples_per_round=20` + `var_threthod=1` + 非 pack + `lr_decay`）首次出现收敛（PPL 51.4→40.2→38.9），但 round 3 的 data_id 9 触发 var 平台期（重采样 199 次），最终 CUDA OOM 后僵死。本次定位并修复泄漏源，并在 log 开头加训练时间戳；不启动训练，等待用户确认后再跑。

## 泄漏根因（已定位）

`perturbation_sampling` 的 `grad_pool` 用 `append(self.grad)` 累积，而 `self.grad` 每次 `train_model` 都重新初始化为 8.4M 参数的 GPU tensor（`tc_transformer_trainer_distribute.py:151`）：

- 每次重采样 `grad_pool` 多累积一份 16MB 的 grad 引用；
- data_id 9 重采样 199 次 → `grad_pool` 累积 ~3.2GB；
- 显存从 12.6GB 单调涨到 19GB（泄漏 ~13MB/次重采样），最终叠加 `sys21021` 占用的 21.4GB → OOM。

## 文件操作

### 修改 (3)

| 文件 | 改动 |
|---|---|
| `forward_training/tc_transformer_trainer_distribute.py` | `grad_pool` 从 `append` 改为**累积和**（`grad_pool[i] += g`，O(1) 显存），语义等价（old_grad 方向不变） |
| `FedML/.../FedSgdClientManager.py` | 两处 `old_grad = grad_aggregete(grad_pool)` 改为 `old_grad = grad_pool`（grad_pool 已是累积和，无需再求和） |
| `experiments/.../run_lm_exps/smell_main.py` | 新增 `import time`；FedML_init 后打印 `[TRAIN-START] <时间戳>`（rank 0），log 开头记录真实训练开始时间 |

### 归档

`log/smell-v2-debug-longrun2.log` → `log/smell-v2-debug-longrun2-converged-then-OOM.log`（保留收敛数据供后续分析）。

## 结论 / 待办

- [ ] 泄漏修复已就绪，但**未启动训练**（等待用户确认）。
- [ ] `grad_for_var_check_list` 仍有次要累积（重采样期间 ~38MB，var 达标即清理，可接受）。
- [ ] 后续训练观察点：显存是否稳定在 ~13GB（不再单调涨）、`max_resample` 是否还需要（泄漏修好后可能不再 OOM）。
