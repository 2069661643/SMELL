# SMELL-v2-debug — 梯度归一化 + lr_decay + var_threthod=0.1（存档，未收敛）

## 概述

本次任务（延续 **SMELL-v2-debug**，针对 B1 不收敛/NaN）：在梯度归一化（方案 2）基础上，回滚 v 归一化，改用非 pack 短数据，新增线性递减学习率 `--lr_decay`，并将 `var_threthod` 调回 0.1。实验 r0-19 **全程无 NaN**，但 **loss 不降反升（3.72→4.23）**，前向梯度方向噪声问题仍未解决。此为存档 commit，下轮对话继续改进。

## 文件操作

### 修改 (2)

| 文件 | 改动 |
|---|---|
| `FedML/.../FedSgdAggregator.py` | (1) `var_threthod` 1 → 0.1；(2) 全局 L2 梯度归一化 + `clamp(-1,1)`（防噪声梯度爆炸）；(3) `--lr_decay` 线性递减逻辑 0.01→0.00001 |
| `experiments/.../initializer.py` | 新增 `--lr_decay` 开关（store_true） |

### 回滚 (1)

`forward_training/tc_transformer_trainer_distribute.py`：回滚上一版加的 **v 归一化**（`v_dict[n] = v_dict[n] / norm`），恢复 `randn_like`。原因是归一化后 `var` 量级骤降导致 `var_threthod` 失配。

## 实验配置与结果

配置：`llama_sparse` + `--sparse 0.4 --enable_consensus`，`--lr 0.0001 --server_lr 0.001`，`--comm_round 31`（r30），`--samples_per_round 1 --lr_decay`，`var_threthod=0.1`，数据 **非 pack** `goemotions_1k_data.h5`（43410 train 短样本，有效 token ~15）。

结果（r0-19，20 轮，在 round 19 按指令停止）：

1. **NaN = 0 次**（梯度归一化 + clamp + lr_decay + var_threthod=0.1 使训练彻底稳定）。
2. **loss 不收敛**：前 10 轮均值 3.715，后 10 轮均值 4.231（不降反升）。
3. **avg_resample = 41.5 次**（波动 0~160，var 在 0.1 阈值附近有平台期，r7/r9/r12 分别 160/110/96 次）。
4. `--lr_decay` 生效：learning rate 0.01 → 0.003673（round 19 时线性递减）。

## 结论 / 待办

- [ ] 稳定性已解决（无 NaN），但收敛未解决。根因不变：前向梯度方向噪声（信噪比 ≈ sqrt(M/8.4M)），MORE_V 平均 ~41 次远不足以逼近真实梯度方向。
- [ ] 下一步候选（下轮讨论）：① 降低有效维度（前向梯度只作用于 lora_B / 逐层 block 扰动）；② 换梯度来源（LoRA 反向梯度 / SPSA 方差缩减）。
