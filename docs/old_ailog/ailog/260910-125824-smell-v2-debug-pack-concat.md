# SMELL-v2-debug — sparse attention_mask 修复 + pack 简单拼接 + samples_per_round

## 概述

本次任务（延续 **SMELL-v2-debug**）：参照 `src/jenga` 修复 sparse 模型 token elimination 与 flash attention `attention_mask` 脱节的 bug，改用 pack 简单拼接数据 + `samples_per_round=1`，跑通 r30/c3 实验并统计 loss / avg_resample。实验跑通（无崩溃），但 loss 未收敛、round 8 起 NaN 退化（B1 依然存在）。

## 关键根因（flash attention 越界）

sparse 模型层 15-30 做 token elimination（topk 0.4，把 seq 1024 稀疏到 384），但 `attention_mask`（`_update_causal_mask` 在 flash_attention_2 下有 padding 时返回 2D `[bsz, seq]`）未同步 slice。`_upad_input` 用 1024 长度的 mask 的非零位置去 index 稀疏后 384 长度的 k/v，有效 token 超过 384 时索引越界（device-side assert）。

- 之前非 pack 数据（GoEmotions 短句，有效 token ~15 < 384）恰好不越界，掩盖了 bug。
- pack 简单拼接后有效 token ~1020 > 384，必然崩溃。

参照 jenga：`src/jenga/src/jenga/models/modeling_llama.py` 的 `_update_causal_mask` 同样返回 2D mask，但 jenga 评测数据无 padding（attention_mask=None）避开了 unpad。SMELL 训练数据必有 padding，故需在 token elimination 后同步 slice。

## 文件操作

### 修改 (4)

| 文件 | 改动 |
|---|---|
| `FwdLLM/model/modeling_llama_sparse.py` | `LlamaFlashAttention2.forward` token elimination 后同步 slice 2D attention_mask：`attention_mask = attention_mask[:, idx]` |
| `script/preprocess_phase0.py` | `encode_blocks` pack 分支改为简单拼接（去掉 eos 分隔符 + 去掉 `+1` 预留） |
| `FwdLLM/experiments/.../initializer.py` | 新增 `--samples_per_round` 参数（None=全量，1=每轮每 client 采 1 样本） |
| `FwdLLM/FedML/.../FedSgdTrainer.py` | `update_dataset` 里按 `samples_per_round` 随机采样 `train_local_list` |

### 回滚 (1)

`FwdLLM/data_manager/causal_lm_data_manager.py`：回滚上一版为 pack eos 加的 `_make_attn_mask`（尾部连续 PAD 判 padding），恢复 `(tokens != PAD_ID).long()`。简单拼接后中间无 eos，原逻辑即可。

## 实验配置与结果

配置：`llama_sparse` + `--sparse 0.4 --enable_consensus --vote_threshold 0.3`，`--lr 0.0001 --server_lr 0.001`，`--comm_round 31`（r30），`--model_max_length 1024`，`--samples_per_round 1`，`--var_control --perturbation_sampling`，`var_threthod=1`，数据 `goemotions_1k_pack_data.h5`（简单拼接 907 blocks，3 clients train=458/181/167）。

结果：

1. **跑通 r30（round 0-29），无崩溃**，checkpoint 正常落盘（`global_lora_final.h5`、`consensus_history.h5`）。
2. **loss 未持续下降**：round 0-7 正常（首值 3.9~4.7，波动大），round 8 起 loss=0、jvp=nan（3 次/round）。
3. **avg_resample = 19.7 次/round**：前 8 轮重采样 0~134 次（round 0/1/3/4/7 为 105/108/99/110/134，round 2/5/6 为 18/12/0）；round 8 起 var=0（loss 已 NaN）不再重采样。
4. **NaN 时间线**：round 0-6 各偶发 1 次 NaN，round 8 起模型退化（loss=0）。根因同 summary.md B1：jvp/grad_norm 过大（2000~3500），即使 server_lr=0.001 累积后仍打坏 LoRA 权重。

## 结论 / 待办

- [ ] sparse 模型长序列 attention_mask 越界 bug 已修复（参照 jenga），长上下文（>384 有效 token）现在可跑。
- [ ] **B1 仍未解决**：loss 不收敛、round 8 起 NaN。下一步需针对 forward gradient 噪声（jvp 过大）做梯度裁剪 / 归一化，或进一步调小 server_lr、增大重采样上限，再观察 loss 收敛。
- [ ] `avg_resample=19.7` 说明 `var_threthod=1` 在正常（未 NaN）数据上平均需 ~20 次重采样才能达标，可作为后续阈值标定的参考。
