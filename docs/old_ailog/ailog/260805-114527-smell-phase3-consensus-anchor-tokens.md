# SMELL Phase 3 — Consensus Anchor Tokens

## 概述

在 SMELL/FwdLLM 中集成 Consensus Anchor Tokens 投票。Client 采集预测器 sum_q，与梯度同步发送。Server 跨 data_id 累计，round 末尾 ranking → consensus_mask。所有行为由 `--enable_consensus` 门控。

## 文件操作

新建: run_phase3_consensus.sh

修改 (9): modeling_llama_sparse.py (_last_sum_q capture + consensus_mask injection), tc_transformer_trainer_distribute.py (sum_q collector + vote), fwdgrad_utils.py (calculate_jvp + model param + _capture_sum_q), message_define.py (type 9 + keys), FedSgdClientManager.py (send_vote + mask handling), FedSgdServerManager.py (type 9 handler + round-end consensus), FedSgdAggregator.py (pending_votes + round_vote_sum + ranking methods), FedSgdTrainer.py (update_consensus_mask), initializer.py (--enable_consensus --vote_threshold)

## 关键设计

- 投票基于每层 predictor 原始 sum_q（正负方向平均），跨 batch 累加
- Server 收到后 pending[data_id] 覆盖 → var 达标时归一化累加到 round_vote_sum
- round 结束: ranking → vote_threshold * 256 block 标记 (+inf/-inf)
- 第一轮: next_consensus_mask=None → 不注入
- 后续轮: consensus_mask_R-1 在 R 的 type 2 中附带 → forward 中 sum_q += mask
- --enable_consensus 默认 False，Phase 1/2 完全不受影响

## 静态检查

9/9 Python 文件 py_compile 通过。Phase 1/2/3 兼容性已分析，所有门控正确。
