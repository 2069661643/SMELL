# Ailog 261021-194500 — SMELL-v3：VAR + CATV + VQ 组合系统评测（挂载在 Method A 之后）

> 用户指令：在 Method A VAR ON r30 结束后，挂一个 **VAR ON + CATV ON + VQ 0.5 ON** 的 30r
> 测试作为整体系统评测。
> 承接：`261021-111547`（Method A raw var）、`261020-162000`（VQ per-result）。

## 一、组合配置

- BERT a03、CATV **on**、VQ **rate=0.5 / bf16**、VAR **Method A**（raw var、min D=4、no patience）
- L=8、r30、eval 500、lr=1.4e-3、clip=1.0、cycle
- T0=0.587230920791626；r10/r20 探针 8 clients × 8 samples
- 基线参考：a03 off（无 CATV/VQ/VAR）r29 **1.588699 / 0.466**

## 二、代码改动

1. **允许组合**：去掉 `--var-control on` 对 `--catv on` / `--vq-rate<1` 的硬拒绝，改为 WARN。
2. **VQ per-client C**：VAR ON 时每条 client 的实际结果数 `C_i = sum(D)` 逐 client 变化；
   `c_dim=rate*d` 不变；为每个 client 单独构造 `CountSketch(C_i, c_dim)`。
   - round metrics：`vq_channels_mean/min/max`；`dense_all_results_communication_bytes = sum(C_i*d*2)`；
     `vq_communication_bytes = sum(C_i*c_dim*b_s)`。
   - `vq_seeds.json`：`channels="per_client"`、`max_channels=128`，per-client C_i 使用前缀 seed。
3. **CATV + VAR**：`_run_zoo_var_per_sample` 现在安装 vote callback（每次 forward 仍投票）；
   probe 不收集 votes，并在 probe 前设置本轮 `consensus_mask`。
4. **vote 逐 client 归一化**：`--catv-normalize client`（默认）对每个 client 的每层 vote 做
   **L1 归一化**，server 再求和 → 每个 client 等权。`off` 保留旧 paper raw 口径，`sum` 为 v2 legacy
   的 signed-sum。旧 CATV 消融是 `off`，不回改；组合 run 用 `client`。
   - D 逐 sample 变化只会影响 client 内 vote 的 averaged precision，不再改变 client 间权重；
     但由于 forward 数不同，仍不是固定 D 的 CATV 消融。

## 三、smoke（1 client × 1 sample，2k，组合）

- CATV on：`vote_bytes=384`，mask 正常；
- VAR：D=4（T0=0.587），hit_low=1；
- VQ：C=4、c_dim=18,432、bf16、`vq_cos_mean=0.8892`；
  `vq_communication_bytes=147,456` vs `dense_all=294,912`（ratio 0.5）；
- 无异常。

## 四、挂载

- chain：`temp/run_chain_methoda_to_combo_bg.sh`
  - 等 `last_bert_var_methoda_a03_dir.txt` 的 `stage_done`；
  - 校验 Method A archive 后，调用 `temp/run_bert_var_catv_vq_a03_bg.sh`。
- combo run：`logs/fed/bert_var_catv_vq_a03_r30/`
- archive：`checkpoints/distilbert_var_catv_vq_a03_r30/`
- ETA：Method A 约 20:20 结束；combo 约 **04:00–04:30**（次日）。

## 五、caveats

- CATV votes 的 per-forward 加权不等价于 per-sample 等权；不能用于 CATV 的干净消融。
- VQ `C_i=sum(D)` 使每 client 的通信量与 D 绑定；报告时必须用 `vq_channels_mean/min/max` 和
  `dense_all_results_communication_bytes`，不能拿旧的固定 C=64 口径。
- 组合 run 的 VAR 阈值仍按 Method A 的 T0/探针；CATV mask 会改变 loss/gradient，D 分布可能不同。
EOF
