# Ailog 261020-162000 — SMELL-v3：VQ 改为 per-result 架构（`c_dim=rate*dense`）

> 用户指令：停掉当前训练；改为 `c_dim = rate * dense`；架构上**每条训练成果都会发给 server**。
> 即每条 ZOO 训练结果单独压缩/上传，channel 数 = 本轮结果数，`c_dim` 不再除以 C。
> 旧口径（总预算 `C*c_dim/d=rate`）的 0.25/0.5 run 全部停止，partial 存档。

## 一、新架构

| 量 | 公式 |
|---|---|
| 每条结果的 dense 维度 | `d_active` |
| 每条结果的 sketch 桶数 | `c_dim = max(1, floor(rate * d_active))` |
| 本轮结果数 / channel 数 | `C = local_steps * zo_directions` |
| 单条结果压缩率 | `c_dim / d_active = rate` |
| 本轮总 sketch | `C * c_dim` |
| 相对“只发一条 dense delta”的总量 | `C * rate`（因为每条结果都发） |
| dense per-result baseline | `clients * C * d_active * 2B` |
| VQ 总量 | `clients * C * c_dim * bytes_per_element` |

- 每条结果用独立 hash/sign seed（跨 client 共享 seed 序列），server 解压后聚合。
- 当前实现仍是对 **最终 delta** 做 C-channel sketch；由于 Count Sketch 线性且最终 delta = 各步结果之和，数学上等价于逐结果 sketch 后 server 求和（未做 per-result clip）。
- `rate=1` 直接绕过 VQ；每条结果无压缩上传的 baseline 是 `C * d_active`。

## 二、代码改动

| 文件 | 改动 |
|---|---|
| `src/fed/run_fed.py` | `vq_c_dim = max(1, int(rate*d_active))`（不再 `/C`）；`vq_actual_rate = c_dim/d_active`；新增 `vq_total_rate = C*c_dim/d_active` |
| 同上 | metrics 新增 `vq_total_ratio_elements`、`vq_total_ratio_bytes_vs_one_delta`、`dense_all_results_communication_bytes`；`vq_ratio_elements/bytes` 改为单条口径 |
| 同上 | `--vq-rate` help 改为 per-result；`vq_seeds.json` 记录 `total_ratio_elements` |
| `AGENTS.md` | VQ 段改为 per-result 口径；当前 run/partial 状态 |

## 三、smoke（BERT d=36,864、C=64、rate 0.5、bf16）

命令：1 client × 2 samples、L8/D8、`--zo-layers 0-5`、`--truncate 2048`。

| 项 | 值 |
|---|---|
| `c_dim` | 18,432（= 0.5 × 36,864） |
| `vq_channels` | 64 |
| `per_result` | 0.5 |
| `total_ratio_elements` | 32.0 |
| `vq_ratio_bytes`（单条） | 0.5（bf16 2B vs dense 2B） |
| `vq_total_ratio_bytes_vs_one_delta` | 32.0 |
| `vq_communication_bytes` | 2,359,296 B（1 client） |
| `dense_all_results_communication_bytes` | 4,718,592 B（1 client） |
| `vq_cos_mean`（聚合） | **0.9945** |

- 单条结果的理论 cos 仍是 ~0.577；C=64 聚合后理论 ~0.985。`vq_cos_mean` 是聚合口径，不能反推单条 rate。
- 总量相对“只发一条 dense delta”是 32×；相对 per-result dense baseline 是 0.5（2× 压缩）。
- 这正是“每条结果都上传”的代价：通信总量随 C 线性增长，压缩率是单条口径。

## 四、旧 run 停止与 partial 存档

| run | 停止点 | partial 归档 |
|---|---|---|
| 0.25 fp32（旧 total 口径） | r20 eval 后 | `checkpoints/distilbert_vq025_a03_r20_partial/` |
| 0.5 bf16（旧 total 口径） | r0 eval 后 | `checkpoints/distilbert_vq050_bf16_a03_r0_partial/` |

旧口径的 `vq_ratio_*` 是“总量/一条 delta”，与新架构不可直接比；旧 0.25 的 r20 结果（NLL 1.6692 vs baseline 1.6275）只作为历史参考。

## 五、新 run

- 待启动：`logs/fed/bert_vq050_bf16_perresult_a03/`，`rate=0.5`、bf16、C=64、c_dim=18,432、r30、CATV off、a03。
- 归档目标：`checkpoints/distilbert_vq050_bf16_perresult_a03_r30/`。
- 首轮重点检查：`c_dim=18432`、`per_result=0.5`、`total_ratio=32`、`vq_ratio_bytes=0.5`、`vq_cos_mean≈0.98–0.99`、`vq_communication_bytes` 是 `dense_all_results_communication_bytes` 的 0.5。

## 六、caveat

1. **总量**：每条结果都发意味着总通信是“一条 delta”的 C 倍；`vq_total_ratio_*` 必须和 `dense_all_results_communication_bytes` 一起看，不能拿旧 `communication_bytes`（一条 delta）当 baseline。
2. **clip**：当前仍是“最终 delta 全局 clip 后压缩”，不是逐结果 clip；若未来做真正流式上传，需要重新标定 per-result clip。
3. **cos**：`vq_cos_mean` 是 C 条聚合后的方向 cos；单条结果的 cos 更低（rate 0.5 时 ~0.577）。论文里要明确报告的是哪一个。
4. **rate=1**：绕过 VQ；如果按 per-result baseline 计，未压缩上传就是 `C * d_active`，不是 `d_active`。

## 七、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 16:13 | 停止 | 0.5 bf16（旧口径） | r0 后停；partial 归档 |
| 16:15 | 修改 | `run_fed.py` | `c_dim=rate*d`；per-result/total metrics |
| 16:20 | 修改 | `AGENTS.md` | VQ 段改 per-result |
| 16:20 | 新增 | 本 ailog | 架构 + smoke + 旧 run 存档 |
