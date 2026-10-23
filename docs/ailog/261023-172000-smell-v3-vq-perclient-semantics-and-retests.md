# Ailog 261023-172000 — VQ 语义改为 **per-client（1 channel/client，30 channel/轮）** + 停机 ABC + 重测计划

> 用户指令（261023）："先将 ABC 全部停机；VQ 改为**旧的压缩比且 per-client**（仅 30 channel，c_dim 不变），只收集**最后每 client 梯度结果**；
> 这要求重测 (VQ on else off) 和 (3on)。"

## 一、停机

- `temp/run_bert_inserted_queue_bg.sh`（A/B/C 队列）已整组 kill（含当时在跑的 A）。
- **A（VAR+CATV+VQ 续训，per-result 旧语义）停在 27 轮**：`logs/fed/bert_var_catv_vq_a03_cont30/a03/`（`adapter_round056` 已存，无归档）——
  作为 per-result 语义的**部分 run**，不用于论文，重测走新语义。
- B（稠密续训 30r）、C（BERT 4 臂续训）**未开始**，等待后续指令重新排队。
- 旧 watcher/监听器已全部关闭（用户要求不用 TUI 监听）。

## 二、代码改动（`src/fed/run_fed.py`）

| 项 | 旧 | 新 |
|---|---|---|
| channel 粒度 | per-result：`C = local_steps*zo_directions`（VAR 下 `C_i=sum(D)`） | **per-client（默认）**：每 client 只压**最终 delta** 1 条 → 30 clients = **30 channel/轮** |
| `c_dim` | `max(1, floor(rate*d_active))` | **不变** |
| 传输比 | `C*rate`（相对一条 delta） | **`rate`**（旧压缩比，bf16 下 `vq_ratio_bytes = c_dim*2/(d*2) = rate`） |
| 新增 CLI | — | `--vq-channels-mode {per_client,per_result}`，默认 **per_client**；`per_result` 保留用于复现旧 run |
| 新增 metrics | — | `vq_channels_total`（= Σ 每 client channel 数，per_client 下 = num_clients）、`vq_channels_mode` |
| config | `vq_channels_mode` 原表示 VAR 可变性 | 语义改为粒度；新增 `vq_channel_variability` 保留原义 |
| seed 文件 | `channels=C` | `channels=1, channels_mode=per_client` |

实现点：per_client 模式在轮初构造单个共享 `CountSketch(d, 1, c_dim)`；client 循环里 `client_vq = vq`（不再按 `sum(D)` 构造）；per_result 分支原样保留。

## 三、Smoke（2 clients × 1 轮，rate 0.5 bf16）

| 项 | VQ-only（VAR off） | VAR+VQ |
|---|---|---|
| `vq_channels` / `vq_channels_total` | 1 / 2 | 1 / 2 |
| `vq_c_dim` / `vq_active_dim` / `actual_rate` | 18432 / 36864 / 0.5 | 同 |
| `vq_ratio_bytes` / `total_ratio_bytes` | **0.5 / 0.5** | 同 |
| `vq_communication_bytes` / dense baseline | 73728 / 147456（2×18432×2 / 2×36864×2） | 同 |
| `vq_cos_mean`（单 channel） | 0.7064 | 0.7058 |
| VAR 样本 | — | D=[4,16],[4,8]，`var_total_samples=16`，channel 仍 1 |

## 四、重测（已启动 17:18:52，串行）

脚本 `temp/run_bert_vq_perclient_bg.sh`（无 TUI 监听）：

| # | 臂 | 配置 | 产物 |
|---|---|---|---|
| 1 | **VQ on, else off** | a03、VQ 0.5 bf16 per_client、VAR off、CATV off、L8 D8 lr1.4e-3/cycle/eval500、30r | `logs/fed/bert_vq_perclient_a03_r30/` → `checkpoints/distilbert_vq_perclient_a03_r30/` |
| 2 | **3-on** | a03、VAR on + CATV on(`client`) + VQ 0.5 bf16 per_client、30r | `logs/fed/bert_var_catv_vq_perclient_a03_r30/` → `checkpoints/distilbert_var_catv_vq_perclient_a03_r30/` |

对照基线（已有、协议一致）：
- VQ 臂 vs `r30_a03_catv_off`（VQ off/VAR off/CATV off）r29 = 1.588699/0.466；
- 3-on 臂 vs 历史 3-on（per-result VQ）1.559838/0.450 与 CATV-on 基线 1.548441/0.454。

预期：per_client 的聚合 cos ≈ √(30·rate/(1+30·rate)) ≈ 0.968（rate 0.5），低于 per-result C=64 的 0.9945 —— 精度/通信权衡待测。
报告：`temp/logs/bert_vq_perclient_report.txt`（两臂 r0→r29、配对、%/h、与基线对照）。

## 五、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 17:16 | 停机 | 队列 A/B/C | A 停 27 轮；B/C 未启 |
| 17:1x | 修改 | `src/fed/run_fed.py` | +`--vq-channels-mode`（默认 per_client）、per-client 单 channel、metrics/config/seed 字段 |
| 17:1x | 修改 | `AGENTS.md` | VQ 段更新为 per_client 默认 + legacy per_result 说明 |
| 17:18 | 启动 | `temp/run_bert_vq_perclient_bg.sh` | 两臂重测 30r |
