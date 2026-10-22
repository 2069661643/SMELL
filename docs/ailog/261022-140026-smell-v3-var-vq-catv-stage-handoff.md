# Ailog 261022-140026 — SMELL-v3：VAR / VQ / CATV 阶段性总结与交接

> 本文件是 **交接给下一个 agent** 的总览。分支 `exp/vq-compression`，HEAD `3a3ee4e`（261022）。
> 当前无训练/评测进程，GPU 空闲。所有实验均为 BERT a03、16k、LoRA r=1/α=2、fp32+sdpa_prune、
> sparse 0.4、cycle、eval 500；基线为 a03 CATV off：r29 **1.588699 / 0.466**。
> 权重/数据 sha 见 `docs/weight-manifest.md` §8/§9；硬坑见 `AGENTS.md`。

## 一、四条主线的最终结果

| 线 | 配置 | r29 NLL / acc | 对照 | 结论 |
|---|---|---|---|---|
| **VQ per-result** | rate 0.5 bf16、CATV off、VAR off | **1.587080 / 0.466** | a03 off 1.588699 / 0.466；配对 ΔNLL **−0.001618**（p=0.0157） | 基本等价/略好；单条 2× 字节压缩 |
| **VAR Method A** | raw var、min D=4、no patience、CATV off、VQ off | **1.584451 / 0.464** | 配对 ΔNLL **−0.004248**（p=2.3e-5） | NLL 略好，acc −0.2pt；D 均值 8.79（目标 8） |
| **组合系统** | VAR Method A + CATV ON（`client` 归一化）+ VQ 0.5 bf16 | **1.559838 / 0.450** | CATV-on 基线 1.548441 / 0.454：**+0.0114 NLL**（p=9.1e-6）；CATV-off 基线：−0.0289 NLL | 功能打通；相对 CATV-on 基线有小幅 NLL 代价 |
| **组合 eval-off** | 同组合 adapter，无 CATV mask | **1.577730 / 0.464** | CATV-on 基线 off 1.577225 / 0.462：Δ +0.000506（p=0.735） | **VAR+VQ 权重与基线等价；差距来自 mask 归一化** |

## 二、VQ per-result（已完成，USE）

- 架构：每条训练结果单独上传；`c_dim = rate*d_active`，`C=64`（L8×D8，无 VAR 时）。
- `--vq-rate 0.5 --vq-sketch-dtype bf16`：单条结果字节比 0.5；相对“一条 dense delta”总量 32×。
- 归档：`checkpoints/distilbert_vq050_bf16_perresult_a03_r30/`
  adapter sha256 `71ddc9c889cd11c03f233c5a6b4a158850ef133463231ed9b20385099e581291`。
- 旧 total 口径 0.25/0.5 partial 仅历史：
  - `distilbert_vq025_a03_r20_partial/`（0.25 fp32 r20：1.669249/0.434；明显落后）
  - `distilbert_vq050_bf16_a03_r0_partial/`（旧 0.5 r0）
- VQ 关键 caveat：
  - `vq_ratio_bytes` 是**单条结果**口径；总量看 `vq_total_ratio_*` / `vq_communication_bytes`。
  - VAR ON 时 `C_i=sum(D)` 逐 client 变化；见 §五。
  - rate=1 直接绕过 VQ；Count Sketch 在 N=d 仍有损。

## 三、VAR Method A（已完成，USE）

- 方案：逐 sample 的 FwdLLM half-mean **raw var**，`D=4..16`，no patience，逐 sample 采样后立即更新。
- 标定：30 clients × 8 samples = 240 histories；T0=**0.587230920791626**；
  mean D=7.967、median 4、hit_low(D=4)=123/240、hit_high(D=16)=59/240（用户选 A，接受该分布）。
- 分段阈值：r0–r9 T0；r10 探针 **0.749424**（meanD 8.0）；r20 探针 **1.297361**（meanD 8.0）。
- 实际：mean D **8.791**、samples **70.3**；hit_low 3156/7200（43.8%）、
  hit_high 2090/7200（29.0%）；分段 D 均值 8.03 / 8.91 / 9.43（段内仍上漂）。
- 归档：`checkpoints/distilbert_var_methoda_a03_r30/`
  adapter sha256 `3a84cd1e3be5a9993a1d91eb03cb13c380f7ff95e80921810d815c41ccf28f74`。
- 关键 caveat：raw var **非单调**，D=4 容易假早停；绝对值阈值跨训练漂移；探针只有 8 clients。
  若继续改 VAR，优先考虑 patience=2 / min D=6 / 归一化或静态 per-client D。

## 四、CATV vote 归一化（改造，默认启用）

- `--catv-normalize {off,sum,client}`，默认 **client**：
  - `client`：每个 client 的每层 vote 做 **L1 归一化**，server 再求和 → 每个 client 等权；
  - `off`：旧 paper raw summed votes（历史 CATV 消融口径）；
  - `sum`：v2 legacy signed-sum。
- VAR 下 D 逐 sample 变化，per-forward vote 数不同；`client` 归一化保证 client 间等权。
- 代价（组合 eval-off 已验证）：等权 mask 的 NLL 收益从约 −0.0288 降到 −0.0179，
  acc 多掉约 0.6pt；这是 §六 的主要解释。

## 五、组合系统评测（VAR + CATV + VQ）

- 配置：VAR Method A + CATV on（`client`）+ VQ 0.5 bf16；L8、r30、eval500。
- 通信：VQ `C_i=sum(D)` 逐 client；r29 C=68.2、cos=0.99469、单条字节比 0.5；
  VQ 75.4MB vs per-result dense 150.8MB；相对一条 delta 约 34×。
- VAR：mean D **9.003**、samples **72.0**；hit_low 2896/7200（40.2%）、
  hit_high 2128/7200（29.6%）；T10=0.611885、T20=2.210161。
- CATV：anchor 51/51，IR mean 0.922，`catv_normalize=client` 已生效。
- 单轮均值 903s；总耗时 ~8.3h。
- 归档：`checkpoints/distilbert_var_catv_vq_a03_r30_rerun/`
  adapter sha256 `d537b37eef4f1ab632fe962d2a1e1e5b2adf25cd5c7a80550308498a721a9c3a`。
- 第一次组合 run 在 r5 后（21:37）被外部 kill；r0–r5 partial 在
  `checkpoints/distilbert_var_catv_vq_a03_r6_partial/`，**不要使用**。原因：chain 是通过 harness
  background shell 挂的，约 2h 生命周期；后来改用 `setsid nohup` 重跑成功。
  **长任务必须 setsid nohup，不要用 background shell。**

## 六、eval-off 分解（261022 13:36，全 30 轮）

| 指标 | on-mask | off-mask | mask 效应 |
|---|---|---|---|
| 30 轮均值 NLL | 1.656088 | 1.633698 | **+0.022390**（on 更差） |
| 30 轮均值 acc | 0.4336 | 0.4497 | **−0.0161** |
| r29 NLL / acc | 1.559838 / 0.450 | 1.577730 / 0.464 | ΔNLL **−0.017892**（p=0.28 n.s.），acc −1.4pt |

- mask 的 NLL 收益从 r0 **+0.052（on 更差）** 逐步下降到 r25 后转负；acc 全程为负。
- **组合 off vs CATV-on 基线 off**：30 轮 mean ΔNLL **+0.003181**、Δacc **+0.0006**；
  r29 +0.000506（p=0.735） ⇒ **VAR+VQ 权重与基线等价**。
- 组合 on vs CATV-on 基线 on 的 +0.0114 NLL **全部来自 mask 选择/归一化**，不是 VQ/VAR。
- eval-off 归档：`checkpoints/distilbert_var_catv_vq_a03_r30_rerun/evaloff/`（30 JSON+persample+SHA）。

## 七、代码与 CLI 状态

| 文件 | 内容 |
|---|---|
| `src/compress/count_sketch.py` | Count Sketch（VQ 复刻；整数 hash/gcd；bf16 传输在外部 cast） |
| `src/fed/run_fed.py` | VQ per-result/per-client C；VAR Method A + r10/r20 探针；CATV client 归一化；metrics |
| `src/fed/serial_fedavg.py` | `_run_zoo_var_per_sample`（raw var、D=4..16、逐 sample 更新、CATV vote callback）；旧 pool 路径仅历史 |
| `src/train/zoo.py` | `half_mean_var` / `half_mean_var_stats` / `zo_grad` |
| `AGENTS.md` | 当前分支/结果/VQ/VAR/CATV 口径与硬坑 |
| `docs/weight-manifest.md` | §8 VQ per-result；§9 组合 + eval-off；sha 清单 |

CLI 默认：

```text
--var-control off
--var-threshold 0.587230920791626    # Method A T0；仅 BERT a03 口径
--var-segment-rounds 10
--var-probe-clients 8
--catv-normalize client              # 重要：默认已改
--vq-rate 1.0                        # 1=off
--vq-sketch-dtype bf16
--vq-bytes-per-element 0             # auto: bf16=2, fp32=4
```

组合复现命令：

```bash
$PY src/fed/run_fed.py --arch distilbert --tag a03 --gpu 0 \
  --trainer zoo --attn sdpa_prune --dtype fp32 --sparse 0.4 --doc-order cycle \
  --pos-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/pos_embed.pt \
  --head-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/head.pt \
  --predictor checkpoints/predictor_bert/a01_warmup_400step/predictor.pth \
  --pruned-config checkpoints/predictor_bert/a01_warmup_400step/pruned_config.pth \
  --catv on --catv-normalize client --lora-r 1 --lora-alpha 2 \
  --zo-subspace layers --zo-layers 0-5 --zo-directions 8 --zo-eps 1e-3 \
  --local-steps 8 --rounds 30 --lr 1.4e-3 --delta-clip 1.0 \
  --max-clients 30 --eval-every 1 --eval-batch-size 8 --eval-max-samples 500 \
  --var-control on --var-threshold 0.587230920791626 \
  --var-segment-rounds 10 --var-probe-clients 8 \
  --vq-rate 0.5 --vq-sketch-dtype bf16 --vq-rescale on --vq-seed 0
```

## 八、产物清点

### USE

| 路径 | 内容 | sha256（adapter_model） |
|---|---|---|
| `checkpoints/distilbert_vq050_bf16_perresult_a03_r30/` | VQ per-result 0.5 bf16 | `71ddc9c8...` |
| `checkpoints/distilbert_var_methoda_a03_r30/` | VAR Method A | `3a84cd1e...` |
| `checkpoints/distilbert_var_catv_vq_a03_r30_rerun/` | 组合系统 + evaloff/ | `d537b37e...` |

### 仅历史/partial（不要用于论文）

- `checkpoints/distilbert_vq025_a03_r20_partial/`：旧 total-rate 0.25，r20 明显落后。
- `checkpoints/distilbert_vq050_bf16_a03_r0_partial/`：旧 total-rate 0.5，r0。
- `checkpoints/distilbert_var_catv_vq_a03_r6_partial/`：组合首次 run 外部 kill，r0–r5。
- `checkpoints/distilbert_var_a03_r30/`：pool 级 VAR（诊断），r29 1.590681/0.462；非最终方案。
- 空目录（勿用）：`distilbert_vq025_a03_r30/`、`distilbert_vq050_bf16_a03_r30/`、
  `distilbert_var_persample_a03_r30/`、`distilbert_var_catv_vq_a03_r30/`。

## 九、已知问题与建议

1. **VAR raw var 非单调**：D=4 假早停多（hit_low 40–51%），D 分布双峰；
   `mean D` 靠 D=4 与 D=16 抵消。若要更稳定的 D，试 patience=2（标定上 hit_low=0，
   70% 落 D=6、11% D=16）或 min D=6；能量归一化能稳定 D=8 但会抹掉 sample 区分度。
2. **阈值漂移**：Method A 的 T10/T20 逐段上调；段内 D 仍从 8.0 漂到 9–10。
   可缩短段长（5 轮）或每段重标定；绝对阈值不可跨模型/精度复用。
3. **CATV `client` 归一化 trade-off**：等权 client 是需求，但 mask 的 NLL 收益变小、
   acc 损失变大；如果目标是最大 NLL/acc，需要和 `off` 做显式对照，或设计折中归一化。
4. **组合归因**：当前组合 vs CATV-on 基线的 +0.0114 NLL 已由 eval-off 证明来自 mask 归一化；
   若论文需要干净归因，补 **CATV on + client、VAR/VQ off** 的匹配基线（约 7h）。
5. **VQ 通信口径**：per-result 0.5 是单条字节比；总量是 C_i 倍一条 delta；不要和旧固定 C=64
   的 total 口径混用。
6. **CATV/VQ 组合**：VAR 下 VQ `C_i=sum D` 已验证；CATV votes 逐 forward，再 client L1 等权；
   D 影响的是 vote 精度，不再影响 client 权重。
7. **长任务**：必须 `setsid nohup`；harness background shell 约 2h 会杀进程（组合首次 run 的教训）。

## 十、下一步（建议优先级）

1. **补匹配基线**（若需要干净归因）：CATV on + client normalize，VAR/VQ off，r30。
2. **VAR 改进**：patience=2 / min D=6 / 每 5 轮探针；目标把 mean D 稳定在 8。
3. **CATV 归一化**：比较 `off` / `client` / 折中（例如先 per-client 等权再温度缩放），
   明确论文要等权还是最大 NLL/acc。
4. **VQ**：在 per-result 架构下补 0.25（单条 4×）和 bf16/fp32 对照；报告 C_i、cos、bytes。
5. **写论文表**：VQ per-result、Method A、组合三行；明确通信基线（per-result dense）与
   mask 归一化 caveat。

## 十一、必读

- `docs/weight-manifest.md` §8（VQ per-result）、§9（组合 + eval-off）。
- `AGENTS.md` 的 VQ / VAR / CATV 段。
- `261020-103040`（VQ 初版）、`261020-155000`（bf16）、`261020-162000`（VQ per-result）。
- `261021-095531`（VAR 讨论）、`261021-111547`（Method A）。
- `261021-194500`（组合系统评测 + eval-off）。
