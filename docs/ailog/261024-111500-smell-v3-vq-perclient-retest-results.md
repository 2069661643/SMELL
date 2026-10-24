# Ailog 261024-111500 — VQ **per-client** 重测结果（VQ-only + 3-on）+ 组合系统代价

> 承接 `261023-172000`（per-client 语义改造与两臂重测启动）、`261020-103040`（旧总预算口径）。
> 两臂均 30/30 完成并归档；无 TUI 监听。基线均为 a03（L8/D8、lr 1.4e-3、clip 1.0、cycle、eval 500）。

## 一、结果总表（r0 → r29）

| 臂 | r0 NLL/acc | r29 NLL/acc | own NLL 降幅 | acc 增益 | 轮速 | 归档 |
|---|---|---|---|---|---:|---|
| **VQ per-client**（VAR off/CATV off） | 1.690814/.424 | **1.627393/.448** | **3.75%** | +2.4pt | 773s | `distilbert_vq_perclient_a03_r30` |
| VQ off（基线） | 1.690632/.426 | 1.588699/.466 | 6.03% | +4.0pt | 772s | `distilbert_ablation_r30_fixed` 的 `r30_a03_catv_off` |
| VQ per-result（参考） | 1.690728/.426 | 1.587080/.466 | 6.13% | +4.0pt | 774s | `distilbert_vq050_bf16_perresult_a03_r30` |
| **3-on per-client**（VAR+CATV client+VQ） | 1.743194/.410 | **1.620426/.450** | **7.04%** | +4.0pt | 882s | `distilbert_var_catv_vq_perclient_a03_r30` |
| 3-on per-result（历史） | 1.742484/.410 | 1.559838/.450 | 10.48% | +4.0pt | 903s | `distilbert_var_catv_vq_a03_r30_rerun` |
| CATV on 基线 | 1.751061/.410 | 1.548441/.454 | 11.57% | +4.4pt | 778s | `distilbert_ablation_r30_fixed` 的 `r30_a03_catv_on` |
| CATV off 基线 | 1.690632/.426 | 1.588699/.466 | 6.03% | +4.0pt | 772s | 同上 |

## 二、VQ-only：per-client 的代价（r0/r29 差值写明）

- 自身：1.690814 → 1.627393，**Δ −0.063421 nats（3.75%）**；acc 0.424 → 0.448（+2.4pt）。
- vs VQ off：基线 1.690632 → 1.588699，**Δ −0.101933（6.03%）**；acc 0.426 → 0.466（+4.0pt）。
- **差值分解**：r0 起点差 **+0.000182**；降幅差 0.101933 − 0.063421 = **+0.038512**；终点差 **+0.038694**（= 两者之和）。
- 配对（n=500）：ΔNLL **+0.038694**，t=7.745，**p=9.55e-15**。
- 损失占比：**+2.44%**（对基线终点）；**38.0%**（占基线总降幅 0.101933）；NLL 降幅完成度 **62.2%**；acc 增益完成度 **60.0%**（+2.4 vs +4.0pt）。
- 速率：0.58 %/h vs 0.94 %/h（慢 38%）。

## 三、3-on：组合系统里的代价更大

- 自身：1.743194 → 1.620426，Δ −0.122768（7.04%）；acc +4.0pt。
- vs 3-on per-result：1.559838/0.450 → **ΔNLL +0.060588**（**+3.88%**），t=8.47，**p<1e-16**；acc **0.0pt**。
- vs CATV-on 基线：+0.071985（+4.65%），p<1e-16；acc −0.4pt。
- vs CATV-off 基线：+0.031727（+2.00%），**p=0.039**；acc −1.6pt。
- 降幅完成度（对 3-on per-result）：0.122768 / 0.182645 = **67.2%** ⇒ 损失 **32.8%** 的可得下降。
- 结论：per-client VQ（cos 0.707/client）在组合系统里比 VQ-only 场景代价更大（**0.0606 vs 0.0387 nats**），且 **acc 不再有 per-result 组合的优势之外的变化**（同为 0.450）。
- **confound**：VQ 噪声改变 VAR 轨迹——mean D **9.858**、samples **78.87**（per-result 组合 9.003 / 72.0）；T10 0.440958、T20 1.048238（per-result 0.611885 / 2.210161）。因此不是严格单变量对照。

## 四、等同通信预算下的 channel 布局（重要）

同一 18,432 元素/客户端（= 旧 total-0.5 口径，1,105,920 B/轮）：

| 布局 | 元素 | cos（gauss） | cos（sparse） | 真实 run |
|---|---:|---:|---:|---:|
| **1 × 18,432（per-client）** | 18,432 | **0.7111** | **0.7024** | **0.7060** |
| 64 × 288（旧 total-0.5 bf16） | 18,432 | 0.5704 | 0.5735 | 0.5698（旧 r0） |
| 4 × 9,216（2× 预算） | 36,864 | 0.7735 | 0.7721 | — |
| 8 × 4,608（2× 预算） | 36,864 | 0.6948 | 0.6928 | — |

⇒ per-client 不是简单“退回旧方案”：在**同等字节**下，单宽 sketch 的保真度显著高于旧的多窄 channel 布局（0.71 vs 0.57）。旧 total-0.5 run 只跑到 r0，正好可作该布局对照。

## 五、产物与校验

| 归档 | sha256（adapter_model） | 内容 |
|---|---|---|
| `checkpoints/distilbert_vq_perclient_a03_r30/` | `804606a66ad940157f27c7c6907071a6546a2753e28b89da6c2ae6b29967ef6d` | r29 + metrics + config + vq_seeds + SHA |
| `checkpoints/distilbert_var_catv_vq_perclient_a03_r30/` | `af0776484af65e96cd601c4618c88e24eb9960dd40b8bcc15f3eb3eeb979efcc` | r29 + metrics + config + vq_seeds + var_thresholds + var_probe 010/020 + SHA |

- 报告：`temp/logs/bert_vq_perclient_report.txt`（两臂 r0→r29、配对、last-10 曲线）。
- 归档时补了 `vq_seeds.json`（run 1 首次归档漏拷）；run 2 的 var 探针文件由独立 waiter 补齐。

## 六、并行状态（cont30，10/24 11:00）

- `a01_off` 续训完成：r52→r59（offset 53、7 轮），r59 **1.5068/.484**（r29 1.5836/.468 → −4.85%）；归档 `checkpoints/distilbert_ablation_r30_fixed_cont30/r30_a01_catv_off/`（sha `c93bbd4f…`）；r52 eval 已回填（30..59 共 29 条 eval 记录 + 回填 JSON）。
- `a01_on` 续训中（r31：1.5532/1.5539，.462）；随后 a03_off、a03_on，ETA **10/25 ~06:00**。

## 七、待办

1. **报告包**：`reports/bert_and_other/make_report.py` 增加 `a03_vq_perclient`、`a03_combo_perclient` 两个 TECH_RUNS + 对应 CSV + 对账（现在加会因 3-on 已完成而可做）。
2. **可选**：3-on per-client 的 eval-off（30 个 adapter）以分解 mask/权重；GPU 现被 cont30 占用，可在 cont30 完成后排队。
3. **建议下一档**：rate 0.75 或 2–4 channel/client（同等预算下 4×9,216 的 cos 0.77 明显更好，2× 字节），在 VQ-only 与 3-on 各跑一臂，定位“可接受损失”点。

## 八、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10/24 00:34 | 完成 | VQ per-client（VQ-only） | r29 1.627393/.448；归档 |
| 10/24 08:44 | 完成 | 3-on per-client | r29 1.620426/.450；归档 |
| 10/24 11:0x | 修改 | `AGENTS.md` / `docs/weight-manifest.md` | 组合 per-client 行 + §10.1 |
| 10/24 11:0x | 新增 | 本 ailog | 两臂结果 + 差值分解 + 布局对照 |
