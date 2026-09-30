# Ailog 260930-104500 — Phase N: Step1–3 执行（选块漂移量化 / r29 predictor 刷新 / E2–E3 有限差分）

> 承接 handoff `260930-100000`。按用户确认顺序执行 Step 1→2→3（+ 汇总报告）；本 session **未启动长跑**。
> 环境：AutoDL 4090（`/root/smell/SMELL`，env `~/miniconda3/envs/SMELL`）；起点 git `b904a07`（clean）。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 10:05 | 新增 Step1 探针（gitignore） | `temp/probe_selection_drift.py` | 全文 | monkeypatch `_prune_block_scores` 记录每层打分，按 forward 规则复现 kept 块 |
| 10:11 | 跑 Step1 | `temp/diag_selection_drift/r0_vs_r29.json` | — | 8 条 global_test × (r0, r29) 状态 |
| 10:11 | 启动 Step2 刷新 | `temp/run_refresh_r29_bg.sh` | — | train 400 步 + eval + 配对检验 |
| 10:35 | Step2 训练完成 | `checkpoints/predictor/refresh_r29_a01a03/` | — | final eval_loss 0.000663（step5 0.000669） |
| 10:41 | Step2 评测完成 | `temp/diag_refresh/{r029_refresh.json,r029_refresh_persample.json,paired_*.json}` | — | answer 1842.5 / full 33.252 |
| 10:42 | 新增/跑 Step3 探针 | `temp/probe_e2e3.py` → `temp/diag_e2e3/e2e3_r29.json` | — | E2/E3：r29、layers 20–23、D=4 |
| 10:42 | 新增汇总报告 | `temp/report_step123.py` → `temp/logs/step123_report.txt` | — | 单文件汇总 |

## Step 1 — predictor 选块漂移（r0 vs r29）

- 口径：r29 run 的 `adapter_round000` vs `adapter_round029`；fp32+`sdpa_prune`+step5 predictor+pos_only；8 条 a03 `global_test`（16k，纯读取）。
- 结果：12 个剪枝层（11–22）kept 块 overlap **0.966–0.996**（每层 ~2/102 块换位）；first-half 占比几乎不变。
- 逐样本 overlap 0.981–0.991；`corr(mean_overlap, Δanswer_nll) = −0.52`（n=8，弱相关，方向与假设一致）。
- **结论：冻结 predictor 的选块在 r0→r29 基本稳定；仅凭选块漂移难以解释 answer +6.4%。**

## Step 2 — r29 predictor 刷新（数据隔离）

- 训练：`train_predictor.py --data-glob a01/a03 client train_input_ids.npy --adapter adapter_round029`，400 步、lr 2e-5、seed 42（除 `--adapter` 外与 step5 完全同口径）；**未用 global_test/local_test/warmup**。
- 训练曲线：final eval_loss 0.000663（step5：0.000669）。
- 评测（a03 global，fp32+sdpa_prune+sparse 0.4，500 样本）：

| 口径 | answer PPL | full PPL |
|---|---:|---:|
| sparse r0（step5） | 1743.76 | 33.6538 |
| sparse r29（step5） | 1855.04 | 33.2851 |
| **sparse r29（refresh）** | **1842.46** | **33.2524** |
| dense r0 | 3593.96 | 26.4667 |
| dense r29 | 3154.68 | 26.0624 |

- 配对 refresh vs step5（n=500）：answer ΔNLL **−0.0067**（p=0.014，ratio 0.9933）；full ΔNLL −0.0010（p=5.8e-27）。
- **结论：刷新统计显著但幅度极小（answer −0.67%），未闭环**（距 r0 仍 +5.7%）；结合 Step 1 ⇒ 选块漂移不是 answer 退化的主因。
- 产物 sha256：`predictor.pth` = `3d3dd15dfbdc0deac609faac7df3aaaf270fce7a792c6060412ae998fafdc1f2`；`pruned_config.pth` = `784e0391e0fb26b86e1b32c22161760537a35a7664f1d127f03475cda393b7f3`。

## Step 3 — E2/E3（r29、layers 20–23、D=4、eps=1e-3）

- E2（同方向同种子，sparse 0.4 vs dense 1.0）：sparse |ΔL| mean **1.79e-7** / max 4.77e-7；dense mean 3.38e-7 / max 9.54e-7 —— 均在 fp32 损失底噪量级，**无重尾**（未复现 r24–26 的 δ~1e4 离群）。
- E3：θ±hν 两次前向的选块翻转数 **flips = 0.00**（3 client × 2 模式 × 4 方向 = 24 个方向对），相关不可计算。
- **结论：r29/晚层没有选块翻转证据；离群重尾更可能是「特定轮 × 早期层」的稀发现象（r24 client_21 发生在 layers 0–3），需对准对应轮/层状态复现。**

## 影响 / 下一步候选

1. answer 退化的候选主因回到**训练目标 token 均值淹没 answer**（answer ≈3 token / 16384），可做 E4/E5 或 answer 加权 loss 实验。
2. 刷新机制（周期性 predictor 重适配）收益过小（−0.67%），暂不建议作为独立方法贡献；如需保留只作为稳定性配套。
3. 离群机制复核若要继续：在 r24 状态 + layers 0–3 上复跑 E2/E3（当前探针支持 `--adapter adapter_round024 --layers 0-3`）。

## 产物索引

- 探针：`temp/probe_selection_drift.py`、`temp/probe_e2e3.py`；驱动：`temp/run_step1_selection_drift_bg.sh`、`temp/run_refresh_r29_bg.sh`、`temp/run_after_refresh_e2e3_bg.sh`；报告：`temp/report_step123.py`。
- 数据：`temp/diag_selection_drift/r0_vs_r29.json`、`temp/diag_refresh/*`、`temp/diag_e2e3/e2e3_r29.json`、`temp/logs/step123_report.txt`。
- 权重：`checkpoints/predictor/refresh_r29_a01a03/`（sha256 已录 `docs/weight-manifest.md` §2）。

## 待办

- [ ] 用户决定 answer 机制继续深挖（E4/E5/answer 加权）或按另一 session 的 PPL 加速方向走。
- [ ] 云端 A40：按 `260930-094100` §一 重跑 TD-3（refresh 非必需；如需带上，用本 ailog sha256 校验）。
