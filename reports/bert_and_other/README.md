# BERT + OPT fixed results package (post self-loop fix)

由 `reports/bert_and_other/make_report.py` 从原始 `logs/fed/` 与 `temp/` 产物自动生成；
纯 CPU、确定性；**不含图片**（论文图由论文 Agent 统一生成）。
禁止手工改本目录数字；改脚本后重跑。原始 `logs/`、`temp/` 不进 git；
换机只带 CSV/README 时可直接用，重跑脚本需要原始产物。

## 范围：只收自循环样本 bug 修复后的实验

- 缺陷（261012 前）：`run_fed.py` 每轮从 client 分片第 0 行取 `L` 条；
  BERT `L=8`、30 轮只重复前 8 篇/人（30 clients 合计 240 个固定 client-doc 位）。
  见 `docs/ailog/261012-151805-...bert-fixed-60docs-protocol-flaw-and-ld-plan-handoff.md`。
- 修复：`--doc-order cycle`，`start=(round*local_steps)%总行数`；本包主表全部为 `cycle`。
- 唯一 `fixed` 行是 `bert_ld_grid.csv` 的 `fixed_L8D8`，作为 H2 语料对照故意保留。
- 修复前缺陷臂、旧 fixed 协议、partial/诊断 run 均列入 `excluded_runs.csv`，不混入主表。
- **OPT fixed r30 四臂也纳入本包**（`opt_r30_*`）：cycle、k=24/D=32/L=2/lr=0.30/clip=13.34、
  eval every 5；`reports/opt/` 保留旧 OPT 图表/长跑，二者口径不同，不要混用。

## 文件索引

| 文件 | 行数 | 作用 | 原始来源 |
|---|---:|---|---|
| `data/bert_r30_rounds.csv` | 120 | 4 臂 × 30 轮逐轮 NLL / acc / 训练统计（主曲线数据） | logs/fed/bert_da_fixed/<arm>/<tag>/metrics.jsonl + eval_roundNNN.json |
| `data/bert_r30_endpoints.csv` | 4 | 4 臂 r0/r29 端点、相对下降、墙钟、adapter sha256 | 同上 + adapter_round029/adapter_model.safetensors |
| `data/bert_r30_paired.csv` | 6 | 4 臂 r0→r29 与 CATV on-vs-off 的逐样本配对检验 | logs/fed/bert_da_fixed/<arm>/<tag>/eval_round{000,029}_persample.json |
| `data/bert_r30_evaloff_rounds.csv` | 60 | CATV on 臂逐轮带掩码/去掩码评测（mask 效应曲线） | logs/fed/bert_da_fixed/r30_<tag>_catv_on/<tag>/eval{off,}_roundNNN.json |
| `data/bert_r30_evaloff_endpoints.csv` | 6 | off / on-evaloff / on-evalon 三栏端点 | 同上 |
| `data/bert_r30_evaloff_paired.csv` | 8 | 权重效应、推理掩码效应、总效应、r0 起点核验 | 同上 + eval{off,}_round{000,029}_persample.json |
| `data/opt_r30_rounds.csv` | 28 | OPT fixed 4 臂 × r0/4/9/14/19/24/29 的 full/answer PPL 曲线 | logs/fed/opt_r30_fixed/<arm>/<tag>/metrics.jsonl + eval_roundNNN.json |
| `data/opt_r30_endpoints.csv` | 4 | OPT fixed 4 臂 r0/r4/r29 端点、vs fresh、墙钟、adapter sha256 | 同上 + temp/eval_fixed_fresh_<tag>.json |
| `data/opt_r30_paired.csv` | 28 | OPT fixed 的 r0→r29、r4→r29、fresh→r29、CATV on-vs-off 配对 | logs/fed/opt_r30_fixed/.../eval{,_round}_persample.json + temp/eval_fixed_fresh_<tag>_persample.json |
| `data/opt_r30_evaloff_rounds.csv` | 12 | OPT CATV on 臂逐评测点带掩码/去掩码 full/answer PPL | logs/fed/opt_r30_fixed/r30_<tag>_catv_on/<tag>/eval{off,}_roundNNN.json |
| `data/opt_r30_evaloff_endpoints.csv` | 6 | OPT off / on-evaloff / on-evalon 三栏端点（r4/r29） | 同上 |
| `data/opt_r30_evaloff_paired.csv` | 12 | OPT 权重效应、推理掩码效应、部署净效应（full/answer） | 同上 + eval{off,}_round029_persample.json |
| `data/opt_baselines.csv` | 2 | OPT fresh pre-training 基线（a01/a03，500 样本） | temp/eval_fixed_fresh_{a01,a03}.json |
| `data/bert_bp_cycle.csv` | 5 | BP 参考系（cycle，250 样本）：dense/sparse × lr × 轮数 | logs/fed/bert_bp_ref_cycle/<run>/a01/metrics.jsonl |
| `data/bert_ld_grid.csv` | 10 | L×D 网格 + fixed / iso-corpus / dense 对照（cycle，250 样本） | logs/fed/bert_ld/<cell>/a01/metrics.jsonl |
| `data/bert_lr_scan.csv` | 5 | lr 扫描 r0–r2 训练损失 / delta（无 eval） | logs/fed/bert_lr_scan/<lr>/a01/metrics.jsonl |
| `data/bert_vq_rounds.csv` | 30 | VQ per-result 0.5 bf16 逐轮 NLL/acc、cos、通信量 | logs/fed/bert_vq050_bf16_perresult_a03/a03/metrics.jsonl + eval_roundNNN.json |
| `data/bert_var_rounds.csv` | 30 | VAR Method A 逐轮 NLL/acc、D 分布、hit_low/high、阈值 | logs/fed/bert_var_methoda_a03_r30/a03/metrics.jsonl + eval_roundNNN.json |
| `data/bert_combo_rounds.csv` | 30 | VAR+CATV(client)+VQ 组合逐轮 NLL/acc、VAR/VQ/CATV 统计 | logs/fed/bert_var_catv_vq_a03_r30_rerun/a03/metrics.jsonl + eval_roundNNN.json |
| `data/bert_combo_evaloff_rounds.csv` | 30 | 组合臂逐轮带掩码/去掩码评测（eval-off 分解） | logs/fed/bert_var_catv_vq_a03_r30_rerun/a03/eval{off,}_roundNNN.json |
| `data/bert_technique_endpoints.csv` | 6 | a03 基线与 VQ / VAR / 组合 / 组合 eval-off 端点汇总（含训练/评测 mask 标记） | 上述 run 的 metrics.jsonl、eval JSON、adapter sha256 |
| `data/bert_technique_paired.csv` | 7 | VQ/VAR/组合相对基线的逐样本配对检验 | 上述 run 的 eval_round029_persample.json / evaloff_round029_persample.json |
| `data/bert_baselines.csv` | 3 | 冻结 Step1 / dense 250 样本锚点（无 adapter） | temp/eval_step1_16k_{final,best}.json、temp/eval_baseline_250.json |
| `data/provenance.csv` | 34 | 每个被纳入 run 的原始路径、协议、轮数、状态 | logs/fed/、temp/ 原始产物清点 |
| `data/excluded_runs.csv` | 12 | 未纳入的缺陷臂 / 旧协议 / partial / 诊断 run 及原因 | checkpoints/ 与 logs/fed/ 清点 |

## 口径

- `nll` = global_test 上的逐样本交叉熵均值（BERT 分类 NLL），不是 PPL；
  r30 四臂/VQ/VAR/组合用 500 样本；LD/BP 用 250 样本，绝对 NLL 不可直接与 500 样本表混用。
- `accuracy` = 11 类样本准确率；`macro_f1` = 11 类 macro-F1。
- `delta_nll_pct = (nll_r0 - nll_last) / nll_r0 * 100`；正值 = 变好。
- `delta_acc_points = acc_last - acc_r0`（0–1 绝对点）；`delta_acc_pct` 为相对 r0 的百分比。
- `err_reduction_pct = (1 - (1-acc_last)/(1-acc_r0)) * 100`。
- 配对检验：`b - a` 的逐样本 NLL；负值 = b 更好。`p_two_sided` 用 scipy t 分布；
  `p_normal_approx` 用正态近似，和 261015/261021/261022 ailog 的历史 p 值一致。
  `exp_delta_nll = exp(mean ΔNLL)`，是逐样本似然比，不是 PPL 比。
  `p=0` 表示 double 下溢（t 很大），不是缺失；OPT 的 full 侧常见。
- `wall_hours` = `config.json` mtime → 最后一个 `adapter_round*` mtime；含评测时间，
  只适合同协议内比较。`nll_pct_per_hour = delta_nll_pct / wall_hours`。
- `adapter_sha256` 是 `adapter_round029/adapter_model.safetensors` 的 sha256。
- **OPT 口径**：`full_ppl_token` / `answer_ppl_token` 是 token 加权 PPL，不是 BERT 的 NLL/acc；
  OPT eval 每 5 轮（r4/9/14/19/24/29），r0 是 post-round-0 补测（训练前 fresh = 33.6653/1740.63）；
  `answer_ppl_token` 随训练通常变差，CATV 的收益在 answer 侧；不要与 BERT 500 样本 NLL 混表。
  `full_pct_per_hour_r4_r29` 用 `train_hours`（各轮 `round_seconds` 求和），与 ailog 的 %/h 一致。
- `bert_technique_endpoints.csv` 的 `vq_bytes_total` / `dense_all_bytes_total` 是 30 轮求和；
  `mean_vq_channels` / `mean_vq_total_ratio` 是 30 轮均值。VQ 无 VAR 时 `vq_channels=64`；
  组合的 `C_i=sum(D)` 逐 client 变化，因此 `mean_vq_channels≈72`。
  VAR/VQ/CATV 的 r29 单轮值请取 `bert_var_rounds.csv` / `bert_combo_rounds.csv`；
  端点表里的 `mean_D`/`mean_vq_*`/`catv_ir_mean` 是 30 轮均值。
- 技术线端点均为 **own-path**（各自 r0 基线），不是跨臂共享 r0；VQ/VAR 的 r0 也略有差异。

## 关键 caveat

- **CATV on 臂的 r0 已带 mask**（训练第 0 轮无 mask，但 r0 eval 用 round-0 掩码）；
  off 臂 r0 无 mask。跨臂比较必须用 r29 同轮配对，不要用 r0 跨臂比较。
- **CATV 归一化不同**：r30 四臂是 `catv_normalize=off`（paper raw summed votes）；
  组合是 `client`（每 client 每层 L1 等权）。组合 vs CATV-on 基线的 +0.0114 NLL
  来自 mask 选择/归一化，不是 VAR/VQ 权重；eval-off 证明组合权重与基线等价。
- **VAR**：raw var 非单调，D=4 假早停多；mean D 会随训练上漂。Method A 的 T10/T20
  是分段探针阈值，不能跨模型/精度复用。
- **VQ**：`vq_ratio_bytes` 是单条结果口径；`vq_total_ratio_bytes_vs_one_delta` 是相对
  “一条 dense delta”的总量（C 倍）。`vq_cos_mean` 是聚合后 cos，不能反推 per-result rate。
- **LD/BP**：sparse 臂用 `sdpa_prune`，自身基线比 dense 高约 0.18 nats（剪枝惩罚）；
  `fixed_L8D8` 是 fixed 对照，不是修复后主结果；`B1/B2` 是等 40 篇唯一语料对照。
- **OPT fixed**：full 侧 CATV 部署净效应略差（a01 +0.0032 / a03 +0.0076 nats，显著但量级小）；
  answer 侧仅 a03 显著更好（−0.0932，p=5.4e-8），a01 +0.0262（p=0.14 n.s.）。
  on 臂 r0 已带 mask，跨臂比较用 r29；OPT 结果不与 BERT 的 NLL/acc 混表。
- **单 seed**：所有 r30 端点都是单次 run，无跨 seed 误差棒。

## 主要端点（r29，500 样本）

| run | NLL r0 → r29 | acc r0 → r29 | 关键配对 |
|---|---:|---:|---|
| a03 CATV off | 1.6906 → 1.5887 | 0.426 → 0.466 | baseline |
| a03 CATV on | 1.7511 → 1.5484 | 0.410 → 0.454 | on vs off ΔNLL −0.0403 (p=0.0025) |
| a01 CATV off | 1.6910 → 1.5836 | 0.424 → 0.468 | baseline |
| a01 CATV on | 1.7527 → 1.5563 | 0.408 → 0.458 | on vs off ΔNLL −0.0272 (p=0.030) |
| VQ 0.5 bf16 | 1.6907 → 1.5871 | 0.426 → 0.466 | vs a03 off ΔNLL −0.0016 (p=0.016) |
| VAR Method A | 1.6906 → 1.5845 | 0.426 → 0.464 | vs a03 off ΔNLL −0.0042 (p=2.7e-5) |
| combo | 1.7425 → 1.5598 | 0.410 → 0.450 | vs a03 on ΔNLL +0.0114；vs a03 off −0.0289 |
| combo eval-off | 1.6904 → 1.5777 | 0.426 → 0.464 | vs a03 on eval-off ΔNLL +0.0005 (p=0.735) |

表中 p 值用历史 ailog 的正态近似；CSV 同时给 `p_two_sided`（scipy t）与 `p_normal_approx`。

## OPT fixed r30 主要端点（PPL，500 样本）

| run | full r0 → r29 | answer r0 → r29 | 关键配对 |
|---|---:|---:|---|
| a01 CATV off | 33.525 → 31.533 | 1782 → 2436 | baseline |
| a01 CATV on | 33.921 → 31.635 | 1745 → 2474 | on vs off full +0.0032（p≈0）；answer +0.0262（p=0.14 n.s.） |
| a03 CATV off | 33.530 → 31.457 | 1765 → 2449 | baseline |
| a03 CATV on | 33.940 → 31.696 | 1598 → 2292 | on vs off full +0.0076（p≈0）；answer −0.0932（p=5.4e-8） |

OPT 的 `own%` 起点是 r4（首个 in-run eval）；vs fresh 用训练前 33.6653/1740.63。

## 复现

```bash
PY=~/miniconda3/envs/SMELL/bin/python
$PY reports/bert_and_other/make_report.py            # 全量 + 对账闸门
$PY reports/bert_and_other/make_report.py --no-check # 仅调试
```

## 对账结果

- 断言：99/99 PASS
