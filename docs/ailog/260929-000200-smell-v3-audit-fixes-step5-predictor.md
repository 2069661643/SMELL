# Ailog 260929-000200 — Phase N: 审计修复（ZOO dropout + 评测随机 predictor/随机 bias）+ step5 predictor 重训

> 承接 260928 用户本地审计。两条「致命链路」在代码中复核、修复并 smoke 验证；另发现并修复第三条隐患
> （训练/推理 predictor 的 bias 不一致）。旧代码 TD-3@a03 已停机（13 轮留档 `logs/fed/zoo_k4_sparse_causal_a03/a03/`）。

## 一、复核与证据

| # | 问题 | 代码证据 | 本地实测 |
|---|---|---|---|
| 1 | ZOO 在 `model.train()` 下做有限差分 → dropout=0.1 吞掉差分 | `run_fed.py` `.cuda().train()`；`serial_fedavg.py` ZOO 前向无 `.eval()`；`no_grad` 不关 dropout | `temp/probe_zoo_eval_fix.json`：loss std train **0.0215** vs eval **0.0**；ZOO grad norm train **2.24e6** vs eval **28.7**（差 1.3e-5 倍）；历史 lr=1e-3 第 2 轮 NaN 由此解释 |
| 2 | 评测从不加载 predictor，Jenga 对非末层用随机 predictor 剪枝 | `ppl.py` 无 predictor/attn/dtype 参数；`run_fed.run_global_eval` 不传；Jenga `modeling_opt.py` 用 `self.predictor` | 16k×4：sparse random full_ppl 31.90 / answer 1761.7 vs **dense 26.21 / 584.7**；训练后 predictor answer 1056（好 40%） |
| 3 | （新发现）训练侧 `PrunableAttnPredictor` Linear `bias=False`，推理侧 `PrunableAttnPredictorInfer` `bias=True` → checkpoint 只有 144 weight，推理侧 144 bias 随机 | `jenga/models/predictor.py` 79-85 vs 309-316 | 修复前 step4/step5 均为「144 权重 + 144 随机 bias」；加载器修复后 `zero_filled=144` |

## 二、代码修复

| 文件 | 修复 |
|---|---|
| `src/fed/serial_fedavg.py` | ClientRunner ZOO 分支强制 `model.eval()` + `zo_grad` 前断言 `not training` |
| `src/eval/ppl.py` | 新增 `--attn/--dtype/--predictor/--pruned-config/--per-sample-out/--seed`；sdpa* 用本仓建模；按 pruned_config 重建并加载 predictor；逐样本 mean-NLL 落盘 |
| `src/fed/run_fed.py` | eval 子进程传 attn/dtype/predictor/pruned + per-sample；`load_predictor_weights` 缺权重报错、**缺 bias 零填充**（等价 bias=False）；CATV×非 flash fail-fast |
| `src/train/train_predictor.py` | 新增可重复 `--data-glob`（client 分片拼接训练，修分布漂移） |
| `scripts/run_k4_td2_td3.sh` | TAG 支持；门控 `--data-root .../$TAG/clients`；探针 2 样本对齐 2 步；δ≤0 中止 |
| `scripts/run_ablation_cloud.sh` | 接 fp32/sdpa_prune/step5 predictor/r1；CATV×prune fail-fast |
| `scripts/diag_cos_grid.py` | predictor 默认改 step5；适配 3 元组加载返回值 |
| `scripts/paired_eval.py`（新） | 逐样本配对 t 检验 + PPL 比值（selftest PASS） |

## 三、step5 predictor（client 分布重训，用户将同步 A40）

- **数据**：a01+a03 全部 client `train_input_ids.npy`（6000 条，含 query+label），替代 warmup-only。
- **训练**：400 步、lr 2e-5、prune 100 步 0.8→0.65、144 张量、24 层剪枝；final train **0.000676** / eval **0.000669**（~23.6 min）。
- **产物 sha256**：
  - `checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth` = `1fabeab15e81b3786ca0f88d891295d10868f21b9bee567fd67b092eaf646478`（102M）
  - `checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth` = `38e94fba64899d18381be76a99e82d39e6612e72fad2e7cb4fa1b50ef58b3d2a`
- step4 保留为历史（warmup-only；bias 需零填充才可用），新运行默认用 step5。

## 四、验证（smoke）

- `temp/probe_zoo_eval_fix.json`：eval loss 确定性 std=0、ClientRunner 结束于 eval、ZOO grad norm 正常量级 → **PASS**。
- 2k ZOO smoke（D=4, lr 4e-6, step5）：train_loss **3.5754**、δ=**2.03e-5**、14.2s，无 NaN。
- 16k×4 predictor 质量（a03 global_test，fp32+sdpa_prune）：

| case | full_ppl | answer_ppl |
|---|---:|---:|
| random（无 predictor） | 31.90 | 1761.7 |
| step5 | 33.29 | **1056.4** |
| step4（零填充后） | 33.21 | 1050.5 |
| dense（sparse=1.0） | **26.21** | **584.7** |

  结论：训练后 predictor 显著改善 answer 质量（≈好 40%），full-text 略差于 random；dense 仍最优（剪枝本身有代价，属既有口径）。
  2k 截断评测把 query/answer 切掉，不能作依据（已弃用该口径做判断）。

## 五、影响面

- **作废**：所有 run_fed ZOO 运行（本地 TD-3 13 轮、云端旧 TD-3）与其 lr 校准；所有 `--eval-every` 的 G-PPL 曲线（随机 predictor + 随机 bias）；「ZOO 平在 32」作为路线判据。
- **存活**：bf16 损失量化根因（eval 模式诊断）；cos 理论/TC；BP k-scan 相对结论与 client 数曲线（绝对值需新协议复测）；数据/环境/管线。
- **待复核**：位置适配 B/C 的绝对 G-PPL（同属 Jenga 稀疏路径）；消融脚本（现已接线）。

## 六、下一步

1. 用 step5 + 零填充重跑 **TD-2' 门控（a03）→ lr 校准 → TD-3@a03** 后台长跑（`logs/fed/zoo_k4_sparse_causal_a03_step5/`，每轮逐样本落盘）。
2. 用 `scripts/paired_eval.py` 对相邻/首末轮做配对检验，替代「看均值」的判据。
3. 云端 A40 停止旧 TD-3、pull 本分支后按同参数重跑。
