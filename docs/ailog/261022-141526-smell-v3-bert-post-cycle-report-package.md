# Ailog 261022-141526 — SMELL-v3: BERT post-cycle CSV 交付包（reports/bert_and_other）

> 用户任务：阅读项目架构、更新 `AGENTS.md`、把**自循环样本 bug（固定前 L 篇）修复后**的
> BERT 实验生成 CSV 到 `reports/bert_and_other/`；本次不生成图片（论文 Agent 统一出图），
> 但 README 要写清各数据作用/来源。分支 `exp/vq-compression`，HEAD `09a9141`。

## 一、范围判定

- Bug 指 261012 前 `run_fed.py` 每轮从 client 分片第 0 行取 `L` 条：BERT L8、30 轮只重复
  前 8 篇/人；见 `docs/ailog/261012-151805-...bert-fixed-60docs-...md`。
- 修复 = `--doc-order cycle`（`start=(round*local_steps)%总行数`），自 261012 起生效。
- 本包主表只收 cycle run：BERT r30 四臂、VQ per-result、VAR Method A、VAR+CATV+VQ 组合、
  BP cycle 参考、L×D 网格、lr 扫描；`fixed_L8D8` 仅作 H2 对照。
- 修复前 defect 臂、旧 fixed BP、旧 total-rate VQ partial、组合 r6 partial、pool VAR 诊断
  均不进主表，写入 `excluded_runs.csv`。

## 二、新增交付包

| 文件 | 内容 |
|---|---|
| `reports/bert_and_other/make_report.py` | 唯一入口；纯 CPU、无 matplotlib；读 `logs/fed/`、`temp/`；内置 62 条 ailog 对账断言 |
| `reports/bert_and_other/README.md` | 自动生成：范围/文件索引/口径/caveat/主要端点/复现/对账 |
| `reports/bert_and_other/data/*.csv` | 18 个 tidy CSV（见下） |

CSV 清单：

- `bert_r30_rounds.csv`（120 行）、`bert_r30_endpoints.csv`（4）、`bert_r30_paired.csv`（6）
- `bert_r30_evaloff_rounds.csv`（60）、`bert_r30_evaloff_endpoints.csv`（6）、
  `bert_r30_evaloff_paired.csv`（8）
- `bert_bp_cycle.csv`（5）、`bert_ld_grid.csv`（10）、`bert_lr_scan.csv`（5）
- `bert_vq_rounds.csv`（30）、`bert_var_rounds.csv`（30）、`bert_combo_rounds.csv`（30）、
  `bert_combo_evaloff_rounds.csv`（30）
- `bert_technique_endpoints.csv`（6）、`bert_technique_paired.csv`（7）
- `bert_baselines.csv`（3）、`provenance.csv`（30）、`excluded_runs.csv`（11）

## 三、口径与对账

- NLL = global_test 逐样本 CE 均值；r30/VQ/VAR/组合 500 样本，LD/BP 250 样本。
- 配对 = 逐样本 `b-a`；`p_two_sided` 用 scipy t；`p_normal_approx` 与历史 ailog 的正态近似一致。
- 62/62 断言 PASS：四臂 r29、CATV on/off、eval-off 三栏、BP 5 格、LD 10 格、
  VQ/VAR/combo 端点与 7 条技术线配对。
- 关键 caveat：CATV on 臂 r0 已带 mask；四臂 `catv_normalize=off`、组合 `client`；
  VQ 通信量单条/总量口径；VAR D 漂移；单 seed。

## 四、AGENTS.md 更新

- 目录布局补 `reports/bert_and_other/`；新增「结果包（reports/）」小节与 BERT 包命令。
- 结果表更新到 261022，补 VAR Method A、组合系统两行。
- 硬坑明确：261012 前 fixed 结果不可与 cycle 混表；`fixed_L8D8` 仅 H2 对照。
- 分支/HEAD 与 Git 小节补 `09a9141`；必读 ailog 增加本条。

## 五、复现 / 校验

```bash
PY=~/miniconda3/envs/SMELL/bin/python
$PY -m py_compile reports/bert_and_other/make_report.py
$PY reports/bert_and_other/make_report.py       # 62/62 PASS 才写 README
```

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 14:05 | 新增 | `reports/bert_and_other/make_report.py` | CSV+README 生成器 + 对账闸门 |
| 14:14 | 生成 | `reports/bert_and_other/{README.md,data/*.csv}` | 18 个 CSV、62/62 PASS |
| 14:15 | 修改 | `AGENTS.md` | 范围/结果/报告包/硬坑/分支 |
| 14:15 | 新增 | 本 ailog | 交付记录 |
