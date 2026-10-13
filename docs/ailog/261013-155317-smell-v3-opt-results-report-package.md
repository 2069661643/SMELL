# Ailog 261013-155317 — Phase N: OPT-350M 结果出图/表格交付（reports/opt 自动报告包）

> 用户任务：把 OPT-350M 实验结果做成论文用图/表。范围据用户拍板：**完整包**（图+表+报告）、
> **只留最终结果**（不含 bf16/lr/clip/失败臂等方法学诊断）、英文标注、产出放顶层 `reports/`、
> 云端 BP/4 卡消融暂不纳入。耗时约 1 session，纯 CPU 完成（GPU 被 BERT 链占用，未干扰）。

## 产物

| 路径 | 内容 |
|---|---|
| `reports/opt/make_report.py` | 唯一入口（stdlib+matplotlib/scipy；读 `logs/fed`、`temp/`、`checkpoints/posemb_step1` 原始产物；内置 arm 注册表 + 32 条 ailog 对账断言，FAIL 则拒绝构建） |
| `reports/opt/figures/fig1_a03_convergence.{pdf,png}` | a03 主曲线：k4/D22(30r)、k24/D16(60r)、k24/D32(74r)、k24/D32+CATV(74r) 的 full/answer G-PPL（同口径） |
| `reports/opt/figures/fig2_catv_alpha.{pdf,png}` | CATV × alpha：a03 r0→74 / a01 r0→44，off / on 同口径 / on 部署，四面板 |
| `reports/opt/figures/fig3_efficiency.{pdf,png}` | 左：full G-PPL vs 实测墙钟小时；右：PPL/h 条形（统一 0–29 轮窗口） |
| `reports/opt/figures/fig4_pos_adaptation.{pdf,png}` | 位置适配 B/C：16k/2k G-PPL 398.8→31.4→18.1 |
| `reports/opt/tables/table1..5_{md,tex}` + `reports/opt/data/*.csv` | 主结果 / CATV×alpha×口径（含 paired ratio/p）/ 稀疏路径质量 / 位置适配 / 覆盖度微扫；底层 tidy CSV |
| `reports/opt/README.md` | 产物索引、口径与数据来源、结论摘要、caveat、复现命令（对账 32/32 PASS） |

## 关键口径（写死在脚本内）

- 曲线取 `metrics.jsonl` eval 事件（token 加权 `full_ppl_token`/`answer_ppl_token`）；r0 锚点与
  ON 臂同口径点取 `temp/eval_*.json` 事后评测；续训块全局轮偏移 30/45/60。
- 墙钟小时：block `config.json` mtime → 最后 `adapter_round*` mtime（链=各块之和）；
  PPL/h = Δfull PPL / 墙钟小时。与 ailog 约数略有出入（定义差异，README 注明）。
- 排除清单：260929 审计前 run、lr clamp bug run（`zoo_k4_sparse_causal_a03`、`..._step5`）。
- 对账断言覆盖 k4/D16/D32/CATV 各端点 PPL 与 7 条 paired ratio（全部 PASS，容差 2e-3 / 5e-4）。

## 验证与自测

- `py_compile` 通过；`--only figures` / `--only tables` / `--no-check` 路径均验证。
- 对账闸门注入错误值测试：可正确 FAIL 并拒绝构建（退出码非 0）。
- 图中所有数字与 ailog `260930-094100` / `261001-034500` / `261007-005000` / `261012-093000` 一致。

## 已知 caveat（README 已写）

- T3a predictor/random/dense 对照 n=4（历史小样本，待 GPU 空闲可 500 样本重评）。
- pos_lora 训练曲线不在本机（cloud A40），T4/F4 用 MANIFEST 终值。
- 云端 BP/4 卡消融未纳入（用户决定）；后续可扩展同一脚本。

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 15:20 | 新增 | `reports/opt/make_report.py` | 唯一入口 + 对账闸门 |
| 15:25 | 生成 | `reports/opt/{figures,tables,data}`、`README.md` | 4 图 + 5 表 + CSV |
| 15:50 | 修改 | `AGENTS.md` | 目录布局补 `reports/` |
| 15:53 | 新增 | 本 ailog | 交付记录 |

## 复现

```bash
PY=~/miniconda3/envs/SMELL/bin/python
$PY reports/opt/make_report.py            # 全量 + 对账
$PY reports/opt/make_report.py --only figures
```
