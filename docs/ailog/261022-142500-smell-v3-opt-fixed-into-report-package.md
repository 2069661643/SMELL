# Ailog 261022-142500 — SMELL-v3: 把 OPT fixed r30 四臂并入 reports/bert_and_other

> 用户指令：把 **OPT fixed** 加入 `reports/bert_and_other/`。承接 `261022-141526`
> （BERT post-cycle CSV 包初版，18 CSV / 62 对账）。本文件只记录 OPT 增量；BERT 包口径不变。

## 一、纳入范围

- 运行：`logs/fed/opt_r30_fixed/{r30_a01_catv_off,r30_a01_catv_on,r30_a03_catv_off,r30_a03_catv_on}`
  （cycle、k=24/D=32/L=2/c=30/r=30、LoRA r1/α2、lr=0.30、clip=13.34、fp32+sdpa_prune、eval every 5）。
- 基线：`temp/eval_fixed_fresh_{a01,a03}.json`（fresh 33.6653 / 1740.63，500 样本；
  a01/a03 构造性相同）。
- r0 使用 `eval_round000.json`（post-round-0 补测；on 臂带 `consensus_mask_round0`）。
- eval-off 使用 on 臂 `evaloff_round{4,9,14,19,24,29}.json`。
- 归档 sha 与 `checkpoints/opt_ablation_r30_fixed/*/SHA256SUMS` 一致。

## 二、新增 CSV（7 个，92 行）

| 文件 | 行数 | 内容 |
|---|---:|---|
| `opt_r30_rounds.csv` | 28 | 4 臂 × r0/4/9/14/19/24/29 的 full/answer PPL、训练统计 |
| `opt_r30_endpoints.csv` | 4 | r0/r4/r29 端点、own% / vs fresh、train_hours、adapter sha256 |
| `opt_r30_paired.csv` | 28 | r0→r29、r4→r29、fresh→r29、CATV on-vs-off（full/answer） |
| `opt_r30_evaloff_rounds.csv` | 12 | on 臂逐评测点带 mask / 去 mask 的 full/answer |
| `opt_r30_evaloff_endpoints.csv` | 6 | off / on-evaloff / on-evalon 三栏（r4/r29） |
| `opt_r30_evaloff_paired.csv` | 12 | 权重效应 / 掩码效应 / 部署净效应（full/answer） |
| `opt_baselines.csv` | 2 | fresh pre-training a01/a03 |

同时更新 `provenance.csv`（+4 个 OPT run）与 `excluded_runs.csv`（+1 条旧 fixed OPT 协议）。

## 三、口径要点

- OPT 指标是 `full_ppl_token` / `answer_ppl_token`（token 加权 PPL），不是 BERT 的 NLL/acc；
  不与 BERT 500 样本表混用。
- eval 每 5 轮（r4 是首个 in-run 点）；`own%` 在 OPT 表里按 r4→r29，vs fresh 用训练前
  33.6653/1740.63。r0 仅作 post-round-0 参考。
- `answer_ppl_token` 随训练上升（answer 退化）；CATV 的收益在 answer 侧：
  - a03 on vs off answer **−0.0932**（p=5.4e-8，显著）；
  - a01 on vs off answer +0.0262（p=0.14 n.s.）；
  - full 侧 on 略差（a01 +0.0032、a03 +0.0076 nats，显著但量级小）。
- eval-off 分解：a03 mask effect answer −0.1021（p=2.7e-15），deploy net −0.0932；
  a01 deploy net answer +0.0262（n.s.）。

## 四、对账

- `make_report.py` 断言从 62 增至 **99/99 PASS**：新增 OPT 4 臂 r0/r29 full/answer（16）、
  CATV on/off（4）、fresh（4）、eval-off 三效应（12）、a03 deploy p（1）等。
- 产物：`reports/bert_and_other/README.md` 已更新为「BERT + OPT fixed」；
  文件索引 25 个 CSV，总数据行 498。

## 五、复现

```bash
PY=~/miniconda3/envs/SMELL/bin/python
$PY -m py_compile reports/bert_and_other/make_report.py
$PY reports/bert_and_other/make_report.py       # 99/99 PASS 才写 README
```

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 14:22 | 新增 | `reports/bert_and_other/make_report.py` OPT 函数 | 7 个 CSV + 校验 |
| 14:24 | 生成 | `reports/bert_and_other/data/opt_*.csv` | 92 行 OPT 数据 |
| 14:25 | 修改 | `reports/bert_and_other/README.md` | BERT + OPT fixed 口径/索引/端点 |
| 14:25 | 修改 | `AGENTS.md` | 报告包描述与必读 ailog |
| 14:25 | 新增 | 本 ailog | OPT 增量记录 |
