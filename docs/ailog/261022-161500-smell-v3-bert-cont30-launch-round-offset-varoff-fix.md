# Ailog 261022-161500 — BERT 四臂续训 r30→r59（后台链）+ `--round-offset` + VAR-off 崩溃修复

> 用户指令（261022）："BERT 4 arm 续训 30 轮，后台挂起"。
> 承接 `261015-213000`（BERT fixed r30 四臂，归档 `checkpoints/distilbert_ablation_r30_fixed/`）。
> 环境重启后（服务器重启，旧 watcher 全部取消）在本 checkout（`exp/vq-compression`）执行。

## 一、代码改动

| 文件 | 改动 | 说明 |
|---|---|---|
| `src/fed/run_fed.py` | **`--round-offset`（默认 0）** | 续训时首个本地轮的**绝对轮号**：用于 ① cycle 取数游标 `start=(abs_round*L)%rows`；② 输出编号 `adapter_roundNNN` / `eval_roundNNN` / `consensus_mask_roundN` / metrics `round`；③ OPT 层轮转与 rank 调度的相位。`eval_every`/`save_every` 仍按本地轮计。默认 0 与旧行为逐位一致。 |
| `src/fed/run_fed.py` | **VAR-off 崩溃修复** | `var_hit_low_samples = sum(stats.get("var_hit_low", 0) ...)` 在 VAR 关闭时值为 `None` → `TypeError`（`exp/vq-compression` 上所有 var-off 长跑会在第 1 轮记录时崩）。改为 `(stats.get(...) or 0)`。 |

复现/验证：
- `py_compile` OK；
- smoke（`temp/smoke_bert_cont30.sh`，1 client × 2 samples × 1 轮，L=2）：`adapter_init loaded…`（635915 张量，随后冻结到 LoRA 36864）；`round 30/30`；`adapter_round030` / `eval_round030.json` 落盘；metrics `round=30, doc_order=cycle`；`client round_start=60`（= (30×2)%100 ✓）。

## 二、续训协议（与首段完全一致）

- 起始权重：`checkpoints/distilbert_ablation_r30_fixed/<arm>/adapter_round029`（`--adapter-init`，is_trainable=True）；
- `--round-offset 30 --rounds 30`：输出轮号 030–059（cycle 游标从第 30 轮继续：(30×8)%100=40 起）；
- 超参不变：L=8、D=8、lr=1.4e-3、clip=1.0、cycle、eval 500（每轮）、`--catv-normalize off`（首段为 off，保持口径）、`keep_block0=True`、LoRA r1/α2、step5-BERT predictor + pos/head；
- 4 臂：a01/a03 × CATV{off,on}；单臂 ~6.5h → 全链 ~26h。
- **已知接缝**：CATV on 臂的 `consensus_mask` 在续训起点重置为 None（第 30 轮训练无掩码，随后按新票生成）——与首段第 0 轮语义一致；报告里注明。

## 三、运行与产物

- 脚本：`temp/run_bert_cont30_bg.sh`；PID `temp/logs/last_bert_cont30.pid`；运行目录 `temp/logs/bert_da_fixed_cont30_<ts>/driver.log`。
- 输出：`logs/fed/bert_da_fixed_cont30/<arm>/<tag>/{metrics.jsonl,adapter_round030..059,eval_round030..059*.json}`；
- 归档：`checkpoints/distilbert_ablation_r30_fixed_cont30/<arm>/{adapter_round059, metrics.jsonl, config.json, SHA256SUMS}`；
- 报告：`temp/bert_cont30_report.py` → `temp/logs/bert_cont30_report.txt`（r29→r59 衔接、配对检验、%/h）。

## 四、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 16:10 | 修改 | `src/fed/run_fed.py` | +`--round-offset`；修 VAR-off None sum |
| 16:12 | 新增 | `temp/smoke_bert_cont30.sh` | 续训机制 smoke（通过） |
| 16:15 | 启动 | `temp/run_bert_cont30_bg.sh` | 4 臂续训 r30→r59，后台 |
| 16:15 | 新增 | `temp/bert_cont30_report.py` | 续训报告脚本 |
