# Ailog 261012-153000 — SMELL-v3 DistilBERT：`--doc-order {fixed,cycle}` 实现 + smoke，BP cycle 补测已启动

> 承接交接文档 `261012-151805-...fixed-60docs-protocol-flaw-and-ld-plan-handoff.md`（P0 第 1 项）。
> 本文件记录：接口实现与 smoke 证据；BP cycle 协议补测的启动与结果（结果段完成时回填）。

## 1. 实现（`src/fed/run_fed.py`）
- 新增 `--doc-order {fixed,cycle}`（默认 `fixed`，旧行为零变化）：
  - `fixed`：每轮从分片第 0 行顺序取 L 条（OPT/discovery 老协议，legacy 保持可比）；
  - `cycle`：本轮起始行 `start = (round_idx * local_steps) % 分片总行数`，从 start 起取 `local_steps` 条、**按分片总行数卷绕**；
- `iter_batches` / `iter_classification_batches` 增 `start=0` 参数（`(start + k) % total`）；
- 记录：`run_config["doc_order"]`；每条 round 记录 `doc_order` + `round_start`；`client_stats[name]["round_start"]`（复现/核对）。

## 2. Smoke 证据（本机，2026-10-12 15:25）
- 单元：4 行合成分片，`start=0/2/4/6` → 行号 `[0,1]/[2,3]/[0,1]/[2,3]`（卷绕正确）；LM 版 `iter_batches(start=6)` → `[2,3]`。输出 `UNIT_OK`。
- 端到端：`run_fed --arch distilbert --trainer bp --doc-order cycle --rounds 4 --local-steps 2`（a01mini 4 行/人）→
  round_start 序列 `0, 2, 0, 2`，metrics/config 字段齐全，产物 `temp/smoke_docorder/`。
- `fixed` 路径未改动（默认参数；旧调用点不传 start）。

## 3. BP cycle 补测（已启动，`temp/run_bert_bpref_cycle_bg.sh`）
| 格 | 协议 | 目的 | 估时 |
|---|---|---|---|
| dense_L8_lr3e4_r10 | cycle, c30, 250 样本评测 | cycle 主参考（δ 锚 + 速率） | ~30 min |
| dense_L8_lr1e3_r10 | 同上 | lr 复查（旧「1e-3 崩」判定是否在 cycle 下成立） | ~30 min |
| sparse_L8_lr3e4_r10 | cycle + sdpa_prune/predictor | ZOO 同路径参考 | ~40 min |
| dense_L8_lr3e4_r5 | cycle, 5 轮 | iso-语料 40 篇（对照下格） | ~15 min |
| dense_L2_lr3e4_r20 | cycle, 20 轮 | iso-语料 40 篇（低成本），H1/H2 早期判读 | ~40 min |
| probe_zoo_L8 / L16 | 1 client × L 篇, D22, lr 4e-6 | ZOO δ 锚（lr0 标定） | ~5 min |
- 产物：`logs/fed/bert_bp_ref_cycle/*`；driver：`temp/logs/bert_bpref_cycle_261012-152811/`。
- 对照素材（旧 fixed 协议）：`logs/fed/bert_bp_ref/lr3e4_ls8_r10`（+1.50%/10r，8 篇唯一）用于 H2 的 fixed-vs-cycle 匹配对照。
- 注意：LD 表（P1）按用户要求**等本补测结果后再定参数**（旧「L=8 才有效」结论已被 60 篇缺陷推翻）。

## 4. 结果（回填区）
- （待补测完成后填写：各格 ΔNLL%/ΔAcc/%/h/δ、H1/H2 早期判读、lr 复查、lr0。）

## 5. 变更列表
| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 15:25 | 修改 | `src/fed/run_fed.py`（+26/−7） | `--doc-order {fixed,cycle}` + 取数卷绕 + 记录字段 |
| 15:28 | 新增 | `temp/run_bert_bpref_cycle_bg.sh`（gitignore） | BP cycle 补测链（5 格 + 2 探针） |
