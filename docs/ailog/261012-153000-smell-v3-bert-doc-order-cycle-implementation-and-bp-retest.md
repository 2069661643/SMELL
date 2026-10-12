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

## 4. 结果（2026-10-12 17:41 全部完成）
基线（冻结头，同 250 样本子集）：acc 0.4880 / NLL 1.5317。

| 格 | 轮 | 唯一文档/人 | ΔNLL% | ERR↓ | ΔAcc% | 末 acc | δ/轮 | 秒/轮 | %/h(含评测) |
|---|---|---|---|---|---|---|---|---|---|
| **dense L8 lr3e-4** | 10 | 80 | **+8.90%**（单调） | +9.4% | +9.8% | 0.536 | 0.230 | 108 | **+20.9** |
| dense L8 lr1e-3 | 10 | 80 | −0.11%（r3 触底 1.372 后震荡） | +15.6% | +16.4% | **0.568** | 0.673 | 108 | −0.3 |
| sparse L8 lr3e-4 | 10 | 80 | +4.57%（自身路径 **+14.58%**） | +3.9% | +4.1% | 0.508 | 0.243 | 67 | +14.6 |
| dense L8 lr3e-4 | 5 | 40 | +3.89% | +6.2% | +6.6% | 0.520 | 0.228 | 108 | +18.3 |
| dense L2 lr3e-4 | 20 | 40 | +1.65% | +4.7% | +4.9% | 0.512 | 0.089 | 27 | +4.1 |

对照（旧 fixed 协议，同 L8/lr/轮数）：ΔNLL +1.50% / ΔAcc +4.4 点 / +3.2%·h⁻¹。

### 判定
- **H2（语料轮转）成立且是主杠杆**：同 L8×10 仅换 cycle → +8.90% vs +1.50%（6×）；
- **H1（L 本身）也真实**：等语料 40 篇、等训练时间（~9 min）下，L8×5 +3.89% vs L2×20 +1.65%（2.4×）；
- **lr 复查（cycle 下旧判定失效）**：lr=1e-3 不再崩（acc 0.568 最高），但 NLL 剧烈震荡（过自信）；lr=3e-4 单调稳定 → **稳定主选 3e-4**；
- **sparse 路径**：自身起点 1.7112（剪枝惩罚 ~0.18 nats），10 轮后 1.4617（自身 +14.6%），与 dense 差值缩到 0.066 nats（惩罚随训练缩小但不消失）；ZOO 臂基线须用 sparse 自身口径；
- **ZOO lr₀ 锚**（目标 δ≈0.230）：探针 L8 δ=6.871e-4 → **lr₀≈1.34e-3**；L16 δ=9.516e-4 → 9.67e-4（clamp [1e-4,1.0]，clip 1.0）。

### 对 LD 表（P1）的含义
- 中心配置取 **L8（可加 L16），弃 L2**（H1+速率双重证据）；
- D 轴按成本反比取舍：L8·D4 ≈ 3.2h/臂、L8·D8 ≈ 6.4h/臂（30 轮 ZOO 估）；D 越小 cos 越低但 %/h 可能更优——正需 P1 表分离；
- lr 轴保留（lr₀ 与 2×lr₀；1e-3 档已证非崩但 NLL 不稳）。

## 5. 变更列表
| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 15:25 | 修改 | `src/fed/run_fed.py`（+26/−7） | `--doc-order {fixed,cycle}` + 取数卷绕 + 记录字段 |
| 15:28 | 新增 | `temp/run_bert_bpref_cycle_bg.sh`（gitignore） | BP cycle 补测链（5 格 + 2 探针） |
