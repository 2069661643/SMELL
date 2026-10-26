# Ailog 261026-130000 — cont30 eval-off 分解（掩码收益变大/权重代价变大）+ cont60 out-root 事故与修复

> 两条线：① 10/25 完成的 cont30 eval-off（on 臂 r30–r59）；② 10/26 发现 cont60 续训脚本 **out-root 误用**，
> 4 臂写进同一个 `a01`/`a03` 目录，造成部分权重丢失；已停机并按确定性重跑修复。

## 一、cont30 eval-off（10/25 13:42 完成）

产物：`checkpoints/distilbert_ablation_r30_fixed_cont30/<arm>/evaloff/`（2 臂 × 30 轮 × 2 文件 + SHA256SUMS）；
汇总：`temp/logs/bert_cont30_evaloff_summary.txt`。

| α | r29 掩码效应 | r59 掩码效应 | seg2 均值 mask ΔNLL | r59 权重差（on-arm off − off 臂） | r59 净差 |
|---|---|---|---:|---:|---:|
| a01 | −0.020501（t=−1.33，p=0.184） | **−0.038582**（t=−2.46，p=0.0137） | −0.032941 | **+0.024271** | −0.01431 |
| a03 | −0.028784（t=−1.75，p=0.0805） | **−0.047220**（t=−3.07，p=0.00215） | −0.042616 | **+0.021419** | −0.02580 |

解读（修正此前“CATV 优势缩小”的单因叙述）：
- **掩码本身的价值随训练上升**（−0.02 → −0.039/−0.047，r59 均显著）；
- **on 臂的原始权重随训练变差**（r29 时 on 臂 off-mask 甚至略好 ~−0.007；r59 变差 +0.021~+0.024）；
- 两者相抵 ⇒ 净优势缩小（−0.027/−0.040 → −0.014/−0.026）。
- 论文口径：CATV 的部署收益 = 推理掩码收益（增长）+ 权重代价（增长）；单看净差会低估掩码项。

## 二、cont60 事故（10/26 12:30 发现）

### 2.1 原因

`temp/run_bert_cont60_after_evaloff_bg.sh` 里 `OUT="$REPO/logs/fed/bert_da_fixed_cont60"` 是**共享** out-root，
`--tag` 只有 a01/a03 → 同 tag 的两个臂写同一目录（`.../a01` 或 `.../a03`），后跑的臂把前一个臂同轮号的
`adapter_roundNNN` / `eval_roundNNN*.json` 覆盖（metrics.jsonl 是追加，未丢）。

### 2.2 时间线与损失

| 臂 | 实际运行 | 结果 | 损失 |
|---|---|---|---|
| a01_off | 10/25 13:43 → 20:55 | 30/30，r60 1.5047 → r89 1.4703 | **adapters + eval JSON 全部被 a01_on 覆盖**；metrics 保留 |
| a01_on | 20:55 → 10/26 04:10 | 30/30，r60 1.4871 → r89 1.4659 | 无（最后写入者） |
| a03_off | 04:10 → 11:24 | 30/30，r60 1.5097 → r89 1.4724 | **r60–64 被 a03_on 覆盖**；r65–89 + metrics 完好 |
| a03_on | 11:24 → 12:40（kill） | 5/30 | 作废（partial 已移走） |

- 12:40 整组 kill，GPU 释放；a03_off 的 r65–89 因此保住。
- 已按 metrics 的 eval 序列切分：a01 前 30 条 = a01_off、后 30 条 = a01_on；a03 前 30 = a03_off、后 5 = a03_on。

### 2.3 修复

1. 目录重建（已完成）：`logs/fed/bert_da_fixed_cont60/<arm>/<tag>/`
   - `r30_a01_catv_on/a01`：30 adapters + 30 eval + 30 persample + metrics（后 30 条）；
   - `r30_a03_catv_off/a03`：r65–89（25 adapters/evals）+ metrics（前 30 条）+ config（catv=off 手工修正）；
   - 混合残留移到 `temp/cont60_mixed_{a01,a03}_<ts>/`，a03_on partial 移到 `temp/cont60_partial_a03_on_<ts>/`。
2. 重跑（`temp/run_bert_cont60_fix_bg.sh`，12:41:43 起）：
   - a01_off：30 轮（offset 60，lr 4.6667e-4）；
   - a03_on：30 轮（干净目录）。
   - 因为两段都是**从 r59 + offset 60 + 固定 seed 新起进程**，方向序列与原 run 相同，重跑应逐位复现。
3. a03_off r60–64 的确定性恢复（`temp/run_cont60_a03off_restore5_bg.sh`，等主链结束）：
   - 从 r59 跑 5 轮（offset 60），与已存 metrics 的 r60–64 对比；**|ΔNLL|<1e-4 才补回** adapter+eval；
   - 不论是否补回，都会重生成报告。
4. ETA：a01_off ~19:10（10/26）→ a03_on ~01:35（10/27）→ 归档 + 报告 → restore 复查 ~02:40。

### 2.4 教训

- 续训脚本的 out-root 必须**每臂独立**（cont30 脚本就是 `$ROOT/$NAME`）；共享 out-root 只会靠 `--tag` 区分，同 tag 的臂会互踩。
- 修复脚本加了两道保护：a03_off 的 5 轮恢复做 **eval 一致性校验**；归档前检查 `adapter_round089` 是否存在（本次已因缺失触发 ARCHIVE SKIPPED，暴露问题）。

## 三、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10/24 13:42 | 完成 | cont30 eval-off | 2 臂 × 30 轮；掩码效应分解 |
| 10/25 20:55 | 启动 | cont60 4 臂（错误 out-root） | 12:40 发现 |
| 10/26 12:40 | 停机 | cont60 链 | 保护 a03_off r65–89 |
| 10/26 12:41 | 启动 | `run_bert_cont60_fix_bg.sh` | 重跑 a01_off + a03_on |
| 10/26 12:42 | 启动 | `run_cont60_a03off_restore5_bg.sh` | 等主链后恢复 a03_off r60–64 |
| 10/26 13:0x | 修改 | `docs/weight-manifest.md` §12 | eval-off 分解表 |
