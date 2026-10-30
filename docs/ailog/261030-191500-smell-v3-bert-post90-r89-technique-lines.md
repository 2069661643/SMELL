# Ailog 261030-191500 — BERT 技术线第三段续训 r60→r89（lr=1.4e-3/3）完成

> 用户指令（261029）："续训技术线到 r90"（= 第三个 30 轮块 r60–r89，端点与 4 臂 cont60 的 r89 对齐）。
> 三条线均从各自 cont30 的 `adapter_round059` 起，`--round-offset 60 --rounds 30`，**lr=4.6667e-4**（与 4 臂 cont60 同口径）。
> 链 15:24:32 启动，10/30 18:38:30 完成（dense 10.83h + 3-on 7.17h + VQ 6.43h，含等待/归档）。

## 一、结果

| 线 | r59 | r60 | **r89** | 段内% | r59→r89% | acc r89 | %/h | 配对（n=500） |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense VAR（sparse=1.0） | 1.3437 | 1.3417 | **1.3096** | +2.39 | **+2.53** | **.520** | +0.22 | −0.0341（t=−4.38，p=1.2e-5） |
| 3-on（VAR+CATV client+VQ pr） | 1.4896 | 1.4921 | **1.4598** | +2.17 | **+2.00** | .480 | +0.30 | −0.0299（t=−6.19，p=6.1e-10） |
| VQ 0.5 per-result | 1.5139 | 1.5130 | **1.4780** | +2.31 | **+2.37** | .494 | +0.36 | −0.0359（t=−7.55，p=4.4e-14） |

## 二、r89 横向（全部 lr/3 段后，同协议）

| 线 | r89 NLL | acc | 备注 |
|---|---:|---:|---|
| **dense VAR（sparse=1.0）** | **1.3096** | **.520** | 非 Jenga（不剪枝），不可与 sparse 线直接同表 |
| 3-on（VAR+CATV+VQ pr） | 1.4598 | .480 | vs a03_on **−0.0012 NLL**、acc −0.8pt |
| 4 臂 a03_on（CATV） | 1.4610 | .488 | |
| 4 臂 a01_on（CATV） | 1.4659 | .490 | |
| 4 臂 a01_off | 1.4703 | .502 | |
| 4 臂 a03_off | 1.4724 | .490 | |
| VQ 0.5 per-result | 1.4780 | .494 | vs a03_off **+0.0056（+0.38%）**、acc +0.4pt |

**解读**
1. **收敛显著变慢**：本段三条线都只再降 2.0–2.5%（上一段 4.4–10.0%），%/h 0.22–0.36；但仍全部显著（p≤1.2e-5）。
2. **3-on 在 r89 与 CATV-on 单技术基本持平**：NLL −0.0012（可忽略）、acc −0.8pt。结合 4 臂在 r89 的 CATV 净效应已不显著（a01 p=0.675、a03 p=0.281），说明**长训后 CATV 的优势窗口已过**。
3. **VQ per-result 的代价随轮数缓慢累积**：r29 时 −0.0016（略优）→ r89 时 +0.0056（+0.38%），仍远小于 per-client 的 +0.0387（r29）。
4. **dense VAR 仍是最强 BERT 线**（1.3096/.520），领先最好的 sparse 线 0.15 nats（−10.4%）。

## 三、技术统计（r89）

| 线 | VAR | VQ | 其他 |
|---|---|---|---|
| dense VAR | D=8.08、samples=64.7（目标 64）；T60/T70/T80 = 13.756/15.094/18.581 | — | `delta_norm_clipped_mean` 0.749 |
| 3-on | D=9.37、samples=74.9；T60/T70/T80 = 8.167/5.321/5.194 | C=74.9、cos **0.9952**、82.87 MB/轮 | CATV r=0.2、client 归一化；clip 0.531 |
| VQ pr | — | C=64、cos **0.9945**、70.78 MB/轮 | clip 0.613 |

- VAR 阈值在本段继续上漂（dense 11.38 → 13.76 → 15.09 → 18.58），但 D 稳定在 8.0–8.1；
  3-on 的阈值在 r70/r80 回落到 5.2（D 仍 8.0），说明 CATV mask 让 raw var 尺度更低。

## 四、caveat

1. **dense VAR 的 sparse=1.0**：走 `sdpa_prune` 的 dense fallback（无 Jenga 剪枝），与 sparse 0.4 线不是同一推理成本；论文里要单列。
2. **CATV mask 在 r60 重置**（跨 run 不继承），与之前每段续训一致；3-on 的 r60 首轮无掩码。
3. **`vq_seeds.json` 只在 round 0 写**：续训段（offset>0）归档里没有该文件；VQ seed 恒为默认 0、c_dim=18432，可离线重建（本次已把 var 探针文件补进归档）。
4. 3-on 的 VQ 仍是 **per_result** 语义（C=sum(D)≈74.9），不是 per-client。

## 五、产物

| 归档 | 内容 | adapter_round089 sha256 |
|---|---|---|
| `checkpoints/distilbert_fwdllm_a03_var_cont60/` | r89 + metrics + config + var_thresholds + probe60/70/80 + SHA | `ef77447f0abe2c5151e93046dbeb6117c55e63b61acfa5eb8e7544beff63dce6` |
| `checkpoints/distilbert_var_catv_vq_a03_cont60/` | 同上 | `831cb7f0b2757187586a1d31fc6a1a647525397e2da4f77dccdb7590836914bf` |
| `checkpoints/distilbert_vq050_perresult_a03_cont60/` | r89 + metrics + config + SHA | `362e473a23910273da5a732c6bba6541119985e68da584e86cb45417f6594879` |

- 报告：`temp/logs/bert_post90_report.txt`；日志：`temp/logs/bert_post90_261029-152432/`。

## 六、待办

1. **4 臂 cont90 未跑**（脚本 `temp/run_bert_cont90_bg.sh` 就绪；按边际收益 ~2%/30 轮，性价比需评估）。
2. 报告包 `reports/bert_and_other`：并入 cont60（4 臂）与 post90（3 线）的 r89 端点/曲线/配对。
3. 论文口径：BERT 主表建议以 **r89** 为长训端点；CATV 结论限定"中前期有效"，长训后不显著。

## 七、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10/29 15:24 | 启动 | `temp/run_bert_post90_chain_bg.sh` | 3 线 r60→r89 |
| 10/30 18:38 | 完成 | 3 线 r89 + 归档 + 报告 | 全部 rc=0 |
| 10/30 19:1x | 修改 | `AGENTS.md` / `docs/weight-manifest.md` | 技术线 cont60 行 + §14 |
| 10/30 19:1x | 新增 | 本 ailog | 结果 + 横向 + caveat |
