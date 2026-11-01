# Ailog 261101-222000 — OPT cont30 out-root 冲突事故 + 恢复链

> 事故由 **10/30 的 OPT 四臂续训启动脚本**引入（`temp/run_opt_cont30_4arm_bg.sh`）：
> 四个臂共用同一个 `--out-root logs/fed/opt_r30_fixed_cont30` + `--tag a01/a03`，
> 因此 **a01_off 与 a01_on 写同一个目录** `.../opt_r30_fixed_cont30/a01`，后跑的 a01_on
> 覆盖了 a01_off 的 `adapter_round030..059` 与 `eval_round034..059{,_persample}.json`。

## 一、影响

| 臂 | 状态 | 数值 |
|---|---|---|
| a01_off | **权重丢失**（仅 metrics 有效，`LOST_WEIGHTS.md`） | r59 full_ppl **30.9529** / answer **2174.81**（metrics 行 append 保留） |
| a01_on | 完整（adapter_round059 + 6 evals） | r59 full **31.1256** / answer **2277.07** |
| a03_off | 旧链 11/1 20:07 起跑，21:16 被恢复链取代（旧目录只剩 config.json，作废） | 未完成 |
| a03_on | 未开始 | — |

- 旧共享目录状态：`logs/fed/opt_r30_fixed_cont30/a01`（空）、`.../a03`（仅 config.json）→ **均标注作废，不要引用**。

## 二、修复

`temp/run_opt_cont30_resume_bg.sh`：

- **每臂独立 out-root**：`logs/fed/opt_r30_fixed_cont30/<arm>/<tag>`；
- 顺序：**a03_off → a03_on → a01_off（重跑，仅为补权重/persample）**；
- 协议与 r30 fixed 完全一致（k24、sdpa_prune、fp32、sparse 0.4、L2、D32、rotate24、clip 13.34、cycle、lr 0.30、eval every 5）；
- 起点：`checkpoints/opt_ablation_r30_fixed/<arm>/adapter_round029`。

## 三、时间线（实测 ~24.4h/臂）

| 臂 | 状态 | 预计完成 |
|---|---|---|
| a03_off | 11/1 21:16 起跑 | **11/2 ~21:40** |
| a03_on | 排队 | **11/3 ~22:00** |
| a01_off（重跑） | 排队 | **11/4 ~22:20** |

- 报告：`temp/opt_cont30_report.py` → 链末自动写 `temp/logs/opt_cont30_report.txt`；
- 归档：`checkpoints/opt_ablation_r30_fixed_cont30/<arm>/`（新目录，含 adapter_round059 + metrics + config + SHA）。

## 四、教训

1. 多臂实验的 out-root **必须按臂隔离**（`<root>/<arm>/<tag>`），不能只靠 `--tag` 区分 α；
2. 复用 `--tag` 的并行/串行链在启动前要显式检查目标目录是否已被前一臂写过；
3. 事故后先核对 `metrics` 与 `adapter` 是否同源（本次 metrics append 保留了 a01_off 的数值，权重却被覆盖）。

## 五、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10/30 19:11 | 启动 | `temp/run_opt_cont30_4arm_bg.sh` | 共享 out-root（缺陷） |
| 10/31 19:35 | 完成 | a01_off 30 轮 | 权重随后被覆盖 |
| 11/1 20:07 | 完成 | a01_on 30 轮 | 覆盖 a01_off |
| 11/1 21:16 | 修复启动 | `temp/run_opt_cont30_resume_bg.sh` | 每臂独立 out-root |
| 11/1 22:1x | 新增 | 本 ailog | 事故记录 + 计划 |
