# Ailog 261014-153500 — Phase N: 训练前共享基线补测（fresh，与 BERT 并行）+ T7 改共享基线口径

> 用户指出审稿人会问「为什么各臂不用同一基线」。此前所有 r0 均为**第 0 轮训练后**状态
> （代码顺序 + lora_B≠0 双重确认，见 §证据），训练前基线从未测过。本次补测并把 T_D 改为共享基线。

## 1. 训练前基线（2026-10-14 15:06–15:18，与 BERT 任务同卡并行）

- 命令：`temp/run_fresh_baseline_bg.sh`（= `ppl.py --tag a03 --attn sdpa_prune --dtype fp32 --sparse 0.4`
  + `--pos-checkpoint a01_pos_only_500step` + step5 predictor，**无 --adapter**，500 条全局测试）。
- 原理：PEFT LoRA 初始化 B=0 ⇒ `get_peft_model` 后模型与 base 逐位等价，故「无 adapter + pos_only」
  就是训练起点。
- 结果：**full G-PPL 33.6653**（sample-mean 33.6858）/ **answer G-PPL 1740.63**；
  对比各臂第 0 轮后 r0（33.521–33.533）⇒ 第 0 轮降 ~0.13 full，answer 略升（与已知 answer 退化一致）。
- 产物：`temp/eval_fresh_baseline.json`（+ `_persample.json`）；日志 `temp/logs/fresh_baseline_261014-150650/`。
- 并行说明：BERT 任务 1.6GB + 评测 3.3GB，显存无压力；评测墙钟 ~12 min（共享算力），BERT 正常。

## 2. 口径证据（为何以前的 r0 不是训练前）

1. `run_fed.py`：save/eval 块均在「本轮 client 训练 + 聚合」之后 ⇒ `adapter_round000` = 第 0 轮后。
2. 实读 `adapter_round000/adapter_model.safetensors`：lora_B 非零（k4 6.66 / D16 236.6 / D32 158.4）。
3. 三臂 r0 互不相同（33.5332/33.5211/33.5296）；训练前状态应逐位相同。
4. `temp/ppl16k_compare/*.json`（唯一的无 adapter 评测）为 **n=4 且 pos_checkpoint=None**，不可用作基线。

## 3. T7（D 对照）改为共享基线

| D | 窗口 | baseline(pre-r0) | 端点 | Δ vs baseline | Δ% |
|---|---|---|---|---|---|
| 16 | pre-r0→r7 | 33.6653 | 32.7144 | −0.9510 | −2.82% |
| 22 | pre-r0→r7 | 33.6653 | 32.5335 | −1.1318 | −3.36% |
| 32 | pre-r0→r7 | 33.6653 | 32.5071 | −1.1582 | **−3.44%** |

- 行内保留各臂 own-r0（33.5332/33.5211/33.5296）及其 delta（−0.819/−0.988/−1.023）作脚注，两口径可互查。
- 对账新增 `fresh baseline full = 33.6653`，总 39/39 PASS。

## 待办 / 待作者定夺

- 是否把共享基线口径推广到 **table1（主结果表）**：加一列 `Δ vs baseline 33.6653`（保留 own-r0 列）？
- table6（CATV r44/r74 端点）的 deploy 行有自己的掩码锚点（33.9393/33.9211），训练前无掩码不可比；建议保留 own-r0 并注脚。
- fig1/fig5/fig6 是否加一条水平虚线 baseline（33.6653）？

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 15:06 | 启动 | `temp/run_fresh_baseline_bg.sh`（与 BERT 并行） | 训练前基线评测 |
| 15:18 | 完成 | `temp/eval_fresh_baseline.json` | full 33.6653 / answer 1740.6 |
| 15:25 | 修改 | `reports/opt/make_report.py` | `fresh_baseline()` + T7 改共享基线 + 新断言 |
| 15:27 | 生成 | `reports/opt/tables/table7_d_comparison.*`、`data/*.csv`、README | 共享基线口径 |

## 4. 追加（15:45）：table3b 数字订正（论文引用发现）

- 作者转述论文句"dense 26.5 vs 稀疏（ρ=0.4）33.5"→ 与 `table3b` 当时的显示值一致，但 **33.5296 是贴错的值**
  （误用 D32 臂的共享 r0；同一适配器应为 TD-3 的 r0=33.6538）。paired ratio（0.9890/1.0671）本身一直用的
  是 TD-3 的 r0↔r29 配对，故修表后 ratio 与数值自洽。
- 修正：`table3_quality` 的 sparse 行改为 `logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/eval_round000.json`
  （33.6538/1743.8）；新增 4 条对账（dense r0/r29、sparse TD-3 r0/r29），总 43/43 PASS。
- 环境注脚补齐：OPT-350M + pos_only、TD-3 适配器、fp32/sdpa_prune、n=500；dense=同适配器 sparse=1.0 重评。
- T3a 注脚补"未加载位置适配（pos_checkpoint=None）"。
- 待办：dense JSON 未记录 pos 标志（09-30 手动评测）；如要完全可审计可重跑一次 dense 评测（~10 min）。
