# SMELL-v2-debug — 回顾 D3/PPPL 平躺结论 + 过时文档同步（summary/AGENTS/README）

## 概述

承接 D3 lm_head 诊断运行（`smell-v2-debug-len4k-d3lmhead_20260913_103351.log`）：用户要求回顾「PPL 几乎不动」的发现、判断是否为 **lm_head 链路问题**，并一次性更新过时文档。

本次仅改文档 + 新增本 ailog，未改代码。

## 回顾：D3 诊断结论（round 17 时点）

| 观察 | 结果 | 判定 |
|---|---|---|
| `[DBG-SENDPARAM]`/`[DBG-SETPARAM]` | 成对、checksum 一致（131.109…）、`translated matched=2` | server→client 消息与 key 翻译正常 |
| `post-load trainable |sum|` | 每轮变化（15 轮 131.1 → 135.4，单调） | 聚合结果**确实加载进 client**，M1 对 D3 生效 |
| `[EVAL-PPL]` | baseline 45.0380，round 0-10 恒为 45.0380，round 11-14 45.0379，round 15-17 回到 45.0380 | 参数有更新但**几乎无学习信号**（1e-4 级摆动，随机游走） |

**结论：不是 lm_head 链路/同步问题**，而是 B1（前向梯度 `SNR=√(M/d)` 过低）——与 train/test PPL 同时平躺（`260912-094314`）、r=1 仍平躺（`260912-100520`）一致。lm_head LoRA 只是把 d 从 1.05M 降到 144K，SNR 提升有限（r=4 时 `cosθ≈14.6%`），且聚合步长下仍有噪声主导。

## 变更表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 11:06 | 重写 | `docs/summary.md` | 全文 | 日期 09-09→09-13；§1.3 补 Phase 4/5；§1.4 改为 M1「已验证生效」+ M2 现方案（归一化/clamp/lr_decay）；§2/§3/§4 更新 B1 根因（SNR）、D3 结论、降 d 路线；§5 索引更新 |
| 11:06 | 修改 | `AGENTS.md` | 环境；运行实验；关键架构；状态；文件参考 | 环境 `smell`→`smell-v2`；补 v2 脚本；补 D3/聚合归一化/batch=1 限制；状态改为「同步已验、瓶颈在信号」；参考表补 `eval_train_vs_test.py`/`FedSgdTrainer.py`/v2 脚本/`phase5_task_head_plan.md` |
| 11:06 | 修改 | `README.md` | Quick Start 环境 + 脚本段 | 环境 `smell`→`smell-v2`；加「尚未收敛」现状说明；补 Phase 4/5 调试脚本 |
| 11:06 | 新增 | `CLAUDE.md`（上一轮） | 仓库根 | 面向 Claude Code 的命令/架构/规范/现状（已含 D3 结论） |

## 说明 / 待办

- 本次未修改 `docs/SMELL/phase{1,2,3}*.md`（历史设计记录，保留原貌）。
- 文档结论以「D3 诊断进行中、round 17」为准；运行结束后如 PPL 出现趋势需再同步。
- 待办（代码层，未动）：D3 r=4 容量/步长复核；`var_threthod` 相对阈值化；Phase 3 vote 按 data_id 重置。
