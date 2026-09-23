# SMELL-v2-debug — D3「PPL 极小摆动」归因：确认聚合更新为随机游走 + 观测盲区修复

## 概述

针对用户问题「高 epoch vs 之前 LoRA 的 PPL 差异，与当前噪声摆动极小，两者归因是否一致」，先用**零训练成本**的离线分析判定，再修观测盲区。结论：

- **两者终极根因一致**（都 ≈ 未训练基座、都不学习，B1）。
- **但摆动幅度的差异不是同源**：D3 的 `~4e-5` vs 之前 LoRA 的 `~0.13`，由**参数子空间/参数化**决定，不是「学得多/少」。
- 高 epoch D3 的「逐位冻结」与当前「4e-5 摆动」是**同一现象**（更新都在打印分辨率以下）。

## A2 方向分析（新增脚本，读已有 checkpoint）

`FwdLLM/.../run_lm_exps/analyze_update_direction.py`，读 `checkpoint_20260913_103356_r1..r59.h5`：

| 指标 | 值 | 解读 |
|---|---|---|
| `‖Δθ‖` | 9.95e-3 → 7.05e-3（线性降） | **恰等于当轮 lr**（lr_decay 0.01→1e-5，r59≈0.00705）→ 位移 = lr × 单位化随机方向 |
| `cos(Δθ_r, Δθ_{r-1})` | mean **-0.0014**, mean\|cos\| 0.0053 | 近似随机 |
| `cos(Δθ_r, Δθ_1)` | mean\|cos\| 0.0021 | ≈ 随机基准 `√(2/πd)=0.0021`（d=144384） |
| `Δloss` | 全 0.000000 | 日志 4 位分辨率盲区被直接暴露 |

**判定**：前向梯度未提供有效下降方向（B1 成立）；聚合后每轮是「步长=lr 的随机游走」→ **增大 lr 只会加大扩散，不会带来学习**（解释了 M2 放大步长导致 NaN）。

## 变更表

| 时间 | 操作 | 文件 | 说明 |
|---|---|---|---|
| 12:20 | 新增 | `FwdLLM/.../run_lm_exps/analyze_update_direction.py` | A2：逐轮 Δθ 方向/模长分析，输出 cos 与随机基准对照 |
| 12:20 | 新增 | `FwdLLM/.../run_lm_exps/eval_ppl_sensitivity.py` | A1：注入已知 ‖Δθ‖ 测 PPL 响应曲线（12:35 于 GPU1 运行完成） |
| 12:35 | 归档 | `log/`、`checkpoints/lm_head_d3_sample1_e16_cr201_len4k/` | 结束 D3 诊断 run（r61，12:21 停止），62 个 r*.h5 + consensus_history + 日志归档 |

## A1 结果（灵敏度标定，r=4，50 样本 test）

| 注入 ‖Δθ‖ | PPL | d_ppl |
|---|---|---|
| 0 | 36.741016 | 0 |
| 1e-5 | 36.741016 | 0 |
| 1e-4 | 36.741016 | 0 |
| 1e-3 | 36.741016 | 0 |
| 1e-2 | 36.741011 | -0.000005 |

- **eval 链路未断**：1e-2 扰动确有响应（5e-6）→ 排除「LoRA 未进 eval forward」。
- **lm_head LoRA 的 PPL 杠杆极低**：`delta_W=(α/r)B@A` 摊到 32000 vocab 行，每行 ~1e-5。
- **与 A2 定量自洽**：随机扰动 `ΔL≈c‖Δθ‖²`，A1 得 c≈0.05；A2 累计位移 `‖θ_r−θ_0‖≈6.5e-2`（r59）→ 预测 `ΔL≈2.1e-4`，与实测漂移 2e-4 吻合。
- **结论**：D3 的 PPL 漂移 = 随机游走扩散 × 二次低杠杆，无学习成分；PPL 判据对 D3 不灵敏（方向随机 + 参数化杠杆低，双重叠加）。
- 注：A1 baseline 36.74 ≠ 训练内 `[EVAL-PPL]` 45.04，因两者评估子集不同（`H5PPLDataset` 前 50 vs `eval_ppl_fixed` seed42）；仅档间相对量有效。
| 12:20 | 修改 | `FwdLLM/FedML/.../fedsgd/FedSgdClientManager.py` | B3：`[EVAL-PPL]` 精度 `.4f`→`.6f`，新增相对 baseline 的 `d_ppl/d_loss` |
| 12:20 | 修改 | `FwdLLM/training/fed_trainer_transformer.py` | B4：`[DBG-SETPARAM]` 新增 `delta_norm`（与上轮参数比较）与分张量 Δ，替代对随机游走单调的 `\|sum\|` |

## 备注 / 待办

- `fed_trainer_transformer.py` 工作区此前被整体转为 CRLF（其余文件为 LF）；本次已从 HEAD 重建、仅保留本改动行，commit diff 保持 19+/3- 干净。
- A1 已在 D3 run 停止、GPU1 释放后运行完成（见上）。
- B3/B4 改动对已结束的 D3 进程不生效（进程加载旧模块）；下次 run 才体现。
- 下一步：PPL 对 D3 不灵敏，建议改用一阶敏感判据（训练 batch loss 高精度 / next-token 准确率）或做 C5（exact head-gradient 对照），直接判定瓶颈在 ZOO 估计器还是聚合/参数化。
