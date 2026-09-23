# SMELL-v2-debug — 固定测试集 eval PPL 长跑完成：B1 收敛问题确定性确认

## 概述

`a0.1_sample10_len4k_datagoe`（sparse 0.4 / c3 / alpha 0.1 / len 4k / samples 10 / max_resample 200 / `--eval_ppl 50`）跑满 25 轮，约 5h3m。这是首个用**确定性固定测试集 eval**（`[EVAL-PPL]`）的完整 run，得到干净的跨轮 PPL 曲线，**确定性确认 B1：前向梯度无法驱动 LoRA 收敛**。

## 结果

### 跨轮 EVAL-PPL（24 个完整点，round 0–23；round 24 被 MPI_ABORT 截断）

- PPL：`min=44.730 max=45.355 mean=44.966 std=0.128`
- eval_loss：`mean=3.8059 std=0.0028`
- 线性斜率：`-0.0057 PPL/round`（24 轮累计仅 -0.13，完全在噪声带内）

### 结论

- 旧 `[SMOKE-PPL]`（随机 30 训练样本 + 重采样噪声）掩盖了真相；新固定测试集口径下噪声极小（std=0.13），PPL 平躺 → **训练不收敛，B1 成立**。
- 根因：前向梯度方向噪声过大（`grad_norm` ~600、`var_threthod=1` 卡 4k 噪声地板），聚合后 LoRA 几乎不动。

### resample / 开销

- 总训练事件 2417，实际重采样 ≈2167 → avg **~87 次/round**（~8.7/data_id）。
- `MAX-RESAMPLE`（200 上限）仅命中 1 次。
- eval 开销稳定 ~35.8s/轮（50 样本 / 862 token）。

### 发现 bug（次要）

最后一轮（round 24）eval 被截断：Server 发完最终 model + FINISH 后立即 `self.finish()` → MPI_ABORT，客户端 36s eval 中途被 kill（`[EVAL-PPL] round=24 START` 后无 END）。不影响已得结论，后续可修。

## 文件操作

### 归档

`log/smell-v2-debug-len4k_20260911_183330.log` → `log/a0.1_sample10_len4k_datagoe_eval/`

`checkpoints/checkpoint_20260911_183335_r1..r25.h5` + `global_lora_final_20260911_183335.h5` + `consensus_history_20260911_183335.h5` → `checkpoints/a0.1_sample10_len4k_datagoe_eval/`

## 下一步

- [x] 启动下一轮：samples 10→**20**，epochs 1→**2**，其余不变（len 4k / sparse 0.4 / alpha 0.1 / max_resample 200 / eval_ppl 50），观察更大样本量 + 多 epoch 能否改善方向噪声。
- [ ] 修 round 24 eval 截断（可选）。
