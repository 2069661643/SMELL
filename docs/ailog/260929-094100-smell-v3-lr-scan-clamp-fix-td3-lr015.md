# Ailog 260929-094100 — Phase N: lr 标定扫描（D）+ clamp 修复 + TD-3@a03 以 lr=0.15 重启

> 承接 `260929-000200`（审计修复）。上一轮 TD-3@a03（`zoo_k4_sparse_causal_a03_step5`）**空转作废**：
> lr 校准值 0.15 被脚本 clamp `[1e-7,1e-4]` 夹低 1500×，13 轮 loss/PPL 恒定、配对检验 NOT-SIGNIFICANT。

## 一、空转证据（13 轮，已停）

- `loss_mean` 恒 = 3.5205、`full_ppl_token` 恒 = 33.665（逐轮一致）；`delta_norm` 后期仅 ~2e-4。
- 配对检验（`scripts/paired_eval.py` r0 vs r12）：mean ΔNLL=+5.8e-6，p=0.23，ppl_ratio=1.000006。
- 原因：修复后的真实探针 δ=1.0147e-5 @lr 4e-6 → 校准 lr = 4e-6×0.38/δ = **0.150**，被 clamp 上限 1e-4 夹低。

## 二、D：lr 标定扫描（`temp/run_lr_scan_bg.sh`，1 client × 2 samples × 3 轮，k4/D22/ls2/step5）

| lr | loss r0→r2 | δ r0/r1/r2 | δ/lr | 备注 |
|---|---|---|---|---|
| 0.01 | 3.5754→3.5754 | 2.44e-2/2.98e-2/2.60e-2 | ~2.5 | 无变化 |
| 0.03 | 3.5754→3.5749 | 7.27e-2/8.83e-2/8.38e-2 | ~2.7 | −0.0005 |
| 0.10 | 3.5754→3.5742 | 2.52e-1/3.05e-1/3.37e-1 | ~2.9 | −0.0012 |
| 0.15 | 3.5757→3.5735 | 3.62e-1/4.28e-1/3.98e-1 | ~2.7 | −0.0022 |
| 0.30 | 3.5758→3.5724 | 8.28e-1/1.10e0/9.06e-1 | ~3.1 | −0.0034 |

- **δ∝lr 线性成立**（30× 范围内 δ/lr 恒定 2.4–3.7）；0.3 档无 NaN/inf；峰值显存 4.39 GiB；66–67 s/轮/client。
- loss 随 lr 单调下降 ⇒ 修复后的 ZOO 更新**方向有效**（单 client 三轮量级 −1e-3~−3e-3）。
- 选定 **lr=0.15**（对齐 δ≈0.38 的 BP k=4 目标）；0.3 作为后续可选档位。

## 三、代码修复

| 文件 | 修复 |
|---|---|
| `scripts/run_k4_td2_td3.sh` | lr clamp `[1e-7,1e-4]` → `[1e-7,1.0]`，输出 `raw` 并在触界时 WARNING（旧上界曾致 1500× 夹低空转） |

## 四、重启

- `temp/run_td3_a03_lr015_bg.sh`（09:40:31 起）：30 client、k4 rotate4、D22、ls2、step5、fp32+sdpa_prune、**lr=0.15**、30 轮、每轮 eval + `eval_roundNNN_persample.json`。
- 产物：`logs/fed/zoo_k4_sparse_causal_a03_lr0.15/`；日志 `temp/logs/last_td3_a03_lr015_dir.txt`。
- 预计 ~39–40 min/轮（33 min 训练 + ~6 min eval）→ 30 轮 ≈ 20 h。

## 五、待办

- [ ] 前 3 轮盯 `delta_norm`（~0.38 量级）、loss/NaN、G-PPL；
- [ ] round 0 vs N 配对检验（`scripts/paired_eval.py`）；
- [ ] 云端 A40 待用户处理：pull（含 clamp 修复 + step5）后按同参数重跑；旧 TD-3 作废。
