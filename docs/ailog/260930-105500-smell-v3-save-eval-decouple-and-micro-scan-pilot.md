# Ailog 260930-105500 — Phase N: save/eval 解耦（`--save-every`）+ 先导微扫描（D×L / lr×clip）

> 承接首个 ZOO 正结果（`260930-094100`）与 Step1–3（`260930-104500`）。本 session：
> （1）把「每轮存 adapter（秒级）」从「每轮 eval（分钟级）」中解耦，长跑可降 eval 频率而不丢中间权重；
> （2）启动先导微扫描，为「加速 full PPL」的三条改进（lr↑ / 稳健聚合 / 减 LD）定量选参。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 10:45 | 新增 `--save-every` | `src/fed/run_fed.py` | `parse_args`（默认 1；0=关） | 每 N 轮 `save_pretrained` + `{"event":"save"}` 指标行（秒级） |
| 10:45 | eval 复用已存 adapter | `src/fed/run_fed.py` | `run_global_eval` | `adapter_roundNNN` 已存在则不再重复保存 |
| 10:46 | eval 转发 `--truncate` | `src/fed/run_fed.py` | `run_global_eval` 命令拼装 | smoke 时 eval 与训练同截断，避免全 16k 评测 |
| 10:50 | smoke 验证 | `temp/smoke_save_decouple/` | 2 轮、2k、`--save-every 1 --eval-every 2` | round0 仅 save；round1 save+eval；eval 复用 adapter；指标事件正确 |

## 设计说明

- 现状（改前）：`run_global_eval` 同时负责「存 `adapter_roundNNN`」和「跑 `ppl.py`」；因此 `--eval-every>1` 会导致**未 eval 的轮次没有权重**，无法事后追评；而 `--eval-every 1` 时 eval（实测 367s/轮）占墙钟 15–27%。
- 改后：权重按 `--save-every` 落盘（r=1 adapter 很小），eval 频率独立；事后可用 `src/eval/ppl.py --adapter <adapter_roundNNN> --per-sample-out ...` 补评，再交给 `scripts/paired_eval.py`。
- 兼容性：`--save-every 0` + `--eval-every 1` 等价旧行为；`--eval-every 0` + `--save-every 1` 为「只存不评」新组合。

## 先导微扫描（`temp/run_pilot_micro_scan_bg.sh`，10:52 启动）

口径：1 client（client_00）× 2 samples × 3 轮、fp32+`sdpa_prune`、step5 predictor、k4 rotate4、r=1；每轮重建 iterator（每轮各 1–2 步）。

| Phase | 网格 | 目的 |
|---|---|---|
| A（6 格） | D∈{8,16,22} × L∈{1,2}，lr=0.15、clip=1.0 | 检验「减 D 是否保住单轮进度、只买吞吐」（成本∝LD） |
| B（4 格） | (D22,lr0.30,clip1)/(D22,lr0.30,clip2)/(D16,lr0.30,clip1)/(D16,lr0.30,clip2) | lr↑ 与 clip 抬升的交互；clip 对单 client 步长的截断折扣 |

产物 `temp/pilot_micro_scan/<cell>/a03/metrics.jsonl`；日志入口 `temp/logs/last_pilot_micro_scan_dir.txt`。

## 微扫描结果（11:15 完成）与决策

Phase A（lr0.15/clip1.0，1 client×2 样本×3 轮）：

| cell | loss r0→r2 | Δloss | Δloss/秒 | δ_r0 | sec/轮 |
|---|---|---|---|---|---|
| A_D8_L1 | 3.5826→3.5815 | −0.00111 | −2.90e-5 | 0.367 | 13 |
| A_D8_L2 | 3.5761→3.5735 | −0.00263 | −3.47e-5 | 0.580 | 26 |
| A_D16_L1 | 3.5826→3.5800 | −0.00262 | −3.55e-5 | 0.294（r1 原始 δ=239 重尾，被 clip） | 25 |
| A_D16_L2 | 3.5756→3.5747 | −0.00088 | −6.0e-6 | 0.441 | 49 |
| A_D22_L1 | 3.5826→3.5810 | −0.00159 | −1.59e-5 | 0.246 | 34 |
| A_D22_L2（现基线） | 3.5757→3.5735 | −0.00219 | −1.10e-5 | 0.362 | 67 |

Phase B（L2，lr=0.30，clip 对比）：

| cell | Δloss | Δloss/秒 | δ_r0 |
|---|---|---|---|
| B_D16_lr0.30_c1 | −0.00299 | −2.04e-5 | 0.957 |
| B_D16_lr0.30_**c2** | **−0.00342** | −2.33e-5 | 0.957 |
| B_D22_lr0.30_c1 | −0.00281 | −1.41e-5 | 0.828 |
| B_D22_lr0.30_**c2** | **−0.00342** | −1.71e-5 | 0.828 |

- **lr 0.15→0.30 ≈ ×1.5**（与 `260929-094100` 旧扫描一致）；**clip 2.0 > 1.0**（同 D 下 −0.00342 vs −0.0028~0.0030）。
- 小 D 的每秒降幅更好（D8/D16 最高），但 1 client×2 样本噪声大、且 L1 与 L2 的 r0 基线不同（3.5826 vs 3.5757，样本组成差异）⇒ **D 的排序只当弱证据**。
- 重尾再次出现（A_D16_L1 r1 原始 δ=239），clip 兜住 ⇒ 保留 clip，暂缓 zo_grad 逐方向截尾（E2/E3@r29 未翻转为弱证据）。

据此启动全协议先导（`temp/run_pilot_full_bg.sh`，11:15，pid 132351）：

| arm | D | L | lr | clip | 单轮估算 | 8 轮估算 |
|---|---|---|---|---|---|---|
| `P_D16_L2_lr0.30_c2`（保守推力） | 16 | 2 | 0.30 | 2.0 | ~30 min | ~4.0 h |
| `P_D11_L2_lr0.30_c2`（吞吐推力） | 11 | 2 | 0.30 | 2.0 | ~23 min | ~3.0 h |

对照基线（成功率配置 D22/lr0.15/c1）：−0.0117 PPL/轮、−0.0179 PPL/h（30 轮 33.654→33.285）。
双指标：full PPL 斜率/h + answer NLL 斜率/h（`paired_eval` r0 vs r7 自动跑）。

## 待办

- [ ] 读全协议 pilot（~7h 后）：两 arm 的 PPL 斜率/h、answer ΔNLL、clipped/δ 重尾，与基线对照。
- [ ] 跳变舍弃（zo_grad 逐方向截尾）暂缓：E2/E3@r29 未见翻转；若 pilot 中 clip 事件仍多再启动。
- [ ] 云端：同步 `--save-every` 用法（长跑可 `--eval-every 5 --save-every 1` 省 ~25% 墙钟）。
