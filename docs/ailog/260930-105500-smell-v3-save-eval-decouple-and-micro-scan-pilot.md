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

## 待办

- [ ] 读微扫描汇总：Δloss（r0→r2）、Δloss/秒、δ、clipped；据此选 8 轮全协议 pilot 配置。
- [ ] 全协议 pilot 双指标：full PPL 斜率/h + answer NLL 斜率/h（caveat：answer 退化仍未闭环，Step1–3 把主因指回「token 均值淹没 answer」）。
- [ ] 跳变舍弃（zo_grad 逐方向截尾）暂缓：E2/E3@r29 未见翻转；若 pilot 中 clip 事件仍多再启动。
