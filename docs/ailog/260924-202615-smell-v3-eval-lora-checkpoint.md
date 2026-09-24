# Ailog 260924-202615 — SMELL-v3: 每次 eval 记录 LoRA checkpoint + ZOO r=1 pre-flight 通过

## 概述

在 `exp/rank-rotation-lora` 分支上，为 `run_fed` 的评测路径增加「每次 eval 的 LoRA checkpoint 可观测性」（metrics 记录 + print），并用 **ZOO r=1 + pos_only** 的 pre-flight 完整验证：δ 标定正确、无 NaN、每次 eval 确实落盘 LoRA checkpoint。pre-flight 通过后已启动正式长跑（rounds=30、eval-every=2，后台运行中）。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 20:15 | 修改 | `src/fed/run_fed.py` | `run_global_eval` | ok/failed 两分支 metrics 增 `lora_checkpoint` 路径字段 + 各自 print；保留 `adapter_round{NNN}` 的 `model.save_pretrained`；加 `# SMELL 3 run_fed eval_lora_checkpoint ADD` 标注 |
| 20:05-20:19 | 运行 | `logs/fed/zoo_r1_preflight`（gitignore） | — | pre-flight：ZOO r=1 pos_only，rounds=2 c=30 local=2 dir=8 lr=1e-7 eval-every=2 |
| 20:21 | 运行 | `logs/fed/zoo_r1_formal`（gitignore） | — | 正式长跑：rounds=30、eval-every=2，后台运行中 |

## Pre-flight 结果（`ALL DONE` 20:19:04）

- **δ**：round0=0.138、round1=0.131（目标 0.1–0.15 ✅）→ `lr=1e-7` 对 r=1（dim 196,608）标定正确。
- **稳定性**：无 NaN/OOM；峰值显存 2.19GiB。
- **eval**：round1 `full_ppl_token=32.28`（基线 ~31.4，2 轮内持平）。
- **LoRA checkpoint 验证 ✅**：`logs/fed/zoo_r1_preflight/a01/adapter_round001/` 含 `adapter_config.json`(r=1, alpha=2, LORA) + `adapter_model.safetensors`(812KB) + `README.md`；metrics.jsonl 已记录 `lora_checkpoint`。

## 正式长跑（进行中）

- 20:21:52 启动，`logs/fed/zoo_r1_formal/a01/`；配置 ZOO r=1 alpha=2 pos_only、16k、c=30、local=2、dir=8、lr=1e-7、eps=1e-3、**rounds=30、eval-every=2**；GPU2 单卡。
- 每 2 轮一次 eval → 落一个 `adapter_round{NNN}/` LoRA checkpoint + `eval_round{NNN}.json`；预计 ~3.3h。

## 待办 / 风险

- 正式长跑完成后需汇总：δ 稳定性、`full_ppl_token` 趋势、各轮 LoRA checkpoint 完整性。
- 若 G-PPL 长期不动，问题在 ZOO 方向质量（历史 `cos≈0`），杠杆为 directions/eps/估计器而非 rank。
- 分支 `exp/rank-rotation-lora`，便于回滚 main。
