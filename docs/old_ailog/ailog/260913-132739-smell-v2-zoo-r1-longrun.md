# SMELL-v2 — ZOO 长跑启动：lm_head r=1, epochs=1024

## 概述

在 C5 确认「head-only BP 精确可学、但 ZOO 方向质量 = √(M/d)」后，按用户要求跑一次**真实 FwdLLM ZOO 长跑**（rank=1），取跨轮 PPL / 更新方向作为对照。epochs 在推荐值 256 基础上 ×4 → **1024**。

## 配置

| 项 | 值 |
|---|---|
| 模型 | `llama_sparse` + peft LoRA，**lm_head-only r=1**（d=36,096） |
| ZOO | `--forward_mode`，epochs=1024（M=3×1024=3072，理论 cos≈0.29），samples_per_round=1 |
| 聚合 | `--lr 1e-4 --server_lr 1e-3 --lr_decay`（线性 0.01→1e-5） |
| 联邦 | 3 clients，comm_round=21（20 实际轮），worker_num=1，`-np 2` |
| 稀疏/共识 | `--sparse 0.4 --enable_consensus --vote_threshold 0.3` |
| 降噪 | `--var_control --perturbation_sampling --max_resample 50` |
| 评估 | `--eval_ppl 50 --eval_seed 42`，`--save_per_round 1` |
| 数据 | `goemotions_4k_{data,partition}.h5`，`niid_dirichlet_clients=3_alpha=0.1` |

## 新增脚本

`script/RUNME-v2-len4k-d3-zoo-r1.sh`（照 `RUNME-v2-len4k-d3-lmhead.sh` 改：r=1、epochs=1024、comm_round=21）。

## 运行

- 启动 2026-09-13 13:26:51，日志 `log/smell-v2-zoo-r1_e1024_20260913_132651.log`
- 启动校验：3 clients / 43410 train 样本 / trainable=2（lm_head lora A/B）
- **ETA ≈ 77min/轮 × 20 ≈ 26h**（可随时停止，逐轮 checkpoint 便于中途分析）

## 判据 / 预期

- 按 C5/A2 结论，cos≈0.29 仍以噪声为主，**PPL 大概率仍不降**；价值在于拿到 M=3072 下 ZOO 的真实跨轮证据。
- 看 `[EVAL-PPL]` 斜率 + `analyze_update_direction.py` 的 `cos(Δθ_r, Δθ_{r-1})`；若仍随机游走 → 坐实需改扰动结构/方法，而非调 lr。
