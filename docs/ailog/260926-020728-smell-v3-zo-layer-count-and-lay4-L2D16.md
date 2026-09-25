# Ailog 260926-020728 — SMELL-v3: --zo-layer-count + 前4层 L2D16 长程启动（GPU2）

## 概述

为「仅前 4 层、每轮轮转 1 层」的新方案新增 `--zo-layer-count`（把层轮转的取模范围限制在前 N 层），并在 **GPU2** 启动 `L=2, D=16, lr=4e-6, c=30, 每轮 eval` 的长程（可切片停机）。

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 02:05 | 修改 | `src/fed/run_fed.py` | 新增 `--zo-layer-count N`（0=全部层）：`rotated_layers_for` 取模基准改为 `N`；超过 num_layers 报错；config 记录 |

## 实验配置（已启动，GPU2）

| 项 | 值 |
|---|---|
| 方案 | ZOO，仅 **前 4 层**，每轮轮转 1 层（`layer=round%4`） |
| lora | r=1, alpha=2 |
| pos / adapter | pos_only 的 `pos_embed.pt` / 无 adapter-init |
| zo | `--zo-subspace layers --zo-layer-rotate --zo-layer-count 4`，`D=16, eps=1e-3` |
| 本地/联邦 | `L=2`, `c=30`, `lr=4e-6` |
| eval | **每轮**（`--eval-every 1`，每轮存 LoRA ckpt + eval JSON） |
| rounds | 480（可中途停机） |
| GPU | **2** |

**cos 预估**：块=单层 `d_eff=8192`，`M=c·L·D=960` ⇒ `cos≈0.342`。
**ETA**：1980 前向/轮 ⇒ ~11min/轮（GPU2）+ 每轮 eval ~2–3min ⇒ **~13–14min/轮**；480 轮上限约 100h，将切片停机。

## 注意

- 相对上次（`L=1,D=8,lr=4e-6`，cos 0.171，149 轮平在 ~32）：本次 **cos 翻倍（0.342）+ 覆盖快 6×（前4层轮转）**，lr 更激进（δ 预计 >0.15）。为判断 ZOO 能否在「更优方向 + 更密覆盖」下动 G-PPL。
- `--zo-layer-count` 代码已入此提交；运行进程已加载其代码。
