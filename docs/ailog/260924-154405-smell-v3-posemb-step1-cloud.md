# Ailog 260924-154405 — SMELL-v3: 云端 A40 首次跑通 + 位置适配 Step1（pos_only/pos_lora）+ 2k 回归

## 概述

在云端 **A40 服务器（host `amax`）** 首次把 SMELL-v3 跑起来：复用 `smell-v2` conda 环境（导入/前向/反传均通），拉取 `facebook/opt-350m`，在云端重建 Discovery 16k a01 数据，并用 **单卡 GPU2 串行** 完成位置适配两方案（B `pos_only`、C `pos_lora`，各 500 步）与配套 2k 回归。结论：**选 C（pos_lora）**，16k G-PPL 18.14（B 为 31.42），2k 无回归（18.13 vs base 40.01）。500 步权重已稳定归档到 `checkpoints/posemb_step1/`。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 15:00 | 修改 | `src/data/build_discovery_16k.py` | L6-7 | `HF_ENDPOINT`/`HF_HUB_DISABLE_XET` 由硬写改 `setdefault`，云端可覆盖（hf-mirror 不可达 → huggingface.co + XET），修复首次 `ChunkedEncodingError` |
| 15:40 | 修改 | `src/eval/ppl.py` | CLI + `build_model()` | 新增 `--pos-checkpoint`，加载 longctx 训得的 `embed_positions` 权重（复用 run_fed 形状自适应逻辑），支持 2k/16k 适配后评测 |
| 15:42 | 构建 | `dataset_v3/discovery_16k/a01/`（gitignore） | — | 云端重建：30 client×(100+16) + 500 G-PPL + 512 warmup；`check_partition` PASSED（0 errors） |
| 15:10/15:20 | 训练 | `logs/longctx/260924-151049`、`-152022`（gitignore） | — | B/C 两臂各 500 步；产物 `pos_embed.pt`（+pos_lora `adapter/`） |
| 15:42 | 评测 | `logs/longctx/step1_2k/*.json`（gitignore） | — | base/pos_only/pos_lora 的 2k 回归 |
| 15:44 | 归档 | `checkpoints/posemb_step1/`（gitignore） | — | 稳定命名权重目录 + `MANIFEST.json`（含超参/结果/加载命令） |

## 关键数据（global_test=500，full-text token PPL）

| step | pos_only full / tail | pos_lora full / tail |
|---|---|---|
| 0（base, 16k） | 398.77 / 302.64 | 398.77 / 302.64 |
| 100 | 41.42 / 40.24 | 22.43 / 21.47 |
| 500（final 16k） | 31.42 / 31.20 | **18.14 / 17.71** |

2k 回归（seq=2048）：base **40.01**、pos_only **32.49**（-18.8%）、pos_lora **18.13**（-54.7%）。
判据：G-PPL<50 ✅；`ppl_tail` 不劣化 ✅；2k 回归≤10% ✅（两臂反而更好）。

## 云端环境（A40，重要）

- 服务器 `amax`：4× **A40 46GB**（sm_86，driver 570.211.01 / CUDA 12.8），144 CPU / 251GB RAM；本次全程**只用 GPU2 串行**（GPU0/1/3 被他人占用）。
- 环境复用 **`smell-v2`**（`~/applications/anaconda3/envs/smell-v2`）：python 3.10.20 / **torch 2.1.2+cu118** / **flash-attn 2.4.2** / transformers 4.45.2 / tokenizers 0.20.1 / **peft 0.19.1** / datasets 5.0.0 / numpy 1.26.4。**未使用** pinned 的 v3 栈（torch 2.8.0+cu128 / FA 2.8.3 / peft 0.13.2）。
- 数据/权重均云端重建或下载：`opt-350m` 经 ModelScope（~3.9MB/s）；`sileod/discovery` 经 huggingface.co + XET。`third_party/Jenga` 子模块仍空，运行时 jenga 来自 `smell-v2` 的 editable 安装（`JengaForMemoryTest/Jenga/src`），**非本仓子模块**，正式实验前应 `git submodule update --init`。

## 产物定位（500 步权重）

- **首选 C（pos_lora）**：`checkpoints/posemb_step1/a01_pos_lora_500step/{pos_embed.pt, adapter/, config.json, metrics.jsonl}`
- B（pos_only）：`checkpoints/posemb_step1/a01_pos_only_500step/{pos_embed.pt, config.json, metrics.jsonl}`
- 源 run（含逐 step 日志）：`logs/longctx/260924-152022`（C）、`260924-151049`（B）
- 索引：`checkpoints/posemb_step1/MANIFEST.json`（超参、结果、加载命令）

## 设计决策

- **先建稳定归档再提交**：`logs/`、`temp/`、`dataset_v3/` 均被 gitignore，故把交付权重复制到语义化路径 `checkpoints/posemb_step1/<arm>_500step/` 并写 `MANIFEST.json`，确保「500 步权重可被找到」。
- **XET 下载**：`HF_HUB_DISABLE_XET=1` 在云端会走易断的普通 HTTP（首次构建即 `ChunkedEncodingError`）；启用 XET（默认）后下载稳定。
- **复用 smell-v2 而非新建 env**：为快速推进，先用既有环境；版本漂移（peft 0.19 / datasets 5.0 / torch 2.1.2）记为风险，正式消融前建议新建 `smell-v3`。
- **C 优于 B**：C 收敛更快（step50 27.1 vs 54.9）、final PPL 低 42%，且附带 `adapter/` 可作后续 FedAvg 初始化。

## 待办 / 风险

- `third_party/Jenga` 子模块未初始化，jenga 指向他人 checkout；正式实验前修正。
- smell-v2 栈与 v3 pinned 版本不一致（尤其 peft 0.19 对 `PeftModel.from_pretrained(..., is_trainable=True)` 的行为未验证）。
- ZOO 步长仍未重调（历史 lr=1e-3 第 2 轮 NaN），Step3 BP 时一并校准。
- `AGENTS.md` 已更新为云端工作区并把 `check_partition` 的正确参数修为 `--root`。
- 下一步：Step2 用 `checkpoints/posemb_step1/a01_pos_lora_500step/pos_embed.pt` 训 predictor。
