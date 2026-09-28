# SMELL v3 权重 / 数据清单（交接用）

- 生成日期：2026-09-28；仓库根：`/home/yangyongbo/projects/smell/SMELLv3`（下称 `$REPO`）
- Git：`git@github.com:2069661643/SMELL.git`，分支 `exp/rank-rotation-lora` @ `dfea2bc`
- **重要**：`checkpoints/`、`dataset_v3/`、`third_party/Jenga/checkpoints/` 均在 `.gitignore`，**不在 GitHub**，需用 rsync/scp 单独传输。本文件（`docs/weight-manifest.md`）随 git 一起走。

## 1. 必需产物（USE）

| 用途 | 相对路径 | 大小 | sha256 |
|---|---|---|---|
| **pos_only 位置表**（当前主用底座） | `checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt` | 33M | `414394f8b71971ec58e1886e47587f5ddd4ff79d40df984c0eba2319f004ea77` |
| **predictor（step5，client 分布重训，主用 260929）** | `checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth` | 102M | `1fabeab15e81b3786ca0f88d891295d10868f21b9bee567fd67b092eaf646478` |
| **pruned_config（必须与 step5 成对）** | `checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth` | 1.9K | `38e94fba64899d18381be76a99e82d39e6612e72fad2e7cb4fa1b50ef58b3d2a` |
| predictor（step4，历史；warmup-only 且推理侧 bias 曾随机） | `checkpoints/predictor/step4_a01_pos_only_causal/predictor.pth` | 98M | `819246e52e37ffe09db438293aa70d78b5957fc0d9a3345eaf18fe84a3662e72` |
| pruned_config（step4 配对，历史） | `checkpoints/predictor/step4_a01_pos_only_causal/pruned_config.pth` | 1.5K | `b194158bb53c12112b85b7497431c7a1cc0b7107358cf7fcc859a8cea6f26a7e` |
| base 模型（OPT-350M，含 config/tokenizer） | `third_party/Jenga/checkpoints/opt-350m/`（权重 `pytorch_model.bin`） | 632M | `a5223ae6f3c26c6d90003f96a6bcd9a4aaaef0d36fca6469112efeeb985f2842` |
| discovery 16k 数据（a01，warmup 池） | `dataset_v3/discovery_16k/a01/warmup_input_ids.npy` | 17M | `c11926723f10a8386b25d45611935a1610046e06d1787c577503065e948f9a85` |

`step4_a01_pos_only_causal/` 同目录另含 `config.json`（记录 `pos_checkpoint=a01_pos_only_500step`、`adapter=None`、`sparse=0.4`、144 张量）。

## 2. 参考 / 历史产物

| 用途 | 相对路径 | 大小 | sha256 |
|---|---|---|---|
| pos_lora 位置表（备用变体） | `checkpoints/posemb_step1/a01_pos_lora_500step/pos_embed.pt` | 33M | `ac6a30a30d3965aaaac9f145258683921b3733ad64335e1e73bd6138a9e665e6` |
| pos_lora adapter（备用） | `checkpoints/posemb_step1/a01_pos_lora_500step/adapter/adapter_model.safetensors` | 6.3M | `aee8c195fafa77fc28fcf22c8fb4fe964a7b2c7d28bd3127d02a3c2debb7b24d` |
| predictor step2（**勿用**：非 causal，pos_lora） | `checkpoints/predictor/step2_a01_pos_lora/predictor.pth` | 99M | `02c84eab2aa491062aecc255fd34c0965dc2cc8b8398c79d4191579db60745d3` |
| predictor step3（**勿用**：非 causal，pos_only） | `checkpoints/predictor/step3_a01_pos_only/predictor.pth` | 99M | `ec9ec761bf0bfdacfc580a2e7d56263382600ed17b1b9de0056c5831caf58237` |

> ⚠️ **只用 step2/3 以外的可用版**：step2/step3 用 Jenga 原非 causal 目标训练，对 OPT 会退化为「只选尾部块」，16k LM loss 5.08（step4 为 3.44），详见 `docs/ailog/260928-110233-...md`。**step5** 在 step4 基础上改为 **client 训练分片**（含 query+label）训练，修正 warmup-only 分布漂移；加载器修复后缺失的推理侧 bias 会零填充（训练侧 `bias=False` 的等价语义）。

## 3. 运行口径（复现 TD-2' / TD-3）

必需成对传入 `--predictor` + `--pruned-config`（`run_fed.py` 强制二者同给）：

```bash
--dtype fp32 --attn sdpa_prune --sparse 0.4 \
--pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
--predictor      checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth \
--pruned-config  checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth
```

- TD-3（k=4 sparsity ZOO）：`--trainer zoo --zo-subspace layers --zo-layer-rotate --zo-layer-group 4 --max-clients 30 --local-steps 2 --zo-directions 22 --rounds 30 --eval-every 1`。
- 诊断（TD-2' k4 稀疏 cos）：`python temp/diag_cos_grid.py --attn sdpa_prune --dtype fp32 --seq 16384 --block-layers 20-23 --blocks k4 --ls 2 --ds 22 --cs 1,5,10,20,30 --n-clients 30 ...`（见 `temp/run_k4_td2_td3_bg.sh`）。

## 4. 传输与校验

```bash
# 代码 + 文档（含本清单与 AGENTS.md）
git clone git@github.com:2069661643/SMELL.git && cd SMELL && git checkout exp/rank-rotation-lora

# 权重 + 数据（git 外）
rsync -av --relative \
  ./checkpoints/posemb_step1/a01_pos_only_500step/ \
  ./checkpoints/predictor/step4_a01_pos_only_causal/ \
  ./dataset_v3/discovery_16k/a01/ \
  ./third_party/Jenga/checkpoints/opt-350m/ \
  <user>@<other-host>:<dest>/SMELLv3/

# 校验（与上表逐一比对）
cd <dest>/SMELLv3 && sha256sum \
  checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
  checkpoints/predictor/step4_a01_pos_only_causal/predictor.pth \
  checkpoints/predictor/step4_a01_pos_only_causal/pruned_config.pth \
  third_party/Jenga/checkpoints/opt-350m/pytorch_model.bin \
  dataset_v3/discovery_16k/a01/warmup_input_ids.npy
```

## 5. 环境

`smell-v2`（torch 2.1.2+cu118 / flash-attn 2.4.2 / transformers 4.45.2 / peft 0.19.1）；jenga 为 editable 安装（源 `JengaForMemoryTest/Jenga/src`），需保证 `import jenga` 可用。规范见 `AGENTS.md` 与 `docs/standard.md`。
