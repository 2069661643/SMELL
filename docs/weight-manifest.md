# SMELL v3 权重 / 数据清单（交接用）

- 生成日期：2026-09-28；仓库根：`/home/yangyongbo/projects/smell/SMELLv3`（下称 `$REPO`）
- Git：`git@github.com:2069661643/SMELL.git`，分支 `exp/rank-rotation-lora`（260930 @ `b904a07`+）
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
| r29 刷新 predictor（实验：answer −0.67%，p=0.014，未闭环；见 ailog `260930-104500`） | `checkpoints/predictor/refresh_r29_a01a03/predictor.pth` | 102M | `3d3dd15dfbdc0deac609faac7df3aaaf270fce7a792c6060412ae998fafdc1f2` |
| 刷新配对 pruned_config（必须与上者成对） | `checkpoints/predictor/refresh_r29_a01a03/pruned_config.pth` | 1.9K | `784e0391e0fb26b86e1b32c22161760537a35a7664f1d127f03475cda393b7f3` |
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

## 6. DistilBERT（BERT 线）产物与运行口径（261014）

> 权重隔离：全部 `distilbert_*` 前缀，与 OPT 产物零混用；数据在 `dataset_v3/arxiv_16k/`（α=0.1/0.3，30 clients × (100+16)，seq 16384）。

### 6.1 底座（USE）

| 用途 | 相对路径 | 大小 | sha256 |
|---|---|---|---|
| Step1 位置表（pos+head 1000 步，16k） | `checkpoints/distilbert_step1/a01_pos_head_1000step/pos_embed.pt` | 48M | `bbf3f805b66532338b480e714c47824e2cb61a9428a28b9ad52a54777de8e58f` |
| Step1 分类头（FL 冻结；D3） | `checkpoints/distilbert_step1/a01_pos_head_1000step/head.pt` | 2.3M | `dbab4ff8bd0b632ede96f533d2591fe389a01415a2f037dd6b027f68f0de0c20` |
| predictor（400 步 warmup，双向池化目标） | `checkpoints/predictor_bert/a01_warmup_400step/predictor.pth` | 45M | `e0925b222396543e7d1e07d48c5316e33cb4cf33c5aec02e6f0db8190c1c7b1c` |
| pruned_config（必须成对） | `checkpoints/predictor_bert/a01_warmup_400step/pruned_config.pth` | 1.5K | `922bbce0463489834b5d95fa21d735442f2062ed96ce43e7c60f66ed54809929` |

### 6.2 r30 四臂消融 —— **含缺陷臂存档**（D1/D4：仅存档，禁用于论文）

`checkpoints/distilbert_ablation_r30_defect/`（含 `README.md` 缺陷说明 + 每臂 `SHA256SUMS`）。缺陷：CATV 投票 padding 占位（掩码冻结 {0..50}/{205..255}）+ 无 block0 强保；修复见 ailog `261014-111800`（P0a valid-mean + D2 block0）。

| 臂 | adapter sha256 | 缺陷臂 own ΔNLL%（30 轮） |
|---|---|---|
| `r30_a01_catv_off` | `b43978eec592de086d32a1c2462fa72f93beb036ce000e75bda47ace45f1b98b` | +8.29% |
| `r30_a01_catv_on` | `d3be4e4e9ac2142e2c76480d4b69530b6add11dd27ab383c35b1fea9ac69a925` | +1.72%（掩码假象注水） |
| `r30_a03_catv_off` | `5160c96d1b9eea22f3a867997e7d5642e7fb9e09f53099e8eddfbd4cf818c6bb` | +11.86% |
| `r30_a03_catv_on` | （链上运行，跑完后补录） | — |

### 6.3 r30 四臂消融 —— **修复版**（USE；CATV 显著有效，见 ailog `261015-213000`）

`checkpoints/distilbert_ablation_r30_fixed/`（每臂 `adapter_round029/` + `metrics.jsonl` + `config.json` + `SHA256SUMS`；`keep_block0=True`）。

| 臂 | adapter sha256 | r0→r29 NLL | ΔNLL%（own） | acc |
|---|---|---|---|---|
| `r30_a01_catv_off` | `6b83f8fa104cdbcbd00c977eb5529a1d53cea0038afcaabb234cfa2a5485f15b` | 1.6910→1.5836 | +6.35% | .424→.468 |
| `r30_a01_catv_on` | `ed72c112464e0dec0b7f611b4749911248dfb1a0c7efc1aea91801e51c058a2a` | 1.7527→1.5563 | +11.20% | .408→.458 |
| `r30_a03_catv_off` | `ab80899cd03f5c5f4e1469070fb764cc3e0b566af99720a09a2d16cdd7436741` | 1.6906→1.5887 | +6.03% | .426→.466 |
| `r30_a03_catv_on` | `8684622738f960cf13a2678141d8b9cb891b60b796acaee8e96fa3446fd52a20` | 1.7511→1.5484 | +11.57% | .410→.454 |

配对（n=500）：on vs off @r29 a01 p=0.030 / a03 p=0.0025（均显著）。

### 6.4 运行口径（复制即用）

```bash
--arch distilbert --attn sdpa_prune --dtype fp32 --sparse 0.4 --doc-order cycle \
--pos-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/pos_embed.pt \
--head-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/head.pt \
--predictor checkpoints/predictor_bert/a01_warmup_400step/predictor.pth \
--pruned-config checkpoints/predictor_bert/a01_warmup_400step/pruned_config.pth \
--trainer zoo --zo-subspace layers --zo-layers 0-5 --zo-directions 8 --zo-eps 1e-3 \
--local-steps 8 --lr 1.4e-3 --delta-clip 1.0 --lora-r 1 --lora-alpha 2 \
--max-clients 30 --rounds 30 --catv {off,on} --eval-every 1 --eval-max-samples 500
```

## 7. OPT-350M 修复版 r30 四臂（cycle 语料轮转；USE，261020）

> 配置：k=24/D=32/L=2/c=30/r=30、LoRA r=1/α=2、lr=0.30、clip=13.34、fp32+sdpa_prune+sparse 0.4、step5 predictor+pos_only、**`--doc-order cycle`**、eval-every 5。
> 结果/分解见 ailog `261020-090000`：full 与旧协议持平（+6.3~6.6% vs fresh 33.6653），**answer 大幅改善（−12~−16%）**；CATV 部署效应 α 依赖（a03 answer −0.093 nats，p=5e-8；a01 n.s.），且由**推理掩码**主导。

`checkpoints/opt_ablation_r30_fixed/`：

| 臂 | adapter sha256 | full r29 | answer r29 |
|---|---|---|---|
| `r30_a01_catv_off` | `792f66131ad46483582b7f13103a77f6396524cf3cf73a51450d86308093d9e5` | 31.533 | 2436.1 |
| `r30_a01_catv_on` | `32e46b70adf1a0eb5c84d160c860fde09a3c030d61bf5b8cc5b059ea7dc7eb07` | 31.635 | 2474.4 |
| `r30_a03_catv_off` | `e1d4acda62440c2cf529273cff418df77b5ca2a6d4e1d28ff94e420a64923b7a` | 31.457 | 2448.8 |
| `r30_a03_catv_on` | `b3cc1cc5f875d93e9487986926c054c1c76cd2a7df7754df947fa8cc966b0c23` | 31.696 | 2291.7 |

评测口径：`temp/logs/opt_r30_fixed_report.txt`（主表/配对）、`temp/logs/opt_r30_evaloff_report.txt`（权重 vs 掩码分解）；逐样本在各臂 `eval_roundNNN_persample.json` / `evaloff_roundNNN_persample.json`。

## 8. BERT VQ per-result r30（a03，CATV off；USE，261020）

> 架构：per-result（每条 ZOO 训练结果 1 channel），`c_dim = rate*d_active`；`rate=0.5`、`C=64`、`c_dim=18,432`、bf16 sketch、rescale on、seed 0。
> 配置：a03、CATV off、L8/D8、lr=1.4e-3、clip=1.0、cycle、r30、eval 500。
> 结果：r0 1.690728 / acc 0.426 → r29 **1.587080 / acc 0.466**（−6.13% NLL）；同轮无 VQ a03 off = 1.588699 / 0.466，配对 ΔNLL=−0.001618（t=−2.416，p=0.0157，n=500）。
> VQ：`vq_cos_mean` 0.9945（聚合口径）；`vq_ratio_elements=0.5`（单条）；`vq_ratio_bytes=0.5`（bf16 vs dense bf16）；`total_ratio=32`（相对一条 dense delta）。
> 通信：per-result dense `141,557,760 B/round` vs VQ `70,778,880 B/round`（0.5）；相对“一条 delta”`2,211,840 B` 是 32×。

`checkpoints/distilbert_vq050_bf16_perresult_a03_r30/`（adapter_round029 + metrics.jsonl + config.json + vq_seeds.json + SHA256SUMS）：

| 文件 | sha256 |
|---|---|
| `adapter_round029/adapter_model.safetensors` | `71ddc9c889cd11c03f233c5a6b4a158850ef133463231ed9b20385099e581291` |
| `adapter_round029/adapter_config.json` | `f348df57fb6bf45c23e3be21f97d7747d30c25654f5aa930fc267ec7e71aa5a2` |

- 旧 total 口径的 0.25 fp32 partial：`checkpoints/distilbert_vq025_a03_r20_partial/`（仅历史，勿与新架构混比）。
- 旧 total 口径的 0.5 bf16 partial：`checkpoints/distilbert_vq050_bf16_a03_r0_partial/`（仅历史）。

## 9. BERT VAR+CATV+VQ 组合系统评测 r30（a03；261022）

> 配置：VAR Method A（raw var、min D=4、no patience、T0=0.5872309208、r10/r20 探针）、CATV on（`--catv-normalize client`，每 client 每层 L1）、VQ rate 0.5 bf16（per-result，`C_i=sum(D)`）、L=8、r30、eval 500。
> 结果：r29 **NLL 1.559838 / acc 0.450**；vs CATV on 历史基线 1.548441/0.454 → **ΔNLL +0.011398**（t=4.44，p=9.1e-6）；vs CATV off 1.588699/0.466 → −0.028861（p=0.039，acc −1.6pt）。
> VAR：mean D 9.003、samples 72.0；hit_low 2896/7200、hit_high 2128/7200；T10=0.611885、T20=2.210161。
> VQ：mean C 72.02、cos 0.99456、ratio_bytes 0.5；r29 75.4MB vs per-result dense 150.8MB。

`checkpoints/distilbert_var_catv_vq_a03_r30_rerun/`（adapter_round029 + metrics + config + vq_seeds + var_thresholds + var_probe_round010/020 + SHA256SUMS）：

| 文件 | sha256 |
|---|---|
| `adapter_round029/adapter_model.safetensors` | `d537b37eef4f1ab632fe962d2a1e1e5b2adf25cd5c7a80550308498a721a9c3a` |
| `adapter_round029/adapter_config.json` | `ce404fc8fda832620b6e0286fbc9b33ee723e556210a278b9f0ec18f1aa603fc` |

- 口径 caveat：历史 CATV on 基线是 `--catv-normalize off`；本组合是 `client` + VAR + VQ，+0.0114 NLL 不能单独归因。
- 旧异常终止 partial（r0–r5）：`checkpoints/distilbert_var_catv_vq_a03_r6_partial/`，仅历史。

### 9.1 eval-off 复核（261022 13:36）

- r29 on-mask 1.559838/0.450 vs off-mask **1.577730/0.464**；mask ΔNLL −0.017892（p=0.28 n.s.）、acc −1.4pt。
- 30 轮均值：on 1.656088/0.4336 vs off 1.633698/0.4497；mask 平均 ΔNLL +0.02239、Δacc −0.0161（mask 的 NLL 收益只在 r25 后出现）。
- 组合 off vs CATV-on 基线 off：30 轮 mean ΔNLL +0.003181、Δacc +0.0006；r29 +0.000506（p=0.735）⇒ VAR+VQ 权重与基线等价。
- +0.0114 on-mask NLL 差来自 `client` 等权 mask：基线 `off` 归一化 r29 mask 收益 −0.0288，组合 `client` −0.0179，acc 多掉 0.6pt。
- eval-off 产物：`checkpoints/distilbert_var_catv_vq_a03_r30_rerun/evaloff/`（30 × JSON + persample + SHA256SUMS）。

## 10. BERT VQ **per-client** r30（a03，CATV off、VAR off；USE，261023）

> 架构：`--vq-channels-mode per_client`（261023 起默认）——每 client 只对**最终 delta** 压 **1 条** sketch；`c_dim = rate*d_active` **不变**、`rate=0.5`、bf16、rescale on、seed 0；30 clients → **30 channel/轮**。
> 配置：a03、CATV off、VAR off、L8/D8、lr=1.4e-3、clip=1.0、cycle、r30、eval 500。
> 结果：r0 1.690814 / acc 0.424 → r29 **1.627393 / acc 0.448**（−3.75%）；同轮无 VQ a03 off = 1.588699 / 0.466 → 配对 ΔNLL **+0.038694**（t=7.745，p=9.55e-15，n=500）、acc −1.8pt。
> VQ：`vq_channels=1`、`vq_channels_total=30`、`vq_cos_mean=0.7060`、`vq_ratio_bytes=0.5`、`vq_communication_bytes=1,105,920 B/round` vs `dense_all_results=2,211,840 B/round`（0.5× 最终 delta；比 per-result 少 64×）。
> 训练 6.45h（773s/轮），与 VQ-off/per-result 相同。

`checkpoints/distilbert_vq_perclient_a03_r30/`（adapter_round029 + metrics + config + vq_seeds + SHA256SUMS）：

| 文件 | sha256 |
|---|---|
| `adapter_round029/adapter_model.safetensors` | `804606a66ad940157f27c7c6907071a6546a2753e28b89da6c2ae6b29967ef6d` |
| `adapter_round029/adapter_config.json` | `ef8bd42fbaa09ee0eaa0688efa94cc194f9986436feb0f2d14d007d736fea1ab` |

- 对照：per-result（1920 channel/轮、cos 0.9945）r29 1.587080/0.466、ΔNLL −0.0016；per-client 把通信降 64×，代价 **+0.0387 NLL / −1.8pt acc**。
- 3-on（VAR+CATV(client)+VQ per_client）r30 重测：`logs/fed/bert_var_catv_vq_perclient_a03_r30/`（进行中，完成后补 §10.1）。

### 10.1 BERT VAR+CATV+VQ **per-client** r30（a03；261024）

> 配置：VAR Method A（T0=0.5872309208、r10/r20 探针）、CATV on（`client`）、VQ rate 0.5 bf16 **per-client**（1 channel/client，30 channel/轮，1.11 MB/轮）；L8、r30、eval 500。
> 结果：r0 1.743194 / acc 0.410 → r29 **1.620426 / acc 0.450**（own −7.04%）。
> 对照：3-on per-result 1.559838/0.450 → **ΔNLL +0.060588**（t=8.47，p<1e-16，n=500）、acc 0.0pt；CATV-on 基线 1.548441/0.454 → +0.071985（+4.65%）；CATV-off 基线 1.588699/0.466 → +0.031727（+2.00%，p=0.039）、acc −1.6pt。
> VAR：mean D **9.858**、samples **78.87**（per-result 组合为 9.003 / 72.0）；T10=0.440958、T20=1.048238；hit_low 33/64、hit_high 14/64（r10 探针）。
> VQ：channels=1、total=30、cos=0.7076、ratio_bytes=0.5、1,105,920 B/round（vs per-result 3-on 70.8MB）。
> 训练 882s/轮（7.35h）。

`checkpoints/distilbert_var_catv_vq_perclient_a03_r30/`（adapter_round029 + metrics + config + vq_seeds + var_thresholds + var_probe_round010/020 + SHA256SUMS）：

| 文件 | sha256 |
|---|---|
| `adapter_round029/adapter_model.safetensors` | `af0776484af65e96cd601c4618c88e24eb9960dd40b8bcc15f3eb3eeb979efcc` |
| `adapter_round029/adapter_config.json` | `054232a31ab36aee6003843b6701fef1cc0d75c33614f4675598d7ab17c5ef48` |

- 口径 caveat：3-on per-client 的 VAR 轨迹被 VQ 噪声改变（D 9.86 vs 9.00、样本 78.9 vs 72.0），与 per-result 3-on 不是严格单变量对照；CATV 归一化同为 `client`。
> **eval-off（10/24 12:25，30 个 adapter）**：r29 on 1.620426/.450 vs off **1.603422/.462** → mask ΔNLL **+0.017004**（t=1.10，p=0.27 n.s.）、acc −1.2pt；30 轮均值 mask **+0.037188**、Δacc −0.0173。
> 权重效应：vs 3-on per-result off 1.577730/.464 → **+0.025692**（t=3.18，**p=0.0015**）；vs CATV-on 基线 off 1.577225/.462 → +0.026198（p=0.0027）。
> 归因（on-mask 总差 +0.060588）：权重 **+0.025692（42.4%）** + mask 效应差 **+0.034896（57.6%）**；单轮 mask 项 n.s.，以 30 轮均值（本 run +0.0372 vs per-result 组合 +0.0224）为准。
> evaloff 归档：`checkpoints/distilbert_var_catv_vq_perclient_a03_r30/evaloff/`（30 JSON + 30 persample + SHA256SUMS）。

## 11. BERT FwdLLM 稠密（sparse=1.0）VAR r30（a03；USE，261023）

> 配置：`--sparse 1.0`（sdpa_prune 走 dense fallback，全部块保留）、VAR Method A（T0=0.5872309208；T10=1.477083、T20=3.225293）、CATV off、VQ off、L8/D8、r30、eval 500、cycle、lr 1.4e-3、clip 1.0。
> 结果：r0 **1.592325 / acc 0.454** → r29 **1.493650 / acc 0.472**（−6.20%）；**1346s/轮、11.39h**（sparse 0.4 的 1.59×）。
> 对照：sparse 0.4 + VAR 1.584451/0.464 → dense **−0.090801 nats（−5.73%）、acc +0.8pt**；VQ-off sparse 基线 1.588699/0.466 → **−0.095048**。
> VAR：mean D **9.108**、samples **72.87**；hit_low 32/64、hit_high 16/64（r10 探针）。
> caveat：dense 的 r0 起点（1.5923）显著低于 sparse 0.4（1.6906），**跨稀疏度不可直接比终点绝对值**；应看 own% 或同协议基线。
> 未测：dense 续训 r30–r59（原 B 计划，未启动）、dense+VQ、dense+CATV；eval-off 不适用（CATV off，评测本身无掩码）。

`checkpoints/distilbert_fwdllm_a03_var_r30/`（adapter_round029 + metrics + config + var_thresholds + var_probe_round010/020 + SHA256SUMS）：

| 文件 | sha256 |
|---|---|
| `adapter_round029/adapter_model.safetensors` | `8f9435fadec173ac3ef43c09a18c7a38885a572d3b7d424e90ec678540e8cd91` |
| `adapter_round029/adapter_config.json` | `1ffbaab68918ce3032c514280520a4b55c864fb1fba0fdd1886b4a9febbc7bd0` |

### 10.2 3-on per-client 的 eval-off（261024 12:25，30 个 adapter）

> 协议同历史 eval-off：sdpa_prune / fp32 / sparse 0.4 / predictor / a03 global / 500 样本 / bs8，无 CATV mask。

| 指标 | on-mask | off-mask | mask 效应 |
|---|---|---|---|
| r29 NLL / acc | 1.620426 / .450 | **1.603422 / .462** | **+0.017004**（mask 更差）、acc −1.2pt |
| 30 轮均值 ΔNLL | — | — | **+0.037188**（mask 全程伤 NLL） |
| 30 轮均值 Δacc | — | — | **−0.0173** |

- 与 3-on per-result 的分解（on-mask 总差 +0.060588）：
  - **权重差**（off vs off）：1.603422 vs 1.577730 = **+0.025692（+1.63%）** → 占 42.4%；
  - **mask 效应差**：per-client +0.017004 vs per-result −0.017892 = **+0.034896** → 占 57.6%。
- 即 per-client VQ 在组合系统里有双重代价：①权重更差；②**CATV mask 从“有效”变成“有害”**（per-result 的 mask 在 r25 后转正，per-client 全程为正）。
- 其他对照（off-mask）：vs CATV-on 基线 off 1.577225 → +0.026197（+1.66%）；vs CATV-off 基线 1.588699/0.466 → +0.014723（+0.93%，acc −0.4pt）。
- 产物：`checkpoints/distilbert_var_catv_vq_perclient_a03_r30/evaloff/`（30 JSON + 30 persample + SHA256SUMS，61 文件）。

## 12. BERT r30 固定四臂**续训 r30→r59**（cycle；USE，261025）

> 协议与首段完全一致：L8/D8、lr 1.4e-3、clip 1.0、cycle、eval 500、`--catv-normalize off`、`keep_block0=True`、LoRA r1/α2、fp32+sdpa_prune、sparse 0.4。
> `--round-offset 30`（a01_off 接前次 partial：r52 → offset 53、7 轮）；CATV on 臂首轮掩码按设计重建（a01_on 在 r33 恢复、a03_on 在 r30 起）。
> 每臂 ~6.45h；r30–r59 逐轮曲线见 `metrics.jsonl` 与 `temp/logs/bert_cont30_report.txt`。

| 臂 | r0（首段） | r29（首段） | **r59** | r29→r59 | r0→r59 | acc r59 |
|---|---|---|---|---|---|---|
| r30_a01_catv_off | 1.6910/.424 | 1.5836/.468 | **1.5068/.484** | −4.85% | −10.89% | .484 |
| r30_a01_catv_on | 1.7527/.408 | 1.5563/.458 | **1.4925/.482** | −4.10% | −14.85% | .482 |
| r30_a03_catv_off | 1.6906/.426 | 1.5887/.466 | **1.5110/.474** | −4.89% | −10.62% | .474 |
| r30_a03_catv_on | 1.7511/.410 | 1.5484/.454 | **1.4852/.472** | −4.08% | −15.18% | .472 |

- 配对（n=500）：各臂 r29→r59 ΔNLL −0.063~−0.078，t=−6.8~−8.7，**p<1e-9**。
- **eval-off（10/25 13:42，on 臂 r30–r59 共 60 次）**：掩码效应随训练**变大**，同时 on 臂权重变差——
  | α | r29 掩码效应 | r59 掩码效应 | seg2 均值 | r59 权重差（on-offmask − off 臂） | r59 净差 |
  |---|---|---|---|---|---|
  | a01 | −0.020501（p=0.184） | **−0.038582（p=0.0137）** | −0.032941 | **+0.024271** | −0.01431 |
  | a03 | −0.028784（p=0.0805） | **−0.047220（p=0.00215）** | −0.042616 | **+0.021419** | −0.02580 |
  ⇒ CATV 的净优势缩小 = **掩码收益变大**（−0.02→−0.04/−0.047）+ **权重代价变大**（r29 约 −0.007 → r59 +0.021~+0.024）两者相抵。
- CATV on vs off @r59：a01 **−0.0143**（t=−1.22，p=0.222，n.s.）；a03 **−0.0258**（t=−2.25，p=0.0245）。
  （@r29 为 a01 −0.0272/p=0.030、a03 −0.0403/p=0.0025 ⇒ CATV 优势随续训缩小，a03 仍显著。）

`checkpoints/distilbert_ablation_r30_fixed_cont30/<arm>/`（adapter_round059 + metrics.jsonl + config.json + SHA256SUMS）：

| 臂 | adapter_model sha256 |
|---|---|
| r30_a01_catv_off | `c93bbd4f330498a6719d4b8a280a7da91b6d696066b4829b2039bb15b13507a4` |
| r30_a01_catv_on | `f816c24c64875866a99b94190f3f5c3b28181a006bbad6f6c5841cac3626dedf` |
| r30_a03_catv_off | `b770dcfb65e2ae08d64769f161001faa14ad1765afcfded4fa958b71f3e7aedc` |
| r30_a03_catv_on | `7ea56840ebcf0e5ef3dfa7da90f8e2ebb8395f6b4cb97ef85a1511aafa15885c` |
