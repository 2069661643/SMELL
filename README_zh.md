# SMELL-v3

**基于 token 级稀疏 × 前向梯度（ZOO）的联邦长上下文 LLM 微调。**
前向做 Jenga 风格的上下文 token 稀疏，用 FwdLLM 风格零阶优化实现无反向传播的联邦更新，
并辅以上行压缩（Count Sketch，CPUR）与方差自适应方向数（VAR）。

[English](README.md) | **中文**

> 研究工件（目标 MobiCom 2027 / ICDE 2027）。以下数字全部来自归档 run
> （16K 上下文、30 设备、Dirichlet α=0.1/0.3、seed 42）。

## 结果速览

| 线 | 设置 | 端点 |
|---|---|---|
| **BERT 四臂, r89** | a01 off / a01 on / a03 off / a03 on | NLL/acc **1.4703/.502**、**1.4659/.490**、**1.4724/.490**、**1.4610/.488** |
| **BERT dense VAR, r89** | `--sparse 1.0` + VAR Method A | **1.3096/.520**（最强 BERT 线；非 Jenga） |
| **BERT 3-on, r89** | VAR + CASC(client) + CPUR 0.5 per-result | **1.4598/.480** |
| **BERT CPUR 0.5, r89** | per-result（64 channel/client） | **1.4780/.494**（vs 无压缩 1.4724/.490 → +0.38% NLL） |
| **CPUR per-client, r29** | 1 channel/client，1.11 MB/轮（0.5× dense，比 per-result 少 64×） | 1.6274/.448（vs 关闭 1.5887/.466，**ΔNLL +0.0387**，p=9.6e-15） |
| **VAR Method A, r29** | per-sample raw var、D=4..16、T0/T10/T20 | 1.5845/.464（vs 关闭 1.5887/.466，ΔNLL −0.0042） |
| **OPT 四臂, r59** | a01 off / a01 on / a03 off / a03 on（full PPL / answer PPL） | **30.953/2174.8**、**31.126/2277.1**、**30.864/2362.6**、**30.959/1504.2** |
| **显存（16K）** | ZOO（真实 `zo_grad`）vs BP | **2.66 / 2.21 GiB vs 14.51 / 12.69 GiB**（dense / sparse 0.4；ZOO 省 **5.5×**） |

关键结论：CASC 的收益在 **OPT a03 上主要来自权重（89%）**，在 **BERT 上主要来自推理期掩码**；
长训后 CASC 净优势衰减（BERT r89 / OPT r59 的 a01 均不显著）；CPUR per-client 用 1/64 的上行换 ~0.04 nats 累积误差。

## 仓库结构

```
src/
  models/    modeling_opt_smell.py、modeling_distilbert_smell.py、position_embed*.py、token_selector.py (CASC)
  data/      build_discovery_16k.py、build_warmup_16k.py、build_arxiv_16k.py、build_warmup_arxiv_16k.py、check_partition.py
  train/     zoo.py、lora.py、longctx_adapt.py、train_predictor*.py、train_pos_head_bert.py、lowrank_reparam.py
  fed/       run_fed.py（入口）、serial_fedavg.py（client runner + FedAvg）
  compress/  count_sketch.py（CPUR）
  eval/      ppl.py（OPT）、classify.py（DistilBERT）
  bench/     memory.py（ZOO/BP 显存矩阵）
  hello_world.py            环境自检（Jenga hello_world 风格）
scripts/
  check-env.sh              → src/hello_world.py（依赖/GPU/底座/数据/权重/前向 smoke）
  setup_server.sh           云端 A40 bootstrap（幂等；--dry-run/--check）
  run_ablation_cloud.sh     4 卡并行跑 OPT/BERT 四臂
  repro/                    6 个论文复现脚本（见 scripts/repro/README.md）
  export_endpoints.py       汇总关键 endpoint + 前置权重（checkpoints/out）
reports/
  bert_and_other/           BERT + OPT fixed 的 tidy CSV/README（脚本生成，含对账闸门）
  opt/                      OPT 图/表（论文用）
  r60/                      BERT r60–r89 续训包
  mem/                      显存交付包（ZOO vs BP，1K–16K）
docs/
  standard.md               代码/标注规范
  weight-manifest.md        权重/数据路径 + sha256 + 运行 flags（换机交接）
  ailog/                    按时间戳的实验日志（每个改动/结论的出处）
third_party/                Jenga / FwdLLM / VQ（只读 submodule）
dataset_v3/  checkpoints/  logs/  temp/    运行时目录（gitignore）
```

## 环境

```bash
# 本机（AutoDL RTX 4090）：conda env SMELL（Python 3.10）
PY=~/miniconda3/envs/SMELL/bin/python
$PY -V && $PY -c "import torch; print(torch.__version__, torch.version.cuda)"
# torch 2.8.0+cu128、flash-attn 2.8.3、transformers 4.45.2、peft 0.13.2
bash scripts/check-env.sh            # 34 项自检：依赖 / GPU / 底座 / 数据 / 权重 / 前向 smoke
bash scripts/check-env.sh --no-gpu   # 纯导入（无 GPU 的机器）
```

- 依赖清单：`requirements-wsl-cu128.txt`；云端从零搭建：`bash scripts/setup_server.sh`（先 `--dry-run`）。
- `third_party/Jenga` 以源码方式加入 `sys.path`（`jenga_src.pth`），不修改 submodule。
- 数据/权重不走 git：见 `docs/weight-manifest.md`（`dataset_v3/`、BERT step1/predictor_bert、
  OPT posemb/predictor step5、`opt-350m`）。

## 快速开始

```bash
# DistilBERT smoke：2 clients × 1 轮 × 2 样本 × 2K
PY=~/miniconda3/envs/SMELL/bin/python
S1=checkpoints/distilbert_step1/a01_pos_head_1000step
P=checkpoints/predictor_bert/a01_warmup_400step
$PY src/fed/run_fed.py --arch distilbert --tag a01 --gpu 0 \
  --trainer zoo --attn sdpa_prune --dtype fp32 --sparse 0.4 --doc-order cycle \
  --pos-checkpoint $S1/pos_embed.pt --head-checkpoint $S1/head.pt \
  --predictor $P/predictor.pth --pruned-config $P/pruned_config.pth \
  --zo-subspace layers --zo-layers 0-5 --zo-directions 2 --local-steps 2 \
  --rounds 1 --lr 1.4e-3 --delta-clip 1.0 --max-clients 2 --max-train-samples 2 \
  --truncate 2048 --eval-every 0 --save-every 0 --out-root temp/smoke_bert
```

## 复现论文实验

环境变量、产物与对照归档见 [`scripts/repro/README.md`](scripts/repro/README.md)：

| # | 实验 | 脚本 | 参考时长（单卡 4090） |
|---|---|---|---|
| 1 | BERT r30 四臂 | `scripts/repro/01_bert_r30_four_arm.sh` | ~26h |
| 2 | BERT 续训（r30→r59 / r60→r89） | `scripts/repro/02_bert_cont.sh` | ~26h/段 |
| 3 | BERT 技术线（CPUR / VAR / 3-on） | `scripts/repro/03_bert_techniques.sh` | ~6.5h/线 |
| 4 | OPT r30 四臂 | `scripts/repro/04_opt_r30_four_arm.sh` | ~4 天（串行） |
| 5 | OPT 续训 + eval-off | `scripts/repro/05_opt_cont30.sh` | ~74h + 评测 |
| 6 | 显存矩阵（ZOO vs BP） | `scripts/repro/06_mem_bench.sh` | ~10min |

长任务请用 `setsid nohup bash <script> >/dev/null 2>&1 &`（harness 前台 shell 约 2h 会被回收）。

## 评测 endpoints 包

`scripts/export_endpoints.py` 把论文 Evaluation 各表所需的关键 endpoint adapter、对应评测 JSON、
前置权重（`pos_embed.pt`、`head.pt`、`predictor.pth`、`pruned_config.pth`）以及显存 CSV 汇总到 `checkpoints/out/`：

- `adapters/<group>/<name>/adapter_roundNNN/` — PEFT LoRA adapter（r29 / r59 / r7）；
- `evals/<group>/<name>/` — 端点评测（`eval_*` 带 CASC 掩码，`evaloff_*` 无掩码）；
- `weights/opt|bert/` — 位置嵌入、predictor、剪枝配置、Step-1 分类头；
- `memory/` — `tab:mem_peak` 的 OPT-350M 显存数据；
- `MANIFEST.csv` / `WEIGHTS.csv` / `SHA256SUMS` — 论文表 → 产物 → 指标 → sha256，逐项对账
  （任一指标与论文表相差超过 1e-3 时脚本直接失败）。

## 结果包

| 包 | 内容 |
|---|---|
| `reports/bert_and_other/` | BERT 四臂/续训/技术线 + OPT fixed 的 tidy CSV、README、`BERT_CONFIG.md`（脚本生成 + 对账闸门） |
| `reports/opt/` | OPT 图/表（论文用） |
| `reports/r60/` | BERT r60–r89 续训曲线/端点/配对 |
| `reports/mem/` | 显存交付包：ZOO vs BP × 1K–16K × dense/sparse（含 `make_report.py` 闸门） |

## 工作口径与硬坑

- **ZOO 必须在 `model.eval()` 下**做有限差分（dropout 会吞掉差分信号）。
- **精度**：正式训练 `--dtype fp32 --attn sdpa_prune`；bf16 的损失量化会吞掉 ZOO 信号。
- **predictor 必须加载**（OPT step5 / BERT predictor_bert）；缺失 key 硬报错、bias 零填充。
- **`--delta-clip` 是逐 client 全局 L2 裁剪**，不是梯度裁剪（OPT 13.34 / BERT 1.0）。
- **`--doc-order cycle`** 是语料轮转协议，长跑必须显式加；261012 前的 fixed 结果不可混表。
- **CASC**：`--catv-normalize client`（默认）保证 client 等权；历史消融用 `off`；DistilBERT 需 `keep_block0=True`。
- **CPUR/VQ**：默认 `--vq-channels-mode per_client`（1 channel/client，旧总预算比例 = rate）；
  `per_result` 仅用于复现旧 run。
- 规范/标注：`docs/standard.md`；每个改动的出处：`docs/ailog/`；换机交接：`docs/weight-manifest.md`。
- 面向 Agent 的工作手册：`AGENTS.md`（结果表、硬坑、运行口径）。

## 第三方

- Jenga（ATC'25 artifact）：contextual token sparsity 与 OPT/DistilBERT 建模；
- FwdLLM：forward-gradient ZOO 参考实现；
- VQ：Count Sketch 上行压缩参考实现。

第三方代码以只读 submodule 引入，许可与引用遵循各自仓库。
