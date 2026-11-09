# SMELL-v3

**Federated long-context LLM fine-tuning with token-level sparsity × forward-gradient (ZOO).**
Jenga-style contextual token sparsity on the forward pass, FwdLLM-style zeroth-order optimization for
gradient-free federated updates, plus uplink compression (Count Sketch) and VARIANCE-aware adaptive directions (VAR).

**English** | [中文](README_zh.md)

> Research artifact (MobiCom 2027 / ICDE 2027 target). All numbers below are reproduced from archived runs
> (16K context, 30 devices, Dirichlet α=0.1/0.3, seed 42).

## Results at a glance

| Line | Setting | Endpoint |
|---|---|---|
| **BERT 4-arm, r89** | a01 off / a01 on / a03 off / a03 on | NLL/acc **1.4703/.502**, **1.4659/.490**, **1.4724/.490**, **1.4610/.488** |
| **BERT dense VAR, r89** | `--sparse 1.0` + VAR Method A | **1.3096/.520** (strongest BERT line; not Jenga) |
| **BERT 3-on, r89** | VAR + CATV(client) + VQ 0.5 per-result | **1.4598/.480** |
| **BERT VQ 0.5, r89** | per-result (64 channels/client) | **1.4780/.494** (vs VQ off 1.4724/.490 → +0.38% NLL) |
| **VQ per-client, r29** | 1 channel/client, 1.11 MB/round (0.5× dense, 64× less than per-result) | 1.6274/.448 (vs off 1.5887/.466, **ΔNLL +0.0387**, p=9.6e-15) |
| **VAR Method A, r29** | per-sample raw var, D=4..16, T0/T10/T20 | 1.5845/.464 (vs off 1.5887/.466, ΔNLL −0.0042) |
| **OPT 4-arm, r59** | a01 off / a01 on / a03 off / a03 on (full PPL / answer PPL) | **30.953/2174.8**, **31.126/2277.1**, **30.864/2362.6**, **30.959/1504.2** |
| **Memory (16K)** | ZOO (real `zo_grad`) vs BP | **2.66 / 2.21 GiB vs 14.51 / 12.69 GiB** (dense / sparse 0.4; ZOO saves **5.5×**) |

Key findings: the CASC (CATV) gain comes mainly from **weights on OPT a03 (89%)** but from the **inference mask on
BERT**; the net CASC advantage decays with long training (not significant at BERT r89 / OPT r59 for a01);
VQ per-client trades 1/64 of the uplink for ~0.04 nats of accumulated error.

## Repository layout

```
src/
  models/    modeling_opt_smell.py, modeling_distilbert_smell.py, position_embed*.py, token_selector.py (CASC)
  data/      build_discovery_16k.py, build_warmup_16k.py, build_arxiv_16k.py, build_warmup_arxiv_16k.py, check_partition.py
  train/     zoo.py, lora.py, longctx_adapt.py, train_predictor*.py, train_pos_head_bert.py, lowrank_reparam.py
  fed/       run_fed.py (entry), serial_fedavg.py (client runner + FedAvg)
  compress/  count_sketch.py (CPUR)
  eval/      ppl.py (OPT), classify.py (DistilBERT)
  bench/     memory.py (ZOO/BP memory matrix)
  hello_world.py            environment checker (Jenga hello_world style)
scripts/
  check-env.sh              → src/hello_world.py (deps/GPU/base model/data/weights/forward smoke)
  setup_server.sh           cloud A40 bootstrap (idempotent; --dry-run/--check)
  run_ablation_cloud.sh     four-arm OPT/BERT ablation across 4 GPUs
  repro/                    six paper-reproduction scripts (see scripts/repro/README.md)
  export_endpoints.py       bundles key endpoints + prerequisite weights (checkpoints/out)
reports/
  bert_and_other/           BERT + OPT fixed tidy CSVs/README (script-generated, reconciliation gate)
  opt/                      OPT figures/tables (paper)
  r60/                      BERT r60–r89 continuation package
  mem/                      memory package (ZOO vs BP, 1K–16K)
docs/
  standard.md               code/annotation conventions
  weight-manifest.md        weight/data paths + sha256 + run flags (machine hand-over)
  ailog/                    timestamped experiment logs (provenance of every change/conclusion)
third_party/                Jenga / FwdLLM / VQ (read-only submodules)
dataset_v3/  checkpoints/  logs/  temp/    runtime directories (gitignored)
```

## Environment

```bash
# Local machine (AutoDL RTX 4090): conda env SMELL (Python 3.10)
PY=~/miniconda3/envs/SMELL/bin/python
$PY -V && $PY -c "import torch; print(torch.__version__, torch.version.cuda)"
# torch 2.8.0+cu128, flash-attn 2.8.3, transformers 4.45.2, peft 0.13.2
bash scripts/check-env.sh            # 34 checks: deps / GPU / base model / data / weights / forward smoke
bash scripts/check-env.sh --no-gpu   # import-only (machines without a GPU)
```

- Dependencies: `requirements-wsl-cu128.txt`; from scratch on a cloud host: `bash scripts/setup_server.sh` (`--dry-run` first).
- `third_party/Jenga` is added to `sys.path` from source (`jenga_src.pth`); submodules are never modified.
- Data/weights are not tracked by git: see `docs/weight-manifest.md` (`dataset_v3/`, BERT step1/predictor_bert,
  OPT posemb/predictor step5, `opt-350m`).

## Quick start

```bash
# DistilBERT smoke: 2 clients × 1 round × 2 samples × 2K
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

## Reproduce the paper experiments

See [`scripts/repro/README.md`](scripts/repro/README.md) for environment variables, outputs and reference archives:

| # | Experiment | Script | Reference time (single 4090) |
|---|---|---|---|
| 1 | BERT r30 four arms | `scripts/repro/01_bert_r30_four_arm.sh` | ~26h |
| 2 | BERT continuation (r30→r59 / r60→r89) | `scripts/repro/02_bert_cont.sh` | ~26h per segment |
| 3 | BERT technique lines (VQ / VAR / 3-on) | `scripts/repro/03_bert_techniques.sh` | ~6.5h per line |
| 4 | OPT r30 four arms | `scripts/repro/04_opt_r30_four_arm.sh` | ~4 days (serial) |
| 5 | OPT continuation + eval-off | `scripts/repro/05_opt_cont30.sh` | ~74h + evaluations |
| 6 | Memory matrix (ZOO vs BP) | `scripts/repro/06_mem_bench.sh` | ~10 min |

Long jobs: `setsid nohup bash <script> >/dev/null 2>&1 &` (a foreground harness shell is reclaimed after ~2h).

## Evaluation endpoints bundle

`scripts/export_endpoints.py` assembles the key endpoint adapters used in the paper's Evaluation section, the
matching evaluation JSONs, the prerequisite weights (`pos_embed.pt`, `head.pt`, `predictor.pth`,
`pruned_config.pth`) and the memory CSVs into `checkpoints/out/`:

- `adapters/<group>/<name>/adapter_roundNNN/` — PEFT LoRA adapters (r29 / r59 / r7);
- `evals/<group>/<name>/` — endpoint evaluations (`eval_*` with the CASC mask, `evaloff_*` without);
- `weights/opt|bert/` — positional embeddings, predictor weights, pruning configs, Step-1 head;
- `memory/` — OPT-350M peak-memory data for `tab:mem_peak`;
- `MANIFEST.csv` / `WEIGHTS.csv` / `SHA256SUMS` — paper table → artifact → metric → sha256, verified
  against the paper tables (the script exits if any value differs by more than 1e-3).

## Reports

| Package | Contents |
|---|---|
| `reports/bert_and_other/` | BERT four-arm/continuation/technique + OPT fixed tidy CSVs, README, `BERT_CONFIG.md` (generated, reconciliation gate) |
| `reports/opt/` | OPT figures/tables (paper) |
| `reports/r60/` | BERT r60–r89 continuation curves/endpoints/paired tests |
| `reports/mem/` | Memory package: ZOO vs BP × 1K–16K × dense/sparse (with a `make_report.py` gate) |

## Working conventions & pitfalls

- **ZOO must run under `model.eval()`** for finite differences (dropout swallows the signal).
- **Precision**: formal training uses `--dtype fp32 --attn sdpa_prune`; bf16 loss quantization destroys the ZOO signal.
- **The predictor must be loaded** (OPT step5 / BERT predictor_bert); missing keys fail hard, biases are zero-filled.
- **`--delta-clip` is a per-client global L2 clip**, not gradient clipping (OPT 13.34 / BERT 1.0).
- **`--doc-order cycle`** is the corpus-rotation protocol and is required for long runs; pre-261012 fixed results must not be mixed in.
- **CASC**: `--catv-normalize client` (default) keeps devices equally weighted; historical ablations use `off`;
  DistilBERT requires `keep_block0=True`.
- **CPUR/VQ**: `--vq-channels-mode per_client` is the default (1 channel/client, old total-budget ratio = rate);
  `per_result` only reproduces legacy runs.
- Conventions/annotations: `docs/standard.md`; provenance of every change: `docs/ailog/`; machine hand-over: `docs/weight-manifest.md`.
- Agent-facing working manual: `AGENTS.md` (result tables, pitfalls, run protocols).

## Third-party

- Jenga (ATC'25 artifact): contextual token sparsity and OPT/DistilBERT modeling;
- FwdLLM: forward-gradient ZOO reference implementation;
- VQ: Count Sketch uplink-compression reference implementation.

Third-party code is vendored as read-only submodules; licenses and citations follow the respective repositories.
