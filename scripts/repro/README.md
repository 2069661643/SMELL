# 复现脚本（scripts/repro）

每个脚本都对应论文里的一条实验线；默认参数即为论文口径。
所有脚本都以仓库根为工作目录、用 `PY`（默认 `~/miniconda3/envs/SMELL/bin/python`）运行，
`GPU`（默认 0）、`DRY_RUN=1`（只打印命令）、`ARCHIVE_ROOT`（非空则把最后一轮 adapter 归档）可覆盖。

```bash
cd <repo>
bash scripts/hello-world.sh          # 环境自检（Jenga hello_world 风格）
DRY_RUN=1 bash scripts/repro/01_bert_r30_four_arm.sh   # 先看命令
```

## 索引

| # | 实验 | 脚本 | 关键环境变量 | 主要产物 / 对照 |
|---|---|---|---|---|
| 1 | **BERT r30 四臂**（a01/a03 × CATV off/on） | `01_bert_r30_four_arm.sh` | — | `logs/fed/repro_bert_r30_four_arm/`；对照 `checkpoints/distilbert_ablation_r30_fixed/`、`reports/bert_and_other/` |
| 2 | **BERT 续训**（r30→r59 / r60→r89） | `02_bert_cont.sh` | `OFFSET`（30/60）、`LR`、`INIT_ARCH` | `logs/fed/repro_bert_cont<OFFSET>/`；对照 `checkpoints/distilbert_ablation_r30_fixed_cont30|cont60/` |
| 3 | **BERT 技术线**（VQ/VAR/组合） | `03_bert_techniques.sh` | `CELL=all|vq_perresult|vq_perclient|var_methoda|combo_3on` | `logs/fed/repro_bert_techniques/`；对照 `checkpoints/distilbert_{vq050_bf16_perresult,vq_perclient,var_methoda,var_catv_vq}_a03_r30/` |
| 4 | **OPT r30 四臂**（k24/D32/L2） | `04_opt_r30_four_arm.sh` | — | `logs/fed/repro_opt_r30_four_arm/`；对照 `checkpoints/opt_ablation_r30_fixed/`、`reports/bert_and_other/`（`opt_r30_*`） |
| 5 | **OPT 续训 r30→r59**（+可选 eval-off） | `05_opt_cont30.sh` | `EVALOFF=1`、`OFFSET`、`LR` | `logs/fed/repro_opt_cont30/`；对照 `checkpoints/opt_ablation_r30_fixed_cont30/` |
| 6 | **显存矩阵**（ZOO vs BP，1k–16k） | `06_mem_bench.sh` | `SEQS`、`MODES`、`ZOO_DIRECTIONS` | `temp/mem_sweep_repro/`；交付表 `reports/mem/README.md` |

## 常跑组合

```bash
# BERT 主表（~26h）
bash scripts/repro/01_bert_r30_four_arm.sh

# BERT 第三段（r60→r89，lr=1/3，~26h）
bash scripts/repro/02_bert_cont.sh

# BERT 第二段（r30→r59，lr=1.4e-3）
OFFSET=30 LR=0.0014 INIT_ARCH="$PWD/checkpoints/distilbert_ablation_r30_fixed" \
  bash scripts/repro/02_bert_cont.sh

# BERT 技术线（VQ per-result / per-client / VAR / 3-on，各 ~6.5h）
CELL=all bash scripts/repro/03_bert_techniques.sh

# OPT 四臂（单卡串行 ~4 天；云端 4 卡可并行拆开跑）
bash scripts/repro/04_opt_r30_four_arm.sh

# OPT 续训 + eval-off（~74h + 12 次评测）
EVALOFF=1 bash scripts/repro/05_opt_cont30.sh

# 显存（~10 min）
bash scripts/repro/06_mem_bench.sh
```

## 约定与注意

- **输出目录**：脚本一律写 `logs/fed/repro_*`，不覆盖历史归档；`ARCHIVE_ROOT=$PWD/checkpoints/repro_x`
  时才会把最后一轮 adapter 按历史结构（`adapter_roundNNN + metrics + config + SHA256SUMS`）归档。
- **数据/权重**：前置产物见 `docs/weight-manifest.md`（数据集 `dataset_v3/`、BERT step1/predictor_bert、
  OPT posemb/predictor step5、`third_party/Jenga/checkpoints/opt-350m`）。
- **长任务**：>1 min 请用 `setsid nohup bash <script> >/dev/null 2>&1 &`，不要用 harness 的前台 shell。
- **单卡**：`--gpu` 只认物理 0（脚本已 `GPU=0`）；云端 4 卡请把 `02/04` 的循环拆成 4 份并行。
- **协议连续性**：BERT 四臂/续训必须 `--catv-normalize off`；组合线（3-on）用 `client`；
  BERT 主表 r30 段用 `lr=1.4e-3`、续训第二段起 `lr=1.4e-3/3`。
