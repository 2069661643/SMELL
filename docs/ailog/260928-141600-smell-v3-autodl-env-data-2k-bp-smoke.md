# Ailog 260928-141600 — Phase N: AutoDL 4090 环境+数据就绪，2k 稀疏 BP smoke 双配置通过

> 承接 `260928-135000`（新机 checkout）。本 session 在 AutoDL 4090 完成：建 env、重建 a01 数据、跑通 2k 稀疏 BP smoke（bf16+FA2 与 fp32+sdpa_prune 各一次），并修复两处跨机可运行性 bug。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 13:53 | 新增 env 驱动（gitignore） | `temp/run_env_setup_bg.sh` | — | 后台跑 `setup_server.sh`，`[HH:MM:SS] STEP` 打点 + `last_env_setup_*` |
| 13:56 | 修复 torch wheel 文件名 | `scripts/setup_server.sh` | `TORCH_WHEEL_FILE` 定义、download/install 段 | 旧名 `torch-2.8.0+cu128.whl` 缺 cp310 tag → pip `Invalid wheel filename`；改完整名 + 已存在则复用 |
| 13:57 | 修复 build_discovery 默认输出 | `src/data/build_discovery_16k.py` | `parse_args --out` | 旧默认写死云端家目录 `/home/yangyongbo118/...`；改相对 `dataset_v3/discovery_16k` |
| 13:57–14:07 | 重建环境 | `~/miniconda3/envs/SMELL` | — | py3.10.21 + torch 2.8.0+cu128 + flash-attn 2.8.3 + transformers 4.45.2 / peft 0.13.2 + `jenga_src.pth` + opt-350m；`setup_server.sh` verify OK |
| 14:07–14:15 | 重建 a01 数据 | `dataset_v3/discovery_16k/a01` | `build_discovery_16k` → `build_warmup_16k` → `check_partition` | PASSED；**warmup sha256 = 清单 `c1192672…`** |
| 14:15 | 2k 稀疏 BP smoke ×2 | `temp/run_data_smoke_bg.sh` → `src/fed/run_fed.py` | — | 见下表；GPU 峰值正常、进程干净退出 |

## Smoke 结果（1 client × 2 samples × 1 step，`--truncate 2048 --sparse 0.4 --lr 1e-3 --bp-clip 1.0`，均带 pos_only）

| 配置 | train_loss | delta_norm | round_seconds | 产物 |
|---|---|---|---|---|
| `--trainer bp`（bf16 + FA2，Jenga 原生 thresh） | 3.6245 | 0.886 | 1.53 | `temp/smoke_bp_flash/a01/metrics.jsonl` |
| `--trainer bp --dtype fp32 --attn sdpa_prune` + step4 predictor/pruned_config | 3.6442 | 0.886 | 1.45 | `temp/smoke_bp_sdpa_prune/a01/metrics.jsonl` |

- sdpa_prune 配置 `predictor_loaded_tensors=144`（与云端一致）；两配置均无 NaN、无 OOM。
- 复现命令：
  ```bash
  PY=~/miniconda3/envs/SMELL/bin/python
  $PY src/fed/run_fed.py --tag a01 --gpu 0 --trainer bp --catv off \
    --sparse 0.4 --truncate 2048 --rounds 1 --local-steps 1 --max-clients 1 --max-train-samples 2 \
    --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
    --out-root temp/smoke_bp_flash
  $PY src/fed/run_fed.py --tag a01 --gpu 0 --trainer bp --catv off \
    --dtype fp32 --attn sdpa_prune --sparse 0.4 --truncate 2048 --rounds 1 --local-steps 1 \
    --max-clients 1 --max-train-samples 2 \
    --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
    --predictor checkpoints/predictor/step4_a01_pos_only_causal/predictor.pth \
    --pruned-config checkpoints/predictor/step4_a01_pos_only_causal/pruned_config.pth \
    --out-root temp/smoke_bp_sdpa_prune
  ```

## 关键结论 / 备注

- **数据可完全重建**：`sileod/discovery`（HF/hf-mirror）+ opt-350m tokenizer，无需 Jenga 的 `dataset.zip`（那是 LongBench/PPL 用的）；a01 产物与云端清单 sha256 一致。
- `jenga` 为 namespace package（`third_party/Jenga/src/jenga` 无 `__init__.py`，`jenga.__file__=None`），但 `jenga.utils.config_utils` 等子模块导入正常，无需 editable 安装。
- `setup_server.sh` 现已可整机复现（Amax 仍需覆盖 CONDA/env 路径；AutoDL 默认适配）。

## 待办

- [ ] 需要 a03 时：`build_discovery_16k.py --tag a03 --alpha 0.3` + `build_warmup_16k.py --tag a03`。
- [ ] 用 step4 predictor 跑 TD-2'（`temp/diag_cos_grid.py` 不在 git，需从云端带或重建）→ 过门控再议 TD-3。
- [ ] `hello-world.sh` 尚未在本机跑（可选环境自检）。
