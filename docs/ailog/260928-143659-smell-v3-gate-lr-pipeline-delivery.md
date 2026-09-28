# Ailog 260928-143659 — Phase N: 门控 + lr 校准流水线交付（source 入 scripts）

> 承接 `260928-110233`（predictor causal 修复）。本 session（1）FF 同步另一个 server 的提交（`2ade9bb`）；
> （2）把「稀疏 cos 门控 + lr 校准」从 gitignore 的 temp 脚本**提升为 tracked 交付件** `scripts/`，并把源码设计写入本 ailog。

## 变更列表

| 时间 | 操作 | 文件 | 说明 |
|---|---|---|---|
| 14:36 | FF 合并另一 server 更新 | (git) `f4e879d..2ade9bb` | 对方新增 AutoDL bootstrap/env/smoke 3 提交（AGENTS.md、setup_server.sh、build_discovery_16k.py、2 ailog），无冲突 |
| 14:36 | 新增 tracked 诊断脚本 | `scripts/diag_cos_grid.py` | 由 `temp/diag_cos_grid.py` COPY；默认 predictor/pruned 改为 step4 causal；`py_compile` OK |
| 14:36 | 新增 tracked 流水线脚本 | `scripts/run_k4_td2_td3.sh` | 门控+lr 校准源码；REPO 自动定位、路径/阈值可用环境变量覆盖；`bash -n` OK |
| 14:37 | 新增本 ailog | `docs/ailog/260928-143659-...md` | 交付说明 |

## 一、交付件

- **`scripts/run_k4_td2_td3.sh`**：k=4 sparsity-ZOO 全流水线（TD-2' → 门控 → lr 校准 → TD-3）。
- **`scripts/diag_cos_grid.py`**：TD-2' 的块级 cos 诊断（`--blocks k4 --block-layers 20-23` 支持多层子空间）。

两者均为 tracked（区别于 `temp/` 下的 gitignore 版本，temp 版继续用于本地临时实验）。`REPO` 由脚本位置自推，`PY/POS/PRED/PCS/...` 可用环境变量覆盖；`GPU` 默认 2。

## 二、门控 + lr 设计（源码口径）

1. **门控（cos gate）**：TD-2' 测 `cos(k4, L2, D22, c30)`，`>= COS_GATE`（默认 **0.05**）才启动 TD-3；否则约 1.4h 提前终止。
   - 依据：dense 理论 `cos ≈ 0.5·√(cLD/d_eff)`；k=4 子空间 `d_eff = 4×8192 = 32768`；causal 修复后实测 ratio≈0.44（贴近理论 0.5）。
2. **lr 校准**：1 client/1 sample 探针，固定 `probe lr = 4e-6`，读回 `delta_norm`，令
   ```
   lr = probe_lr × TARGET_DNORM / delta_norm = 4e-6 × 0.38 / delta_norm     (clamp [1e-7, 1e-4])
   ```
   - `TARGET_DNORM = 0.38` = BP k=4 实测参考（对齐 BP 每轮参数位移量级）。
   - 原理：ZOO delta 与 lr 成正比（`δ ∝ lr`），故按目标 δ 线性反解 lr。
3. **predictor 配对约束**：必须 `--predictor/--pruned-config = step4_a01_pos_only_causal`（causal 修复版）；用非 causal 版会只选尾部块（16k LM loss 5.08 vs 3.44）。
4. **TD-3 参数**：`--trainer zoo --zo-subspace layers --zo-layer-rotate --zo-layer-group 4 --max-clients 30 --local-steps 2 --zo-directions 22 --zo-eps 1e-3 --rounds 30 --eval-every 1`，`--dtype fp32 --attn sdpa_prune --sparse 0.4`。

## 三、本轮实测（cos + lr）

TD-2'（causal predictor，`temp/diag_cos_sparse_td2_k4_causal.json`）：

| c | cos | formula(√(cLD/d)) | ratio |
|---|---|---|---|
| 1 | 0.0249 | 0.0366 | 0.68 |
| 10 | 0.0491 | 0.1159 | 0.42 |
| **30** | **0.0885** | 0.2007 | **0.441** |

- 门控：`cos=0.0885 >= 0.05` → **通过**（注意：非 causal 版为 0.197，修正后回落到理论标度）。
- lr 校准：`delta_norm=0.1452` → `lr = 4e-6 × 0.38 / 0.1452 = **1.0467e-5**`（TD-3 实际 lr，写入 `logs/fed/zoo_k4_sparse_causal/a01/config.json`）。
- 观测：修正后 δ 从 1.61 降到 0.145（信号变弱 → lr 自动放大 ~11×），需在 TD-3 前几轮盯 `delta_norm`/loss/G-PPL 是否发散。

## 四、复现命令

```bash
# 全流水线（后台）：GPU2、默认 30 轮
setsid nohup bash scripts/run_k4_td2_td3.sh >/dev/null 2>&1 < /dev/null &
# 仅诊断
python scripts/diag_cos_grid.py --gpu 2 --attn sdpa_prune --dtype fp32 --seq 16384 \
  --block-layers 20-23 --blocks k4 --n-clients 30 --ls 2 --ds 22 --cs 1,5,10,20,30 \
  --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
  --predictor      checkpoints/predictor/step4_a01_pos_only_causal/predictor.pth \
  --pruned-config  checkpoints/predictor/step4_a01_pos_only_causal/pruned_config.pth \
  --out temp/diag_cos_sparse_td2_k4_causal.json
```

## 待办

- [ ] 另一 server pull `exp/rank-rotation-lora`（含本交付件）后，按其显存/卡号覆盖 `GPU`/`PY` 复跑。
- [ ] TD-3 前 0–2 轮检查 `delta_norm` 稳定性与 loss/G-PPL 是否发散（lr 放大 11× 的风险）。
- [ ] 若 TD-3 收敛：出 G-PPL 曲线并与 BP k=4（ppl 26.8）/全量（~21.5）对比。

## 环境

host `amax`（A40×4，仅 GPU2）；`smell-v2`（torch 2.1.2+cu118 / FA 2.4.2）。当前长跑 TD-3 于 14:03 起（`logs/fed/zoo_k4_sparse_causal/`），ETA ~9/30 09:20。
