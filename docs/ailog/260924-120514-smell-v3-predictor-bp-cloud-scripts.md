# Ailog 260924-120514 — SMELL-v3: predictor 训练器 + BP/适配加载 runner + 云端脚本交付

## 概述

补齐交付表里剩余 4 项 TODO：(1) `src/train/train_predictor.py`（Jenga predictor 单机训练器，仅训 q/k 线性分支；含动态剪枝、`predictor.pth`+`pruned_config.pth` 导出）；(2) `src/fed/run_fed.py`/`serial_fedavg.py` 扩展 `--trainer {zoo,bp}`、`--truncate`、`--pos-checkpoint`、`--adapter-init`、`--act-pack`（ZOO 路径保持逐位兼容）；(3) `scripts/setup_server.sh`（云端幂等 bootstrap，`--dry-run/--check/--skip-data`）；(4) `scripts/run_ablation_cloud.sh`（4 卡并发消融启动器，GPU0-3 = a01 off/on、a03 off/on，遵循 AGENTS.md 长任务约定）。**重要发现**：Jenga `third_party/Jenga/src/jenga/ops/flash_block.py:171` 把 `HEAD_DIM` 硬编码为 128，OPT-350M（head_dim=64）的 `pooling_gt` 因此错误且偶发 illegal access——trainer 用等价 torch 实现 `block_attn_pool_fixed` monkeypatch 规避（Jenga 自身的 OPT predictor 训练同样受影响，属上游 bug）。

## 变更列表

| 时间 | 操作 | 文件 | 规模 | 说明 |
|---|---|---|---|---|
| 11:58 | 新增 | `src/train/train_predictor.py` | 478 行 | 冻结 base（trainable=62,914,560 / 144 张量，恰好 predictor q/k linears）、SmoothL1(predict_mask, pooling_gt) 逐层平均、每 100 步 `prune_neurons`（0.8→0.75…）、`predictor.pth`+`pruned_config.pth`+metrics |
| 12:00 | 修改 | `src/fed/serial_fedavg.py` | +31/-13 | `ClientRunner(trainer="zoo"\|"bp", bp_clip=1.0)`；BP = zero_grad→bf16 前向→backward→clip→AdamW.step，delta 口径不变 |
| 12:00 | 修改 | `src/fed/run_fed.py` | +93 | 新 CLI：`--trainer/--bp-clip/--truncate/--pos-checkpoint/--adapter-init/--act-pack`；`effective_seq_len` 驱动 `ensure_positions`；mask 注入在 PeftModel 重载后重新解析 `base_config` |
| 12:04 | 新增 | `scripts/setup_server.sh` | 196 行 | Miniconda(USTC) → env SMELL(py3.10) → torch 2.8.0+cu128(sha256 校验) → FA 2.8.3 按 ABI(ghfast 代理) → pinned deps(USTC PyPI) → `jenga_src.pth` → hf-mirror 拉 opt-350m；`--check` 本机验证通过 |
| 12:04 | 新增 | `scripts/run_ablation_cloud.sh` | 122 行 | 4 并发 setsid nohup；每实验独立 driver.log/run.sh、`temp/logs/last_ablation_dir.txt`；`--dry-run` 输出 4 条命令（映射已验证） |

## 验证结果

- **predictor 训练器**：`py_compile` OK；512-token smoke（4 步）loss 0.0041→0.0031、144 张量全为 `predictor.{q,k}linear*`；1024-token + 剪枝 smoke：`pruned=true`，layer0 `q2: 4096→3575→3353`、`k2: 4096→3617→3402`，`predictor.pth` 62.9M→56.3M 参数；2048-token 在 8GB 跑通（~1.1s/step）。
- **run_fed 扩展**：ZOO 回归与改动前记录逐位一致（`loss=7.5, delta_norm=44282.61`）；BP smoke（truncate 1024）`loss=3.888, delta_norm=0.1537`，峰值 1.90GiB；`--pos-checkpoint`（16k 表）+ `--adapter-init`（v1.57M）加载 smoke 通过，trainable 数与 adapter 完全一致。
- **云端脚本**：`bash -n` 双过；`--dry-run` 输出正确（4 条命令 GPU0-3 映射 a01 off/on、a03 off/on）；`setup_server.sh --check` 在本机 env 上验证 OK（torch 2.8.0+cu128 / FA 2.8.3 / jenga import / opt-350m config 存在）。

## 关键发现

- **Jenga `flash_block.py` HEAD_DIM 硬编码 128**（`third_party/.../ops/flash_block.py:171`）：OPT-350M head_dim=64，`block_attn_pool` 产出错误 `pooling_gt`（probe 最大绝对误差 20.16）且偶发 CUDA illegal access。trainer 使用等价 torch 分块实现（head_dim 64/128 误差 ≤0.012）。若后续要复用 Jenga 的 OPT predictor 训练代码，必须先修此 bug（不改 third_party，保持 monkeypatch/自研路径）。
- **PEFT 适配器初始化方式**：`PeftModel.from_pretrained(build_lora_model(...))` 会双重包装且权重未载入；正确做法是 `PeftModel.from_pretrained(base_opt, path, is_trainable=True)`（`run_fed --adapter-init` 采用此路径）。
- **Jenga act-pack hooks 污染 BP**：BP 训练需 `--act-pack off`（默认），否则后一半 token 的梯度被静默置零。

## 交付表（更新）

| # | 交付物 | 位置 | 状态 |
|---|---|---|---|
| 1 | Discovery 16k 分片（a01/a03） | `dataset_v3/discovery_16k/`（gitignore） | ✅ |
| 2 | warmup 池（512×16k，零重叠） | 同上 | ✅ |
| 3 | 位置扩展模块 | `src/models/position_embed.py` | ✅ |
| 4 | 长上下文适配器 B/C | `src/train/longctx_adapt.py` | ✅（16k 待 A40） |
| 5 | CATV（r<s） | `src/models/token_selector.py` 等 | ✅ |
| 6 | predictor 训练器 | `src/train/train_predictor.py` | ✅ |
| 7 | BP/适配加载 runner | `src/fed/run_fed.py`（`--trainer bp` 等） | ✅ |
| 8 | 云端 bootstrap | `scripts/setup_server.sh` | ✅ |
| 9 | 消融编排（4 并发） | `scripts/run_ablation_cloud.sh` | ✅ |

## 接下来的流程（云端）

```
Step1 位置适配两方案: src/train/longctx_adapt.py（pos_only / pos_lora，16k warmup 池）
      判据: G-PPL<50、ppl_tail 不劣化
Step2 predictor: src/train/train_predictor.py --pos-checkpoint <pos_embed.pt> [--adapter <warmup adapter/>]
      产出 predictor.pth + pruned_config.pth（供 JengaSparse/CATV）
Step3 BP 可行性: run_fed --trainer bp --pos-checkpoint ... --adapter-init ...（c=2~3，重调 ZOO 步长）
Step4 消融: scripts/run_ablation_cloud.sh --pos-checkpoint ... --adapter-init ...（4 卡并发，c=30，ZOO+LoRA）
Step5 调参/出图
```

## 待办 / 风险

- 16k 的 pos_emb/predictor/正式实验均在云端 A40；本机只做 smoke。
- ZOO 步长仍待重调（当前 lr=1e-3 第 2 轮 NaN）。
- `--adapter-init` 在正式消融中的语义（warmup adapter 作为 FedAvg 初始 LoRA）需要在 Step3 固化并记录。
