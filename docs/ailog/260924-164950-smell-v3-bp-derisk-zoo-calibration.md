# Ailog 260924-164950 — SMELL-v3: Step3 BP 去风险 + ZOO 步长校准（云端 A40/GPU2）

## 概述

在选定 `pos_lora` 权重（`checkpoints/posemb_step1/a01_pos_lora_500step/`）上做 Step3：**BP 路径去风险**（16k、c=3、真 LoRA）与 **ZOO 步长校准**（16k、c=2、rounds=2）。结论：BP 机制在 16k 稳定（δ≈0.30，loss 下降）；ZOO 更新范数 `δ ∝ lr`，历史 NaN 只是 lr 太大，`lr≤1e-6` 两轮均稳定；按「匹配 BP δ≈0.3」把消融默认 ZOO lr 由 `1e-4` 暂定为 **`1e-7`**（待 BP 收敛复核）。`cos(ZOO,BP)≈0`，与 v2 结论一致。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 16:38-16:42 | 运行 | `logs/fed/step3_*`（gitignore） | — | 1×BP + 3×ZOO 校准 run，全部无 NaN/OOM |
| 16:49 | 修改 | `scripts/run_ablation_cloud.sh` | L10 | ZOO 默认 `LR` `1e-4`→`1e-7`（标定值，待 BP 收敛复核） |
| 16:49 | 修改 | `scripts/run_ablation_cloud.sh` | L6 | `PY` 默认由不存在的 `$HOME/miniconda3/envs/SMELL` 改为 `$HOME/applications/anaconda3/envs/smell-v2`，并可用 `PY=` 覆盖 |

## 关键数据（16k，catv off，`--pos-checkpoint`+`--adapter-init`）

| run | trainer | lr | dir | round0 loss / δ | round1 loss / δ | 稳定性 |
|---|---|---|---|---|---|---|
| step3_bp | bp (clip=1.0, local=4, c=3) | 1e-4 | — | 3.0119 / 0.3043 | 2.9980 / 0.2953 | 稳定，loss↓ |
| step3_zoo_lr1e-6 | zoo (local=1, c=2) | 1e-6 | 8 | 3.0 / 3.8882 | 3.0 / 2.4167 | 稳定 |
| step3_zoo_lr1e-8 | zoo | 1e-8 | 8 | 3.0 / 0.03888 | 3.0 / 0.03708 | 稳定 |
| step3_zoo_lr1e-6_dir64 | zoo | 1e-6 | 64 | 2.9922 / 1.2753 | 2.9922 / 1.2333 | 稳定（85s/轮） |
| （历史 v3） | zoo | 1e-3 | 8 | ~5.5e4 | NaN | 发散 |

- **δ 近似 ∝ lr**（1e-3→5.5e4、1e-6→3.9、1e-8→0.039）。根源：`src/train/zoo.py` 的单位方向估计带 `dim` 补偿（`grad += dim·(ΔL/2ε)·v`，量级 ~√dim≈1250），故 ZOO 可用 lr 比常规 GD 小 ~3 个数量级。
- **dir64 vs dir8**：同 lr=1e-6 下 δ 由 3.89 降到 1.28（~3×，方差减小、估计更准），代价单轮 85s（~7.5×）。
- **cos(ZOO,BP)≈0**（ZOO 各 run 的 `cos_mean_sampled` 在 ±0.001），BP 两轮 cos=0.157→0.062，均与 v2「ZOO 与 BP 近乎正交」一致 → 仅对齐 δ 范数不足以保证 ZOO 有进度，需 BP 收敛做参照。

## 设计决策

- **BP 作为收敛参照**：既然 cos≈0，先看 BP 在 16k/联邦协议下能到何种 loss/G-PPL，再据此判断 ZOO 是否「收敛但慢」或「方向无效」。故 Step4 前应先做一次较长 BP（多轮）收敛观察。
- **ZOO lr 暂定 1e-7**：以 δ 匹配 BP（0.30）为准则的保守起点；`--zo-directions`/`--zo-eps` 维持 8/1e-3，待 BP 收敛后再定是否需要 dir≥32。
- **`run_ablation_cloud.sh` PY 修正**：原默认路径在云端不存在，属既有云端阻塞项，本次一并修掉并支持 `PY=` 覆盖。

## 云端环境

host `amax`（A40×4，driver 570/CUDA 12.8），全程**仅 GPU2**（GPU0/1/3 被他人占用）。环境复用 `smell-v2`（torch 2.1.2+cu118 / FA 2.4.2 / transformers 4.45.2 / peft 0.19.1 / datasets 5.0.0）。jenga 来自 editable 安装 `JengaForMemoryTest/Jenga/src`（非本仓子模块）。

## 待办 / 风险

- **待调参数清单**见下一步讨论；核心未定项：ZOO `{lr, zo-eps, zo-directions, local-steps, rounds}` 与 BP `{lr, bp-clip, local-steps, rounds}`。
- ZOO lr 标定以 δ 范数匹配为准，**未经收敛验证**；需 BP 收敛参照。
- 现无 `consensus_mask_*`（catv off），Step4 CATV on 须带 `--predictor/--pruned-config`。
- GPU0/1/3 被占，4 卡并发 Step4 暂不可行。
- smell-v2 栈与 v3 pinned 版本漂移（peft 0.19 / datasets 5.0 / torch 2.1.2）仍未收敛。
