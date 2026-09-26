# Ailog 260926-1849xx — SMELL-v3: TB 完成（fp32 SDPA 16k 前向）+ TC 启动

## 概述

**TB 完成**：为自定义 OPT 新增 **SDPA 注意力路径**（`F.scaled_dot_product_attention`，mem-efficient，支持 **fp32**、O(seq)），**16k fp32 前向跑通（峰值 19.5 GiB，无 OOM）**；`ΔL` 与真方向导数同号同量级，layer0 cos 明显高于 bf16。**TC（fp32 cos 网格）已启动**（ETA ~5.8h）。

## 变更列表

| 时间 | 操作 | 文件 | 位置/影响 |
|---|---|---|---|
| 18:43 | 新增 | `src/models/modeling_opt_smell.py` | `OptSdpaAttention`（L445）：改造 `OPTAttention`，用 `F.scaled_dot_product_attention(is_causal=True)`；注册 `"sdpa"`（L511）；`_supports_sdpa=True`(L641)、`_use_sdpa`(L758)、2d-mask 分支免构造 O(seq²) 4d mask（L857） |
| 18:43 | 修改(临时) | `temp/diag_bf16_swallow.py`、`temp/diag_cos_grid.py` | 加 `--attn {flash,eager,sdpa}`；cos_grid 加 `--dtype {bf16,fp32}`（gitignore） |

## TB smoke（GPU2，16k fp32 SDPA，A1 r=1 adapter）

- **16k fp32 SDPA 无 OOM**：峰值 allocated **19.53 GiB**；单次 forward **2.89 s**，BP fwd+bwd 3.89 s。
- `loss0=2.879734`，`noise_floor=0`；`dL_meas vs dL_pred`：ε=1e-3 `+2.86e-6 vs +2.90e-6`、1e-2 `-4.77e-6 vs -6.10e-6`、1e-1 `-3.36e-4 vs -3.39e-4` ⇒ **ε≥1e-3 同号同量级**（ε=1e-4 反号，低于 fp32 分辨率）。
- ZOO cos（单 client）：D=8 **0.0314**、D=32 **0.0674**（bf16 同条件 ~0.013）。
- TC-lite（3 client, layer0）：D=8 c1 0.0383/c3 0.0405；D=32 c1 0.0489/c3 0.0474（**~3–4× bf16**）。

## TC（进行中）
- 配置：`--dtype fp32 --attn sdpa --blocks layer0 --n-clients 30 --ls 1,2 --ds 8,32 --cs 1,5,10,20,30`，16k，GPU2；输出 `temp/diag_cos_fp32.json`，日志 `temp/logs/diag_cos_fp32.log`。
- **ETA ~5.8h**（每 client ~11.6min）。
- 判据：fp32 下 cos 是否显著（目标 ≥0.1）。

## 风险 / 未决
- SDPA 不返回注意力权重（`output_attentions=True` → None）；`is_causal` 假定 q_len==k_len、无 KV-cache 解码路径；train 模式 dropout 分支未测（smoke 为 eval）。
- **稀疏仍待 TD 接入**（td-1 top-k 掩码 → SDPA；td-2 稀疏下复验 cos；td-3 生产）。
- fp32 前向 ~2.9s（bf16 FA ~0.33s，~9×）⇒ 生产 ZOO 成本高，需评估。

## 环境
host `amax`（A40×4）。GPU2 跑 TC（36.6GB/100%，含邻居）。`smell-v2` 栈。
