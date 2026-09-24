# Ailog 260924-112549 — Phase 3: CATV（Consensus Anchor Tokens Vote）实装 + GPU 冒烟

## 概述

按 260923-210703 计划与作者确认的 CATV 设计（论文 `04-design-v1.tex`），把占位的 `CATVSelector` 换成可运行实现：客户端在每次 forward（含 ZOO 中心差分两次探测）收集 Jenga predictor 原始块分数并取均值上传；服务器逐层求和、排序，对 top/bottom `k=floor(r*N)` 块写 `+inf/-inf` 锚掩码，下一轮在 top-k 前注入原始分数。硬约束 `r < s` 与 `s + r <= 1` 在 CLI 与掩码函数两处校验。GPU mini（16k，2 client，2 轮）冒烟通过，round 1 指标含 `anchor_in=anchor_out=51`、`vote_bytes=12288`、`ir_mean/ir_min`，掩码文件按轮落盘。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 11:10 | 重写 | `src/models/token_selector.py` | L1-251 | `CATVSelector`（mask/set_mask/apply）；新增 `compute_consensus_mask`（r<s、s+r<=1、k=floor(r*N)、stable argsort tie-break）与 `mask_intersection_rate`；含 16 项 CPU 自测 |
| 11:12 | 修改 | `src/models/modeling_opt_smell.py` | `OptFlashAttention2.forward` L300-311 | topk 前上报 `vote_callback(layer_idx, sum_q.detach())`；按 `config.consensus_mask` 注入 ±inf 掩码；4 个 SMELL 3 标记 |
| 11:14 | 修改 | `src/fed/serial_fedavg.py` | L14-62, L86-138, L175-186 | `resolve_model_config`、`_VoteAccumulator`（逐层 batch 总和/计数→CPU fp32 均值，仅 sparse 层 11..22）；`ClientRunner(collect_votes=)` 安装/移除回调并返回 `votes`；`ServerAggregator.accumulate_votes` |
| 11:16 | 修改 | `src/fed/run_fed.py` | L23-41, L158-181, L201-208, L242-356 | 模型换成 `src.models.modeling_opt_smell`；新增 `--catv-r/--catv-normalize`；移除 `--catv on` 报错；轮循环注入上轮掩码、收集投票、算掩码（存 `consensus_mask_round{r}.pt`）并记录 `catv_r/anchor_in/anchor_out/vote_bytes/ir_mean/ir_min` |

新增（temp，gitignore）：`temp/run_catv_smoke_bg.sh`、`temp/catv_cpu_checks.py`、`temp/catv_mask_check.py`、`temp/catv_wiring_check.py`。

## 验证结果

**a. py_compile**：4 个修改文件全过（后台 driver 记录 `OK py_compile (rc=0)`）。

**b. `python src/models/token_selector.py`**：16/16 PASS + `PASS`（mask shape/±inf 计数、tie-break `+inf={0,1,2}/-inf={7,8,9}`、`r>=s` 与 `s+r>1` 抛 ValueError、IR overlap/partial/disjoint、apply 带/不带掩码）。

**c. CPU 约束检查**（`temp/catv_cpu_checks.py`）：
```
PASS r=0.4 s=0.4 raised ValueError: CATV requires anchor ratio r < sparse budget s
PASS r=0.5 s=0.4 raised ValueError: ...
PASS r=0.2 s=0.9 raised ValueError: CATV requires s + r <= 1 ...
[PASS] accumulate_votes sum
IR overlap=1.0 disjoint=0.0
```

**d. GPU CATV on 冒烟**（16k mini a01、2 client、2 轮、local_steps=1、zo_dirs=2、sparse=0.4、r=0.2；RTX 5060 8GB，走 sysmem spill）：
```
[fed] catv round 0 mask=consensus_mask_round0.pt anchor_in=51 anchor_out=51 vote_bytes=12288 ir_mean=0.9158 ir_min=0.8235
[fed] round 0/1 loss=7.578125 delta_norm=5.5213e+04 time=9.9s
[fed] catv round 1 mask=consensus_mask_round1.pt anchor_in=51 anchor_out=51 vote_bytes=12288 ir_mean=1.0000 ir_min=1.0000
[fed] round 1/1 loss=nan delta_norm=nan time=7.3s
```
round 0 无掩码（`consensus_mask=None` 初始化，round 0 结束后才生成 `consensus_mask_round0.pt`，round 1 使用）。掩码校验：12 层（11..22）、每层 float32 `[256]`、51 `+inf`/51 `-inf`/154 `0`、dtype fp32。

**e. CATV off 冒烟**：1 轮正常（`loss=7.578125`、`round_seconds=8.3`），metrics 无任何 `catv_*` 字段。

**f. 配置共享连线检查**（`temp/catv_wiring_check.py`，防 `from_pretrained` deepcopy 静默失效）：`base_config is layer11/23.config`、`resolve_model_config(peft) is base_config`、LoRA 包装后掩码仍可见——5/5 PASS。

## 设计决策 / 发现

- **`base_config = model.config` 必须在 `from_pretrained` 之后取**：transformers 4.45 `from_pretrained` 对传入 config 做 `deepcopy`（modeling_utils.py:3879），传参对象与模型内部不是同一实例；若挂错对象掩码会静默失效。
- **客户端投票只收 sparse 层**：模型对 layer 0..22 都调用回调，`_VoteAccumulator` 只记 `[num_hidden_layers//2-1, num_hidden_layers-2]`（OPT-350M 为 11..22），与 `vote_bytes=layers*N*4` 口径一致。
- **tie-break**：`torch.argsort(descending=True, stable=True)` 等价于按 `(-score, index)`，top/bottom 同时确定性。
- **IR 口径**：`M_local` = 客户端自身 raw 投票的 top-`floor(s*N)`（s=0.4→102），`M_central` = 新掩码 `+inf` 集（r*N=51）；对全部 client×layer 取 `ir_mean/ir_min`。
- **round 1 NaN 为既有 ZOO 步长问题**：2 轮 `--catv off` 对照同样在 round 1 NaN（`delta_norm≈5.5e4`），与 260924-103741 记录的「lr/eps/directions 需重调」一致，非 CATV 引入。

## 待办 / 风险

- 正式实验前重调 ZOO 步长（lr/eps/directions），否则第 2 轮即发散 NaN。
- `--catv-normalize sum` 仅按 legacy v2 语义实现，尚未做 on/off 对照。
- 云端 4 并发消融脚本未编写；CATV 质量结论需预训练 predictor 或先解决位置适配（P2）。
