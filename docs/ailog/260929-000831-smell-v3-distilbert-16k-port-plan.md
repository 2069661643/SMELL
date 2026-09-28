# Ailog 260929-000831 — SMELL-v3: DistilBERT 66M × 16k port 计划（仿 FwdLLM 换小模型基座）

> 背景：FwdLLM 原教旨审计（`docs/ailog/260928-231421-...fwdllm-original-route-audit.md`）与 k4/a03 负结果后，
> 决定保留 16k 长上下文叙事，仿照 FwdLLM 把基座换成更小参数的 BERT 系模型以降低单步成本、放大 ZO 方向预算。
> 本 session 只做调研与计划：用户已拍板基座 **DistilBERT-base-uncased（66M）**、**接受评测协议改变**（不再与 OPT PPL 直接可比）。
> 本 ailog 随新分支 `exp/distilbert-16k` 推送；尚未写任何代码。

## 一、用户决策（锁定）

| # | 决策 | 说明 |
|---|---|---|
| D1 | **保留 16k 长上下文叙事** | sparsity + predictor + CATV + 联邦/ ZO 栈全部保留，不做短序列降级 |
| D2 | **基座 = DistilBERT-base-uncased 66M** | 对比候选见 §2；BERT-base 110M / ALBERT 12M / OPT-125M 均被否 |
| D3 | **评测协议可改** | Discovery 从生成式 PPL 改为 masked-span 打分（accuracy 为主指标） |

## 二、调研结论（代码实证）

### 2.1 Jenga 零 BERT 支持（核心约束）
- `third_party/Jenga/src/jenga/utils/config_utils.py` 仅 OPT/LLaMA（Mistral 只有 config 桩）；`models/` 无 encoder。
- 剪枝语义（`modeling_opt.py:294-321`）：64-token 块、`config.sparse`=保留率 0.4、前半层不剪/后半剪/末层豁免；
  Q/K 同时收缩到 token 子集；`is_causal=True` 写死（`modeling_opt.py:124`）。
- 已知上游坑（移植时要处理）：`flash_block.py` HEAD_DIM=128 硬编码（`train_predictor.py:60-88` 已用 torch 版绕过）、
  `output[0].scatter_`/`cuda:0` batch=1 硬编码、predictor 要求 seq_len%64==0。
- 结论：BERT port = COPY 进 `src/`（不改 third_party），模型副本 + 剪枝端口 + predictor trainer + 评测，共 4 块新代码。

### 2.2 参数量与成本（qkvo LoRA；位置表 16k）

| 模型 | 类型/总参 | L×H | 位置上限 | r=1 | r=4 | r=8 | ZO 4层块 r=1 | 16k 位置表 | 16k attn 成本 |
|---|---|---|---|---|---|---|---|---|---|
| **DistilBERT**（选定） | enc 66M | 6×768 | 512 | 36.9k | 147k | 295k | 24.6k | 12.6M | **0.19×** |
| BERT-base | enc 110M | 12×768 | 512 | 73.7k | 295k | 590k | 24.6k | 12.6M | 0.38× |
| ALBERT-base-v2 | enc 12M | 12 层共享 1 物理层 | 512 | 6.1k | 24.6k | 49.2k | ⚠️ 轮转失效 | 2.1M | 0.38× |
| OPT-125M | dec 125M | 12×768 | 2048 | 73.7k | 295k | 590k | 24.6k | 12.6M | 0.38× |
| OPT-350M（现状） | dec 331M | 24×1024 | 2048 | 197k | 786k | 1.57M | 32.8k | 16.8M | 1.0×（实测 2.9s/前向） |

- FwdLLM 对照：DistilBERT output-adapter ≈0.45M 可训练 ≈ 本方案 LoRA r=8（0.59M）。
- **预期收益**：成本 0.19×、全层 LoRA r=1 的 `d_eff` 也 0.19×（36.9k vs 196.6k）；
  同墙钟 D 约 +5.3×、`cos≈0.5√(M/d)` 潜力 ~2.6–5×（乐观上界；k4a03 的协议/门控问题仍需另治）。

### 2.3 组件可行性

| 组件 | 结论 | 依据 |
|---|---|---|
| 16k pos-emb | 可行，需新写 encoder 版 + 重训；风险高于 OPT | `position_embed.py` 为 OPT 专用（offset=2、`model.decoder`）；BERT 512→16k=32×（OPT 8×） |
| sparsity | 需 port `OptSdpaPruneAttention:717-868`；双向无 causal、batch-safe | Jenga 无 encoder；6 层只剩 3 层可剪（收益 ~0.75× vs OPT 0.58×） |
| predictor | MLP 架构无关、形状兼容（head_dim=64）；原生 `block_attn_pool` 无 causal **恰是双向正确目标** | `predictor.py:282-351`；`train_predictor.py` 的 causal 修复仅 OPT 需要 |
| LoRA | PEFT 可用；target 名与 TaskType 需改 | `lora.py:14-22` 写死 `CAUSAL_LM`+`q_proj...`；DistilBERT 为 `q_lin/k_lin/v_lin/out_lin` |
| 数据 | 必须用 DistilBERT tokenizer 重建全部 npy | `build_discovery_16k.py:23` 写死 OPT tokenizer；partition 行索引可复用 |
| 评测 | Discovery 为 **174 类连接词**且标签多 token ⇒ 改 span-mask PLL 打分 | `dataset_v3/discovery_16k/a01/meta.json` label_names；首 token 分类有 ~20 个 "in_/by_/for_" 冲突 |

## 三、执行计划（里程碑 + 验收门）

| 里程碑 | 内容 | 验收 / Gate | 机器 |
|---|---|---|---|
| **M0 预检** | 下载 `distilbert-base-uncased`；统计 174 类 token 序列/首 token 冲突；定剪枝层调度 | 统计 + 决策 ailog | WSL |
| **M1 数据重建** | build/tokenizer CLI 化（默认不变）+ [CLS]/[SEP] 选项；重建 a01 + warmup；更新 manifest sha256 | `check_partition` PASSED | AutoDL |
| **M2 位置适配** | `src/models/position_embed_bert.py` + `longctx_adapt --arch distilbert`（MLM on warmup 512×16k, pos_only 500 step） | **Go/No-Go**：16k MLM loss 逼近 2k 且 2k 无回归；失败备选 sinusoidal 重初始化/ALiBi | AutoDL |
| **M3 BP 基线+评测** | `src/eval/mlm.py`（span-mask PLL 174 类）；dense vs sparse、LoRA r=1/4/8、全层 k-scan | 稀疏精度不回退；口径对齐现 BP k-scan | AutoDL |
| **M4 稀疏+predictor** | `src/models/modeling_distilbert_smell.py`（SDPA + prune，去 causal、batch-safe）；`src/train/train_predictor_bert.py` | predictor 加载后 16k 精度/吞吐达标；36 张量 | AutoDL |
| **M5 Fed+ZO** | `run_fed --arch` 分派（工厂/位置/LoRA/zoo 正则/诊断），补 SDPA 剪枝的 CATV 投票 | TD-2' cos 诊断 → 门控 → ZO 长跑；同墙钟 D×5 兑现 | amax GPU2 |

关键 touchpoints：`run_fed.py:21-22,366-431,437-455,231-232,492-502`；`zoo.py:56-82`；`lora.py:14-22`；
`train_predictor.py:50-88,182-205`；`build_discovery_16k.py:23`；`build_warmup_16k.py:24`；`ppl.py:19`。

## 四、风险与默认决策

- **M2 是唯一 Go/No-Go**：512→16k 外推若 MLM loss 不降，先试插值/dup 缩放与 sinusoidal 重初始化，再考虑 ALiBi（改动大）。
- 6 层剪枝面窄：默认沿用 Jenga 规则（0-1 不剪 / 2-4 剪 / 5 豁免），M3 若收益 <10% 再议自定义调度。
- 协议变化：accuracy 与 OPT G-PPL 不可比 ⇒ 论文需保留 dense 对照与协议说明。
- 现存缺口：CATV 投票只在 `OptFlashAttention2`（`modeling_opt_smell.py:321-329`），SDPA 剪枝路径无投票 → M5 一并补。
- 默认：先 a01、a03 后置；OPT-350M 结果留作附录对照。

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 00:08 | 新增 | `docs/ailog/260929-000831-...distilbert-16k-port-plan.md` | 本计划（决策 + 调研 + 里程碑） |
| 00:08 | 新增分支 | git `exp/distilbert-16k`（基 `90b1fe3`） | 承载 DistilBERT port 工作；已推送 origin |
| （00:08 同时纳入上一 session 未提交的 FwdLLM 审计） | `docs/ailog/260928-231421-...fwdllm-original-route-audit.md` | 原教旨路线审计结论 |

## 待办

- [ ] M0：DistilBERT 下载 + 标签 tokenization 统计 + 剪枝调度决策
- [ ] M1：`--tokenizer`/特殊 token 参数化，重建 a01 + warmup，check_partition
- [ ] M2：encoder 位置扩展 + MLM 适配，过 Go/No-Go
- [ ] M3：span-mask 评测 + BP 基线（dense/sparse、LoRA k-scan）
- [ ] M4：DistilBERT 稀疏注意力 + predictor 训练器
- [ ] M5：run_fed `--arch` + CATV 投票补洞 + ZO cos 诊断与长跑
