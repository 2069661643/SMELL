# Ailog 260928-231421 — Phase N: FwdLLM 原教旨路线代码审计（k4/a03 负结果后的路线评估）

> 背景：TD-3 的 k=4 rotate sparsity ZOO（a01 及 a03 变体）实测无收益；用户要求只读审计
> `third_party/FwdLLM`（@36ecdbc）原版实现 + 论文 arXiv:2308.13894v2，判断「照原教旨跑」能否救 ZOO。
> 本 session 不改代码；对照物为 `src/train/zoo.py`、`src/fed/run_fed.py`、TD-2'/TD-3 口径
> （`scripts/run_k4_td2_td3.sh`）。k4a03 的云端日志不在本 checkout，未能核对其失败模式。

## 一、原版路线到底是什么（代码实证）

| 组件 | 实现口径 | 位置 |
|---|---|---|
| 估计器 | 中心差分，**h=0.01 硬编码**；v~N(0,I)，不归一化、无 dim 补偿 | `forward_training/utils/fwdgrad_utils.py:67-77` |
| 客户端计算 | 每 batch 1 个方向；`grad += jvp·v` 逐 batch 累加，**不做参数更新** | `forward_training/tc_transformer_trainer_distribute.py:127-141` |
| PEFT | 只训 Adapter（仅 output adapter，reduction 16）/ BitFit / LoRA；其余冻结 | `experiments/distributed/transformer_exps/initializer.py:93-106` |
| 聚合/更新 | FedSgd：client 发梯度，**server 每 mini-batch 更新** `θ -= lr·mean_c(g_c)`；`lr=0.01×ratio`（warmup_ratio=0，线性衰减到 0） | `FedML/fedml_api/distributed/fedsgd/FedSgdAggregator.py:76-139` |
| 方差门控 | 单层方差 > 阈值（distilbert 0.1 / bert 0.2）→ **回滚本轮更新** + 暂存 cached_v + 让 client 继续算更多 v | `FedSgdAggregator.py:96-133`；`FedSgdClientManager.py:106-109` |
| 判别采样 | 候选 `randn(v_num·10, shape)` 按与上一轮聚合梯度的 cos 排序，取 top `v_num`；`old_grad` 来自 `grad_aggregete(grad_pool)` | `forward_training/...py:83-104`、`155-159` |
| 路由 | `FedFwd` → `FedML_FedSgd_distributed`（不是 FedAvg，也不是 `fedsgd_beifen`） | `initializer.py:43-44`；`fedavg_main_tc.py:136-138` |
| 实验档位 | DistilBERT 66M / seq 64(AGNEWS) / bs8 / 100 clients/轮 / `comm_round=3000` / 每轮按 batch 数次 server 微步 | `run_tc_exps/run_text_classification.sh:113-141` |

**关键解读**：原教旨的每步 cos 也很低。用论文自己的 global-PS 口径（100 clients × 最多 50
perturbations/client，adapter 可训练维 d≈0.45M）代入 `cos≈0.5√(N/d)`：**约 0.0075–0.05/步**。
与 TD-2' 的 k4 cos=0.0885 同量级甚至更低。

## 二、它 work 的原因是协议，不是估计器

- 用**海量低 cos 微步 + 方差门控 + 上一轮方向对齐 + lr 衰减**把噪声平均掉；论文 §3.3 明确
  99% 相对精度需要 global-PS=50/client（≈5000 方向/步），80% 只需 3/client。
- 生效边界很清晰：≤340M encoder、seq 64–256、PEFT 小参数子空间、100–1000 clients。论文卖的是
  time-to-accuracy / 显存（14.6×），**不是精度优势**：LLaMA-7B INT4 AGNEWS 85.8 vs 集中式 BP
  FP16 89.7（-3.9 pt，Table 7）；sub-1B 上也只是追平 BP-PEFT 的目标精度（AGNEWS 0.88）。
- 本 artifact **无法复现 LLaMA 结论**：全 repo 无 llama/gptq 代码与依赖，模型工厂仅 BERT 系
  （bert/distilbert/roberta/albert/deberta）；论文的 LLaMA 走 llama.cpp + AutoGPTQ，未开源在此。

## 三、与 SMELL-v3 TD-3 k4 的对照

| 维度 | FwdLLM 原版 | SMELL k4（TD-3） |
|---|---|---|
| 序列 / 模型 | 64 / 66M | 16384（×256）/ OPT-350M |
| 可训练子空间 | adapter≈0.45M / bitfit | k=4 LoRA ≈ 32.7k 维（更小） |
| 更新粒度 | FedSgd，**每 mini-batch 一次 server 更新** | 每轮 1 次 Δθ 加权平均 |
| 更新次数 | 每轮 ~15 次（AGNEWS），数十~数千轮 | 30 轮 × local_steps 2 = 60 次 |
| 噪声控制 | 方差门控 + 回滚 + 判别采样 | 无 |
| lr 口径 | 0.01 固定 + 线性衰减 | 校准到 BP δ=0.38（大且恒定） |
| 每步 cos | ~0.0075（1 dir/client）→0.05（门控攒到 50） | 0.0885（TD-2' c=30, D=22, L2） |
| 单方向成本 | 2×64 token 前向 | 2×16384 token 前向 |

**结论性对照**：SMELL 的估计器并不弱（归一化 + dim 补偿 = 无偏；fp32 已解决 bf16 量化），
k4 的每步 cos 甚至**高于**原版的工作点；差距全在「更新次数 × 噪声平均 × 方差门控 × 调度」的协议层。

## 四、结论（回答：原教旨能否救 k4/a03）

1. **不能。** 原教旨的有效工作点本身就是 cos≈0.01–0.09 的高噪声区间；它靠的是每轮多次
   server 微步 + 方差不达标就不提交 + 方向复用。16k 下每个方向的前向成本是 64-token 的 256×，
   且 client 串行（本 harness），无法复刻这个步数。把 h=0.01、lr=0.01、per-batch FedSgd
   照搬到 16k 在算力上直接不可行。
2. **k4/a03 的差不是「实现不够原教旨」**，而是优化预算与协议问题：cos=0.0885 配 δ=0.38 的
   大恒定步长（≈91% 是噪声），无门控/回滚/衰减；比 FwdLLM 单步还「激进」。
3. **值得移植的只有 3 个廉价协议件**（按性价比）：
   - **方差门控 + 回滚**（server 端几十行；直接针对 lr 放大 11× 的发散风险）；
   - **判别性扰动采样**（用上一轮聚合方向对齐；等效把 D 提高 `(cos'/cos)²` 倍，可能把所需
     directions 降一个量级，是唯一可能改变 cos 量级的机制）；
   - **lr warmup→decay + 微步化**（受 16k 成本限制，只能有限做）。
4. **主线建议不变**：BP-based FL 承担精度叙事；ZOO 降级为 BP-free/成本叙事（原论文自身也不
   主张精度优势，LLaMA 差 4 pt）。若仍要投资 ZOO，先做 3 的单变量消融再谈长跑。

## 五、审计发现（不要照抄的坑）

- `calculate_jvp` 用 `autocast()`（fp16 混合精度）且 h=0.01 固定；论文口径是 N+1 次前向
  （复用 f(θ)），artifact 是 **2N 次**（更贵）。论文的 loss 量化通信在本 repo 未用于 FedFwd
  （`use_quantize=False`）。
- server 更新 `next(old_param).detach().to("cpu").sub_(...)`：server 模型若在 CUDA，`.to("cpu")`
  产生副本，更新**静默丢失**（artifact 在 CPU server 上跑所以未暴露）。
- `calculate_cos_sim` 最后 `return similarity`（最后一批）而非 `result`：候选 >1000 时排序只对
  最后一批生效（本实验 10×batches<1000 未触发）。
- `grad_aggregete` 原地改 `grad_list[0]`；`cached_v`/`model_dict` 多处 deepcopy+in-place，脆弱。
- `--forward_mode` 在分布式 FedFwd 路径未被使用（vestigial）；`--server_lr` 对 FedFwd 无效。
- 回滚依赖 `set_global_model_params(origin_param)` 与参数顺序一致，有 v2 M1 式**静默失效**风险。

## 待办

- [ ] 若采纳方差门控/判别采样：先在 2k fp32+sdpa_prune smoke 上做单变量消融（只加 var gate；
        只加 directed sampling），与 TD-2' 的 cos/delta_norm 对照，再决定是否上 16k。
- [ ] 补齐 k4a03 的 `metrics.jsonl` 后核对失败模式（发散/平台/无信号），确认 lr 校准与
        30-client 实跑的 delta_norm 是否一致。
- [ ] （可选）在 a01 上测 global-PS 曲线（cos vs directions/client），给出「cos≥0.15 需
        多少小时/轮」的实测数，进论文 cost 讨论。
