# SMELL-v2 — 上游 FwdLLM 实证核查 + ZO / head-only 文献版图

## 概述

起因：讨论"仅训 lm_head（D3）"这一路线的出处与可行性。原先的工作假设是"仅训头沿用自 FwdLLM"，需要先把它查实，再据此判断 (a) D3 是否是 FwdLLM 的忠实复刻、(b) 有没有相关文献已占位、(c) 文献对我们当前"37.0 墙"的解释力。

本文档只做**核查与归档**（不是实验报告），共三块：① 上游代码实证 FwdLLM 到底训练什么；② 原论文口径；③ 文献版图与对 SMELL 的贡献/挑战/冲突/联系。本轮唯一的代码改动是给 BP sweep 脚本加 `LR_R` 旋钮（见变更表）。

**证据等级标注**（全文遵守）：
- **A** = 本仓库代码 / 原文 / 本机日志实证
- **B** = 检索摘要（未取到原文校验）
- **C** = 本文档的估算或推导（我方计算，非文献结论）

### 三条主要结论

1. **"仅训头"不是 FwdLLM 的做法（A）**。上游 FwdLLM 的可训集**永远在骨干内部**：DistilBERT 用 Houlsby output adapter（逐层插入）、LLaMA 用 LoRA、其它模型用 BitFit（bias+classifier）。D3 是 SMELL 自创，动机是 `cos=√(M/d)` 的 SNR 论证（`docs/summary.md:78`）。精神上确实继承自 FwdLLM（"代价随可训参数数走 ⇒ 尽量缩小可训集"），但配置不是。
2. **FwdLLM 发表的可用系统，方向质量大约只有 cos≈0.08–0.11（C）**，却能在 AGNEWS 上达到 87.0%（A/B，论文 Table 7）。这**与我们此前"cos 太低 ⇒ 学不动"的结论存在张力**：需要限定为"在我们的 3-client、36K 可训、causal-LM PPL 口径下"。差异候选见 §2.5。
3. **head-only 有先例，而且证据方向对 D3 不利（B）**：ZO 下"单层主导"成立，但主导层是**残差流早期层**而非输出头（Dominant-Layer ZO）；头-only 的 PPL 天花板明显低于"头+骨干"（Split-FG 的冻结骨干对照 668 vs 387）；且头层恰恰是前向训练最难的地方（FineFed）。

---

## 1. 上游 FwdLLM 实际训练什么（A：代码实证）

| 维度 | FwdLLM 上游做法 | 证据 |
|---|---|---|
| 任务/模型 | DistilBERT/BERT **序列分类**（AGNEWS），非 causal LM | `FwdLLM/experiments/distributed/transformer_exps/run_tc_exps/run_text_classification.sh:9-10` |
| **主可训集** | **Houlsby output adapter**（`output_adapter=True`、`mh_adapter=False`、`reduction_factor=64`）；`train_adapter()` 冻结 base、只解冻 adapter | `experiments/centralized/transformer_exps/initializer.py:72-74`；adapter config `experiments/distributed/transformer_exps/test.ipynb:1501-1504` |
| 备选可训集 | **bitfit** = 所有 bias + classifier；或 `--freeze_layers` 只训未冻结层 | `experiments/distributed/.../initializer.py:280-283`；`training/tc_transformer_trainer.py:284-305` |
| 任务头 | **不新建**，用 HF 自带 `classifier`；`pre_classifier` 被替换成空 `nn.Sequential` | `training/tc_transformer_trainer.py:41-42` |
| 扰动 v | 形状 = **全部参数**，非可训位置填 0 ⇒ 实际扰动 = 全部 `requires_grad`（adapter 全模块） | `training/tc_transformer_trainer.py:73,123-125` |
| 差分式 | `jvp = (f(θ+hv) − f(θ))/(2h)` —— **单边差分 / 2h** | `training/tc_transformer_trainer.py:124-125` |
| 更新 | `p.sub_(lr · Σ_jvp·v / v_num)` —— 朴素梯度步 | `training/tc_transformer_trainer.py:136-147` |
| client/server 同步 | 交换**完整 `state_dict()`**（含冻结权重） | `training/fed_trainer_transformer.py:15-19` |

⇒ `trainable_param_names`、参数名翻译表（`_translate_lora_param_names`）、server 侧 `LoRAOnlyModel`、聚合里的全局 L2 归一化 + `clamp(-1,1)`（`FedSgdAggregator.py:231,243-259`）**全部是 SMELL 新增**，上游没有 ⇒ 这些改动必须写进论文的"我们做了什么"而不是"沿用 FwdLLM"。

### 1.1 与上游的四处偏离（复刻时必须知道）

| 项 | 上游 | SMELL 现状 | 影响 |
|---|---|---|---|
| 差分式 | 单边 `(f(θ+hv) − f(θ))/(2h)` | 双边中心差分 | 我方的估计器是无偏的；上游那个 /2h 相当于把梯度整体缩放 1/2（被 lr 吸收，但**方向方差不同**），不可直接搬数字 |
| 聚合 | 平均后朴素梯度步，梯度尺度自由 | 全局 L2 归一化 + clamp ⇒ **‖Δθ‖ ≡ lr** | 我方步长是"参数空间绝对长度"，与数据/样本数无关 ⇒ 扫 lr 就是在扫位移 |
| 同步 | 完整 `state_dict` | 仅可训 LoRA 张量（server 侧 LoRAOnlyModel） | 通信量低 3-4 个数量级，但引入"参数枚举顺序/名字翻译"这一独立失效面 |
| 可训集 | adapter / bitfit / 未冻结层（骨干内部） | D3 = lm_head LoRA（骨干外） | 这是我们能"缓存 h"的前提，也是与文献最不一致的一点（见 §5） |

---

## 2. 原论文口径（A/B：arXiv 2308.13894 原文）

### 2.1 训练什么

- PEFT 按模型选（Table 4）：ALBERT/BERT/RoBERTa = **BitFit**、DistilBERT = **Adapter**、**LLaMA-7B = LoRA**（正文**未给** r / target_modules / 参数量）。
- 选择由一个**离线 PEFT profiler** 决定，判据是"**前向梯度与 BP 梯度的相似度**"。
  ⇒ **"用 cos(FG, BP) 选 PEFT"是人家论文的原始方法论**；我们的 C5（`test_exact_head_grad.py`）与 `estimate_lr.py` 的 cos 测量属于同一思路，不能当首创。
- 第一性观察："BP-free 的复杂度随**可训参数数**而非总参数数增长"，并称"所需扰动数随参数规模指数增长"（无公式）。
  ⇒ 这是我们 D3/A1/C1 方向（缩小 d）的**理论依据来源**，可以正面引用。

### 2.2 可训练参数量估算（C：我方按架构手算，论文未给）

| 配置 | 计算式 | 可训参数 | 占模型 |
|---|---|---|---|
| DistilBERT + adapter(r=64) | b=768/64=12；每层 (768·12+12)+(12·768+768)=19,212；×6 层 | **≈115K** | 0.17%（66M） |
| LLaMA-7B LoRA r=8, q,v | 8·8192×2×32 | 4.19M | 0.06% |
| LLaMA-7B LoRA r=8, q,k,v,o | 8·8192×4×32 | 8.39M | 0.12% |
| LLaMA-7B LoRA r=16, q,k,v,o | 16·8192×4×32 | 16.8M | 0.24% |
| **SMELL 原 Phase 1**（LoRA r=8 qkvo） | 同上 | **8.4M** | 0.12% ← **与 FwdLLM 同量级** |
| **SMELL D3**（lm_head LoRA r=1） | 4096 + 32000 | **36,096** | **0.0005%** ← 小 100–400× |

⇒ 结论：**我们最早那版 r=8 qkvo 才是 FwdLLM 的忠实复刻**；D3 是把"缩小可训集"推到极限的产物。

### 2.3 扰动、调度与采样（A/B）

- 每 client 每轮扰动数**很少**：早期 **3 个**即达 80% 相对精度，**50 个**才到 99%（算力差 16.7×）。
- **client 数 100（部分实验 1000）**；策略"先加设备，再加每设备扰动数"；复用 f(x) 把前向从 `2N` 降到 `N+1`。
- **判别式扰动采样**：服务器预生成随机种子，按"与上一轮 FG 的 cos"过滤低价值扰动；实测 **>60% 的扰动 cos<0.03**、只贡献 29.6% 的最终 FG；**20% 采样率最优**（2.33×）。
- **方差控制**：方差低于阈值才聚合；阈值经验范围 **0.1–0.5**，"唯一要调的参数"；实测梯度方差 0.078 → 1.182（15.2×）⇒ **越收敛方差越大**。
- 论文未给 h 的数值；v 的分布前后不一致（公式 N(0,1) vs §3.1 uniform）。

### 2.4 LLaMA-7B 结果（B：Table 7）

AGNEWS / Pixel 7 Pro：FP16 87.0%（240 轮，1.5h）；INT8 86.9%（0.8h）；INT4 手机 CPU 0.19h；INT4+模拟 NPU 85.8%（130 轮，0.07h）；量化用 GPTQ，**骨干 INT8/INT4 + LoRA 权重保持 FP32**。生成任务 SQuAD：312 轮达 83% F1，比集中式 BP 低 1.9%。

### 2.5 冲突：cos≈0.1 也能 work，我们需要限定自己的结论

按 `cos≈√(M/d)`（C：我方公式）反推 FwdLLM 的训练 regime：全局扰动数 `M = 1000 client × 50 = 5×10⁴`、`d≈4–8M` ⇒ **cos ≈ 0.08–0.11**。

而我们 D3 在 M=3072 时**实测** cos=0.2794（A：`log/estimate_lr_20260914.log:206`）。**也就是说我们已经比人家发表的可用系统方向质量好 3 倍**。

⇒ **"cos 太低所以学不动"这条结论必须加限定条件**（我们此前的表述见 `docs/ailog/260915-163135-...md:15,103`）。最可能的差异来源：

| 候选差异 | FwdLLM | SMELL | 为什么可能才是真因 |
|---|---|---|---|
| **client 数** | 100–1000 | **3** | 每个 client 的噪声独立 ⇒ 聚合方向质量约有 ×√K 的增益；1000 个 client 相当于把 cos 0.1 抬到接近 1。我们 3 个 client 吃不到这条红利 |
| **容量** | 4–8M | **36K** | 差 100–400× 可训自由度；我们的 BP 臂证明"即使精确梯度也只有 −17.8%"，指向容量/配置而非估计器 |
| **任务口径** | 分类（每样本 1 个预测） | causal-LM PPL（每样本 ~17 个 token 预测） | 分类的梯度是 pooled 表示上的单点监督，信号集中；PPL 的 17 个 shift 任务共享一个 rank-1 头，有效自由度极低 |
| 数据规模 | 见论文（AGNEWS 全量） | 4k 条 goemotions | 训练信号总量差很多 |

### 2.6 论文里含糊/缺失（引用时要小心）

h 无数值；v 分布自相矛盾；单边差分写成 /2h；LLaMA LoRA 配置未给；profiler 的相似度指标未给公式；只报"per client 扰动数"而全局 M 需要从 client 数推算。

---

## 3. 后续与相邻工作

### 3.1 FwdLLM+（B：IEEE TMC 2026, Vol.25 No.8, pp.12735–12749, DOI 10.1109/TMC.2026.3672496）

- **定位**：FwdLLM 的期刊扩展，针对"BP-free/ZO 理论收敛步数更多"这一个短板。
- **做法**：advanced zeroth-order optimization + **low-rank perturbation decomposition** 降低收敛步数。
- **报告**：4 个模型（110M–7B）× 8 数据集，最高 **151× 训练加速**、**93% 显存下降**，优于原版 FwdLLM。
- **对 SMELL**：**挑战**——"压缩扰动维度/低秩扰动"这条线已被原作者团队做成期刊版；我们的相关工作必须引用它，且不能把"降维"当首创。

### 3.2 P-RGE（B：arXiv 2409.15520，"Thinking Forward"）

- **做法**：冻结 W 与 LoRA-A，**只扰动/训练 LoRA-B**（r=16, alpha=32）；RGE 需 **2q 次前向**。
- **它的"并行"不是简单加 batch**（B，检索摘要）：三重组合 ——
  1. **loop 嵌套重排**：外层遍历数据 query、内层遍历 ±noise；
  2. **用数据 batch 换扰动数**：固定有效 batch `E=q·B=16`，q 变大就把 B 变小（q=4/B=4、q=16/B=1）⇒ **峰值激活显存不随 q 增长**；
  3. **per-copy 权重**：每个副本用它自己的（扰动后）权重 ⇒ 靠"复制 LoRA-B 副本 + dual-forwarding 模块"实现（等价 grouped GEMM）。
- **收益性质**：2q 行 backbone 仍要算 ⇒ 赚的是**利用率/显存形状**，不是 FLOP。实测内循环单独 1.79×、端到端最高 4.3×（vs MeZO Full）、1.9×（vs MeZO LoRA-FA）；Llama2-7B 峰值显存 0.99 GB（外循环）→ 1.97 GB（内循环），而 FO 训练 >30 GB。
- **理论**：给出 `Var(RGE) ≈ O(d/q)` ⇒ 与我们的 `cos≈√(M/d)` 相互印证（方差 ∝ d/M 与 cos ∝ √(M/d) 同源）。

### 3.3 我方推导（C）：秩结构让 M 基本免费 —— 供"缓存 h"方案引用

前提：可训集只有 lm_head 的 A(1,4096)、B(32000,1)。对扰动 i：

$$\text{logits}_i = \underbrace{h\,W_0^{\top}}_{\text{所有 } i \text{ 共享，每样本每轮只算一次}} + \underbrace{(hA_i^{\top})\,B_i^{\top}}_{\text{每扰动 } \sim 1.2\ \text{MFLOP}}$$

- 共享项：17 个有效位置 × 4096 × 32000 × 2 ≈ **4.5 GFLOP（一次性）**
- 每扰动项：17×32000×2 ≈ **1.2 MFLOP**
- 对照"每个扰动都算全宽 head"（4.5 GFLOP/扰动）⇒ **省约 4000×**
- M=36,096 全量（双边差分）≈ **90 GFLOP ⇒ 秒级**；而 backbone 侧只需每样本 1 次前向（30 条 ≈ 45 s，与 BP 臂同级）

⇒ 含义：缓存 h + 切片到有效位置 + factorized evaluation 三件套齐备时，**M 可以推 1–2 个数量级到 ~d 量级，cos 逼近 1，而边际成本≈0**。这条是**我方推导，不在任何检索到的论文里**（ZO-Act 的"激活低秩子空间"最接近，但它是把扰动限制在子空间，而不是利用参数化的秩结构把 FLOP 消掉）。

---

## 4. 文献版图：文本/LLM + ZO（问题 1）

**结论：不是一个空白，是一整片活跃领域。** 四类：

### 4.1 集中式 ZO for LLM

| 文献 | 出处 | 是什么（意义） | 对 SMELL 的贡献/挑战/冲突/联系 |
|---|---|---|---|
| MeZO | arXiv 2302.06677 系（奠基，B） | SPSA 两点估计，推理级显存 | **贡献**：BP-free 可行性的地基；**联系**：我们的 ZOO 就是它的联邦变体 |
| ZO Benchmark | arXiv 2402.11592（B） | 5 族 LLM 系统比较 + block-wise / ZO-FO 混合 / 梯度稀疏 | **贡献**：给出"ZO 在什么条件下接近 FO"的经验边界，可直接引作对照 |
| SubZero | Yu et al. 2024（B） | 随机低秩子空间扰动，降方差 | **挑战**：占据"低秩扰动"生态位 |
| ZO-Act | arXiv 2607.01125（B） | 激活引导的低秩子空间（basis 一次性算好并冻结，只优化系数矩阵）；**rank-1 往往最好**（RTE 87.0 vs rank32 83.0） | **挑战/冲突**：与我们"缓存 h + 秩结构"最接近；它已给出"rank-1 最好"的实证；**贡献**：可引作"限制扰动维度反而更好"的独立支持 |
| AGZO | ICML 2026（B） | 激活引导扰动方向 | **挑战**：同上，属最近邻 |
| Sparse MeZO / CurvZO / HELENE / KerZOO / LOZO / HiZOO / ZO-Muon | 见出处（B） | 稀疏扰动 / 曲率引导 / Hessian 分层 / kernel 降偏 / 低秩 / 分层 / Muon 化 | **贡献**：都可以作为"ZO 改进"的对照基线 |

### 4.2 联邦 ZO for LLM

| 文献 | 出处 | 是什么 | 对 SMELL |
|---|---|---|---|
| **FedKSeed** | ICML 2024（B） | 只传**有限种子集 + 标量梯度**，通信 <18KB/轮，全参数调优；K 取内在维度 10³–10⁴；含概率化种子采样（Pro） | **最大挑战**：通信口径我们打不过它（它 <18KB/轮）；**联系**："种子+标量"与我们的可训 LoRA 张量可组合 |
| FedZO | 相关基线（B） | 每轮多次局部更新 + 部分设备参与 | 对照 |
| **ZorBA** | INFOCOM 2026（B） | 异构 block 激活的 ZO 联邦微调 | 对照（设备异构口径） |
| Ferret | arXiv 2409.06277（B） | 共享随机性的一阶全参数联邦 | 对照（一阶） |
| **FineFed** | OpenReview（B） | forward-only 联邦 + **专治"头层更新落后"** | **挑战**：见 §5，直接质疑头层在 FG 下的可训性 |

### 4.3 参数稀疏 × ZO

| 文献 | 出处 | 是什么 | 对 SMELL |
|---|---|---|---|
| Guo et al. | ICLR 2025（B） | 迁移静态稀疏：0.1% 敏感参数 + 4-bit，Llama2-7B 在 **<8GB** 显存微调，胜过全参 ZO/ICL | **挑战**：与我方"降 d"目标同向且更强；**联系**：可组合（稀疏可训集 + 我们的 token 稀疏） |
| Sparse MeZO / CurvZO | 见出处（B） | 稀疏扰动 / 曲率引导预算 | 贡献：基线 |

### 4.4 token 级稀疏 × 长上下文

| 文献 | 出处 | 是什么 | 对 SMELL |
|---|---|---|---|
| **LeMo** | arXiv 2501.09767（B） | "Contextual Token Sparsity"：Token Elimination + Pattern Prediction + kernel 优化；显存 ↓1.93×、速度 ↑1.36× | **最大冲突**：它的 **Token Elimination** 与我们的 Jenga 半边几乎同名同义；但它**是一阶（BP）** ⇒ 只能做相关工作，不能当"我们的独特机制" |

### 4.5 版图结论：真正的空白

检索结论（B）原话：ZO/sparsity 与 token pruning/long-context **两条线基本分离**。

⇒ **SMELL 的差异化位置 = 「token 级稀疏 × BP-free/ZO × 联邦 × 设备侧」的交集**。单看任一项都有前人（LeMo 的 token 淘汰、ZO-Act 的低秩扰动、FedKSeed 的通信、Guo 的参数稀疏）；**组合起来目前是空的**。这解释了为什么项目定位是 Jenga ⊗ FwdLLM 而非单一创新。

---

## 5. head-only / 单层 ZO：先例与反证（问题 2）

**有先例，且证据方向对 D3 不利。** 三篇关键：

### 5.1 Dominant-Layer ZO（arXiv 2606.05516, 2026-06, B）

- **结论**：ZO 微调被**单个层**主导；只训该层可**匹配甚至超过**全模型 ZO（LLaMA2-7B、Qwen3-8B，9 个 benchmark），最高 **4.52×** 加速。
- **关键在选哪层**：主导层任务无关、模型相关，**位于残差流早期**（与"第一个激活离群层"一致），可用**仅推理**的激活分析在训练前识别。机制：高扰动敏感性 + 位置靠前 ⇒ 扰动效应沿后续层**累积放大**。
- **对 SMELL（冲突）**：这是对我们 260914 记录的"**D3 r=1 抓手太弱：即便沿精确梯度位移 1×‖θ‖，loss 也只降 6.9%**"的文献版对应结论。lm_head 是**最后一层**，扰动没有下游可放大 ⇒ ZO 语境下它是最差的选择之一。
- **注意边界**：该文的比较前提是"骨干可训、选哪一层训"；我们冻结骨干是联邦设备侧的动机 ⇒ **不能直接推翻 D3 的存在理由**，但"把头换成早期层"变成一个**用现有 BP 臂就能便宜验证**的实验。

### 5.2 Split-FG（"Backpropagation-Free Trunk Training via the Split Forward Gradients"，B）

- **做法**：在中间表示处切开网络，**输出头梯度算精确的**，只对骨干用 JVP 估计。⇒ 与我们"头可精确、可缓存"的推论同源。
- **对照数据**（16M GPT-2 / WikiText-103，val PPL）：

| 方案 | val PPL |
|---|---|
| BP | 150 |
| Split-FG（头精确 + 骨干 FG） | **387** |
| **冻结骨干 = 只训头** | **668** |
| 纯 FG（全参） | 2885 |

- **附加发现**：朴素 FG 训骨干甚至**不如冻结随机骨干**（Adam 会放大噪声坐标），必须配极小 lr 才反转。
- **对 SMELL（冲突）**：**"头-only"的天花板明显低于"也训骨干"**，与我们的 BP 臂（精确梯度、头-only、也只到 −17.8%）方向一致。

### 5.3 FineFed（forward-only 联邦，多类任务，B）

- **结论**：前向训练在**头层更新上显著落后于反向**——头未训练时预测接近随机，不同扰动方向的投影差别很小 ⇒ 方向引导不准、收敛慢；**类别越多越糟**。据此专门做 forward-only head tuning。
- **对 SMELL（冲突）**：我们的头是 **32000 类**的 lm_head ⇒ 正落在它描述的最坏情形里。

### 5.4 相邻的 forward-only 工作（B）

SharpZO（VLM prompt，两阶段 sharpness-aware）、FROST（逐层轮流训 attention 投影）、ZeroFlow、FFGAF-SNN —— 都属"前向训练可行但需要专门设计"的证据群。

### 5.5 对 D3 的三条含义

1. **"头-only"不是新叙事，而且文献倾向相反**：ZO 下单层最好选**早期层**（Dominant-Layer ZO）；头-only 的天花板低于"头+骨干"（Split-FG）；头层本身是 FG 最难训的位置（FineFed）。我们 37.0 的墙很可能就是这条线的具体表现。
2. **但我们的约束不同**：冻结骨干是设备侧通信动机，不能照搬"训骨干"的结论。
3. **可便宜验证的一步**：把 D3 的 lm_head 换成**早期层的一个同尺寸 LoRA**（d≈36K 预算，如 layer 0/1 或 Dominant-Layer 识别的激活离群层），用**现有 BP 臂**同 n/lr/调度对照。若早期层显著更好 ⇒ D3 的可训位置选错，比继续调 M/估计器更值得做。

---

## 6. 今日其它核查（备查，均已在本机验证）

### 6.1 BP 臂全景（A）

| 臂（SPU=10 ⇒ n=30，D3 r=1） | 轮数 | r99 test PPL | 备注 |
|---|---|---|---|
| lr=0.15 linear | 100 | 39.11 (min @r79) | |
| **lr=0.3 linear** | 100 | **37.03** | 尾段仍在缓降 |
| **lr=0.45 linear** | 100 | **36.96** | 与 0.3 差 0.2% ⇒ 平顶 |
| lr=0.6 linear | 100 | 37.31 | 中途峰值 258.3（r21），**被衰减救回** |
| lr=1.0 linear | 5 | 532（已崩） | r1 train PPL **195,706** ⇒ 杀掉 |
| lr=0.05 constant | 50 | 39.08 | |
| spu30(n=90) lr=0.05 constant | 50 | 38.37 | 同 lr 下 n 大三倍更好 |
| lr=0.3 **constant** 100 轮 | 进行中 | — | 验"linear 是否浪费后 1/3" |
| **r=2, lr=0.3 linear** 100 轮 | 进行中 | — | 本轮新增，验秩/容量 |

**读法**：步长在 0.3–0.45 平顶 ⇒ **lr 已见底**；所有跑够轮数的臂都落到 ~37.0 ⇒ 像吸引子。

### 6.2 train 侧四流与"是否记忆化"（A）

日志有四条 eval 流：`[EVAL-PPL-TRAINBATCH-BEFORE/AFTER]`（本轮被训练的 n=30 条）、`[EVAL-PPL]`（留出 test 50）、`[EVAL-PPL-TRAIN]`（留出 train-split 50）。

- lr=0.3：batch 44.06→34.96（−9.1），test 45.06→37.03（−8.0），train-split 44.33→35.82（−8.5）
  ⇒ **批的降幅只比留出多 1.14×** ⇒ 更新带的是**分布信息**，不是把样本背下来。
- lr=0.05（50 轮）：比值 **2.05**（batch −11.9 vs test −5.8）⇒ 小步长长跑反而更容易贴样本。
- **r25 附近所有臂的 batch PPL 都齐刷刷升到 51–57 再回落** ⇒ 是数据伪影（各臂"轮次↔data_id"映射相同，那一批特别难），不是训练病态。

### 6.3 唯一一次 ZOO-vs-BP 的 cos 实测（A）

- 出处：`log/estimate_lr_20260914.log:206`，探针 `run_lm_exps/estimate_lr.py`。
- 口径：D3 r=1（d=36,096），θ = 训练过的 checkpoint（`checkpoint_20260913_132655_r1.h5`，‖θ‖=0.5817），3 条样本，`g_exact` = 同 3 条样本的精确 head-only 反传均值，`g_zoo` = Σ (L(θ−hv)−L(θ+hv))/(2h)·v，v~N(0,I) 未归一化、h=1e-2、M=3072（每样本 1024）。
- 结果：**cos=0.2794 vs 理论 √(M/d)=0.2917**；独立自洽校验：沿精确方向 dL/dε=−0.2686、沿 ZOO 方向 −0.0779，比值 **0.290 ≈ cos**。
- 限制：**单 θ、单步、离线**（backbone 冻结、只算 head），不是沿训练轨迹、不是逐轮；原始 LoRA r=1 qkvo（d=1.05M）的 cos=5.4% 只是**公式**，从未实测。
- 另：C5 的 M 扫描（`test_exact_head_grad.py`，d=144,384）M=10/100/1000/10000 → 0.0069/0.0266/0.0839/0.2565（理论 0.0083/0.0263/0.0832/0.2632）。
- **`FwdLLM/training/tc_transformer_trainer_distribute_check_cosine.py:155` 是死代码**（算 `cos(true_grad, forward_grad)`），全仓库无引用、任何日志无 `cos_sim=` 输出 ⇒ **训练环内从未打过 cos**，这是一个可直接补的观测位（`--grad_source bp` 已提供同 θ 同 batch 的精确梯度）。

### 6.4 "1 条样本怎么训练"（A：代码实证）

- 样本 = `(input_ids, attention_mask)`，pad 到 `--model_max_length`=4096，**有效 token 仅 ~17–19**（日志：n_samples=30 → n_tokens=557）。
- BP 支路（`forward_training/tc_transformer_trainer_distribute.py:352-441`）逐条：`labels = input_ids.masked_fill(attn_mask==0, -100)`（:404）→ 一次 fp32 前向（:413-415）→ `loss.backward()`（:427）→ `self.grad[i].add_()`（:434）。
- **不是循环喂前缀**：loss 是 HF `LlamaForCausalLM` 的标准 shift-CE（`model/modeling_llama_sparse.py:1269-1277`），一次前向内**对所有 n 并行**完成"用前 n 个 token 猜第 n+1 个"。
- 计数：1 条样本 ≈ 17 个预测项；n=30 ⇒ 约 **510 项/更新**；`epochs` 只作用于 ZOO（每条样本扰 epochs 次），BP 支路 `epochs` 置空。
- 更新落地（`FedSgdAggregator.py:230-259`）：Σ 梯度 → 全局 L2 归一化 → clamp → `param.sub_(lr·ĝ)` ⇒ **‖Δθ‖ ≡ lr**。
- `num_logits_to_keep` 默认 0，而代码写 `hidden_states[:, -num_logits_to_keep:, :]`（Python `-0 == 0`）⇒ **始终算全 4096 位置的 logits**（fp32 下 524 MB/样本），其中 99.5% 是 padding ⇒ 算力优化线索。

### 6.5 权重共享（供写作时表述）

`lm_head` 是**一个** `nn.Linear(4096→32000, bias=False)`，被**所有位置共享**（`modeling_llama_sparse.py:1142,1259`）；一条 18 token 样本是"同一个头被调用 17 次"，不是一个 token 一个头。r=1 LoRA 下 `ΔW = B·A` ⇒ **32000 个词行只能沿同一个 4096 维方向 A 修改**，只有每行幅度 B_i 不同 ⇒ 这才是"抓手太弱"的结构性原因。本仓库 `config.json` 为 `tie_word_embeddings: False` ⇒ head 与 embedding **不绑定**。

---

## 7. 结论与可行动项

| 优先级 | 行动 | 依据 | 成本 |
|---|---|---|---|
| 1 | **可训位置消融**：lm_head → 早期层（layer 0/1 或激活离群层）的同尺寸 LoRA，BP 臂同 n/lr/调度对照 | Dominant-Layer ZO（§5.1）+ 我们自己的 6.9% 上限观测 | 与现有 BP 臂同（~6h/臂） |
| 2 | **r 消融**（r=2，进行中）：区分"秩-1 约束"与"d/容量" | §6.1 平顶 ⇒ 需要换杠杆 | 同上 |
| 3 | **补训练环内 cos 日志**（复用 `--grad_source bp`，零额外前向） | §6.3 死代码 | 极小 |
| 4 | **缓存-h 方案**：定位为**诊断/消融**（把 cos 0.28→1，分离"估计器"与"配置天花板"），卖点不能是"低秩扰动" | §3.1/3.2 已占位 + §3.3 我方推导 | 中（需实现 factorized eval） |
| 5 | **论文差异化**：token 稀疏 × ZO × 联邦 的交集（§4.5），并把 §1.1 四处偏离写成"我们做了什么" | §4 版图 | — |

---

## 参考文献（出处）

**上游代码/论文**
- [FwdLLM: Efficient FedLLM using Forward Gradient (arXiv 2308.13894)](https://ar5iv.labs.arxiv.org/html/2308.13894)
- [FwdLLM+: Accelerating Forward-Only FedLLM With Low-Rank Perturbations (IEEE TMC 2026)](https://ieeexplore.ieee.org/document/11429542) — 也见 [Semantic Scholar](https://www.semanticscholar.org/paper/FwdLLM%2B%3A-Accelerating-Forward-Only-FedLLM-With-Peng-Xu/c127f6db0b15eac9c933d6f896bbbf328e79b188)
- [P-RGE / Thinking Forward: Memory-Efficient Federated Finetuning of Language Models (arXiv 2409.15520)](https://ar5iv.labs.arxiv.org/html/2409.15520)

**联邦 ZO**
- [FedKSeed（ICML 2024）](https://icml.cc/virtual/2024/poster/33581)
- [ZorBA: Zeroth-order Federated Fine-tuning of LLMs with Heterogeneous Block Activation](https://www.semanticscholar.org/paper/ZorBA%3A-Zeroth-order-Federated-Fine-tuning-of-LLMs-Meng-Tang/efef487c00c9a98352e5b294082eb73c51ae5fae)
- [Ferret: Federated Full-Parameter Tuning at Scale for LLMs (arXiv 2409.06277)](https://ar5iv.labs.arxiv.org/html/2409.06277)
- [FineFed: Forward-Only Federated Fine-Tuning for Many-Class Tasks under Non-IID Heterogeneity](https://openreview.net/pdf?id=ciO8Bt4oUS)

**ZO for LLM（集中式）**
- [Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark (arXiv 2402.11592)](https://ar5iv.labs.arxiv.org/html/2402.11592)
- [ZO-Act: One-Shot Activation-Informed Low-Rank Subspaces (arXiv 2607.01125)](https://huggingface.co/papers/2607.01125)
- [CurvZO: Adaptive Curvature-Guided Sparse ZO (arXiv 2603.21725)](https://ar5iv.labs.arxiv.org/html/2603.21725)
- [KerZOO: Kernel Function Informed ZO (arXiv 2505.18886)](https://ar5iv.labs.arxiv.org/html/2505.18886)
- [Dominant-Layer ZO: A Single Layer Dominates ZO Fine-Tuning of LLMs (arXiv 2606.05516)](https://papers.cool/arxiv/2606.05516)
- [Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity (ICLR 2025)](https://mlanthology.org/iclr/2025/guo2025iclr-zerothorder/)

**前向梯度 / BP-free 相邻**
- [Backpropagation-Free Trunk Training via the Split Forward Gradients](https://www.machinebrief.com/news/backpropagation-free-trunk-training-via-the-split-forward-gr-8tbh)
- [ZeroFlow（ICML 2025）](https://mlanthology.org/icml/2025/feng2025icml-zeroflow/)
- [SharpZO（NeurIPS 2025）](https://mlanthology.org/neurips/2025/yang2025neurips-sharpzo/)

**token 稀疏 × 长上下文**
- [LeMo: Enabling LEss Token Involvement for MOre Context Fine-tuning (arXiv 2501.09767)](https://fugumt.com/fugumt/paper_check/2501.09767v1)

## 变更表

| 时间 | 操作 | 文件 | 说明 |
|---|---|---|---|
| 16:50 | ADD | `script/RUNME-v2-BP-sweep.sh` | 新增 `LR_R` 旋钮（默认 1，保持历史臂命名与行为逐字一致；`R≠1` 时日志/ckpt 名加 `_r$R`），`--lora_lm_head_r` 由硬编码 1 改为 `$LR_R`；用法注释补一行 |
| 17:0x | NEW | `docs/ailog/260916-165417-smell-v2-fwdllm-upstream-and-zo-literature.md` | 本文档 |

## 待办

1. 等 **r=2 / lr=0.3 linear 100 轮**（GPU1）与 **lr=0.3 constant 100 轮**（GPU3）出结果，与 r=1/lr=0.3 linear 的 37.03 对照。
2. 设计"早期层 vs lm_head"的 BP 消融脚本（同 d≈36K、同 n/lr/调度）。
3. 决定是否补训练环内 cos 日志（§6.3）。
4. 缓存-h 方案：先做**等价性测试**（同 θ 同 v，缓存路径 vs 全量前向路径，要求 cos(g_a,g_b)≈1.0000），通过再谈 M 扫描。
