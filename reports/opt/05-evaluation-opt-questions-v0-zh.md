# 05/01 OPT-350M 结果对齐问题单 v0（实验侧已填答）

> 背景：已阅读 `reports/opt/`（README、fig1–fig6、table1–table6、data/*.csv、make_report.py）。
> 用途：对齐写作用到的实验参数、章节主张与图表取舍；回收后据此产出《修改表（中文）》，再进入中文稿。
> 用法：直接在每题下方 `答：` 处填写；可只答编号。已有建议的题目，采纳请写「采纳」或直接写你的版本。
> **状态：实验侧已回填 v1（标注来源与 caveat；写作决策题为建议，待作者定稿）。**
> 备注：本 checkout 没有 `docs/paper/`，问题单里引用的 05 正文与 `figure_design/pre2` 不在此机，涉及正文内容的题只能给数据侧答案或建议。

---

## 0. 我已确认的事实（如与事实不符请纠正）

- 模型/环境：OPT-350M，16k 上下文，ZOO+LoRA，fp32 + `sdpa_prune`，稀疏率 `sparse=0.4`，predictor `step5_a01a03_clients_causal`，c=30 客户端串行 FedAvg，AutoDL 1×RTX 4090。
- a01 = α=0.1、a03 = α=0.3（Dirichlet 非 IID 划分）。
- 指标：full G-PPL、answer G-PPL（token 加权）、PPL/h、实测墙钟小时、clipped（delta-clip 计数）、逐样本配对 ratio 与 p 值（n=500）。
- CATV 三种口径：off / on（eval off，同口径）/ on（eval on，部署口径）。
- 位置适配：none → pos_only (B) → pos_lora (C)，16k G-PPL 398.8 → 31.4 → 18.1（2k 无回归）。
- 主要结论：覆盖度 k 是主杠杆（PPL/h：k4 0.019 vs k24 0.09–0.10）；CATV 收益依赖 α（a03 同口径 −0.20 PPL、answer 早期显著改善；a01 训练效应不显著）。
- 数据包对账 32/32 PASS；T3a 为 n=4 支持行；无 OPT-350M 上的 BP 对照（按此前决定暂不做）。

纠错/补充：

答：以上核对无误。**必须补进论文的三条 caveat/事实**：

1) **固定语料缺陷（OPT 全量长跑）**：所有 OPT 实验的取数协议是「每轮固定取分片前 `local_steps=2` 行」——30 轮 × 30 client = 1800 次 doc-view 全部落在每个 client 的前 2 篇上（每 tag 3000 篇中约 60 篇被使用；`local_test` 16 篇从未参与训练）。`--doc-order {fixed,cycle}` 接口 10-12 才加入，OPT run 均在此之前。臂间对比仍内部有效（同一底料），但绝对降幅受 60 篇封顶，部分"下降"可能是对固定 mini-batch 的适应。已在 `docs/ailog/261012-151805` 确认，用户已决定 **OPT 不重做**、论文/附录注明。
2) **a01 数据瑕疵**（α=0.1 特有）：train 行 3000 条中 **101 行内容重复**，`client_17` 仅 10 篇唯一（90 行重复），fallback_reuse=133；a03 仅 1 行重复（`261012-151805` §1.3）。
3) c=30 = 总客户端数且每轮全部参与（无采样/无 dropout）；每 client-轮 2 步、每步 1 条 16384-token 样本；LoRA r=1/α=2。

---

## 1. 实验参数定义（写 Setup 必需）

**1.1 k 的准确定义。** k 是「每轮参与更新的层数」（layers/round）吗？k 层的选择是固定轮转还是随机/按重要性？k=24 是否等于 OPT-350M 的全部层？「coverage」的正式定义是什么（k × 轮次 / 总层数，或其他）？

答：k = `--zo-layer-group` = **每轮参入 ZO 更新的层数**（layers/round）。层集合**确定性轮转**（`--zo-layer-rotate`）：第 r 轮的层 = {(r·k+offset) mod 24, offset=0..k−1}——k=4 时按 0–3 → 4–7 → … 循环，k=24 时每轮都是全部 0–23。OPT-350M `num_hidden_layers=24`（hidden 1024、16 heads、FFN 4096），所以 **k=24 ≡ 全部层**。coverage 可定义为 `k × rounds / 24` 的层-轮覆盖倍数（k4×30 轮 = 5×，k24×30 轮 = 30×）；注意这是 **ZO 扰动的参数覆盖**，与剪枝层覆盖（只在上半层 11–22，见 1.5）是两个维度。

**1.2 D 的准确定义。** D 是「每轮/每层/每样本的扰动采样数」中的哪一个？具体地：一个客户端一轮对一层做 D 次扰动前向（θ+εᵢvᵢ）并聚合？还是对同一扰动做 D 次重复？D 与设计 §4.1 公式中「n 次估计」的关系是什么（D = n？）？

答：D = `--zo-directions` = **每个 local step 内独立采样的随机方向数**（不是重复同一方向）。实现（`src/train/zoo.py::zo_grad`）：在**当轮激活子空间**上采 D 个 N(0,I) 单位方向 v，每个方向做一次中心差分（θ+εv 与 θ−εv 各 1 次前向），`g = mean_i[ dim · (L₊−L₋)/(2ε) · v_i ]`（dim=子空间元素数，单位向量需 dim 补偿才无偏）。一个方向向量是**跨当轮全部 k 层的平坦向量**（不是每层独立）。每 client-轮代价 = `local_steps × D × 2` 次前向（主跑：2×32×2=128）。与设计 §4.1 的「n 次估计」关系：**D = n**（每个 local step 的估计样本数）；设计里的自适应判据在实现中不存在的（见 2.5）。

**1.3 一轮内的梯度聚合方式。** D 个采样估计是直接平均、还是带方差控制/裁剪后再聚合？k 层之间是否共享同一组扰动方向？

答：D 个估计**直接平均**（方向级无裁剪、无控制变量）；方差只在 **client-delta 层**处理：每轮聚合前对每个 client 的 Δθ 做 L2 范数裁剪 `--delta-clip`（见 1.4）。k 层**共享同一组 D 个平坦方向**（每个方向覆盖当轮激活子空间的全部参数）。ε=`--zo-eps`=1e-3。

**1.4 训练超参。** lr 链式 0.30→0.10 的切换依据与时刻；clip（12.61、19.41、7.3–8.5 等）的定义（3×median 标定？）；`clipped` 列要不要写进论文；batch size、LoRA rank、目标模块（Q/K/V/O 还是全部线性层）。

答：
- **batch/local steps**：每步 1 条 16384-token 序列（batch=1），`local_steps=2`（每 client-轮 2 步）；**LoRA r=1、alpha=2、targets=q/k/v/out_proj**（4 个 attention 投影；不含 FFN），可训练参数 **196,608**（24 层 × 4 投影 × 2 × 1024）。
- **lr 链**：k4 臂 lr=0.15（δ 匹配，见 09-29 ailog）；k24 臂起 lr=0.30（k-scan 选定，微扫中 k24@0.30 单 client 不稳但 30-client 聚合稳定）。30 轮后段均降幅 0.064→0.007/轮（D16 臂），诊断"饱和=lr 限制还是数据地板"，在 **r30 用 `--adapter-init adapter_round029` + lr=0.10 续训**：续训仍解锁 −1.33%（p≈0）⇒ 部分是 lr 限制、部分地板；随后 a03 以 15 轮一块续到 r74（每块重新探测 clip），a01 到 r44。切换时刻即 r30/r45/r60（各块边界）。
- **clip 定义**：开跑/续训前用 **1 轮 30-client δ 探测**（`temp/*probe/`）取每 client 原始 Δθ 的 L2 范数 **中位数 × 3**。已验证：12.619=3×4.204（D32/a03）、19.407=3×6.469（D16）、13.336=3×4.445（D32/a01）、10.208=3×3.403（D16 续训）、7.316/7.796/8.480（a03 on 三块）、7.028/7.449（a01 两块）等，可用 `logs/fed/*/config.json` 与 `temp/*probe/*/metrics.jsonl` 复算。
- **`clipped` 列**：建议**不进正文主表**（它反映 FD 估计的重尾噪声水平，不是失败计数；主跑累计 51–358 client-轮）。可放附录一段"离群 client 由 delta-clip 拦截"。

**1.5 稀疏设置。** `sparse=0.4` 是什么口径（保留比例/丢弃比例）？与设计 §4.2 的「sparsity budget 0.2、块大小 64」如何统一——设计节改成 0.4，还是说明实验设置不同于示例？

答：`sparse=0.4` 是**保留比例**：上 12 层（layer 11–22）每层保留 **top-40% 的 KV 块**（n_blocks=256、pool=64，即保留 102/256 块；块分由 predictor 沿 query 维求和得到，topk 后按原序子集化 token）；**下 11 层（0–10）全留、末层（23）豁免**。实现见 `src/models/modeling_opt_smell.py::OptSdpaPruneAttention`（Jenga 真语义：`modeling_opt.py:294-321`）。与设计 §4.2 的 0.2/64 **不一致**：实验用 **0.4**；建议设计节改 0.4（或写"示例预算；实验取 0.4"）。块大小 64 一致。

**1.6 数据与任务。** 16k 训练的语料是什么（RedPajama？如何切分/构造）？full 与 answer 两个 PPL 的「answer」指什么（问答对/文档问答构造？）；「G-PPL」的 G 是什么（generative？）；每个客户端样本量、训练/评测划分。

答：**不是 RedPajama**。数据 = `sileod/discovery`（HF，apache-2.0）连词分类，改写成 ICL 生成式样本（模板照抄 LongICLBench）：
- 每条样本 = instruction（原文） + 演示（`{s1} ( ) {s2}\nthe most suitable conjunction word in the previous ( ) is  {label},`）**顺序填充到 16384 token**（约 260–270 条演示，条数随样本变化） + query（同格式、留空待预测） + GT label（+逗号）。
- **full PPL** = 全文 16384 token 的 LM NLL（token 加权）；**answer PPL** = 只统计末尾 **GT label token**（`answer_spans.npy` 记录的 label 区间，不含逗号）的 NLL。**G-PPL** = generative PPL（生成式困惑度；与分类 accuracy 对应的生成式口径）。
- 规模：30 clients × (**100 train + 16 local test**)，全局评测 **500 条**（官方 test query + 独立均匀演示池；client 池/全局演示池/官方 query 三池互斥）；per-class Dirichlet α，**seed=42**。每个 16k 样本的 token 预算固定（演示按顺序删/补）。
- 又一次提醒：训练实际只用到前 2 篇/人（§0 caveat 1）。

**1.7 术语正式化。** 论文里如何称呼 `TD-3`、`step5` predictor、`sdpa_prune`、`delta-clip`？是否有可引用的来源（FwdLLM/Jenga 对应组件）？

答：建议正文不出现内部代号：
- `TD-3` → **k=4 ZOO 臂**（`logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip`；论文写 "k=4 layer-rotating ZOO"）。
- `step5` → **block-importance predictor**（第 5 版：causal 训练目标 + client 训练分片分布；144 tensors；用于选 KV 块）。上游 = Jenga 的 predictor 设计（`jenga/models/predictor.py`）。
- `sdpa_prune` → **token-subset block-sparse attention（SDPA 实现，fp32）**；语义等价 Jenga 的 sparse attention（块分 → top-k KV 块 → token 子集化）。上游 = Jenga `modeling_opt.py`。
- `delta-clip` → **per-client delta-norm clipping**（聚合前对 Δθ 做 3×median L2 裁剪）。
- 可引用：FwdLLM（前向梯度联邦训练）、Jenga（块稀疏与 predictor）；本项目的 `docs/standard.md` 记录组件编号规范（内部用）。

**1.8 位置适配 B/C。** 能否给一句可写入论文的描述？B = learned offset（interp init）具体指什么；C = B + LoRA adapter 训练在什么数据上、是否参与联邦。

答：可写："OPT 的学习式绝对位置嵌入（2048）无法零样本外推 16k（G-PPL 398.8）。我们把位置表延长为可训练参数并以线性插值初始化（B：**pos_only**），或在此基础上同时对 attention 四投影加 LoRA 联合训练（C：**pos_lora**），在 512 条 16k 预热样本上 BP 训练 500 步。适配后 16k G-PPL → 31.4（B）/ **18.1**（C），2k 无回归，取 C。"——**不参与联邦**：是单进程预热（`src/train/longctx_adapt.py`），产物 `pos_embed.pt(+adapter)` 固定后由所有联邦 run 加载（`--pos-checkpoint`）。

**1.9 客户端与协议。** c=30 是总客户端数还是每轮参与数？串行 FedAvg 的轮次定义（一轮=每客户端一次本地更新？）；a01/a03 的划分种子与 α 的来源。

答：c=30 **既是总数也是每轮参与数**（无采样、全部串行）。一轮 = 30 个 client 各自做 `local_steps=2` 步本地更新（步=1 条样本）后，服务器按 **样本数加权平均**（各 client 样本数相同 ⇒ 等价等权）更新全局；即"一轮 = 每个客户端 2 次本地更新"。α=0.1/0.3 = per-class Dirichlet 浓度参数（`build_discovery_16k.py::build_partition`，`seed=42`；a01/a03 为 tag 名）。**`doc_order=fixed`**（history caveat 见 §0）。

---

## 2. 章节结构与主张（**写作决策题：以下为实验侧建议，待你定稿**）

**2.1 05 的章节顺序。** 我的建议：Setup → Memory → ZOO 优化（覆盖度 k + 多采样 D + 效率）→ CATV（α=0.1/0.3）→ Compression。新增的「多采样（D）」小节放在 ZOO 优化节内，还是独立小节？给出你想要的顺序。

答（建议）：**Setup → Memory → ZOO 优化 { 覆盖度 k → 多采样 D → 效率 } → CATV → Compression**。D 作为 ZOO 内的独立小节（设计 §4.1 本有多采样位置；但正文主张仍以 k 为主，D 用于解释"固定预算怎么花"）。

**2.2 D 小节的核心主张与配图。** 从现有数据看，D 的效应被 k 与 lr 混淆（table1 中 k4/D22 与 k24/D16/D32 不同 lr；table5 微扫 n 极小）。你希望这一节主张什么：
(a) 「在同一 k 下增大 D 可降低方差、改善稳定性」——但需要新图/新数据；
(b) 「固定采样预算下，D 与 k 的配置权衡」——用 table1/fig1/fig3 说话；
(c) 用 table5 微扫支撑「D 的选择」；
(d) 其他。请指定配图（现有 fig1/fig3？还是需要我基于 `data/*.csv` 新画 D 相关图）。

答（建议）：**(b) + (c) 组合**。主张："在前向预算固定时，k（覆盖度）比 D（单步估计）更值价；D 主要用于稳定单步估计（微扫：k4→k12 同 D 下降幅更大，k24 单 client 不稳但 30-client 聚合稳定）"。配 fig1/fig3 + table5（标注 n=1 client/2 样本）。若你要 (a) 的干净证据，需要新实验（同 k 扫 D，会占 GPU；当前被 BERT 链占用，可排期）——**这是唯一缺的实验**，请明确要不要补。

**2.3 CATV 结论口径（a01 负结果）。** a01（α=0.1）训练效应不显著（r29 full p=0.97），a03 显著。正文如何表述：
(a) 以 a03 为主结果，a01 作为「收益随非 IID 强度（α↓）减弱」的诚实讨论；
(b) 两臂并列平铺；
(c) 只报 a03，a01 放表格不展开；
(d) 其他。

答（建议）：**(a)**：a03 为主 + a01 一行并列 + 一句"在更弱异质性（α=0.1）下训练效应不显著（p=0.97），收益随非 IID 强度增强"。表 2/6 已含两臂，不需要隐藏 (c)。

**2.4 deploy 口径是否入正文。** 部署口径（评测也带掩码）在 r74 与 off 打平，且残余 answer 优势主要来自评测掩码本身。你要：
(a) 只报同口径（干净的训练效应）；
(b) 同口径为主 + 一段 deploy 口径说明（含 mask-effect 行）；
(c) 两口径都进主表。
选哪个，正文该如何措辞？

答（建议）：**(b)**。实验侧强烈建议保留 mask-effect（r0 同 adapter 加/不加掩码：full +0.30~0.41、answer −2~−15%），否则 deploy 的 answer 优势会被误读为训练收益；措辞："CATV 的训练收益在同口径下成立；部署口径（评测也带掩码）在 r64–r69 与基线打平（r74 ratio 1.0001, p=0.65），残余 answer 差异主要来自评测掩码本身（占 X%）。"

**2.5 §4.1「自适应方差阈值」怎么改。** 实现中 ZOO-GR 没有自适应模块，改用固定 D 多采样。设计节的处理：
(a) 删除自适应判据，改为「固定采样数 D + 方差随 D 下降」的表述；
(b) 保留 var 判据作为理论动机，说明实现采用固定 D；
(c) 保留判据但降级为「可选的停止条件」；
(d) 其他。请确认口径（这决定我改 §4.1 与 01 的写法）。

答（建议）：**(b)** 最稳（理论动机 + 实现事实兼容，且与 01 的公式不冲突）。实现事实：`zo_grad` 固定 D 方向、方差 ∝ 1/D（σ²/D）；没有任何自适应阈值代码。（(a) 也可以，但会丢掉 §4.1 的动机链条。）

**2.6 位置适配的定位。** 它是「SMELL 为 16k 采用的必要组件（写进设计/流程）」还是「实验环境的适配细节（只写 Setup）」？fig4/table4 是否入正文？

答（建议）：**必要组件**（写进流程/Setup 一句话 + 附录细节）：不写会被问"16k 怎么跑起来的"；fig4/table4 **入附录**（主文不必占篇幅）。

**2.7 Table3 取舍。** 3a（predictor/random/dense，n=4）与 3b（dense vs sparse，n=500）是否入正文？dense 显著优于 sparse（full 26.5 vs 33.5）是内存-精度的 trade-off，正文是否明说，还是只作为支持行？

答（建议）：3a **只作支持行/脚注**（n=4，弱）；3b **入正文或附录**并**明说** trade-off（dense 26.5 vs sparse 33.5 full；memory 换精度），否则审稿人必问"为什么不 dense"。注意 3a 的 random predictor 的 **full PPL（31.9）反而低于 step5（33.3）**，这是"随机选块破坏性更小/被剪 token 更少"的非常规现象（n=4），不建议在正文引用该行（answer 上 step5 1056 明显优于 random 1762）。

**2.8 Table5 取舍。** 微扫（1 client、2 样本、3 轮）是否入正文？还是仅作为「参数选择依据」在正文一句话引用？

答（建议）：正文一句话引用 + 附录整表（n 太小，不宜作为主证据）。

**2.9 效率节。** fig3（PPL vs 墙钟、PPL/h）是否入正文？「PPL/h（0–29 轮窗口）」这一定义是否在正文解释？

答（建议）：**入正文**（这是"效率/可扩展"主张的直接证据）；PPL/h 定义放图注：`Δfull PPL / 实测墙钟小时`，右图统一 0–29 轮窗口，链式实验的墙钟=各块之和。

---

## 3. 图表入文（「至少要放 a01、a03 的图和表」）

**3.1 最小图集。** 现有图中你指定必须入正文的集合（可多选）：fig1（a03 主曲线）、fig2（CATV×α 四联）、fig3（效率）、fig4（位置适配）、fig5（α=0.1）、fig6（α=0.3）。之外的图如何处置（丢弃/备用）？

答（建议）：**fig6（a03）+ fig5（a01）入正文**（你已经定的两张）+ **fig1**（多 k 主曲线）与 **fig3**（效率）视篇幅入正文；fig2 四联作为 fig5/6 的合并/附录；fig4 附录。丢弃：无（都保留在包内备用）。

**3.2 最小表集。** 同样列出必须入正文的表：table1（主结果）、table2（CATV×α×口径）、table3、table4、table5、table6。table2 与 table6 是否有重复、是否合并？

答（建议）：正文 **table1 + table6**（table6 是 table2 的 r44/r74 子集且含 r0 下降率，信息更贴论文主叙事）；table2 的配对 p 值可作为 table6 的补充列或放附录。table3/4/5 附录。

**3.3 table1 列改造。** 原始列：arm / k / D / lr / clip / rounds / full PPL 起止 / dFull% / answer PPL 起止 / dAns% / hours / PPL/h / clipped。正文保留哪些列、删哪些（如 clip、clipped、answer 起止）？是否要按 k/D 分组重排？「table 仅为原始数据，可基于 data 改动」——请确认哪些列可以合并/新增。

答（建议）：正文保留 `arm / k / D / lr / rounds / full 起止 / Δfull% / answer 起止 / Δans% / hours / PPL/h`；**删 clip 与 clipped**（放附录，或只在脚注写"clip=3×median"）；建议按 **{k4 → D16 → D32 → D32+CATV} × {a01|a03}** 分组行序，chain 行（0–59/0–74/0–44）与 30 轮行分开两个小块。

**3.4 图的样式。** 直接复制现有 PDF/PNG，还是需要我重新生成（fig1–fig4 目前是否统一为论文风格；fig5/6 已按 Times New Roman、无标题、图例白底黑框生成）？目标字号/尺寸（单栏 3.5in 或跨栏 7in）？

答（建议）：**fig1–fig4 目前是 DejaVu + 无框图例，与 fig5/6 不统一**；建议我按 fig5/6 规范（Times、无标题、图例白底黑框）重生成一遍（纯 CPU，~10 分钟）。尺寸：fig5/6 现状 3.5 in 单栏；fig1/fig3 现状 7.2 in 跨栏；fig2 7.2×4.4 跨栏。

**3.5 排版方式。** a01/a03 是否用子图 (a)(b) 合并；fig2 的四联图是否拆成两图；跨栏（figure*）还是单栏？

答（建议）：a01/a03 用 **(a)(b) 双子图合并**（3.5×2.7 → 7.2×2.7 跨栏即可合并 fig5+fig6）；fig2 四联**拆成两图**或只保留 full 一行（answer 的两行可能冗余）；表格跨栏。

---

## 4. 与既有论文内容的同步

**4.1 模型切换范围。** 05 的 Memory 节目前是 LLaMA2-7B 表（88.35% 等）。本次 OPT-350M 实验上线后：
(a) Memory 保持 LLaMA2-7B（作为动机性证据）+ 主实验用 OPT-350M；
(b) Memory 改为 OPT-350M（用 `figure_design/pre2` 的 opt 数据重写）；
(c) 其他。请指认哪个是「本次实验用的模型，需要同步改」的范围（02/04/05 哪些段落）。

答：**本 checkout 没有 `docs/paper/` 与 `figure_design/pre2`**，无法核对 05 现状。数据侧事实：本机只有 `src/bench/memory.py`（可跑 OPT 显存矩阵），**没有 OPT 显存结果落盘**；88.35% 属 v2/LLaMA2-7B 时代。建议：**(a)**（Memory 保留 7B 动机表），并在 05 Setup 明写主实验模型=OPT-350M；若要把 Memory 换成 OPT，需我补跑 `src/bench/memory.py`（GPU 空闲时，当前被 BERT 占用）→ 请指定。

**4.2 §4.2 架构与稀疏参数。** 现文以 LLaMA2-7B 为例（32 层、后 16 层、块 64、预算 0.2、18.75%）。改为 OPT-350M 的话，请给：层数、hidden、heads、做稀疏化的层范围、块大小、预算（0.4？）、保留比例。或者改为与模型无关的通用描述 + 实验节给具体值？

答：OPT-350M 具体值：**24 层、hidden 1024、16 heads、FFN 4096、vocab 50272（word_embed_proj_dim 512）**；稀疏化层 = **11–22（上 12 层）**；块大小 **64 token**（n_blocks=256）；**保留比例 0.4**（⇒ 剪掉 60% KV 块）；末层 23 豁免。建议：设计节保持**模型无关描述**（"upper-half layers, block size 64, retention budget ρ"），实验节给这套具体值。注意 LLaMA2-7B 的"后 16 层"与 OPT 的"上 12 层"是同一规则（upper half）。

**4.3 01 Introduction 的更新点。** 请指定你希望引言的三个主打卖点（例如：内存节省 / 覆盖度 k 与 D 的可扩展性 / CATV 在强非 IID 下的收益），以及必须更新的数字或表述（模型名、16k、88.35%、α 依赖等）。a01 不显著是否需要在引言就说明局限？

答（建议卖点）：① **16k 端侧联邦长上下文可行**（ZOO 前向梯度，无需反传）；② **覆盖度 k 是 ZOO 联邦的主杠杆**（k4→k24，速率 0.019→0.087 PPL/h，×4.6）；③ **CATV 在强非 IID（α=0.3）下改善 answer 并保持 full**（α 依赖作为诚实边界）。必须更新：模型名 OPT-350M、16k、PPL/h、α 依赖；**a01 n.s. 建议在 intro 或 limitation 里一句带过**（不要藏）。

**4.4 基线。** OPT-350M 上只有 ZOO 的 off/on 与 k/D 消融，无 BP 对照。05 的 Baselines 段如何写：只声明「对照为 ZOO 基线（无 CATV）」？Memory 节的四基线（Full FT/LoRA/Jenga/FwdLLM）是否保留？

答：OPT 上如实写 "ZOO-only with in-family ablations (k/D/lr/CATV/α)；no BP baseline"；Memory 的四基线**保留**（那是 7B 动机实验，属于不同实验组）。**注意**：若审稿问"ZOO vs BP 效率"，我们有 7B/旧数据与 BERT 侧新数据可引用，但 OPT 上没有——需要的话可排期补（GPU 排队）。

**4.5 压缩实验（VQ）去留。** 05 现有的 Compression 节保留吗？与新的 CATV/ZOO 节顺序如何排？VQ 实验是 LLaMA2-7B 尺寸（8M）做的，是否需要说明与 OPT-350M 实验的关系？

答：写作决策，实验侧只给事实：本 checkout 有 `third_party/VQ`（Count Sketch）与 v2 的 VQ 结果；与 OPT-350M 主实验无交叉。建议保留但标注尺度（7B/8M）并说明与主实验的关系（如"独立小节/附录"）。要补 OPT 尺度 VQ 需要新实验。

---

## 5. 流程与其他

**5.1 写作流程。** 是否仍走「中文修订稿 → 你审 → 翻译英文 tex」？本次范围是 05 重写 + 01 修改 + 02/04 同步，还是先只做 05？

答：流程由你定。实验侧可随时支持：数字复核、新图/新表、Times 重生成、补实验（GPU 排队）。**建议先只做 05**，01/02/04 的同步等 05 定稿后一次性改，减少来回。

**5.2 参考资料。** 是否有与本次 OPT-350M 实验对应的 ailog / 任务文档路径（便于引用与数字对账）？`make_report.py` 中提到「脚本内置 ailog 已发表关键值断言」，请给 ailog 目录。

答：**`docs/ailog/`**（最新相关：`261013-161052` CATV 图/表增补、`261013-155317` 结果包交付、`261012-093000` a03 r74 长训、`261007-005000` a01 D32+CATV、`261001-034500` k24 覆盖度与 D16/D32/r60、`260930-105500` 微扫与 k24 链、`260930-094100` TD-3 首个正结果、`260930-104500` Step1–3（漂移/刷新/dense 对照）、`260929-000200` 审计修复、`260928-110233` predictor causal、`260927-121756` fp32 cos、`260923-210703` 实验方案锁定、`261012-151805` 固定语料缺陷）。**对账**：`reports/opt/make_report.py` 内置 32 条 ailog 已发表数字断言（重跑输出 32/32 PASS）。权重/数据清单：`docs/weight-manifest.md`。

**5.3 图表文件去向。** 是否将 `evaluation/CATV/opt/figures|tables` 中选定文件复制到 `docs/paper/figures|tables`（沿用现有命名），还是引用原路径？

答：本 checkout 的包在 **`reports/opt/`**（没有 `docs/paper/`；你引用的 `docs/paper/evaluation/CATV/opt/` 应是另一侧 checkout）。是否镜像到 `docs/paper/...` 由你定；要的话我可以建同名目录并复制选定文件（保持文件名），或按你给的命名规则重命名。

**5.4 其他未尽事项。**

答：
1) **请明确 D 小节是否要补"同 k 扫 D"实验**（2.2 的唯一缺项；GPU 空闲时可做）。
2) **固定语料缺陷 + a01 数据瑕疵**请务必在论文/附录体现（表注一句 + 附录一小段），避免审稿人复现时发现。
3) 需要我把 **fig1–fig4 统一成 Times/无标题/图例白底黑框** 的话请说一声（会重生成并更新包）。
4) 若要把 a01/a03 合成双子图 (a)(b)，我可以在 `make_report.py` 加一个合并版 fig7，原 fig5/6 保留。

---

## F. 引用补录（实验侧回填，2026-10-14）

**F1. 请提供以下可引用来源（HF 链接或 BibTeX）。**

### F1.1 `sileod/discovery` 数据集

- HF：<https://huggingface.co/datasets/sileod/discovery>（本仓 `build_discovery_16k.py` 用 id `sileod/discovery`，config `default`；license apache-2.0）
- 代码/数据：<https://github.com/sileod/Discovery>
- 论文（作者官方 BibTeX，来自 GitHub README）：Sileo et al., NAACL 2019, "Mining Discourse Markers for Unsupervised Sentence Representation Learning"（ACL Anthology N19-1351）。注意：**HF 卡片与论文是同一作者**；引用该论文即为该数据集的出处。

### F1.2 LongICLBench（ICL 模板来源）

- 论文：Li, Zhang, Do, Yue, Chen. "Long-context LLMs Struggle with Long In-context Learning." **TMLR 2025**（OpenReview `Cw2xlg0e46`，2025-03-26）；arXiv:2404.02060。
- 代码：<https://github.com/TIGER-AI-Lab/LongICLBench>（MIT；Discovery 任务的模板在 `discovery_infer.py` / `data_preprocess.py`）
- 数据：<https://huggingface.co/datasets/TIGER-Lab/LongICLBench>
- 我们只**照抄模板与任务设置**（instruction/demo/query 三段式与 `the most suitable conjunction word in the previous ( ) is`），数据仍取 `sileod/discovery` 原始池自建 16k 分片。

### F1.3 位置适配 B/C 与 Position Interpolation 的关系

答：**思想对应、对象不同**，需要分开引用：
- 我们的 B/C（`src/models/position_embed.py::extend_opt_positions(mode="interpolate")`）是对 **OPT 的学习式绝对位置表**做线性插值（`F.interpolate(..., mode="bilinear", align_corners=False)`，即半像素线性重采样 2048→16384，仅位置轴变化）再 BP 微调——这是 **ViT 那套"换长度就插值 position embedding 再微调"** 的标准做法（Dosovitskiy et al., ICLR 2021, Appendix B.1 有明确描述）。
- **Position Interpolation（PI）**（Chen et al., 2023, arXiv:2306.15595）是同一"插值而非外推"思想，但其公式针对 **RoPE**；我们的模型没有 RoPE，**不宜写成"我们用了 PI"**，写成"we follow the position-interpolation principle" 并同时引 ViT 更严谨。
- "学习式绝对嵌入不能外推"这一问题本身可引：ALiBi（Press et al., ICLR 2022）与 Kazemnejad et al.（NeurIPS 2023）。
- 备选替代方案（正文若做 related work 可提）：Longformer（Beltagy et al., ACL 2020）用**复制**位置表扩展到 4096；RoPE/YaRN 等只适用于旋转位置。

**建议写法（英文）**："Since OPT uses learned absolute position embeddings trained up to 2,048 tokens, zero-shot 16k evaluation collapses (G-PPL 398.8). Following the position-interpolation principle [Chen et al., 2023] — implemented, as in ViT, by linearly interpolating the learned embedding table and fine-tuning [Dosovitskiy et al., 2021] — we adapt the table (B) and additionally fine-tune LoRA adapters (C), recovering 31.4 / 18.1 G-PPL at 16k."

**BibTeX**：

```bibtex
@inproceedings{sileo2019mining,
  title = "Mining Discourse Markers for Unsupervised Sentence Representation Learning",
  author = "Sileo, Damien and Van De Cruys, Tim and Pradel, Camille and Muller, Philippe",
  booktitle = "Proceedings of NAACL-HLT 2019",
  month = jun, year = "2019", address = "Minneapolis, Minnesota",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/N19-1351", pages = "3477--3486",
}

@misc{li2024longcontext,
  title={Long-context LLMs Struggle with Long In-context Learning},
  author={Tianle Li and Ge Zhang and Quy Duc Do and Xiang Yue and Wenhu Chen},
  year={2024}, eprint={2404.02060}, archivePrefix={arXiv}, primaryClass={cs.CL},
}
% TMLR 2025 版（若以正式期刊格式引用）：
@article{li2025longcontext,
  title={Long-context LLMs Struggle with Long In-context Learning},
  author={Li, Tianle and Zhang, Ge and Do, Quy Duc and Yue, Xiang and Chen, Wenhu},
  journal={Transactions on Machine Learning Research}, year={2025},
  url={https://openreview.net/forum?id=Cw2xlg0e46},
}

@article{chen2023positional,
  title={Extending Context Window of Large Language Models via Positional Interpolation},
  author={Chen, Shouyuan and Wong, Sherman and Chen, Liangjian and Tian, Yuandong},
  journal={arXiv preprint arXiv:2306.15595}, year={2023},
}

@inproceedings{dosovitskiy2021vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021},
}

@inproceedings{press2022train,
  title={Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation},
  author={Press, Ofir and Smith, Noah A. and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2022},
}

@inproceedings{kazemnejad2023impact,
  title={The Impact of Positional Encoding on Length Generalization in Transformers},
  author={Kazemnejad, Amirhossein and Padhi, Inkit and Natesan Ramamurthy, Karthikeyan and Das, Payel and Reddy, Siva},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2023},
}
```
