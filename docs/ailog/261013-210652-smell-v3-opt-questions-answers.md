# Ailog 261013-210652 — Phase N: 05/01 OPT 对齐问题单回填（实验侧 v1）

> 用户提供 `reports/opt/05-evaluation-opt-questions-v0-zh.md`（面向实验侧的参数对齐问题单），
> 要求填写可确认的事实项。本 session 逐题回填：事实题给出代码/数据级答案与复算路径，
> 写作决策题给建议并标注"待作者定稿"。

## 回填要点（事实）

| 题 | 结论 |
|---|---|
| 1.1 k | `--zo-layer-group`，每轮激活层数；确定性轮转 `{(r·k+i) mod 24}`；k=24 ≡ 全部 24 层；coverage=k×rounds/24 |
| 1.2 D | `--zo-directions`，每 local step 独立单位方向数；中心差分 2 前向/方向；`g=mean_i[dim·ΔL/(2ε)·v_i]`；方向跨层共享；D=设计 §4.1 的 n |
| 1.3 | D 个估计直接平均；无方向级裁剪；client-delta 层 L2 裁剪（3×median） |
| 1.4 | LoRA r=1/α=2/q-k-v-o；可训练 196,608；batch=1×16384；local_steps=2；lr 0.30→0.10 于 r30（饱和诊断后 `--adapter-init`）；clip=3×probe median（12.619=3×4.204 等已复算） |
| 1.5 | sparse=0.4=**保留** top-40% KV 块（层 11–22、pool 64、102/256）；设计节 0.2 应改 0.4 |
| 1.6 | Discovery→ICL 生成式（非 RedPajama）；full=全 16k token NLL，answer=GT label token NLL；100 train/16 local-test/500 全局；seed 42 |
| 1.7 | TD-3/step5/sdpa_prune/delta-clip 的论文措辞与上游（FwdLLM/Jenga） |
| 1.8 | 位置适配 B/C 为单进程预热（不参与联邦），512×16k、500 step；一段可直接入文的描述 |
| 1.9 | c=30 全参与；一轮=30 client×2 步；`doc_order=fixed` |

## 关键 caveat（要进论文/附录）

1. **固定语料缺陷**（复述 `261012-151805`）：OPT 长跑每轮固定取前 2 篇/client，1800 doc-view 落在 ~60 篇上；用户已定 OPT 不重做。
2. **a01 数据瑕疵**：101 行重复、client_17 仅 10 篇唯一、fallback 133。
3. **deploy 口径 mask-effect**：同 adapter 加掩码 full +0.30~0.41 / answer −2~−15%，部署 answer 优势主因。

## 建议（待作者定稿，非事实）

- 章节：Setup → Memory → ZOO{k→D→效率} → CATV → Compression；D 为 ZOO 内小节。
- D 主张：以 k 为主 + 微扫支撑；唯一缺实验=同 k 扫 D（GPU 排队）。
- CATV：a03 主 + a01 诚实并列；deploy 用 (b) 口径。
- §4.1 自适应阈值：保留理论动机 + 说明实现固定 D（最稳）。
- 图表：正文 fig5/fig6(+fig1/fig3)、table1+table6；其余附录；fig1–4 建议按 Times/无标题/图例白底黑框重生成。

## 变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 21:00 | 回填 | `reports/opt/05-evaluation-opt-questions-v0-zh.md` | 19 题全部填写（事实+建议），标注来源与复算路径 |
| 21:06 | 新增 | 本 ailog | 记录回填结论与待作者决策项 |
