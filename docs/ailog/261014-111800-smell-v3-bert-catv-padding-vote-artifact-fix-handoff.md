# Ailog 261014-135500 — SMELL-v3 DistilBERT：CATV padding 投票假象（确认）+ **block0 强制保留（BERT 特供，已拍板）** + 修复交接 v2

> 交接对象：修复 agent。本文档 = 决定记录 + 缺陷机理/复现证据 + block0/[CLS] 论证 + 修复规格（含实现点） + 当前状态（勿冲突）。
> v1（11:18）只含 vote 假象；v2（13:55）并入 2026-10-14 讨论结论与 block0 决定。

## §0 决定记录（用户 2026-10-14 拍板，锁定）
| # | 决定 | 说明 |
|---|---|---|
| D1 | **投票 padding 假象仅存档**，不列为研究主线、不作为独立 baseline 交付 | 现象（掩码恒为 {0..50}/{205..255}）写入本文档即可；论文不展开 |
| D2 | **block0 强制保留 = 采用**（**BERT 特供**，OPT 路径不动） | 所有剪枝层（2/3/4）、topk 之后强制 union block0；理由见 §3；**影响所有后续 sparse 臂基线** |
| D3 | fixed-CATV 对照：**先 a03 一臂**（L8D8 / lr=1.4e-3 / 30 轮 ≈7.4h），再定 a01 | 修复落地后执行 |
| D4 | a03 双臂的**停机由用户处理**（不是本文档/修复 agent 的施工内容） | 产物保留为"含缺陷臂"存档，禁用于论文 |
| D5 | **投票归一化规格定稿**：**P0a=逐块有效均值（NaN-aware）**；**覆盖资格（P0b）经数据检验取消**（最坏块有效样本 83/240，均值可用）；打分边界按块百分比 `k=floor(catv_r·Ne)`（无资格过滤 ⇒ Ne=256 ⇒ 维持 51/51）；**P1=两阶段 client 聚合，核查后确认已实现**；**P2（q 侧归一）不做** | 见 §4.1；依据 2026-10-14 的 padding/覆盖实测 |

## §1 缺陷机理（vote padding 假象，代码级）
1. `src/models/modeling_distilbert_smell.py::DistilBertSdpaPruneAttention.forward`：
   ```python
   scores = scores.float().sum(dim=-2)
   block_keep = keep_mask.view(bsz, n_blocks, pool).amax(dim=-1)
   scores = scores.masked_fill(block_keep < 0.5, torch.finfo(scores.dtype).min)  # pad 块 = -3.4e38
   vote_callback(self.layer_idx, scores.detach())                               # 占位符随分数进入投票
   ```
2. `src/fed/serial_fedavg.py::_VoteAccumulator`：按层累计 `values.sum(dim=0)` 后取均值 → 被 −3.4e38 主导：
   某块在任一篇里 pad 一次 → 均值 ≈ −1.4e37；≥2–3 次 → float32 溢出 `-inf`。有效分 ~1e3–1e4，相差 34 个数量级。
3. `compute_consensus_mask` 的 `argsort(descending, stable=True)` → 顺序 = [有效块按分数降序] → [−1.4e37 组按索引升序] → [−inf 组按索引升序]：
   - 前 51（锚）= 永不 pad 前缀（≤51 块）+ 并列组最小索引补齐 → **恒为 {0..50}**；
   - 后 51（反锚）→ **恒为 {205..255}**；与真实分数无关。
4. 触发条件 = 每轮投票的 240 篇里存在 ≥1 篇 < 51 块（3264 token）。a01 cycle 实测每轮最短 1029–3238 token（16–51 块），**30/30 轮满足** → 掩码 30 轮 Jaccard=1.000。

## §2 复现证据（可照抄核验）
- **精确复现**：同轮真实 240 篇重算投票 + 同款排序 → r0/r1、三层 **锚 51/51、反锚 51/51** 完全一致（对比 `logs/fed/bert_da/r30_a01_catv_on/a01/consensus_mask_round{0,1}.pt`）。
- **占位符实测**（24 篇模拟，layer2）：`block25 pad=0 avg=7.2e3 | block30 pad=1 avg=−1.4e37 | block255 pad=14 avg=−inf`。
- **OPT 对照**：`logs/fed/a01_zoo_k24_d32_catv_lr0.30_r30` 掩码缓慢漂移（+inf J=0.82–1.0、−inf J=0.79–0.96）——OPT 投票路径无 pad 补丁，是真分数驱动。**OPT 结论不受影响。**
- **效应侧（r0，同权重；free=1.8060/0.3860）**：
  | 评测选择 | NLL | acc |
  |---|---|---|
  | 只 +inf 锚（=保前 51 块） | 1.5367 | 0.4840（≈全部收益 −0.269） |
  | 只 −inf 反锚 | 1.7203 | 0.4060 |
  | 只 +inf block0 | 1.6917 | 0.4240（占 42%） |
  | 完整掩码 | 1.5424 | 0.4740 |

## §3 block0/[CLS] 论证（D2 的依据）
- **`[CLS]` 是唯一 readout**：分类头只读 `last_hidden_state[:, 0]`（layer5 输出，768 维单向量）；encoder 冻结 + LoRA 微调 ⇒ 分类适应全部经此通道。中间层 `[CLS]` 状态不参与读出。（BERT 系传统；NSP 头当年也读它，DistilBERT 无 NSP 但保留输入格式。）
- **位置布局**：输入 `[CLS] + tokens + [SEP]`，`[CLS]` 在位置 0；块 64 → **block0 = [CLS] + 前 63 个 token（标题/摘要）**；64 粒度下"保 `[CLS]`"与"保头部内容"不可分。
- **剪枝语义**（共享子集）：入选块同时当 Q 与 K/V；block0 被丢时，该层 `[CLS]` 不做 query（无更新）也不做 key（他人读不到）；layer5 豁免（dense）会兜底重新混合，代价 = 少一层混合 + layer5 读到旧一层状态。
- **实测**：free 选择下 layer2/3 六样本全保留、**layer4 五样本丢 5 次**；只强保 block0（2/3/4）回收 r0 掩码收益的 42%。predictor 的目标是通用注意力池化（ReLU qk 池），对 `[CLS]` 无偏置，故会丢它。
- **结论**：BERT 分类特供策略 = **所有剪枝层强制保留 block0**（keep 预算不变，挤掉最低分入选块）；与 D1 的假象解耦（假象是 vote 记账 bug，block0 是剪枝策略选择）。

## §4 修复规格（交给修复 agent；**分批提交，勿与运行中任务冲突**）
### 4.1 vote padding 修复 + 归一化规格（P0；`serial_fedavg.py` + `modeling_distilbert_smell.py`）
#### 4.1.1 P0a：逐块 valid-mean（NaN-aware）
1. BERT 投票出口：给 callback 传**独立副本**，pad 位置置 NaN（topk 仍用 `finfo.min` 的 `scores`）：
   ```python
   if vote_callback is not None:
       vote_payload = scores.masked_fill(block_keep < 0.5, float("nan"))
       vote_callback(self.layer_idx, vote_payload.detach())
   ```
2. `_VoteAccumulator`：`__call__` 改 `valid = ~torch.isnan(values)`；`sums += torch.nansum(values, dim=0)`；`counts += valid.sum(dim=0)`（counts 由标量改**逐块向量**）；`average()` 改 `sums / counts.clamp(min=1)`（零覆盖块→0 + WARN）。
3. **OPT 路径零改动**（无 NaN → `nansum==sum`、counts 全等 → 逐位等价），selftest 锁住等价性。

#### 4.1.2 P0b：覆盖资格——**经数据检验后取消**（2026-10-14 用户决定）
- 原动机：防止"有效样本数极小的块"靠运气进榜。**数据检验否定了其必要性**：
  - 实测前向级有效计数（30 轮平均，240 forwards/轮）最低也有 **83/240**（block 255），典型尾部块 83–118，中部 ≥166，头部 ≈240；
  - 参考系：满长样本（16384 token，填满全部 256 块）占比 34.5%（client train）/ 34.7%（emb global_train）/ 39.8%（predictor warmup）⇒ 块 255 的有效样本永远 ≈ 满长文档占比×240；
  - 结论：**不存在"样本数少到均值不可用"的块**（std 比最优覆盖块 ≤1.7×），逐块有效均值（P0a）已足够，无需资格阈值。
- 附（保留作历史证据）：client 级覆盖饱和（每 client 8 篇中≥1 篇满长 ⇒ 所有块 client 覆盖 ≥27/30），τ_c 类阈值无区分度；前向覆盖曲线 = 长度分布生存函数：

  | block | 0 | 50 | 91 | 100 | 150 | 199 | 200 | 250 | 255 |
  |---|---|---|---|---|---|---|---|---|---|
  | 前向覆盖 | 1.000 | 0.982 | 0.892 | 0.870 | 0.693 | 0.493 | 0.489 | 0.359 | 0.347 |
  | 有效样本 n | 240 | 236 | 214 | 209 | 166 | 118 | 117 | 86 | 83 |

- **规则**：不做覆盖过滤；仅保留零覆盖兜底（counts=0 → 该块 mask=0 + WARN；本数据不存在此情况）。

#### 4.1.3 P0c：打分边界（**合格块百分比的退化形式**）
- 用户规则："打分边界在 BERT 体系按块百分比"：`k = max(1, floor(catv_r × Ne))`，其中 `Ne` = 参与排序的块数；
- 由于 P0b 取消（无资格过滤），**Ne ≡ N = 256** ⇒ **k = floor(0.2×256) = 51，即维持既有 51/51 预算**（公式保留以便审计/未来数据变更时复用）；
- 在全部块内取 top-k → +inf、bottom-k → -inf，其余 0；`r<s`、`s+r≤1` 校验语义不变。

#### 4.1.4 P1：两阶段聚合——**核查结论：已实现，无需改动**
- 现状链路：`ClientRunner` 内 `_VoteAccumulator` 先对本 client 的前向取均值 → `result["votes"]` → 服务器 `accumulate_votes` 对 30 个 client 的均值**求和** → `compute_consensus_mask` 排序。
- 即"client 均值 → client 间等权（和=30×均值）"，**与 FedAvg 的 client 等权语义一致**，排名对常数因子不变；P1 的唯一前提（逐块有效计数）由 P0a 提供。
- 修正此前的错误表述：不存在"forward 池化"问题。
### 4.2 block0 强制保留（`modeling_distilbert_smell.py`，BERT 特供）
- 在 `_, idx = torch.topk(scores, keep_blocks, ...); idx = idx.sort(dim=-1).values` **之后**、展开 `expanded` 之前：
  ```python
  # SMELL 3 modeling_distilbert_smell keep_block0 ADD — BERT 特供：block0 载 [CLS]（唯一 readout），强制保留
  if bool(getattr(self.config, "keep_block0", True)):
      miss = (idx == 0).any(dim=-1) == False                 # (bsz,) 本轮未含 block0 的行
      if bool(miss.any()):
          idx = idx.clone()
          # 挤位规则：在已入选块中丢“分数最低”者（而不是按块号取末列——idx 是按块号升序的）
          sel = scores.gather(1, idx).masked_fill(idx == 0, float("inf"))   # block0 自身不可被丢
          idx[miss, sel[miss].argmin(dim=-1)] = 0
          idx = idx.sort(dim=-1).values
  ```
  说明：`scores` 此处仍含 pad 占位（−3.4e38），短文档非 pad 块不足时占位块会被优先挤掉（无害）；keep 预算不变。
- 默认 `config.keep_block0 = True`（在 `load_distilbert_classifier`/`patch_distilbert_sdpa_prune` 设置，写入 run `config.json` 供复现）；`last_prune_stats` 增加 `block0_kept`（与 `block0_forced`）。
- 生效位置在 consensus_mask + topk 之后（即使掩码把 block0 置 −inf 也会被救回）；对 dense（`sparse>=1`）/未剪层无影响；**OPT 路径不动**。
### 4.3 测试/验证
1. 单测：VoteAccumulator NaN 均值（含"逐块有效计数不同"的用例）+ 无 NaN 等价（逐位）；block0 强制（构造 mask 把 block0 置 −inf → 断言 kept 含 0）；零覆盖兜底（构造全 pad 块 → 断言 mask=0 不参与 top/bottom）；
2. 回归：`modeling_distilbert_smell` 16 项 selftest、`serial_fedavg._selftest`、`py_compile`；
3. 复现测试：真实 240 篇重算投票（同一 round）→ 断言掩码 ≠ {0..50}/{205..255} 且 r0 与 r1 不再相同；输出锚/反的位置分布（预期不再纯前缀/后缀）与逐块有效计数（预期 83–240）；
4. **两版掩码对照**（同一份 round-0 数据 + 同一 free/mask 评测口径）：旧实现 vs P0a（+block0）→ 掩码差异 + 评测 NLL；
5. 覆盖率/有效计数诊断落盘：逐块 counts 直方图与零覆盖块计数（写入 round 记录或独立诊断 json）；
6. block0 诊断三连（500 样本，均为纯评测/回放，无训练）：
   a. 分层归因：只强保 layer4 的 block0 vs 只保 layer2+3；
   b. 相关性：per-sample "layer4 是否保 block0" × free 口径 per-sample NLL；
   c. free 选择位置画像：每层 kept-102 的索引分布（头部占比/均值索引）。

## §5 影响与论文口径
- 本 run（a01 双臂）的 "CATV" 实质 = **固定头部前缀先验（{0..50}）**，**不是 client-voted consensus**；其部署收益约 42% 来自"顺带补了 [CLS]/头部"。修复（4.1+4.2）后：
  - 掩码变成真·分数驱动（预期会随轮漂移）；CATV 增益预期收缩到"真共识选块"部分；
  - free 口径的 block0 缺陷被 4.2 消除，所有 sparse 臂基线改变（**这是 D2 的主要动机：让对照干净**）。
- a01 已观测结论（同协议训练效应、free 迁移截断）在"前缀先验"解释下仍成立，但叙述必须改；a03 双臂（含缺陷）只作废臂存档。

## §6 当前状态（2026-10-14 13:52；修复 agent 勿动运行中任务）
- `temp/run_bert_d_a_chain_bg.sh`（pid `temp/logs/last_bert_da.pid`）仍在跑：`r30_a03_catv_off` **28/30**（06:46 起）；其后排有 `r30_a03_catv_on`（未开始）。
- **D4：停机由用户处理**。若 a03_off 跑完，产物留档；`temp/logs/bert_r30_report.txt` 为自动报告（含 a03 数据）。
- a01 双臂产物：`logs/fed/bert_da/r30_a01_catv_{off,on}/`；交叉评测 `temp/eval_{off,on}{00,09,19,29}_*.json`；掩码变体 `temp/mask_{anchor_in_only,anchor_out_only,block0_only}.pt`。
- 修复完成后：按 D3 跑 fixed-CATV（a03 一臂）；报告口径按 §5。

## §7 变更列表
| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10-14 11:18 | 新增 | 本 ailog v1 | vote 假象确认 + 修复交接 |
| 10-14 13:55 | 修订 | 本 ailog v2 | 并入 D1–D4 决定 + §3 block0 论证 + §4.2 block0 规格 + §6 状态 |
| 10-14 15:20 | 修订 | 本 ailog v3 | D5：P0+P1 定稿、P2 不做；§4.1 升级（P0a valid-mean / P0b 前向级覆盖资格 τ_f=0.5（数据表）/ P0c 合格块百分比预算；P1 核查=已实现）；§4.3 加四版掩码对照与覆盖诊断 |
| 10-14 15:45 | 修订 | 本 ailog v4 | **P0b 覆盖资格取消**（数据：最坏块有效样本 83/240，无需阈值）；P0c 退化为原 51/51；§4.3 改两版对照；D5 更新 |
| 待办 | — | 代码修复由修复 agent 执行（4.1 P0a + 4.1.3 退化预算 + 4.2 + 4.3） | 不影响运行中任务 |
