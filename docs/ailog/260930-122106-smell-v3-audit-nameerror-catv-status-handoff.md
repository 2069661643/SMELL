# Ailog 260930-122106 — Phase N: 审计 `exp/rank-rotation-lora` @ `0cf218f`（run_fed NameError 必修 + 风险清单 + CATV 实现现状）

> 本 ailog 由 **WSL checkout 的审计 session** 产出，**交接给下一个执行 session**（AutoDL / amax 均可）。
> 审计对象：`exp/rank-rotation-lora` @ `0cf218f`（260930 11:16，已 fetch 最新；工作树 clean）。
> 方法：10 个新提交（`90b1fe3..0cf218f`）逐文件 diff 阅读 + 19/19 `py_compile` + 4 项 selftest +
> 4 个脚本 `bash -n` + 1 条实跑复现（2k smoke，本机 8GB）。
> 现场处置：本 checkout 原在 `exp/distilbert-16k`，已切到本分支；切分支时发现 41 文件「CRLF 行尾噪声」
> （逐文件 `tr -d '\r'` 后与 HEAD 逐字节一致，41/41 无内容差异），已 stash→pull→drop。根因：Windows 侧
> git `core.autocrlf=true`、WSL 侧未设。**本审计未改动任何代码或权重。**

## 0. 一句话结论

1. **必修（P0）**：`src/fed/run_fed.py` **不带 `--predictor` 必定 `UnboundLocalError` 崩溃**（已实跑复现）——
   文档化的冒烟命令、BP/dense 无 predictor 运行全被堵；修复 = 一行。
2. 上一轮修复链本身（ZOO 强制 eval / fp32+SDPA / predictor bias 零填充 / `--delta-clip` / save-eval 解耦）
   经审计**未发现新问题**，逻辑与落盘字段核对一致。
3. **CATV 只接在 `flash` 路径**（`OptFlashAttention2`），在当前主力协议（fp32 + `sdpa_prune`）下被 fail-fast
   硬禁用 ⇒ **α×CATV 四点消融的 CATV 两臂目前跑不了**；补齐约 6 行 + 评测掩码参数（见 §四）。
4. `scripts/run_ablation_cloud.sh` 的 `LR=1e-7` 是 9/24 旧标定，且为全参 ZO、无 eval/save 参数，
   与 TD-3 成功配置（k4 rotate4 / lr 0.15 / clip 1.0）**非同一口径**——照原样启动大概率产出无效 run。

## 一、P0：`run_fed.py` 不带 `--predictor` 必崩（已复现）

| 项 | 内容 |
|---|---|
| 位置 | `src/fed/run_fed.py`：`predictor_loaded = 0`（**L468**）；`predictor_zero_filled` 只在 `if args.predictor:` 内赋值（**L479**）；却无条件写入 config（**L583**） |
| 引入提交 | `5b2af10`（"fix ZOO dropout probes, wire trained predictor into eval (zero-fill bias), step5 client-dist predictor"） |
| 触发条件 | 任何**未传 `--predictor`** 的 `run_fed.py` 调用；崩在写 `config.json` 处（训练开始前，fail-fast ~3min 内） |
| 不受影响 | `scripts/run_k4_td2_td3.sh`、`scripts/run_ablation_cloud.sh`（两者总传 `--predictor`）；`ppl.py` 的加载器（局部变量在块内使用） |
| 修复 | L468 后补一行 `predictor_zero_filled = 0`（或改 `run_config` 用 `locals().get`，二选一，推荐前者） |

实跑复现（本机 8GB，2k、1 client；修复前必崩）：

```bash
$PY -u src/fed/run_fed.py --tag a01 --gpu 0 --trainer zoo --catv off \
  --sparse 0.4 --truncate 2048 --rounds 1 --local-steps 1 --zo-directions 2 \
  --max-clients 1 --max-train-samples 1 --dtype fp32 --attn sdpa_prune \
  --out-root temp/audit_smoke_nopredictor
# 实际输出（模型加载成功后）：
#   File "src/fed/run_fed.py", line 583, in main
#     "predictor_zero_filled_tensors": predictor_zero_filled,
#   UnboundLocalError: local variable 'predictor_zero_filled' referenced before assignment
```

修复后验证建议：同命令（不带 predictor）跑通 1 轮；再带 `--predictor/--pruned-config`（step5）跑 1 轮，
确认 `predictor_zero_filled_tensors` 与 `zero_filled=144` 正常落盘。

## 二、任务进度快照（截至 260930 12:21，供执行 session 对齐）

| 项 | 状态 |
|---|---|
| TD-3@a03（lr=0.15 + clip 1.0，30 轮） | ✅ 首个 ZOO 正结果：full PPL 33.654→33.285（−1.10%，30 点全单调，配对 p≈0）；**caveat：answer NLL +6.7% 单调退化**；51 client-轮被 clip；~40.3min/轮 |
| dense 对照（同 adapter，`--sparse 1.0`） | answer PPL **−12.2%** vs sparse **+6.4%** ⇒ 口径×训练状态交互 |
| Step 1–3（漂移/刷新/E2E3） | 选块漂移小（overlap 0.966–0.996）；r29 刷新仅 −0.67%（未闭环）；E2/E3 无翻转、无重尾 ⇒ 主因候选回到**训练目标 token 均值淹没 answer（~3/16384）** |
| 微扫描 → 全协议 pilot | 已启动 2 臂（`P_D16_L2_lr0.30_c2`、`P_D11_L2_lr0.30_c2`，260930 11:15，pid 132351，AutoDL；各 ~3–4h，跑两个）→ **以 AutoDL 现场 `temp/logs/last_pilot_full_dir.txt` 为准** |
| 云端 A40 | 旧 TD-3 未停、新协议未重跑（待用户处理） |

## 三、风险 / 口径清单（P1–P3）

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| R1 | `scripts/run_ablation_cloud.sh:10` | `LR=1e-7` 为 9/24 旧标定（当时 ZOO 还在 dropout+bf16 病理下）；且**全参 ZO**（无 `--zo-subspace layers --zo-layer-rotate`）、无 `--eval-every/--save-every` ⇒ 与 TD-3 成功配置不同口径 | 启动前先按 `260929-094100` 同法重标定（或直接改为 k4 rotate4 口径）；把结论写进脚本头注释 |
| R2 | `run_fed.py:372-375`、`run_ablation_cloud.sh:76-80` | CATV×非 flash 硬 fail-fast ⇒ 消融 CATV 臂不可用 | 见 §四移植（移植后放宽 guard） |
| R3 | `src/eval/ppl.py`（L48 附近，无 catv 参数） | 评测侧不加载共识掩码 ⇒ 即使 flash 下也是「训练有掩码、评测无掩码」 | 增 `--catv-mask PATH`（run 目录已有 `consensus_mask_roundN.pt` 可直接载入 `config.consensus_mask`） |
| R4 | `run_fed.py:302-304` | eval 复用已存在 adapter：**同 out-root 重启 + `--save-every 0`（或 >1 且轮次错位）时会静默评旧权重** | 复用条件收紧为 `save_every>0 and (round_idx+1)%save_every==0`；或只认本轮 `event:save` 记录 |
| R5 | `src/eval/ppl.py:48` | `--attn` 集合不含 `sdpa_gather`（run_fed L45 有）⇒ `--attn sdpa_gather --eval-every>0` 评测子进程报错 | 补 choice（或 run_fed 侧禁止该组合） |
| R6 | `run_fed.py:56-58` | `--delta-clip` 默认 **1.0**，对 BP/dense 老口径是隐式行为变化 | 文档标注；复现老口径显式 `--delta-clip 0` |
| R7 | `run_fed.py:713` | `delta_norm_clipped_mean` 实为「裁剪后**全体** client 均值」，非「被裁者均值」（命名误导） | 改名/加注释（低优先） |

## 四、CATV 实现现状（审计重点）与移植清单

### 4.1 组件清单（HEAD `0cf218f`）

| 组件 | 位置 | 状态 |
|---|---|---|
| `compute_consensus_mask`（±inf 锚掩码；校验 r>0、r<s、s+r≤1、k=floor(rN)≥1；stable tie-break） | `src/models/token_selector.py:115-156` | ✅ selftest 覆盖 |
| `mask_intersection_rate`（IR） | `:159-165` | ✅ |
| `CATVSelector.apply/set_mask` | `:169-189` | ✅ 接口保留（实际掩码在注意力内部生效） |
| 客户端投票累积 `_VoteAccumulator`（层 `num_layers//2-1 .. num_layers-2` = **11–22**，即被剪层；batch 行均值） | `src/fed/serial_fedavg.py:31-62` | ✅ `try/finally` 安装/卸载，不跨 client 泄漏 |
| 服务器聚合 `ServerAggregator.accumulate_votes`（跨 client 求和；可选 per-layer sum 归一化） | `:229-238`；`run_fed.py:667-676` | ✅ |
| 掩码逐轮生成/落盘/指标（`consensus_mask_roundN.pt`、`anchor_in/out`、`vote_bytes`、`ir_mean/min`；**round 0 无掩码**） | `run_fed.py:733-767`；参数校验 `:363-375` | ✅ |
| **注意力接线（唯一实现点）**：投票取掩码前原始分；±inf 加在 topk 之前；末层豁免 | `src/models/modeling_opt_smell.py:321-336`（`OptFlashAttention2`） | ✅ |
| `OptSdpaAttention` / `OptSdpaPruneAttention` / `OptSdpaSparseGatherAttention` / `eager` | — | ❌ **完全无接线** |
| 评测侧 | `src/eval/ppl.py` | ❌ 无 catv/掩码参数 |

**历史验证状态**：单元 selftest PASS；唯一端到端证据 = 9/24 旧代码 smoke（16k、2 client、2 轮）——
投票→掩码→IR 链路跑通（`ir_mean` 0.92→1.00、`anchor_in/out=51`、`vote_bytes=12288`），但该 run
第 2 轮 NaN（旧 lr=1e-3 + dropout 噪声 + 无裁剪，非 CATV 专属）。**在 260929 修复后的新协议下从未重跑**。

### 4.2 移植到 `sdpa_prune`（解锁消融 CATV 臂）

`OptSdpaPruneAttention.forward`（`modeling_opt_smell.py:813-820`）在 topk 前补与 flash 同序的两步；
`scores` 已是 **float32**、形状 `(bsz, n_blocks)`（`n_blocks = tgt_len // pool_size`，与 `compute_consensus_mask`
的 N、与 flash 路径口径一致），±inf 语义正确：

```python
scores, scored_by = self._prune_block_scores(hidden_states, bsz, n_blocks, pool)
# SMELL 3 sdpa_prune vote_callback ADD — 与 OptFlashAttention2 同：上报掩码前原始分
vote_callback = getattr(self.config, "vote_callback", None)
if vote_callback is not None:
    vote_callback(self.layer_idx, scores.detach())
# SMELL 3 sdpa_prune consensus_mask ADD — ±inf 锚掩码加在 topk 之前
consensus_mask = getattr(self.config, "consensus_mask", None)
if consensus_mask is not None and self.layer_idx in consensus_mask:
    scores = scores + consensus_mask[self.layer_idx].to(device=scores.device, dtype=scores.dtype)
_, idx = torch.topk(scores, q_len_blocks, largest=True, dim=-1)
```

配套项：
1. 放宽 `run_fed.py:372-375` 与 `run_ablation_cloud.sh:76-80` 的 fail-fast（移到「已移植」后）；
2. `ppl.py` 增 `--catv-mask`（载入 `consensus_mask_roundN.pt` → `config.consensus_mask`），`run_global_eval` 转发
   （评测口径用**最后一轮**掩码，需在 ailog 写明）；
3. smoke 判据：2k/16k 1–2 client × 2 轮 —— 断言 vote 非空、mask 文件生成、`anchor_in/out == floor(r·N)`、
   `ir∈(0,1]`、无 NaN；catv off/on loss 曲线无异常；
4. 验收命令（沿用手册口径）：`--dtype fp32 --attn sdpa_prune --sparse 0.4 --catv on --catv-r 0.2 --predictor step5 …`。

**工作量估计**：主路径 ~0.5 天（含 smoke）。`exp/distilbert-16k` 分支的 SDPA 剪枝路径存在**同款缺口**
（其管线 ailog D4 已列为待办）——建议把 hook 抽成小工具/统一模式，两边复用。

## 五、审计验证记录（本机 WSL，Python 3.10.21 / torch 2.8.0+cu128 / jenga OK）

| 项 | 结果 |
|---|---|
| `py_compile`（src+scripts 全部 19 个 .py） | 19 OK / 0 FAIL |
| `src/models/token_selector.py`（CATV 掩码/约束/IR/tie-break/apply） | PASS |
| `src/fed/serial_fedavg.py`（加权聚合/一致性/JSONL） | PASS |
| `src/train/zoo.py`（前向梯度参数索引） | PASS |
| `scripts/paired_eval.py --selftest` | PASS（p=6.17e-85, ratio=0.9792） |
| `bash -n`（4 个脚本） | 全 OK |
| P0 实跑复现 | ✅ 复现（错误见 §一） |

审计脚本（本机 `temp/`，**gitignore 不随仓库走**，如需复现请按上文命令重建）：
`temp/audit_crlf.sh`（行尾验证）、`temp/audit_checks.sh`（环境+compile+selftest）、
`temp/audit_checks2.sh`（AST 粗查）、`temp/audit_repro_nopredictor.sh`（P0 复现）、`temp/audit_final.sh`。

## 六、给执行 session 的建议顺序

- [ ] **P0**：修 `predictor_zero_filled`（一行）→ 按 §一 命令跑「无 predictor / 有 predictor」两个 1 轮 smoke → 记 ailog；
- [ ] **先与用户确认消融协议（R1）**：重标定 lr 或改 k4 口径；协议未定前**不要**启动 4 臂消融；
- [ ] **CATV 移植（§4.2）** + 评测掩码（R3） + smoke → 解锁 α×CATV 四点；
- [ ] 小项：R4（eval 复用条件）、R5（`--attn sdpa_gather`）、R6/R7（标注/命名）；
- [ ] 云端 A40：停旧 TD-3 → `git pull` → 按 `260930-094100` §一 重跑（旧运行含 dropout 噪声 + 随机 predictor，全部作废）；
- [ ] AutoDL：读全协议 pilot（两个 arm）结果，与基线（−0.0117 PPL/轮、−0.0179 PPL/h）对照；
- [ ] 提交规范：`SMELL v3: <English summary>`，勿 amend 已 push 提交；修复带 `# SMELL 3 ... FIXED` 标注。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 12:21 | 新增本 ailog（审计交接） | `docs/ailog/260930-122106-smell-v3-audit-nameerror-catv-status-handoff.md` | 全文 | P0 bug + 风险清单 + CATV 现状/移植清单 |
| 12:21 | （`temp/`，不进 git）审计脚本 5 个 | `temp/audit_*.sh` | — | 行尾验证/静态检查/P0 复现 |

> 本 session 未修改任何代码、权重、数据；仅切换 checkout 分支与清理行尾噪声（见题记）。

## 环境

- 审计机：WSL checkout（`\\wsl.localhost\Ubuntu\home\yangyongbo118\projects\SMELL-v3`，host `YangYongbo118`；
  8GB RTX 5060 laptop，`~/miniconda3/envs/SMELL`，torch 2.8.0+cu128 / transformers 4.45.2 / peft 0.13.2）。
- 仓库：`git@github.com:2069661643/SMELL.git`，分支 `exp/rank-rotation-lora` @ `0cf218f`（clean，与 origin 同步）。
- 本机无 `checkpoints/`（无 pos_embed / predictor 权重），未做权重相关评测。
