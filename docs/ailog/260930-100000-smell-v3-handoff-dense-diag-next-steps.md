# Ailog 260930-100000 — Phase N: HANDOFF（TD-3@a03 成功 + dense 对照发现；下一步 1→2→E2/E3）

> **交接对象：下一个 session。** 本机：AutoDL 4090（`/root/smell/SMELL`，env `~/miniconda3/envs/SMELL/bin/python`）。
> git：`exp/rank-rotation-lora` @ **`421517e`**（clean）；GPU 空闲；**无后台任务在跑**。上一 session 上下文已很长，故在此截断交接。

## 0. 一句话现状

1. **TD-3@a03（lr=0.15 + `--delta-clip 1.0`）30 轮完成**：full PPL 33.654→33.285（**−1.10%，30 点全单调**）——**首个 ZOO 正结果**（配置见 ailog `260930-094100`）。
2. **dense 对照（本 session 新增，关键）**：同一 adapter 在 dense 口径（`sparse=1.0`）下 answer PPL **−12.2%（改善）**，而 sparse 口径（`0.4`+step5 predictor）是 +6.4%（退化） ⇒ **answer 退化源于 frozen predictor 的选块随模型漂移失配，而非训练本身**。
3. 下一步顺序（用户已确认）：**Step 1 选择漂移量化 → Step 2 predictor 刷新（数据隔离！）→ Step 3 E2/E3 跳变诊断**。

## 1. 已完成实验与产物（可直接复用）

| 内容 | 路径 |
|---|---|
| TD-3@a03 run（30 轮） | `logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/`（`metrics.jsonl`、`adapter_round000–029`、`eval_roundNNN{,_persample}.json`）|
| 完成报告（整表+配对） | `temp/logs/td3_a03_lr015_clip_report.txt` |
| **dense 对照评测** | `temp/diag_dense_eval/r000.json`、`r029.json`、`r000_persample.json`、`r029_persample.json` |
| 成功配置 ailog | `docs/ailog/260930-094100-smell-v3-td3-a03-lr015-clip-first-success.md` |
| 修复链 ailog | `260929-000200`（审计 3 修）/ `260929-094100`（D 扫描+clamp 修复）/ `260929-111500`（离群爆炸+delta-clip）|

**两口径 × r0/r29（500 样本 global_test，配对检验 n=500）**：

| 口径 | full PPL | answer PPL | 配对 answer ΔNLL |
|---|---|---|---|
| sparse（`0.4`） | 33.654 → 33.285（−1.10%） | 1743.8 → **1855.0（+6.4%）** | +0.0650（REGRESSED，p≈0） |
| dense（`1.0`） | 26.467 → 26.062（−1.53%） | 3594.0 → **3154.7（−12.2%）** | **−0.1383（IMPROVED，p≈0）** |

- run 期间：**51 个 client-轮被 clip**（1.7/轮，0–6）；无 NaN/爆炸；~40.3 min/轮。
- 离群呈**层位置模式**（早期层大）：r0–1/r6–7（层 0–7，dense）δ_raw 2.4e4–1.7e5；r3–5（层 12–23，sparse）≤~300 ⇒ 支持"早期层扰动经下游离散选择级联放大"，支持 E2/E3。
- 另注一个反直觉现象：**r0 时 sparse 的 answer PPL（1744）远好于 dense（3594）** ⇒ predictor 选 40% KV 在初态是"滤干扰"，随训练漂移退化为"丢关键信息"。

## 2. 下一步实现要点（按 1→2→3）

### Step 1：选择漂移量化（纯诊断，~20 min）
- 目的：同一批样本，在 **r0 vs r29 模型状态**下，predictor 的 kept-KV-block 集合变化。
- 实现（`src/models/modeling_opt_smell.py`）：
  - `_prune_block_scores`（L760）返回 `scores`（predictor 路径下 `predict_attn.sum(dim=-2)`，形状 `(1, n_blocks)`）；monkeypatch 该函数记录 per-layer `scores`；
  - 按 forward 规则（L811–823）复现 idx：`layer_idx < num_layers//2-1` 全留；否则 `q_len_blocks = max(1, min(int(n_blocks*sparse), n_blocks))`，`torch.topk(scores, q_len_blocks).sort()`；
  - 两个 adapter 各前向 N=8 条 global_test（16k，纯读取不训练），dump kept idx → 计算每层 overlap、first-half 占比；可与逐样本 answer ΔNLL 关联。
  - 骨架：新建 `temp/probe_selection_drift.py`；模型加载参数与 `ppl.py` 一致（fp32+sdpa_prune+step5+pos）。
- 判据：r29 的 kept 集合相对 r0 明显漂移（overlap 显著下降 / first-half 占比变化），且漂移大的层与 answer 退化相关。

### Step 2：predictor 刷新（**数据隔离硬约束**）
- **必须沿用 predictor 块训练当时的数据集**（用户明确）：只用 client train 分片，**不得用 global_test/local_test/warmup 训练**：
  ```
  $PY src/train/train_predictor.py --gpu 0 \
    --data-glob 'dataset_v3/discovery_16k/a01/clients/client_*/train_input_ids.npy' \
    --data-glob 'dataset_v3/discovery_16k/a03/clients/client_*/train_input_ids.npy' \
    --adapter logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/adapter_round029 \
    --pos-init interpolate --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
    --sparse 0.4 --lr 2e-5 --steps 400 --prune-interval 100 --prune-until 620 \
    --eval-every 50 --eval-samples 4 --seed 42 \
    --save-dir checkpoints/predictor/refresh_r29_a01a03
  ```
  （除 `--adapter` 外与 step5 对齐；对 r29 状态做刷新）
- 评测闭环：
  ```
  $PY src/eval/ppl.py --tag a03 --split global --dtype fp32 --attn sdpa_prune --sparse 0.4 \
    --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
    --adapter logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/adapter_round029 \
    --predictor checkpoints/predictor/refresh_r29_a01a03/predictor.pth \
    --pruned-config checkpoints/predictor/refresh_r29_a01a03/pruned_config.pth \
    --per-sample-out temp/diag_refresh/r029_refresh_persample.json --out temp/diag_refresh/r029_refresh.json
  ```
- 对比目标：step5 口径 **1855.0** / dense **3154.7** / r0 sparse **1743.8**。若 refresh 后回落至 ~1700–1800 → 闭环，写"周期性 predictor 重适配"进方法；否则查 refresh 训练质量（mask loss）或考虑训练中周期性刷新。
- 完成后：sha256 记入 `docs/weight-manifest.md`。

### Step 3：E2/E3 跳变诊断（~30 min）
- **E2**：在 r29 状态、同 client、同方向种子下比较 `sdpa_prune(0.4)` vs `sdpa_prune(1.0)`（或 `sdpa`）的 `zo_grad` 范数；候选离群 client：r24 `client_21`（δ=1.68e5）、r25 `client_10/13`、r26 `client_09/15/29`（2.6e4–1.1e5）。
- **E3**：复用 Step 1 的 monkeypatch，记录 θ+hv 与 θ−hv 两次前向的 kept-block 集合差异（翻转数），与 |ΔL| 相关；若翻转数↔|ΔL| 强相关 ⇒ 有限差分跳变伪影坐实（解释 delta-clip 的必要性）。

## 3. 必须遵守的前置/口径（踩过的坑）

- **5 项前置修复**缺一不可：ZOO 强制 eval / fp32+SDPA / step5+缺 bias 零填充+评测接线 / clamp 放宽 / delta-clip（详见 `260929-000200`、`260929-094100`、`260929-111500`）。**历史 G-PPL 与 260929 前所有 ZOO 结果全部作废**，勿引用。
- 评测口径：训练一致 = `--dtype fp32 --attn sdpa_prune --sparse 0.4 --predictor/--pruned-config`；dense 对照 = `--sparse 1.0`（predictor 可传可不传）。
- **answer 指标是 teacher-forced NLL，不是 accuracy**；"argmax 猜词正确率"评测是待办（可给 `ppl.py` 加 logits/argmax 统计）。
- `delta-clip` 默认 1.0（`run_fed`），`clipped_clients`/`delta_clip_applied` 在 metrics；lr clamp `[1e-7,1.0]` + 触界警告。
- 长任务按仓库约定：`setsid nohup` + `temp/logs/last_*` + `metrics.jsonl`（driver.log 有块缓冲）。
- 云端 A40：仍需停止旧 TD-3、pull 本分支（含全部修复+step5），按 `260930-094100` §一 重跑——由用户处理。

## 4. 交接清单（下个 session 起步顺序）

- [ ] 读本 ailog + `260930-094100` + `260929-111500` + `260929-000200`；确认 GPU 空闲、git @ `421517e`。
- [ ] Step 1：`temp/probe_selection_drift.py`（r0 vs r29，8 条 global_test，~20 min）→ 汇报 overlap/first-half 与逐样本关联。
- [ ] Step 2：refresh 训练（数据隔离：a01+a03 client train 分片 + r29 adapter，400 步，~25 min）→ 评测 r29 sparse answer（~10 min）→ 与 1855 / 3154.7 / 1743.8 对比。
- [ ] Step 3：E2/E3（同 monkeypatch 思路）。
- [ ] 收尾：新 ailog（含 refresh sha256、数据隔离说明与结论）；必要时更新 `docs/weight-manifest.md` 与 `AGENTS.md`。
