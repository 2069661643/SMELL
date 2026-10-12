# Ailog 260930-094100 — Phase N: TD-3@a03（lr=0.15 + delta-clip 1.0）30 轮完成：首个 ZOO 正结果（full PPL 显著下降）

> **成功口径**：a03 30 client × k4 rotate4 sparsity ZOO（fp32+sdpa_prune、D22、ls2、r=1）30 轮，
> **full-text G-PPL 单调下降 33.654 → 33.285（−1.10%）**，r0 vs r29 逐样本配对检验 p≈0。
> （同期 answer NLL +6.7% 的退化另列 caveat 与诊断队列——本 ailog 固定"成功配置"以便复现与云端重跑。）

## 一、精确配置（复现命令）

```bash
PY=~/miniconda3/envs/SMELL/bin/python   # AutoDL：~/miniconda3/envs/SMELL；云端 amax 用其 env
$PY -u src/fed/run_fed.py --tag a03 --gpu 0 --trainer zoo --catv off \
  --dtype fp32 --attn sdpa_prune --sparse 0.4 \
  --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
  --predictor      checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth \
  --pruned-config  checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth \
  --lora-r 1 --lora-alpha 2 \
  --zo-subspace layers --zo-layer-rotate --zo-layer-group 4 \
  --max-clients 30 --local-steps 2 --zo-directions 22 --zo-eps 1e-3 \
  --rounds 30 --eval-every 1 --lr 0.15 --delta-clip 1.0 \
  --out-root logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip
```

| 超参 | 值 | 说明 |
|---|---|---|
| tag / clients | `a03` / 30 | α=0.3 分片，client_00–29 串行 |
| dtype / attn / sparse | fp32 / `sdpa_prune` / 0.4 | bf16 会吞 ZOO 差分；SDPA 提供 fp32 高精度 |
| LoRA | `r=1, alpha=2`（q/k/v/out） | d_eff(k4)=32768 |
| ZO | `subspace=layers, rotate, group=4, D=22, eps=1e-3, local_steps=2` | 每轮 4 层轮转 |
| lr | **0.15** | δ-校准：Δδ_target≈0.38（clamp [1e-7,1.0]）|
| **delta-clip** | **1.0** | 逐 client 聚合前范数裁剪（防重尾离群毒化）|
| 轮数 / eval | 30 / 每轮 | 每轮存 adapter + eval json + per-sample json |

## 二、结果

| rnd | PPL | loss | clipped |  | rnd | PPL | loss | clipped |
|---|---|---|---|---|---|---|---|---|
| 0 | 33.654 | 3.6186 | 2 |  | 18 | 33.449 | 3.7229 | 3 |
| 5 | 33.599 | 3.5187 | 0 |  | 22 | 33.376 | 3.5697 | 1 |
| 10 | 33.536 | 3.5170 | 0 |  | 25 | 33.340 | 3.7177 | 4 |
| 14 | 33.478 | 3.5154 | 5 |  | 28 | 33.287 | 3.5097 | 0 |
| 17 | 33.460 | 3.5145 | 0 |  | **29** | **33.285** | 3.5093 | 0 |

- **full PPL：33.654 → 33.285（−0.369，−1.10%），30 点全单调**；r0 vs r29 配对 ΔNLL=−0.01102（p≈0，ratio 0.989）；
- answer NLL：r0 vs r29 **+0.06498（answer PPL +6.7%，p≈8e-68）**，单调恶化（r5 +0.0082 / r17 +0.0291 / r29 +0.0650）；
- 离群与裁剪：**51 个 client-轮被裁（1.7/轮，0–6）**，多波（r6–7、r13–15、r18、r24–27）；裁剪有效，全程无 NaN、无爆炸；
- 成本：~40.3 min/轮（30 轮 ≈ 20.4h），峰值显存 ~8.3 GiB（含 eval）。

## 三、缺一不可的前置修复（本配置成立的前提）

1. **ZOO 前向强制 eval**（dropout=0.1 会吞差分；audit `260929-000200`）；
2. **fp32 + SDPA**（bf16 损失量化吞差分）；
3. **step5 predictor + 缺失 bias 零填充 + 评测接线**（`--predictor/--pruned-config/--attn/--dtype`）；
4. **lr clamp 放宽**（旧 `[1e-7,1e-4]` 把校准值 0.15 夹低 1500× 致空转；`260929-094100`）；
5. **`--delta-clip 1.0`**：无裁剪时 round 0 即被单离群 client（δ=2.4e4）毒化爆炸（`260929-111500`）。

## 四、Caveat（必须与结果同时引用）

- 同一 run 的 **answer 指标反向退化**（+6.7%，单调、晚期仍在扩大）。候选机制：① 训练目标（16k token 均值）淹没 answer（~3 token/样本）；② predictor 随 LoRA 漂移致选块失配。
- 诊断队列（进行中/待办）：**dense 对照评测（`temp/diag_dense_eval/`，运行中）**、answer argmax accuracy 评测、E2 dense-vs-sparse 有限差分、E3 选择翻转计数、E4 层位置扫描、E5 BP 对照、**BP 对照跑（同协议）**。
- 所有历史 G-PPL 数字（260929 前）协议不同，**不可与本结果直接比较**。

## 五、产物与复现

- run dir：`logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/`（`metrics.jsonl`、`adapter_round000–029`、`eval_roundNNN{,_persample}.json`）；
- 完成报告：`temp/logs/td3_a03_lr015_clip_report.txt`；
- 驱动：`temp/run_td3_a03_lr015_clip_bg.sh`；tracked pipeline：`scripts/run_k4_td2_td3.sh`（`DELTA_CLIP=1.0`、clamp [1e-7,1.0]）；
- 配对检验：`scripts/paired_eval.py --run-dir <dir> --rounds 0,29 --field full_nll_mean|answer_nll_mean`。

## 六、待办

- [ ] dense 对照（跑完）：判断 answer 退化是否与剪枝/选块相关；
- [ ] answer accuracy（argmax）评测：区分"似然下降"与"猜对率下降"；
- [ ] E2–E5 + BP 对照跑；
- [ ] 云端 A40：pull 本分支（含全部前置修复 + step5 + delta-clip），按 §一 命令重跑 a01/a03。
