# Ailog 260924-114152 — SMELL-v3: 16k 长上下文适配（B/C）落地 + warmup 池交付与后续流程

## 概述

按作者确认的 B/C 对照方案，交付两件代码 + 两套数据：(1) `src/train/longctx_adapt.py`（BP warmup：`pos_only` 只训位置嵌入 / `pos_lora` 位置嵌入+LoRA qkvo r=8，`interpolate|duplicate|jenga_dup_scaled` 三种初始化，含 `ppl_full/ppl_tail` 诊断与 checkpoint）；(2) `src/data/build_warmup_16k.py` + `dataset_v3/discovery_16k/{a01,a03}/warmup_input_ids.npy`（各 512×16384，与 client 池/G-PPL 演示池完全不相交）。本机 smoke：B/C 两臂 2k 通过、4k（grad-ckpt on）也能在 8GB 上跑（step ~1.4s）；16k 正式训练待云端 A40。另发现并规避了一个 Jenga 模型的 BP 梯度污染问题（act-pack hooks）。文末给出**交付表**与**端到端流程**（训位置→训 predictor→BP 可行性→ZOO 消融）。

## 变更列表

| 时间 | 操作 | 文件 | 位置/规模 | 影响 |
|---|---|---|---|---|
| 11:38 | 新增 | `src/train/longctx_adapt.py` | 396 行 | longctx warmup：冻结校验（pos_only=16,779,264 / pos_lora=18,352,128 参数）、metrics.jsonl（loss/grad_norm/ppl_full/ppl_tail）、`pos_embed.pt`+`adapter/`+`config.json` 落盘 |
| 11:37 | 新增 | `src/data/build_warmup_16k.py` | 238 行 | 复用主 builder 模板/TokenCache/RoundRobinDemoPool；从 train 池剔除 client train/local/G-PPL demo 后采样；产出 `warmup_input_ids.npy`+meta |
| 11:38 | 构建 | `dataset_v3/discovery_16k/{a01,a03}/warmup_*`（gitignore，各 16MB） | 512×16384 uint16 | a01 draws=139,385 unique=139,385 overlap=0/0/0；a03 draws=139,371 unique=139,371 overlap=0/0/0 |
| — | 发现 | `src/models/modeling_opt_smell.py:509` | — | Jenga `saved_tensors_hooks` 只保存前一半 relu 输出、反向补零 ⇒ **BP 第二半 token 梯度被静默污染**；trainer 以 `--act-pack off`（默认）monkeypatch 为 identity 规避 |

## 验证结果（本机 8GB）

- `py_compile` 两文件通过。
- **B 臂**（`--mode pos_only --truncate 2048 --steps 4`）：trainable 仅 `model.decoder.embed_positions.weight`；step loss 6.04→6.21，`ppl_full` 418.4→393.1。
- **C 臂**（`--mode pos_lora`）：trainable = pos-emb + 192 个 `lora_*`；step loss 6.04→6.21，`ppl_full` 418.4→372.8。
- **4k**（`--truncate 4096 --grad-ckpt`）：**在 8GB 内跑通**，step 1.4s，`ppl_tail` 跟随优化（390.3→387.9）。16k 仍需 A40。
- 注：smoke 的 PPL 数值高是"随机 predictor + sparse 0.4"所致（见 260924-103741），不是位置方案的评价。

## 交付表

| # | 交付物 | 位置 | 状态 | 说明 |
|---|---|---|---|---|
| 1 | Discovery 16k 分片（α=0.1/0.3） | `dataset_v3/discovery_16k/{a01,a03}`（gitignore） | ✅ | 30×100 训练 + 30×16 local + 500 G-PPL；checker PASSED |
| 2 | warmup 池（16k） | 同上 `warmup_input_ids.npy` | ✅ | 512 条/套；与 client/G-PPL/test 零重叠 |
| 3 | 位置扩展模块 | `src/models/position_embed.py` | ✅ | interpolate / duplicate / jenga_dup_scaled |
| 4 | 长上下文适配器（B/C） | `src/train/longctx_adapt.py` | ✅ smoke（2k/4k） | 16k 待 A40；判据 G-PPL<50、tail 不劣化 |
| 5 | CATV（r<s 约束） | `src/models/token_selector.py` + `modeling_opt_smell.py` + `src/fed/*` | ✅ | IR/vote_bytes/anchor 指标、逐轮掩码落盘 |
| 6 | predictor 训练脚本 | `src/train/train_predictor.py` | ⏳ TODO | 移植 Jenga `modeling_opt_train_predictor.py`+`opt_la.py`；冻结 base、400 步、16k；输出 `predictor.pth`+`pruned_config` |
| 7 | 主线 BP 可行性 runner | `src/fed/run_fed.py --trainer bp`（待加） | ⏳ TODO | 现仅 ZOO；BP 用于机制去风险（c=2~3） |
| 8 | 云端 bootstrap | `scripts/setup_server.sh` | ⏳ TODO | conda env + requirements + FA wheel + hf-mirror 数据重建 |
| 9 | 消融编排（4 并发） | `scripts/run_ablation_cloud.sh` | ⏳ TODO | 每实验 1 卡、30 client 串行；4 卡跑 α×CATV 矩阵 |

## 接下来的流程（云端开展）

```
[Step 1 · A40] 位置适配两方案（数据用 warmup_input_ids.npy）
 ①B: python src/train/longctx_adapt.py --tag a01 --mode pos_only --pos-init interpolate --lr 1e-3 --steps 500 --eval-every 50 --gpu 0
 ①C: python src/train/longctx_adapt.py --tag a01 --mode pos_lora --pos-init interpolate --lr 2e-4 --steps 500 --eval-every 50 --gpu 0
   判据: 16k ppl_full < 50（理想 25-35）、ppl_tail 不劣化、2k 回归不超 10%
   ↓ 选定方案，产出 pos_embed.pt（+ adapter/）
[Step 2 · A40] 训 predictor（冻结 base，400 步，16k；数据用 warmup+训练池文本）
   → predictor.pth + pruned_config.pth → 接入 JengaSparse/CATV（CATV 才有意义）
   ↓
[Step 3 · 本机/云端] 主线 BP 可行性（BP+LoRA，c=2~3、少量轮）
   验证 sparse + CATV + LoRA 的 loss/G-PPL 能下降；同时校准 ZOO 步长（当前 lr=1e-3 第 2 轮 NaN）
   ↓
[Step 4 · A40×4 并发] 主线消融 (α∈{0.1,0.3}) × (CATV off/on)，c=30、ZOO+LoRA、16k
   4 个脚本并发（每个 1 卡、30 client 串行），每轮记 G-PPL(全文/answer)、IR、‖Δ‖、通信字节
   ↓
[Step 5] 预算内调参（lr/eps/directions）与论文图表
```

- 数据/权重/checkpoint 不进 git：**warmup 池可用 `build_warmup_16k.py` 在云端重建**，`pos_embed.pt`/`adapter/`/`predictor.pth` 由云端重训或单独传输。

## 风险 / 遗留

- **act-pack 隐患**：`modeling_opt_smell.py:509` 的 hooks 会污染 BP 梯度（ZOO 纯前向不受影响）；当前靠 trainer 的 `--act-pack off` 规避，后续应在模型副本内正式修掉并加 SMELL 标记。
- opt-350m 的 predictor 需按新位置方案复训，否则 CATV 投票仍是噪声。
- 16k BP 本机放不下（4k 可以）；正式实验一律 A40。
