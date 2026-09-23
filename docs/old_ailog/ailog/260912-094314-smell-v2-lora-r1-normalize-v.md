# SMELL-v2-debug — B1 定论（train/test 全平）+ 方案 A3（LoRA r=1）+ B（归一化 v）

## 概述

承接上一轮：`[EVAL-PPL]` 显示 25 轮 PPL 平躺（mean 45.0, std 0.14）。本次先做**区分实验**，判定「没学习」还是「过拟合」；再落地减参方案 A3 与归一化 v 方案 B。

## 区分实验（新增 `eval_train_vs_test.py` + `eval_ppl.py --split`）

对 samples=20 run 的 checkpoint r1/r7/r13/r19/r25，在同一模型上分别算固定 100 条 **train** / **test** 子集 PPL：

| ckpt | train_ppl | test_ppl | gap |
|---|---|---|---|
| r1 | 52.39 | 38.30 | −14.09 |
| r7 | 52.01 | 38.50 | −13.51 |
| r13 | 52.03 | 38.40 | −13.63 |
| r19 | 52.00 | 38.37 | −13.63 |
| r25 | 52.00 | 38.65 | −13.35 |

**结论**：train 与 test PPL **都平躺** → **不是过拟合/泛化问题，是连训练集都没学进去**。前向梯度 SNR = M/d ≈ 10⁻⁵ → 方向纯噪声，B1 根因钉死。（train>test 的恒定 ~14 gap 是数据 split 难度属性，r1≈基座即存在。）

附带发现：sparse 模型 **batch>1 会崩**（`query_states.view([4,4096,32,128])` 形状错误，`modeling_llama_sparse.py:538`，token elimination 只支持 batch=1）。

## 代码改动（A、B 分开 commit）

### commit `d136e0a`（方案 B：归一化 v）

`forward_training/tc_transformer_trainer_distribute.py`：构造 `v_dict` 后，对**每个可训张量**将扰动 `v` 归一化到单位 L2 范数，把有效步长 `h·‖v‖` 从 ~29 降到 ~`h·√num_tensors`(~0.16)，回到有限差分线性区。
> 注：用**逐张量**而非全局归一化——全局归一化使逐元素扰动 ~1/√d 低于 bf16 ulp 而被抹掉；r=1 时所有张量同为 4096 元素，逐张量 ≈ 各向同性。

### commit `74ce5d9`（方案 A3：可配置 LoRA r/targets/alpha + 脚本）

- `initializer.py`：新增 `--lora_r`(默认8) / `--lora_alpha`(默认0→2r) / `--lora_target_modules`(默认 q,k,v,o)；抽 `_build_lora_config(args)` 统一 3 处硬编码 LoraConfig。
- `run_lm_exps/smell_main.py`：Server `LoRAOnlyModel` 从 config 读 `num_hidden_layers`/`hidden_size`，从 args 读 `lora_r`/`lora_target_modules`，与 Client 严格对齐。
- `script/RUNME-v2-len4k-sparse0.4.sh`：`--lora_r 1 --lora_alpha 2`，`--samples_per_round 100`。

**A3 参数规模**：d = 32 层 × 4 proj × 2 × r(=1) × 4096 = **1,048,576**（8.4M → 1.05M，**8× 降低**）。配合 samples 20→100（M 5×），合计 SNR 理论提升 **√(8×5) ≈ 6.3×**。

## 风险 / 待验证

- [ ] 冒烟确认 `trainable=1,048,576`（Client）与 Server LoRAOnlyModel 参数量一致。
- [ ] 确认 eval 加载 by index 顺序仍正确（r=1 时 server 枚举 layer→proj→A,B 与 peft 一致）。
- [ ] B 归一化后 jvp 仍非零且分布合理（逐元素扰动 ~1.6e-4 > bf16 ulp）。
- [ ] 若仍不学习，下一步考虑：A4（只 q_proj, d=262K）或改 exact JVP / 结构化低秩扰动（方案 D）。

## 新 run 配置

sparse 0.4 / c3 / alpha 0.1 / len 4k / **samples 100** / **epochs 2** / **lora_r 1, alpha 2** / max_resample 200 / eval_ppl 50 / round 25。因 samples 5×，单轮时间预计 ~5×，总时长会显著拉长。
