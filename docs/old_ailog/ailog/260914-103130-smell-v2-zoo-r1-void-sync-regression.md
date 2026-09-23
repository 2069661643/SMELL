# SMELL-v2 — ZOO r=1 长跑作废：两个独立阻塞（同步回归 + bf16 吞掉 head LoRA）

## 概述

ZOO r=1 长跑（`epochs=1024`，comm_round=21）跑到第 17 轮被终止。排查中发现**两个互相独立**的阻塞，任一都足以让「训练不学」，且都不是之前归因的 B1（ZOO SNR）：

- **阻塞一**：server→client 参数同步被 `569d94b` 误删成 no-op，client 参数全程冻结（见下）。
- **阻塞二**：lm_head LoRA 的增量在 bf16 前向里被**舍入吞掉**——把 B 整个置零，loss 变化是位级的 0。这条与同步无关，修好同步也照样不学。

---

# 阻塞一：server→client 参数同步回归（569d94b 误删 `load_state_dict`）

## 概述

ZOO r=1 长跑（`epochs=1024`，comm_round=21）跑到第 17 轮被终止。**终止原因不是 ZOO 不学，而是 client 从未收到 server 的参数**——跨轮 `[EVAL-PPL]` 17 次评估全部为 `45.037962 / eval_loss=3.807506`，连小数第 6 位都不动。

根因是一次**误删**：commit `569d94b`（2026-09-13 12:20:33）在 “rebuild fed_trainer_transformer.py from HEAD to drop unrelated whole-file CRLF churn” 的过程中，把 `FedTransformerTrainer.set_model_params` 里唯一一行真正加载参数的语句删掉了：

```diff
         # SMELL 3 load_state_dict COMMENTED — 翻译 key 后仍走同一 load_state_dict(strict=False)
         # self.model.load_state_dict(model_parameters, strict=False)
-        self.model.load_state_dict(model_parameters, strict=False)
```

删除后只剩注释版，函数在 `fed_trainer_transformer.py:70` 结束，**没有任何替代加载路径**（无 `super().set_model_params`、无 `copy_`、无 `load_state_dict`）。函数只做 key 翻译 + 打日志，然后丢弃 `model_parameters`。

## 调用链（确认路径是活的，终点是 no-op）

```
FedSgdClientManager.handle_message_{init,downstream}()   # FedSgdClientManager.py:52 / 108 / 145
  -> FedSGDTrainer.update_model(weights)                 # FedSgdTrainer.py:25
    -> FedTransformerTrainer.set_model_params(weights)   # fed_trainer_transformer.py:21  ← 丢弃参数
```

日志显示消息确实到达：`[DBG-SETPARAM] recv dict n_keys=2 keys[:4]=['lora_A_lm_head', 'lora_B_lm_head'] checksum=33.727552`，`translated n=2 matched=2`——key 翻译与匹配都正确，只是在最后一步没有写回模型。

## 证据（三重独立）

| # | 证据 | 观测 |
|---|---|---|
| 1 | 代码 | `git show 569d94b` 显示删除该行；当前函数无加载语句 |
| 2 | 客户端参数 | `[DBG-SETPARAM] post-load trainable \|sum\|=32.035362 delta_norm=0.0000e+00 per=[lora_A:0.000e+00, lora_B:0.000e+00]` —— **每一轮 delta_norm 都精确为 0**，即本轮 post-load 参数与上轮完全相同 |
| 3 | 评价指标 | `[EVAL-PPL]` baseline + round 0..15 共 17 次，全部 `ppl=45.037962 d_ppl=+0.000000` |

同时 server 侧在正常演化：`[DBG-SENDPARAM] checksum` 单调增长 33.727552 → 35.571584（round 0..7），说明**聚合端在更新，只是下发后没落地**。

> 注意 `delta_norm` 这个指标本身是本 commit 为替代 `\|sum\|` 而新加的（B4），定义为「本轮 post-load 参数 vs 上轮 post-load 参数」，不存在 B2 那种跨轮未重置的假指标问题——它的 0 是真实的 0。

## 影响面

- **ZOO r=1 run 作废**（2026-09-13 13:26:51 启动 > 12:20:33 提交）。21 小时算力全部无效：client 一直停在初始 LoRA，eval 一直在初始 θ 上。
- **D3 r=4 run 不受影响**（2026-09-13 10:33 启动 < 12:20:33 提交）。它用的是 `e878d40` 及之前的版本，包含该行，所以当时观测到的 `\|sum\|` 131.1→135.4 变化、以及 M1 同步验证结论仍然有效。
- **A2「随机游走 / 方向无下降信号」结论需复核。** 该结论基于 server 侧逐轮 checkpoint 的 `cos(Δθ_r, Δθ_{r-1})`，而 server checkpoint 本身是真实的聚合结果，方向上仍可参考；但由于 client 冻结，每轮梯度都在**同一个 θ_init** 上估计、跨轮相互独立，这会系统性放大「cos≈0」的观感。在同步修复前，不能用它给「ZOO 估计器 SNR 不足」下定论。
- **B1 结论本身仍有独立支撑**：C5（`test_exact_head_grad.py`）是纯本地仿真、不经过联邦同步，其「精确 head BP 可降 loss、ZOO 方向质量 ≈ √(M/d)」的结论不受本次回归影响。但「端到端训练不学」目前**没有任何一次有效实验验证过**。

## 阻塞一修复与验证（已实施）

**改动**：`FwdLLM/training/fed_trainer_transformer.py` 的 `set_model_params`，在被注释的
`load_state_dict` 行之后、debug 块之前，补回参数落地逻辑。

**不用 `load_state_dict(model_parameters, strict=False)`，改为按 key 显式 `copy_`**：前者在 key 不匹配时
静默忽略，正是 M1 老 bug（“Server 聚合结果永远到不了 Client”）的形态，出问题没有任何征兆。
显式 `copy_` 会统计实际落地数量，落地为 0 时直接 `logging.error`。

放在 debug 块**之前**很关键——`delta_norm` 是「本轮 post-load vs 上轮 post-load」，只有先落地才有意义。

**验证**：`run_lm_exps/check_param_sync.py`（新脚本，3 项断言 + 1 个负例），日志 `log/paramsync_verify_20260914.log`：

```
[SYNC] before: ||A||=5.815825e-01 ||B||=9.431067e-03 loss=2.676983118
[DBG-SETPARAM] translated n=2 matched=2
[DBG-SETPARAM] applied 2/2 tensors
[DBG-SETPARAM] post-load trainable |sum|=50.591326 delta_norm=nan        ← 首轮

[PASS] A. 短名 dict 逐元素落地       ||ΔA||=2.907912e-01 (期望 2.907912e-01)
[PASS] B. 变更传导到前向(loss 改变)   loss 2.676983118 -> 2.676956177 (dL=-2.694e-05)
[PASS] C. 反例不误改参数             applied 0/2 + warning + error，参数不变

[DBG-SETPARAM] post-load trainable |sum|=16.863776 delta_norm=5.8166e-01  ← 第二次同步
```

两个额外的强证据：

1. **`recv checksum` 现在等于 `post-load |sum|`**（50.591326 == 50.591326）。修复前二者不等
   （33.727552 vs 32.035362），正是“下发的值和模型里的值不是同一份”的直接体现。
2. **`delta_norm` 不再是恒定 0**：第二次同步给出 5.8166e-01。修复前每一轮都是 `0.0000e+00`。
3. 负例（全不匹配的 key）会打出 `applied 0/2` + warning + error，**这条路径再也不会静默失败**。

---

# 阻塞二：lm_head LoRA 在 bf16 下被舍入吞掉（独立于阻塞一）

## 概述

排查 `h·‖v‖` 量级问题时写的新诊断脚本 `check_jvp_scale.py` 扫遍 `h` 与三种扰动方案，**没有任何一个 `h` 能让 FD 可信**（rel_err 最小 0.41，见下表）。进一步做「把 LoRA 整个删掉」的对照，才发现问题不在 FD，而在**参数化本身在前向里不可见**。

## 决定性证据

`/tmp/dbg_logits.py`（固定同一 batch、同一 `h_in`，只改 A/B，直接看 loss 与 logits 的变化）：

| 变体 | L | dL | ‖dz‖/‖z‖ |
|---|---|---|---|
| baseline (A0, B0) | 2.682932854 | — | — |
| **B=0（删掉 LoRA 分支）** | 2.682932854 | **0.000e+00** | 6.705e-07 |
| **A=0（删掉 LoRA 分支）** | 2.682932854 | **0.000e+00** | 6.705e-07 |
| **A=B=0（纯 base）** | 2.682932854 | **0.000e+00** | 6.705e-07 |
| A×10 | 2.682794094 | -1.388e-04 | 2.439e-05 |
| B×10 | 2.682794094 | -1.388e-04 | 2.439e-05 |
| A×100, B×100 | 8.622855186 | +5.940e+00 | 1.043e-01 |

量级：

```
model.dtype                        = torch.bfloat16
lm_head.base_layer.weight          = bfloat16 (32000, 4096)   ‖W‖ = 1.9787e+02
lm_head.lora_A/B.weight            = float32   (1,4096)/(32000,1)
‖ΔW‖ = 2·‖B‖·‖A‖ = 1.0970e-02     ‖ΔW‖/‖W‖ = 5.54e-05
```

**机理（已直接验证，非推断）**：`/tmp/dbg_sumdtype.py` 在 head 前向内部逐层取 dtype：

```
x         torch.bfloat16
base out  torch.bfloat16
full out  torch.bfloat16        ← base + lora 的加法结果也是 bf16
scaling   2.0
‖full−base‖/‖full‖        = 6.7e-07      ← 实际活下来的增量
‖manual lora_fp32‖/‖base‖ = 1.04e-05    ← 增量本该有的量级（fp32 手算）
‖full − (base + lora_fp32)‖ = 3.89e-01  ← 手算增量范数 0.389，被整体抹掉
(full−base) 不同取值个数   = 158 / 131,072,000   ← 量化指纹
bf16 ulp @ |z|rms=3.26    = 1.27e-02
```

LoRA 增量的逐元素量级 `0.389/√1.31e8 ≈ 3.4e-5`，比输出 bf16 ulp `1.27e-2` **低约 375 倍**，因此 `base + lora_delta` 这一步把增量整个舍掉。放大 10~100× 越过阈值后才可见，与 ulp 位置吻合。

**为什么梯度却不为零**：addition 之后的 dtype cast 在 backward 里是 **straight-through**（导数按 1 传递，不体现舍入），所以 autograd 仍给出非零梯度。于是形成「梯度非零 + loss 不响应」的错配——优化器会持续移动参数而 loss 纹丝不动。这正是 A2 观测到 `‖Δθ‖=lr` 且 `cos(Δθ_r, Δθ_{r-1})≈0` 随机游走的成因，**不是 ZOO 估计器 SNR 不足**。

| 维度 | 「低梯度平坦区」 | 本次实际情况 |
|---|---|---|
| loss 对 (A,B) 不响应 | 是 | 是（表象相同） |
| autograd 梯度 | ≈ 0 | 0.614 / 2.5e-3，全非零 |
| 成因 | 目标函数本身平坦 | 前向 bf16 cast 舍掉增量 |
| 优化器行为 | 原地不动 | 一直在动，但 loss 不响应 |
| 对症修法 | 调 lr / 加动量 | 均无效，必须让增量在前向可见 |

## FD 扫描结果（`check_jvp_scale.py`，D3 r=1，4 样本 × 2 方向）

| mode | h | h·‖δ‖ | h·‖δ‖/‖θ‖ | rel_err_med |
|---|---|---|---|---|
| raw | 1e-2 | 1.90 | 3.27 | 0.759 |
| raw | 3e-2 | 5.71 | 9.81 | **0.446**（全表最小） |
| raw | 1e-1 | 19.0 | 32.7 | 2.459 |
| raw | 1e-3 | 0.190 | 0.327 | 0.990 |
| unit | 所有 h | — | — | 1.000（fd 恒为 0） |
| rel | 所有 h | — | — | 1.000（fd 恒为 0） |

- `--fd_dtype fp32`（关闭 autocast）与 bf16 **结果逐行几乎相同** ⇒ 失真不来自 FD 的运算精度，而来自 base 权重本身就是 bf16（`dbg_logits` 已确认 `base_layer.weight: bfloat16`）。
- 之前记录的 `h=0.01`、`h·‖v‖=3.27×‖θ‖` 是**真实存在**的问题，但在本参数化下是次要矛盾：loss 对 (A,B) 根本不敏感，FD 无论取什么 h 都测不到东西。

## 连带影响

- **D3（lm_head-only LoRA）这条线整体无效**。C5 的「head-only BP 精确可学」负结果（commit `9cf4c69`）由此得到解释：不是 BP 不行，是这个参数化的输出被 bf16 抹掉了。
- **C5 的 `cos ≈ √(M/d)` 方向质量结论**是在「用精确梯度 `g_head` + 精确 `<g,v>` 做离线仿真」下得到的，它刻画的是估计器在理想设定下的性质；但真实 loss 对 (A,B) 无响应，因此「ZOO vs BP」的对比在该参数化下不成立。
- **A2 的随机游走结论**同样测在这个 loss 对 (A,B) 近平坦的区间内，需在修好参数化后重测。
- 该风险不限于 head：任何 `‖ΔW‖/‖W‖ ≲ 3.9e-3` 的 LoRA 增量在 bf16 前向里都会被吞。Phase 1/2 的 r=8 q_proj 是否也受影响需要单独量一下。

## 修复与验证（已实施）

**两处改动，缺一不可**：

1. `experiments/distributed/transformer_exps/initializer.py:88` 新增 `_promote_lm_head_fp32(model)`，把 `lm_head.base_layer.weight` 换成 fp32 副本；在 llama_sparse 分支的 peft 注入后调用（`initializer.py:234`）。
   - 用**新建 Parameter** 而非 in-place `.data` 转换：`tie_word_embeddings=True` 下 `lm_head.weight` 与 `embed_tokens.weight` 共享同一张量，in-place 转换会连带把输入 embedding 变成 fp32、与 bf16 主干不匹配。替换后解绑，但二者皆冻结且初值相同 ⇒ 前向行为不变。
   - `requires_grad=False`，**不改动任何可训参数** ⇒ trainable 集合/形状/顺序与 server 侧保持一致。
2. `model/modeling_llama_sparse.py:1256` head 前向改为在 `torch.autocast(device_type=..., enabled=False)` 下、把输入 `.float()` 后喂给 `self.lm_head`。

只做第 2 步不够：`F.linear` **不做类型提升**，fp32 输入 × bf16 权重直接抛
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`（首次验证时就是这样失败的）。
两步合起来才使 peft `layer.py:980` 的 `result.to(torch_result_dtype)` 成为 fp32→fp32 空操作。

**验证**：`experiments/distributed/transformer_exps/run_lm_exps/check_lm_head_fp32.py`（新脚本，3 项断言，退出码即结果），日志 `log/fp32head_verify_20260914.log`：

```
[SMOKE] lm_head promoted to fp32: base_model.model.lm_head.base_layer.weight bfloat16 -> float32
[FP32HEAD] lm_head hook: input dtype=torch.float32 output dtype=torch.float32

[PASS] A. lm_head 输出为 fp32                input=float32 output=float32   （修复前 bfloat16）
[PASS] B. B=0 / A=0 改变 loss                max|dL|=2.122e-05               （修复前 0.000e+00）
[PASS] C. h=0.01 rel_err 降到 O(1e-2)        rel_err_med=1.909e-02            （修复前 0.759）

  h=1.0e-02   rel_err_med=1.909e-02
  h=3.0e-03   rel_err_med=1.735e-03
  h=1.0e-03   rel_err_med=4.581e-04
```

**关键副产品**：rel_err 现在**随 h 减小而单调下降**（1.9e-2 → 1.7e-3 → 4.6e-4），是教科书式的 O(h²) 有限差分行为。修复前恰好相反——h 越小 fd 越趋近于 0（被 bf16 量化地板吃掉）。这独立佐证了 FD 已回到线性区、量化地板已消失。

同时说明 `h=0.01` 这个硬编码值（`fwdgrad_utils.py:189`）**不再是灾难但仍非最优**：h=1e-3 的 rel_err 比它好 40 倍。`h` 的选值可作为后续独立优化项。

## 未实施的备选（记录备查）

1. 提高 LoRA 输出尺度（改 `lora_lm_head_alpha` / 初始化 B 非零），把 `‖ΔW‖/‖W‖` 抬过 bf16 阈值——治标，且会改变优化动力学。若要跨过 ~375× 的差距需 scaling≈750，属退化配置。
2. 换更"大"的可训练参数化（如 A1 task head / C1 soft prompt），让增量天然可见。
3. **`model/modeling_llama_base.py` 未同步修改**（Phase 1 用的 `--model_type llama`）。若跑 Phase 1 需要同样处理，否则其 head LoRA 仍会被 bf16 吞掉。

---

## 归档内容

| 项 | 位置 |
|---|---|
| 完整日志（17MB，含 3072×17 次 fwdgrad 明细） | `log/smell-v2-zoo-r1_e1024_20260913_132651.log` |
| server 逐轮 checkpoint r0..r16 | `checkpoints/checkpoint_20260913_132655_r{0..16}.h5` |
| consensus 历史 | `checkpoints/consensus_history_20260913_132655.h5` |
| 启动记录 | `docs/ailog/260913-132739-smell-v2-zoo-r1-longrun.md` |
| 启动脚本 | `script/RUNME-v2-len4k-d3-zoo-r1.sh` |

运行区间：2026-09-13 13:26:51 → 2026-09-14 10:30（约 21h），完成 16 轮更新 + 17 次评估（约 76min/轮）。

## 顺带确认的其它事实（本次读码所得）

- 训练脚本里的 `--lr 0.0001` / `--learning_rate` 在 ZOO 路径**完全无效**：client 全程 `torch.no_grad()`、无 optimizer，唯一更新点是 `FedSgdAggregator.py:238-240` 的 `param -= lr * normalize(grad)`。
- 因为脚本都带 `--lr_decay`，`--server_lr 0.001` 也被忽略，实际 lr 是 `FedSgdAggregator.py:167-171` 里硬编码的 0.01→1e-5 线性斜坡（日志实测 0.01 → 0.0095005 → …）。
- `--var_control` / `--max_resample 50` 实际从未触发：日志每轮 `var: 0.0`（首轮 1.70e-4），恒 ≤ 硬编码阈值 `var_threthod=1` ⇒ 立即 accept，M 恒为 3072。
- `h=0.01` 非 CLI 参数（`fwdgrad_utils.py:189` 函数默认值），`v` 为未归一化 `randn_like`（`tc_transformer_trainer_distribute.py:227`），导致 `h·‖v‖=1.90 ≈ 3.27×‖θ‖`。C5 的 JVP 校验 rel_err=2.222 与此一致。**已诊断，结论见阻塞二**：真实原因是 LoRA 增量被 bf16 吞掉，`h` 是次要矛盾。

## 重新估算 lr（两阻塞修复后实测）

脚本 `run_lm_exps/estimate_lr.py`，日志 `log/estimate_lr_20260914.log`。前提：`FedSgdAggregator.py:238-240`
是唯一更新点，梯度先全局 L2 归一化再 clamp ⇒ **`‖Δθ‖ ≡ learning_rate`**（步长是参数空间里的绝对长度）。

### 方向质量：实测等于理论

```
[LR] (3) ZOO 方向 (M=3072, fd_h=1e-2): ||g_zoo||=2.908018e+03
          cos(g_zoo, g_exact)=0.2794      sqrt(M/d)=0.2917
```

**实测 0.2794 对上理论 0.2917**——阻塞二修复前这个数是不可能对上的（彼时损失对 (A,B) 无响应）。
这独立确认估计器已恢复正常。内部自洽性也对：`dL/lr` 沿 ZOO 方向 = -0.0779，沿精确方向 = -0.2686，
比值 0.290 ≈ cos。

### 步长响应：几乎全程线性

| 方向 | L0 | 线性区(±10%)上界 | eps=1.0 时线性比 |
|---|---|---|---|
| 精确 head 梯度 | 3.578512669 | **> 1.0**（即 >172%‖θ‖） | 0.918 |
| ZOO 估计方向 (M=3072) | 3.578512669 | **≈ 0.3**（≈52%‖θ‖） | 0.650 |

ZOO 方向 eps=1.0（‖Δθ‖=1.72‖θ‖）时 dL=-0.0506（-1.4%）；精确方向同 eps 时 dL=-0.2465（-6.9%）。

### 当前调度的实测总位移

`lr_decay` 硬编码 0.01→1e-5 线性（`FedSgdAggregator.py:167-171`，`--lr`/`--server_lr` 都被忽略）：

```
全程 21 轮: Σlr=0.105105 = 0.1807×||theta||      ← ‖Δθ‖≡lr，故这也是总位移
参考: 恒定 lr=0.01 跑 21 轮 Σlr=0.2100 = 0.3610×||theta||（当前斜坡只用了 50%）
```

且**严重前重后轻**：50% 的位移集中在前 6 轮，第 20 轮只移动 0.0017%‖θ‖。响应是线性的、
没有任何收敛迹象可退火，所以这个衰减没有依据。

### 预期收益（ΔL = -0.0779 × Σlr）

| 方案 | Σlr | ΔL | PPL | 相对 |
|---|---|---|---|---|
| 当前斜坡 0.01→1e-5 | 0.1051 | -0.0082 | 35.82→35.53 | **-0.8%** |
| 恒定 0.01 | 0.2100 | -0.0164 | 35.82→35.24 | -1.6% |
| **恒定 0.05（建议）** | 1.0500 | -0.0818 | 35.82→33.0 | **-7.8%** |
| 恒定 0.10 | 2.1000 | -0.1636 | 35.82→30.4 | -15%（外推，未验证） |

### 结论

1. **建议 lr ≈ 0.05、恒定，去掉 `--lr_decay`。** 每步 `‖Δθ‖/‖θ‖`=8.6%，对比实测线性区上界 52% 有 6 倍余量；
   同样墙钟时间下的预期收益是当前调度的 ~10 倍。改法：改 `FedSgdAggregator.py:168` 的 `start_lr`，
   或关掉 `--lr_decay` 改用 `--server_lr`（注意此时 `warmup_rounds=0`，ratio 退化为 `(comm_round-r)/comm_round`）。
2. **上限不在 lr。** 即便沿**精确**梯度、位移达 1×‖θ‖，loss 也只降 6.9%；ZOO 的 cos=0.28 再打一个折。
   所以 D3 r=1 这条参数化本身的「抓手太弱」才是天花板——这与阻塞二是同一个根源的两面。
3. **`lr ≡ ‖Δθ‖` 是绝对量**，以上数字绑定在 d=36096、‖θ‖=0.582 上；换参数化必须重测。
4. **M 与轮数的权衡（推导，未实测）**：每轮相干位移 ∝ √(M/d)，每轮墙钟 ∝ M，故**单位时间位移 ∝ 1/√M**
   —— 减小 M、增加轮数在时间效率上更优（M=256 约为 M=3072 的 3.6 倍）。而方向质量 SNR
   `cos·√R/√(1-cos²)` 与 M 无关（单位算力的 SNR 不变量），所以这是绝对位移的取舍，不是方向质量的取舍。

## lr_schedule CLI 链路改造与小验证（已实施）

### 改动

1. `initializer.py` `add_federated_args` 新增三个参数：
   - `--lr_schedule {constant,linear}`，**默认 `constant`**
   - `--lr_end`（默认 `1e-5`）——linear 的终点，替掉原来硬编码的 1e-5
   - `--warmup_ratio`（默认 `0.0`）——**此前该参数从未定义**，`getattr(...,0)` 恒取 0，
     导致 `warmup_rounds=0`、warmup 分支永远打不开，是个 latent bug
2. `FedSgdAggregator.aggregate()`：`--server_lr` 成为唯一入口。`constant` 时全程取 `server_lr`；
   `linear` 时以 `server_lr` 为起点、`--lr_end` 为终点。`--lr_decay` 保留为 `linear` 的别名（向后兼容）。

   **注意语义变化**：传 `--lr_decay` 的旧脚本，起点从硬编码的 `0.01` 变为脚本实际传入的 `--server_lr`
   （与其上方 SMELL 3 注释里声明的意图一致，但确实是行为改变）。

### 小验证

`script/RUNME-v2-lr-schedule-check.sh`（照 ZOO 长跑脚本缩小：epochs 1024→64、comm_round 21→4，
`--lr_schedule constant --server_lr 0.05`，**不传** `--lr_decay`；输出 `--checkpoint_dir /tmp/...` 不污染主目录）。
日志 `log/smell-v2-lrcheck_20260914_114454.log`。

| 判据 | 结果 |
|---|---|
| P1 lr 恒定 | `learning rate: 0.05` 三轮完全相同（旧斜坡为 0.01 → 0.0095 → 0.009） |
| **P2 ‖Δθ‖ == server_lr** | **`delta_norm=5.0000e-02` 精确等于 0.05**（lora_A:1.714e-02, lora_B:4.697e-02，√(A²+B²)=0.05） |
| P3 同步落地 | `applied 2/2 tensors` 每轮 |
| P4 PPL 脱离位级恒定 | 45.032131 → 45.032571 → 45.033605（d_ppl = +0.000440, +0.001474） |

**P2 是最强的一条**：聚合端先全局 L2 归一化再乘 lr，故 `‖Δθ‖` 应恒等于 lr。实测精确到 `5.0000e-02`，
说明 CLI → 聚合 → 下发 → client 落地 → 参数位移整条链路一致。

**附带确认 fp32 head 在真实流水线里生效**：本次 baseline PPL = **45.032131**，而修复前长跑的 baseline 是
**45.037962**（差 5.8e-3，相对 1.3e-4，与 LoRA 增量对 logits 的 ~1e-5 相对贡献同量级）。这是阻塞二修复
在端到端流水线（而非离线脚本）里的独立证据。

**PPL 方向不能用来判 lr 取值**：M = 3×64 = 192 ⇒ `cos = √(192/36096) = 0.073`，
`drift/noise = cos·√R/√(1-cos²) ≈ 0.073×1.41/0.997 = 0.10` —— 随机游走比相干漂移大 10 倍；
且 `[EVAL-PPL]` 在 **test** split 上，而梯度只针对 3 个 **train** 样本。故 PPL 上漂属预期，
与 lr 取值无关。**lr=0.05 的实际收益仍需全量 M=3072 的长跑验证。**

## 全量长跑重跑 + train/test 双评估（已启动）

### 新增改动：train split 评估

`FedSgdClientManager._maybe_eval_ppl` 拆为 `_maybe_eval_ppl` + `_eval_split`，每轮同时评两个 split：

- test：前缀 `[EVAL-PPL]`（**行格式完全不变**，避免破坏既有 grep）
- train：前缀 `[EVAL-PPL-TRAIN]`（新增，取 `self.trainer.train_local` 展平；test 用 `test_local`）

两者使用同一 `--eval_ppl 50 --eval_seed 42`，样本集固定 ⇒ 跨轮可比；baseline 按 split 分别缓存，
故每行的 `d_ppl/d_loss` 是相对**各自**起点。走 `eval_ppl_fixed`（无扰动、token 加权、经 fp32 head）。

判读：**train 降而 test 不降 = 过拟合；两者同降 = 泛化；两者都不动 = 更新无效果。**

启动校验（`log/smell-v2-zoo-r1-lr005_e1024_20260914_125239.log`）：

```
[SMOKE] LoRA config: r=1, alpha=2, targets=['lm_head'], lm_head_mode=True
[SMOKE] After LoRA(sparse): trainable=36,096
[EVAL-PPL]       round=baseline ppl=45.032131 eval_loss=3.807376 n_tokens=862
[EVAL-PPL-TRAIN] round=baseline ppl=44.320965 eval_loss=3.791458 n_tokens=892
```

### 配置

| 项 | 值 |
|---|---|
| 模型 | `llama_sparse` + LoRA lm_head-only r=1 / alpha=2（scaling=2），d=36,096 |
| ZOO | `--forward_mode`，epochs=1024（M=3×1024=3072，理论 cos=0.2917） |
| **lr** | **`--lr_schedule constant --server_lr 0.05`**（每轮 ‖Δθ‖=0.05=8.6%‖θ‖；全程 Σlr=1.0=1.72‖θ‖） |
| 联邦 | 3 clients，comm_round=21（20 次更新），worker_num=1，`-np 2` |
| 稀疏/共识 | `--sparse 0.4 --enable_consensus --vote_threshold 0.3` |
| 降噪 | `--var_control --perturbation_sampling --max_resample 50` |
| 评估 | `--eval_ppl 50 --eval_seed 42` × {test, train}，`--save_per_round 1` |
| 脚本 | `script/RUNME-v2-len4k-d3-zoo-r1-lr005.sh` |
| 预算 | ~76 min/轮 × 21 ≈ 27h（每轮多一次 train eval，+33s，可忽略） |

### 预期与判读

- 训练目标：`ΔL = -0.0779 × Σlr = -0.078`。
- 但 `-0.0779` 是在**训练样本**上测的斜率，**test PPL 的降幅取决于泛化，应明显更小**——这正是新增 train split 要回答的问题。若 test 完全跟随，PPL 45.03→~41.5（-7.8%），这是乐观上界。
- 轮数 20 > 12，满足 `cos·√R > 1`，相干漂移应能压过随机游走（上次长跑在 R 和 cos 上都远不满足）。
- 与上次长跑的对照锚点：上次 baseline（bf16 head）为 45.037962，本次（fp32 head）为 45.032131。

### 首次启动中途被杀与重跑（诊断修正）

- 首次启动 12:52:39，**13:22:05 死亡**，停在 `epoch = 352` / 1024（round 0 的 34%），未落 checkpoint。
- 初步归因**是错的**：曾判断为脚本内 `trap ... EXIT` 开火。证据反驳——该日志
  `grep -c "Cleanup done"` = **0**，trap 里的 echo 会写进同一 LOG_FILE，却一次都没出现。
- **真实死因**：启动方式是会话绑定的后台 shell（工具后台任务），会话回收时整个进程组被
  SIGKILL，trap 接不住 SIGKILL ⇒ 训练被一起带走。即**运行从未真正脱离会话**，与 trap 无关。
- 修法：
  1. 用 `nohup ... &` 脱离会话启动（`setsid` 非必需，`nohup` 已足够）；用法写进脚本头注释。
  2. 仍然 **REMOVED** 脚本内那个 `EXIT trap`——这次没开火，但仍是隐患：脚本 shell 一旦收到
     SIGTERM / 终端关闭，它就会 `pkill -f smell_main` 把训练杀掉。
     （4 个旧 RUNME 脚本都带这一行，尚未清理。）
- 重跑：16:39:36 启动 → `log/smell-v2-zoo-r1-lr005_e1024_20260914_163936.log`。
- 重跑校验（两次启动的 baseline **逐位相同** ⇒ 评估确定、两次运行等价）：

  | split | ppl | eval_loss | n_tokens |
  |---|---|---|---|
  | test | 45.032131 | 3.807376 | 862 |
  | train | 44.320965 | 3.791458 | 892 |

  train 略低于 test（44.32 vs 45.03）符合预期；两者各自记 baseline，故每行 `d_ppl` 是相对各自起点。

### 运维注意

- 长跑一律 `nohup ... &` 脱离会话启动，**不要挂成会话绑定的后台任务**。
- 巡检用 **30s** 粒度起步；长跑中途再逐步放宽。

## 长跑停止（23:32）+ 测量问题排查

长跑于 23:32 主动停止（round 4 的 epoch 744/1024）。原因：4 轮后 test `d_loss=+4.5e-5`、
train-split `+1.1e-5`，外推跑满 20 轮收益上界仅 `Δloss≈-2.3e-4`（`ΔPPL≈-0.01`），
远低于样本内斜率许诺的 `Δloss=-0.078`（`ΔPPL≈-3.5`）。

### 1. `data_id` 每轮归零 —— **不是 bug**

用户质疑「本次连 BP 那次的过拟合都没出现」。追查 `data_id`：

- `FedSgdClientManager.py:141` 每轮无条件 `self.data_id = 0`，把 `:228` 的 `data_id += 1` 撤销。
- 但日志实证 `len(train_local_list[0]) = 1`（每轮），因为 `--samples_per_round 1` 把每个 client
  的本地列表裁到 1 条；`train_with_data_id` 里 `data_id == len(train_local_list[0])` 即发 model
  结束本轮。⇒ 归零是「**重启 sweep**」的设计，不是 bug。
- **但** `FedSgdTrainer.py:36-42` 用的是**无种子** `random.sample`：
  ```python
  self.train_local_list = [random.sample(tl, min(samples_per_round, len(tl))) for tl in ...]
  ```
  而 `update_dataset` **每轮都被调用**（`handle_message_receive_model_from_server`）
  ⇒ **每轮抽的是全新的 1 条样本**。

⇒ 20 轮 × 3 条 = **60 条互不重复的样本，每条只曝光一次** ⇒ 结构上**不存在**重复曝光型
（memorization）过拟合。BP 那次的 4.195→3.421 是在**固定 128 样本上反复走 40 步**得到的
样本内拟合曲线，两者不是同一个量，**不可比**。

**修正先前错误**：我此前说「整个长跑在同一批 3 条样本上走 20 步」（只看了 `data_id=0`
未查重采样）是**错的**；据此算出的三列「逐样本轨迹」实际是三条不同样本，不是轨迹。
同时先前那句「既不是泛化也不是过拟合，是更新无效」也**过度声称**：`[EVAL-PPL-TRAIN]` 测的是
从未被训练的 50 条，是**泛化代理**，它看不见训练批。

### 2. 测量缺口：run 里没有任何量能测训练批的 `L(θ)`

| 想问的量 | 原仪器 | 能否回答 |
|---|---|---|
| 训练批 `L(θ)` 降没降 | 每轮 `loss=` | ✗ |
| 同一批 50 条 train 的留出 PPL | `[EVAL-PPL-TRAIN]` | ✓（但这批从未训练） |
| BP 那种训练批 loss 曲线 | 无 | **缺失** |

每轮 `loss=` 是 `(L(θ+hv)+L(θ−hv))/2`，而 `h=0.01` 硬编码（`fwdgrad_utils.py:75`）、
`v = torch.randn_like` σ=1（`tc_transformer_trainer_distribute.py:227`，v 归一化方案 B 已注释
`:228-233`），d=36,096 ⇒ `‖v‖=190` ⇒ **`‖hv‖ = 1.90 = 3.27‖θ‖`**（`‖θ‖=0.5817`）。
即采样点离 θ 有 3.27‖θ‖ 远，测的是那里的曲率项 `(h²/2)tr(H)`。

一致性校验：`estimate_lr.py` 的无扰动 line search 显示沿 ZOO 方向挪 `eps=1.0`（=172%‖θ‖）
`L` 才降 0.0506 ⇒ 每 0.05 步长真值只动 ~0.004；而实测每轮均值摆 **±1.3**
⇒ **仪器噪声是信号的 ~300×**。

### 3. `shutdown` 竞态：末轮 eval 一直是丢的（新发现并已修）

`FedSgdServerManager.py:118-142`：最后一轮 `send_message_sync_model_to_client` 之后**立刻**
`send_finish_notification` → `finish()` → 退出 → `MPI_ABORT`。而客户端收到 sync 后还要跑
`[EVAL-PPL-TRAINBATCH-AFTER] + [EVAL-PPL] + [EVAL-PPL-TRAIN]`（~70s），**被一并杀掉**。
smoke 实证：`AFTER ... START` 之后下一行就是 `Done!`，没有结果行。

⇒ **每一轮长跑的最后一轮 eval 都是丢的**，之前没跑到末轮故未暴露。已加
`--final_eval_grace_s`（默认 120s）等待客户端跑完；更干净的做法是加一条 ACK 消息，暂不引入。

### 4. 修复内容

| 文件 | 改动 |
|---|---|
| `FedSgdClientManager.py` | 新增 `_eval_train_batch`；`train_with_data_id` 训练前钉住本批样本并打 `BEFORE`；`handle_message_receive_model_from_server` 参数落地后打 `AFTER`（用**缓存的张量**，因 `update_dataset` 会重建 `train_local_list`）；`_eval_split` 加 `use_baseline=False` |
| `FedSgdServerManager.py` | 末轮加 `final_eval_grace_s` 等待 |

### 5. 验证（`log/trainbatch-smoke_20260914_233827.log`）

```
[EVAL-PPL-TRAINBATCH-BEFORE] round=0 ppl=81.799300 eval_loss=4.404269 n_samples=3 n_tokens=50 cost=2.0s
[EVAL-PPL-TRAINBATCH-AFTER]  round=0 ppl=81.803956 eval_loss=4.404326 n_samples=3 n_tokens=50 cost=2.0s
[EVAL-PPL-TRAINBATCH-BEFORE] round=1 ppl=42.000317 eval_loss=3.737677 n_samples=3 n_tokens=39 cost=2.0s
[SMOKE] Waiting 120s for client post-sync evals before shutdown
[EVAL-PPL-TRAINBATCH-AFTER]  round=1 ppl=41.998880 eval_loss=3.737643 n_samples=3 n_tokens=39 cost=2.0s
```

- BEFORE/AFTER 的 `n_samples`/`n_tokens` 一致 ⇒ 确认是**同一批**样本，配对有效。
- 两次 smoke 的 `BEFORE round=0` 逐位相同（`81.799300 / 4.404269`）⇒ 仪器确定可复现。
- 末轮 `AFTER` 现在完整产出 ⇒ 竞态修复生效。
- cost 2.0s/次，对 82 min/轮可忽略。

### 6. 顺带量化：监督 token 极少

| 评估对象 | n_samples | n_tokens | tokens/样本 |
|---|---|---|---|
| trainbatch（3 条训练样本） | 3 | 39~50 | ~13~17 |
| `[EVAL-PPL-TRAIN]` | 20 | 363 | ~18 |
| `[EVAL-PPL]`（test） | 20 | 368 | ~18 |

⇒ 每轮 M=3072 个方向的 ZOO 梯度，只由 **~50 个监督 token** 估计。训练批配对评估的分辨率
约 `1e-4`，而预期样本内效应 `-3.9e-3/轮` ⇒ **该仪器足以检出**，可作为 F2 判据。

## 【根因】perturbation_sampling 的 v 缓冲区被逐 epoch 复用 —— 有效扰动数 M=3，不是 3072

### 结论先行

`--perturbation_sampling + --var_control` 下，扰动方向缓冲区只存 `len(train_dl)=3` 个向量，
取用时**只按 `batch_idx` 索引、不含 `epoch`**，于是 1024 个 epoch 反复用同 3 个方向。
**有效扰动数 M_eff = 3**，`cos(g_zoo, g_true) ≈ √(3/36096) = 0.0091`，而设计假设为
`√(3072/36096) = 0.2917` —— **差 32 倍**。这一个因子就解释了此前一直无法解释的 26~35× 缺口。

### 代码链

`tc_transformer_trainer_distribute.py:131-149`（缓冲区只存 `v_num` 个）：

```python
v_num = len(self.train_dl)                 # = 3（samples_per_round=1 ⇒ 每轮 3 个样本）
candidate_v = torch.randn((v_num * 10, *shape), device="cpu")   # 30 个候选
v_buffer[n] = [candidate_v[i].reshape(shape) for i in sorted_indices[:v_num]]   # 只留 3 个
```

`tc_transformer_trainer_distribute.py:167-168, 223-227`（循环结构 + 只按 batch_idx 取用）：

```python
for epoch in range(0, self.args.epochs):          # 外层 1024
    for batch_idx, batch in enumerate(self.train_dl):   # 内层 3
        ...
        v_dict[n] = v_buffer[n][batch_idx].to(device)   # ← 没有 epoch ⇒ 每 3 次迭代精确重复
```

`FedSgdClientManager.py:136-143`（决定 `old_grad` 是否为 None ⇒ 缓冲区是否生效）：

```python
if self.args.perturbation_sampling:
    if self.data_id % 2:                     # 此处 data_id 仍是【上一轮】留下的 1
        ...old_grad = ...grad_pool           # → 非 None ⇒ v_buffer 生效
    else:
        ...old_grad = None                   # round 0 ⇒ 空 ⇒ 每次全新 randn
```

注意 `self.data_id = 0` 在 `:171`，**晚于** `:136` 这个块，所以该块看到的是 `data_id=1`（奇数）。

### 日志里的逐位指纹（两条独立证据）

```
round 0: num of fwdgrad: 3072, var: 0.00016877132293302566
round 1: num of fwdgrad: 3072, var: 0.0
round 2..6:                    var: 0.0
```

- `calculate_var`（`fwdgrad_utils.py:83-100`）= 「前半均值 vs 后半均值的方差」。
  3072 项按周期 3 重复 ⇒ 两个 1536 项的半段各含 512 个完整周期 ⇒ 均值**逐位相同** ⇒ **var 恰为 0.0**。
  `var=0.0` 是「精确周期性」的充分证据，随机梯度给不出 0.0。
- round 0 的 `old_grad=None` ⇒ 每次全新 `randn_like` ⇒ 无周期性 ⇒ var=1.69e-4（非零）。**两边同时对上。**

### 后果

1. **方向质量被削 32×。** 对账：预测每轮样本内效应
   `0.05 × (0.0091×⟨g,ĝ⟩ ± ‖g‖/√d)`，用 θ_r7 实测 client 0（`‖g‖=0.538`）
   = **−7.8e-5 ± 1.4e-4**；**实测 −73e-6/轮。吻合，连噪声量级都对上。**
2. **`var_control` 被静默关闭。** var 恒为 0.0 ≤ `var_threthod=1` ⇒ 永远走「达标」分支 ⇒
   重采样安全网（唯一存在目的就是拒掉低质梯度）**恒报"完美"**。
   `resample_count=0` 不是"梯度质量好"，是安全网死了。
3. **跨轮方向相关 ⇒ 解释相干漂移。** 缓冲区按「与**上一轮梯度**的 cos」从 30 个候选里选 top-3，
   方向被系统性锚在过去的梯度上 ⇒ `|θ|₁` 39.15→49.01 的单调、无损增长不是随机游走。

### 为什么前面三轮诊断（E1/E2）没抓到

`diag_theta_drift.py` 每次扰动都用全新 `randn`（M=3072），即**测的是「此 bug 不存在时的 run」**。
故测出 `cos(v_c,g_c)=0.278` 贴着理论 0.2917 —— 那是估计器**本该**有的样子。
⇒ E1/E2 的结论本身没错（θ 漂移被否、相消只有 1/√3、三 client 梯度近正交），
但它们**没有覆盖 run 的真实估计器**。

### 需要作废重测的结论

Phase 5 此前关于「ZOO 方向不可用 / 该换参数化（D3→A1→C1）」的推论，
全部建立在**方向质量被削 32×** 的更新之上：

- lr=0.05 的标定（`estimate_lr.py` 的 −0.0779）从未在「设计质量的梯度」上被测过；
- 「F2≈0，泛化不迁移」是在 32× 退化的方向上测的，它平躺**完全在意料之内**，
  对参数化/容量**什么都没说明**；
- 「60 条不重复样本仍抹掉 90% 记忆」同样来自这种更新。

### 处置

- `var_control` 暂时不用（改用 epochs 直接控制 M）⇒ 该路径下 `v_buffer` 恒空 ⇒
  `M = epochs × samples` 本就正确，bug 不触发。
- 但 bug 仍修：缓冲区改为**每 epoch 重填**，保证 M_eff = epochs × samples。

## A 阶段补测（修复后）—— A1 修复生效 / A1b 根因转数字 / A2 否掉 H1

三个实验均为满窗口 4096（M=3 那个 run 停掉后 GPU1 空出 21.2GB，截断 caveat 消除），
日志 `log/diag/{E_theta_init,E_theta_r7,A1_varbuffer}.log`，脚本
`script/RUNME-v2-diag-theta-drift.sh`、`script/RUNME-v2-A1-varbuffer-check.sh`。

### A1：修复生效 —— var 指纹翻转

同一配置（`--perturbation_sampling --var_control`，epochs=64 ⇒ M=64×3=192）：

| | round 0 | round 1 | round 2 |
|---|---|---|---|
| **修复前** | var = 1.69e-4 | var = **0.0（精确零）** | **0.0** |
| **修复后** | var = 2.478e-3 | var = 4.526e-3 | var = 5.613e-3 |

`num of fwdgrad: 192` 逐轮一致（= epochs × samples）。**精确零消失 ⇒ 周期复用被打破 ⇒ 修复生效。**
重采样未触发（var ≪ `var_threthod=1`），总耗时 941s，与按 1.579 s/pair 外推的 ETA 吻合到分钟。

var 在此仅作「周期性是否被打断」的定性指标 —— 其量纲是绝对的、不随梯度实际大小缩放，
故按约定**不**当阈值判据使用（A4）。

### A1b：根因从代码推理变成可复现数字

同一批样本、同一 M=3072，**只改方向是否复用**：

```
每次全新 randn (M=3072)   cos(v, ḡ) = 0.2029   （理论 √(3072/d) = 0.2917）
只 3 个方向循环复用        cos(v, ḡ) = -0.0004  （理论 √(3/d)    = 0.0091）
```

复用版落到 **≈ 0**，比我按 `√(3/d)=0.0091` 估的还差 —— 因为 3 个固定方向构成一个
固定的 3 维投影，它与真实梯度的 cos 是零均值随机变量，标准差才是 `√(3/d)`。
⇒ 修复前拿到的不是「弱方向」，而是**实质上没有方向**。

### A2：H1（容量 → 泛化不足）**被否**

沿精确梯度 `−ĝ` 扫 ε，同时记 in-sample（3 条 42 token）与 held-out（test 前 50 条 / 1012 token）：

| ε | θ_init dL(train) | θ_init dL(holdout) | θ_r7 dL(train) | θ_r7 dL(holdout) |
|---|---|---|---|---|
| 0.01 | -3.88e-03 | **+4.13e-05** | -2.84e-03 | -5.57e-05 |
| 0.05 | -1.94e-02 | **+2.40e-04** | -1.48e-02 | -3.00e-04 |
| 0.25 | -9.53e-02 | **+2.11e-03** | -8.87e-02 | -1.95e-03 |
| 0.50 | -1.86e-01 | **+6.71e-03** | -2.13e-01 | -4.32e-03 |
| 1.00 | -3.52e-01 | **+2.47e-02** | -5.38e-01 | +3.37e-03 |
| **ho/train** | — | **-0.011 ~ -0.070** | — | **+0.020 ~ +0.022** |

- **θ_init：沿下降方向走，训练 loss 降、留出 loss 升** —— 方向是反泛化的。
- **θ_r7：留出也单调下降**（ε ≤ 0.5），`ho/train ≈ 0.02`。

⇒ **36K 维的 lm_head-only 子空间里确实存在能降低留出 loss 的方向** ⇒ 「容量不足以泛化」
不成立，**H1 死**。真实情况是**只有约 2% 的样本内收益能迁移**，瓶颈在**方向质量**，
不是可训练容量 —— 这也再次指向已定位的 M=3 根因。

### 补充检查：`holdout L0` 两次打印相同（第 7 位小数）

`check_holdout_sensitivity.py`（同进程、同模型对象内切 θ）结论 —— **不是 bug**：

- `h` 在 θ_init 与 θ_r7 之间**逐位相同**（`|dh|_2 = 0`）。这是**正确**的：backbone 冻结、
  LoRA 只挂在 lm_head ⇒ `h` 与 A/B 无关（正是本诊断所依赖的那条性质）。
- 阳性对照：`dloss(train) = -5.154e-04` ⇒ 参数确实换了、loss 确实随参数变。
- `dloss(holdout) = -1.574e-07` —— 差异在第 7 位小数，6 位打印自然看不出来。

**这组数还是 M=3 根因的第三个独立佐证**：7 轮实际更新（总 ‖Δθ‖=0.35）让训练 loss 动
−5.2e-4、留出动 −1.6e-7；而 A2 显示沿精确梯度走同样的 0.25 步，训练动 −8.9e-2、
留出动 −2.0e-3。**实际更新只拿到「同样位移下精确梯度收益」的约 0.5%** ⇒ 等效 cos ≈ 0.02，
与 A1b 的 −0.0004 一致。且留出并非本征不敏感（精确梯度能推动它 ⇒ 子空间有货）。

## 待办

1. ~~处置阻塞二：lm_head 改走 fp32~~ —— **已完成**，见阻塞二「修复与验证」。
2. ~~恢复被误删的参数落地逻辑~~ —— **已完成**，见阻塞一「修复与验证」。
3. 量一下 Phase 1/2 的注意力 LoRA 是否同样被 bf16 吞（判据 `该层 LoRA 增量 / 该层输出 bf16 ulp`）。
   `modeling_llama_base.py` 的 head **不需要**同步修改：Phase 1 不传 `--lora_lm_head`，
   其 LoRA 挂在 q/k/v/o_proj 上，lm_head 是无 adapter 的普通冻结 Linear。
4. 两个阻塞都修好后，先跑短程（comm_round 3~5）验证 `delta_norm ≠ 0` 且 `[EVAL-PPL]` 不再是位级恒定。
5. ~~把 lr 改成恒定并可配置~~ —— **CLI 已支持**；全量长跑已启动（见上节）。
   待观察：`[EVAL-PPL]`/`[EVAL-PPL-TRAIN]` 斜率是否与「-0.0779 × Σlr」相符，以及 train/test
   是否分离（分离 = 过拟合）。
6. A2 / C5 的结论在修复后重测。
7. `h=0.01` 的硬编码值（`fwdgrad_utils.py:189`）可选优化项：fp32 head 后 rel_err 随 h 单调下降，
   h=1e-3 比 h=1e-2 好约 40 倍。
8. M / 轮数的时间效率权衡（见上「结论 4」）值得单独做一次小实验确认。
