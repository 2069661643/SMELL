# Ailog 260924-162828 — SMELL-v3: predictor/pruned_config 接线进 run_fed + CATV smoke

## 概述

补齐 Step2 遗留缺口：`src/` 无任何入口把训练好的 `predictor.pth` + `pruned_config.pth` 载入推理模型，
CATV 的 `vote_callback` 因此只能吃到随机初始化的 predictor。本次调研 Jenga 加载约定后在 `run_fed.py`
增加 `--predictor` / `--pruned-config` 并实现按剪枝 outdim 重建 + 逐键载入；同时修复 `modeling_opt_smell.py`
中 OPTAttention 忽略 `config.predictor_layers` 的**上游 OPT bug**。GPU2 单轮 CATV smoke 通过。

## Jenga predictor 加载 API 调研结论

1. **保存格式**（`jenga/trainer/predictor_trainer.py:95-108`）：
   `predictor.pth` = `{name: param for requires_grad}`，键名 `model.decoder.layers.N.self_attn.predictor.{q,k}linear{1,2,3}.weight`（OPT：144 张量 = 24 层 × 6）；
   `pruned_config.pth` = `{"layers": [{q1_outdim,q2_outdim,k1_outdim,k2_outdim}, ...]}`（来自 `PrunableAttnPredictor.get_current_outdims()`）。
2. **推理加载约定**（`Jenga/src/experiment/end2end/*/llama_jenga.py` / `hello_world.py:137-174`）：
   先 `pruned_cfg=torch.load(...); config.predictor_layers = pruned_cfg["layers"]`
   → `from_pretrained(config=config)`（建模时按 `config.predictor_layers[layer_idx]` 传 `q*_outdim/k*_outdim` 给 `PrunableAttnPredictorInfer`）
   → `torch.load(predictor.pth)` 后对每个 key 判断 `if k in model.state_dict(): model.state_dict()[k].copy_(v)`。
3. **上游 OPT bug**：`jenga/models/modeling_llama.py:393` / `modeling_llama_2D.py` / `modeling_llama_offload.py` 均消费
   `config.predictor_layers`，但 `jenga/models/modeling_opt.py:138` **未消费**，恒以默认（未剪枝）形状
   （`q1_out=hidden_dim=128`、`q2_out=dim*32`）建 predictor。因此直接 `copy_` 会在 `qlinear2/3`、`klinear2/3`
   形状不匹配（保存 ~2751–2993 vs 默认 2048）——实测若不做本仓修复，`run_fed` 会 shape mismatch。
   `get_current_outdims()` 仅用于保存；OPT 推理无 `set_attn_predictor` 类 API，**唯一约定就是 config + copy_**。

## 变更列表

| 时间 | 操作 | 文件 | 规模 | 说明 |
|---|---|---|---|---|
| 16:20 | 修改 | `src/models/modeling_opt_smell.py` | +17/-1 | `OPTAttention.__init__` 读取 `config.predictor_layers[layer_idx]`，透传 `q1/q2/k1/k2_outdim` 重建剪枝形状；无配置时退回默认形状（等价旧行为） |
| 16:22 | 修改 | `src/fed/run_fed.py` | +60 | 新 CLI `--predictor`/`--pruned-config`（成对校验）；建模前 `config.predictor_layers=...`；建模后 `load_predictor_weights()` 逐键 `copy_`（shape mismatch / 0 载入即报错）；`config.json` 记录路径与载入张量数 |

## Smoke 命令与结果（GPU2）

驱动：`temp/run_smoke_predictor_bg.sh`（setsid nohup，日志 `temp/logs/smoke_predictor_260924-162752/driver.log`）

```
$PY src/fed/run_fed.py --tag a01 --gpu 2 --trainer zoo --catv on --sparse 0.4 --catv-r 0.2 \
  --rounds 1 --local-steps 1 --zo-directions 2 --max-clients 1 --max-train-samples 1 --truncate 1024 \
  --predictor checkpoints/predictor/step2_a01_pos_lora/predictor.pth \
  --pruned-config checkpoints/predictor/step2_a01_pos_lora/pruned_config.pth \
  --out-root temp/smoke_predictor
```

关键输出：

```
[fed] pruned_config loaded path=.../pruned_config.pth layers=24
[fed] predictor loaded tensors=144 skipped=0 path=.../predictor.pth
[fed] catv round 0 mask=consensus_mask_round0.pt anchor_in=3 anchor_out=3 vote_bytes=768 ir_mean=1.0000 ir_min=1.0000
[fed] round 0/0 loss=4.34375 delta_norm=8.2443e+04 cos=None time=0.5s
[fed] cuda peak allocated=1.01GiB reserved=1.05GiB
rc=0
```

结论：`144/144` 全载入且 `skipped=0`，证明 `config.predictor_layers` 重建的形状与 checkpoint 完全一致
（若未生效会 shape mismatch 报错）；CATV `anchor_in/out/vote_bytes/ir_mean` 正常；loss 无 NaN。

## 静态检查 / 自测

- `py_compile src/fed/run_fed.py src/models/modeling_opt_smell.py` → OK
- `$PY src/models/token_selector.py` → PASS（exit 0）
- `$PY src/fed/serial_fedavg.py` → PASS（exit 0）

## 风险 / 待办

- **predictor 只在 CATV 路径被“使用”**：`modeling_opt_smell.forward` 里 predictor 用于块打分/topk（`consensus_mask` 注入），
  vote_callback 上报原始分数。若 `--catv off`，predictor 仍参与块稀疏前向、但投票不收集；加载本身对两种模式都生效。
- **jenga 非本仓子模块**：实际 import 自 `smell-v2` editable 安装的
  `/home/yangyongbo/projects/smell/JengaForMemoryTest/Jenga/src/jenga`（`third_party/Jenga/src` 为空）。
  本仓改动不依赖修改 jenga，但环境漂移（editable 指向变化）会让行为不可复现。
- **peft 0.19.1 漂移**：`build_lora_model`/`PeftModel.from_pretrained` 会冻结非 adapter 参数（含 predictor），
  保证 predictor 不进入 FedAvg trainable state（smoke `trainable_params=1572864` = 仅 LoRA，符合预期）。
  升级 peft 后需复核冻结语义。
- **ZOO 步长**：smoke `delta_norm=8.24e+04` 仍是 lr=1e-3 的已知问题（>1 轮会 NaN），与本次接线无关。
- `--predictor`/`--pruned-config` 成对且须与模型层数一致；`pruned_config.layers` 中的 `None` 层按未剪枝重建（当前 24 层均非 None）。
