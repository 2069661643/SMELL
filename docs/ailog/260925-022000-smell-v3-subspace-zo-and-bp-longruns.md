# Ailog 260925-022000 — SMELL-v3: 逐模块 δ 记录 + 子空间 ZO（`--zo-subspace`）+ 长跑改序

## 概述

在 `exp/rank-rotation-lora` 分支上归档两类工作：(1) 为诊断「ZOO 该更新哪些块」新增
**逐层/逐投影 δ 记录**与**块/子空间 ZO**（`--zo-subspace {all,layers,modules}`），复用既有
`active_index` 管道把 ZO 更新限制在选定层或投影；(2) 因今晚 GPU0/1/3 被其他用户占满、原
A1/A34 曾 OOM，改用 **GPU2 串行新顺序长跑**：先 A34（BP r=1 `c=3` → `c=10`），再 A1（`c=30`）。
两条烟测（layers/modules）均通过，长跑已后台启动。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 02:05 | 新增逐层/逐投影 δ 拆解 | `src/fed/run_fed.py` | `delta_norm_by_group()`（`_LAYER_PATTERN`/`_PROJ_PATTERN`），标注 `per_module_delta BEGIN/END` | 把 client δ 拆成逐层 L2（`layers.{i}.`，无匹配归 `other`）与逐投影 L2（`<layer>.<proj>`，仅 self_attn q/k/v/out） |
| 02:06 | 修改 round 记录 | `src/fed/run_fed.py` | client 循环 `client_stats[...]`、round `record` | client_stats 增 `delta_norm_by_layer`/`delta_norm_by_module`；round 增跨 client 均值 `delta_norm_by_layer_mean`（缺失层按 0） |
| 02:08 | 新增子空间 CLI | `src/fed/run_fed.py` | `parse_args()`（标注 `subspace_zo ADD`） | `--zo-subspace {all,layers,modules}`、`--zo-layers`（如 `'0-2,5'`）、`--zo-modules`（如 `'0.q_proj,1.k_proj'`） |
| 02:09 | 新增参数校验/解析 | `src/fed/run_fed.py` | `parse_layer_spec()`/`parse_module_spec()`、`main()` 校验（`subspace_zo BEGIN/END`） | 与 `--rank-mode` 互斥（非 all 时 rank_mode 必须 all）；`layers` 缺 `--zo-layers`、`modules` 缺 `--zo-modules` 报 `SystemExit`；空解析报错 |
| 02:10 | 新增子空间索引构造 | `src/fed/run_fed.py` | `main()` 建模型后、round 循环内（`subspace_zo MODIFIED`） | 调 `build_param_index` 得固定 `subspace_index`，非空断言；round 内覆盖 `active_index`（rank_mode=all）；`active_index_size` 写入 config.json |
| 02:03 | 新增按名选参 | `src/train/zoo.py` | `build_param_index(model, layers, modules)`（标注 `subspace_index BEGIN/END`） | 按参数名 `layers.{i}.` / `layers.{i}.self_attn.{proj}.` 匹配，返回 CPU `LongTensor` flat 索引；同时给出取并集；两者皆 None 返回 None；无匹配 `ValueError` |
| 02:04 | 新增自测 | `src/train/zoo.py` | `_run_selftest()`（`subspace_index ADD`） | 8 项 `build_param_index` 用例（单层/多层/投影/多投影/全层/None/无匹配报错/取值） |

## 验证证据

- `py_compile`（`~/applications/anaconda3/envs/smell-v2/bin/python`）：`src/fed/run_fed.py`、`src/train/zoo.py` 均 OK。
- `src/train/zoo.py` 自测 **17/17 PASS**（原 9 项 + 新增 8 项 `build_param_index`）。
- `src/fed/serial_fedavg.py`、`src/models/token_selector.py` 自测 PASS（回归基线）。
- config.json 记录 `zo_subspace`/`zo_layers`/`zo_modules`/`active_index_size`，可复现。

## 子空间 ZO smoke 结果

| smoke | 选择 | `active_index_size` | δ 行为 |
|---|---|---|---|
| `logs/smoke_blockzo/layers/a01` | `--zo-subspace layers --zo-layers 0-2` | **24576** | 仅层 0/1/2 有 δ；层 3–23 全 0；无 NaN |
| `logs/smoke_blockzo/modules/a01` | `--zo-subspace modules --zo-modules 0.q_proj,1.k_proj` | **4096** | 仅 `0.q_proj`/`1.k_proj` 有 δ，其余投影/层全 0；无 NaN |

- 两例均 2 轮、c=2、truncate=1024、ZOO dir=8、lr=1e-7，`delta_norm_by_layer(_mean)` 与
  `delta_norm_by_module` 落盘正确，掩码粒度符合预期（元素数 = 层数×4 投影×LoRA r1 参数）。
- 证明 `build_param_index` → `active_index` 管道端到端可用，块级 ZO 通信量随子空间线性下降。

## 长跑启动配置与新顺序（`temp/run_a34_then_a1_bg.sh`，GPU2）

- 02:19:13 启动，`temp/logs/a34_then_a1_260925-021913/`，pid 见 `temp/logs/last_a34_then_a1.pid`。
- 公共配置：`--tag a01 --gpu 2 --sparse 0.4 --catv off --pos-checkpoint <pos_only_500step> --lora-r 1 --lora-alpha 2 --trainer bp --bp-clip 1.0 --local-steps 4`，**无 `--adapter-init`**。
- 顺序：`a34_c3`（c=3, rounds=300, eval20）→ `a34_c10`（c=10, rounds=300, eval20）→ `a1_bp_r1_c30`（c=30, rounds=300, eval10）。
- 改序原因：今晚 GPU0/1/3 被他用户占满（`lijh2125` VLLM / `sys21021` / `aeonia`），原 A1/A34 曾 OOM；改为单卡 GPU2 串行、先小 client 数确认稳定性再上 c=30。
- 启动日志含 bitsandbytes `libcusparse.so.11` 原生库加载失败（非致命，bitsandbytes 内部捕获降级），主流程正常加载模型并进入训练。

## 待办 / Caveat（用户构思，原样保留）

- **想法2**：把全部/部分 LoRA 权重重组为矩阵，再在其上做 LoRA/低秩（文献：FwdLLM+ 摘要 + SubZero / LoRA-XS / LoRA-SB；疑无 ZO 版）。
- **想法3**：每次前向加随机掩码、只更新部分元素（Sparse MeZO / MeZO-BCD / ZO-BCD）。
- **想法4**：BP 统计各模块更新幅度 → 概率分布 → ZOO 按分布采样要更新的模块（无直接先例，风险最高）。
- **信息泄露 caveat**：想法4 的分布**不能来自真实 c30 测试集**，须来自 pos/predictor 训练集；本次先沿用 A1 的 BP 统计。
- **ZOO lr 重设计待办**：`cos≈0` 下 lr 探针无意义，待方向修复后再设计。

## 风险

- 子空间 ZO 仅验证「掩码/索引/通信/δ 落盘」正确性，未验证其 PPL/收敛收益；块级更新可能因丢失其它模块梯度而变慢。
- 长跑为单卡串行，GPU2 若被抢占或再次 OOM 会中断；需在完成后按 `last_a34_then_a1_dir.txt` 校验残留与产物。
- `delta_norm_by_layer_mean` 把未激活层按 0 计入均值，跨不同子空间的实验不宜直接横向比较。
- 想法4 的概率分布存在训练/测试信息泄露风险，须严格限定统计来源。
