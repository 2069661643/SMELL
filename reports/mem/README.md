# SMELL 显存交付包：ZOO（真实 `zo_grad`）vs BP，1k–16k

由 `reports/mem/make_report.py` 从 `temp/mem_sweep_zoo/` 与 `temp/mem_sweep/` 的原始 JSON 自动生成；纯 CPU、确定性；**不含图片**。禁止手改本目录数字，改脚本后重跑。

## 范围与来源

- 模型：**OPT-350M**（`third_party/Jenga/checkpoints/opt-350m`；24L / hidden 1024 / 16 heads / vocab 50272，max_pos 2048 由 `ensure_positions` 扩到 16k），bf16 权重 + autocast，flash/SDPA；
- LoRA：qkvo，r=8（bench 默认，**不是**主线的 r=1）；单卡 RTX 4090 24GB，`cap_fraction=0.9`；
- 输入：单条随机 `input_ids`，长度 1024/2048/4096/8192/16384，单步；指标为 `torch.cuda.max_memory_allocated()`（peak_allocated，非 reserved）。

| 模式 | 实现 | 说明 |
|---|---|---|
| **ZOO** | `src/train/zoo.py::zo_grad` | 真实实现；每次 forward 在 `torch.no_grad()` 下；D=8、eps=1e-3；D 只影响时间不影响峰值 |
| **BP** | training-mode forward + `loss.backward()` | 标准反传，autograd 保留 activation |
| *graph-forward*（参考） | training-mode forward（不 backward） | 仍建图，≈BP−0.2 GiB；**不是** ZOO 路径，仅用于口径对照 |

## 主表（GiB）

| seq | ZOO dense | ZOO sparse | ZOO 省 | BP dense | BP sparse | BP 省 | BP/ZOO (dense) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 1.02 | 0.99 | 2.9% | 2.00 | 1.88 | 5.9% | 1.96× |
| 2k | 1.31 | 1.26 | 3.7% | 3.28 | 3.07 | 6.5% | 2.50× |
| 4k | 1.50 | 1.39 | 7.4% | 4.89 | 4.45 | 9.0% | 3.25× |
| 8k | 1.89 | 1.66 | 11.9% | 8.10 | 7.19 | 11.2% | 4.28× |
| 16k | 2.66 | 2.21 | 16.9% | 14.51 | 12.69 | 12.5% | 5.45× |

## 关键结论

1. **ZOO 比 BP 省 1.96×（1k）→ 5.45×（16k）**；16k 时 ZOO 2.66 GiB（sparse 2.21）vs BP 14.51（sparse 12.69）。
2. ZOO 斜率 ≈ 0.1 GiB/1k（权重 + 单次激活，线性）；BP 斜率 ≈ 0.9 GiB/1k（autograd 图）。
3. 稀疏 0.4 的节省：ZOO 2.9% → 16.9%（随 L 上升），BP 5.9% → 12.5%；**相对值 ZOO 更大、绝对值 BP 更大**（16k 绝对节省：ZOO 0.45 GiB vs BP 1.82 GiB）。
4. ZOO 16k 仅 ~2.2–2.7 GiB → 8GB 卡可跑；BP 16k 需 12.7–14.5 GiB。

## 口径 caveat

- bench 口径（bf16/flash/LoRA r8）≠ 主线（fp32 + LoRA r1）；横向对比（ZOO vs BP、dense vs sparse）有效，绝对值不可直接当主线显存。
- `peak_allocated` 是 PyTorch 分配器口径（非 reserved，也不含 CUDA context）；不同驱动/框架版本会有 ±0.1 GiB 级差异。
- ZOO 的 D 只影响耗时（每次方向 2 次前向），不影响峰值；本表 D=8。
- 模型 provenance：ZOO 10 组 JSON 自带 `model_dir/dtype/lora_r/device_name`（`meta_source=json`）；BP 10 组为同一 bench 默认值，脚本按默认补记（`meta_source=bench_default`），两者同脚本同口径。
- 早期 `mem_sweep` 的 “forward” 模式是建图前向（≈BP−0.2 GiB），只以 `mem_tidy.csv` 的 `mode=forward` 行保留作口径对照，**不应**用于 ZOO 结论。

## 文件索引

| 文件 | 行数 | 作用 |
|---|---:|---|
| `data/mem_tidy.csv` | 30 | 逐 run（mode × seq × kind）的峰值/耗时/状态/来源 |
| `data/mem_summary.csv` | 5 | 宽表：ZOO/BP × dense/sparse、节省 %、BP/ZOO 比 |
| `data/provenance.csv` | 30 | 每个 JSON 的路径、mtime、状态（含缺失行） |

## 复现

```bash
PY=~/miniconda3/envs/SMELL/bin/python
# 1) 测量（GPU；ZOO 只跑 5 长度 × dense/sparse）
bash temp/run_mem_sweep_zoo_only_bg.sh
# 2) 生成交付包（CPU，含对账闸门）
$PY reports/mem/make_report.py
```

## 对账

- ZOO 10/10、BP 10/10 行均 `status=ok`；16k 四个端点与 ailog `261105` §八 一致（±0.02 GiB）；
- 所有长度 ZOO < BP；ZOO/BP 的 dense→sparse 节省率随 L 单调上升。
