# Ailog 261105-153500 — OPT fixed 四臂续训 r30→r59 完成 + eval-off 归因 + 显存矩阵（1k–16k dense vs sparse）

> 用户指令（10/30）："给 opt fixed 4arm 续训到 60r"；10/31 追加"eval-off 挂在链条后"；11/4 追加
> "1/2/4/8/16k 上下文下 dense v.s. sparsity 0.4 的内存使用"。
> 主线：`temp/run_opt_cont30_resume_bg.sh`（修复 out-root 后），10/30 19:11 启动 → 11/4 23:25 完成；
> eval-off 链 11/5 00:40 完成；显存矩阵 11/5 00:47 完成。GPU 现已空闲。

## 一、OPT fixed 四臂 r30→r59（lr=0.30，首段续训保持原 lr）

| 臂 | r29 full / answer | **r59 full / answer** | full Δ | answer Δ | 时长 |
|---|---:|---:|---:|---:|---:|
| a01_off | 31.5334 / 2436.1 | **30.9529 / 2174.8** | −1.84% | −10.73% | 23.9h |
| a01_on | 31.6351 / 2474.4 | **31.1256 / 2277.1** | −1.61% | −7.98% | 23.9h |
| a03_off | 31.4568 / 2448.8 | **30.8639 / 2362.6** | −1.88% | −3.52% | 24.4h |
| a03_on | 31.6957 / 2291.7 | **30.9586 / 1504.2** | −2.33% | **−34.37%** | 24.0h |

配对（n=500，r29→r59）：
- full：−0.0162 ~ −0.0235，全部 p<1e-16；
- answer：a01_off −0.1021（p=2.0e-8）、a01_on −0.1118（p=8.2e-6）、a03_off −0.0249（**p=0.30 n.s.**）、a03_on **−0.4190**（p<1e-16）。

CATV on-vs-off @r59：
- a01：full +0.0056（p<1e-16，量级<0.02%）、answer +0.0166（**p=0.54 n.s.**）→ 无收益；
- a03：full +0.0031（p<1e-16，量级<0.01%）、answer **−0.4873（p<1e-16）** → 巨大收益。

## 二、eval-off 归因（on 臂无掩码评测，r34/39/44/49/54/59）

口径：`dNLL = on-mask − off-mask`，负 = mask 更好；权重效应用 `on-arm(off-mask) − off-arm`。

| r59 | mask 效应 full | mask 效应 answer | 权重效应 full | 权重效应 answer | 净 answer |
|---|---:|---:|---:|---:|---:|
| a01 | +0.0072（更差） | −0.1641（更好） | −0.0005 | **+0.1807（更差）** | +0.0166（n.s.） |
| a03 | +0.0072（更差） | −0.0528（更好） | −0.0042 | **−0.4345（更好）** | **−0.4873** |

**结论（与 BERT 相反）**：OPT a03 的 CATV answer 收益 **主要是权重效应**（−0.435 / −0.487 = 89%），mask 只贡献 −0.053；
而 a01 是"权重变差 +0.181、mask 变好 −0.164"相互抵消 → 净 0。
两个 α 上 mask 对 **full PPL 都是轻微负作用**（+0.007 nats，逐轮稳定）。

## 三、显存矩阵（1k/2k/4k/8k/16k，dense vs sparse 0.4）

`src/bench/memory.py`，OPT-350M，bf16 + flash，LoRA qkvo r=8（bench 默认），单卡 4090，cap 0.9，单步。

| seq | forward dense | forward sparse | 省 | backward dense | backward sparse | 省 |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 1.90 | 1.78 | 6.2% | 2.00 | 1.88 | 5.9% |
| 2k | 3.08 | 2.87 | 7.0% | 3.28 | 3.07 | 6.5% |
| 4k | 4.68 | 4.24 | 9.4% | 4.89 | 4.45 | 9.0% |
| 8k | 7.89 | 6.98 | 11.5% | 8.10 | 7.19 | 11.2% |
| **16k** | **14.30** | **12.48** | **12.7%** | **14.51** | **12.69** | **12.5%** |

- 节省随上下文单调上升（6.2% @1k → 12.7% @16k），因为 NLL 部分的注意力/激活随 L² 增长而 Jenga 只保留 40% 块。
- 16k backward dense 在 24GB 上可跑（14.5 GiB），未 OOM；8GB 卡会 OOM（历史结论）。
- 产物：`temp/mem_sweep/summary.md` / `summary.csv` / `mem_<mode>_<kind>_<seq>.json`。

## 四、事故与修复（必须记录）

第一版脚本 `run_opt_cont30_4arm_bg.sh` 把四臂写到**同一个 out-root**，导致：
- a01_on 的第二次运行覆盖了 a01_off 的 `adapter_round030..059` 与 `eval_round*`；**metrics 因 append 存活**；
- 归档路径同样写错 → 两臂 ARCHIVE SKIPPED；
- a03_off 当时 0 轮完成（kill 只损失 ~1h）。

**抢救**：
- a01_on：adapter + metrics + 6 eval JSON/persample 归档（sha `33f444e3…`）；
- a01_off：`metrics`-only 归档 + `LOST_WEIGHTS.md`；原始目录保留 `logs/fed/opt_r30_fixed_cont30/a01_collided_off+on/`；
- 重跑 a01_off 验证**逐位确定性**：r34（31.398/2657.5）、r39（31.255/3099.6）与原 run 完全一致；r59 复现 30.9529/2174.8，归档补齐（sha `9570d21a…`）。

## 五、产物

`checkpoints/opt_ablation_r30_fixed_cont30/<arm>/`（adapter_round059 + metrics + config + 6 eval JSON + 6 persample + SHA）：

| 臂 | adapter_round059 sha256 |
|---|---|
| r30_a01_catv_off | `9570d21ae4658a9b3252e421a71bc692b9b7f731b2267fa321ab84eaafe09c12` |
| r30_a01_catv_on | `33f444e367d3ef64ab937cc4726e84ea99661e0655b4ca1170c64777e1ef28d1` |
| r30_a03_catv_off | `99ce218eedf2da5f01002dd5f3f0c406c1491e772b382a8f498c0f285a78421b` |
| r30_a03_catv_on | `639d3f2bb935a8e37f99b7a3ef939bdf6ef389daee7fad3f1f0917192182735a` |

- on 臂另有 `evaloff_round034..059{,_persample}.json`（各 12 文件）。
- 报告：`temp/logs/opt_cont30_report.txt`、`temp/logs/opt_cont30_evaloff_report.txt`、`temp/mem_sweep/summary.md`。

## 六、caveat

1. **answer PPL 基数小**：500 样本仅 1492 个 answer token，单值波动大（a01_off 在 r39 到过 3099 又回落 2175）；结论用配对 p 值，但论文建议补 bootstrap CI。
2. **显存 bench 口径**：bf16 + flash + LoRA r8，不是主线 fp32 + r1；`peak_allocated`（非 reserved）；单步、单进程。
3. **eval-off 的 mask 效应**：on 臂 adapter 固定，仅切换推理期掩码；mask 由该轮 votes 生成，r59 的 mask 与权重不是"同轮最优"配对。
4. a01_on 的 adapter 在事故目录（r59 sha 与归档一致，已核验）。

## 七、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10/30 19:11 | 启动 | `run_opt_cont30_4arm_bg.sh` | 4 臂续训（out-root 冲突） |
| 11/1 21:16 | 重启 | `run_opt_cont30_resume_bg.sh` | 每臂独立 out-root；a03×2 + a01_off 重跑 |
| 11/4 23:25 | 完成 | 4 臂 r59 + 归档 + 报告 | — |
| 11/5 00:40 | 完成 | eval-off（2 臂 × 6 点） | mask/权重分解 |
| 11/5 00:47 | 完成 | 显存矩阵（20 组） | `temp/mem_sweep/summary.md` |
| 11/5 15:3x | 修改 | `AGENTS.md` / `docs/weight-manifest.md` | cont30 行 + §15 |
| 11/5 15:3x | 新增 | 本 ailog | — |

## 八、显存口径修正（261105 15:4x）：ZOO vs BP

用户指出 "BP v.s. 我们实现的 zoo"，重测 ZOO 路径（仅 ZOO，BP 用既有数据）：

| seq | ZOO dense | ZOO sparse | ZOO 省 | BP dense | BP sparse | BP 省 | BP/ZOO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 1.02 | 0.99 | 2.9% | 2.00 | 1.88 | 5.9% | 1.96× |
| 2k | 1.31 | 1.26 | 3.7% | 3.28 | 3.07 | 6.5% | 2.50× |
| 4k | 1.50 | 1.39 | 7.4% | 4.89 | 4.45 | 9.0% | 3.25× |
| 8k | 1.89 | 1.66 | 11.9% | 8.10 | 7.19 | 11.2% | 4.28× |
| **16k** | **2.66** | **2.21** | **16.9%** | **14.51** | **12.69** | **12.5%** | **5.45×** |

- 代码：`src/bench/memory.py` 新增 `--mode zoo`（直接调用 `src/train/zoo.py::zo_grad`，D=8、eps=1e-3，前向在 `no_grad` 下）；`--zoo-directions/--zo-eps`。
- 结论：**ZOO 的显存随上下文近似线性（~0.1 GiB/1k），BP 在长上下文快 0.9 GiB/1k** → 16k 时 ZOO 仅 2.66 GiB（BP 14.51），**5.5×**；且 ZOO 在 8GB 卡上可跑 16k。
- 稀疏节省：相对值 ZOO 更大（16.9% vs 12.5%），绝对值 BP 更大（1.82 vs 0.45 GiB）。
- 原 "forward" 列实为 training-mode 前向（建图，≈BP−0.2 GiB），不能当作 ZOO。
- 产物：`temp/mem_sweep_zoo/summary.md`。
