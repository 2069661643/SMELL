# Ailog 260929-111500 — Phase N: TD-3@a03 离群爆炸（client_22）根因 + per-client delta clip 修复

> lr=0.15 的 TD-3@a03 在 round 0 即被**单个离群 client** 毒化爆炸（PPL 33.7 → 10752 → 33254），
> 已停机（2 轮留档 `logs/fed/zoo_k4_sparse_causal_a03_lr0.15/`）。本 ailog 记录根因、修复（`--delta-clip`）与重跑。

## 一、失败证据（lr=0.15，30 client）

| 轮 | loss | δ_mean | PPL | 说明 |
|---|---|---|---|---|
| 0 | 3.6186 | **810.5** | **10752** | 29 个 client δ∈[0.33, 1.05]（正常）；**client_22 单点 δ=2.43e4** |
| 1 | 9.4684 | 5.15e4 | 33254 | 毒化后全体 δ 2e4–1.5e5；模型已毁 |

- `client_22` round-0 平均 loss=6.5（其余 ~3.5）；其数据核查**正常**（query/answer/span 完整、词表/重复度/标签与其他 client 一致），非数据损坏。
- 反推 `zo_grad` 的有限差分 `ΔL≈δ·2h/(lr·dim)`：正常 client ~1e-7（≈fp32 loss 量化底噪），client_22 ~**0.02**（高 5 个量级）。

## 二、根因

1. **估计器重尾**：`g = (1/D)·Σ dim·(ΔL/2h)·v`，dim=32768 与 1/(2h) 把 ΔL 异常放大 3e4×；一个高 loss 点的 ΔL 异常即可产生 δ~1e4。
2. **最可能的微观机制**：稀疏前向 `OptSdpaPruneAttention` 的 **离散 top-k token 选择在 θ±hv 扰动下翻转**——一次 64-token 块换入/换出造成的 loss 跳变 ≈ `(64/16384)×~5 ≈ 0.02`，与反推值同量级。（备选解释：该点真实梯度/曲率大。二者都需要冻结选择 vs 稀疏对照 probe 才能定论。）
3. **放大因素**：① 全链路无裁剪；② v3 改为加权平均（v2 的 `‖Δθ‖≡lr` 归一化保护被弃用），单个离群 client 可直接毒全局；③ D=22 估计方差大；④ lr=0.15 再乘上去。
4. **D 扫描盲区**：只用 client_00（正常点），未暴露跨 client 离群——preflight 需要 client 级 δ 审计或自带裁剪。

## 三、修复（本案）

- `src/fed/run_fed.py` 新增 **`--delta-clip`（默认 1.0，<=0 关闭）**：
  - 聚合前对每个 client 的 delta 做全局 L2 范数裁剪（保方向、仅缩放超限者）；
  - `client_stats` 新增 `delta_clip_applied`；round 记录新增 `clipped_clients`；`[fed] round` 日志打印 `clipped=N`；
  - `delta_norm` 统计保留**原始**范数（供诊断离群）。
- `scripts/run_k4_td2_td3.sh`：TD-3 阶段加 `--delta-clip`（env `DELTA_CLIP`，默认 1.0）。
- 阈值依据：16k round-0 正常 δ∈[0.33, 1.05]，裁剪 1.0 只影响离群；被裁 client 对聚合的相对贡献 ~1/30（原 810/0.45≈1800×）。

## 四、验证（smoke）

- `temp/smoke_delta_clip_1.0` / `_1e-4`（2k、1 client、D=4）：`clipped_clients=1`、`delta_clip_applied=True` 正确落盘；两次结果一致（确定性），代码路径生效。
- `py_compile` OK；pipeline 脚本 `bash -n` OK。

## 五、重跑

- `temp/run_td3_a03_lr015_clip_bg.sh`：lr=0.15、`--delta-clip 1.0`、30 client × 30 轮、每轮 eval+逐样本；
- 产物 `logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/`；入口 `temp/logs/last_td3_a03_lr015_clip_dir.txt`；
- **首轮检查点**：`clipped_clients` 预期=1（client_22），`delta_norm_mean` 应回落 ~0.4 量级，PPL 应从 33.7 附近正常起步。

## 六、待办

- [ ] 首 3 轮核对 `clipped_clients`/δ/loss/PPL；round 0 vs N 配对检验（`scripts/paired_eval.py`）；
- [ ] 可选加固：trimmed-mean/median 聚合、`zo_grad` 逐方向 ΔL 裁剪、dense-vs-sparse 选择稳定性 probe；
- [ ] 云端 pull 后按同参数（含 `--delta-clip 1.0`）重跑。
