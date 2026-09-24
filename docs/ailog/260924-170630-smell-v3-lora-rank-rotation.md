# Ailog 260924-170630 — Phase 3: SMELL-v3 LoRA 秩轮换/锁定（掩码 ZOO/BP）

> 分支 `exp/rank-rotation-lora`。目标：r=8 LoRA 支持「每轮只激活若干秩」，并让 ZOO 的差分
> 维度只含激活秩（1,572,864 → 196,608），降低前向梯度估计方差；非激活秩逐位冻结。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 17:00 | 新增 `build_active_index` | `src/train/zoo.py` | L13-51 | 按平坦偏移把 active_ranks 映射为 LongTensor 索引（A 行/B 列） |
| 17:00 | 修改 `flatten_trainable/apply_flat_delta/_restore_flat` | `src/train/zoo.py` | 全函数 | 增 `index` 参数；None 走原路径（逐位兼容），否则取子空间/散射回全长 |
| 17:00 | 修改 `zo_grad` | `src/train/zoo.py` | L107-150 | 增 `index`；方向/扰动/还原只在子空间，`dim=len(index)`；返回仍为全形状（非激活为 0） |
| 17:00 | 扩展 `_run_selftest` | `src/train/zoo.py` | L188+ | 新增偏移/行-列取值/掩码往返/非激活冻结 4 项断言 |
| 17:02 | 新增 `num_lora_ranks` / `active_ranks_for` | `src/train/lora.py` | L29-58 | 取秩数；rotate（轮换）/lock（固定前 k）/all（None）生成激活秩 |
| 17:03 | 修改 `ClientRunner` | `src/fed/serial_fedavg.py` | `__init__` / `run` | 增 `active_index/active_ranks`；ZOO 传 index；BP step 后把非激活位置还原为初始快照；result 增 `active_ranks` |
| 17:04 | 修改 `run_fed.py` | `src/fed/run_fed.py` | CLI/config/round loop | 新增 `--rank-mode/--rank-k`；每轮算 active_ranks/index 并传入；通信量按激活子空间计；config/record 记录 |

## 语义（已定）

- 每模块 r=8：秩 i = `lora_A[i,:]`（行）+ `lora_B[:,i]`（列）。
- `rotate`：`S = {(round_idx*rank_k + j) % num_ranks | j in range(rank_k)}`；`lock`：`{0..rank_k-1}`；`all`：None（无掩码）。
- 全客户端同一调度（仅由 `round_idx` 决定），保证服务器端 FedAvg 聚合语义正确。
- 聚合未改：仍对**全形状 delta** 加权平均，非激活 delta=0。

## Smoke 证据（GPU2，tag=a01，truncate=1024，pos_lora adapter）

日志：`temp/logs/rankrot_smoke_260924-170521/driver.log`（`temp/` 不入库）。

- ZOO `--rank-mode rotate --rank-k 1 --rounds 2`：`active_ranks=[0]`→`[1]`，`comm_bytes=786432`（=2 client×196608×2），loss 3.20→9.25（无 NaN）。
- BP `--trainer bp --local-steps 2 --rank-mode rotate`：`active_ranks=[0]`→`[1]`，loss 3.218→3.154，无 NaN。
- ZOO `--rank-mode all`：`active_ranks=None`，`comm_bytes=3145728`（=1×1572864×2），与旧行为一致。
- 一次性脚本（`temp/rankrot_check.py`）：`num_ranks=8`、round0 `active_ranks=[0]`、**ZOO dim=196608**；
  ZOO/BP 各自 `torch.equal(after[inactive], before[inactive])=True`（非激活逐位不变）；result `active_ranks=[0]`。

## 静态检查 / 自测

- `py_compile`：zoo.py / lora.py / serial_fedavg.py / run_fed.py 全部 OK。
- `src/train/zoo.py` 9 项全 PASS（含原有用例，index=None 路径未变）。
- `src/fed/serial_fedavg.py`、`src/models/token_selector.py` 均 PASS。

## 风险 / 已知限制

- **基依赖未做 SVD 重整**：秩轮换只改 `A`/`B` 中激活秩的行列，`BA = Σ_i B[:,i]A[i,:]` 的贡献可分解，
  但每轮只更新 1/8 秩 ⇒ 单轮有效学习率/覆盖下降，收敛需更多轮；若后续要做秩合并需 SVD 重整。
- **每轮只训 1/8 参数**：预计收敛变慢，需在正式消融中对比轮数/质量。
- **ZOO 算力不变**：中心差分数值次数由 `--zo-directions` 决定，掩码只降维（方差 ↓、单次 dim 收缩），
  总前向次数不变。
- BP 非激活靠 step 后快照-还原，AdamW 的动量缓冲仍会为非激活参数更新（仅参数值被还原）；
  若后续要严格省算力，应按 index 只建激活参数优化器。

## 待办

- 在正式消融中加入 `rank-mode ∈ {all, rotate, lock} × rank_k` 组，评估收敛轮数与质量。
- 视需要实现「激活秩 SVD 合并」与「BP 仅激活参数优化器」。
