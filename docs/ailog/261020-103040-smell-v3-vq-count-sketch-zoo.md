# Ailog 261020-103040 — SMELL-v3：VQ/Count Sketch 接入 ZOO 上行（`--vq-rate`，默认关闭）

> **口径已被 `261020-162000` 取代**：本文的 rate=总预算（`C*c_dim/d`）；新架构为 per-result
> `c_dim=rate*d`（每条训练结果都上传）。旧的 0.25/0.5 长跑已停，partial 存档。

> 用户指令：把 `third_party/VQ` 的量化压缩加入训练，设计压缩率 CLI（默认 1 = 不开）；
> 压缩率按“总 sketch 元素 / 活跃参数”定义；channel 数 = 一轮 ZOO 前向样本数（`local_steps * zo_directions`）；
> 开工前先定 clip 语义。分支：`exp/vq-compression`（从 `exp/distilbert-16k`/本地 `main` `37e5708` 拉出）。

## 一、定稿口径

| 项 | 决定 |
|---|---|
| `--vq-rate` | 默认 `1.0` = 关闭；`0<rate<1` 开启。总预算 `C*c_dim/d_active`，与 `third_party/VQ/script.py` 一致 |
| channel 数 | `C = local_steps * zo_directions`（每个中心差分方向样本 1 channel；`--vq-channels` 仅诊断覆盖） |
| `c_dim` | `max(1, floor(rate * d_active / C))`；实际 `vq_actual_rate = C*c_dim/d_active` 落盘 |
| hash | 每个 channel 独立 hash/sign seed；所有 client 共享同一 seed 序列（`--vq-seed` 默认 `--seed`）；`vq_seeds.json` 落盘 |
| clip | **先按现有 `--delta-clip` 对完整 delta 做全局 L2 clip，再 VQ**；解压后默认 `--vq-rescale on` 缩放到 clip 后范数 |
| trainer | **仅 ZOO**；`--trainer bp --vq-rate<1` 直接 `SystemExit` |
| 通信 | sketch fp32（4 B/element）；`vq_communication_bytes` 单独记录，`communication_bytes` 保留 dense bf16 口径 |

**为什么不使用“C 个 channel 完全同 hash”**：同 hash 且都压缩同一最终 delta 时，C 个 channel 会退化成 1 个只有 `c_dim` 个桶的 sketch，有效 cos≈√(rate/C)，比名义 rate 差 C 倍。本实现按“跨 client 共享同一套 seed、channel 间独立 seed”落地。

## 二、代码改动

| 文件 | 改动 |
|---|---|
| `src/compress/count_sketch.py` | 新增 `CountSketch`：整数 hash/sign、`gcd(a,c_dim)=1`、`compress/decompress/roundtrip/cosine`、seed 缓存、`_selftest` |
| `src/fed/run_fed.py` | 新增 `--vq-rate/--vq-rescale/--vq-channels/--vq-seed/--vq-bytes-per-element`；`flatten_delta/scatter_delta/resolve_vq_channels`；client clip→VQ→decompress→rescale；round/client metrics；`vq_seeds.json` |
| `AGENTS.md` | 更新分支/当前 OPT+BERT r30 结果/运行口径；新增 VQ 段与硬坑 |

- 未修改 `third_party/`；算法语义从 `third_party/VQ/main.py` 复制到 `src/compress/`。
- `rate=1` 时完全不构造 `CountSketch`，基线路径逐位不变（独立 `numpy.random.default_rng`，不消耗 torch 全局 RNG）。
- active 子空间：只压缩 `active_index` 选中的参数；非 active 位置严格为 0，不引入 hash 噪声。

## 三、CLI 与 metrics

```text
--vq-rate 1.0                 # 总预算/d_active；1=off
--vq-rescale {on,off}         # 默认 on
--vq-channels 0               # 0=auto L*D
--vq-seed None                # 默认 --seed
--vq-bytes-per-element 4      # fp32 sketch
```

每轮 `metrics.jsonl` 新增：`vq_enabled/vq_rate/vq_channels/vq_c_dim/vq_active_dim/vq_actual_rate/vq_rescale/vq_seed/vq_cos_mean/vq_cos_min/vq_cos_max/vq_communication_bytes/vq_ratio_elements/vq_ratio_bytes`；每 client 新增 `vq_cos/vq_delta_norm/vq_client_bytes`。

## 四、smoke 结果（AutoDL 4090，`--gpu 0`）

配置：BERT 6 层、`--zo-layers 0-5`、L=2、D=2、C=4、2 clients × 2 samples、`--truncate 2048`、LoRA 默认 r=8（d_active=294,912；仅 smoke，正式跑用 r=1）。OPT：1 client × 1 sample、`--zo-layer-rotate --zo-layer-group 1`、L=1、D=2、C=2（d_active=65,536）。

| 模型 | rate | C | c_dim | actual | `vq_cos_mean` | `vq_ratio_elements` | `vq_ratio_bytes` | 说明 |
|---|---|---|---|---|---|---|---|---|
| BERT | 0.5 | 4 | 36,864 | 0.500000 | **0.6302** | 0.500 | 1.000 | fp32 sketch 4B vs dense bf16 2B |
| BERT | 0.25 | 4 | 18,432 | 0.250000 | **0.4838** | 0.250 | 0.500 | |
| BERT | 0.1 | 4 | 7,372 | 0.099946 | **0.3104** | 0.100 | 0.200 | |
| OPT | 0.1 | 2 | 3,276 | 0.099976 | **0.2987** | 0.100 | 0.200 | layer 0 active |

- **理论对照**：总预算 `N=rate*d` 时 Count Sketch 的全向量重建 `cos≈√(rate/(1+rate))`（0.577/0.447/0.301），实测 0.630/0.484/0.310，吻合。**不是 √rate**；`√rate` 只在 `N≫d` 时近似成立。
- `rate=1.0` 与不传 `--vq-rate` 的 smoke `train_loss/delta_norm/delta_norm_clipped_mean` 逐位相同。
- `--vq-rescale on`：被 clip 的 client（raw 2.79>1.0）VQ 后范数精确回到 1.0；未 clip 的 client 回到 0.7036。
- BP guard：`--trainer bp --vq-rate 0.5` 报 `--vq-rate < 1 is ZOO-only`；`--vq-rate 1.5` 报 `must be in (0, 1]`。

## 五、关键发现 / caveat

1. **Count Sketch 在 `rate≤1` 时必然有损**：即使 `N=d`（rate=1），cos 也只有 ~0.7；因此 CLI 在 `rate=1` 直接绕过 VQ，而不是把 identity 当“无压缩”。`_selftest` 的 100% 预算阈值因此设为 0.6（不是 0.9）。
2. **`third_party/VQ` 原 hash 有退化风险**：`h=(i*a)%c_dim` 未约束 `gcd(a,c_dim)=1`，小 `c_dim` 时可能只用部分桶。本实现强制 `gcd=1`；这是有意的质量修正，不是逐位复刻。
3. **字节口径**：`--vq-rate` 是元素比；fp32 sketch 4B vs 当前 dense bf16 2B，所以 `vq_ratio_bytes = 2 * vq_ratio_elements`。若论文要“字节压缩”，需要 bf16/fp16 sketch（尚未实现），或明确用元素比。
4. **post-hoc 模拟**：当前在 client 本地训练结束后对最终 delta 做 VQ；由于 Count Sketch 线性且 clip 在压缩前，等价于把各前向样本的 sketch 在 server 上求和（不建模逐前向延迟）。真正的流式上传可作为后续扩展。
5. **C 按名义 `local_steps*zo_directions`**：若 smoke 的 `--max-train-samples` 导致实际步数少于 `local_steps`，仍按名义 C 分配（部分 channel 为 0）；正式跑 100×16k 不会触发。
6. `vq_cos_mean` 是 rescale 前的方向 cos；rescale 只保幅度，不改变方向误差。若看到 `cos≈√(rate/C)`，说明 hash 被错误地跨 channel 复用。

## 六、复现

```bash
PY=~/miniconda3/envs/SMELL/bin/python
$PY src/compress/count_sketch.py
$PY -m py_compile src/fed/run_fed.py src/compress/count_sketch.py
# BERT smoke（完整命令见 AGENTS.md「VQ」段）
$PY src/fed/run_fed.py --arch distilbert --tag a01 --gpu 0 --trainer zoo \
  --attn sdpa_prune --dtype fp32 --sparse 0.4 --doc-order cycle \
  --pos-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/pos_embed.pt \
  --head-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/head.pt \
  --predictor checkpoints/predictor_bert/a01_warmup_400step/predictor.pth \
  --pruned-config checkpoints/predictor_bert/a01_warmup_400step/pruned_config.pth \
  --zo-subspace layers --zo-layers 0-5 --zo-directions 2 --local-steps 2 \
  --rounds 1 --lr 1.4e-3 --delta-clip 1.0 --max-clients 2 --max-train-samples 2 \
  --truncate 2048 --eval-every 0 --save-every 0 --vq-rate 0.5 \
  --out-root temp/smoke_vq_bert
```

## 七、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 10:10 | 新建分支 | `exp/vq-compression` | 从 `37e5708` 拉出；本地 `main` 已 fast-forward 到同提交 |
| 10:15 | 新增 | `src/compress/count_sketch.py` | Count Sketch + selftest |
| 10:25 | 修改 | `src/fed/run_fed.py` | VQ CLI/接线/metrics/seeds |
| 10:30 | 更新 | `AGENTS.md` | 分支/当前结果/运行口径/VQ 段 |
| 10:30 | 新增 | 本 ailog | 设计 + smoke + caveat |

## 八、后续

- [ ] 在 BERT r30 / OPT r30 上跑 `--vq-rate {0.5,0.25,0.1}`（先 1 轮 smoke，再决定长跑队列）。
- [ ] 若要字节级压缩：加 bf16/fp16 sketch（accumulate fp32，传输前 cast），并重测 cos。
- [ ] 若要让 server 直接聚合 sketch（而非 per-client decompress），可加“聚合后解压”路径；当前 per-client decompress 便于 `vq_cos` 与 rescale 诊断。
