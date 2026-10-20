# Ailog 261020-155000 — SMELL-v3：VQ sketch 增加 bf16 传输口径；停 0.25，改跑 0.5 bf16

> 承接 `261020-103040`（VQ 初版，fp32 sketch）。用户指出：`--vq-rate 0.5` 在 fp32 sketch (4B) vs
> dense bf16 (2B) 下 `vq_ratio_bytes=1.0`，并不是真正的字节压缩。本 session 加 `--vq-sketch-dtype`，
> 默认 bf16（内部 fp32 累加、传输前 cast），停掉 0.25 run，改跑 0.5 bf16。

## 一、口径

| 量 | 定义 |
|---|---|
| `r_e`（`--vq-rate`） | `C*c_dim/d_active`，元素压缩率 |
| `r_b`（`vq_ratio_bytes`） | `C*c_dim*bytes_per_element / (d_active*2)`；dense 按 bf16 2B |
| `--vq-sketch-dtype bf16` | 传输 2B/element；内部 bucket 累加仍 fp32，cast 后再 decompress |
| `--vq-sketch-dtype fp32` | 传输 4B/element（初版行为） |

- `--vq-bytes-per-element 0` = auto：bf16→2，fp32→4；显式 >0 覆盖。
- 默认改为 bf16 后：`r_b = r_e`（0.5→0.5/2×，0.25→0.25/4×）。
- `rate=1` 仍直接绕过 VQ（Count Sketch 在 N=d 仍有 ~0.7 cos，不是 identity）。

## 二、bf16 cast 对 fidelity 的影响（真实 adapter）

用 `logs/fed/bert_da_fixed/r30_a03_catv_off/a03/adapter_round019` 的 LoRA 权重（d=36,864，C=64，seed=0）：

| rate | c_dim | fp32 sketch cos | bf16 sketch cos | fp16 sketch cos |
|---|---|---|---|---|
| 0.5 | 288 | 0.57046 | **0.57047** | 0.57046 |
| 0.25 | 144 | 0.44148 | **0.44147** | 0.44148 |
| 0.1 | 57 | 0.29276 | **0.29275** | 0.29276 |

- 差异 <1e-4；Count Sketch 碰撞噪声远大于 bf16 量化噪声。
- dense bf16 cast 的方向 cos ≈ 1.0（几乎无损）。

## 三、0.25 fp32 run：已停，保留部分结果

- 启动 10:36:58，停止 15:47:10（r20 eval 完成后；r21 刚启动）。
- r0→r20：NLL 1.692205→**1.669249**（−1.36%），acc 0.424→0.434。
- 同轮无 VQ a03 off 基线 r20 = **1.627528 / acc 0.448**；VQ 0.25 落后 **+0.0417 NLL**。
- r19 逐样本配对：ΔNLL +0.0380，t=9.87，p=5.8e-23（差距随轮数扩大）。
- 部分归档：`checkpoints/distilbert_vq025_a03_r20_partial/`（adapter_round020 + metrics/config/vq_seeds + eval_round020_persample + SHA256SUMS）。
- 结论：0.25（cos≈0.44）明显拖慢收敛；不是发散/NaN，但效果不佳。

## 四、0.5 bf16 run（当前）

- 启动 15:48:08；`logs/fed/bert_vq050_bf16_a03/a03`；归档目标 `checkpoints/distilbert_vq050_bf16_a03_r30/`。
- 配置：a03、CATV off、L8/D8、lr=1.4e-3、clip=1.0、cycle、r30、eval 500。
- VQ：`rate=0.5`、C=64、c_dim=288、`actual=0.5`、`sketch=bf16`、`bytes/elem=2`。
- 预期：`vq_ratio_elements=0.5`、`vq_ratio_bytes=0.5`（真正 2× 字节压缩）、cos≈0.57。
- 首轮 smoke（2 clients/2 samples）已验证 bf16 路径：cos 0.6302，bytes 589,824 vs dense 1,179,648（0.5）。

## 五、CLI

```text
--vq-sketch-dtype {bf16,fp32}   # 默认 bf16；内部 fp32 累加，传输前 cast
--vq-bytes-per-element 0        # 0=auto（bf16=2, fp32=4）
```

`metrics.jsonl` 新增 `vq_sketch_dtype/vq_bytes_per_element`；`config.json`/`vq_seeds.json` 同步记录。

## 六、变更列表

| 时间 | 操作 | 文件 | 影响 |
|---|---|---|---|
| 15:45 | 停止 | `bert_vq025_a03` run | r20 完成后停；partial 归档 |
| 15:47 | 新增 | `--vq-sketch-dtype` / auto bytes | run_fed CLI + client cast + metrics |
| 15:48 | 启动 | `bert_vq050_bf16_a03` | r30、CATV off、rate 0.5 bf16 |
| 15:49 | 修改 | `AGENTS.md` | VQ 通信口径/默认 bf16/0.25 结果 |
| 15:49 | 新增 | 本 ailog | bf16 口径 + 0.25 partial + 0.5 启动 |

## 七、待办

- [ ] 0.5 bf16 r30 完成后，与 a03 off 基线（r29 1.5887 / acc 0.466）做同轮配对；重点看 r29 和逐轮斜率。
- [ ] 若 0.5 仍明显落后，下一步考虑 0.75 / 只压缩部分层 / error feedback；0.1 不建议直接长跑。
