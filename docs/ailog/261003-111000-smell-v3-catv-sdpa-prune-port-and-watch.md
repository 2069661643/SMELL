# Ailog 261003-111000 — Phase N: CATV→`sdpa_prune` 移植 + D32+CATV 监听链（脱离 TUI 自动接续）

> 承接审计 `260930-122106` §4.2（R2：CATV 只在 flash 接线；R3：评测无掩码）。
> 本 session 在等 D32（k24/D32/lr0.30，预计 16:30 完）的同时：① 完成 CATV 在主力协议（fp32+`sdpa_prune`）下的接线；
> ② 挂一个 `setsid` 监听链，使 TUI 关闭后也能在 D32 结束的瞬间自动接上 **D32+CATV** 的 smoke → δ 探测 → 30 轮正式跑。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 11:00 | 移植 vote_callback + consensus_mask | `src/models/modeling_opt_smell.py` | `OptSdpaPruneAttention.forward`（`_prune_block_scores` 之后、topk 之前） | sdpa_prune 下 CATV 生效，与 flash 完全同序（原始分上报 / ±inf 注入） |
| 11:00 | 放宽 CATV guard | `src/fed/run_fed.py` | `catv_sdpa_guard`（L372-375） | 允许 `flash|sdpa_prune`；`sdpa`/`eager`/`sdpa_gather` 仍 fail-fast |
| 11:00 | eval 转发掩码 | `src/fed/run_fed.py` | `run_global_eval` | `--catv-mask consensus_mask_roundN.pt`（本轮最新掩码，最后一轮口径） |
| 11:00 | 新增 `--catv-mask` | `src/eval/ppl.py` | `parse_args` + `build_model` | 载入 payload → `config.consensus_mask`；与训练同语义 |
| 11:00 | 同步云端 guard | `scripts/run_ablation_cloud.sh` | `catv_prune_guard` | 放行 `sdpa_prune` 组合 |

接线口径核对（无 GPU 验证）：
- `_VoteAccumulator` 自带层过滤 `[11, 22]`，sdpa_prune 对所有非末层上报不会污染票集；
- `compute_consensus_mask` 输出 `{layer: float32[N]}`，可与 `(bsz, N)` 的 `scores` 广播相加；
- flash 与 prune 的 topk 前流程逐行同构。

## 监听链（`temp/run_d32_catv_watch_bg.sh`，11:03 启动，pid 298880）

| 阶段 | 动作 | 判据 |
|---|---|---|
| 1 等待 | 轮询 `last_k24_d32_r30.pid` 退出（60s 间隔） | 双方 setsid，脱 TUI 存活 |
| 2 校验 | D32 `driver.log` 含 `k24_d32_r30 ALL DONE` | 否则 ABORT rc=2 |
| 3 smoke | 2k、1 client、2 轮、`--catv on --catv-r 0.2`、`--attn sdpa_prune` | 断言 `consensus_mask_round0.pt` 生成、`ir_mean` 非空、`anchor_in≥1`；失败 ABORT rc=3 |
| 4 δ 探测 | 30-client 1 轮（catv on）→ `clip=3×median`（clamp[2,20]） | 输出 `[stats]` |
| 5 正式跑 | D32+CATV 30 轮、`--eval-every 5 --save-every 1` | 产物 `logs/fed/zoo_k24_d32_catv_lr0.30_r30/` |

状态入口：`temp/logs/last_d32_catv_watch_dir.txt` → `watcher.log`。

## 待办

- [ ] smoke 结果（D32 结束后自动执行）；通过后正式跑 ETA ~25h。
- [ ] D32+CATV vs D32（无 CATV）同轮数对照：IR/anchor 统计 + full PPL 斜率。
- [ ] 评测掩码口径（最后一轮）在报告中注明；如需与训练严格同步可后续改逐轮掩码。
