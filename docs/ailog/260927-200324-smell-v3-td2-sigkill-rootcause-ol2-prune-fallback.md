# Ailog 260927-200324 — Phase N: TD-2' SIGKILL 根因定位（外部 kill + 末层 O(L²) 回退）

> 本 session **仅调查与记录，未改任何代码**。分支 `exp/rank-rotation-lora`；云端 A40 host `amax`。
> 触发问题：`temp/diag_cos_grid.py --attn sdpa_prune --dtype fp32 --seq 8192` 被 `Killed`（rc=137），需判断是否 OOM。

## 一句话结论

- **rc=137 的直接死因是外部 `SIGKILL`，不是 OOM**：内核与 cgroup 的 OOM 计数全为 0，脚本无 kill 逻辑；两个违规跑在 **GPU1** 的任务在 19:36 同时静默消失，随后 19:37:57 改写为 GPU2 队列 ⇒ 即上一个 Agent 为遵守「仅 GPU2」手动 kill。
- **但存在一个真 bug，会让该测试无论如何 OOM**：`OptSdpaPruneAttention` **豁免末层** → 末层回退进 `OptSdpaAttention._build_sparse_attn_mask`，构造 dense `(1,L,L)` fp32 加性掩码，torch 2.1.2 fp32 无 mem-efficient 加性掩码核 → math 核实体化每头 `(h,L,L)`，L=8192 ≈ 4 GiB、L=16384 ≈ 16 GiB。**未修**（本 session 决定只记录）。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 19:40 | 无（只读调查） | `temp/logs/diag_cos_td2_260927-193452/{driver.log,diag_seq8192.log}` | — | 确认 rc=137 无 Python traceback |
| 19:50 | 无（只读调查） | `/proc/vmstat`、`/sys/fs/cgroup/user.slice/user-1034.slice/memory.events` | — | `oom_kill=0`、无 `memory.max` 限制 |
| 19:50 | 无（只读调查） | `src/models/modeling_opt_smell.py` | L456-521, L505, L567, L738-739, L793-802 | 定位末层回退触发的 O(L²) 掩码 |
| 19:50 | 无（只读调查） | `temp/logs/smoke_td1p_260927-192544/driver.log` | — | 铁证：`Tried to allocate 16.00 GiB` @ `modeling_opt_smell.py:567` |
| 19:52 | 无（只读调查） | `temp/run_diag_cos_td2_bg.sh`、`temp/run_after_kscan_k1_td2_bg.sh` | — | 前者 `--gpu 1`、后者 `--gpu 2`；时间线闭环 |
| 20:03 | 新增本 ailog | `docs/ailog/260927-200324-smell-v3-td2-sigkill-rootcause-ol2-prune-fallback.md` | — | 记录根因，供后续修复 |

## 一、直接死因：外部 SIGKILL（非 OOM）

时间线（`temp/logs/diag_cos_td2_260927-193452/driver.log`）：

```
[19:34:52] STEP diag_cos_td2 start          （run_diag_cos_td2_bg.sh，--gpu 1）
[19:34:52] STEP try seq=8192
            424366 Killed   python ... --gpu 1 ...       ← SIGKILL，无 traceback
[19:36:18] STEP FAIL(non-OOM) seq=8192 rc=137 -> stop
```

排除项：

| 假说 | 证据 | 结论 |
|---|---|---|
| 内核 host OOM | `grep -iE oom /proc/vmstat` → `oom_kill 0`（开机 45 天未触发） | **排除** |
| cgroup OOM | 用户全树 `memory.events.oom_kill=0`；各层 `memory.max=max` | **排除** |
| systemd-oomd | 本用户 `memory.pressure total≈1ms`、`swap.current≈1MB`；且若触发会杀整个 session scope（但 wrapper/tee 存活） | 不像 |
| 进程内自毁 | `run_*.sh` grep 无 `kill/pkill/timeout/ulimit/setrlimit`；`src/` 无 | **排除** |
| 真实 CUDA OOM | 对照 `smoke_td1p` 会打出完整 `torch.cuda.OutOfMemoryError`；本例无 | **排除** |

判定：**同 UID/管理员显式 `kill -9`**。两个跑在 **GPU1** 的任务（TD-2' pid 424366 + `bp_k1_lr1e-3`，19:32 启动）于 19:36 前后**一起静默死**；`run_after_kscan_k1_td2_bg.sh`（`--gpu 2`）写于 **19:37:57**、19:38:08 启动。结合本仓硬性约定「仅 GPU2 允许」，即上一个 Agent 中止了违规的 GPU1 运行并改到 GPU2。

> 注：内核日志 `/var/log/{kern,syslog}` 与 `dmesg` 对本用户不可读（`kernel.dmesg_restrict=1`、无 sudo）。但 `/proc/vmstat:oom_kill=0` 已足以反证内核 OOM 路径未运行。

## 二、真 bug：稀疏剪枝「末层回退」重新触发 O(L²)

调用链（`src/models/modeling_opt_smell.py`）：

```
OptSdpaPruneAttention._prune_ok()          L738-739  末层 (layer_idx == num_layers-1) 被豁免 → False
  └─ forward() 回退                          L793-802  return super().forward(...)  (= OptSdpaAttention.forward)
       └─ _build_sparse_attn_mask()          L456-521  分配 dense (bsz,1,L,L) fp32 掩码 @ L505
            └─ F.scaled_dot_product_attention L567     加性掩码 + fp32 → 无 mem-efficient 核 → math 核 O(L²)
```

- 除末层外，`OptSdpaPruneAttention` 本体是 **O(L)**：top-k 块 `L815-818`、token 子集化 `L822-827`、`is_causal=True` SDPA（无掩码）`L835-842`。**O(L²) 每 forward 仅在末层发生一次。**
- math 核实体化每头 `(h=16, L, L)` fp32：**L=8192 → 16·8192²·4 B ≈ 4 GiB**；**L=16384 → 16 GiB**。
- 铁证：`temp/logs/smoke_td1p_260927-192544/driver.log`
  `torch.cuda.OutOfMemoryError: Tried to allocate 16.00 GiB ... modeling_opt_smell.py:567`（经 `:795` 回退）。
- **历史澄清**：TC 的 16k fp32 成功（`temp/logs/diag_cos_fp32.log`）发生在 `_build_sparse_attn_mask` **尚未加入**之前——该函数由 `2e03bc7`（09-27 13:06）引入，而 `OptSdpaAttention` 由 `2cecfe9`（09-26 19:03）引入。故 TC 当时走的是纯 dense causal（`is_causal=True`，O(L)）。
- **连带影响**：只要 `config.thresh∈(0,1)`（`get_opt_qk` 必设 `config.thresh=config.sparse`），`--attn sdpa` 本身在任何层都会建这个 dense 掩码 ⇒ 当前 16k fp32 `sdpa` 也会 O(L²)。

## 三、对「GPU3 全量 ZOO」评估的修正

- GPU3 现状：pid 236794（他人 `VLLM::EngineCore`）占 39142 MiB，**仅剩 ~6.7 GB**；`util 0%` 只是空闲，显存已被占。按约定不共享。
- forward-only 全 24 层 fp32 ZOO（含分块 lm_head loss，`modeling_opt_smell.py:1557-1562`）估算 **~2.5–4 GB**，6.7 GB 理论可容纳；
- **但前提是先修掉上面的末层 O(L²) 回退**，否则末层 16 GiB 直接爆。
- `run_fed.py` 仍**硬编码 bf16 + FA2**（`run_fed.py:411-422`），无 `--dtype/--attn`，全量 fp32 ZOO 目前无法复现（见待办）。

## 未实施的修复清单（仅记录，未改）

1. `OptSdpaPruneAttention` 末层/回退路径改为**纯 dense causal SDPA**（`is_causal=True, attn_mask=None`），不进 `_build_sparse_attn_mask`（`modeling_opt_smell.py:793-802`，最小改动）。
2. `OptSdpaAttention.forward` 的稀疏掩码改为**显式开关**（如 `config.sdpa_sparse`，默认关），令 `--attn sdpa` 回到 O(L) dense causal；稀疏只由 `sdpa_prune` 承担（`modeling_opt_smell.py:564-574`）。
3. `run_fed.py` 新增 `--dtype {bf16,fp32}` 与 `--attn {flash,sdpa,sdpa_prune,sdpa_gather}` 接线，使 fp32 SDPA 全量 ZOO 可复现。

## 待办

- [ ] 与用户确认：是否接受**不修**时排队的 TD-2' 只能在 seq≈2048 出稀疏 cos（8192/4096 将 CUDA OOM）；或改做上述**最小修复 1**。
- [ ] k-scan 完成后读拐点：`logs/fed/bp_kscan/k{4,12}/a01/metrics.jsonl`（k=4 已 18/60，train_loss 3.6136→3.5123，cos 0.554，收敛趋势明确）。
- [ ] k=1@lr1e-3 是否仍平（判「平」是覆盖率还是 lr）→ `logs/fed/bp_k1_lr1e-3`。
- [ ] 修掉 O(L²) 后于 GPU2 重跑 TD-2'，测实际可跑 seq 与稀疏 cos。

## 环境 / 约定

- conda：`~/applications/anaconda3/envs/smell-v2`（torch 2.1.2+cu118 / FA 2.4.2）。
- jenga 源：`/home/yangyongbo/projects/smell/JengaForMemoryTest/Jenga/src`（editable，非本仓子模块）。
- 仅 GPU2、长任务后台化、driver 打点 `[HH:MM:SS] STEP`、看 `metrics.jsonl`（driver.log 有块缓冲滞后）。
