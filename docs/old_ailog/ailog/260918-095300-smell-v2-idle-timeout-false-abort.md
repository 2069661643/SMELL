# SMELL-v2 — Phase 6 client 空闲看门狗误杀 M_tot=1500 长程（根因：空闲基准记在"消息出队时刻"）

## 概述

2026-09-18 00:02:10/00:02:12 启动 **AB 正式两臂**（`ARM=A e=15 spu=1` GPU1、`ARM=B e=3 spu=5` GPU3，均 M_tot=1500、N=100/500、rounds=100、consensus=1）。两臂 round 0 的 1500 次迭代都正常跑完（wall 2364.8 s / 2376.2 s，it_s_avg 0.634/0.631，与冒烟一致、**无退化**），随后在 **00:43:28 / 00:43:42 被自己刚加的看门狗判"空闲 2464.6 s / 2475.5 s > 1800 s"并 `MPI.COMM_WORLD.Abort(1)`**。

**根因（日志直接证实）**：Phase 6 新增的 client 侧 idle-timeout 把空闲基准记在**消息出队时刻**，而 `notify()` 在**同一线程里同步跑完整轮**（3 次 eval + 1500 次迭代 ≈ 2497 s）。于是每轮训练结束、控制流回到接收循环时，`now − _last_msg_ts` 恰好等于"上一轮的全部本地计算时间"，必然超过阈值。

**定性**：不是训练/估计器/调度问题，也不是僵尸（进程干净退出、GPU 已释放），是 **Phase 6 看门狗自身的计时语义 bug**。触发条件是纯算术的：**单轮客户端计算时间 > `--client_idle_timeout_s`**。

**修复已实施并提交**（`5ee5186`），长程于 09:46:08/09:46:10 重启。

---

## 1. 时间线

| 时刻 | 事件 | 证据 |
|---|---|---|
| 09-17 21:21 / 21:31 | Phase 6 通路冒烟（M=100、rounds=3）：A 臂 consensus=1、B 臂 consensus=0，均 3 轮正常收尾 | `log/smell-v2-AB-B-c100_20260917_{212121,213128}.log` |
| 09-18 00:02:10 / 00:02:12 | AB 正式两臂启动（rounds=100） | `log/smell-v2-AB-{A,B}-c100_20260918_0002{10,12}.log` 首行配置 |
| 00:02:23 / 00:02:26 | 客户端收到**唯一**一条服务器消息 → baseline eval 开跑 | A 日志 `:171` |
| 00:04:03 | BEFORE eval 结束 → round 0 训练开始 | A 日志 `:177` |
| 00:43:28 / 00:43:42 | round 0 训练结束（1500 iters），客户端发 MODEL(type 3)，随即被看门狗判空闲超时自杀 | A 日志 `:361,363,364,366` |
| 09:43 | 人工巡检发现日志 9 小时未更新、进程早已消失 | 本文档触发点 |
| 09:46:08 / 09:46:10 | 修复后重启两臂 | `log/smell-v2-AB-{A,B}-c100_20260918_0946{08,10}.log` |

## 2. 证据链

### 2.1 日志（A 臂 `log/smell-v2-AB-A-c100_20260918_000210.log`）

```
171: [EVAL-PPL] round=baseline START time=2026-09-18 00:02:23
180: [PROF-ROUND] start: profile_every=50 threads=4
361: [PROF-ROUND] iters=1500 wall_s=2364.8 it_s_avg=0.634 rss_peak_mb=19686 threads=4 load1=5.70 ... consensus=True
363: [SMOKE] Sending MODEL (type 3) to server, round=0
364: ERROR:[comm] client idle for 2464.6s (> 1800s) with no message from server — assuming the server is gone, stopping the receive loop
366: ERROR:[SMELL 6] client idle timeout (1800.0s) — aborting the MPI job
```

B 臂同构（`..._000212.log:364`：`client idle for 2475.5s`）。**364 与 363 相邻**是关键：超时判定紧跟在"发完本轮模型"之后，即发生在控制流回到接收循环的那一瞬间。

### 2.2 算术核对

`2464.6 s = 00:43:28 − 00:02:23`，即**距最后一条收到的消息**（00:02:23 那条启动消息）的墙钟时间，而不是"空转等待"时间。这一条消息之后，客户端在**同一个 observer 回调里**依次做完：baseline eval 33.2 s → baseline-TRAIN eval 33.2 s → BEFORE eval 33.3 s → round 0 的 1500 次迭代 2364.8 s，合计 **2464.5 s**，与日志报的 2464.6 s 逐位吻合。B 臂同理（2476 s）。

### 2.3 修前代码（`FwdLLM/FedML/fedml_core/distributed/communication/mpi/com_manager.py:103-107`）

```python
while self.is_running:
    if self.q_receiver.qsize() > 0:
        msg_params = self.q_receiver.get()
        _last_msg_ts = time.time()      # ← 基准记在出队时刻
        self.notify(msg_params)         # ← 本线程同步跑完整个 round（~2497 s）
```

`notify()` → `observer.receive_message()` → handler 回调 → 整轮训练，**全程占用接收线程**，因此：
- 训练期间看门狗不会检查（所以不是 00:32 触发，而是轮末触发）；
- 轮末检查时 `time.time() − _last_msg_ts` ≡ 本轮本地计算时间。

### 2.4 触发条件（为什么冒烟躲过了）

| 配置 | 单轮客户端计算 | vs 1800 s | 结果 |
|---|---|---|---|
| 冒烟 e=1/spu=1（M=100） | 训练 158 s + 3×33 s ≈ **190 s** | 远小于 | 3 轮全过 ✓ |
| 正式 e=15/spu=1、e=3/spu=5（M_tot=1500） | 训练 2365 s + 3×33 s ≈ **2497 s** | 超出 1.39× | 两臂同在 round 0 末尾被自杀 ✗ |

⇒ 该守护**在 M_tot=1500 的任何配置下都必然误杀**，与 consensus 开关、e/spu 分配、数据分区无关。

## 3. 修法（commit `5ee5186`）

基准改在 `notify()` **返回之后**重置，使计时只覆盖真正空转轮询的等待；对"服务端真死"的检测能力不变（客户端在轮内结束后 1800 s 收不到消息仍会退出）。

```python
            if self.q_receiver.qsize() > 0:
                msg_params = self.q_receiver.get()
                self.notify(msg_params)
                # SMELL 6 idle-timeout FIXED — 基准须在【处理完】这条消息之后重置。...
                _last_msg_ts = time.time()
```

- 只改 1 文件、5 增 1 删；旧注解（`ADD`）全部保留，按约定以 `FIXED` 追加。
- `python -m py_compile` 通过。
- 回滚：`git revert 5ee5186`（或 `git checkout 5ee5186^ -- <该文件>`）。含 bug 的基线提交为 `9246932`。
- 顺带确认 Phase 6 B 项（zombie-proof）本身**有效**：这次 abort 后进程全部消失、GPU 释放，没有出现 260916 那种"进程不退、显存不还"的僵尸形态。

## 4. 本 session 早些时候的 Phase 6 验收（M=100 冒烟，09-17 21:21–21:47）

| 检查项 | 结果 |
|---|---|
| 收尾 | `Received FINISH (type 10)` → `__finish server`，无残留 `smell_main`/`orted`，GPU 释放 ✓ |
| 100 client 路径 | `[SMOKE] samples_per_update=1 ⇒ n=100` ✓ |
| Server→Client 同步 | `[DBG-SETPARAM] matched=2 / applied 2/2`、checksum 一致 ✓ |
| Phase 6 D 埋点 | `[PROF-ROUND]`/`[PROF-ITER]` 正常输出 ✓ |
| 每轮配速 | `it_s_avg` 0.633→0.631，3 轮**平**（rss 19.67 GB、threads=4 恒定）；对照 09-17 12:19 那次（**早于全部 Phase 6 修复**）round 1 曾退化到 0.0439 it/s（22.8 s/iter）✓ |
| consensus 两态 | c=1 与 c=0 两条通路都跑通 ✓ |
| 磁盘 | `/home` 205 GB 可用（预检门槛 5 GB），使用率 94% |

唯一红色是历史遗留的 `bitsandbytes libcusparse.so.11` traceback —— 走 bf16+flash-attn，不用 bnb，无影响。

## 5. 当前长程状态与 ETA

- 配置：`A: e=15 spu=1 → M=1500, N=100` / `B: e=3 spu=5 → M=1500, N=500`，rounds=100、lr 0.3→1e-5 linear、consensus=1、profile_every=50、threads=8；GPU1/GPU3。
- 单轮预算（取自 00:43 那次的 round 0 实测）：训练 2364.8 s + 4×33.2 s eval ≈ **2497 s ≈ 41.6 min**。
- 100 轮 ≈ 69.4 h ⇒ **约 09-21 07:00–07:30 收尾**（含末轮不训练、仅同步+终评+120 s 宽限）。
- 中途航点：脚本头注明"前 20–30 轮即可读"⇒ 20 轮 ≈ 09-18 23:40、30 轮 ≈ 09-19 06:35 可看 A/B 斜率比是否接近 √5≈2.2。
- **前提**：round 1 起维持 0.63 it/s。09-17 12:19 那次（早于 Phase 6 E 项 token cache 向量化）round 1 曾退化 14×，故约 10:35 需回读两臂 round 1 的 `[PROF-ITER] it=50 it_s_avg` 确认（任务 #6）。若退化，则上述 ETA 不成立。

## 6. 排查配方（下次"进程没了/日志停更"照这个查）

```bash
date '+%H:%M:%S'                                  # 1) 先对时：日志 mtime 与当前差多少
ps -eo pid,stat,etime,pcpu,args | grep -E "[s]mell_main"   # 2) 进程是否还在（本次：早已消失）
tail -12 log/<run>.log                            # 3) 日志尾部是否留下 Abort/超时/traceback
grep -nE "idle for|MPI_ABORT|PROF-ROUND|Sending MODEL" log/<run>.log   # 4) 定位死亡时刻与相邻事件
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader  # 5) 显存是否释放
```

## 变更表

| 时间 | 操作 | 对象 | 说明 |
|---|---|---|---|
| 00:02 | NEW | `log/smell-v2-AB-{A,B}-c100_20260918_0002{10,12}.log` | 正式两臂首次启动（后被看门狗误杀） |
| 09:46 | MODIFIED | `FwdLLM/FedML/fedml_core/distributed/communication/mpi/com_manager.py:107` | 修法：`_last_msg_ts` 移到 `notify()` 之后重置（`# SMELL 6 idle-timeout FIXED`） |
| 09:46 | NEW | `log/smell-v2-AB-{A,B}-c100_20260918_0946{08,10}.log` | 修复后重启（rounds=100） |
| 09:50 | COMMIT | `5ee5186` | `SMELL Phase 6: fix client idle-timeout false abort on long rounds`（1 文件，仅该文件入库） |
| 09:53 | NEW | `docs/ailog/260918-095300-smell-v2-idle-timeout-false-abort.md` | 本文档 |

## 待办

1. **约 10:35 回读 round 1 配速**（`[PROF-ITER] it=50 it_s_avg`）；若退化 10× 以上则 ETA 失效，需回到 Phase 6 E 项（token cache/线程）复查。
2. 长程收尾（预计 09-21 上午）后读 A/B 判决量（斜率比 ≈√5≈2.2 ⇒ 样本噪声主导；≈1 ⇒ 估计器噪声主导）。
3. 若后续还要把 `--client_idle_timeout_s` 用于新脚本：记住其语义现为"**空转等待**上限"，与单轮计算时长无关；server 侧恒传 0。
4. 09-17 那两条 AB 正式臂（`..._121900.log`，12:19–20:15）日志仍在，其 round 1 的 14× 退化是否即为 E 项所修，尚未逐条对账。
