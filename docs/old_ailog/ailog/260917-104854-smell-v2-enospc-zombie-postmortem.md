# SMELL-v2 — 两条 BP 长跑同时僵死 15.5 小时的 postmortem（根因：磁盘写满 ENOSPC）

## 概述

2026-09-16 18:24–18:25，两条正在跑的 BP 臂（`lr0.3 constant` 至 round 57、`r=2 lr0.3 linear` 至 round 39）**同时**停止工作，GPU 使用率归零但进程不退出，一直僵到 09-17 10:12 被人工清理，共 15.5 小时，两个进程各自抱着 ~19.8 GB 显存。

**根因（已用日志 traceback 直接证实，非推断）**：`/home` 被写满（`errno = 28, 'No space left on device'`），服务端写 consensus history 的 `h5py flush()` 抛 `RuntimeError`，该异常在 FedML 的消息处理链上一路无人接管，**打死了服务端主线程**；随后进程被"非 daemon 收信线程"永久卡在解释器退出阶段，而客户端在自己的轮询循环里永远等一条不会来的消息。

**结论定性**：不是训练逻辑 bug，不是 OOM，不是估计器/调度问题 —— 是**基础设施故障 + 通信层缺少看门狗**。两条臂此前的实验结论不受影响。

**修复未实施**（按用户决定，本文档只记录证据链与修法）：见 §6。

---

## 1. 时间线与证据

| 时刻 | 事件 | 证据 |
|---|---|---|
| 09-16 16:15 | `lr0.3 constant` 臂启动（GPU3） | `log/smell-v2-BP-spu10_lr0.3_constant_20260916_161509.log` |
| 09-16 16:55 | `r=2 lr0.3 linear` 臂启动（GPU1） | `log/smell-v2-BP-spu10_lr0.3_linear_r2_20260916_165517.log` |
| 09-16 18:24:30 | constant 臂服务端 `flush()` 失败，主线程带异常退出（round 57） | `...constant....log` 中的 `errno = 28` traceback + `consensus_history_20260916_161514.h5` mtime |
| 09-16 18:25:51 | r=2 臂同样失败（round 39） | 同上，`...linear_r2...log` + `consensus_history_20260916_165522.h5` mtime |
| → 09-17 10:12 | 两对进程（mpirun + 2×python 各）僵持，GPU 0%、显存不还 | GPU1/GPU3 各 19.8 GB 常驻、`nvidia-smi` util=0 |
| 09-17 10:12 | 人工 kill（python → mpirun → bash），GPU 释放 | 见 §7 |

两条臂相隔 **80 秒**先后死亡，因为它们写的是**同一个文件系统**。

## 2. 证据链

### 2.1 日志里就写着根因（`log/smell-v2-BP-spu10_lr0.3_linear_r2_20260916_165517.log:4966,4999`）

```
RuntimeError: Unable to flush file (file write failed: time = Wed Sep 16 18:25:51 2026,
  filename = '.../checkpoints/bp-spu-sweep-spu10-lr0.3-linear_r2/consensus_history_20260916_165522.h5',
  file descriptor = 63, errno = 28, error message = 'No space left on device',
  total write size = 2920, bytes actually written = 18446744073709551615, offset = 0)

Traceback (most recent call last):
  ...
  File "FedSgdServerManager.py", line 34, in run           → super().run()
  File "server_manager.py", line 41, in run                → self.com_manager.handle_receive_message()
  File "com_manager.py", line 76, in handle_receive_message → self.notify(msg_params)
  File "com_manager.py", line 93, in notify                 → observer.receive_message(...)
  File "server_manager.py", line 51, in receive_message     → handler_callback_func(msg_params)
  File "FedSgdServerManager.py", line 222, in get_vote      → self.aggregator.collect_vote(...)
  File "FedSgdAggregator.py", line 413, in collect_vote     → self._save_consensus_vote_history(...)
  File "FedSgdAggregator.py", line 392, in _save_consensus_vote_history → self.consensus_history_file.flush()
RuntimeError: Unable to flush file (... errno = 28 ...)
```

### 2.2 h5 侧证：最后一次写入被截断

进程被杀（锁释放）后读文件，两个 history h5 的**最后一个 round 组都是坏的**：

```
rounds 数= 40 最后 3 轮: ['round_0037', 'round_0038', 'round_0039']   ← r=2 臂
KeyError: 'Unable to open object (len not positive after adjustment for EOA)'
```

即 HDF5 头已分配、数据没写完 —— 与 `flush()` 抛 ENOSPC 一致；mtime 精确停在 18:25:51 / 18:24:30。

### 2.3 进程状态：服务端已"死"但退不出，客户端在等（`py-spy dump`）

```
rank0 (server, pid 4073140)
  MainThread: _shutdown (threading.py:1567)          ← 主循环已带异常结束，卡在解释器退出
  ServerSendThread / ServerReceiveThread: active     ← 非 daemon，永远 join 不掉
rank1 (worker, pid 4073141)
  MainThread: handle_receive_message (com_manager.py:78)   ← 每 0.3s 轮询收件队列，无超时
```

### 2.4 排除项与**被推翻的中间假设**（保留以免下次重犯）

| 假设 | 结论 | 依据 |
|---|---|---|
| GPU 0% 是因为进程空转 | **错** | 每线程 CPU 时间显示烧 CPU 的是 OpenMPI 的 async progress 线程，**从启动起就在烧**（≈ 整个寿命），与僵死无关 |
| 是 OOM | 排除 | `free -g` 显示 88 GB available，日志无 OOM marker |
| 是 /dev/shm 满 | 排除 | 70 M / 126 G |
| 日志尾部缺失是 mpirun/C stdio 缓冲未 flush | **错（曾据此写过长篇推理）** | 真因就是 ENOSPC：日志文件本身写不进去，文件停在 4096 边界；空间恢复后补写（634880 → 641014 字节）。`python -u` 对本次无帮助 |
| `tee` 进程消失 | **错** | 该脚本用 `exec >> "$LOG_FILE" 2>&1`，**根本没有 tee**（我误读了脚本） |
| 服务端走到了末轮正常收尾 | 排除 | 末轮分支要求 `round_idx==100`，而存档只到 r39/r57；且无 `global_lora_final_*.h5`、无 `Waiting 120s`、无 `__finish server` |

### 2.5 顺带查清的两件通信层事实

- **日志管道**：rank 的 **stderr**（`logging`，Python 3.10 行缓冲，实测逐行落盘）→ 私有 pipe → **mpirun**（block-buffered stdio）→ 日志文件；rank 的 **stdout** 被挂到 `/dev/pts/5x`（master 由 mpirun 持有）⇒ 代码里的 `print('done running')` / `print('indexes of clients: ...')` **在任何日志里都不出现**（实测 114 个日志文件中 `done running` 命中 0 次），别再用它们判断服务端是否收尾。
- **潜在雷**：`MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4` 在 `message_define.py:15` 有定义，但服务端只注册了 3/6/8/9；`server_manager.receive_message` 是 `self.message_handler_dict[msg_type]` 直接索引 ⇒ **哪天客户端一发 type 4 就是 KeyError 打死主线程**（当前客户端只发 3/6/8/9，未触发）。

## 3. 根因链（四步）

1. **`/home` 写满**（`df`：3.1T/3.5T，95%，共享盘，非本项目占用；我们自己的全部产出只有 checkpoints 3.3 G + log 202 M）。共享盘上任何一个用户把盘写满，都会命中我们。
2. **纯诊断用的写操作把训练打死**：`consensus_history*.h5` 只是 Phase 3 的记录文件，其 `flush()` 抛的 `RuntimeError` 在 `get_vote → notify → handle_receive_message → server_manager.run()` 链上**没有任何 try/except**，直接终止主线程。
3. **退不掉**：`mpi_receive_thread.py:19-28` 是 `while True: self.comm.recv()` 的**非 daemon** 线程，卡在 C 层阻塞 recv；`threading._shutdown()` 要 join 它 ⇒ 永远退不出。`stop_receive_message()` 的 `raise_exception()+join()` 也无效（异步异常投不进 C 调用）。
4. **客户端永远等**：`com_manager.py:71-78` 是 `while self.is_running: 轮询 q_receiver; sleep(0.3)`，**没有任何超时**；服务端死了它不知道，永远等 round-39/round-58 的 model sync。

⇒ 结果：GPU 空转、显存不还、进程不退。**任何服务端异常都会变成这个形态**，磁盘满只是这次的扳机。

## 4. 为什么 15.5 小时没人发现

- 没有看门狗，也没有"服务端死了"的检测（客户端静默等待）。
- 日志尾部因 ENOSPC 丢失，`[EVAL-PPL]` 停更看起来像"跑得很慢"。
- 没有磁盘水位监控/预检。

## 5. 本次诊断用的工具与配方（下次挂了照这个顺序查）

```bash
# 1) 是不是真在算：GPU util + 显存归属
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
# 2) 进程状态：谁在睡、等在哪个内核点
for t in /proc/<pid>/task/*; do awk '{print $3, $35}' $t/stat; cat $t/wchan; done
# 3) 决定性一步：Python 级栈（本环境已装）
py-spy dump --pid <pid>          # 或 --nonblocking
# 4) 每线程 CPU 时间，区分"空转"与"OpenMPI async progress 常态"
awk '{print ($14+$15)/100}' /proc/<pid>/task/<tid>/stat
# 5) 日志/存档时间戳定位停止时刻；h5 尾部完整性
ls -lt --time-style=+%F_%H:%M:%S checkpoints/<dir>/ | head
# 6) 磁盘
df -h /home
```

附：临时工具 `/tmp/memscan.py`（用 `process_vm_readv` 扫进程内存找未落盘日志）本次**未被需要**（traceback 就在日志里），且对 mpirun 无效（EPERM），未入库。

## 6. 修法清单（**1 已实施**，其余待批）

| 优先级 | 改动 | 位置 | 效果 |
|---|---|---|---|
| 1 ✅ **已实施** 09-17 | history 写入包 try/except：记 `ERROR` + **完整 traceback**（`exc_info=True`）⇒ 关闭并停用 history ⇒ 训练继续；新增 `_disable_consensus_history()`（幂等） | `FedSgdAggregator._save_consensus_vote_history` / `close_consensus_history`（后者补 traceback） | 磁盘满只损失诊断数据，不再杀训练；磁盘正常时行为逐字不变 |
| 2 | 服务端主循环包异常：记录 traceback 后 `MPI.COMM_WORLD.Abort()` | `server_manager.run()` 或 `FedSgdServerManager.run()` | 异常立刻 fail-fast，客户端同时死，不再留僵尸 |
| 3 | 客户端轮询加超时（N 秒无消息 ⇒ 记日志退出） | `com_manager.handle_receive_message` | 服务端死了能自己退出 |
| 4 | `MPIReceiveThread` 设 `daemon=True` | `mpi_receive_thread.py` | 保证任何路径下进程都能退出 |
| 5 | 脚本启动前 `df` 预检（<X GB 拒绝启动）+ `python -u` | `script/RUNME-v2-BP-sweep.sh` | 提前失败而不是跑两小时后炸 |
| 6 | 注册 type 4（STATS）handler 或从 `message_define` 删掉 | `FedSgdServerManager.register_message_receive_handlers` | 消除 §2.5 的潜在 KeyError 雷 |

## 7. 本次运维动作

- **杀进程**：`kill -9` 掉 4 个 python rank → 2 个 mpirun → 2 个 bash 脚本（3847400 段与 4073100 段），GPU1/GPU3 释放（残余 2.4 GB 属其他用户进程）。
- **归档清理**（用户确认，删除不可恢复）：删掉 ZOO + **全 LoRA**（8.39M/1.05M 参数、256 张量）的六个项 —— `checkpoints/a0.3_sample20_len1k_datagoe`(889M)、`a0.1_sample20_len4k_datagoe`(860M)、`a0.1_sample10_len4k_datagoe_eval`(850M)、`a0.1_sample10_len4k_datagoe`(459M)、`checkpoint_20260910_193611_r{1,2,3}.h5`(3×33.8M)、`checkpoint_20260912_{100314,110404}_r*.h5`(3×4.4M)。
  结果：`checkpoints/` **3.3 G → 209 M**，/home 可用 177 G → 181 G。
  保留：`a0.1_sample1_len4k_datagoe`（260912 task-head 计划引用的 d=1.05M 臂）、全部 D3 lm_head 档（63+16 个）、13 个 `bp-*` 目录、10 个 `consensus_history_*.h5`、`FwdLLM/checkpoints/{llama2(16G 基座权重),predictor(Phase2/3 必需),phase3}`、全部 `log/`（202 M，结论原始证据）。
- - **修法 1 落地**（09-17，本 session）：`FedSgdAggregator._save_consensus_vote_history` 全函数体包 `try/except`，失败时 `logging.error(..., exc_info=True)` 落完整 traceback 并调用新增的 `_disable_consensus_history()`（关闭+置 `None`，使后续写入走 early-return、`close_consensus_history()` 变 no-op）；`close_consensus_history` 的 except 也补了 `exc_info=True`。`py_compile` 通过；靶向测试 5 项全过 —— ①ENOSPC 不冒泡 ②日志含 ERROR+完整 traceback（含 errno 28 与出错行）③`close()` 亦失败时不冒泡 ④正常路径数据集/attrs 与改前一致 ⑤停用后 no-op。
  已知取舍：`except Exception` 是宽口径，连"配置类异常"（如 `_get_sparse_ratio` 找不到 `args.sparse`）也会转为"停用 history"而不再中止训练——但 traceback 完整落盘，可事后发现（本次测试夹具就误触过一次，日志里一目了然）。
- ⚠️ **清理不等于解决**：我们只占 3.5 T 盘的 3.3 G（0.1%），ENOSPC 会复发；真正的保险是 §6 的 1（已做）+ 2。

### 7.1 共享盘事实（回答"3.4T 是不是全服务器共享"）

| 挂载 | 设备 | 容量/可用 | 结论 |
|---|---|---|---|
| **`/home`** | `/dev/nvme0n1p1` ext4 | 3.5 T / **193 G**（95%） | **全服务器共享**：同一分区挂着所有用户家目录（`/home/aeonia`、`sys21021`、`tangchuang`… 20+ 个）。可读到的占用：aeonia 239 G、tangchuang 168 G、sunny 152 G、lzj 66 G（其余目录无读权限）。ext4 默认 root 保留 5%（~175 G）**对非 root 不可见** ⇒ 我们实际能用的就是那 193 G |
| `/data` | `/dev/sda1` ext4 | 15 T / **5.5 T**（61%） | 另有大盘，但 `/data` 本身 root:root 755、按用户建子目录（`/data/aeonia` 等），**没有 `/data/yangyongbo`，我们写不进去** ⇒ 想把写量大的产物（ZOO 全 LoRA 每轮 ~30 MB）迁过去需管理员开目录 |
| `/` | `/dev/sdb3` ext4 | 1.8 T / 1.3 T（25%） | 系统盘；`/tmp` 就在它上面，容量够但会被定期清理 ⇒ 只适合临时中转，不适合归档 |

⇒ 结论：**离争议盘只有"搬到 /data"一条路，且需要管理员**；在拿到之前，靠 §6 的容错（已做 1、待批 2）把"全盘写满"降级为"丢诊断数据"而不是"丢 15 小时"。

## 8. 可抢救的实验结果（两条臂死于基础设施，结论不受影响）

| 臂 | 完成 | round 38 test PPL | round 39 | 备注 |
|---|---|---|---|---|
| r=1 lr=0.3 constant（GPU3） | 57/100 | 41.56 | 41.21 | 存档到 `_r57.h5` |
| r=1 lr=0.3 linear（历史） | 100/100 | 40.99 | 39.42 | 最终 37.03 |
| r=1 lr=0.45 linear（历史） | 100/100 | 45.00 | 41.06 | 最终 36.96 |
| **r=2 lr=0.3 linear（GPU1）** | **39/100** | **38.39**（d_ppl −6.64） | — | 存档到 `_r39.h5`；等轮次明显快于 r=1 |

⇒ r=2 在等轮次上更快（38.39 vs 40.99 / 41.56），秩/容量杠杆有效，**值得在磁盘有保障后重跑满 100 轮**（另见 `docs/ailog/260916-165417-...md` §6.1 的 lr 平顶结论）。

## 变更表

| 时间 | 操作 | 对象 | 说明 |
|---|---|---|---|
| 10:12 | REMOVED | 4×python / 2×mpirun / 2×bash | 清理僵死进程，释放 GPU1/GPU3 |
| 10:40 | REMOVED | `checkpoints/` 六个 ZOO 全 LoRA 项（3.2 G） | 用户确认；数值已在 docs 归档，无脚本/文档引用被破坏 |
| 10:48 | NEW | `docs/ailog/260917-104854-smell-v2-enospc-zombie-postmortem.md` | 本文档 |
| 10:5x | MODIFIED | `FwdLLM/FedML/fedml_api/distributed/fedsgd/FedSgdAggregator.py` | 修法 1：`_save_consensus_vote_history` 包 try/except + `exc_info=True`；新增 `_disable_consensus_history()`；`close_consensus_history` 的 except 补 traceback（均带 `# SMELL 5 consensus-h5-resilience` 注解） |
| 10:5x | 其他 | `script/RUNME-v2-BP-sweep.sh` 的 `LR_R` 旋钮（09-16） | 记在 `260916-165417` 文档，本文不重复 |

## 待办

1. ~~实施 §6 的 1~~（**已完成 09-17**）；§6 的 2（服务端 fail-fast）待批，3/4/5/6 待议。
2. 向管理员申请 `/data/yangyongbo`（见 §7.1），把写量大的产物迁出共享的 `/home`。
3. 重跑 `r=2 lr0.3 linear` 满 100 轮（对照 r=1 的 37.03），以及 `lr0.3 constant` 的剩余 43 轮价值评估（constant 是否值得重跑需另判）。
4. 关注 `/home` 水位；至少加一条启动预检，避免再次"跑两小时后炸"。
5. 若还要用 consensus history 做指标（B2：`_sum_q_acc` 跨轮不清零的问题仍在），注意其 h5 已经不可靠——本次两条都留下损坏的尾对象。
