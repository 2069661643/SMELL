# Ailog 260924-105926 — SMELL-v3: AGENTS.md 补充「长任务与双平台约定」

## 概述

把本 session 反复使用的长任务操作规范固化进 `AGENTS.md`：后台化（setsid nohup）、每步时间戳打点、`sleep 30` 启动确认、聊天触发式结果检查、禁止内联 shell 变量，以及本机 8GB / 云端 A40×4 的双平台分工（此前用户在对话中要求写入）。纯文档变更，无代码。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 10:59 | 新增 | `AGENTS.md` | 环境段之后，新增 `## 长任务与双平台约定` | 6 条约定：nohup 后台化 / 日志与 pid 打点 / sleep 30 确认 / 聊天触发检查 / 禁止内联 `$VAR` / 本机 smoke vs A40 正式实验 |

## 设计决策

- 以本 session 实际踩坑为准写入：内联 `$VAR` 被 Windows→WSL 包装吞掉导致多次命令失败；16k 在 8GB 上（纯前向）依赖 WSL 系统内存溢出，硬上限会 OOM，故明确本机只跑 ≤4k smoke。
- 约定与现有 ailog 制度互补：长任务 driver 脚本放 `temp/`（gitignore），结果与结论仍按规范记 `docs/ailog/`。

## 待办 / 风险

- 云端 `scripts/setup_server.sh` 尚未编写（服务器就绪后补）。
