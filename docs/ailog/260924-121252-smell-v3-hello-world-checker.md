# Ailog 260924-121252 — SMELL-v3: Jenga 风格环境自检 hello_world

## 概述

新增 `src/hello_world.py`（358 行）+ `scripts/hello-world.sh` 包装脚本：仿 Jenga `hello_world.py` 的环境验证器，35 项检查覆盖 Python/torch/FA2 版本、pinned deps、jenga 与 `src/` 模块导入、GPU bf16 matmul、opt-350m 加载 + 位置扩展到 16k + 256-token 前向 loss、资源/数据集清单，输出 `[OK]/[WARN]/[FAIL]` 分类报告与汇总；无 FAIL 则打印 "Congratulations!" 并退出 0。支持 `--no-gpu`（纯导入/清单模式）、`--json`（机器可读），默认 < 10s。

## 变更列表

| 时间 | 操作 | 文件 | 规模 | 影响 |
|---|---|---|---|---|
| 12:10 | 新增 | `src/hello_world.py` | 358 行 | 分类检查器；FAIL 项 = python<3.10、torch/FA2 缺失、jenga/src 导入失败、模型文件缺失、前向非有限；WARN 项 = 版本漂移、GPU 非 sm_120、llama2/llama3 等可选资源/数据集缺失 |
| 12:10 | 新增 | `scripts/hello-world.sh` | 9 行 | `cd` 仓库根后用 `${PYTHON:-~/miniconda3/envs/SMELL/bin/python}` 执行并透传参数 |

## 验证结果（本机）

- `bash scripts/hello-world.sh`：exit 0，`checks=35 OK=34 WARN=1 FAIL=0`，6.2s；唯一 WARN 为 gated 的 `llama2/llama3/config.json` 缺失（预期）。
- `--no-gpu`：exit 0，`checks=31 OK=30 WARN=1`，2.1s（跳过 GPU/Forward 段）。
- 负例（`--model-dir temp/hello_empty`）：exit 1，`FAIL=3`（model.files / tokenizer / forward.smoke），汇总打印 "Setup Incomplete"。
- `--json temp/hello_world.json`：写入 35 项结果，`ok=34 warn=1 fail=0`。
- Forward 段实际验证了：opt-350m 加载、`ensure_positions` 扩展到 16384、FA2 bf16 前向 loss=13.36（有限）。

## 设计决策

- **FAIL/WARN 分级**：把"环境不可用"与"可选资源缺失"分开，保证在云端 A40（capability 8.6）上只会出现 capability WARN 而非 FAIL；gated 模型/未构建数据集一律 WARN。
- **短前向但真验证**：默认 seq_len=256，但位置表仍按 `--ctx 16384` 扩展，因此同时覆盖"位置扩展 + FA2 前向"两个此前最容易踩坑的路径，且保持 <10s。
- 与 Jenga hello_world 的差异：Jenga 用 llama2-7B 做兼容性测试（本机不可行），我们用 opt-350m + 位置扩展，等价地验证真实技术栈。

## 待办 / 风险

- 云端首次运行预期出现 `gpu.capability` WARN（A40 = (8,6)）与相同的 llama2/llama3 WARN，属设计内。
