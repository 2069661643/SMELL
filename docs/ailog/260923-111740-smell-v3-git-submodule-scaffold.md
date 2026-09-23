# Ailog 260923-111740 — SMELL-v3: Git/submodule 框架搭建 + OPT-350M 工作区说明

## 概述

SMELL-v3 从「带 `.git` 的普通 clone 目录」正式搭建为独立 git 仓库，`third_party/FwdLLM`、`third_party/Jenga` 转为固定 submodule；写 `SMELL-v3/AGENTS.md`（v3 工作区 brief，含 OPT-350M 迁移要点与 v2 历史文档索引），并同步更新父仓库 `../AGENTS.md` 的布局/拓扑描述。

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 11:10 | 初始化仓库 | `SMELL-v3/.git` | `git init -b main` | 独立仓库，父仓库 FedLLM 不跟踪 |
| 11:12 | 新增 | `.gitignore` | 根 | 忽略 `checkpoints/ dataset/ logs/ temp/`、`docs/reference/`（~64 MB 论文）、`__pycache__/` |
| 11:13 | 新增 | `.gitmodules` | 根 | `third_party/FwdLLM` @`36ecdbc`、`third_party/Jenga` @`23764a6`（`git submodule add --force` 复用现有 clone，再 `absorbgitdirs`，git dir 移入 `.git/modules/`） |
| 11:20 | 新增 | `AGENTS.md` | 根 | v3 brief：项目方向、目录、submodule 工作流、环境、OPT-350M 迁移要点、v2 教训、必读 ailog |
| 11:25 | 修改 | `../AGENTS.md`（父仓库） | 布局表 / Git 拓扑 / SMELL 段 / 跨库说明 | 新增 SMELL-v3 行；SMELL/ 标为 v2 且本地只到 Phase 3；修正「root .gitignore 忽略 docs」（实际 `./docs` 是 no-op 模式）、「SMELL 有 remote」（实际只有 HeimdaLLM 有） |
| 11:30 | 提交 | `4fec078` | 仓库首次提交 | `SMELL v3: workspace scaffold with FwdLLM/Jenga submodules and v2 historical docs`（36 文件） |

## 设计决策

- **third_party 保持上游纯净、只读**：SMELL 新代码写 `src/`（用户确认），不沿用 v2「直接改 FwdLLM 副本」的做法；这样 submodule 指针可保持干净，上游可随时升级。
- **submodule 固定 commit（无 branch 跟踪）**：避免 `submodule update --remote` 意外漂移。
- `docs/reference/`（论文 PDF/LaTeX）不进 git；`docs/old_ailog/`、`docs/standard.md` 入库，保证新 clone 自带历史与规范。

## 待办

- 在 `src/` 落地 OPT-350M 版模型/训练代码（参考 `third_party/Jenga/src/jenga/models/modeling_opt*.py`；v2 的 Llama 版只作参考，勿照抄）。
- 补 v3 的 requirements.txt（Jenga 栈：py3.10 / torch 2.1.2 / transformers 4.45.2 / peft / flash-attn）。
- 若后续需要上游改动：fork 后更新 submodule 指针（流程见 `AGENTS.md`），不要直接改 `third_party/` 内文件。
