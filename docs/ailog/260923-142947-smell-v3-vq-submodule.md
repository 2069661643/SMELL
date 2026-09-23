# Ailog 260923-142947 — SMELL-v3: VQ（Count Sketch 压缩）纳入 third_party submodule

## 概述

`third_party/VQ` 原先是一个未跟踪的嵌套 git 仓库（无任何远程；SMELL-v3 无法跟踪它，clone/迁移会丢失）。本次先在父仓库 `../VQ` 提交其工作区未跟踪产出（`script_multidim.py`、`plot_multidim.py`、`figs/*.png`），再把该 commit 同步到 v3 副本，最后以本地 `file://` URL 注册为固定 submodule（`bc2c4ad`）并 `absorbgitdirs`，与 FwdLLM/Jenga 拓扑一致。VQ 内容：Count Sketch 对高维向量的有损压缩/解压（numpy-only micro-benchmark，面向 FedLLM 权重/梯度传输压缩）。

## 变更列表

| 时间 | 操作 | 位置 | 影响 |
|---|---|---|---|
| 14:26 | 提交 | `../VQ`（父仓库，`bc2c4ad`） | `feat: add script_multidim.py grid search and plot_multidim.py figures`（5 文件，+251 行） |
| 14:27 | 同步 | `third_party/VQ` | 核对比对后删除副本内同名未跟踪文件 → `fetch file:///.../VQ main` → `merge --ff-only` 到 `bc2c4ad` |
| 14:28 | 新增 | `.gitmodules`、`third_party/VQ`（gitlink） | `git submodule add --force file:///D:/GitRepository/Projects/FedLLM/VQ third_party/VQ`（复用现有 clone，未重新下载） |
| 14:28 | 迁移 gitdir | `.git/modules/third_party/VQ` | `git submodule absorbgitdirs third_party/VQ`；worktree `.git` 变 gitfile；补 `origin` remote |
| 14:29 | 修改 | `AGENTS.md` | 目录树新增 VQ 行；Git 段「两个 submodule」→「三个」+ VQ 更新流程；环境段注明 sm_120 的 torch 偏离 |
| 14:29 | 修改 | `docs/ailog/260923-142534-...-plan.md` | Step 0 注明期望 3 个 submodule |

## 设计决策

- **本地 URL（用户选定）**：VQ 无远程（父仓库 AGENTS.md 亦确认「只有 HeimdaLLM 有 remote」），暂用 `file:///D:/GitRepository/Projects/FedLLM/VQ`。将来 VQ 有 GitHub/GitLab 远程后，改 `.gitmodules` 的 URL 并 `git submodule sync` 即可，历史 commit 不变。
- **钉死 commit、不跟踪 branch**：`bc2c4ad`（`refs/heads/main`），与 FwdLLM/Jenga 一致，避免意外漂移。
- **未跟踪产出归 VQ 历史**：`plot_multidim.py` / `script_multidim.py` / `figs/*.png` 是 VQ 自己的工作产出，提交到 VQ 仓库而非 v3 主仓库。
- v3 中的注册（`.gitmodules` + gitlink）**仅 staged，未 commit**（用户未要求提交）；建议提交信息 `SMELL v3: register VQ as third_party submodule`。

## 遗留 / 风险

- `file:///D:/...` 是机器本地路径：若 `../VQ` 被移动/删除，`submodule update --init` 会失败（已 checkout 的 worktree 与 `.git/modules` 不受影响，可继续读写）。迁到 WSL 后如需 update，把 `.gitmodules` URL 改为 `file:///mnt/d/GitRepository/Projects/FedLLM/VQ`（或先给 VQ 建真正的远程）。
- VQ 的 `out/`、`log/`、`__pycache__/` 仍按其自带 `.gitignore` 忽略，不进历史。
- 迁移 WSL 时建议连同 `.git/` 一起复制（而非重新 clone），submodule gitdir（`.git/modules/`）随行即可离线使用。
