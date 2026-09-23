# Ailog 260923-174807 — SMELL-v3: 项目上传 GitHub（origin = 2069661643/SMELL）

## 概述

SMELL-v3 仓库首次上传到 `git@github.com:2069661643/SMELL.git`（`main`）。上传前把此前 Windows→WSL 迁移遗留的 CRLF-only 脏文件清干净（根仓库 3 个 old_ailog、VQ 子模块 4 个文件），把 VQ submodule 注册与本日 WSL 环境验证拆成两个提交，推送后 `origin/main = dcdeeba`。WSL 侧原本没有 SSH 凭证，本次把 Windows 的 `C:\Users\yangy\\.ssh/id_rsa` 复制到 `~/.ssh`（600）后认证成功（`Hi 2069661643!`）。

## 变更列表

| 时间 | 操作 | 位置 | 影响 |
|---|---|---|---|
| 17:46 | 复制 SSH key | `/mnt/c/Users/yangy/.ssh/id_rsa{,.pub}` → `~/.ssh/` | WSL 可 `ssh -T git@github.com`，账号 2069661643 |
| 17:47 | 修复 | `docs/old_ailog/{ailog/260915-163135-...,tasks/ex1-2.md,tasks/ex5/phase3.md}` | `--ignore-cr-at-eol` 确认 0 内容差异，`checkout -- .` 后归零 |
| 17:47 | 提交 `f607b58` | `.gitmodules`、`third_party/VQ`、`docs/ailog/260923-142947-...vq-submodule.md` | `SMELL v3: register VQ as third_party submodule` |
| 17:47 | 提交 `dcdeeba` | `AGENTS.md`、`requirements-wsl-cu128.txt`、`src/smoke_jenga_opt.py`、plan/result 两篇 ailog | `SMELL v3: WSL conda env SMELL with Jenga hello-world and OPT-350M smoke` |
| 17:47 | 新增 remote | `origin git@github.com:2069661643/SMELL.git` | `git push -u origin main` 成功（空仓库 → new branch） |
| 17:48 | 修复 | `third_party/VQ` 工作区 4 文件（`.gitignore`/`.vscode/settings.json`/`plot_multidim.py`/`script_multidim.py`） | 同为 CRLF-only，`checkout -- .` 后 submodule 全干净 |
| 17:48 | 校验 | 远端 refs | `HEAD = refs/heads/main = dcdeeba`；根仓库 `git status` 空，三个 submodule dirty=0 |

## 设计决策

- **提交身份不写 config**：WSL 无 `user.name/email`，用 `git -c user.name=YangYongBo118 -c user.email=jackjack5233@mail.ustc.edu.cn commit`（镜像 Windows 全局配置），避免改动任何 git config。
- **拆两个提交**：VQ 注册（沿用 142947 计划里的 message）与 WSL 环境验证分开，保持历史可读。
- **Submodule 可达性**：已确认 `FwdLLM@36ecdbc`、`Jenga@23764a6` 都在上游 `origin/master` 上，clone 可 fetch；VQ 按既有文档状态原样保留（见遗留）。
- **上传前 sanity**：tracked files 44、`.git` 93MB（含 submodule gitdir），无 checkpoints/dataset/temp 入库。

## 遗留 / 风险

- **VQ submodule 对别的机器不可用**：gitlink `bc2c4ad` 无远程，且 `.gitmodules` 的 URL 是 `file:///D:/GitRepository/Projects/FedLLM/VQ`（仅本机 Windows 有效）。后续建议：给 VQ 建独立 GitHub 仓库并 push `main`，再 `git submodule set-url third_party/VQ <new-url>` 并提交。
- WSL `~/.ssh/id_rsa` 是 Windows 私钥的副本（无 passphrase 则等于多了一份明文凭证）；不想要可删除，改用 `ssh-keygen` + GitHub 新 key。
- GitHub 仓库暂无顶层 `README.md`/`LICENSE`（入口是 `AGENTS.md`）；如需对外展示可补。
