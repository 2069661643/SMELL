# Ailog 260923-175708 — SMELL-v3: VQ 上传 GitHub，submodule URL 切到 SMELL-VQ

## 概述

此前 `third_party/VQ` 是本地 submodule（`file:///D:/GitRepository/Projects/FedLLM/VQ`，gitlink `bc2c4ad` 无任何远程），别的机器 clone SMELL-v3 时 VQ 无法 init。本次把 VQ 全部 5 个 commit（`main`，12 个跟踪文件，~1 MB）推送到新建的 `git@github.com:2069661643/SMELL-VQ.git`，并把 superproject `.gitmodules` 的 VQ URL 换成该 GitHub 地址；submodule 本地 `origin` 仍是父仓库 `../VQ`（保留既有开发上游工作流）。推送 VQ 前用 grep 扫过 password/secret/token 等关键字，无命中。

## 变更列表

| 时间 | 操作 | 位置 | 影响 |
|---|---|---|---|
| 17:56 | 新增 remote | `third_party/VQ` → `github = git@github.com:2069661643/SMELL-VQ.git` | submodule 内双 remote：`origin`=本地父仓库，`github`=发布 |
| 17:56 | 推送 | `SMELL-VQ main`（bc2c4ad，new branch） | `ls-remote` 确认 `HEAD = refs/heads/main = bc2c4ad` |
| 17:57 | 修改 | `.gitmodules` VQ `url` | `file:///D:/...` → `git@github.com:2069661643/SMELL-VQ.git`（clone 可用） |
| 17:57 | 修改 | `AGENTS.md` 目录树 + Git/submodule 段 | VQ 标注 GitHub 地址；补「发布时 push github」流程与 submodule sync 注意事项 |
| 17:57 | 提交 | superproject（见下） | `SMELL v3: point VQ submodule at SMELL-VQ GitHub remote` |

## 设计决策

- **remote 命名 `github` 而非改写 `origin`**：`origin` 保持 `file:///D:/...`，AGENTS.md 既有的「父仓库 ../VQ 开发 → fetch origin → 更新 gitlink」流程不变；GitHub 只作为发布出口。未设 upstream tracking，避免 `git push` 默认行为漂移。
- **手动改 `.gitmodules` 而不跑 `git submodule set-url`**：后者会同时把 `.git/modules/third_party/VQ/config` 里的 `origin` 覆盖为 GitHub URL，破坏本地开发工作流。
- **不改写历史 ailog**：`260923-142947` 记录的 file:// 临时状态是当时的真实历史，本次变更另记一篇。

## 遗留 / 风险

- VQ 仓库双上游（本地 `../VQ` 开发、GitHub 发布），每次更新后需手动 `git -C third_party/VQ push github main`；忘记 push 会导致 superproject gitlink 指向 GitHub 上不存在的 commit。
- `git submodule sync` 会把 submodule 的 `origin` 改写为 `.gitmodules` 里的 GitHub URL（AGENTS.md 已注明）；届时本地 fetch origin 流程会失效，需要手动改回或改用 `github`/`upstream` 命名。
- `SMELL-VQ` 无 CI，`figs/`、`README*.md` 等均为源码与文档，未做远端构建验证。
