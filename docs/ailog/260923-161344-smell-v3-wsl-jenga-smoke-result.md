# Ailog 260923-161344 — SMELL-v3: WSL2 conda env SMELL + Jenga hello-world/OPT-350M smoke 跑通

## 概述

在 WSL2 Ubuntu 24.04 + RTX 5060 Laptop（8 GB，sm_120）上完成 Jenga 技术栈落地：新建 conda 环境 **`SMELL`**（用户指定命名，取代计划中的 `smell-v3`），实装 **torch 2.8.0+cu128**（计划写的 2.8.1 不存在）、flash-attn 2.8.3（cxx11abiTRUE wheel）、transformers 4.45.2 / peft 0.13.2 / tokenizers 0.20.1 / numpy 1.26.4。Jenga 三个资源包（dataset/predictor/peft_model）从 `/mnt/d/.../download/*.zip` 解压进 `third_party/Jenga/{dataset,checkpoints}`（均被 gitignore），并从 hf-mirror 拉齐 `opt-350m`（含 663 MB 权重）与 opt-1.3b/2.7b/6.7b 的 `config.json`。官方安装检查脚本 `hello-world.sh` **退出码 0**：PEFT artifacts、datasets 全过，仅 gated 的 llama2/llama3 config 缺失（hf-mirror 返回 401 拒绝）。计划的真正 GPU 验收——`src/smoke_jenga_opt.py`——FA2 与 eager 两条路径均 `[smoke] PASS`。落盘 `requirements-wsl-cu128.txt`；顺手把 Jenga（371 文件）与 FwdLLM（1228 文件）此前 Windows→WSL 迁移产生的 **CRLF-only 脏工作区**恢复干净（`git checkout -- .`，submodule 指针未动）。

## 变更列表

| 时间 | 操作 | 文件/位置 | 影响 |
|---|---|---|---|
| 15:00 | 新增 | `~/miniconda3`（USTC 镜像，`Miniconda3-py310_26.7.1-0`） | TUNA 下载被限速且 md5 mismatch，改 USTC 12MB/s 成功 |
| 15:05 | 新增 | conda env `SMELL`（Python 3.10.21，pip 26.2.1） | 后续所有 Jenga/SMELL 运行环境 |
| 15:14–15:16 | 下载 | `~/wheels/flash_attn-2.8.3+cu12torch2.8cxx11abi{FALSE,TRUE}-cp310-cp310-linux_x86_64.whl` | 经 `ghfast.top` GitHub 代理各 244MB，zip 完整 |
| 15:15 | 下载 | `~/wheels/torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl`（889MB） | 来自 `download-r2.pytorch.org`，sha256 校验通过（2–4MB/s） |
| 15:47–15:53 | 安装 | env `SMELL` | torch 2.8.0+cu128 / CUDA 12.8 / ABI=True；FA 2.8.3；transformers 4.45.2、tokenizers 0.20.1、peft 0.13.2、accelerate 1.0.1、datasets 2.21.0、numpy 1.26.4；nvidia-*/triton 依赖经 USTC PyPI（~10MB/s） |
| 15:01–15:03 | 解压 | `third_party/Jenga/{dataset,checkpoints/predictor,checkpoints/peft_model}` | `dataset.zip`(2.1GB) / `predictor.zip`(54MB) / `peft_model.zip`(5.3GB) 从 `/mnt/d/GitRepository/Projects/FedLLM/download/` 解压，12 个 adapter + predictor.pth 就位；落在 gitignore 目录，submodule 无脏文件 |
| 15:53 | 新增 | `envs/SMELL/lib/python3.10/site-packages/jenga_src.pth` | 指向 `third_party/Jenga/src`，无需 `pip install -e`（避免 third_party 脏化）即可 `import jenga` |
| 15:56–16:11 | 下载 | `Jenga/checkpoints/opt-350m/`、`opt-{1.3b,2.7b,6.7b}/config.json` | hf-mirror；`pytorch_model.bin` 663MB（2–13MB/s）；误下的 `flax_model.msgpack`/`tf_model.h5` 已删除 |
| 16:11 | 删除 | `Jenga/checkpoints/{llama2,llama3}/config.json` | hf-mirror 对 gated 仓库返回 401 错误体（非 JSON），删除以保持 checker 语义（缺失即报缺失） |
| 16:12 | 运行 | `third_party/Jenga/hello-world.sh` | 退出码 0；PEFT/datasets 全过；仅 llama2/llama3 config 缺失 → `Setup Incomplete`（本机预期） |
| 16:12 | 新增 | `src/smoke_jenga_opt.py`（计划附录 A，头部 `# SMELL 3 smoke_jenga_opt NEW`） | OPT-350M Jenga 自定义实现前向+反向冒烟；`py_compile` 通过 |
| 16:12–16:13 | 运行 | `src/smoke_jenga_opt.py` / `--no-flash` | FA2：loss=12.98，13.7s，peak 3.08GiB，PASS；eager：loss=11.99，19.8s，peak **15.26GiB**（>8GB，走系统内存回退），PASS |
| 16:13 | 新增 | `requirements-wsl-cu128.txt` | `pip freeze` 87 行，实际安装版本快照 |
| 16:13 | 修复 | `third_party/Jenga`、`third_party/FwdLLM` | `git status` 371/1228 个 M 全为 CRLF-only（迁移产物），`git checkout -- .` 后均归零；submodule 指针与文件内容未改 |

## 验证输出（摘要）

```
hello-world.sh:
ERROR: Base model 'llama2' config not found at: checkpoints/llama2/config.json
ERROR: Base model 'llama3' config not found at: checkpoints/llama3/config.json
--- Checking for PEFT artifacts ---  All checked PEFT artifacts seem to be present.
--- Checking for datasets ---       All datasets seem to be present.
Setup Incomplete: One or more checks failed.        (exit code 0)

smoke_jenga_opt.py (FA2):  torch=2.8.0+cu128 cap=(12,0) params=375.3M loss=12.9789 step=13.71s peak_vram=3.08GiB [smoke] PASS
smoke_jenga_opt.py (eager): attn_implementation=eager params=375.3M loss=11.9902 step=19.80s peak_vram=15.26GiB [smoke] PASS
```

## 设计决策

- **env 命名 `SMELL`**：用户指定；计划文档中 `smell-v3` 作废。
- **torch 2.8.0 而非 2.8.1**：经 pypi.org / SJTU / download.pytorch.org 索引核实 **2.8.1 不存在**（2.8.0 → 2.9.0 直跳）；FA 2.8.3 的 `torch2.8` wheel 与 2.8.0 兼容。
- **FA wheel 选 `cxx11abiTRUE`**：实测 `torch._C._GLIBCXX_USE_CXX11_ABI == True`；FALSE 变体留 `~/wheels` 备用。
- **用 `.pth` 而非仅 conda env var 注入 PYTHONPATH**：`bash hello-world.sh` 直接 `python ...`（不经过 conda activate）也能 import jenga；conda env var 亦已设置。
- **不伪造 llama2/llama3 config**：错误体内容为 “Please enable access to public gated repositories…” / “request … has been rejected”，说明 hf-mirror 不能绕过 gating；保留它们会让 checker 误判存在并在 compat test 阶段抛 traceback，删除后输出干净且符合计划 §5 预判。
- **CRLF 修复**：`git diff --ignore-cr-at-eol` 显示 0 差异，确认全部 M 为行尾问题；`checkout -- .` 只恢复工作区，不改 index/HEAD，不违反 third_party 只读约定。
- **eager 15.26GiB 现象**：8GB 物理显存下 CUDA on WSL 允许溢到系统内存（因此不 OOM 但 step 从 13.7s 涨到 19.8s）；SMELL 后续必须走 FA2，且控制 seq_len。

## 遗留 / 风险

- **hello_world 无法全绿**：llama2/llama3 是 gated 仓库（申请被拒），且 7B bf16≈13.5GB > 8GB，本机物理不可行；本文件按计划 §5 语义验收（脚本无异常跑完 + 缺失项符合预期）。
- **根仓库（SMELL-v3）仍有迁移期 CRLF 脏文件**（`AGENTS.md`、`docs/old_ailog/*` 等，均未提交），且 `third_party/VQ` 注册仍 staged、两份 ailog 未跟踪——均为本次任务之前的状态，未处理；若清理需先确认 AGENTS.md 的手改内容。
- `hf-mirror` 单线程下载波动大（opt-350m 从 13MB/s 掉到 1MB/s）；大批量素材建议直接 `/mnt/d` 拷贝。
- 重装环境时切勿 `pip install -r third_party/Jenga/requirements.txt`（会把 torch 降回 2.1.2，无 sm_120 kernel）；用 `requirements-wsl-cu128.txt`，FA wheel 按 ABI 从 `~/wheels` 装。
