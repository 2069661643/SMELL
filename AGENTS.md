# AGENTS.md — SMELL-v3 云端 A40 服务器工作区

> **你的角色：云端服务器 Agent。** 本工作区运行在 **A40×4 云端服务器（host `amax`）**，承担 16k
> 正式实验（位置适配 / predictor / BP 去风险 / 4 卡消融）与论文图表。本机 WSL（8GB RTX 5060）只做
> ≤4k smoke，不承担正式训练。命令、显存、后台化约定以本文件为准。

## 项目与方向

SMELL = Jenga token 级稀疏 × FwdLLM 前向梯度（ZOO）→ 联邦端侧长上下文 LLM 微调；目标 MobiCom 2027 / ICDE 2027。

- **v3 用 OPT-350M（`facebook/opt-350m`）**，取代 v2（`../SMELL/`，Llama2-7B；v2 ZOO 单轮 ≈42 min、50 轮 ≈135 h，不可接受）。
- Jenga 原生支持 OPT：`third_party/Jenga/src/jenga/utils/config_utils.py` 的 `get_opt_qk` 默认 `facebook/opt-350m`；实现见 `modeling_opt*.py`。
- 现行规范 `docs/standard.md`；v2 权威 brief `docs/old_ailog/AGENTS.md`；v2 全部结论**只**在 `docs/old_ailog/ailog/`（`5ee5186` 等修复不在任何本地 checkout，代码与 ailog 冲突以 ailog 为准）。

## 目录布局

```
src/        新代码全部写这里（third_party 只读，允许 COPY 进 src/）：
  models/   modeling_opt_smell.py(OPT 副本)、position_embed.py、token_selector.py(CATV)
  data/     build_discovery_16k.py、build_warmup_16k.py、check_partition.py
  train/    zoo.py、lora.py、longctx_adapt.py、train_predictor.py
  fed/      serial_fedavg.py、run_fed.py
  eval/ppl.py   bench/memory.py   hello_world.py(环境自检)
scripts/    setup_server.sh(云端 bootstrap)、run_ablation_cloud.sh(4 卡)、hello-world.sh
third_party/ 只读 submodule：FwdLLM / Jenga(OPT+数据+权重) / VQ(Count Sketch)
docs/       standard.md、ailog/、old_ailog/、reference/(gitignore)
dataset_v3/ checkpoints/ logs/ temp/   运行时目录，gitignore
```

- `third_party/FwdLLM/FedML` 是 1168 个**已跟踪普通文件**（非真子模块），递归 clone 不必再拉 FedML。
- v2 可跑参考代码在 `../SMELL/FwdLLM/`，但**只到 Phase 3**；B1/BP/Phase 4–6 修复不在其中。

## 当前进度与后续流程

| # | 交付物 | 位置 | 状态 |
|---|---|---|---|
| 1 | Discovery 16k 分片（α=0.1/0.3） | `dataset_v3/discovery_16k/{a01,a03}`（gitignore） | 代码 ✅，**数据须云端重建** |
| 2 | warmup 池（512×16k，零重叠） | 同上 `warmup_input_ids.npy` | 代码 ✅ |
| 3 | 位置扩展模块 | `src/models/position_embed.py` | ✅ |
| 4 | 长上下文适配 B/C | `src/train/longctx_adapt.py` | **云端 A40 16k 完成**：B ppl_full 31.4 / C 18.1，2k 无回归；选 C；权重见 `checkpoints/posemb_step1/MANIFEST.json` |
| 5 | CATV（r<s） | `src/models/token_selector.py` 等 | smoke ✅ |
| 6 | predictor 训练器 | `src/train/train_predictor.py` | smoke ✅ |
| 7 | BP/适配 runner | `src/fed/run_fed.py --trainer bp --pos-checkpoint --adapter-init` | smoke ✅ |
| 8 | 云端 bootstrap | `scripts/setup_server.sh` | ✅（路径按下面改） |
| 9 | 4 卡消融编排 | `scripts/run_ablation_cloud.sh` | ✅ |

云端执行顺序（详见 `docs/ailog/260924-120514-...` / `260924-114152-...`）：

1. **位置适配**（`longctx_adapt.py`，warmup 池）：B `--mode pos_only`、C `--mode pos_lora`；判据 16k G-PPL<50、`ppl_tail` 不劣化 ⇒ 产出 `pos_embed.pt`(+`adapter/`)。
2. **predictor**（`train_predictor.py --pos-checkpoint ...`）⇒ `predictor.pth`+`pruned_config.pth`。
3. **BP 去风险**（`run_fed --trainer bp`，c=2~3）同时校准 ZOO 步长。
4. **消融**（`run_ablation_cloud.sh`）：α∈{0.1,0.3} × CATV off/on，c=30、ZOO+LoRA、16k，4 卡并发。
5. 调参 + 出图。

**已知阻塞**：ZOO 当前 `lr=1e-3` 第 2 轮即 NaN（须重调 lr/eps/directions）；CATV 需预训练 predictor 或先解决位置适配；随机 predictor 的 PPL 不能作质量依据。

## 云端环境（与 WSL 脚本**不一致**，先读再跑）

- 仓库：`/home/yangyongbo/projects/smell/SMELLv3`（= `~/projects/smell/SMELLv3`）。
- conda 在 **`~/applications/anaconda3/bin/conda`**（**不是** `~/miniconda3`）。已有 env：`base`、`fwdllm`、`smell`(损坏/空)、`smell-v1`、`smell-v2`(torch 2.1.2+cu118 / FA 2.4.2，v2 栈，A40 可用)。
- 硬件：**4× A40 46GB**（sm_86，driver 570 / CUDA 12.8）、144 CPU、251GB RAM、`/home` 已用 93%（**仅 ~246G 空闲，及时清理**）。GPU 0/2/3 常有他人任务，用前 `nvidia-smi` 挑空闲卡并 `--gpu` 固定。
- **首次云端就绪清单**：
  1. `git submodule update --init --recursive`（当前 3 个 submodule 均未初始化）。
  2. 解压 `~/download/{dataset.zip,peft_model.zip,predictor.zip}` 到 `third_party/Jenga/`；缺的从 `third_party/Jenga/README.md` 的清华云链接取。
  3. `scripts/setup_server.sh` **硬编码 `$HOME/miniconda3` 与 env `SMELL`（大写）**，云端须改 `CONDA`/`--env-name`（建议新建 `smell-v3`，勿复用损坏的 `smell`）；或手动 `conda create -n smell-v3 python=3.10` 后按 `requirements-wsl-cu128.txt` 装 torch 2.8.0+cu128 + flash-attn 2.8.3（A40 sm_86 兼容）。
  4. 写 `jenga_src.pth`（`third_party/Jenga/src` 进 site-packages）；从 hf-mirror 拉 opt-350m。
  5. 重建数据：`build_discovery_16k.py --tag a01 --alpha 0.1`（a03=0.3）→ `build_warmup_16k.py --tag a0X` → `check_partition.py` 必须 PASSED。
- **`scripts/hello-world.sh` 也硬编码 `$HOME/miniconda3/envs/SMELL/bin/python`**：云端用 `PYTHON=~/applications/anaconda3/envs/smell-v3/bin/python bash scripts/hello-world.sh` 覆盖。A40 上 `gpu.capability` 与 llama2/llama3 资源 WARN 属预期。
- 数据/权重/checkpoint 不进 git：warmup 池用 `build_warmup_16k.py` 重建，`pos_embed.pt`/`adapter/`/`predictor.pth` 云端重训或单独传输。

## 构建 / 静态检查 / 测试命令

**无 CI / lint / 测试框架 / formatter**，不要引入。唯一强制静态检查是逐文件 `py_compile`：

```bash
PY=~/applications/anaconda3/envs/smell-v3/bin/python
$PY -m py_compile src/fed/run_fed.py            # 单文件检查（“单个测试”等价物）
$PY -m py_compile $(git diff --name-only -- '*.py')   # 仅改动的 py 文件
bash -n scripts/setup_server.sh scripts/run_ablation_cloud.sh   # shell 语法
```

自测（各文件内置 `_selftest`，退出码 0=PASS，是回归基线）：

```bash
$PY src/train/zoo.py             # 前向梯度 vs autograd + FedAvg 聚合
$PY src/fed/serial_fedavg.py     # 加权聚合 / 一致性断言 / JSONL
$PY src/models/token_selector.py # CATV 掩码 / r<s / s+r<=1 / IR
```

环境与脚本自检（改动 environment/CLI 后必跑）：

```bash
PYTHON=$PY bash scripts/hello-world.sh            # 35 项；--no-gpu 纯导入；--json OUT
bash scripts/setup_server.sh --check              # 验证 python/torch/cuda/FA/jenga/权重
bash scripts/setup_server.sh --dry-run            # 只打印安装步骤
bash scripts/run_ablation_cloud.sh --dry-run      # 打印 4 条命令与 GPU 映射
$PY src/data/check_partition.py --root dataset_v3/discovery_16k/a01   # 数据不变量
$PY src/fed/run_fed.py --help                     # CLI 参数一览
```

冒烟（本机 ≤4k；云端可用 `--max-clients/--max-train-samples/--rounds` 缩小）：

```bash
$PY src/fed/run_fed.py --tag a01 --gpu 1 --trainer zoo --catv off \
  --sparse 0.4 --rounds 1 --local-steps 1 --zo-directions 2 --max-clients 1 --out-root temp/smoke
```

## 代码风格

- **Python 3.10，无类型注解**（项目现状：函数签名不带 `typing`；用 docstring 说明，中英文皆可）。PEP 8、4 空格缩进、双引号、行宽不强求但贴近邻文件。
- **导入顺序**：stdlib → third-party → 本地。`torch` 等重依赖放在使用它的函数内部（保持 `--help`/CPU 路径可导入）。本地模块一律**绝对导入** `from src.foo import bar`；每个入口先用
  `REPO = Path(__file__).resolve().parents[N]; sys.path.insert(0, str(REPO))` 固定仓库根。
- **命名**：`snake_case` 函数/变量、`PascalCase` 类、`UPPER_CASE` 常量、`_prefix` 私有；路径一律 `pathlib.Path`。
- **错误处理**：不变量用 `assert cond, f"...got {x}"`；CLI 用户错误用 `raise SystemExit("...")`；库函数参数错误用 `ValueError`。**禁止裸 `except`**；仅在可选导入/GPU 探测处捕获具体异常并降级为 WARN/占位。断言信息必须含实际值。
- **CLI**：每个入口 `parse_args()` + `main()` + `if __name__ == "__main__": main()`；实验输出写 `config.json` + `metrics.jsonl`（用 `append_metrics`，`json.dumps(..., default=str)`）。
- **注释**：**默认不写注释**。所有对既有代码的改动必须带 `# SMELL N <组件> <动词> — 说明`（动词 `NEW/COPY/BEGIN/END/ADD/MODIFIED/REMOVED/COMMENTED/FIXED/REWRITTEN`，格式见 `docs/standard.md`）；新建文件头置 `# SMELL N <name> NEW/COPY — ...`。不要删旧标注。
- **不修改 `third_party/`**；上游 bug 用 `src/` 内 monkeypatch/自研路径规避（如 Jenga `flash_block.py` HEAD_DIM 硬编码 128、act-pack hooks）。

## 长任务与云端约定

- **长任务一律后台化**（>1 min：数据构建、smoke、训练、评测）：写成 `temp/run_<name>_bg.sh`（`temp/` gitignore），用 `setsid nohup bash temp/run_<name>_bg.sh >/dev/null 2>&1 < /dev/null &` 启动。
- **每步打点**：driver 对每阶段输出 `[HH:MM:SS] STEP ...`，日志写 `temp/logs/<name>_YYMMDD-HHMMSS/driver.log`，并留 `last_<name>.pid` / `last_<name>_dir.txt`。
- **启动确认**：`sleep 30` 后查日志/进程/`nvidia-smi`；起不来或立刻报错必须当场修复，不留悬空任务。
- **结果检查聊天触发**：任务完成后由用户在聊天唤醒 agent，按 `last_<name>_dir.txt` 读日志、校验产物、汇报结论。
- 4 卡并发用 `bash scripts/run_ablation_cloud.sh`（GPU0-3 = a01 off/on、a03 off/on；每实验 1 卡、30 client 串行）；**跑前确认目标卡空闲**，勿抢他人 GPU。

## OPT-350M / v2 高价值教训

- OPT 的 `lm_head` 与词嵌入**绑定**（`modeling_opt.py:943` `_tied_weights_keys`），head 用 `word_embed_proj_dim`（≠hidden_size，经 project_in/out）；v2 的 D3「lm_head LoRA」假设不成立。
- OPT 学习式绝对位置嵌入**零样本无法外推**：PPL 从 2048 的 20.2 恶化到 16k 的 588；必须做位置适配（见流程 Step 1）。
- Server 聚合（v2）= `Σg → 全局 L2 归一化 → clamp(-1,1) → θ-=lr·ĝ` ⇒ ‖Δθ‖≡lr；v3 改为**加权平均**，v2 的 lr 结论不迁移。
- Server/Client 参数枚举顺序与名字翻译必须严格一致，否则聚合**静默失效**（v2 M1 bug）。
- ZOO 中心差分 `grad≈(L(θ+hv)−L(θ−hv))/(2h)·v`；v2 cos(ZOO,BP)≈0.2–0.28；有效杠杆是每次更新样本数 n 与轮数，非扰动数 M。
- `var_threshold` 是绝对阈值，任何梯度缩放改动都会让它静默失效；client idle-timeout 计时基准须在 `notify()` 返回后重置。

## Git / submodule

- 独立仓库（`main`），提交信息 `SMELL v3: <English summary>`；不要 amend 已 push 的提交；提交前跑完相关文件 `py_compile`。
- 三个 submodule 只读；需上游改动时 fetch/checkout 后 `git add third_party/<x>` 更新指针，**不直接改文件**（脏指针）。
- VQ 已上 GitHub（`2069661643/SMELL-VQ`，remote 名 `github`）；开发上游是父仓库 `../VQ`。`git submodule sync` 会改写 submodule origin，慎用。
- 改完必须记 `docs/ailog/YYMMDD-HHMMSS-<desc>.md`（模板见 `docs/standard.md`）。

## 必读 ailog

| 文件 | 内容 |
|---|---|
| `docs/ailog/260924-120514-...predictor-bp-cloud-scripts.md` | 交付表、Jenga HEAD_DIM=128 bug、PEFT 正确加载、云端 5 步流程 |
| `docs/ailog/260924-114152-...longctx-adapt-and-delivery.md` | 位置适配 B/C、warmup 池、act-pack 梯度污染 |
| `docs/ailog/260924-103741-...framework-smoke-and-position-finding.md` | 位置外推长度曲线、框架冒烟、ZOO 步长问题 |
| `docs/ailog/260923-210703-...discovery-16k-fedavg-plan.md` | **实验方案权威依据**（数据/口径/协议/分工全部锁定） |
| `docs/ailog/260924-121252-...hello-world-checker.md` | 环境自检器用法与 FAIL/WARN 分级 |
| `docs/old_ailog/ailog/260915-163135-...b1-verdict-and-matched-bp-arm.md` | v2 ZOO vs BP 定论、n/lr 杠杆、~38 PPL 平台 |
