# AGENTS.md — SMELL-v3 多机工作区（A40 `amax` + AutoDL 4090）

> **你的角色：SMELL-v3 实验 Agent。** 本仓库有 3 类 checkout：云端 **A40×4（host `amax`）** 承担 16k
> 正式实验与论文图表；**AutoDL 1×RTX 4090 24G（见「AutoDL 4090 工作机」节）**；WSL（8GB RTX 5060）只做 ≤4k smoke。
> **开工先跑 `hostname` + `git branch --show-current`**：实验工作分支是 **`exp/rank-rotation-lora`**，
> `main` 只是 260924 的旧交付快照（落后 19 个提交），别在 main 上找近期实验代码。

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
docs/       standard.md、weight-manifest.md(换机交接)、ailog/、old_ailog/、reference/(gitignore)
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
| 6 | predictor 训练器 | `src/train/train_predictor.py` | ✅ **唯一可用权重：`checkpoints/predictor/step4_a01_pos_only_causal/`**（causal target 修复；step2/step3 作废） |
| 7 | BP/适配 runner | `src/fed/run_fed.py --trainer bp --pos-checkpoint --adapter-init` | smoke ✅ |
| 8 | 云端 bootstrap | `scripts/setup_server.sh` | ✅（`amax` 需改 CONDA/env 路径；AutoDL 默认适配） |
| 9 | 4 卡消融编排 | `scripts/run_ablation_cloud.sh` | ✅ |

云端执行顺序（详见 `docs/ailog/260924-120514-...` / `260924-114152-...`）：

1. **位置适配**（`longctx_adapt.py`，warmup 池）：B `--mode pos_only`、C `--mode pos_lora`；判据 16k G-PPL<50、`ppl_tail` 不劣化 ⇒ 产出 `pos_embed.pt`(+`adapter/`)。
2. **predictor**（`train_predictor.py --pos-checkpoint ...`；**必须带 causal target 修复**，见 ailog `260928-110233`）⇒ `predictor.pth`+`pruned_config.pth`，**只用 step4 causal 版**。
3. **BP 去风险**（`run_fed --trainer bp`，c=2~3）同时校准 ZOO 步长。
4. **消融**（`run_ablation_cloud.sh`）：α∈{0.1,0.3} × CATV off/on，c=30、ZOO+LoRA、16k，4 卡并发。
5. 调参 + 出图。

**已知阻塞 / 最新结论（ZOO 精度根因，详见 handoff `docs/ailog/260927-194011-smell-v3-handoff.md`）**：

- **ZOO 长期「平在 ~32」的根因 = bf16 损失量化**（有限差分信号 ~1e-6 ≪ 损失量化台阶 0.0156），**不是维度/秩/覆盖率**。old_ailog `260914-103130` 同款（bf16 吞 LoRA 增量）。验证脚本 `temp/diag_bf16_swallow.py`。
- **修复**：`src/models/modeling_opt_smell.py` 新增 **`OptSdpaAttention`**（fp32 `F.scaled_dot_product_attention`、O(seq)）；16k fp32 前向 2.9s、BP ~13s/样本。fp32 下 **`cos≈0.5·√(cLD/d_eff)`**（与理论吻合；bf16 仅 0.01–0.08×）。FA2 只支持 bf16/fp16，故高精度必须走 SDPA。
- **新瓶颈 = 成本/覆盖权衡**：fp32 比 bf16 慢 ~9×；全层 ZOO 要 cos≈0.15 需大 D（~28h/轮，不可行）；「少层/层轮转（k=1）」在 **BP 下就不收敛**（仅全 24 层收敛）。
- **稀疏口径**：`thresh` = `config.sparse` = **保留 top-40% 的 query 块**（Jenga 仅**上半层**剪、末层豁免）；`OptSdpaPruneAttention` 为其真语义（token 子集化）实现。predictor 须 `load_predictor_weights` 加载训练权重，随机 predictor 的 PPL 不能作质量依据。
- ZOO `lr=1e-3` 第 2 轮 NaN；跑前先按 `delta_norm` 校准 lr（流程见 ailog `260928-004251`）。

## 最新进展 / 下一步（截至 exp HEAD `f4e879d`，260928 12:41）

- **predictor 病根已修（step4）**：Jenga `block_attn_pool` 目标**无 causal mask**，对 OPT（post-norm、未归一化残差）退化为「只选尾部块」⇒ 16k LM loss 5.08；`train_predictor.py` 加 causal mask 后重训 `step4_a01_pos_only_causal`（loss 3.44，first-half 选块 0.52–0.59）。**step2/step3 作废**，详见 ailog `260928-110233`、`docs/weight-manifest.md`。
- **BP k-scan 已出**：收敛随覆盖单调（k=1 −0.038 / k=4 −0.169 / k=12 −0.280，同轮 0–32）⇒ 无可用中间粒度；用户选定 **k=4 rotate4 sparsity ZOO 为 TD-3**（L2 D22, c30, cos≈0.10），由 TD-2' 门控（cos ≥ 0.05）。
- **TD-3 已停**（云端 `logs/fed/zoo_k4_sparse` 留 adapter_round000–004）；**长跑暂不启动**（用户要求）。
- 下一步：① 用 step4 predictor 重跑 **TD-2'**（`diag_cos_grid.py --attn sdpa_prune --dtype fp32`）→ 过门控再起 TD-3；② 若 ZOO 粒度不可用 → 主线转 **BP-based FL**（客户端数曲线 c3→26.1 / c10→22.3 / c30→21.5）；③ 成本缓解走 Jenga fused block-sparse / `flex_attention`，gather 已证不划算。
- 云端跑批检查入口：`temp/logs/last_*_dir.txt`；**读 `metrics.jsonl`（driver.log 有 Python 块缓冲，会滞后）**。

## AutoDL 4090 工作机（本 checkout：`/root/smell/SMELL`）

- 硬件：1×**RTX 4090 24GB**（sm_89，driver 595 / CUDA 13.2）、20 CPU、系统盘 **30G（紧张，装前先 `df -h`）**。**单卡：示例命令里的 `--gpu 1` 在本机要改 `--gpu 0`**（`run_fed.py --gpu` 直接写 `CUDA_VISIBLE_DEVICES`，索引 1+ 看不到卡）。
- conda：**`~/miniconda3`**（base py3.12 已装 torch 2.8.0+cu128）。`scripts/setup_server.sh` 的默认路径（`$HOME/miniconda3`、env `SMELL`）**正好适配本机**，但尚未跑过：transformers/peft/flash-attn/jenga/opt-350m 全未装。
- GitHub：**HTTPS git 会挂死，SSH 正常**（`git@github.com` 已认证）。三个 submodule 的 URL 已在本机 `.git/config` 改成 SSH；**别跑 `git submodule sync`**（会改回 HTTPS 再挂死）。
- 已就位（sha256 与 `docs/weight-manifest.md` 完全一致）：`checkpoints/posemb_step1/a01_pos_only_500step/`、`checkpoints/predictor/step4_a01_pos_only_causal/`。
- 尚缺：`third_party/Jenga/checkpoints/opt-350m/`（632M，`setup_server.sh` 可从 hf-mirror 拉）、`dataset_v3/discovery_16k/a01/`（warmup 池，传输或重建）。
- 本机尚未跑过训练；`python3 -m py_compile` 全 `src/` 已通过（base 无 jenga，跑脚本前先建 env）。

## 云端 A40 环境（host `amax`；与 WSL 脚本**不一致**，先读再跑）

- 仓库：`/home/yangyongbo/projects/smell/SMELLv3`（= `~/projects/smell/SMELLv3`）。
- conda 在 **`~/applications/anaconda3/bin/conda`**（**不是** `~/miniconda3`）。已有 env：`base`、`fwdllm`、`smell`(损坏/空)、`smell-v1`、`smell-v2`(torch 2.1.2+cu118 / FA 2.4.2，v2 栈，A40 可用)。
- 硬件：**4× A40 46GB**（sm_86，driver 570 / CUDA 12.8）、144 CPU、251GB RAM、`/home` 已用 93%（**仅 ~246G 空闲，及时清理**）。**约定只用 GPU2**（GPU0/1/3 被他人占用）；跑前 `nvidia-smi` 确认。
- **首次云端就绪清单**：
  1. `git submodule update --init --recursive`（未初始化时；HTTPS 挂死时用 SSH URL，见「AutoDL 4090 工作机」节）。
  2. 解压 `~/download/{dataset.zip,peft_model.zip,predictor.zip}` 到 `third_party/Jenga/`；缺的从 `third_party/Jenga/README.md` 的清华云链接取。
  3. `scripts/setup_server.sh` **硬编码 `$HOME/miniconda3` 与 env `SMELL`（大写）**，云端须改 `CONDA`/`--env-name`（建议新建 `smell-v3`，勿复用损坏的 `smell`）；或手动 `conda create -n smell-v3 python=3.10` 后按 `requirements-wsl-cu128.txt` 装 torch 2.8.0+cu128 + flash-attn 2.8.3（A40 sm_86 兼容）。
  4. 写 `jenga_src.pth`（`third_party/Jenga/src` 进 site-packages）；从 hf-mirror 拉 opt-350m。
  5. 重建数据：`build_discovery_16k.py --tag a01 --alpha 0.1`（a03=0.3）→ `build_warmup_16k.py --tag a0X` → `check_partition.py` 必须 PASSED。
- **`scripts/hello-world.sh` 也硬编码 `$HOME/miniconda3/envs/SMELL/bin/python`**：云端用 `PYTHON=~/applications/anaconda3/envs/smell-v3/bin/python bash scripts/hello-world.sh` 覆盖。A40 上 `gpu.capability` 与 llama2/llama3 资源 WARN 属预期。
- 数据/权重/checkpoint 不进 git：warmup 池用 `build_warmup_16k.py` 重建，`pos_embed.pt`/`adapter/`/`predictor.pth` 云端重训或单独传输。

## 构建 / 静态检查 / 测试命令

**无 CI / lint / 测试框架 / formatter**，不要引入。唯一强制静态检查是逐文件 `py_compile`：

```bash
PY=~/applications/anaconda3/envs/smell-v3/bin/python   # amax；AutoDL 4090 用 ~/miniconda3/envs/SMELL/bin/python
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

- **工作分支 = `exp/rank-rotation-lora`**（clone/换机后先 `git checkout exp/rank-rotation-lora`）；`main` 是旧交付快照，别在上面开发或找近期实验代码。提交信息 `SMELL v3: <English summary>`；不要 amend 已 push 的提交；提交前跑完相关文件 `py_compile`。
- 新机未配置 git 身份时提交会失败：`git config user.name/user.email` 与历史作者一致（`git log -1 --format='%an <%ae>'` 查看）。
- 三个 submodule 只读；需上游改动时 fetch/checkout 后 `git add third_party/<x>` 更新指针，**不直接改文件**（脏指针）。GitHub HTTPS 挂死时改用 SSH URL（见「AutoDL 4090 工作机」节）。
- VQ 已上 GitHub（`2069661643/SMELL-VQ`，remote 名 `github`）；开发上游是父仓库 `../VQ`。`git submodule sync` 会改写 submodule origin，慎用。
- 改完必须记 `docs/ailog/YYMMDD-HHMMSS-<desc>.md`（模板见 `docs/standard.md`）。

## 必读 ailog

| 文件 | 内容 |
|---|---|
| `docs/weight-manifest.md`（非 ailog） | **换机交接清单**：权重/数据路径 + sha256 + TD-2'/TD-3 运行 flags |
| `docs/ailog/260928-110233-...predictor-rootcause-noncausal-target-causal-fix.md` | **predictor causal 修复（step4 唯一可用）**：Jenga 非 causal target 对 OPT 不适配 |
| `docs/ailog/260928-004251-...fix-ol2-prune-fallback-wire-fp32-sdpa-launch-k4-td3.md` | sdpa_prune 末层 O(L²) 修复、run_fed `--attn/--dtype`、k-scan 结论与 TD-3 |
| `docs/ailog/260927-194011-...handoff.md` | ZOO 精度根因 + fp32 SDPA 修复、运行队列（上一轮交接） |
| `docs/ailog/260927-121756-...tc-fp32-cos-validated.md` | **TC：fp32 下 `cos≈0.5√(cLD/d)`，ZOO 打通**；成本约束 |
| `docs/ailog/260927-130020-...thresh-verify-gather-bench-td2b1-pilotbp.md` | `thresh` 口径、token-prune gather 不划算、TD-2/先导BP |
| `docs/ailog/260926-184900-...tb-sdpa-fp32-and-tc.md` | 新增 fp32 SDPA 注意力（16k 可行） |
| `docs/ailog/260926-182711-...bf16-loss-quant-and-stepA-todo.md` | Step A：bf16 损失量化根因确认 |
| `docs/ailog/260926-015115-...zo-layerrotate-negative-and-clientcount.md` | 层轮转 ZO 负结果 + BP 客户端数曲线 |
| `docs/ailog/260924-120514-...predictor-bp-cloud-scripts.md` | 交付表、Jenga HEAD_DIM=128 bug、PEFT 正确加载、云端 5 步流程 |
| `docs/ailog/260924-114152-...longctx-adapt-and-delivery.md` | 位置适配 B/C、warmup 池、act-pack 梯度污染 |
| `docs/ailog/260924-103741-...framework-smoke-and-position-finding.md` | 位置外推长度曲线、框架冒烟、ZOO 步长问题 |
| `docs/ailog/260923-210703-...discovery-16k-fedavg-plan.md` | **实验方案权威依据**（数据/口径/协议/分工全部锁定） |
| `docs/ailog/260924-121252-...hello-world-checker.md` | 环境自检器用法与 FAIL/WARN 分级 |
| `docs/old_ailog/ailog/260915-163135-...b1-verdict-and-matched-bp-arm.md` | v2 ZOO vs BP 定论、n/lr 杠杆、~38 PPL 平台 |
