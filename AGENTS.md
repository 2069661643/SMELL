# AGENTS.md — SMELL-v3 多机工作区（A40 `amax` + AutoDL 4090）

> **你的角色：SMELL-v3 实验 Agent。** 仓库有 3 类 checkout：云端 **A40×4（host `amax`）**、**AutoDL 1×RTX 4090 24G**（本 checkout：`/root/smell/SMELL`）、WSL（8GB RTX 5060，只做 ≤4k smoke）。
> **开工先跑 `hostname` + `git branch --show-current`**：
> - 主实验线 `exp/distilbert-16k` @ `37e5708`（261020）；`main` 已 fast-forward 到同一提交（本地）。
> - VQ 线 `exp/vq-compression`（从 `37e5708` 拉出）；`exp/rank-rotation-lora` 是祖先/旧线。
> - `main` 已不再是 260924 旧快照；origin/main 可能仍落后，push 前先确认。

## 项目与方向

SMELL = Jenga token 级稀疏 × FwdLLM 前向梯度（ZOO）→ 联邦端侧长上下文 LLM 微调；目标 MobiCom 2027 / ICDE 2027。

- **OPT-350M（`facebook/opt-350m`）**：24 层、hidden 1024、16 heads、vocab 50272；Jenga 稀疏 = 上半层 11–22、pool 64、`sparse=0.4`（保留 top-40% query 块，末层豁免）。
- **DistilBERT（`distilbert-base-uncased`）**：6 层、hidden 768、11 类 arXiv 分类；`keep_block0=True`（BERT-only），LoRA 目标 `q_lin/k_lin/v_lin/out_lin`。
- v2（`../SMELL/`，Llama2-7B）只作历史参考：v2 ZOO 单轮 ≈42 min、50 轮 ≈135 h，不可接受；v2 结论**只**在 `docs/old_ailog/ailog/`。
- 现行规范 `docs/standard.md`；权重/数据路径与 sha256 以 `docs/weight-manifest.md` 为准（不在本文件重复）。

## 目录布局

```
src/        新代码全部写这里（third_party 只读，允许 COPY 进 src/）：
  models/   modeling_opt_smell.py、modeling_distilbert_smell.py、position_embed*.py、token_selector.py(CATV)
  data/     build_discovery_16k.py、build_warmup_16k.py、build_arxiv_16k.py、build_warmup_arxiv_16k.py、check_partition.py
  train/    zoo.py、lora.py、longctx_adapt.py、train_predictor*.py、train_pos_head_bert.py、lowrank_reparam.py
  fed/      serial_fedavg.py、run_fed.py
  compress/ count_sketch.py（VQ/Count Sketch，训练上行压缩）
  eval/     ppl.py、classify.py   bench/memory.py   hello_world.py(环境自检)
scripts/    setup_server.sh、run_ablation_cloud.sh、hello-world.sh、diag_cos_*.py、paired_eval.py
third_party/ 只读 submodule：FwdLLM / Jenga(OPT+数据+权重) / VQ(Count Sketch)
reports/    交付包（git 跟踪）：reports/opt/ = OPT 图/表/报告，入口 make_report.py
docs/       standard.md、weight-manifest.md、ailog/、paper/evaluation/redundancy/、old_ailog/、reference/(gitignore)
dataset_v3/ checkpoints/ logs/ temp/   运行时目录，gitignore
```

- `third_party/FwdLLM/FedML` 是 1168 个**已跟踪普通文件**（非真子模块），递归 clone 不必再拉 FedML。
- 数据：`dataset_v3/discovery_16k/{a01,a03}`（OPT）、`dataset_v3/arxiv_16k/{a01,a03}`（BERT），均 30 clients × (100 train + 16 local-test)、16k、α=0.1/0.3。

## 当前结果（截至 261020）

| 线 | 协议 | 结果 | 归档 |
|---|---|---|---|
| OPT r30 fixed | k=24、D=32、L=2、c=30、r=30、LoRA r1/α2、lr=0.30、clip=13.34、`--doc-order cycle` | full r29 31.46–31.70 vs fresh 33.665（+5.9~6.6%）；answer r29 2292–2474；CATV a03 answer −0.093 nats（p=5e-8），a01 n.s.；CATV full 略差 | `checkpoints/opt_ablation_r30_fixed/` |
| BERT r30 fixed | 6 层、L=8、D=8、lr=1.4e-3、clip=1.0、cycle、eval 500 | NLL r0 1.69–1.75 → r29 **1.548–1.589**；CATV on vs off 显著（a01 p=0.030 / a03 p=0.0025） | `checkpoints/distilbert_ablation_r30_fixed/` |
| VQ | ZOO-only、`--vq-rate` 默认 1（off） | smoke 已过；rate 0.5/0.25/0.1 的 cos≈0.63/0.48/0.31；**尚无长跑** | `temp/smoke_vq_*`（gitignore） |

关键口径 caveat：

- **CATV 的收益主要是推理期掩码效应**（OPT a03 answer 显著；a01 修复协议下 n.s.）；on 臂 r0 自带 mask，**不要用 r0 跨臂比较**，用 r29 配对。
- OPT 旧「固定前 2 篇/client」缺陷已由 `--doc-order cycle` 修复；旧 fixed 协议结果不可与本表直接混用。
- BERT `checkpoints/distilbert_ablation_r30_defect/` 仅存档、禁用于论文；`keep_block0` 修复见 `261014-111800`。
- 权重/数据 sha256 见 `docs/weight-manifest.md` §6/§7；OPT step5 predictor 与 BERT predictor 不可混用。

## 当前运行口径

BERT（`--gpu 0` on AutoDL）：

```bash
$PY src/fed/run_fed.py --arch distilbert --tag a01 --gpu 0 \
  --trainer zoo --attn sdpa_prune --dtype fp32 --sparse 0.4 --doc-order cycle \
  --pos-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/pos_embed.pt \
  --head-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/head.pt \
  --predictor checkpoints/predictor_bert/a01_warmup_400step/predictor.pth \
  --pruned-config checkpoints/predictor_bert/a01_warmup_400step/pruned_config.pth \
  --catv off --lora-r 1 --lora-alpha 2 \
  --zo-subspace layers --zo-layers 0-5 --zo-directions 8 --zo-eps 1e-3 \
  --local-steps 8 --rounds 30 --lr 1.4e-3 --delta-clip 1.0 \
  --max-clients 30 --eval-every 1 --eval-batch-size 8 --eval-max-samples 500
```

OPT（`amax` 用 `--gpu 2`；AutoDL 用 `--gpu 0`）：

```bash
$PY src/fed/run_fed.py --arch opt --tag a01 --gpu 2 \
  --trainer zoo --attn sdpa_prune --dtype fp32 --sparse 0.4 --doc-order cycle \
  --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \
  --predictor checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth \
  --pruned-config checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth \
  --catv off --lora-r 1 --lora-alpha 2 \
  --zo-subspace layers --zo-layer-rotate --zo-layer-group 24 \
  --zo-directions 32 --zo-eps 1e-3 --local-steps 2 --rounds 30 --lr 0.30 --delta-clip 13.34 \
  --max-clients 30 --eval-every 5 --save-every 1
```

## VQ（Count Sketch）上行压缩

- 入口：`src/compress/count_sketch.py`；`--vq-rate` 默认 **1.0 = 关闭**，`0<rate<1` 开启。**仅 ZOO**；`--trainer bp --vq-rate<1` 直接报错。
- **rate 口径 = 总 sketch 元素 / 活跃可训练参数**（与 `third_party/VQ/script.py` 一致），不是 per-channel。
- **channel 数 C = `local_steps * zo_directions`**（每个中心差分方向样本 1 channel；`--vq-channels` 仅诊断覆盖）。`c_dim = max(1, floor(rate * d_active / C))`；实际 `vq_actual_rate = C*c_dim/d_active` 写入 metrics。
- **每个 channel 独立 hash/sign seed，所有 client 共享同一 seed 序列**（`--vq-seed` 默认 `--seed`）；`vq_seeds.json` 落盘。不要改成 C 个 channel 完全同 hash：那会退化成一个 `c_dim` sketch，cos≈√(rate/C)。
- **clip 顺序：先按现有 `--delta-clip` 对完整 delta 做全局 L2 clip，再 VQ**；server 解压后默认 `--vq-rescale on` 缩放到 clip 后范数。`delta_norm` 仍是完整 delta 口径；方向误差看 `vq_cos_mean`。
- **fidelity**：Count Sketch 在总预算 N=rate*d 时 cos≈√(rate/(1+rate))，不是 √rate。实测（BERT d=294912、C=4）：rate 0.5→0.630、0.25→0.484、0.1→0.310；OPT d=65536、C=2、rate 0.1→0.299。rate=1 直接绕过 VQ，基线逐位不变（已验证）。
- **通信口径**：`vq_communication_bytes = clients * C * c_dim * 4`（fp32 sketch）；`communication_bytes` 保留 dense bf16 口径。注意 fp32 sketch 4B vs dense 2B ⇒ `--vq-rate 0.5` 的 `vq_ratio_bytes` = 1.0；要真实字节压缩需 bf16 sketch（暂未实现，先看 `vq_ratio_elements`）。
- metrics/config 字段：`vq_rate/vq_channels/vq_c_dim/vq_active_dim/vq_actual_rate/vq_cos_mean|min|max/vq_communication_bytes/vq_ratio_elements/vq_ratio_bytes`；`config.json` 记录 `vq_seed/vq_bytes_per_element`。
- 自检：`$PY src/compress/count_sketch.py`；smoke：

```bash
$PY src/fed/run_fed.py --arch distilbert --tag a01 --gpu 0 --trainer zoo \
  --attn sdpa_prune --dtype fp32 --sparse 0.4 --doc-order cycle \
  --pos-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/pos_embed.pt \
  --head-checkpoint checkpoints/distilbert_step1/a01_pos_head_1000step/head.pt \
  --predictor checkpoints/predictor_bert/a01_warmup_400step/predictor.pth \
  --pruned-config checkpoints/predictor_bert/a01_warmup_400step/pruned_config.pth \
  --zo-subspace layers --zo-layers 0-5 --zo-directions 2 --local-steps 2 \
  --rounds 1 --lr 1.4e-3 --delta-clip 1.0 --max-clients 2 --max-train-samples 2 \
  --truncate 2048 --eval-every 0 --save-every 0 \
  --vq-rate 0.5 --out-root temp/smoke_vq_bert
```

## 硬坑（ZOO / 精度 / 数据）

- **ZOO 必须在 `model.eval()` 下做有限差分**：train 模式 OPT dropout=0.1 会让 L+/L- 拿到不同 mask，差分被噪声吞掉（`260929-000200`）。
- **bf16 损失量化会吞掉 ZOO 信号**（~1e-6 ≪ 0.0156 台阶）；正式训练用 `--dtype fp32 --attn sdpa_prune`（FA2 只支持 bf16/fp16）。16k fp32 前向 ~2.9s、BP ~13s/样本。
- **predictor 必须加载**：OPT 用 step5（`step5_a01a03_clients_causal`），BERT 用 `predictor_bert/a01_warmup_400step`；随机 predictor 的 PPL 不能作质量依据。推理侧 predictor bias 由加载器零填充，其余缺 key 硬报错。
- `--delta-clip` 是逐 client 全局 L2 裁剪，**不是梯度裁剪**；OPT raw norm 可达 1e6，clip=13.34；BERT clip=1.0。重尾离群见 `260929-111500`。
- CATV：`r < s` 且 `s+r <= 1`（默认 `r=s/2`）；BERT 必须 `keep_block0=True`；投票用 NaN-aware valid-mean（`261014-111800`），否则 pad 占位符会让掩码冻结。
- `--arch distilbert` 只支持 `sdpa/sdpa_prune`；`--act-pack` 仅 OPT；`--lora-r 1 --lora-alpha 2` 是正式口径（默认 r=8 只用于一般实验）。
- `--doc-order cycle` 是修复固定语料的协议；新长跑必须显式加上，否则每 client 只重复前 L 篇。
- 读结果以 `metrics.jsonl` 为准（driver.log 有块缓冲）；`config.json` 落盘实际生效参数。

## AutoDL 4090 工作机（本 checkout：`/root/smell/SMELL`）

- 硬件：1×**RTX 4090 24GB**（sm_89，driver 595 / CUDA 13.2）、20 CPU、系统盘 **30G（紧张，装前 `df -h`）**。**`--gpu` 必须用 0**（`run_fed.py` 直接写 `CUDA_VISIBLE_DEVICES`，索引 1+ 看不到卡）。
- conda：`~/miniconda3`，env `SMELL`（py3.10.21 + torch 2.8.0+cu128 + flash-attn 2.8.3 + transformers 4.45.2 / peft 0.13.2 + `jenga_src.pth`）；一律用 `~/miniconda3/envs/SMELL/bin/python`。
- 数据/权重：`dataset_v3/{discovery_16k,arxiv_16k}` 已建；OPT-350M、pos_only、step5、BERT step1/predictor 已在 `checkpoints/`（sha 见 `docs/weight-manifest.md`）。长跑前确认 GPU 空闲。
- GitHub：**HTTPS 会挂死，SSH 正常**；三个 submodule 的 URL 已在本机 `.git/config` 改成 SSH。**别跑 `git submodule sync`**（会改回 HTTPS）。慢时可临时用 `https://ghfast.top/https://github.com/<owner>/<repo>.git`。

## 云端 A40 环境（host `amax`）

- 仓库：`/home/yangyongbo/projects/smell/SMELLv3`；conda 在 `~/applications/anaconda3/bin/conda`（**不是** `~/miniconda3`）。
- 硬件：4× A40 46GB（sm_86，driver 570 / CUDA 12.8）、144 CPU、251GB RAM；**约定只用 GPU2**（GPU0/1/3 常被占用），跑前 `nvidia-smi`。
- 首次就绪：`git submodule update --init --recursive`（HTTPS 挂死用 SSH）→ 解压 `~/download/{dataset.zip,peft_model.zip,predictor.zip}` 到 `third_party/Jenga/` → 建 `smell-v3` env（`setup_server.sh` 硬编码 `$HOME/miniconda3`/`SMELL`，须改 `CONDA`/`--env-name`）→ 写 `jenga_src.pth` → 从 hf-mirror 拉 opt-350m → 重建数据并跑 `check_partition.py`。
- `scripts/hello-world.sh` 也硬编码 `$HOME/miniconda3/envs/SMELL/bin/python`；云端用 `PYTHON=~/applications/anaconda3/envs/smell-v3/bin/python bash scripts/hello-world.sh` 覆盖。
- 数据/权重/checkpoint 不进 git；`pos_embed.pt`/`adapter/`/`predictor.pth` 单独传输并核对 sha256。

## 构建 / 静态检查 / 测试

**无 CI / lint / 测试框架 / formatter**，不要引入。唯一强制静态检查是逐文件 `py_compile`：

```bash
PY=~/miniconda3/envs/SMELL/bin/python   # AutoDL；amax 用 ~/applications/anaconda3/envs/smell-v3/bin/python
$PY -m py_compile $(git diff --name-only -- '*.py')
$PY src/compress/count_sketch.py         # VQ 数学/确定性/覆盖自测
$PY src/train/zoo.py                     # 前向梯度 vs autograd + FedAvg
$PY src/fed/serial_fedavg.py             # 加权聚合 / 断言 / JSONL
$PY src/models/token_selector.py         # CATV 掩码 / r<s / IR
$PY src/data/check_partition.py --root dataset_v3/arxiv_16k/a01
$PY src/fed/run_fed.py --help
bash -n scripts/setup_server.sh scripts/run_ablation_cloud.sh
```

环境自检：`PYTHON=$PY bash scripts/hello-world.sh`（35 项；`--no-gpu` 纯导入）；`bash scripts/setup_server.sh --check`；`bash scripts/run_ablation_cloud.sh --dry-run`。

## 代码风格

- **Python 3.10，无类型注解**；PEP 8、4 空格、双引号。stdlib → third-party → 本地；重依赖在使用处导入；本地一律绝对导入 `from src.foo import bar`，入口先 `REPO = Path(__file__).resolve().parents[N]`。
- `snake_case` / `PascalCase` / `UPPER_CASE` / `_prefix`；路径用 `pathlib.Path`。
- 不变量 `assert cond, f"...got {x}"`；CLI 错误 `raise SystemExit(...)`；库参数错误 `ValueError`；**禁止裸 except**。
- 每个入口 `parse_args()` + `main()`；实验输出 `config.json` + `metrics.jsonl`（`append_metrics`）。
- **默认不写注释**；所有既有代码改动带 `# SMELL N <组件> <动词> — 说明`（BEGIN/END 成对，动词见 `docs/standard.md`）；新文件头 `# SMELL N <name> NEW/COPY — ...`。不删旧标注。
- **不修改 `third_party/`**；上游 bug 在 `src/` 内 monkeypatch/自研规避（如 Jenga HEAD_DIM=128、act-pack hooks）。

## 长任务与 GPU 约定

- >1 min 的任务后台化：写 `temp/run_<name>_bg.sh`（gitignore），`setsid nohup bash temp/run_<name>_bg.sh >/dev/null 2>&1 < /dev/null &`。
- 每阶段打点 `[HH:MM:SS] STEP ...`；日志 `temp/logs/<name>_YYMMDD-HHMMSS/driver.log`；留 `last_<name>.pid` / `last_<name>_dir.txt`。
- 启动 30s 后查日志/进程/`nvidia-smi`；失败当场修。完成后由用户在聊天唤醒 agent，按 `last_*_dir.txt` 读日志、校验产物、汇报。
- 4 卡并发用 `bash scripts/run_ablation_cloud.sh`（GPU0-3 = a01 off/on、a03 off/on）；跑前确认目标卡空闲。

## OPT / v2 高价值教训

- OPT `lm_head` 与词嵌入**绑定**（`_tied_weights_keys`），head 用 `word_embed_proj_dim`（≠hidden_size）；v2 的「lm_head LoRA」假设不成立。
- OPT 学习式绝对位置嵌入零样本无法外推（2k PPL 20.2 → 16k 588）；必须位置适配（`pos_only` 主用）。
- v2 Server 聚合 = `Σg → 全局 L2 归一化 → clamp → θ-=lr·ĝ` ⇒ ‖Δθ‖≡lr；v3 是**加权平均**，v2 lr 结论不迁移。
- Server/Client 参数枚举顺序与名字翻译必须严格一致，否则聚合**静默失效**（v2 M1 bug）。
- ZOO 中心差分 `grad≈(L(θ+hv)−L(θ−hv))/(2h)·v`；有效杠杆是样本数 n/轮数，不是扰动数 M；`var_threshold` 是绝对阈值，缩放改动会让它静默失效。

## Git / submodule

- 当前 VQ 分支 `exp/vq-compression`；主实验线 `exp/distilbert-16k` 已合入本地 `main`（`37e5708`）。提交信息 `SMELL v3: <English summary>`；不要 amend 已 push 的提交；提交前跑相关 `py_compile`。
- 新机 git 身份：`git config user.name/user.email` 与历史作者一致（`git log -1 --format='%an <%ae>'`）。
- 三个 submodule 只读；上游改动 fetch/checkout 后 `git add third_party/<x>` 更新指针，**不直接改文件**。VQ 发布 remote `github`（`2069661643/SMELL-VQ`），开发上游 `../VQ`；`git submodule sync` 会改写 origin，慎用。
- 每次改动写 `docs/ailog/YYMMDD-HHMMSS-<desc>.md`（模板见 `docs/standard.md`）。

## 必读 ailog

| 文件 | 内容 |
|---|---|
| `docs/weight-manifest.md`（非 ailog） | **换机交接**：权重/数据路径 + sha256 + OPT/BERT 运行 flags |
| `docs/ailog/261020-090000-...opt-r30-fixed-results-catv-mask-alpha.md` | **OPT r30 fixed 四臂 + r0 补测**：CATV 为掩码主导、α 依赖 |
| `docs/ailog/261015-213000-...bert-r30-fixed-four-arm-results.md` | **BERT r30 fixed 四臂**：CATV 显著；weight/mask 分解 |
| `docs/ailog/261014-111800-...bert-catv-padding-vote-artifact-fix-handoff.md` | BERT CATV padding 伪影 + valid-mean/block0 修复 |
| `docs/ailog/261012-151805-...bert-fixed-60docs-protocol-flaw-and-ld-plan-handoff.md` | 固定 60 篇协议缺陷 + `doc-order cycle` |
| `docs/ailog/261012-093000-...a03-r74-catv-parity-answer-dynamics.md` | a03 r74 CATV 收敛/answer 动态 |
| `docs/ailog/261007-005000-...a01-d32-catv-tables.md` | a01 D32 CATV 曲线与起点/掩码分解 |
| `docs/ailog/261001-034500-...k24-coverage-first-result.md` | k24 覆盖度首个正结果 |
| `docs/ailog/260930-104500-...step123-selection-drift-refresh-e2e3.md` | 选块漂移小 / predictor refresh 未闭环 / E2-E3 无翻转 |
| `docs/ailog/260929-000200-...audit-fixes-step5-predictor.md` | **审计修复（必读）**：ZOO dropout / 评测随机 predictor / bias 零填充 |
| `docs/ailog/260929-111500-...td3-outlier-explosion-delta-clip.md` | 离群爆炸 + delta-clip 根因 |
| `docs/ailog/260928-110233-...predictor-rootcause-noncausal-target-causal-fix.md` | predictor causal 修复（step4/5 基础） |
| `docs/ailog/260927-121756-...tc-fp32-cos-validated.md` | fp32 下 `cos≈0.5√(cLD/d)`，ZOO 打通 |
| `docs/ailog/260923-210703-...discovery-16k-fedavg-plan.md` | 实验方案权威依据（数据/口径/协议） |
| `docs/old_ailog/ailog/260915-163135-...b1-verdict-and-matched-bp-arm.md` | v2 ZOO vs BP 定论、n/lr 杠杆、~38 PPL 平台 |
