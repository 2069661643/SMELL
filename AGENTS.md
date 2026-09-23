# AGENTS.md — SMELL-v3（当前工作区）

## 项目与方向

SMELL = Jenga token 级稀疏 × FwdLLM 前向梯度（ZOO）→ 联邦端侧长上下文 LLM 微调；目标 MobiCom 2027 / ICDE 2027。

- **v3 取代 v2（`../SMELL/`，Llama2-7B 时代）**，核心变化：**改用 OPT-350M（`facebook/opt-350m`）**。原因：Llama2-7B 时间开销不可接受 —— v2 ZOO（M_tot=1500）单轮 ≈42 min（训练 2365 s + 3 次 eval），100 轮 ≈69 h；B1 4 轮 5.5 h；估算 50 轮 ZOO ≈135 h（同轮数 BP 臂仅 2–3 h）。
- Jenga 原生支持 OPT：`third_party/Jenga/src/jenga/utils/config_utils.py` 的 `get_opt_baseline/get_opt_qk/get_opt_llora` 默认就是 `facebook/opt-350m`；实现见 `third_party/Jenga/src/jenga/models/modeling_opt*.py`。
- 历史文档（必读）：`docs/old_ailog/AGENTS.md`（v2 权威 brief，但描述老目录结构）、`docs/old_ailog/ailog/`（26 篇 v2 变更/实验记录）、`docs/old_ailog/tasks/`；现行规范 `docs/standard.md`。
- v2 引用的 `docs/summary.md`、`docs/SMELL/phaseN.md` **不在本目录**；phase1–3 设计在父仓库 `../docs/SMELL/phase{1,2,3}.md`，后续结论（Phase 4–6、B1）只存在于 `docs/old_ailog/ailog/`。

## 目录布局

```
src/                 SMELL 包：所有新代码写这里（third_party 不改）
scripts/             运行脚本
third_party/         只读 git submodule
  FwdLLM/            UbiquitousLearning/FwdLLM @36ecdbc（FedML 已 vendored 为普通跟踪文件）
  Jenga/             Pairshoe/Jenga-AE @23764a6（ATC'25 artifact，含 OPT 模型与 predictor）
docs/                standard.md、ailog/（新日志）、old_ailog/（v2 历史）、reference/（论文，gitignore）
checkpoints/ dataset/ logs/ temp/   运行时目录，已 gitignore
```

- `third_party/FwdLLM/FedML` 虽然被 `.gitmodules` 声明为子模块，实际是 **1168 个已跟踪普通文件**；递归 clone 无需再 `submodule init` 拉 FedML。
- v2 可运行参考代码在 `../SMELL/FwdLLM/`，但它**只到 Phase 3**；B1/BP/Phase 4–6 的修复（`5ee5186` 等 commit）**不在本地任何 checkout** —— 代码与 ailog 冲突时以 `docs/old_ailog/ailog/` 为准。

## Git / submodule

- SMELL-v3 是独立 git 仓库（`main`，2026-09-23 初始化），父仓库 FedLLM 不跟踪它。
- 两个 submodule 已固定（git dir 在 `.git/modules/`）；clone 后 `git submodule update --init --recursive`。
- **不要直接改 third_party 里的文件**（会产生脏指针）。需要上游改动时 fork/branch 后更新指针：
  `git -C third_party/FwdLLM fetch; git -C third_party/FwdLLM checkout <commit>; git add third_party/FwdLLM; git commit`
- 提交信息：`SMELL v3: <English summary>`（沿用旧规范 `SMELL Phase N: ...` 的风格）；不要 amend 已 push 的提交。
- 无 CI / lint / 测试框架；唯一静态检查 `python -m py_compile <file>`；改完必须记 `docs/ailog/YYMMDD-HHMMSS-<desc>.md`，代码改动用 `# SMELL N <组件> <动词> — 说明` 标注（格式见 `docs/standard.md`）。

## 环境

- 用 **Jenga 技术栈**：Python 3.10 · torch 2.1.2 · transformers 4.45.2 · peft · bf16 + flash-attn。**不要**用 FwdLLM 的 Python 3.7 / functorch 栈。
- v2 conda 环境名 `smell-v2`；v3 尚无 requirements.txt（2026-09-23）。
- 权重：`third_party/Jenga/checkpoints/opt-350m/`（来自 `facebook/opt-350m`）+ `predictor/`、`peft_model/`；压缩包在父仓库 `../download/`（`dataset.zip`、`peft_model.zip`、`predictor.zip`）。

## OPT-350M 迁移要点（v2 代码是 Llama 专用）

- 在 `src/` 新建 OPT 版模型/训练代码；v2 的 `modeling_llama_base.py` / `modeling_llama_sparse.py` / `predictor.py` 不能直接复用，Jenga 的 `modeling_opt.py`（`OPTForCausalLM`）是主要参考。上游 FwdLLM 只支持 DistilBERT（`initializer.py` 的 `MODEL_CLASSES`），ZOO/JVP 与联邦管线要自己接。
- **OPT 的 lm_head 与词嵌入绑定**：Jenga `modeling_opt.py:943` 有 `_tied_weights_keys = ["lm_head.weight"]`，且 head 用 `config.word_embed_proj_dim`（opt-350m 上 ≠ hidden_size，经 project_in/out）。v2 的 D3「lm_head LoRA」方案假设 head 独立，移植前先验证绑定/投影的影响。
- v2 实测：序列 pad 到 `--model_max_length`（4096）而有效 token 仅 ~17，99.7% 算力浪费（attention mask 只影响 loss，前向仍全长）；OPT-350M 实验要控制实际序列长度。

## v2 高价值教训（重踩会浪费数小时）

- Server 聚合 = `Σg → 全局 L2 归一化 → clamp(-1,1) → θ -= lr·ĝ` ⇒ **‖Δθ‖ ≡ lr**，与梯度尺度/样本数无关；调 lr 就是调位移。
- Server/Client 参数枚举顺序与名字翻译必须严格一致，否则聚合**静默失效**（v2 M1 bug，修复为 `fed_trainer_transformer.py::_translate_lora_param_names`；该改动不在本地任何 checkout，需在 `src/` 重实现时照做）。
- ZOO 中心差分 `grad ≈ (L(θ+hv) − L(θ−hv))/(2h)·v`；v2 实测 cos(ZOO, BP) ≈ 0.2–0.28（≈√(M/d)），n=3 时留出改善比 BP 差 ~80×；有效杠杆是每次更新的样本数 n 与轮数，不是扰动数 M。
- `var_threshold` 是绝对阈值，任何梯度缩放改动（归一化、重参数化）都会让它静默失效。
- 联邦 MPI 实际只能 `-np 2`（1 个 worker 模拟多 client）；多 worker 有 var-overwrite bug；`_sum_q_acc` 跨 data_id/round 从不清零（共识率指标不可信）。
- client idle-timeout 计时基准必须在 `notify()` 返回后重置，否则长轮误判自杀（见 old_ailog 260918）。

## 必读 ailog

| 文件 | 内容 |
|---|---|
| `docs/old_ailog/ailog/260915-163135-...b1-verdict-and-matched-bp-arm.md` | ZOO vs BP 定论、n/lr 杠杆、~38 PPL 平台 |
| `docs/old_ailog/ailog/260916-165417-...fwdllm-upstream-and-zo-literature.md` | 上游 FwdLLM 实证核查 + ZO 文献版图（差异化定位） |
| `docs/old_ailog/ailog/260912-192014-...d3-lmhead-lora.md` | D3 lm_head-LoRA 方案与撞墙 |
| `docs/old_ailog/ailog/260918-095300-...idle-timeout-false-abort.md` | 看门狗 bug +「进程没了/日志停更」排查配方 |
