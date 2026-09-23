# AGENTS.md — SMELL 项目

## 概述

SMELL（Sparsity-aware Memory Efficient Federated On-device Fine-tuning for Long-context LLMs）将 **Jenga token 级稀疏化** 与 **FwdLLM 前向梯度（ZOO）** 结合，对 Llama2-7B 做联邦端侧长上下文微调。

- **目标**：MobiCom 2027 / ICDE 2027
- **仓库根**：`FwdLLM/`（原 FwdLLM 代码，大量改造）；`src/jenga/` 为移植参考的 Jenga 源码
- **权威现状**：`docs/summary.md`（含当前瓶颈 B1/B2/B3，改动前必读）；阶段设计 `docs/SMELL/phaseN.md`；规范 `docs/tasks/standard.md`

**技术栈**：Python 3.10 · torch 2.1.2 · transformers 4.45.2 · peft LoRA (r=8, q/k/v/o, ~8.4M 参数) · bf16 + flash_attention_2 · h5py 数据 · FedML/MPI。

---

## 构建 / 检查 / 测试命令

### 环境

```bash
conda create --name smell-v2 python=3.10 && conda activate smell-v2
pip install -r requirements.txt                  # SMELL 核心依赖（根目录，pipreqs 整理）
pip install -r FwdLLM/requirements.txt           # FwdLLM 全量依赖
pip install flash-attn --no-build-isolation      # 需 CUDA toolkit
```

> 环境名以运行脚本为准：当前脚本用 `smell-v2`（个别旧脚本仍写 `smell_env`/`smell-v1`）。

### 单文件 / 单测（唯一的静态验证方式）

本项目**没有正式测试框架**。验证单个文件改动用 `py_compile`：

```bash
# 单个文件的静态语法检查
python -m py_compile FwdLLM/<path>/file.py

# 改动后一次性检查所有 SMELL 核心模块
for f in FwdLLM/model/modeling_llama_sparse.py FwdLLM/model/predictor.py \
         FwdLLM/forward_training/tc_transformer_trainer_distribute.py \
         FwdLLM/forward_training/utils/fwdgrad_utils.py \
         FwdLLM/experiments/distributed/transformer_exps/initializer.py \
         FwdLLM/experiments/distributed/transformer_exps/run_lm_exps/smell_main.py \
         FwdLLM/data_manager/causal_lm_data_manager.py \
         FwdLLM/data_preprocessing/causal_lm_preprocessor.py \
         FwdLLM/FedML/fedml_api/distributed/fedsgd/*.py; do python -m py_compile "$f" || exit 1; done
```

现有 ad-hoc 单测脚本直接运行（从 `FwdLLM/` 目录，脚本内已处理 sys.path）：

```bash
python FwdLLM/data/raw_data_loader/test/test_rawdataloader.py   # 数据加载器
python FwdLLM/test_dm.py                                        # CausalLMDataManager 冒烟
python check_token_ids.py --data_file dataset/goemotions_data.h5  # 校验 token id 词表范围
```

新测试用 `pytest`（`test_*.py` 放 `tests/`）；单测：`pytest tests/test_x.py::test_name -q`。

### 运行实验（MPI，单机）

数据文件与 predictor checkpoint 必须存在：`dataset/goemotions_{data,partition}.h5`（`FwdLLM/dataset/` 或脚本内绝对路径）、`checkpoints/predictor/{predictor.pth,pruned_config.pth}`（Phase 2/3）。**注意 `-np 2` 固定（1 个 MPI worker 模拟多个逻辑 client；多 worker 有 var-overwrite bug）。**

```bash
# 冒烟测试（从仓库根，推荐首选）
bash script/RUNME-SmokeTest-Phase0.sh   # 数据预处理
bash script/RUNME-SmokeTest-Phase1.sh   # Llama + LoRA + Forward Gradient
bash script/RUNME-SmokeTest-Phase2.sh   # Token Elimination
bash script/RUNME-SmokeTest-Phase3.sh   # Consensus Anchor Tokens

# 当前调试/诊断脚本（从仓库根，会自动 cd FwdLLM；Phase 4/5 使用）
bash script/RUNME-v2-len4k-sparse0.4.sh      # LoRA r=1, len4k
bash script/RUNME-v2-len4k-d3-lmhead.sh      # D3: 只训 lm_head LoRA (Phase 5)

# 正式实验（从 FwdLLM/ 目录，路径参数在脚本内）
bash experiments/distributed/transformer_exps/run_lm_exps/run_phase1_baseline.sh
bash experiments/distributed/transformer_exps/run_lm_exps/run_phase2_sparse.sh
bash experiments/distributed/transformer_exps/run_lm_exps/run_phase3_consensus.sh
```

### 评估 / 工具脚本

```bash
# 计算 PPL（读 global_lora_final.h5）
python experiments/distributed/transformer_exps/run_lm_exps/eval_ppl.py \
  --checkpoint <global_lora_final.h5> --eval_data_file <goemotions_data.h5>

# 共识 top-k 交集率（读 consensus_history.h5）
python experiments/distributed/transformer_exps/run_lm_exps/eval_top_k.py --history <consensus_history.h5>

# 检查 LoRA 权重是否含 NaN/Inf
python experiments/distributed/transformer_exps/run_lm_exps/check.py --checkpoint <global_lora_final.h5>
```

无 `.flake8` / `pyproject.toml` / `.pre-commit-config.yaml`，也无 Cursor（`.cursorrules`/`.cursor/rules/`）或 Copilot（`.github/copilot-instructions.md`）规则，风格请手动遵循下文。

---

## 代码风格

### 目录结构

```
FwdLLM/model/            Llama 建模（base=基线 / sparse=稀疏 / predictor）
FwdLLM/forward_training/ ZOO 训练器 + JVP 工具
FwdLLM/data_manager/     h5py 数据管理（fedavg 风格）
FwdLLM/FedML/            FedML fork（fedml_api/distributed/fedsgd/）
FwdLLM/experiments/      入口、run 脚本、eval 脚本
FwdLLM/evaluation/       LongBench 评估（后续）
FwdLLM/training/         fed_trainer_transformer.py（FedML ↔ 模型训练器桥接，含 LoRA key 翻译）
script/                  冒烟测试脚本 + 数据预处理
src/jenga/               Jenga 参考源码（只读参考，勿改）
```

模块导入是扁平的：`from model.modeling_llama_sparse import ...`、`from forward_training.utils.fwdgrad_utils import ...`。

### 导入与格式化

- 顺序：**stdlib → 第三方 → 本地项目**，4 空格缩进，行宽 ~120，双引号，UTF-8。
- 统一 `import numpy as np`、`import torch.nn.functional as F`；避免 `from X import *`。
- FedML 双路径导入（框架要求）：

```python
try:
    from fedml_core.distributed.client.client_manager import ClientManager
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
```

- 部分模块在函数内惰性导入（如 `initializer.py` 的 `create_model()` 内 import LlamaConfig）。

### 命名

| 类别 | 约定 | 示例 |
|------|------|------|
| 类 | PascalCase | `ForwardCausalLMTrainer`, `FedSgdAggregator` |
| 函数/变量 | snake_case | `calculate_jvp()`, `train_dl`, `num_clients` |
| 常量 | UPPER_SNAKE | `MSG_TYPE_C2S_SEND_VOTE`, `IGNORE_INDEX` |
| 私有成员 | `_prefix` | `_param_to_idx`, `_sum_q_acc` |
| 文件 | snake_case | `causal_lm_data_manager.py` |

### 类型标注与 Docstring

- 类型标注可选、公共接口鼓励；用 Python 3.10 `typing`（`Callable`/`Dict`/`Optional`/`Tuple`…），参考 `fwdgrad_utils.py`。
- 公共类/函数用 **Google 风格** docstring（`Args:`/`Returns:`）；类 docstring 说明用途与关键设计决策。

### 错误处理

- 不用自定义异常，用内建（`ValueError`/`RuntimeError`）。
- 训练核心代码让异常自然传播，勿加 try/except（除非有明确恢复策略）；仅框架级导入用 FedML `try/except ImportError`。

---

## SMELL 代码标注规范（必须遵守）

所有改动加 `# SMELL N <组件名> <动词> — <说明>` 追踪来源，`N` 为 Phase 编号。动词：

- `BEGIN`/`END`（成对，包裹新增代码块）、`ADD`、`COMMENTED`（注释掉的旧代码）、`REMOVED`、`MODIFIED`、`COPY`/`NEW`/`REWRITTEN`（文件头标注来源）、`FIXED`。

```python
# SMELL 2 LlamaFlashAttention2 BEGIN
class LlamaFlashAttention2(nn.Module): ...
# SMELL 2 LlamaFlashAttention2 END
# SMELL 3 consensus_mask injection ADD
# SMELL 1 functorch COMMENTED: torch 2.1.2 内置 torch.func
```

**规则**：删除=注释原代码（保留回溯）；不删旧标注；新 Phase 在原处**追加**不覆盖；标注精确定位到改动行。

### Git / Ailog / Phase.md 规范

- **Git commit**：`SMELL Phase N: <英文摘要>`（如 `SMELL Phase 1/2: h5py DataManager integration`）。提交前跑完相关文件 `py_compile`；不提交 checkpoints/dataset/secrets（`.gitignore` 已配置）。
- **Ailog**：每次会话一条，`docs/ailog/YYMMDD-HHMMSS-<desc>.md`（变更表：时间/操作/文件/位置/影响）。
- **Phase.md**：`docs/SMELL/phaseN.md`，必需章节见 `docs/tasks/standard.md` §4。

---

## 关键架构细节

- **模型**：Llama2-7B + peft LoRA，秩/目标/α 可配置（`--lora_r/--lora_target_modules/--lora_alpha`，默认 r=8 q/k/v/o ≈8.4M；r=1 时 ≈1.05M）；bf16 + flash_attention_2。
- **D3 lm_head LoRA（Phase 5）**：`--lora_lm_head/--lora_lm_head_r/--lora_lm_head_alpha` 冻结 backbone、只训 `lm_head` 低秩增量（r=4 → d=144,384；r=1 → 36,096）。Server `LoRAOnlyModel` 含 lm_head 分支，`_translate_lora_param_names` 增加 `lora_{A,B}_lm_head` 翻译，`var_control` 检查目标改为 `lm_head.lora_A`。**新增任何可训参数方案时，务必保持 Server 参数枚举顺序/形状与 Client 一致并扩展翻译表，否则聚合静默失效。**
- **梯度**：前向梯度 ZOO 中心差分 `grad ≈ (L(θ+hv) − L(θ−hv))/(2h) · v`。
- **训练**：`torch.func.functional_call` 替换 functorch；`torch.no_grad()` 最外层包裹，无 autograd 图。
- **聚合**：`FedSgdAggregator.aggregate` 对 `trainable_params` 做全局 L2 归一化 + `clamp(-1,1)`；步长 `--lr_decay` 时线性 0.01→1e-5，否则 `server_lr*ratio`。
- **稀疏**：层 0-14 全保留、15-30 topk(sparse=0.4)、31 跳过；Predictor 对 64-token block 打分。注意 sparse 模型只支持 batch=1。
- **共识**：`--enable_consensus` 门控（默认 False）；client 发 `sum_q`（type 9 消息）→ server 累计阈值化 → `±inf/0` mask 下发。Phase 1/2 不受影响。
- **通信**：FedML MPI，`-np 2` 单 worker。梯度以 list 存（`self.grad = [None] * N`）兼容 FedSGD。
- **数据**：h5py 预 tokenize uint16 数组（`CausalLMDataManager`）；GoEmotions 有 1k/4k 变体（`dataset/goemotions_4k_{data,partition}.h5`），seq 可 1024/4096。

## 当前状态与已知问题（详见 docs/summary.md，最新以 docs/ailog/ 为准）

- **B1 训练尚未真正收敛（最高优先）**：根因已钉死为**前向梯度信噪比过低** `SNR=cosθ=√(M/d)`——train/test PPL 同时平躺、25 轮斜率仅 -0.006/round，排除过拟合。M1（Server→Client LoRA key 翻译）**已修复且经 D3 诊断验证生效**（`[DBG-SENDPARAM]`/`[DBG-SETPARAM]` checksum 一致、`matched=2`、client 参数每轮变化），**瓶颈不在同步链路**。M2 步长现改为梯度归一化+clamp+`--lr_decay`，已无 NaN 但仍不学习。降噪 `var_control/MORE_V` 效果有限且 `var_threthod` 为绝对阈值（缩放改动会使其静默失效，方案 B 已回退）。
- **B2 共识率指标不可信**：`_sum_q_acc` 跨 data_id/round 从不清零；`-np 2` 下仅一个 `client_0000`，`eval_top_k.py` 无 pair 可算。
- **B3 工程限制**：多 worker MPI 有 var-overwrite bug；无正式测试框架（仅 `py_compile`）；sparse 模型只支持 batch=1；部分文档行号/参数规模已过期。

## 文件参考

| 文件 | 作用 |
|------|------|
| `FwdLLM/forward_training/tc_transformer_trainer_distribute.py` | 核心 ZOO 训练器 |
| `FwdLLM/forward_training/utils/fwdgrad_utils.py` | JVP 计算、functional_call |
| `FwdLLM/model/modeling_llama_sparse.py` | Token Elimination 模型 |
| `FwdLLM/model/modeling_llama_base.py` | 基线 Llama（Phase 1） |
| `FwdLLM/model/predictor.py` | PrunableAttnPredictorInfer |
| `FwdLLM/experiments/.../initializer.py` | CLI 参数、模型创建、PEFT |
| `FwdLLM/experiments/.../run_lm_exps/smell_main.py` | 入口（所有 phase） |
| `FwdLLM/data_manager/causal_lm_data_manager.py` | h5py DataManager |
| `FwdLLM/FedML/.../fedsgd/FedSgd{Aggregator,ClientManager,ServerManager,Trainer}.py` | 联邦聚合/通信（含 consensus 投票） |
| `FwdLLM/FedML/.../fedsgd/message_define.py` | type 9 消息定义 |
| `FwdLLM/training/fed_trainer_transformer.py` | LoRA key 翻译（M1 修复处） |
| `FwdLLM/experiments/.../run_lm_exps/{eval_ppl,eval_top_k,eval_train_vs_test,check}.py` | 评估/校验工具 |
| `FwdLLM/FedML/.../fedsgd/FedSgdTrainer.py` | client 组装/发送 `model_trainer.grad` |
| `script/RUNME-v2-len4k-{sparse0.4,d3-lmhead}.sh` | Phase 4/5 调试/诊断脚本 |
| `docs/summary.md` | 项目现状与瓶颈（权威来源） |
| `docs/SMELL/phase5_task_head_plan.md` | Phase 5 降 d 方案对比（A1/C1/D3…） |
