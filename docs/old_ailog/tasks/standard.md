# SMELL 开发规范与约定

> 本文档定义 SMELL 项目的开发规范：代码注释风格、Ailog 变更日志、Git 提交、Phase 设计文档格式。
> 所有 SMELL 新功能开发必须遵循这些约定。

---

## 1. 代码注释 — SMELL 标注规范

所有代码修改使用 `# SMELL N ...` 标记，追踪每行代码由哪个 Phase 引入。

### 1.1 标注动词

| 动词 | 用途 | 示例 |
|---|---|---|
| `BEGIN` / `END` | 包裹新增代码块（成对使用） | `# SMELL 2 LlamaFlashAttention2 BEGIN` ... `# SMELL 2 LlamaFlashAttention2 END` |
| `ADD` | 新增单行/少量代码 | `# SMELL 3 consensus_mask ADD — 在 topk 前注入共识掩码` |
| `COMMENTED` | 原有代码被注释 | `# SMELL 2 PrunedMLP COMMENTED — Phase 2 不使用` |
| `REMOVED` | 原有代码被删除 | `# SMELL 2 prune_ratio REMOVED — Phase 2 不使用 PrunedMLP` |
| `MODIFIED` | 原有代码被修改 | `# SMELL 2 imports MODIFIED — 本地 import 替代 jenga 路径` |
| `COPY` | 文件从外部复制（文件头部用） | `# SMELL 2 predictor COPY — 从 Jenja/src/jenga/models/predictor.py 复制` |
| `NEW` | 全新文件（文件头部用） | `# SMELL 3 h5py preprocessor NEW — Phase 1/2 causal LM 的 h5py 数据预处理` |
| `REWRITTEN` | 文件被重写（文件头部用） | `# SMELL 3 smell_main REWRITTEN — Phase 1/2 h5py DataManager 集成` |
| `FIXED` | Bug 修复 | `# SMELL 3 FedSGDAggregator FIXED — 正确的 12 参数构造函数` |

### 1.2 格式

```python
# SMELL N <组件名> <动词> — <简要说明>
```

- `N`：Phase 编号（1/2/3/...）
- `<组件名>`：功能或模块名（英文或中文均可，与前后代码一致）
- 描述用中文或英文均可，要求简洁（尽量一行内）
- `BEGIN`/`END` 必须成对使用，标记新增代码块的开始和结束

### 1.3 完整示例

```python
# 文件头部标注来源（选其一）
# SMELL 2 modeling_llama_sparse COPY — 从 Jenja/src/jenga/models/modeling_llama.py 复制
# SMELL 3 DataManager NEW — 全新的 h5py DataManager 类

# 修改现有代码
# SMELL 1 causal_lm MODEL_CLASSES ADD — Phase 1 Llama2-7B causal LM
# SMELL 2 llama_sparse MODEL_CLASSES ADD — Token Elimination 稀疏化版本

# 注释掉的旧代码
# SMELL 1 functorch COMMENTED: torch 2.1.2 内置 torch.func，不再需要 functorch
# import functorch

# 新增代码块（BEGIN/END 包裹）
# SMELL 1 ForwardCausalLMTrainer BEGIN — Phase 1 新增
class ForwardCausalLMTrainer:
    """Forward gradient (ZOO) causal LM trainer."""
    def __init__(self, ...):
        ...
    def train_model(self, ...):
        ...
# SMELL 1 ForwardCausalLMTrainer END

# 删除代码（原地留注释说明删除原因）
# SMELL 2 prune_ratio REMOVED — Phase 2 不使用 PrunedMLP
```

### 1.4 注意事项

- **删除 = 注释原代码**，使用 `COMMENTED` 标记，保留原代码以备回溯
- 不要删除旧标注（它们记录代码演进历史）
- 新增 Phase 的修改在原标注处**追加**新标注，不覆盖旧标注
- 标注应精确定位到被修改的代码行/块，不要标注在无关位置
- 评论代码中使用 `—` 作为分隔符，描述用中文或英文均可

---

## 2. Ailog — 变更日志

### 2.1 目录与命名

- 存放目录：`docs/ailog/`
- 文件命名：`YYMMDD-HHMMSS-<description>.md`
  - `YYMMDD`：日期（如 `260804` = 2026-08-04）
  - `HHMMSS`：时间（如 `175011` = 17:50:11）
  - `<description>`：简短英文描述，以 `smell-` 前缀开头
  - 示例：`260804-175011-smell-phase2-token-elimination.md`
- 标题格式：`# Ailog YYMMDD-HHMMSS — Phase N: 简要描述`

### 2.2 记录内容

每次开发会话（至多一个 Phase 的完整变更）一条 Ailog。每条变更记录包含：

| 字段 | 说明 | 示例 |
|---|---|---|
| 时间戳 | `HH:MM` 格式，记录每条变更的具体时刻 | `14:30` |
| 操作 | 概括性动词短语（新增/修改/删除/修复） | `新增 ForwardCausalLMTrainer` / `修改 initializer.py` / `修复 grad list 格式` |
| 文件 | 涉及的具体文件路径 | `FwdLLM/forward_training/tc_transformer_trainer_distribute.py` |
| 位置 | 写入位置（行号/函数名） | `L42-230` / `ForwardCausalLMTrainer.__init__` |
| 影响 | 简述影响范围 | `替换旧 ForwardTextClassificationTrainer` |

### 2.3 模板

```markdown
# Ailog YYMMDD-HHMMSS — Phase N: 简要描述

## 变更列表

| 时间 | 操作 | 文件 | 位置 | 影响 |
|---|---|---|---|---|
| 14:30 | 新增 ForwardCausalLMTrainer | tc_transformer_trainer_distribute.py | L42-230 | 核心 ZOO 训练器 |
| 14:45 | 修改 create_model() | initializer.py | L85-130 | 新增 llama_sparse 分支 |
| 15:00 | 修改 layer_id_for_check | tc_transformer_trainer_distribute.py | L90 | 16 → 14（最后一个非稀疏层） |

## 设计决策（可选）

- 为什么选择方案 A 而非方案 B
- 已知限制和边界情况

## 待办（可选）

- 后续需要跟进的项
```

---

## 3. Git 提交规范

### 3.1 格式

```
SMELL Phase N: <English summary>
```

- 必须包含 `SMELL` 前缀（区分原始 FwdLLM 代码）
- 必须标注 `Phase N`
- 摘要用英文
- 一行即可（不需要 body），如有必要可在第二段补充细节

### 3.2 示例

```
SMELL Phase 1: Llama2-7B causal LM replaces DistilBERT classification
SMELL Phase 2: Token Elimination sparsity integration
SMELL Phase 1/2: h5py DataManager integration
SMELL Phase 3: Consensus Anchor Tokens voting
```

### 3.3 规则

- **一个 Phase 可以多 commit**（如 Phase 1 的模型替换 + h5py 集成是分开提交的）
- **跨 Phase 的修复**标注涉及的 Phase（如 `Phase 1/2`）
- **不要 amend 已 push 的 commit**
- **提交前至少跑完所有相关文件的 `py_compile`**
- **不要提交 secrets、checkpoints、dataset 到 git**（.gitignore 已配置 `checkpoints/`、`dataset/`）

---

## 4. Phase.md — 设计文档

### 4.1 存放位置

`docs/SMELL/phaseN.md`（N = 1, 2, 3, ...）

### 4.2 必需章节

| 章节 | 说明 | 必需？ |
|---|---|---|
| `# SMELL Phase N: Title` | 标题 | ✅ |
| `> 更新日期` 标注 | 如有后续修改，标注更新日期 | 可选 |
| `## 概述` | 本 Phase 的目标和范围（3-5 句话） | ✅ |
| `## 环境要求` | 依赖版本、前置条件 | 有依赖变化时必填 |
| `## 文件清单` | 新建文件表 + 修改文件表 | ✅ |
| `## 关键设计决策` | 每个决策一个编号小节，含代码片段 | ✅ |
| `## 精确修改表` | 每个修改文件的行号→操作→内容对照 | 可选（复杂 Phase 必填） |
| `## 使用方式` | 启动命令、CLI 参数 | 可选 |
| `## 静态检查` | py_compile 结果 + 各文件状态 | ✅ |
| `## 与 Phase N 的对比` | 与前一 Phase 的差异表 | 可选 |
| `## 未完成（后续 Phase）` | 本 Phase 未完成、留待后续的事项 | ✅ |

### 4.3 文件清单格式

```markdown
### 新建 (N)

| # | 文件 | 行数 | 作用 |
|---|---|---|---|
| N1 | `FwdLLM/model/predictor.py` | 351 | PrunableAttnPredictorInfer |

### 修改 (M)

| # | 文件 | 修改内容 |
|---|---|---|
| M1 | `FwdLLM/.../initializer.py` | MODEL_CLASSES 新增 llama_sparse; --sparse CLI arg |
```

### 4.4 关键设计决策格式

```markdown
### 1. 决策标题

简短说明为什么做这个决策（1-2 句话）。

```python
# 关键代码片段（如有）
```

额外解释（影响范围、与其他组件的交互等）。
```

---

## 5. 检查清单（开发完成前）

- [ ] 所有 Python 文件通过 `py_compile` 静态检查
- [ ] 所有代码修改均有 `# SMELL N ...` 标注
- [ ] 新建文件头部有 `# SMELL N ... COPY/NEW` 标注
- [ ] 更新 `docs/SMELL/phaseN.md`（如有新 Phase）
- [ ] 记录 Ailog 到 `docs/ailog/YYMMDD-HHMMSS-<description>.md`
- [ ] Git commit message 符合 `SMELL Phase N: ...` 格式
- [ ] 未提交 checkpoints/dataset/secrets 到 git
- [ ] 更新 AGENTS.md（如有新人或新规范）
