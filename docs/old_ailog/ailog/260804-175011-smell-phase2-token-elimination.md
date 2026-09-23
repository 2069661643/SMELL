# SMELL Phase 2 — Token Elimination 稀疏化集成

## 概述

在 SMELL/FwdLLM 中集成 Jenga 的 Token Elimination 机制，使用预训练的 Predictor 对 attention 层的 token 做 block 级稀疏化。PrunedMLP 和 Segment Loss 已注释。

## 文件操作

### 新建 (2)

| 文件 | 用途 |
|---|---|
| `FwdLLM/model/modeling_llama_sparse.py` | 从 Jenga modeling_llama.py 复制，保留 Token Elimination，注释 PrunedMLP + Segment Loss |
| `FwdLLM/model/predictor.py` | 从 Jenga predictor.py 复制，提供 PrunableAttnPredictorInfer 类 |

### 修改 (3)

| 文件 | 修改内容 |
|---|---|
| `FwdLLM/model/modeling_llama_sparse.py` | imports 改为本地 predictor; PrunedMLP 注释+标准 MLP 替代; Segment Loss 注释+标准 loss; logits=None→logits=logits |
| `FwdLLM/experiments/.../initializer.py` | 新增 llama_sparse 分支: 加载 pruned_config.pth + predictor.pth; 新增 --sparse CLI arg |
| `FwdLLM/forward_training/tc_transformer_trainer_distribute.py` | layer_id_for_check: 16→14 |

## 前置依赖

需手动复制 Predictor 文件:
```
SMELL/FwdLLM/checkpoints/predictor/
  ├── predictor.pth         ← 从 Jenja/checkpoints/predictor/ 复制
  └── pruned_config.pth     ← 从 Jenja/checkpoints/predictor/ 复制
```

## 稀疏化行为

- layer 0-14: 全保留（predictor 运行但不裁剪）
- layer 15-30: predictor 选 Top-K block → sparse 比例保留
- layer 31: 跳过（标准 flash attention）
- sparse=0.4: 保留 40% token block

## 静态检查

所有 4 个 Python 文件 py_compile 通过。
