# SMELL Phase 1 — Llama2-7B Causal LM 模型替换

## 概述

在 `SMELL/FwdLLM` 内将原始 FwdLLM (DistilBERT 文本分类 + functorch) 替换为 Llama2-7B 因果语言模型 + peft LoRA + torch.func，不加 Jenga 稀疏性。

## 环境变更

- Python 3.7 → 3.10
- torch 1.10.0 → 2.1.2
- functorch → torch.func.functional_call
- adapter-transformers 3.1.0 → peft >= 0.5.0
- 新增: flash-attn, accelerate, datasets

## 文件操作清单

### 新建文件 (7个)

| 文件 | 用途 |
|---|---|
| `FwdLLM/model/modeling_llama_base.py` | 从 Jenja 复制，基线 Llama 模型（无 sparsity） |
| `FwdLLM/data/data_loader_causal_lm.py` | RedPajama 加载 + tokenize |
| `FwdLLM/evaluation/longbench.py` | 从 Jenga 复制，LongBench 零样本评估 |
| `FwdLLM/evaluation/eval.py` | 从 Jenga 复制，LongBench 指标计算 |
| `FwdLLM/evaluation/__init__.py` | 空 |
| `FwdLLM/experiments/distributed/transformer_exps/run_lm_exps/smell_main.py` | Phase 1 入口点 |
| `FwdLLM/experiments/distributed/transformer_exps/run_lm_exps/run_causal_lm.sh` | mpirun 启动脚本 |

### 修改文件 (5个)

| 文件 | 修改内容 |
|---|---|
| `FwdLLM/requirements.txt` | 版本对齐 Jenga，注释 adapter-transformers + functorch |
| `FwdLLM/forward_training/utils/fwdgrad_utils.py` | 新增 functional_get_loss_causal_lm + calculate_jvp_causal_lm |
| `FwdLLM/forward_training/tc_transformer_trainer_distribute.py` | ForwardTextClassificationTrainer → ForwardCausalLMTrainer |
| `FwdLLM/experiments/distributed/transformer_exps/initializer.py` | 添加 causal_lm 分支 + peft LoRA + --model_max_length |
| `FwdLLM/FedML/fedml_api/distributed/fedsgd/FedSgdAggregator.py` | var_threthod 添加 llama 条目 |

## 关键设计决策

1. **functional_call 替代 make_functional_with_buffers**: 因为 peft LoRA 仅 ~4M 可训参数，字典格式更自然
2. **no_grad 保留**: 外循环包裹整个 train_model，保证不构建 autograd 图
3. **loss 用 HF 内部计算**: labels=input_ids → forward 内部自动 shift+CE
4. **Flash Attention 2**: config._attn_implementation = "flash_attention_2"
5. **bfloat16**: 与 Jenga 一致

## 未完成事项（后续 phase）

- Jenga token sparsity 集成
- Consensus Anchor Tokens
- Pipeline-ZOO (梯度压缩、计算-通信并行)
- LongBench 评估 pipeline 对接
- 多 GPU MPI Worker 方差控制修复

## 验证

- 静态检查: Python 语法、import 可用性
- 不进行实际训练（模型权重缺失）
- RedPajama dataset 可后续通过 git submodule 加入
