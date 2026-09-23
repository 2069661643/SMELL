# SMELL Phase 1/2 h5py DataManager 改造

## 概述

将 SMELL Phase 1/2 的数据路径从直接 DataLoader（`datasets.load_dataset` 在线加载）改为 h5py 预分区 DataManager（对齐 FwdLLM 联邦数据架构）。

## 文件操作

### 新建 (2)

| 文件 | 用途 |
|---|---|
| `FwdLLM/data_preprocessing/causal_lm_preprocessor.py` | h5py uint16 token IDs → TensorDataset |
| `FwdLLM/data_manager/causal_lm_data_manager.py` | 继承 BaseDataManager，read_instance_from_h5 读 token IDs |

### 修改 (3)

| 文件 | 修改内容 |
|---|---|
| `FwdLLM/forward_training/tc_transformer_trainer_distribute.py` | self.grad dict→list（FedSGDTrainer.train_with_data_id 兼容），新增 _param_to_idx |
| `FwdLLM/experiments/.../run_lm_exps/smell_main.py` | 重写数据链路: CausalLMDataManager → FedSGDTrainer → FedSGDClientManager |
| `FwdLLM/experiments/.../run_lm_exps/run_phase1_baseline.sh` | 数据集检查改为 h5py; 新增 --data_file_path/--partition_file_path/--partition_method |
| `FwdLLM/experiments/.../run_lm_exps/run_phase2_sparse.sh` | 同上 |

## 关键修复

| 问题 | 修复 |
|---|---|
| P1: 绕过 DataManager | 接入 CausalLMDataManager 继承 BaseDataManager |
| P2: FedTransformerTrainer 直接给 FedSGDClientManager | 中间加 FedSGDTrainer 层 |
| P3: FedSgdAggregator 拼写错误 | 改回 FedSGDAggregator |
| P4: FedTransformerTrainer 构造参数错误 | 改回 (trainer, model) |
| P5: FedSGDAggregator 构造参数缺失 | 12 参数全量传入 |
| P6: self.grad dict 在 FedSGDTrainer 中迭代出错 | 改为 list + _param_to_idx 映射 |

## 数据接口

--data_file_path      ./dataset/smell_data.h5
--partition_file_path ./dataset/smell_partition.h5
--partition_method    uniform_client_100

h5py 期望结构:
  data.h5:      tokens[str(idx)] = uint16 numpy array [context_length]
  partition.h5: <method>/partition_data/<client_id>/train = [idx1, idx2, ...]
