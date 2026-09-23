# SMELL Phase 0 GoEmotions 数据预处理 + Dirichlet 预划分

## 概述

Phase 0 目标：将 GoEmotions 数据集从 HuggingFace 缓存转换为 FwdLLM 联邦数据链路所需的 h5py 文件，并做 **Dirichlet label-skew 预划分**，为后续实验（含 Phase 3 的"共识率"数据集对比：高共识 GoEmotions vs 低共识 Red…）准备数据。

本次先实现 **GoEmotions 版本**；RedPajama（Red…原数据集）留待后续版本（缓存中只有 GPT-2 预分词 `input_ids`，需 decode→Llama2 重分词 + quantity-skew Dirichlet）。

## 文件操作

### 新建 (3)

| 文件 | 用途 |
|---|---|
| `script/preprocess_phase0.py` | Phase 0 主脚本：GoEmotions → `goemotions_data.h5`（预 tokenize uint16）+ Dirichlet label-skew 划分 → `goemotions_partition.h5`，内置统计与校验 |
| `script/RUNME-Phase0.sh` | 运行脚本（conda 激活、参数透传、日志） |
| `docs/ailog/260807-163000-smell-phase0-goemotions-preprocessing.md` | 本变更日志 |

### 修改 (0)

无。保留已跑通的 `script/preprocess_goemotions.py` 冒烟测试不动。

## 数据源

| 数据集 | 位置（HF 缓存，离线读取） | 结构 | 规模 |
|---|---|---|---|
| GoEmotions (simplified) | `~/.cache/huggingface/datasets/go_emotions/simplified/<ver>/<hash>/go_emotions-{train,validation,test}.arrow` | `text` / `labels`(28 类多标签) / `id` | train 43,410 / val 5,426 / test 5,427 |

- 用 pyarrow IPC stream 读取 arrow 缓存，不访问网络。
- 多标签 → 主标签取 `labels[0]`（用于 Dirichlet 划分与统计）。

## 数据接口（h5 结构）

```
data.h5:
  attributes             = JSON 字符串 {index_list, train_index_list, test_index_list,
                                        label_vocab{name:id}, dataset, max_length, n_classes}
  tokens/<idx>           = uint16 [max_length]   （Llama2 预 tokenize，pad 到 max_length）
  labels/<idx>           = int32 主标签

partition.h5:
  <method>/n_clients
  <method>/alpha
  <method>/partition_data/<client_id>/train   （全局索引 int64）
  <method>/partition_data/<client_id>/test    （全局索引 int64）
```

全局索引约定：train 样本占 `0..T-1`，test 样本占 `T..T+E-1`（对齐 `FwdLLM/data/advanced_partition/niid_label.py`）。
`--method_name` 默认 `niid_dirichlet_clients=N_alpha=A`，与训练脚本 `--partition_method` 参数对齐。

## 关键设计

| 设计点 | 说明 |
|---|---|
| Dirichlet label-skew（per-class LDA） | 参考 FedML `non_iid_partition_with_dirichlet_distribution` / `partition_class_samples_with_dirichlet_distribution`；每个类别单独 `dirichlet(alpha)` 分配到各 client，α 越小分布越偏（Non-IID 越强） |
| min_size 重试 | 任意 client 样本数 < 1 时整体重采（最多 200 次），避免空 client |
| split 点防御 | `np.unique` 去重保证严格递增，避免 α 过小 / client 满员时 `np.split` 报错 |
| 可复现 | `--seed`（默认 42），test 划分使用 `seed+1000` 区分随机流 |
| 打包（可选）| `--pack`：多条短样本以 eos 分隔 concat 成 `max_length` 块，适配长上下文实验；默认关闭，per-sample truncate+pad 对齐原冒烟测试 |

## 用法

```bash
# 冒烟（小样本）
bash script/RUNME-Phase0.sh
MAX_TRAIN=12 MAX_TEST=6 bash script/RUNME-Phase0.sh   # 更小

# 全量
CLIENTS=10 ALPHA=0.5 bash script/RUNME-Phase0.sh

# 长上下文打包
PACK=1 bash script/RUNME-Phase0.sh
```

## 验证

- 脚本内置 `verify()`：回读 h5，校验 `tokens/<idx>` dtype==uint16、划分索引不越界 / 不重叠 / 全覆盖（train==0..T-1，test==T..T+E-1）。
- 打印每 client train/test 样本数与 top 标签分布，供 α 灵敏度 sanity check（α=0.1 vs 1.0）。

## 待办

- [ ] RedPajama 版本：GPT-2 `input_ids` → 无损 decode → Llama2 重分词 → quantity-skew Dirichlet → `redpajama_data.h5` / `redpajama_partition.h5`
- [ ] 全量 h5 生成后与 Phase 1/2/3 冒烟测试联调（`CausalLMDataManager` 当前读 `.pt`；真实实验走标准 h5 链路）
