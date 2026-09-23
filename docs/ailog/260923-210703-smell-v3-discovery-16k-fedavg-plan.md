# Ailog 260923-210703 — SMELL-v3: Discovery 16k 数据 + 串行 FedAvg 执行计划（锁定）

## 概述

本文件把 2026-09-23 讨论定稿的 SMELL-v3 实验方案固化下来，作为后续编码与实验的权威依据：数据用 **Discovery**（`sileod/discovery`，HF，apache-2.0）自建 **16k ICL 分类样本**，per-class Dirichlet 划给 **30 clients**（α=0.1/0.3，方案 A：演示也取自 client 自己的偏斜池），训练用 **ZOO+LoRA**，联邦协议改为**串行 FedAvg（n_i 加权）**，评测仅在同一 Discovery 内部做 **G-PPL（全文 + answer-only，token 加权）**。`third_party` 保持只读（允许复制到 `src/` 重建简化框架）。

## 锁定决策

| 维度 | 决定 |
|---|---|
| 模型 | OPT-350M（Jenga `OPTForCausalLM`），仅 16k，位置嵌入用 Jenga 复制基线 |
| 训练 | ZOO + LoRA（主线）；CATV on/off 对照（先留 `--catv` 接口）；无其他基线 |
| 联邦 | c=30 串行，FedAvg 加权平均，权重默认 n_i/Σn（当前各 client 等量 ⇒ 等价 1/30） |
| 数据 | Discovery 自建 16k ICL 样本，纯文本拼接（无 doc mask/无 `[SEP]`），query 末尾补 GT label |
| 划分 | per-class Dirichlet α∈{0.1, 0.3}，各出一套 static 分片（FwdLLM 式预构建） |
| 规模 | 30 clients × (100 train + 16 local-test)；G-PPL 500 条（官方 test query + 独立均匀演示池） |
| 评测 | 仅 Discovery 内部：全文 PPL + answer-only PPL（只统计末尾 GT label token），均报 token 加权与 per-sample；暂不做 L-PPL，local test 仅 eval |
| 平台 | 本机 8GB 只跑 smoke；云端每次实验 1 张 A40、30 client 严格串行；可 4 个脚本并发做 (α, CATV) 消融 |

## 数据管线（`src/data/`）

- 来源：`sileod/discovery`（hf-mirror），train 1,566,000 / validation 87,000 / test 87,000，174 类（含 `[no-conn]`），平均 ~61 tokens/句对。
- 预算与互斥：train 按 idx 切成 `client_source`（~85%）与 `global_demo_source`（~15%），前者做 per-class Dirichlet 分给 30 clients；client 池、global 演示池、官方 test query 三池 id 互斥。需求 ≈ 930k（client，100+16 条 ×30）+ 134k（G-PPL 演示）< 1.33M，可行；per-client 配额 seed 检查，不足时后备允许同 client 内复用并告警。
- 构造（方案 A）：每条训练样本 = instruction + ~267 条演示（从本 client 池不放回抽）+ 1 条 query（同池，其 label 即样本标签）+ GT label；tokenize（OPT）后恰好 16384（超出时从最早的演示开始删，不足继续加）。
- 模板照抄 LongICLBench（instruction 原文 + `s1 ( ) s2` + `the most suitable conjunction word in the previous ( ) is  <label>,`）。
- 产物：`dataset_v3/discovery_16k/<tag>/{global_test_input_ids.npy, global_test_labels.npy, global_test_answer_spans.npy, clients/client_{00..29}/{train_input_ids.npy, train_labels.npy, train_answer_spans.npy, local_test_*.npy}, meta.json, partition.json}`；uint16 存 input_ids，int32 存 answer span（start,end），int64 存 label。
- `check_partition.py`：长度=16384 且 %64=0、三池互斥、标签直方图、client 间 JS 散度、可复现 hash。

## 评测口径（`src/eval/ppl.py`）

- 全文 PPL：`exp(Σ NLL_all / Σ 16384)`（token 加权），同时报 per-sample 平均。
- answer-only PPL：只对 `answer_spans` 内 token 算 NLL，`exp(Σ NLL_label / Σ L_label)`，同时报 per-sample 平均。
- 固定 eval 顺序、batch=1、`torch.no_grad()`；logits 分块投影避免 16k×50272 爆显存。

## 简化框架（`src/`，third_party 只读、允许复制）

| 文件 | 说明 |
|---|---|
| `src/models/modeling_opt_smell.py` | 从 Jenga `modeling_opt.py` **原样复制** + `# SMELL 3 ... COPY` 标注，待接 TokenSelector/CATV 时再改 |
| `src/models/token_selector.py` | `JengaSparseSelector`（predictor+块 topk）、`CATVSelector`（占位） |
| `src/models/position_embed.py`（P2） | Jenga 复制方案 + 纯重复 + 线性插值 |
| `src/train/lora.py` | LoRA 配置工厂（r=8，targets 可配）；参数量对照：attn 1.57M / +ffn 3.54M |
| `src/train/zoo.py` | 中心差分前向梯度估计 + 参数枚举/聚合一致性工具 |
| `src/eval/ppl.py` | 上述 PPL 口径 |
| `src/bench/memory.py` | 16k 显存矩阵 `{sparse,dense}×{LoRA-attn,+ffn}×{ckpt on/off}`，`memory_fraction(0.9)` 防 WSL 溢出假象 |
| `src/fed/serial_fedavg.py` | ClientRunner / ServerAggregator（n_i 加权、状态字典一致性断言、JSONL 指标） |
| `src/fed/run_fed.py` | CLI：`--alpha --catv --local-steps --rounds --lr --seed --tag --weight`，写 `config.json` + `metrics.jsonl` |

## 串行 FedAvg 协议

```
for round r:
  for i in 1..30 (串行):
      θ_i = local_train(copy(θ_global), shard_i, local_steps)   # ZOO+LoRA，可选 CATV
      Δ_i = θ_i − θ_global                                       # 仅 LoRA A/B
  θ_global += Σ_i w_i Δ_i,  w_i = n_i/Σn_j                       # FedAvg（n_i 加权）
```
- 与 v2 的 `Σg→归一化→clamp（‖Δθ‖≡lr）`不同：v3 是加权平均，步长语义变化，v2 的 lr 结论不直接迁移。
- 单进程串行 ⇒ 参数名/顺序天然一致，仍保留聚合前断言；每轮记录两种 G-PPL、`‖Δ_i‖`、`cos(Δ_i,Δ_j)` 抽样、通信字节。

## 执行与分工

1. 本 ailog 提交后，用子智能体并行搭框架：A=数据管线（`src/data/`），B=模型/训练（`src/models/`、`src/train/`），C=评测/联邦（`src/eval/`、`src/bench/`、`src/fed/`）；每个子任务只回「文件+验收输出」，主聊天保上下文。
2. 验收：全部 `py_compile`；数据 `--mini` 跑通 + `check_partition` 通过；框架 `--help` 与 CPU 聚合单测；随后本机 8GB smoke（c=2、2 steps、2 rounds、16k、FA2+sparse+LoRA+ckpt）。
3. 正式实验（云端 1×A40/实验，4 并发消融）与打点调参由 CLI 参数支持，不改代码。

## 风险与后备

- α=0.1 的 per-client 配额可能偏紧 → seed 检查 + 同 client 内复用后备。
- 8GB 峰值不确定 → P1b 矩阵定（预期 ckpt on + Jenga sparse 可 <7.5GB）。
- answer span 定位（多 token、含逗号）→ 模板切分 + tokenizer 对齐，单测覆盖。
- G-PPL 演示池用**均匀类分布**（与 client 偏斜刻意不同）；三池互斥防泄漏。
