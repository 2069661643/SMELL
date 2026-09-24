# Ailog 260924-105611 — SMELL-v3: Discovery 16k 全量数据构建完成 + checker/显存矩阵修复

## 概述

Discovery 16k 数据集全量构建完成（a01/a03 两套，各约 141s、142MB），`check_partition` 复检 **PASSED**（修复了一个跨 split 索引比较的假阳性）。本机显存矩阵（16k × 8 组合）在**硬 8GB 上限**（`set_per_process_memory_fraction(0.9)`）下全部 OOM——说明 16k 前向+LoRA 在本机实际依赖 WSL 的显存↔系统内存溢出才能跑通（此前 `run_fed`/`ppl` 冒烟成功即属此种"慢速溢出"），本地只适合短序列 smoke；云端 A40（48GB）不受影响。同时修复了 `memory.py` 的 `--only` 逻辑（组内应为 AND）并新增 `--mode forward|backward`（ZOO 主线为纯前向）。

## 变更列表

| 时间 | 操作 | 文件/位置 | 影响 |
|---|---|---|---|
| 10:40–10:45 | 构建 | `dataset_v3/discovery_16k/{a01,a03}/`（gitignore） | 各 3980 条 16k 样本：30 clients × (100 train + 16 local) + 500 G-PPL；token_cache_miss≈1.09M/套 |
| 10:45 | 校验 | `check_partition` | 两套均报 1 error：`test queries overlap global demos n=44`（假阳性） |
| 10:52 | 修复 | `src/data/check_partition.py` | 删除跨 split（test idx vs train idx）的交集比较；复检两套 PASSED；a01 仅 client_26 有 783 次（~2.9%）演示复用 WARN，a03 零复用 |
| 10:53 | 修复 | `src/bench/memory.py` | `--only` 组内 AND/组间 OR；新增 `--mode forward|backward`（默认 forward）；None 字段打印不再崩溃；JSON 先于表格落盘 |
| 10:55 | 运行 | `src/bench/memory.py --mode forward --cap-fraction 0.9` | 8 组合全 OOM：**16k 在本机 8GB 物理上限内不可行**（含 LoRA qkvo/ckpt） |

## 关键数据

- 每套：`total_train≈82.0万` 源样本（30×116 条样本 ≈ 3530 条 16k + 500 G-PPL），共 3980 条 16k 序列；磁盘 142MB/套。
- a01：`client_26` 池不足触发复用 783 次（占其演示 ~2.9%）；a03 无复用。
- 校验：seq_len=16384（%64=0）、span 解码==label、三池互斥/唯一、JS 散度、可复现 hash 全过。

## 设计决策 / 发现

- **跨 split 索引不可比较**：`global_test_query_idx` 属 test split（max 87k），`global_demo_idx` 属 train split（max 1.57M），数字交集是伪泄漏；真正的隔离靠"不同 split + 不同池分配"。
- **16k 超出本机 8GB 物理显存（即使纯前向）**：矩阵在 0.9 上限下全部 OOM；此前冒烟能跑是因为 WSL 驱动允许溢到系统内存（表现为 step 时间变长）。结论：本机 smoke 建议 ≤4k 或接受慢速溢出；正式 16k 一律 A40。
- `--only` 语义修正后，后续单组合诊断可用 `sparse=1,lora=attn,ckpt=0` 精确匹配。

## 待办 / 风险

- **位置适配决策仍阻塞正式训练**（见 `260924-103741` ailog 的 PPL 长度曲线）：可训练位置嵌入 vs 适配阶段 vs RoPE retrofit。
- 显存矩阵若需"真实峰值"，建议用 `--mode forward` 不设 cap 再跑一轮，以 `nvidia-smi` 观察溢出（本地意义有限，云端不需要）。
- CATV 选择器仍是占位；服务器环境未就绪。
