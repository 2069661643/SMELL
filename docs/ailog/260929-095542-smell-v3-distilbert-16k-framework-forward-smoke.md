# Ailog 260929-095542 — SMELL-v3: DistilBERT 16k 框架落地 + 前向 smoke（M0/M1/M2/M3 骨架）

> 承接 `260929-000831`（DistilBERT 16k port 计划）。本轮按「本机 8GB 显存只做框架 + 前向 smoke，不做 BP 训练」执行，
> 由两个子智能体分别完成数据管线与模型/评测，主 session 做集成审阅与验证。数据源切换为 `ccdv/arxiv-classification`（no_ref，11 类），
> 长度策略 = 头尾截断（75/25）+ 右 padding 到 seq_len，评测 = accuracy / macro-F1 / NLL。

## 变更列表

| 时间 | 操作 | 文件 | 说明 |
|---|---|---|---|
| 09:41 | 新增 | `src/data/build_arxiv_16k.py` | arXiv 16k 联邦分片构建：streaming 拉取、按 token 头尾截断、`lengths.npy`、per-class Dirichlet（复用 build_discovery 的 `build_partition`/`ClientPoolStream`） |
| 09:45 | 新增 | `src/data/build_warmup_arxiv_16k.py` | warmup 池（512×seq_len），读 partition 推导未用行，零重叠断言；为 M2 位置适配准备无标签语料 |
| 09:46 | 修改 | `src/data/check_partition.py` | +53/−25：识别 `*_lengths.npy` 新风格（int32、右 padding==pad_id），旧 Discovery span 路径逐行不变 |
| 09:47 | 新增 | `src/models/position_embed_bert.py` | DistilBERT 位置表扩展（interpolate/duplicate/dup_scaled，offset=0）；同步 `position_ids` buffer 与 `config.max_position_embeddings` |
| 09:47 | 新增 | `src/models/modeling_distilbert_smell.py` | fp32 SDPA 注意力（`__class__` 免复制 patch）+ 分类加载器 + 9 项 selftest |
| 09:47 | 新增 | `src/eval/classify.py` | 分类评测 CLI（accuracy/macro-F1/NLL；global/local split；可选 pos-checkpoint/adapter） |

## 关键实现事实（踩坑记录）

- **transformers 4.45.2 没有 DistilBERT SDPA**：`DistilBertSdpaAttention` 不存在、`_supports_sdpa=False`、类名是 `MultiHeadSelfAttention`
  ⇒ 自写子类（`getattr(..., "DistilBertAttention", MultiHeadSelfAttention)` 兜底），运行时 `attention.__class__ = DistilBertSdpaAttention` 免复制替换。
- 该版 mask 是 **1=keep/0=pad 的 2D 张量**（非 additive）⇒ `_sdpa_additive_mask` 自动转换；additive 3D/4D 直接透传。
- `DistilBertEmbeddings` 缓存 512 长的 `position_ids` buffer ⇒ 扩展位置表时必须同步重写，否则 >512 前向 size mismatch（已修）。
- `output_attentions=True` 返回 `(context, None)` 占位（SDPA 不产 attn 权重）；`head_mask` 通过 context 置零实现（本流程暂不用）。
- 模型下载：本机 `HF_HUB_DISABLE_XET=1` 仍会走 `cas-bridge.xethub.hf.co` 并 ReadTimeout；实际用 `HF_HUB_DISABLE_XET=0` + hf-mirror 拿到
  `third_party/Jenga/checkpoints/distilbert-base-uncased/`（tokenizer + `model.safetensors` 267,954,768 B）。

## 数据格式（新）

`dataset_v3/arxiv_16k/<tag>/`（gitignore）：
`clients/client_XX/{train,local_test}_{input_ids,labels,lengths}.npy`、`global_test_*`、`warmup_input_ids.npy|warmup_lengths.npy|warmup_meta.json`、`partition.json`、`meta.json`。
- `input_ids` uint16 `[N, seq_len]` 右 padding；`lengths` int32 真实 token 数；截断 = `ids[:0.75*L] + ids[-0.25*L:]`；`[CLS]/[SEP]` 由 tokenizer 加。
- mini 构建（tag=`a01mini`，seq 2048）：4 clients × (5 train + 2 local) + global 10 + warmup 8；来源 300 文档 streaming。

## 验证（命令与结果）

`bash temp/verify_bert_framework.sh`（WSL env `SMELL`，py3.10 + torch 2.8.0+cu128 + transformers 4.45.2）：

| 项 | 结果 |
|---|---|
| `py_compile`（6 个 py） | PASS |
| `python src/models/modeling_distilbert_smell.py` | **9/9 PASS**（SDPA vs eager atol 1e-5 含 padding、patch 幂等、位置三模式、两种 pos key 加载） |
| `check_partition.py --root dataset_v3/arxiv_16k/a01mini` | **PASSED**（lengths/pad/分区不变量全 OK） |
| `check_partition.py --root dataset_v3/discovery_16k/a01`（回归） | **PASSED**（旧 span 路径未破坏） |
| `classify.py --tag a01mini --split global`（随机分类头） | acc 0.1000 / macro-F1 0.0227 / NLL 2.3962（n=10，符合未训练预期；端到端链路通） |
| `temp/smoke_bert_forward.py`（真实权重，fp32，RTX 5060 8GB） | **16k：logits (1,11) finite，2.02s，峰值 0.88GB**；2k：0.05s，0.40GB |

## 已知限制 / 风险

- 只建了 mini 数据；全量 a01（30×100+、16k）未跑（网络 streaming + 16k 截断待 AutoDL/amax 执行）。
- `classify.py` 硬编码 `.cuda()`；`--adapter` 分支未实测（暂无 DistilBERT adapter）。
- 未启动：M2 全量位置适配训练（BP，按约定不在本机跑）、M4 稀疏 + predictor port、M5 `run_fed --arch`。
- 剪枝层调度（6 层只剩 3 层可剪）与 full-build 的 `global_demo_frac` 语义沿用 Discovery，待 M1 全量时复核。

## 待办

- [ ] M1 全量：AutoDL/amax 跑 `build_arxiv_16k.py --tag a01` + warmup，更新 manifest sha256
- [ ] M2：`longctx_adapt --arch distilbert`（MLM on warmup）→ pos_embed.pt，过 Go/No-Go
- [ ] M3：BP 分类基线（dense/sparse、LoRA k-scan）与评测协议固化
- [ ] M4：`modeling_distilbert_smell` 加 `DistilBertSdpaPruneAttention` + predictor 训练器
- [ ] M5：`run_fed --arch distilbert` 分派 + CATV 投票 + ZO cos 诊断
