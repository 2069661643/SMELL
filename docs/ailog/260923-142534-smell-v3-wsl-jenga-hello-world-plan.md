# Ailog 260923-142534 — SMELL-v3: WSL2 Jenga 启动验证计划（hello_world → OPT-350M smoke）

> **执行结果（260923-161344）：已完成。** 结果 ailog 见 `docs/ailog/260923-161344-smell-v3-wsl-jenga-smoke-result.md`。
> 偏差：环境名改为 **`SMELL`**；torch 实装 **2.8.0**+cu128（2.8.1 不存在）；FA wheel 用 **cxx11abiTRUE**；`hello-world.sh` 退出码 0（仅 gated llama2/llama3 config 缺失）；`src/smoke_jenga_opt.py` FA2/eager 双 PASS。

> **状态：计划（待 WSL session 执行）** — 本文件是交接用执行计划，不是完成记录。
> 下一个 session（在 WSL Ubuntu 内）按本文件执行；完成后另写结果 ailog，并在本文件顶部补「执行结果」。
> 决策前提（用户已定）：Windows 侧文件全部迁移到 WSL；在 WSL 内新建 conda 环境；Windows 的 4 个 conda 环境**不动**。

---

## 1. 目标与验收

1. **主目标**：在 WSL2 Ubuntu 24.04 + RTX 5060 Laptop（8 GB，sm_120）上跑通 Jenga 官方 `hello_world` 的资源/导入检查。
2. **实际验收**（因为 hello_world 的 CUDA 兼容测试硬编码 llama2-7B，见 §5）：OPT-350M 版 smoke —— Jenga 自定义 `OPTForCausalLM` + `flash_attention_2` 在 sm_120 上完成前向+反向，loss 有限、显存不爆。
3. 产出一个可复现的环境（conda env `smell-v3` + 明确 pip 版本），后续 SMELL-v3 ZOO 开发直接用它。

---

## 2. 已核实事实（Windows 侧，2026-09-23）

### 2.1 conda 环境（均不适配 Jenga）

| 环境 | Python | 关键包 | 结论 |
|---|---|---|---|
| `base`（D:\Applications\Anaconda） | 3.13.9 | 无 torch | 不可用 |
| `base0` | 3.13.9 | 无 torch | 不可用 |
| `USTC_AB_26` | 3.12.13 | 无 torch | 不可用 |
| `USTC_CG_26` | 3.13.9 | torch 2.11.0+cu130 / transformers 5.10.1 / peft 0.19.1 / bitsandbytes 0.49.2 | 通用环境，与 Jenga pin 全冲突，且无 deepspeed/einops/fire/flash-attn |

`smell-v2` 环境在**本机不存在**（AGENTS.md 提到的是 v2 机器上的环境名）。

### 2.2 硬件 / 驱动

- GPU：**RTX 5060 Laptop，8151 MiB，sm_120（Blackwell）**；Windows driver 616.92 / CUDA 13.0。
- WSL2 Ubuntu 24.04 已启用且 GPU 直通正常（WSL 内 nvidia-smi 可见 5060，driver 615.71.08 / CUDA UMD 13.4）。
- WSL 内**无 conda、无 nvcc**；系统 Python 3.12.3；ext4 空闲 ~945 GB；内存 15 GiB。

### 2.3 关键兼容性结论

- **torch 2.1.2（Jenga requirements pin）无 sm_120 kernel** → 在 5060 上不可用（CPU 回退/报 no kernel image）。Blackwell 需要 **torch ≥ 2.7 + cu128**，因此环境必须偏离 Jenga pin。
- **flash-attn 有预编译 wheel**：v2.8.3 release 提供 `cp310 + cu12 + torch2.8` 的 Linux wheel（`cxx11abiFALSE/TRUE` 两种），**无需 nvcc 编译**。
- `third_party/Jenga/src/jenga/models/modeling_opt.py:46` 仅在 `is_flash_attn_2_available()` 时导入 `transformers.modeling_flash_attention_utils._flash_attention_forward`（transformers 4.45.2 存在该函数），因此 torch 2.8 + transformers 4.45.2 + FA 2.8.3 的组合预期兼容，需 smoke 验证。
- `modeling_opt.py:409` 硬编码 `device='cuda:0'`、稀疏路径假定 batch=1、seq_len 为 64 的倍数 —— 单卡/小 batch 冒烟没问题。

### 2.4 资源清单（迁移前，Windows 侧）

| 资源 | 位置/大小 | 现状 |
|---|---|---|
| Jenga 权重解压目录 `third_party/Jenga/checkpoints/` | — | 只有 `.gitkeep`（`checkpoints/` 被 Jenga `.gitignore` 覆盖） |
| Jenga 数据集目录 `third_party/Jenga/dataset/` | — | 只有 `.gitkeep`（同样被 gitignore） |
| `dataset.zip` | `../download/`（2.0 GB），内含 `dataset/{LongAlign,longbench,PPL,RedPajama-Data-1T-Sample}` | 未解压 |
| `predictor.zip` | `../download/`（51 MB），内含 `predictor/{predictor.pth,pruned_config.pth}` | 未解压 |
| `peft_model.zip` | `../download/`（5.0 GB），内含 `peft_model/{la,rp}/{jenga,lora}/...` | 未解压（**冒烟验证可延后**） |
| `facebook/opt-350m` | 未下载（Windows HF cache 无此模型） | 需下载 |
| llama2 / llama3 / opt-1.3b/2.7b/6.7b | 未下载（HF cache 的 Llama-2-7b 目录 blobs 为空） | hello_world 文件检查需要 config；llama2 权重缺失（见 §5） |

素材 zip 清单来自父仓库 `D:\GitRepository\Projects\FedLLM\download\`；若 WSL 侧位置不同，先 `find` 定位。

---

## 3. 环境决策（已定，勿改）

| 组件 | Jenga pin | 本环境采用 | 理由 |
|---|---|---|---|
| OS | — | WSL2 Ubuntu 24.04（ext4） | GPU 直通已验证；Linux 栈是 Jenga 原生环境 |
| Python | 3.10 | **3.10** | 与 wheel（cp310）匹配 |
| torch | 2.1.2 | **2.8.1+cu128** | 2.1.2 无 sm_120；2.8 是 FA cp310 wheel 覆盖的最新版 |
| transformers | 4.45.2 | **4.45.2** | 不变 |
| tokenizers | 0.20.1 | **0.20.1** | transformers 4.45.2 要求 `>=0.20,<0.21` |
| flash-attn | 单独装 | **2.8.3 预编译 wheel** | 免编译；按 `torch._C._GLIBCXX_USE_CXX11_ABI` 选 abi TRUE/FALSE |
| peft | >=0.5.0 | **0.13.2**（回退 0.14.0） | 与 transformers 4.45.2 同期，避免新版隐式要求 |
| numpy | >=1.26.0 | **1.26.4** | 避开 numpy 2.x 兼容风险 |
| accelerate / datasets | >=0.34 / >=2.14.5 | **1.0.1 / 2.21.0** | 同代版本 |
| deepspeed / bitsandbytes | 0.14.0 / 0.41.1 | **不装** | SMELL ZOO 管线不需要；后续要用时再补 |

- **Jenga 不 `pip install -e`**：避免在 submodule 里生成 `*.egg-info` 脏文件（AGENTS.md 要求 third_party 干净）。改用 `PYTHONPATH` 指向 `third_party/Jenga/src`。
- **绝对不要** `pip install -r third_party/Jenga/requirements.txt`：会把 torch 降回 2.1.2 并拉 deepspeed/bitsandbytes。
- 环境全部放 WSL ext4（`~/miniconda3`），代码/数据也可整体迁移到 ext4（用户已定）。

---

## 4. 执行步骤（给 WSL session）

### Step 0 — 定位与盘点

```bash
# 1) 确认仓库根目录（用户迁入 WSL 后的实际路径），后续记为 $SMELL_V3
cd ~ && find . -maxdepth 3 -type d -name "SMELL-v3" 2>/dev/null
export SMELL_V3=$(realpath ~/SMELL-v3)   # 按实际结果替换
cd $SMELL_V3

# 2) 基础盘点
pwd; ls; ls third_party/Jenga
git status --short                       # v3 主仓库
git -C third_party/Jenga status --short  # 应干净（checkpoints/、dataset/ 被 ignore）
git submodule status                    # 期望 FwdLLM / Jenga / VQ 三个

# 3) GPU / 磁盘
nvidia-smi
df -h ~ $SMELL_V3
```

**注意**：本节所有命令基于「仓库在 `$SMELL_V3`」假设；若实际不同（例如只迁了 Jenga），先修正路径再继续，并把差异记入结果 ailog。

### Step 1 — 安装 Miniconda + 建环境

```bash
cd ~
wget -O miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
exec bash
conda create -n smell-v3 python=3.10 -y
conda activate smell-v3
python -V   # 期望 Python 3.10.x
```

### Step 2 — 装 torch(cu128) + 依赖

```bash
pip install torch==2.8.1 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.45.2" "tokenizers==0.20.1" "peft==0.13.2" \
  "accelerate==1.0.1" "datasets==2.21.0" "numpy==1.26.4" \
  "sentencepiece" "fire" "einops" "scipy" "protobuf" \
  "torchmetrics" "rouge_score" "rouge" "jieba" "fuzzywuzzy" "matplotlib"
```

若 pip 官方源慢：加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（torch 那步除外，torch 必须来自 cu128 索引；若 pytorch 官方源被墙，换阿里云/上交 `pytorch-wheels` 镜像中 `cu128` 目录，或 `--index-url` + `--trusted-host`，具体地址以当时可用为准，记入 ailog）。

### Step 3 — 装 flash-attn（预编译 wheel）

```bash
# 先确认 ABI 选择：
python -c "import torch; print('cxx11_abi =', torch._C._GLIBCXX_USE_CXX11_ABI)"
# False -> 用 cxx11abiFALSE；True -> 用 cxx11abiTRUE

pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
```

> GitHub release 下载慢/失败时的备选：① 用代理/镜像拉 wheel；② 源码编译（需 CUDA toolkit 12.8：`conda install -c nvidia cuda-toolkit=12.8`，再 `pip install flash-attn==2.8.3 --no-build-isolation`，耗时 30–60 min）。选哪条都记入 ailog。

### Step 4 — Jenga 接入（PYTHONPATH）

```bash
conda env config vars set PYTHONPATH=$SMELL_V3/third_party/Jenga/src
conda deactivate && conda activate smell-v3
python -c "import jenga; print(jenga.__file__)"   # 期望 .../third_party/Jenga/src/jenga/__init__.py
```

### Step 5 — GPU 自检

```bash
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
print("capability", torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (x @ x).sum().item())
PY
```

**判据**：`capability == (12, 0)`、matmul 不报错（证明 sm_120 kernel 生效）。

### Step 6 — 资源就位

```bash
cd $SMELL_V3/third_party/Jenga

# 6.1 解压三个 zip（先确认 zip 在 WSL 内的实际位置，$DOWNLOAD 替换之）
export DOWNLOAD=/path/to/download
unzip -q -o $DOWNLOAD/dataset.zip   -x "__MACOSX/*" -d .           # -> dataset/
mkdir -p checkpoints
unzip -q -o $DOWNLOAD/predictor.zip -x "__MACOSX/*" -d checkpoints # -> checkpoints/predictor/
unzip -q -o $DOWNLOAD/peft_model.zip -x "__MACOSX/*" -d checkpoints # -> checkpoints/peft_model/  （可延后）

# 6.2 下载 opt-350m（HF 直连慢时用 hf-mirror）
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "facebook/opt-350m",
    local_dir="checkpoints/opt-350m",
    allow_patterns=["config.json", "*.safetensors", "tokenizer*",
                    "vocab.json", "merges.txt", "special_tokens_map.json", "*.model"],
)
PY

# 6.3 检查
ls checkpoints/opt-350m/config.json checkpoints/predictor/predictor.pth
git -C $SMELL_V3/third_party/Jenga status --short   # 期望空（全部落 gitignore 目录）
```

> hello_world 的文件检查还要求 `checkpoints/{llama2,llama3,opt-1.3b,opt-2.7b,opt-6.7b}/config.json`、完整 peft_model、dataset 全部子目录。若用户未迁这些文件：opt-350m（必需）+ opt-1.3b/2.7b/6.7b 的 `config.json` 可从 HF 补；**llama2/llama3 是 gated 仓库**，且 llama2 权重本机没有（见 §5），不要为此花时间。

### Step 7 — 跑官方 hello_world（启动/文件检查）

```bash
cd $SMELL_V3/third_party/Jenga
python src/experiment/hello_world.py
```

**预期**（这台 8 GB 的 5060）：
- 脚本依次打印 base model / PEFT / dataset 检查结果；缺哪个文件会明确列出（这是**正常且有信息量**的输出）。
- `run_environment_compatibility_test()` 只在前面全过时执行；而它硬编码 **llama2-7B**（`hello_world.py:136-163`），bf16 权重 ≈13.5 GB **必然 OOM**，即使文件齐全。因此 hello_world 在本机只能验证「导入 + 环境 + 文件盘点」，不能作为 GPU 验收。
- 判定通过：脚本无异常跑完（返回码 0）、`import` 链路（peft/transformers/jenga）全通、缺失项符合预期。

### Step 8 — OPT-350M smoke（真正的验收）

1. 在 `$SMELL_V3/src/smoke_jenga_opt.py` 新建脚本（草案见附录 A；文件头必须带 `# SMELL 3 smoke_jenga_opt NEW — ...` 标注，遵循 `docs/standard.md`）。
2. 运行：

```bash
conda activate smell-v3
cd $SMELL_V3
python src/smoke_jenga_opt.py              # flash_attention_2 路径
python src/smoke_jenga_opt.py --no-flash   # eager 对照组
```

**判据**：两次都打印 `[smoke] PASS`；FA 路径下 `loss` 有限、`peak_vram < 8 GiB`、无 dtype/设备报错。

### Step 9 — 收尾

- 写结果 ailog：`docs/ailog/<新时间戳>-smell-v3-wsl-jenga-smoke-result.md`（环境版本 `pip freeze` 摘要、验证输出、失败点与修复、遗留风险）。
- 在 v3 根目录落 `requirements-wsl-cu128.txt`（本次实际安装版本）。
- 不要自动 commit（AGENTS.md：仅在用户要求时提交；若提交用 `SMELL v3: ...` 格式）。

---

## 5. hello_world 的硬伤（为什么需要 OPT smoke 兜底）

`third_party/Jenga/src/experiment/hello_world.py`：

- `check_base_models_exist()` 要求 6 个模型目录下都有 `config.json`（llama2/llama3/opt-350m/opt-1.3b/opt-2.7b/opt-6.7b）。
- `run_environment_compatibility_test()`（L102-213）只用 **llama2**：`get_llama_qk(..., flash_attention=True)` 加载 `LlamaForCausalLM`，`model.to('cuda')`，4096 token 前向+反向。
- 本机没有 llama2 权重（Windows HF cache 的 `models--meta-llama--Llama-2-7b-hf` 只有 `refs/main`，blobs 为 0），且 7B bf16 超出 8 GB 显存 → **无法在 RTX 5060 上完成**。
- 所以「验证 Jenga 能正确启动」拆成：Step 7（官方脚本的导入/环境/文件层）+ Step 8（OPT-350M 自定义模型的 CUDA 层）。若下个 session 发现用户其实提供了完整 llama2 权重且显存 ≥24 GB（例如换了机器），可把 Step 7 升级为完整跑通，并把 §3 版本按该机 GPU 调整。

---

## 6. 风险与备选

| 风险 | 处置 |
|---|---|
| torch cu128 下载慢/失败 | 换阿里云/上交 pytorch-wheels 镜像；或退回 **2.7.1+cu128 + FA 2.7.4.post1（cp310 torch2.7 wheel）**；记录实际版本 |
| FA wheel 下载失败 | 见 Step 3 备选（镜像 or 源码编译） |
| peft 0.13.2 与 4.45.2 不兼容 | 改 peft 0.14.0（仍与 4.45 兼容） |
| smoke OOM（8 GB） | 降 `--seq-len` 到 1024/512；设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Jenga 稀疏路径仅支持 batch=1、seq_len%64==0、单卡 | smoke 固定 `(1, 2048)`；不要多卡/变长尝试 |
| model_max_length > 2046 需位置嵌入复制 | smoke 只用 2048（`opt_jenga.py:88-97` 的复制逻辑留给后续长上下文阶段） |
| `/mnt/d` I/O 慢 | 用户已决定整体迁 WSL ext4；若不迁，则把 dataset/checkpoints 放 ext4 再软链 |
| 跑冒烟时改动 third_party | 禁止；所有新文件写 `src/`，Jenga 只读 |

---

## 7. 附录 A — OPT-350M smoke 脚本草案

```python
# SMELL 3 smoke_jenga_opt NEW — WSL/cu128 环境自检：Jenga OPT-350M + FA2 前向/反向
"""在 RTX 5060 (sm_120) 上验证 Jenga 自定义 OPT 实现可加载/前向/反向。

用法（在仓库根目录、conda env smell-v3 下）:
    python src/smoke_jenga_opt.py              # flash_attention_2
    python src/smoke_jenga_opt.py --no-flash   # eager 对照
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
JENGA_SRC = REPO / "third_party" / "Jenga" / "src"
if str(JENGA_SRC) not in sys.path:
    sys.path.insert(0, str(JENGA_SRC))

from jenga.models.modeling_opt import OPTForCausalLM  # noqa: E402
from jenga.utils.config_utils import get_opt_qk  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(REPO / "third_party" / "Jenga" / "checkpoints" / "opt-350m"))
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--no-flash", action="store_true")
    args = parser.parse_args()

    assert args.seq_len % 64 == 0, "Jenga 稀疏路径要求 seq_len 为 pool_size(64) 的整数倍"
    assert torch.cuda.is_available(), "CUDA 不可用"
    print(f"[smoke] torch={torch.__version__} cuda={torch.version.cuda} "
          f"cap={torch.cuda.get_device_capability()} dev={torch.cuda.get_device_name(0)}")

    flash = not args.no_flash
    config = get_opt_qk(model_name=args.model_dir, flash_attention=flash,
                        pool_size=64, thresh=0.4)
    print(f"[smoke] attn_implementation={config.attn_implementation} sparse={config.sparse}")

    model = OPTForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, config=config)
    model.to("cuda").train()
    print(f"[smoke] params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    vocab = int(getattr(config, "vocab_size", 50272))
    input_ids = torch.randint(0, vocab, (1, args.seq_len), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids, labels=input_ids).loss
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[smoke] loss={loss.item():.4f} step={dt:.2f}s peak_vram={peak:.2f}GiB")
    assert torch.isfinite(loss), "loss is not finite"
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
```

---

## 8. 附录 B — 本计划产生的结论/依据（速查）

- Windows 侧检查命令与输出：conda env 表、`nvidia-smi`、WSL `nvidia-smi`、FA release assets、Jenga `requirements.txt`、`hello_world.py`、`modeling_opt.py:46/138/295-413`、`config_utils.py:54-68`。
- 关键路径：`third_party/Jenga/src/experiment/hello_world.py`、`third_party/Jenga/src/jenga/models/modeling_opt.py`、`third_party/Jenga/src/experiment/end2end/time/opt_jenga.py`（OPT 加载/位置嵌入复制参考）。
- 规范：`docs/standard.md`（SMELL 标注、ailog 命名）；`AGENTS.md`（third_party 只读、ailog 必写、勿提交 checkpoints/dataset）。
