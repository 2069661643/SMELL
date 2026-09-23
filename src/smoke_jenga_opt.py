# SMELL 3 smoke_jenga_opt NEW — WSL/cu128 环境自检：Jenga OPT-350M + FA2 前向/反向
"""在 RTX 5060 (sm_120) 上验证 Jenga 自定义 OPT 实现可加载/前向/反向。

用法（在仓库根目录、conda env SMELL 下）:
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
