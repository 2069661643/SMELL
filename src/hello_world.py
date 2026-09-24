# SMELL 3 hello_world NEW — SMELL-v3 环境自检（Jenga hello_world 风格）

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JENGA_ROOT = REPO / "third_party" / "Jenga"
JENGA_SRC = JENGA_ROOT / "src"
for extra in (str(REPO), str(JENGA_SRC)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

DEFAULT_MODEL_DIR = JENGA_ROOT / "checkpoints" / "opt-350m"
EXPECTED_PYTHON = (3, 10)
EXPECTED_TORCH = "2.8.0"
EXPECTED_CUDA = "12.8"
EXPECTED_FLASH_ATTN = "2.8.3"
EXPECTED_CAPABILITY = (12, 0)
PINNED_DEPS = {
    "transformers": "4.45.2",
    "tokenizers": "0.20.1",
    "peft": "0.13.2",
    "accelerate": "1.0.1",
    "datasets": "2.21.0",
    "numpy": "1.26.4",
}
JENGA_IMPORTS = (
    ("jenga.models.modeling_opt.OPTForCausalLM", "jenga.models.modeling_opt", "OPTForCausalLM"),
    ("jenga.utils.config_utils.get_opt_qk", "jenga.utils.config_utils", "get_opt_qk"),
    ("jenga.models.predictor.PrunableAttnPredictor", "jenga.models.predictor", "PrunableAttnPredictor"),
)
SRC_MODULES = (
    "src.models.modeling_opt_smell",
    "src.models.position_embed",
    "src.models.token_selector",
    "src.train.lora",
    "src.train.zoo",
    "src.train.longctx_adapt",
    "src.train.train_predictor",
    "src.eval.ppl",
    "src.fed.serial_fedavg",
    "src.fed.run_fed",
)
JENGA_RESOURCES = (
    "checkpoints/predictor/predictor.pth",
    "checkpoints/predictor/pruned_config.pth",
    "checkpoints/peft_model/la/jenga/adapter_model.safetensors",
    "checkpoints/peft_model/la/lora/adapter_model.safetensors",
)
BASE_MODEL_CONFIGS = (
    "checkpoints/opt-1.3b/config.json",
    "checkpoints/opt-2.7b/config.json",
    "checkpoints/opt-6.7b/config.json",
    "checkpoints/llama2/config.json",
    "checkpoints/llama3/config.json",
)
JENGA_DATASETS = (
    "dataset/LongAlign",
    "dataset/PPL/proof_pile.bin",
    "dataset/RedPajama-Data-1T-Sample",
    "dataset/longbench",
)
SMELL_ARTIFACTS = (
    "dataset_v3/discovery_16k/a01/global_test_input_ids.npy",
    "dataset_v3/discovery_16k/a01/warmup_input_ids.npy",
    "dataset_v3/discovery_16k/a03/global_test_input_ids.npy",
    "dataset_v3/discovery_16k/a03/warmup_input_ids.npy",
)

RESULTS = []


def check(name, status, detail=""):
    print(f"[{status}] {name}: {detail}")
    RESULTS.append({"name": name, "status": status, "detail": detail})


def section(title):
    print(f"\n--- {title} ---")


def human_size(path):
    size = float(path.stat().st_size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


def check_python():
    section("Python / env")
    version = platform.python_version()
    if sys.version_info >= EXPECTED_PYTHON:
        check("python.version", "OK", version)
    else:
        check("python.version", "FAIL", f"{version} (>= 3.10 required)")
    try:
        import torch
    except Exception as exc:
        check("torch.import", "FAIL", f"{type(exc).__name__}: {exc}")
        return
    check("torch.import", "OK", f"{torch.__version__} cuda={torch.version.cuda}")
    if torch.__version__.startswith(EXPECTED_TORCH) and str(torch.version.cuda) == EXPECTED_CUDA:
        check("torch.version", "OK", f"{EXPECTED_TORCH}+cu{EXPECTED_CUDA.replace('.', '')}")
    else:
        check("torch.version", "WARN",
              f"{torch.__version__} cuda={torch.version.cuda} "
              f"(expected {EXPECTED_TORCH}+cu{EXPECTED_CUDA.replace('.', '')})")


def check_flash_attn():
    section("flash-attn")
    try:
        import flash_attn
    except Exception as exc:
        check("flash_attn.import", "FAIL", f"{type(exc).__name__}: {exc}")
        return
    version = getattr(flash_attn, "__version__", "unknown")
    if version == EXPECTED_FLASH_ATTN:
        check("flash_attn.version", "OK", version)
    else:
        check("flash_attn.version", "WARN", f"{version} (expected {EXPECTED_FLASH_ATTN})")


def check_pinned_deps():
    section("Pinned deps")
    for module_name, expected in PINNED_DEPS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            check(f"dep.{module_name}", "FAIL", f"import failed: {type(exc).__name__}: {exc}")
            continue
        version = getattr(module, "__version__", "unknown")
        if version == expected:
            check(f"dep.{module_name}", "OK", version)
        else:
            check(f"dep.{module_name}", "WARN", f"{version} (expected {expected})")


def check_jenga():
    section("Jenga")
    try:
        import jenga
        check("jenga", "OK", str(getattr(jenga, "__file__", None) or "namespace package"))
    except Exception as exc:
        check("jenga", "FAIL", f"{type(exc).__name__}: {exc}")
    for name, module_name, attribute in JENGA_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            getattr(module, attribute)
            check(name, "OK", "imported")
        except Exception as exc:
            check(name, "FAIL", f"{type(exc).__name__}: {exc}")


def check_src_modules():
    section("SMELL src modules")
    for module_name in SRC_MODULES:
        try:
            module = importlib.import_module(module_name)
            path = getattr(module, "__file__", None)
            check(module_name, "OK", str(Path(path).relative_to(REPO)) if path else "imported")
        except Exception as exc:
            check(module_name, "FAIL", f"{type(exc).__name__}: {exc}")


def check_gpu():
    section("GPU")
    try:
        import torch
        available = torch.cuda.is_available()
    except Exception as exc:
        check("gpu.cuda", "FAIL", f"{type(exc).__name__}: {exc}")
        return
    if not available:
        check("gpu.cuda", "FAIL", "torch.cuda.is_available() is False")
        return
    check("gpu.cuda", "OK", torch.cuda.get_device_name(0))
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability == EXPECTED_CAPABILITY:
        check("gpu.capability", "OK", f"sm_{capability[0]}{capability[1]}")
    else:
        check("gpu.capability", "WARN",
              f"{capability}; expected {EXPECTED_CAPABILITY} locally (A40 is (8, 6))")
    try:
        left = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        right = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        product = left @ right
        finite = bool(torch.isfinite(product).all().item())
        check("gpu.bf16_matmul", "OK" if finite else "FAIL",
              f"1024x1024 bf16 matmul {'finite' if finite else 'non-finite'}")
    except Exception as exc:
        check("gpu.bf16_matmul", "FAIL", f"{type(exc).__name__}: {exc}")


def check_base_model(model_dir):
    section("Base model")
    config_path = model_dir / "config.json"
    weights = [candidate.name for candidate in
               (model_dir / "pytorch_model.bin", model_dir / "model.safetensors")
               if candidate.exists()]
    if config_path.exists() and weights:
        check("model.files", "OK", f"config.json + {weights[0]} in {model_dir}")
    else:
        missing = []
        if not config_path.exists():
            missing.append("config.json")
        if not weights:
            missing.append("pytorch_model.bin|model.safetensors")
        check("model.files", "FAIL", f"missing {', '.join(missing)} in {model_dir}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        check("model.tokenizer", "OK", f"{type(tokenizer).__name__} vocab={tokenizer.vocab_size}")
    except Exception as exc:
        check("model.tokenizer", "FAIL", f"{type(exc).__name__}: {exc}")
    try:
        from jenga.utils.config_utils import get_opt_qk
        config = get_opt_qk(model_name=str(model_dir), flash_attention=True, pool_size=64, thresh=0.4)
        check("model.get_opt_qk", "OK",
              f"attn={config.attn_implementation} pool={config.pool_size} sparse={config.sparse}")
        return config
    except Exception as exc:
        check("model.get_opt_qk", "FAIL", f"{type(exc).__name__}: {exc}")
        return None


def check_forward(model_dir, config, seq_len, ctx):
    section("Forward smoke")
    if config is None:
        check("forward.smoke", "FAIL", "skipped: get_opt_qk config unavailable")
        return
    try:
        import warnings

        import torch
        from transformers import logging as hf_logging
        from jenga.models.modeling_opt import OPTForCausalLM
        from src.models.position_embed import ensure_positions

        hf_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = OPTForCausalLM.from_pretrained(
                str(model_dir), torch_dtype=torch.bfloat16, config=config)
            model = ensure_positions(model, ctx)
            model = model.cuda().eval()
            vocab_size = int(getattr(config, "vocab_size", 50272))
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device="cuda")
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids, labels=input_ids).loss
        if loss is not None and bool(torch.isfinite(loss).item()):
            check("forward.smoke", "OK", f"seq_len={seq_len} ctx={ctx} loss={float(loss):.4f}")
        else:
            check("forward.smoke", "FAIL", f"non-finite loss: {loss}")
        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        check("forward.smoke", "FAIL", f"{type(exc).__name__}: {exc}")


def check_inventory():
    section("Inventory (WARN only)")
    groups = (
        ("inventory.jenga_resources", JENGA_ROOT, JENGA_RESOURCES, False),
        ("inventory.base_model_configs", JENGA_ROOT, BASE_MODEL_CONFIGS, False),
        ("inventory.datasets", JENGA_ROOT, JENGA_DATASETS, False),
        ("inventory.smell_artifacts", REPO, SMELL_ARTIFACTS, True),
    )
    for name, base, relative_paths, with_sizes in groups:
        found, missing = [], []
        for relative in relative_paths:
            path = base / relative
            if path.exists():
                found.append(f"{relative}={human_size(path)}" if with_sizes else relative)
            else:
                missing.append(relative)
        if not missing:
            detail = f"all {len(relative_paths)} present"
            if with_sizes:
                detail += ": " + ", ".join(found)
            check(name, "OK", detail)
        else:
            detail = f"missing {len(missing)}/{len(relative_paths)}: {', '.join(missing)}"
            if found:
                detail += f"; found: {', '.join(found)}"
            check(name, "WARN", detail)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SMELL-v3 environment checker (Jenga hello_world style); relative paths resolve against the repo root")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=16384, help="position-extension target")
    parser.add_argument("--no-gpu", action="store_true", help="skip GPU and forward checks")
    parser.add_argument("--json", default=None, help="path to write machine-readable JSON results")
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO / path
    return path


def main():
    started = time.time()
    args = parse_args()
    print("=" * 54)
    print(" SMELL-v3 Environment Checker (Jenga hello_world style)")
    print("=" * 54)
    try:
        check_python()
        check_flash_attn()
        check_pinned_deps()
        check_jenga()
        check_src_modules()
        if not args.no_gpu:
            check_gpu()
        model_dir = resolve_path(args.model_dir)
        config = check_base_model(model_dir)
        if not args.no_gpu:
            check_forward(model_dir, config, args.seq_len, args.ctx)
        check_inventory()
    except Exception as exc:
        check("hello_world.unhandled", "FAIL", f"{type(exc).__name__}: {exc}")

    section("Summary")
    ok = sum(1 for item in RESULTS if item["status"] == "OK")
    warn = sum(1 for item in RESULTS if item["status"] == "WARN")
    fail = sum(1 for item in RESULTS if item["status"] == "FAIL")
    print(f"checks={len(RESULTS)}  OK={ok}  WARN={warn}  FAIL={fail}  "
          f"elapsed={time.time() - started:.1f}s")
    if fail == 0:
        print("Congratulations! SMELL-v3 environment appears to be set up correctly.")
    else:
        print("Setup Incomplete: review FAIL lines above.")

    if args.json:
        out_path = resolve_path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"checks": RESULTS, "ok": ok, "warn": warn, "fail": fail}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[hello_world] wrote {out_path}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
