# SMELL 3 memory NEW — 16k 显存矩阵：Jenga sparse/dense × LoRA attn/+ffn × grad ckpt

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
for extra in (str(REPO), str(REPO / "third_party" / "Jenga" / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

DEFAULT_MODEL_DIR = REPO / "third_party" / "Jenga" / "checkpoints" / "opt-350m"
SPARSE_THRESH = 0.4
LORA_TARGETS = {
    "qkvo": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "qkvo+fc": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="16k memory matrix for sparse/dense + LoRA + ckpt")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--matrix", action="store_true",
                        help="run all 8 combos (default; kept for explicit CLI)")
    parser.add_argument("--only", default=None,
                        help='filter, e.g. "sparse=1,dense=0;lora=attn;ckpt=1"')
    parser.add_argument("--cap-fraction", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--mode", choices=("forward", "backward"), default="forward",
                        help="forward = ZOO 主线显存；backward = BP 对照（16k 8GB 会 OOM）")
    parser.add_argument("--out", default=None, help="JSON output path")
    return parser.parse_args()


def build_combos():
    return [
        {"sparse": sparse, "lora": lora, "ckpt": ckpt}
        for sparse in (True, False)
        for lora in ("qkvo", "qkvo+fc")
        for ckpt in (False, True)
    ]


def parse_only(text):
    groups = []
    for group in text.split(";"):
        conditions = []
        for item in group.split(","):
            if not item.strip():
                continue
            key, _, value = item.partition("=")
            conditions.append((key.strip().lower(), value.strip().lower()))
        if conditions:
            groups.append(conditions)
    return groups


def _as_bool(value):
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"expected 0/1, got '{value}'")


def condition_match(combo, key, value):
    if key == "sparse":
        return combo["sparse"] == _as_bool(value)
    if key == "dense":
        return (not combo["sparse"]) == _as_bool(value)
    if key == "ckpt":
        return combo["ckpt"] == _as_bool(value)
    if key == "lora":
        aliases = {
            "attn": "qkvo", "qkvo": "qkvo",
            "ffn": "qkvo+fc", "fc": "qkvo+fc", "mlp": "qkvo+fc",
            "attn+ffn": "qkvo+fc", "qkvo+ffn": "qkvo+fc", "qkvo+fc": "qkvo+fc",
            "attn+fc": "qkvo+fc", "all": "qkvo+fc",
        }
        wanted = value.replace(" ", "")
        return aliases.get(wanted, wanted) == combo["lora"]
    raise SystemExit(f"unknown --only key: {key}")


def combo_matches(combo, groups):
    # SMELL 3 memory FIXED — 组内 AND、组间 OR（首版写成组内 OR，--only 失效）
    return any(all(condition_match(combo, key, value) for key, value in group) for group in groups)


def run_combo(combo, args):
    from jenga.models.modeling_opt import OPTForCausalLM
    from jenga.utils.config_utils import get_opt_qk
    from src.models.position_embed import ensure_positions
    from src.train.lora import build_lora_model

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    thresh = SPARSE_THRESH if combo["sparse"] else 1.0
    config = get_opt_qk(model_name=args.model_dir, flash_attention=True, pool_size=64, thresh=thresh)
    model = OPTForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, config=config)
    # SMELL 3 position_embed ADD — seq_len 可能超过 OPT 原生 2048 位置表
    model = ensure_positions(model, args.seq_len)
    model = build_lora_model(model, r=8, targets=LORA_TARGETS[combo["lora"]])
    if combo["ckpt"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = model.cuda().train()

    input_ids = torch.randint(0, int(config.vocab_size), (1, args.seq_len), device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, _ = torch.cuda.mem_get_info()
    started = time.time()
    loss = None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(args.steps):
            loss = model(input_ids, labels=input_ids).loss
            if args.mode == "backward":
                loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    free_after, _ = torch.cuda.mem_get_info()

    record = dict(combo)
    record.update({
        "status": "ok",
        "mode": args.mode,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "loss": float(loss.detach().cpu()),
        "peak_allocated_gb": peak_allocated / 1024 ** 3,
        "free_before_gb": free_before / 1024 ** 3,
        "free_after_gb": free_after / 1024 ** 3,
        "elapsed_s": elapsed,
    })
    del model, input_ids, loss
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.cuda.set_per_process_memory_fraction(args.cap_fraction, 0)
    combos = build_combos()
    if args.only:
        groups = parse_only(args.only)
        combos = [combo for combo in combos if combo_matches(combo, groups)]
    if not combos:
        raise SystemExit("no combos selected")

    print(f"[memory] seq_len={args.seq_len} steps={args.steps} cap_fraction={args.cap_fraction} "
          f"combos={len(combos)}")
    print(f"{'sparse':<8}{'lora':<9}{'ckpt':<6}{'status':<10}"
          f"{'peak_GiB':>10}{'free_before':>13}{'free_after':>12}{'time_s':>9}")
    records = []
    for combo in combos:
        try:
            record = run_combo(combo, args)
        except Exception as exc:  # noqa: BLE001 — keep the matrix alive after OOM/errors
            record = dict(combo)
            record.update({"status": f"error: {exc}"[:160], "seq_len": args.seq_len,
                           "steps": args.steps, "peak_allocated_gb": None,
                           "free_before_gb": None, "free_after_gb": None, "elapsed_s": None})
            gc.collect()
            torch.cuda.empty_cache()
        records.append(record)
        print(f"[memory] {combo['sparse']} {combo['lora']} ckpt={combo['ckpt']} -> "
              f"{record['status'][:120]}", flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"[memory] wrote {out_path}")

    def cell(value, digits=2):
        # SMELL 3 memory FIXED — None 字段不能直接套格式符（首版在此崩溃）
        return "-" if value is None else round(value, digits)

    print(f"{'sparse':<8}{'lora':<9}{'ckpt':<6}{'status':<10}"
          f"{'peak_GiB':>10}{'free_before':>13}{'free_after':>12}{'time_s':>9}")
    for record in records:
        print(f"{str(record['sparse']):<8}{record['lora']:<9}{str(record['ckpt']):<6}"
              f"{str(record['status'])[:9]:<10}"
              f"{cell(record['peak_allocated_gb']):>10}"
              f"{cell(record['free_before_gb']):>13}"
              f"{cell(record['free_after_gb']):>12}"
              f"{cell(record['elapsed_s']):>9}")


if __name__ == "__main__":
    main()
