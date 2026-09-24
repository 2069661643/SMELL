# SMELL 3 longctx_adapt NEW — 长上下文 BP 预热训练器（pos_only B 臂 / pos_lora C 臂），位置表扩展 + 全序列/尾窗 PPL 诊断
#
# 背景：OPT 学习式绝对位置嵌入零样本无法外推（docs/ailog/260924-103741：2048 PPL 20.2 → 16384 PPL 588）。
# 本脚本用反向传播适配 embed_positions（可选叠加 LoRA qkvo r=8），为后续 ZOO 联邦长上下文实验准备权重。
# 用法示例：
#   python src/train/longctx_adapt.py --mode pos_only --pos-init interpolate --steps 500 --gpu 0
#
# SMELL 3 longctx_adapt act_pack ADD — 默认旁路 Jenga modeling_opt 的激活 pack/unpack hooks：
#   pack_hook 只保存 activation 输出前一半行、unpack 用 0 补齐 → BP 反传到后一半 token 的 MLP 梯度被静默清零。
#   该 hook 不影响前向数值（ZOO/评测无感），本训练器默认 bypass 以保证 BP 梯度正确；--act-pack on 恢复 Jenga 原行为。

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MODEL_DIR = REPO / "third_party" / "Jenga" / "checkpoints" / "opt-350m"
DEFAULT_DATA_ROOT = REPO / "dataset_v3" / "discovery_16k"
POS_SUFFIX = "embed_positions.weight"


# SMELL 3 longctx_adapt gpu_preparse BEGIN — --gpu 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES
def _preparse_gpu():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default=None)
    known, _ = parser.parse_known_args()
    if known.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_preparse_gpu()
# SMELL 3 longctx_adapt gpu_preparse END

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jenga.utils.config_utils import get_opt_qk  # noqa: E402
from src.eval.ppl import get_decoder, get_lm_head, sequence_nll  # noqa: E402
from src.models.modeling_opt_smell import OPTForCausalLM  # noqa: E402
from src.models.position_embed import ensure_positions  # noqa: E402
from src.train.lora import build_lora_model, count_trainable  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Long-context BP warmup trainer (pos_only / pos_lora)")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--data", default=None,
                        help="train input_ids npy [N, L]; default dataset_v3/discovery_16k/<tag>/warmup_input_ids.npy")
    parser.add_argument("--tag", default="a01")
    parser.add_argument("--eval-data", default=None,
                        help="eval input_ids npy [N, L]; default dataset_v3/discovery_16k/<tag>/global_test_input_ids.npy")
    parser.add_argument("--mode", choices=("pos_only", "pos_lora"), default="pos_only")
    parser.add_argument("--pos-init", choices=("interpolate", "duplicate", "jenga_dup_scaled"),
                        default="interpolate")
    parser.add_argument("--sparse", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--truncate", type=int, default=0, help="smoke only: use first N tokens; 0 = full seq")
    parser.add_argument("--eval-every", type=int, default=50, help="0 = no periodic eval (baseline + final still run)")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--tail-window", type=int, default=2048)
    parser.add_argument("--save-dir", default=None, help="default logs/longctx/<timestamp>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--log-every", type=int, default=1)
    # SMELL 3 longctx_adapt act_pack ADD — off = 旁路 Jenga 半丢弃 hooks（正确 BP）；on = Jenga 原行为
    parser.add_argument("--act-pack", choices=("off", "on"), default="off")
    args = parser.parse_args()
    if args.data is None:
        args.data = str(DEFAULT_DATA_ROOT / args.tag / "warmup_input_ids.npy")
    if args.eval_data is None:
        args.eval_data = str(DEFAULT_DATA_ROOT / args.tag / "global_test_input_ids.npy")
    if args.save_dir is None:
        args.save_dir = str(REPO / "logs" / "longctx" / time.strftime("%y%m%d-%H%M%S"))
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise SystemExit("--batch-size/--grad-accum must be >= 1")
    if not 0.0 < args.sparse <= 1.0:
        raise SystemExit(f"--sparse must be in (0, 1], got {args.sparse}")
    return args


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def load_ids(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing data file: {path}")
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise SystemExit(f"expected 2D [N, L] input_ids at {path}, got shape {array.shape}")
    return array


def row_to_tensor(array, index, truncate, device):
    row = np.asarray(array[index], dtype=np.int64)
    if truncate > 0:
        row = row[:truncate]
    return torch.from_numpy(row).unsqueeze(0).to(device)


class ShuffledStream:
    # SMELL 3 longctx_adapt ShuffledStream NEW — 带种子的确定性样本流（epoch 级 reshuffle）
    def __init__(self, num_samples, seed):
        self.num_samples = int(num_samples)
        self.rng = np.random.default_rng(seed)
        self.order = None
        self.cursor = 0

    def next_index(self):
        if self.order is None or self.cursor >= self.order.size:
            self.order = self.rng.permutation(self.num_samples)
            self.cursor = 0
        index = int(self.order[self.cursor])
        self.cursor += 1
        return index


# SMELL 3 longctx_adapt act_pack BEGIN — 默认把 Jenga pack/unpack hooks 换成恒等映射（保 BP 梯度正确）
def configure_activation_pack(mode):
    if mode == "on":
        return "jenga"
    import src.models.modeling_opt_smell as modeling
    modeling.pack_hook = lambda tensor: tensor
    modeling.unpack_hook = lambda tensor: tensor
    return "bypassed"
# SMELL 3 longctx_adapt act_pack END


def build_model(args, pos_len):
    config = get_opt_qk(model_name=args.model_dir, flash_attention=True, pool_size=64, thresh=args.sparse)
    model = OPTForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, config=config)
    model = ensure_positions(model, pos_len, mode=args.pos_init)
    for param in model.parameters():
        param.requires_grad_(False)
    if args.mode == "pos_lora":
        model = build_lora_model(model)
    pos_param = None
    for name, param in model.named_parameters():
        if name.endswith(POS_SUFFIX):
            param.requires_grad_(True)
            pos_param = param
    if pos_param is None:
        raise RuntimeError(f"cannot find trainable {POS_SUFFIX}")
    return model, pos_param


def collect_trainable_names(model):
    return sorted(name for name, param in model.named_parameters() if param.requires_grad)


def validate_trainable_names(names, mode):
    pos_names = [name for name in names if name.endswith(POS_SUFFIX)]
    lora_names = [name for name in names if "lora_" in name]
    expected = set(pos_names) | set(lora_names)
    other = [name for name in names if name not in expected]
    assert len(pos_names) == 1, f"expected exactly one trainable {POS_SUFFIX}, got {pos_names}"
    assert not other, f"unexpected trainable params: {other[:8]}"
    if mode == "pos_only":
        assert not lora_names, f"pos_only must not contain LoRA params: {lora_names[:5]}"
    else:
        assert lora_names, "pos_lora expected lora_* trainable params"
    return pos_names, lora_names


def evaluate(model, eval_array, num_samples, truncate, tail_window, device):
    model.eval()
    full_sum = 0.0
    full_count = 0
    tail_sum = 0.0
    tail_count = 0
    used = 0
    started = time.time()
    with torch.no_grad():
        for index in range(num_samples):
            input_ids = row_to_tensor(eval_array, index, truncate, device)
            if input_ids.size(1) < 2:
                continue
            nll = sequence_nll(model, input_ids)
            if nll.numel() == 0:
                continue
            full_sum += float(nll.sum())
            full_count += int(nll.numel())
            tail = nll[-tail_window:] if tail_window > 0 else nll
            tail_sum += float(tail.sum())
            tail_count += int(tail.numel())
            used += 1
    model.train()
    return {
        "samples": used,
        "scored_tokens": full_count,
        "tail_tokens": tail_count,
        "ppl_full": math.exp(full_sum / full_count) if full_count else None,
        "ppl_tail": math.exp(tail_sum / tail_count) if tail_count else None,
        "seconds": time.time() - started,
    }


def print_eval_row(step, stats):
    ppl_full = f"{stats['ppl_full']:.4f}" if stats["ppl_full"] is not None else "nan"
    ppl_tail = f"{stats['ppl_tail']:.4f}" if stats["ppl_tail"] is not None else "nan"
    print(f"[eval] step={step} samples={stats['samples']} scored_tokens={stats['scored_tokens']} "
          f"tail_tokens={stats['tail_tokens']} ppl_full={ppl_full} ppl_tail={ppl_tail} "
          f"time={stats['seconds']:.1f}s")


def grad_norm_if_clipped(trainable_params, max_grad_norm):
    if max_grad_norm > 0:
        return float(torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm))
    total = 0.0
    for param in trainable_params:
        if param.grad is not None:
            total += float((param.grad.detach().float() ** 2).sum())
    return math.sqrt(total)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.model_dir = str(resolve(args.model_dir))
    data_path = resolve(args.data)
    eval_path = resolve(args.eval_data)
    save_dir = resolve(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    train_array = load_ids(data_path)
    eval_array = load_ids(eval_path)
    train_full_len = int(train_array.shape[1])
    eval_full_len = int(eval_array.shape[1])
    # SMELL 3 longctx_adapt position_len ADD — 位置表按数据全长扩展（truncate 仅截断训练/评测 token，不影响容量）
    pos_len = max(train_full_len, eval_full_len)
    train_len = min(args.truncate, train_full_len) if args.truncate > 0 else train_full_len
    eval_len = min(args.truncate, eval_full_len) if args.truncate > 0 else eval_full_len
    eval_count = min(args.eval_samples, int(eval_array.shape[0])) if args.eval_samples > 0 else int(eval_array.shape[0])

    pack_mode = configure_activation_pack(args.act_pack)
    print(f"[longctx] act_pack={args.act_pack} ({pack_mode})")

    model, pos_param = build_model(args, pos_len)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_grads):
            enable_input_grads()

    names = collect_trainable_names(model)
    pos_names, lora_names = validate_trainable_names(names, args.mode)
    trainable = count_trainable(model)
    decoder = get_decoder(model)
    lm_head = get_lm_head(model)
    print(f"[longctx] decoder={type(decoder).__name__} lm_head={type(lm_head).__name__} "
          f"pos_rows={int(decoder.embed_positions.num_embeddings) - int(decoder.embed_positions.offset)}")
    print(f"[longctx] mode={args.mode} pos_init={args.pos_init} trainable_params={trainable} "
          f"trainable_tensors={len(names)} pos_tensors={len(pos_names)} lora_tensors={len(lora_names)}")
    print(f"[longctx] trainable first={names[:3]}")
    print(f"[longctx] trainable last={names[-3:]}")
    print(f"[longctx] data={data_path} (n={train_array.shape[0]} len={train_full_len} used={train_len})")
    print(f"[longctx] eval={eval_path} (n={eval_array.shape[0]} len={eval_full_len} used={eval_len} "
          f"samples={eval_count}) save_dir={save_dir}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    model.train()

    config_payload = {
        **vars(args),
        "resolved_model_dir": args.model_dir,
        "resolved_data": str(data_path),
        "resolved_eval_data": str(eval_path),
        "resolved_save_dir": str(save_dir),
        "train_full_seq_len": train_full_len,
        "eval_full_seq_len": eval_full_len,
        "position_table_len": pos_len,
        "train_seq_len": train_len,
        "eval_seq_len": eval_len,
        "mode": args.mode,
        "pos_init": args.pos_init,
        "trainable_params": trainable,
        "trainable_tensors": len(names),
        "trainable_names": names,
        "act_pack": pack_mode,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    (save_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    started = time.time()
    metrics_handle = metrics_path.open("w", encoding="utf-8")

    def log_metrics(step, loss, grad_norm, ppl_full, ppl_tail):
        record = {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "elapsed_s": time.time() - started,
            "ppl_full": ppl_full,
            "ppl_tail": ppl_tail,
        }
        metrics_handle.write(json.dumps(record) + "\n")
        metrics_handle.flush()
        return record

    last_eval = None
    if eval_count > 0:
        last_eval = evaluate(model, eval_array, eval_count, args.truncate, args.tail_window, device)
        log_metrics(0, None, None, last_eval["ppl_full"], last_eval["ppl_tail"])
        print("[eval] baseline")
        print_eval_row(0, last_eval)

    stream = ShuffledStream(int(train_array.shape[0]), args.seed)
    micro_batches = args.batch_size * args.grad_accum
    last_loss = None
    last_grad_norm = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for _ in range(micro_batches):
            index = stream.next_index()
            input_ids = row_to_tensor(train_array, index, args.truncate, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                loss = outputs.loss
            (loss / micro_batches).backward()
            step_losses.append(float(loss.detach().float()))
        last_grad_norm = grad_norm_if_clipped(trainable_params, args.max_grad_norm)
        optimizer.step()
        last_loss = sum(step_losses) / len(step_losses)

        eval_stats = None
        if eval_count > 0 and (args.eval_every > 0 and step % args.eval_every == 0 or step == args.steps):
            eval_stats = evaluate(model, eval_array, eval_count, args.truncate, args.tail_window, device)
            last_eval = eval_stats

        if args.log_every > 0 and (step % args.log_every == 0 or step == args.steps):
            print(f"[longctx] step {step}/{args.steps} loss={last_loss:.6f} "
                  f"grad_norm={last_grad_norm:.4e} elapsed={time.time() - started:.1f}s")
        if eval_stats is not None:
            print_eval_row(step, eval_stats)
        log_metrics(step, last_loss, last_grad_norm,
                    eval_stats["ppl_full"] if eval_stats else None,
                    eval_stats["ppl_tail"] if eval_stats else None)

    torch.save(pos_param.detach().to(device="cpu", dtype=torch.bfloat16).clone(), save_dir / "pos_embed.pt")
    if args.mode == "pos_lora":
        model.save_pretrained(str(save_dir / "adapter"))
    metrics_handle.close()

    summary = {
        "mode": args.mode,
        "pos_init": args.pos_init,
        "steps": args.steps,
        "trainable_params": trainable,
        "final_train_loss": last_loss,
        "final_grad_norm": last_grad_norm,
        "final_ppl_full": last_eval["ppl_full"] if last_eval else None,
        "final_ppl_tail": last_eval["ppl_tail"] if last_eval else None,
        "metrics": str(metrics_path),
        "pos_embed": str(save_dir / "pos_embed.pt"),
    }
    if args.mode == "pos_lora":
        summary["adapter"] = str(save_dir / "adapter")
    print("[longctx] ===== summary =====")
    for key, value in summary.items():
        print(f"[longctx] {key}: {value}")
    print(f"[longctx] done steps={args.steps} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
