# SMELL 3 train_predictor NEW — Jenga predictor 单机训练器（OPT-350M，Discovery 16k）
#
# 只训练每个 decoder layer 的 PrunableAttnPredictor 的 q/k 线性分支（`predictor.klinear*` /
# `predictor.qlinear*`），冻结 base 权重、位置嵌入与可选 PEFT adapter。
# loss = mean over layers of SmoothL1(predict_mask, pooling_gt)；可选 Jenga 动态剪枝
# （每 --prune-interval 步，zero_ratio_threshold 0.8 -> 0.75 -> ...，step < --prune-until）。
# 位置表复用 src/models/position_embed.ensure_positions，可叠加 longctx_adapt 产出的
# pos_embed.pt / adapter/。产物：predictor.pth + pruned_config.pth + config.json + metrics.jsonl。
#
# 用法示例：
#   python src/train/train_predictor.py --tag a01 --pos-init interpolate --steps 400 --gpu 0
#   python src/train/train_predictor.py --data temp/mini_dataset/a01/global_test_input_ids.npy \
#       --truncate 512 --steps 4 --no-prune --eval-every 2 --gpu 0 --save-dir temp/predictor_smoke

import argparse
import json
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
BLOCK_SIZE = 64
PRUNE_ZERO_RATIO = 0.8
PRUNE_DECAY = 0.05
PRUNE_SCOPE_MAX_STEP = 620


# SMELL 3 train_predictor gpu_preparse BEGIN — --gpu 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES
def _preparse_gpu():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default=None)
    known, _ = parser.parse_known_args()
    if known.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_preparse_gpu()
# SMELL 3 train_predictor gpu_preparse END

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import jenga.models.modeling_opt_train_predictor as jenga_train_predictor_module  # noqa: E402
from jenga.models.modeling_opt_train_predictor import (  # noqa: E402
    OPTForCausalLM as OPTTrainPredictorForCausalLM,
    OPTLearnedPositionalEmbedding as OPTTrainPositionalEmbedding,
)
from jenga.models.predictor import PrunableAttnPredictor  # noqa: E402
from jenga.utils.config_utils import get_opt_qk  # noqa: E402
from src.models.position_embed import ensure_positions  # noqa: E402


# SMELL 3 train_predictor block_attn_pool FIXED — third_party Triton 版把 HEAD_DIM 硬编码为 128，
# OPT-350M head_dim=64 时会越界读相邻行：pooling_gt 错误（实测 diff≈20），并偶发 CUDA illegal
# memory access / loss=inf。此处替换为等价分块 torch 实现（sum_h ReLU(QK^T) -> 64x64 max-pool -> /64），
# 对任意 head_dim 正确；只在 train_predictor 内 monkeypatch，不改 third_party。
def block_attn_pool_fixed(q, k, block=BLOCK_SIZE):
    bsz, _, n_ctx, _ = q.shape
    if n_ctx % block != 0:
        raise ValueError(f"sequence length {n_ctx} must be divisible by {block}")
    n_blocks = n_ctx // block
    out = torch.empty((bsz, n_blocks, n_blocks), dtype=q.dtype, device=q.device)
    with torch.no_grad():
        for row in range(n_blocks):
            q_block = q[:, :, row * block:(row + 1) * block, :]
            attn = torch.matmul(q_block, k).float()
            attn.relu_()
            attn = attn.sum(dim=1)
            pooled = attn.view(bsz, block, n_blocks, block).amax(dim=(1, 3)) / float(block)
            out[:, row, :] = pooled.to(out.dtype)
    return out


jenga_train_predictor_module.block_attn_pool = block_attn_pool_fixed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Jenga predictor (q/k linears) for OPT-350M")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tag", default="a01")
    parser.add_argument("--data", default=None,
                        help="train input_ids npy [N, L]; default dataset_v3/discovery_16k/<tag>/warmup_input_ids.npy")
    parser.add_argument("--pos-init", choices=("interpolate", "duplicate", "jenga_dup_scaled"),
                        default="interpolate")
    parser.add_argument("--pos-checkpoint", default=None,
                        help="pos_embed.pt tensor from src/train/longctx_adapt.py")
    parser.add_argument("--adapter", default=None, help="PEFT adapter dir (e.g. pos_lora adapter/)")
    parser.add_argument("--sparse", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--truncate", type=int, default=0,
                        help="smoke only: use first N tokens (multiple of 64); 0 = full seq")
    parser.add_argument("--prune-interval", type=int, default=100)
    parser.add_argument("--prune-until", type=int, default=PRUNE_SCOPE_MAX_STEP)
    parser.add_argument("--no-prune", action="store_true", help="disable dynamic pruning (smoke)")
    parser.add_argument("--eval-every", type=int, default=50, help="0 = no periodic eval")
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--eval-data", default=None,
                        help="eval input_ids npy [N, L]; default = first --eval-samples rows of --data")
    parser.add_argument("--save-dir", default=None, help="default logs/predictor/<timestamp>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    if args.data is None:
        args.data = str(DEFAULT_DATA_ROOT / args.tag / "warmup_input_ids.npy")
    if args.save_dir is None:
        args.save_dir = str(REPO / "logs" / "predictor" / time.strftime("%y%m%d-%H%M%S"))
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not 0.0 < args.sparse <= 1.0:
        raise SystemExit(f"--sparse must be in (0, 1], got {args.sparse}")
    if args.truncate < 0:
        raise SystemExit("--truncate must be >= 0")
    if args.truncate > 0 and args.truncate % BLOCK_SIZE != 0:
        raise SystemExit(f"--truncate must be a multiple of {BLOCK_SIZE}, got {args.truncate}")
    if args.prune_interval < 0:
        raise SystemExit("--prune-interval must be >= 0")
    if args.eval_every > 0 and args.eval_samples < 1:
        raise SystemExit("--eval-samples must be >= 1 when --eval-every > 0")
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
    # SMELL 3 train_predictor ShuffledStream COPY — 带种子的确定性样本流（与 longctx_adapt 一致）
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


# SMELL 3 train_predictor align_position_class ADD — ensure_positions 注入 modeling_opt 的位置类不接受
# position_ids 关键字，而 train-predictor decoder 会传该参数；换回 train-predictor 的位置类并保留权重。
def align_position_class(model):
    decoder = model.model.decoder
    old = decoder.embed_positions
    if isinstance(old, OPTTrainPositionalEmbedding):
        return decoder
    new = OPTTrainPositionalEmbedding(int(old.num_embeddings) - int(old.offset), old.embedding_dim)
    new = new.to(device=old.weight.device)
    with torch.no_grad():
        new.weight.copy_(old.weight)
    new.weight.requires_grad_(bool(old.weight.requires_grad))
    decoder.embed_positions = new
    print("[predictor] position class swapped modeling_opt -> modeling_opt_train_predictor")
    return decoder


def build_model(args, pos_len):
    config = get_opt_qk(model_name=args.model_dir, flash_attention=True, pool_size=BLOCK_SIZE,
                        thresh=args.sparse)
    model = OPTTrainPredictorForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, config=config)
    model = ensure_positions(model, pos_len, mode=args.pos_init)
    decoder = align_position_class(model)
    pos_weight = decoder.embed_positions.weight
    if args.pos_checkpoint:
        pos_state = torch.load(resolve(args.pos_checkpoint), map_location="cpu")
        if isinstance(pos_state, dict):
            tensors = [value for value in pos_state.values() if torch.is_tensor(value)]
            if len(tensors) != 1:
                raise SystemExit(
                    f"--pos-checkpoint dict must hold exactly one tensor, got {len(tensors)}")
            pos_state = tensors[0]
        if not torch.is_tensor(pos_state):
            raise SystemExit(f"--pos-checkpoint must be a tensor, got {type(pos_state).__name__}")
        if tuple(pos_state.shape) != tuple(pos_weight.shape):
            raise SystemExit(
                f"--pos-checkpoint shape {tuple(pos_state.shape)} != position table "
                f"{tuple(pos_weight.shape)}")
        with torch.no_grad():
            pos_weight.copy_(pos_state.to(device=pos_weight.device, dtype=pos_weight.dtype))
        print(f"[predictor] loaded pos-checkpoint {args.pos_checkpoint} -> {tuple(pos_weight.shape)}")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(resolve(args.adapter)))
        print(f"[predictor] loaded PEFT adapter from {args.adapter}")
    return model, decoder


def freeze_predictor_only(model):
    trainable_names = []
    for name, param in model.named_parameters():
        param.requires_grad_("predictor.klinear" in name or "predictor.qlinear" in name)
        if param.requires_grad:
            trainable_names.append(name)
    unexpected = [name for name in trainable_names
                  if not ("predictor.klinear" in name or "predictor.qlinear" in name)]
    assert not unexpected, f"unexpected trainable params: {unexpected[:8]}"
    return sorted(trainable_names)


def all_layer_dims(decoder):
    dims = []
    for layer in decoder.layers:
        predictor = getattr(layer.self_attn, "predictor", None)
        if isinstance(predictor, PrunableAttnPredictor):
            dims.append(predictor.get_current_outdims())
        else:
            dims.append(None)
    return dims


def compute_mask_loss(outputs):
    predict_mask = getattr(outputs, "predict_mask", None)
    pooling_gt = getattr(outputs, "pooling_gt", None)
    if predict_mask is None or pooling_gt is None:
        raise RuntimeError("model output lacks predict_mask/pooling_gt (need train-predictor model)")
    total = None
    layers = 0
    for predict, groundtruth in zip(predict_mask, pooling_gt):
        if predict is None or groundtruth is None:
            continue
        term = F.smooth_l1_loss(predict.float(), groundtruth.float())
        total = term if total is None else total + term
        layers += 1
    if layers == 0:
        raise RuntimeError("no valid (predict_mask, pooling_gt) pairs in model output")
    return total / layers


@torch.no_grad()
def evaluate(model, array, num_samples, truncate, device):
    model.eval()
    losses = []
    for index in range(num_samples):
        input_ids = row_to_tensor(array, index, truncate, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, use_cache=False)
        losses.append(float(compute_mask_loss(outputs)))
    model.train()
    return {"samples": len(losses), "loss": sum(losses) / len(losses) if losses else None}


def grad_norm_if_clipped(trainable_params, max_grad_norm):
    if max_grad_norm > 0:
        return float(torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm))
    total = 0.0
    for param in trainable_params:
        if param.grad is not None:
            total += float((param.grad.detach().float() ** 2).sum())
    return total ** 0.5


# SMELL 3 train_predictor strip_wrappers ADD — PEFT 包装后去掉 base_model 前缀，保证 predictor.pth 键名稳定
def strip_wrappers(name):
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def save_predictor(model, decoder, save_dir):
    params = {
        strip_wrappers(name): param.detach().to("cpu").clone()
        for name, param in model.named_parameters() if param.requires_grad
    }
    torch.save(params, save_dir / "predictor.pth")
    pruned_config = {"layers": all_layer_dims(decoder)}
    torch.save(pruned_config, save_dir / "pruned_config.pth")
    return params, pruned_config


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
    save_dir = resolve(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    train_array = load_ids(data_path)
    train_full_len = int(train_array.shape[1])
    if args.eval_data:
        eval_array = load_ids(resolve(args.eval_data))
    else:
        eval_array = train_array
    eval_full_len = int(eval_array.shape[1])
    pos_len = max(train_full_len, eval_full_len)
    requested_len = args.truncate if args.truncate > 0 else train_full_len
    seq_len = requested_len - requested_len % BLOCK_SIZE
    if seq_len != requested_len:
        print(f"[predictor] WARNING seq_len rounded {requested_len} -> {seq_len} (block size {BLOCK_SIZE})")
    eval_count = min(args.eval_samples, int(eval_array.shape[0])) if args.eval_every > 0 else 0
    do_prune = (not args.no_prune) and args.prune_interval > 0

    model, decoder = build_model(args, pos_len)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    trainable_names = freeze_predictor_only(model)
    num_layers = len(decoder.layers)
    assert len(trainable_names) == num_layers * 6, (
        f"expected {num_layers * 6} trainable predictor tensors, got {len(trainable_names)}")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_count = int(sum(param.numel() for param in trainable_params))
    predictor_types = {type(module).__name__ for module in model.modules()
                       if isinstance(module, PrunableAttnPredictor)}

    print(f"[predictor] decoder={type(decoder).__name__} layers={num_layers} "
          f"predictor={sorted(predictor_types)}")
    print(f"[predictor] pos_init={args.pos_init} pos_rows="
          f"{int(decoder.embed_positions.num_embeddings) - int(decoder.embed_positions.offset)} "
          f"pos_checkpoint={args.pos_checkpoint} adapter={args.adapter}")
    print(f"[predictor] trainable_params={trainable_count} trainable_tensors={len(trainable_names)}")
    print(f"[predictor] trainable first={trainable_names[:3]}")
    print(f"[predictor] trainable last={trainable_names[-3:]}")
    print(f"[predictor] prune={do_prune} interval={args.prune_interval} until={args.prune_until} "
          f"lr={args.lr} weight_decay={args.weight_decay} steps={args.steps} "
          f"batch_size={args.batch_size} max_grad_norm={args.max_grad_norm}")
    print(f"[predictor] data={data_path} (n={train_array.shape[0]} len={train_full_len} used={seq_len})")
    print(f"[predictor] eval={'first ' + str(eval_count) + ' rows of ' if not args.eval_data else ''}"
          f"{'eval_data' if args.eval_data else 'train data'} every={args.eval_every} "
          f"save_dir={save_dir}")

    config_payload = {
        **vars(args),
        "resolved_model_dir": args.model_dir,
        "resolved_data": str(data_path),
        "resolved_eval_data": str(resolve(args.eval_data)) if args.eval_data else str(data_path),
        "resolved_save_dir": str(save_dir),
        "train_full_seq_len": train_full_len,
        "eval_full_seq_len": eval_full_len,
        "position_table_len": pos_len,
        "seq_len": seq_len,
        "do_prune": do_prune,
        "trainable_params": trainable_count,
        "trainable_tensors": len(trainable_names),
        "trainable_names": trainable_names,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    (save_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    stream = ShuffledStream(int(train_array.shape[0]), args.seed)
    started = time.time()
    metrics_handle = metrics_path.open("w", encoding="utf-8")

    last_loss = None
    last_grad_norm = None
    last_eval = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for _ in range(args.batch_size):
            index = stream.next_index()
            input_ids = row_to_tensor(train_array, index, args.truncate, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, use_cache=False)
                loss = compute_mask_loss(outputs) / args.batch_size
            loss.backward()
            step_losses.append(float(loss.detach().float()) * args.batch_size)
        last_grad_norm = grad_norm_if_clipped(trainable_params, args.max_grad_norm)
        optimizer.step()
        last_loss = sum(step_losses) / len(step_losses)

        prune_info = None
        if do_prune and step % args.prune_interval == 0 and step < args.prune_until:
            times = step // args.prune_interval
            threshold = PRUNE_ZERO_RATIO - (times - 1) * PRUNE_DECAY
            before_dims = all_layer_dims(decoder)
            for module in model.modules():
                if isinstance(module, PrunableAttnPredictor):
                    module.prune_neurons(
                        step_count_threshold=args.prune_interval,
                        zero_ratio_threshold=threshold,
                    )
            after_dims = all_layer_dims(decoder)
            # SMELL 3 train_predictor optimizer_refresh ADD — 剪枝替换的新 Linear 需重新挂入优化器参数组
            trainable_params = [param for param in model.parameters() if param.requires_grad]
            optimizer.param_groups[0]["params"] = trainable_params
            prune_info = {"times": times, "threshold": threshold, "changed": before_dims != after_dims}
            print(f"[prune] step={step} times={times} threshold={threshold:.2f} "
                  f"changed={prune_info['changed']}")
            print(f"[prune] layer0 before={before_dims[0]} after={after_dims[0]}")

        eval_stats = None
        if eval_count > 0 and (step % args.eval_every == 0 or step == args.steps):
            eval_stats = evaluate(model, eval_array, eval_count, args.truncate, device)
            last_eval = eval_stats

        if args.log_every > 0 and (step % args.log_every == 0 or step == args.steps):
            eval_text = f" eval_loss={eval_stats['loss']:.6f}" if eval_stats else ""
            print(f"[predictor] step {step}/{args.steps} loss={last_loss:.6f} "
                  f"grad_norm={last_grad_norm:.4e} elapsed={time.time() - started:.1f}s{eval_text}")

        record = {
            "step": step,
            "loss": last_loss,
            "grad_norm": last_grad_norm,
            "elapsed_s": time.time() - started,
            "pruned": bool(prune_info),
            "prune_threshold": prune_info["threshold"] if prune_info else None,
            "prune_changed": prune_info["changed"] if prune_info else None,
            "eval_loss": eval_stats["loss"] if eval_stats else None,
        }
        metrics_handle.write(json.dumps(record) + "\n")
        metrics_handle.flush()

    params, pruned_config = save_predictor(model, decoder, save_dir)
    metrics_handle.close()

    summary = {
        "steps": args.steps,
        "trainable_params": trainable_count,
        "trainable_tensors": len(trainable_names),
        "final_train_loss": last_loss,
        "final_grad_norm": last_grad_norm,
        "final_eval_loss": last_eval["loss"] if last_eval else None,
        "predictor": str(save_dir / "predictor.pth"),
        "pruned_config": str(save_dir / "pruned_config.pth"),
        "config": str(save_dir / "config.json"),
        "metrics": str(metrics_path),
        "saved_tensors": len(params),
        "pruned_layers": sum(1 for layer in pruned_config["layers"] if layer is not None),
    }
    print("[predictor] ===== summary =====")
    for key, value in summary.items():
        print(f"[predictor] {key}: {value}")
    print(f"[predictor] done steps={args.steps} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
