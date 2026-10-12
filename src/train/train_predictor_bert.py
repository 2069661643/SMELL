# SMELL 3 train_predictor_bert NEW — DistilBERT 16k predictor 训练器（双向 block_attn_pool 目标）
#
# 仿 train_predictor.py（OPT）但适配 DistilBERT：
#   - 目标 = 双向 ReLU(qk^T) 块池化（非 causal；Jenga 原 Triton 目标对 encoder 本就双向）
#   - 只训练每层 attention 的 PrunableAttnPredictor q/k 两层线性（qlinear1/2、klinear1/2，bias=False）
#   - 前向整体 no_grad（encoder 冻结），predictor 以 detach 输入 + enable_grad 局部建图：
#     仅 predictor 反传，省显存/时间；pooling 目标在 no_grad 下算
#   - 位置表复用 Step1 产物；动态剪枝（zero_ratio 0.8 -> 0.75 -> ...）与 OPT 同调度
# 产物：predictor.pth + pruned_config.pth + config.json + metrics.jsonl
#
# 用法示例：
#   python src/train/train_predictor_bert.py --tag a01 --steps 400 --gpu 0 \
#       --pos-checkpoint checkpoints/distilbert_step1/a01_pos_head/pos_embed.pt

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

DEFAULT_MODEL_DIR = REPO / "third_party" / "Jenga" / "checkpoints" / "distilbert-base-uncased"
DEFAULT_DATA_ROOT = REPO / "dataset_v3" / "arxiv_16k"
BLOCK_SIZE = 64
PRUNE_ZERO_RATIO = 0.8
PRUNE_DECAY = 0.05
PRUNE_SCOPE_MAX_STEP = 620


# SMELL 3 train_predictor_bert gpu_preparse BEGIN — --gpu 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES
def _preparse_gpu():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default=None)
    known, _ = parser.parse_known_args()
    if known.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_preparse_gpu()
# SMELL 3 train_predictor_bert gpu_preparse END

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train Jenga predictor (q/k linears) for DistilBERT-16k")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tag", default="a01")
    parser.add_argument("--data", default=None,
                        help="train input_ids npy [N, L]; default dataset_v3/arxiv_16k/<tag>/warmup_input_ids.npy")
    parser.add_argument("--lengths", default=None,
                        help="lengths npy [N] for --data; default dataset_v3/arxiv_16k/<tag>/warmup_lengths.npy")
    parser.add_argument("--pos-init", choices=("interpolate", "duplicate", "dup_scaled"), default="interpolate")
    parser.add_argument("--pos-checkpoint", default=None,
                        help="pos_embed.pt from Step1 (pos+head) training")
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
    parser.add_argument("--save-dir", default=None, help="default logs/predictor_bert/<timestamp>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    if args.data is None:
        args.data = str(DEFAULT_DATA_ROOT / args.tag / "warmup_input_ids.npy")
    if args.lengths is None:
        args.lengths = str(DEFAULT_DATA_ROOT / args.tag / "warmup_lengths.npy")
    if args.save_dir is None:
        args.save_dir = str(REPO / "logs" / "predictor_bert" / time.strftime("%y%m%d-%H%M%S"))
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


def load_lengths(path, ids_array):
    path = Path(path)
    if path.exists():
        lengths = np.load(path).astype(np.int64)
        assert lengths.shape[0] == ids_array.shape[0], (
            f"lengths rows {lengths.shape[0]} != input_ids rows {ids_array.shape[0]}")
        return lengths
    print(f"[predictor-bert] lengths missing at {path}; inferring from pad id 0")
    return (np.asarray(ids_array) != 0).sum(axis=1)


def row_to_tensors(ids_array, lengths_array, index, truncate, device):
    row = np.asarray(ids_array[index], dtype=np.int64)
    if truncate > 0:
        row = row[:truncate]
    seq_len = int(row.shape[0])
    length = int(min(max(int(lengths_array[index]), 1), seq_len))
    input_ids = torch.from_numpy(row).unsqueeze(0).to(device)
    attention_mask = (torch.arange(seq_len, device=device).view(1, -1) < length).long()
    return input_ids, attention_mask


class ShuffledStream:
    # SMELL 3 train_predictor_bert ShuffledStream COPY — 带种子的确定性样本流（与 long_ctx 训练器一致）
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


# SMELL 3 train_predictor_bert trainer_attn NEW — 训练期注意力：记录 predict_mask 与双向块池化目标，前向全程 no_grad + predictor 局部 enable_grad
def make_trainer_attention_class():
    from src.models.modeling_distilbert_smell import (DistilBertSdpaAttention,
                                                      _keep_mask_2d, block_attn_pool_bidir)

    class DistilBertPredictorTrainAttention(DistilBertSdpaAttention):
        layer_idx = None
        predict_mask = None
        pooling_gt = None

        def __init__(self, config):
            super().__init__(config)
            self.layer_idx = None
            self.predict_mask = None
            self.pooling_gt = None

        @property
        def predictor(self):
            return self.__dict__.get("_modules", {}).get("predictor")

        def forward(self, query, key, value, mask, head_mask=None, output_attentions=False, **kwargs):
            bsz, q_len, _ = query.size()
            dim_per_head = self.dim // self.n_heads
            if self.predictor is not None:
                with torch.no_grad():
                    keep = _keep_mask_2d(mask, bsz, q_len, query.device) if mask is not None else None
                    q = self.q_lin(query).view(bsz, -1, self.n_heads, dim_per_head).transpose(1, 2) \
                        / math.sqrt(dim_per_head)
                    k = self.k_lin(query).view(bsz, -1, self.n_heads, dim_per_head).transpose(1, 2)
                    self.pooling_gt = block_attn_pool_bidir(q, k, keep, block=BLOCK_SIZE)
                with torch.enable_grad():
                    # SMELL 3 train_predictor_bert local_graph ADD — detach 输入：仅 predictor 建图，encoder 不反传
                    self.predict_mask = self.predictor(query.detach()) / 100.0
            return super().forward(query, key, value, mask, head_mask=head_mask,
                                   output_attentions=output_attentions, **kwargs)

    return DistilBertPredictorTrainAttention


def build_model(args, pos_len):
    from transformers import DistilBertForSequenceClassification

    from src.models.modeling_distilbert_smell import patch_distilbert_sdpa
    from src.models.position_embed_bert import ensure_positions_bert

    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir, num_labels=11)
    model = model.float()
    model = ensure_positions_bert(model, pos_len, mode=args.pos_init)
    if args.pos_checkpoint:
        payload = torch.load(resolve(args.pos_checkpoint), map_location="cpu")
        if isinstance(payload, dict):
            for key in ("position_embeddings.weight", "embed_positions.weight", "weight"):
                if key in payload:
                    payload = payload[key]
                    break
        assert torch.is_tensor(payload), f"unsupported pos checkpoint at {args.pos_checkpoint}"
        target = model.distilbert.embeddings.position_embeddings.weight
        assert tuple(payload.shape) == tuple(target.shape), (
            f"pos checkpoint shape {tuple(payload.shape)} != {tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(payload.to(device=target.device, dtype=target.dtype))
        print(f"[predictor-bert] loaded pos checkpoint {args.pos_checkpoint} shape={tuple(target.shape)}")
    model = patch_distilbert_sdpa(model)
    attn_class = make_trainer_attention_class()
    for layer_idx, layer in enumerate(model.distilbert.transformer.layer):
        attention = layer.attention
        attention.__class__ = attn_class
        attention.layer_idx = int(layer_idx)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def attach_predictors(model):
    from jenga.models.predictor import PrunableAttnPredictor

    built = 0
    for layer_idx, layer in enumerate(model.distilbert.transformer.layer):
        attention = layer.attention
        if attention.predictor is None:
            attention.predictor = PrunableAttnPredictor(
                dim=int(attention.dim // attention.n_heads), hidden_dim=128,
                n_head=int(attention.n_heads))
            built += 1
    return built


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


def all_layer_dims(model):
    from jenga.models.predictor import PrunableAttnPredictor

    dims = []
    for layer in model.distilbert.transformer.layer:
        predictor = layer.attention.predictor
        if isinstance(predictor, PrunableAttnPredictor):
            dims.append(predictor.get_current_outdims())
        else:
            dims.append(None)
    return dims


def compute_mask_loss(model):
    total = None
    layers = 0
    for layer in model.distilbert.transformer.layer:
        predict = layer.attention.predict_mask
        groundtruth = layer.attention.pooling_gt
        if predict is None or groundtruth is None:
            continue
        term = F.smooth_l1_loss(predict.float(), groundtruth.float())
        total = term if total is None else total + term
        layers += 1
    if layers == 0:
        raise RuntimeError("no valid (predict_mask, pooling_gt) pairs; predictors not attached?")
    return total / layers


@torch.no_grad()
def evaluate(model, ids_array, lengths_array, num_samples, truncate, device):
    losses = []
    for index in range(num_samples):
        input_ids, attention_mask = row_to_tensors(ids_array, lengths_array, index, truncate, device)
        model(input_ids=input_ids, attention_mask=attention_mask)
        losses.append(float(compute_mask_loss(model)))
    return {"samples": len(losses), "loss": sum(losses) / len(losses) if losses else None}


def grad_norm_if_clipped(trainable_params, max_grad_norm):
    if max_grad_norm > 0:
        return float(torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm))
    total = 0.0
    for param in trainable_params:
        if param.grad is not None:
            total += float((param.grad.detach().float() ** 2).sum())
    return total ** 0.5


# SMELL 3 train_predictor_bert strip_wrappers COPY — 去掉常见包装前缀，保证 predictor.pth 键名与 run_fed 推理模型一致
def strip_wrappers(name):
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def save_predictor(model, save_dir):
    params = {
        strip_wrappers(name): param.detach().to("cpu").clone()
        for name, param in model.named_parameters() if param.requires_grad
    }
    torch.save(params, save_dir / "predictor.pth")
    pruned_config = {"layers": all_layer_dims(model)}
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
    lengths_path = resolve(args.lengths)
    save_dir = resolve(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    train_array = load_ids(data_path)
    train_lengths = load_lengths(lengths_path, train_array)
    train_full_len = int(train_array.shape[1])
    requested_len = args.truncate if args.truncate > 0 else train_full_len
    seq_len = requested_len - requested_len % BLOCK_SIZE
    if seq_len != requested_len:
        print(f"[predictor-bert] WARNING seq_len rounded {requested_len} -> {seq_len} (block {BLOCK_SIZE})")
    eval_count = min(args.eval_samples, int(train_array.shape[0])) if args.eval_every > 0 else 0
    do_prune = (not args.no_prune) and args.prune_interval > 0

    if train_full_len % BLOCK_SIZE != 0:
        raise SystemExit(f"training seq len {train_full_len} must be divisible by {BLOCK_SIZE}")

    model = build_model(args, train_full_len)
    built = attach_predictors(model)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    trainable_names = freeze_predictor_only(model)
    num_layers = len(model.distilbert.transformer.layer)
    assert len(trainable_names) == num_layers * 6, (
        f"expected {num_layers * 6} trainable predictor tensors, got {len(trainable_names)}")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_count = int(sum(param.numel() for param in trainable_params))

    print(f"[predictor-bert] layers={num_layers} predictors_built={built} "
          f"trainable_params={trainable_count} trainable_tensors={len(trainable_names)}")
    print(f"[predictor-bert] first={trainable_names[:2]}")
    print(f"[predictor-bert] prune={do_prune} interval={args.prune_interval} until={args.prune_until} "
          f"lr={args.lr} steps={args.steps} batch={args.batch_size} seq={seq_len} data={data_path}")

    config_payload = {
        **vars(args),
        "resolved_model_dir": args.model_dir,
        "resolved_data": str(data_path),
        "resolved_lengths": str(lengths_path),
        "resolved_save_dir": str(save_dir),
        "train_full_seq_len": train_full_len,
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
    # SMELL 3 train_predictor_bert eval_mode ADD — 全程 eval：dropout 关闭，predictor 目标与损失确定性
    model.eval()
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
            input_ids, attention_mask = row_to_tensors(train_array, train_lengths, index,
                                                       args.truncate, device)
            # SMELL 3 train_predictor_bert fwd_no_grad ADD — encoder 前向 no_grad；predictor 在注意力内局部 enable_grad
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            loss = compute_mask_loss(model) / args.batch_size
            loss.backward()
            step_losses.append(float(loss.detach().float()) * args.batch_size)
        last_grad_norm = grad_norm_if_clipped(trainable_params, args.max_grad_norm)
        optimizer.step()
        last_loss = sum(step_losses) / len(step_losses)

        prune_info = None
        if do_prune and step % args.prune_interval == 0 and step < args.prune_until:
            from jenga.models.predictor import PrunableAttnPredictor
            times = step // args.prune_interval
            threshold = PRUNE_ZERO_RATIO - (times - 1) * PRUNE_DECAY
            before_dims = all_layer_dims(model)
            for module in model.modules():
                if isinstance(module, PrunableAttnPredictor):
                    module.prune_neurons(step_count_threshold=args.prune_interval,
                                         zero_ratio_threshold=threshold)
            after_dims = all_layer_dims(model)
            trainable_params = [param for param in model.parameters() if param.requires_grad]
            optimizer.param_groups[0]["params"] = trainable_params
            prune_info = {"times": times, "threshold": threshold, "changed": before_dims != after_dims}
            print(f"[prune] step={step} times={times} threshold={threshold:.2f} "
                  f"changed={prune_info['changed']} layer0={before_dims[0]} -> {after_dims[0]}")

        eval_stats = None
        if eval_count > 0 and (step % args.eval_every == 0 or step == args.steps):
            eval_stats = evaluate(model, train_array, train_lengths, eval_count, args.truncate, device)
            last_eval = eval_stats

        if args.log_every > 0 and (step % args.log_every == 0 or step == args.steps):
            eval_text = f" eval_loss={eval_stats['loss']:.6f}" if eval_stats else ""
            print(f"[predictor-bert] step {step}/{args.steps} loss={last_loss:.6f} "
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

    params, pruned_config = save_predictor(model, save_dir)
    metrics_handle.close()

    summary = {
        "steps": args.steps,
        "trainable_params": trainable_count,
        "final_train_loss": last_loss,
        "final_grad_norm": last_grad_norm,
        "final_eval_loss": last_eval["loss"] if last_eval else None,
        "predictor": str(save_dir / "predictor.pth"),
        "pruned_config": str(save_dir / "pruned_config.pth"),
        "saved_tensors": len(params),
        "pruned_layers": sum(1 for layer in pruned_config["layers"] if layer is not None),
        "metrics": str(metrics_path),
    }
    print("[predictor-bert] ===== summary =====")
    for key, value in summary.items():
        print(f"[predictor-bert] {key}: {value}")
    print(f"[predictor-bert] done steps={args.steps} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
