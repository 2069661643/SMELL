# SMELL 3 train_pos_head_bert NEW — DistilBERT 16k Step1：冻结 encoder，联合训练位置表 + 分类头（CE + MLM 辅助）
#
# 口径（D2 = A）：每个 batch 对输入做 15% BERT 式 mask，单遍前向；[CLS] 出 11 类 CE，
# mask 位 gather 后过冻结的预训练 MLM 头出 MLM CE；loss = CE + mlm_weight * MLM。
# --mlm-weight 0 退化为 B（干净输入、仅 CE）；--freeze-head / --freeze-pos 可复现 C 两段式。
# 可训练：位置表（lr-pos）+ pre_classifier/classifier（lr-head）；encoder 与 MLM 头冻结。
# 产物：pos_embed.pt（{"position_embeddings.weight": ...}）+ head.pt + config.json + metrics.jsonl。
#
# 用法示例（本机 smoke）：
#   python src/train/train_pos_head_bert.py --tag a01mini --truncate 512 --steps 4 --gpu 0

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

DEFAULT_MODEL_DIR = REPO / "third_party" / "Jenga" / "checkpoints" / "distilbert-base-uncased"
DEFAULT_DATA_ROOT = REPO / "dataset_v3" / "arxiv_16k"
POS_KEY = "position_embeddings.weight"
HEAD_PREFIX = "smell_cls_head."
IGNORE_INDEX = -100


# SMELL 3 train_pos_head_bert gpu_preparse BEGIN — --gpu 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES
def _preparse_gpu():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default=None)
    known, _ = parser.parse_known_args()
    if known.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)


_preparse_gpu()
# SMELL 3 train_pos_head_bert gpu_preparse END

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class DistilBertClsHead(torch.nn.Module):
    # SMELL 3 train_pos_head_bert cls_head NEW — HF DistilBertForSequenceClassification 同构头（pre_classifier + ReLU + dropout + classifier）
    def __init__(self, dim, num_labels, dropout=0.2):
        super().__init__()
        self.pre_classifier = torch.nn.Linear(dim, dim)
        self.classifier = torch.nn.Linear(dim, num_labels)
        self.dropout = float(dropout)

    def forward(self, cls_hidden):
        hidden = self.pre_classifier(cls_hidden)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.classifier(hidden)


class ShuffledStream:
    # SMELL 3 train_pos_head_bert ShuffledStream COPY — 带种子的确定性样本流（与 longctx_adapt 一致）
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


def parse_args():
    parser = argparse.ArgumentParser(description="DistilBERT 16k Step1: joint position-table + head (CE + MLM)")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--tag", default="a01")
    parser.add_argument("--train-prefix", default="global_train")
    parser.add_argument("--eval-prefix", default="global_val")
    parser.add_argument("--num-labels", type=int, default=11)
    parser.add_argument("--pos-init", choices=("interpolate", "duplicate", "dup_scaled"), default="interpolate")
    parser.add_argument("--pos-checkpoint", default=None)
    parser.add_argument("--head-checkpoint", default=None)
    parser.add_argument("--mlm-weight", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--lr-pos", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--truncate", type=int, default=0, help="smoke only: use first N tokens; 0 = full seq")
    parser.add_argument("--freeze-pos", action="store_true", help="C 段2：只训 head")
    parser.add_argument("--freeze-head", action="store_true", help="C 段1：只训 pos")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--eval-every", type=int, default=100, help="0 = no periodic eval")
    parser.add_argument("--eval-samples", type=int, default=0, help="0 = all eval rows")
    parser.add_argument("--save-dir", default=None, help="default logs/pos_head/<timestamp>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = str(REPO / "logs" / "pos_head" / time.strftime("%y%m%d-%H%M%S"))
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise SystemExit("--batch-size/--grad-accum must be >= 1")
    if args.mlm_weight < 0:
        raise SystemExit("--mlm-weight must be >= 0")
    if not 0.0 < args.mask_ratio < 1.0:
        raise SystemExit("--mask-ratio must be in (0, 1)")
    if args.freeze_pos and args.freeze_head:
        raise SystemExit("--freeze-pos and --freeze-head cannot both be set")
    if args.truncate < 0:
        raise SystemExit("--truncate must be >= 0")
    return args


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def load_split(root_dir, prefix):
    stem = root_dir / prefix
    ids_path = Path(f"{stem}_input_ids.npy")
    labels_path = Path(f"{stem}_labels.npy")
    if not ids_path.exists() or not labels_path.exists():
        raise SystemExit(f"missing data files: {ids_path} / {labels_path}")
    ids = np.load(ids_path, mmap_mode="r")
    labels = np.load(labels_path)
    lengths_path = Path(f"{stem}_lengths.npy")
    if lengths_path.exists():
        lengths = np.load(lengths_path).astype(np.int64)
    else:
        lengths = (np.asarray(ids) != 0).sum(axis=1).astype(np.int64)
        print(f"[pos_head] lengths file missing, inferred from pad id: {lengths_path}")
    assert ids.shape[0] == labels.shape[0] == lengths.shape[0], (
        f"row mismatch ids={ids.shape[0]} labels={labels.shape[0]} lengths={lengths.shape[0]}")
    return ids, labels, lengths


def load_pos_into(model, pos_checkpoint):
    from src.models.position_embed_bert import ensure_positions_bert

    payload = torch.load(resolve(pos_checkpoint), map_location="cpu")
    if isinstance(payload, dict):
        for key in (POS_KEY, "embed_positions.weight", "weight"):
            if key in payload:
                payload = payload[key]
                break
    assert torch.is_tensor(payload), f"unsupported pos checkpoint payload at {pos_checkpoint}"
    embeddings = model.distilbert.embeddings
    target = embeddings.position_embeddings.weight
    if tuple(payload.shape) != tuple(target.shape):
        if payload.dim() == 2 and payload.shape[1] == target.shape[1] and payload.shape[0] > target.shape[0]:
            model = ensure_positions_bert(model, int(payload.shape[0]), mode="interpolate")
            target = embeddings.position_embeddings.weight
    assert tuple(payload.shape) == tuple(target.shape), (
        f"pos checkpoint shape {tuple(payload.shape)} != {tuple(target.shape)}")
    with torch.no_grad():
        target.copy_(payload.to(device=target.device, dtype=target.dtype))
    print(f"[pos_head] loaded pos checkpoint {pos_checkpoint} shape={tuple(target.shape)}")
    return model


def load_head_into(head, head_checkpoint):
    payload = torch.load(resolve(head_checkpoint), map_location="cpu")
    assert isinstance(payload, dict), f"unsupported head checkpoint at {head_checkpoint}"
    state = {key[len(HEAD_PREFIX):] if key.startswith(HEAD_PREFIX) else key: value
             for key, value in payload.items() if torch.is_tensor(value)}
    missing, unexpected = head.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    assert loaded > 0, f"no head tensors loaded from {head_checkpoint}"
    print(f"[pos_head] loaded head checkpoint {head_checkpoint} tensors={loaded} "
          f"missing={list(missing)} unexpected={list(unexpected)}")
    return head


def build_model(args, pos_len):
    from transformers import AutoTokenizer, DistilBertForMaskedLM

    from src.models.modeling_distilbert_smell import patch_distilbert_sdpa
    from src.models.position_embed_bert import ensure_positions_bert

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = DistilBertForMaskedLM.from_pretrained(args.model_dir)
    model = model.to(dtype=dtype)
    model = ensure_positions_bert(model, pos_len, mode=args.pos_init)
    model = patch_distilbert_sdpa(model)
    head = DistilBertClsHead(model.config.dim, args.num_labels,
                             dropout=getattr(model.config, "seq_classif_dropout", 0.2))
    model.add_module(HEAD_PREFIX.rstrip("."), head)
    if args.pos_checkpoint:
        model = load_pos_into(model, args.pos_checkpoint)
    if args.head_checkpoint:
        head = load_head_into(head, args.head_checkpoint)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_grads):
            enable_input_grads()

    for param in model.parameters():
        param.requires_grad_(False)
    pos_weight = model.distilbert.embeddings.position_embeddings.weight
    if not args.freeze_pos:
        pos_weight.requires_grad_(True)
    if not args.freeze_head:
        for param in head.parameters():
            param.requires_grad_(True)
    return model, head, tokenizer


def make_mlm_batch(input_ids, lengths, tokenizer, mask_ratio, generator):
    # SMELL 3 train_pos_head_bert mlm_mask ADD — BERT 式 15% mask（80/10/10），只在有效 token 且非特殊 token 上采样
    device = input_ids.device
    bsz, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    special = ((input_ids == tokenizer.cls_token_id) | (input_ids == tokenizer.sep_token_id)
               | (input_ids == tokenizer.pad_token_id))
    candidate = valid & ~special
    rand = torch.rand(input_ids.shape, device=device, generator=generator)
    mask = candidate & (rand < mask_ratio)
    empty = candidate.any(dim=1) & ~mask.any(dim=1)
    if bool(empty.any()):
        first_valid = candidate.float().argmax(dim=1)
        rows = empty.nonzero(as_tuple=True)[0]
        mask[rows, first_valid[rows]] = True
    mlm_labels = input_ids.clone()
    mlm_labels[~mask] = IGNORE_INDEX
    masked_ids = input_ids.clone()
    rand2 = torch.rand(input_ids.shape, device=device, generator=generator)
    to_mask = mask & (rand2 < 0.8)
    to_random = mask & ~to_mask & (rand2 < 0.9)
    random_ids = torch.randint(0, tokenizer.vocab_size, input_ids.shape, device=device, generator=generator)
    masked_ids[to_mask] = tokenizer.mask_token_id
    masked_ids[to_random] = random_ids[to_random]
    return masked_ids, mask, mlm_labels


def mlm_logits_at(model, hidden_states, mask):
    # SMELL 3 train_pos_head_bert mlm_head ADD — mask 位 gather 后过冻结 MLM 头，避免全 16k×30522 logits
    # SMELL 3 train_pos_head_bert mlm_head FIXED — transformers 4.45 的 DistilBertForMaskedLM 是扁平头（vocab_transform/activation/vocab_layer_norm/vocab_projector），无 model.cls
    gathered = hidden_states[mask]
    transformed = model.vocab_transform(gathered)
    transformed = model.activation(transformed)
    transformed = model.vocab_layer_norm(transformed)
    return model.vocab_projector(transformed)


def build_batch(ids_array, labels_array, lengths_array, indices, truncate, device):
    chunk = np.asarray(ids_array[indices], dtype=np.int64)
    if truncate > 0:
        chunk = chunk[:, :truncate]
    seq_len = int(chunk.shape[1])
    lengths = np.maximum(np.minimum(lengths_array[indices], seq_len), 1)
    input_ids = torch.from_numpy(chunk).to(device)
    attention_mask = (torch.arange(seq_len, device=device).unsqueeze(0)
                      < torch.from_numpy(lengths).unsqueeze(1).to(device)).long()
    labels = torch.from_numpy(np.asarray(labels_array[indices], dtype=np.int64)).to(device)
    lengths_t = torch.from_numpy(lengths).to(device)
    return input_ids, attention_mask, labels, lengths_t


def forward_losses(model, head, tokenizer, input_ids, attention_mask, labels, lengths, args, generator):
    if args.mlm_weight > 0:
        masked_ids, mask, mlm_labels = make_mlm_batch(
            input_ids, lengths, tokenizer, args.mask_ratio, generator)
    else:
        masked_ids, mask, mlm_labels = input_ids, None, None
    outputs = model.distilbert(input_ids=masked_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state
    logits = head(hidden[:, 0])
    loss_ce = F.cross_entropy(logits.float(), labels)
    loss = loss_ce
    loss_mlm = None
    if args.mlm_weight > 0:
        logits_mlm = mlm_logits_at(model, hidden, mask)
        loss_mlm = F.cross_entropy(logits_mlm.float(), mlm_labels[mask])
        loss = loss + args.mlm_weight * loss_mlm
    return loss, loss_ce, loss_mlm, logits


@torch.no_grad()
def evaluate(model, head, ids_array, labels_array, lengths_array, num_samples, truncate, device, batch_size):
    model.eval()
    correct = 0
    total = 0
    ce_sum = 0.0
    for start in range(0, num_samples, batch_size):
        indices = list(range(start, min(num_samples, start + batch_size)))
        input_ids, attention_mask, labels, _ = build_batch(
            ids_array, labels_array, lengths_array, indices, truncate, device)
        outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = head(outputs.last_hidden_state[:, 0]).float()
        ce_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += len(indices)
    model.train()
    return {"n": total, "accuracy": correct / total if total else None,
            "ce": ce_sum / total if total else None}


def clip_grad(trainable_params, max_grad_norm):
    if max_grad_norm > 0:
        return float(torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm))
    total = 0.0
    for param in trainable_params:
        if param.grad is not None:
            total += float((param.grad.detach().float() ** 2).sum())
    return total ** 0.5


def save_artifacts(model, head, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    pos_weight = model.distilbert.embeddings.position_embeddings.weight
    torch.save({POS_KEY: pos_weight.detach().to("cpu").clone()}, save_dir / "pos_embed.pt")
    torch.save(head.state_dict(), save_dir / "head.pt")
    return int(pos_weight.numel()), int(sum(p.numel() for p in head.parameters()))


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = resolve(args.data_root) / args.tag
    save_dir = resolve(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    train_ids, train_labels, train_lengths = load_split(data_dir, args.train_prefix)
    eval_ids, eval_labels, eval_lengths = load_split(data_dir, args.eval_prefix)
    pos_len = max(int(train_ids.shape[1]), int(eval_ids.shape[1]))
    eval_count = int(eval_ids.shape[0]) if args.eval_samples <= 0 else min(
        args.eval_samples, int(eval_ids.shape[0]))

    model, head, tokenizer = build_model(args, pos_len)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    head.to(device)

    pos_params = [model.distilbert.embeddings.position_embeddings.weight]
    head_params = list(head.parameters())
    groups = []
    if not args.freeze_pos:
        groups.append({"params": pos_params, "lr": args.lr_pos, "name": "pos"})
    if not args.freeze_head:
        groups.append({"params": head_params, "lr": args.lr_head, "name": "head"})
    assert groups, "no trainable parameter groups"
    trainable_params = [param for group in groups for param in group["params"]]
    trainable_count = int(sum(param.numel() for param in trainable_params))
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    model.train()

    config_payload = {
        **vars(args),
        "resolved_model_dir": str(resolve(args.model_dir)),
        "resolved_data_dir": str(data_dir),
        "resolved_save_dir": str(save_dir),
        "position_table_len": pos_len,
        "train_rows": int(train_ids.shape[0]),
        "eval_rows": int(eval_ids.shape[0]),
        "eval_count": eval_count,
        "trainable_params": trainable_count,
        "trainable_groups": [group["name"] for group in groups],
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    (save_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"[pos_head] mode={'freeze-pos' if args.freeze_pos else 'freeze-head' if args.freeze_head else 'pos+head'} "
          f"trainable_params={trainable_count} groups={[group['name'] for group in groups]}")
    print(f"[pos_head] pos_rows={pos_len} pos_init={args.pos_init} dtype={args.dtype} "
          f"mlm_weight={args.mlm_weight} mask_ratio={args.mask_ratio} grad_ckpt={args.grad_ckpt}")
    print(f"[pos_head] data={data_dir} train={train_ids.shape} eval={eval_ids.shape} "
          f"steps={args.steps} batch={args.batch_size} grad_accum={args.grad_accum} "
          f"lr_pos={args.lr_pos} lr_head={args.lr_head} save_dir={save_dir}")

    stream = ShuffledStream(int(train_ids.shape[0]), args.seed)
    started = time.time()
    metrics_handle = metrics_path.open("w", encoding="utf-8")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.eval_every > 0:
        stats = evaluate(model, head, eval_ids, eval_labels, eval_lengths, eval_count,
                         args.truncate, device, args.batch_size)
        print(f"[pos_head] baseline eval n={stats['n']} acc={stats['accuracy']:.4f} ce={stats['ce']:.4f}")

    best_acc = None
    best_dir = save_dir / "best"
    last_loss = None
    last_grad_norm = None
    micro_batches = args.batch_size * args.grad_accum
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        step_ce = []
        step_mlm = []
        for _ in range(micro_batches):
            indices = [stream.next_index() for _ in range(args.batch_size)]
            input_ids, attention_mask, labels, lengths = build_batch(
                train_ids, train_labels, train_lengths, indices, args.truncate, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(args.dtype == "bf16")):
                loss, loss_ce, loss_mlm, _ = forward_losses(
                    model, head, tokenizer, input_ids, attention_mask, labels, lengths, args, generator)
            (loss / micro_batches).backward()
            step_losses.append(float(loss.detach().float()))
            step_ce.append(float(loss_ce.detach().float()))
            if loss_mlm is not None:
                step_mlm.append(float(loss_mlm.detach().float()))
        last_grad_norm = clip_grad(trainable_params, args.max_grad_norm)
        optimizer.step()
        last_loss = sum(step_losses) / len(step_losses)

        eval_stats = None
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            eval_stats = evaluate(model, head, eval_ids, eval_labels, eval_lengths, eval_count,
                                  args.truncate, device, args.batch_size)
            if best_acc is None or eval_stats["accuracy"] > best_acc:
                best_acc = eval_stats["accuracy"]
                save_artifacts(model, head, best_dir)

        if args.log_every > 0 and (step % args.log_every == 0 or step == args.steps):
            mlm_text = f" mlm={sum(step_mlm) / len(step_mlm):.4f}" if step_mlm else ""
            eval_text = (f" eval_acc={eval_stats['accuracy']:.4f} eval_ce={eval_stats['ce']:.4f}"
                         if eval_stats else "")
            print(f"[pos_head] step {step}/{args.steps} loss={last_loss:.6f} "
                  f"ce={sum(step_ce) / len(step_ce):.4f}{mlm_text} grad_norm={last_grad_norm:.4e} "
                  f"elapsed={time.time() - started:.1f}s{eval_text}")

        record = {
            "step": step,
            "loss": last_loss,
            "ce": sum(step_ce) / len(step_ce),
            "mlm": sum(step_mlm) / len(step_mlm) if step_mlm else None,
            "grad_norm": last_grad_norm,
            "elapsed_s": time.time() - started,
            "eval_accuracy": eval_stats["accuracy"] if eval_stats else None,
            "eval_ce": eval_stats["ce"] if eval_stats else None,
        }
        metrics_handle.write(json.dumps(record) + "\n")
        metrics_handle.flush()

    pos_count, head_count = save_artifacts(model, head, save_dir)
    metrics_handle.close()

    summary = {
        "steps": args.steps,
        "trainable_params": trainable_count,
        "final_train_loss": last_loss,
        "final_grad_norm": last_grad_norm,
        "best_eval_accuracy": best_acc,
        "pos_tensors": pos_count,
        "head_tensors": head_count,
        "pos_embed": str(save_dir / "pos_embed.pt"),
        "head": str(save_dir / "head.pt"),
        "best_dir": str(best_dir) if best_acc is not None else None,
        "config": str(save_dir / "config.json"),
        "metrics": str(metrics_path),
    }
    print("[pos_head] ===== summary =====")
    for key, value in summary.items():
        print(f"[pos_head] {key}: {value}")
    print(f"[pos_head] done steps={args.steps} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
