# SMELL 3 classify NEW — DistilBERT 分类评测（accuracy / macro-F1 / NLL；global/local split）

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MODEL_DIR = REPO / "third_party" / "Jenga" / "checkpoints" / "distilbert-base-uncased"


def parse_args():
    parser = argparse.ArgumentParser(description="DistilBERT sequence-classification eval")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--data-root", default="dataset_v3/arxiv_16k")
    parser.add_argument("--tag", default="a01")
    parser.add_argument("--split", choices=("global", "local"), default="global")
    parser.add_argument("--client", default=None, help="client id for --split local, e.g. 07")
    parser.add_argument("--pos-checkpoint", default=None, help="position_embeddings weight .pt")
    # SMELL 3 classify head_checkpoint ADD — Step1 head.pt（FL 评测必需；缺失则头为随机初始化）
    parser.add_argument("--head-checkpoint", default=None, help="Step1 head.pt (pre_classifier/classifier)")
    parser.add_argument("--adapter", default=None, help="optional PEFT adapter dir")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    # SMELL 3 classify attn ADD — 与 run_fed 同后端：sdpa_prune 需 predictor+pruned_config（Jenga 真语义稀疏）
    parser.add_argument("--attn", choices=("sdpa", "sdpa_prune", "eager"), default="sdpa")
    # SMELL 3 classify predictor ADD — 训练后 predictor + 逐层形状（缺失时剪枝退化为随机打分）
    parser.add_argument("--predictor", default=None, help="predictor.pth (requires --pruned-config)")
    parser.add_argument("--pruned-config", default=None, help="pruned_config.pth")
    # SMELL 3 classify catv ADD — 服务器共识锚掩码（与训练同轮口径；仅 sdpa_prune 路径生效）
    parser.add_argument("--catv-mask", default=None, help="consensus_mask_roundNNN.pt")
    # SMELL 3 classify persample ADD — 逐样本 NLL/正确性落盘（配对检验）
    parser.add_argument("--per-sample-out", default=None)
    # SMELL 3 classify sparse ADD — 与训练同 sparse（剪枝预算；0.4=Jenga 口径）
    parser.add_argument("--sparse", type=float, default=0.4, help="block keep ratio for sdpa_prune")
    parser.add_argument("--truncate", type=int, default=0, help="testing only: use first N tokens")
    parser.add_argument("--out", default=None, help="JSON output path")
    args = parser.parse_args()
    if bool(args.predictor) != bool(args.pruned_config):
        raise SystemExit("--predictor and --pruned-config must be provided together")
    if args.attn != "sdpa_prune" and (args.predictor or args.catv_mask):
        raise SystemExit("--predictor/--catv-mask require --attn sdpa_prune")
    return args


def resolve_data_paths(args):
    root = Path(args.data_root)
    if not root.is_absolute():
        root = REPO / root
    tag_dir = root / args.tag
    if args.split == "global":
        stem = tag_dir / "global_test"
    else:
        if args.client is None:
            raise SystemExit("--client is required with --split local")
        client = f"client_{int(str(args.client).replace('client_', '')):02d}"
        stem = tag_dir / "clients" / client / "local_test"
    return (Path(f"{stem}_input_ids.npy"), Path(f"{stem}_labels.npy"),
            Path(f"{stem}_lengths.npy"), tag_dir / "meta.json")


def load_label_names(meta_path):
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    names = meta.get("label_names")
    return [str(name) for name in names] if names else None


def macro_f1(preds, labels, num_classes):
    scores = []
    for cls in range(num_classes):
        tp = sum(1 for p, y in zip(preds, labels) if p == cls and y == cls)
        fp = sum(1 for p, y in zip(preds, labels) if p == cls and y != cls)
        fn = sum(1 for p, y in zip(preds, labels) if p != cls and y == cls)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else None


def main():
    args = parse_args()
    import numpy as np
    import torch

    from src.models.modeling_distilbert_smell import load_distilbert_classifier

    ids_path, labels_path, lengths_path, meta_path = resolve_data_paths(args)
    for path in (ids_path, labels_path):
        if not path.exists():
            raise SystemExit(f"missing data file: {path}")
    input_ids_all = np.load(ids_path)
    labels_all = np.load(labels_path)
    assert input_ids_all.shape[0] == labels_all.shape[0], (
        f"input_ids rows {input_ids_all.shape[0]} != labels rows {labels_all.shape[0]}")

    label_names = load_label_names(meta_path)
    num_labels = len(label_names) if label_names else int(labels_all.max()) + 1
    num_total = int(input_ids_all.shape[0])
    num_eval = min(num_total, args.max_samples) if args.max_samples > 0 else num_total
    if num_eval <= 0:
        raise SystemExit("no samples to evaluate")
    effective_seq_len = args.truncate if args.truncate > 0 else int(input_ids_all.shape[1])

    model = load_distilbert_classifier(args.model_dir, num_labels=num_labels, dtype=args.dtype,
                                       attn=args.attn, seq_len=effective_seq_len,
                                       pos_checkpoint=args.pos_checkpoint,
                                       head_checkpoint=args.head_checkpoint,
                                       predictor=args.predictor,
                                       pruned_config=args.pruned_config,
                                       sparse=args.sparse)
    model = model.cuda().eval()
    if args.catv_mask:
        mask = torch.load(args.catv_mask, map_location="cpu")
        assert isinstance(mask, dict), f"unsupported catv mask payload {type(mask).__name__}"
        model.config.consensus_mask = {int(key): value for key, value in mask.items()}
        print(f"[classify] catv mask loaded layers={sorted(model.config.consensus_mask)}")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()

    if lengths_path.exists():
        lengths_all = np.load(lengths_path).astype(np.int64)
        assert lengths_all.shape[0] == input_ids_all.shape[0], (
            f"lengths rows {lengths_all.shape[0]} != input_ids rows {input_ids_all.shape[0]}")
    else:
        pad_id = getattr(model.config, "pad_token_id", 0)
        pad_id = int(pad_id) if pad_id is not None else 0
        lengths_all = (input_ids_all != pad_id).sum(axis=1).astype(np.int64)
        print(f"[classify] lengths file missing, inferred with pad_token_id={pad_id}")

    preds = []
    labels = []
    per_sample = []
    nll_sum = 0.0
    n = 0
    batch_size = max(1, int(args.batch_size))
    for start in range(0, num_eval, batch_size):
        end = min(num_eval, start + batch_size)
        chunk_ids = torch.from_numpy(input_ids_all[start:end].astype(np.int64))
        if args.truncate > 0:
            chunk_ids = chunk_ids[:, :args.truncate]
        seq_len = int(chunk_ids.size(1))
        chunk_lengths = np.maximum(np.minimum(lengths_all[start:end], seq_len), 1)
        attention_mask = (torch.arange(seq_len).unsqueeze(0)
                          < torch.from_numpy(chunk_lengths).unsqueeze(1)).long()
        chunk_labels = torch.from_numpy(labels_all[start:end].astype(np.int64))
        with torch.no_grad():
            logits = model(input_ids=chunk_ids.cuda(), attention_mask=attention_mask.cuda()).logits.float()
        # SMELL 3 classify persample ADD — 逐样本 NLL（reduction=none）与是否正确
        chunk_nll = torch.nn.functional.cross_entropy(logits, chunk_labels.cuda(), reduction="none")
        chunk_pred = logits.argmax(dim=-1)
        nll_sum += float(chunk_nll.sum().item())
        preds.extend(chunk_pred.cpu().tolist())
        labels.extend(chunk_labels.tolist())
        for index in range(int(chunk_labels.numel())):
            per_sample.append({"nll": float(chunk_nll[index].item()),
                               "correct": bool(int(chunk_pred[index].item()) == int(chunk_labels[index].item()))})
        n += int(chunk_labels.numel())

    n_classes = max(num_labels, max(labels) + 1, max(preds) + 1)
    accuracy = sum(1 for p, y in zip(preds, labels) if p == y) / n if n > 0 else None
    f1 = macro_f1(preds, labels, n_classes)
    mean_nll = nll_sum / n if n > 0 else None
    print(f"[classify] data={ids_path}")
    print(f"[classify] model={args.model_dir} dtype={args.dtype} adapter={args.adapter} "
          f"n={n} seq_len={effective_seq_len}")
    print(f"[classify] accuracy={accuracy:.4f} macro_f1={f1:.4f} nll={mean_nll:.4f}")

    result = {
        "model_dir": str(args.model_dir),
        "adapter": args.adapter,
        "pos_checkpoint": args.pos_checkpoint,
        "head_checkpoint": args.head_checkpoint,
        "attn": args.attn,
        "predictor": args.predictor,
        "pruned_config": args.pruned_config,
        "catv_mask": args.catv_mask,
        "input_ids": str(ids_path),
        "labels": str(labels_path),
        "lengths": str(lengths_path) if lengths_path.exists() else None,
        "meta": str(meta_path) if meta_path.exists() else None,
        "split": args.split,
        "tag": args.tag,
        "client": args.client,
        "dtype": args.dtype,
        "truncate": args.truncate,
        "batch_size": batch_size,
        "seq_len": effective_seq_len,
        "n": n,
        "num_labels": n_classes,
        "accuracy": accuracy,
        "macro_f1": f1,
        "nll": mean_nll,
        "label_names": label_names,
    }
    if args.per_sample_out:
        per_sample_path = Path(args.per_sample_out)
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        per_sample_path.write_text(json.dumps(per_sample, indent=2), encoding="utf-8")
        print(f"[classify] wrote {per_sample_path} n={len(per_sample)}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[classify] wrote {out_path}")


if __name__ == "__main__":
    main()
