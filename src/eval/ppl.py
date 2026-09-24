# SMELL 3 ppl NEW — Discovery 16k 生成式 PPL 评测（全文 + answer-only，token 加权 / per-sample）

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
JENGA_SRC = REPO / "third_party" / "Jenga" / "src"
for extra in (str(REPO), str(JENGA_SRC)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from jenga.models.modeling_opt import OPTForCausalLM  # noqa: E402
from jenga.utils.config_utils import get_opt_qk  # noqa: E402

from src.models.position_embed import ensure_positions  # noqa: E402

DEFAULT_MODEL_DIR = REPO / "third_party" / "Jenga" / "checkpoints" / "opt-350m"
LOGIT_CHUNK = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Discovery 16k G-PPL: full-text + answer-only")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--data-dir", default=None,
                        help="dir containing input_ids.npy / labels.npy / answer_spans.npy")
    parser.add_argument("--split", choices=("global", "local"), default="global")
    parser.add_argument("--data-root", default="dataset_v3/discovery_16k")
    parser.add_argument("--tag", default="a01")
    parser.add_argument("--client", default=None, help="client id for --split local, e.g. 07")
    parser.add_argument("--adapter", default=None, help="optional PEFT adapter dir")
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--sparse", type=float, default=0.4)
    parser.add_argument("--pos-mode", choices=("jenga_dup_scaled", "duplicate", "interpolate"),
                        default="jenga_dup_scaled")
    # SMELL 3 ppl pos_checkpoint ADD — 加载 longctx 适配后的 embed_positions 权重（2k 回归 / 16k 评测）
    parser.add_argument("--pos-checkpoint", default=None, help="pos_embed.pt loaded into embed_positions")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--truncate", type=int, default=0, help="testing only: use first N tokens")
    parser.add_argument("--out", default=None, help="JSON output path")
    return parser.parse_args()


def resolve_data_paths(args):
    if args.data_dir:
        data_dir = Path(args.data_dir)
        return (data_dir / "input_ids.npy", data_dir / "labels.npy", data_dir / "answer_spans.npy")
    root = Path(args.data_root)
    if not root.is_absolute():
        root = REPO / root
    if args.split == "global":
        stem = root / args.tag / "global_test"
    else:
        if args.client is None:
            raise SystemExit("--client is required with --split local")
        client = f"client_{int(str(args.client).replace('client_', '')):02d}"
        stem = root / args.tag / "clients" / client / "local_test"
    return (Path(f"{stem}_input_ids.npy"), Path(f"{stem}_labels.npy"),
            Path(f"{stem}_answer_spans.npy"))


def get_decoder(model):
    try:
        return model.get_decoder()
    except AttributeError:
        return model.base_model.model.get_decoder()


def get_lm_head(model):
    try:
        return model.get_output_embeddings()
    except AttributeError:
        return model.base_model.model.get_output_embeddings()


def build_model(args, effective_max_len):
    config = get_opt_qk(model_name=args.model_dir, flash_attention=not args.no_flash,
                        pool_size=64, thresh=args.sparse)
    model = OPTForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, config=config)
    # SMELL 3 position_embed ADD — 16k 序列需扩展 embed_positions（OPT 原表仅 2050 行）
    model = ensure_positions(model, effective_max_len, mode=args.pos_mode)
    # SMELL 3 ppl pos_checkpoint BEGIN — 覆盖为 warmup 训得的 embed_positions（形状不符时按行数再扩展）
    if args.pos_checkpoint:
        pos_payload = torch.load(args.pos_checkpoint, map_location="cpu")
        if isinstance(pos_payload, dict):
            pos_payload = pos_payload.get("weight", pos_payload)
        assert torch.is_tensor(pos_payload), f"unsupported pos checkpoint payload: {type(pos_payload)}"
        target_weight = model.model.decoder.embed_positions.weight
        if tuple(pos_payload.shape) != tuple(target_weight.shape):
            same_width = pos_payload.dim() == 2 and pos_payload.shape[1] == target_weight.shape[1]
            if same_width and pos_payload.shape[0] > target_weight.shape[0]:
                offset = int(model.model.decoder.embed_positions.offset)
                model = ensure_positions(model, int(pos_payload.shape[0]) - offset)
                target_weight = model.model.decoder.embed_positions.weight
        assert tuple(pos_payload.shape) == tuple(target_weight.shape), (
            f"pos checkpoint shape {tuple(pos_payload.shape)} != embed_positions {tuple(target_weight.shape)}")
        with torch.no_grad():
            target_weight.copy_(pos_payload.to(device=target_weight.device, dtype=target_weight.dtype))
    # SMELL 3 ppl pos_checkpoint END
    model = model.cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    return model


def sequence_nll(model, input_ids):
    decoder = get_decoder(model)
    lm_head = get_lm_head(model)
    with torch.no_grad():
        outputs = decoder(input_ids=input_ids, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state[0]
        length = hidden.size(0)
        labels = input_ids[0, 1:length]
        nll = torch.empty(length - 1, dtype=torch.float32)
        for start in range(0, length - 1, LOGIT_CHUNK):
            end = min(length - 1, start + LOGIT_CHUNK)
            logits = lm_head(hidden[start:end]).float()
            nll[start:end] = F.cross_entropy(logits, labels[start:end], reduction="none").cpu()
            del logits
    return nll


def main():
    args = parse_args()
    ids_path, labels_path, spans_path = resolve_data_paths(args)
    for path in (ids_path, labels_path, spans_path):
        if not path.exists():
            raise SystemExit(f"missing data file: {path}")
    input_ids_all = np.load(ids_path)
    spans_all = np.load(spans_path)
    num_total = int(input_ids_all.shape[0])
    num_eval = min(num_total, args.max_samples) if args.max_samples > 0 else num_total
    if num_eval <= 0:
        raise SystemExit("no samples to evaluate")

    # SMELL 3 position_embed ADD — truncate>0 时按其长度扩展，否则用数据实际 seq_len
    effective_max_len = args.truncate if args.truncate > 0 else int(input_ids_all.shape[1])
    model = build_model(args, effective_max_len)
    full_sum = 0.0
    full_count = 0
    full_per_sample = []
    ans_sum = 0.0
    ans_count = 0
    ans_per_sample = []
    seq_lengths = []
    for index in range(num_eval):
        input_ids = torch.from_numpy(input_ids_all[index].astype(np.int64)).unsqueeze(0).cuda()
        if args.truncate > 0:
            input_ids = input_ids[:, :args.truncate]
        length = input_ids.size(1)
        if length < 2:
            continue
        span_start, span_end = int(spans_all[index][0]), int(spans_all[index][1])
        assert span_start >= 1, f"sample {index}: answer span start {span_start} < 1"
        nll = sequence_nll(model, input_ids)
        full_sum += float(nll.sum())
        full_count += int(nll.numel())
        full_per_sample.append(float(torch.exp(nll.mean())))
        seq_lengths.append(length)
        answer_end = min(span_end, length)
        if answer_end > span_start:
            answer_nll = nll[span_start - 1:answer_end - 1]
            ans_sum += float(answer_nll.sum())
            ans_count += int(answer_nll.numel())
            ans_per_sample.append(float(torch.exp(answer_nll.mean())))

    full_ppl_token = math.exp(full_sum / full_count) if full_count else None
    full_ppl_sample = sum(full_per_sample) / len(full_per_sample) if full_per_sample else None
    ans_ppl_token = math.exp(ans_sum / ans_count) if ans_count else None
    ans_ppl_sample = sum(ans_per_sample) / len(ans_per_sample) if ans_per_sample else None

    print(f"[ppl] data={ids_path}")
    print(f"[ppl] samples={len(seq_lengths)} total_tokens={full_count} "
          f"mean_len={sum(seq_lengths) / len(seq_lengths):.1f} sparse={args.sparse} flash={not args.no_flash}")
    print(f"[ppl] full-text   token_ppl={full_ppl_token:.4f} "
          f"sample_mean_ppl={full_ppl_sample:.4f} scored_tokens={full_count}")
    if ans_count:
        print(f"[ppl] answer-only token_ppl={ans_ppl_token:.4f} "
              f"sample_mean_ppl={ans_ppl_sample:.4f} answer_tokens={ans_count}")
    else:
        print("[ppl] answer-only no answer tokens found")

    result = {
        "model_dir": str(args.model_dir),
        "adapter": args.adapter,
        "input_ids": str(ids_path),
        "answer_spans": str(spans_path),
        "sparse": args.sparse,
        "flash": not args.no_flash,
        "truncate": args.truncate,
        "num_samples": len(seq_lengths),
        "mean_seq_len": (sum(seq_lengths) / len(seq_lengths)) if seq_lengths else None,
        "full_ppl_token": full_ppl_token,
        "full_ppl_sample_mean": full_ppl_sample,
        "answer_ppl_token": ans_ppl_token,
        "answer_ppl_sample_mean": ans_ppl_sample,
        "total_scored_tokens": full_count,
        "total_answer_tokens": ans_count,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[ppl] wrote {out_path}")


if __name__ == "__main__":
    main()
