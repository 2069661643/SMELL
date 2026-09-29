# SMELL 3 build_warmup_arxiv_16k NEW — arxiv 长上下文 warmup LM 语料（与 client 分区 / 预留池严格互斥）

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

import build_arxiv_16k as base  # noqa: E402  (import 时先设置 HF mirror 环境再 import datasets)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SOURCE = base.SOURCE
SOURCE_CONFIG = base.SOURCE_CONFIG
TOKENIZER_PATH = base.TOKENIZER_PATH
DEFAULT_OUT = base.DEFAULT_OUT
log = base.log


def parse_args():
    ap = argparse.ArgumentParser(description="Build arxiv long-context warmup LM corpus (disjoint from client partition)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output root; reads <out>/<tag>/partition.json, writes <out>/<tag>/warmup_*")
    ap.add_argument("--tag", default="a01")
    ap.add_argument("--partition-root", default=None, help="root containing <tag>/partition.json (default: --out)")
    ap.add_argument("--partition", default=None, help="explicit partition.json path (overrides --partition-root/--out)")
    ap.add_argument("--warmup-size", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--head-frac", type=float, default=0.75)
    ap.add_argument("--tokenizer", default=TOKENIZER_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mini", action="store_true", help="8 samples; seq_len 2048")
    args = ap.parse_args()
    if not 0.0 < args.head_frac < 1.0:
        raise SystemExit(f"--head-frac must be in (0,1), got {args.head_frac}")
    if args.mini:
        args.warmup_size = 8
        args.seq_len = 2048
    if args.partition is None:
        proot = args.partition_root if args.partition_root is not None else args.out
        args.partition = os.path.join(proot, args.tag, "partition.json")
    return args


# SMELL 3 build_warmup_arxiv_16k load_partition ADD — 汇总排除集（client train/local + 预留 train 池 [+ train 空间 global test]）与可用池
def load_partition(path):
    with open(path, encoding="utf-8") as f:
        part = json.load(f)
    train_size = int(part["train_size"])
    test_size = int(part.get("test_size", 0))
    excl = np.zeros(train_size, dtype=bool)
    n_clients = 0
    n_client_train = 0
    n_client_local = 0
    for cname, d in part["clients"].items():
        idx_train = np.asarray(d["train"], dtype=np.int64)
        idx_local = np.asarray(d["local_test"], dtype=np.int64)
        assert idx_train.size == 0 or int(idx_train.max()) < train_size, f"{cname}: train idx out of range"
        assert idx_local.size == 0 or int(idx_local.max()) < train_size, f"{cname}: local idx out of range"
        excl[idx_train] = True
        excl[idx_local] = True
        n_clients += 1
        n_client_train += int(idx_train.size)
        n_client_local += int(idx_local.size)
    demo_idx = np.asarray(part.get("global_demo_idx", []), dtype=np.int64)
    if demo_idx.size:
        assert int(demo_idx.max()) < train_size, "global_demo_idx out of range"
        excl[demo_idx] = True
    global_test_idx = np.asarray(part.get("global_test_query_idx", []), dtype=np.int64)
    global_test_split = part.get("global_test_split", "train")
    n_excluded_global_test = 0
    if global_test_split == "train" and global_test_idx.size:
        assert int(global_test_idx.max()) < train_size, "global_test_query_idx out of train range"
        excl[global_test_idx] = True
        n_excluded_global_test = int(global_test_idx.size)
    elif global_test_split == "test" and global_test_idx.size:
        assert test_size > 0 and int(global_test_idx.max()) < test_size, "global_test_query_idx out of test range"
    avail = np.flatnonzero(~excl)
    stats = {
        "partition_path": os.path.abspath(path),
        "n_clients": n_clients,
        "train_size": train_size,
        "test_size": test_size,
        "excluded_clients_train": n_client_train,
        "excluded_clients_local_test": n_client_local,
        "excluded_global_demo": int(demo_idx.size),
        "excluded_global_test": n_excluded_global_test,
        "excluded_total": int(excl.sum()),
        "available_pool": int(avail.size),
        "global_test_split": global_test_split,
    }
    return part, excl, avail, stats


def main():
    t0 = time.time()
    args = parse_args()
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log(f"warmup build start: out={out_dir} partition={args.partition} warmup-size={args.warmup_size} "
        f"seq_len={args.seq_len} head_frac={args.head_frac} seed={args.seed} mini={args.mini}")

    part, excl, avail, stats = load_partition(args.partition)
    log(f"disjoint pool: excluded={stats['excluded_total']} "
        f"(clients_train={stats['excluded_clients_train']} clients_local={stats['excluded_clients_local_test']} "
        f"global_demo={stats['excluded_global_demo']}) available={stats['available_pool']} "
        f"global_test_split={stats['global_test_split']} elapsed={time.time() - t0:.1f}s")

    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.partition)), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            build_meta = json.load(f)
        if int(build_meta["seq_len"]) != args.seq_len:
            raise SystemExit(f"--seq-len {args.seq_len} != build meta seq_len {build_meta['seq_len']}")
        if abs(float(build_meta["head_frac"]) - args.head_frac) > 1e-9:
            raise SystemExit(f"--head-frac {args.head_frac} != build meta head_frac {build_meta['head_frac']}")
        if build_meta.get("tokenizer") != args.tokenizer:
            log(f"WARNING tokenizer mismatch: build={build_meta.get('tokenizer')} warmup={args.tokenizer}")

    if args.warmup_size > avail.size:
        raise SystemExit(f"--warmup-size {args.warmup_size} > available pool {avail.size}")

    rng = np.random.default_rng([args.seed, 404])
    chosen = np.sort(avail[rng.permutation(avail.size)[:args.warmup_size]])
    chosen_set = {int(x) for x in chosen.tolist()}

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        raise SystemExit(f"tokenizer has no pad token: {args.tokenizer}")
    pad_id = int(tok.pad_token_id)
    tok.model_max_length = args.seq_len  # SMELL 3 build_warmup_arxiv_16k model_max_length ADD — 自行头尾截断到 seq_len，避免 512 上限告警
    tok.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True  # SMELL 3 build_warmup_arxiv_16k tokenizer_warning ADD — 长文由 encode_doc 截断，静音误导告警
    log(f"tokenizer ready: {type(tok).__name__} vocab={tok.vocab_size} pad_token_id={pad_id}")

    ids_arr = np.empty((args.warmup_size, args.seq_len), dtype=np.uint16)
    lengths_arr = np.empty(args.warmup_size, dtype=np.int32)
    pos = {int(idx): k for k, idx in enumerate(chosen.tolist())}

    def fill(idx, text):
        ids, length = base.encode_doc(tok, text, args.seq_len, args.head_frac, pad_id)
        k = pos[idx]
        ids_arr[k] = ids
        lengths_arr[k] = length

    seen = base.stream_wanted("train", stats["train_size"], chosen_set, fill)
    base.validate_ids_lengths("warmup", ids_arr, lengths_arr, pad_id)
    log(f"warmup tokens filled: stream_rows={seen} elapsed={time.time() - t0:.1f}s")

    client_train_idx, client_local_idx = [], []
    for cname, d in part["clients"].items():
        client_train_idx.extend(d["train"])
        client_local_idx.extend(d["local_test"])
    demo_idx = list(part.get("global_demo_idx", []))
    global_test_idx = list(part.get("global_test_query_idx", []))

    def overlap(idx_list):
        if not idx_list:
            return 0
        return int(np.intersect1d(chosen, np.asarray(idx_list, dtype=np.int64)).size)

    over_train = overlap(client_train_idx)
    over_local = overlap(client_local_idx)
    over_demo = overlap(demo_idx)
    over_global_test = overlap(global_test_idx) if stats["global_test_split"] == "train" else 0
    assert over_train == 0, f"overlap with clients.train: {over_train}"
    assert over_local == 0, f"overlap with clients.local_test: {over_local}"
    assert over_demo == 0, f"overlap with global_demo_idx: {over_demo}"
    assert over_global_test == 0, f"overlap with global_test_query_idx: {over_global_test}"

    digest = hashlib.sha256(np.asarray(chosen, dtype=np.uint32).tobytes()).hexdigest()[:16]
    length_stats_out = base.length_stats(lengths_arr, args.seq_len)
    np.save(os.path.join(out_dir, "warmup_input_ids.npy"), ids_arr)
    np.save(os.path.join(out_dir, "warmup_lengths.npy"), lengths_arr)
    warmup_meta = {
        "source": SOURCE,
        "config": SOURCE_CONFIG,
        "split": "train",
        "tag": args.tag,
        "seed": args.seed,
        "samples": args.warmup_size,
        "seq_len": args.seq_len,
        "head_frac": args.head_frac,
        "truncation": base.TRUNCATION,
        "dtype": "uint16",
        "tokenizer": args.tokenizer,
        "add_special_tokens": True,
        "pad_token_id": pad_id,
        "partition": stats,
        "source_idx": [int(x) for x in chosen.tolist()],
        "source_idx_count": int(chosen.size),
        "source_idx_sha256": digest,
        "total_tokens": int(lengths_arr.sum()),
        "total_token_slots": int(args.warmup_size * args.seq_len),
        "length_stats": length_stats_out,
        "disjointness": {
            "overlap_clients_train": over_train,
            "overlap_clients_local_test": over_local,
            "overlap_global_demo": over_demo,
            "overlap_global_test": over_global_test,
            "global_test_split": stats["global_test_split"],
            "note": "global_test rows live in the test split; train-space warmup rows cannot overlap by construction",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "warmup_meta.json"), "w", encoding="utf-8") as f:
        json.dump(warmup_meta, f, ensure_ascii=False, indent=2)
    log("=" * 64)
    log(f"warmup done: samples={args.warmup_size} seq_len={args.seq_len} dtype=uint16 path={out_dir}")
    log(f"  disjointness: overlap(train/local/demo/global_test)="
        f"{over_train}/{over_local}/{over_demo}/{over_global_test} sha256={digest}")
    log(f"  source pool: available={stats['available_pool']} used={chosen.size} "
        f"({100.0 * chosen.size / stats['available_pool']:.1f}%) "
        f"lengths min/mean/max={length_stats_out['min']}/{length_stats_out['mean']}/{length_stats_out['max']} "
        f"total_tokens={int(lengths_arr.sum())}")
    log(f"  elapsed={time.time() - t0:.1f}s out={out_dir}")


if __name__ == "__main__":
    main()
