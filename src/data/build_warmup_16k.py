# SMELL 3 build_warmup_16k NEW — Discovery 长上下文 warmup LM 语料构建（与评测/训练池严格互斥）

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

import build_discovery_16k as base  # noqa: E402  (import 时先设置 HF mirror 环境再 import datasets)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SOURCE = base.SOURCE
SOURCE_CONFIG = base.SOURCE_CONFIG
TOKENIZER_PATH = base.TOKENIZER_PATH
INSTRUCTION = base.INSTRUCTION
DEMO_FMT = base.DEMO_FMT
TEMPLATE_HASH = base.TEMPLATE_HASH
log = base.log
TokenCache = base.TokenCache
RoundRobinDemoPool = base.RoundRobinDemoPool
get_demo_ids = base.get_demo_ids

DEFAULT_OUT = os.path.join("dataset_v3", "discovery_16k")


def parse_args():
    ap = argparse.ArgumentParser(description="Build Discovery long-context warmup LM corpus (disjoint from eval/client pools)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output root; writes <out>/<tag>/warmup_*")
    ap.add_argument("--tag", default="a01")
    ap.add_argument("--partition-root", default=None,
                    help="root containing <tag>/partition.json (default: --out)")
    ap.add_argument("--partition", default=None,
                    help="explicit partition.json path (overrides --partition-root/--out)")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mini", action="store_true", help="8 samples; partition defaults to temp/mini_dataset")
    args = ap.parse_args()
    if args.mini:
        args.samples = 8
        if args.partition is None and args.partition_root is None:
            args.partition_root = os.path.join(_ROOT, "temp", "mini_dataset")
    if args.partition is None:
        proot = args.partition_root if args.partition_root is not None else args.out
        args.partition = os.path.join(proot, args.tag, "partition.json")
    return args


# SMELL 3 warmup_pool ADD — 从 partition.json 的三类排除集构建布尔掩码与类别分桶
def load_disjoint_pool(partition_path, labels_all, n_classes):
    with open(partition_path, encoding="utf-8") as f:
        part = json.load(f)
    n = labels_all.shape[0]
    m_train = np.zeros(n, dtype=bool)
    m_local = np.zeros(n, dtype=bool)
    for cname, d in part["clients"].items():
        m_train[np.asarray(d["train"], dtype=np.int64)] = True
        m_local[np.asarray(d["local_test"], dtype=np.int64)] = True
    m_demo = np.zeros(n, dtype=bool)
    m_demo[np.asarray(part["global_demo_idx"], dtype=np.int64)] = True
    m_excl = m_train | m_local | m_demo
    avail = np.flatnonzero(~m_excl)
    avail_labels = labels_all[avail]
    by_class = [[] for _ in range(n_classes)]
    for idx, lab in zip(avail.tolist(), avail_labels.tolist()):
        by_class[int(lab)].append(int(idx))
    per_class = np.bincount(avail_labels, minlength=n_classes)
    stats = {
        "partition_path": os.path.abspath(partition_path),
        "clients": sorted(part["clients"].keys()),
        "n_clients": len(part["clients"]),
        "excluded_clients_train": int(m_train.sum()),
        "excluded_clients_local_test": int(m_local.sum()),
        "excluded_global_demo": int(m_demo.sum()),
        "excluded_total": int(m_excl.sum()),
        "available_pool": int(avail.size),
        "available_per_class_min": int(per_class.min()),
        "available_per_class_max": int(per_class.max()),
        "available_per_class_mean": float(per_class.mean()),
    }
    return by_class, m_train, m_local, m_demo, stats


# SMELL 3 warmup_assemble ADD — instruction + 均匀类别 round-robin 演示块，截断演示区至恰好 seq_len（无 query/无 pad）
def assemble_warmup(seq_len, instr_ids, draw_demo):
    demos = []
    total = len(instr_ids)
    while total < seq_len:
        demo_ids = draw_demo()
        if not demo_ids:
            raise RuntimeError("empty demo tokenization")
        demos.append(demo_ids)
        total += len(demo_ids)
    while demos and total - len(demos[0]) >= seq_len:
        total -= len(demos.pop(0))
    flat = [token for demo in demos for token in demo]
    cut = len(instr_ids) + len(flat) - seq_len
    if cut > 0:
        if cut > len(flat):
            raise RuntimeError(f"cannot truncate demo region: cut={cut} flat={len(flat)}")
        flat = flat[cut:]
    input_ids = instr_ids + flat
    if len(input_ids) != seq_len:
        raise RuntimeError(f"assembled {len(input_ids)} tokens != seq_len {seq_len}")
    return input_ids, len(demos)


def main():
    t0 = time.time()
    args = parse_args()
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log(f"warmup build start: out={out_dir} partition={args.partition} samples={args.samples} "
        f"seq_len={args.seq_len} seed={args.seed} mini={args.mini}")

    log(f"loading {SOURCE} [{SOURCE_CONFIG}] train via mirror ...")
    train_ds = load_dataset(SOURCE, SOURCE_CONFIG, split="train").select_columns(
        ["sentence1", "sentence2", "label"]
    )
    label_names = list(train_ds.features["label"].names)
    n_classes = len(label_names)
    labels_all = np.asarray(train_ds["label"], dtype=np.int64)
    n_train = labels_all.shape[0]
    log(f"loaded: train={n_train} classes={n_classes} elapsed={time.time() - t0:.1f}s")

    by_class, m_train, m_local, m_demo, pool_stats = load_disjoint_pool(
        args.partition, labels_all, n_classes
    )
    log(f"disjoint pool: excluded={pool_stats['excluded_total']} "
        f"(train={pool_stats['excluded_clients_train']} local={pool_stats['excluded_clients_local_test']} "
        f"demo={pool_stats['excluded_global_demo']}) available={pool_stats['available_pool']} "
        f"per-class min={pool_stats['available_per_class_min']} mean={pool_stats['available_per_class_mean']:.0f} "
        f"elapsed={time.time() - t0:.1f}s")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
    if getattr(tok, "add_bos_token", False):
        tok.add_bos_token = False
    instr_ids = tok.encode(INSTRUCTION, add_special_tokens=False)
    log(f"tokenizer ready: {type(tok).__name__} vocab={tok.vocab_size} instr_tokens={len(instr_ids)}")

    rng = np.random.default_rng([args.seed, 404])
    rr_pool = RoundRobinDemoPool(by_class, rng, n_classes)
    cache = TokenCache()
    ids_arr = np.empty((args.samples, args.seq_len), dtype=np.uint16)
    demo_counts = []
    for k in range(args.samples):
        rr_pool.start_sample()

        def draw_demo():
            didx = rr_pool.draw()
            return get_demo_ids(cache, tok, train_ds, "train", didx, label_names[int(labels_all[didx])])

        ids, n_demos = assemble_warmup(args.seq_len, instr_ids, draw_demo)
        ids_arr[k] = ids
        demo_counts.append(n_demos)
        if (k + 1) % 50 == 0 or (k + 1) == args.samples:
            log(f"warmup {k + 1}/{args.samples} elapsed={time.time() - t0:.1f}s "
                f"last_demos={n_demos} reuse={rr_pool.reuse}")

    used = np.asarray(rr_pool.used, dtype=np.int64)
    unique = np.unique(used)
    dup_count = int(used.shape[0] - unique.shape[0])
    over_train = int(m_train[unique].sum())
    over_local = int(m_local[unique].sum())
    over_demo = int(m_demo[unique].sum())
    assert dup_count == 0, f"sampled indices not unique: {dup_count} duplicate draws"
    assert rr_pool.reuse == 0, f"demo pool reuse occurred: {rr_pool.reuse}"
    assert over_train == 0, f"overlap with clients.*.train: {over_train}"
    assert over_local == 0, f"overlap with clients.*.local_test: {over_local}"
    assert over_demo == 0, f"overlap with global_demo_idx: {over_demo}"

    digest = hashlib.sha256(unique.astype(np.uint32).tobytes()).hexdigest()[:16]
    per_class_draws = np.bincount(labels_all[unique], minlength=n_classes)
    np.save(os.path.join(out_dir, "warmup_input_ids.npy"), ids_arr)
    demo_counts = np.asarray(demo_counts)
    meta = {
        "source": SOURCE,
        "config": SOURCE_CONFIG,
        "split": "train",
        "tag": args.tag,
        "seed": args.seed,
        "samples": args.samples,
        "seq_len": args.seq_len,
        "dtype": "uint16",
        "tokenizer": TOKENIZER_PATH,
        "add_special_tokens": False,
        "template_hash": TEMPLATE_HASH,
        "instruction": INSTRUCTION,
        "demo_template": DEMO_FMT,
        "partition": pool_stats,
        "disjointness": {
            "sampled_draws": int(used.shape[0]),
            "sampled_unique": int(unique.shape[0]),
            "sampled_duplicates": dup_count,
            "sampled_reuse": int(rr_pool.reuse),
            "overlap_clients_train": over_train,
            "overlap_clients_local_test": over_local,
            "overlap_global_demo": over_demo,
            "sampled_idx_sha256": digest,
            "sampled_idx_min": int(unique.min()),
            "sampled_idx_max": int(unique.max()),
            "sampled_per_class_min": int(per_class_draws.min()),
            "sampled_per_class_max": int(per_class_draws.max()),
            "sampled_per_class_mean": float(per_class_draws.mean()),
        },
        "demos_per_sample": {
            "min": int(demo_counts.min()),
            "max": int(demo_counts.max()),
            "mean": float(demo_counts.mean()),
        },
        "token_cache": {"hits": int(cache.hits), "misses": int(cache.misses), "capacity": cache.capacity},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "warmup_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("=" * 64)
    log(f"warmup done: samples={args.samples} seq_len={args.seq_len} dtype=uint16 "
        f"path={os.path.join(out_dir, 'warmup_input_ids.npy')}")
    log(f"  disjointness: draws={used.shape[0]} unique={unique.shape[0]} dup={dup_count} reuse={rr_pool.reuse} "
        f"overlap(train/local/demo)={over_train}/{over_local}/{over_demo} sha256={digest}")
    log(f"  source pool: available={pool_stats['available_pool']} "
        f"used={unique.shape[0]} ({100.0 * unique.shape[0] / pool_stats['available_pool']:.1f}%) "
        f"demos/sample min/mean/max={int(demo_counts.min())}/{demo_counts.mean():.1f}/{int(demo_counts.max())}")
    log(f"  elapsed={time.time() - t0:.1f}s out={out_dir}")


if __name__ == "__main__":
    main()
