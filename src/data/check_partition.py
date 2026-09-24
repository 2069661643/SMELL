# SMELL 3 check_partition NEW — 校验 Discovery 16k 数据分片（形状/跨度/直方图/JS 散度/互斥）

import argparse
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

SPLITS = ("train", "local_test")


def log(msg):
    print(msg, flush=True)


def js_divergence(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def fmt_topk(counts, names, k=5):
    order = np.argsort(-counts)[:k]
    return ", ".join(f"{names[int(i)]}:{int(counts[int(i)])}" for i in order if counts[int(i)] > 0)


def main():
    ap = argparse.ArgumentParser(description="Check Discovery 16k federated shard invariants")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))

    with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(root, "partition.json"), encoding="utf-8") as f:
        part = json.load(f)

    seq_len = int(meta["seq_len"])
    names = list(meta["label_names"])
    n_classes = len(names)
    counts_meta = meta["counts"]
    errors = []

    def expect(cond, ok, bad, quiet=False):
        if cond:
            if not quiet:
                log("  OK   " + ok)
        else:
            log("  FAIL " + bad)
            errors.append(bad)

    log(f"[check] root={root}")
    log(f"[check] source={meta.get('source')} alpha={meta['alpha']} seed={meta['seed']} "
        f"seq_len={seq_len} template_hash={meta['template_hash']}")
    expect(seq_len % 64 == 0, f"seq_len={seq_len} divisible by 64", f"seq_len={seq_len} not divisible by 64")

    tok = AutoTokenizer.from_pretrained(meta["tokenizer"], use_fast=True)
    pad_id = tok.pad_token_id

    def check_matrix(tag, dirpath, prefix, expect_rows, print_decoded=False):
        ids = np.load(os.path.join(dirpath, f"{prefix}_input_ids.npy"))
        labels = np.load(os.path.join(dirpath, f"{prefix}_labels.npy"))
        spans = np.load(os.path.join(dirpath, f"{prefix}_answer_spans.npy"))
        n_before = len(errors)
        expect(ids.dtype == np.uint16, f"{tag}: input_ids dtype=uint16",
               f"{tag}: dtype={ids.dtype} != uint16", quiet=True)
        expect(ids.shape == (expect_rows, seq_len), f"{tag}: shape=({expect_rows},{seq_len})",
               f"{tag}: shape={ids.shape} != ({expect_rows},{seq_len})", quiet=True)
        expect(labels.dtype == np.int64, f"{tag}: labels dtype=int64",
               f"{tag}: labels dtype={labels.dtype}", quiet=True)
        expect(labels.shape == (expect_rows,), f"{tag}: labels shape correct",
               f"{tag}: labels shape={labels.shape}", quiet=True)
        expect(spans.dtype == np.int32, f"{tag}: spans dtype=int32",
               f"{tag}: spans dtype={spans.dtype}", quiet=True)
        expect(spans.shape == (expect_rows, 2), f"{tag}: spans shape correct",
               f"{tag}: spans shape={spans.shape}", quiet=True)
        counts = np.zeros(n_classes)
        if ids.shape == (expect_rows, seq_len) and labels.shape == (expect_rows,):
            expect(int(labels.min()) >= 0 and int(labels.max()) < n_classes,
                   f"{tag}: labels within [0,{n_classes})",
                   f"{tag}: labels out of range [{int(labels.min())},{int(labels.max())})", quiet=True)
            start, end = spans[:, 0].astype(np.int64), spans[:, 1].astype(np.int64)
            bad_spans = [(r, int(start[r]), int(end[r])) for r in range(expect_rows)
                         if not (0 <= start[r] < end[r] <= seq_len)]
            expect(not bad_spans, f"{tag}: all {expect_rows} spans in range",
                   f"{tag}: bad spans {bad_spans[:5]}", quiet=True)
            decoded = [tok.decode(ids[r, start[r]:end[r]].tolist(), skip_special_tokens=False).strip()
                       for r in range(expect_rows)]
            mism = [(r, decoded[r], names[int(labels[r])]) for r in range(expect_rows)
                    if decoded[r] != names[int(labels[r])]]
            expect(not mism, f"{tag}: decoded spans == label names (all {expect_rows})",
                   f"{tag}: decode mismatch {mism[:5]}", quiet=True)
            expect(not bool(np.any(ids == pad_id)), f"{tag}: no pad token (id={pad_id})",
                   f"{tag}: contains pad token id={pad_id}", quiet=True)
            counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
            if print_decoded:
                for r in range(min(5, expect_rows)):
                    log(f"    {tag}[{r}] label={names[int(labels[r])]!r} "
                        f"span=[{int(start[r])},{int(end[r])}) decoded={decoded[r]!r}")
        if len(errors) == n_before:
            log(f"  OK   {tag}: n={expect_rows} seq_len={seq_len} dtype/span/decode/pad all consistent")
        return counts

    expect(os.path.exists(os.path.join(root, "global_test_input_ids.npy")),
           "global_test files present", "global_test files missing")
    global_counts = check_matrix(
        "global_test", root, "global_test", int(counts_meta["global_test"]), print_decoded=True
    )

    clients = sorted(part["clients"].keys())
    expect(len(clients) == int(counts_meta["clients"]),
           f"partition.json has {len(clients)} clients",
           f"partition.json clients={len(clients)} != meta {counts_meta['clients']}")
    client_counts = {}
    for key in clients:
        cdir = os.path.join(root, "clients", key)
        expect(os.path.isdir(cdir), f"{key}: dir present", f"{key}: missing dir {cdir}")
        train_counts = check_matrix(f"{key}/train", cdir, "train", int(counts_meta["train_per_client"]))
        local_counts = check_matrix(
            f"{key}/local_test", cdir, "local_test", int(counts_meta["local_test_per_client"]),
            print_decoded=(key == clients[0]),
        )
        client_counts[key] = (train_counts, local_counts)

    labels_by_client = {k: v[0] + v[1] for k, v in client_counts.items()}
    log("[check] label histogram (client train+local_test) top5:")
    for key in clients:
        log(f"    {key} n={int(labels_by_client[key].sum())} top5=[{fmt_topk(labels_by_client[key], names)}]")
    log(f"[check] label histogram (global_test) top5: [{fmt_topk(global_counts, names)}]")

    pairs = []
    for a in range(len(clients)):
        for b in range(a + 1, len(clients)):
            pa, pb = labels_by_client[clients[a]], labels_by_client[clients[b]]
            if pa.sum() > 0 and pb.sum() > 0:
                pairs.append((js_divergence(pa, pb), clients[a], clients[b]))
    if pairs:
        mn = min(pairs, key=lambda x: x[0])
        mx = max(pairs, key=lambda x: x[0])
        mean_js = float(np.mean([p[0] for p in pairs]))
        log(f"[check] JS pairwise (clients): min={mn[0]:.4f} ({mn[1]} vs {mn[2]}) "
            f"max={mx[0]:.4f} ({mx[1]} vs {mx[2]}) mean={mean_js:.4f}")
    else:
        log("[check] JS pairwise (clients): N/A")
    vs_global = []
    for key in clients:
        p = labels_by_client[key]
        if p.sum() > 0 and global_counts.sum() > 0:
            vs_global.append((js_divergence(p, global_counts), key))
    if vs_global:
        mn = min(vs_global, key=lambda x: x[0])
        mx = max(vs_global, key=lambda x: x[0])
        log(f"[check] JS client-vs-global: min={mn[0]:.4f} ({mn[1]}) max={mx[0]:.4f} ({mx[1]})")
    else:
        log("[check] JS client-vs-global: N/A")

    train_size = int(part["train_size"])
    test_size = int(part["test_size"])
    all_train, all_local = set(), set()
    for key in clients:
        tr = part["clients"][key]["train"]
        lo = part["clients"][key]["local_test"]
        st, sl = set(tr), set(lo)
        expect(tr == sorted(set(tr)) and lo == sorted(set(lo)),
               f"{key}: partition lists sorted/unique", f"{key}: partition lists not sorted/unique")
        expect(not (st & sl), f"{key}: train/local_test disjoint",
               f"{key}: train/local_test overlap n={len(st & sl)}")
        expect(not (st & all_train), f"{key}: train disjoint from previous clients",
               f"{key}: train overlap with previous clients n={len(st & all_train)}")
        expect(not (sl & all_local), f"{key}: local_test disjoint from previous clients",
               f"{key}: local_test overlap with previous clients n={len(sl & all_local)}")
        over = (st | sl) - set(range(train_size))
        expect(not over, f"{key}: all idx < train_size={train_size}", f"{key}: idx out of range n={len(over)}")
        all_train |= st
        all_local |= sl
    global_demo = set(part["global_demo_idx"])
    global_q = set(part["global_test_query_idx"])
    expect(len(global_demo) == len(part["global_demo_idx"]),
           f"global_demo_idx unique n={len(global_demo)}", "global_demo_idx has duplicates")
    expect(not (global_demo & (all_train | all_local)),
           "global_demo disjoint from all client pools",
           f"global_demo overlaps clients n={len(global_demo & (all_train | all_local))}")
    expect(len(global_q) == len(part["global_test_query_idx"]),
           f"global_test_query_idx unique n={len(global_q)}", "global_test_query_idx has duplicates")
    expect(len(global_q) == int(counts_meta["global_test"]),
           f"global_test_query_idx n={len(global_q)} == meta",
           f"global_test_query_idx n={len(global_q)} != meta {counts_meta['global_test']}")
    expect(global_q and max(global_q) < test_size,
           f"global_test_query_idx within test_size={test_size}",
           f"global_test_query_idx out of range (test_size={test_size})")
    expect(not (global_q & global_demo),
           "global_test_query disjoint from global_demo_idx",
           f"test queries overlap global demos n={len(global_q & global_demo)}")

    log(f"[check] fallback/reuse counters: {json.dumps(meta.get('fallback_reuse', {}), ensure_ascii=False)}")
    client_reuse = int(meta.get("fallback_reuse", {}).get("total_client_reuse", 0))
    if client_reuse > 0:
        log(f"  WARN client pool reuse occurred ({client_reuse} draws); partition lists record first-use only")

    log("=" * 64)
    log(f"[check] summary: clients={len(clients)} total_train={int(sum(len(part['clients'][k]['train']) for k in clients))} "
        f"total_local={int(sum(len(part['clients'][k]['local_test']) for k in clients))} "
        f"global_demo_used={len(global_demo)} global_test={len(global_q)} errors={len(errors)}")
    if errors:
        log(f"[check] FAILED with {len(errors)} error(s)")
        sys.exit(1)
    log("[check] PASSED (0 errors)")
    sys.exit(0)


if __name__ == "__main__":
    main()
