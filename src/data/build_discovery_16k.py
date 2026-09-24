# SMELL 3 build_discovery_16k NEW — Discovery 16k ICL 联邦分片构建（per-class Dirichlet + 方案 A）

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, "temp", "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_ROOT, "temp", "hf_cache", "datasets"))

import argparse
import collections
import hashlib
import json
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SOURCE = "sileod/discovery"
SOURCE_CONFIG = "default"
TOKENIZER_PATH = os.path.join(_ROOT, "third_party", "Jenga", "checkpoints", "opt-350m")

INSTRUCTION = (
    "Given two sentence1 and sentence2, please predict the conjunction word between the two sentences. "
    "The predict answer must come from the demonstration examples with the exact format. "
    "The examples are as follows: \n"
)
DEMO_FMT = "{s1} ( ) {s2}\nthe most suitable conjunction word in the previous ( ) is  {label},\n"
QUERY_FMT = "{s1} ( ) {s2}\nthe most suitable conjunction word in the previous ( ) is  {label},"
QUERY_PREFIX_FMT = "{s1} ( ) {s2}\nthe most suitable conjunction word in the previous ( ) is  "
TEMPLATE_HASH = hashlib.sha256(
    json.dumps([INSTRUCTION, DEMO_FMT, QUERY_FMT], ensure_ascii=False).encode("utf-8")
).hexdigest()[:16]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TokenCache:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.data = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        value = self.data.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return value

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


class ClientPoolStream:
    def __init__(self, flat_pool, rng):
        self.pool = flat_pool
        self.rng = rng
        self.pos = 0
        self.reuse = 0
        self.used = []

    def draw(self):
        if self.pos < len(self.pool):
            idx = int(self.pool[self.pos])
            self.pos += 1
        else:
            idx = int(self.pool[int(self.rng.integers(0, len(self.pool)))])
            self.reuse += 1
        self.used.append(idx)
        return idx


class RoundRobinDemoPool:
    def __init__(self, by_class, rng, n_classes):
        self.rng = rng
        self.n_classes = n_classes
        self.by_class = []
        for c in range(n_classes):
            lst = list(by_class[c])
            rng.shuffle(lst)
            self.by_class.append(lst)
        self.ptr = [0] * n_classes
        self.class_ptr = 0
        self.reuse = 0
        self.used = []
        self.nonempty = [c for c in range(n_classes) if self.by_class[c]]
        if not self.nonempty:
            raise RuntimeError("global demo source is empty")

    def start_sample(self):
        self.class_ptr = 0

    def draw(self):
        c = self.class_ptr % self.n_classes
        self.class_ptr += 1
        lst = self.by_class[c]
        if not lst:
            c = self.nonempty[int(self.rng.integers(0, len(self.nonempty)))]
            lst = self.by_class[c]
        p = self.ptr[c]
        if p < len(lst):
            idx = int(lst[p])
            self.ptr[c] = p + 1
        else:
            idx = int(lst[int(self.rng.integers(0, len(lst)))])
            self.reuse += 1
        self.used.append(idx)
        return idx


def parse_args():
    ap = argparse.ArgumentParser(description="Build Discovery 16k ICL federated shards (scheme A)")
    ap.add_argument("--out", default="/home/yangyongbo118/projects/SMELL-v3/dataset_v3/discovery_16k")
    ap.add_argument("--tag", default="a01")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--clients", type=int, default=30)
    ap.add_argument("--train-per-client", type=int, default=100)
    ap.add_argument("--local-test-per-client", type=int, default=16)
    ap.add_argument("--global-test", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--global-demo-frac", type=float, default=0.15)
    ap.add_argument("--mini", action="store_true")
    args = ap.parse_args()
    if args.mini:
        args.clients = 3
        args.train_per_client = 4
        args.local_test_per_client = 2
        args.global_test = 8
    return args


def get_demo_ids(cache, tok, ds, split_tag, idx, label_name):
    key = ("demo", split_tag, idx)
    ids = cache.get(key)
    if ids is None:
        row = ds[idx]
        text = DEMO_FMT.format(s1=row["sentence1"], s2=row["sentence2"], label=label_name)
        ids = tok.encode(text, add_special_tokens=False)
        cache.put(key, ids)
    return ids


def get_prefix_ids(cache, tok, ds, split_tag, idx):
    key = ("prefix", split_tag, idx)
    ids = cache.get(key)
    if ids is None:
        row = ds[idx]
        text = QUERY_PREFIX_FMT.format(s1=row["sentence1"], s2=row["sentence2"])
        ids = tok.encode(text, add_special_tokens=False)
        cache.put(key, ids)
    return ids


def assemble_sample(seq_len, instr_ids, prefix_ids, label_ids, comma_ids, draw_demo):
    query_ids = list(prefix_ids) + list(label_ids) + list(comma_ids)
    demos = []
    total = len(instr_ids) + len(query_ids)
    while total < seq_len:
        demo_ids = draw_demo()
        if not demo_ids:
            raise RuntimeError("empty demo tokenization")
        demos.append(demo_ids)
        total += len(demo_ids)
    while demos and total - len(demos[0]) >= seq_len:
        total -= len(demos.pop(0))
    flat = [token for demo in demos for token in demo]
    cut = len(instr_ids) + len(flat) + len(query_ids) - seq_len
    if cut > 0:
        if cut > len(flat):
            raise RuntimeError(f"cannot truncate demo region: cut={cut} flat={len(flat)}")
        flat = flat[cut:]
    input_ids = instr_ids + flat + query_ids
    if len(input_ids) != seq_len:
        raise RuntimeError(f"assembled {len(input_ids)} tokens != seq_len {seq_len}")
    start = len(instr_ids) + len(flat) + len(prefix_ids)
    end = start + len(label_ids)
    if not 0 <= start < end <= seq_len:
        raise RuntimeError(f"bad answer span [{start},{end})")
    return input_ids, (start, end), len(demos)


def build_partition(labels, n_clients, n_classes, alpha, global_demo_frac, seed):
    n = labels.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_client_source = int(round(n * (1.0 - global_demo_frac)))
    client_source = perm[:n_client_source]
    global_demo_source = perm[n_client_source:]
    cs_labels = labels[client_source]
    pools = [dict() for _ in range(n_clients)]
    for c in range(n_classes):
        cls_idx = client_source[cs_labels == c]
        rng.shuffle(cls_idx)
        p = rng.dirichlet(np.full(n_clients, alpha, dtype=np.float64))
        counts = rng.multinomial(cls_idx.shape[0], p)
        diff = int(cls_idx.shape[0]) - int(counts.sum())
        if diff != 0:
            counts[int(np.argmax(p))] += diff
        offset = 0
        for i in range(n_clients):
            cnt = int(counts[i])
            if cnt > 0:
                pools[i][c] = cls_idx[offset:offset + cnt].tolist()
            offset += cnt
    return client_source.tolist(), global_demo_source.tolist(), pools


def group_by_class(indices, labels, n_classes):
    by_class = [[] for _ in range(n_classes)]
    for idx, lab in zip(indices, labels.tolist()):
        by_class[int(lab)].append(int(idx))
    return by_class


def write_matrix(dirpath, prefix, ids, labels, spans):
    os.makedirs(dirpath, exist_ok=True)
    np.save(os.path.join(dirpath, f"{prefix}_input_ids.npy"), ids)
    np.save(os.path.join(dirpath, f"{prefix}_labels.npy"), labels)
    np.save(os.path.join(dirpath, f"{prefix}_answer_spans.npy"), spans)


def main():
    t0 = time.time()
    args = parse_args()
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log(f"build start: out={out_dir} mini={args.mini} alpha={args.alpha} clients={args.clients} "
        f"train/client={args.train_per_client} local-test/client={args.local_test_per_client} "
        f"global-test={args.global_test} seq-len={args.seq_len} seed={args.seed}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
    if getattr(tok, "add_bos_token", False):
        tok.add_bos_token = False
    instr_ids = tok.encode(INSTRUCTION, add_special_tokens=False)
    comma_ids = tok.encode(",", add_special_tokens=False)

    log(f"tokenizer ready: {type(tok).__name__} vocab={tok.vocab_size} instr_tokens={len(instr_ids)}")
    log(f"loading {SOURCE} [{SOURCE_CONFIG}] train/test via mirror ...")
    train_ds = load_dataset(SOURCE, SOURCE_CONFIG, split="train").select_columns(
        ["sentence1", "sentence2", "label"]
    )
    test_ds = load_dataset(SOURCE, SOURCE_CONFIG, split="test").select_columns(
        ["sentence1", "sentence2", "label"]
    )
    label_names = list(train_ds.features["label"].names)
    n_classes = len(label_names)
    labels_all = np.asarray(train_ds["label"], dtype=np.int64)
    test_labels = np.asarray(test_ds["label"], dtype=np.int64)
    log(f"loaded: train={labels_all.shape[0]} test={test_labels.shape[0]} classes={n_classes} "
        f"elapsed={time.time() - t0:.1f}s")

    client_source, global_demo_source, pools = build_partition(
        labels_all, args.clients, n_classes, args.alpha, args.global_demo_frac, args.seed
    )
    log(f"partition done: client_source={len(client_source)} global_demo_source={len(global_demo_source)} "
        f"elapsed={time.time() - t0:.1f}s")

    client_source_labels = labels_all[np.asarray(client_source, dtype=np.int64)]
    global_demo_labels = labels_all[np.asarray(global_demo_source, dtype=np.int64)]
    global_by_class = group_by_class(global_demo_source, global_demo_labels, n_classes)

    rng_test = np.random.default_rng([args.seed, 303])
    global_query_idx = rng_test.permutation(test_labels.shape[0])[:args.global_test].tolist()
    log(f"global test queries sampled: {len(global_query_idx)}")
    train_label_ids = {
        c: tok.encode(label_names[c], add_special_tokens=False) for c in range(n_classes)
    }
    empty_labels = [c for c in range(n_classes) if not train_label_ids[c]]
    if empty_labels:
        log(f"WARNING: {len(empty_labels)} label(s) tokenize to empty ids: {empty_labels[:5]}")

    cache = TokenCache()
    clients_dir = os.path.join(out_dir, "clients")
    partition_clients = {}
    reuse_per_client = {}
    demo_counts = []
    total_sequences = 0

    for i in range(args.clients):
        cname = f"client_{i:02d}"
        cdir = os.path.join(clients_dir, cname)
        c_t0 = time.time()
        flat = [idx for c in range(n_classes) for idx in pools[i].get(c, ())]
        if not flat:
            raise RuntimeError(f"{cname} has empty pool")
        rng_i = np.random.default_rng([args.seed, 101, i])
        arr = np.asarray(flat, dtype=np.int64)
        rng_i.shuffle(arr)
        stream = ClientPoolStream(arr, rng_i)

        def make_sample(split_tag):
            qidx = stream.draw()
            qlabel = int(labels_all[qidx])
            prefix_ids = get_prefix_ids(cache, tok, train_ds, split_tag, qidx)
            def draw_demo():
                didx = stream.draw()
                dlabel = int(labels_all[didx])
                return get_demo_ids(cache, tok, train_ds, split_tag, didx, label_names[dlabel])
            ids, span, n_demos = assemble_sample(
                args.seq_len, instr_ids, prefix_ids, train_label_ids[qlabel], comma_ids, draw_demo
            )
            return ids, qlabel, span, n_demos

        n_train = args.train_per_client
        n_local = args.local_test_per_client
        train_ids = np.empty((n_train, args.seq_len), dtype=np.uint16)
        train_labels = np.empty(n_train, dtype=np.int64)
        train_spans = np.empty((n_train, 2), dtype=np.int32)
        for k in range(n_train):
            ids, label, span, n_demos = make_sample("train")
            train_ids[k] = ids
            train_labels[k] = label
            train_spans[k] = span
            demo_counts.append(n_demos)
            if (k + 1) % 10 == 0:
                log(f"{cname} train {k + 1}/{n_train} elapsed={time.time() - c_t0:.1f}s")
        train_used = sorted(set(stream.used))

        local_ids = np.empty((n_local, args.seq_len), dtype=np.uint16)
        local_labels = np.empty(n_local, dtype=np.int64)
        local_spans = np.empty((n_local, 2), dtype=np.int32)
        for k in range(n_local):
            ids, label, span, n_demos = make_sample("local_test")
            local_ids[k] = ids
            local_labels[k] = label
            local_spans[k] = span
            demo_counts.append(n_demos)
        local_used = sorted(set(stream.used) - set(train_used))

        write_matrix(cdir, "train", train_ids, train_labels, train_spans)
        write_matrix(cdir, "local_test", local_ids, local_labels, local_spans)
        partition_clients[cname] = {"train": train_used, "local_test": local_used}
        reuse_per_client[cname] = int(stream.reuse)
        total_sequences += n_train + n_local
        log(f"{cname} done: pool={len(flat)} train={n_train} local_test={n_local} "
            f"reuse={stream.reuse} elapsed={time.time() - c_t0:.1f}s")

    global_ids = np.empty((args.global_test, args.seq_len), dtype=np.uint16)
    global_labels_arr = np.empty(args.global_test, dtype=np.int64)
    global_spans = np.empty((args.global_test, 2), dtype=np.int32)
    rng_global = np.random.default_rng([args.seed, 202])
    rr_pool = RoundRobinDemoPool(global_by_class, rng_global, n_classes)
    for j, qi in enumerate(global_query_idx):
        qlabel = int(test_labels[qi])
        prefix_ids = get_prefix_ids(cache, tok, test_ds, "test", qi)
        rr_pool.start_sample()
        def draw_demo():
            didx = rr_pool.draw()
            dlabel = int(labels_all[didx])
            return get_demo_ids(cache, tok, train_ds, "train", didx, label_names[dlabel])
        ids, span, n_demos = assemble_sample(
            args.seq_len, instr_ids, prefix_ids, train_label_ids[qlabel], comma_ids, draw_demo
        )
        global_ids[j] = ids
        global_labels_arr[j] = qlabel
        global_spans[j] = span
        demo_counts.append(n_demos)
        if (j + 1) % 10 == 0 or (j + 1) == args.global_test:
            log(f"global_test {j + 1}/{args.global_test} elapsed={time.time() - t0:.1f}s")
    write_matrix(out_dir, "global_test", global_ids, global_labels_arr, global_spans)
    total_sequences += args.global_test

    counts = {
        "clients": args.clients,
        "train_per_client": args.train_per_client,
        "local_test_per_client": args.local_test_per_client,
        "global_test": args.global_test,
        "train_split_size": int(labels_all.shape[0]),
        "test_split_size": int(test_labels.shape[0]),
        "client_source_size": len(client_source),
        "global_demo_source_size": len(global_demo_source),
        "total_sequences": total_sequences,
        "total_tokens": total_sequences * args.seq_len,
        "mean_demos_per_sample": float(np.mean(demo_counts)),
        "min_demos_per_sample": int(np.min(demo_counts)),
        "max_demos_per_sample": int(np.max(demo_counts)),
        "empty_labels": empty_labels,
    }
    fallback_reuse = {
        "per_client": reuse_per_client,
        "total_client_reuse": int(sum(reuse_per_client.values())),
        "global_test_reuse": int(rr_pool.reuse),
        "token_cache_hits": int(cache.hits),
        "token_cache_misses": int(cache.misses),
    }
    meta = {
        "source": SOURCE,
        "config": SOURCE_CONFIG,
        "alpha": args.alpha,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "tokenizer": TOKENIZER_PATH,
        "template_hash": TEMPLATE_HASH,
        "instruction": INSTRUCTION,
        "demo_template": DEMO_FMT,
        "query_template": QUERY_FMT,
        "add_special_tokens": False,
        "label_names": label_names,
        "counts": counts,
        "fallback_reuse": fallback_reuse,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    partition = {
        "seed": args.seed,
        "alpha": args.alpha,
        "train_size": int(labels_all.shape[0]),
        "test_size": int(test_labels.shape[0]),
        "client_source_size": len(client_source),
        "global_demo_source_size": len(global_demo_source),
        "clients": partition_clients,
        "global_demo_idx": sorted(set(rr_pool.used)),
        "global_test_query_idx": sorted(global_query_idx),
    }
    with open(os.path.join(out_dir, "partition.json"), "w", encoding="utf-8") as f:
        json.dump(partition, f, ensure_ascii=False, indent=2)
    log(f"build done: sequences={total_sequences} elapsed={time.time() - t0:.1f}s out={out_dir}")


if __name__ == "__main__":
    main()
