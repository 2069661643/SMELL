# SMELL 3 build_arxiv_16k NEW — ccdv/arxiv-classification 16k 联邦分片构建（per-class Dirichlet + 头尾截断）

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # SMELL 3 build_arxiv_16k HF_ENDPOINT MODIFIED — 允许环境变量覆盖（云端 huggingface.co / 本地 hf-mirror）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # SMELL 3 build_arxiv_16k HF_HUB_DISABLE_XET MODIFIED — 允许环境变量覆盖（Xet 桥不可达时设 0 走 xet 客户端）
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, "temp", "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_ROOT, "temp", "hf_cache", "datasets"))

import build_discovery_16k as base  # noqa: E402  (import 时先设置 HF mirror 环境再 import datasets)

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SOURCE = "ccdv/arxiv-classification"
SOURCE_CONFIG = "no_ref"
TOKENIZER_PATH = os.path.join(_ROOT, "third_party", "Jenga", "checkpoints", "distilbert-base-uncased")
TEXT_KEY = "text"
LABEL_KEY = "label"
TRUNCATION = "head_tail"
DEFAULT_OUT = os.path.join("dataset_v3", "arxiv_16k")
log = base.log
ClientPoolStream = base.ClientPoolStream
build_partition = base.build_partition


def parse_args():
    ap = argparse.ArgumentParser(description="Build ccdv/arxiv-classification 16k federated shards (per-class Dirichlet)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output root; writes <out>/<tag>/")
    ap.add_argument("--tag", default="a01")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--clients", type=int, default=30)
    ap.add_argument("--train-per-client", type=int, default=100)
    ap.add_argument("--local-test-per-client", type=int, default=16)
    ap.add_argument("--global-test", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--head-frac", type=float, default=0.75, help="head share of seq_len in head+tail truncation")
    ap.add_argument("--tokenizer", default=TOKENIZER_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--global-demo-frac", type=float, default=0.15, help="train share held out of client pools (Discovery semantics)")
    ap.add_argument("--global-val-frac", type=float, default=0.1, help="share of held-out train pool encoded as global_val (rest = global_train)")
    ap.add_argument("--no-global-pool", action="store_true", help="skip encoding held-out train pool into global_train/global_val")
    ap.add_argument("--max-source-docs", type=int, default=0, help="streaming row cap per split; 0 = no limit")
    ap.add_argument("--mini", action="store_true", help="4 clients x 5 train x 2 local, global 10, seq_len 2048, cap 300 docs")
    args = ap.parse_args()
    if not 0.0 < args.head_frac < 1.0:
        raise SystemExit(f"--head-frac must be in (0,1), got {args.head_frac}")
    if not 0.0 < args.global_val_frac < 1.0:
        raise SystemExit(f"--global-val-frac must be in (0,1), got {args.global_val_frac}")
    if args.mini:
        args.clients = 4
        args.train_per_client = 5
        args.local_test_per_client = 2
        args.global_test = 10
        args.seq_len = 2048
        if args.max_source_docs <= 0:
            args.max_source_docs = 300
    return args


# SMELL 3 build_arxiv_16k stream_split ADD — streaming 读取（可选 take 上限 + 列裁剪）
def stream_split(split, max_docs, columns):
    ds = load_dataset(SOURCE, SOURCE_CONFIG, split=split, streaming=True)
    if max_docs and max_docs > 0:
        ds = ds.take(max_docs)
    if columns is not None:
        ds = ds.select_columns(columns)
    return ds


# SMELL 3 build_arxiv_16k collect_labels ADD — 单遍流式收集标签（只读 label 列）
def collect_labels(split, max_docs):
    ds = stream_split(split, max_docs, [LABEL_KEY])
    label_names = list(ds.features[LABEL_KEY].names)
    labels = np.asarray([int(row[LABEL_KEY]) for row in ds], dtype=np.int64)
    assert labels.shape[0] > 0, f"{split}: empty stream"
    return labels, label_names


# SMELL 3 build_arxiv_16k stream_wanted ADD — 单遍流式，对命中索引回调 (idx, text)
def stream_wanted(split, max_docs, wanted, callback):
    n_seen = 0
    n_hit = 0
    for i, row in enumerate(stream_split(split, max_docs, [TEXT_KEY])):
        n_seen = i + 1
        if i in wanted:
            callback(i, row[TEXT_KEY])
            n_hit += 1
    missing = len(wanted) - n_hit
    assert missing == 0, f"{split}: wanted rows missing={missing} (seen={n_seen})"
    return n_seen


# SMELL 3 build_arxiv_16k encode_doc ADD — [CLS]/[SEP] 编码 + 头尾截断 + 右 padding（断言 length==min(raw,seq_len)）
def encode_doc(tok, text, seq_len, head_frac, pad_id):
    ids = tok(text, add_special_tokens=True)["input_ids"]
    raw_len = len(ids)
    if raw_len > seq_len:
        head = int(seq_len * head_frac)
        tail = seq_len - head
        ids = list(ids[:head]) + list(ids[-tail:])
    length = len(ids)
    assert length == min(raw_len, seq_len), f"length={length} raw={raw_len} seq_len={seq_len}"
    assert 0 < length <= seq_len, f"empty tokenization: length={length}"
    assert max(ids) < 65536, f"token id out of uint16 range: max={max(ids)}"
    out = np.full(seq_len, pad_id, dtype=np.uint16)
    out[:length] = np.asarray(ids, dtype=np.uint16)
    if length < seq_len:
        assert bool(np.all(out[length:] == pad_id)), f"right padding not pad_id={pad_id}"
    return out, length


# SMELL 3 build_arxiv_16k validate_ids_lengths ADD — 形状/dtype/lengths/右 padding 不变量
def validate_ids_lengths(tag, ids, lengths, pad_id):
    seq_len = ids.shape[1]
    assert ids.dtype == np.uint16, f"{tag}: ids dtype={ids.dtype}"
    assert lengths.dtype == np.int32, f"{tag}: lengths dtype={lengths.dtype}"
    assert lengths.shape == (ids.shape[0],), f"{tag}: lengths shape={lengths.shape}"
    assert int(lengths.min()) >= 1 and int(lengths.max()) <= seq_len, \
        f"{tag}: lengths range [{int(lengths.min())},{int(lengths.max())}] vs [1,{seq_len}]"
    bad = [r for r in range(ids.shape[0]) if not bool(np.all(ids[r, int(lengths[r]):] == pad_id))]
    assert not bad, f"{tag}: non-pad after length in rows {bad[:5]}"
    assert int(ids.max()) < 65536, f"{tag}: token id out of uint16 range: max={int(ids.max())}"


def validate_matrix(tag, ids, labels, lengths, pad_id, n_classes):
    validate_ids_lengths(tag, ids, lengths, pad_id)
    assert labels.dtype == np.int64, f"{tag}: labels dtype={labels.dtype}"
    assert labels.shape == (ids.shape[0],), f"{tag}: labels shape={labels.shape}"
    assert int(labels.min()) >= 0 and int(labels.max()) < n_classes, \
        f"{tag}: labels range [{int(labels.min())},{int(labels.max())}] vs [0,{n_classes})"


def length_stats(lengths, seq_len):
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(round(float(lengths.mean()), 1)),
        "truncated": int(np.sum(lengths >= seq_len)),
        "rows": int(lengths.shape[0]),
    }


# SMELL 3 build_arxiv_16k label_histogram ADD — 标签计数直方图（按 label_names 顺序，JSON 友好 dict）
def label_histogram(labels, label_names):
    counts = np.bincount(labels, minlength=len(label_names))
    return {name: int(c) for name, c in zip(label_names, counts)}


def write_split(dirpath, prefix, ids, labels, lengths):
    os.makedirs(dirpath, exist_ok=True)
    np.save(os.path.join(dirpath, f"{prefix}_input_ids.npy"), ids)
    np.save(os.path.join(dirpath, f"{prefix}_labels.npy"), labels)
    np.save(os.path.join(dirpath, f"{prefix}_lengths.npy"), lengths)


def main():
    t0 = time.time()
    args = parse_args()
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log(f"build start: out={out_dir} mini={args.mini} alpha={args.alpha} clients={args.clients} "
        f"train/client={args.train_per_client} local-test/client={args.local_test_per_client} "
        f"global-test={args.global_test} seq-len={args.seq_len} head-frac={args.head_frac} "
        f"max-source-docs={args.max_source_docs} seed={args.seed}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        raise SystemExit(f"tokenizer has no pad token: {args.tokenizer}")
    pad_id = int(tok.pad_token_id)
    tok.model_max_length = args.seq_len  # SMELL 3 build_arxiv_16k model_max_length ADD — 自行头尾截断到 seq_len，避免 512 上限告警
    tok.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True  # SMELL 3 build_arxiv_16k tokenizer_warning ADD — 长文由 encode_doc 截断，静音误导告警
    log(f"tokenizer ready: {type(tok).__name__} vocab={tok.vocab_size} pad_token_id={pad_id}")

    log(f"loading {SOURCE} [{SOURCE_CONFIG}] train/test (streaming) ...")
    labels_all, label_names = collect_labels("train", args.max_source_docs)
    test_labels, test_label_names = collect_labels("test", args.max_source_docs)
    assert test_label_names == label_names, f"label names mismatch: {test_label_names} != {label_names}"
    n_classes = len(label_names)
    if args.global_test > test_labels.shape[0]:
        raise SystemExit(f"--global-test {args.global_test} > test rows {test_labels.shape[0]}")
    log(f"loaded: train={labels_all.shape[0]} test={test_labels.shape[0]} classes={n_classes} "
        f"elapsed={time.time() - t0:.1f}s")

    client_source, global_demo_source, pools = build_partition(
        labels_all, args.clients, n_classes, args.alpha, args.global_demo_frac, args.seed
    )
    log(f"partition done: client_source={len(client_source)} held_out={len(global_demo_source)} "
        f"elapsed={time.time() - t0:.1f}s")

    clients_dir = os.path.join(out_dir, "clients")
    plans = []
    slot_map = collections.defaultdict(list)
    reuse_per_client = {}
    n_train = args.train_per_client
    n_local = args.local_test_per_client
    for i in range(args.clients):
        cname = f"client_{i:02d}"
        flat = [idx for c in range(n_classes) for idx in pools[i].get(c, ())]
        if not flat:
            raise RuntimeError(f"{cname} has empty pool")
        rng_i = np.random.default_rng([args.seed, 101, i])
        arr = np.asarray(flat, dtype=np.int64)
        rng_i.shuffle(arr)
        stream = ClientPoolStream(arr, rng_i)
        draws = [int(stream.draw()) for _ in range(n_train + n_local)]
        train_used = sorted(set(draws[:n_train]))
        local_used = sorted(set(draws[n_train:]) - set(train_used))
        plan = {
            "cname": cname,
            "dir": os.path.join(clients_dir, cname),
            "pool_size": len(flat),
            "draws": draws,
            "train_used": train_used,
            "local_used": local_used,
            "train_ids": np.empty((n_train, args.seq_len), dtype=np.uint16),
            "train_labels": np.empty(n_train, dtype=np.int64),
            "train_lengths": np.empty(n_train, dtype=np.int32),
            "local_ids": np.empty((n_local, args.seq_len), dtype=np.uint16),
            "local_labels": np.empty(n_local, dtype=np.int64),
            "local_lengths": np.empty(n_local, dtype=np.int32),
        }
        for k in range(n_train):
            slot_map[draws[k]].append((i, "train", k))
        for k in range(n_local):
            slot_map[draws[n_train + k]].append((i, "local", k))
        reuse_per_client[cname] = int(stream.reuse)
        plans.append(plan)

    wanted_train = set(slot_map.keys())
    log(f"client plan: wanted train rows={len(wanted_train)} "
        f"(draws={args.clients * (n_train + n_local)}) elapsed={time.time() - t0:.1f}s")

    def fill_train(idx, text):
        ids, length = encode_doc(tok, text, args.seq_len, args.head_frac, pad_id)
        for ci, kind, k in slot_map[idx]:
            plan = plans[ci]
            plan[f"{kind}_ids"][k] = ids
            plan[f"{kind}_lengths"][k] = length
            plan[f"{kind}_labels"][k] = int(labels_all[idx])

    seen = stream_wanted("train", args.max_source_docs, wanted_train, fill_train)
    log(f"train tokens filled: stream_rows={seen} elapsed={time.time() - t0:.1f}s")

    partition_clients = {}
    total_sequences = 0
    train_lengths_all = []
    local_lengths_all = []
    for plan in plans:
        validate_matrix(f"{plan['cname']}/train", plan["train_ids"], plan["train_labels"],
                        plan["train_lengths"], pad_id, n_classes)
        validate_matrix(f"{plan['cname']}/local_test", plan["local_ids"], plan["local_labels"],
                        plan["local_lengths"], pad_id, n_classes)
        write_split(plan["dir"], "train", plan["train_ids"], plan["train_labels"], plan["train_lengths"])
        write_split(plan["dir"], "local_test", plan["local_ids"], plan["local_labels"], plan["local_lengths"])
        partition_clients[plan["cname"]] = {"train": plan["train_used"], "local_test": plan["local_used"]}
        train_lengths_all.append(plan["train_lengths"])
        local_lengths_all.append(plan["local_lengths"])
        total_sequences += n_train + n_local
        log(f"{plan['cname']} done: pool={plan['pool_size']} train={n_train} local_test={n_local} "
            f"reuse={reuse_per_client[plan['cname']]}")

    rng_test = np.random.default_rng([args.seed, 303])
    global_query_idx = rng_test.permutation(test_labels.shape[0])[:args.global_test].tolist()
    global_ids = np.empty((args.global_test, args.seq_len), dtype=np.uint16)
    global_labels_arr = np.empty(args.global_test, dtype=np.int64)
    global_lengths = np.empty(args.global_test, dtype=np.int32)
    gpos = {int(idx): j for j, idx in enumerate(global_query_idx)}

    def fill_global(idx, text):
        ids, length = encode_doc(tok, text, args.seq_len, args.head_frac, pad_id)
        j = gpos[idx]
        global_ids[j] = ids
        global_lengths[j] = length
        global_labels_arr[j] = int(test_labels[idx])

    seen = stream_wanted("test", args.max_source_docs, set(gpos.keys()), fill_global)
    validate_matrix("global_test", global_ids, global_labels_arr, global_lengths, pad_id, n_classes)
    write_split(out_dir, "global_test", global_ids, global_labels_arr, global_lengths)
    total_sequences += args.global_test
    log(f"global_test done: n={args.global_test} from test rows={seen} elapsed={time.time() - t0:.1f}s")

    # SMELL 3 build_arxiv_16k global_pool ADD — 预留 train 池（global_demo_source）编码为 Step 1 全局训练/验证集
    global_pool_meta = {
        "source": "train(global_demo_idx)",
        "val_frac": args.global_val_frac,
        "train": 0,
        "val": 0,
        "skipped": bool(args.no_global_pool),
    }
    global_train_lengths = None
    global_val_lengths = None
    if args.no_global_pool:
        log(f"global pool skipped: --no-global-pool held_out={len(global_demo_source)} elapsed={time.time() - t0:.1f}s")
    else:
        gpool_idx = np.asarray(global_demo_source, dtype=np.int64)
        n_gpool = int(gpool_idx.shape[0])
        n_gval = int(np.floor(args.global_val_frac * n_gpool + 0.5))
        if not 0 < n_gval < n_gpool:
            raise SystemExit(f"global val split n={n_gval} not in (0,{n_gpool}); adjust --global-val-frac")
        order = np.random.default_rng([args.seed, 505]).permutation(n_gpool)
        global_train_idx = gpool_idx[order[n_gval:]]
        global_val_idx = gpool_idx[order[:n_gval]]
        n_gtrain = int(global_train_idx.shape[0])
        gslots = {}
        for k, idx in enumerate(global_train_idx.tolist()):
            assert int(idx) not in gslots, f"duplicate global train idx={idx}"
            gslots[int(idx)] = ("global_train", k)
        for k, idx in enumerate(global_val_idx.tolist()):
            assert int(idx) not in gslots, f"duplicate global val idx={idx}"
            gslots[int(idx)] = ("global_val", k)
        assert len(gslots) == n_gtrain + n_gval, f"global pool slots={len(gslots)} != {n_gtrain + n_gval}"
        global_train_ids = np.empty((n_gtrain, args.seq_len), dtype=np.uint16)
        global_train_labels_arr = np.empty(n_gtrain, dtype=np.int64)
        global_train_lengths = np.empty(n_gtrain, dtype=np.int32)
        global_val_ids = np.empty((n_gval, args.seq_len), dtype=np.uint16)
        global_val_labels_arr = np.empty(n_gval, dtype=np.int64)
        global_val_lengths = np.empty(n_gval, dtype=np.int32)

        def fill_global_pool(idx, text):
            ids, length = encode_doc(tok, text, args.seq_len, args.head_frac, pad_id)
            prefix, k = gslots[idx]
            if prefix == "global_train":
                global_train_ids[k] = ids
                global_train_lengths[k] = length
                global_train_labels_arr[k] = int(labels_all[idx])
            else:
                global_val_ids[k] = ids
                global_val_lengths[k] = length
                global_val_labels_arr[k] = int(labels_all[idx])

        seen_g = stream_wanted("train", args.max_source_docs, set(gslots.keys()), fill_global_pool)
        validate_matrix("global_train", global_train_ids, global_train_labels_arr, global_train_lengths, pad_id, n_classes)
        validate_matrix("global_val", global_val_ids, global_val_labels_arr, global_val_lengths, pad_id, n_classes)
        write_split(out_dir, "global_train", global_train_ids, global_train_labels_arr, global_train_lengths)
        write_split(out_dir, "global_val", global_val_ids, global_val_labels_arr, global_val_lengths)
        global_pool_meta.update({
            "train": n_gtrain,
            "val": n_gval,
            "train_label_hist": label_histogram(global_train_labels_arr, label_names),
            "val_label_hist": label_histogram(global_val_labels_arr, label_names),
        })
        log(f"global pool done: train={n_gtrain} val={n_gval} val_frac={args.global_val_frac} "
            f"from train rows={seen_g} elapsed={time.time() - t0:.1f}s")

    counts = {
        "clients": args.clients,
        "train_per_client": n_train,
        "local_test_per_client": n_local,
        "global_test": args.global_test,
        "train_split_size": int(labels_all.shape[0]),
        "test_split_size": int(test_labels.shape[0]),
        "client_source_size": len(client_source),
        "global_demo_source_size": len(global_demo_source),
        "total_sequences": total_sequences,
        "total_tokens": total_sequences * args.seq_len,
        "max_source_docs": args.max_source_docs,
    }
    length_stats_all = {
        "train": length_stats(np.concatenate(train_lengths_all), args.seq_len),
        "local_test": length_stats(np.concatenate(local_lengths_all), args.seq_len),
        "global_test": length_stats(global_lengths, args.seq_len),
    }
    # SMELL 3 build_arxiv_16k global_pool ADD — 开启 global 池时补充 train/val 长度统计
    if global_train_lengths is not None:
        length_stats_all["global_train"] = length_stats(global_train_lengths, args.seq_len)
        length_stats_all["global_val"] = length_stats(global_val_lengths, args.seq_len)
    fallback_reuse = {
        "per_client": reuse_per_client,
        "total_client_reuse": int(sum(reuse_per_client.values())),
        "global_test_reuse": 0,
    }
    meta = {
        "source": SOURCE,
        "config": SOURCE_CONFIG,
        "text_column": TEXT_KEY,
        "label_column": LABEL_KEY,
        "alpha": args.alpha,
        "global_demo_frac": args.global_demo_frac,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "head_frac": args.head_frac,
        "truncation": TRUNCATION,
        "tokenizer": args.tokenizer,
        "add_special_tokens": True,
        "pad_token_id": pad_id,
        "label_names": label_names,
        "counts": counts,
        "length_stats": length_stats_all,
        "global_pool": global_pool_meta,
        "fallback_reuse": fallback_reuse,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # SMELL 3 build_arxiv_16k partition ADD — global_demo_idx 为预留 train 行池（arxiv 无 ICL demo），warmup 据此互斥
    partition = {
        "seed": args.seed,
        "alpha": args.alpha,
        "global_demo_frac": args.global_demo_frac,
        "train_size": int(labels_all.shape[0]),
        "test_size": int(test_labels.shape[0]),
        "client_source_size": len(client_source),
        "global_demo_source_size": len(global_demo_source),
        "clients": partition_clients,
        "global_demo_idx": sorted(int(x) for x in global_demo_source),
        "global_test_query_idx": sorted(int(x) for x in global_query_idx),
        "global_test_split": "test",
        "max_source_docs": args.max_source_docs,
    }
    with open(os.path.join(out_dir, "partition.json"), "w", encoding="utf-8") as f:
        json.dump(partition, f, ensure_ascii=False, indent=2)
    log(f"build done: sequences={total_sequences} elapsed={time.time() - t0:.1f}s out={out_dir}")


if __name__ == "__main__":
    main()
