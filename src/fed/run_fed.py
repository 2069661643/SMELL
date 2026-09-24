# SMELL 3 run_fed NEW — 串行 FedAvg（ZOO+LoRA）实验入口：CLI、逐轮指标 JSONL、可选 G-PPL 子进程评测

import argparse
import json
import os
import random
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MODEL_DIR = REPO / "third_party" / "Jenga" / "checkpoints" / "opt-350m"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def parse_args():
    parser = argparse.ArgumentParser(description="Serial FedAvg with ZOO+LoRA on Discovery 16k")
    parser.add_argument("--data-root", default="dataset_v3/discovery_16k")
    parser.add_argument("--tag", default="a01")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight", choices=("samples", "equal"), default="samples")
    parser.add_argument("--catv", choices=("on", "off"), default="off")
    parser.add_argument("--sparse", type=float, default=0.4)
    parser.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES before importing torch")
    parser.add_argument("--max-clients", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--zo-eps", type=float, default=1e-3)
    parser.add_argument("--zo-directions", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=0, help="0 = off; else run ppl.py every N rounds")
    parser.add_argument("--out-root", default="logs/fed")
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def discover_clients(data_root, tag, max_clients=0):
    clients_dir = data_root / tag / "clients"
    if not clients_dir.exists():
        return []
    names = sorted(entry.name for entry in clients_dir.iterdir()
                   if entry.is_dir() and entry.name.startswith("client_"))
    if max_clients > 0:
        names = names[:max_clients]
    return names


def _count_of(entry):
    if isinstance(entry, (list, tuple)):
        return len(entry)
    if not isinstance(entry, dict):
        return None
    for key in ("num_samples", "sample_count", "count", "n"):
        if key in entry:
            return int(entry[key])
    for key in ("train", "source_idx", "indices", "idx"):
        if key in entry and isinstance(entry[key], (list, tuple)):
            return len(entry[key])
    return None


def load_partition_counts(path):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("clients", payload) if isinstance(payload, dict) else payload
    counts = {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("client_id") or entry.get("name") or entry.get("id")
            count = _count_of(entry)
            if name is not None and count is not None:
                counts[str(name).replace("client_", "")] = count
    elif isinstance(entries, dict):
        for name, entry in entries.items():
            count = _count_of(entry)
            if count is not None:
                counts[str(name).replace("client_", "")] = count
    return counts


def iter_batches(input_ids_np, num_samples, device):
    import torch
    for index in range(num_samples):
        row = torch.from_numpy(input_ids_np[index].astype(np.int64))
        yield row.unsqueeze(0).to(device)


def delta_norm(delta):
    import torch
    return float(sum(torch.sum(tensor.float() ** 2) for tensor in delta.values()) ** 0.5)


def sample_pairwise_cos(deltas, max_clients=5):
    import torch
    vectors = [torch.cat([delta[key].reshape(-1).float() for key in delta])
               for delta in deltas[:max_clients]]
    values = []
    for left, right in combinations(range(len(vectors)), 2):
        denom = float(vectors[left].norm() * vectors[right].norm())
        if denom > 0:
            values.append(float(torch.dot(vectors[left], vectors[right]) / denom))
    return (sum(values) / len(values)) if values else None


def run_global_eval(args, model, out_dir, metrics_path, round_idx):
    from src.fed.serial_fedavg import append_metrics
    adapter_dir = out_dir / f"adapter_round{round_idx:03d}"
    model.save_pretrained(str(adapter_dir))
    eval_out = out_dir / f"eval_round{round_idx:03d}.json"
    command = [
        sys.executable, str(REPO / "src" / "eval" / "ppl.py"),
        "--model-dir", args.model_dir,
        "--data-root", str(resolve(args.data_root)),
        "--tag", args.tag,
        "--split", "global",
        "--adapter", str(adapter_dir),
        "--sparse", str(args.sparse),
        "--out", str(eval_out),
    ]
    proc = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0 or not eval_out.exists():
        append_metrics(metrics_path, {
            "event": "eval", "round": round_idx, "status": "failed",
            "returncode": proc.returncode, "stderr_tail": proc.stderr[-500:],
        })
        print(f"[fed] eval round {round_idx} FAILED rc={proc.returncode}: {proc.stderr.strip()[-200:]}")
        return None
    payload = json.loads(eval_out.read_text(encoding="utf-8"))
    append_metrics(metrics_path, {"event": "eval", "round": round_idx, "status": "ok", **payload})
    print(f"[fed] eval round {round_idx} full_ppl_token={payload['full_ppl_token']} "
          f"answer_ppl_token={payload['answer_ppl_token']}")
    return payload


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.catv == "on":
        raise SystemExit(
            "CATVSelector is not implemented yet (placeholder in src/models/token_selector.py) — "
            "run with --catv off for now"
        )

    import torch
    from jenga.models.modeling_opt import OPTForCausalLM
    from jenga.utils.config_utils import get_opt_qk

    from src.fed.serial_fedavg import ClientRunner, ServerAggregator, append_metrics
    from src.models.position_embed import ensure_positions
    from src.train.lora import build_lora_model, count_trainable, get_trainable_state_dict, load_trainable_state_dict

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_root = resolve(args.data_root)
    out_dir = resolve(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    client_names = discover_clients(data_root, args.tag, args.max_clients)
    if not client_names:
        raise SystemExit(f"no client dirs under {data_root / args.tag / 'clients'}")
    partition_counts = load_partition_counts(data_root / args.tag / "partition.json")
    # SMELL 3 position_embed ADD — 以 client 训练序列长度扩展位置嵌入（16k 数据）
    first_train_ids = np.load(data_root / args.tag / "clients" / client_names[0] / "train_input_ids.npy",
                              mmap_mode="r")
    train_seq_len = int(first_train_ids.shape[1])

    config = get_opt_qk(model_name=args.model_dir, flash_attention=True, pool_size=64,
                        thresh=args.sparse)
    model = OPTForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, config=config)
    model = ensure_positions(model, train_seq_len)
    model = build_lora_model(model, r=8, targets=LORA_TARGETS)
    model = model.cuda().train()
    global_state = get_trainable_state_dict(model)
    trainable_params = count_trainable(model)
    aggregator = ServerAggregator(args.weight)
    bytes_per_param = 2

    run_config = {
        **vars(args),
        "resolved_data_root": str(data_root),
        "resolved_out_dir": str(out_dir),
        "model_dir": args.model_dir,
        "client_names": client_names,
        "partition_counts": partition_counts,
        "trainable_params": trainable_params,
        "lora_targets": LORA_TARGETS,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    print(f"[fed] tag={args.tag} clients={len(client_names)} trainable_params={trainable_params} "
          f"weight={args.weight} sparse={args.sparse} lr={args.lr} seed={args.seed}")
    print(f"[fed] data={data_root} out={out_dir}")
    if partition_counts:
        covered = sum(1 for name in client_names
                      if partition_counts.get(name.replace("client_", "")) is not None)
        print(f"[fed] partition.json counts loaded for {covered}/{len(client_names)} clients")

    for round_idx in range(args.rounds):
        round_started = time.time()
        deltas, counts, losses, client_stats = [], [], [], {}
        for name in client_names:
            client_dir = data_root / args.tag / "clients" / name
            input_ids_np = np.load(client_dir / "train_input_ids.npy")
            num_samples = len(input_ids_np)
            if args.max_train_samples > 0:
                num_samples = min(num_samples, args.max_train_samples)

            load_trainable_state_dict(model, global_state)
            runner = ClientRunner(model, local_steps=args.local_steps, lr=args.lr,
                                  zo_eps=args.zo_eps, zo_directions=args.zo_directions)
            result = runner.run(iter_batches(input_ids_np, num_samples, model.device))
            deltas.append(result["delta"])
            counts.append(num_samples)
            losses.append(result["train_loss"])
            client_stats[name] = {
                "samples": num_samples,
                "steps": result["steps"],
                "train_loss": result["train_loss"],
                "delta_norm": delta_norm(result["delta"]),
            }

        aggregated = aggregator.aggregate(deltas, counts)
        global_state = {key: global_state[key] + aggregated[key] for key in global_state}
        load_trainable_state_dict(model, global_state)

        valid = [(loss, count) for loss, count in zip(losses, counts) if loss is not None]
        train_loss_mean = (
            sum(loss * count for loss, count in valid) / sum(count for _, count in valid)
            if valid else None
        )
        norms = [stats["delta_norm"] for stats in client_stats.values()]
        record = {
            "event": "round",
            "round": round_idx,
            "num_clients": len(client_names),
            "total_samples": int(sum(counts)),
            "train_loss_mean": train_loss_mean,
            "client_stats": client_stats,
            "delta_norm_mean": sum(norms) / len(norms),
            "delta_norm_min": min(norms),
            "delta_norm_max": max(norms),
            "cos_mean_sampled": sample_pairwise_cos(deltas),
            "communication_bytes": len(client_names) * trainable_params * bytes_per_param,
            "weight": args.weight,
            "lr": args.lr,
            "local_steps": args.local_steps,
            "zo_directions": args.zo_directions,
            "round_seconds": time.time() - round_started,
        }
        append_metrics(metrics_path, record)
        print(f"[fed] round {round_idx}/{args.rounds - 1} loss={train_loss_mean} "
              f"delta_norm={record['delta_norm_mean']:.4e} "
              f"cos={record['cos_mean_sampled']} time={record['round_seconds']:.1f}s")

        if args.eval_every > 0 and (round_idx + 1) % args.eval_every == 0:
            run_global_eval(args, model, out_dir, metrics_path, round_idx)

    print(f"[fed] done; metrics={metrics_path}")


if __name__ == "__main__":
    main()
