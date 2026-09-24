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
# SMELL 3 CATV block size ADD — Jenga pool_size（16k / 64 = 256 块/层）
BLOCK_SIZE = 64


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
    # SMELL 3 run_fed catv args ADD — CATV anchor ratio r（默认 sparse/2）与投票归一化方式
    parser.add_argument("--catv-r", type=float, default=None,
                        help="CATV anchor ratio r; default None => sparse/2 (requires r < s and s + r <= 1)")
    parser.add_argument("--catv-normalize", choices=("off", "sum"), default="off",
                        help="off = raw summed votes (paper); sum = each client per-layer vote / its sum (v2 legacy)")
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
    # SMELL 3 run_fed catv REWRITTEN — 解析/校验 CATV 锚比例（r < s 且 s + r <= 1，默认 r = s/2）
    catv_on = args.catv == "on"
    catv_r = None
    if catv_on:
        catv_r = args.catv_r if args.catv_r is not None else args.sparse / 2.0
        if catv_r <= 0.0 or catv_r >= args.sparse:
            raise SystemExit(f"CATV requires 0 < r < s, got r={catv_r} s={args.sparse}")
        if args.sparse + catv_r > 1.0 + 1e-9:
            raise SystemExit(f"CATV requires s + r <= 1, got s={args.sparse} r={catv_r}")

    import torch
    # SMELL 3 run_fed imports MODIFIED — 使用 src/ 可编辑的 OPT 拷贝（含 CATV 掩码注入）
    from src.models.modeling_opt_smell import OPTForCausalLM
    from jenga.utils.config_utils import get_opt_qk

    from src.fed.serial_fedavg import ClientRunner, ServerAggregator, append_metrics
    from src.models.position_embed import ensure_positions
    # SMELL 3 run_fed imports ADD — CATV 服务器端掩码计算与 IR 指标
    from src.models.token_selector import compute_consensus_mask, mask_intersection_rate
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
    # SMELL 3 run_fed n_blocks ADD — CATV: 每层块数 N = seq_len / pool_size
    n_blocks = train_seq_len // BLOCK_SIZE

    config = get_opt_qk(model_name=args.model_dir, flash_attention=True, pool_size=64,
                        thresh=args.sparse)
    model = OPTForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, config=config)
    # SMELL 3 run_fed base_config ADD — 保存 LoRA 包装前的 OPTConfig 引用（vote_callback/consensus_mask 挂载点）
    base_config = model.config
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
        # SMELL 3 run_fed catv config ADD — 生效的锚比例与块数
        "resolved_catv_r": catv_r,
        "n_blocks": n_blocks,
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

    # SMELL 3 run_fed catv round0 ADD — 第 0 轮无掩码（None），掩码由上一轮投票生成
    consensus_mask = None
    for round_idx in range(args.rounds):
        round_started = time.time()
        # SMELL 3 run_fed catv round start ADD — 注入上一轮掩码；回调由 ClientRunner 自行安装/移除
        base_config.consensus_mask = consensus_mask
        base_config.vote_callback = None
        deltas, counts, losses, client_stats = [], [], [], {}
        round_vote_sum = {}
        client_votes = {}
        for name in client_names:
            client_dir = data_root / args.tag / "clients" / name
            input_ids_np = np.load(client_dir / "train_input_ids.npy")
            num_samples = len(input_ids_np)
            if args.max_train_samples > 0:
                num_samples = min(num_samples, args.max_train_samples)

            load_trainable_state_dict(model, global_state)
            runner = ClientRunner(model, local_steps=args.local_steps, lr=args.lr,
                                  zo_eps=args.zo_eps, zo_directions=args.zo_directions,
                                  collect_votes=catv_on)
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
            # SMELL 3 run_fed catv votes ADD — 收集本轮客户端投票（可选 per-layer sum 归一化）
            if catv_on:
                votes = result["votes"]
                if args.catv_normalize == "sum":
                    votes = {
                        layer: (tensor / tensor.sum() if float(tensor.sum()) != 0.0 else tensor)
                        for layer, tensor in votes.items()
                    }
                round_vote_sum = ServerAggregator.accumulate_votes(round_vote_sum, votes)
                client_votes[name] = votes

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
        # SMELL 3 run_fed catv mask BEGIN — 本轮投票 → 下一轮共识锚掩码 + 掩码文件 + IR 指标
        if catv_on:
            consensus_mask = compute_consensus_mask(round_vote_sum, n_blocks, catv_r, args.sparse)
            mask_path = out_dir / f"consensus_mask_round{round_idx}.pt"
            torch.save(consensus_mask, mask_path)
            first_layer_mask = next(iter(consensus_mask.values()))
            anchor_in = int(torch.isposinf(first_layer_mask).sum())
            anchor_out = int(torch.isneginf(first_layer_mask).sum())
            local_keep = max(1, min(int(n_blocks * args.sparse), n_blocks))
            ir_values = []
            for votes in client_votes.values():
                for layer, tensor in votes.items():
                    local_order = torch.argsort(tensor, descending=True, stable=True)[:local_keep]
                    local_set = {int(index) for index in local_order.tolist()}
                    central_set = {
                        int(index)
                        for index in torch.isposinf(consensus_mask[layer]).nonzero().reshape(-1).tolist()
                    }
                    ir_values.append(mask_intersection_rate(local_set, central_set))
            ir_mean = (sum(ir_values) / len(ir_values)) if ir_values else None
            ir_min = min(ir_values) if ir_values else None
            record.update({
                "catv_r": catv_r,
                "anchor_in": anchor_in,
                "anchor_out": anchor_out,
                "vote_bytes": len(round_vote_sum) * n_blocks * 4,
                "ir_mean": ir_mean,
                "ir_min": ir_min,
                "consensus_mask_file": mask_path.name,
            })
            print(f"[fed] catv round {round_idx} mask={mask_path.name} anchor_in={anchor_in} "
                  f"anchor_out={anchor_out} vote_bytes={record['vote_bytes']} "
                  f"ir_mean={ir_mean if ir_mean is not None else float('nan'):.4f} "
                  f"ir_min={ir_min if ir_min is not None else float('nan'):.4f}")
        # SMELL 3 run_fed catv mask END
        append_metrics(metrics_path, record)
        print(f"[fed] round {round_idx}/{args.rounds - 1} loss={train_loss_mean} "
              f"delta_norm={record['delta_norm_mean']:.4e} "
              f"cos={record['cos_mean_sampled']} time={record['round_seconds']:.1f}s")

        if args.eval_every > 0 and (round_idx + 1) % args.eval_every == 0:
            run_global_eval(args, model, out_dir, metrics_path, round_idx)

    print(f"[fed] done; metrics={metrics_path}")


if __name__ == "__main__":
    main()
