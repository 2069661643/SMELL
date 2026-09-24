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
    # SMELL 3 run_fed trainer ADD — zoo=前向梯度本地更新（默认）；bp=标准反传本地训练（Step 3 机制去风险）
    parser.add_argument("--trainer", choices=("zoo", "bp"), default="zoo")
    parser.add_argument("--bp-clip", type=float, default=1.0, help="BP grad clip norm; <=0 disables clipping")
    # SMELL 3 run_fed truncate ADD — smoke 用：每条训练序列只取前 N token（0 = 关闭，全长）
    parser.add_argument("--truncate", type=int, default=0, help="smoke only: train on first N tokens; 0 = full seq")
    # SMELL 3 run_fed init-path ADD — warmup 产物初始化：位置表 + LoRA 适配器（云端 Step 3/4 复用）
    parser.add_argument("--pos-checkpoint", default=None, help="pos_embed.pt loaded into embed_positions BEFORE LoRA")
    parser.add_argument("--adapter-init", default=None, help="PEFT adapter dir used to initialize LoRA (kept trainable)")
    # SMELL 3 run_fed act_pack ADD — BP 默认旁路 Jenga 半丢弃 hooks（避免梯度静默污染）；ZOO 不受影响
    parser.add_argument("--act-pack", choices=("off", "on"), default="off")
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


def iter_batches(input_ids_np, num_samples, device, truncate=0):
    import torch
    for index in range(num_samples):
        row = input_ids_np[index]
        # SMELL 3 run_fed truncate ADD — smoke：只保留前 N 个 token（0 = 不截断）
        if truncate > 0:
            row = row[:truncate]
        row = torch.from_numpy(row.astype(np.int64))
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
            # SMELL 3 run_fed trainer ADD — eval 记录也标注本地训练器
            "trainer": args.trainer,
        })
        print(f"[fed] eval round {round_idx} FAILED rc={proc.returncode}: {proc.stderr.strip()[-200:]}")
        return None
    payload = json.loads(eval_out.read_text(encoding="utf-8"))
    append_metrics(metrics_path, {"event": "eval", "round": round_idx, "status": "ok",
                                  "trainer": args.trainer, **payload})
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

    from src.fed.serial_fedavg import ClientRunner, ServerAggregator, append_metrics, resolve_model_config
    from src.models.position_embed import ensure_positions
    # SMELL 3 run_fed imports ADD — CATV 服务器端掩码计算与 IR 指标
    from src.models.token_selector import compute_consensus_mask, mask_intersection_rate
    from src.train.lora import build_lora_model, count_trainable, get_trainable_state_dict, load_trainable_state_dict

    # SMELL 3 run_fed act_pack BEGIN — BP 旁路 Jenga modeling_opt 的 pack/unpack hooks（否则后一半 token 梯度被静默置零）
    if args.trainer == "bp" and args.act_pack == "off":
        import src.models.modeling_opt_smell as modeling
        modeling.pack_hook = lambda tensor: tensor
        modeling.unpack_hook = lambda tensor: tensor
    # SMELL 3 run_fed act_pack END

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
    # SMELL 3 run_fed truncate ADD — 有效序列长度 = min(truncate, 全长)（0 = 关闭）；位置表按有效长度扩展
    effective_seq_len = min(train_seq_len, args.truncate) if args.truncate > 0 else train_seq_len
    # SMELL 3 run_fed n_blocks ADD — CATV: 每层块数 N = seq_len / pool_size（按有效序列长度）
    n_blocks = effective_seq_len // BLOCK_SIZE

    config = get_opt_qk(model_name=args.model_dir, flash_attention=True, pool_size=64,
                        thresh=args.sparse)
    model = OPTForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, config=config)
    # SMELL 3 run_fed base_config ADD — 保存 LoRA 包装前的 OPTConfig 引用（vote_callback/consensus_mask 挂载点）
    base_config = model.config
    model = ensure_positions(model, effective_seq_len)
    # SMELL 3 run_fed pos_checkpoint BEGIN — warmup 位置表注入（LoRA 包装前）；checkpoint 更大时按行数扩展后精确形状校验
    if args.pos_checkpoint:
        pos_path = resolve(args.pos_checkpoint)
        pos_payload = torch.load(pos_path, map_location="cpu")
        if isinstance(pos_payload, dict):
            pos_payload = pos_payload.get("weight", pos_payload)
        assert torch.is_tensor(pos_payload), f"unsupported pos checkpoint payload: {type(pos_payload)}"
        target_weight = model.model.decoder.embed_positions.weight
        if tuple(pos_payload.shape) != tuple(target_weight.shape):
            if (pos_payload.dim() == 2 and pos_payload.shape[1] == target_weight.shape[1]
                    and pos_payload.shape[0] > target_weight.shape[0]):
                pos_offset = int(model.model.decoder.embed_positions.offset)
                model = ensure_positions(model, int(pos_payload.shape[0]) - pos_offset)
                target_weight = model.model.decoder.embed_positions.weight
        assert tuple(pos_payload.shape) == tuple(target_weight.shape), (
            f"pos checkpoint shape {tuple(pos_payload.shape)} != embed_positions {tuple(target_weight.shape)}")
        with torch.no_grad():
            target_weight.copy_(pos_payload.to(device=target_weight.device, dtype=target_weight.dtype))
        print(f"[fed] pos_checkpoint loaded path={pos_path} shape={tuple(target_weight.shape)}")
    # SMELL 3 run_fed pos_checkpoint END
    # SMELL 3 run_fed adapter_init BEGIN — 载入 warmup 训练好的 LoRA 适配器（is_trainable=True，继续参与 FedAvg）
    if args.adapter_init:
        from peft import PeftModel
        adapter_path = resolve(args.adapter_init)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        # SMELL 3 run_fed adapter_init FIXED — from_pretrained 返回新 PeftModel；重解析共享 config，保证 consensus_mask/vote_callback 可挂载
        base_config = resolve_model_config(model)
        print(f"[fed] adapter_init loaded path={adapter_path} trainable_params={count_trainable(model)}")
    else:
        model = build_lora_model(model, r=8, targets=LORA_TARGETS)
    # SMELL 3 run_fed adapter_init END
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
        # SMELL 3 run_fed config ADD — 有效序列长度与初始化路径（可复现性）
        "train_full_seq_len": train_seq_len,
        "effective_seq_len": effective_seq_len,
        "resolved_pos_checkpoint": str(resolve(args.pos_checkpoint)) if args.pos_checkpoint else None,
        "resolved_adapter_init": str(resolve(args.adapter_init)) if args.adapter_init else None,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    print(f"[fed] tag={args.tag} clients={len(client_names)} trainable_params={trainable_params} "
          f"weight={args.weight} sparse={args.sparse} lr={args.lr} seed={args.seed} "
          f"trainer={args.trainer} truncate={args.truncate} seq_len={effective_seq_len}")
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
                                  collect_votes=catv_on,
                                  # SMELL 3 run_fed trainer ADD — 本地训练器与 BP 裁剪阈值
                                  trainer=args.trainer, bp_clip=args.bp_clip)
            result = runner.run(iter_batches(input_ids_np, num_samples, model.device,
                                             truncate=args.truncate))
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
            # SMELL 3 run_fed trainer ADD — 每条 round 记录本地训练器（zoo|bp）
            "trainer": args.trainer,
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

    # SMELL 3 run_fed peak_mem ADD — 报告 CUDA 峰值显存（8GB 卡 BP smoke 的 OOM 判据）
    if torch.cuda.is_available():
        print(f"[fed] cuda peak allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
              f"reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB")
    print(f"[fed] done; metrics={metrics_path}")


if __name__ == "__main__":
    main()
