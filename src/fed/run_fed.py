# SMELL 3 run_fed NEW — 串行 FedAvg（ZOO+LoRA）实验入口：CLI、逐轮指标 JSONL、可选 G-PPL 子进程评测

import argparse
import json
import os
import random
# SMELL 3 run_fed per_module_delta ADD — 解析参数名中的层号/投影名
import re
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
    # SMELL 3 run_fed rank_rotation ADD — LoRA 秩调度：all=旧行为；rotate=每轮激活 rank_k 个秩轮换；lock=固定前 rank_k 秩
    parser.add_argument("--rank-mode", choices=("all", "rotate", "lock"), default="all")
    parser.add_argument("--rank-k", type=int, default=1, help="每轮激活秩数（>0；rotate/lock 生效）")
    # SMELL 3 run_fed subspace_zo ADD — 块/子空间 ZO：按层或投影选择激活子空间（与 rank-mode 互斥）
    parser.add_argument("--zo-subspace", choices=("all", "layers", "modules"), default="all",
                        help="all=全参数（旧行为）；layers=按层选；modules=按层+投影选")
    parser.add_argument("--zo-layers", default=None, help="区间/单值列表，如 '0-2,5'（--zo-subspace layers 时必填）")
    parser.add_argument("--zo-modules", default=None, help="<layer>.<proj> 列表，如 '0.q_proj,1.k_proj'（--zo-subspace modules 时必填）")
    # SMELL 3 run_fed zo_layer_rotate ADD — 每轮轮转激活层子空间（仅 --zo-subspace layers，与 rank-mode 互斥）
    parser.add_argument("--zo-layer-rotate", action="store_true",
                        help="rotate active layer set every round (requires --zo-subspace layers)")
    parser.add_argument("--zo-layer-group", type=int, default=1,
                        help="layers activated per round when --zo-layer-rotate (default 1)")
    # SMELL 3 run_fed truncate ADD — smoke 用：每条训练序列只取前 N token（0 = 关闭，全长）
    parser.add_argument("--truncate", type=int, default=0, help="smoke only: train on first N tokens; 0 = full seq")
    # SMELL 3 run_fed init-path ADD — warmup 产物初始化：位置表 + LoRA 适配器（云端 Step 3/4 复用）
    parser.add_argument("--pos-checkpoint", default=None, help="pos_embed.pt loaded into embed_positions BEFORE LoRA")
    parser.add_argument("--adapter-init", default=None, help="PEFT adapter dir used to initialize LoRA (kept trainable)")
    # SMELL 3 run_fed lora_rank ADD — LoRA 秩与 alpha（无 adapter-init 时生效；r=1 建议 alpha≈2 以保持 alpha/r 缩放）
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    # SMELL 3 run_fed predictor ADD — predictor.pth + pruned_config.pth 接线（None = 不加载，保持随机初始化）
    parser.add_argument("--predictor", default=None,
                        help="predictor.pth (Jenga PrunableAttnPredictorInfer weights); requires --pruned-config")
    parser.add_argument("--pruned-config", default=None,
                        help="pruned_config.pth (per-layer q/k outdims) used to rebuild predictor shapes")
    # SMELL 3 run_fed act_pack ADD — BP 默认旁路 Jenga 半丢弃 hooks（避免梯度静默污染）；ZOO 不受影响
    parser.add_argument("--act-pack", choices=("off", "on"), default="off")
    parser.add_argument("--eval-every", type=int, default=0, help="0 = off; else run ppl.py every N rounds")
    parser.add_argument("--out-root", default="logs/fed")
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


# SMELL 3 run_fed subspace_zo BEGIN — 解析 --zo-layers / --zo-modules 选择串
def parse_layer_spec(spec):
    """把 '0-2,5' 解析为升序去重的层号列表（闭区间）。空结果报错。"""
    layers = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(chunk))
    if not layers:
        raise SystemExit(f"--zo-layers parsed to empty list from {spec!r}")
    return sorted(set(layers))


def parse_module_spec(spec):
    """把 '0.q_proj,1.k_proj' 解析为 [(layer, proj), ...]。格式错误报错。"""
    modules = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        layer, _, proj = chunk.partition(".")
        if not proj:
            raise SystemExit(f"--zo-modules entry {chunk!r} must be <layer>.<proj>")
        modules.append((int(layer), proj))
    if not modules:
        raise SystemExit(f"--zo-modules parsed to empty list from {spec!r}")
    return modules
# SMELL 3 run_fed subspace_zo END


# SMELL 3 run_fed predictor BEGIN — 把 predictor.pth 的逐层 q/k 线性权重拷入模型 self_attn.predictor
def load_predictor_weights(model, path):
    """按 key 把 predictor.pth 载入模型（形状须与 --pruned-config 重建的 predictor 一致）。

    上游 Jenga 约定（llama_jenga.py / hello_world.py）：predictor.pth 键名与
    `model.decoder.layers.N.self_attn.predictor.*` 一致，逐键 copy_ 进 state_dict。
    形状不匹配（例如未按 pruned_config 重建）直接报错，避免静默加载随机权重。
    """
    import torch
    state = torch.load(path, map_location="cpu")
    assert isinstance(state, dict), f"predictor payload is {type(state).__name__}, expected dict"
    model_state = model.state_dict()
    loaded, skipped, mismatched = 0, 0, []
    with torch.no_grad():
        for key, value in state.items():
            if key not in model_state:
                skipped += 1
                continue
            target = model_state[key]
            if tuple(target.shape) != tuple(value.shape):
                mismatched.append((key, tuple(value.shape), tuple(target.shape)))
                continue
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded += 1
    if mismatched:
        raise RuntimeError(
            f"predictor shape mismatch for {len(mismatched)} tensors, e.g. {mismatched[:3]}; "
            f"--pruned-config must match --predictor")
    if loaded == 0:
        raise RuntimeError(f"predictor loaded 0 tensors from {path}; check key prefixes")
    print(f"[fed] predictor loaded tensors={loaded} skipped={skipped} path={path}")
    return loaded, skipped
# SMELL 3 run_fed predictor END


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


# SMELL 3 run_fed per_module_delta BEGIN — 逐层/逐投影拆分 client δ 的 L2 范数（诊断 LoRA 更新分布）
_LAYER_PATTERN = re.compile(r"layers\.(\d+)\.")
_PROJ_PATTERN = re.compile(r"self_attn\.(q_proj|k_proj|v_proj|out_proj)\.")


def _layer_sort_key(layer):
    return (0, int(layer)) if layer.isdigit() else (1, 0)


def delta_norm_by_group(delta):
    """把 `{param_name: tensor}` 的 δ 拆成（逐层, 逐投影）L2 范数。

    层号取自 `layers.{i}.`，无匹配（如 embed_positions）归入 `other`；
    投影名取自 `self_attn.{proj}.`，键形如 `0.q_proj`，无投影匹配的只进逐层结果。
    """
    import torch
    sq_by_layer = {}
    sq_by_module = {}
    for name, tensor in delta.items():
        sq = float(torch.sum(tensor.detach().float() ** 2))
        layer_match = _LAYER_PATTERN.search(name)
        layer = layer_match.group(1) if layer_match else "other"
        sq_by_layer[layer] = sq_by_layer.get(layer, 0.0) + sq
        proj_match = _PROJ_PATTERN.search(name)
        if proj_match is not None and layer_match is not None:
            key = f"{layer}.{proj_match.group(1)}"
            sq_by_module[key] = sq_by_module.get(key, 0.0) + sq
    return (
        {key: sq_by_layer[key] ** 0.5 for key in sorted(sq_by_layer, key=_layer_sort_key)},
        {key: sq_by_module[key] ** 0.5 for key in sq_by_module},
    )
# SMELL 3 run_fed per_module_delta END


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
    # SMELL 3 run_fed eval_lora_checkpoint ADD — 记录每次 eval 保存的 LoRA checkpoint 路径
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
    # SMELL 3 run_fed eval pos_checkpoint ADD — 评测须加载同一 pos_embed，否则位置表错位导致 G-PPL 失真
    if getattr(args, "pos_checkpoint", None):
        command += ["--pos-checkpoint", str(args.pos_checkpoint)]
    proc = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0 or not eval_out.exists():
        append_metrics(metrics_path, {
            "event": "eval", "round": round_idx, "status": "failed",
            "returncode": proc.returncode, "stderr_tail": proc.stderr[-500:],
            # SMELL 3 run_fed trainer ADD — eval 记录也标注本地训练器
            "trainer": args.trainer,
            # SMELL 3 run_fed eval_lora_checkpoint ADD — 记录每次 eval 保存的 LoRA checkpoint 路径
            "lora_checkpoint": str(adapter_dir),
        })
        print(f"[fed] eval round {round_idx} FAILED rc={proc.returncode}: {proc.stderr.strip()[-200:]}")
        print(f"[fed] eval round {round_idx} lora_checkpoint={adapter_dir}")
        return None
    payload = json.loads(eval_out.read_text(encoding="utf-8"))
    append_metrics(metrics_path, {"event": "eval", "round": round_idx, "status": "ok",
                                  "trainer": args.trainer,
                                  # SMELL 3 run_fed eval_lora_checkpoint ADD — 记录每次 eval 保存的 LoRA checkpoint 路径
                                  "lora_checkpoint": str(adapter_dir), **payload})
    print(f"[fed] eval round {round_idx} full_ppl_token={payload['full_ppl_token']} "
          f"answer_ppl_token={payload['answer_ppl_token']}")
    print(f"[fed] eval round {round_idx} lora_checkpoint={adapter_dir}")
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

    # SMELL 3 run_fed predictor ADD — predictor/pruned-config 必须成对提供
    if bool(args.predictor) != bool(args.pruned_config):
        raise SystemExit("--predictor and --pruned-config must be provided together")
    # SMELL 3 run_fed rank_rotation ADD — rank_k 必须为正
    if args.rank_k <= 0:
        raise SystemExit(f"--rank-k must be > 0, got {args.rank_k}")
    # SMELL 3 run_fed subspace_zo BEGIN — 子空间 ZO 参数校验、与秩轮换互斥、解析选择串
    if args.zo_subspace != "all" and args.rank_mode != "all":
        raise SystemExit(
            f"--zo-subspace {args.zo_subspace} and --rank-mode {args.rank_mode} are mutually exclusive")
    zo_layers, zo_modules = None, None
    if args.zo_subspace == "layers":
        # SMELL 3 run_fed zo_layer_rotate MODIFIED — rotate 模式按轮生成层集合，不需要固定 --zo-layers
        if args.zo_layer_rotate:
            if args.zo_layers is not None:
                raise SystemExit("--zo-layer-rotate is incompatible with --zo-layers")
            if args.zo_layer_group <= 0:
                raise SystemExit(f"--zo-layer-group must be > 0, got {args.zo_layer_group}")
        else:
            if args.zo_layers is None:
                raise SystemExit("--zo-subspace layers requires --zo-layers, e.g. '0-2,5'")
            zo_layers = parse_layer_spec(args.zo_layers)
    elif args.zo_subspace == "modules":
        if args.zo_modules is None:
            raise SystemExit("--zo-subspace modules requires --zo-modules, e.g. '0.q_proj,1.k_proj'")
        zo_modules = parse_module_spec(args.zo_modules)
    # SMELL 3 run_fed zo_layer_rotate ADD — --zo-layer-rotate 仅支持 layers 子空间（已由上一分支覆盖），显式兜底
    if args.zo_layer_rotate and args.zo_subspace != "layers":
        raise SystemExit("--zo-layer-rotate requires --zo-subspace layers")
    # SMELL 3 run_fed subspace_zo END

    import torch
    # SMELL 3 run_fed imports MODIFIED — 使用 src/ 可编辑的 OPT 拷贝（含 CATV 掩码注入）
    from src.models.modeling_opt_smell import OPTForCausalLM
    from jenga.utils.config_utils import get_opt_qk

    from src.fed.serial_fedavg import ClientRunner, ServerAggregator, append_metrics, resolve_model_config
    from src.models.position_embed import ensure_positions
    # SMELL 3 run_fed imports ADD — CATV 服务器端掩码计算与 IR 指标
    from src.models.token_selector import compute_consensus_mask, mask_intersection_rate
    from src.train.lora import (
        active_ranks_for,
        build_lora_model,
        count_trainable,
        get_trainable_state_dict,
        load_trainable_state_dict,
        num_lora_ranks,
    )
    # SMELL 3 run_fed rank_rotation ADD — 秩掩码索引构造
    # SMELL 3 run_fed subspace_zo MODIFIED — 同源导入按层/投影构造的 build_param_index
    from src.train.zoo import build_active_index, build_param_index

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
    # SMELL 3 run_fed predictor BEGIN — 载入 pruned_config 并在建模前挂到 config（OPTAttention 据此重建剪枝形状）
    predictor_loaded = 0
    if args.predictor:
        pruned_path = resolve(args.pruned_config)
        pruned_payload = torch.load(pruned_path, map_location="cpu")
        assert isinstance(pruned_payload, dict) and "layers" in pruned_payload, (
            f"unsupported pruned_config payload from {pruned_path}")
        config.predictor_layers = pruned_payload["layers"]
        print(f"[fed] pruned_config loaded path={pruned_path} layers={len(config.predictor_layers)}")
    model = OPTForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, config=config)
    if args.predictor:
        predictor_loaded, _ = load_predictor_weights(model, resolve(args.predictor))
    # SMELL 3 run_fed predictor END
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
        model = build_lora_model(model, r=args.lora_r, lora_alpha=args.lora_alpha, targets=LORA_TARGETS)
    # SMELL 3 run_fed adapter_init END
    model = model.cuda().train()
    global_state = get_trainable_state_dict(model)
    trainable_params = count_trainable(model)
    # SMELL 3 run_fed rank_rotation ADD — LoRA 秩数；rotate/lock 时校验 rank_k 不超秩数
    num_ranks = num_lora_ranks(model)
    if args.rank_mode != "all" and args.rank_k > num_ranks:
        raise SystemExit(f"--rank-k={args.rank_k} exceeds LoRA num_ranks={num_ranks}")
    aggregator = ServerAggregator(args.weight)
    bytes_per_param = 2

    # SMELL 3 run_fed subspace_zo BEGIN — 建好模型后构造固定子空间索引（跨轮不变，复用 active_index 管道）
    subspace_index = None
    # SMELL 3 run_fed zo_layer_rotate ADD — rotate 模式每轮重建索引，不预构造固定 subspace_index
    if args.zo_subspace != "all" and not args.zo_layer_rotate:
        try:
            subspace_index = build_param_index(model, layers=zo_layers, modules=zo_modules)
        except ValueError as error:
            raise SystemExit(f"--zo-subspace selection failed: {error}")
        assert subspace_index is not None and int(subspace_index.numel()) > 0, (
            f"empty subspace index for layers={zo_layers} modules={zo_modules}")
        print(f"[fed] zo_subspace={args.zo_subspace} active_index={int(subspace_index.numel())}")
    # SMELL 3 run_fed subspace_zo END
    # SMELL 3 run_fed zo_layer_rotate BEGIN — OPT 层数（轮转取模基准）与轮转集合计算
    num_layers = int(resolve_model_config(model).num_hidden_layers)

    def rotated_layers_for(round_idx):
        return sorted({(round_idx * args.zo_layer_group + offset) % num_layers
                       for offset in range(args.zo_layer_group)})
    # SMELL 3 run_fed zo_layer_rotate END

    run_config = {
        **vars(args),
        "resolved_data_root": str(data_root),
        "resolved_out_dir": str(out_dir),
        "model_dir": args.model_dir,
        "client_names": client_names,
        "partition_counts": partition_counts,
        "trainable_params": trainable_params,
        "lora_targets": LORA_TARGETS,
        # SMELL 3 run_fed rank_rotation config ADD — LoRA 秩数（rank_mode/rank_k 来自 vars(args)）
        "num_lora_ranks": num_ranks,
        # SMELL 3 run_fed subspace_zo ADD — 解析后的子空间选择与实际激活元素数（可复现性）
        "zo_subspace": args.zo_subspace,
        "zo_layers": zo_layers,
        "zo_modules": zo_modules,
        "active_index_size": int(subspace_index.numel()) if subspace_index is not None else None,
        # SMELL 3 run_fed zo_layer_rotate config ADD — 层轮转参数与层数（可复现性）
        "zo_layer_rotate": args.zo_layer_rotate,
        "zo_layer_group": args.zo_layer_group,
        "num_layers": num_layers,
        # SMELL 3 run_fed catv config ADD — 生效的锚比例与块数
        "resolved_catv_r": catv_r,
        "n_blocks": n_blocks,
        # SMELL 3 run_fed config ADD — 有效序列长度与初始化路径（可复现性）
        "train_full_seq_len": train_seq_len,
        "effective_seq_len": effective_seq_len,
        "resolved_pos_checkpoint": str(resolve(args.pos_checkpoint)) if args.pos_checkpoint else None,
        "resolved_adapter_init": str(resolve(args.adapter_init)) if args.adapter_init else None,
        # SMELL 3 run_fed predictor config ADD — predictor 加载路径与已载入张量数（可复现性）
        "resolved_predictor": str(resolve(args.predictor)) if args.predictor else None,
        "resolved_pruned_config": str(resolve(args.pruned_config)) if args.pruned_config else None,
        "predictor_loaded_tensors": predictor_loaded,
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
        # SMELL 3 run_fed rank_rotation BEGIN — 全客户端同调度：由 round_idx 决定本轮激活秩与平坦索引
        active_ranks = active_ranks_for(round_idx, args.rank_k, num_ranks, args.rank_mode)
        active_index = build_active_index(model, active_ranks) if active_ranks else None
        # SMELL 3 run_fed subspace_zo MODIFIED — 子空间模式覆盖固定索引；与 rank 模式互斥（rank_mode=all）
        if subspace_index is not None:
            active_index = subspace_index
        # SMELL 3 run_fed zo_layer_rotate BEGIN — 本轮轮转层集合及其 ZO 平坦索引（每轮重建）
        round_active_layers = None
        if args.zo_layer_rotate:
            round_active_layers = rotated_layers_for(round_idx)
            active_index = build_param_index(model, layers=round_active_layers)
            assert active_index is not None and int(active_index.numel()) > 0, (
                f"empty rotated layer index for layers={round_active_layers}")
        elif args.zo_subspace == "layers":
            round_active_layers = zo_layers
        # SMELL 3 run_fed zo_layer_rotate END
        comm_params = int(active_index.numel()) if active_index is not None else trainable_params
        # SMELL 3 run_fed rank_rotation END
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
                                  trainer=args.trainer, bp_clip=args.bp_clip,
                                  # SMELL 3 run_fed rank_rotation ADD — 本轮激活秩掩码
                                  active_index=active_index, active_ranks=active_ranks)
            result = runner.run(iter_batches(input_ids_np, num_samples, model.device,
                                             truncate=args.truncate))
            deltas.append(result["delta"])
            counts.append(num_samples)
            losses.append(result["train_loss"])
            # SMELL 3 run_fed per_module_delta ADD — 逐层/逐投影拆解 client δ 范数
            delta_by_layer, delta_by_module = delta_norm_by_group(result["delta"])
            client_stats[name] = {
                "samples": num_samples,
                "steps": result["steps"],
                "train_loss": result["train_loss"],
                "delta_norm": delta_norm(result["delta"]),
                "delta_norm_by_layer": delta_by_layer,
                "delta_norm_by_module": delta_by_module,
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
        # SMELL 3 run_fed per_module_delta ADD — 各层 δ 范数的跨 client 均值（缺失层按 0 计）
        layer_keys = sorted(
            {key for stats in client_stats.values() for key in stats["delta_norm_by_layer"]},
            key=_layer_sort_key,
        )
        delta_norm_by_layer_mean = {
            key: sum(stats["delta_norm_by_layer"].get(key, 0.0) for stats in client_stats.values())
            / len(client_stats)
            for key in layer_keys
        }
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
            # SMELL 3 run_fed per_module_delta ADD — 逐层 δ 范数跨 client 均值
            "delta_norm_by_layer_mean": delta_norm_by_layer_mean,
            # SMELL 3 run_fed zo_layer_rotate ADD — 本轮实际激活层（rotate=轮转集合；固定 layers=zo_layers）
            "zo_active_layers": round_active_layers,
            "cos_mean_sampled": sample_pairwise_cos(deltas),
            # SMELL 3 run_fed rank_rotation MODIFIED — 有掩码时通信量只计激活子空间
            "communication_bytes": len(client_names) * comm_params * bytes_per_param,
            "weight": args.weight,
            "lr": args.lr,
            "local_steps": args.local_steps,
            "zo_directions": args.zo_directions,
            # SMELL 3 run_fed trainer ADD — 每条 round 记录本地训练器（zoo|bp）
            "trainer": args.trainer,
            # SMELL 3 run_fed rank_rotation ADD — 本轮秩调度记录
            "rank_mode": args.rank_mode,
            "rank_k": args.rank_k,
            "active_ranks": active_ranks,
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
              f"cos={record['cos_mean_sampled']} time={record['round_seconds']:.1f}s "
              # SMELL 3 run_fed rank_rotation ADD — 打印本轮激活秩/通信量
              f"active_ranks={active_ranks} comm_bytes={record['communication_bytes']} "
              # SMELL 3 run_fed zo_layer_rotate ADD — 打印本轮激活层（rotate 核对用）
              f"active_layers={round_active_layers}")

        if args.eval_every > 0 and (round_idx + 1) % args.eval_every == 0:
            run_global_eval(args, model, out_dir, metrics_path, round_idx)

    # SMELL 3 run_fed peak_mem ADD — 报告 CUDA 峰值显存（8GB 卡 BP smoke 的 OOM 判据）
    if torch.cuda.is_available():
        print(f"[fed] cuda peak allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
              f"reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB")
    print(f"[fed] done; metrics={metrics_path}")


if __name__ == "__main__":
    main()
