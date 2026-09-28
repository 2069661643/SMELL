# SMELL 3 diag_cos_grid COPY — 从 temp/diag_cos_grid.py 提升为 tracked 交付件（TD-2' k=4 稀疏 cos 门控）
# 块级真实 cos 打点：cos(ZOO client-delta, BP grad)，网格 (block, L, D, c)
# 单脚本串行：30 client × {layer0, k4, full} × L/D 网格；c 由前 c 个 client 子集平均免费得到
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")
DTYPES = {"bf16": "bfloat16", "fp32": "float32"}


def build_model(args, torch):
    from src.models.modeling_opt_smell import OPTForCausalLM
    from jenga.utils.config_utils import get_opt_qk
    from src.models.position_embed import ensure_positions
    from src.train.lora import build_lora_model

    config = get_opt_qk(model_name=args.model_dir, flash_attention=(args.attn == "flash"),
                        pool_size=64, thresh=args.sparse)
    # SMELL 3 diag_cos_grid attn ADD — sdpa 路径覆盖 get_opt_qk 的 eager/flash 默认
    if args.attn == "sdpa":
        config.attn_implementation = "sdpa"
    # SMELL 3 diag_cos_grid sdpa_prune ADD — Jenga 真语义 token 子集化：注册表键 + debug 统计
    elif args.attn == "sdpa_prune":
        config.attn_implementation = "sdpa_prune"
        config.sdpa_prune_debug = True
    # SMELL 3 diag_cos_grid predictor BEGIN — 建模前载入 pruned_config 的逐层 predictor 形状（复用 run_fed 口径）
    if args.predictor:
        pruned_payload = torch.load(args.pruned_config, map_location="cpu")
        assert isinstance(pruned_payload, dict) and "layers" in pruned_payload, (
            f"unsupported pruned_config payload from {args.pruned_config}")
        config.predictor_layers = pruned_payload["layers"]
        print(f"[grid] pruned_config loaded path={args.pruned_config} layers={len(config.predictor_layers)}", flush=True)
    torch_dtype = getattr(torch, DTYPES[args.dtype])
    model = OPTForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch_dtype, config=config)
    if args.predictor:
        from src.fed.run_fed import load_predictor_weights
        predictor_loaded, _ = load_predictor_weights(model, args.predictor)
        print(f"[grid] predictor loaded tensors={predictor_loaded} path={args.predictor}", flush=True)
    # SMELL 3 diag_cos_grid predictor END
    model = ensure_positions(model, args.seq)
    if args.pos_checkpoint:
        pos = torch.load(args.pos_checkpoint, map_location="cpu")
        if isinstance(pos, dict):
            pos = pos.get("weight", pos)
        tw = model.model.decoder.embed_positions.weight
        if (tuple(pos.shape) != tuple(tw.shape) and pos.dim() == 2
                and pos.shape[1] == tw.shape[1] and pos.shape[0] > tw.shape[0]):
            offset = int(model.model.decoder.embed_positions.offset)
            model = ensure_positions(model, int(pos.shape[0]) - offset)
            tw = model.model.decoder.embed_positions.weight
        assert tuple(pos.shape) == tuple(tw.shape)
        with torch.no_grad():
            tw.copy_(pos.to(device=tw.device, dtype=tw.dtype))
    import src.models.modeling_opt_smell as modeling
    modeling.pack_hook = lambda tensor: tensor
    modeling.unpack_hook = lambda tensor: tensor
    if args.adapter_init:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_init, is_trainable=True)
    else:
        model = build_lora_model(model, r=args.lora_r, lora_alpha=args.alpha, targets=TARGETS)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--model-dir", default=str(REPO / "third_party/Jenga/checkpoints/opt-350m"))
    ap.add_argument("--pos-checkpoint", default=str(REPO / "checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt"))
    # SMELL 3 diag_cos_grid predictor ADD — 训练后 predictor + 剪枝配置（sdpa_prune 正确稀疏语义所需）
    ap.add_argument("--predictor", default=str(REPO / "checkpoints/predictor/step4_a01_pos_only_causal/predictor.pth"))
    ap.add_argument("--pruned-config", default=str(REPO / "checkpoints/predictor/step4_a01_pos_only_causal/pruned_config.pth"))
    ap.add_argument("--data-root", default=str(REPO / "dataset_v3/discovery_16k/a01/clients"))
    ap.add_argument("--block-layer", type=int, default=0)
    # SMELL 3 diag_cos_grid block_layers ADD — k=4 等多层块：'20-23' -> 名为 'k4' 的子空间（d_eff=4*8192）
    ap.add_argument("--block-layers", default=None, help="e.g. '20-23' -> multi-layer block named 'k4'")
    ap.add_argument("--blocks", default="layer0,full")
    ap.add_argument("--adapter-init", default=None)
    ap.add_argument("--lora-r", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--seq", type=int, default=16384)
    # SMELL 3 diag_cos_grid attn ADD — 注意力后端 flash / eager / sdpa（默认 flash）
    # SMELL 3 diag_cos_grid sdpa_prune ADD — sdpa_prune = Jenga 真语义 token 子集化（fp32 SDPA）
    ap.add_argument("--attn", choices=["flash", "eager", "sdpa", "sdpa_prune"], default="flash")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--sparse", type=float, default=0.4)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=4e-6)
    ap.add_argument("--n-clients", type=int, default=30)
    ap.add_argument("--ls", default="1,2")
    ap.add_argument("--ds", default="8,16,32")
    ap.add_argument("--cs", default="1,5,10,20,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "temp/diag_cos_grid.json"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    from src.train.zoo import (_restore_flat, _trainable_named_parameters, apply_flat_delta,
                               build_param_index, flatten_trainable, zo_grad)

    torch.manual_seed(args.seed)
    model = build_model(args, torch).cuda().eval()
    named = _trainable_named_parameters(model)
    names, init_flat = flatten_trainable(model)
    blocks = {}
    for bname in args.blocks.split(","):
        if bname == "layer0":
            blocks[bname] = build_param_index(model, layers=[args.block_layer])
        elif bname == "k4":
            # SMELL 3 diag_cos_grid block_layers ADD — 多层块（rotate k 层子空间）
            from src.fed.run_fed import parse_layer_spec
            assert args.block_layers, "--blocks k4 requires --block-layers, e.g. '20-23'"
            blocks[bname] = build_param_index(model, layers=parse_layer_spec(args.block_layers))
        else:
            blocks[bname] = None
    Ls = [int(x) for x in args.ls.split(",")]
    Ds = [int(x) for x in args.ds.split(",")]
    Cs = [int(x) for x in args.cs.split(",")]
    print(f"[grid] dtype={args.dtype} attn={args.attn} r={args.lora_r} seq={args.seq} eps={args.eps} lr={args.lr} blocks={list(blocks)} "
          f"Ls={Ls} Ds={Ds} Cs={Cs} n_clients={args.n_clients}", flush=True)
    for bname, bidx in blocks.items():
        n = init_flat.numel() if bidx is None else int(bidx.numel())
        print(f"[grid] block {bname}: d_eff={n}", flush=True)

    bp = {b: [] for b in blocks}
    acc = {(b, L, D): [] for b in blocks for L in Ls for D in Ds}
    t0 = time.time()
    for i in range(args.n_clients):
        cdir = Path(args.data_root) / f"client_{i:02d}" / "train_input_ids.npy"
        ids = torch.tensor(np.asarray(np.load(cdir, mmap_mode="r")[0][:args.seq]).astype("int64"),
                           dtype=torch.long, device="cuda").unsqueeze(0)
        _restore_flat(model, init_flat)
        # BP reference
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        bp_full = torch.cat([p.grad.detach().reshape(-1).float() for _, p in named]).cpu()
        for b, bidx in blocks.items():
            bp[b].append(bp_full if bidx is None else bp_full[bidx])
        model.zero_grad(set_to_none=True)
        # ZOO
        for b, bidx in blocks.items():
            for L in Ls:
                for D in Ds:
                    _restore_flat(model, init_flat)
                    a = torch.zeros(init_flat.numel() if bidx is None else int(bidx.numel()), dtype=torch.float32)
                    for _l in range(L):
                        g = zo_grad(model, lambda: model(input_ids=ids, labels=ids, use_cache=False).loss,
                                    args.eps, D, index=bidx)
                        g_flat = torch.cat([g[n].reshape(-1).float() for n in names])
                        a += (g_flat if bidx is None else g_flat[bidx]).cpu()
                        apply_flat_delta(model, -args.lr * g_flat)
                    _restore_flat(model, init_flat)
                    acc[(b, L, D)].append(a)
        print(f"[grid] client {i:02d} done  elapsed={time.time()-t0:.1f}s", flush=True)

    def cos(x, y):
        return float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-12))

    rows = []
    for b in blocks:
        for L in Ls:
            for D in Ds:
                for c in Cs:
                    zs = torch.stack(acc[(b, L, D)][:c]).mean(0)
                    gs = torch.stack(bp[b][:c]).mean(0)
                    rows.append({"block": b, "L": L, "D": D, "c": c, "cos": cos(zs, gs),
                                 "M": c * L * D, "d_eff": int(zs.numel()),
                                 "cos_formula": (c * L * D / zs.numel()) ** 0.5})
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"{'block':7} {'L':>2} {'D':>3} {'c':>3} {'cos_meas':>9} {'cos_formula':>11}", flush=True)
    for r in rows:
        print(f"{r['block']:7} {r['L']:>2} {r['D']:>3} {r['c']:>3} {r['cos']:>9.4f} {r['cos_formula']:>11.4f}", flush=True)
    # SMELL 3 diag_cos_grid peak_mem ADD — 记录实际 seq 与峰值显存
    print(f"[grid] seq={args.seq} peak_alloc_gb={torch.cuda.max_memory_allocated()/1e9:.2f} "
          f"peak_reserved_gb={torch.cuda.max_memory_reserved()/1e9:.2f}", flush=True)
    print(f"[grid] DONE saved={args.out} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
