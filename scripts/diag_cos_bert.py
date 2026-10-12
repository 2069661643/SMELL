# SMELL 3 diag_cos_bert NEW — DistilBERT 16k 稀疏 ZOO 门控：cos(ZOO client-delta, BP grad) 网格
# 仿 scripts/diag_cos_grid.py（OPT）：单脚本串行 n_clients × {full6, kN 层块} × L/D 网格，
# c 由前 c 个 client 子集平均免费得到；loss = 冻结头分类 CE（与 FL 训练目标一致）。
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_MODEL_DIR = REPO / "third_party/Jenga/checkpoints/distilbert-base-uncased"
DEFAULT_DATA_ROOT = REPO / "dataset_v3/arxiv_16k"


def build_model(args, torch):
    from src.fed.run_fed import load_predictor_weights
    from src.models.modeling_distilbert_smell import load_distilbert_classifier
    from src.train.lora import build_lora_model
    from peft import TaskType

    model = load_distilbert_classifier(
        args.model_dir, num_labels=args.num_labels, dtype=args.dtype,
        attn=("sdpa_prune" if args.attn == "sdpa_prune" else "sdpa"),
        seq_len=args.seq, pos_checkpoint=args.pos_checkpoint, head_checkpoint=args.head_checkpoint,
        predictor=args.predictor, pruned_config=args.pruned_config, sparse=args.sparse)
    model.config.num_hidden_layers = int(model.config.n_layers)
    if args.adapter_init:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_init, is_trainable=True)
    else:
        model = build_lora_model(model, r=args.lora_r, lora_alpha=args.alpha,
                                 targets=["q_lin", "k_lin", "v_lin", "out_lin"],
                                 task_type=TaskType.SEQ_CLS)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--pos-checkpoint", default=None)
    ap.add_argument("--head-checkpoint", required=True)
    ap.add_argument("--predictor", default=None)
    ap.add_argument("--pruned-config", default=None)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT / "a01" / "clients"))
    ap.add_argument("--tag", default="a01")
    ap.add_argument("--blocks", default="full6",
                    help="comma list: full6 | layer<N> | k<M> (uses --block-layers)")
    ap.add_argument("--block-layers", default=None, help="e.g. '2-4' -> k3 subspace named 'k3'")
    ap.add_argument("--adapter-init", default=None)
    ap.add_argument("--lora-r", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--attn", choices=["sdpa", "sdpa_prune", "eager"], default="sdpa")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    ap.add_argument("--sparse", type=float, default=0.4)
    ap.add_argument("--num-labels", type=int, default=11)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=4e-6)
    ap.add_argument("--n-clients", type=int, default=30)
    ap.add_argument("--ls", default="2")
    ap.add_argument("--ds", default="8,22,64")
    ap.add_argument("--cs", default="1,5,10,20,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "temp/diag_cos_bert.json"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    from src.fed.run_fed import iter_classification_batches, parse_layer_spec
    from src.train.zoo import (_restore_flat, _trainable_named_parameters, apply_flat_delta,
                               build_param_index, flatten_trainable, zo_grad)

    torch.manual_seed(args.seed)
    model = build_model(args, torch).cuda().eval()
    named = _trainable_named_parameters(model)
    names, init_flat = flatten_trainable(model)
    blocks = {}
    for bname in args.blocks.split(","):
        bname = bname.strip()
        if bname in ("full", "full6", "full4"):
            blocks[bname] = None
        elif bname.startswith("layer"):
            blocks[bname] = build_param_index(model, layers=[int(bname[len("layer"):])])
        elif bname.startswith("k"):
            assert args.block_layers, "--blocks kN requires --block-layers, e.g. '2-4'"
            blocks[bname] = build_param_index(model, layers=parse_layer_spec(args.block_layers))
        else:
            raise SystemExit(f"unknown block spec {bname!r}")
    Ls = [int(x) for x in args.ls.split(",")]
    Ds = [int(x) for x in args.ds.split(",")]
    Cs = [int(x) for x in args.cs.split(",")]
    print(f"[cos-bert] dtype={args.dtype} attn={args.attn} r={args.lora_r} seq={args.seq} eps={args.eps} "
          f"lr={args.lr} blocks={list(blocks)} Ls={Ls} Ds={Ds} Cs={Cs} n_clients={args.n_clients}",
          flush=True)
    for bname, bidx in blocks.items():
        n = init_flat.numel() if bidx is None else int(bidx.numel())
        print(f"[cos-bert] block {bname}: d_eff={n}", flush=True)

    def classification_loss(batch):
        input_ids, attention_mask, label = batch
        return lambda: model(input_ids=input_ids, attention_mask=attention_mask, labels=label).loss

    bp = {b: [] for b in blocks}
    acc = {(b, L, D): [] for b in blocks for L in Ls for D in Ds}
    t0 = time.time()
    for i in range(args.n_clients):
        cdir = Path(args.data_root) / f"client_{i:02d}"
        batch = next(iter(iter_classification_batches(cdir, 1, "cuda", truncate=args.seq)))
        _restore_flat(model, init_flat)
        # BP reference（同 batch、同冻结头 CE）
        model.zero_grad(set_to_none=True)
        classification_loss(batch)().backward()
        bp_full = torch.cat([p.grad.detach().reshape(-1).float() for _, p in named]).cpu()
        for b, bidx in blocks.items():
            bp[b].append(bp_full if bidx is None else bp_full[bidx])
        model.zero_grad(set_to_none=True)
        # ZOO
        for b, bidx in blocks.items():
            for L in Ls:
                for D in Ds:
                    _restore_flat(model, init_flat)
                    a = torch.zeros(init_flat.numel() if bidx is None else int(bidx.numel()),
                                    dtype=torch.float32)
                    for _l in range(L):
                        g = zo_grad(model, classification_loss(batch), args.eps, D, index=bidx)
                        g_flat = torch.cat([g[n].reshape(-1).float() for n in names])
                        a += (g_flat if bidx is None else g_flat[bidx]).cpu()
                        apply_flat_delta(model, -args.lr * g_flat)
                    _restore_flat(model, init_flat)
                    acc[(b, L, D)].append(a)
        print(f"[cos-bert] client {i:02d} done  elapsed={time.time()-t0:.1f}s", flush=True)

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
        print(f"{r['block']:7} {r['L']:>2} {r['D']:>3} {r['c']:>3} {r['cos']:>9.4f} "
              f"{r['cos_formula']:>11.4f}", flush=True)
    print(f"[cos-bert] seq={args.seq} peak_alloc_gb={torch.cuda.max_memory_allocated()/1e9:.2f} "
          f"peak_reserved_gb={torch.cuda.max_memory_reserved()/1e9:.2f}", flush=True)
    print(f"[cos-bert] DONE saved={args.out} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
