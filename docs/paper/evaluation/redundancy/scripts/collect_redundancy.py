# SMELL 3 redundancy_stats NEW — 长上下文 token 冗余统计（引言 teaser 数据包，纯前向）
"""Collect CR(rho) / SR(theta) statistics for the SMELL intro redundancy figure.

Setup (fixed by reports/opt/requests/01-intro-figure-delivery-v0-zh.md):
  - OPT-350M + position adaptation (pos_only, same as all main runs);
  - block-importance predictor = step5 (a01a03 clients causal), layers 11-22, pool = 64;
  - dense-equivalent forward (sparse=1.0 -> keeps all blocks; no mask);
  - Discovery 16K global test, fixed sample subset, prefix truncation to L in {4K, 8K, 16K}.
Metrics:
  - CR(rho, L, layer) = positive score mass of the top-rho blocks / total positive mass;
  - SR(theta, L, layer) = fraction of blocks with score < max(score) * theta.
Outputs (all deterministic): data/*.csv, figures/fig0_redundancy_preview.{pdf,png},
summary.json (hashes + key numbers).

Usage:
  PY=~/miniconda3/envs/SMELL/bin/python
  $PY docs/paper/evaluation/redundancy/scripts/collect_redundancy.py --gpu 0
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = None  # set in main()


def find_repo(start, override=None):
    if override:
        return Path(override).resolve()
    path = Path(start).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "dataset_v3").is_dir() and \
                (candidate / "third_party" / "Jenga" / "checkpoints" / "opt-350m").is_dir():
            return candidate
    raise SystemExit("cannot locate the SMELL repo root (need dataset_v3 + third_party/Jenga/checkpoints/opt-350m); "
                     "pass --repo")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect token-redundancy statistics (intro figure)")
    parser.add_argument("--repo", default=None, help="SMELL repo root (auto-detected by default)")
    parser.add_argument("--out-dir", default=None, help="package root (default: parent of scripts/)")
    parser.add_argument("--tag", default="a03")
    parser.add_argument("--data-root", default="dataset_v3/discovery_16k")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lengths", default="4096,8192,16384")
    parser.add_argument("--rho", default="0.1,0.2,0.4,0.6,0.8")
    parser.add_argument("--theta", default="0.1,0.3,0.5")
    parser.add_argument("--model-dir", default="third_party/Jenga/checkpoints/opt-350m")
    parser.add_argument("--pos-checkpoint", default="checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt")
    parser.add_argument("--predictor", default="checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth")
    parser.add_argument("--pruned-config", default="checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--no-verify-dense", action="store_true", help="skip the dense-equivalence check")
    return parser.parse_args()


def cr_for_rho(scores, rho):
    """Coverage ratio: positive mass of top-k blocks (k = floor(rho * n_blocks), model's rule)."""
    n_blocks = int(scores.numel())
    k = max(1, min(int(n_blocks * rho), n_blocks))
    positive = scores.clamp(min=0.0)
    total = float(positive.sum())
    if total <= 0.0:
        return float("nan")
    top = scores.topk(k, largest=True).indices
    return min(1.0, float(positive[top].sum()) / total)


def sr_for_theta(scores, theta):
    """Sparsity ratio: fraction of blocks below max(score) * theta."""
    threshold = float(scores.max()) * theta
    return float((scores < threshold).float().mean())


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    args = parse_args()
    global REPO
    REPO = find_repo(Path(__file__).resolve().parent, args.repo)
    sys.path.insert(0, str(REPO))
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from argparse import Namespace
    import numpy as np
    import torch
    from src.models.modeling_opt_smell import OptSdpaPruneAttention
    from src.eval.ppl import build_model, get_decoder

    lengths = [int(x) for x in args.lengths.split(",")]
    rhos = [float(x) for x in args.rho.split(",")]
    thetas = [float(x) for x in args.theta.split(",")]
    package = Path(args.out_dir).resolve() if args.out_dir else Path(__file__).resolve().parents[1]
    data_dir = package / "data"
    fig_dir = package / "figures"

    ids_path = resolve(args.data_root) / args.tag / "global_test_input_ids.npy"
    ids = np.load(ids_path, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(int(ids.shape[0]), size=min(args.n_samples, int(ids.shape[0])), replace=False))
    print(f"[redundancy] repo={REPO}")
    print(f"[redundancy] samples: n={len(chosen)} seed={args.seed} "
          f"first={chosen[:5].tolist()} last={chosen[-3:].tolist()}")

    torch.manual_seed(0)
    ns = Namespace(model_dir=str(resolve(args.model_dir)), attn="sdpa_prune", dtype="fp32", sparse=1.0,
                   predictor=str(resolve(args.predictor)), pruned_config=str(resolve(args.pruned_config)),
                   catv_mask=None, pos_mode="jenga_dup_scaled", pos_checkpoint=str(resolve(args.pos_checkpoint)),
                   adapter=None)
    model = build_model(ns, max(lengths)).eval()
    num_layers = int(model.config.num_hidden_layers)
    layers = list(range(num_layers // 2 - 1, num_layers - 1))
    pool = int(model.config.pool_size)
    print(f"[redundancy] model loaded; num_layers={num_layers} scored_layers={layers} pool={pool}")

    # --- dense-equivalence check: sparse=1.0 subset path must not perturb the forward ---
    verify = {}
    if not args.no_verify_dense:
        dense_ns = Namespace(**vars(ns))
        dense_ns.attn = "sdpa"
        dense_model = build_model(dense_ns, lengths[0]).eval()

        def nll(model_obj, sample_ids):
            with torch.no_grad():
                return float(model_obj(input_ids=sample_ids, labels=sample_ids).loss)

        probe_ids = torch.from_numpy(np.asarray(ids[chosen[0], :lengths[0]], dtype=np.int64)).unsqueeze(0).cuda()
        nll_prune, nll_dense = nll(model, probe_ids), nll(dense_model, probe_ids)
        verify = {"sample": int(chosen[0]), "length": lengths[0],
                  "nll_sparse1.0": nll_prune, "nll_dense": nll_dense,
                  "abs_diff": abs(nll_prune - nll_dense)}
        assert verify["abs_diff"] < 1e-3, f"sparse=1.0 forward differs from dense: {verify}"
        print(f"[redundancy] dense-equivalence PASS: |dNLL|={verify['abs_diff']:.2e}")
        del dense_model
        torch.cuda.empty_cache()

    captured = {}
    original = OptSdpaPruneAttention._prune_block_scores

    def patched(self, hidden_states, bsz, n_blocks, pool_size):
        scores, scored_by = original(self, hidden_states, bsz, n_blocks, pool_size)
        captured[int(self.layer_idx)] = scores.detach().float().cpu().clone()
        return scores, scored_by

    OptSdpaPruneAttention._prune_block_scores = patched

    raw_rows = []
    per_layer_values = {}
    decoder = get_decoder(model)
    try:
        for length in lengths:
            expected_blocks = length // pool
            for index in chosen:
                captured.clear()
                input_ids = torch.from_numpy(np.asarray(ids[int(index), :length], dtype=np.int64)).unsqueeze(0).cuda()
                with torch.no_grad():
                    decoder(input_ids=input_ids, use_cache=False, return_dict=True)
                assert set(captured) >= set(layers), (
                    f"missing layers for sample {index} L={length}: got {sorted(captured)}")
                for layer in layers:
                    row = captured[layer][0]
                    assert row.numel() == expected_blocks, (
                        f"sample {index} L={length} layer {layer}: n_blocks={row.numel()} expected {expected_blocks}")
                    assert torch.isfinite(row).all(), f"non-finite score: sample {index} L={length} layer {layer}"
                    for rho in rhos:
                        value = cr_for_rho(row, rho)
                        raw_rows.append(["coverage", length, int(index), layer, rho, f"{value:.6f}"])
                        per_layer_values.setdefault(("coverage", length, layer, rho), []).append(value)
                    for theta in thetas:
                        value = sr_for_theta(row, theta)
                        raw_rows.append(["sparsity", length, int(index), layer, theta, f"{value:.6f}"])
                        per_layer_values.setdefault(("sparsity", length, layer, theta), []).append(value)
                print(f"[redundancy] L={length} sample {int(index)} done "
                      f"(max_score={float(captured[layers[0]][0].max()):.4f})")
    finally:
        OptSdpaPruneAttention._prune_block_scores = original

    write_csv(data_dir / "raw_metrics.csv",
              ["metric", "length", "sample_index", "layer", "param", "value"], raw_rows)

    coverage_rows = []
    sparsity_rows = []
    import statistics
    for (metric, length, layer, param), values in sorted(per_layer_values.items()):
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        assert -1e-9 <= mean <= 1.0 + 1e-9, (
            f"mean out of range: {metric} L={length} layer={layer} {param} -> {mean}")
        row = [length, layer, param, f"{mean:.6f}", f"{std:.6f}", len(values)]
        (coverage_rows if metric == "coverage" else sparsity_rows).append(row)

    write_csv(data_dir / "coverage_curve.csv",
              ["length", "layer", "rho", "mean", "std", "n"], coverage_rows)
    write_csv(data_dir / "sparsity_ratio.csv",
              ["length", "layer", "theta", "mean", "std", "n"], sparsity_rows)
    write_csv(data_dir / "samples.csv", ["sample_index"], [[int(i)] for i in chosen])

    # cross-layer summary (mean +/- std over the 12 scored layers)
    def cross_layer(metric, length, param):
        rows = [r for r in (coverage_rows if metric == "coverage" else sparsity_rows)
                if r[0] == length and abs(float(r[2]) - param) < 1e-9]
        means = [float(r[3]) for r in rows]
        assert means, f"no rows for {metric} L={length} param={param}"
        return statistics.mean(means), statistics.pstdev(means)

    summary = {
        "tag": args.tag,
        "n_samples": int(len(chosen)),
        "seed": args.seed,
        "lengths": lengths,
        "rho": rhos,
        "theta": thetas,
        "layers": layers,
        "pool": pool,
        "model": {
            "model_dir": str(resolve(args.model_dir)),
            "pos_checkpoint": str(resolve(args.pos_checkpoint)),
            "predictor": str(resolve(args.predictor)),
            "sparse_forward": 1.0,
        },
        "cross_layer": {},
        "hashes": {
            "pos_embed.pt": sha256_file(resolve(args.pos_checkpoint)),
            "predictor.pth": sha256_file(resolve(args.predictor)),
            "pruned_config.pth": sha256_file(resolve(args.pruned_config)),
            "global_test_input_ids.npy": sha256_file(ids_path),
        },
        "dense_equivalence": verify,
    }
    for length in lengths:
        for rho in rhos:
            mean, std = cross_layer("coverage", length, rho)
            summary["cross_layer"][f"CR_rho{rho:g}_L{length}"] = {"mean": mean, "std": std}
        for theta in thetas:
            mean, std = cross_layer("sparsity", length, theta)
            summary["cross_layer"][f"SR_theta{theta:g}_L{length}"] = {"mean": mean, "std": std}
    (package / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[redundancy] cross-layer summary")
    for key, value in summary["cross_layer"].items():
        print(f"  {key:22s} {value['mean']:.4f} +/- {value['std']:.4f}")

    # ---- preview figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })
    colors = {4096: "#1f77b4", 8192: "#ff7f0e", 16384: "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for length in lengths:
        means, stds = [], []
        for rho in rhos:
            mean, std = cross_layer("coverage", length, rho)
            means.append(mean)
            stds.append(std)
        ax.errorbar(rhos, means, yerr=stds, marker="o", ms=3, lw=1.2, capsize=2,
                    color=colors[length], label=f"{length // 1024}K")
    ax.set_xlabel(r"retention budget $\rho$")
    ax.set_ylabel("coverage ratio CR")
    ax.grid(alpha=0.25, lw=0.5)
    legend = ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="black",
                       framealpha=1.0, fancybox=False)
    legend.get_frame().set_linewidth(0.7)

    ax = axes[1]
    for theta, style, width in [(0.1, ":", 0.9), (0.3, "-", 1.4), (0.5, "--", 0.9)]:
        means, stds = [], []
        for length in lengths:
            mean, std = cross_layer("sparsity", length, theta)
            means.append(mean)
            stds.append(std)
        if theta == 0.3:
            ax.errorbar(lengths, means, yerr=stds, marker="s", ms=3, lw=width, capsize=2,
                        color="#d62728", label=r"$\theta$=0.3")
        else:
            ax.plot(lengths, means, style, lw=width, color="#d62728", alpha=0.55, label=rf"$\theta$={theta}")
    ax.set_xlabel("prefix length")
    ax.set_ylabel("sparsity ratio SR")
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{length // 1024}K" for length in lengths])
    ax.grid(alpha=0.25, lw=0.5)
    legend = ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="black",
                       framealpha=1.0, fancybox=False)
    legend.get_frame().set_linewidth(0.7)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig0_redundancy_preview.png", dpi=300)
    fig.savefig(fig_dir / "fig0_redundancy_preview.pdf", dpi=300, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"[redundancy] wrote {package}")


if __name__ == "__main__":
    main()
