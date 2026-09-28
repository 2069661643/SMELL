# SMELL 3 paired_eval NEW — 逐样本配对检验（ppl.py --per-sample-out 产物；判断两轮 adapter 差异是否显著）
#
# 用法：
#   python scripts/paired_eval.py --run-dir logs/fed/zoo_k4_sparse_causal_a03/a03 --rounds 0,29
#   python scripts/paired_eval.py --a eval_round000_persample.json --b eval_round029_persample.json \
#       --field full_nll_mean --out temp/paired_r0_r29.json
# 语义：diff = new - old（NLL），负值 = 新轮更好；报告配对 t 检验与 PPL 比值。

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_rows(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: no 'rows' list (need ppl.py --per-sample-out output)")
    out = {}
    for row in rows:
        out[int(row["index"])] = row
    return out


def paired_stats(a_rows, b_rows, field):
    common = sorted(set(a_rows) & set(b_rows))
    diffs = []
    for index in common:
        old = a_rows[index].get(field)
        new = b_rows[index].get(field)
        if old is None or new is None:
            continue
        diffs.append(float(new) - float(old))
    n = len(diffs)
    if n < 2:
        raise SystemExit(f"paired {field}: only {n} usable samples")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    if se > 0:
        t = mean / se
        try:
            from scipy import stats
            p_two = float(2.0 * stats.t.sf(abs(t), df=n - 1))
        except ImportError:
            p_two = math.erfc(abs(t) / math.sqrt(2.0))
    else:
        # SMELL 3 paired_eval zero_variance FIXED — 所有差值相同（含全 0）时无方差：均值=0 视为无差异 p=1
        t = 0.0 if mean == 0 else math.copysign(float("inf"), mean)
        p_two = 1.0 if mean == 0 else 0.0
    p_one = p_two / 2.0 if mean < 0 else 1.0 - p_two / 2.0
    return {
        "field": field,
        "n": n,
        "mean_delta_nll": mean,
        "sd": sd,
        "se": se,
        "t": t,
        "p_two_sided": p_two,
        "p_one_sided_improve": p_one,
        "ci95_delta_nll": [mean - 1.96 * se, mean + 1.96 * se],
        "ppl_ratio_new_over_old": math.exp(mean),
    }


def _selftest():
    a = {"rows": [{"index": i, "full_nll_mean": 5.0} for i in range(60)]}
    b = {"rows": [{"index": i, "full_nll_mean": 5.0 - 0.02 - (i % 3) * 0.001} for i in range(60)]}
    stats = paired_stats({r["index"]: r for r in a["rows"]}, {r["index"]: r for r in b["rows"]},
                         "full_nll_mean")
    assert stats["n"] == 60, stats
    assert stats["mean_delta_nll"] < 0, stats
    assert stats["p_two_sided"] < 0.01, stats
    same = paired_stats({r["index"]: r for r in a["rows"]}, {r["index"]: r for r in a["rows"]},
                        "full_nll_mean")
    assert abs(same["mean_delta_nll"]) < 1e-12 and same["p_two_sided"] == 1.0, same
    print(f"paired_eval selftest PASS improved_p={stats['p_two_sided']:.2e} "
          f"ratio={stats['ppl_ratio_new_over_old']:.4f}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Paired per-sample eval (NLL) between two rounds")
    parser.add_argument("--a", default=None, help="old per-sample json")
    parser.add_argument("--b", default=None, help="new per-sample json")
    parser.add_argument("--run-dir", default=None, help="run dir; builds eval_roundNNN_persample.json")
    parser.add_argument("--rounds", default="0,29", help="old,new round indices with --run-dir")
    parser.add_argument("--field", choices=("full_nll_mean", "answer_nll_mean"),
                        default="full_nll_mean")
    parser.add_argument("--out", default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    if args.run_dir:
        rounds = [int(x) for x in args.rounds.split(",")]
        if len(rounds) != 2:
            raise SystemExit(f"--rounds expects old,new, got {args.rounds!r}")
        run_dir = Path(args.run_dir)
        path_a = run_dir / f"eval_round{rounds[0]:03d}_persample.json"
        path_b = run_dir / f"eval_round{rounds[1]:03d}_persample.json"
    else:
        if not args.a or not args.b:
            raise SystemExit("provide --a/--b or --run-dir/--rounds")
        path_a, path_b = Path(args.a), Path(args.b)
    for path in (path_a, path_b):
        if not path.exists():
            raise SystemExit(f"missing per-sample file: {path}")
    stats = paired_stats(load_rows(path_a), load_rows(path_b), args.field)
    stats.update({"a": str(path_a), "b": str(path_b),
                  "verdict": "IMPROVED" if stats["mean_delta_nll"] < 0 and stats["p_two_sided"] < 0.05
                  else ("REGRESSED" if stats["mean_delta_nll"] > 0 and stats["p_two_sided"] < 0.05
                        else "NOT-SIGNIFICANT")})
    print(json.dumps(stats, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[paired_eval] wrote {out_path}")


if __name__ == "__main__":
    main()
