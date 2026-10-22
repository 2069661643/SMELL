# SMELL 3 bert_and_other_report NEW — BERT + OPT fixed post-cycle CSV/README package (no figures)
"""Build reports/bert_and_other from raw logs (CPU-only, deterministic).

Scope: BERT experiments after the 261012 fixed-60-doc self-loop fix
(`--doc-order cycle`) plus the OPT fixed r30 four-arm chain (also cycle).
Pre-fix defect arms, old fixed-order runs, and non-final partial runs are
listed in excluded_runs.csv but never mixed into the main CSVs. No figures
are produced; the paper agent renders them later.

Usage:
  PY=~/miniconda3/envs/SMELL/bin/python
  $PY reports/bert_and_other/make_report.py
  $PY reports/bert_and_other/make_report.py --no-check
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from scipy import stats
except ImportError:
    stats = None

RUN_ROOT = REPO / "logs" / "fed"
OUT_ROOT = REPO / "reports" / "bert_and_other"
DATA_DIR = OUT_ROOT / "data"

R30_ARMS = [
    {"run_id": "r30_a01_catv_off", "tag": "a01", "catv": "off"},
    {"run_id": "r30_a01_catv_on", "tag": "a01", "catv": "on"},
    {"run_id": "r30_a03_catv_off", "tag": "a03", "catv": "off"},
    {"run_id": "r30_a03_catv_on", "tag": "a03", "catv": "on"},
]

OPT_ARMS = [
    {"run_id": "r30_a01_catv_off", "tag": "a01", "catv": "off"},
    {"run_id": "r30_a01_catv_on", "tag": "a01", "catv": "on"},
    {"run_id": "r30_a03_catv_off", "tag": "a03", "catv": "off"},
    {"run_id": "r30_a03_catv_on", "tag": "a03", "catv": "on"},
]
OPT_EVAL_ROUNDS = [4, 9, 14, 19, 24, 29]

BP_RUNS = [
    "dense_L8_lr3e4_r10",
    "dense_L8_lr1e3_r10",
    "sparse_L8_lr3e4_r10",
    "dense_L8_lr3e4_r5",
    "dense_L2_lr3e4_r20",
]

LD_RUNS = [
    ("L8D4", "main"),
    ("L8D8", "main"),
    ("fixed_L8D8", "control_fixed_order"),
    ("L16D4", "main"),
    ("L16D8", "main"),
    ("L8D16", "main"),
    ("L8D8_lr2x", "main"),
    ("dense_L8D8", "anchor_dense"),
    ("B1_L8D4_r5", "control_iso_corpus"),
    ("B2_L2D4_r20", "control_iso_corpus"),
]

TECH_RUNS = {
    "a03_catv_off_baseline": {
        "run_id": "r30_a03_catv_off", "group": "bert_da_fixed",
        "tag": "a03", "technique": "CATV off baseline",
        "catv": "off", "catv_normalize": "off", "var_control": "off", "vq_rate": 1.0,
    },
    "a03_catv_on_baseline": {
        "run_id": "r30_a03_catv_on", "group": "bert_da_fixed",
        "tag": "a03", "technique": "CATV on baseline",
        "catv": "on", "catv_normalize": "off", "var_control": "off", "vq_rate": 1.0,
    },
    "a03_vq_perresult": {
        "run_id": "bert_vq050_bf16_perresult_a03", "group": "",
        "tag": "a03", "technique": "VQ 0.5 bf16 per-result",
        "catv": "off", "catv_normalize": "off", "var_control": "off", "vq_rate": 0.5,
    },
    "a03_var_methoda": {
        "run_id": "bert_var_methoda_a03_r30", "group": "",
        "tag": "a03", "technique": "VAR Method A",
        "catv": "off", "catv_normalize": "off", "var_control": "on", "vq_rate": 1.0,
    },
    "a03_combo": {
        "run_id": "bert_var_catv_vq_a03_r30_rerun", "group": "",
        "tag": "a03", "technique": "VAR+CATV(client)+VQ 0.5",
        "catv": "on", "catv_normalize": "client", "var_control": "on", "vq_rate": 0.5,
    },
}

FILE_NOTES = [
    ("bert_r30_rounds.csv", "4 臂 × 30 轮逐轮 NLL / acc / 训练统计（主曲线数据）",
     "logs/fed/bert_da_fixed/<arm>/<tag>/metrics.jsonl + eval_roundNNN.json"),
    ("bert_r30_endpoints.csv", "4 臂 r0/r29 端点、相对下降、墙钟、adapter sha256",
     "同上 + adapter_round029/adapter_model.safetensors"),
    ("bert_r30_paired.csv", "4 臂 r0→r29 与 CATV on-vs-off 的逐样本配对检验",
     "logs/fed/bert_da_fixed/<arm>/<tag>/eval_round{000,029}_persample.json"),
    ("bert_r30_evaloff_rounds.csv", "CATV on 臂逐轮带掩码/去掩码评测（mask 效应曲线）",
     "logs/fed/bert_da_fixed/r30_<tag>_catv_on/<tag>/eval{off,}_roundNNN.json"),
    ("bert_r30_evaloff_endpoints.csv", "off / on-evaloff / on-evalon 三栏端点",
     "同上"),
    ("bert_r30_evaloff_paired.csv", "权重效应、推理掩码效应、总效应、r0 起点核验",
     "同上 + eval{off,}_round{000,029}_persample.json"),
    ("opt_r30_rounds.csv", "OPT fixed 4 臂 × r0/4/9/14/19/24/29 的 full/answer PPL 曲线",
     "logs/fed/opt_r30_fixed/<arm>/<tag>/metrics.jsonl + eval_roundNNN.json"),
    ("opt_r30_endpoints.csv", "OPT fixed 4 臂 r0/r4/r29 端点、vs fresh、墙钟、adapter sha256",
     "同上 + temp/eval_fixed_fresh_<tag>.json"),
    ("opt_r30_paired.csv", "OPT fixed 的 r0→r29、r4→r29、fresh→r29、CATV on-vs-off 配对",
     "logs/fed/opt_r30_fixed/.../eval{,_round}_persample.json + temp/eval_fixed_fresh_<tag>_persample.json"),
    ("opt_r30_evaloff_rounds.csv", "OPT CATV on 臂逐评测点带掩码/去掩码 full/answer PPL",
     "logs/fed/opt_r30_fixed/r30_<tag>_catv_on/<tag>/eval{off,}_roundNNN.json"),
    ("opt_r30_evaloff_endpoints.csv", "OPT off / on-evaloff / on-evalon 三栏端点（r4/r29）",
     "同上"),
    ("opt_r30_evaloff_paired.csv", "OPT 权重效应、推理掩码效应、部署净效应（full/answer）",
     "同上 + eval{off,}_round029_persample.json"),
    ("opt_baselines.csv", "OPT fresh pre-training 基线（a01/a03，500 样本）",
     "temp/eval_fixed_fresh_{a01,a03}.json"),
    ("bert_bp_cycle.csv", "BP 参考系（cycle，250 样本）：dense/sparse × lr × 轮数",
     "logs/fed/bert_bp_ref_cycle/<run>/a01/metrics.jsonl"),
    ("bert_ld_grid.csv", "L×D 网格 + fixed / iso-corpus / dense 对照（cycle，250 样本）",
     "logs/fed/bert_ld/<cell>/a01/metrics.jsonl"),
    ("bert_lr_scan.csv", "lr 扫描 r0–r2 训练损失 / delta（无 eval）",
     "logs/fed/bert_lr_scan/<lr>/a01/metrics.jsonl"),
    ("bert_vq_rounds.csv", "VQ per-result 0.5 bf16 逐轮 NLL/acc、cos、通信量",
     "logs/fed/bert_vq050_bf16_perresult_a03/a03/metrics.jsonl + eval_roundNNN.json"),
    ("bert_var_rounds.csv", "VAR Method A 逐轮 NLL/acc、D 分布、hit_low/high、阈值",
     "logs/fed/bert_var_methoda_a03_r30/a03/metrics.jsonl + eval_roundNNN.json"),
    ("bert_combo_rounds.csv", "VAR+CATV(client)+VQ 组合逐轮 NLL/acc、VAR/VQ/CATV 统计",
     "logs/fed/bert_var_catv_vq_a03_r30_rerun/a03/metrics.jsonl + eval_roundNNN.json"),
    ("bert_combo_evaloff_rounds.csv", "组合臂逐轮带掩码/去掩码评测（eval-off 分解）",
     "logs/fed/bert_var_catv_vq_a03_r30_rerun/a03/eval{off,}_roundNNN.json"),
    ("bert_technique_endpoints.csv", "a03 基线与 VQ / VAR / 组合 / 组合 eval-off 端点汇总（含训练/评测 mask 标记）",
     "上述 run 的 metrics.jsonl、eval JSON、adapter sha256"),
    ("bert_technique_paired.csv", "VQ/VAR/组合相对基线的逐样本配对检验",
     "上述 run 的 eval_round029_persample.json / evaloff_round029_persample.json"),
    ("bert_baselines.csv", "冻结 Step1 / dense 250 样本锚点（无 adapter）",
     "temp/eval_step1_16k_{final,best}.json、temp/eval_baseline_250.json"),
    ("provenance.csv", "每个被纳入 run 的原始路径、协议、轮数、状态",
     "logs/fed/、temp/ 原始产物清点"),
    ("excluded_runs.csv", "未纳入的缺陷臂 / 旧协议 / partial / 诊断 run 及原因",
     "checkpoints/ 与 logs/fed/ 清点"),
]

PAIRED_HEADER = [
    "comparison", "tag", "a_run", "b_run", "a_round", "b_round", "field", "n",
    "mean_delta_nll", "sd", "se", "t", "p_two_sided", "p_normal_approx",
    "exp_delta_nll", "ci95_low", "ci95_high", "verdict",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value == 0.0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e6:
            return f"{value:.6e}"
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([fmt(value) for value in row])
    return len(rows)


def mean_optional(values):
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def sum_optional(values):
    nums = [float(value) for value in values if value is not None]
    return sum(nums) if nums else None


def eval_map(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"missing metrics: {path}")
    out = {}
    for row in read_jsonl(path):
        if row.get("event") != "eval" or row.get("status") != "ok":
            continue
        if row.get("round") is None or row.get("nll") is None:
            continue
        out[int(row["round"])] = row
    if not out:
        raise SystemExit(f"no eval rows in {path}")
    return out


def round_map(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    out = {}
    for row in read_jsonl(path):
        if row.get("event") == "round" and row.get("round") is not None:
            out[int(row["round"])] = row
    return out


def eval_file(run_dir, round_idx, evaloff=False):
    stem = "evaloff" if evaloff else "eval"
    path = Path(run_dir) / f"{stem}_round{round_idx:03d}.json"
    if not path.exists():
        raise SystemExit(f"missing eval json: {path}")
    return read_json(path)


def persample_path(path, field):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing per-sample json: {path}")
    payload = read_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise SystemExit(f"unsupported per-sample payload: {path}")
    out = {}
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            key = int(row.get("index", index))
            value = row.get(field)
        else:
            key = index
            value = row
        if value is not None:
            out[key] = float(value)
    if not out:
        raise SystemExit(f"empty per-sample data: {path}")
    return out


def persample_field(run_dir, round_idx, field, evaloff=False):
    stem = "evaloff" if evaloff else "eval"
    path = Path(run_dir) / f"{stem}_round{round_idx:03d}_persample.json"
    return persample_path(path, field)


def persample_file(run_dir, round_idx, evaloff=False):
    return persample_field(run_dir, round_idx, "nll", evaloff=evaloff)


def opt_eval_map(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"missing metrics: {path}")
    out = {}
    for row in read_jsonl(path):
        if row.get("event") != "eval" or row.get("status") != "ok":
            continue
        if row.get("round") is None or row.get("full_ppl_token") is None:
            continue
        out[int(row["round"])] = row
    if not out:
        raise SystemExit(f"no OPT eval rows in {path}")
    return out


def opt_eval_file(run_dir, round_idx, evaloff=False):
    stem = "evaloff" if evaloff else "eval"
    path = Path(run_dir) / f"{stem}_round{round_idx:03d}.json"
    if not path.exists():
        raise SystemExit(f"missing OPT eval json: {path}")
    payload = read_json(path)
    if payload.get("full_ppl_token") is None:
        raise SystemExit(f"OPT eval json missing full_ppl_token: {path}")
    return payload


def block_span_hours(run_dir):
    run_dir = Path(run_dir)
    config = run_dir / "config.json"
    adapters = sorted(
        [path for path in run_dir.glob("adapter_round*") if path.is_dir()],
        key=lambda path: int(path.name.split("round")[-1]),
    )
    if not config.exists() or not adapters:
        return None
    return (adapters[-1].stat().st_mtime - config.stat().st_mtime) / 3600.0


def sha256_file(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_sha(run_dir, round_idx):
    return sha256_file(Path(run_dir) / f"adapter_round{round_idx:03d}" / "adapter_model.safetensors")


def endpoint_metrics(run_dir):
    evals = eval_map(run_dir)
    rounds = round_map(run_dir)
    keys = sorted(evals)
    first, last = keys[0], keys[-1]
    nll0 = float(evals[first]["nll"])
    nll_last = float(evals[last]["nll"])
    acc0 = float(evals[first].get("accuracy"))
    acc_last = float(evals[last].get("accuracy"))
    delta_nll = nll_last - nll0
    delta_nll_pct = (nll0 - nll_last) / nll0 * 100.0 if nll0 else None
    delta_acc_points = acc_last - acc0
    delta_acc_pct = delta_acc_points / acc0 * 100.0 if acc0 else None
    err_reduction_pct = (1.0 - (1.0 - acc_last) / (1.0 - acc0)) * 100.0 if acc0 < 1.0 else None
    round_seconds = [rounds[key].get("round_seconds") for key in sorted(rounds)]
    delta_norms = [rounds[key].get("delta_norm_mean") for key in sorted(rounds)]
    clipped = [rounds[key].get("clipped_clients") for key in sorted(rounds)]
    hours = block_span_hours(run_dir)
    return {
        "run_dir": Path(run_dir),
        "first_round": first,
        "last_round": last,
        "n_evals": len(keys),
        "eval_n": int(evals[first].get("n") or 0),
        "nll_r0": nll0,
        "nll_last": nll_last,
        "acc_r0": acc0,
        "acc_last": acc_last,
        "macro_f1_r0": evals[first].get("macro_f1"),
        "macro_f1_last": evals[last].get("macro_f1"),
        "delta_nll": delta_nll,
        "delta_nll_pct": delta_nll_pct,
        "delta_acc_points": delta_acc_points,
        "delta_acc_pct": delta_acc_pct,
        "err_reduction_pct": err_reduction_pct,
        "mean_round_seconds": mean_optional(round_seconds),
        "mean_delta_norm": mean_optional(delta_norms),
        "peak_delta_norm": max([float(x) for x in delta_norms if x is not None], default=None),
        "clipped_clients_total": sum_optional(clipped),
        "wall_hours": hours,
        "nll_pct_per_hour": (delta_nll_pct / hours) if delta_nll_pct is not None and hours else None,
        "adapter_sha256": adapter_sha(run_dir, last),
    }


def opt_endpoint_metrics(run_dir, tag):
    evals = opt_eval_map(run_dir)
    records = round_map(run_dir)
    r0 = opt_eval_file(run_dir, 0)
    r4 = evals[OPT_EVAL_ROUNDS[0]]
    r29 = evals[OPT_EVAL_ROUNDS[-1]]
    fresh = read_json(REPO / f"temp/eval_fixed_fresh_{tag}.json")
    full_r0 = float(r0["full_ppl_token"])
    full_r4 = float(r4["full_ppl_token"])
    full_r29 = float(r29["full_ppl_token"])
    answer_r0 = float(r0["answer_ppl_token"])
    answer_r4 = float(r4["answer_ppl_token"])
    answer_r29 = float(r29["answer_ppl_token"])
    round_seconds = [records[key].get("round_seconds") for key in sorted(records)]
    train_hours = sum(float(value) for value in round_seconds if value is not None) / 3600.0
    wall_hours = block_span_hours(run_dir)
    clipped = [records[key].get("clipped_clients") for key in sorted(records)]
    full_delta_r0 = (full_r0 - full_r29) / full_r0 * 100.0
    answer_delta_r0 = (answer_r0 - answer_r29) / answer_r0 * 100.0
    full_delta_r4 = (full_r4 - full_r29) / full_r4 * 100.0
    answer_delta_r4 = (answer_r4 - answer_r29) / answer_r4 * 100.0
    return {
        "run_dir": Path(run_dir),
        "eval_n": int(r0.get("num_samples") or 500),
        "rounds": len(records),
        "full_r0": full_r0,
        "full_r4": full_r4,
        "full_r29": full_r29,
        "answer_r0": answer_r0,
        "answer_r4": answer_r4,
        "answer_r29": answer_r29,
        "full_delta_pct_r0_r29": full_delta_r0,
        "answer_delta_pct_r0_r29": answer_delta_r0,
        "full_delta_pct_r4_r29": full_delta_r4,
        "answer_delta_pct_r4_r29": answer_delta_r4,
        "vs_fresh_full_pct": (float(fresh["full_ppl_token"]) - full_r29)
        / float(fresh["full_ppl_token"]) * 100.0,
        "vs_fresh_answer_pct": (float(fresh["answer_ppl_token"]) - answer_r29)
        / float(fresh["answer_ppl_token"]) * 100.0,
        "clipped_clients_total": sum_optional(clipped),
        "mean_round_seconds": mean_optional(round_seconds),
        "train_hours": train_hours,
        "wall_hours": wall_hours,
        "full_pct_per_hour_r4_r29": full_delta_r4 / train_hours if train_hours else None,
        "adapter_sha256": adapter_sha(run_dir, 29),
    }


def paired_stats(a_rows, b_rows):
    common = sorted(set(a_rows) & set(b_rows))
    diffs = [b_rows[index] - a_rows[index] for index in common]
    n = len(diffs)
    if n < 2:
        raise SystemExit(f"paired stats need >=2 samples, got {n}")
    mean = sum(diffs) / n
    var = sum((value - mean) ** 2 for value in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    if se > 0:
        t = mean / se
        if stats is not None:
            p_t = float(2.0 * stats.t.sf(abs(t), df=n - 1))
        else:
            p_t = math.erfc(abs(t) / math.sqrt(2.0))
        p_normal = math.erfc(abs(t) / math.sqrt(2.0))
    else:
        t = 0.0 if mean == 0 else math.copysign(float("inf"), mean)
        p_t = 1.0 if mean == 0 else 0.0
        p_normal = p_t
    verdict = "NOT-SIGNIFICANT"
    if p_t < 0.05 and mean < 0:
        verdict = "IMPROVED"
    elif p_t < 0.05 and mean > 0:
        verdict = "REGRESSED"
    return {
        "n": n,
        "mean_delta_nll": mean,
        "sd": sd,
        "se": se,
        "t": t,
        "p_two_sided": p_t,
        "p_normal_approx": p_normal,
        "exp_delta_nll": math.exp(mean),
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "verdict": verdict,
    }


def paired_row(comparison, tag, a_run, b_run, a_round, b_round,
               a_rows, b_rows, field="nll"):
    stats_row = paired_stats(a_rows, b_rows)
    return [
        comparison, tag, a_run, b_run, a_round, b_round, field,
        stats_row["n"], stats_row["mean_delta_nll"], stats_row["sd"],
        stats_row["se"], stats_row["t"], stats_row["p_two_sided"],
        stats_row["p_normal_approx"], stats_row["exp_delta_nll"],
        stats_row["ci95_low"], stats_row["ci95_high"], stats_row["verdict"],
    ]


def r30_package():
    rounds_rows = []
    endpoint_rows = []
    endpoints = {}
    eval_rows = {}
    for arm in R30_ARMS:
        run_dir = RUN_ROOT / "bert_da_fixed" / arm["run_id"] / arm["tag"]
        config = read_json(run_dir / "config.json")
        evals = eval_map(run_dir)
        records = round_map(run_dir)
        metrics = endpoint_metrics(run_dir)
        endpoints[arm["run_id"]] = metrics
        eval_rows[arm["run_id"]] = evals
        for round_idx in sorted(evals):
            row = evals[round_idx]
            record = records.get(round_idx, {})
            rounds_rows.append([
                arm["run_id"], arm["tag"], arm["catv"], config.get("catv_normalize"),
                round_idx, row.get("nll"), row.get("accuracy"), row.get("macro_f1"),
                row.get("n"), record.get("round_seconds"), record.get("train_loss_mean"),
                record.get("delta_norm_mean"), record.get("delta_norm_max"),
                record.get("clipped_clients"), record.get("catv_r"),
                record.get("anchor_in"), record.get("anchor_out"), record.get("ir_mean"),
                config.get("doc_order"), rel(run_dir),
            ])
        endpoint_rows.append([
            arm["run_id"], arm["tag"], arm["catv"], config.get("catv_normalize"),
            config.get("doc_order"), metrics["n_evals"], metrics["eval_n"],
            metrics["nll_r0"], metrics["nll_last"], metrics["delta_nll"],
            metrics["delta_nll_pct"], metrics["acc_r0"], metrics["acc_last"],
            metrics["macro_f1_r0"], metrics["macro_f1_last"],
            metrics["delta_acc_points"], metrics["delta_acc_pct"], metrics["err_reduction_pct"],
            metrics["clipped_clients_total"], metrics["mean_round_seconds"],
            metrics["wall_hours"], metrics["nll_pct_per_hour"],
            metrics["adapter_sha256"], rel(run_dir),
        ])
    write_csv(
        DATA_DIR / "bert_r30_rounds.csv",
        ["run_id", "tag", "catv", "catv_normalize", "round", "nll", "accuracy",
         "macro_f1", "n", "round_seconds", "train_loss_mean", "delta_norm_mean",
         "delta_norm_max", "clipped_clients", "catv_r", "catv_anchor_in",
         "catv_anchor_out", "catv_ir_mean", "doc_order", "source_dir"],
        rounds_rows,
    )
    write_csv(
        DATA_DIR / "bert_r30_endpoints.csv",
        ["run_id", "tag", "catv", "catv_normalize", "doc_order", "rounds", "eval_n",
         "nll_r0", "nll_r29", "delta_nll", "delta_nll_pct", "acc_r0", "acc_r29",
         "macro_f1_r0", "macro_f1_r29", "delta_acc_points", "delta_acc_pct",
         "err_reduction_pct", "clipped_clients_total", "mean_round_seconds", "wall_hours",
         "nll_pct_per_hour", "adapter_sha256", "source_dir"],
        endpoint_rows,
    )

    paired_rows = []
    run_dirs = {
        arm["run_id"]: RUN_ROOT / "bert_da_fixed" / arm["run_id"] / arm["tag"]
        for arm in R30_ARMS
    }
    for arm in R30_ARMS:
        run_dir = run_dirs[arm["run_id"]]
        first = endpoints[arm["run_id"]]["first_round"]
        last = endpoints[arm["run_id"]]["last_round"]
        paired_rows.append(paired_row(
            "r0_to_r29", arm["tag"], arm["run_id"], arm["run_id"], first, last,
            persample_file(run_dir, first), persample_file(run_dir, last),
        ))
    for tag in ("a01", "a03"):
        off_id = f"r30_{tag}_catv_off"
        on_id = f"r30_{tag}_catv_on"
        last = endpoints[off_id]["last_round"]
        paired_rows.append(paired_row(
            "catv_on_vs_off_r29", tag, off_id, on_id, last, last,
            persample_file(run_dirs[off_id], last), persample_file(run_dirs[on_id], last),
        ))
    write_csv(DATA_DIR / "bert_r30_paired.csv", PAIRED_HEADER, paired_rows)
    return {"endpoints": endpoints, "evals": eval_rows,
            "paired": {(row[0], row[1]): row for row in paired_rows}}


def r30_evaloff_package(r30):
    round_rows = []
    endpoint_rows = []
    paired_rows = []
    endpoints = {}
    for tag in ("a01", "a03"):
        on_dir = RUN_ROOT / "bert_da_fixed" / f"r30_{tag}_catv_on" / tag
        off_dir = RUN_ROOT / "bert_da_fixed" / f"r30_{tag}_catv_off" / tag
        for round_idx in range(30):
            on_row = eval_file(on_dir, round_idx, evaloff=False)
            off_row = eval_file(on_dir, round_idx, evaloff=True)
            round_rows.append([
                tag, round_idx, on_row.get("nll"), on_row.get("accuracy"),
                off_row.get("nll"), off_row.get("accuracy"),
                float(on_row["nll"]) - float(off_row["nll"]),
                float(on_row["accuracy"]) - float(off_row["accuracy"]),
            ])
        variants = [
            ("off", off_dir, False),
            ("on_evaloff", on_dir, True),
            ("on_evalon", on_dir, False),
        ]
        for label, run_dir, eo in variants:
            first = eval_file(run_dir, 0, evaloff=eo)
            last = eval_file(run_dir, 29, evaloff=eo)
            nll0 = float(first["nll"])
            nll29 = float(last["nll"])
            acc0 = float(first["accuracy"])
            acc29 = float(last["accuracy"])
            endpoints[(tag, label)] = {"nll_r0": nll0, "nll_r29": nll29,
                                       "acc_r0": acc0, "acc_r29": acc29,
                                       "macro_f1_r0": first.get("macro_f1"),
                                       "macro_f1_r29": last.get("macro_f1")}
            mask_label = "none" if label == "off" else ("evaloff" if eo else "evalon")
            endpoint_rows.append([
                tag, label, mask_label,
                nll0, nll29, (nll0 - nll29) / nll0 * 100.0,
                acc0, acc29, first.get("macro_f1"), last.get("macro_f1"),
                acc29 - acc0, (acc29 - acc0) / acc0 * 100.0,
                rel(run_dir),
            ])
        pairs = [
            ("weight_effect", f"r30_{tag}_catv_off", f"r30_{tag}_catv_on",
             29, 29, off_dir, on_dir, False, True),
            ("mask_effect", f"r30_{tag}_catv_on(evaloff)", f"r30_{tag}_catv_on(evalon)",
             29, 29, on_dir, on_dir, True, False),
            ("total_effect", f"r30_{tag}_catv_off", f"r30_{tag}_catv_on",
             29, 29, off_dir, on_dir, False, False),
            ("start_check", f"r30_{tag}_catv_off", f"r30_{tag}_catv_on(evaloff)",
             0, 0, off_dir, on_dir, False, True),
        ]
        for label, a_name, b_name, a_round, b_round, a_dir, b_dir, a_eo, b_eo in pairs:
            paired_rows.append(paired_row(
                label, tag, a_name, b_name, a_round, b_round,
                persample_file(a_dir, a_round, evaloff=a_eo),
                persample_file(b_dir, b_round, evaloff=b_eo),
            ))
    write_csv(
        DATA_DIR / "bert_r30_evaloff_rounds.csv",
        ["tag", "round", "nll_on", "accuracy_on", "nll_off", "accuracy_off",
         "delta_nll_on_minus_off", "delta_acc_on_minus_off"],
        round_rows,
    )
    write_csv(
        DATA_DIR / "bert_r30_evaloff_endpoints.csv",
        ["tag", "variant", "eval_mask", "nll_r0", "nll_r29", "delta_nll_pct",
         "acc_r0", "acc_r29", "macro_f1_r0", "macro_f1_r29",
         "delta_acc_points", "delta_acc_pct", "source_dir"],
        endpoint_rows,
    )
    write_csv(DATA_DIR / "bert_r30_evaloff_paired.csv", PAIRED_HEADER, paired_rows)
    return {"endpoints": endpoints,
            "paired": {(row[0], row[1]): row for row in paired_rows}}


def opt_package():
    round_rows = []
    endpoint_rows = []
    endpoints = {}
    run_dirs = {}
    for arm in OPT_ARMS:
        run_dir = RUN_ROOT / "opt_r30_fixed" / arm["run_id"] / arm["tag"]
        run_dirs[arm["run_id"]] = run_dir
        config = read_json(run_dir / "config.json")
        metrics = opt_endpoint_metrics(run_dir, arm["tag"])
        endpoints[arm["run_id"]] = metrics
        evals = opt_eval_map(run_dir)
        records = round_map(run_dir)
        for round_idx in [0] + OPT_EVAL_ROUNDS:
            if round_idx == 0:
                payload = opt_eval_file(run_dir, 0)
            else:
                payload = evals[round_idx]
            record = records.get(round_idx, {})
            round_rows.append([
                arm["run_id"], arm["tag"], arm["catv"], config.get("catv_normalize"),
                round_idx, "on" if arm["catv"] == "on" else "none",
                payload.get("full_ppl_token"), payload.get("answer_ppl_token"),
                payload.get("num_samples"), record.get("round_seconds"),
                record.get("train_loss_mean"), record.get("delta_norm_mean"),
                record.get("delta_norm_max"), record.get("clipped_clients"),
                record.get("catv_r"), record.get("anchor_in"), record.get("anchor_out"),
                record.get("ir_mean"), config.get("doc_order"), rel(run_dir),
            ])
        endpoint_rows.append([
            arm["run_id"], arm["tag"], arm["catv"], config.get("catv_normalize"),
            config.get("doc_order"), metrics["rounds"], metrics["eval_n"],
            metrics["full_r0"], metrics["full_r4"], metrics["full_r29"],
            metrics["answer_r0"], metrics["answer_r4"], metrics["answer_r29"],
            metrics["full_delta_pct_r0_r29"], metrics["answer_delta_pct_r0_r29"],
            metrics["full_delta_pct_r4_r29"], metrics["answer_delta_pct_r4_r29"],
            metrics["vs_fresh_full_pct"], metrics["vs_fresh_answer_pct"],
            metrics["clipped_clients_total"], metrics["mean_round_seconds"],
            metrics["train_hours"], metrics["wall_hours"],
            metrics["full_pct_per_hour_r4_r29"], metrics["adapter_sha256"],
            rel(run_dir),
        ])
    write_csv(
        DATA_DIR / "opt_r30_rounds.csv",
        ["run_id", "tag", "catv", "catv_normalize", "round", "eval_mask",
         "full_ppl_token", "answer_ppl_token", "num_samples", "round_seconds",
         "train_loss_mean", "delta_norm_mean", "delta_norm_max", "clipped_clients",
         "catv_r", "catv_anchor_in", "catv_anchor_out", "catv_ir_mean",
         "doc_order", "source_dir"],
        round_rows,
    )
    write_csv(
        DATA_DIR / "opt_r30_endpoints.csv",
        ["run_id", "tag", "catv", "catv_normalize", "doc_order", "rounds", "eval_n",
         "full_r0", "full_r4", "full_r29", "answer_r0", "answer_r4", "answer_r29",
         "full_delta_pct_r0_r29", "answer_delta_pct_r0_r29",
         "full_delta_pct_r4_r29", "answer_delta_pct_r4_r29",
         "vs_fresh_full_pct", "vs_fresh_answer_pct", "clipped_clients_total",
         "mean_round_seconds", "train_hours", "wall_hours",
         "full_pct_per_hour_r4_r29", "adapter_sha256", "source_dir"],
        endpoint_rows,
    )

    paired_rows = []
    for arm in OPT_ARMS:
        run_dir = run_dirs[arm["run_id"]]
        tag = arm["tag"]
        fresh_path = REPO / f"temp/eval_fixed_fresh_{tag}_persample.json"
        for field in ("full_nll_mean", "answer_nll_mean"):
            paired_rows.append(paired_row(
                "r0_to_r29", tag, arm["run_id"], arm["run_id"], 0, 29,
                persample_field(run_dir, 0, field),
                persample_field(run_dir, 29, field), field=field,
            ))
            paired_rows.append(paired_row(
                "r4_to_r29", tag, arm["run_id"], arm["run_id"], 4, 29,
                persample_field(run_dir, 4, field),
                persample_field(run_dir, 29, field), field=field,
            ))
            paired_rows.append(paired_row(
                "fresh_to_r29", tag, f"fresh_{tag}", arm["run_id"], 29, 29,
                persample_path(fresh_path, field),
                persample_field(run_dir, 29, field), field=field,
            ))
    for tag in ("a01", "a03"):
        off_dir = run_dirs[f"r30_{tag}_catv_off"]
        on_dir = run_dirs[f"r30_{tag}_catv_on"]
        for field in ("full_nll_mean", "answer_nll_mean"):
            paired_rows.append(paired_row(
                "catv_on_vs_off_r29", tag, f"r30_{tag}_catv_off",
                f"r30_{tag}_catv_on", 29, 29,
                persample_field(off_dir, 29, field),
                persample_field(on_dir, 29, field), field=field,
            ))
    write_csv(DATA_DIR / "opt_r30_paired.csv", PAIRED_HEADER, paired_rows)
    return {"endpoints": endpoints,
            "paired": {(row[0], row[1], row[6]): row for row in paired_rows}}


def opt_evaloff_package():
    round_rows = []
    endpoint_rows = []
    paired_rows = []
    endpoints = {}
    for tag in ("a01", "a03"):
        off_dir = RUN_ROOT / "opt_r30_fixed" / f"r30_{tag}_catv_off" / tag
        on_dir = RUN_ROOT / "opt_r30_fixed" / f"r30_{tag}_catv_on" / tag
        for round_idx in OPT_EVAL_ROUNDS:
            on_row = opt_eval_file(on_dir, round_idx, evaloff=False)
            off_row = opt_eval_file(on_dir, round_idx, evaloff=True)
            round_rows.append([
                tag, round_idx, on_row.get("full_ppl_token"),
                on_row.get("answer_ppl_token"), off_row.get("full_ppl_token"),
                off_row.get("answer_ppl_token"),
                float(on_row["full_ppl_token"]) - float(off_row["full_ppl_token"]),
                float(on_row["answer_ppl_token"]) - float(off_row["answer_ppl_token"]),
            ])
        variants = [
            ("off", off_dir, False),
            ("on_evaloff", on_dir, True),
            ("on_evalon", on_dir, False),
        ]
        for label, run_dir, eo in variants:
            r4 = opt_eval_file(run_dir, 4, evaloff=eo)
            r29 = opt_eval_file(run_dir, 29, evaloff=eo)
            full_r4 = float(r4["full_ppl_token"])
            full_r29 = float(r29["full_ppl_token"])
            answer_r4 = float(r4["answer_ppl_token"])
            answer_r29 = float(r29["answer_ppl_token"])
            endpoints[(tag, label)] = {"full_r4": full_r4, "full_r29": full_r29,
                                        "answer_r4": answer_r4, "answer_r29": answer_r29}
            endpoint_rows.append([
                tag, label, "none" if label == "off" else ("evaloff" if eo else "evalon"),
                full_r4, full_r29, answer_r4, answer_r29,
                (full_r4 - full_r29) / full_r4 * 100.0,
                (answer_r4 - answer_r29) / answer_r4 * 100.0, rel(run_dir),
            ])
        specs = [
            ("weight_effect", f"r30_{tag}_catv_off", f"r30_{tag}_catv_on(evaloff)",
             29, 29, off_dir, on_dir, False, True),
            ("mask_effect", f"r30_{tag}_catv_on(evaloff)", f"r30_{tag}_catv_on(evalon)",
             29, 29, on_dir, on_dir, True, False),
            ("deploy_net", f"r30_{tag}_catv_off", f"r30_{tag}_catv_on(evalon)",
             29, 29, off_dir, on_dir, False, False),
        ]
        for label, a_name, b_name, a_round, b_round, a_dir, b_dir, a_eo, b_eo in specs:
            for field in ("full_nll_mean", "answer_nll_mean"):
                paired_rows.append(paired_row(
                    label, tag, a_name, b_name, a_round, b_round,
                    persample_field(a_dir, a_round, field, evaloff=a_eo),
                    persample_field(b_dir, b_round, field, evaloff=b_eo),
                    field=field,
                ))
    write_csv(
        DATA_DIR / "opt_r30_evaloff_rounds.csv",
        ["tag", "round", "full_on", "answer_on", "full_off", "answer_off",
         "delta_full_on_minus_off", "delta_answer_on_minus_off"],
        round_rows,
    )
    write_csv(
        DATA_DIR / "opt_r30_evaloff_endpoints.csv",
        ["tag", "variant", "eval_mask", "full_r4", "full_r29", "answer_r4",
         "answer_r29", "full_delta_pct_r4_r29", "answer_delta_pct_r4_r29",
         "source_dir"],
        endpoint_rows,
    )
    write_csv(DATA_DIR / "opt_r30_evaloff_paired.csv", PAIRED_HEADER, paired_rows)
    return {"endpoints": endpoints,
            "paired": {(row[0], row[1], row[6]): row for row in paired_rows}}


def opt_baselines_package():
    rows = []
    for tag in ("a01", "a03"):
        path = f"temp/eval_fixed_fresh_{tag}.json"
        payload = read_json(REPO / path)
        rows.append([
            f"fresh_pretrain_{tag}", payload.get("num_samples"), tag,
            payload.get("attn"), payload.get("sparse"), payload.get("full_ppl_token"),
            payload.get("answer_ppl_token"), path,
            "OPT pre-training fresh baseline; a01/a03 identical by construction",
        ])
    write_csv(
        DATA_DIR / "opt_baselines.csv",
        ["baseline", "n", "tag", "attn", "sparse", "full_ppl_token",
         "answer_ppl_token", "source_file", "note"],
        rows,
    )
    return {row[0]: row for row in rows}


def bp_package():
    rows = []
    endpoints = {}
    for run_id in BP_RUNS:
        run_dir = RUN_ROOT / "bert_bp_ref_cycle" / run_id / "a01"
        config = read_json(run_dir / "config.json")
        metrics = endpoint_metrics(run_dir)
        endpoints[run_id] = metrics
        rows.append([
            run_id, config.get("trainer"), config.get("attn"), config.get("sparse"),
            config.get("doc_order"), config.get("local_steps"),
            config.get("zo_directions"), config.get("lr"), config.get("rounds"),
            metrics["eval_n"], metrics["nll_r0"], metrics["nll_last"],
            metrics["delta_nll_pct"], metrics["acc_r0"], metrics["acc_last"],
            metrics["delta_acc_points"], metrics["delta_acc_pct"],
            metrics["err_reduction_pct"], metrics["mean_round_seconds"],
            metrics["wall_hours"], metrics["nll_pct_per_hour"],
            metrics["mean_delta_norm"], metrics["clipped_clients_total"], rel(run_dir),
        ])
    write_csv(
        DATA_DIR / "bert_bp_cycle.csv",
        ["run_id", "trainer", "attn", "sparse", "doc_order", "local_steps",
         "zo_directions", "lr", "rounds", "eval_n", "nll_r0", "nll_last",
         "delta_nll_pct", "acc_r0", "acc_last", "delta_acc_points", "delta_acc_pct",
         "err_reduction_pct", "mean_round_seconds", "wall_hours", "nll_pct_per_hour",
         "mean_delta_norm", "clipped_clients_total", "source_dir"],
        rows,
    )
    return endpoints


def ld_package():
    rows = []
    endpoints = {}
    for run_id, role in LD_RUNS:
        run_dir = RUN_ROOT / "bert_ld" / run_id / "a01"
        config = read_json(run_dir / "config.json")
        metrics = endpoint_metrics(run_dir)
        endpoints[run_id] = metrics
        local_steps = int(config.get("local_steps") or 0)
        directions = int(config.get("zo_directions") or 0)
        doc_order = config.get("doc_order")
        rounds = int(config.get("rounds") or metrics["n_evals"])
        unique_docs = min(rounds * local_steps, 100) if doc_order == "cycle" else local_steps
        rows.append([
            run_id, role, config.get("attn"), config.get("sparse"), doc_order,
            local_steps, directions, local_steps * directions,
            len(range(metrics["first_round"], metrics["last_round"] + 1)),
            unique_docs, config.get("lr"), metrics["eval_n"],
            metrics["nll_r0"], metrics["nll_last"], metrics["delta_nll_pct"],
            metrics["acc_r0"], metrics["acc_last"], metrics["delta_acc_points"],
            metrics["delta_acc_pct"], metrics["err_reduction_pct"],
            metrics["mean_round_seconds"], metrics["wall_hours"],
            metrics["nll_pct_per_hour"], metrics["mean_delta_norm"],
            metrics["peak_delta_norm"], metrics["clipped_clients_total"], rel(run_dir),
        ])
    write_csv(
        DATA_DIR / "bert_ld_grid.csv",
        ["run_id", "role", "attn", "sparse", "doc_order", "local_steps",
         "zo_directions", "L_times_D", "rounds", "unique_docs_per_client", "lr",
         "eval_n", "nll_r0", "nll_last", "delta_nll_pct", "acc_r0", "acc_last",
         "delta_acc_points", "delta_acc_pct", "err_reduction_pct",
         "mean_round_seconds", "wall_hours", "nll_pct_per_hour", "mean_delta_norm",
         "peak_delta_norm", "clipped_clients_total", "source_dir"],
        rows,
    )
    return endpoints


def lr_scan_package():
    rows = []
    for run_dir in sorted((RUN_ROOT / "bert_lr_scan").glob("*")):
        metrics_path = run_dir / "a01" / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        records = round_map(run_dir / "a01")
        config = read_json(run_dir / "a01" / "config.json")
        ordered = [records[index] for index in sorted(records)]
        losses = [record.get("train_loss_mean") for record in ordered] + [None, None, None]
        deltas = [record.get("delta_norm_mean") for record in ordered] + [None, None, None]
        lr = float(config.get("lr"))
        rows.append([
            run_dir.name, lr, len(ordered), losses[0], losses[1], losses[2],
            deltas[0], deltas[1], deltas[2],
            (deltas[0] / lr) if deltas[0] is not None and lr else None,
            mean_optional([record.get("round_seconds") for record in ordered]),
            rel(run_dir / "a01"),
        ])
    rows.sort(key=lambda row: row[1])
    write_csv(
        DATA_DIR / "bert_lr_scan.csv",
        ["run_id", "lr", "rounds", "loss_r0", "loss_r1", "loss_r2",
         "delta_r0", "delta_r1", "delta_r2", "delta_per_lr_r0",
         "mean_round_seconds", "source_dir"],
        rows,
    )
    return {row[0]: row for row in rows}


def series_package(run_id, tag, extra_fields, filename):
    run_dir = RUN_ROOT / run_id / tag
    evals = eval_map(run_dir)
    records = round_map(run_dir)
    rows = []
    for round_idx in sorted(evals):
        row = evals[round_idx]
        record = records.get(round_idx, {})
        base = [
            round_idx, row.get("nll"), row.get("accuracy"), row.get("macro_f1"),
            record.get("round_seconds"), record.get("train_loss_mean"),
            record.get("delta_norm_mean"), record.get("clipped_clients"),
        ]
        extra = [record.get(field) for field in extra_fields]
        rows.append(base + extra + [rel(run_dir)])
    header = ["round", "nll", "accuracy", "macro_f1", "round_seconds",
              "train_loss_mean", "delta_norm_mean", "clipped_clients"]
    header += list(extra_fields) + ["source_dir"]
    write_csv(DATA_DIR / filename, header, rows)
    return {"run_dir": run_dir, "evals": evals, "records": records}


def vq_package():
    extra = ["vq_rate", "vq_sketch_dtype", "vq_c_dim", "vq_channels", "vq_cos_mean",
             "vq_cos_min", "vq_cos_max", "vq_communication_bytes",
             "dense_all_results_communication_bytes", "vq_ratio_bytes",
             "vq_total_ratio_bytes_vs_one_delta"]
    info = series_package("bert_vq050_bf16_perresult_a03", "a03", extra,
                          "bert_vq_rounds.csv")
    return endpoint_metrics(info["run_dir"]), info


def var_package():
    extra = ["var_threshold", "var_D_mean", "var_D_min", "var_D_max",
             "var_total_samples_mean", "var_total_samples_min",
             "var_total_samples_max", "var_hit_low_samples", "var_hit_high_samples",
             "var_norm_mean", "var_raw_mean"]
    info = series_package("bert_var_methoda_a03_r30", "a03", extra,
                          "bert_var_rounds.csv")
    return endpoint_metrics(info["run_dir"]), info


def combo_package():
    extra = ["catv_r", "anchor_in", "anchor_out", "ir_mean",
             "var_threshold", "var_D_mean", "var_D_min", "var_D_max",
             "var_total_samples_mean", "var_total_samples_min",
             "var_total_samples_max", "var_hit_low_samples", "var_hit_high_samples",
             "vq_rate", "vq_sketch_dtype", "vq_c_dim", "vq_channels_mean",
             "vq_channels_min", "vq_channels_max", "vq_cos_mean", "vq_cos_min",
             "vq_cos_max", "vq_communication_bytes",
             "dense_all_results_communication_bytes", "vq_ratio_bytes",
             "vq_total_ratio_bytes_vs_one_delta"]
    info = series_package("bert_var_catv_vq_a03_r30_rerun", "a03", extra,
                          "bert_combo_rounds.csv")
    return endpoint_metrics(info["run_dir"]), info


def combo_evaloff_package():
    run_dir = RUN_ROOT / "bert_var_catv_vq_a03_r30_rerun" / "a03"
    rows = []
    for round_idx in range(30):
        on_row = eval_file(run_dir, round_idx, evaloff=False)
        off_row = eval_file(run_dir, round_idx, evaloff=True)
        rows.append([
            round_idx, on_row.get("nll"), on_row.get("accuracy"),
            off_row.get("nll"), off_row.get("accuracy"),
            float(on_row["nll"]) - float(off_row["nll"]),
            float(on_row["accuracy"]) - float(off_row["accuracy"]),
        ])
    write_csv(
        DATA_DIR / "bert_combo_evaloff_rounds.csv",
        ["round", "nll_on", "accuracy_on", "nll_off", "accuracy_off",
         "delta_nll_on_minus_off", "delta_acc_on_minus_off"],
        rows,
    )
    return {"nll_r0": float(eval_file(run_dir, 0, evaloff=True)["nll"]),
            "nll_r29": float(eval_file(run_dir, 29, evaloff=True)["nll"]),
            "acc_r0": float(eval_file(run_dir, 0, evaloff=True)["accuracy"]),
            "acc_r29": float(eval_file(run_dir, 29, evaloff=True)["accuracy"]),
            "macro_f1_r0": eval_file(run_dir, 0, evaloff=True).get("macro_f1"),
            "macro_f1_r29": eval_file(run_dir, 29, evaloff=True).get("macro_f1")}


def technique_package(vq, var, combo, combo_evaloff, r30):
    rows = []
    endpoints = {}
    run_specs = dict(TECH_RUNS)
    for key, spec in run_specs.items():
        run_dir = (RUN_ROOT / spec["group"] / spec["run_id"] / spec["tag"]
                   if spec["group"] else RUN_ROOT / spec["run_id"] / spec["tag"])
        metrics = r30["endpoints"][spec["run_id"]] if spec["run_id"] in r30["endpoints"] else endpoint_metrics(run_dir)
        records = round_map(run_dir)
        endpoints[key] = metrics
        eval_mask = "on" if spec["catv"] == "on" else "none"
        rows.append([
            key, spec["technique"], spec["tag"], spec["catv"], spec["catv_normalize"],
            eval_mask, spec["var_control"], spec["vq_rate"],
            metrics["nll_r0"], metrics["nll_last"],
            metrics["delta_nll_pct"], metrics["acc_r0"], metrics["acc_last"],
            metrics["macro_f1_r0"], metrics["macro_f1_last"],
            metrics["delta_acc_points"], metrics["delta_acc_pct"],
            metrics["err_reduction_pct"], metrics["mean_round_seconds"],
            metrics["wall_hours"], metrics["nll_pct_per_hour"],
            mean_optional([record.get("var_D_mean") for record in records.values()]),
            mean_optional([record.get("var_total_samples_mean") for record in records.values()]),
            mean_optional([record.get("vq_cos_mean") for record in records.values()]),
            mean_optional([record.get("vq_channels_mean", record.get("vq_channels"))
                           for record in records.values()]),
            sum_optional([record.get("vq_communication_bytes") for record in records.values()]),
            sum_optional([record.get("dense_all_results_communication_bytes")
                          for record in records.values()]),
            mean_optional([record.get("vq_total_ratio_bytes_vs_one_delta")
                           for record in records.values()]),
            mean_optional([record.get("ir_mean") for record in records.values()]),
            metrics["adapter_sha256"], rel(run_dir),
        ])
    evaloff_key = "a03_combo_evaloff"
    metrics = endpoints["a03_combo"]
    combo_dir = RUN_ROOT / "bert_var_catv_vq_a03_r30_rerun" / "a03"
    combo_records = round_map(combo_dir)
    combo_means = {
        "mean_D": mean_optional([record.get("var_D_mean") for record in combo_records.values()]),
        "mean_total_samples": mean_optional([record.get("var_total_samples_mean")
                                             for record in combo_records.values()]),
        "mean_vq_cos": mean_optional([record.get("vq_cos_mean")
                                      for record in combo_records.values()]),
        "mean_vq_channels": mean_optional([record.get("vq_channels_mean",
                                                      record.get("vq_channels"))
                                           for record in combo_records.values()]),
        "vq_bytes_total": sum_optional([record.get("vq_communication_bytes")
                                        for record in combo_records.values()]),
        "dense_all_bytes_total": sum_optional([record.get("dense_all_results_communication_bytes")
                                               for record in combo_records.values()]),
        "mean_vq_total_ratio": mean_optional([record.get("vq_total_ratio_bytes_vs_one_delta")
                                              for record in combo_records.values()]),
        "catv_ir_mean": mean_optional([record.get("ir_mean")
                                       for record in combo_records.values()]),
    }
    endpoints[evaloff_key] = {
        "nll_r0": combo_evaloff["nll_r0"], "nll_last": combo_evaloff["nll_r29"],
        "acc_r0": combo_evaloff["acc_r0"], "acc_last": combo_evaloff["acc_r29"],
        "macro_f1_r0": combo_evaloff["macro_f1_r0"],
        "macro_f1_last": combo_evaloff["macro_f1_r29"],
        "mean_round_seconds": metrics["mean_round_seconds"],
        "wall_hours": metrics["wall_hours"], "adapter_sha256": metrics["adapter_sha256"],
        **combo_means,
    }
    rows.append([
        evaloff_key, "VAR+CATV(client)+VQ 0.5 (eval off)", "a03", "on", "client",
        "off", "on", 0.5, combo_evaloff["nll_r0"], combo_evaloff["nll_r29"],
        (combo_evaloff["nll_r0"] - combo_evaloff["nll_r29"]) / combo_evaloff["nll_r0"] * 100.0,
        combo_evaloff["acc_r0"], combo_evaloff["acc_r29"],
        combo_evaloff["macro_f1_r0"], combo_evaloff["macro_f1_r29"],
        combo_evaloff["acc_r29"] - combo_evaloff["acc_r0"],
        (combo_evaloff["acc_r29"] - combo_evaloff["acc_r0"]) / combo_evaloff["acc_r0"] * 100.0,
        (1.0 - (1.0 - combo_evaloff["acc_r29"]) / (1.0 - combo_evaloff["acc_r0"])) * 100.0,
        metrics["mean_round_seconds"], metrics["wall_hours"],
        ((combo_evaloff["nll_r0"] - combo_evaloff["nll_r29"]) / combo_evaloff["nll_r0"] * 100.0)
        / metrics["wall_hours"] if metrics["wall_hours"] else None,
        combo_means["mean_D"], combo_means["mean_total_samples"],
        combo_means["mean_vq_cos"], combo_means["mean_vq_channels"],
        combo_means["vq_bytes_total"], combo_means["dense_all_bytes_total"],
        combo_means["mean_vq_total_ratio"], combo_means["catv_ir_mean"],
        metrics["adapter_sha256"], rel(combo_dir),
    ])
    write_csv(
        DATA_DIR / "bert_technique_endpoints.csv",
        ["run", "technique", "tag", "catv", "catv_normalize", "eval_mask",
         "var_control", "vq_rate", "nll_r0", "nll_r29", "delta_nll_pct", "acc_r0", "acc_r29",
         "macro_f1_r0", "macro_f1_r29", "delta_acc_points", "delta_acc_pct",
         "err_reduction_pct", "mean_round_seconds", "wall_hours", "nll_pct_per_hour",
         "mean_D",
         "mean_total_samples", "mean_vq_cos", "mean_vq_channels", "vq_bytes_total",
         "dense_all_bytes_total", "mean_vq_total_ratio", "catv_ir_mean",
         "adapter_sha256", "source_dir"],
        rows,
    )

    base_off_dir = RUN_ROOT / "bert_da_fixed" / "r30_a03_catv_off" / "a03"
    base_on_dir = RUN_ROOT / "bert_da_fixed" / "r30_a03_catv_on" / "a03"
    vq_dir = RUN_ROOT / "bert_vq050_bf16_perresult_a03" / "a03"
    var_dir = RUN_ROOT / "bert_var_methoda_a03_r30" / "a03"
    combo_dir = RUN_ROOT / "bert_var_catv_vq_a03_r30_rerun" / "a03"
    pair_specs = [
        ("vq_vs_catv_off_r29", "a03", "r30_a03_catv_off", "vq_perresult", 29, 29,
         base_off_dir, vq_dir, False, False),
        ("var_vs_catv_off_r29", "a03", "r30_a03_catv_off", "var_methoda", 29, 29,
         base_off_dir, var_dir, False, False),
        ("combo_vs_catv_on_r29", "a03", "r30_a03_catv_on", "combo", 29, 29,
         base_on_dir, combo_dir, False, False),
        ("combo_vs_catv_off_r29", "a03", "r30_a03_catv_off", "combo", 29, 29,
         base_off_dir, combo_dir, False, False),
        ("combo_evaloff_vs_catv_on_evaloff_r29", "a03",
         "r30_a03_catv_on(evaloff)", "combo(evaloff)", 29, 29,
         base_on_dir, combo_dir, True, True),
        ("combo_mask_effect_r29", "a03", "combo(evaloff)", "combo(evalon)", 29, 29,
         combo_dir, combo_dir, True, False),
        ("combo_evaloff_vs_catv_off_r29", "a03", "r30_a03_catv_off",
         "combo(evaloff)", 29, 29, base_off_dir, combo_dir, False, True),
    ]
    paired_rows = []
    for label, tag, a_name, b_name, a_round, b_round, a_dir, b_dir, a_eo, b_eo in pair_specs:
        paired_rows.append(paired_row(
            label, tag, a_name, b_name, a_round, b_round,
            persample_file(a_dir, a_round, evaloff=a_eo),
            persample_file(b_dir, b_round, evaloff=b_eo),
        ))
    write_csv(DATA_DIR / "bert_technique_paired.csv", PAIRED_HEADER, paired_rows)
    return {"endpoints": endpoints,
            "paired": {(row[0], row[1]): row for row in paired_rows}}


def baselines_package():
    specs = [
        ("step1_final_global500", "temp/eval_step1_16k_final.json",
         "Step1 最终头；dense 推理；r30 的初始化/上下文锚点"),
        ("step1_best_global500", "temp/eval_step1_16k_best.json",
         "Step1 best 头；仅作备选锚点，未用于正式链"),
        ("frozen_dense_250_subset", "temp/eval_baseline_250.json",
         "冻结 Step1 头在前 250 个 global_test 样本；LD/BP 表的共同 250 样本锚点"),
    ]
    rows = []
    for name, path, note in specs:
        payload = read_json(REPO / path)
        rows.append([
            name, payload.get("n"), payload.get("tag"), payload.get("attn"),
            payload.get("sparse"), payload.get("catv_mask"), payload.get("nll"),
            payload.get("accuracy"), payload.get("macro_f1"), path, note,
        ])
    write_csv(
        DATA_DIR / "bert_baselines.csv",
        ["baseline", "n", "tag", "attn", "sparse", "catv_mask", "nll",
         "accuracy", "macro_f1", "source_file", "note"],
        rows,
    )
    return {row[0]: row for row in rows}


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def bert_config_payload():
    run_dir = RUN_ROOT / "bert_da_fixed" / "r30_a03_catv_off" / "a03"
    config = read_json(run_dir / "config.json")
    model_config = read_json(Path(config["model_dir"]) / "config.json")
    on_config = read_json(RUN_ROOT / "bert_da_fixed" / "r30_a03_catv_on" / "a03" / "config.json")
    vq_config = read_json(RUN_ROOT / "bert_vq050_bf16_perresult_a03" / "a03" / "config.json")
    var_config = read_json(RUN_ROOT / "bert_var_methoda_a03_r30" / "a03" / "config.json")
    combo_config = read_json(RUN_ROOT / "bert_var_catv_vq_a03_r30_rerun" / "a03" / "config.json")
    meta_a01 = read_json(REPO / "dataset_v3" / "arxiv_16k" / "a01" / "meta.json")
    meta_a03 = read_json(REPO / "dataset_v3" / "arxiv_16k" / "a03" / "meta.json")
    warmup_meta = read_json(REPO / "dataset_v3" / "arxiv_16k" / "a01" / "warmup_meta.json")
    on_round = round_map(RUN_ROOT / "bert_da_fixed" / "r30_a03_catv_on" / "a03").get(0, {})
    var_records = round_map(RUN_ROOT / "bert_var_methoda_a03_r30" / "a03")
    combo_records = round_map(RUN_ROOT / "bert_var_catv_vq_a03_r30_rerun" / "a03")
    var_mean_d = mean_optional([record.get("var_D_mean") for record in var_records.values()])
    combo_mean_d = mean_optional([record.get("var_D_mean") for record in combo_records.values()])
    combo_mean_c = mean_optional([record.get("vq_channels_mean") for record in combo_records.values()])

    train_a01 = meta_a01["length_stats"]["train"]
    train_a03 = meta_a03["length_stats"]["train"]
    payload = {
        "generated_from": {
            "run_config": rel(run_dir / "config.json"),
            "model_config": rel(Path(config["model_dir"]) / "config.json"),
            "data_meta": ["dataset_v3/arxiv_16k/a01/meta.json",
                          "dataset_v3/arxiv_16k/a03/meta.json",
                          "dataset_v3/arxiv_16k/a01/warmup_meta.json"],
            "note": "Canonical values are from the r30 four-arm BERT run config; variants differ as listed.",
        },
        "model": {
            "name": "distilbert-base-uncased",
            "task": "11-class arXiv sequence classification",
            "num_layers": model_config["n_layers"],
            "num_attention_heads": model_config["n_heads"],
            "hidden_size": model_config["dim"],
            "ffn_size": model_config["hidden_dim"],
            "vocab_size": model_config["vocab_size"],
            "num_labels": config["num_labels"],
            "base_max_position_embeddings": model_config["max_position_embeddings"],
            "position_adaptation": "distilbert_step1/a01_pos_head_1000step/pos_embed.pt (16k)",
            "classification_head": "distilbert_step1/a01_pos_head_1000step/head.pt (frozen in FL)",
            "predictor": "predictor_bert/a01_warmup_400step/predictor.pth",
            "pruned_config": "predictor_bert/a01_warmup_400step/pruned_config.pth",
            "predictor_tensors": config["predictor_loaded_tensors"],
            "predictor_zero_filled_tensors": config["predictor_zero_filled_tensors"],
            "trainable_parameters": config["trainable_params"],
        },
        "sequence_and_attention": {
            "seq_len": config["effective_seq_len"],
            "truncation": "head_tail",
            "head_fraction": meta_a01["head_frac"],
            "head_tokens": int(config["effective_seq_len"] * meta_a01["head_frac"]),
            "tail_tokens": config["effective_seq_len"] - int(config["effective_seq_len"] * meta_a01["head_frac"]),
            "padding": "right, pad_token_id=0",
            "storage_dtype": "uint16",
            "attention_backend": config["attn"],
            "sparse_keep_ratio": config["sparse"],
            "n_blocks": config["n_blocks"],
            "block_size": config["effective_seq_len"] // config["n_blocks"],
            "keep_block0": config["keep_block0"],
            "pruned_layers": "2,3,4 (top-40% query blocks); 0,1 full; 5 exempt",
            "dtype": config["dtype"],
        },
        "lora": {
            "rank_r": config["lora_r"],
            "alpha": config["lora_alpha"],
            "scaling_alpha_over_r": config["lora_alpha"] / config["lora_r"],
            "dropout": 0.0,
            "bias": "none",
            "task_type": "SEQ_CLS",
            "target_modules": config["lora_targets"],
            "head_frozen": True,
            "trainable_parameters": config["trainable_params"],
            "parameter_formula": "6 layers x 4 modules x (768x1 + 1x768) = 36,864",
        },
        "federated": {
            "trainer": config["trainer"],
            "clients": config["max_clients"],
            "local_steps": config["local_steps"],
            "local_batch_size": 1,
            "rounds": config["rounds"],
            "learning_rate": config["lr"],
            "zo_subspace": config["zo_subspace"],
            "zo_layers": config["zo_layers"],
            "zo_directions": config["zo_directions"],
            "zo_epsilon": config["zo_eps"],
            "client_update": "ZOO central-difference; immediate update per sample; no optimizer",
            "server_update": "FedAvg; sample-weighted (all clients have 100 docs -> equal); per-client delta L2-clipped to 1.0 before aggregation",
            "delta_clip": config["delta_clip"],
            "weighting": config["weight"],
            "doc_order": config["doc_order"],
            "seed": config["seed"],
            "forwards_per_round": f"{config['max_clients']} clients x {config['local_steps']} samples x {config['zo_directions']} directions x 2 = "
                                  f"{config['max_clients'] * config['local_steps'] * config['zo_directions'] * 2} forwards",
        },
        "evaluation": {
            "split": "global_test",
            "samples": config["eval_max_samples"],
            "batch_size": config["eval_batch_size"],
            "every_rounds": config["eval_every"],
            "metrics": "NLL (mean per-sample CE), accuracy, macro-F1",
        },
        "data": {
            "source": meta_a01["source"],
            "config": meta_a01["config"],
            "label_names": meta_a01["label_names"],
            "seq_len": meta_a01["seq_len"],
            "truncation": meta_a01["truncation"],
            "head_frac": meta_a01["head_frac"],
            "pad_token_id": meta_a01["pad_token_id"],
            "seed": meta_a01["seed"],
            "global_demo_frac": meta_a01["global_demo_frac"],
            "train_split_size": meta_a01["counts"]["train_split_size"],
            "test_split_size": meta_a01["counts"]["test_split_size"],
            "client_source_size": meta_a01["counts"]["client_source_size"],
            "global_demo_source_size": meta_a01["counts"]["global_demo_source_size"],
            "clients": meta_a01["counts"]["clients"],
            "train_per_client": meta_a01["counts"]["train_per_client"],
            "local_test_per_client": meta_a01["counts"]["local_test_per_client"],
            "global_test": meta_a01["counts"]["global_test"],
            "global_test_role": "random test rows (seed [42,303]); all FL evaluation",
            "global_train": meta_a01["global_pool"]["train"],
            "global_train_role": "Step1 position/head training",
            "global_val": meta_a01["global_pool"]["val"],
            "global_val_role": "Step1 validation",
            "global_val_frac": meta_a01["global_pool"]["val_frac"],
            "warmup": warmup_meta["samples"],
            "warmup_role": "predictor training (400 steps); disjoint from client/global pools",
            "excluded_total": warmup_meta["partition"]["excluded_total"],
            "available_pool": warmup_meta["partition"]["available_pool"],
            "tags": {
                "a01": {
                    "alpha": meta_a01["alpha"],
                    "train_length_mean": train_a01["mean"],
                    "train_truncated": train_a01["truncated"],
                    "train_rows": train_a01["rows"],
                    "fallback_reuse_total": meta_a01["fallback_reuse"]["total_client_reuse"],
                },
                "a03": {
                    "alpha": meta_a03["alpha"],
                    "train_length_mean": train_a03["mean"],
                    "train_truncated": train_a03["truncated"],
                    "train_rows": train_a03["rows"],
                    "fallback_reuse_total": meta_a03["fallback_reuse"]["total_client_reuse"],
                },
            },
            "global_split_note": "global_test / global_train / global_val rows are shared across a01/a03; only the client partition differs",
            "doc_order": "cycle: start = (round * local_steps) % 100; 30 rounds x 8 docs/client = 240 doc-views (~2.4 passes over 100 docs/client)",
            "fixed_legacy": "fixed (first L rows every round) is the pre-261012 self-loop bug; use cycle for all reported BERT/OPT results",
            "quirks": [
                "a01: total_client_reuse=133; client_17 has only 10 unique docs (90 duplicate storage rows)",
                "a03: total_client_reuse=10 (client_29)",
                "a01/a03 global_train/global_val/global_test rows are row-identical; alpha only changes client partition",
            ],
        },
        "variants": {
            "catv_on": {
                "catv_r": on_config["resolved_catv_r"],
                "sparse_s": on_config["sparse"],
                "anchor_in": on_round.get("anchor_in"),
                "anchor_out": on_round.get("anchor_out"),
                "normalize_four_arm": on_config["catv_normalize"],
                "normalize_combo": combo_config["catv_normalize"],
            },
            "vq_perresult": {
                "rate": vq_config["vq_rate"],
                "sketch_dtype": vq_config["vq_sketch_dtype"],
                "channels_C": vq_config["vq_channels"],
                "c_dim": 18432,
                "total_ratio_vs_one_delta": 32.0,
            },
            "var_methoda": {
                "control": var_config["var_control"],
                "threshold_T0": var_config["var_threshold"],
                "D_range": "4..16, even-D checks, no patience",
                "segment_rounds": var_config["var_segment_rounds"],
                "probe_clients": var_config["var_probe_clients"],
                "measured_mean_D": var_mean_d,
            },
            "combo": {
                "catv": combo_config["catv"],
                "catv_normalize": combo_config["catv_normalize"],
                "var_control": combo_config["var_control"],
                "vq_rate": combo_config["vq_rate"],
                "measured_mean_D": combo_mean_d,
                "measured_mean_C": combo_mean_c,
            },
        },
        "notes": [
            "Sequence length 16k requires the learned position table (pos_embed.pt); the base DistilBERT config still says 512.",
            "Only LoRA parameters are trainable; the Step1 classification head is frozen and not federated.",
            "The four-arm CATV runs use raw summed votes (catv_normalize=off); the system combo uses per-client L1 normalization (client).",
            "Do not compare BERT NLL/accuracy with OPT token PPL; see opt_* CSVs for OPT.",
        ],
    }
    return payload


def bert_config_markdown(payload):
    m = payload["model"]
    s = payload["sequence_and_attention"]
    l = payload["lora"]
    f = payload["federated"]
    e = payload["evaluation"]
    d = payload["data"]
    v = payload["variants"]
    lines = [
        "# BERT experiment configuration (paper-facing)",
        "",
        "由 `reports/bert_and_other/make_report.py` 从实际 `config.json` 与 `dataset_v3/arxiv_16k/*/meta.json` 生成；",
        "对应 BERT r30 四臂正式口径。机器可读版见 `bert_config.json`。",
        "",
        "## 1. Model",
        "",
        markdown_table(["item", "value"], [
            ("model", m["name"]),
            ("task", m["task"]),
            ("layers / heads / hidden / FFN", f"{m['num_layers']} / {m['num_attention_heads']} / {m['hidden_size']} / {m['ffn_size']}"),
            ("vocab / labels", f"{m['vocab_size']} / {m['num_labels']}"),
            ("position adaptation", m["position_adaptation"]),
            ("classification head", m["classification_head"]),
            ("predictor", m["predictor"]),
            ("pruned config", m["pruned_config"]),
            ("trainable params", f"{m['trainable_parameters']:,} (LoRA only)"),
        ]),
        "",
        "## 2. Sequence and sparse attention",
        "",
        markdown_table(["item", "value"], [
            ("sequence length", f"{s['seq_len']:,}"),
            ("truncation", f"{s['truncation']}: {s['head_fraction']:.0%} head + {1 - s['head_fraction']:.0%} tail"),
            ("head / tail tokens", f"{s['head_tokens']:,} / {s['tail_tokens']:,}"),
            ("padding", s["padding"]),
            ("storage dtype", s["storage_dtype"]),
            ("attention backend", s["attention_backend"]),
            ("sparse keep ratio s", s["sparse_keep_ratio"]),
            ("blocks / block size", f"{s['n_blocks']} / {s['block_size']}"),
            ("pruned layers", s["pruned_layers"]),
            ("keep block0", s["keep_block0"]),
            ("compute dtype", s["dtype"]),
        ]),
        "",
        "## 3. LoRA",
        "",
        markdown_table(["item", "value"], [
            ("rank r", l["rank_r"]),
            ("alpha", l["alpha"]),
            ("scaling alpha/r", l["scaling_alpha_over_r"]),
            ("dropout", l["dropout"]),
            ("bias", l["bias"]),
            ("task type", l["task_type"]),
            ("target modules", ", ".join(l["target_modules"])),
            ("classification head", "frozen (not federated)"),
            ("trainable params", f"{l['trainable_parameters']:,}"),
            ("parameter formula", l["parameter_formula"]),
        ]),
        "",
        "## 4. Federated optimization",
        "",
        markdown_table(["item", "value"], [
            ("trainer", f["trainer"]),
            ("clients", f["clients"]),
            ("local steps (samples/client/round)", f["local_steps"]),
            ("local batch size", f["local_batch_size"]),
            ("rounds", f["rounds"]),
            ("learning rate", f["learning_rate"]),
            ("ZOO subspace", f["zo_subspace"]),
            ("ZOO layers", "0-5"),
            ("ZOO directions D", f["zo_directions"]),
            ("ZOO epsilon", f["zo_epsilon"]),
            ("client update", f["client_update"]),
            ("server update", f["server_update"]),
            ("delta clip (L2)", f["delta_clip"]),
            ("weighting", f["weighting"]),
            ("doc order", f["doc_order"]),
            ("seed", f["seed"]),
            ("forwards per round", f["forwards_per_round"]),
        ]),
        "",
        "## 5. Data construction",
        "",
        markdown_table(["item", "value"], [
            ("source", f"{d['source']} [{d['config']}]"),
            ("classes", ", ".join(d["label_names"])),
            ("data root", "dataset_v3/arxiv_16k/{a01,a03}"),
            ("sequence length", f"{d['seq_len']:,}"),
            ("truncation", f"{d['truncation']}, head_frac={d['head_frac']}"),
            ("partition", "per-class Dirichlet"),
            ("alpha", f"a01={d['tags']['a01']['alpha']}, a03={d['tags']['a03']['alpha']}"),
            ("partition seed", d["seed"]),
            ("clients", d["clients"]),
            ("train / local-test per client", f"{d['train_per_client']} / {d['local_test_per_client']}"),
            ("global test", f"{d['global_test']} — {d['global_test_role']}"),
            ("global train / val", f"{d['global_train']} / {d['global_val']} — Step1 position/head train / val"),
            ("warmup", f"{d['warmup']} — predictor training (400 steps), disjoint"),
            ("global_demo_frac", d["global_demo_frac"]),
            ("global_val_frac", d["global_val_frac"]),
            ("train / test split size", f"{d['train_split_size']} / {d['test_split_size']}"),
            ("client source / held-out", f"{d['client_source_size']} / {d['global_demo_source_size']}"),
            ("excluded / available pool", f"{d['excluded_total']} / {d['available_pool']}"),
            ("doc order", d["doc_order"]),
        ]),
        "",
        "| tag | train rows | mean length | truncated at 16k | fallback reuse |",
        "|---|---:|---:|---:|---:|",
        f"| a01 | {d['tags']['a01']['train_rows']} | {d['tags']['a01']['train_length_mean']} | {d['tags']['a01']['train_truncated']} | {d['tags']['a01']['fallback_reuse_total']} |",
        f"| a03 | {d['tags']['a03']['train_rows']} | {d['tags']['a03']['train_length_mean']} | {d['tags']['a03']['train_truncated']} | {d['tags']['a03']['fallback_reuse_total']} |",
        "",
        f"- Global split: {d['global_split_note']}.",
        f"- `warmup`: {d['warmup']} samples, disjoint from client/global pools.",
        "- Data quirks:",
    ]
    for quirk in d["quirks"]:
        lines.append(f"  - {quirk}")
    lines += [
        "",
        "## 6. Evaluation",
        "",
        markdown_table(["item", "value"], [
            ("split", e["split"]),
            ("samples", e["samples"]),
            ("batch size", e["batch_size"]),
            ("cadence", f"every {e['every_rounds']} round(s)"),
            ("metrics", e["metrics"]),
        ]),
        "",
        "## 7. Variants",
        "",
        markdown_table(["variant", "key settings"], [
            ("CATV on (four-arm)", f"r={v['catv_on']['catv_r']}, s={v['catv_on']['sparse_s']}, "
                                    f"anchors {v['catv_on']['anchor_in']}/{v['catv_on']['anchor_out']}, "
                                    f"normalize={v['catv_on']['normalize_four_arm']}"),
            ("VQ per-result", f"rate={v['vq_perresult']['rate']}, {v['vq_perresult']['sketch_dtype']}, "
                              f"C={v['vq_perresult']['channels_C']}, c_dim={v['vq_perresult']['c_dim']}, "
                              f"total={v['vq_perresult']['total_ratio_vs_one_delta']}x one delta"),
            ("VAR Method A", f"control={v['var_methoda']['control']}, T0={v['var_methoda']['threshold_T0']}, "
                             f"{v['var_methoda']['D_range']}, segments={v['var_methoda']['segment_rounds']} rounds, "
                             f"probe={v['var_methoda']['probe_clients']} clients, mean D={v['var_methoda']['measured_mean_D']:.3f}"),
            ("Combo", f"CATV={v['combo']['catv']} ({v['combo']['catv_normalize']}), VAR={v['combo']['var_control']}, "
                      f"VQ={v['combo']['vq_rate']}, mean D={v['combo']['measured_mean_D']:.3f}, "
                      f"mean C={v['combo']['measured_mean_C']:.2f}"),
        ]),
        "",
        "## 8. Notes",
        "",
    ]
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def bert_config_package(out_root):
    payload = bert_config_payload()
    (Path(out_root) / "bert_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (Path(out_root) / "BERT_CONFIG.md").write_text(
        bert_config_markdown(payload), encoding="utf-8")
    return payload


def provenance_package(r30, bpp, ldp, lr_scan, opt):
    rows = []

    def add(group, run_id, role, raw_path, doc_order, rounds, eval_n, status, note):
        rows.append([group, run_id, role, raw_path, doc_order, rounds, eval_n, status, note])

    for arm in R30_ARMS:
        run_dir = RUN_ROOT / "bert_da_fixed" / arm["run_id"] / arm["tag"]
        metrics = r30["endpoints"][arm["run_id"]]
        add("r30_four_arm", arm["run_id"], "headline", rel(run_dir),
            "cycle", metrics["n_evals"], metrics["eval_n"], "USE",
            "CATV 四臂消融；eval-off 文件在同目录")
    for arm in OPT_ARMS:
        run_dir = RUN_ROOT / "opt_r30_fixed" / arm["run_id"] / arm["tag"]
        metrics = opt["endpoints"][arm["run_id"]]
        add("opt_r30_fixed", arm["run_id"], "headline", rel(run_dir),
            "cycle", metrics["rounds"], metrics["eval_n"], "USE",
            "OPT fixed r30；eval every 5；r0 为 post-round-0 补测")
    for run_id in BP_RUNS:
        run_dir = RUN_ROOT / "bert_bp_ref_cycle" / run_id / "a01"
        metrics = bpp[run_id]
        add("bp_cycle", run_id, "reference", rel(run_dir), "cycle",
            metrics["n_evals"], metrics["eval_n"], "USE_REFERENCE",
            "BP 参考系；250 样本评测；不与 r30 绝对 NLL 混用")
    for run_id, role in LD_RUNS:
        run_dir = RUN_ROOT / "bert_ld" / run_id / "a01"
        metrics = ldp[run_id]
        doc_order = "fixed" if run_id == "fixed_L8D8" else "cycle"
        add("ld_grid", run_id, role, rel(run_dir), doc_order,
            metrics["n_evals"], metrics["eval_n"], "USE_CONTROL",
            "L×D 网格/对照；250 样本评测")
    for run_id in sorted(lr_scan):
        run_dir = RUN_ROOT / "bert_lr_scan" / run_id / "a01"
        add("lr_scan", run_id, "calibration", rel(run_dir), "cycle",
            3, None, "USE_CALIBRATION", "无 eval；仅 train_loss/delta")
    for run_id, tag in [
        ("bert_vq050_bf16_perresult_a03", "a03"),
        ("bert_var_methoda_a03_r30", "a03"),
        ("bert_var_catv_vq_a03_r30_rerun", "a03"),
    ]:
        run_dir = RUN_ROOT / run_id / tag
        add("technique", run_id, "headline", rel(run_dir), "cycle",
            30, 500, "USE", "VQ/VAR/组合系统线")
    for name, path, _note in [
        ("step1_final_global500", "temp/eval_step1_16k_final.json", ""),
        ("step1_best_global500", "temp/eval_step1_16k_best.json", ""),
        ("frozen_dense_250_subset", "temp/eval_baseline_250.json", ""),
    ]:
        add("baseline", name, "anchor", path, "n/a", 0, None, "USE_ANCHOR",
            "冻结权重锚点")
    write_csv(
        DATA_DIR / "provenance.csv",
        ["group", "run_id", "role", "raw_path", "doc_order", "rounds", "eval_n",
         "status", "note"],
        rows,
    )
    return rows


def excluded_package():
    rows = [
        ["distilbert_ablation_r30_defect", "checkpoints/distilbert_ablation_r30_defect/",
         "EXCLUDED", "CATV padding 伪影修复前的 r30 四臂；论文禁用"],
        ["bert_da", "logs/fed/bert_da/",
         "EXCLUDED", "修复前 r30 链；与 defect 归档同缺陷"],
        ["bert_bp_ref", "logs/fed/bert_bp_ref/",
         "EXCLUDED", "fixed（自循环样本）协议的 BP 参考；已被 bert_bp_ref_cycle 取代"],
        ["opt_legacy_fixed_runs", "logs/fed/zoo_k24_d32*, a01_zoo*, a03_zoo*",
         "EXCLUDED", "旧 fixed（前 2 篇/client）协议；reports/opt 历史图表使用，本包不混入"],
        ["bert_var_a03_r30", "logs/fed/bert_var_a03_r30/",
         "DIAGNOSTIC", "旧 pool 级 VAR；非最终 Method A，不作结果"],
        ["distilbert_vq025_a03_r20_partial", "checkpoints/distilbert_vq025_a03_r20_partial/",
         "EXCLUDED", "旧 total-rate 0.25 口径，r20 partial"],
        ["distilbert_vq050_bf16_a03_r0_partial", "checkpoints/distilbert_vq050_bf16_a03_r0_partial/",
         "EXCLUDED", "旧 total-rate 0.5 口径，仅 r0"],
        ["distilbert_var_catv_vq_a03_r6_partial", "checkpoints/distilbert_var_catv_vq_a03_r6_partial/",
         "EXCLUDED", "组合首次 run 被外部 kill，仅 r0–r5"],
        ["distilbert_vq025_a03_r30", "checkpoints/distilbert_vq025_a03_r30/",
         "EXCLUDED", "空目录"],
        ["distilbert_vq050_bf16_a03_r30", "checkpoints/distilbert_vq050_bf16_a03_r30/",
         "EXCLUDED", "空目录"],
        ["distilbert_var_persample_a03_r30", "checkpoints/distilbert_var_persample_a03_r30/",
         "EXCLUDED", "空目录"],
        ["distilbert_var_catv_vq_a03_r30", "checkpoints/distilbert_var_catv_vq_a03_r30/",
         "EXCLUDED", "空目录；成功 run 在 ..._rerun"],
    ]
    write_csv(DATA_DIR / "excluded_runs.csv",
              ["run_id", "raw_path", "status", "reason"], rows)
    return rows


def check(name, got, want, tol):
    ok = got is not None and abs(float(got) - float(want)) <= tol
    return (name, got, want, tol, ok)


def validation(r30, evaloff, opt, opt_evaloff, opt_baselines,
               bpp, ldp, vq, var, combo, combo_evaloff, tech):
    checks = []

    def ep_endpoint(metrics, field):
        return metrics["nll_last"] if field == "nll_last" else metrics["acc_last"]

    for run_id, want_nll, want_acc in [
        ("r30_a01_catv_off", 1.583558, 0.468),
        ("r30_a01_catv_on", 1.556317, 0.458),
        ("r30_a03_catv_off", 1.588699, 0.466),
        ("r30_a03_catv_on", 1.548441, 0.454),
    ]:
        checks.append(check(f"{run_id} r29 nll", ep_endpoint(r30["endpoints"][run_id], "nll_last"),
                            want_nll, 5e-4))
        checks.append(check(f"{run_id} r29 acc", ep_endpoint(r30["endpoints"][run_id], "acc_last"),
                            want_acc, 1e-6))
    checks.append(check("a01 CATV on-vs-off dNLL",
                        r30["paired"][("catv_on_vs_off_r29", "a01")][8], -0.0272, 5e-4))
    checks.append(check("a03 CATV on-vs-off dNLL",
                        r30["paired"][("catv_on_vs_off_r29", "a03")][8], -0.0403, 5e-4))
    checks.append(check("a01 CATV on-vs-off p",
                        r30["paired"][("catv_on_vs_off_r29", "a01")][13], 0.0304, 2e-3))
    checks.append(check("a03 CATV on-vs-off p",
                        r30["paired"][("catv_on_vs_off_r29", "a03")][13], 0.00253, 5e-4))

    for tag, want_nll, want_acc in [("a01", 1.556317, 0.458), ("a03", 1.548441, 0.454)]:
        checks.append(check(f"{tag} on-evalon r29 nll", evaloff["endpoints"][(tag, "on_evalon")]["nll_r29"],
                            want_nll, 5e-4))
        checks.append(check(f"{tag} on-evaloff r29 nll", evaloff["endpoints"][(tag, "on_evaloff")]["nll_r29"],
                            1.5768 if tag == "a01" else 1.5772, 5e-4))
    checks.append(check("a03 mask effect dNLL",
                        evaloff["paired"][("mask_effect", "a03")][8], -0.0288, 5e-4))
    checks.append(check("a03 weight effect dNLL",
                        evaloff["paired"][("weight_effect", "a03")][8], -0.0115, 5e-4))

    for run_id, full_r0, full_r29, answer_r0, answer_r29 in [
        ("r30_a01_catv_off", 33.525272, 31.533372, 1782.0397, 2436.1133),
        ("r30_a01_catv_on", 33.921101, 31.635054, 1744.9838, 2474.4120),
        ("r30_a03_catv_off", 33.529889, 31.456821, 1765.3404, 2448.8251),
        ("r30_a03_catv_on", 33.939862, 31.695741, 1598.1708, 2291.7071),
    ]:
        metrics = opt["endpoints"][run_id]
        checks.append(check(f"OPT {run_id} full_r0", metrics["full_r0"], full_r0, 5e-4))
        checks.append(check(f"OPT {run_id} full_r29", metrics["full_r29"], full_r29, 5e-4))
        checks.append(check(f"OPT {run_id} answer_r0", metrics["answer_r0"], answer_r0, 0.02))
        checks.append(check(f"OPT {run_id} answer_r29", metrics["answer_r29"], answer_r29, 0.02))
    for tag, want_full, want_answer in [
        ("a01", 0.003219, 0.026221),
        ("a03", 0.007566, -0.093214),
    ]:
        checks.append(check(f"OPT {tag} CATV on-vs-off full",
                            opt["paired"][("catv_on_vs_off_r29", tag, "full_nll_mean")][8],
                            want_full, 5e-5))
        checks.append(check(f"OPT {tag} CATV on-vs-off answer",
                            opt["paired"][("catv_on_vs_off_r29", tag, "answer_nll_mean")][8],
                            want_answer, 5e-5))
    for tag, want_full, want_answer in [("a01", 33.665317, 1740.6320),
                                        ("a03", 33.665317, 1740.6320)]:
        row = opt_baselines[f"fresh_pretrain_{tag}"]
        checks.append(check(f"OPT fresh {tag} full", row[5], want_full, 5e-4))
        checks.append(check(f"OPT fresh {tag} answer", row[6], want_answer, 0.02))
    for tag, label, want_full, want_answer in [
        ("a01", "weight_effect", -0.004937, 0.053245),
        ("a01", "mask_effect", 0.008156, -0.027024),
        ("a01", "deploy_net", 0.003219, 0.026221),
        ("a03", "weight_effect", -0.001283, 0.008884),
        ("a03", "mask_effect", 0.008849, -0.102097),
        ("a03", "deploy_net", 0.007566, -0.093214),
    ]:
        checks.append(check(f"OPT {tag} {label} full",
                            opt_evaloff["paired"][(label, tag, "full_nll_mean")][8],
                            want_full, 5e-5))
        checks.append(check(f"OPT {tag} {label} answer",
                            opt_evaloff["paired"][(label, tag, "answer_nll_mean")][8],
                            want_answer, 5e-5))
    checks.append(check("OPT a03 deploy answer p",
                        opt_evaloff["paired"][("deploy_net", "a03", "answer_nll_mean")][13],
                        5.4e-8, 2e-8))

    for run_id, want_nll, want_acc in [
        ("dense_L8_lr3e4_r10", 1.395443, 0.536),
        ("dense_L8_lr1e3_r10", 1.533402, 0.568),
        ("sparse_L8_lr3e4_r10", 1.461707, 0.508),
        ("dense_L8_lr3e4_r5", 1.472105, 0.520),
        ("dense_L2_lr3e4_r20", 1.506369, 0.512),
    ]:
        checks.append(check(f"BP {run_id} nll_last", bpp[run_id]["nll_last"], want_nll, 5e-4))
        checks.append(check(f"BP {run_id} acc_last", bpp[run_id]["acc_last"], want_acc, 1e-6))

    for run_id, want_nll, want_acc in [
        ("L8D4", 1.728417, 0.432),
        ("L8D8", 1.727120, 0.436),
        ("fixed_L8D8", 1.729784, 0.436),
        ("L16D4", 1.719170, 0.444),
        ("L16D8", 1.711493, 0.448),
        ("L8D16", 1.730032, 0.432),
        ("L8D8_lr2x", 1.711552, 0.440),
        ("dense_L8D8", 1.521502, 0.484),
        ("B1_L8D4_r5", 1.730357, 0.432),
        ("B2_L2D4_r20", 1.731751, 0.432),
    ]:
        checks.append(check(f"LD {run_id} nll_last", ldp[run_id]["nll_last"], want_nll, 5e-4))
        checks.append(check(f"LD {run_id} acc_last", ldp[run_id]["acc_last"], want_acc, 1e-6))

    checks.append(check("VQ r29 nll", vq["nll_last"], 1.587080, 5e-4))
    checks.append(check("VQ r29 acc", vq["acc_last"], 0.466, 1e-6))
    checks.append(check("VQ vs catv-off dNLL",
                        tech["paired"][("vq_vs_catv_off_r29", "a03")][8], -0.001618, 5e-5))
    checks.append(check("VAR r29 nll", var["nll_last"], 1.584451, 5e-4))
    checks.append(check("VAR r29 acc", var["acc_last"], 0.464, 1e-6))
    checks.append(check("VAR vs catv-off dNLL",
                        tech["paired"][("var_vs_catv_off_r29", "a03")][8], -0.004248, 5e-5))
    checks.append(check("combo r29 nll", combo["nll_last"], 1.559838, 5e-4))
    checks.append(check("combo r29 acc", combo["acc_last"], 0.450, 1e-6))
    checks.append(check("combo vs catv-on dNLL",
                        tech["paired"][("combo_vs_catv_on_r29", "a03")][8], 0.011398, 5e-5))
    checks.append(check("combo vs catv-off dNLL",
                        tech["paired"][("combo_vs_catv_off_r29", "a03")][8], -0.028861, 5e-5))
    checks.append(check("combo evaloff r29 nll", combo_evaloff["nll_r29"], 1.577730, 5e-4))
    checks.append(check("combo evaloff r29 acc", combo_evaloff["acc_r29"], 0.464, 1e-6))
    checks.append(check("combo evaloff vs catv-on evaloff dNLL",
                        tech["paired"][("combo_evaloff_vs_catv_on_evaloff_r29", "a03")][8],
                        0.000506, 5e-5))
    checks.append(check("combo mask effect dNLL",
                        tech["paired"][("combo_mask_effect_r29", "a03")][8],
                        -0.017892, 5e-5))
    return checks


def readme_text(file_counts, checks):
    passed = sum(1 for item in checks if item[-1])
    failed = [item for item in checks if not item[-1]]
    lines = []
    lines.append("# BERT + OPT fixed results package (post self-loop fix)")
    lines.append("")
    lines.append("由 `reports/bert_and_other/make_report.py` 从原始 `logs/fed/` 与 `temp/` 产物自动生成；")
    lines.append("纯 CPU、确定性；**不含图片**（论文图由论文 Agent 统一生成）。")
    lines.append("禁止手工改本目录数字；改脚本后重跑。原始 `logs/`、`temp/` 不进 git；")
    lines.append("换机只带 CSV/README 时可直接用，重跑脚本需要原始产物。")
    lines.append("")
    lines.append("## 范围：只收自循环样本 bug 修复后的实验")
    lines.append("")
    lines.append("- 缺陷（261012 前）：`run_fed.py` 每轮从 client 分片第 0 行取 `L` 条；")
    lines.append("  BERT `L=8`、30 轮只重复前 8 篇/人（30 clients 合计 240 个固定 client-doc 位）。")
    lines.append("  见 `docs/ailog/261012-151805-...bert-fixed-60docs-protocol-flaw-and-ld-plan-handoff.md`。")
    lines.append("- 修复：`--doc-order cycle`，`start=(round*local_steps)%总行数`；本包主表全部为 `cycle`。")
    lines.append("- 唯一 `fixed` 行是 `bert_ld_grid.csv` 的 `fixed_L8D8`，作为 H2 语料对照故意保留。")
    lines.append("- 修复前缺陷臂、旧 fixed 协议、partial/诊断 run 均列入 `excluded_runs.csv`，不混入主表。")
    lines.append("- **OPT fixed r30 四臂也纳入本包**（`opt_r30_*`）：cycle、k=24/D=32/L=2/lr=0.30/clip=13.34、")
    lines.append("  eval every 5；`reports/opt/` 保留旧 OPT 图表/长跑，二者口径不同，不要混用。")
    lines.append("")
    lines.append("## 文件索引")
    lines.append("")
    lines.append("| 文件 | 行数 | 作用 | 原始来源 |")
    lines.append("|---|---:|---|---|")
    for name, purpose, source in FILE_NOTES:
        lines.append(f"| `data/{name}` | {file_counts.get(name, 0)} | {purpose} | {source} |")
    lines.append("")
    lines.append("配置（论文侧直接引用）：`BERT_CONFIG.md` 是 BERT r30 四臂的可读配置表（序列长度、LoRA、local steps、lr、数据构造、变体）；")
    lines.append("`bert_config.json` 是同一内容的机器可读版，均由实际 `config.json` + `meta.json` 生成。")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- `nll` = global_test 上的逐样本交叉熵均值（BERT 分类 NLL），不是 PPL；")
    lines.append("  r30 四臂/VQ/VAR/组合用 500 样本；LD/BP 用 250 样本，绝对 NLL 不可直接与 500 样本表混用。")
    lines.append("- `accuracy` = 11 类样本准确率；`macro_f1` = 11 类 macro-F1。")
    lines.append("- `delta_nll_pct = (nll_r0 - nll_last) / nll_r0 * 100`；正值 = 变好。")
    lines.append("- `delta_acc_points = acc_last - acc_r0`（0–1 绝对点）；`delta_acc_pct` 为相对 r0 的百分比。")
    lines.append("- `err_reduction_pct = (1 - (1-acc_last)/(1-acc_r0)) * 100`。")
    lines.append("- 配对检验：`b - a` 的逐样本 NLL；负值 = b 更好。`p_two_sided` 用 scipy t 分布；")
    lines.append("  `p_normal_approx` 用正态近似，和 261015/261021/261022 ailog 的历史 p 值一致。")
    lines.append("  `exp_delta_nll = exp(mean ΔNLL)`，是逐样本似然比，不是 PPL 比。")
    lines.append("  `p=0` 表示 double 下溢（t 很大），不是缺失；OPT 的 full 侧常见。")
    lines.append("- `wall_hours` = `config.json` mtime → 最后一个 `adapter_round*` mtime；含评测时间，")
    lines.append("  只适合同协议内比较。`nll_pct_per_hour = delta_nll_pct / wall_hours`。")
    lines.append("- `adapter_sha256` 是 `adapter_round029/adapter_model.safetensors` 的 sha256。")
    lines.append("- **OPT 口径**：`full_ppl_token` / `answer_ppl_token` 是 token 加权 PPL，不是 BERT 的 NLL/acc；")
    lines.append("  OPT eval 每 5 轮（r4/9/14/19/24/29），r0 是 post-round-0 补测（训练前 fresh = 33.6653/1740.63）；")
    lines.append("  `answer_ppl_token` 随训练通常变差，CATV 的收益在 answer 侧；不要与 BERT 500 样本 NLL 混表。")
    lines.append("  `full_pct_per_hour_r4_r29` 用 `train_hours`（各轮 `round_seconds` 求和），与 ailog 的 %/h 一致。")
    lines.append("- `bert_technique_endpoints.csv` 的 `vq_bytes_total` / `dense_all_bytes_total` 是 30 轮求和；")
    lines.append("  `mean_vq_channels` / `mean_vq_total_ratio` 是 30 轮均值。VQ 无 VAR 时 `vq_channels=64`；")
    lines.append("  组合的 `C_i=sum(D)` 逐 client 变化，因此 `mean_vq_channels≈72`。")
    lines.append("  VAR/VQ/CATV 的 r29 单轮值请取 `bert_var_rounds.csv` / `bert_combo_rounds.csv`；")
    lines.append("  端点表里的 `mean_D`/`mean_vq_*`/`catv_ir_mean` 是 30 轮均值。")
    lines.append("- 技术线端点均为 **own-path**（各自 r0 基线），不是跨臂共享 r0；VQ/VAR 的 r0 也略有差异。")
    lines.append("")
    lines.append("## 关键 caveat")
    lines.append("")
    lines.append("- **CATV on 臂的 r0 已带 mask**（训练第 0 轮无 mask，但 r0 eval 用 round-0 掩码）；")
    lines.append("  off 臂 r0 无 mask。跨臂比较必须用 r29 同轮配对，不要用 r0 跨臂比较。")
    lines.append("- **CATV 归一化不同**：r30 四臂是 `catv_normalize=off`（paper raw summed votes）；")
    lines.append("  组合是 `client`（每 client 每层 L1 等权）。组合 vs CATV-on 基线的 +0.0114 NLL")
    lines.append("  来自 mask 选择/归一化，不是 VAR/VQ 权重；eval-off 证明组合权重与基线等价。")
    lines.append("- **VAR**：raw var 非单调，D=4 假早停多；mean D 会随训练上漂。Method A 的 T10/T20")
    lines.append("  是分段探针阈值，不能跨模型/精度复用。")
    lines.append("- **VQ**：`vq_ratio_bytes` 是单条结果口径；`vq_total_ratio_bytes_vs_one_delta` 是相对")
    lines.append("  “一条 dense delta”的总量（C 倍）。`vq_cos_mean` 是聚合后 cos，不能反推 per-result rate。")
    lines.append("- **LD/BP**：sparse 臂用 `sdpa_prune`，自身基线比 dense 高约 0.18 nats（剪枝惩罚）；")
    lines.append("  `fixed_L8D8` 是 fixed 对照，不是修复后主结果；`B1/B2` 是等 40 篇唯一语料对照。")
    lines.append("- **OPT fixed**：full 侧 CATV 部署净效应略差（a01 +0.0032 / a03 +0.0076 nats，显著但量级小）；")
    lines.append("  answer 侧仅 a03 显著更好（−0.0932，p=5.4e-8），a01 +0.0262（p=0.14 n.s.）。")
    lines.append("  on 臂 r0 已带 mask，跨臂比较用 r29；OPT 结果不与 BERT 的 NLL/acc 混表。")
    lines.append("- **单 seed**：所有 r30 端点都是单次 run，无跨 seed 误差棒。")
    lines.append("")
    lines.append("## 主要端点（r29，500 样本）")
    lines.append("")
    lines.append("| run | NLL r0 → r29 | acc r0 → r29 | 关键配对 |")
    lines.append("|---|---:|---:|---|")
    lines.append("| a03 CATV off | 1.6906 → 1.5887 | 0.426 → 0.466 | baseline |")
    lines.append("| a03 CATV on | 1.7511 → 1.5484 | 0.410 → 0.454 | on vs off ΔNLL −0.0403 (p=0.0025) |")
    lines.append("| a01 CATV off | 1.6910 → 1.5836 | 0.424 → 0.468 | baseline |")
    lines.append("| a01 CATV on | 1.7527 → 1.5563 | 0.408 → 0.458 | on vs off ΔNLL −0.0272 (p=0.030) |")
    lines.append("| VQ 0.5 bf16 | 1.6907 → 1.5871 | 0.426 → 0.466 | vs a03 off ΔNLL −0.0016 (p=0.016) |")
    lines.append("| VAR Method A | 1.6906 → 1.5845 | 0.426 → 0.464 | vs a03 off ΔNLL −0.0042 (p=2.7e-5) |")
    lines.append("| combo | 1.7425 → 1.5598 | 0.410 → 0.450 | vs a03 on ΔNLL +0.0114；vs a03 off −0.0289 |")
    lines.append("| combo eval-off | 1.6904 → 1.5777 | 0.426 → 0.464 | vs a03 on eval-off ΔNLL +0.0005 (p=0.735) |")
    lines.append("")
    lines.append("表中 p 值用历史 ailog 的正态近似；CSV 同时给 `p_two_sided`（scipy t）与 `p_normal_approx`。")
    lines.append("")
    lines.append("## OPT fixed r30 主要端点（PPL，500 样本）")
    lines.append("")
    lines.append("| run | full r0 → r29 | answer r0 → r29 | 关键配对 |")
    lines.append("|---|---:|---:|---|")
    lines.append("| a01 CATV off | 33.525 → 31.533 | 1782 → 2436 | baseline |")
    lines.append("| a01 CATV on | 33.921 → 31.635 | 1745 → 2474 | on vs off full +0.0032（p≈0）；answer +0.0262（p=0.14 n.s.） |")
    lines.append("| a03 CATV off | 33.530 → 31.457 | 1765 → 2449 | baseline |")
    lines.append("| a03 CATV on | 33.940 → 31.696 | 1598 → 2292 | on vs off full +0.0076（p≈0）；answer −0.0932（p=5.4e-8） |")
    lines.append("")
    lines.append("OPT 的 `own%` 起点是 r4（首个 in-run eval）；vs fresh 用训练前 33.6653/1740.63。")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("PY=~/miniconda3/envs/SMELL/bin/python")
    lines.append("$PY reports/bert_and_other/make_report.py            # 全量 + 对账闸门")
    lines.append("$PY reports/bert_and_other/make_report.py --no-check # 仅调试")
    lines.append("```")
    lines.append("")
    lines.append("## 对账结果")
    lines.append("")
    lines.append(f"- 断言：{passed}/{len(checks)} PASS")
    if failed:
        lines.append("- FAIL：")
        for name, got, want, tol, _ok in failed:
            lines.append(f"  - {name}: got {got} want {want} (tol {tol})")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Build the BERT + OPT fixed post-cycle CSV package")
    parser.add_argument("--out", default=str(OUT_ROOT), help="output directory")
    parser.add_argument("--no-check", action="store_true", help="skip ailog cross-checks")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out)
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    global DATA_DIR
    DATA_DIR = data_dir

    r30 = r30_package()
    evaloff = r30_evaloff_package(r30)
    opt = opt_package()
    opt_evaloff = opt_evaloff_package()
    opt_baselines = opt_baselines_package()
    bpp = bp_package()
    ldp = ld_package()
    lr_scan = lr_scan_package()
    vq, _vq_info = vq_package()
    var, _var_info = var_package()
    combo, _combo_info = combo_package()
    combo_evaloff = combo_evaloff_package()
    tech = technique_package(vq, var, combo, combo_evaloff, r30)
    baselines_package()
    bert_config_package(out_root)
    provenance_package(r30, bpp, ldp, lr_scan, opt)
    excluded_package()

    checks = validation(r30, evaloff, opt, opt_evaloff, opt_baselines,
                        bpp, ldp, vq, var, combo, combo_evaloff, tech)
    print("== ailog cross-checks ==")
    bad = 0
    for name, got, want, tol, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:42s} got={got} want={want} tol={tol}")
        bad += 0 if ok else 1
    if bad and not args.no_check:
        raise SystemExit(f"{bad} cross-checks FAILED; refusing to write README (use --no-check to override)")

    file_counts = {}
    for name, _purpose, _source in FILE_NOTES:
        path = data_dir / name
        file_counts[name] = max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)
    (out_root / "README.md").write_text(readme_text(file_counts, checks), encoding="utf-8")
    print(f"== done: {out_root} ==")
    print(f"== checks: {len(checks) - bad}/{len(checks)} PASS ==")


if __name__ == "__main__":
    main()
