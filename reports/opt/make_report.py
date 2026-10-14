# SMELL 3 opt_report NEW — OPT-350M 实验结果出图/表格（纯 CPU；读 logs/fed 与 temp 原始产物）
"""One-command builder for the OPT-350M result package.

Reads raw artifacts only:
  - logs/fed/<run>/<tag>/metrics.jsonl        (per-round eval events: full/answer G-PPL)
  - temp/eval_*.json                          (post-hoc cross-protocol evals + per-sample)
  - checkpoints/posemb_step1/MANIFEST.json    (position adaptation B/C)
  - temp/k_scan2/, temp/pilot_micro_scan/     (coverage micro-scans)

Writes to reports/opt/: figures/ (pdf+png), tables/ (md+tex), data/ (csv), README.md.
All headline numbers are checked against the values published in docs/ailog/ (--no-check
skips; a failing check aborts with a non-zero exit code).

Usage:
  PY=~/miniconda3/envs/SMELL/bin/python
  $PY reports/opt/make_report.py                  # full package + validation
  $PY reports/opt/make_report.py --no-check       # skip validation gate
  $PY reports/opt/make_report.py --only figures
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from paired_eval import load_rows, paired_stats  # noqa: E402


# ---------------------------------------------------------------- loading

def metrics_path(run, tag):
    return REPO / "logs" / "fed" / run / tag / "metrics.jsonl"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def round_evals(run, tag, offset=0):
    """{round: (full_ppl_token, answer_ppl_token)} from in-run eval events."""
    out = {}
    path = metrics_path(run, tag)
    if not path.exists():
        raise SystemExit(f"missing metrics: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "eval" or row.get("status") != "ok":
            continue
        if row.get("round") is None:
            continue
        out[int(row["round"]) + offset] = (
            float(row["full_ppl_token"]),
            float(row["answer_ppl_token"]),
        )
    return out


def round_records(run, tag):
    out = []
    path = metrics_path(run, tag)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "round":
            out.append(row)
    return out


def load_round_rows(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "round":
            out.append(row)
    if not out:
        raise SystemExit(f"no round rows in {path}")
    return out


def xeval(path):
    data = load_json(REPO / path)
    return float(data["full_ppl_token"]), float(data["answer_ppl_token"])


def xeval_map(pairs, offset=0):
    """pairs: [(round, 'temp/eval_*.json')] -> {round+offset: (full, answer)}"""
    return {int(r) + offset: xeval(p) for r, p in pairs}


def block_span_hours(run, tag):
    """Wall clock of one run block: config.json mtime -> last adapter_round mtime."""
    base = REPO / "logs" / "fed" / run / tag
    cfg = base / "config.json"
    adapters = sorted(
        [p for p in base.glob("adapter_round*") if p.is_dir()],
        key=lambda p: int(p.name.split("round")[-1]),
    )
    if not cfg.exists() or not adapters:
        raise SystemExit(f"cannot measure wall clock for {run}/{tag}")
    return (adapters[-1].stat().st_mtime - cfg.stat().st_mtime) / 3600.0


def block_hours_at(run, tag, local_round, base_hours=0.0):
    base = REPO / "logs" / "fed" / run / tag
    cfg = base / "config.json"
    adapter = base / f"adapter_round{local_round:03d}"
    if not adapter.is_dir():
        raise SystemExit(f"missing adapter dir: {adapter}")
    return base_hours + (adapter.stat().st_mtime - cfg.stat().st_mtime) / 3600.0


def chain_hours(blocks):
    """{global_round: cumulative wall hours} for a chain of run blocks."""
    out = {}
    base = 0.0
    for block in blocks:
        run, tag, offset = block["run"], block["tag"], block["offset"]
        for gr in round_evals(run, tag, offset):
            out[gr] = block_hours_at(run, tag, gr - offset, base_hours=base)
        base += block_span_hours(run, tag)
    return out


def chain_round_records(blocks):
    rows = []
    for block in blocks:
        rows += round_records(block["run"], block["tag"])
    return rows


def run_block(run, tag, offset=0):
    return {"run": run, "tag": tag, "offset": offset}


# ---------------------------------------------------------------- arm definitions

A03 = "a03"
A01 = "a01"

K4_BLOCKS = [run_block("zoo_k4_sparse_causal_a03_lr0.15_clip", A03)]
D16_BLOCKS = [
    run_block("zoo_k24_d16_lr0.30_r30", A03),
    run_block("zoo_k24_d16_r60_cont_lr0.10", A03, offset=30),
]
D16_MAIN = [run_block("zoo_k24_d16_lr0.30_r30", A03)]
D32_BLOCKS = [
    run_block("zoo_k24_d32_lr0.30_r30", A03),
    run_block("a03_zoo_k24_d32_lr0.10_cont15", A03, offset=30),
    run_block("a03_zoo_k24_d32_lr0.10_cont15b", A03, offset=45),
    run_block("a03_zoo_k24_d32_lr0.10_cont15c", A03, offset=60),
]
D32_MAIN = [run_block("zoo_k24_d32_lr0.30_r30", A03)]
D32_CATV_BLOCKS = [
    run_block("zoo_k24_d32_catv_lr0.30_r30", A03),
    run_block("a03_zoo_k24_d32_catv_lr0.10_cont15", A03, offset=30),
    run_block("a03_zoo_k24_d32_catv_lr0.10_cont15b", A03, offset=45),
    run_block("a03_zoo_k24_d32_catv_lr0.10_cont15c", A03, offset=60),
]
D32_CATV_MAIN = [run_block("zoo_k24_d32_catv_lr0.30_r30", A03)]
A01_D32_BLOCKS = [
    run_block("a01_zoo_k24_d32_lr0.30_r30", A01),
    run_block("a01_zoo_k24_d32_lr0.10_cont15", A01, offset=30),
]
A01_D32_MAIN = [run_block("a01_zoo_k24_d32_lr0.30_r30", A01)]
A01_CATV_BLOCKS = [
    run_block("a01_zoo_k24_d32_catv_lr0.30_r30", A01),
    run_block("a01_zoo_k24_d32_catv_lr0.10_cont15", A01, offset=30),
]
A01_CATV_MAIN = [run_block("a01_zoo_k24_d32_catv_lr0.30_r30", A01)]

R0_A03_SAME = ("temp/eval_shared_r0_nomask.json", (33.5296, 1766.4))
R0_A03_DEPL = ("temp/eval_shared_r0_mask0.json", (33.9393, 1598.9))
R0_A01_SAME = ("temp/eval_a01_shared_r0_nomask.json", (33.5253, 1782.0))
R0_A01_DEPL = ("temp/eval_a01_shared_r0_mask0.json", (33.9211, 1745.0))
R0_D16 = ("temp/eval_run1_r0.json", (33.5332, 1768.8))

A03_CATV_SAME_MAIN = [
    (4, "temp/eval_a03_catv_r4_nomask.json"),
    (9, "temp/eval_a03_catv_r9_nomask.json"),
    (14, "temp/eval_a03_catv_r14_nomask.json"),
    (19, "temp/eval_a03_catv_r19_nomask.json"),
    (24, "temp/eval_a03_catv_r24_nomask.json"),
    (29, "temp/eval_catv_r29_nomask.json"),
]
A03_CATV_SAME_CONT = [
    (4, "temp/eval_a03_catv_cont_c4_nomask.json"),
    (9, "temp/eval_a03_catv_cont_c9_nomask.json"),
    (14, "temp/eval_a03_catv_cont_c14_nomask.json"),
    (19, "temp/eval_a03_catv_b2_c4_nomask.json"),
    (24, "temp/eval_a03_catv_b2_c9_nomask.json"),
    (29, "temp/eval_a03_catv_b2_c14_nomask.json"),
    (34, "temp/eval_a03_catv_b3_c4_nomask.json"),
    (39, "temp/eval_a03_catv_b3_c9_nomask.json"),
    (44, "temp/eval_a03_catv_b3_c14_nomask.json"),
]
A01_CATV_SAME_MAIN = [
    (4, "temp/eval_a01_catv_r4_nomask.json"),
    (9, "temp/eval_a01_catv_r9_nomask.json"),
    (14, "temp/eval_a01_catv_r14_nomask.json"),
    (19, "temp/eval_a01_catv_r19_nomask.json"),
    (24, "temp/eval_a01_catv_r24_nomask.json"),
    (29, "temp/eval_a01_catv_r29_nomask.json"),
]
A01_CATV_SAME_CONT = [
    (4, "temp/eval_a01_catv_cont_c4_nomask.json"),
    (9, "temp/eval_a01_catv_cont_c9_nomask.json"),
    (14, "temp/eval_a01_catv_cont_c14_nomask.json"),
]


def merge_points(*parts):
    pts = {}
    for part in parts:
        for key, value in part.items():
            pts[int(key)] = value
    return dict(sorted(pts.items()))


def arm_summary(name, points, hours, records):
    rounds = sorted(points)
    full0, ans0 = points[rounds[0]]
    fullN, ansN = points[rounds[-1]]
    span = hours[rounds[-1]] if rounds[-1] in hours else 0.0
    clip_total = sum(int(r.get("clipped_clients") or 0) for r in records)
    sec = statistics.mean([r.get("round_seconds", 0.0) for r in records]) if records else 0.0
    delta_full = full0 - fullN
    return {
        "name": name,
        "rounds": rounds,
        "points": points,
        "hours": hours,
        "hours_total": span,
        "full0": full0, "fullN": fullN, "ans0": ans0, "ansN": ansN,
        "d_full_pct": (fullN / full0 - 1.0) * 100.0,
        "d_ans_pct": (ansN / ans0 - 1.0) * 100.0,
        "rate": (delta_full / span) if span > 0 else float("nan"),
        "sec_per_round": sec,
        "clip_total": clip_total,
        "n_rounds": len(records),
    }


def build_arms():
    arms = {}

    pts = merge_points(round_evals("zoo_k4_sparse_causal_a03_lr0.15_clip", A03))
    hrs = chain_hours(K4_BLOCKS)
    arms["k4"] = arm_summary("ZOO k=4/D=22 (TD-3)", pts, hrs, chain_round_records(K4_BLOCKS))

    pts = merge_points(
        {0: xeval(REPO / R0_D16[0])},
        round_evals("zoo_k24_d16_lr0.30_r30", A03),
        round_evals("zoo_k24_d16_r60_cont_lr0.10", A03, offset=30),
    )
    hrs = chain_hours(D16_BLOCKS)
    hrs[0] = 0.0
    arms["d16_60"] = arm_summary("ZOO k=24/D=16 (60 r)", pts, hrs, chain_round_records(D16_BLOCKS))
    arms["d16_30"] = arm_summary(
        "ZOO k=24/D=16 (30 r)",
        merge_points({0: xeval(REPO / R0_D16[0])}, round_evals("zoo_k24_d16_lr0.30_r30", A03)),
        {0: 0.0, **chain_hours(D16_MAIN)},
        chain_round_records(D16_MAIN),
    )

    pts = merge_points(
        {0: xeval(REPO / R0_A03_SAME[0])},
        round_evals("zoo_k24_d32_lr0.30_r30", A03),
        round_evals("a03_zoo_k24_d32_lr0.10_cont15", A03, offset=30),
        round_evals("a03_zoo_k24_d32_lr0.10_cont15b", A03, offset=45),
        round_evals("a03_zoo_k24_d32_lr0.10_cont15c", A03, offset=60),
    )
    hrs = chain_hours(D32_BLOCKS)
    hrs[0] = 0.0
    arms["d32_74"] = arm_summary("ZOO k=24/D=32 (74 r)", pts, hrs, chain_round_records(D32_BLOCKS))
    arms["d32_30"] = arm_summary(
        "ZOO k=24/D=32 (30 r)",
        merge_points({0: xeval(REPO / R0_A03_SAME[0])}, round_evals("zoo_k24_d32_lr0.30_r30", A03)),
        {0: 0.0, **chain_hours(D32_MAIN)},
        chain_round_records(D32_MAIN),
    )

    pts = merge_points(
        {0: xeval(REPO / R0_A03_SAME[0])},
        xeval_map(A03_CATV_SAME_MAIN),
        xeval_map(A03_CATV_SAME_CONT, offset=30),
    )
    hrs = {0: 0.0, **chain_hours(D32_CATV_BLOCKS)}
    arms["d32_catv_74"] = arm_summary("ZOO k=24/D=32+CATV (74 r)", pts, hrs, chain_round_records(D32_CATV_BLOCKS))
    arms["d32_catv_30"] = arm_summary(
        "ZOO k=24/D=32+CATV (30 r)",
        merge_points({0: xeval(REPO / R0_A03_SAME[0])}, xeval_map(A03_CATV_SAME_MAIN)),
        {0: 0.0, **chain_hours(D32_CATV_MAIN)},
        chain_round_records(D32_CATV_MAIN),
    )

    pts = merge_points(
        {0: xeval(REPO / R0_A01_SAME[0])},
        round_evals("a01_zoo_k24_d32_lr0.30_r30", A01),
        round_evals("a01_zoo_k24_d32_lr0.10_cont15", A01, offset=30),
    )
    hrs = chain_hours(A01_D32_BLOCKS)
    hrs[0] = 0.0
    arms["a01_d32_44"] = arm_summary("ZOO k=24/D=32 a01 (44 r)", pts, hrs, chain_round_records(A01_D32_BLOCKS))
    arms["a01_d32_30"] = arm_summary(
        "ZOO k=24/D=32 a01 (30 r)",
        merge_points({0: xeval(REPO / R0_A01_SAME[0])}, round_evals("a01_zoo_k24_d32_lr0.30_r30", A01)),
        {0: 0.0, **chain_hours(A01_D32_MAIN)},
        chain_round_records(A01_D32_MAIN),
    )

    pts = merge_points(
        {0: xeval(REPO / R0_A01_SAME[0])},
        xeval_map(A01_CATV_SAME_MAIN),
        xeval_map(A01_CATV_SAME_CONT, offset=30),
    )
    hrs = {0: 0.0, **chain_hours(A01_CATV_BLOCKS)}
    arms["a01_catv_44"] = arm_summary("ZOO k=24/D=32+CATV a01 (44 r)", pts, hrs, chain_round_records(A01_CATV_BLOCKS))
    arms["a01_catv_30"] = arm_summary(
        "ZOO k=24/D=32+CATV a01 (30 r)",
        merge_points({0: xeval(REPO / R0_A01_SAME[0])}, xeval_map(A01_CATV_SAME_MAIN)),
        {0: 0.0, **chain_hours(A01_CATV_MAIN)},
        chain_round_records(A01_CATV_MAIN),
    )

    pts = merge_points(
        {0: xeval(REPO / R0_A03_DEPL[0])},
        round_evals("zoo_k24_d32_catv_lr0.30_r30", A03),
        round_evals("a03_zoo_k24_d32_catv_lr0.10_cont15", A03, offset=30),
        round_evals("a03_zoo_k24_d32_catv_lr0.10_cont15b", A03, offset=45),
        round_evals("a03_zoo_k24_d32_catv_lr0.10_cont15c", A03, offset=60),
    )
    hrs = {0: 0.0, **chain_hours(D32_CATV_BLOCKS)}
    arms["a03_catv_depl"] = arm_summary("a03 CATV deployment", pts, hrs, chain_round_records(D32_CATV_BLOCKS))

    pts = merge_points(
        {0: xeval(REPO / R0_A01_DEPL[0])},
        round_evals("a01_zoo_k24_d32_catv_lr0.30_r30", A01),
        round_evals("a01_zoo_k24_d32_catv_lr0.10_cont15", A01, offset=30),
    )
    hrs = {0: 0.0, **chain_hours(A01_CATV_BLOCKS)}
    arms["a01_catv_depl"] = arm_summary("a01 CATV deployment", pts, hrs, chain_round_records(A01_CATV_BLOCKS))

    return arms


# ---------------------------------------------------------------- validation

EXPECTED = [
    ("k4 r29 full", lambda a: a["k4"]["points"][29][0], 33.2851, 2e-3),
    ("k4 r29 answer", lambda a: a["k4"]["points"][29][1], 1855.0, 0.5),
    ("k4 r0 full", lambda a: a["k4"]["points"][0][0], 33.6538, 2e-3),
    ("d16 r29 full", lambda a: a["d16_30"]["points"][29][0], 32.2595, 2e-3),
    ("d16 r59 full", lambda a: a["d16_60"]["points"][59][0], 31.8292, 2e-3),
    ("d16 r0 full", lambda a: a["d16_60"]["points"][0][0], 33.5332, 2e-3),
    ("d32 r29 full", lambda a: a["d32_30"]["points"][29][0], 31.4103, 2e-3),
    ("d32 r0 full", lambda a: a["d32_74"]["points"][0][0], 33.5296, 2e-3),
    ("d32 r44 full", lambda a: a["d32_74"]["points"][44][0], 31.1189, 2e-3),
    ("d32 r59 full", lambda a: a["d32_74"]["points"][59][0], 30.9644, 2e-3),
    ("d32 r74 full", lambda a: a["d32_74"]["points"][74][0], 30.8090, 2e-3),
    ("d32 r74 answer", lambda a: a["d32_74"]["points"][74][1], 2201.7, 0.5),
    ("d32+catv same r29 full", lambda a: a["d32_catv_30"]["points"][29][0], 31.2223, 2e-3),
    ("d32+catv same r29 answer", lambda a: a["d32_catv_30"]["points"][29][1], 2404.0, 0.5),
    ("d32+catv same r74 full", lambda a: a["d32_catv_74"]["points"][74][0], 30.6095, 2e-3),
    ("d32+catv same r74 answer", lambda a: a["d32_catv_74"]["points"][74][1], 2468.4, 0.5),
    ("d32+catv depl r0 full", lambda a: a["a03_catv_depl"]["points"][0][0], 33.9393, 2e-3),
    ("d32+catv depl r29 full", lambda a: a["a03_catv_depl"]["points"][29][0], 31.4847, 2e-3),
    ("d32+catv depl r69 full", lambda a: a["a03_catv_depl"]["points"][69][0], 30.8527, 2e-3),
    ("d32+catv depl r74 full", lambda a: a["a03_catv_depl"]["points"][74][0], 30.8118, 2e-3),
    ("a01 d32 r29 full", lambda a: a["a01_d32_30"]["points"][29][0], 31.4668, 2e-3),
    ("a01 d32 r44 full", lambda a: a["a01_d32_44"]["points"][44][0], 31.1653, 2e-3),
    ("a01 catv same r29 full", lambda a: a["a01_catv_30"]["points"][29][0], 31.4666, 2e-3),
    ("a01 catv same r44 full", lambda a: a["a01_catv_44"]["points"][44][0], 31.1572, 2e-3),
    ("a01 catv depl r29 full", lambda a: a["a01_catv_depl"]["points"][29][0], 31.6960, 2e-3),
]


def d_checks():
    checks = []
    specs = [
        ("D16 r7 full", "temp/pilot_k24/K24_D16_lr0.30_c19.41/a03/eval_round007.json", 32.7144),
        ("D22 r7 full", "temp/pilot_k24/K24_D22_lr0.30_c16.08/a03/eval_round007.json", 32.5335),
        ("D32 r0 full", "temp/eval_shared_r0_nomask.json", 33.5296),
        ("D32 r7 full", "temp/eval_d32_r7.json", 32.5071),
        ("D32 r8 full", "temp/eval_d32_r8.json", 32.4151),
        ("D32 r9 control", "temp/eval_d32_r9.json", 32.3408),
        ("fresh baseline full", "temp/eval_fresh_baseline.json", 33.6653),
        ("dense r0 full", "temp/diag_dense_eval/r000.json", 26.4667),
        ("dense r29 full", "temp/diag_dense_eval/r029.json", 26.0624),
        ("sparse TD-3 r0 full", "logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/eval_round000.json", 33.6538),
        ("sparse TD-3 r29 full", "logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/eval_round029.json", 33.2851),
    ]
    for label, path, expected in specs:
        if (REPO / path).exists():
            got = xeval(REPO / path)[0]
            checks.append((label, got, expected, 2e-3, abs(got - expected) <= 2e-3))
    return checks


def run_validation(arms):
    results = []
    for name, getter, expected, tol in EXPECTED:
        got = getter(arms)
        ok = abs(got - expected) <= tol
        results.append((name, got, expected, tol, ok))
    return results


# ---------------------------------------------------------------- paired helpers

PERSAMPLE = {
    "a03_off_r0": {
        "summary": "temp/eval_shared_r0_nomask.json",
        "rows": "temp/eval_shared_r0_nomask_persample.json"},
    "a03_mask_r0": {
        "summary": "temp/eval_shared_r0_mask0.json",
        "rows": "temp/eval_shared_r0_mask0_persample.json"},
    "a03_off_r29": {
        "summary": "logs/fed/zoo_k24_d32_lr0.30_r30/a03/eval_round029.json",
        "rows": "logs/fed/zoo_k24_d32_lr0.30_r30/a03/eval_round029_persample.json"},
    "a03_off_r44": {
        "summary": "logs/fed/a03_zoo_k24_d32_lr0.10_cont15/a03/eval_round014.json",
        "rows": "logs/fed/a03_zoo_k24_d32_lr0.10_cont15/a03/eval_round014_persample.json"},
    "a03_off_r59": {
        "summary": "logs/fed/a03_zoo_k24_d32_lr0.10_cont15b/a03/eval_round014.json",
        "rows": "logs/fed/a03_zoo_k24_d32_lr0.10_cont15b/a03/eval_round014_persample.json"},
    "a03_off_r74": {
        "summary": "logs/fed/a03_zoo_k24_d32_lr0.10_cont15c/a03/eval_round014.json",
        "rows": "logs/fed/a03_zoo_k24_d32_lr0.10_cont15c/a03/eval_round014_persample.json"},
    "a03_on_same_r29": {
        "summary": "temp/eval_catv_r29_nomask.json",
        "rows": "temp/eval_catv_r29_nomask_persample.json"},
    "a03_on_same_r44": {
        "summary": "temp/eval_a03_catv_cont_c14_nomask.json",
        "rows": "temp/eval_a03_catv_cont_c14_nomask_persample.json"},
    "a03_on_same_r59": {
        "summary": "temp/eval_a03_catv_b2_c14_nomask.json",
        "rows": "temp/eval_a03_catv_b2_c14_nomask_persample.json"},
    "a03_on_same_r74": {
        "summary": "temp/eval_a03_catv_b3_c14_nomask.json",
        "rows": "temp/eval_a03_catv_b3_c14_nomask_persample.json"},
    "a03_on_depl_r29": {
        "summary": "logs/fed/zoo_k24_d32_catv_lr0.30_r30/a03/eval_round029.json",
        "rows": "logs/fed/zoo_k24_d32_catv_lr0.30_r30/a03/eval_round029_persample.json"},
    "a03_on_depl_r44": {
        "summary": "logs/fed/a03_zoo_k24_d32_catv_lr0.10_cont15/a03/eval_round014.json",
        "rows": "logs/fed/a03_zoo_k24_d32_catv_lr0.10_cont15/a03/eval_round014_persample.json"},
    "a03_on_depl_r59": {
        "summary": "logs/fed/a03_zoo_k24_d32_catv_lr0.10_cont15b/a03/eval_round014.json",
        "rows": "logs/fed/a03_zoo_k24_d32_catv_lr0.10_cont15b/a03/eval_round014_persample.json"},
    "a03_on_depl_r74": {
        "summary": "logs/fed/a03_zoo_k24_d32_catv_lr0.10_cont15c/a03/eval_round014.json",
        "rows": "logs/fed/a03_zoo_k24_d32_catv_lr0.10_cont15c/a03/eval_round014_persample.json"},
    "a01_off_r0": {
        "summary": "temp/eval_a01_shared_r0_nomask.json",
        "rows": "temp/eval_a01_shared_r0_nomask_persample.json"},
    "a01_mask_r0": {
        "summary": "temp/eval_a01_shared_r0_mask0.json",
        "rows": "temp/eval_a01_shared_r0_mask0_persample.json"},
    "a01_off_r29": {
        "summary": "logs/fed/a01_zoo_k24_d32_lr0.30_r30/a01/eval_round029.json",
        "rows": "logs/fed/a01_zoo_k24_d32_lr0.30_r30/a01/eval_round029_persample.json"},
    "a01_off_r44": {
        "summary": "logs/fed/a01_zoo_k24_d32_lr0.10_cont15/a01/eval_round014.json",
        "rows": "logs/fed/a01_zoo_k24_d32_lr0.10_cont15/a01/eval_round014_persample.json"},
    "a01_on_same_r29": {
        "summary": "temp/eval_a01_catv_r29_nomask.json",
        "rows": "temp/eval_a01_catv_r29_nomask_persample.json"},
    "a01_on_same_r44": {
        "summary": "temp/eval_a01_catv_cont_c14_nomask.json",
        "rows": "temp/eval_a01_catv_cont_c14_nomask_persample.json"},
    "a01_on_depl_r29": {
        "summary": "logs/fed/a01_zoo_k24_d32_catv_lr0.30_r30/a01/eval_round029.json",
        "rows": "logs/fed/a01_zoo_k24_d32_catv_lr0.30_r30/a01/eval_round029_persample.json"},
    "a01_on_depl_r44": {
        "summary": "logs/fed/a01_zoo_k24_d32_catv_lr0.10_cont15/a01/eval_round014.json",
        "rows": "logs/fed/a01_zoo_k24_d32_catv_lr0.10_cont15/a01/eval_round014_persample.json"},
}


def key_ppl(key):
    return xeval(REPO / PERSAMPLE[key]["summary"])


def paired(key_a, key_b, field):
    path_a = REPO / PERSAMPLE[key_a]["rows"]
    path_b = REPO / PERSAMPLE[key_b]["rows"]
    return paired_stats(load_rows(path_a), load_rows(path_b), field)


def catv_rows():
    rows = []
    cases = [
        ("a01", "r0", "anchor", "a01_off_r0", "a01_off_r0"),
        ("a01", "r0", "mask-effect", "a01_off_r0", "a01_mask_r0"),
        ("a01", "r29", "same", "a01_off_r29", "a01_on_same_r29"),
        ("a01", "r29", "deploy", "a01_off_r29", "a01_on_depl_r29"),
        ("a01", "r44", "same", "a01_off_r44", "a01_on_same_r44"),
        ("a01", "r44", "deploy", "a01_off_r44", "a01_on_depl_r44"),
        ("a03", "r0", "anchor", "a03_off_r0", "a03_off_r0"),
        ("a03", "r0", "mask-effect", "a03_off_r0", "a03_mask_r0"),
        ("a03", "r29", "same", "a03_off_r29", "a03_on_same_r29"),
        ("a03", "r29", "deploy", "a03_off_r29", "a03_on_depl_r29"),
        ("a03", "r44", "same", "a03_off_r44", "a03_on_same_r44"),
        ("a03", "r44", "deploy", "a03_off_r44", "a03_on_depl_r44"),
        ("a03", "r59", "same", "a03_off_r59", "a03_on_same_r59"),
        ("a03", "r59", "deploy", "a03_off_r59", "a03_on_depl_r59"),
        ("a03", "r74", "same", "a03_off_r74", "a03_on_same_r74"),
        ("a03", "r74", "deploy", "a03_off_r74", "a03_on_depl_r74"),
    ]
    for tag, rnd, proto, key_off, key_on in cases:
        off_full, off_ans = key_ppl(key_off)
        on_full, on_ans = key_ppl(key_on)
        full_stats = paired(key_off, key_on, "full_nll_mean")
        ans_stats = paired(key_off, key_on, "answer_nll_mean")
        rows.append({
            "tag": tag, "round": rnd, "protocol": proto,
            "off_full": off_full, "on_full": on_full,
            "full_ratio": full_stats["ppl_ratio_new_over_old"], "full_p": full_stats["p_two_sided"],
            "off_ans": off_ans, "on_ans": on_ans,
            "ans_ratio": ans_stats["ppl_ratio_new_over_old"], "ans_p": ans_stats["p_two_sided"],
        })
    return rows


CATV_CHECKS = [
    ("a03 r29 same full ratio", 0.9940, 5e-4),
    ("a03 r29 same answer ratio", 0.8313, 5e-4),
    ("a03 r29 deploy full ratio", 1.0024, 5e-4),
    ("a03 r74 same full ratio", 0.9935, 5e-4),
    ("a03 r74 deploy full ratio", 1.0001, 5e-4),
    ("a01 r29 same full ratio", 1.0000, 5e-4),
    ("a01 r44 same full ratio", 0.9997, 5e-4),
]


# ---------------------------------------------------------------- figures

PALETTE = {
    "k4": "#1f77b4",
    "d16": "#ff7f0e",
    "d32": "#2ca02c",
    "catv": "#d62728",
    "gray": "#555555",
    "violet": "#9467bd",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def plot_series(ax, arm, metric, label, color, style="-", marker="o"):
    xs = sorted(arm["points"])
    ys = [arm["points"][r][metric] for r in xs]
    ax.plot(xs, ys, style, marker=marker, ms=3.0, lw=1.2, color=color, label=label)


def save_fig(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = outdir / f"{stem}.{ext}"
        if ext == "pdf":
            fig.savefig(path, dpi=300, metadata={"CreationDate": None})
        else:
            fig.savefig(path, dpi=300)
    return [outdir / f"{stem}.pdf", outdir / f"{stem}.png"]


def fig1_convergence(arms, figdir):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    entries = [
        (arms["k4"], "k=4/D=22 (TD-3)", PALETTE["k4"], "-", "o"),
        (arms["d16_60"], "k=24/D=16 (60 r)", PALETTE["d16"], "-", "s"),
        (arms["d32_74"], "k=24/D=32 (74 r)", PALETTE["d32"], "-", "^"),
        (arms["d32_catv_74"], "k=24/D=32+CATV (74 r)", PALETTE["catv"], "--", "D"),
    ]
    for metric, ax, ylab in [(0, axes[0], "Full G-PPL"), (1, axes[1], "Answer G-PPL")]:
        for arm, label, color, style, marker in entries:
            plot_series(ax, arm, metric, label, color, style, marker)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_xlim(0, 76)
    axes[1].set_xlim(0, 76)
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    paths = save_fig(fig, figdir, "fig1_a03_convergence")
    plt.close(fig)
    return paths


def fig2_catv(arms, figdir):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4))
    panels = [
        (0, 0, "a03 (alpha=0.3)", "full"),
        (0, 1, "a03 (alpha=0.3)", "answer"),
        (1, 0, "a01 (alpha=0.1)", "full"),
        (1, 1, "a01 (alpha=0.1)", "answer"),
    ]
    for row, col, tag, metric in panels:
        ax = axes[row][col]
        m = 0 if metric == "full" else 1
        if tag.startswith("a03"):
            off, on_same, on_depl = arms["d32_74"], arms["d32_catv_74"], arms["a03_catv_depl"]
        else:
            off, on_same, on_depl = arms["a01_d32_44"], arms["a01_catv_44"], arms["a01_catv_depl"]
        plot_series(ax, off, m, "CATV off", PALETTE["gray"], "-", "o")
        plot_series(ax, on_same, m, "CATV on (same protocol)", PALETTE["catv"], "--", "s")
        plot_series(ax, on_depl, m, "CATV on (deployment)", PALETTE["violet"], ":", "^")
        ax.set_xlabel("Round")
        ax.set_ylabel("Full G-PPL" if metric == "full" else "Answer G-PPL")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, 76 if tag.startswith("a03") else 48)
        ax.set_title(tag, loc="left", fontweight="bold")
        if row == 0 and col == 0:
            ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    paths = save_fig(fig, figdir, "fig2_catv_alpha")
    plt.close(fig)
    return paths


def fig3_efficiency(arms, figdir):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.35, 1.0]})
    entries = [
        (arms["k4"], "k=4/D=22 (30 r)", PALETTE["k4"]),
        (arms["d16_60"], "k=24/D=16 (60 r)", PALETTE["d16"]),
        (arms["d32_74"], "k=24/D=32 (74 r)", PALETTE["d32"]),
        (arms["d32_catv_74"], "k=24/D=32+CATV (74 r)", PALETTE["catv"]),
        (arms["a01_d32_44"], "a01: k=24/D=32 (44 r)", PALETTE["violet"]),
    ]
    ax = axes[0]
    for arm, label, color in entries:
        xs = sorted(arm["points"])
        ys = [arm["points"][r][0] for r in xs]
        hs = [arm["hours"].get(r, float("nan")) for r in xs]
        ax.plot(hs, ys, marker="o", ms=3.0, lw=1.2, color=color, label=label)
    ax.set_xlabel("Wall-clock hours (measured)")
    ax.set_ylabel("Full G-PPL")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    common = [
        (arms["k4"], "k=4/D=22", PALETTE["k4"]),
        (arms["d16_30"], "k=24/D=16", PALETTE["d16"]),
        (arms["d32_30"], "k=24/D=32", PALETTE["d32"]),
        (arms["d32_catv_30"], "k=24/D=32+CATV", PALETTE["catv"]),
        (arms["a01_d32_30"], "a01: k=24/D=32", PALETTE["violet"]),
    ]
    labels, rates, colors = [], [], []
    for arm, label, color in common[::-1]:
        labels.append(label)
        rates.append(arm["rate"])
        colors.append(color)
    ypos = list(range(len(labels)))
    ax.barh(ypos, rates, color=colors, height=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Full-PPL reduction per hour (rounds 0-29)")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for y, rate in zip(ypos, rates):
        ax.text(rate + 0.002, y, f"{rate:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, max(rates) * 1.25)
    fig.tight_layout()
    paths = save_fig(fig, figdir, "fig3_efficiency")
    plt.close(fig)
    return paths


def fig4_pos_adaptation(figdir):
    manifest = load_json(REPO / "checkpoints/posemb_step1/MANIFEST.json")
    base = manifest["baseline"]
    arms = manifest["arms"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    labels = ["No adaptation", "pos_only (B)", "pos_lora (C)"]
    ppl16 = [base["ppl_16k"], arms["a01_pos_only_500step"]["final_16k"]["ppl_full"],
             arms["a01_pos_lora_500step"]["final_16k"]["ppl_full"]]
    ppl2k = [base["ppl_2k"], arms["a01_pos_only_500step"]["ppl_2k"],
             arms["a01_pos_lora_500step"]["ppl_2k"]]
    xpos = list(range(len(labels)))
    width = 0.36
    bars1 = ax.bar([x - width / 2 for x in xpos], ppl16, width, label="16k", color="#1f77b4")
    bars2 = ax.bar([x + width / 2 for x in xpos], ppl2k, width, label="2k", color="#ff7f0e")
    ax.set_yscale("log")
    ax.set_ylabel("G-PPL (log scale)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["none", "pos_only (B)", "pos_lora (C)"])
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper right")
    for bars in (bars1, bars2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.1f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)
    fig.tight_layout()
    paths = save_fig(fig, figdir, "fig4_pos_adaptation")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- tables

def fmt(value, digits):
    return f"{value:.{digits}f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def tex_table(caption, label, header, rows, align=None):
    if align is None:
        align = "l" + "r" * (len(header) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def fmt_p(value):
    if value is None:
        return "-"
    if value == 0.0:
        return "<1e-300"
    return f"{value:.1e}"


def tex_p(value):
    if value is None:
        return "-"
    if value == 0.0:
        return "$<10^{-300}$"
    exponent = math.floor(math.log10(value))
    mantissa = value / (10 ** exponent)
    return f"${mantissa:.1f}\\times 10^{{{exponent}}}$"


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def ensure_dirs(*dirs):
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def table1_main(arms, tabledir, datadir):
    entries = [
        ("ZOO k=4/D=22 (TD-3, a03)", arms["k4"], "4", "22", "0.15", "1.0", "0-29"),
        ("ZOO k=24/D=16 (a03)", arms["d16_30"], "24", "16", "0.30", "19.41", "0-29"),
        ("ZOO k=24/D=16 chain (a03)", arms["d16_60"], "24", "16", "0.30->0.10", "19.41/10.21", "0-59"),
        ("ZOO k=24/D=32 (a03)", arms["d32_30"], "24", "32", "0.30", "12.61", "0-29"),
        ("ZOO k=24/D=32+CATV (a03)", arms["d32_catv_30"], "24", "32", "0.30", "12.61", "0-29"),
        ("ZOO k=24/D=32 chain (a03)", arms["d32_74"], "24", "32", "0.30->0.10", "12.61/7.3-8.5", "0-74"),
        ("ZOO k=24/D=32+CATV chain (a03)", arms["d32_catv_74"], "24", "32", "0.30->0.10", "12.61/7.3-8.5", "0-74"),
        ("ZOO k=24/D=32 (a01)", arms["a01_d32_30"], "24", "32", "0.30", "13.34", "0-29"),
        ("ZOO k=24/D=32+CATV (a01)", arms["a01_catv_30"], "24", "32", "0.30", "13.34", "0-29"),
        ("ZOO k=24/D=32 chain (a01)", arms["a01_d32_44"], "24", "32", "0.30->0.10", "13.34/7.03", "0-44"),
        ("ZOO k=24/D=32+CATV chain (a01)", arms["a01_catv_44"], "24", "32", "0.30->0.10", "13.34/7.45", "0-44"),
    ]
    header = ["arm", "k", "D", "lr", "clip", "rounds", "full PPL (start -> end)", "dFull %",
              "answer PPL (start -> end)", "dAns %", "hours", "PPL/h", "clipped"]
    rows = []
    for name, arm, k, d, lr, clip, rng in entries:
        rows.append([
            name, k, d, lr, clip, rng,
            f"{arm['full0']:.4f} -> {arm['fullN']:.4f}", f"{arm['d_full_pct']:+.2f}",
            f"{arm['ans0']:.1f} -> {arm['ansN']:.1f}", f"{arm['d_ans_pct']:+.2f}",
            f"{arm['hours_total']:.2f}", f"{arm['rate']:.4f}", str(arm["clip_total"]),
        ])
    (tabledir / "table1_main_results.md").write_text(
        "# Table 1 - OPT-350M main results (same-protocol, no mask)\n\n" + md_table(header, rows) + "\n",
        encoding="utf-8")
    tex = tex_table(
        "OPT-350M main results (same-protocol G-PPL, 30 clients, 16k, ZOO+LoRA).",
        "tab:opt-main", ["Arm", "k", "D", "lr", "clip", "rounds", "Full PPL", "dFull (\\%)",
                         "Answer PPL", "dAns (\\%)", "Hours", "PPL/h", "Clip"],
        [[r[0], r[1], r[2], r[3].replace("->", "$\\to$"), r[4], r[5].replace("-", "--"),
          r[6].replace(" -> ", "$\\to$"), r[7].replace("%", "\\%"),
          r[8].replace(" -> ", "$\\to$"), r[9].replace("%", "\\%"),
          r[10], r[11], r[12]] for r in rows])
    (tabledir / "table1_main_results.tex").write_text(tex, encoding="utf-8")
    write_csv(datadir / "table1_main_results.csv", header, rows)
    return header, rows


def table2_catv(rows, tabledir, datadir):
    header = ["tag", "round", "protocol", "off full", "on full", "full ratio", "p(full)",
              "off ans", "on ans", "ans ratio", "p(ans)"]
    md_rows = []
    for row in rows:
        md_rows.append([
            row["tag"], row["round"], row["protocol"],
            f"{row['off_full']:.4f}", f"{row['on_full']:.4f}",
            "-" if row["full_ratio"] is None else f"{row['full_ratio']:.4f}",
            "-" if row["full_p"] is None else f"{row['full_p']:.1e}",
            f"{row['off_ans']:.1f}", f"{row['on_ans']:.1f}",
            "-" if row["ans_ratio"] is None else f"{row['ans_ratio']:.4f}",
            "-" if row["ans_p"] is None else f"{row['ans_p']:.1e}",
        ])
    note = ("`anchor` = shared r0 adapter evaluated without mask; `mask-effect` = same r0 adapter\n"
            "with vs without the eval mask (isolates the mask cost, not a training effect);\n"
            "`same`/`deploy` compare CATV on vs off under identical / deployment evaluation.\n"
            "Ratios are paired per-sample (n=500); ratio < 1 means the `on` column has lower NLL.\n")
    (tabledir / "table2_catv_alpha.md").write_text(
        "# Table 2 - CATV x alpha x evaluation protocol\n\n" + note + "\n" + md_table(header, md_rows) + "\n",
        encoding="utf-8")
    tex_rows = []
    for row in rows:
        tex_rows.append([
            row["tag"], row["round"], row["protocol"],
            f"{row['off_full']:.4f}", f"{row['on_full']:.4f}",
            f"{row['full_ratio']:.4f}", tex_p(row["full_p"]),
            f"{row['off_ans']:.1f}", f"{row['on_ans']:.1f}",
            f"{row['ans_ratio']:.4f}", tex_p(row["ans_p"]),
        ])
    tex = tex_table(
        "CATV ablation by Dirichlet alpha and evaluation protocol (paired per-sample, n=500).",
        "tab:opt-catv", ["alpha", "round", "protocol", "off full", "on full", "ratio", "$p$",
                         "off ans", "on ans", "ratio", "$p$"], tex_rows)
    (tabledir / "table2_catv_alpha.tex").write_text(tex, encoding="utf-8")
    write_csv(datadir / "table2_catv_alpha.csv", header, md_rows)
    return header, md_rows


def table3_quality(tabledir, datadir):
    selection = [
        ("step5 predictor (main)", "temp/ppl16k_compare/ppl_step5.json"),
        ("random predictor", "temp/ppl16k_compare/ppl_random.json"),
        ("dense (no prune)", "temp/ppl16k_compare/ppl_dense.json"),
    ]
    header1 = ["selection", "full PPL", "answer PPL", "n"]
    rows1 = []
    for name, path in selection:
        data = load_json(REPO / path)
        rows1.append([name, f"{data['full_ppl_token']:.3f}", f"{data['answer_ppl_token']:.1f}",
                      str(data["num_samples"])])

    dense_r0 = load_json(REPO / "temp/diag_dense_eval/r000.json")
    dense_r29 = load_json(REPO / "temp/diag_dense_eval/r029.json")
    header2 = ["variant", "r0 full", "r29 full", "full ratio", "p(full)",
               "r0 ans", "r29 ans", "ans ratio", "p(ans)"]
    full_pair = load_json(REPO / "temp/diag_dense_eval/paired_r0_r29_full.json")
    ans_pair = load_json(REPO / "temp/diag_dense_eval/paired_r0_r29_answer.json")
    sparse_full = load_json(REPO / "temp/diag_compare/paired_sparse_r0_r29_full.json")
    sparse_ans = load_json(REPO / "temp/diag_compare/paired_sparse_r0_r29_answer.json")
    sparse_r0 = xeval(REPO / "logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/eval_round000.json")
    sparse_r29 = xeval(REPO / "logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip/a03/eval_round029.json")
    rows2 = [
        ["dense (sparse=1.0)",
         f"{dense_r0['full_ppl_token']:.3f}", f"{dense_r29['full_ppl_token']:.3f}",
         f"{full_pair['ppl_ratio_new_over_old']:.4f}", fmt_p(full_pair["p_two_sided"]),
         f"{dense_r0['answer_ppl_token']:.1f}", f"{dense_r29['answer_ppl_token']:.1f}",
         f"{ans_pair['ppl_ratio_new_over_old']:.4f}", fmt_p(ans_pair["p_two_sided"])],
        ["sparse (sparse=0.4, TD-3)",
         f"{sparse_r0[0]:.4f}", f"{sparse_r29[0]:.4f}",
         f"{sparse_full['ppl_ratio_new_over_old']:.4f}", fmt_p(sparse_full["p_two_sided"]),
         f"{sparse_r0[1]:.1f}", f"{sparse_r29[1]:.1f}",
         f"{sparse_ans['ppl_ratio_new_over_old']:.4f}", fmt_p(sparse_ans["p_two_sided"])],
    ]
    md = ("# Table 3 - Sparse-path quality\n\n"
          "## 3a. Predictor/selection quality at 16k (n=4; small-n support row, not paired; "
          "evaluated without the position checkpoint, so not comparable to the 500-sample rows)\n\n"
          + md_table(header1, rows1) + "\n\n"
          "## 3b. Dense vs sparse on the same adapters (paired per-sample, n=500)\n\n"
          + md_table(header2, rows2) + "\n\n"
          "Environment: OPT-350M + pos_only table, TD-3 adapters (LoRA r=1, a03, trained on the "
          "sparse rho=0.4 path), fp32 sdpa_prune, 500-sample global test; dense = the same adapters "
          "re-evaluated with sparse=1.0 (no pruning).\n")
    (tabledir / "table3_sparse_quality.md").write_text(md, encoding="utf-8")
    tex = tex_table("Sparse-path quality: predictor/selection variants at 16k (n=4).",
                    "tab:opt-quality", header1, rows1)
    (tabledir / "table3_sparse_quality.tex").write_text(tex, encoding="utf-8")
    write_csv(datadir / "table3_predictor_quality.csv", header1, rows1)
    write_csv(datadir / "table3_dense_vs_sparse.csv", header2, rows2)
    return header1, rows1


def table4_pos_adapt(tabledir, datadir):
    manifest = load_json(REPO / "checkpoints/posemb_step1/MANIFEST.json")
    base = manifest["baseline"]
    header = ["arm", "mode", "16k G-PPL", "2k G-PPL", "judgement"]
    rows = [
        ["no adaptation", "-", f"{base['ppl_16k']:.2f}", f"{base['ppl_2k']:.2f}", "reference"],
        ["pos_only (B)", "learned offset (interp init)",
         f"{manifest['arms']['a01_pos_only_500step']['final_16k']['ppl_full']:.2f}",
         f"{manifest['arms']['a01_pos_only_500step']['ppl_2k']:.2f}",
         "16k G-PPL < 50 PASS; no 2k regression"],
        ["pos_lora (C)", "pos_only + LoRA adapter",
         f"{manifest['arms']['a01_pos_lora_500step']['final_16k']['ppl_full']:.2f}",
         f"{manifest['arms']['a01_pos_lora_500step']['ppl_2k']:.2f}",
         "selected; best 16k and 2k"],
    ]
    md = "# Table 4 - Position adaptation (a01 warmup pool)\n\n" + md_table(header, rows) + "\n"
    (tabledir / "table4_pos_adaptation.md").write_text(md, encoding="utf-8")
    tex = tex_table("Position-embedding adaptation for 16k context (a01).", "tab:opt-pos", header, rows)
    (tabledir / "table4_pos_adaptation.tex").write_text(tex, encoding="utf-8")
    write_csv(datadir / "table4_pos_adaptation.csv", header, rows)
    return header, rows


def micro_rows():
    out = []
    cells = [
        ("k=4", "D=11", "0.30", "A_k4_D11_e1"),
        ("k=4", "D=22", "0.30", "A_k4_D22_e1"),
        ("k=12", "D=11", "0.30", "A_k12_D11_e1"),
        ("k=12", "D=22", "0.30", "A_k12_D22_e1"),
        ("k=24", "D=11", "0.30", "A_k24_D11_e1"),
        ("k=24", "D=22", "0.30", "A_k24_D22_e1"),
    ]
    for k, d, lr, name in cells:
        rows = load_round_rows(REPO / "temp/k_scan2" / name / A03 / "metrics.jsonl")
        loss0 = rows[0]["train_loss_mean"]
        lossN = rows[-1]["train_loss_mean"]
        delta = lossN - loss0
        unstable = "yes" if abs(delta) > 0.01 else "no"
        out.append([k, d, lr, f"{loss0:.5f}", f"{lossN:.5f}", f"{delta:+.5f}",
                    f"{rows[0].get('delta_norm_mean', 0.0):.3g}", unstable])

    pilot = [
        ("L=1", "D=8", "0.15", "1.0", "A_D8_L1_lr0.15_c1"),
        ("L=2", "D=8", "0.15", "1.0", "A_D8_L2_lr0.15_c1"),
        ("L=1", "D=16", "0.15", "1.0", "A_D16_L1_lr0.15_c1"),
        ("L=2", "D=16", "0.15", "1.0", "A_D16_L2_lr0.15_c1"),
        ("L=1", "D=22", "0.15", "1.0", "A_D22_L1_lr0.15_c1"),
        ("L=2", "D=22", "0.15", "1.0", "A_D22_L2_lr0.15_c1"),
        ("L=2", "D=16", "0.30", "1.0", "B_D16_L2_lr0.30_c1"),
        ("L=2", "D=16", "0.30", "2.0", "B_D16_L2_lr0.30_c2"),
        ("L=2", "D=22", "0.30", "1.0", "B_D22_L2_lr0.30_c1"),
        ("L=2", "D=22", "0.30", "2.0", "B_D22_L2_lr0.30_c2"),
    ]
    pilot_rows = []
    for block, d, lr, clip, name in pilot:
        path = REPO / "temp" / "pilot_micro_scan" / name / A03 / "metrics.jsonl"
        rows = load_round_rows(path)
        delta = rows[-1]["train_loss_mean"] - rows[0]["train_loss_mean"]
        pilot_rows.append([block, d, lr, clip, f"{delta:+.5f}", f"{rows[0].get('delta_norm_mean', 0.0):.3g}"])
    return out, pilot_rows


def fresh_baseline():
    """Pre-training baseline (base + pos_only, identity LoRA) on the shared 500-sample test."""
    path = REPO / "temp/eval_fresh_baseline.json"
    if not path.exists():
        return None
    return xeval(path)


def table7_d_comparison(tabledir, datadir):
    """D comparison over the unified 8-round window (full G-PPL).

    Primary deltas use the shared pre-training baseline (measured 2026-10-14); each arm's
    own post-round-0 r0 is kept in the footnote. D16/D22 from the 8-round pilots; D32
    endpoint = post-hoc eval of the 30-round run's adapter_round007.
    """
    cells = [
        ("16", "temp/pilot_k24/K24_D16_lr0.30_c19.41/a03/eval_round000.json",
         "temp/pilot_k24/K24_D16_lr0.30_c19.41/a03/eval_round007.json"),
        ("22", "temp/pilot_k24/K24_D22_lr0.30_c16.08/a03/eval_round000.json",
         "temp/pilot_k24/K24_D22_lr0.30_c16.08/a03/eval_round007.json"),
        ("32", "temp/eval_shared_r0_nomask.json",
         "temp/eval_d32_r7.json"),
    ]
    fresh = fresh_baseline()
    header = ["D", "window", "baseline (pre-r0, shared)", "endpoint full PPL", "delta", "delta %"]
    rows = []
    own_notes = []
    for d, r0_path, end_path in cells:
        own_r0 = xeval(REPO / r0_path)[0]
        end = xeval(REPO / end_path)[0]
        base = fresh[0] if fresh is not None else own_r0
        delta = end - base
        rows.append([d, "pre-r0->r7", f"{base:.4f}", f"{end:.4f}",
                     f"{delta:+.4f}", f"{delta / base * 100.0:+.2f}"])
        own_notes.append(f"D{d} own r0 {own_r0:.4f} ({end - own_r0:+.4f})")
    extras = []
    for label, path in [("r8", "temp/eval_d32_r8.json"), ("r9", "temp/eval_d32_r9.json")]:
        if (REPO / path).exists():
            extras.append(f"{label} {xeval(REPO / path)[0]:.4f}")
    extra_txt = (" D32 extended points (not in the unified window): " + ", ".join(extras) + "."
                 if extras else "")
    base_note = ("Shared pre-training baseline = base model + pos_only table with identity LoRA "
                 "(PEFT B=0), same 500-sample protocol; measured once (full 33.6653 / answer 1740.6), "
                 "applies to all arms." if fresh is not None else
                 "Per-arm post-round-0 r0 used (pre-training baseline not measured).")
    note = (f"{base_note} All rows use the 8-round window (r7 = after round 8; eval_every=1). "
            "D16/D22 from the 8-round pilots; D32 endpoint is a post-hoc eval of the 30-round run's "
            "adapter_round007 (the in-run eval grid is every 5 rounds). "
            + "; ".join(own_notes) + "." + extra_txt + "\n")
    md = "# Table 7 - D comparison (unified 8-round window, full G-PPL)\n\n" + note + "\n" + md_table(header, rows) + "\n"
    (tabledir / "table7_d_comparison.md").write_text(md, encoding="utf-8")
    tex = tex_table("D comparison over the unified 8-round window (full G-PPL; shared pre-training baseline).",
                    "tab:opt-d", ["D", "window", "baseline", "endpoint", "$\\Delta$", "$\\Delta$ \\%"],
                    [[r[0], r[1], r[2], r[3], r[4], r[5].replace("%", "\\%")] for r in rows])
    (tabledir / "table7_d_comparison.tex").write_text(tex, encoding="utf-8")
    write_csv(datadir / "table7_d_comparison.csv", header, rows)
    return header, rows


def _times_plt():
    plt = _plt()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    return plt


def fig5_catv_onoff(arms, figdir):
    paths = []
    specs = [
        ("fig5_catv_alpha01", r"$\alpha$=0.1",
         arms["a01_d32_44"], arms["a01_catv_44"], arms["a01_catv_depl"], (0, 46)),
        ("fig6_catv_alpha03", r"$\alpha$=0.3",
         arms["d32_74"], arms["d32_catv_74"], arms["a03_catv_depl"], (0, 76)),
    ]
    for stem, alpha, off, on, on_depl, xlim in specs:
        plt = _times_plt()
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        plot_series(ax, off, 0, "CATV off (baseline)", PALETTE["gray"], "-", "o")
        plot_series(ax, on, 0, "CATV on (eval off)", PALETTE["catv"], "--", "s")
        plot_series(ax, on_depl, 0, "CATV on (eval on)", PALETTE["violet"], ":", "^")
        ax.set_xlabel("Round")
        ax.set_ylabel("Full G-PPL")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(*xlim)
        legend = ax.legend(loc="upper right", frameon=True, facecolor="white",
                           edgecolor="black", framealpha=1.0, fancybox=False,
                           title=alpha, title_fontsize=7)
        legend.get_frame().set_linewidth(0.7)
        fig.tight_layout()
        paths += save_fig(fig, figdir, stem)
        plt.close(fig)
    return paths


def table6_endpoints(arms, tabledir, datadir):
    specs = [
        ("0.1", "CATV off (baseline)", arms["a01_d32_44"]),
        ("0.1", "CATV on (eval off)", arms["a01_catv_44"]),
        ("0.1", "CATV on (eval on)", arms["a01_catv_depl"]),
        ("0.3", "CATV off (baseline)", arms["d32_74"]),
        ("0.3", "CATV on (eval off)", arms["d32_catv_74"]),
        ("0.3", "CATV on (eval on)", arms["a03_catv_depl"]),
    ]
    header = ["\u03b1", "CATV / eval", "r44 PPL", "r44 drop %", "r74 PPL", "r74 drop %"]
    rows = []
    for alpha, label, arm in specs:
        full0 = arm["points"][0][0]
        cells = []
        for rnd in (44, 74):
            if rnd in arm["points"]:
                ppl = arm["points"][rnd][0]
                cells += [f"{ppl:.4f}", f"{(ppl / full0 - 1.0) * 100.0:+.2f}"]
            else:
                cells += ["-", "-"]
        rows.append([alpha, label] + cells)
    note = ("Full G-PPL at r44/r74; drop % is relative to each arm's own r0 anchor "
            "(same-protocol: 33.5296 for a03 / 33.5253 for a01; deployment: masked r0 "
            "33.9393 / 33.9211). a01 runs stopped at r44, so r74 is not available.\n")
    md = "# Table 6 - CATV endpoints at r44 / r74 (full G-PPL)\n\n" + note + "\n" + md_table(header, rows) + "\n"
    (tabledir / "table6_catv_r44_r74.md").write_text(md, encoding="utf-8")
    tex_rows = [[r[0], r[1], r[2], r[3].replace("%", "\\%"), r[4], r[5].replace("%", "\\%")] for r in rows]
    tex = tex_table("CATV endpoint comparison at r44/r74 (full G-PPL; drops vs own r0 anchor).",
                    "tab:opt-endpoints", ["$\\alpha$", "CATV / eval", "r44 PPL", "r44 drop \\%",
                                          "r74 PPL", "r74 drop \\%"], tex_rows)
    (tabledir / "table6_catv_r44_r74.tex").write_text(tex, encoding="utf-8")
    write_csv(datadir / "table6_catv_r44_r74.csv", header, rows)
    return header, rows


def table5_micro(tabledir, datadir):
    scan_rows, pilot_rows = micro_rows()
    header1 = ["k (layers/round)", "D", "lr", "loss r0", "loss r2", "dLoss (r2-r0)", "delta r0", "unstable"]
    header2 = ["layers/round L", "D", "lr", "clip", "dLoss (r2-r0)", "delta r0"]
    md = ("# Table 5 - Coverage micro-scans (1 client, 2 samples, 3 rounds, 16k)\n\n"
          "## 5a. k sweep at fixed budget (L=2, lr=0.30)\n\n" + md_table(header1, scan_rows) + "\n\n"
          "## 5b. L x D pilot\n\n" + md_table(header2, pilot_rows) + "\n")
    (tabledir / "table5_micro_scans.md").write_text(md, encoding="utf-8")
    tex = tex_table("Coverage micro-scan: k sweep (1 client, 2 samples, 3 rounds).",
                    "tab:opt-micro", header1, scan_rows)
    (tabledir / "table5_micro_scans.tex").write_text(tex, encoding="utf-8")
    write_csv(datadir / "table5_k_sweep.csv", header1, scan_rows)
    write_csv(datadir / "table5_ld_pilot.csv", header2, pilot_rows)
    return header1, scan_rows


# ---------------------------------------------------------------- data dumps

def dump_series(arms, datadir):
    header = ["arm", "round", "full_ppl", "answer_ppl", "hours"]
    rows = []
    for name, arm in arms.items():
        for r in sorted(arm["points"]):
            full, ans = arm["points"][r]
            rows.append([name, r, f"{full:.4f}", f"{ans:.2f}",
                         f"{arm['hours'].get(r, 0.0):.3f}"])
    write_csv(datadir / "arm_points.csv", header, rows)


# ---------------------------------------------------------------- README

README_TEMPLATE = """# OPT-350M results package

由 `reports/opt/make_report.py` 从原始产物自动生成（纯 CPU、确定性；禁止手工改本目录数字）。

## 产物

- `figures/fig1_a03_convergence.{pdf,png}` — a03 主曲线：k=4/D=22(30r)、k=24/D=16(60r)、k=24/D=32(74r)、k=24/D=32+CATV(74r)，full 与 answer G-PPL。同口径（无掩码）。
- `figures/fig2_catv_alpha.{pdf,png}` — CATV × alpha：a03/a01 的 off / on(同口径) / on(部署) 双指标曲线。
- `figures/fig3_efficiency.{pdf,png}` — 左：full G-PPL vs 实测墙钟小时（完整链）；右：PPL/h 条形（统一取 0–29 轮口径）。
- `figures/fig4_pos_adaptation.{pdf,png}` — 位置适配 B/C 的 16k/2k G-PPL。
- `figures/fig5_catv_alpha01.{pdf,png}` — α=0.1 的 CATV 三条曲线（off (baseline) / on (eval off) / on (eval on)），16k full G-PPL；Times New Roman、无标题、图例白底黑框。
- `figures/fig6_catv_alpha03.{pdf,png}` — α=0.3 同款。
- `tables/table1_main_results.{md,tex}` — 主结果汇总（同口径）。
- `tables/table2_catv_alpha.{md,tex}` — CATV × alpha × 口径，含逐样本配对 ratio/p。
- `tables/table3_sparse_quality.{md,tex}` — predictor/随机/dense 对照（n=4）+ dense vs sparse（n=500）。
- `tables/table4_pos_adaptation.{md,tex}` — 位置适配判据表。
- `tables/table5_micro_scans.{md,tex}` — 覆盖度微扫（k 扫描 + L×D 先导）。
- `tables/table6_catv_r44_r74.{md,tex}` — CATV 三口径（off / on 同口径 / on 部署）在 r44/r74 的实际值与相对 r0 下降率。
- `tables/table7_d_comparison.{md,tex}` — D 对照（D16/D22/D32，统一 8 轮窗口；**以训练前共享基线 33.6653 起算**，行内保留各臂 own-r0 备注）。
- `data/*.csv` — 上述图表底层 tidy 数据（`arm_points.csv` 为全部曲线点）。

## 口径与数据来源

- 运行均在 AutoDL 1×RTX 4090：ZOO+LoRA、fp32 + `sdpa_prune`、`sparse=0.4`、
  `--predictor step5_a01a03_clients_causal`、c=30 串行 FedAvg、16k、a03/a01。
- 曲线取 `metrics.jsonl` 的 eval 事件（token 加权 `full_ppl_token`/`answer_ppl_token`）；
  r0 锚点与 ON 臂同口径点取 `temp/eval_*.json` 事后评测（脚本内 `PERSAMPLE`/`R0_*` 注册表）。
- 墙钟小时：每个 run block `config.json` mtime → 最后一个 `adapter_round*` mtime；链式
  实验为各 block 之和（不含 block 间探测间隙），与 ailog 中的约数略有出入属定义差异。
- PPL/h = (full PPL 起点 − 终点) / 墙钟小时；F3 右图统一 0–29 轮窗口。
- fig5/fig6 按用户指定样式生成：Times New Roman（serif）、无标题、**图例**白底黑框、
  α 作为图例标题；表中 α 一律用希腊字母（Markdown 用 `α`，TeX 源用 `$\alpha$`）。
- 主运行目录：`logs/fed/zoo_k4_sparse_causal_a03_lr0.15_clip`、`zoo_k24_d16_lr0.30_r30`、
  `zoo_k24_d16_r60_cont_lr0.10`、`zoo_k24_d32[_catv]_lr0.30_r30`、
  `a03_zoo_k24_d32[_catv]_lr0.10_cont15{,b,c}`、`a01_zoo_k24_d32[_catv]_lr0.30_r30`、
  `a01_zoo_k24_d32[_catv]_lr0.10_cont15`。
- 对账：脚本内置 ailog 已发表关键值断言，运行输出 32/32 PASS 才写本文件。

## 结论摘要

- 覆盖度是 ZOO 的主杠杆：k=4→24 使每轮 full-PPL 降幅约提高一个数量级，
  0–29 轮速率 k4 0.019、D16 0.101、D32 0.087、D32+CATV 0.094、a01 0.084 PPL/h。
- CATV 的收益依赖 alpha：a03（0.3）同口径全程领先约 0.20 PPL 且 answer 早期显著改善
  （r29 answer ratio 0.83，p=5e-31）；a01（0.1）训练效应不显著（r29 full p=0.97）。
- a03 部署口径（训练+评测均带掩码）在 r64–r69 打平（r74 full ratio 1.0001，p=0.65），
  同口径仍领先 0.20；answer 的同口径优势在长训后反转（r74 ratio 1.13，p=6.7e-8），
  部署口径残余 answer 优势（0.93）主要来自评测掩码本身。
- 位置适配：16k G-PPL 398.8 → 31.4（B）→ 18.1（C），2k 无回归，选 C。

## 已知 caveat（写盘时保留，勿在正文中抹平）

- T3a 的 predictor/random/dense 对照只有 n=4（历史小样本，且未加载位置适配），仅作支持行；
  需要正式口径可用 GPU 空闲时以 500 样本重评（当前 GPU 被 BERT 链占用）。
- pos_lora 训练曲线与权重不在本机（cloud A40 产物），T4/F4 用 `posemb_step1/MANIFEST.json`
  的终值；本机仅有 pos_only 的 metrics。
- 云端 A40 的 BP 对照与 4 卡 alpha×CATV 全协议消融未纳入本包（按用户决定暂不做）。
- 260929 审计前的历史 ZOO run 与 lr clamp bug run 已排除；`zoo_k4_sparse_causal_a03`
  （lr≈9e-6）与 `..._step5`（lr=1e-4 被 clamp）不作结果使用。
- clipped 列是 30 client-轮累计被 `--delta-clip` 裁剪的数量（按 3×median 探测标定），
  表示稀疏路径 FD 噪声水平，不是失败计数。

## 复现

```bash
PY=~/miniconda3/envs/SMELL/bin/python
$PY reports/opt/make_report.py            # 全量 + 对账闸门
$PY reports/opt/make_report.py --only figures
$PY reports/opt/make_report.py --only tables
$PY reports/opt/make_report.py --no-check # 仅调试用
```

字体：fig5/fig6 指定 Times New Roman，本机经
`apt-get install -y ttf-mscorefonts-installer` 安装（华为云镜像）；换机缺失时脚本按
`Times New Roman → Nimbus Roman → Liberation Serif → STIXGeneral → DejaVu Serif` 自动回退。
"""


def write_readme(outdir, check_results):
    passed = sum(1 for *_rest, ok in check_results if ok)
    failed = [r for r in check_results if not r[-1]]
    body = README_TEMPLATE
    body += "\n## 对账结果\n\n"
    body += f"- 断言：{passed}/{len(check_results)} PASS\n"
    if failed:
        body += "- FAIL：\n"
        for name, got, want, tol, _ok in failed:
            body += f"  - {name}: got {got} want {want} (tol {tol})\n"
    (outdir / "README.md").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description="Build the OPT-350M results package")
    parser.add_argument("--out", default=str(REPO / "reports/opt"), help="output directory")
    parser.add_argument("--only", choices=("all", "figures", "tables"), default="all")
    parser.add_argument("--no-check", action="store_true", help="skip ailog cross-checks")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.out)
    figdir = outdir / "figures"
    tabledir = outdir / "tables"
    datadir = outdir / "data"
    ensure_dirs(figdir, tabledir, datadir)

    arms = build_arms()
    checks = run_validation(arms)
    checks.extend(d_checks())
    catv = catv_rows()

    print("== ailog cross-checks (PPL) ==")
    bad = 0
    for name, got, want, tol, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} got={got:.4f} want={want} tol={tol}")
        bad += 0 if ok else 1
    catv_checks = []
    for name, expected, tol in CATV_CHECKS:
        field = "ans_ratio" if "answer" in name else "full_ratio"
        tag = name[:3]
        rnd = "r29" if "r29" in name else ("r74" if "r74" in name else "r44")
        proto = "deploy" if "deploy" in name else "same"
        row = next(r for r in catv if r["tag"] == tag and r["round"] == rnd and r["protocol"] == proto)
        got = row[field]
        ok = got is not None and abs(got - expected) <= tol
        catv_checks.append((name, got, expected, tol, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} got={got:.4f} want={expected} tol={tol}")
        bad += 0 if ok else 1
    checks.extend(catv_checks)

    if bad and not args.no_check:
        raise SystemExit(f"{bad} cross-checks FAILED; refusing to build (use --no-check to override)")

    if args.only in ("all", "figures"):
        print("== figures ==")
        for path in fig1_convergence(arms, figdir):
            print(f"  {path.relative_to(REPO)}")
        for path in fig2_catv(arms, figdir):
            print(f"  {path.relative_to(REPO)}")
        for path in fig3_efficiency(arms, figdir):
            print(f"  {path.relative_to(REPO)}")
        for path in fig4_pos_adaptation(figdir):
            print(f"  {path.relative_to(REPO)}")
        for path in fig5_catv_onoff(arms, figdir):
            print(f"  {path.relative_to(REPO)}")

    if args.only in ("all", "tables"):
        print("== tables ==")
        table1_main(arms, tabledir, datadir)
        table2_catv(catv, tabledir, datadir)
        table3_quality(tabledir, datadir)
        table4_pos_adapt(tabledir, datadir)
        table5_micro(tabledir, datadir)
        table6_endpoints(arms, tabledir, datadir)
        table7_d_comparison(tabledir, datadir)
        dump_series(arms, datadir)
        for path in sorted(tabledir.iterdir()):
            print(f"  {path.relative_to(REPO)}")

    write_readme(outdir, checks)
    print(f"== done: {outdir} ==")


if __name__ == "__main__":
    main()
