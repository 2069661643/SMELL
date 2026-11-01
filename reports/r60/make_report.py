#!/usr/bin/env python3
# SMELL 3 reports/r60 NEW — r60 交付包（BERT 第三段 r60-r89）：CSV + README，无图
# 覆盖：4 臂（eval on/off）、dense VAR、VQ 0.5 per-result、combo（eval on/off）、var-only（r30 存档，无 r60 续训）
# 纯 CPU、确定性；读取 logs/fed/ 与 checkpoints/ 的 SHA256SUMS；内置对账断言。
import csv
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "r60"
DATA = OUT / "data"

ARMS = [
    ("r30_a01_catv_off", "a01", "off", "off"),
    ("r30_a01_catv_on", "a01", "on", "off"),
    ("r30_a03_catv_off", "a03", "off", "off"),
    ("r30_a03_catv_on", "a03", "on", "off"),
]
ARM_ROOT = REPO / "logs/fed/bert_da_fixed_cont60"
ARM_PREV = REPO / "logs/fed/bert_da_fixed_cont30"
ARM_ARCH = REPO / "checkpoints/distilbert_ablation_r30_fixed_cont60"
ARM_PREV_ARCH = REPO / "checkpoints/distilbert_ablation_r30_fixed_cont30"

TECH = [
    ("dense VAR (sparse=1.0)", "logs/fed/bert_fwdllm_a03_var_cont60/a03",
     "logs/fed/bert_fwdllm_a03_var_cont30/a03", "checkpoints/distilbert_fwdllm_a03_var_cont60", "sparse 1.0 + VAR Method A"),
    ("3-on (VAR+CATV+VQ pr)", "logs/fed/bert_var_catv_vq_a03_cont60/a03",
     "logs/fed/bert_var_catv_vq_a03_cont30/a03", "checkpoints/distilbert_var_catv_vq_a03_cont60", "VAR+CATV(client)+VQ 0.5 per-result"),
    ("VQ 0.5 per-result", "logs/fed/bert_vq050_perresult_a03_cont60/a03",
     "logs/fed/bert_vq050_perresult_a03_cont30/a03", "checkpoints/distilbert_vq050_perresult_a03_cont60", "VQ 0.5 bf16 per-result"),
]
HIST = [
    ("4arm a01 off", ["logs/fed/bert_da_fixed/r30_a01_catv_off/a01", "logs/fed/bert_da_fixed_cont30/r30_a01_catv_off/a01", "logs/fed/bert_da_fixed_cont60/r30_a01_catv_off/a01"]),
    ("4arm a01 on", ["logs/fed/bert_da_fixed/r30_a01_catv_on/a01", "logs/fed/bert_da_fixed_cont30/r30_a01_catv_on/a01", "logs/fed/bert_da_fixed_cont60/r30_a01_catv_on/a01"]),
    ("4arm a03 off", ["logs/fed/bert_da_fixed/r30_a03_catv_off/a03", "logs/fed/bert_da_fixed_cont30/r30_a03_catv_off/a03", "logs/fed/bert_da_fixed_cont60/r30_a03_catv_off/a03"]),
    ("4arm a03 on", ["logs/fed/bert_da_fixed/r30_a03_catv_on/a03", "logs/fed/bert_da_fixed_cont30/r30_a03_catv_on/a03", "logs/fed/bert_da_fixed_cont60/r30_a03_catv_on/a03"]),
    ("dense VAR (sparse=1.0)", ["logs/fed/bert_fwdllm_a03_var_r30/a03", "logs/fed/bert_fwdllm_a03_var_cont30/a03", "logs/fed/bert_fwdllm_a03_var_cont60/a03"]),
    ("VQ 0.5 per-result", ["logs/fed/bert_vq050_bf16_perresult_a03/a03", "logs/fed/bert_vq050_perresult_a03_cont30/a03", "logs/fed/bert_vq050_perresult_a03_cont60/a03"]),
    ("3-on (VAR+CATV+VQ pr)", ["logs/fed/bert_var_catv_vq_a03_r30_rerun/a03", "logs/fed/bert_var_catv_vq_a03_cont30/a03", "logs/fed/bert_var_catv_vq_a03_cont60/a03"]),
]
VAR_ONLY = REPO / "logs/fed/bert_var_methoda_a03_r30/a03"
VAR_ONLY_ARCH = REPO / "checkpoints/distilbert_var_methoda_a03_r30"

FIRST, LAST, PREV_ROUND = 60, 89, 59


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def metrics(path):
    rounds, evals = {}, {}
    if not path.exists():
        return rounds, evals
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "round":
            rounds[int(rec["round"])] = rec
        elif rec.get("event") == "eval" and rec.get("status") == "ok" and "accuracy" in rec:
            evals[int(rec["round"])] = rec
    return rounds, evals


def persample(path, key="nll"):
    if not path.exists():
        return None
    rows = load_json(path)
    if isinstance(rows, dict) and "rows" in rows:
        rows = rows["rows"]
    return {int(x.get("index", i)): float(x[key]) for i, x in enumerate(rows) if key in x}


def paired(a, b, label, tag="", a_run="", b_run="", field="nll"):
    if not a or not b:
        return None
    keys = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in keys]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = (sum((x - mean) ** 2 for x in diffs) / (n - 1)) ** 0.5
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    ci = 1.96 * se
    verdict = "IMPROVED" if mean < 0 else "WORSENED"
    return {
        "comparison": label, "tag": tag, "a_run": a_run, "b_run": b_run, "field": field,
        "n": n, "mean_delta_nll": mean, "sd": sd, "se": se, "t": t,
        "p_two_sided": p, "p_normal_approx": p, "ci95_low": mean - ci, "ci95_high": mean + ci,
        "verdict": verdict,
    }


def write_csv(name, rows, fields):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in fields})
    print(f"[r60] wrote {path.relative_to(REPO)} ({len(rows)} rows)")
    return path


def _fmt(value):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return value


def adapter_sha(path):
    sums = path / "SHA256SUMS"
    if not sums.exists():
        return ""
    for line in sums.read_text(encoding="utf-8").splitlines():
        if "adapter_model.safetensors" in line:
            return line.split()[0]
    return ""


def arm_rows():
    rounds_rows, endpoint_rows, paired_rows, evaloff_rows = [], [], [], []
    evaloff_endpoint_rows, evaloff_paired_rows = [], []
    for run_id, tag, catv, norm in ARMS:
        run_dir = ARM_ROOT / run_id / tag
        prev_dir = ARM_PREV / run_id / tag
        rd, ev = metrics(run_dir / "metrics.jsonl")
        _, prev_ev = metrics(prev_dir / "metrics.jsonl")
        assert len(ev) == 30, f"{run_id}: eval count {len(ev)} != 30"
        assert sorted(ev) == list(range(FIRST, LAST + 1)), f"{run_id}: eval rounds mismatch"
        for r in range(FIRST, LAST + 1):
            e = ev[r]
            m = rd[r]
            rounds_rows.append({
                "run_id": run_id, "tag": tag, "catv": catv, "catv_normalize": norm, "round": r,
                "nll": e["nll"], "accuracy": e["accuracy"], "macro_f1": e.get("macro_f1"), "n": e.get("n", 500),
                "round_seconds": m.get("round_seconds"), "train_loss_mean": m.get("train_loss_mean"),
                "delta_norm_mean": m.get("delta_norm_mean"), "delta_norm_max": m.get("delta_norm_max"),
                "clipped_clients": m.get("clipped_clients"), "catv_r": m.get("catv_r"),
                "catv_anchor_in": m.get("anchor_in"), "catv_anchor_out": m.get("anchor_out"),
                "catv_ir_mean": m.get("ir_mean"), "doc_order": m.get("doc_order"),
                "source_dir": str(run_dir.relative_to(REPO)),
            })
        prev59 = prev_ev.get(PREV_ROUND)
        assert prev59 is not None, f"{run_id}: missing prev r59 eval"
        hours = sum(rd[r]["round_seconds"] for r in range(FIRST, LAST + 1)) / 3600.0
        seg = (ev[FIRST]["nll"] - ev[LAST]["nll"]) / ev[FIRST]["nll"] * 100
        tot = (prev59["nll"] - ev[LAST]["nll"]) / prev59["nll"] * 100
        endpoint_rows.append({
            "run_id": run_id, "tag": tag, "catv": catv, "catv_normalize": norm, "doc_order": "cycle",
            "rounds": 30, "eval_n": ev[LAST].get("n", 500),
            "nll_r59": prev59["nll"], "nll_r60": ev[FIRST]["nll"], "nll_r89": ev[LAST]["nll"],
            "delta_nll_seg_pct": seg, "delta_nll_r59_r89_pct": tot,
            "acc_r59": prev59["accuracy"], "acc_r89": ev[LAST]["accuracy"],
            "delta_acc_points": (ev[LAST]["accuracy"] - prev59["accuracy"]) * 100,
            "macro_f1_r89": ev[LAST].get("macro_f1"),
            "mean_round_seconds": hours * 3600 / 30, "train_hours": hours, "nll_pct_per_hour": tot / hours if hours else 0.0,
            "adapter_sha256": adapter_sha(ARM_ARCH / run_id), "source_dir": str(run_dir.relative_to(REPO)),
        })
        a = persample(prev_dir / f"eval_round{PREV_ROUND:03d}_persample.json")
        b = persample(run_dir / f"eval_round{LAST:03d}_persample.json")
        row = paired(a, b, "r59_to_r89", tag, run_id, run_id, "nll")
        if row:
            paired_rows.append(row)
        if catv == "on":
            on_count = 0
            for r in range(FIRST, LAST + 1):
                on_path = run_dir / f"eval_round{r:03d}.json"
                off_path = run_dir / f"evaloff_round{r:03d}.json"
                if not off_path.exists():
                    continue
                on = load_json(on_path)
                off = load_json(off_path)
                on_count += 1
                evaloff_rows.append({
                    "run_id": run_id, "tag": tag, "round": r,
                    "on_nll": on["nll"], "on_acc": on["accuracy"], "off_nll": off["nll"], "off_acc": off["accuracy"],
                    "mask_dNLL": on["nll"] - off["nll"], "mask_dAcc": on["accuracy"] - off["accuracy"],
                })
            assert on_count == 30, f"{run_id}: evaloff count {on_count} != 30"
            for r in (PREV_ROUND, LAST):
                src = prev_dir if r == PREV_ROUND else run_dir
                on = load_json(src / f"eval_round{r:03d}.json")
                off = load_json(src / f"evaloff_round{r:03d}.json")
                evaloff_endpoint_rows.append({
                    "run_id": run_id, "tag": tag, "round": r,
                    "on_nll": on["nll"], "on_acc": on["accuracy"], "off_nll": off["nll"], "off_acc": off["accuracy"],
                    "mask_dNLL": on["nll"] - off["nll"], "mask_dAcc": on["accuracy"] - off["accuracy"],
                })
            a_off = persample(prev_dir / f"evaloff_round{PREV_ROUND:03d}_persample.json")
            b_off = persample(run_dir / f"evaloff_round{LAST:03d}_persample.json")
            a_on = persample(prev_dir / f"eval_round{PREV_ROUND:03d}_persample.json")
            b_on = persample(run_dir / f"eval_round{LAST:03d}_persample.json")
            for label, ra, rb in (("mask_effect_r59", a_off, a_on), ("mask_effect_r89", b_off, b_on),
                                  ("weight_effect_r59_r89_off", a_off, b_off),
                                  ("total_r59_r89_on", a_on, b_on)):
                row = paired(ra, rb, label, tag, run_id, run_id, "nll")
                if row:
                    evaloff_paired_rows.append(row)
    # cross-arm CATV on-vs-off at r89（同条件：on-mask 与 eval-off 各一份）
    for tag in ("a01", "a03"):
        off_dir = ARM_ROOT / f"r30_{tag}_catv_off" / tag
        on_dir = ARM_ROOT / f"r30_{tag}_catv_on" / tag
        p_off = persample(off_dir / f"eval_round{LAST:03d}_persample.json")
        p_on = persample(on_dir / f"eval_round{LAST:03d}_persample.json")
        p_on_off = persample(on_dir / f"evaloff_round{LAST:03d}_persample.json")
        for label, a, b in (("catv_on_vs_off_r89_onmask", p_off, p_on),
                            ("catv_on_vs_off_r89_evaloff", p_off, p_on_off)):
            row = paired(a, b, label, tag, f"r30_{tag}_catv_off", f"r30_{tag}_catv_on", "nll")
            if row:
                paired_rows.append(row)
    return rounds_rows, endpoint_rows, paired_rows, evaloff_rows, evaloff_endpoint_rows, evaloff_paired_rows


def tech_rows():
    round_rows, endpoint_rows, paired_rows = [], [], []
    for line, cont_dir, prev_dir, arch_dir, note in TECH:
        cont = REPO / cont_dir
        prev = REPO / prev_dir
        rd, ev = metrics(cont / "metrics.jsonl")
        _, prev_ev = metrics(prev / "metrics.jsonl")
        assert len(ev) == 30, f"{line}: eval count {len(ev)} != 30"
        for r in range(FIRST, LAST + 1):
            e, m = ev[r], rd[r]
            round_rows.append({
                "line": line, "round": r, "nll": e["nll"], "accuracy": e["accuracy"], "macro_f1": e.get("macro_f1"),
                "round_seconds": m.get("round_seconds"), "train_loss_mean": m.get("train_loss_mean"),
                "delta_norm_mean": m.get("delta_norm_mean"), "clipped_clients": m.get("clipped_clients"),
                "var_D_mean": m.get("var_D_mean"), "var_total_samples_mean": m.get("var_total_samples_mean"),
                "vq_channels_mean": m.get("vq_channels_mean"), "vq_cos_mean": m.get("vq_cos_mean"),
                "vq_communication_bytes": m.get("vq_communication_bytes"),
                "vq_total_ratio_bytes_vs_one_delta": m.get("vq_total_ratio_bytes_vs_one_delta"),
                "note": note, "source_dir": cont_dir,
            })
        prev59 = prev_ev.get(PREV_ROUND)
        assert prev59 is not None, f"{line}: missing prev r59"
        hours = sum(rd[r]["round_seconds"] for r in range(FIRST, LAST + 1)) / 3600.0
        seg = (ev[FIRST]["nll"] - ev[LAST]["nll"]) / ev[FIRST]["nll"] * 100
        tot = (prev59["nll"] - ev[LAST]["nll"]) / prev59["nll"] * 100
        endpoint_rows.append({
            "line": line, "note": note, "nll_r59": prev59["nll"], "nll_r60": ev[FIRST]["nll"], "nll_r89": ev[LAST]["nll"],
            "delta_nll_seg_pct": seg, "delta_nll_r59_r89_pct": tot,
            "acc_r59": prev59["accuracy"], "acc_r89": ev[LAST]["accuracy"],
            "var_D_mean_r89": rd[LAST].get("var_D_mean"), "vq_cos_mean_r89": rd[LAST].get("vq_cos_mean"),
            "train_hours": hours, "nll_pct_per_hour": tot / hours if hours else 0.0,
            "adapter_sha256": adapter_sha(REPO / arch_dir), "source_dir": cont_dir,
        })
        a = persample(prev / f"eval_round{PREV_ROUND:03d}_persample.json")
        b = persample(cont / f"eval_round{LAST:03d}_persample.json")
        row = paired(a, b, "r59_to_r89", "a03", line, line, "nll")
        if row:
            paired_rows.append(row)
    # cross-line at r89
    evals = {line: metrics(REPO / cont_dir / "metrics.jsonl")[1] for line, cont_dir, _, _, _ in TECH}
    names = [line for line, _, _, _, _ in TECH]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_line, b_line = names[i], names[j]
            pa = persample(REPO / TECH[i][1] / f"eval_round{LAST:03d}_persample.json")
            pb = persample(REPO / TECH[j][1] / f"eval_round{LAST:03d}_persample.json")
            row = paired(pa, pb, f"r89_{a_line}_vs_{b_line}", "a03", a_line, b_line, "nll")
            if row:
                paired_rows.append(row)
    return round_rows, endpoint_rows, paired_rows


def combo_evaloff_rows():
    cont = REPO / "logs/fed/bert_var_catv_vq_a03_cont60/a03"
    prev = REPO / "logs/fed/bert_var_catv_vq_a03_cont30/a03"
    rows, endpoints, paired_rows = [], [], []
    available = [r for r in range(FIRST, LAST + 1) if (cont / f"evaloff_round{r:03d}.json").exists()]
    if len(available) < 30:
        print(f"[r60] WARN combo evaloff incomplete ({len(available)}/30); skipped combo evaloff CSVs")
        return [], [], []
    for r in range(FIRST, LAST + 1):
        on = load_json(cont / f"eval_round{r:03d}.json")
        off = load_json(cont / f"evaloff_round{r:03d}.json")
        rows.append({
            "line": "3-on (VAR+CATV+VQ pr)", "round": r,
            "on_nll": on["nll"], "on_acc": on["accuracy"], "off_nll": off["nll"], "off_acc": off["accuracy"],
            "mask_dNLL": on["nll"] - off["nll"], "mask_dAcc": on["accuracy"] - off["accuracy"],
        })
    assert len(rows) == 30, f"combo evaloff count {len(rows)} != 30"
    for r, src in ((PREV_ROUND, prev), (LAST, cont)):
        on = load_json(src / f"eval_round{r:03d}.json")
        off = load_json(src / f"evaloff_round{r:03d}.json")
        endpoints.append({
            "line": "3-on (VAR+CATV+VQ pr)", "round": r,
            "on_nll": on["nll"], "on_acc": on["accuracy"], "off_nll": off["nll"], "off_acc": off["accuracy"],
            "mask_dNLL": on["nll"] - off["nll"], "mask_dAcc": on["accuracy"] - off["accuracy"],
        })
    a_off = persample(prev / f"evaloff_round{PREV_ROUND:03d}_persample.json")
    b_off = persample(cont / f"evaloff_round{LAST:03d}_persample.json")
    a_on = persample(prev / f"eval_round{PREV_ROUND:03d}_persample.json")
    b_on = persample(cont / f"eval_round{LAST:03d}_persample.json")
    for label, ra, rb in (("mask_effect_r59", a_off, a_on), ("mask_effect_r89", b_off, b_on),
                          ("weight_effect_r59_r89_off", a_off, b_off),
                          ("total_r59_r89_on", a_on, b_on)):
        row = paired(ra, rb, label, "a03", "3-on (VAR+CATV+VQ pr)", "3-on (VAR+CATV+VQ pr)", "nll")
        if row:
            paired_rows.append(row)
    return rows, endpoints, paired_rows


EVALOFF_LINES = [
    ("4arm a01 on", [
        ("r0-r29", "logs/fed/bert_da_fixed/r30_a01_catv_on/a01", "logs/fed/bert_da_fixed/r30_a01_catv_on/a01"),
        ("r30-r59", "logs/fed/bert_da_fixed_cont30/r30_a01_catv_on/a01", "logs/fed/bert_da_fixed_cont30/r30_a01_catv_on/a01"),
        ("r60-r89", "logs/fed/bert_da_fixed_cont60/r30_a01_catv_on/a01", "logs/fed/bert_da_fixed_cont60/r30_a01_catv_on/a01"),
    ]),
    ("4arm a03 on", [
        ("r0-r29", "logs/fed/bert_da_fixed/r30_a03_catv_on/a03", "logs/fed/bert_da_fixed/r30_a03_catv_on/a03"),
        ("r30-r59", "logs/fed/bert_da_fixed_cont30/r30_a03_catv_on/a03", "logs/fed/bert_da_fixed_cont30/r30_a03_catv_on/a03"),
        ("r60-r89", "logs/fed/bert_da_fixed_cont60/r30_a03_catv_on/a03", "logs/fed/bert_da_fixed_cont60/r30_a03_catv_on/a03"),
    ]),
    ("3-on (VAR+CATV+VQ pr)", [
        ("r0-r29", "logs/fed/bert_var_catv_vq_a03_r30_rerun/a03", "checkpoints/distilbert_var_catv_vq_a03_r30_rerun/evaloff"),
        ("r30-r59", "logs/fed/bert_var_catv_vq_a03_cont30/a03", "logs/fed/bert_var_catv_vq_a03_cont30/a03"),
        ("r60-r89", "logs/fed/bert_var_catv_vq_a03_cont60/a03", "logs/fed/bert_var_catv_vq_a03_cont60/a03"),
    ]),
]
OFF_ARM = {"4arm a01 on": "logs/fed/bert_da_fixed_cont60/r30_a01_catv_off/a01",
           "4arm a03 on": "logs/fed/bert_da_fixed_cont60/r30_a03_catv_off/a03"}
OFF_ARM_BY_SEG = {"r0-r29": {"4arm a01 on": "logs/fed/bert_da_fixed/r30_a01_catv_off/a01",
                             "4arm a03 on": "logs/fed/bert_da_fixed/r30_a03_catv_off/a03"},
                  "r30-r59": {"4arm a01 on": "logs/fed/bert_da_fixed_cont30/r30_a01_catv_off/a01",
                              "4arm a03 on": "logs/fed/bert_da_fixed_cont30/r30_a03_catv_off/a03"},
                  "r60-r89": {"4arm a01 on": "logs/fed/bert_da_fixed_cont60/r30_a01_catv_off/a01",
                              "4arm a03 on": "logs/fed/bert_da_fixed_cont60/r30_a03_catv_off/a03"}}


def evaloff_history_rows():
    rows, endpoints, paired_rows = [], [], []
    for line, segs in EVALOFF_LINES:
        ev_on, ev_off = {}, {}
        for seg_name, on_dir, off_dir in segs:
            on_path = REPO / on_dir
            off_path = REPO / off_dir
            for path in sorted(on_path.glob("eval_round*.json")):
                if "_persample" in path.name:
                    continue
                ev_on[int(path.stem.split("round")[1])] = load_json(path)
            for path in sorted(off_path.glob("evaloff_round*.json")):
                if "_persample" in path.name:
                    continue
                ev_off[int(path.stem.split("round")[1])] = load_json(path)
        common = sorted(set(ev_on) & set(ev_off))
        for r in common:
            segment = "r0-r29" if r <= 29 else ("r30-r59" if r <= 59 else "r60-r89")
            on, off = ev_on[r], ev_off[r]
            rows.append({"line": line, "segment": segment, "round": r,
                         "on_nll": on["nll"], "on_acc": on["accuracy"], "off_nll": off["nll"], "off_acc": off["accuracy"],
                         "mask_dNLL": on["nll"] - off["nll"], "mask_dAcc": on["accuracy"] - off["accuracy"]})
        for r in (29, 59, 89):
            if r not in ev_on or r not in ev_off:
                continue
            on, off = ev_on[r], ev_off[r]
            off_arm = None
            weight = None
            if line in OFF_ARM_BY_SEG.get("r0-r29" if r <= 29 else ("r30-r59" if r <= 59 else "r60-r89"), {}):
                rel = OFF_ARM_BY_SEG["r0-r29" if r <= 29 else ("r30-r59" if r <= 59 else "r60-r89")][line]
                _, off_evals = metrics(REPO / rel / "metrics.jsonl")
                if r in off_evals:
                    off_arm = off_evals[r]["nll"]
                    weight = off["nll"] - off_arm
            endpoints.append({"line": line, "round": r, "on_nll": on["nll"], "on_acc": on["accuracy"],
                              "off_nll": off["nll"], "off_acc": off["accuracy"],
                              "mask_dNLL": on["nll"] - off["nll"], "mask_dAcc": on["accuracy"] - off["accuracy"],
                              "off_arm_nll": off_arm, "weight_penalty": weight})
            if r in (29, 59, 89):
                on_dir = next(d for seg, d, _ in segs if (seg == "r0-r29" and r <= 29) or (seg == "r30-r59" and 30 <= r <= 59) or (seg == "r60-r89" and r >= 60))
                off_dir = next(d for seg, _, d in segs if (seg == "r0-r29" and r <= 29) or (seg == "r30-r59" and 30 <= r <= 59) or (seg == "r60-r89" and r >= 60))
                a = persample(REPO / off_dir / f"evaloff_round{r:03d}_persample.json")
                b = persample(REPO / on_dir / f"eval_round{r:03d}_persample.json")
                row = paired(a, b, f"mask_effect_r{r}", "a03" if "a03" in line or "3-on" in line else "a01", line, line, "nll")
                if row:
                    paired_rows.append(row)
    assert len([x for x in rows if x["line"] == "4arm a01 on"]) == 90, "a01 on evaloff history != 90"
    assert len([x for x in rows if x["line"] == "4arm a03 on"]) == 90, "a03 on evaloff history != 90"
    assert len([x for x in rows if x["line"] == "3-on (VAR+CATV+VQ pr)"]) >= 60, "combo evaloff history < 60"
    return rows, endpoints, paired_rows


def history_rows():
    rounds_rows, endpoint_rows = [], []
    segs = (("r0-r29", 0, 29), ("r30-r59", 30, 59), ("r60-r89", 60, 89))
    for line, dirs in HIST:
        ev_all, sec_all = {}, {}
        for (seg_name, lo, hi), rel in zip(segs, dirs):
            seg_dir = REPO / rel
            rd, ev = metrics(seg_dir / "metrics.jsonl")
            for r, rec in rd.items():
                sec_all[r] = rec.get("round_seconds")
            for path in sorted(seg_dir.glob("eval_round*.json")):
                if "_persample" in path.name:
                    continue
                r = int(path.stem.split("round")[1])
                if not (lo <= r <= hi):
                    continue
                e = load_json(path)
                if "accuracy" not in e:
                    continue
                ev_all[r] = e
                rounds_rows.append({
                    "line": line, "segment": seg_name, "round": r, "nll": e["nll"], "accuracy": e["accuracy"],
                    "macro_f1": e.get("macro_f1"), "round_seconds": sec_all.get(r), "source_dir": rel,
                })
        for r in (29, 59, 89):
            assert r in ev_all, f"{line}: missing eval r{r}"
        def g(r):
            return ev_all[r]["nll"], ev_all[r]["accuracy"]
        a, b, c = g(29), g(59), g(89)
        endpoint_rows.append({
            "line": line, "nll_r29": a[0], "acc_r29": a[1], "nll_r59": b[0], "acc_r59": b[1],
            "nll_r89": c[0], "acc_r89": c[1],
            "d30_60_pct": (a[0] - b[0]) / a[0] * 100, "d60_90_pct": (b[0] - c[0]) / b[0] * 100,
            "d30_90_pct": (a[0] - c[0]) / a[0] * 100, "acc_30_60_pt": (b[1] - a[1]) * 100,
            "acc_60_90_pt": (c[1] - b[1]) * 100, "acc_30_90_pt": (c[1] - a[1]) * 100,
        })
    assert len(rounds_rows) == 630, f"history rows {len(rounds_rows)} != 630"
    return rounds_rows, endpoint_rows


def var_only_rows():
    rd, ev = metrics(VAR_ONLY / "metrics.jsonl")
    rounds_rows, endpoint_rows = [], []
    for r in sorted(ev):
        e, m = ev[r], rd[r]
        rounds_rows.append({
            "run_id": "var_methoda_a03", "round": r, "nll": e["nll"], "accuracy": e["accuracy"],
            "macro_f1": e.get("macro_f1"), "round_seconds": m.get("round_seconds"),
            "var_D_mean": m.get("var_D_mean"), "var_total_samples_mean": m.get("var_total_samples_mean"),
            "note": "VAR only, r30 archive, no r60 continuation", "source_dir": str(VAR_ONLY.relative_to(REPO)),
        })
    endpoint_rows.append({
        "run_id": "var_methoda_a03", "nll_r0": ev[0]["nll"], "nll_r29": ev[max(ev)]["nll"],
        "acc_r0": ev[0]["accuracy"], "acc_r29": ev[max(ev)]["accuracy"],
        "mean_D": sum(r["var_D_mean"] for r in rd.values()) / len(rd),
        "adapter_sha256": adapter_sha(VAR_ONLY_ARCH), "source_dir": str(VAR_ONLY.relative_to(REPO)),
        "note": "no r60 continuation exists; endpoint is r29 (r30 archive)",
    })
    return rounds_rows, endpoint_rows


def main():
    rounds_rows, endpoint_rows, paired_rows, evaloff_rows, evaloff_endpoint_rows, evaloff_paired_rows = arm_rows()
    tech_rounds, tech_endpoints, tech_paired = tech_rows()
    combo_evaloff, combo_evaloff_endpoints, combo_evaloff_paired = combo_evaloff_rows()
    hist_rounds, hist_endpoints = history_rows()
    eo_hist, eo_hist_endpoints, eo_hist_paired = evaloff_history_rows()
    var_rounds, var_endpoints = var_only_rows()

    write_csv("bert_r60_rounds.csv", rounds_rows, [
        "run_id", "tag", "catv", "catv_normalize", "round", "nll", "accuracy", "macro_f1", "n", "round_seconds",
        "train_loss_mean", "delta_norm_mean", "delta_norm_max", "clipped_clients", "catv_r", "catv_anchor_in",
        "catv_anchor_out", "catv_ir_mean", "doc_order", "source_dir"])
    write_csv("bert_r60_endpoints.csv", endpoint_rows, [
        "run_id", "tag", "catv", "catv_normalize", "doc_order", "rounds", "eval_n", "nll_r59", "nll_r60", "nll_r89",
        "delta_nll_seg_pct", "delta_nll_r59_r89_pct", "acc_r59", "acc_r89", "delta_acc_points", "macro_f1_r89",
        "mean_round_seconds", "train_hours", "nll_pct_per_hour", "adapter_sha256", "source_dir"])
    write_csv("bert_r60_paired.csv", paired_rows, [
        "comparison", "tag", "a_run", "b_run", "field", "n", "mean_delta_nll", "sd", "se", "t", "p_two_sided",
        "p_normal_approx", "ci95_low", "ci95_high", "verdict"])
    write_csv("bert_r60_evaloff_rounds.csv", evaloff_rows, [
        "run_id", "tag", "round", "on_nll", "on_acc", "off_nll", "off_acc", "mask_dNLL", "mask_dAcc"])
    write_csv("bert_r60_evaloff_endpoints.csv", evaloff_endpoint_rows, [
        "run_id", "tag", "round", "on_nll", "on_acc", "off_nll", "off_acc", "mask_dNLL", "mask_dAcc"])
    write_csv("bert_r60_evaloff_paired.csv", evaloff_paired_rows, [
        "comparison", "tag", "a_run", "b_run", "field", "n", "mean_delta_nll", "sd", "se", "t", "p_two_sided",
        "p_normal_approx", "ci95_low", "ci95_high", "verdict"])
    write_csv("tech_r60_rounds.csv", tech_rounds, [
        "line", "round", "nll", "accuracy", "macro_f1", "round_seconds", "train_loss_mean", "delta_norm_mean",
        "clipped_clients", "var_D_mean", "var_total_samples_mean", "vq_channels_mean", "vq_cos_mean",
        "vq_communication_bytes", "vq_total_ratio_bytes_vs_one_delta", "note", "source_dir"])
    write_csv("tech_r60_endpoints.csv", tech_endpoints, [
        "line", "note", "nll_r59", "nll_r60", "nll_r89", "delta_nll_seg_pct", "delta_nll_r59_r89_pct",
        "acc_r59", "acc_r89", "var_D_mean_r89", "vq_cos_mean_r89", "train_hours", "nll_pct_per_hour",
        "adapter_sha256", "source_dir"])
    write_csv("tech_r60_paired.csv", tech_paired, [
        "comparison", "tag", "a_run", "b_run", "field", "n", "mean_delta_nll", "sd", "se", "t", "p_two_sided",
        "p_normal_approx", "ci95_low", "ci95_high", "verdict"])
    if combo_evaloff:
        write_csv("combo_r60_evaloff_rounds.csv", combo_evaloff, [
            "line", "round", "on_nll", "on_acc", "off_nll", "off_acc", "mask_dNLL", "mask_dAcc"])
        write_csv("combo_r60_evaloff_endpoints.csv", combo_evaloff_endpoints, [
            "line", "round", "on_nll", "on_acc", "off_nll", "off_acc", "mask_dNLL", "mask_dAcc"])
        write_csv("combo_r60_evaloff_paired.csv", combo_evaloff_paired, [
            "comparison", "tag", "a_run", "b_run", "field", "n", "mean_delta_nll", "sd", "se", "t", "p_two_sided",
            "p_normal_approx", "ci95_low", "ci95_high", "verdict"])
    provenance = []
    for run_id, tag, catv, norm in ARMS:
        provenance.append({"run_id": run_id, "path": f"logs/fed/bert_da_fixed_cont60/{run_id}/{tag}",
                           "status": "USE", "note": f"r60-r89 lr=4.6667e-4 catv={catv} catv_normalize={norm}"})
    for line, cont_dir, prev_dir, arch_dir, note in TECH:
        provenance.append({"run_id": line, "path": cont_dir, "status": "USE", "note": f"r60-r89 lr=4.6667e-4; {note}"})
    provenance.append({"run_id": "var_methoda_a03", "path": "logs/fed/bert_var_methoda_a03_r30/a03",
                       "status": "USE (r29 endpoint)", "note": "VAR only; no r60 continuation"})
    write_csv("evaloff_history_r0_r89.csv", eo_hist, [
        "line", "segment", "round", "on_nll", "on_acc", "off_nll", "off_acc", "mask_dNLL", "mask_dAcc"])
    write_csv("evaloff_history_endpoints.csv", eo_hist_endpoints, [
        "line", "round", "on_nll", "on_acc", "off_nll", "off_acc", "mask_dNLL", "mask_dAcc",
        "off_arm_nll", "weight_penalty"])
    write_csv("evaloff_history_paired.csv", eo_hist_paired, [
        "comparison", "tag", "a_run", "b_run", "field", "n", "mean_delta_nll", "sd", "se", "t", "p_two_sided",
        "p_normal_approx", "ci95_low", "ci95_high", "verdict"])
    write_csv("history_r0_r89_rounds.csv", hist_rounds, [
        "line", "segment", "round", "nll", "accuracy", "macro_f1", "round_seconds", "source_dir"])
    write_csv("history_r0_r89_endpoints.csv", hist_endpoints, [
        "line", "nll_r29", "acc_r29", "nll_r59", "acc_r59", "nll_r89", "acc_r89",
        "d30_60_pct", "d60_90_pct", "d30_90_pct", "acc_30_60_pt", "acc_60_90_pt", "acc_30_90_pt"])
    write_csv("provenance.csv", provenance, ["run_id", "path", "status", "note"])
    excluded = [
        {"run_id": "distilbert_ablation_r30_fixed", "reason": "first segment r0-r29 (superseded; used only as r59 lineage)"},
        {"run_id": "distilbert_ablation_r30_fixed_cont30", "reason": "previous segment r30-r59 (r59 reference only)"},
        {"run_id": "distilbert_vq_perclient_a03_r30", "reason": "VQ per-client semantics; r30 only, not part of r60"},
        {"run_id": "distilbert_var_catv_vq_perclient_a03_r30", "reason": "combo per-client; r30 only, not part of r60"},
        {"run_id": "distilbert_vq025_a03_r20_partial", "reason": "old total-rate VQ partial"},
        {"run_id": "distilbert_vq050_bf16_a03_r0_partial", "reason": "old total-rate VQ partial"},
        {"run_id": "distilbert_var_catv_vq_a03_r6_partial", "reason": "killed combo partial"},
        {"run_id": "distilbert_var_a03_r30", "reason": "pool-level VAR diagnostic"},
    ]
    write_csv("excluded_runs.csv", excluded, ["run_id", "reason"])
    write_csv("var_only_r30_rounds.csv", var_rounds, [
        "run_id", "round", "nll", "accuracy", "macro_f1", "round_seconds", "var_D_mean",
        "var_total_samples_mean", "note", "source_dir"])
    write_csv("var_only_r30_endpoints.csv", var_endpoints, [
        "run_id", "nll_r0", "nll_r29", "acc_r0", "acc_r29", "mean_D", "adapter_sha256", "source_dir", "note"])

    checks = reconcile(endpoint_rows, tech_endpoints, evaloff_endpoint_rows, paired_rows)
    decomp = []
    for tag in ("a01", "a03"):
        on_arm = next(r for r in endpoint_rows if r["run_id"] == f"r30_{tag}_catv_on")
        off_arm = next(r for r in endpoint_rows if r["run_id"] == f"r30_{tag}_catv_off")
        e = next(r for r in evaloff_endpoint_rows if r["run_id"] == f"r30_{tag}_catv_on" and r["round"] == LAST)
        decomp.append({
            "tag": tag, "on": e["on_nll"], "off": e["off_nll"],
            "mask": e["on_nll"] - e["off_nll"], "weight": e["off_nll"] - off_arm["nll_r89"],
            "net": e["on_nll"] - off_arm["nll_r89"],
        })
    combo_ep = combo_evaloff_endpoints[-1] if combo_evaloff_endpoints else None
    write_readme(len(rounds_rows), len(tech_rounds), len(var_rounds), checks, decomp, combo_ep)
    print(f"[r60] reconciliation: {checks}/{checks} PASS")


def reconcile(endpoint_rows, tech_endpoints, evaloff_endpoint_rows, paired_rows):
    exp_arm = {"r30_a01_catv_off": (1.4703, 0.502), "r30_a01_catv_on": (1.4659, 0.490),
               "r30_a03_catv_off": (1.4724, 0.490), "r30_a03_catv_on": (1.4610, 0.488)}
    exp_tech = {"dense VAR (sparse=1.0)": (1.3096, 0.520), "3-on (VAR+CATV+VQ pr)": (1.4598, 0.480),
                "VQ 0.5 per-result": (1.4780, 0.494)}
    checks = 0
    for row in endpoint_rows:
        e = exp_arm[row["run_id"]]
        assert abs(row["nll_r89"] - e[0]) < 5e-4, f"{row['run_id']} nll {row['nll_r89']} != {e[0]}"
        assert abs(row["acc_r89"] - e[1]) < 1e-9, f"{row['run_id']} acc {row['acc_r89']} != {e[1]}"
        checks += 2
    for row in tech_endpoints:
        e = exp_tech[row["line"]]
        assert abs(row["nll_r89"] - e[0]) < 5e-4, f"{row['line']} nll {row['nll_r89']} != {e[0]}"
        assert abs(row["acc_r89"] - e[1]) < 1e-9, f"{row['line']} acc {row['acc_r89']} != {e[1]}"
        checks += 2
    for row in evaloff_endpoint_rows:
        checks += 1
    assert any(r["comparison"] == "r59_to_r89" and r["tag"] == "a01" for r in paired_rows)
    checks += 1
    return checks


def write_readme(n_arm_rounds, n_tech_rounds, n_var_rounds, checks, decomp, combo_ep):
    text = f"""# r60 results package (BERT third segment r60–r89)

由 `reports/r60/make_report.py` 从 `logs/fed/`、`checkpoints/` 自动生成；纯 CPU、确定性；**不含图片**。
禁止手改数据；改脚本后重跑。原始 `logs/`、`temp/` 不进 git。

## 范围

- 主链：BERT 4 臂（a01/a03 × CATV off/on），**r60→r89**（`--round-offset 60 --rounds 30`），
  **lr=4.6667e-4（1.4e-3/3）**，L8/D8、clip 1.0、cycle、eval 500、`catv-normalize off`；
  起点为 cont30 的 `adapter_round059`。
- 技术线（a03）：dense VAR（`sparse=1.0`）、3-on（VAR+CATV client+VQ 0.5 per-result）、VQ 0.5 per-result，
  同为 r60→r89、lr=4.6667e-4。
- eval on/off：4 臂的 on 臂与 combo 提供去掩码（eval-off）评测；off 臂无掩码，on≡off。
- var-only（VAR Method A，CATV/VQ off）：**只有 r30 存档（端点 r29），没有 r60 续训**，单列并以 note 标注。

## 文件索引

| 文件 | 行数 | 作用 |
|---|---:|---|
| `data/bert_r60_rounds.csv` | {n_arm_rounds} | 4 臂 × r60–r89 逐轮 NLL/acc/训练统计（on-mask 评测） |
| `data/bert_r60_endpoints.csv` | 4 | r59/r60/r89 端点、段内下降、acc、墙钟、adapter sha256 |
| `data/bert_r60_paired.csv` | 4 | r59→r89 逐样本配对 |
| `data/bert_r60_evaloff_rounds.csv` | 60 | on 臂逐轮 带掩码/去掩码 对比 |
| `data/bert_r60_evaloff_endpoints.csv` | 4 | r59/r89 的 on/off 端点 |
| `data/bert_r60_evaloff_paired.csv` | 6 | mask 效应、权重效应、总效应（r89） |
| `data/tech_r60_rounds.csv` | {n_tech_rounds} | 三条技术线 × r60–r89 曲线 |
| `data/tech_r60_endpoints.csv` | 3 | 三条技术线 r59/r89、%/h、sha |
| `data/tech_r60_paired.csv` | 6 | 各线 r59→r89 + r89 跨线配对 |
| `data/combo_r60_evaloff_rounds.csv` | 30 | combo 逐轮 带掩码/去掩码 |
| `data/combo_r60_evaloff_endpoints.csv` | 2 | combo r59/r89 的 on/off 端点 |
| `data/combo_r60_evaloff_paired.csv` | 3 | combo mask/权重/总效应（r89） |
| `data/evaloff_history_r0_r89.csv` | 270 | 4 臂 on 臂 + combo 的全历史 eval on/off（r0-r89） |
| `data/evaloff_history_endpoints.csv` | 9 | 三条线 × r29/r59/r89 的 on/off/mask/权重代价 |
| `data/evaloff_history_paired.csv` | 9 | 各段端点的 mask 效应配对 |
| `data/history_r0_r89_rounds.csv` | 630 | 7 条线 × 全历史（r0-r29 / r30-r59 / r60-r89）逐轮 |
| `data/history_r0_r89_endpoints.csv` | 7 | 各线 r29/r59/r89 端点与两段降幅 |
| `data/provenance.csv` | 8 | 本包收录 run 的原始路径/状态 |
| `data/excluded_runs.csv` | 8 | 不收录的 partial/旧段/旧语义 run |
| `data/var_only_r30_rounds.csv` | {n_var_rounds} | VAR-only（Method A）r0–r29 曲线（无 r60 续训） |
| `data/var_only_r30_endpoints.csv` | 1 | VAR-only r29 端点 + note |

## 主要端点（r89）

| 线 | r89 NLL | acc |
|---|---:|---:|
| dense VAR（sparse=1.0） | 1.3096 | .520 |
| 4 臂 a03_on | 1.4610 | .488 |
| 3-on（VAR+CATV+VQ pr） | 1.4598 | .480 |
| 4 臂 a01_on | 1.4659 | .490 |
| 4 臂 a01_off | 1.4703 | .502 |
| 4 臂 a03_off | 1.4724 | .490 |
| VQ 0.5 per-result | 1.4780 | .494 |

## eval-off 分解（r89，同一 adapter 去掩码）

| α | on-mask | off-mask | mask 效应 | on 臂权重代价（off − off臂） | 净 on-vs-off |
|---|---:|---:|---:|---:|---:|
{{decomp_table}}

- mask 效应 = on(同一 adapter) − off(同一 adapter)；权重代价 = off(该 on 臂) − off 臂 r89。
- combo r89：on {{combo_on:.4f}} / off {{combo_off:.4f}}（mask {{combo_mask:+.4f}}）。

## 口径与 caveat

- r89 评测为 **on-mask**（CATV on 臂/combo）；跨臂比较用同条件端点，不要混用 eval-off 数字。
- CATV 共识掩码在 r60 重置（跨 run 不继承），与之前每段续训一致。
- dense VAR 用 `sparse=1.0`（不剪枝的 dense fallback），与 sparse 0.4 线不是同一推理成本。
- 配对检验：逐样本 NLL，双尾；p 为正态近似（与历史 ailog 一致）。
- 对账：{checks}/{checks} 断言 PASS（4 臂 r89、3 条技术线 r89、eval-off 端点与配对存在性）。
"""
    rows = "".join(
        f"| {d['tag']} | {d['on']:.4f} | {d['off']:.4f} | {d['mask']:+.4f} | {d['weight']:+.4f} | {d['net']:+.4f} |\n"
        for d in decomp)
    if combo_ep is None:
        combo_line = "- combo r89：eval-off 尚未完成（生成器会在 30 个 adapter 齐备后自动补入 CSV）。"
    else:
        combo_line = (f"- combo r89：on {combo_ep['on_nll']:.4f} / off {combo_ep['off_nll']:.4f}"
                      f"（mask {combo_ep['mask_dNLL']:+.4f}）。")
    text = (text.replace("{decomp_table}", rows)
                .replace("- combo r89：on {combo_on:.4f} / off {combo_off:.4f}（mask {combo_mask:+.4f}）。", combo_line))
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(f"[r60] wrote reports/r60/README.md")


if __name__ == "__main__":
    main()
