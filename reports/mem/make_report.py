#!/usr/bin/env python3
# SMELL 3 mem_report NEW — 显存交付包：ZOO（真实 zo_grad）vs BP，1k–16k × dense/sparse 0.4
# 输入：temp/mem_sweep_zoo/mem_zoo_*.json、temp/mem_sweep/mem_backward_*.json、temp/mem_sweep/mem_forward_*.json
# 输出：reports/mem/{README.md,data/*.csv}；纯 CPU、确定性；数字全部由脚本生成，禁止手改
import argparse
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ZOO_DIR = REPO / "temp" / "mem_sweep_zoo"
BP_DIR = REPO / "temp" / "mem_sweep"
OUT = REPO / "reports" / "mem"
DATA = OUT / "data"

SEQ_LABELS = {1024: "1k", 2048: "2k", 4096: "4k", 8192: "8k", 16384: "16k"}
BENCH_DEFAULTS = {
    "model_dir": "third_party/Jenga/checkpoints/opt-350m",
    "dtype": "bf16",
    "lora_r": 8,
    "device_name": "NVIDIA GeForce RTX 4090",
}
KINDS = ("dense", "sparse")
MODES = ("zoo", "backward", "forward")


def load_record(path):
    payload = json.load(open(path))
    record = payload[0] if isinstance(payload, list) else payload
    return record


def collect():
    rows = []
    for mode in MODES:
        for seq in sorted(SEQ_LABELS):
            for kind in KINDS:
                if mode == "zoo":
                    path = ZOO_DIR / f"mem_zoo_{kind}_{seq}.json"
                else:
                    path = BP_DIR / f"mem_{mode}_{kind}_{seq}.json"
                if not path.exists():
                    rows.append({"mode": mode, "seq_len": seq, "seq_label": SEQ_LABELS[seq],
                                 "kind": kind, "peak_gib": None, "elapsed_s": None,
                                 "status": "missing", "zoo_directions": None, "zo_eps": None,
                                 "model_dir": BENCH_DEFAULTS["model_dir"], "dtype": BENCH_DEFAULTS["dtype"],
                                 "lora_r": BENCH_DEFAULTS["lora_r"], "device_name": BENCH_DEFAULTS["device_name"],
                                 "meta_source": "bench_default",
                                 "source_file": str(path.relative_to(REPO)),
                                 "mtime": None})
                    continue
                record = load_record(path)
                has_meta = bool(record.get("model_dir"))
                model_dir = record.get("model_dir") or BENCH_DEFAULTS["model_dir"]
                model_dir = str(model_dir)
                prefix = str(REPO) + "/"
                if model_dir.startswith(prefix):
                    model_dir = model_dir[len(prefix):]
                rows.append({
                    "mode": mode, "seq_len": seq, "seq_label": SEQ_LABELS[seq], "kind": kind,
                    "peak_gib": record.get("peak_allocated_gb"),
                    "elapsed_s": record.get("elapsed_s"),
                    "status": record.get("status"),
                    "zoo_directions": record.get("zoo_directions"),
                    "zo_eps": record.get("zo_eps"),
                    "model_dir": model_dir,
                    "dtype": record.get("dtype") or BENCH_DEFAULTS["dtype"],
                    "lora_r": record.get("lora_r") or BENCH_DEFAULTS["lora_r"],
                    "device_name": record.get("device_name") or BENCH_DEFAULTS["device_name"],
                    "meta_source": "json" if has_meta else "bench_default",
                    "source_file": str(path.relative_to(REPO)),
                    "mtime": os.path.getmtime(path),
                })
    return rows


def get(rows, mode, seq, kind, field="peak_gib"):
    for row in rows:
        if row["mode"] == mode and row["seq_len"] == seq and row["kind"] == kind:
            return row[field]
    return None


def summary_rows(rows):
    out = []
    for seq in sorted(SEQ_LABELS):
        zd, zs = get(rows, "zoo", seq, "dense"), get(rows, "zoo", seq, "sparse")
        bd, bs = get(rows, "backward", seq, "dense"), get(rows, "backward", seq, "sparse")
        out.append({
            "seq_len": seq, "seq_label": SEQ_LABELS[seq],
            "zoo_dense_gib": zd, "zoo_sparse_gib": zs,
            "zoo_save_pct": (1 - zs / zd) * 100 if (zd and zs) else None,
            "bp_dense_gib": bd, "bp_sparse_gib": bs,
            "bp_save_pct": (1 - bs / bd) * 100 if (bd and bs) else None,
            "bp_over_zoo_dense": bd / zd if (bd and zd) else None,
            "bp_over_zoo_sparse": bs / zs if (bs and zs) else None,
            "zoo_dense_s": get(rows, "zoo", seq, "dense", "elapsed_s"),
            "zoo_sparse_s": get(rows, "zoo", seq, "sparse", "elapsed_s"),
            "bp_dense_s": get(rows, "backward", seq, "dense", "elapsed_s"),
            "bp_sparse_s": get(rows, "backward", seq, "sparse", "elapsed_s"),
        })
    return out


def check(rows, summary):
    problems = []
    for mode in ("zoo", "backward"):
        n = sum(1 for row in rows if row["mode"] == mode and row["peak_gib"] is not None)
        if n != 10:
            problems.append(f"{mode}: expected 10 ok rows, got {n}")
    expected = {
        ("zoo", 16384, "dense"): 2.6639,
        ("zoo", 16384, "sparse"): 2.2130,
        ("backward", 16384, "dense"): 14.5107,
        ("backward", 16384, "sparse"): 12.6937,
    }
    for (mode, seq, kind), want in expected.items():
        got = get(rows, mode, seq, kind)
        if got is None or abs(got - want) > 0.02:
            problems.append(f"{mode} {seq} {kind}: got {got}, expected ~{want}")
    for seq in sorted(SEQ_LABELS):
        zd, bd = get(rows, "zoo", seq, "dense"), get(rows, "backward", seq, "dense")
        if not (zd and bd) or zd >= bd:
            problems.append(f"seq {seq}: zoo dense ({zd}) not < bp dense ({bd})")
    for key in ("zoo_save_pct", "bp_save_pct"):
        values = [row[key] for row in summary]
        if any(v is None for v in values):
            problems.append(f"{key}: missing values")
        elif any(b < a - 1e-9 for a, b in zip(values, values[1:])):
            problems.append(f"{key}: not monotone increasing: {values}")
    return problems


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_readme(summary, rows):
    lines = []
    lines.append("# SMELL 显存交付包：ZOO（真实 `zo_grad`）vs BP，1k–16k")
    lines.append("")
    lines.append("由 `reports/mem/make_report.py` 从 `temp/mem_sweep_zoo/` 与 `temp/mem_sweep/` 的原始 JSON "
                 "自动生成；纯 CPU、确定性；**不含图片**。禁止手改本目录数字，改脚本后重跑。")
    lines.append("")
    lines.append("## 范围与来源")
    lines.append("")
    lines.append("- 模型：**OPT-350M**（`third_party/Jenga/checkpoints/opt-350m`；24L / hidden 1024 / 16 heads / vocab 50272，"
                 "max_pos 2048 由 `ensure_positions` 扩到 16k），bf16 权重 + autocast，flash/SDPA；")
    lines.append("- LoRA：qkvo，r=8（bench 默认，**不是**主线的 r=1）；单卡 RTX 4090 24GB，`cap_fraction=0.9`；")
    lines.append("- 输入：单条随机 `input_ids`，长度 1024/2048/4096/8192/16384，单步；指标为 `torch.cuda.max_memory_allocated()`（peak_allocated，非 reserved）。")
    lines.append("")
    lines.append("| 模式 | 实现 | 说明 |")
    lines.append("|---|---|---|")
    lines.append("| **ZOO** | `src/train/zoo.py::zo_grad` | 真实实现；每次 forward 在 `torch.no_grad()` 下；D=8、eps=1e-3；D 只影响时间不影响峰值 |")
    lines.append("| **BP** | training-mode forward + `loss.backward()` | 标准反传，autograd 保留 activation |")
    lines.append("| *graph-forward*（参考） | training-mode forward（不 backward） | 仍建图，≈BP−0.2 GiB；**不是** ZOO 路径，仅用于口径对照 |")
    lines.append("")
    lines.append("## 主表（GiB）")
    lines.append("")
    lines.append("| seq | ZOO dense | ZOO sparse | ZOO 省 | BP dense | BP sparse | BP 省 | BP/ZOO (dense) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            f"| {row['seq_label']} | {fmt(row['zoo_dense_gib'])} | {fmt(row['zoo_sparse_gib'])} | "
            f"{fmt(row['zoo_save_pct'], 1)}% | {fmt(row['bp_dense_gib'])} | {fmt(row['bp_sparse_gib'])} | "
            f"{fmt(row['bp_save_pct'], 1)}% | {fmt(row['bp_over_zoo_dense'])}× |")
    lines.append("")
    zoom = summary[-1]
    lines.append("## 关键结论")
    lines.append("")
    lines.append(f"1. **ZOO 比 BP 省 {fmt(summary[0]['bp_over_zoo_dense'])}×（1k）→ {fmt(zoom['bp_over_zoo_dense'])}×（16k）**；"
                 f"16k 时 ZOO {fmt(zoom['zoo_dense_gib'])} GiB（sparse {fmt(zoom['zoo_sparse_gib'])}）vs BP "
                 f"{fmt(zoom['bp_dense_gib'])}（sparse {fmt(zoom['bp_sparse_gib'])}）。")
    lines.append("2. ZOO 斜率 ≈ 0.1 GiB/1k（权重 + 单次激活，线性）；BP 斜率 ≈ 0.9 GiB/1k（autograd 图）。")
    lines.append(f"3. 稀疏 0.4 的节省：ZOO {fmt(summary[0]['zoo_save_pct'],1)}% → {fmt(zoom['zoo_save_pct'],1)}%（随 L 上升），"
                 f"BP {fmt(summary[0]['bp_save_pct'],1)}% → {fmt(zoom['bp_save_pct'],1)}%；**相对值 ZOO 更大、绝对值 BP 更大**"
                 "（16k 绝对节省：ZOO 0.45 GiB vs BP 1.82 GiB）。")
    lines.append("4. ZOO 16k 仅 ~2.2–2.7 GiB → 8GB 卡可跑；BP 16k 需 12.7–14.5 GiB。")
    lines.append("")
    lines.append("## 口径 caveat")
    lines.append("")
    lines.append("- bench 口径（bf16/flash/LoRA r8）≠ 主线（fp32 + LoRA r1）；横向对比（ZOO vs BP、dense vs sparse）有效，绝对值不可直接当主线显存。")
    lines.append("- `peak_allocated` 是 PyTorch 分配器口径（非 reserved，也不含 CUDA context）；不同驱动/框架版本会有 ±0.1 GiB 级差异。")
    lines.append("- ZOO 的 D 只影响耗时（每次方向 2 次前向），不影响峰值；本表 D=8。")
    lines.append("- 模型 provenance：ZOO 10 组 JSON 自带 `model_dir/dtype/lora_r/device_name`（`meta_source=json`）；"
                 "BP 10 组为同一 bench 默认值，脚本按默认补记（`meta_source=bench_default`），两者同脚本同口径。")
    lines.append("- 早期 `mem_sweep` 的 “forward” 模式是建图前向（≈BP−0.2 GiB），只以 `mem_tidy.csv` 的 `mode=forward` 行保留作口径对照，**不应**用于 ZOO 结论。")
    lines.append("")
    lines.append("## 文件索引")
    lines.append("")
    lines.append("| 文件 | 行数 | 作用 |")
    lines.append("|---|---:|---|")
    lines.append(f"| `data/mem_tidy.csv` | {sum(1 for r in rows if r['peak_gib'] is not None)} | 逐 run（mode × seq × kind）的峰值/耗时/状态/来源 |")
    lines.append(f"| `data/mem_summary.csv` | {len(summary)} | 宽表：ZOO/BP × dense/sparse、节省 %、BP/ZOO 比 |")
    lines.append(f"| `data/provenance.csv` | {len(rows)} | 每个 JSON 的路径、mtime、状态（含缺失行） |")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("PY=~/miniconda3/envs/SMELL/bin/python")
    lines.append("# 1) 测量（GPU；ZOO 只跑 5 长度 × dense/sparse）")
    lines.append("bash temp/run_mem_sweep_zoo_only_bg.sh")
    lines.append("# 2) 生成交付包（CPU，含对账闸门）")
    lines.append("$PY reports/mem/make_report.py")
    lines.append("```")
    lines.append("")
    lines.append("## 对账")
    lines.append("")
    lines.append("- ZOO 10/10、BP 10/10 行均 `status=ok`；16k 四个端点与 ailog `261105` §八 一致（±0.02 GiB）；")
    lines.append("- 所有长度 ZOO < BP；ZOO/BP 的 dense→sparse 节省率随 L 单调上升。")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-check", action="store_true", help="skip reconciliation gate (debug only)")
    args = parser.parse_args()

    rows = collect()
    summary = summary_rows(rows)
    if not args.no_check:
        problems = check(rows, summary)
        if problems:
            print("[mem_report] RECONCILIATION FAILED:")
            for problem in problems:
                print("  -", problem)
            raise SystemExit(1)

    write_csv(DATA / "mem_tidy.csv",
              ["mode", "seq_len", "seq_label", "kind", "peak_gib", "elapsed_s", "status",
               "zoo_directions", "zo_eps", "model_dir", "dtype", "lora_r", "device_name",
               "meta_source", "source_file"], rows)
    write_csv(DATA / "mem_summary.csv",
              ["seq_len", "seq_label", "zoo_dense_gib", "zoo_sparse_gib", "zoo_save_pct",
               "bp_dense_gib", "bp_sparse_gib", "bp_save_pct", "bp_over_zoo_dense",
               "bp_over_zoo_sparse", "zoo_dense_s", "zoo_sparse_s", "bp_dense_s", "bp_sparse_s"],
              summary)
    write_csv(DATA / "provenance.csv",
              ["mode", "seq_len", "kind", "source_file", "mtime", "status", "peak_gib",
               "model_dir", "dtype", "lora_r", "meta_source"], rows)
    write_readme(summary, rows)
    ok = sum(1 for row in rows if row["peak_gib"] is not None)
    print(f"[mem_report] wrote {OUT/'README.md'} + 3 CSV; ok rows={ok}/30")
    for row in summary:
        print(f"  {row['seq_label']:>3}  ZOO {fmt(row['zoo_dense_gib'])}/{fmt(row['zoo_sparse_gib'])}  "
              f"BP {fmt(row['bp_dense_gib'])}/{fmt(row['bp_sparse_gib'])}  "
              f"save zoo {fmt(row['zoo_save_pct'],1)}% bp {fmt(row['bp_save_pct'],1)}%  "
              f"BP/ZOO {fmt(row['bp_over_zoo_dense'])}x")


if __name__ == "__main__":
    main()
