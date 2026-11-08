#!/usr/bin/env python3
# SMELL 3 export_endpoints NEW — 汇总论文 Evaluation 各表所需的关键 endpoint 权重 + 评测 JSON
# 输出：checkpoints/out/{README.md,MANIFEST.csv,SHA256SUMS,adapters/,evals/,memory/} + tar.gz
# 只收 r29 / r59 / r7 的关键 adapter（不做全轮次归档）；评测数值与论文表逐项对账（不符即退出）。
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "checkpoints" / "out"

DBERT_ARCH = REPO / "checkpoints" / "distilbert_ablation_r30_fixed"
DBERT_CONT = REPO / "checkpoints" / "distilbert_ablation_r30_fixed_cont30"
OPT_ARCH = REPO / "checkpoints" / "opt_ablation_r30_fixed"
LOGS = REPO / "logs" / "fed"

# (paper, group, name, model, adapter dir, eval json, evaloff json, expected metrics, source run)
ENTRIES = [
    # ---------------- Table main: CASC, DistilBERT r29 ----------------
    ("tab:main", "distilbert_casc_r29", "a01_casc_off", "DistilBERT-a01",
     DBERT_ARCH / "r30_a01_catv_off" / "adapter_round029",
     LOGS / "bert_da_fixed" / "r30_a01_catv_off" / "a01" / "eval_round029.json", None,
     {"nll": 1.5836, "accuracy": 0.468}, "bert_da_fixed/r30_a01_catv_off"),
    ("tab:main", "distilbert_casc_r29", "a01_casc_train_mask", "DistilBERT-a01",
     DBERT_ARCH / "r30_a01_catv_on" / "adapter_round029",
     LOGS / "bert_da_fixed" / "r30_a01_catv_on" / "a01" / "evaloff_round029.json", None,
     {"nll": 1.5768, "accuracy": 0.460}, "bert_da_fixed/r30_a01_catv_on"),
    ("tab:main", "distilbert_casc_r29", "a01_casc_train_eval_mask", "DistilBERT-a01",
     DBERT_ARCH / "r30_a01_catv_on" / "adapter_round029",
     LOGS / "bert_da_fixed" / "r30_a01_catv_on" / "a01" / "eval_round029.json", None,
     {"nll": 1.5563, "accuracy": 0.458}, "bert_da_fixed/r30_a01_catv_on"),
    ("tab:main", "distilbert_casc_r29", "a03_casc_off", "DistilBERT-a03",
     DBERT_ARCH / "r30_a03_catv_off" / "adapter_round029",
     LOGS / "bert_da_fixed" / "r30_a03_catv_off" / "a03" / "eval_round029.json", None,
     {"nll": 1.5887, "accuracy": 0.466}, "bert_da_fixed/r30_a03_catv_off"),
    ("tab:main", "distilbert_casc_r29", "a03_casc_train_mask", "DistilBERT-a03",
     DBERT_ARCH / "r30_a03_catv_on" / "adapter_round029",
     LOGS / "bert_da_fixed" / "r30_a03_catv_on" / "a03" / "evaloff_round029.json", None,
     {"nll": 1.5772, "accuracy": 0.462}, "bert_da_fixed/r30_a03_catv_on"),
    ("tab:main", "distilbert_casc_r29", "a03_casc_train_eval_mask", "DistilBERT-a03",
     DBERT_ARCH / "r30_a03_catv_on" / "adapter_round029",
     LOGS / "bert_da_fixed" / "r30_a03_catv_on" / "a03" / "eval_round029.json", None,
     {"nll": 1.5484, "accuracy": 0.454}, "bert_da_fixed/r30_a03_catv_on"),
    # ---------------- Table main: CASC, OPT r29 ----------------
    ("tab:main", "opt_casc_r29", "a01_casc_off", "OPT-350M-a01",
     OPT_ARCH / "r30_a01_catv_off" / "adapter_round029",
     LOGS / "opt_r30_fixed" / "r30_a01_catv_off" / "a01" / "eval_round029.json", None,
     {"full_ppl_token": 31.5334}, "opt_r30_fixed/r30_a01_catv_off"),
    ("tab:main", "opt_casc_r29", "a01_casc_train_mask", "OPT-350M-a01",
     OPT_ARCH / "r30_a01_catv_on" / "adapter_round029",
     LOGS / "opt_r30_fixed" / "r30_a01_catv_on" / "a01" / "evaloff_round029.json", None,
     {"full_ppl_token": 31.3781}, "opt_r30_fixed/r30_a01_catv_on"),
    ("tab:main", "opt_casc_r29", "a01_casc_train_eval_mask", "OPT-350M-a01",
     OPT_ARCH / "r30_a01_catv_on" / "adapter_round029",
     LOGS / "opt_r30_fixed" / "r30_a01_catv_on" / "a01" / "eval_round029.json", None,
     {"full_ppl_token": 31.6351}, "opt_r30_fixed/r30_a01_catv_on"),
    ("tab:main", "opt_casc_r29", "a03_casc_off", "OPT-350M-a03",
     OPT_ARCH / "r30_a03_catv_off" / "adapter_round029",
     LOGS / "opt_r30_fixed" / "r30_a03_catv_off" / "a03" / "eval_round029.json", None,
     {"full_ppl_token": 31.4568}, "opt_r30_fixed/r30_a03_catv_off"),
    ("tab:main", "opt_casc_r29", "a03_casc_train_mask", "OPT-350M-a03",
     OPT_ARCH / "r30_a03_catv_on" / "adapter_round029",
     LOGS / "opt_r30_fixed" / "r30_a03_catv_on" / "a03" / "evaloff_round029.json", None,
     {"full_ppl_token": 31.4165}, "opt_r30_fixed/r30_a03_catv_on"),
    ("tab:main", "opt_casc_r29", "a03_casc_train_eval_mask", "OPT-350M-a03",
     OPT_ARCH / "r30_a03_catv_on" / "adapter_round029",
     LOGS / "opt_r30_fixed" / "r30_a03_catv_on" / "a03" / "eval_round029.json", None,
     {"full_ppl_token": 31.6957}, "opt_r30_fixed/r30_a03_catv_on"),
    # ---------------- Table e2e: DistilBERT a03 r59 ----------------
    ("tab:e2e", "distilbert_e2e_r59", "fwdllm_dense", "DistilBERT-a03",
     REPO / "checkpoints" / "distilbert_fwdllm_a03_var_cont30" / "adapter_round059",
     LOGS / "bert_fwdllm_a03_var_cont30" / "a03" / "eval_round059.json", None,
     {"nll": 1.3437, "accuracy": 0.520}, "bert_fwdllm_a03_var_cont30"),
    ("tab:e2e", "distilbert_e2e_r59", "fwdllm_jenga", "DistilBERT-a03",
     DBERT_CONT / "r30_a03_catv_off" / "adapter_round059",
     LOGS / "bert_da_fixed_cont30" / "r30_a03_catv_off" / "a03" / "eval_round059.json", None,
     {"nll": 1.5110, "accuracy": 0.474}, "bert_da_fixed_cont30/r30_a03_catv_off"),
    ("tab:e2e", "distilbert_e2e_r59", "cpur_0.5", "DistilBERT-a03",
     REPO / "checkpoints" / "distilbert_vq050_perresult_a03_cont30" / "adapter_round059",
     LOGS / "bert_vq050_perresult_a03_cont30" / "a03" / "eval_round059.json", None,
     {"nll": 1.5139, "accuracy": 0.474}, "bert_vq050_perresult_a03_cont30"),
    ("tab:e2e", "distilbert_e2e_r59", "combo_train", "DistilBERT-a03",
     REPO / "checkpoints" / "distilbert_var_catv_vq_a03_cont30" / "adapter_round059",
     LOGS / "bert_var_catv_vq_a03_cont30" / "a03" / "evaloff_round059.json", None,
     {"nll": 1.5231, "accuracy": 0.470}, "bert_var_catv_vq_a03_cont30"),
    ("tab:e2e", "distilbert_e2e_r59", "combo_train_eval", "DistilBERT-a03",
     REPO / "checkpoints" / "distilbert_var_catv_vq_a03_cont30" / "adapter_round059",
     LOGS / "bert_var_catv_vq_a03_cont30" / "a03" / "eval_round059.json", None,
     {"nll": 1.4896, "accuracy": 0.478}, "bert_var_catv_vq_a03_cont30"),
    # ---------------- Table resample: OPT D16/D32 (r7 and r29) ----------------
    ("tab:resample", "opt_resample_r29", "D16_r29", "OPT-350M-a03",
     LOGS / "zoo_k24_d16_lr0.30_r30" / "a03" / "adapter_round029",
     LOGS / "zoo_k24_d16_lr0.30_r30" / "a03" / "eval_round029.json", None,
     {"full_ppl_token": 32.259}, "zoo_k24_d16_lr0.30_r30"),
    ("tab:resample", "opt_resample_r29", "D32_r29", "OPT-350M-a03",
     LOGS / "zoo_k24_d32_lr0.30_r30" / "a03" / "adapter_round029",
     LOGS / "zoo_k24_d32_lr0.30_r30" / "a03" / "eval_round029.json", None,
     {"full_ppl_token": 31.410}, "zoo_k24_d32_lr0.30_r30"),
    ("tab:resample", "opt_resample_r7", "D16_r7", "OPT-350M-a03",
     LOGS / "zoo_k24_d16_lr0.30_r30" / "a03" / "adapter_round007", None, None,
     {}, "zoo_k24_d16_lr0.30_r30", "r7 端点值见 resample/table7_d_comparison.csv（无 per-run eval JSON）"),
    ("tab:resample", "opt_resample_r7", "D32_r7", "OPT-350M-a03",
     LOGS / "zoo_k24_d32_lr0.30_r30" / "a03" / "adapter_round007", None, None,
     {}, "zoo_k24_d32_lr0.30_r30", "r7 端点值见 resample/table7_d_comparison.csv（无 per-run eval JSON）"),
    ("tab:resample", "opt_resample_r29", "D22_r29", "OPT-350M-a03",
     None, None, None, {}, "", "D22 run/权重不在本 checkout；表值 32.5335 见 resample/table7_d_comparison.csv"),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval(path):
    if path is None:
        return None, {}
    data = json.load(open(path))
    return data, {key: data.get(key) for key in ("nll", "accuracy", "full_ppl_token", "answer_ppl_token")}


def copy_adapter(src, dst_root, name):
    if not src.exists():
        return None
    last = src.name
    dst = dst_root / name / last
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
    return dst


PREREQ = [
    ("OPT-350M", "pos_embed", REPO / "checkpoints" / "posemb_step1" / "a01_pos_only_500step" / "pos_embed.pt",
     "weights/opt/pos_embed.pt", "run_fed/ppl 的 --pos-checkpoint"),
    ("OPT-350M", "predictor", REPO / "checkpoints" / "predictor" / "step5_a01a03_clients_causal" / "predictor.pth",
     "weights/opt/predictor.pth", "run_fed/ppl 的 --predictor"),
    ("OPT-350M", "pruned_config", REPO / "checkpoints" / "predictor" / "step5_a01a03_clients_causal" / "pruned_config.pth",
     "weights/opt/pruned_config.pth", "run_fed/ppl 的 --pruned-config"),
    ("DistilBERT", "pos_embed", REPO / "checkpoints" / "distilbert_step1" / "a01_pos_head_1000step" / "pos_embed.pt",
     "weights/bert/pos_embed.pt", "run_fed/classify 的 --pos-checkpoint"),
    ("DistilBERT", "head", REPO / "checkpoints" / "distilbert_step1" / "a01_pos_head_1000step" / "head.pt",
     "weights/bert/head.pt", "run_fed/classify 的 --head-checkpoint"),
    ("DistilBERT", "predictor", REPO / "checkpoints" / "predictor_bert" / "a01_warmup_400step" / "predictor.pth",
     "weights/bert/predictor.pth", "run_fed/classify 的 --predictor"),
    ("DistilBERT", "pruned_config", REPO / "checkpoints" / "predictor_bert" / "a01_warmup_400step" / "pruned_config.pth",
     "weights/bert/pruned_config.pth", "run_fed/classify 的 --pruned-config"),
]


def export_prereq():
    rows = []
    for model, kind, src, dest_rel, used_by in PREREQ:
        if not src.exists():
            raise SystemExit(f"[export] missing prerequisite: {src}")
        dest = OUT / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        rows.append({"model": model, "kind": kind, "dest": dest_rel, "source": str(src.relative_to(REPO)),
                     "bytes": dest.stat().st_size, "sha256": sha256(dest), "used_by": used_by})
    with open(OUT / "WEIGHTS.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "adapters").mkdir(parents=True)
    (OUT / "evals").mkdir(parents=True)

    rows, problems = [], []
    for entry in ENTRIES:
        paper, group, name, model, adapter_src, eval_path, evaloff_path, expected, source = entry[:9]
        note = entry[9] if len(entry) > 9 else ""
        dst = copy_adapter(adapter_src, OUT / "adapters" / group, name) if adapter_src else None
        if dst is None:
            if adapter_src is None:
                rows.append({"paper": paper, "group": group, "name": name, "model": model,
                             "adapter_round": "", "adapter_dir": "", "adapter_sha256": "",
                             "eval_json": "", "eval_metrics": "", "evaloff_json": "",
                             "evaloff_metrics": "", "source_run": source, "note": note})
                continue
            problems.append(f"missing adapter: {adapter_src}")
            continue
        eval_data, eval_vals = load_eval(eval_path)
        if eval_path is not None and eval_data is None:
            problems.append(f"missing eval: {eval_path}")
        evaloff_data, evaloff_vals = load_eval(evaloff_path) if evaloff_path is not None else (None, {})
        if not args.no_check:
            for key, want in expected.items():
                got = (eval_vals or {}).get(key)
                if got is None or abs(float(got) - float(want)) > 1e-3:
                    problems.append(f"{group}/{name}: {key} got {got}, expected {want}")
        eval_dst = OUT / "evals" / group / name
        eval_dst.mkdir(parents=True, exist_ok=True)
        for label, path in (("eval", eval_path), ("evaloff", evaloff_path)):
            if path is not None and path.exists():
                shutil.copy2(path, eval_dst / f"{label}_{path.name}")
        adapter_file = dst / "adapter_model.safetensors"
        rows.append({
            "paper": paper, "group": group, "name": name, "model": model,
            "adapter_round": dst.name,
            "adapter_dir": str(dst.relative_to(OUT)),
            "adapter_sha256": sha256(adapter_file) if adapter_file.exists() else "",
            "eval_json": (str((eval_dst / f"eval_{eval_path.name}").relative_to(OUT))
                          if eval_path is not None and eval_path.exists() else ""),
            "eval_metrics": json.dumps(eval_vals, sort_keys=True),
            "evaloff_json": (str((eval_dst / f"evaloff_{evaloff_path.name}").relative_to(OUT))
                             if evaloff_path is not None and evaloff_path.exists() else ""),
            "evaloff_metrics": json.dumps(evaloff_vals, sort_keys=True) if evaloff_path else "",
            "source_run": source,
            "note": note,
        })

    (OUT / "resample").mkdir(parents=True, exist_ok=True)
    table7 = REPO / "reports" / "opt" / "data" / "table7_d_comparison.csv"
    if table7.exists():
        shutil.copy2(table7, OUT / "resample" / "table7_d_comparison.csv")

    prereq_rows = export_prereq()

    # memory table (no weights)
    mem_src = REPO / "reports" / "mem"
    (OUT / "memory").mkdir(parents=True, exist_ok=True)
    for item in ("README.md", "data/mem_summary.csv", "data/mem_tidy.csv"):
        src = mem_src / item
        if src.exists():
            (OUT / "memory").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, OUT / "memory" / Path(item).name)

    if problems and not args.no_check:
        print("[export] FAILED:")
        for problem in problems:
            print("  -", problem)
        raise SystemExit(1)

    with open(OUT / "MANIFEST.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# SMELL-v3 evaluation endpoints",
        "",
        "论文 Evaluation 各表所需的关键 adapter（r29 / r59 / r7）与对应评测 JSON。",
        "由 `scripts/export_endpoints.py` 从归档/运行目录自动汇总，数值与论文表逐项对账（不符即脚本退出）。",
        "",
        "## 目录",
        "",
        "- `adapters/<group>/<name>/adapter_roundNNN/` — PEFT LoRA adapter（`adapter_model.safetensors` + `adapter_config.json`）",
        "- `evals/<group>/<name>/` — 对应端点评测 JSON（`eval_*` 带 mask；`evaloff_*` 无 mask）",
        "- `memory/` — Table `tab:mem_peak` 的 OPT-350M 数据（`mem_summary.csv` / `mem_tidy.csv`）",
        "- `resample/table7_d_comparison.csv` — Table `tab:resample` 的 r7 端点值（D16/D22/D32）",
        "- `weights/opt|bert/` — 复现所需的前置权重：`pos_embed.pt`、`head.pt`（BERT）、`predictor.pth`、`pruned_config.pth`",
        "- `WEIGHTS.csv` — 前置权重的 sha256 / 大小 / 用途",
        "- `MANIFEST.csv` — 论文表 → adapter → eval JSON → 指标数值 → 来源 run",
        "- `SHA256SUMS` — 全部文件校验和",
        "",
        "## 加载方式",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from peft import PeftModel",
        "base = AutoModelForCausalLM.from_pretrained(<base model>, torch_dtype=torch.bfloat16)",
        "model = PeftModel.from_pretrained(base, 'adapters/<group>/<name>/adapter_roundNNN')",
        "```",
        "",
        "OPT-350M 评测：`src/eval/ppl.py --tag a03 --attn sdpa_prune --dtype fp32 --sparse 0.4 \\",
        "  --pos-checkpoint checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt \\",
        "  --predictor checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth \\",
        "  --pruned-config checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth \\",
        "  --adapter <adapter_dir> --out <out.json>`",
        "DistilBERT 评测：`src/eval/classify.py ... --adapter <adapter_dir>`（见 `README.md` 复现一节）。",
        "",
        "## 前置权重（pos_embed / predictor / head / pruned_config）",
        "",
        "| 模型 | 文件 | 用途 |",
        "|---|---|---|",
        "| OPT-350M | `weights/opt/pos_embed.pt` | `--pos-checkpoint`（位置适配） |",
        "| OPT-350M | `weights/opt/predictor.pth` + `pruned_config.pth` | `--predictor` / `--pruned-config`（Jenga 块打分） |",
        "| DistilBERT | `weights/bert/pos_embed.pt` | `--pos-checkpoint` |",
        "| DistilBERT | `weights/bert/head.pt` | `--head-checkpoint`（Step-1 分类头） |",
        "| DistilBERT | `weights/bert/predictor.pth` + `pruned_config.pth` | `--predictor` / `--pruned-config` |",
        "",
        "底座模型（`facebook/opt-350m`、`distilbert-base-uncased`）不随包分发，按 `docs/weight-manifest.md` 放到",
        "`third_party/Jenga/checkpoints/` 即可；本包的 adapter 均在上述底座 + 前置权重下评测。",
        "",
        "## 与论文表的对应",
        "",
        "| 论文表 | group | 说明 |",
        "|---|---|---|",
        "| `tab:main`（CASC） | `distilbert_casc_r29` / `opt_casc_r29` | `*_casc_off` 用 `eval_*`；`*_casc_train_mask` 用 `evaloff_*`（训练有 mask、评测无）；`*_casc_train_eval_mask` 用 `eval_*`（带 mask） |",
        "| `tab:e2e`（端到端 60 轮） | `distilbert_e2e_r59` | `combo_train` 用 `evaloff_*`，`combo_train_eval` 用 `eval_*`；其余为 `eval_*` |",
        "| `tab:resample`（D 扫描） | `opt_resample_r29` / `opt_resample_r7` | D16/D32 的 r29 与 r7 adapter；**D22 权重不在本 checkout**，仅表值存在 |",
        "| `tab:mem_peak`（显存） | `memory/` | OPT-350M 的 ZOO（Ours）与 BP（FwdLLM+LoRA）显存 CSV |",
        "",
        "## 备注",
        "",
        "- 只含**关键 endpoint**（不做全轮次导出）；完整训练曲线在各 `checkpoints/*` 归档与 `reports/` 包内。",
        "- 所有 adapter 为 LoRA r=1/α=2（OPT 与 DistilBERT 相同），加载时无需重新合并。",
        "- `tab:mem_peak` 的 LLaMA2-7B 列不在本仓库（另一实验环境），本包只含 OPT-350M 列。",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    entries = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in ("SHA256SUMS",) and path.suffix != ".gz":
            entries.append(f"{sha256(path)}  {path.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")

    tarball = REPO / "checkpoints" / "out_smell_v3_endpoints.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(OUT, arcname="out")
    (tarball.parent / (tarball.name + ".sha256")).write_text(f"{sha256(tarball)}  {tarball.name}\n",
                                                              encoding="utf-8")

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"[export] {len(rows)} entries -> {OUT}")
    print(f"[export] files={sum(1 for p in OUT.rglob('*') if p.is_file())}  bytes={total}  "
          f"tarball={tarball} ({tarball.stat().st_size} B)")


if __name__ == "__main__":
    main()
