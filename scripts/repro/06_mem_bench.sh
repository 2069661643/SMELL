#!/usr/bin/env bash
# 实验 6：显存矩阵（交付于 reports/mem）
#   ZOO = src/train/zoo.py::zo_grad（no_grad 前向，D=8）; BP = forward + backward
#   seq 1k/2k/4k/8k/16k × dense/sparse 0.4，OPT-350M、bf16 + flash、LoRA qkvo r8
# 用法：GPU=0 bash scripts/repro/06_mem_bench.sh
# 产物：temp/mem_sweep_repro/mem_<mode>_<kind>_<seq>.json（再跑 reports/mem/make_report.py 可对账）
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$REPO/temp/mem_sweep_repro"
mkdir -p "$OUT"
SEQS="${SEQS:-1024 2048 4096 8192 16384}"
MODES="${MODES:-zoo backward}"
ZOO_DIRECTIONS="${ZOO_DIRECTIONS:-8}"

echo "[$(date +%H:%M:%S)] memory bench -> $OUT (log $LOG_ROOT)"
for SEQ in $SEQS; do
  for MODE in $MODES; do
    for KIND in sparse dense; do
      if [ "$KIND" = "sparse" ]; then ONLY="sparse=1,lora=qkvo,ckpt=0"; else ONLY="dense=1,lora=qkvo,ckpt=0"; fi
      EXTRA=()
      [ "$MODE" = "zoo" ] && EXTRA=(--zoo-directions "$ZOO_DIRECTIONS" --zo-eps 1e-3)
      JSON="$OUT/mem_${MODE}_${KIND}_${SEQ}.json"
      echo "[$(date +%H:%M:%S)] $MODE $KIND seq=$SEQ"
      launch "$LOG_ROOT/mem_${MODE}_${KIND}_${SEQ}.log" "$PY" -u src/bench/memory.py \
        --seq-len "$SEQ" --mode "$MODE" --steps 1 --only "$ONLY" "${EXTRA[@]}" --out "$JSON"
      "$PY" -c "
import json
d=json.load(open('$JSON'))[0]
print('  peak_GiB=%s status=%s' % (
  round(d['peak_allocated_gb'],2) if d['peak_allocated_gb'] is not None else None, d['status'][:60]))
" 2>/dev/null
    done
  done
done
echo "[$(date +%H:%M:%S)] memory bench finished; JSON under $OUT"
echo "  参考结果：reports/mem/README.md（16k ZOO 2.66/2.21 GiB vs BP 14.51/12.69 GiB）"
