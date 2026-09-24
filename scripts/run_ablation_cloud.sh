#!/usr/bin/env bash
# SMELL 3 run_ablation_cloud NEW — 4 并发消融启动器（α0.1/0.3 × CATV off/on；每实验 1 卡、30 client 串行）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/miniconda3/envs/SMELL/bin/python"
GPU_LIST="0,1,2,3"
ROUNDS=30
LOCAL_STEPS=20
LR=1e-4
ZO_DIRS=8
ZO_EPS=1e-3
SPARSE=0.4
CATV_R=0.2
POS_CKPT=""
ADAPTER_INIT=""
DATA_ROOT="dataset_v3/discovery_16k"
OUT_ROOT="logs/fed"
DRY_RUN=0

usage() {
  cat <<'EOF'
usage: bash scripts/run_ablation_cloud.sh [--gpus 0,1,2,3] [--rounds 30] [--local-steps 20]
       [--lr 1e-4] [--zo-directions 8] [--zo-eps 1e-3] [--sparse 0.4] [--catv-r 0.2]
       [--pos-checkpoint PATH] [--adapter-init PATH] [--data-root DIR] [--out-root DIR] [--dry-run]
mapping: GPU0=a01 CATV off, GPU1=a01 CATV on, GPU2=a03 CATV off, GPU3=a03 CATV on
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --local-steps) LOCAL_STEPS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --zo-directions) ZO_DIRS="$2"; shift 2 ;;
    --zo-eps) ZO_EPS="$2"; shift 2 ;;
    --sparse) SPARSE="$2"; shift 2 ;;
    --catv-r) CATV_R="$2"; shift 2 ;;
    --pos-checkpoint) POS_CKPT="$2"; shift 2 ;;
    --adapter-init) ADAPTER_INIT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1"; usage; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NAMES=(a01_off a01_on a03_off a03_on)
TAGS=(a01 a01 a03 a03)
CATVS=(off on off on)
TS="$(date +%y%m%d-%H%M%S)"
LOG_BASE="$REPO/temp/logs"
mkdir -p "$LOG_BASE"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  gpu="${GPUS[$i]:-}"
  tag="${TAGS[$i]}"
  catv="${CATVS[$i]}"
  [[ -n "$gpu" ]] || { echo "missing GPU for $name"; exit 2; }

  extra=()
  [[ -n "$POS_CKPT" ]] && extra+=(--pos-checkpoint "$POS_CKPT")
  [[ -n "$ADAPTER_INIT" ]] && extra+=(--adapter-init "$ADAPTER_INIT")

  CMD=("$PY" "$REPO/src/fed/run_fed.py"
       --data-root "$DATA_ROOT" --tag "$tag" --gpu "$gpu"
       --trainer zoo --catv "$catv" --catv-r "$CATV_R" --sparse "$SPARSE"
       --rounds "$ROUNDS" --local-steps "$LOCAL_STEPS" --lr "$LR"
       --zo-directions "$ZO_DIRS" --zo-eps "$ZO_EPS"
       --out-root "$OUT_ROOT" "${extra[@]}")

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    continue
  fi

  run_dir="$LOG_BASE/ablation_${TS}_${name}"
  mkdir -p "$run_dir"
  wrapper="$run_dir/run.sh"
  {
    printf '#!/usr/bin/env bash\nset -uo pipefail\n'
    printf 'cd %q\n' "$REPO"
    printf 'echo "[$(date +%%H:%%M:%%S)] start %s gpu=%s tag=%s catv=%s"\n' "$name" "$gpu" "$tag" "$catv"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    printf 'rc=$?\n'
    printf 'echo "[$(date +%%H:%%M:%%S)] exit=$rc"\n'
  } > "$wrapper"
  chmod +x "$wrapper"

  setsid nohup bash "$wrapper" >"$run_dir/driver.log" 2>&1 </dev/null &
  pid=$!
  echo "launched $name pid=$pid gpu=$gpu log=$run_dir/driver.log"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] no tasks launched"
  exit 0
fi

printf '%s\n' "$TS" > "$LOG_BASE/last_ablation_dir.txt"
echo
echo "name       gpu tag  catv  log"
for i in "${!NAMES[@]}"; do
  printf '%-10s %-3s %-4s %-5s %s\n' "${NAMES[$i]}" "${GPUS[$i]}" "${TAGS[$i]}" "${CATVS[$i]}" \
    "$LOG_BASE/ablation_${TS}_${NAMES[$i]}/driver.log"
done
echo "confirm startup per AGENTS.md: sleep 30; tail -n 5 <log>"
