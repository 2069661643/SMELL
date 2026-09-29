#!/usr/bin/env bash
# SMELL 3 run_k4_td2_td3 NEW — k=4 sparsity-ZOO 门控 + lr 校准流水线（TD-2' -> gate -> lr -> TD-3）
#
# 设计（详细说明见 docs/ailog/260928-143659-smell-v3-gate-lr-pipeline-delivery.md）：
#   1) TD-2'：scripts/diag_cos_grid.py 测 k=4 子空间（d_eff=4*8192=32768）在 fp32+sdpa_prune 下的稀疏 cos
#   2) 门控：cos(k4,L2,D22,c30) >= COS_GATE（默认 0.05）才继续；否则 ~1.4h 时提前终止
#   3) lr 校准：1 client/2 samples 探针（probe lr=4e-6）读 delta_norm
#              -> lr = 4e-6 * TARGET_DNORM / delta_norm（TARGET_DNORM 默认 0.38 = BP k=4 参考；钳制 [1e-7, 1.0]，触界告警）
#   4) TD-3：run_fed --trainer zoo（rotate4, c30, L2, D22, fp32+sdpa_prune, 每轮 eval/save）
#
# 约束：predictor 必须用 causal 修复版（Jenga 原非 causal 目标会选尾部块）；ZOO 前向在 ClientRunner 内强制 eval（dropout=0.1 会吞差分）。
# 可用环境变量覆盖：PY TAG POS PRED PCS BDIR TD2_OUT PROBE_ROOT ZOO_ROOT ROUNDS COS_GATE TARGET_DNORM GPU
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/applications/anaconda3/envs/smell-v2/bin/python}"
TAG="${TAG:-a01}"
POS="${POS:-$REPO/checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt}"
PRED="${PRED:-$REPO/checkpoints/predictor/step5_a01a03_clients_causal/predictor.pth}"
PCS="${PCS:-$REPO/checkpoints/predictor/step5_a01a03_clients_causal/pruned_config.pth}"
BDIR="${BDIR:-$REPO/logs/fed}"
TD2_OUT="${TD2_OUT:-$REPO/temp/diag_cos_sparse_td2_k4_causal.json}"
PROBE_ROOT="${PROBE_ROOT:-$BDIR/zoo_k4_sparse_causal_preflight}"
ZOO_ROOT="${ZOO_ROOT:-$BDIR/zoo_k4_sparse_causal}"
ROUNDS="${ROUNDS:-30}"
COS_GATE="${COS_GATE:-0.05}"
TARGET_DNORM="${TARGET_DNORM:-0.38}"
GPU="${GPU:-2}"
cd "$REPO"
LOG_BASE="$REPO/temp/logs"; RUN_DIR="$LOG_BASE/k4_td2_td3_$(date +%y%m%d-%H%M%S)"; mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1
echo "$$" > "$LOG_BASE/last_k4_td2_td3.pid"
echo "$RUN_DIR" > "$LOG_BASE/last_k4_td2_td3_dir.txt"
echo "[$(date +%H:%M:%S)] STEP start run_dir=$RUN_DIR rounds=$ROUNDS cos_gate=$COS_GATE target_dnorm=$TARGET_DNORM predictor=$PRED"

# ---- Stage 1: TD-2' k4 稀疏 cos @16k ----
echo "[$(date +%H:%M:%S)] STEP td2_k4 start (sdpa_prune fp32 seq=16384 k4=layers20-23 L2 D22 c=1..30)"
"$PY" -u "$REPO/scripts/diag_cos_grid.py" --gpu "$GPU" --attn sdpa_prune --dtype fp32 --seq 16384 \
  --data-root "$REPO/dataset_v3/discovery_16k/$TAG/clients" \
  --block-layers 20-23 --blocks k4 --n-clients 30 --ls 2 --ds 22 --cs 1,5,10,20,30 \
  --lora-r 1 --alpha 2.0 --pos-checkpoint "$POS" --predictor "$PRED" --pruned-config "$PCS" \
  --out "$TD2_OUT"
RC=$?
echo "[$(date +%H:%M:%S)] STEP td2_k4 done rc=$RC out=$TD2_OUT"
[ "$RC" -eq 0 ] || { echo "[$(date +%H:%M:%S)] STEP ABORT td2 rc=$RC"; exit 1; }

COS=$("$PY" -c "import json;rows=json.load(open('$TD2_OUT'));h=[r for r in rows if r['block']=='k4' and r['L']==2 and r['D']==22 and r['c']==30];print(h[0]['cos'] if h else -1.0)")
echo "[$(date +%H:%M:%S)] STEP gate cos(k4,L2,D22,c30)=$COS"
if ! awk "BEGIN{exit !($COS >= $COS_GATE)}"; then
  echo "[$(date +%H:%M:%S)] STEP ABORT: sparse cos=$COS < gate=$COS_GATE -> TD-3 skipped"; exit 0
fi

# ---- Stage 2: lr 校准（probe lr=4e-6 -> 对齐 BP k=4 delta_norm） ----
# SMELL 3 run_k4_td2_td3 probe FIXED — 探针须与 TD-3 同步数（2 samples × local-steps 2 = 2 步）；delta_norm<=0 直接中止
echo "[$(date +%H:%M:%S)] STEP lr_probe start (1 client/2 samples, probe lr=4e-6)"
rm -rf "$PROBE_ROOT"
mkdir -p "$PROBE_ROOT"
"$PY" src/fed/run_fed.py --tag "$TAG" --gpu "$GPU" --trainer zoo --catv off \
  --dtype fp32 --attn sdpa_prune --sparse 0.4 \
  --pos-checkpoint "$POS" --predictor "$PRED" --pruned-config "$PCS" \
  --lora-r 1 --lora-alpha 2 \
  --zo-subspace layers --zo-layer-rotate --zo-layer-group 4 \
  --max-clients 1 --max-train-samples 2 --local-steps 2 --zo-directions 22 --zo-eps 1e-3 \
  --rounds 1 --lr 4e-6 --out-root "$PROBE_ROOT" > "$PROBE_ROOT/driver.log" 2>&1
DN=$("$PY" -c "import json;print(json.loads(open('$PROBE_ROOT/$TAG/metrics.jsonl').readline())['delta_norm_mean'])" 2>/dev/null || echo 0)
echo "[$(date +%H:%M:%S)] STEP lr_probe delta_norm=$DN"
if ! awk "BEGIN{exit !($DN > 0)}"; then
  echo "[$(date +%H:%M:%S)] STEP ABORT lr_probe delta_norm=$DN <= 0; tail $PROBE_ROOT/driver.log:"
  tail -n 5 "$PROBE_ROOT/driver.log" 2>/dev/null
  exit 1
fi
LR_RAW=$("$PY" -c "d=float('$DN'); print(4e-6*$TARGET_DNORM/d)")
LR=$("$PY" -c "raw=float('$LR_RAW'); print(min(max(raw,1e-7),1.0))")
# SMELL 3 run_k4_td2_td3 lr_clamp FIXED — 上界 1e-4 曾把校准值 0.15 夹低 1500×（TD-3 空转 13 轮）；放宽到 1.0 并告警
if awk "BEGIN{exit !(($LR_RAW) < 1e-7 || ($LR_RAW) > 1.0)}"; then
  echo "[$(date +%H:%M:%S)] STEP WARNING lr_clamped raw=$LR_RAW -> $LR (check probe/delta)"
fi
echo "[$(date +%H:%M:%S)] STEP lr_calibrated=$LR (raw=$LR_RAW target delta_norm=$TARGET_DNORM)"

# ---- Stage 3: TD-3 k=4 sparsity ZOO（rotate4, L2 D22, c30, eval/save 每轮） ----
echo "[$(date +%H:%M:%S)] STEP td3 start lr=$LR rounds=$ROUNDS eval_every=1"
"$PY" src/fed/run_fed.py --tag "$TAG" --gpu "$GPU" --trainer zoo --catv off \
  --dtype fp32 --attn sdpa_prune --sparse 0.4 \
  --pos-checkpoint "$POS" --predictor "$PRED" --pruned-config "$PCS" \
  --lora-r 1 --lora-alpha 2 \
  --zo-subspace layers --zo-layer-rotate --zo-layer-group 4 \
  --max-clients 30 --local-steps 2 --zo-directions 22 --zo-eps 1e-3 \
  --rounds "$ROUNDS" --eval-every 1 --lr "$LR" \
  --out-root "$ZOO_ROOT"
echo "[$(date +%H:%M:%S)] STEP ALL DONE"
