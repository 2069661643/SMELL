#!/usr/bin/env bash
# 实验 5：OPT 续训 r30→r59（lr=0.30 首段续训）+ 可选 eval-off
#   EVALOFF=1 时，对两个 on 臂的 r34/39/44/49/54/59 做无掩码评测（src/eval/ppl.py）
# 用法：
#   GPU=0 bash scripts/repro/05_opt_cont30.sh
#   GPU=0 EVALOFF=1 bash scripts/repro/05_opt_cont30.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OFFSET="${OFFSET:-30}"
ROUNDS="${ROUNDS:-30}"
LR="${LR:-0.30}"
INIT_ARCH="${INIT_ARCH:-$REPO/checkpoints/opt_ablation_r30_fixed}"
ARMS="${ARMS:-r30_a01_catv_off r30_a01_catv_on r30_a03_catv_off r30_a03_catv_on}"
EVALOFF="${EVALOFF:-0}"
OUT="$REPO/logs/fed/repro_opt_cont${OFFSET}"
POS="$REPO/checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt"
PRED="$REPO/checkpoints/predictor/step5_a01a03_clients_causal"
PREV=$((OFFSET - 1))

BASE=(
  --arch opt --gpu "$GPU" --trainer zoo --attn sdpa_prune --dtype fp32
  --sparse 0.4 --doc-order cycle --round-offset "$OFFSET" --rounds "$ROUNDS"
  --pos-checkpoint "$POS" --predictor "$PRED/predictor.pth" --pruned-config "$PRED/pruned_config.pth"
  --catv-normalize off --lora-r 1 --lora-alpha 2
  --zo-subspace layers --zo-layer-rotate --zo-layer-group 24
  --zo-directions 32 --zo-eps 1e-3 --local-steps 2 --lr "$LR" --delta-clip 13.34
  --max-clients 30 --eval-every 5 --save-every 1
)

echo "[$(date +%H:%M:%S)] OPT continuation offset=$OFFSET rounds=$ROUNDS lr=$LR evaloff=$EVALOFF"
for ARM in $ARMS; do
  TAG="${ARM:4:3}"
  CATV="${ARM##*_}"
  INIT=$(printf "%s/%s/adapter_round%03d" "$INIT_ARCH" "$ARM" "$PREV")
  if [ ! -d "$INIT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $ARM: missing init $INIT"
    continue
  fi
  echo "[$(date +%H:%M:%S)] $ARM start (catv=$CATV)"
  launch "$LOG_ROOT/$ARM.log" "$PY" -u src/fed/run_fed.py "${BASE[@]}" \
    --tag "$TAG" --catv "$CATV" --adapter-init "$INIT" --out-root "$OUT/$ARM"
  echo "[$(date +%H:%M:%S)] $ARM done"
  maybe_archive "$OUT/$ARM" "$TAG" $((OFFSET + ROUNDS - 1)) "$ARM"

  if [ "$EVALOFF" = "1" ] && [ "$CATV" = "on" ]; then
    for R in 34 39 44 49 54 59; do
      RN=$(printf "%03d" "$R")
      ADP="$OUT/$ARM/$TAG/adapter_round$RN"
      [ -d "$ADP" ] || continue
      echo "[$(date +%H:%M:%S)] $ARM r$R eval-off"
      launch "$LOG_ROOT/${ARM}_evaloff_r$RN.log" "$PY" -u src/eval/ppl.py \
        --tag "$TAG" --attn sdpa_prune --dtype fp32 --sparse 0.4 \
        --pos-checkpoint "$POS" --predictor "$PRED/predictor.pth" --pruned-config "$PRED/pruned_config.pth" \
        --adapter "$ADP" \
        --out "$OUT/$ARM/$TAG/evaloff_round$RN.json" \
        --per-sample-out "$OUT/$ARM/$TAG/evaloff_round${RN}_persample.json"
    done
  fi
done
echo "[$(date +%H:%M:%S)] OPT continuation finished; metrics under $OUT/<arm>/<tag>/metrics.jsonl"
echo "  对照：checkpoints/opt_ablation_r30_fixed_cont30/（r59 full/answer + evaloff）"
