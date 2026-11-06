#!/usr/bin/env bash
# 实验 2：BERT 续训（r30 之后，30 轮/段）
#   默认：r60→r89（从 cont30 归档 r59 起，lr=4.6667e-4 = 1.4e-3/3）
#   也可：OFFSET=30 INIT_ARCH=checkpoints/distilbert_ablation_r30_fixed LR=1.4e-3（r30→r59）
# 用法：
#   GPU=0 bash scripts/repro/02_bert_cont.sh
#   OFFSET=30 LR=1.4e-3 INIT_ARCH="$PWD/checkpoints/distilbert_ablation_r30_fixed" \
#     bash scripts/repro/02_bert_cont.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OFFSET="${OFFSET:-60}"
ROUNDS="${ROUNDS:-30}"
LR="${LR:-0.0004666666666666667}"
INIT_ARCH="${INIT_ARCH:-$REPO/checkpoints/distilbert_ablation_r30_fixed_cont30}"
ARMS="${ARMS:-r30_a01_catv_off r30_a01_catv_on r30_a03_catv_off r30_a03_catv_on}"
OUT="$REPO/logs/fed/repro_bert_cont${OFFSET}"
S1="$REPO/checkpoints/distilbert_step1/a01_pos_head_1000step"
PRED="$REPO/checkpoints/predictor_bert/a01_warmup_400step"
PREV=$((OFFSET - 1))

echo "[$(date +%H:%M:%S)] BERT continuation offset=$OFFSET rounds=$ROUNDS lr=$LR init=$INIT_ARCH"

BASE=(
  --arch distilbert --gpu "$GPU" --trainer zoo --attn sdpa_prune --dtype fp32
  --sparse 0.4 --doc-order cycle --round-offset "$OFFSET" --rounds "$ROUNDS"
  --pos-checkpoint "$S1/pos_embed.pt" --head-checkpoint "$S1/head.pt"
  --predictor "$PRED/predictor.pth" --pruned-config "$PRED/pruned_config.pth"
  --catv-normalize off --lora-r 1 --lora-alpha 2
  --zo-subspace layers --zo-layers 0-5 --zo-directions 8 --zo-eps 1e-3
  --local-steps 8 --lr "$LR" --delta-clip 1.0
  --max-clients 30 --eval-every 1 --eval-batch-size 8 --eval-max-samples 500 --save-every 1
)

for ARM in $ARMS; do
  TAG="${ARM:4:3}"
  CATV="${ARM##*_}"
  INIT=$(printf "%s/%s/adapter_round%03d" "$INIT_ARCH" "$ARM" "$PREV")
  if [ ! -d "$INIT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $ARM: missing init $INIT"
    continue
  fi
  echo "[$(date +%H:%M:%S)] $ARM start (catv=$CATV, init=$INIT)"
  launch "$LOG_ROOT/$ARM.log" "$PY" -u src/fed/run_fed.py "${BASE[@]}" \
    --tag "$TAG" --catv "$CATV" --adapter-init "$INIT" \
    --out-root "$OUT/$ARM"
  echo "[$(date +%H:%M:%S)] $ARM done"
  maybe_archive "$OUT/$ARM" "$TAG" $((OFFSET + ROUNDS - 1)) "$ARM"
done
echo "[$(date +%H:%M:%S)] continuation finished; metrics under $OUT/<arm>/<tag>/metrics.jsonl"
echo "  对照：checkpoints/distilbert_ablation_r30_fixed_cont30|cont60/"
