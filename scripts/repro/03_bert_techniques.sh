#!/usr/bin/env bash
# 实验 3：BERT 技术线（a03、30 轮、eval 500）
#   CELL=vq_perresult  VQ 0.5 bf16（per-result，C=L*D=64；legacy 语义）
#   CELL=vq_perclient  VQ 0.5 bf16（per-client，1 channel/client，30 channel/轮）
#   CELL=var_methoda   VAR Method A（raw var、D=4..16、T0=0.5872309208、r10/r20 探针）
#   CELL=combo_3on     VAR + CATV(client) + VQ 0.5 per-result（系统评测）
#   CELL=all          依次跑四个
# 用法：GPU=0 CELL=all bash scripts/repro/03_bert_techniques.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CELL="${CELL:-all}"
OUT_ROOT="$REPO/logs/fed/repro_bert_techniques"
S1="$REPO/checkpoints/distilbert_step1/a01_pos_head_1000step"
PRED="$REPO/checkpoints/predictor_bert/a01_warmup_400step"

BASE=(
  --arch distilbert --tag a03 --gpu "$GPU" --trainer zoo --attn sdpa_prune --dtype fp32
  --sparse 0.4 --doc-order cycle --rounds 30
  --pos-checkpoint "$S1/pos_embed.pt" --head-checkpoint "$S1/head.pt"
  --predictor "$PRED/predictor.pth" --pruned-config "$PRED/pruned_config.pth"
  --lora-r 1 --lora-alpha 2
  --zo-subspace layers --zo-layers 0-5 --zo-directions 8 --zo-eps 1e-3
  --local-steps 8 --lr 1.4e-3 --delta-clip 1.0
  --max-clients 30 --eval-every 1 --eval-batch-size 8 --eval-max-samples 500 --save-every 1
)

run_cell() {
  local name="$1"; shift
  local out="$OUT_ROOT/$name"
  echo "[$(date +%H:%M:%S)] $name start"
  launch "$LOG_ROOT/$name.log" "$PY" -u src/fed/run_fed.py "${BASE[@]}" "$@" --out-root "$out"
  echo "[$(date +%H:%M:%S)] $name done -> $out"
  maybe_archive "$out" a03 29 "$name"
}

[ "$CELL" = "all" ] || [ "$CELL" = "vq_perresult" ] && run_cell vq_perresult \
  --catv off --catv-normalize off \
  --vq-rate 0.5 --vq-channels-mode per_result --vq-sketch-dtype bf16
[ "$CELL" = "all" ] || [ "$CELL" = "vq_perclient" ] && run_cell vq_perclient \
  --catv off --catv-normalize off \
  --vq-rate 0.5 --vq-channels-mode per_client --vq-sketch-dtype bf16
[ "$CELL" = "all" ] || [ "$CELL" = "var_methoda" ] && run_cell var_methoda \
  --catv off --catv-normalize off \
  --var-control on --var-threshold 0.587230920791626 \
  --var-segment-rounds 10 --var-probe-clients 8
[ "$CELL" = "all" ] || [ "$CELL" = "combo_3on" ] && run_cell combo_3on \
  --catv on --catv-normalize client \
  --var-control on --var-threshold 0.587230920791626 \
  --var-segment-rounds 10 --var-probe-clients 8 \
  --vq-rate 0.5 --vq-channels-mode per_result --vq-sketch-dtype bf16

echo "[$(date +%H:%M:%S)] techniques finished; metrics under $OUT_ROOT/<cell>/a03/metrics.jsonl"
echo "  对照：checkpoints/distilbert_vq050_bf16_perresult_a03_r30 / distilbert_vq_perclient_a03_r30 /"
echo "        distilbert_var_methoda_a03_r30 / distilbert_var_catv_vq_a03_r30_rerun"
