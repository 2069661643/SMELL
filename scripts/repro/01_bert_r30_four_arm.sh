#!/usr/bin/env bash
# 实验 1：BERT r30 四臂（a01/a03 × CATV off/on）—— 论文主表（r29 端点）
# 协议：L=8、D=8、lr=1.4e-3、clip=1.0、cycle、eval 500、catv-normalize off、LoRA r1/α2、sparse 0.4
# 对照归档：checkpoints/distilbert_ablation_r30_fixed/（r0 1.69–1.75 → r29 1.548–1.589）
# 用法：GPU=0 bash scripts/repro/01_bert_r30_four_arm.sh
#       单臂约 6.5h（4090），四臂串行 ~26h；DRY_RUN=1 只看命令
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_ID="${RUN_ID:-bert_r30_four_arm}"
OUT="$REPO/logs/fed/repro_$RUN_ID"
S1="$REPO/checkpoints/distilbert_step1/a01_pos_head_1000step"
PRED="$REPO/checkpoints/predictor_bert/a01_warmup_400step"

BASE=(
  --arch distilbert --gpu "$GPU" --trainer zoo --attn sdpa_prune --dtype fp32
  --sparse 0.4 --doc-order cycle
  --pos-checkpoint "$S1/pos_embed.pt" --head-checkpoint "$S1/head.pt"
  --predictor "$PRED/predictor.pth" --pruned-config "$PRED/pruned_config.pth"
  --lora-r 1 --lora-alpha 2
  --zo-subspace layers --zo-layers 0-5 --zo-directions 8 --zo-eps 1e-3
  --local-steps 8 --lr 1.4e-3 --delta-clip 1.0
  --max-clients 30 --eval-every 1 --eval-batch-size 8 --eval-max-samples 500 --save-every 1
)

echo "[$(date +%H:%M:%S)] BERT r30 four-arm -> $OUT (log $LOG_ROOT)"
for TAG in a01 a03; do
  for CATV in off on; do
    ARM="r30_${TAG}_catv_${CATV}"
    echo "[$(date +%H:%M:%S)] $ARM start (catv=$CATV)"
    launch "$LOG_ROOT/$ARM.log" "$PY" -u src/fed/run_fed.py "${BASE[@]}" \
      --tag "$TAG" --catv "$CATV" --catv-normalize off --rounds 30 \
      --out-root "$OUT/$ARM"
    echo "[$(date +%H:%M:%S)] $ARM done"
    maybe_archive "$OUT/$ARM" "$TAG" 29 "$ARM"
  done
done
echo "[$(date +%H:%M:%S)] BERT r30 four-arm finished; metrics under $OUT/<arm>/<tag>/metrics.jsonl"
echo "  对照：checkpoints/distilbert_ablation_r30_fixed/ 与 reports/bert_and_other/README.md"
