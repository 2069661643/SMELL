#!/usr/bin/env bash
# 实验 4：OPT-350M r30 四臂（a01/a03 × CATV off/on）—— 论文 OPT 主表
# 协议：k=24、D=32、L=2、c=30、LoRA r1/α2、lr=0.30、clip=13.34、cycle、eval every 5、catv-normalize off
# 对照归档：checkpoints/opt_ablation_r30_fixed/（full r29 31.46–31.70 vs fresh 33.665）
# 用法：GPU=0 bash scripts/repro/04_opt_r30_four_arm.sh    # 单臂 ~24.5h（单卡 4090），四臂串行 ~4 天
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_ID="${RUN_ID:-opt_r30_four_arm}"
OUT="$REPO/logs/fed/repro_$RUN_ID"
POS="$REPO/checkpoints/posemb_step1/a01_pos_only_500step/pos_embed.pt"
PRED="$REPO/checkpoints/predictor/step5_a01a03_clients_causal"

BASE=(
  --arch opt --gpu "$GPU" --trainer zoo --attn sdpa_prune --dtype fp32
  --sparse 0.4 --doc-order cycle
  --pos-checkpoint "$POS" --predictor "$PRED/predictor.pth" --pruned-config "$PRED/pruned_config.pth"
  --catv-normalize off --lora-r 1 --lora-alpha 2
  --zo-subspace layers --zo-layer-rotate --zo-layer-group 24
  --zo-directions 32 --zo-eps 1e-3 --local-steps 2 --rounds 30 --lr 0.30 --delta-clip 13.34
  --max-clients 30 --eval-every 5 --save-every 1
)

echo "[$(date +%H:%M:%S)] OPT r30 four-arm -> $OUT (log $LOG_ROOT)"
for TAG in a01 a03; do
  for CATV in off on; do
    ARM="r30_${TAG}_catv_${CATV}"
    echo "[$(date +%H:%M:%S)] $ARM start (catv=$CATV)"
    launch "$LOG_ROOT/$ARM.log" "$PY" -u src/fed/run_fed.py "${BASE[@]}" \
      --tag "$TAG" --catv "$CATV" --out-root "$OUT/$ARM"
    echo "[$(date +%H:%M:%S)] $ARM done"
    maybe_archive "$OUT/$ARM" "$TAG" 29 "$ARM"
  done
done
echo "[$(date +%H:%M:%S)] OPT r30 four-arm finished; metrics under $OUT/<arm>/<tag>/metrics.jsonl"
echo "  对照：checkpoints/opt_ablation_r30_fixed/ 与 reports/bert_and_other/README.md（opt_r30_* 表）"
