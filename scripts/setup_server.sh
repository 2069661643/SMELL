#!/usr/bin/env bash
# SMELL 3 setup_server NEW — 云端 A40 服务器 bootstrap（幂等；支持 --dry-run/--check）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="SMELL"
DRY_RUN=0
CHECK_ONLY=0
SKIP_DATA=0
PIP_INDEX="${PIP_INDEX:-https://mirrors.ustc.edu.cn/pypi/simple}"
CONDA_MIRROR="${CONDA_MIRROR:-https://mirrors.ustc.edu.cn/anaconda/miniconda}"
TORCH_WHEEL_URL="${TORCH_WHEEL_URL:-https://download-r2.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl}"
TORCH_SHA256="0c96999d15cf1f13dd7c913e0b21a9a355538e6cfc10861a17158320292f5954"
FA_WHEEL_ABI="${FA_WHEEL_ABI:-auto}"
CONDA="$HOME/miniconda3/bin/conda"
ENV_PY="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
ENV_PIP="$HOME/miniconda3/envs/$ENV_NAME/bin/pip"
WHEELS="$HOME/wheels"
FA_BASE="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"

usage() {
  cat <<'EOF'
usage: bash scripts/setup_server.sh [--repo DIR] [--env-name NAME] [--dry-run] [--check] [--skip-data]
env overrides: TORCH_WHEEL_URL, FA_WHEEL_ABI=auto|TRUE|FALSE, PIP_INDEX, CONDA_MIRROR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1"; usage; exit 2 ;;
  esac
done
ENV_PY="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
ENV_PIP="$HOME/miniconda3/envs/$ENV_NAME/bin/pip"

log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { if [[ $DRY_RUN -eq 1 ]]; then echo "[dry-run] $*"; else "$@"; fi; }

verify() {
  log "verify: python / torch / cuda / flash-attn / jenga"
  [[ -x "$ENV_PY" ]] || { echo "FATAL: env python missing at $ENV_PY"; exit 1; }
  "$ENV_PY" - <<'PY'
import sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("capability", torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))
import flash_attn
print("flash_attn", flash_attn.__version__)
import jenga
print("jenga", jenga.__file__)
PY
  local ckpt_dir="$REPO/third_party/Jenga/checkpoints/opt-350m"
  [[ -f "$ckpt_dir/config.json" ]] || { echo "FATAL: opt-350m config missing at $ckpt_dir"; exit 1; }
  log "verify OK"
}

if [[ $CHECK_ONLY -eq 1 ]]; then verify; exit 0; fi

log "repo=$REPO env=$ENV_NAME dry_run=$DRY_RUN skip_data=$SKIP_DATA"

if [[ ! -x "$CONDA" ]]; then
  log "install Miniconda -> $HOME/miniconda3"
  run bash -lc "curl -fL -o /tmp/miniconda.sh '$CONDA_MIRROR/Miniconda3-latest-Linux-x86_64.sh' && bash /tmp/miniconda.sh -b -u -p '$HOME/miniconda3'"
else
  log "Miniconda present: $("$CONDA" --version 2>/dev/null || echo unknown)"
fi

if ! "$CONDA" env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "create conda env $ENV_NAME (python 3.10)"
  run "$CONDA" create -n "$ENV_NAME" python=3.10 -y
fi

mkdir -p "$WHEELS"

if "$ENV_PY" -c "import torch,sys; sys.exit(0 if (torch.__version__.startswith('2.8.0') and torch.version.cuda=='12.8') else 1)" 2>/dev/null; then
  log "torch 2.8.0+cu128 already installed"
else
  log "download torch wheel"
  run curl -fL --retry 3 -o "$WHEELS/torch-2.8.0+cu128.whl" "$TORCH_WHEEL_URL"
  if [[ $DRY_RUN -eq 0 ]]; then
    echo "$TORCH_SHA256  $WHEELS/torch-2.8.0+cu128.whl" | sha256sum -c -
  fi
  log "pip install torch"
  run "$ENV_PIP" install -i "$PIP_INDEX" "$WHEELS/torch-2.8.0+cu128.whl"
fi

if "$ENV_PY" -c "import flash_attn" 2>/dev/null; then
  log "flash-attn already installed"
else
  ABI="$FA_WHEEL_ABI"
  if [[ "$ABI" == "auto" ]]; then
    ABI="$("$ENV_PY" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
  fi
  FA_FILE="flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"
  log "download flash-attn wheel (abi=$ABI)"
  run bash -lc "curl -fL --retry 3 -o '$WHEELS/$FA_FILE' 'https://ghfast.top/$FA_BASE/$FA_FILE' || curl -fL --retry 3 -o '$WHEELS/$FA_FILE' '$FA_BASE/$FA_FILE'"
  run "$ENV_PIP" install --no-deps "$WHEELS/$FA_FILE"
fi

log "pip install pinned deps"
run "$ENV_PIP" install -i "$PIP_INDEX" \
  "transformers==4.45.2" "tokenizers==0.20.1" "peft==0.13.2" \
  "accelerate==1.0.1" "datasets==2.21.0" "numpy==1.26.4" \
  "sentencepiece" "fire" "einops" "scipy" "protobuf" \
  "torchmetrics" "rouge_score" "rouge" "jieba" "fuzzywuzzy" "matplotlib" "huggingface_hub"

if [[ $DRY_RUN -eq 0 ]]; then
  SITE="$("$ENV_PY" -c 'import site; print(site.getsitepackages()[0])')"
  echo "$REPO/third_party/Jenga/src" > "$SITE/jenga_src.pth"
  log "wrote jenga_src.pth -> $SITE/jenga_src.pth"
else
  echo "[dry-run] write jenga_src.pth into env site-packages"
fi

if [[ $SKIP_DATA -eq 0 ]]; then
  log "stage facebook/opt-350m (hf-mirror)"
  run bash -lc "HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 '$ENV_PY' - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('facebook/opt-350m', local_dir='$REPO/third_party/Jenga/checkpoints/opt-350m',
                  allow_patterns=['config.json','*.bin','*.safetensors','tokenizer*','vocab.json','merges.txt','special_tokens_map.json','generation_config.json'])
print('opt-350m staged')
PY"
  log "NOTE: Jenga zips (dataset/predictor/peft_model) are NOT downloaded automatically."
  log "      Get them from the Tsinghua cloud link in third_party/Jenga/README.md and unzip into third_party/Jenga/."
  log "      Then rebuild Discovery shards:"
  log "        python src/data/build_discovery_16k.py --tag a01 --alpha 0.1"
  log "        python src/data/build_discovery_16k.py --tag a03 --alpha 0.3"
  log "        python src/data/build_warmup_16k.py --tag a01"
  log "        python src/data/build_warmup_16k.py --tag a03"
fi

verify
