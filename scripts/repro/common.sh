#!/usr/bin/env bash
# SMELL-v3 复现脚本公共设置（01–06 均 source 本文件）
#   PY / GPU / DRY_RUN / STAMP / LOG_ROOT 均可由环境变量覆盖
#   ARCHIVE_ROOT 非空时，脚本会把最后一轮 adapter 归档到该目录
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/SMELL/bin/python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAMP="${STAMP:-$(date +%y%m%d-%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$REPO/temp/logs/repro_$STAMP}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-}"
mkdir -p "$LOG_ROOT"

# 前台运行并把输出写日志；DRY_RUN=1 时只打印命令
launch() {
  local log="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "$@"; echo
  else
    "$@" > "$log" 2>&1
  fi
}

# 归档最后一轮 adapter（与历史归档同结构：adapter_roundNNN + metrics + config + SHA256SUMS）
maybe_archive() {
  local out_cell="$1" tag="$2" last_round="$3" name="$4"
  [ -n "$ARCHIVE_ROOT" ] || return 0
  local last; last=$(printf "adapter_round%03d" "$last_round")
  [ -d "$out_cell/$tag/$last" ] || return 0
  mkdir -p "$ARCHIVE_ROOT/$name"
  cp -r "$out_cell/$tag/$last" "$ARCHIVE_ROOT/$name/$last"
  cp -f "$out_cell/$tag/metrics.jsonl" "$out_cell/$tag/config.json" "$ARCHIVE_ROOT/$name/" 2>/dev/null
  (cd "$ARCHIVE_ROOT/$name" && sha256sum "$last"/*.safetensors "$last"/adapter_config.json > SHA256SUMS 2>/dev/null)
  echo "[$(date +%H:%M:%S)] archived -> $ARCHIVE_ROOT/$name"
}
