#!/usr/bin/env bash
# SMELL-v3 环境自检（Jenga hello_world 风格）：依赖/GPU/底座/数据/权重/前向 smoke
# 用法：bash scripts/check-env.sh [--no-gpu] [--json temp/env_report.json]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/SMELL/bin/python}"
cd "$REPO"
exec "$PYTHON" src/hello_world.py "$@"
