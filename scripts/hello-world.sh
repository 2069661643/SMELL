#!/usr/bin/env bash
# SMELL 3 hello_world NEW — SMELL-v3 环境自检（Jenga hello_world 风格）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/SMELL/bin/python}"

cd "$REPO"
exec "$PYTHON" src/hello_world.py "$@"
