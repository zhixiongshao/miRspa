#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#
# Render: bind 0.0.0.0; use Render-injected PORT when present.
# Large assets: app.py will download best_model.pth and .fa.gz from Zenodo
# (https://zenodo.org/records/19586578) if files are missing, unless
# DISABLE_ZENODO_ASSET_FETCH=1. Override URLs with CHECKPOINT_URL / GENOME_GZ_URL.
#
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"

DATA_DIR="${DATA_DIR:-/var/data}"
mkdir -p "${DATA_DIR}"

CHECKPOINT="${CHECKPOINT:-${DATA_DIR}/best_model.pth}"
# 默认使用 Zenodo 上的 .fa.gz；服务只按需解压读取一条染色体到内存，不整库落盘解压。
GENOME="${GENOME:-${DATA_DIR}/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz}"

# Render free instances don't have GPU; default to CPU.
DEVICE="${DEVICE:-cpu}"
PADDING_SIZE="${PADDING_SIZE:-1000}"

python "${SCRIPT_DIR}/app.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --checkpoint "${CHECKPOINT}" \
  --genome "${GENOME}" \
  --device "${DEVICE}" \
  --padding_size "${PADDING_SIZE}"
