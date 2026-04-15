#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#
# Render/Vercel-like environments require binding to 0.0.0.0 and using $PORT.
# You can override any of these via environment variables.
#
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-${PORT:-8765}}"

# Default to a generic data mount path. Adjust or override in Render env vars.
CHECKPOINT="${CHECKPOINT:-/var/data/best_model.pth}"
GENOME="${GENOME:-/var/data/Homo_sapiens.GRCh38.dna.primary_assembly.fa}"

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
