#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/data/models/NAVA}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nava}"

vllm serve "${MODEL}" --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --model-class-name NAVAPipeline \
  --enforce-eager \
  --disable-log-stats
