#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/data/models/NAVA}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nava}"

if command -v vllm-omni >/dev/null 2>&1; then
  VLLM_CLI=(vllm-omni)
elif command -v vllm >/dev/null 2>&1; then
  VLLM_CLI=(vllm)
else
  echo "Could not find vllm-omni or vllm in PATH. Install vLLM-Omni before starting NAVA serving." >&2
  exit 127
fi

"${VLLM_CLI[@]}" serve "${MODEL}" --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --model-class-name NAVAPipeline \
  --enforce-eager \
  --disable-log-stats
