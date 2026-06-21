#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

response="$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -F "model=nava" \
    -F "prompt=A person speaks while standing near the sea." \
    -F 'extra_params={"num_frames":37,"fps":24,"num_inference_steps":50}'
)"

video_id="$(python -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"${response}")"
curl -sS "${BASE_URL}/v1/videos/${video_id}"
curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o nava_output.mp4
