#!/usr/bin/env bash
# NAVA text-to-audio-video curl example using the async video job API.

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${SERVED_MODEL_NAME:-nava}"
OUTPUT_PATH="${OUTPUT_PATH:-nava_output.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "model=${MODEL}" \
    -F "prompt=清晨的海边，一名男子沿着湿润的沙滩慢跑，镜头低角度跟随。背景里有海浪声和微弱的风声。" \
    -F "height=704" \
    -F "width=1280" \
    -F "num_frames=37" \
    -F "fps=24" \
    -F "num_inference_steps=50" \
    -F "seed=100" \
    -F 'extra_params={"video_guidance_scale":3.0,"audio_guidance_scale":2.0}'
)

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create NAVA video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created NAVA video job ${video_id}"
echo "${create_response}" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "NAVA video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq .
      break
      ;;
    failed)
      echo "NAVA video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"
echo "Saved NAVA audio-video output to ${OUTPUT_PATH}"
