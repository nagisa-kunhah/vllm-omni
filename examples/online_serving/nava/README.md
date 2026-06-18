# NAVA Online Serving

NAVA serving uses the same `NAVAPipeline` bridge as offline inference. The model path must be a prepared local directory, not a remote HF ID.

Validate the model directory before starting the server:

```bash
python examples/offline_inference/nava/download_nava.py \
  --local-dir /data/models/NAVA \
  --verify-only
```

If timbre control is enabled, prepare ReDimNet with the offline `download_nava.py --prepare-redimnet` step first. When a custom `--torch-home` was used during setup, export the same `TORCH_HOME` before starting the server.

## Start Server

```bash
MODEL=/data/models/NAVA \
PORT=8000 \
bash examples/online_serving/nava/run_server.sh
```

The script runs:

```bash
vllm serve "$MODEL" --omni \
  --model-class-name NAVAPipeline \
  --served-model-name nava \
  --port "$PORT" \
  --enforce-eager \
  --disable-log-stats
```

## Send Request

```bash
bash examples/online_serving/nava/run_curl_nava.sh
```

For image-conditioned generation, use `image_reference`:

```bash
curl -sS -X POST "http://${HOST:-127.0.0.1}:${PORT:-8000}/v1/videos" \
  -H "Accept: application/json" \
  -F "model=nava" \
  -F "prompt=延续首帧画面，人物看向镜头并轻声说话，背景保持电影感自然光。" \
  -F 'image_reference={"image_url":"data:image/png;base64,<base64-png>"}' \
  -F "height=704" \
  -F "width=1280" \
  -F "num_frames=37" \
  -F "fps=24" \
  -F "num_inference_steps=50"
```

For single-speaker timbre control, use `audio_reference` and one `<S>...<E>` span:

```bash
curl -sS -X POST "http://${HOST:-127.0.0.1}:${PORT:-8000}/v1/videos" \
  -H "Accept: application/json" \
  -F "model=nava" \
  -F "prompt=一个人对镜头说话。<S>Hello, this is a synthetic demo.<E>" \
  -F 'audio_reference={"audio_url":"data:audio/wav;base64,<base64-wav>"}' \
  -F "height=704" \
  -F "width=1280" \
  -F "num_frames=37" \
  -F "fps=24" \
  -F "num_inference_steps=50"
```

Multi-speaker `spk_wavs` is supported by the pipeline contract, but the generic video endpoint currently exposes one `audio_reference`. For multiple reference voices, use the offline API or an application layer that calls `Omni.generate()` with `multi_modal_data["spk_wavs"]`.

The core inference engine does not perform prompt rewrite or image captioning.

The `/v1/videos` endpoint accepts NAVA-specific guidance values through the `extra_params` form field as a JSON string, not `extra_body`.

## Hardware E2E

The online serving E2E test is gated by `NAVA_E2E_MODEL` so it does not run without a prepared local model directory:

```bash
export NAVA_E2E_MODEL=/data/models/NAVA
pytest -q tests/e2e/online_serving/test_nava_expansion.py -m "full_model and diffusion"
```

The test starts `vllm serve` with `--model-class-name NAVAPipeline --enforce-eager` and sends one `/v1/videos` text-to-audio-video request.

## Safety

Use reference images and reference voices only when you have the necessary rights and consent. Label generated audio-video content as synthetic when it is shared outside controlled evaluation.

## Current Scope

- Single request generation is the supported path.
- Custom `negative_prompt`, video negative, and audio negative prompts are rejected in the bridge because upstream NAVA's sample path uses built-in negative prompts.
- Prompt rewrite, Qwen3-VL captioning, Gradio, and ComfyUI belong above the inference engine.
- Native vLLM-Omni continuous batching and parallel acceleration are future work after the upstream NAVA backbone is ported into vLLM-Omni modules. Do not enable `--usp`, Cache-DiT, TP, or HSDP for this bridge unless you have validated the upstream runtime path locally.
