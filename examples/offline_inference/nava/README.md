# NAVA Offline Inference

This example runs `NAVAPipeline`, the vLLM-Omni bridge for Baidu NAVA text/image/reference-voice audio-video generation.

NAVA is a custom upstream project, not a standard Diffusers pipeline. The first vLLM-Omni integration expects:

- A local NAVA weight directory containing `NAVA.safetensors` or `NAVA_fp8.safetensors`.
- `nava.yaml` from the current Hugging Face release, or `configs/nava.yaml` from an upstream source checkout.
- Upstream NAVA Python code installed so `import nava_src` works.
- Wan2.2 video VAE files and LTX audio VAE files in the same layout as the upstream release.
- ReDimNet in the torch hub cache when using reference timbre control.

Prompt rewrite, Qwen3-VL captioning, Gradio, and ComfyUI workflows are intentionally not part of this core pipeline. Run those in your product/application layer before calling vLLM-Omni.

## Prepare Weights

```bash
python examples/offline_inference/nava/download_nava.py \
  --local-dir /data/models/NAVA
```

To validate an existing local directory before starting inference:

```bash
python examples/offline_inference/nava/download_nava.py \
  --local-dir /data/models/NAVA \
  --verify-only
```

For bf16-only or fp8-only downloads:

```bash
python examples/offline_inference/nava/download_nava.py \
  --local-dir /data/models/NAVA \
  --bf16-only

python examples/offline_inference/nava/download_nava.py \
  --local-dir /data/models/NAVA \
  --fp8-only
```

Install upstream NAVA code separately:

```bash
git clone https://github.com/ernie-research/NAVA /data/src/NAVA
pip install -e /data/src/NAVA
```

If the upstream source checkout is also the local model directory, the download script can install it:

```bash
python examples/offline_inference/nava/download_nava.py \
  --local-dir /data/src/NAVA \
  --install-upstream
```

For reference timbre control, upstream NAVA uses ReDimNet for speaker embeddings. Prepare it during setup so runtime inference does not depend on a first-request network download:

```bash
python examples/offline_inference/nava/download_nava.py \
  --local-dir /data/models/NAVA \
  --prepare-redimnet
```

If you pass `--torch-home /data/torch-cache`, export the same `TORCH_HOME` before running offline inference or online serving.

## Text To Audio-Video

`--frames` follows upstream NAVA's temporal latent unit. The default `37`
matches upstream examples and usually decodes to about `(37 - 1) * 4 + 1`
output video frames.

```bash
python examples/offline_inference/nava/end2end.py \
  --model /data/models/NAVA \
  --prompt "清晨的海边，一名男子沿着湿润的沙滩慢跑，镜头低角度跟随。背景里有海浪声和微弱的风声。" \
  --output outputs/nava_t2av.mp4
```

## Image Conditioned Audio-Video

```bash
python examples/offline_inference/nava/end2end.py \
  --model /data/models/NAVA \
  --image infer_cases/timbre/peter.png \
  --prompt "延续首帧画面，人物看向镜头并轻声说话，背景保持电影感自然光。" \
  --output outputs/nava_i2av.mp4
```

## Reference Timbre Control

`spk_wavs[i]` is matched to the i-th `<S>...<E>` span in order.
When no speaker WAV is provided, `NAVAPipeline` disables timbre CFG by default.
If `timbre_cfg=true` is requested explicitly, speaker WAVs must be provided and aligned to the speech spans.

```bash
python examples/offline_inference/nava/end2end.py \
  --model /data/models/NAVA \
  --prompt "两个人在咖啡馆交谈。<S>Hello, nice to meet you.<E><S>Nice to meet you too.<E>" \
  --spk-wavs /path/to/speaker_a.wav /path/to/speaker_b.wav \
  --output outputs/nava_timbre.mp4
```

The pipeline raises a clear error if the number of speaker WAVs does not match the number of `<S>...<E>` spans.

## Hardware E2E

The checkpoint tests are gated by `NAVA_E2E_MODEL` so they do not run without a prepared local model directory:

```bash
export NAVA_E2E_MODEL=/data/models/NAVA
pytest -q tests/e2e/offline_inference/test_nava_expansion.py -m "full_model and diffusion"
```

The test requires upstream NAVA to be importable, the prepared weight directory to pass `download_nava.py --verify-only`, and a GPU with enough memory for the selected checkpoint.

## Safety

Use reference images and reference voices only when you have the necessary rights and consent. Label generated audio-video content as synthetic when it is shared outside controlled evaluation.

## Notes

- The first bridge version supports one prompt per vLLM-Omni request. Multi-request throughput batching should stay in the business layer or a future step-execution scheduler integration.
- The bridge uses upstream NAVA's built-in negative prompts. Custom `negative_prompt`, video negative, and audio negative prompts are rejected until the upstream denoise path is ported or exposed.
- Online continuous batching, Cache-DiT, TP/HSDP, and native vLLM-Omni SP are not claimed by this bridge. Upstream NAVA has its own Ulysses SP path, but this bridge calls the upstream single-process pipeline. Native vLLM-Omni parallelism should be added after the MMDiT backbone is ported.
- Use `--nava-weight-dtype fp8_e4m3fn` with the upstream FP8 checkpoint when upstream `NAVA_FP8` is installed.
