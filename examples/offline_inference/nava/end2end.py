# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline NAVA text/image/speaker-reference audio-video generation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synchronized video+audio with NAVA.")
    parser.add_argument("--model", required=True, help="Local NAVA directory prepared by download_nava.py.")
    parser.add_argument("--prompt", required=True, help="NAVA prompt. Use <S>...<E> spans for reference timbre.")
    parser.add_argument("--image", default=None, help="Optional first-frame image path for I2AV.")
    parser.add_argument("--spk-wavs", nargs="*", default=None, help="Optional reference WAVs aligned to <S>...<E> spans.")
    parser.add_argument("--output", default="nava_output.mp4", help="Output MP4 path.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=37)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--video-guidance-scale", type=float, default=3.0)
    parser.add_argument("--audio-guidance-scale", type=float, default=2.0)
    parser.add_argument("--video-align-guidance-scale", type=float, default=3.0)
    parser.add_argument("--audio-align-guidance-scale", type=float, default=2.0)
    parser.add_argument("--timbre-align-guidance-scale", type=float, default=3.0)
    parser.add_argument(
        "--disable-timbre-cfg",
        action="store_true",
        help="Disable timbre CFG even when --spk-wavs is set.",
    )
    parser.add_argument("--nava-weight-dtype", choices=["auto", "bf16", "fp8_e4m3fn"], default="auto")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    return parser.parse_args()


def _to_uint8_frames(video: np.ndarray) -> np.ndarray:
    """Normalize NAVA [-1, 1] video to [T, H, W, C] uint8."""

    frames = np.asarray(video)
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim != 4:
        raise RuntimeError(f"Unexpected NAVA video shape: {frames.shape}")
    if frames.shape[1] in (1, 3):
        frames = np.transpose(frames, (0, 2, 3, 1))
    return (np.clip((frames.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0) * 255).round().astype(np.uint8)


def _build_prompt(args: argparse.Namespace) -> str | dict[str, Any]:
    prompt: str | dict[str, Any] = {"prompt": args.prompt, "multi_modal_data": {}}
    if args.image:
        prompt["multi_modal_data"]["image"] = Image.open(args.image).convert("RGB")
    if args.spk_wavs:
        prompt["multi_modal_data"]["spk_wavs"] = args.spk_wavs
    if not prompt["multi_modal_data"]:
        prompt = args.prompt
    return prompt


def _build_sampling_params(args: argparse.Namespace) -> OmniDiffusionSamplingParams:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_args={
            "video_guidance_scale": args.video_guidance_scale,
            "audio_guidance_scale": args.audio_guidance_scale,
            "video_align_guidance_scale": args.video_align_guidance_scale,
            "audio_align_guidance_scale": args.audio_align_guidance_scale,
            "timbre_align_guidance_scale": args.timbre_align_guidance_scale,
            "timbre_cfg": bool(args.spk_wavs) and not args.disable_timbre_cfg,
        },
    )


def main() -> None:
    args = parse_args()
    prompt = _build_prompt(args)
    sampling_params = _build_sampling_params(args)

    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(
        model=args.model,
        model_class_name="NAVAPipeline",
        enforce_eager=args.enforce_eager,
        custom_pipeline_args={"nava_weight_dtype": args.nava_weight_dtype},
    )
    started = time.perf_counter()
    outputs = omni.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - started

    if not outputs:
        raise RuntimeError("No output returned from NAVAPipeline.")
    result = outputs[0]
    if not result.images:
        raise RuntimeError("No video payload returned from NAVAPipeline.")
    video_frames = _to_uint8_frames(result.images[0])
    mm = result.multimodal_output or {}
    audio = mm.get("audio")
    if audio is not None:
        audio = np.squeeze(np.asarray(audio)).astype(np.float32)
    audio_sample_rate = int(mm.get("audio_sample_rate", 16000))
    fps = float(mm.get("fps", args.fps))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(mux_video_audio_bytes(video_frames, audio, fps=fps, audio_sample_rate=audio_sample_rate))
    print(f"Saved {output_path} in {elapsed:.2f}s")
    omni.close()


if __name__ == "__main__":
    main()
