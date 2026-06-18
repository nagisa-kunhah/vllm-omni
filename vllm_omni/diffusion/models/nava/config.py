# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_NAVA_MODEL_TYPE = "NAVA"
DEFAULT_NAVA_MODEL_INDEX = {
    "_class_name": "NAVAPipeline",
    "nava_ckpt": "NAVA.safetensors",
    "fp8_ckpt": "NAVA_fp8.safetensors",
    "config": "configs/nava.yaml",
    "wan_dir": "Wan2.2-TI2V-5B",
    "audio_vae_dir": "params",
}

_SPEECH_SPAN_RE = re.compile(r"<S>(.*?)<E>", re.DOTALL)

NAVA_CONFIG_ALIAS_MAP = {
    "nava_ckpt": "ckpt_name",
    "fp8_ckpt": "fp8_ckpt_name",
    "config": "config_name",
    "audio_vae_dir": "audio_vae_ckpt_dir",
    "num_inference_steps": "num_steps",
    "num_frames": "frames",
    "height": "log_height",
    "width": "log_width",
}


@dataclass(frozen=True)
class NAVAConfig:
    """NAVA runtime defaults used by the vLLM-Omni bridge pipeline."""

    model_type: str = DEFAULT_NAVA_MODEL_TYPE
    modality: str = "audio_video"
    use_bf16: bool = True
    audio_latent_ch: int = 128
    video_latent_ch: int = 48
    lambda_ddpm: float = 1.0
    patch_size: int = 16
    log_height: int = 704
    log_width: int = 1280
    frames: int = 37
    fps: int = 24
    num_steps: int = 50
    audio_tokens_per_sec: float = 25.0
    audio_sample_rate: int = 16000
    video_guidance_scale: float = 3.0
    audio_guidance_scale: float = 2.0
    video_align_guidance_scale: float = 3.0
    audio_align_guidance_scale: float = 2.0
    align_3d_cfg: bool = True
    timbre_cfg: bool = True
    timbre_align_guidance_scale: float = 3.0
    negative_prompt_mode: bool = True
    ckpt_name: str = "NAVA.safetensors"
    fp8_ckpt_name: str = "NAVA_fp8.safetensors"
    config_name: str = "configs/nava.yaml"
    audio_vae_ckpt_dir: str = "params"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> NAVAConfig:
        if not raw:
            return cls()

        data = dict(raw)
        model_block = data.get("model")
        if isinstance(model_block, dict) and "audio_vae_ckpt_dir" not in data:
            data["audio_vae_ckpt_dir"] = model_block.get("audio_vae_ckpt_dir", cls.audio_vae_ckpt_dir)

        data_block = data.get("data")
        if isinstance(data_block, dict):
            if "audio_tokens_per_sec" not in data:
                data["audio_tokens_per_sec"] = data_block.get("audio_tokens_per_sec", cls.audio_tokens_per_sec)
            if "fps" not in data and "video_fps" in data_block:
                data["fps"] = data_block["video_fps"]

        for old_key, new_key in NAVA_CONFIG_ALIAS_MAP.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        valid_keys = cls.__dataclass_fields__.keys()
        kwargs = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**kwargs)

    @property
    def target_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.use_bf16 else torch.float16

    def video_latent_hw(self, height: int | None = None, width: int | None = None) -> tuple[int, int]:
        resolved_height = int(height or self.log_height)
        resolved_width = int(width or self.log_width)
        if resolved_height % self.patch_size != 0 or resolved_width % self.patch_size != 0:
            raise ValueError(
                "NAVA height and width must be divisible by "
                f"{self.patch_size}: got height={resolved_height}, width={resolved_width}."
            )
        return resolved_height // self.patch_size, resolved_width // self.patch_size

    def audio_latent_length(self, frames: int | None = None, fps: int | None = None) -> int:
        resolved_frames = int(frames or self.frames)
        resolved_fps = int(fps or self.fps)
        video_duration = ((resolved_frames - 1) * 4 + 1) / resolved_fps
        return max(1, int(video_duration * self.audio_tokens_per_sec))


@dataclass(frozen=True)
class NAVASpeakerCondition:
    wavs: list[Any]
    spans: list[str]


@dataclass(frozen=True)
class NAVARequestContext:
    prompt: str
    image: Any | None
    speaker_condition: NAVASpeakerCondition | None
    height: int
    width: int
    frames: int
    fps: int
    seed: int
    num_steps: int
    video_guidance_scale: float
    audio_guidance_scale: float
    video_align_guidance_scale: float
    audio_align_guidance_scale: float
    align_3d_cfg: bool
    timbre_cfg: bool
    timbre_align_guidance_scale: float
    negative_prompt_mode: bool
    offload_backbone: bool
    tiled_vae: bool
    vae_tile_size: tuple[int, int]
    vae_tile_stride: tuple[int, int]
    save_vid_latent: bool

    @property
    def is_i2v(self) -> bool:
        return self.image is not None


def parse_speech_spans(prompt: str) -> list[str]:
    return [match.group(1) for match in _SPEECH_SPAN_RE.finditer(prompt or "")]


def count_speech_spans(prompt: str) -> int:
    return len(parse_speech_spans(prompt))


def inject_speaker_sentinel(prompt: str) -> str:
    """Match upstream T2AVDataset's T5 speech sentinel insertion."""

    return (prompt or "").replace("<S>", "<S><extra_id_2>")
