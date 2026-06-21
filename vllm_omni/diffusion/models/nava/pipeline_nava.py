# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from inspect import signature
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.nava.audio_vae import NAVAAudioVAE
from vllm_omni.diffusion.models.nava.config import (
    DEFAULT_NAVA_MODEL_INDEX,
    NAVA_CONFIG_ALIAS_MAP,
    NAVAConfig,
    NAVARequestContext,
    NAVASpeakerCondition,
    inject_speaker_sentinel,
    parse_speech_spans,
)
from vllm_omni.diffusion.models.nava.nava_transformer import NAVATransformer
from vllm_omni.diffusion.models.nava.scheduler import NAVAFlowMatchScheduler
from vllm_omni.diffusion.models.nava.speaker import NAVASpeakerEncoder
from vllm_omni.diffusion.models.nava.utils import as_bool, image_to_tensor, move_to_device, resolve_num_frames
from vllm_omni.diffusion.models.nava.video_vae import NAVAVideoVAE
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def get_nava_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(output: dict[str, Any] | Any, output_type: str = "np", sampling_params=None):
        if output_type == "latent":
            return output
        if isinstance(output, dict):
            video = output.get("video")
            audio = output.get("audio")
            audio_sample_rate = output.get("audio_sample_rate", NAVAPipeline.audio_sample_rate)
            fps = output.get("fps", _resolve_output_fps(sampling_params, NAVAPipeline.fps))
        else:
            video = output
            audio = None
            audio_sample_rate = NAVAPipeline.audio_sample_rate
            fps = _resolve_output_fps(sampling_params, NAVAPipeline.fps)
        return {
            "video": _normalize_video_output(video, output_type),
            "audio": audio,
            "audio_sample_rate": int(audio_sample_rate),
            "fps": fps,
        }

    return post_process_func


class NAVAPipeline(
    nn.Module,
    SupportImageInput,
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    support_image_input: ClassVar[bool] = True
    support_audio_input: ClassVar[bool] = True
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 16000
    fps: ClassVar[int] = 24
    dummy_run_num_frames: ClassVar[int] = 0

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "speaker_encoder"]
    _vae_modules: ClassVar[list[str]] = ["video_vae", "audio_vae"]
    _resident_modules: ClassVar[list[str]] = []

    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "align_3d_cfg",
            "audio_align_guidance_scale",
            "audio_guidance_scale",
            "frames",
            "fps",
            "height",
            "negative_prompt",
            "num_frames",
            "num_inference_steps",
            "num_steps",
            "seed",
            "spk_wavs",
            "timbre_align_guidance_scale",
            "timbre_cfg",
            "video_align_guidance_scale",
            "video_guidance_scale",
            "width",
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.nava_config = self._load_nava_config(od_config)
        self.audio_sample_rate = self.nava_config.audio_sample_rate
        self.fps = self.nava_config.fps
        self.scheduler = NAVAFlowMatchScheduler(shift=5.0)
        self.scheduler_audio = NAVAFlowMatchScheduler(shift=5.0)
        self._validate_runtime_features()
        self._init_native_components()
        self._init_weight_sources()
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.transformer.load_weights(weights)

    def forward(self, request: OmniDiffusionRequest, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        ctx = self._parse_request(request)
        generator = self._make_generator(ctx.seed)

        if ctx.speaker_condition is None:
            text_embeds = self._encode_text(ctx.prompt)
            speaker_positions = None
        else:
            text_embeds, speaker_positions = self._encode_text_with_speaker_positions(ctx.prompt)
        negative_text_embeds = self._encode_text(ctx.negative_prompt)
        image_embeds = self._encode_image(ctx)
        speaker_embeds = self._encode_speakers(ctx)
        if speaker_embeds is not None and not speaker_positions:
            raise ValueError("NAVA speaker references require <S>...<E> spans with <extra_id_2> speaker markers.")
        if speaker_embeds is not None and len(speaker_positions[0]) != speaker_embeds.shape[0]:
            raise ValueError(
                "NAVA speaker embedding count must match tokenizer speaker marker count: "
                f"got {speaker_embeds.shape[0]} embedding(s) and {len(speaker_positions[0])} marker(s)."
            )
        latents = self._prepare_latents(ctx, generator)

        video_latents, audio_latents = self._denoise(
            ctx=ctx,
            video_latents=latents["video"],
            audio_latents=latents["audio"],
            text_embeds=text_embeds,
            negative_text_embeds=negative_text_embeds,
            image_embeds=image_embeds,
            speaker_embeds=speaker_embeds,
            speaker_positions=speaker_positions,
        )
        video = self._decode_video(video_latents, ctx)
        audio = self._decode_audio(audio_latents)
        return DiffusionOutput(
            output={
                "video": video,
                "audio": audio,
                "audio_sample_rate": self.audio_sample_rate,
                "fps": ctx.fps,
            }
        )

    def _validate_runtime_features(self) -> None:
        if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload:
            raise ValueError("NAVAPipeline CPU and layerwise offload are not verified yet.")
        pc = self.od_config.parallel_config
        enabled = {
            "tensor_parallel_size": pc.tensor_parallel_size,
            "sequence_parallel_size": pc.sequence_parallel_size,
            "cfg_parallel_size": pc.cfg_parallel_size,
            "vae_patch_parallel_size": pc.vae_patch_parallel_size,
            "pipeline_parallel_size": pc.pipeline_parallel_size,
            "data_parallel_size": pc.data_parallel_size,
        }
        enabled = {key: value for key, value in enabled.items() if int(value or 1) != 1}
        if enabled or pc.use_hsdp or pc.enable_expert_parallel:
            raise ValueError(
                "NAVAPipeline native parallel and sharding modes are not verified yet. "
                f"Unsupported settings: {enabled}, use_hsdp={pc.use_hsdp}, "
                f"enable_expert_parallel={pc.enable_expert_parallel}."
            )

    def _init_native_components(self) -> None:
        overrides = self.od_config.custom_pipeline_args or {}
        if overrides:
            self.text_encoder = overrides.get("text_encoder") or _MissingNAVAComponent("text_encoder")
            self.video_vae = overrides.get("video_vae") or _MissingNAVAComponent("video_vae")
            self.audio_vae = overrides.get("audio_vae") or _MissingNAVAComponent("audio_vae")
            self.speaker_encoder = overrides.get("speaker_encoder") or _MissingNAVAComponent("speaker_encoder")
            self.transformer = overrides.get("transformer") or _MissingNAVAComponent("transformer")
        else:
            model_root = self._require_local_model_root()
            self.text_encoder = _NAVATextEncoder(model_root, self.nava_config, self.device)
            self.video_vae = NAVAVideoVAE(model_root, self.nava_config, self.device)
            self.audio_vae = NAVAAudioVAE(model_root, self.nava_config)
            self.speaker_encoder = NAVASpeakerEncoder(model_root, self.nava_config)
            self.transformer = NAVATransformer(self.nava_config)
        self.model = self.transformer
        self.vae = self.video_vae
        self.to(self.device)

    def _require_local_model_root(self) -> str:
        model = str(self.od_config.model or "")
        if not model or not os.path.isdir(model):
            raise FileNotFoundError(
                "NAVAPipeline requires a local baidu/NAVA directory prepared by "
                "examples/offline_inference/nava/download_nava.py."
            )
        return model

    def _init_weight_sources(self) -> None:
        model = self.od_config.model
        if not model:
            return
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder=None,
                revision=self.od_config.revision,
                prefix="",
                fall_back_to_pt=False,
                allow_patterns_overrides=[self.nava_config.ckpt_name],
            )
        ]

    def _load_nava_config(self, od_config: OmniDiffusionConfig) -> NAVAConfig:
        config_data: dict[str, Any] = dict(DEFAULT_NAVA_MODEL_INDEX)
        if od_config.model and os.path.isdir(str(od_config.model)):
            index_path = os.path.join(str(od_config.model), "model_index.json")
            if os.path.exists(index_path):
                with open(index_path, encoding="utf-8") as f:
                    index_data = json.load(f)
                if isinstance(index_data, dict):
                    config_data.update(index_data)
            config_name = str(config_data.get("config") or DEFAULT_NAVA_MODEL_INDEX["config"])
            config_path = os.path.join(str(od_config.model), config_name)
            if not os.path.exists(config_path) and config_name == "configs/nava.yaml":
                legacy_path = os.path.join(str(od_config.model), "nava.yaml")
                if os.path.exists(legacy_path):
                    config_path = legacy_path
                    config_data["config"] = "nava.yaml"
            if os.path.exists(config_path):
                config_data.update(_load_yaml(config_path))
            joint_config_path = os.path.join(str(od_config.model), "config.json")
            if os.path.exists(joint_config_path):
                with open(joint_config_path, encoding="utf-8") as f:
                    joint_config = json.load(f)
                if isinstance(joint_config, dict):
                    config_data.setdefault("model", {})["joint_config_data"] = joint_config

        explicit = dict(od_config.model_config or {})
        for old_key, new_key in NAVA_CONFIG_ALIAS_MAP.items():
            if old_key in explicit:
                explicit[new_key] = explicit[old_key]
        config_data.update(explicit)
        config_data.update(od_config.additional_config or {})
        return NAVAConfig.from_dict(config_data)

    def _parse_request(self, request: OmniDiffusionRequest) -> NAVARequestContext:
        if len(request.prompts) != 1:
            raise ValueError("NAVAPipeline currently supports one prompt per request. Use request-level batching.")
        prompt_data = request.prompts[0]
        prompt = prompt_data if isinstance(prompt_data, str) else str(prompt_data.get("prompt", ""))
        if not prompt:
            raise ValueError("NAVAPipeline requires a non-empty prompt.")
        multi_modal_data = {} if isinstance(prompt_data, str) else (prompt_data.get("multi_modal_data") or {})
        sp = request.sampling_params
        extra = sp.extra_args or {}
        speaker_condition = self._parse_speaker_condition(prompt, multi_modal_data, extra)
        image = multi_modal_data.get("image")

        frames = self.nava_config.normalize_output_frames(
            resolve_num_frames(sp.num_frames, extra, self.nava_config.frames)
        )
        return NAVARequestContext(
            prompt=prompt,
            negative_prompt=str(extra.get("negative_prompt", self.nava_config.negative_prompt)),
            image=image,
            speaker_condition=speaker_condition,
            height=int(sp.height or extra.get("height") or self.nava_config.log_height),
            width=int(sp.width or extra.get("width") or self.nava_config.log_width),
            frames=frames,
            fps=float(sp.resolved_frame_rate or extra.get("fps") or self.nava_config.fps),
            seed=int(sp.seed if sp.seed is not None else extra.get("seed", 100)),
            num_steps=int(
                sp.num_inference_steps
                or extra.get("num_inference_steps")
                or extra.get("num_steps")
                or self.nava_config.num_steps
            ),
            video_guidance_scale=float(extra.get("video_guidance_scale", self.nava_config.video_guidance_scale)),
            audio_guidance_scale=float(extra.get("audio_guidance_scale", self.nava_config.audio_guidance_scale)),
            video_align_guidance_scale=float(
                extra.get("video_align_guidance_scale", self.nava_config.video_align_guidance_scale)
            ),
            audio_align_guidance_scale=float(
                extra.get("audio_align_guidance_scale", self.nava_config.audio_align_guidance_scale)
            ),
            timbre_align_guidance_scale=float(
                extra.get("timbre_align_guidance_scale", self.nava_config.timbre_align_guidance_scale)
            ),
            align_3d_cfg=as_bool(extra.get("align_3d_cfg", self.nava_config.align_3d_cfg)),
            timbre_cfg=self._resolve_timbre_cfg(extra, speaker_condition),
        )

    def _parse_speaker_condition(
        self,
        prompt: str,
        multi_modal_data: dict[str, Any],
        extra: dict[str, Any],
    ) -> NAVASpeakerCondition | None:
        wavs = multi_modal_data.get("spk_wavs", multi_modal_data.get("audio", extra.get("spk_wavs")))
        if wavs is None:
            return None
        if not isinstance(wavs, list):
            wavs = [wavs]
        spans = parse_speech_spans(prompt)
        if len(wavs) != len(spans):
            raise ValueError(
                "NAVA speaker reference count must match <S>...<E> speech span count: "
                f"got {len(wavs)} reference wav(s) and {len(spans)} span(s)."
            )
        return NAVASpeakerCondition(wavs=list(wavs), spans=spans)

    def _resolve_timbre_cfg(self, extra: dict[str, Any], speaker_condition: NAVASpeakerCondition | None) -> bool:
        if "timbre_cfg" in extra:
            requested = as_bool(extra["timbre_cfg"])
            if requested and speaker_condition is None:
                raise ValueError("NAVA timbre_cfg requires reference speaker WAVs aligned to <S>...<E> spans.")
            return requested
        return bool(self.nava_config.timbre_cfg and speaker_condition is not None)

    def _encode_text(self, prompt: str) -> torch.Tensor:
        return self._run_text_encoder(prompt, return_speaker_positions=False)[0]

    def _encode_text_with_speaker_positions(self, prompt: str) -> tuple[torch.Tensor, list[list[int]] | None]:
        return self._run_text_encoder(prompt, return_speaker_positions=True)

    def _run_text_encoder(
        self,
        prompt: str,
        *,
        return_speaker_positions: bool,
    ) -> tuple[torch.Tensor, list[list[int]] | None]:
        caption = inject_speaker_sentinel(prompt)
        encoder = self.text_encoder
        if hasattr(encoder, "encode"):
            # Text embedding: [batch, text_tokens, text_dim], shared by video
            # and audio denoising branches.
            kwargs: dict[str, Any] = {}
            if "return_speaker_positions" in signature(encoder.encode).parameters:
                kwargs["return_speaker_positions"] = return_speaker_positions
            result = encoder.encode([caption], device=self.device, dtype=self.nava_config.target_dtype, **kwargs)
            if isinstance(result, tuple):
                return result
            return result, None
        raise TypeError("NAVA text_encoder must expose encode(texts, device, dtype).")

    def _encode_image(self, ctx: NAVARequestContext) -> torch.Tensor | None:
        if ctx.image is None:
            return None
        image = move_to_device(image_to_tensor(ctx.image, ctx.height, ctx.width), self.device)
        if not hasattr(self.video_vae, "encode_first_frame"):
            raise TypeError("NAVA video_vae must expose encode_first_frame(image).")
        # Image embedding: first-frame pixels become a conditioning latent for
        # the video denoising branch.
        return self.video_vae.encode_first_frame(image)

    def _encode_speakers(self, ctx: NAVARequestContext) -> torch.Tensor | None:
        if ctx.speaker_condition is None:
            return None
        if not hasattr(self.speaker_encoder, "encode"):
            raise TypeError("NAVA speaker_encoder must expose encode(wavs, device, dtype).")
        # Speaker embedding: reference WAVs are ordered to match <S>...<E>
        # spans and only condition the timbre branch.
        return self.speaker_encoder.encode(
            ctx.speaker_condition.wavs,
            device=self.device,
            dtype=self.nava_config.target_dtype,
        )

    def _prepare_latents(self, ctx: NAVARequestContext, generator: torch.Generator) -> dict[str, torch.Tensor]:
        latent_h, latent_w = self.nava_config.video_latent_hw(ctx.height, ctx.width)
        latent_frames = self.nava_config.video_latent_frames(ctx.frames)
        video_tokens = latent_frames * latent_h * latent_w
        audio_tokens = self.nava_config.audio_latent_length(ctx.frames, ctx.fps)
        # Video/audio latent initialization fixes the generated duration and
        # resolution before denoising starts.
        return {
            "video": torch.randn(
                1,
                video_tokens,
                self.nava_config.video_latent_ch,
                generator=generator,
                device=self.device,
                dtype=self.nava_config.target_dtype,
            ),
            "audio": torch.randn(
                1,
                audio_tokens,
                self.nava_config.audio_latent_ch,
                generator=generator,
                device=self.device,
                dtype=self.nava_config.target_dtype,
            ),
        }

    def _denoise(
        self,
        *,
        ctx: NAVARequestContext,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        text_embeds: torch.Tensor,
        negative_text_embeds: torch.Tensor | None,
        image_embeds: torch.Tensor | None,
        speaker_embeds: torch.Tensor | None,
        speaker_positions: list[list[int]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = self.scheduler.set_timesteps(ctx.num_steps, device=self.device)
        self.scheduler_audio.set_timesteps(ctx.num_steps, device=self.device)
        video_grid = self._video_grid(ctx)
        for step_index, timestep in enumerate(timesteps):
            timestep_tensor = timestep.reshape(1)
            positive = self.transformer(
                video_latents=video_latents,
                audio_latents=audio_latents,
                timestep=timestep_tensor,
                text_embeds=text_embeds,
                image_embeds=image_embeds,
                speaker_embeds=speaker_embeds,
                speaker_positions=speaker_positions,
                video_grid=video_grid,
                step_index=step_index,
            )
            negative = None
            if negative_text_embeds is not None:
                negative = self.transformer(
                    video_latents=video_latents,
                    audio_latents=audio_latents,
                    timestep=timestep_tensor,
                    text_embeds=negative_text_embeds,
                    image_embeds=image_embeds,
                    speaker_embeds=None,
                    speaker_positions=None,
                    video_grid=video_grid,
                    step_index=step_index,
                )
            align = None
            if ctx.align_3d_cfg:
                align = self.transformer(
                    video_latents=video_latents,
                    audio_latents=audio_latents,
                    timestep=timestep_tensor,
                    text_embeds=text_embeds,
                    image_embeds=image_embeds,
                    speaker_embeds=speaker_embeds,
                    speaker_positions=speaker_positions,
                    masking_modality=True,
                    video_grid=video_grid,
                    step_index=step_index,
                )
            timbre = None
            if ctx.timbre_cfg and speaker_embeds is not None:
                timbre = self.transformer(
                    video_latents=video_latents,
                    audio_latents=audio_latents,
                    timestep=timestep_tensor,
                    text_embeds=text_embeds,
                    image_embeds=image_embeds,
                    speaker_embeds=None,
                    speaker_positions=speaker_positions,
                    video_grid=video_grid,
                    step_index=step_index,
                )
            noise = self._combine_guidance(ctx, positive, negative, align=align, timbre=timbre)
            # Denoise step: FlowMatch updates video/audio latents with matching
            # timesteps so generated duration stays synchronized.
            video_latents = self.scheduler.step(
                noise["video"].to(torch.float32), timestep, video_latents.to(torch.float32)
            )
            audio_latents = self.scheduler_audio.step(
                noise["audio"].to(torch.float32), timestep, audio_latents.to(torch.float32)
            )
            video_latents = video_latents.to(self.nava_config.target_dtype)
            video_latents = self._apply_first_frame_condition(video_latents, image_embeds, video_grid)
            audio_latents = audio_latents.to(self.nava_config.target_dtype)
        return video_latents, audio_latents

    def _video_grid(self, ctx: NAVARequestContext) -> tuple[int, int, int]:
        latent_h, latent_w = self.nava_config.video_latent_hw(ctx.height, ctx.width)
        return self.nava_config.video_latent_frames(ctx.frames), latent_h, latent_w

    def _apply_first_frame_condition(
        self,
        video_latents: torch.Tensor,
        image_embeds: torch.Tensor | None,
        video_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        if image_embeds is None:
            return video_latents
        _, latent_h, latent_w = video_grid
        first_frame_tokens = latent_h * latent_w
        if image_embeds.shape[1] != first_frame_tokens:
            raise ValueError(
                "NAVA image latent token count does not match video first-frame grid: "
                f"expected {first_frame_tokens}, got {image_embeds.shape[1]}."
            )
        video_latents = video_latents.clone()
        video_latents[:, :first_frame_tokens] = image_embeds.to(device=video_latents.device, dtype=video_latents.dtype)
        return video_latents

    def _combine_guidance(
        self,
        ctx: NAVARequestContext,
        positive: dict[str, torch.Tensor],
        negative: dict[str, torch.Tensor] | None,
        *,
        align: dict[str, torch.Tensor] | None = None,
        timbre: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if negative is None:
            negative = positive
        # CFG branch combine mirrors NAVA's AV guidance: regular negative
        # prompt CFG, optional modality-mask alignment, and optional timbre CFG.
        if align is None:
            video = negative["video"] + ctx.video_guidance_scale * (positive["video"] - negative["video"])
            audio = negative["audio"] + ctx.audio_guidance_scale * (positive["audio"] - negative["audio"])
        else:
            video = (
                positive["video"]
                + ctx.video_guidance_scale * (positive["video"] - negative["video"])
                + ctx.video_align_guidance_scale * (positive["video"] - align["video"])
            )
            audio = (
                positive["audio"]
                + ctx.audio_guidance_scale * (positive["audio"] - negative["audio"])
                + ctx.audio_align_guidance_scale * (positive["audio"] - align["audio"])
            )
        if timbre is not None:
            audio = audio + ctx.timbre_align_guidance_scale * (positive["audio"] - timbre["audio"])
        return {
            "video": video,
            "audio": audio,
        }

    def _decode_video(self, video_latents: torch.Tensor, ctx: NAVARequestContext) -> torch.Tensor:
        if not hasattr(self.video_vae, "decode"):
            raise TypeError("NAVA video_vae must expose decode(video_latents, height, width, frames).")
        # Video decode maps denoised video tokens back to [B, C, T, H, W].
        return self.video_vae.decode(video_latents, height=ctx.height, width=ctx.width, frames=ctx.frames)

    def _decode_audio(self, audio_latents: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.audio_vae, "decode"):
            raise TypeError("NAVA audio_vae must expose decode(audio_latents).")
        # Audio decode maps denoised audio tokens to waveform samples.
        return self.audio_vae.decode(audio_latents)

    def _make_generator(self, seed: int) -> torch.Generator:
        random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("NAVAPipeline requires PyYAML to read configs/nava.yaml.") from exc
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"NAVA config must be a mapping: {path}")
    return data


class _MissingNAVAComponent(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def encode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise NotImplementedError(f"NAVA component {self.name!r} is not initialized.")


class _NAVATextEncoder(nn.Module):
    def __init__(self, model_root: str, config: NAVAConfig, device: torch.device) -> None:
        super().__init__()
        from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

        self.config = config
        self.device = device
        wan_root = os.path.join(model_root, config.wan_dir)
        tokenizer_path = os.path.join(wan_root, "google", "umt5-xxl")
        checkpoint_path = os.path.join(wan_root, "models_t5_umt5-xxl-enc-bf16.pth")
        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(f"NAVA text tokenizer directory not found: {tokenizer_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"NAVA text encoder checkpoint not found: {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        umt5_config = UMT5Config(
            vocab_size=256384,
            d_model=config.text_embed_dim,
            d_kv=64,
            d_ff=10240,
            num_heads=64,
            num_layers=24,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dense_act_fn="gelu_new",
            is_gated_act=True,
            is_encoder_decoder=False,
        )
        self.model = UMT5EncoderModel(umt5_config)
        state_dict = _convert_wan_t5_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.load_state_dict(state_dict, assign=True)
        self.model = self.model.to(device=device, dtype=config.target_dtype).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        return_speaker_positions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[int]]]:
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.text_len,
            return_tensors="pt",
        )
        speaker_positions = self._speaker_positions(inputs["input_ids"]) if return_speaker_positions else None
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        embeds = outputs.last_hidden_state.to(dtype=dtype)
        if return_speaker_positions:
            return embeds, speaker_positions
        return embeds

    def _speaker_positions(self, input_ids: torch.Tensor) -> list[list[int]]:
        token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_2>")
        if token_id is None or token_id < 0:
            raise ValueError("NAVA tokenizer does not expose <extra_id_2> for speaker binding.")
        return [[int(pos) for pos in (row == token_id).nonzero(as_tuple=True)[0].tolist()] for row in input_ids]


def _convert_wan_t5_state_dict(wan_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    hf_sd: dict[str, torch.Tensor] = {
        "shared.weight": wan_sd["token_embedding.weight"],
        "encoder.embed_tokens.weight": wan_sd["token_embedding.weight"],
        "encoder.final_layer_norm.weight": wan_sd["norm.weight"],
    }
    for i in range(24):
        src = f"blocks.{i}"
        dst = f"encoder.block.{i}"
        for proj in ("q", "k", "v", "o"):
            hf_sd[f"{dst}.layer.0.SelfAttention.{proj}.weight"] = wan_sd[f"{src}.attn.{proj}.weight"]
        hf_sd[f"{dst}.layer.0.SelfAttention.relative_attention_bias.weight"] = wan_sd[
            f"{src}.pos_embedding.embedding.weight"
        ]
        hf_sd[f"{dst}.layer.0.layer_norm.weight"] = wan_sd[f"{src}.norm1.weight"]
        hf_sd[f"{dst}.layer.1.DenseReluDense.wi_0.weight"] = wan_sd[f"{src}.ffn.gate.0.weight"]
        hf_sd[f"{dst}.layer.1.DenseReluDense.wi_1.weight"] = wan_sd[f"{src}.ffn.fc1.weight"]
        hf_sd[f"{dst}.layer.1.DenseReluDense.wo.weight"] = wan_sd[f"{src}.ffn.fc2.weight"]
        hf_sd[f"{dst}.layer.1.layer_norm.weight"] = wan_sd[f"{src}.norm2.weight"]
    return hf_sd


def _resolve_output_fps(sampling_params, default: int) -> int | float:
    value = getattr(sampling_params, "resolved_frame_rate", None)
    if value is None:
        value = getattr(sampling_params, "fps", None)
    fps = float(value if value is not None else default)
    return int(fps) if fps.is_integer() else fps


def _normalize_video_output(video: Any, output_type: str) -> Any:
    if not isinstance(video, torch.Tensor):
        return video
    video = video.detach().cpu()
    if output_type == "pt":
        return video
    if video.ndim == 5 and video.shape[1] in (1, 3):
        return [sample.permute(1, 0, 2, 3).contiguous().numpy() for sample in video]
    return video.numpy()
