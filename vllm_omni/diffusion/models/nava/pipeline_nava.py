# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import json
import os
import random
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.nava.config import (
    DEFAULT_NAVA_MODEL_INDEX,
    NAVA_CONFIG_ALIAS_MAP,
    NAVAConfig,
    NAVARequestContext,
    NAVASpeakerCondition,
    inject_speaker_sentinel,
    parse_speech_spans,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def get_nava_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(
        output: dict[str, Any] | tuple[Any, Any] | Any,
        output_type: str = "np",
        sampling_params=None,
    ):
        if output_type == "latent":
            return output

        if isinstance(output, dict):
            video = output.get("video")
            audio = output.get("audio")
            audio_sample_rate = output.get("audio_sample_rate", NAVAPipeline.audio_sample_rate)
            fps = output.get("fps", _resolve_output_fps(sampling_params, NAVAPipeline.fps))
        elif isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            audio_sample_rate = NAVAPipeline.audio_sample_rate
            fps = _resolve_output_fps(sampling_params, NAVAPipeline.fps)
        else:
            video = output
            audio = None
            audio_sample_rate = NAVAPipeline.audio_sample_rate
            fps = _resolve_output_fps(sampling_params, NAVAPipeline.fps)

        result = {
            "video": _normalize_video_output(video, output_type),
            "audio": audio,
            "audio_sample_rate": int(audio_sample_rate),
            "fps": fps,
        }
        return result

    return post_process_func


def _resolve_output_fps(sampling_params, default: int) -> int | float:
    value = None
    extra = getattr(sampling_params, "extra_args", None)
    if isinstance(extra, dict):
        value = extra.get("fps")
    value = value or getattr(sampling_params, "resolved_frame_rate", None)
    value = value or getattr(sampling_params, "frame_rate", None)
    value = value or getattr(sampling_params, "fps", None)
    try:
        fps = float(value if value is not None else default)
    except (TypeError, ValueError):
        fps = float(default)
    if fps <= 0:
        fps = float(default)
    return int(fps) if fps.is_integer() else fps


def _normalize_video_output(video: Any, output_type: str) -> Any:
    if not isinstance(video, torch.Tensor):
        return video

    video = video.detach().cpu()
    if video.ndim == 5:
        if video.shape[2] in (1, 3):
            items = [sample for sample in video]
        elif video.shape[1] in (1, 3):
            items = [sample.permute(1, 0, 2, 3).contiguous() for sample in video]
        else:
            items = [sample for sample in video]
        return items if output_type == "pt" else [sample.numpy() for sample in items]

    return video if output_type == "pt" else video.numpy()


class NAVAPipeline(
    nn.Module,
    SupportImageInput,
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    """Bridge pipeline for upstream NAVA audio-video inference.

    This first integration keeps the upstream runtime as the execution backend.
    The vLLM-Omni layer owns request parsing, model discovery, clear setup
    errors, postprocess metadata, and a narrow batch contract.
    """

    support_image_input: ClassVar[bool] = True
    support_audio_input: ClassVar[bool] = True
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 16000
    fps: ClassVar[int] = 24
    dummy_run_num_frames: ClassVar[int] = 0
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "align_3d_cfg",
            "audio_align_guidance_scale",
            "audio_guidance_scale",
            "frames",
            "fps",
            "height",
            "image_path",
            "negative_prompt_mode",
            "num_frames",
            "num_inference_steps",
            "num_steps",
            "offload_backbone",
            "save_vid_latent",
            "seed",
            "spk_wavs",
            "timbre_align_guidance_scale",
            "timbre_cfg",
            "tiled_vae",
            "vae_tile_size",
            "vae_tile_stride",
            "video_align_guidance_scale",
            "video_guidance_scale",
            "width",
        }
    )

    _dit_modules: ClassVar[list[str]] = ["pipe.model"]
    _encoder_modules: ClassVar[list[str]] = ["pipe.text_model"]
    _vae_modules: ClassVar[list[str]] = ["pipe.video_vae", "pipe.audio_vae"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self._validate_runtime_features()
        self.device = get_local_device()
        self.nava_config = self._load_nava_config(od_config)
        self.audio_sample_rate = self.nava_config.audio_sample_rate
        self.fps = self.nava_config.fps
        self.pipe = self._create_upstream_pipeline()
        self.model = self.pipe.model
        self.transformer = self.pipe.model
        self.text_model = self.pipe.text_model
        self.text_encoder = getattr(self.pipe.text_model, "model", self.pipe.text_model)
        self.video_vae = self.pipe.video_vae
        self.audio_vae = self.pipe.audio_vae
        self.vae = self.video_vae
        self._load_upstream_checkpoint()
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set()

    def _validate_runtime_features(self) -> None:
        if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload:
            raise ValueError(
                "NAVAPipeline bridge does not support vLLM-Omni CPU/layerwise offload yet. "
                "Use NAVA extra parameter `offload_backbone` for the upstream runtime, or disable "
                "`--enable-cpu-offload` and `--enable-layerwise-offload`."
            )

        parallel_config = self.od_config.parallel_config
        unsupported_parallel = {
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "sequence_parallel_size": parallel_config.sequence_parallel_size,
            "cfg_parallel_size": parallel_config.cfg_parallel_size,
            "vae_patch_parallel_size": parallel_config.vae_patch_parallel_size,
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
        }
        enabled = {name: value for name, value in unsupported_parallel.items() if int(value or 1) != 1}
        if enabled or parallel_config.use_hsdp or parallel_config.enable_expert_parallel:
            raise ValueError(
                "NAVAPipeline bridge does not support native vLLM-Omni parallelism yet. "
                f"Unsupported settings: {enabled}, use_hsdp={parallel_config.use_hsdp}, "
                f"enable_expert_parallel={parallel_config.enable_expert_parallel}. "
                "Run this bridge with single-process diffusion settings until the NAVA backbone is ported."
            )

    def forward(
        self,
        request: OmniDiffusionRequest,
        **kwargs,
    ) -> DiffusionOutput:
        ctx = self._parse_request(request)
        self._set_seed(ctx.seed)

        # Text embedding is performed inside upstream AudioVideoPipeline.sample().
        batch = self._move_batch_to_device(self._build_sample_batch(ctx))

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.device.type == "cuda" and self.nava_config.target_dtype != torch.float32,
            dtype=self.nava_config.target_dtype,
        ):
            # Generation delegates to upstream NAVA's denoise loop so the first
            # integration preserves text/image/speaker conditioning semantics.
            video, audio_items = self.pipe.sample(
                batch,
                num_steps=ctx.num_steps,
                audio_guidance_scale=ctx.audio_guidance_scale,
                video_guidance_scale=ctx.video_guidance_scale,
                negative_prompt_mode=ctx.negative_prompt_mode,
                align_3d_cfg=ctx.align_3d_cfg,
                audio_align_guidance_scale=ctx.audio_align_guidance_scale,
                video_align_guidance_scale=ctx.video_align_guidance_scale,
                is_i2v=ctx.is_i2v,
                save_vid_latent=ctx.save_vid_latent,
                timbre_cfg=ctx.timbre_cfg,
                timbre_align_guidance_scale=ctx.timbre_align_guidance_scale,
                offload_backbone=ctx.offload_backbone,
                tiled_vae=ctx.tiled_vae,
                vae_tile_size=ctx.vae_tile_size,
                vae_tile_stride=ctx.vae_tile_stride,
            )

        # Audio decode returns one dict per sample; normalize single-request
        # output to the waveform payload expected by vLLM-Omni postprocess.
        audio, audio_sample_rate = self._normalize_audio_output(audio_items)
        return DiffusionOutput(
            output={
                "video": video,
                "audio": audio,
                "audio_sample_rate": audio_sample_rate,
                "fps": ctx.fps,
            }
        )

    def _load_nava_config(self, od_config: OmniDiffusionConfig) -> NAVAConfig:
        if not od_config.model or not os.path.isdir(str(od_config.model)):
            raise ValueError(
                "NAVAPipeline requires a local NAVA weight directory. "
                "Run `python examples/offline_inference/nava/download_nava.py --local-dir <dir>` "
                "or pass a prepared local directory with NAVA.safetensors, nava.yaml or configs/nava.yaml, "
                "Wan2.2-TI2V-5B/, and params/."
            )

        config_data = dict(DEFAULT_NAVA_MODEL_INDEX)
        explicit_model_config = _normalize_explicit_model_config(od_config.model_config or {})

        model_root = od_config.model
        index_path = os.path.join(model_root, "model_index.json")
        index_data = {}
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                index_data = json.load(f)
            config_data.update(index_data)

        config_selector = dict(config_data)
        config_selector.update(explicit_model_config)
        config_name = str(config_selector.get("config") or DEFAULT_NAVA_MODEL_INDEX["config"])
        config_path = os.path.join(model_root, config_name)
        if not os.path.exists(config_path) and config_name == "configs/nava.yaml":
            legacy_config_path = os.path.join(model_root, "nava.yaml")
            if os.path.exists(legacy_config_path):
                config_path = legacy_config_path
                config_data["config"] = "nava.yaml"
                index_data = dict(index_data)
                index_data["config"] = "nava.yaml"
        if os.path.exists(config_path):
            config_data.update(_load_yaml(config_path))
            config_data.update(index_data)
        config_data.update(explicit_model_config)

        self._raw_nava_config = dict(config_data)
        return NAVAConfig.from_dict(config_data)

    def _create_upstream_pipeline(self):
        try:
            module = importlib.import_module("nava_src.pipeline_nava")
        except ImportError as exc:
            raise ImportError(
                "NAVAPipeline requires the upstream NAVA package. Install it with "
                "`pip install -e /path/to/NAVA` or run "
                "`python examples/offline_inference/nava/download_nava.py --local-dir <dir> --install-upstream`."
            ) from exc

        pipeline_cls = getattr(module, "AudioVideoPipeline")
        cfg = self._upstream_cfg_dict()
        if "audio" in str(cfg.get("modality", "")) and "video" in str(cfg.get("modality", "")):
            cfg["init_from_meta"] = True

        disable_compile = self.od_config.enforce_eager or _as_bool(
            self._extra_arg("disable_text_encoder_compile", False)
        )
        with _temporarily_disable_torch_compile(disable_compile):
            return pipeline_cls.create(
                model_id=cfg.get("model_id", ""),
                use_bf16=_as_bool(cfg.get("use_bf16", self.nava_config.use_bf16)),
                audio_latent_ch=int(cfg.get("audio_latent_ch", self.nava_config.audio_latent_ch)),
                video_latent_ch=int(cfg.get("video_latent_ch", self.nava_config.video_latent_ch)),
                lambda_ddpm=float(cfg.get("lambda_ddpm", self.nava_config.lambda_ddpm)),
                cfg=cfg,
                device=self.device,
            )

    def _upstream_cfg_dict(self) -> dict[str, Any]:
        cfg = dict(getattr(self, "_raw_nava_config", {}))
        cfg.setdefault("model_type", self.nava_config.model_type)
        cfg.setdefault("modality", self.nava_config.modality)
        cfg.setdefault("use_bf16", self.nava_config.use_bf16)
        cfg.setdefault("audio_latent_ch", self.nava_config.audio_latent_ch)
        cfg.setdefault("video_latent_ch", self.nava_config.video_latent_ch)
        cfg.setdefault("lambda_ddpm", self.nava_config.lambda_ddpm)
        cfg.setdefault("patch_size", 2)
        cfg.setdefault("image_size", 960)
        cfg.setdefault("log_height", self.nava_config.log_height)
        cfg.setdefault("log_width", self.nava_config.log_width)
        cfg.setdefault("scheduler_personalized", True)
        cfg.setdefault("scheduler_unipc", True)
        cfg.setdefault("align_3d_cfg", self.nava_config.align_3d_cfg)
        cfg.setdefault("timbre_cfg", self.nava_config.timbre_cfg)

        data_cfg = dict(cfg.get("data") or {})
        data_cfg.setdefault("video_fps", self.nava_config.fps)
        data_cfg.setdefault("audio_tokens_per_sec", self.nava_config.audio_tokens_per_sec)
        data_cfg.setdefault("use_speech_special_token", False)
        cfg["data"] = data_cfg

        model_cfg = dict(cfg.get("model") or {})
        model_cfg["ckpt_dir"] = self.od_config.model
        model_cfg["audio_vae_ckpt_dir"] = self._join_model_path(self.nava_config.audio_vae_ckpt_dir)
        model_cfg.setdefault("shift", 5)
        model_cfg.setdefault("shift_audio", 5)
        model_cfg.setdefault("num_train_timesteps", 1000)
        cfg["model"] = model_cfg

        cfg.update(self.od_config.additional_config or {})
        return cfg

    def _load_upstream_checkpoint(self) -> None:
        ckpt_path = self._resolve_checkpoint_path()
        if ckpt_path is None:
            raise FileNotFoundError(
                "NAVA checkpoint not found. Expected one of "
                f"{self.nava_config.ckpt_name!r}, {self.nava_config.fp8_ckpt_name!r}, or "
                "`od_config.model_config['nava_ckpt']` under the model directory."
            )

        state_dict = self._load_checkpoint_state_dict(ckpt_path)
        use_fp8 = self._should_use_fp8(state_dict)
        if use_fp8:
            try:
                from NAVA_FP8 import patch_model_to_fp8
            except ImportError as exc:
                raise ImportError(
                    "NAVA FP8 checkpoint requires upstream NAVA_FP8. Install the upstream NAVA package "
                    "or use the bf16 NAVA.safetensors checkpoint."
                ) from exc
            patch_model_to_fp8(self.pipe.model)

        missing, unexpected = self.pipe.model.load_state_dict(state_dict, strict=False)
        loaded_count = len(state_dict) - len(unexpected)
        if loaded_count <= 0:
            raise RuntimeError(
                "NAVA checkpoint did not match the upstream model state dict. "
                f"Loaded 0 parameter(s) from {ckpt_path}; first checkpoint keys: {list(state_dict)[:8]}."
            )
        if missing:
            logger.warning("NAVA checkpoint missing %d keys, first keys: %s", len(missing), missing[:8])
        if unexpected:
            logger.warning("NAVA checkpoint had %d unexpected keys, first keys: %s", len(unexpected), unexpected[:8])

        self.pipe.to(self.device)
        self.pipe.model.eval()
        backbone = getattr(self.pipe.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "set_rope_params"):
            backbone.set_rope_params()

    def _resolve_checkpoint_path(self) -> str | None:
        explicit = (self.od_config.model_config or {}).get("nava_ckpt")
        candidates = []
        if explicit:
            candidates.append(str(explicit))

        weight_dtype = self._extra_arg("nava_weight_dtype", "auto")
        if weight_dtype == "fp8_e4m3fn":
            candidates.append(self.nava_config.fp8_ckpt_name)
        candidates.append(self.nava_config.ckpt_name)
        if weight_dtype != "bf16":
            candidates.append(self.nava_config.fp8_ckpt_name)

        for candidate in candidates:
            path = candidate if os.path.isabs(candidate) else self._join_model_path(candidate)
            if os.path.exists(path):
                return path
            fallback = os.path.splitext(path)[0] + ".ckpt"
            if os.path.exists(fallback):
                return fallback
        return None

    def _load_checkpoint_state_dict(self, ckpt_path: str) -> dict[str, torch.Tensor]:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(ckpt_path, device="cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        if isinstance(ckpt, dict):
            return ckpt
        raise TypeError(f"Unsupported NAVA checkpoint payload: {type(ckpt)!r}")

    def _should_use_fp8(self, state_dict: dict[str, torch.Tensor]) -> bool:
        requested = self._extra_arg("nava_weight_dtype", "auto")
        if requested == "fp8_e4m3fn":
            return True
        if requested == "bf16":
            return False
        return any(
            isinstance(value, torch.Tensor) and value.dtype == torch.float8_e4m3fn
            for value in state_dict.values()
        )

    def _parse_request(self, request: OmniDiffusionRequest) -> NAVARequestContext:
        if len(request.prompts) != 1:
            raise ValueError("NAVAPipeline currently supports one prompt per request. Use business-layer batching.")

        prompt_data = request.prompts[0]
        prompt = prompt_data if isinstance(prompt_data, str) else str(prompt_data.get("prompt", ""))
        if not prompt:
            raise ValueError("NAVAPipeline requires a non-empty prompt.")

        multi_modal_data = {} if isinstance(prompt_data, str) else (prompt_data.get("multi_modal_data") or {})
        extra = request.sampling_params.extra_args or {}
        sp = request.sampling_params

        _reject_custom_negative_prompt(prompt_data, extra)
        speaker_condition = self._parse_speaker_condition(prompt, multi_modal_data, extra)

        image = multi_modal_data.get("image")
        if image is None:
            image = extra.get("image_path")

        return NAVARequestContext(
            prompt=prompt,
            image=image,
            speaker_condition=speaker_condition,
            height=int(sp.height or extra.get("height") or self.nava_config.log_height),
            width=int(sp.width or extra.get("width") or self.nava_config.log_width),
            frames=_resolve_num_frames(sp.num_frames, extra, self.nava_config.frames),
            fps=int(sp.fps or extra.get("fps") or self.nava_config.fps),
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
            align_3d_cfg=_as_bool(extra.get("align_3d_cfg", self.nava_config.align_3d_cfg)),
            timbre_cfg=_resolve_timbre_cfg(extra, self.nava_config.timbre_cfg, speaker_condition),
            timbre_align_guidance_scale=float(
                extra.get("timbre_align_guidance_scale", self.nava_config.timbre_align_guidance_scale)
            ),
            negative_prompt_mode=_as_bool(extra.get("negative_prompt_mode", self.nava_config.negative_prompt_mode)),
            offload_backbone=_as_bool(extra.get("offload_backbone", False)),
            tiled_vae=_as_bool(extra.get("tiled_vae", False)),
            vae_tile_size=_as_pair(extra.get("vae_tile_size", (22, 40))),
            vae_tile_stride=_as_pair(extra.get("vae_tile_stride", (14, 26))),
            save_vid_latent=_as_bool(extra.get("save_vid_latent", False)),
        )

    def _parse_speaker_condition(
        self,
        prompt: str,
        multi_modal_data: dict[str, Any],
        extra: dict[str, Any],
    ) -> NAVASpeakerCondition | None:
        wavs = multi_modal_data.get("spk_wavs")
        if wavs is None:
            wavs = multi_modal_data.get("audio")
        if wavs is None:
            wavs = extra.get("spk_wavs")
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

    def _build_sample_batch(self, ctx: NAVARequestContext) -> dict[str, Any]:
        # Text conditioning: upstream T5 expects <extra_id_2> after each <S>.
        caption = inject_speaker_sentinel(ctx.prompt)
        video_h, video_w = self.nava_config.video_latent_hw(ctx.height, ctx.width)
        video_latents = torch.randn((ctx.frames, video_h, video_w, self.nava_config.video_latent_ch))

        # Audio latent initialization only defines the generation length; the
        # denoise loop replaces it with sampled noise before decode.
        audio_len = self.nava_config.audio_latent_length(ctx.frames, ctx.fps)
        audio_latents = torch.randn((audio_len, self.nava_config.audio_latent_ch))

        first_frames = self._encode_image(ctx)
        if isinstance(first_frames, torch.Tensor):
            video_latents = torch.randn(
                (ctx.frames, first_frames.shape[1], first_frames.shape[2], self.nava_config.video_latent_ch)
            )

        spk_embs = self._encode_speakers(ctx)
        return self._collate_single(
            {
                "idx": 0,
                "video_latents": video_latents,
                "first_frames": first_frames,
                "audio_latents": audio_latents,
                "captions": caption,
                "spk_embs": spk_embs,
                "is_i2v": first_frames is not None,
            }
        )

    def _encode_image(self, ctx: NAVARequestContext):
        if ctx.image is None:
            return None
        image = _prepare_image_input(ctx.image, ctx.height, ctx.width)
        if not hasattr(self.video_vae, "encode"):
            raise RuntimeError("NAVA upstream video VAE does not expose encode(); image conditioning is unavailable.")
        # Image embedding: encode the first frame into Wan video-VAE latents.
        encoded = self.video_vae.encode(
            image,
            rank=-1,
            frame_length=ctx.frames,
            fps=ctx.fps,
            target_height=ctx.height,
            target_width=ctx.width,
        )
        return encoded.latent_dist.sample()

    def _encode_speakers(self, ctx: NAVARequestContext):
        if ctx.speaker_condition is None:
            return None
        if not hasattr(self.audio_vae, "encode"):
            raise RuntimeError("NAVA upstream audio VAE does not expose encode(); speaker conditioning is unavailable.")
        if hasattr(self.audio_vae, "spk_model") and self.audio_vae.spk_model is None:
            raise RuntimeError(
                "NAVA reference timbre control requires ReDimNet speaker embedding. "
                "Run `python examples/offline_inference/nava/download_nava.py --prepare-redimnet` "
                "and set TORCH_HOME consistently before inference."
            )

        speaker_embeddings = []
        for wav in ctx.speaker_condition.wavs:
            # Speaker embedding: ReDimNet embedding is extracted by upstream
            # LocalAudioVAEAdapter and aligned with <S>...<E> span order.
            result = self.audio_vae.encode({"data_path": wav, "use_spk_emb": True}).latent_dist.sample()
            speaker_embeddings.append(result["spk_embs"])
        return speaker_embeddings

    def _collate_single(self, sample: dict[str, Any]) -> dict[str, Any]:
        video_latents = sample["video_latents"]
        t_h_w_list = torch.tensor([(video_latents.shape[0], video_latents.shape[1], video_latents.shape[2])])
        return {
            "idx": [sample["idx"]],
            "video_latents": video_latents.view(1, -1, video_latents.shape[-1]),
            "first_frames": [sample["first_frames"]],
            "audio_latents": [sample["audio_latents"]],
            "captions": [sample["captions"]],
            "spk_embs": None if sample["spk_embs"] is None else [sample["spk_embs"]],
            "is_i2v": [sample["is_i2v"]],
            "t_h_w_list": t_h_w_list,
        }

    def _normalize_audio_output(self, audio_items: Any) -> tuple[Any, int]:
        if isinstance(audio_items, list) and audio_items:
            item = audio_items[0]
        else:
            item = audio_items
        if isinstance(item, dict):
            return item.get("waveform"), int(item.get("sample_rate", self.audio_sample_rate))
        return item, self.audio_sample_rate

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return _move_to_device(batch, self.device)

    def _join_model_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(str(self.od_config.model), path)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _extra_arg(self, key: str, default: Any = None) -> Any:
        custom_args = self.od_config.custom_pipeline_args or {}
        if key in custom_args:
            return custom_args[key]
        extras = self.od_config.extras or {}
        if isinstance(extras, dict):
            return extras.get(key, default)
        return default


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


def _normalize_explicit_model_config(raw: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    for old_key, new_key in NAVA_CONFIG_ALIAS_MAP.items():
        if old_key in data:
            data[new_key] = data[old_key]
    return data


def _reject_custom_negative_prompt(prompt_data: Any, extra: dict[str, Any]) -> None:
    prompt_dict = prompt_data if isinstance(prompt_data, dict) else {}
    if (
        "negative_prompt" in prompt_dict
        or "video_negative_prompt" in prompt_dict
        or "audio_negative_prompt" in prompt_dict
    ):
        raise ValueError(
            "NAVAPipeline bridge uses upstream NAVA default negative prompts; "
            "custom negative prompts are not supported yet."
        )
    if "negative_prompt" in extra or "video_negative_prompt" in extra or "audio_negative_prompt" in extra:
        raise ValueError(
            "NAVAPipeline bridge uses upstream NAVA default negative prompts; "
            "custom negative prompts are not supported yet."
        )


def _resolve_num_frames(value: Any, extra: dict[str, Any], default: int) -> int:
    # OmniDiffusionSamplingParams defaults num_frames to 1 for image models;
    # NAVA should keep its audio-video default unless the caller asks for video length.
    if value is not None and int(value) > 1:
        return int(value)
    extra_value = extra.get("num_frames")
    if extra_value is None:
        extra_value = extra.get("frames")
    if extra_value is not None:
        return int(extra_value)
    return int(default)


def _resolve_timbre_cfg(
    extra: dict[str, Any],
    default: bool,
    speaker_condition: NAVASpeakerCondition | None,
) -> bool:
    if "timbre_cfg" in extra:
        requested = _as_bool(extra["timbre_cfg"])
        if requested and speaker_condition is None:
            raise ValueError("NAVA timbre_cfg requires reference speaker WAVs aligned to <S>...<E> spans.")
        return requested
    return bool(default and speaker_condition is not None)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(f"Expected boolean-like value, got {value!r}.")
    return bool(value)


def _as_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected pair value, got {value!r}")


def _prepare_image_input(image: Any, height: int, width: int) -> str | torch.Tensor:
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError("NAVAPipeline image conditioning accepts exactly one first-frame image.")
        image = image[0]

    if isinstance(image, str):
        return image
    if isinstance(image, Image.Image):
        image = _resize_center_crop(image.convert("RGB"), height, width)
        array = np.array(image, copy=True)
        tensor = torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0)
        return tensor / 255.0 * 2.0 - 1.0
    if isinstance(image, torch.Tensor):
        tensor = image.detach()
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.ndim != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape {tuple(tensor.shape)}.")
        tensor = tensor.float()
        if tensor.numel() > 0 and tensor.min().item() >= 0.0:
            tensor = tensor / 255.0 if tensor.max().item() > 2.0 else tensor
            tensor = tensor * 2.0 - 1.0
        return tensor
    raise TypeError(f"Unsupported NAVA image input type: {type(image)!r}")


def _resize_center_crop(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resize_w = int(round(src_w * scale))
    resize_h = int(round(src_h * scale))
    image = image.resize((resize_w, resize_h), Image.Resampling.LANCZOS)
    left = (resize_w - target_w) // 2
    top = (resize_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


@contextmanager
def _temporarily_disable_torch_compile(enabled: bool):
    if not enabled or not hasattr(torch, "compile"):
        yield
        return

    original_compile = torch.compile

    def _identity_compile(module, *args, **kwargs):
        return module

    torch.compile = _identity_compile
    try:
        yield
    finally:
        torch.compile = original_compile
