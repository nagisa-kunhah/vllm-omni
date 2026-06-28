# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Local Transformer v1.5 vocoder stage."""

from __future__ import annotations

import bisect
import contextlib
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoProcessor
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.moss_tts_local.codec_path import (
    moss_tts_local_processor_kwargs,
)
from vllm_omni.model_executor.models.moss_tts_local.vocoder_cuda_graph import (
    MossTTSLocalVocoderCUDAGraphWrapper,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _vocoder_debug_enabled() -> bool:
    return os.environ.get("MOSS_TTS_LOCAL_VOCODER_DEBUG", "0").lower() in ("1", "true", "yes", "on")


def _shape_numel(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, numel={value.numel()}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def _codec_ids_from_payload_or_input(
    input_ids: torch.Tensor | None,
    runtime_info: dict[str, Any] | None,
) -> torch.Tensor:
    if isinstance(runtime_info, dict):
        codes = runtime_info.get("codes")
        if isinstance(codes, dict):
            audio = codes.get("audio")
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                return audio.reshape(-1).to(dtype=torch.long)
            if isinstance(audio, (list, tuple)) and audio:
                return torch.as_tensor(audio, dtype=torch.long).reshape(-1)
        task_type = runtime_info.get("task_type")
        if task_type == "moss_tts_local" or (isinstance(task_type, list) and "moss_tts_local" in task_type):
            return torch.empty(0, dtype=torch.long)
        if "prompt_rows" in runtime_info:
            return torch.empty(0, dtype=torch.long)
    if input_ids is None:
        return torch.empty(0, dtype=torch.long)
    return input_ids.reshape(-1).to(dtype=torch.long)


def _meta_bool(value: Any) -> bool:
    if isinstance(value, list):
        value = value[0] if value else False
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item()) if value.numel() > 0 else False
    return bool(value)


def _meta_str(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value)


@dataclass
class _StreamingDecodeState:
    exit_stack: contextlib.ExitStack

    @classmethod
    def create(cls, codec: Any) -> _StreamingDecodeState:
        stack = contextlib.ExitStack()
        stack.enter_context(codec.streaming(batch_size=1))
        return cls(exit_stack=stack)

    def close(self) -> None:
        self.exit_stack.close()


@dataclass
class _FullDecodeGroupResult:
    audio: torch.Tensor
    audio_lengths: torch.Tensor | None
    code_lengths: list[int]
    audio_shape: tuple[int, ...]
    pack_ms: float
    decode_ms: float


class _TensorRTCodecDecodeWrapper(nn.Module):
    """TensorRT-friendly wrapper around MOSS full-frame codec decode.

    ``codec.decode`` builds a Python list by calling ``codes_lengths[i].item()``
    and slicing per request, which blocks torch.export/TensorRT.  The
    non-streaming ``batch_decode`` path ultimately calls ``_decode_frame`` with
    already-padded codes and a length tensor, so use that lower-level tensor
    entrypoint directly.
    """

    def __init__(self, codec: Any, *, num_quantizers: int) -> None:
        super().__init__()
        self.codec = codec
        self.num_quantizers = int(num_quantizers)

    def forward(self, audio_codes: torch.Tensor, code_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.codec, "_decode_frame"):
            raise RuntimeError("MOSS-TTS Local TensorRT decode requires codec._decode_frame")
        decoded = self.codec._decode_frame(
            audio_codes[: self.num_quantizers],
            code_lengths.to(dtype=torch.long),
        )
        audio = getattr(decoded, "audio", None)
        audio_lengths = getattr(decoded, "audio_lengths", None)
        if audio is None:
            if isinstance(decoded, tuple) and decoded:
                audio = decoded[0]
                audio_lengths = decoded[1] if len(decoded) > 1 else None
            else:
                audio = decoded
        if not isinstance(audio, torch.Tensor):
            raise TypeError(f"MOSS-TTS Local codec returned {type(audio).__name__}, expected Tensor")
        if not isinstance(audio_lengths, torch.Tensor):
            audio_lengths = torch.full(
                (int(audio.shape[0]),),
                int(audio.shape[-1]),
                dtype=torch.long,
                device=audio.device,
            )
        return audio, audio_lengths


class _TensorRTCodecDecoderWrapper(nn.Module):
    """TensorRT wrapper for the float decoder half of MOSS codec decode.

    TensorRT cannot reliably convert the LFQ codebook path when the graph starts
    from integer code ids: the converter may propagate the int tensor into later
    linear nodes.  Keep ``quantizer.decode_codes`` in PyTorch and compile the
    heavier decoder transformer from float hidden states.
    """

    def __init__(self, codec: Any) -> None:
        super().__init__()
        self.decoder = codec.decoder
        object.__setattr__(self, "_codec_ref", codec)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        code_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codec = getattr(self, "_codec_ref")
        try:
            target_dtype = next(self.decoder.parameters()).dtype
        except StopIteration:
            target_dtype = decoder_hidden_states.dtype
        audio = decoder_hidden_states.to(dtype=target_dtype)
        audio_lengths = code_lengths.to(dtype=torch.long)
        for decoder_module in self.decoder:
            audio, audio_lengths = decoder_module(audio, audio_lengths)
        audio, audio_lengths = codec._restore_channels_from_codec(audio, audio_lengths)
        return audio, audio_lengths


class MossTTSLocalVocoder(nn.Module):
    """Decode raw ``[T, 12]`` Local v1.5 audio codes with MOSS-Audio-Tokenizer-v2."""

    input_modalities = "audio"
    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = False
    enable_update_additional_information: bool = True
    requires_raw_input_tokens: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.model_path = vllm_config.model_config.model
        self.config = vllm_config.model_config.hf_config
        self._n_vq = int(getattr(self.config, "n_vq", 12))
        self._sample_rate = int(getattr(self.config, "sampling_rate", 48000))
        self._processor: Any | None = None
        self._sr_tensor = torch.tensor(self._sample_rate, dtype=torch.int32)
        self._streaming_states: dict[str, _StreamingDecodeState] = {}
        self._decode_audio_codes_buffer: torch.Tensor | None = None
        self._decode_padding_mask_buffer: torch.Tensor | None = None
        self._cuda_graph_wrapper: MossTTSLocalVocoderCUDAGraphWrapper | None = None
        self._cuda_graph_codec: Any | None = None
        self._cuda_graph_disabled = False
        self._cuda_graph_warned = False
        self._compile_codec: Any | None = None
        self._compile_applied = False
        self._trt_codec: dict[tuple[int, int], Any] | Any | None = None
        self._trt_codec_source: Any | None = None
        self._trt_disabled = False
        self._trt_warned = False
        self._multi_stream_warned = False
        self._profile_enabled = os.environ.get("MOSS_TTS_LOCAL_VOCODER_PROFILE") == "1"
        self._profile_sync = os.environ.get("MOSS_TTS_LOCAL_VOCODER_PROFILE_SYNC") == "1"
        self._profile_log_every = max(int(os.environ.get("MOSS_TTS_LOCAL_VOCODER_PROFILE_LOG_EVERY", "20") or 20), 1)
        self._profile_stats: dict[str, float | int | list[int]] = {
            "groups": 0,
            "requests": 0,
            "frames": 0,
            "pack_ms": 0.0,
            "decode_ms": 0.0,
            "d2h_ms": 0.0,
            "batches": [],
        }

    def _profile_mark(self, device: torch.device | None = None) -> float:
        if self._profile_sync and device is not None and device.type == "cuda":
            torch.accelerator.synchronize(device)
        return time.perf_counter()

    def _profile_elapsed_ms(self, start: float, device: torch.device | None = None) -> float:
        if self._profile_sync and device is not None and device.type == "cuda":
            torch.accelerator.synchronize(device)
        return (time.perf_counter() - start) * 1000

    def _record_profile_group(
        self,
        *,
        batch_size: int,
        frame_count: int,
        pack_ms: float,
        decode_ms: float,
        d2h_ms: float,
    ) -> None:
        if not bool(getattr(self, "_profile_enabled", False)):
            return
        stats = getattr(self, "_profile_stats", None)
        if not isinstance(stats, dict):
            return
        stats["groups"] = int(stats["groups"]) + 1
        stats["requests"] = int(stats["requests"]) + int(batch_size)
        stats["frames"] = int(stats["frames"]) + int(frame_count)
        stats["pack_ms"] = float(stats["pack_ms"]) + float(pack_ms)
        stats["decode_ms"] = float(stats["decode_ms"]) + float(decode_ms)
        stats["d2h_ms"] = float(stats["d2h_ms"]) + float(d2h_ms)
        batches = stats["batches"]
        if isinstance(batches, list):
            batches.append(int(batch_size))
            del batches[:-100]
        groups = int(stats["groups"])
        if groups % self._profile_log_every != 0:
            return
        avg_batch = sum(batches) / max(1, len(batches)) if isinstance(batches, list) else 0.0
        logger.info(
            "[moss-vocoder-profile] groups=%d requests=%d avg_batch=%.2f frames=%d "
            "pack=%.2fms decode=%.2fms d2h=%.2fms sync=%s",
            groups,
            int(stats["requests"]),
            avg_batch,
            int(stats["frames"]),
            float(stats["pack_ms"]) / max(1, groups),
            float(stats["decode_ms"]) / max(1, groups),
            float(stats["d2h_ms"]) / max(1, groups),
            self._profile_sync,
        )

    def _ensure_processor(self) -> Any:
        if self._processor is None:
            logger.info("Loading MOSS-TTS Local processor from %s", self.model_path)
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                **moss_tts_local_processor_kwargs(self.model_path),
            )
            audio_tokenizer = getattr(self._processor, "audio_tokenizer", None)
            if audio_tokenizer is not None:
                if hasattr(audio_tokenizer, "eval"):
                    audio_tokenizer.eval()
                if torch.cuda.is_available() and hasattr(audio_tokenizer, "to"):
                    audio_tokenizer.to("cuda")
        return self._processor

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _request_id_from_runtime_info(self, runtime_info: dict[str, Any] | None, index: int) -> str:
        if isinstance(runtime_info, dict):
            request_id = runtime_info.get("request_id")
            if request_id is not None:
                return str(request_id)
            meta = runtime_info.get("meta")
            if isinstance(meta, dict):
                req_id = _meta_str(meta.get("req_id"))
                if req_id is not None:
                    return req_id
        return str(index)

    def _runtime_info_uses_streaming(self, runtime_info: dict[str, Any] | None) -> bool:
        if not isinstance(runtime_info, dict):
            return False
        meta = runtime_info.get("meta")
        if not isinstance(meta, dict):
            return False
        return _meta_bool(meta.get("codec_streaming"))

    def _runtime_info_finished(self, runtime_info: dict[str, Any] | None) -> bool:
        if not isinstance(runtime_info, dict):
            return False
        meta = runtime_info.get("meta")
        if not isinstance(meta, dict):
            return False
        return _meta_bool(meta.get("finished")) or _meta_bool(meta.get("is_segment_finished"))

    def _streaming_codec_device(self, codec: Any) -> torch.device:
        try:
            return next(codec.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_audio_codec(self, processor: Any) -> Any:
        audio_tokenizer = getattr(processor, "audio_tokenizer", None)
        codec = getattr(audio_tokenizer, "model", None) if audio_tokenizer is not None else None
        if codec is None:
            codec = audio_tokenizer
        if codec is None or not hasattr(codec, "decode"):
            raise RuntimeError(
                "MOSS-TTS Local full-sequence decode requires processor.audio_tokenizer.decode "
                "or processor.audio_tokenizer.model.decode"
            )
        return codec

    @staticmethod
    def _env_enabled(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _parse_shape_triplet(name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        raw = os.environ.get(name)
        if not raw:
            return default
        values: list[int] = []
        for item in raw.replace("x", ",").replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(int(item))
            except ValueError:
                return default
        return tuple(values[:3]) if len(values) >= 3 and all(v > 0 for v in values[:3]) else default

    def _maybe_compile_codec_decode(self, codec: Any) -> Any:
        if not self._env_enabled("MOSS_TTS_LOCAL_VOCODER_COMPILE"):
            return codec
        if getattr(self, "_compile_codec", None) is codec and bool(getattr(self, "_compile_applied", False)):
            return codec
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            logger.warning("MOSS-TTS Local vocoder torch.compile requested but torch.compile is unavailable")
            return codec
        try:
            mode = os.environ.get("MOSS_TTS_LOCAL_VOCODER_COMPILE_MODE", "max-autotune-no-cudagraphs")
            compiled_decode = compile_fn(codec.decode, mode=mode)
            setattr(codec, "decode", compiled_decode)
            self.__dict__["_compile_codec"] = codec
            self.__dict__["_compile_applied"] = True
            logger.info("MOSS-TTS Local vocoder torch.compile enabled for codec.decode mode=%s", mode)
        except Exception:
            self.__dict__["_compile_applied"] = False
            logger.warning(
                "MOSS-TTS Local vocoder torch.compile failed; falling back to original decode",
                exc_info=True,
            )
        return codec

    def _warn_trt_fallback(self, message: str, *args: Any) -> None:
        if bool(getattr(self, "_trt_warned", False)):
            return
        self.__dict__["_trt_warned"] = True
        logger.warning(message, *args, exc_info=True)

    @staticmethod
    def _parse_int_list_env(name: str, default: list[int]) -> list[int]:
        raw = os.environ.get(name)
        if not raw:
            return list(default)
        values: set[int] = set()
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                value = int(item)
            except ValueError:
                continue
            if value > 0:
                values.add(value)
        return sorted(values) or list(default)

    @classmethod
    def _trt_bucket(cls, *, batch_size: int, frame_count: int) -> tuple[int, int] | None:
        batch_buckets = cls._parse_int_list_env("MOSS_TTS_LOCAL_VOCODER_TRT_BATCH_BUCKETS", [1, 2, 4, 8])
        frame_buckets = cls._parse_int_list_env("MOSS_TTS_LOCAL_VOCODER_TRT_FRAME_BUCKETS", [32, 48, 64, 80, 96, 128])
        batch_idx = bisect.bisect_left(batch_buckets, int(batch_size))
        frame_idx = bisect.bisect_left(frame_buckets, int(frame_count))
        if batch_idx >= len(batch_buckets) or frame_idx >= len(frame_buckets):
            return None
        return batch_buckets[batch_idx], frame_buckets[frame_idx]

    @staticmethod
    def _vocoder_trt_mode() -> str:
        return os.environ.get("MOSS_TTS_LOCAL_VOCODER_TRT_MODE", "decoder_hidden").strip().lower()

    def _get_or_init_trt_codec(
        self,
        *,
        codec: Any,
        device: torch.device,
        batch_size: int,
        frame_count: int,
    ) -> tuple[Any, tuple[int, int], str, torch.dtype] | None:
        if not self._env_enabled("MOSS_TTS_LOCAL_VOCODER_TRT") or bool(getattr(self, "_trt_disabled", False)):
            return None
        if device.type != "cuda":
            return None
        if not hasattr(codec, "_decode_frame"):
            self.__dict__["_trt_disabled"] = True
            logger.warning("MOSS-TTS Local vocoder TensorRT requires codec._decode_frame; falling back to PyTorch")
            return None
        bucket = self._trt_bucket(batch_size=batch_size, frame_count=frame_count)
        if bucket is None:
            return None
        if getattr(self, "_trt_codec_source", None) is not codec:
            self.__dict__["_trt_codec"] = {}
            self.__dict__["_trt_codec_source"] = codec
        trt_codecs = getattr(self, "_trt_codec", None)
        if not isinstance(trt_codecs, dict):
            trt_codecs = {}
            self.__dict__["_trt_codec"] = trt_codecs
        trt_mode = self._vocoder_trt_mode()
        trt_input_dtype = torch.float32
        trt_key = (bucket, trt_mode, trt_input_dtype)
        trt_codec = trt_codecs.get(trt_key)
        if trt_codec is not None:
            return trt_codec, bucket, trt_mode, trt_input_dtype

        try:
            import torch_tensorrt  # type: ignore[import-not-found]

            bucket_b, bucket_t = bucket
            if trt_mode == "full_codes":
                wrapper = _TensorRTCodecDecodeWrapper(codec, num_quantizers=self._n_vq).eval().to(device)
                inputs = [
                    torch_tensorrt.Input(
                        shape=(self._n_vq, bucket_b, bucket_t),
                        dtype=torch.long,
                        name="audio_codes",
                    ),
                    torch_tensorrt.Input(
                        shape=(bucket_b,),
                        dtype=torch.long,
                        name="code_lengths",
                    ),
                ]
            else:
                wrapper = _TensorRTCodecDecoderWrapper(codec).eval().to(device)
                try:
                    decoder_dtype = next(wrapper.decoder.parameters()).dtype
                except StopIteration:
                    decoder_dtype = torch.float32
                trt_input_dtype = (
                    decoder_dtype if decoder_dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32
                )
                trt_key = (bucket, trt_mode, trt_input_dtype)
                trt_codec = trt_codecs.get(trt_key)
                if trt_codec is not None:
                    return trt_codec, bucket, trt_mode, trt_input_dtype
                inputs = [
                    torch_tensorrt.Input(
                        shape=(bucket_b, 768, bucket_t),
                        dtype=trt_input_dtype,
                        name="decoder_hidden_states",
                    ),
                    torch_tensorrt.Input(
                        shape=(bucket_b,),
                        dtype=torch.long,
                        name="code_lengths",
                    ),
                ]
            compile_kwargs: dict[str, Any] = {"inputs": inputs}
            if self._env_enabled("MOSS_TTS_LOCAL_VOCODER_TRT_ENABLE_PRECISIONS"):
                compile_kwargs["enabled_precisions"] = {torch.float16, torch.float32}
            ir = os.environ.get("MOSS_TTS_LOCAL_VOCODER_TRT_IR")
            if ir:
                compile_kwargs["ir"] = ir
            trt_codec = torch_tensorrt.compile(wrapper, **compile_kwargs)
            trt_codecs[trt_key] = trt_codec
            logger.info(
                "MOSS-TTS Local vocoder TensorRT enabled for bucket batch=%d frames=%d mode=%s input_dtype=%s",
                bucket_b,
                bucket_t,
                trt_mode,
                trt_input_dtype,
            )
            return trt_codec, bucket, trt_mode, trt_input_dtype
        except Exception:
            self.__dict__["_trt_disabled"] = True
            self._warn_trt_fallback("MOSS-TTS Local vocoder TensorRT init failed; falling back to PyTorch")
            return None

    @staticmethod
    def _vocoder_cuda_graph_enabled() -> bool:
        return os.environ.get("MOSS_TTS_LOCAL_VOCODER_DISABLE_CUDA_GRAPH", "0").lower() not in (
            "1",
            "true",
            "yes",
            "on",
        )

    @staticmethod
    def _vocoder_cuda_graph_min_batch() -> int:
        try:
            return max(int(os.environ.get("MOSS_TTS_LOCAL_VOCODER_GRAPH_MIN_BATCH", "2") or 2), 1)
        except ValueError:
            return 2

    def _warn_cuda_graph_fallback(self, message: str, *args: Any) -> None:
        if bool(getattr(self, "_cuda_graph_warned", False)):
            return
        self._cuda_graph_warned = True
        logger.warning(message, *args, exc_info=True)

    def _get_or_init_cuda_graph_wrapper(
        self,
        *,
        codec: Any,
        device: torch.device,
    ) -> MossTTSLocalVocoderCUDAGraphWrapper | None:
        if not self._vocoder_cuda_graph_enabled() or bool(getattr(self, "_cuda_graph_disabled", False)):
            return None

        wrapper = getattr(self, "_cuda_graph_wrapper", None)
        if wrapper is not None and getattr(self, "_cuda_graph_codec", None) is codec:
            return wrapper

        try:
            wrapper = MossTTSLocalVocoderCUDAGraphWrapper.from_env(
                codec=codec,
                num_quantizers=self._n_vq,
            )
            wrapper.warmup(device)
        except Exception:
            self._cuda_graph_disabled = True
            self._warn_cuda_graph_fallback("MOSS-TTS Local vocoder CUDA Graph init failed; falling back to eager")
            return None

        self.__dict__["_cuda_graph_wrapper"] = wrapper
        self.__dict__["_cuda_graph_codec"] = codec
        return wrapper

    def _decode_codec_full_sequence(
        self,
        *,
        codec: Any,
        device: torch.device,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Any:
        batch_size = int(audio_codes.shape[1]) if audio_codes.dim() >= 2 else 1
        frame_count = int(audio_codes.shape[2]) if audio_codes.dim() >= 3 else 0
        trt_entry = self._get_or_init_trt_codec(
            codec=codec,
            device=device,
            batch_size=batch_size,
            frame_count=frame_count,
        )
        if trt_entry is not None:
            try:
                trt_codec, (bucket_b, bucket_t), trt_mode, trt_input_dtype = trt_entry
                code_lengths = padding_mask.sum(dim=-1).to(dtype=torch.long)
                if bucket_b != batch_size or bucket_t != frame_count:
                    padded_codes = torch.zeros(
                        self._n_vq,
                        bucket_b,
                        bucket_t,
                        dtype=audio_codes.dtype,
                        device=audio_codes.device,
                    )
                    padded_lengths = torch.zeros(bucket_b, dtype=torch.long, device=audio_codes.device)
                    padded_codes[:, :batch_size, :frame_count].copy_(audio_codes)
                    padded_lengths[:batch_size].copy_(code_lengths)
                else:
                    padded_codes = audio_codes
                    padded_lengths = code_lengths
                if trt_mode == "full_codes":
                    audio, audio_lengths = trt_codec(padded_codes, padded_lengths)
                else:
                    quantizer = getattr(codec, "quantizer", None)
                    if quantizer is None or not hasattr(quantizer, "decode_codes"):
                        raise RuntimeError("MOSS-TTS Local decoder-hidden TensorRT requires quantizer.decode_codes")
                    decoder_hidden_states = quantizer.decode_codes(padded_codes[: self._n_vq]).to(dtype=trt_input_dtype)
                    audio, audio_lengths = trt_codec(decoder_hidden_states, padded_lengths)
                if isinstance(audio, torch.Tensor):
                    audio = audio[:batch_size].clone()
                if isinstance(audio_lengths, torch.Tensor):
                    audio_lengths = audio_lengths[:batch_size].clone()
                return SimpleNamespace(audio=audio, audio_lengths=audio_lengths)
            except Exception:
                self.__dict__["_trt_disabled"] = True
                self._warn_trt_fallback("MOSS-TTS Local vocoder TensorRT decode failed; falling back to PyTorch")

        batch_size = int(audio_codes.shape[1]) if audio_codes.dim() >= 2 else 1
        graph_wrapper = (
            self._get_or_init_cuda_graph_wrapper(codec=codec, device=device)
            if batch_size >= self._vocoder_cuda_graph_min_batch()
            else None
        )
        if graph_wrapper is not None:
            try:
                return graph_wrapper.decode(audio_codes, padding_mask)
            except Exception:
                self._cuda_graph_disabled = True
                self._warn_cuda_graph_fallback("MOSS-TTS Local vocoder CUDA Graph decode failed; falling back to eager")

        return codec.decode(
            audio_codes,
            padding_mask=padding_mask,
            num_quantizers=self._n_vq,
            return_dict=True,
            chunk_duration=None,
        )

    @staticmethod
    def _vocoder_max_pad_ratio() -> float:
        try:
            return max(float(os.environ.get("MOSS_TTS_LOCAL_VOCODER_MAX_PAD_RATIO", "2.0") or 2.0), 1.0)
        except ValueError:
            return 2.0

    @staticmethod
    def _vocoder_max_batch_size() -> int:
        try:
            return max(int(os.environ.get("MOSS_TTS_LOCAL_VOCODER_MAX_BATCH", "0") or 0), 0)
        except ValueError:
            return 0

    @staticmethod
    def _vocoder_grouping_strategy() -> str:
        strategy = os.environ.get("MOSS_TTS_LOCAL_VOCODER_GROUPING", "graph_bucket")
        return strategy.strip().lower().replace("-", "_")

    @staticmethod
    def _vocoder_multi_stream_enabled() -> bool:
        return os.environ.get("MOSS_TTS_LOCAL_VOCODER_MULTI_STREAM", "0").lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _vocoder_multi_stream_max_groups() -> int:
        try:
            return max(int(os.environ.get("MOSS_TTS_LOCAL_VOCODER_MULTI_STREAM_MAX_GROUPS", "4") or 4), 1)
        except ValueError:
            return 4

    @staticmethod
    def _vocoder_multi_stream_allow_cuda_graph() -> bool:
        return os.environ.get("MOSS_TTS_LOCAL_VOCODER_MULTI_STREAM_ALLOW_CUDA_GRAPH", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @staticmethod
    def _group_by_pad_ratio(
        *,
        lengths: list[int],
        order: list[int],
        max_pad_ratio: float,
        max_batch_size: int,
    ) -> list[list[int]]:
        groups: list[list[int]] = []
        current: list[int] = []
        current_min = 0
        for index in order:
            length = lengths[index]
            next_min = length if not current else current_min
            next_max = length
            would_exceed_ratio = bool(current) and next_max > max(1, int(next_min * max_pad_ratio))
            would_exceed_batch = bool(max_batch_size and len(current) >= max_batch_size)
            if current and (would_exceed_ratio or would_exceed_batch):
                groups.append(current)
                current = []
            if not current:
                current_min = length
            current.append(index)
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _graph_bucket_batch_limit(
        *,
        graph_wrapper: MossTTSLocalVocoderCUDAGraphWrapper,
        max_batch_size: int,
    ) -> int:
        graph_batch_sizes = getattr(graph_wrapper, "capture_batch_sizes", None)
        graph_max_batch = max(graph_batch_sizes) if graph_batch_sizes else 0
        if max_batch_size > 0 and graph_max_batch > 0:
            return min(max_batch_size, int(graph_max_batch))
        if max_batch_size > 0:
            return max_batch_size
        if graph_max_batch > 0:
            return int(graph_max_batch)
        return 0

    @staticmethod
    def _group_by_graph_bucket(
        *,
        lengths: list[int],
        order: list[int],
        graph_wrapper: MossTTSLocalVocoderCUDAGraphWrapper,
        max_batch_size: int,
    ) -> list[list[int]] | None:
        """Pack groups to the largest usable CUDA graph batch/frame bucket.

        The graph wrapper already pads every replay to bucketed static buffers.
        Once the longest request in a group selects a frame bucket, shorter
        requests are essentially free from a launch-count perspective, so this
        strategy fills that bucket before starting another group.
        """
        batch_limit = MossTTSLocalVocoder._graph_bucket_batch_limit(
            graph_wrapper=graph_wrapper,
            max_batch_size=max_batch_size,
        )
        if batch_limit <= 0:
            return None

        pending = sorted(order, key=lambda i: lengths[i], reverse=True)
        groups: list[list[int]] = []
        while pending:
            anchor = pending.pop(0)
            anchor_len = lengths[anchor]
            anchor_bucket = graph_wrapper.get_bucket(batch_size=1, frame_count=anchor_len)
            if anchor_bucket is None:
                groups.append([anchor])
                continue
            _, frame_bucket = anchor_bucket
            group = [anchor]
            keep: list[int] = []
            for index in pending:
                if len(group) >= batch_limit:
                    keep.append(index)
                    continue
                if lengths[index] <= frame_bucket:
                    group.append(index)
                else:
                    keep.append(index)
            pending = keep
            groups.append(group)
        return groups

    def _decode_batch_full_sequence(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode complete ``[T, n_vq]`` code rows in one codec call."""
        if not codes_list:
            return []

        processor = self._ensure_processor()
        codec = self._resolve_audio_codec(processor)
        codec = self._maybe_compile_codec_decode(codec)
        device = self._streaming_codec_device(codec)
        n_vq = self._n_vq

        codes_channels_first: list[torch.Tensor] = []
        for codes in codes_list:
            if codes.dim() != 2 or int(codes.shape[1]) < n_vq:
                raise ValueError(f"Expected MOSS-TTS Local codes shaped [T, >={n_vq}], got {tuple(codes.shape)}")
            codes_channels_first.append(codes[:, :n_vq].transpose(0, 1).contiguous())

        lengths = [int(codes.shape[1]) for codes in codes_channels_first]
        order = sorted(range(len(codes_channels_first)), key=lambda i: lengths[i])
        max_batch_size = self._vocoder_max_batch_size()
        groups: list[list[int]] | None = None
        if self._vocoder_grouping_strategy() == "graph_bucket" and self._vocoder_cuda_graph_enabled():
            graph_wrapper = self._get_or_init_cuda_graph_wrapper(codec=codec, device=device)
            if graph_wrapper is not None:
                groups = self._group_by_graph_bucket(
                    lengths=lengths,
                    order=order,
                    graph_wrapper=graph_wrapper,
                    max_batch_size=max_batch_size,
                )
        if groups is None:
            groups = self._group_by_pad_ratio(
                lengths=lengths,
                order=order,
                max_pad_ratio=self._vocoder_max_pad_ratio(),
                max_batch_size=max_batch_size,
            )

        wavs: list[torch.Tensor | None] = [None for _ in codes_channels_first]
        use_multi_stream = (
            self._vocoder_multi_stream_enabled()
            and device.type == "cuda"
            and len(groups) > 1
            and (not self._vocoder_cuda_graph_enabled() or self._vocoder_multi_stream_allow_cuda_graph())
        )
        if (
            self._vocoder_multi_stream_enabled()
            and device.type == "cuda"
            and len(groups) > 1
            and self._vocoder_cuda_graph_enabled()
            and not self._vocoder_multi_stream_allow_cuda_graph()
        ):
            if not bool(getattr(self, "_multi_stream_warned", False)):
                self.__dict__["_multi_stream_warned"] = True
                logger.warning(
                    "MOSS-TTS Local vocoder multi-stream is disabled for CUDA Graph decode because "
                    "the graph wrapper uses shared static buffers. Set "
                    "MOSS_TTS_LOCAL_VOCODER_MULTI_STREAM_ALLOW_CUDA_GRAPH=1 only for explicit experiments."
                )
        if use_multi_stream:
            group_results = self._decode_full_sequence_groups_multi_stream(
                codec=codec,
                device=device,
                groups=groups,
                codes_channels_first=codes_channels_first,
            )
            for group, group_wavs in group_results:
                for original_index, wav in zip(group, group_wavs):
                    wavs[original_index] = wav
        else:
            for group in groups:
                group_wavs = self._decode_full_sequence_group(
                    codec=codec,
                    device=device,
                    codes_channels_first=[codes_channels_first[i] for i in group],
                )
                for original_index, wav in zip(group, group_wavs):
                    wavs[original_index] = wav
        return [wav if wav is not None else torch.zeros((0,), dtype=torch.float32) for wav in wavs]

    def _decode_full_sequence_groups_multi_stream(
        self,
        *,
        codec: Any,
        device: torch.device,
        groups: list[list[int]],
        codes_channels_first: list[torch.Tensor],
    ) -> list[tuple[list[int], list[torch.Tensor]]]:
        max_groups = self._vocoder_multi_stream_max_groups()
        outputs: list[tuple[list[int], list[torch.Tensor]]] = []
        for chunk_start in range(0, len(groups), max_groups):
            chunk = groups[chunk_start : chunk_start + max_groups]
            main_stream = torch.cuda.current_stream(device)
            launched: list[tuple[list[int], _FullDecodeGroupResult, torch.cuda.Stream]] = []
            for group in chunk:
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(main_stream)
                with torch.cuda.stream(stream):
                    result = self._decode_full_sequence_group_result(
                        codec=codec,
                        device=device,
                        codes_channels_first=[codes_channels_first[i] for i in group],
                        use_shared_buffers=False,
                    )
                launched.append((group, result, stream))
            for _, _, stream in launched:
                main_stream.wait_stream(stream)
            for group, result, _ in launched:
                outputs.append((group, self._finalize_full_sequence_group_result(result)))
        return outputs

    def _decode_full_sequence_group(
        self,
        *,
        codec: Any,
        device: torch.device,
        codes_channels_first: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        result = self._decode_full_sequence_group_result(
            codec=codec,
            device=device,
            codes_channels_first=codes_channels_first,
            use_shared_buffers=True,
        )
        return self._finalize_full_sequence_group_result(result)

    def _decode_full_sequence_group_result(
        self,
        *,
        codec: Any,
        device: torch.device,
        codes_channels_first: list[torch.Tensor],
        use_shared_buffers: bool,
    ) -> _FullDecodeGroupResult:
        max_len = max(int(codes.shape[1]) for codes in codes_channels_first)
        if max_len <= 0:
            empty_audio = torch.zeros(
                (len(codes_channels_first), 1, 0),
                dtype=torch.float32,
                device=device,
            )
            empty_lengths = torch.zeros((len(codes_channels_first),), dtype=torch.long, device=device)
            return _FullDecodeGroupResult(
                audio=empty_audio,
                audio_lengths=empty_lengths,
                code_lengths=[0 for _ in codes_channels_first],
                audio_shape=tuple(empty_audio.shape),
                pack_ms=0.0,
                decode_ms=0.0,
            )

        profile_enabled = bool(getattr(self, "_profile_enabled", False))
        if use_shared_buffers:
            audio_codes, padding_mask = self._decode_pack_buffers(
                batch_size=len(codes_channels_first),
                max_len=max_len,
                device=device,
            )
        else:
            audio_codes = torch.empty(self._n_vq, len(codes_channels_first), max_len, device=device, dtype=torch.long)
            padding_mask = torch.empty(len(codes_channels_first), max_len, device=device, dtype=torch.bool)
        pack_start = self._profile_mark(device) if profile_enabled else 0.0
        audio_codes.zero_()
        padding_mask.zero_()
        code_lengths: list[int] = []
        for index, codes in enumerate(codes_channels_first):
            length = int(codes.shape[1])
            code_lengths.append(length)
            audio_codes[:, index, :length] = codes.to(device=device, dtype=torch.long)
            padding_mask[index, :length] = True
        pack_ms = self._profile_elapsed_ms(pack_start, device) if profile_enabled else 0.0

        decode_start = self._profile_mark(device) if profile_enabled else 0.0
        decoded = self._decode_codec_full_sequence(
            codec=codec,
            device=device,
            audio_codes=audio_codes,
            padding_mask=padding_mask,
        )
        decode_ms = self._profile_elapsed_ms(decode_start, device) if profile_enabled else 0.0
        audio = getattr(decoded, "audio", None)
        audio_lengths = getattr(decoded, "audio_lengths", None)
        if audio is None:
            if isinstance(decoded, tuple) and decoded:
                audio = decoded[0]
                audio_lengths = decoded[1] if len(decoded) > 1 else None
            else:
                audio = decoded
        if not isinstance(audio, torch.Tensor):
            raise TypeError(f"MOSS-TTS Local codec returned {type(audio).__name__}, expected Tensor")
        if audio.dim() == 2:
            if len(codes_channels_first) == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ValueError(f"MOSS-TTS Local codec returned audio shaped {tuple(audio.shape)}, expected [B, C, T]")
        if int(audio.shape[0]) != len(codes_channels_first):
            raise ValueError(
                f"MOSS-TTS Local codec returned batch size {int(audio.shape[0])}, expected {len(codes_channels_first)}"
            )
        return _FullDecodeGroupResult(
            audio=audio,
            audio_lengths=audio_lengths if isinstance(audio_lengths, torch.Tensor) else None,
            code_lengths=code_lengths,
            audio_shape=tuple(audio.shape),
            pack_ms=pack_ms,
            decode_ms=decode_ms,
        )

    def _finalize_full_sequence_group_result(self, result: _FullDecodeGroupResult) -> list[torch.Tensor]:
        audio = result.audio
        audio_lengths = result.audio_lengths
        device = audio.device
        profile_enabled = bool(getattr(self, "_profile_enabled", False))
        if isinstance(audio_lengths, torch.Tensor):
            lengths_cpu = audio_lengths.detach().to("cpu")
        else:
            lengths_cpu = torch.full((audio.shape[0],), int(audio.shape[-1]), dtype=torch.long)
        if _vocoder_debug_enabled():
            logger.warning(
                "[moss-vocoder-debug] full_group batch=%d code_lens=%s audio_shape=%s audio_lengths=%s",
                len(result.code_lengths),
                result.code_lengths,
                result.audio_shape,
                lengths_cpu.reshape(-1).tolist(),
            )
        d2h_start = self._profile_mark(device) if profile_enabled else 0.0
        audio_cpu = audio.detach().to("cpu", torch.float32)
        if device.type == "cuda":
            audio_cpu = audio_cpu.contiguous()
        d2h_ms = self._profile_elapsed_ms(d2h_start, device) if profile_enabled else 0.0
        self._record_profile_group(
            batch_size=len(result.code_lengths),
            frame_count=sum(result.code_lengths),
            pack_ms=result.pack_ms,
            decode_ms=result.decode_ms,
            d2h_ms=d2h_ms,
        )
        wavs: list[torch.Tensor] = []
        for index in range(int(audio_cpu.shape[0])):
            n_samples = int(lengths_cpu.reshape(-1)[index].item())
            if n_samples <= 0 and result.code_lengths[index] > 0:
                logger.warning(
                    "MOSS-TTS Local full vocoder decoded empty audio for non-empty codes: "
                    "index=%d code_frames=%d audio_shape=%s audio_lengths=%s",
                    index,
                    result.code_lengths[index],
                    result.audio_shape,
                    lengths_cpu.reshape(-1).tolist(),
                )
            wavs.append(audio_cpu[index, :, :n_samples].contiguous())
        return wavs

    def _decode_pack_buffers(
        self,
        *,
        batch_size: int,
        max_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_buffer = getattr(self, "_decode_audio_codes_buffer", None)
        mask_buffer = getattr(self, "_decode_padding_mask_buffer", None)
        needs_new = (
            audio_buffer is None
            or mask_buffer is None
            or audio_buffer.device != device
            or mask_buffer.device != device
            or int(audio_buffer.shape[0]) < self._n_vq
            or int(audio_buffer.shape[1]) < batch_size
            or int(audio_buffer.shape[2]) < max_len
            or int(mask_buffer.shape[0]) < batch_size
            or int(mask_buffer.shape[1]) < max_len
        )
        if needs_new:
            new_batch = max(batch_size, int(audio_buffer.shape[1]) if isinstance(audio_buffer, torch.Tensor) else 0)
            new_len = max(max_len, int(audio_buffer.shape[2]) if isinstance(audio_buffer, torch.Tensor) else 0)
            audio_buffer = torch.empty(self._n_vq, new_batch, new_len, device=device, dtype=torch.long)
            mask_buffer = torch.empty(new_batch, new_len, device=device, dtype=torch.bool)
            self._decode_audio_codes_buffer = audio_buffer
            self._decode_padding_mask_buffer = mask_buffer
        return audio_buffer[: self._n_vq, :batch_size, :max_len], mask_buffer[:batch_size, :max_len]

    def _decode_streaming_chunk(
        self,
        seg: torch.Tensor,
        *,
        runtime_info: dict[str, Any] | None,
        index: int,
    ) -> torch.Tensor:
        processor = self._ensure_processor()
        codec = getattr(processor, "audio_tokenizer", None)
        if codec is None:
            raise RuntimeError("MOSS-TTS Local streaming decode requires processor.audio_tokenizer")
        missing = [
            name for name in ("streaming", "_set_streaming_exec_mask", "_decode_frame") if not hasattr(codec, name)
        ]
        if missing:
            raise RuntimeError(
                "MOSS-TTS Local streaming decode requires codec methods "
                f"{missing}; installed MOSS-Audio-Tokenizer-v2 is incompatible"
            )

        req_id = self._request_id_from_runtime_info(runtime_info, index)
        finished = self._runtime_info_finished(runtime_info)
        empty = torch.zeros((0,), dtype=torch.float32)
        if seg.numel() == 0:
            if finished:
                state = self._streaming_states.pop(req_id, None)
                if state is not None:
                    state.close()
            return empty

        if seg.numel() % self._n_vq != 0:
            logger.warning(
                "MOSS-TTS Local streaming vocoder got %d ids not divisible by n_vq=%d",
                seg.numel(),
                self._n_vq,
            )
            return empty

        if req_id not in self._streaming_states:
            if self._streaming_states:
                logger.warning(
                    "MOSS-TTS Local streaming vocoder has %d live request(s); "
                    "current deploy config should keep stage-1 max_num_seqs=1 until batched slots are implemented.",
                    len(self._streaming_states),
                )
            self._streaming_states[req_id] = _StreamingDecodeState.create(codec)

        device = self._streaming_codec_device(codec)
        frames = int(seg.numel() // self._n_vq)
        codes_step = seg.reshape(self._n_vq, frames).to(device=device, dtype=torch.long).unsqueeze(1)
        codes_lengths = torch.tensor([frames], dtype=torch.long, device=device)
        exec_mask = torch.tensor([True], dtype=torch.bool, device=device)

        try:
            codec._set_streaming_exec_mask(exec_mask)
            result = codec._decode_frame(codes_step, codes_lengths)
            audio = getattr(result, "audio", None)
            audio_lengths = getattr(result, "audio_lengths", None)
            if audio is None:
                if isinstance(result, tuple) and result:
                    audio = result[0]
                    audio_lengths = result[1] if len(result) > 1 else None
                else:
                    audio = result
            if not isinstance(audio, torch.Tensor):
                raise TypeError(f"MOSS-TTS Local streaming codec returned {type(audio).__name__}, expected Tensor")
            if isinstance(audio_lengths, torch.Tensor) and audio_lengths.numel() > 0:
                n_samples = int(audio_lengths.reshape(-1)[0].item())
            else:
                n_samples = int(audio.shape[-1])
            if audio.dim() == 3:
                wav = audio[0, :, :n_samples]
            elif audio.dim() == 2:
                wav = audio[:, :n_samples]
            else:
                wav = audio.reshape(-1)[:n_samples]
            return wav.detach().to("cpu", torch.float32).contiguous()
        finally:
            if finished:
                state = self._streaming_states.pop(req_id, None)
                if state is not None:
                    state.close()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        info_list = runtime_additional_information or []
        num_req = max(len(info_list), 1)
        empty = torch.zeros((0,), dtype=torch.float32)
        audios: list[torch.Tensor] = [empty] * num_req
        srs: list[torch.Tensor] = [self._sr_tensor] * num_req

        ids = (
            input_ids.reshape(-1).to(dtype=torch.long)
            if isinstance(input_ids, torch.Tensor)
            else torch.empty(0, dtype=torch.long)
        )
        counts = kwargs.get("num_scheduled_tokens")
        if counts is None:
            counts = kwargs.get("seq_token_counts")
        if isinstance(counts, list) and len(counts) == num_req:
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + int(count))
        else:
            offsets = [0, int(ids.numel())] if num_req == 1 else None

        debug = _vocoder_debug_enabled()
        if debug:
            logger.warning(
                "[moss-vocoder-debug] forward num_req=%d ids=%d counts=%s offsets=%s",
                num_req,
                int(ids.numel()),
                counts,
                offsets,
            )

        decode_items: list[tuple[int, torch.Tensor]] = []
        streaming_items: list[tuple[int, torch.Tensor, dict[str, Any] | None]] = []
        for i in range(num_req):
            runtime_info = info_list[i] if i < len(info_list) and isinstance(info_list[i], dict) else None
            if offsets is None:
                input_slice = torch.empty(0, dtype=torch.long, device=ids.device)
            elif i + 1 < len(offsets):
                input_slice = ids[offsets[i] : offsets[i + 1]]
            else:
                input_slice = torch.empty(0, dtype=torch.long, device=ids.device)
            seg = _codec_ids_from_payload_or_input(
                input_slice,
                runtime_info,
            )
            if debug:
                codes_audio = None
                meta = None
                if isinstance(runtime_info, dict):
                    codes_dict = runtime_info.get("codes")
                    if isinstance(codes_dict, dict):
                        codes_audio = codes_dict.get("audio")
                    meta = runtime_info.get("meta")
                logger.warning(
                    "[moss-vocoder-debug] item=%d req=%s input_slice=%d codes_audio=%s "
                    "seg=%d seg_mod=%d streaming=%s finished=%s meta=%s",
                    i,
                    self._request_id_from_runtime_info(runtime_info, i),
                    int(input_slice.numel()),
                    _shape_numel(codes_audio),
                    int(seg.numel()),
                    int(seg.numel() % self._n_vq) if self._n_vq else -1,
                    self._runtime_info_uses_streaming(runtime_info),
                    self._runtime_info_finished(runtime_info),
                    meta,
                )
            if self._runtime_info_uses_streaming(runtime_info):
                streaming_items.append((i, seg, runtime_info))
                continue
            if seg.numel() == 0:
                continue
            if seg.numel() % self._n_vq != 0:
                logger.warning("MOSS-TTS Local vocoder got %d ids not divisible by n_vq=%d", seg.numel(), self._n_vq)
                continue
            codes_t_nq = seg.view(self._n_vq, -1).transpose(0, 1).contiguous()
            decode_items.append((i, codes_t_nq))

        if decode_items:
            if debug:
                logger.warning(
                    "[moss-vocoder-debug] decode_items=%s",
                    [(i, tuple(codes.shape)) for i, codes in decode_items],
                )
            wavs = self._decode_batch_full_sequence([codes for _, codes in decode_items])
            for (i, _), wav in zip(decode_items, wavs):
                wav_t = torch.as_tensor(wav).detach().to("cpu", torch.float32)
                if debug:
                    logger.warning(
                        "[moss-vocoder-debug] decoded item=%d wav_shape=%s wav_numel=%d",
                        i,
                        tuple(wav_t.shape),
                        int(wav_t.numel()),
                    )
                audios[i] = wav_t.contiguous()

        for i, seg, runtime_info in streaming_items:
            audios[i] = self._decode_streaming_chunk(seg, runtime_info=runtime_info, index=i)

        return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": audios, "sr": srs})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The vocoder owns a separate MOSS-Audio-Tokenizer-v2 loaded by the HF
        # processor. The Local checkpoint weights belong to stage 0.
        return set()


__all__ = ["MossTTSLocalVocoder"]
