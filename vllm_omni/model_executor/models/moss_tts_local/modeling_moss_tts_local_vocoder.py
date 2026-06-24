# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Local Transformer v1.5 vocoder stage."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoProcessor
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.moss_tts_local.codec_path import (
    moss_tts_local_processor_kwargs,
)

logger = init_logger(__name__)


def _codec_ids_from_payload_or_input(input_ids: torch.Tensor | None, runtime_info: dict[str, Any] | None) -> torch.Tensor:
    if isinstance(runtime_info, dict):
        codes = runtime_info.get("codes")
        if isinstance(codes, dict):
            audio = codes.get("audio")
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                return audio.reshape(-1).to(dtype=torch.long)
            if isinstance(audio, (list, tuple)) and audio:
                return torch.as_tensor(audio, dtype=torch.long).reshape(-1)
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
    def create(cls, codec: Any) -> "_StreamingDecodeState":
        stack = contextlib.ExitStack()
        stack.enter_context(codec.streaming(batch_size=1))
        return cls(exit_stack=stack)

    def close(self) -> None:
        self.exit_stack.close()


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
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        missing = [name for name in ("streaming", "_set_streaming_exec_mask", "_decode_frame") if not hasattr(codec, name)]
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
            logger.warning("MOSS-TTS Local streaming vocoder got %d ids not divisible by n_vq=%d", seg.numel(), self._n_vq)
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

        ids = input_ids.reshape(-1).to(dtype=torch.long) if isinstance(input_ids, torch.Tensor) else torch.empty(0, dtype=torch.long)
        counts = kwargs.get("num_scheduled_tokens")
        if isinstance(counts, list) and len(counts) == num_req:
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + int(count))
        else:
            offsets = [0, int(ids.numel())]

        processor = None
        decode_items: list[tuple[int, torch.Tensor]] = []
        streaming_items: list[tuple[int, torch.Tensor, dict[str, Any] | None]] = []
        for i in range(num_req):
            runtime_info = info_list[i] if i < len(info_list) and isinstance(info_list[i], dict) else None
            seg = _codec_ids_from_payload_or_input(
                ids[offsets[i] : offsets[i + 1]] if i + 1 < len(offsets) else ids,
                runtime_info,
            )
            if self._runtime_info_uses_streaming(runtime_info):
                streaming_items.append((i, seg, runtime_info))
                continue
            if seg.numel() == 0:
                continue
            if seg.numel() % self._n_vq != 0:
                logger.warning("MOSS-TTS Local vocoder got %d ids not divisible by n_vq=%d", seg.numel(), self._n_vq)
                continue
            codes_t_nq = seg.view(self._n_vq, -1).transpose(0, 1).contiguous().cpu()
            decode_items.append((i, codes_t_nq))

        if decode_items:
            processor = self._ensure_processor()
            wavs = processor.decode_audio_codes([codes for _, codes in decode_items])
            for (i, _), wav in zip(decode_items, wavs):
                wav_t = torch.as_tensor(wav).detach().to("cpu", torch.float32)
                audios[i] = wav_t.contiguous()

        for i, seg, runtime_info in streaming_items:
            audios[i] = self._decode_streaming_chunk(seg, runtime_info=runtime_info, index=i)

        return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": audios, "sr": srs})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The vocoder owns a separate MOSS-Audio-Tokenizer-v2 loaded by the HF
        # processor. The Local checkpoint weights belong to stage 0.
        return set()


__all__ = ["MossTTSLocalVocoder"]
