# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""CUDA Graph acceleration for MOSS-TTS Local full-sequence vocoder decode."""

from __future__ import annotations

import bisect
import os
import time
from collections import Counter
from types import SimpleNamespace
from typing import Any

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
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


class MossTTSLocalVocoderCUDAGraphWrapper:
    """CUDA Graph wrapper for ``MOSS-Audio-Tokenizer-v2`` full decode.

    The wrapped codec API is:

    ``codec.decode(audio_codes, padding_mask=..., num_quantizers=..., return_dict=True, chunk_duration=None)``

    where ``audio_codes`` is shaped ``[n_vq, batch, frames]`` and
    ``padding_mask`` is shaped ``[batch, frames]``. Graphs are keyed by
    ``(bucket_batch, bucket_frames)``. Smaller actual inputs are left-aligned
    into static buffers and right-padded with zeros / False mask values.
    """

    DEFAULT_FRAME_BUCKETS = [32, 48, 64, 80, 96, 128]
    DEFAULT_BATCH_BUCKETS = [1, 2, 4, 8]

    def __init__(
        self,
        *,
        codec: Any,
        capture_frame_sizes: list[int] | None = None,
        capture_batch_sizes: list[int] | None = None,
        num_quantizers: int = 12,
        enabled: bool = True,
    ) -> None:
        self.codec = codec
        self.capture_frame_sizes = sorted(set(capture_frame_sizes or self.DEFAULT_FRAME_BUCKETS))
        self.capture_batch_sizes = sorted(set(capture_batch_sizes or self.DEFAULT_BATCH_BUCKETS))
        self.num_quantizers = int(num_quantizers)
        self.enabled = bool(enabled)

        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_audio_codes: dict[tuple[int, int], torch.Tensor] = {}
        self.static_padding_masks: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], Any] = {}
        self._warmed_up = False
        self._device: torch.device | None = None

        self._stats_enabled = os.environ.get("MOSS_TTS_LOCAL_VOCODER_GRAPH_STATS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._stats_log_every = int(os.environ.get("MOSS_TTS_LOCAL_VOCODER_GRAPH_STATS_LOG_EVERY", "50") or 50)
        self._stats_total = 0
        self._stats_hits = 0
        self._stats_fallbacks = 0
        self._stats_shapes: Counter[tuple[int, int, int, int]] = Counter()

    @classmethod
    def from_env(cls, *, codec: Any, num_quantizers: int) -> MossTTSLocalVocoderCUDAGraphWrapper:
        return cls(
            codec=codec,
            capture_frame_sizes=_parse_int_list(
                os.environ.get("MOSS_TTS_LOCAL_VOCODER_GRAPH_FRAME_BUCKETS"),
                cls.DEFAULT_FRAME_BUCKETS,
            ),
            capture_batch_sizes=_parse_int_list(
                os.environ.get("MOSS_TTS_LOCAL_VOCODER_GRAPH_BATCH_BUCKETS"),
                cls.DEFAULT_BATCH_BUCKETS,
            ),
            num_quantizers=num_quantizers,
            enabled=os.environ.get("MOSS_TTS_LOCAL_VOCODER_DISABLE_CUDA_GRAPH", "0").lower()
            not in ("1", "true", "yes", "on"),
        )

    def get_bucket(self, *, batch_size: int, frame_count: int) -> tuple[int, int] | None:
        batch_idx = bisect.bisect_left(self.capture_batch_sizes, int(batch_size))
        frame_idx = bisect.bisect_left(self.capture_frame_sizes, int(frame_count))
        if batch_idx >= len(self.capture_batch_sizes) or frame_idx >= len(self.capture_frame_sizes):
            return None
        return self.capture_batch_sizes[batch_idx], self.capture_frame_sizes[frame_idx]

    def warmup(self, device: torch.device) -> None:
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        self._device = device
        if hasattr(self.codec, "eval"):
            self.codec.eval()
        shapes = [(batch, frames) for batch in self.capture_batch_sizes for frames in self.capture_frame_sizes]
        logger.info(
            "MOSS-TTS Local vocoder CUDA Graph warmup: shapes=%s n_vq=%d",
            shapes,
            self.num_quantizers,
        )
        start_s = time.perf_counter()
        for batch_size, frame_count in shapes:
            audio_codes = torch.zeros(
                self.num_quantizers,
                batch_size,
                frame_count,
                dtype=torch.long,
                device=device,
            )
            padding_mask = torch.ones(batch_size, frame_count, dtype=torch.bool, device=device)
            with torch.no_grad():
                _ = self._decode_eager(audio_codes, padding_mask)
        torch.accelerator.synchronize(device)

        for batch_size, frame_count in shapes:
            try:
                self._capture(batch_size, frame_count, device)
                logger.info("  Captured MOSS vocoder CUDA Graph batch=%d frames=%d", batch_size, frame_count)
            except Exception:
                logger.warning(
                    "  Failed to capture MOSS vocoder CUDA Graph batch=%d frames=%d",
                    batch_size,
                    frame_count,
                    exc_info=True,
                )
        self._warmed_up = True
        logger.info(
            "MOSS-TTS Local vocoder CUDA Graph warmup complete: %d/%d captured in %.1f ms",
            len(self.graphs),
            len(shapes),
            (time.perf_counter() - start_s) * 1000.0,
        )

    def _decode_eager(self, audio_codes: torch.Tensor, padding_mask: torch.Tensor) -> Any:
        if hasattr(self.codec, "_decode_frame"):
            code_lengths = (
                padding_mask.sum(dim=-1).to(dtype=torch.long)
                if padding_mask is not None
                else torch.full(
                    (audio_codes.shape[1],),
                    int(audio_codes.shape[-1]),
                    dtype=torch.long,
                    device=audio_codes.device,
                )
            )
            return self.codec._decode_frame(audio_codes[: self.num_quantizers], code_lengths)
        return self.codec.decode(
            audio_codes,
            padding_mask=padding_mask,
            num_quantizers=self.num_quantizers,
            return_dict=True,
            chunk_duration=None,
        )

    def _capture(self, batch_size: int, frame_count: int, device: torch.device) -> None:
        key = (batch_size, frame_count)
        audio_codes = torch.zeros(
            self.num_quantizers,
            batch_size,
            frame_count,
            dtype=torch.long,
            device=device,
        )
        padding_mask = torch.ones(batch_size, frame_count, dtype=torch.bool, device=device)
        with torch.no_grad():
            _ = self._decode_eager(audio_codes, padding_mask)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self._decode_eager(audio_codes, padding_mask)

        self.graphs[key] = graph
        self.static_audio_codes[key] = audio_codes
        self.static_padding_masks[key] = padding_mask
        self.static_outputs[key] = static_output

    def _record(self, *, hit: bool, batch_size: int, frame_count: int, bucket: tuple[int, int] | None) -> None:
        if not self._stats_enabled:
            return
        self._stats_total += 1
        if hit:
            self._stats_hits += 1
        else:
            self._stats_fallbacks += 1
        bucket_batch, bucket_frames = bucket if bucket is not None else (-1, -1)
        self._stats_shapes[(batch_size, frame_count, bucket_batch, bucket_frames)] += 1
        if self._stats_log_every > 0 and self._stats_total % self._stats_log_every == 0:
            hit_rate = 100.0 * self._stats_hits / max(1, self._stats_total)
            logger.info(
                "[moss-vocoder-graph] total=%d hits=%d fallbacks=%d hit_rate=%.1f%% top_shapes=%s",
                self._stats_total,
                self._stats_hits,
                self._stats_fallbacks,
                hit_rate,
                self._stats_shapes.most_common(12),
            )

    @torch.no_grad()
    def decode(self, audio_codes: torch.Tensor, padding_mask: torch.Tensor) -> Any:
        if not self.enabled or not self._warmed_up or audio_codes.device.type != "cuda":
            return self._decode_eager(audio_codes, padding_mask)
        if torch.cuda.is_current_stream_capturing():
            return self._decode_eager(audio_codes, padding_mask)

        batch_size = int(audio_codes.shape[1])
        frame_count = int(audio_codes.shape[2])
        bucket = self.get_bucket(batch_size=batch_size, frame_count=frame_count)
        if bucket is None or bucket not in self.graphs:
            self._record(hit=False, batch_size=batch_size, frame_count=frame_count, bucket=bucket)
            return self._decode_eager(audio_codes, padding_mask)

        bucket_batch, bucket_frames = bucket
        static_codes = self.static_audio_codes[bucket]
        static_mask = self.static_padding_masks[bucket]
        static_codes.zero_()
        static_mask.zero_()
        static_codes[:, :batch_size, :frame_count].copy_(audio_codes)
        static_mask[:batch_size, :frame_count].copy_(padding_mask)

        self.graphs[bucket].replay()
        self._record(hit=True, batch_size=batch_size, frame_count=frame_count, bucket=bucket)

        output = self.static_outputs[bucket]
        audio = getattr(output, "audio", None)
        audio_lengths = getattr(output, "audio_lengths", None)
        if isinstance(audio, torch.Tensor):
            audio = audio[:batch_size].clone()
        if isinstance(audio_lengths, torch.Tensor):
            audio_lengths = audio_lengths[:batch_size].clone()
        if hasattr(output, "__dict__"):
            values = {**output.__dict__, "audio": audio, "audio_lengths": audio_lengths}
            try:
                return type(output)(**values)
            except Exception:
                return SimpleNamespace(**values)
        return output


__all__ = ["MossTTSLocalVocoderCUDAGraphWrapper"]
