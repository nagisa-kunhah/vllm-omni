# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Local Transformer v1.5 AR stage — native vLLM backbone.

Uses vLLM's compiled Qwen3Model with paged attention and CUDA graph
instead of the custom HF-compatible backbone, plus a full-frame CUDA
graph for the local transformer decode (13 sequential RVQ steps captured
as a single graph replay). Achieves ~3ms/token vs ~30ms/token with the
HF-compatible eager path (~9x speedup).

This native path is the default MOSS-TTS Local pipeline. Set
MOSS_TTS_LOCAL_NATIVE=0 to use the HF-compatible fallback backbone.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.moss_tts_local.configuration_moss_tts_local import (
    MossTTSLocalConfig,
)
from vllm_omni.model_executor.models.moss_tts_local.local_transformer import (
    MossTTSLocalTransformer,
    sample_top_k_top_p,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _first_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _copy_sampling_overrides(state: dict[str, Any], info: dict[str, Any]) -> None:
    for key in (
        "text_temperature",
        "audio_temperature",
        "text_top_k",
        "audio_top_k",
        "text_top_p",
        "audio_top_p",
        "max_new_frames",
        "min_new_frames",
    ):
        val = info.get(key)
        if val is not None:
            state[key] = _first_scalar(val)


def _stream_flag_from_info(info: dict[str, Any], default: bool = True) -> bool:
    for key in ("stream", "codec_streaming"):
        value = _first_scalar(info.get(key))
        if value is not None:
            return bool(value)
    return default


def _env_enabled(*names: str) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def _make_sampling_generator(seed: Any, device: torch.device) -> torch.Generator | None:
    seed = _first_scalar(seed)
    if seed is None:
        return None
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_int)
    return generator


def _debug_state_enabled() -> bool:
    return os.environ.get("MOSS_TTS_LOCAL_DEBUG_STATE") == "1"


def _request_label(info: dict[str, Any]) -> str:
    return str(info.get("request_id") or info.get("global_request_id") or "")[-12:]


def _update_debug_code_hash(state: dict[str, Any], codes: torch.Tensor) -> int:
    values = codes.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    h = int(state.get("_debug_code_hash", 1469598103934665603))
    for value in values:
        h ^= int(value) & 0xFFFFFFFF
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    state["_debug_code_hash"] = h
    return h


_BUCKET_SIZES = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def _next_bucket(value: int) -> int:
    for bucket in _BUCKET_SIZES:
        if int(value) <= bucket:
            return int(bucket)
    return int(value)


def _debug_tensor_digest(tensor: torch.Tensor) -> tuple[str, float, list[float]]:
    sample = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:16]
    norm = float(sample.norm().item())
    head = [float(x) for x in sample.reshape(-1)[:4].tolist()]
    return digest, norm, head


class MossTTSLocalRequestStatePool:
    """Request-slot frame state used by output assembly.

    Native v2 leaves backbone KV management to vLLM.  This pool only mirrors
    per-request frame counters and stop state so output assembly can use
    batched tensor ops instead of per-row CPU/GPU synchronization.
    """

    def __init__(self, max_slots: int) -> None:
        self.max_slots = max(1, int(max_slots))
        self._slot_to_request: list[str | None] = [None] * self.max_slots
        self._request_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self.max_slots))
        self.steps_cpu = torch.zeros(self.max_slots, dtype=torch.long)
        self.min_frames_cpu = torch.full((self.max_slots,), 3, dtype=torch.long)
        self.max_frames_cpu = torch.full((self.max_slots,), 150, dtype=torch.long)
        self.is_stopping_cpu = torch.zeros(self.max_slots, dtype=torch.bool)
        self.current_codes_cpu: torch.Tensor | None = None
        self.steps_py = [0 for _ in range(self.max_slots)]
        self.min_frames_py = [3 for _ in range(self.max_slots)]
        self.max_frames_py = [150 for _ in range(self.max_slots)]
        self.is_stopping_py = [False for _ in range(self.max_slots)]
        self._frame_state_device: torch.device | None = None
        self.steps_gpu: torch.Tensor | None = None
        self.min_frames_gpu: torch.Tensor | None = None
        self.max_frames_gpu: torch.Tensor | None = None
        self.is_stopping_gpu: torch.Tensor | None = None
        self.current_codes_gpu: torch.Tensor | None = None

    def allocate_slot(self, request_id: str) -> int:
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(f"No free MOSS-TTS Local state slots (max_slots={self.max_slots})")
        slot_id = self._free_slots.pop(0)
        self._slot_to_request[slot_id] = request_id
        self._request_to_slot[request_id] = slot_id
        return slot_id

    def slot_for_request(self, request_id: str) -> int | None:
        return self._request_to_slot.get(request_id)

    def slot_request_id(self, slot_id: int) -> str | None:
        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self.max_slots:
            return None
        return self._slot_to_request[slot_id]

    def release_slot(self, request_id: str) -> None:
        slot_id = self._request_to_slot.pop(request_id, None)
        if slot_id is None:
            return
        self._slot_to_request[slot_id] = None
        self.steps_cpu[slot_id] = 0
        self.min_frames_cpu[slot_id] = 3
        self.max_frames_cpu[slot_id] = 150
        self.is_stopping_cpu[slot_id] = False
        self.steps_py[slot_id] = 0
        self.min_frames_py[slot_id] = 3
        self.max_frames_py[slot_id] = 150
        self.is_stopping_py[slot_id] = False
        if self.steps_gpu is not None:
            self.steps_gpu[slot_id] = 0
        if self.min_frames_gpu is not None:
            self.min_frames_gpu[slot_id] = 3
        if self.max_frames_gpu is not None:
            self.max_frames_gpu[slot_id] = 150
        if self.is_stopping_gpu is not None:
            self.is_stopping_gpu[slot_id] = False
        if self.current_codes_cpu is not None:
            self.current_codes_cpu[slot_id].fill_(0)
        if self.current_codes_gpu is not None:
            self.current_codes_gpu[slot_id].fill_(0)
        self._free_slots.append(slot_id)

    def init_frame_state(self, *, slot_id: int, min_new_frames: int = 3, max_new_frames: int = 150) -> None:
        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self.max_slots:
            return
        self.steps_cpu[slot_id] = 0
        self.min_frames_cpu[slot_id] = int(min_new_frames)
        self.max_frames_cpu[slot_id] = int(max_new_frames)
        self.is_stopping_cpu[slot_id] = False
        self.steps_py[slot_id] = 0
        self.min_frames_py[slot_id] = int(min_new_frames)
        self.max_frames_py[slot_id] = int(max_new_frames)
        self.is_stopping_py[slot_id] = False
        if self.steps_gpu is not None:
            self.steps_gpu[slot_id] = 0
        if self.min_frames_gpu is not None:
            self.min_frames_gpu[slot_id] = int(min_new_frames)
        if self.max_frames_gpu is not None:
            self.max_frames_gpu[slot_id] = int(max_new_frames)
        if self.is_stopping_gpu is not None:
            self.is_stopping_gpu[slot_id] = False
        if self.current_codes_cpu is not None:
            self.current_codes_cpu[slot_id].fill_(0)
        if self.current_codes_gpu is not None:
            self.current_codes_gpu[slot_id].fill_(0)

    def ensure_frame_state_device(self, device: torch.device) -> None:
        if device.type != "cuda" or not _env_enabled("MOSS_TTS_LOCAL_GPU_FRAME_STATE"):
            return
        if (
            self._frame_state_device == device
            and self.steps_gpu is not None
            and self.min_frames_gpu is not None
            and self.max_frames_gpu is not None
            and self.is_stopping_gpu is not None
        ):
            return
        self._frame_state_device = device
        self.steps_gpu = self.steps_cpu.to(device=device, non_blocking=True)
        self.min_frames_gpu = self.min_frames_cpu.to(device=device, non_blocking=True)
        self.max_frames_gpu = self.max_frames_cpu.to(device=device, non_blocking=True)
        self.is_stopping_gpu = self.is_stopping_cpu.to(device=device, non_blocking=True)

    def ensure_current_codes(self, *, n_vq: int, audio_pad_code: int, device: torch.device) -> None:
        if self.current_codes_cpu is None or tuple(self.current_codes_cpu.shape) != (self.max_slots, int(n_vq)):
            self.current_codes_cpu = torch.full(
                (self.max_slots, int(n_vq)),
                int(audio_pad_code),
                dtype=torch.long,
            )
            self.current_codes_gpu = None
        if device.type != "cuda":
            return
        if (
            self._frame_state_device == device
            and self.current_codes_gpu is not None
            and tuple(self.current_codes_gpu.shape) == (self.max_slots, int(n_vq))
        ):
            return
        self._frame_state_device = device
        self.current_codes_gpu = self.current_codes_cpu.to(device=device, non_blocking=True)

    def set_current_codes(self, slot_id: int, codes: torch.Tensor, *, n_vq: int, audio_pad_code: int) -> None:
        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self.max_slots:
            return
        device = codes.device
        self.ensure_current_codes(n_vq=n_vq, audio_pad_code=audio_pad_code, device=device)
        codes_1d = codes.reshape(-1)[: int(n_vq)].to(dtype=torch.long)
        if self.current_codes_gpu is not None and device.type == "cuda":
            self.current_codes_gpu[slot_id].copy_(codes_1d, non_blocking=True)
        elif self.current_codes_cpu is not None:
            self.current_codes_cpu[slot_id].copy_(codes_1d.detach().to(device="cpu"), non_blocking=False)

    def gather_current_codes(
        self,
        slot_ids: list[int],
        *,
        n_vq: int,
        audio_pad_code: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not slot_ids:
            return torch.empty((0, int(n_vq)), dtype=torch.long, device=device)
        if any(int(slot_id) < 0 or int(slot_id) >= self.max_slots for slot_id in slot_ids):
            return None
        self.ensure_current_codes(n_vq=n_vq, audio_pad_code=audio_pad_code, device=device)
        source = self.current_codes_gpu if device.type == "cuda" else self.current_codes_cpu
        if source is None:
            return None
        slot_t = torch.tensor([int(slot_id) for slot_id in slot_ids], device=source.device, dtype=torch.long)
        return source.index_select(0, slot_t).to(device=device, dtype=torch.long)

    def advance_frame_state(self, slot_ids: list[int], should_stop: list[bool]) -> None:
        for slot_id, stop in zip(slot_ids, should_stop, strict=False):
            slot_id = int(slot_id)
            if slot_id < 0 or slot_id >= self.max_slots:
                continue
            if stop:
                self.is_stopping_py[slot_id] = True
                self.is_stopping_cpu[slot_id] = True
                if self.is_stopping_gpu is not None:
                    self.is_stopping_gpu[slot_id] = True
            else:
                self.steps_py[slot_id] += 1
                self.steps_cpu[slot_id] += 1
                if self.steps_gpu is not None:
                    self.steps_gpu[slot_id] += 1

    def advance_frame_state_gpu(
        self,
        slot_ids: torch.Tensor,
        should_stop: torch.Tensor,
        should_stop_cpu: list[bool],
        slot_ids_cpu: list[int] | None = None,
    ) -> None:
        if (
            self.steps_gpu is None
            or self.is_stopping_gpu is None
            or slot_ids.numel() == 0
            or slot_ids.device.type != "cuda"
        ):
            if slot_ids_cpu is None:
                slot_ids_cpu = [int(x) for x in slot_ids.detach().to("cpu").tolist()]
            self.advance_frame_state(slot_ids_cpu, should_stop_cpu)
            return
        slot_ids = slot_ids.reshape(-1).to(dtype=torch.long)
        should_stop = should_stop.reshape(-1).to(dtype=torch.bool)
        old_steps = self.steps_gpu.index_select(0, slot_ids)
        new_steps = old_steps + (~should_stop).to(dtype=torch.long)
        old_stopping = self.is_stopping_gpu.index_select(0, slot_ids)
        new_stopping = old_stopping.logical_or(should_stop)
        self.steps_gpu.index_copy_(0, slot_ids, new_steps)
        self.is_stopping_gpu.index_copy_(0, slot_ids, new_stopping)
        if slot_ids_cpu is None:
            slot_ids_cpu = [int(x) for x in slot_ids.detach().to("cpu").tolist()]
        for slot_id, stop in zip(slot_ids_cpu, should_stop_cpu, strict=False):
            slot_id = int(slot_id)
            if slot_id < 0 or slot_id >= self.max_slots:
                continue
            if stop:
                self.is_stopping_py[slot_id] = True
                self.is_stopping_cpu[slot_id] = True
            else:
                self.steps_py[slot_id] += 1
                self.steps_cpu[slot_id] += 1

    def mark_stopping(self, slot_id: int) -> None:
        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self.max_slots:
            return
        self.is_stopping_py[slot_id] = True
        self.is_stopping_cpu[slot_id] = True
        if self.is_stopping_gpu is not None:
            self.is_stopping_gpu[slot_id] = True


class MossTTSLocalNativeModel(nn.Module):
    """Stage-0 AR model with vLLM native Qwen3 backbone.

    The backbone uses vLLM's compiled Qwen3Model which handles KV cache
    management, paged attention, and CUDA graph automatically via the
    model runner's forward context.
    """

    input_modalities = "audio"
    have_multimodal_outputs: bool = True
    has_preprocess: bool = True
    has_postprocess: bool = False
    requires_raw_input_tokens: bool = True
    supports_omni_query_start_loc: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: MossTTSLocalConfig = vllm_config.model_config.hf_config
        gpt2_cfg = self.config.gpt2_config

        self.hidden_size = int(self.config.qwen3_config.hidden_size)
        self.vocab_size = int(self.config.qwen3_config.vocab_size)
        self.n_vq = int(self.config.n_vq)
        self.audio_vocab_size = int(self.config.audio_vocab_size)
        self.audio_pad_code = int(self.config.audio_pad_code)
        self.audio_assistant_slot_token_id = int(self.config.audio_assistant_slot_token_id)
        self.audio_end_token_id = int(self.config.audio_end_token_id)

        # Native vLLM Qwen3 backbone — gets paged attn + compile + CUDA graph
        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(self.audio_vocab_size + 1, self.hidden_size) for _ in range(self.n_vq)]
        )
        self.local_transformer = MossTTSLocalTransformer(
            hidden_size=self.hidden_size,
            num_heads=int(getattr(gpt2_cfg, "n_head", 32)),
            inner_size=int(getattr(gpt2_cfg, "n_inner", 4 * self.hidden_size) or 4 * self.hidden_size),
            num_layers=int(getattr(self.config, "local_transformer_layers", 1)),
            max_positions=self.n_vq + 1,
            rope_base=float(getattr(gpt2_cfg, "rope_base", 1_000_000.0)),
            layer_norm_eps=float(getattr(gpt2_cfg, "layer_norm_epsilon", 1e-6)),
            attn_implementation=os.environ.get("MOSS_TTS_LOCAL_ATTN_IMPL")
            or getattr(self.config, "local_transformer_attn_implementation", None)
            or "eager",
        )
        self.local_text_lm_head = nn.Linear(self.hidden_size, 2, bias=False)
        self._batch_state: list[dict[str, Any]] | None = None
        self._audio_embedding_indices = torch.arange(self.n_vq, dtype=torch.long)
        self._audio_embedding_weight_cache: torch.Tensor | None = None
        self._max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        self._request_state_pool = MossTTSLocalRequestStatePool(self._max_num_seqs)

        # Full-frame CUDA graph: captures the entire _decode_frame_eager as one graph
        self._frame_graphs: dict[tuple[int, tuple[float, float, int, float, float, int]], torch.cuda.CUDAGraph] = {}
        self._frame_graph_input: torch.Tensor = torch.empty(0)
        self._frame_graph_rand: torch.Tensor = torch.empty(0)
        self._frame_graph_stop: torch.Tensor = torch.empty(0)
        self._frame_graph_codes: torch.Tensor = torch.empty(0)
        self._frame_graph_max_batch: int = 0
        self._frame_graph_batch_buckets = self._parse_frame_graph_batch_buckets()
        self._delay_stop_sync_enabled = os.environ.get("MOSS_TTS_LOCAL_DELAY_STOP_SYNC", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._pending_stop_result: dict[str, Any] | None = None
        self._output_asm_buffers: dict[str, Any] = {}
        self._delay_delta_routing_enabled = _env_enabled("MOSS_TTS_LOCAL_DELAY_DELTA_ROUTING")
        try:
            self._stop_check_interval = max(int(os.environ.get("MOSS_TTS_LOCAL_STOP_CHECK_INTERVAL", "1") or 1), 1)
        except ValueError:
            self._stop_check_interval = 1
        self._profile_enabled = os.environ.get("MOSS_TTS_LOCAL_PROFILE") == "1"
        self._profile_sync = os.environ.get("MOSS_TTS_LOCAL_PROFILE_SYNC") == "1"
        self._profile_log_every = int(os.environ.get("MOSS_TTS_LOCAL_PROFILE_LOG_EVERY", "100") or 100)
        self._profile_stats: dict[str, Any] = {
            "n_forward": 0,
            "n_decode": 0,
            "qwen_ms": 0.0,
            "local_ms": 0.0,
            "make_output_ms": 0.0,
            "rowmap_ms": 0.0,
            "update_ms": 0.0,
            "output_ms": 0.0,
            "batch_sizes": [],
            "frame_graph_capture": 0,
            "frame_graph_replay": 0,
            "frame_graph_fallback": 0,
            "output_asm_ms": 0.0,
            "stop_sync_ms": 0.0,
            "hidden_rows": 0,
            "sample_rows": 0,
            "logits_rows": 0,
            "info_rows": 0,
            "hidden_pad_rows": 0,
            "shape_samples": 0,
        }
        self._default_min_frames = int(os.environ.get("MOSS_TTS_LOCAL_MIN_FRAMES", "0") or 0)

        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("hidden_states", "last"),
        }

    def _get_request_state_pool(self) -> MossTTSLocalRequestStatePool:
        pool = getattr(self, "_request_state_pool", None)
        if not isinstance(pool, MossTTSLocalRequestStatePool):
            pool = MossTTSLocalRequestStatePool(int(getattr(self, "_max_num_seqs", 1)))
            self._request_state_pool = pool
        return pool

    def _allocate_slot(self, request_id: str) -> int:
        return self._get_request_state_pool().allocate_slot(request_id)

    def _release_slot(self, request_id: str) -> None:
        self._get_request_state_pool().release_slot(request_id)

    def on_requests_finished(self, finished_req_ids: list[str]) -> None:
        for req_id in finished_req_ids:
            self._release_slot(req_id)

    def should_disable_outer_cudagraph(self) -> bool:
        return _env_enabled("MOSS_TTS_LOCAL_DISABLE_OUTER_CUDAGRAPH")

    def _set_slot_current_codes(self, slot_id: int, codes: torch.Tensor) -> None:
        if not _env_enabled("MOSS_TTS_LOCAL_USE_CODE_POOL"):
            return
        self._get_request_state_pool().set_current_codes(
            slot_id,
            codes,
            n_vq=self.n_vq,
            audio_pad_code=self.audio_pad_code,
        )

    def _gather_slot_current_codes(self, slot_ids: list[int], device: torch.device) -> torch.Tensor | None:
        if not _env_enabled("MOSS_TTS_LOCAL_USE_CODE_POOL"):
            return None
        return self._get_request_state_pool().gather_current_codes(
            slot_ids,
            n_vq=self.n_vq,
            audio_pad_code=self.audio_pad_code,
            device=device,
        )

    def _ensure_output_asm_buffers(self, *, batch_size: int, device: torch.device) -> dict[str, Any]:
        buffers = getattr(self, "_output_asm_buffers", None)
        if not isinstance(buffers, dict):
            buffers = {}
            self._output_asm_buffers = buffers
        bucket = _next_bucket(max(1, int(batch_size)))
        if buffers.get("bucket") != bucket or buffers.get("device") != device:
            buffers.clear()
            buffers["bucket"] = bucket
            buffers["device"] = device
            pin_cpu = device.type == "cuda" and torch.cuda.is_available()

            def cpu_empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
                try:
                    return torch.empty(*shape, dtype=dtype, pin_memory=pin_cpu)
                except RuntimeError:
                    return torch.empty(*shape, dtype=dtype)

            buffers["rows_cpu"] = cpu_empty(bucket, dtype=torch.long)
            buffers["slot_cpu"] = cpu_empty(bucket, dtype=torch.long)
            buffers["steps_cpu"] = cpu_empty(bucket, dtype=torch.long)
            buffers["min_cpu"] = cpu_empty(bucket, dtype=torch.long)
            buffers["max_cpu"] = cpu_empty(bucket, dtype=torch.long)
            buffers["stop_cpu"] = cpu_empty(bucket, dtype=torch.bool)
            buffers["poll_cpu"] = cpu_empty(bucket, dtype=torch.bool)
            buffers["rows_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["slot_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["steps_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["min_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["max_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["stop_gpu"] = torch.empty(bucket, device=device, dtype=torch.bool)
            buffers["tmp_stop_gpu"] = torch.empty(bucket, device=device, dtype=torch.bool)
            buffers["poll_gpu"] = torch.empty(bucket, device=device, dtype=torch.bool)
            buffers["next_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            if device.type == "cuda":
                buffers["stop_copy_event"] = torch.cuda.Event()
        return buffers

    def _can_profile_sync(self) -> bool:
        if not self._profile_sync or not torch.cuda.is_available():
            return False
        try:
            return not torch.cuda.is_current_stream_capturing()
        except Exception:
            return True

    def _profile_mark(self) -> float:
        if self._can_profile_sync():
            torch.accelerator.synchronize()
        return time.perf_counter()

    def _profile_elapsed_ms(self, start: float) -> float:
        if self._can_profile_sync():
            torch.accelerator.synchronize()
        return (time.perf_counter() - start) * 1000

    def _record_shape_profile(
        self,
        *,
        hidden_rows: int,
        sample_rows: int,
        logits_rows: int,
        info_rows: int,
    ) -> None:
        if not self._profile_enabled:
            return
        self._profile_stats["hidden_rows"] += int(hidden_rows)
        self._profile_stats["sample_rows"] += int(sample_rows)
        self._profile_stats["logits_rows"] += int(logits_rows)
        self._profile_stats["info_rows"] += int(info_rows)
        self._profile_stats["hidden_pad_rows"] += max(0, int(hidden_rows) - int(sample_rows))
        self._profile_stats["shape_samples"] += 1

    def _maybe_log_profile(self) -> None:
        if not self._profile_enabled:
            return
        n_decode = int(self._profile_stats["n_decode"])
        n_forward = int(self._profile_stats["n_forward"])
        if n_decode == 0 or n_decode % self._profile_log_every != 0:
            return
        batch_sizes = self._profile_stats["batch_sizes"][-self._profile_log_every :]
        avg_bs = sum(batch_sizes) / max(1, len(batch_sizes))
        shape_samples = int(self._profile_stats.get("shape_samples", 0))
        logger.info(
            "[moss-local-profile] forward=%d decode=%d avg_bs=%.2f "
            "qwen=%.3fms local=%.3fms make_output=%.3fms rowmap=%.3fms update=%.3fms output=%.3fms "
            "shape(hidden=%.1f sample=%.1f logits=%.1f infos=%.1f pad=%.1f) "
            "frame_graph(capture=%d replay=%d fallback=%d) sync=%s outer_graph_off=%s",
            n_forward,
            n_decode,
            avg_bs,
            self._profile_stats["qwen_ms"] / max(1, n_forward),
            self._profile_stats["local_ms"] / max(1, n_decode),
            self._profile_stats["make_output_ms"] / max(1, n_decode),
            self._profile_stats["rowmap_ms"] / max(1, n_decode),
            self._profile_stats["update_ms"] / max(1, n_decode),
            self._profile_stats["output_ms"] / max(1, n_decode),
            self._profile_stats.get("hidden_rows", 0) / max(1, shape_samples),
            self._profile_stats.get("sample_rows", 0) / max(1, shape_samples),
            self._profile_stats.get("logits_rows", 0) / max(1, shape_samples),
            self._profile_stats.get("info_rows", 0) / max(1, shape_samples),
            self._profile_stats.get("hidden_pad_rows", 0) / max(1, shape_samples),
            int(self._profile_stats.get("frame_graph_capture", 0)),
            int(self._profile_stats.get("frame_graph_replay", 0)),
            int(self._profile_stats.get("frame_graph_fallback", 0)),
            self._profile_sync,
            self.should_disable_outer_cudagraph(),
        )

    def _consume_pending_stop_result(
        self, info_dicts: list[dict[str, Any]]
    ) -> tuple[list[str], list[torch.Tensor], list[bool]]:
        pending = getattr(self, "_pending_stop_result", None)
        if not pending:
            return [], [], []
        self._pending_stop_result = None
        event = pending.get("event")
        if isinstance(event, torch.cuda.Event):
            event.synchronize()
        stop_cpu = pending.get("stop_cpu")
        slot_ids = pending.get("slot_ids") or []
        req_ids = pending.get("req_ids") or []
        if isinstance(stop_cpu, torch.Tensor) and not slot_ids and req_ids:
            stop_values = [bool(x) for x in stop_cpu[: len(req_ids)].tolist()]
            info_by_req = {
                str(info.get("request_id") or info.get("global_request_id") or ""): info
                for info in info_dicts
                if isinstance(info, dict)
            }
            for req_id, should_stop in zip(req_ids, stop_values, strict=False):
                if not should_stop:
                    continue
                info = info_by_req.get(str(req_id))
                if info is None:
                    continue
                state = info.get("audio_state")
                if not isinstance(state, dict):
                    state = {}
                    info["audio_state"] = state
                state["is_stopping"] = True
                state["next_text"] = self.audio_end_token_id
                state.setdefault("stop_reason", "model_stop")
            return [], [], []
        if not isinstance(stop_cpu, torch.Tensor) or not slot_ids:
            return [], [], []
        stop_values = [bool(x) for x in stop_cpu[: len(slot_ids)].tolist()]
        force_stop = pending.get("force_stop") or []
        stop_values = [
            bool(model_stop or (idx < len(force_stop) and force_stop[idx]))
            for idx, model_stop in enumerate(stop_values)
        ]
        pending_steps = pending.get("steps") or []
        pending_codes = pending.get("codes")
        pending_streaming = pending.get("codec_streaming") or []
        emit_req_ids: list[str] = []
        emit_codes: list[torch.Tensor] = []
        emit_streaming: list[bool] = []

        pool = self._get_request_state_pool()
        state_by_slot: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            try:
                slot_id = int(info.get("_kv_slot_id"))
            except (TypeError, ValueError):
                continue
            state = info.get("audio_state")
            if isinstance(state, dict):
                state_by_slot[slot_id] = (info, state)

        for idx, should_stop in enumerate(stop_values):
            if idx >= len(slot_ids):
                continue
            slot_id = int(slot_ids[idx])
            if slot_id < 0 or slot_id >= pool.max_slots:
                continue
            expected_req_id = str(req_ids[idx]) if idx < len(req_ids) else ""
            current_req_id = pool.slot_request_id(slot_id)
            if expected_req_id and current_req_id != expected_req_id:
                continue
            if not should_stop:
                if isinstance(pending_codes, torch.Tensor) and idx < int(pending_codes.shape[0]):
                    rid = expected_req_id or str(current_req_id or "")
                    if rid or len(info_dicts) == 1:
                        emit_req_ids.append(rid)
                        emit_codes.append(pending_codes[idx : idx + 1])
                        emit_streaming.append(bool(pending_streaming[idx]) if idx < len(pending_streaming) else True)
                continue
            info_state = state_by_slot.get(slot_id)
            stop_step = int(pending_steps[idx]) if idx < len(pending_steps) else pool.steps_py[slot_id]
            pool.steps_py[slot_id] = stop_step
            pool.steps_cpu[slot_id] = stop_step
            if pool.steps_gpu is not None:
                pool.steps_gpu[slot_id] = stop_step
            pool.mark_stopping(slot_id)
            if info_state is None:
                continue
            _info, state = info_state
            state["is_stopping"] = True
            state["next_text"] = self.audio_end_token_id
            state["stop_reason"] = "model_stop"
            state["step"] = stop_step
        return emit_req_ids, emit_codes, emit_streaming

    @staticmethod
    def _compute_stop_and_next_tokens(
        *,
        stop_choices: torch.Tensor,
        steps: torch.Tensor,
        min_frames: torch.Tensor,
        max_frames: torch.Tensor,
        audio_assistant_slot_token_id: int,
        audio_end_token_id: int,
        should_stop_out: torch.Tensor | None = None,
        tmp_out: torch.Tensor | None = None,
        next_tokens_out: torch.Tensor | None = None,
        model_stop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        should_stop = (
            should_stop_out if should_stop_out is not None else torch.empty_like(stop_choices, dtype=torch.bool)
        )
        tmp = tmp_out if tmp_out is not None else torch.empty_like(should_stop)
        next_tokens = next_tokens_out if next_tokens_out is not None else torch.empty_like(steps, dtype=torch.long)

        torch.eq(stop_choices.reshape(-1)[: steps.numel()], 1, out=should_stop)
        if model_stop_mask is not None:
            should_stop.logical_and_(model_stop_mask.reshape(-1)[: steps.numel()])
        torch.ge(steps, min_frames, out=tmp)
        should_stop.logical_and_(tmp)
        torch.ge(steps, max_frames, out=tmp)
        should_stop.logical_or_(tmp)

        next_tokens.fill_(int(audio_assistant_slot_token_id))
        next_tokens.masked_fill_(should_stop, int(audio_end_token_id))
        return should_stop, next_tokens

    def _build_input_embeds(self, text_ids: torch.Tensor, audio_codes: torch.Tensor | None) -> torch.Tensor:
        embeds = self.model.embed_tokens(text_ids)
        if audio_codes is None:
            return embeds
        codes = audio_codes.to(device=text_ids.device, dtype=torch.long)
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        valid = codes.ne(self.audio_pad_code)
        safe_codes = codes.clamp(0, self.audio_vocab_size).masked_fill(~valid, 0)
        weights = self._stacked_audio_embedding_weights(device=text_ids.device)
        codebook_idx = self._audio_embedding_indices.to(device=text_ids.device)
        audio_embeds = weights[codebook_idx.unsqueeze(0), safe_codes]
        return embeds + (audio_embeds * valid.unsqueeze(-1)).sum(dim=1)

    @staticmethod
    def _parse_frame_graph_batch_buckets() -> tuple[int, ...]:
        raw = os.environ.get("MOSS_TTS_LOCAL_FRAME_GRAPH_BATCH_BUCKETS", "").strip()
        values: list[int] = []
        if raw:
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    value = int(part)
                except ValueError:
                    logger.warning("Ignoring invalid MOSS_TTS_LOCAL_FRAME_GRAPH_BATCH_BUCKETS entry: %s", part)
                    continue
                if value > 0:
                    values.append(value)
        if not values:
            values = [1, 2, 4, 8, 16, 32, 64]
        return tuple(sorted(set(values)))

    def _frame_graph_bucket_for_batch(self, batch_size: int) -> int:
        for bucket in self._frame_graph_batch_buckets:
            if batch_size <= bucket:
                return int(bucket)
        return int(batch_size)

    def _stacked_audio_embedding_weights(self, *, device: torch.device) -> torch.Tensor:
        cache = self._audio_embedding_weight_cache
        dtype = self.audio_embeddings[0].weight.dtype
        if cache is None or cache.device != device or cache.dtype != dtype:
            cache = torch.stack([emb.weight for emb in self.audio_embeddings], dim=0).to(device=device)
            self._audio_embedding_weight_cache = cache
        return cache

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del input_embeds
        device = input_ids.device
        span_len = int(input_ids.shape[0])
        audio_state = info_dict.get("audio_state")
        is_first_call = not isinstance(audio_state, dict)

        request_id = str(info_dict.get("request_id") or info_dict.get("global_request_id") or "")
        pool = self._get_request_state_pool()
        slot_id = pool.slot_for_request(request_id) if request_id else None
        if is_first_call and slot_id is None and request_id:
            slot_id = self._allocate_slot(request_id)
        if slot_id is None:
            slot_id = 0

        if span_len > 1 or is_first_call:
            prompt_rows = info_dict.get("prompt_rows")
            if isinstance(prompt_rows, torch.Tensor) and prompt_rows.numel() > 0:
                rows = prompt_rows.to(device=device, dtype=torch.long)
                text_ids = rows[:, 0]
                audio_codes = rows[:, 1 : self.n_vq + 1]
                embeds = self._build_input_embeds(text_ids, audio_codes)
                current_codes = audio_codes[-1].detach()
            else:
                ref_codes = (info_dict.get("codes", {}) or {}).get("ref")
                ref_offset = int(info_dict.get("ref_offset", 0))
                chunk_audio = None
                if isinstance(ref_codes, torch.Tensor) and ref_codes.numel() > 0:
                    if ref_codes.dim() == 1 and ref_codes.numel() % self.n_vq == 0:
                        ref_codes = ref_codes.view(-1, self.n_vq)
                    if isinstance(ref_codes, torch.Tensor) and ref_codes.dim() == 2:
                        sliced = ref_codes[ref_offset : ref_offset + span_len]
                        if sliced.shape[0] == span_len:
                            chunk_audio = sliced.to(device=device, dtype=torch.long)
                embeds = self._build_input_embeds(input_ids, chunk_audio)
                current_codes = (
                    chunk_audio[-1].detach()
                    if chunk_audio is not None
                    else torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
                )
                ref_offset = ref_offset + span_len

            max_new_frames = _first_scalar(info_dict.get("max_new_frames"))
            try:
                max_new_frames = int(max_new_frames) if max_new_frames is not None else -1
            except (TypeError, ValueError):
                max_new_frames = -1
            min_new_frames = _first_scalar(info_dict.get("min_new_frames"))
            try:
                min_new_frames = int(min_new_frames) if min_new_frames is not None else 3
            except (TypeError, ValueError):
                min_new_frames = 3
            state = {
                "step": 0,
                "is_stopping": False,
                "next_text": self.audio_assistant_slot_token_id,
                "max_new_frames": max_new_frames,
                "min_new_frames": min_new_frames,
            }
            pool.init_frame_state(
                slot_id=slot_id,
                min_new_frames=min_new_frames,
                max_new_frames=max_new_frames if max_new_frames > 0 else 150,
            )
            pool.ensure_frame_state_device(device)
            self._set_slot_current_codes(slot_id, current_codes)
            _copy_sampling_overrides(state, info_dict)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {"current": current_codes},
                    "ref_offset": int(info_dict.get("ref_offset", 0)) + span_len,
                    "_kv_slot_id": slot_id,
                },
            )

        prev_codes = None
        if _env_enabled("MOSS_TTS_LOCAL_USE_CODE_POOL"):
            gathered = self._gather_slot_current_codes([int(slot_id)], device)
            if isinstance(gathered, torch.Tensor) and gathered.numel() == self.n_vq:
                prev_codes = gathered.reshape(-1)
        if prev_codes is None:
            prev_codes = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(prev_codes, torch.Tensor) or prev_codes.numel() != self.n_vq:
            prev_codes = torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
        embeds = self._build_input_embeds(input_ids.reshape(-1), prev_codes.to(device=device).unsqueeze(0))
        return input_ids, embeds, {"_kv_slot_id": slot_id}

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Batch single-token decode embedding construction.

        Native Stage0 relies on vLLM for KV/cache routing, so decode
        preprocess only needs to combine the sampled text token with the
        previous frame's RVQ codes.  Doing this once for the whole scheduled
        decode batch avoids one embedding/gather/sum launch sequence per
        request at high concurrency.
        """
        device = input_ids.device
        text_ids = input_ids.reshape(-1)
        pool = self._get_request_state_pool()
        slot_ids: list[int] = []
        prev_codes_list: list[torch.Tensor] = []
        for info in req_infos:
            request_id = str(info.get("request_id") or info.get("global_request_id") or "")
            slot_id = pool.slot_for_request(request_id) if request_id else None
            if slot_id is None:
                slot_id = 0
            slot_ids.append(int(slot_id))
            prev_codes = (info.get("audio_codes", {}) or {}).get("current")
            if not isinstance(prev_codes, torch.Tensor) or prev_codes.numel() != self.n_vq:
                prev_codes = torch.full(
                    (self.n_vq,),
                    self.audio_pad_code,
                    dtype=torch.long,
                    device=device,
                )
            prev_codes_list.append(prev_codes.to(device=device, dtype=torch.long).reshape(-1)[: self.n_vq])

        prev_codes_batch = self._gather_slot_current_codes(slot_ids, device)
        if prev_codes_batch is not None:
            prev_codes_batch = prev_codes_batch.reshape(len(slot_ids), self.n_vq)
        elif prev_codes_list:
            prev_codes_batch = torch.stack(prev_codes_list, dim=0)
        else:
            prev_codes_batch = torch.empty((0, self.n_vq), dtype=torch.long, device=device)
        embeds = self._build_input_embeds(text_ids, prev_codes_batch)
        return text_ids, embeds, [{"_kv_slot_id": slot_id} for slot_id in slot_ids]

    def preprocess_decode_batch_fast(
        self,
        *,
        input_ids: torch.Tensor,
        req_ids: list[str],
        prev_codes: list[torch.Tensor | None],
        slot_ids: list[int | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Fast decode preprocess with only the fields needed by MOSS native.

        The generic runner path attaches per-request metadata dictionaries
        before calling preprocess.  Single-token MOSS decode only needs the
        sampled token, previous RVQ codes and the request slot, so this entry
        point keeps the hot path out of the larger req_info dictionaries.
        """
        device = input_ids.device
        text_ids = input_ids.reshape(-1)
        pool = self._get_request_state_pool()
        resolved_slot_ids: list[int] = []
        prev_codes_list: list[torch.Tensor] = []
        for idx, req_id in enumerate(req_ids):
            slot_id = None
            if slot_ids is not None and idx < len(slot_ids):
                try:
                    slot_id = int(slot_ids[idx]) if slot_ids[idx] is not None else None
                except (TypeError, ValueError):
                    slot_id = None
            if slot_id is None or slot_id < 0:
                slot_id = pool.slot_for_request(str(req_id)) if req_id else None
            if slot_id is None:
                slot_id = 0
            resolved_slot_ids.append(int(slot_id))

            codes = prev_codes[idx] if idx < len(prev_codes) else None
            if not isinstance(codes, torch.Tensor) or codes.numel() != self.n_vq:
                codes = torch.full(
                    (self.n_vq,),
                    self.audio_pad_code,
                    dtype=torch.long,
                    device=device,
                )
            prev_codes_list.append(codes.to(device=device, dtype=torch.long).reshape(-1)[: self.n_vq])

        prev_codes_batch = self._gather_slot_current_codes(resolved_slot_ids, device)
        if prev_codes_batch is not None:
            prev_codes_batch = prev_codes_batch.reshape(len(resolved_slot_ids), self.n_vq)
        elif prev_codes_list:
            prev_codes_batch = torch.stack(prev_codes_list, dim=0)
        else:
            prev_codes_batch = torch.empty((0, self.n_vq), dtype=torch.long, device=device)
        embeds = self._build_input_embeds(text_ids, prev_codes_batch)
        return text_ids, embeds, [{"_kv_slot_id": slot_id} for slot_id in resolved_slot_ids]

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        # Native path: vLLM handles batching, KV cache, attention via forward context
        if not self._profile_enabled:
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        start = self._profile_mark()
        result = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        self._profile_stats["qwen_ms"] += self._profile_elapsed_ms(start)
        self._profile_stats["n_forward"] += 1
        return result

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        next_text_from_output: torch.Tensor | None = None
        if isinstance(hidden_states, OmniOutput):
            meta = hidden_states.multimodal_outputs.get("meta", {}) if hidden_states.multimodal_outputs else {}
            next_text = meta.get("next_text") if isinstance(meta, dict) else None
            if isinstance(next_text, torch.Tensor):
                next_text_from_output = next_text.to(device=hidden_states.text_hidden_states.device, dtype=torch.long)
            hidden_states = hidden_states.text_hidden_states
        if hidden_states.numel() == 0:
            return torch.zeros(1, self.vocab_size, device=hidden_states.device)
        if next_text_from_output is not None:
            tokens = next_text_from_output.reshape(-1).clamp(0, self.vocab_size - 1)
            logits = hidden_states.new_full((int(tokens.numel()), self.vocab_size), -1e9)
            logits.scatter_(1, tokens.unsqueeze(1), 0.0)
            self._batch_state = None
            if _debug_state_enabled():
                logger.info(
                    "[moss-local-state] compute_logits source=meta rows=%d next=%s",
                    int(tokens.numel()),
                    next_text_from_output.reshape(-1).detach().cpu().tolist(),
                )
            return logits
        batch_state = self._batch_state
        if not batch_state:
            num_rows = hidden_states.shape[0] if hidden_states.dim() >= 1 else 1
            if _debug_state_enabled():
                logger.info("[moss-local-state] compute_logits source=zeros rows=%d", num_rows)
            return torch.zeros(num_rows, self.vocab_size, device=hidden_states.device)
        tokens = torch.full(
            (len(batch_state),),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        debug_next: list[int | None] = []
        for i, state in enumerate(batch_state):
            if not isinstance(state, dict):
                debug_next.append(None)
                continue
            next_text = state.get("next_text")
            if next_text is not None:
                token = max(0, min(int(next_text), self.vocab_size - 1))
                tokens[i] = token
                debug_next.append(token)
            else:
                debug_next.append(None)
        self._batch_state = None
        logits = hidden_states.new_full((len(batch_state), self.vocab_size), -1e9)
        forced = tokens.ge(0)
        if forced.any():
            forced_tokens = tokens[forced]
            logits[forced] = -1e9
            logits[forced, forced_tokens] = 0.0
        if (~forced).any():
            logits[~forced] = 0.0
        if _debug_state_enabled():
            logger.info(
                "[moss-local-state] compute_logits source=batch_state rows=%d next=%s",
                len(batch_state),
                debug_next,
            )
        return logits

    @staticmethod
    def _frame_params(info: dict[str, Any]) -> tuple[float, float, int, float, float, int, torch.Generator | None]:
        state = info.get("audio_state", {}) or {}
        generator = state.get("sampling_generator")
        if not isinstance(generator, torch.Generator):
            generator = None
        temperature = _first_scalar(info.get("temperature", 1.0))
        top_p = _first_scalar(info.get("top_p", 1.0))
        top_k = _first_scalar(info.get("top_k", 50))
        return (
            float(state.get("text_temperature", _first_scalar(info.get("text_temperature", temperature)))),
            float(state.get("text_top_p", _first_scalar(info.get("text_top_p", top_p)))),
            int(state.get("text_top_k", _first_scalar(info.get("text_top_k", top_k)))),
            float(state.get("audio_temperature", _first_scalar(info.get("audio_temperature", temperature)))),
            float(state.get("audio_top_p", _first_scalar(info.get("audio_top_p", top_p)))),
            int(state.get("audio_top_k", _first_scalar(info.get("audio_top_k", top_k)))),
            generator,
        )

    def _frame_graph_params(
        self,
        infos: list[dict[str, Any]],
        batch_size: int,
    ) -> tuple[tuple[float, float, int, float, float, int], bool]:
        params = [self._frame_params(info if isinstance(info, dict) else {}) for info in infos[:batch_size]]
        if len(params) != batch_size:
            params.extend([self._frame_params({}) for _ in range(batch_size - len(params))])
        first_params = params[0][:6] if params else (1.0, 1.0, 50, 1.0, 0.95, 50)
        graphable = all(item[:6] == first_params and item[6] is None for item in params)
        return first_params, graphable

    @staticmethod
    def _sample_with_params(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if float(temperature) <= 0:
            return torch.argmax(logits, dim=-1)
        return sample_top_k_top_p(
            logits,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            generator=generator,
        )

    @staticmethod
    def _sample_graph_safe(
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
        rand_val: torch.Tensor,
    ) -> torch.Tensor:
        scores = logits / max(float(temperature), 1e-6)
        vocab = int(scores.shape[-1])
        top_k = int(top_k)
        top_p = float(top_p)
        if 0 < top_k < vocab:
            topk_scores, topk_indices = torch.topk(scores, top_k, dim=-1)
            if 0.0 < top_p < 1.0:
                sorted_scores, sorted_order = torch.sort(topk_scores, descending=True, dim=-1)
                sorted_indices = topk_indices.gather(-1, sorted_order)
                sorted_probs = torch.softmax(sorted_scores, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
                probs = torch.softmax(sorted_scores, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                cdf = torch.cumsum(probs, dim=-1)
                sampled_rel = torch.searchsorted(cdf, rand_val.unsqueeze(-1)).reshape(-1)
                sampled_rel = sampled_rel.clamp(0, top_k - 1).unsqueeze(-1)
                return sorted_indices.gather(-1, sampled_rel).reshape(-1).clamp(0, vocab - 1)

            probs = torch.softmax(topk_scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            cdf = torch.cumsum(probs, dim=-1)
            sampled_rel = torch.searchsorted(cdf, rand_val.unsqueeze(-1)).reshape(-1)
            sampled_rel = sampled_rel.clamp(0, top_k - 1).unsqueeze(-1)
            return topk_indices.gather(-1, sampled_rel).reshape(-1).clamp(0, vocab - 1)

        if 0.0 < top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, sorted_indices, remove)
            scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        cdf = torch.cumsum(probs, dim=-1)
        sampled = torch.searchsorted(cdf, rand_val.unsqueeze(-1)).reshape(-1)
        return sampled.clamp(0, vocab - 1)

    @staticmethod
    def _sample_graph_safe_param(
        logits: torch.Tensor,
        *,
        top_k: int,
        top_p: float,
        temperature: float,
        rand_val: torch.Tensor,
    ) -> torch.Tensor:
        if float(temperature) <= 0:
            return torch.argmax(logits, dim=-1)
        return MossTTSLocalNativeModel._sample_graph_safe(
            logits,
            int(top_k),
            float(top_p),
            float(temperature),
            rand_val,
        )

    def _frame_graph_kernel(
        self,
        batch_size: int,
        params: tuple[float, float, int, float, float, int],
    ) -> None:
        text_temp, text_top_p, text_top_k, audio_temp, audio_top_p, audio_top_k = params
        hidden = self._frame_graph_input[:batch_size]
        rand = self._frame_graph_rand[:, :batch_size]
        local_hidden = self.local_transformer._step_eager(hidden.to(dtype=self.audio_embeddings[0].weight.dtype), 0)
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()
        stop = self._sample_graph_safe_param(
            text_logits,
            temperature=text_temp,
            top_p=text_top_p,
            top_k=text_top_k,
            rand_val=rand[0],
        )
        self._frame_graph_stop[:batch_size].copy_(stop)
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
            logits = F.linear(current, head_weight).float()
            codes = self._sample_graph_safe_param(
                logits,
                temperature=audio_temp,
                top_p=audio_top_p,
                top_k=audio_top_k,
                rand_val=rand[1 + channel],
            )
            self._frame_graph_codes[:batch_size, channel].copy_(codes)
            if channel + 1 < self.n_vq:
                embedded = F.embedding(codes, head_weight).to(dtype=current.dtype)
                current = self.local_transformer._step_eager(embedded, channel + 1)

    def _decode_frame_eager(
        self,
        hidden_batch: torch.Tensor,
        infos: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure computation path (no graph logic) for capture or eager fallback."""
        B = hidden_batch.shape[0]
        infos = infos or [{} for _ in range(B)]
        dtype = self.audio_embeddings[0].weight.dtype
        local_hidden = self.local_transformer._step_eager(hidden_batch.to(dtype=dtype), 0)
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()

        params = [
            self._frame_params(infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}) for i in range(B)
        ]
        first_params = params[0][:6] if params else (1.0, 1.0, 50, 1.0, 0.95, 50)
        homogeneous = all(item[:6] == first_params and item[6] is None for item in params)

        greedy = homogeneous and float(first_params[0]) <= 0 and float(first_params[3]) <= 0
        if greedy:
            text_temp, text_top_p, text_top_k, audio_temp, audio_top_p, audio_top_k = first_params
            stop_choices = torch.argmax(text_logits, dim=-1)
        elif homogeneous:
            text_temp, text_top_p, text_top_k, audio_temp, audio_top_p, audio_top_k = first_params
            stop_choices = self._sample_with_params(
                text_logits,
                temperature=text_temp,
                top_p=text_top_p,
                top_k=text_top_k,
            )
        else:
            stop_choices = torch.zeros(B, dtype=torch.long, device=hidden_batch.device)
            for b, (text_temp, text_top_p, text_top_k, _, _, _, generator) in enumerate(params):
                stop_choices[b] = self._sample_with_params(
                    text_logits[b : b + 1],
                    temperature=text_temp,
                    top_p=text_top_p,
                    top_k=text_top_k,
                    generator=generator,
                )[0]

        if os.environ.get("MOSS_TTS_LOCAL_DEBUG_STOP_LOGITS") == "1":
            logits_cpu = text_logits.detach().to("cpu", dtype=torch.float32)
            stops_cpu = stop_choices.detach().to("cpu", dtype=torch.long).reshape(-1).tolist()
            for b in range(B):
                info = infos[b] if b < len(infos) and isinstance(infos[b], dict) else {}
                state = info.get("audio_state", {}) or {}
                step = int(state.get("step", 0))
                margin = float((logits_cpu[b, 1] - logits_cpu[b, 0]).item())
                if stops_cpu[b] == 1 or step < 5 or os.environ.get("MOSS_TTS_LOCAL_TRACE_STEPS") == "1":
                    probs = torch.softmax(logits_cpu[b], dim=-1).tolist()
                    text_temp, text_top_p, text_top_k, audio_temp, audio_top_p, audio_top_k, generator = params[b]
                    logger.info(
                        "[moss-local-stop] req=%s step=%d stop=%d logits=%s probs=%s "
                        "margin=%.4f text_temp=%.3f audio_temp=%.3f text_top_k=%d "
                        "audio_top_k=%d gen=%s",
                        _request_label(info),
                        step,
                        int(stops_cpu[b]),
                        [float(x) for x in logits_cpu[b].tolist()],
                        [float(x) for x in probs],
                        margin,
                        text_temp,
                        audio_temp,
                        text_top_k,
                        audio_top_k,
                        generator is not None,
                    )

        all_codes = torch.zeros(B, self.n_vq, dtype=torch.long, device=hidden_batch.device)
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
            logits = F.linear(current, head_weight).float()
            if greedy:
                codes = torch.argmax(logits, dim=-1)
            elif homogeneous:
                codes = self._sample_with_params(
                    logits,
                    temperature=audio_temp,
                    top_p=audio_top_p,
                    top_k=audio_top_k,
                )
            else:
                codes = torch.zeros(B, dtype=torch.long, device=hidden_batch.device)
                for b, (_, _, _, audio_temp, audio_top_p, audio_top_k, generator) in enumerate(params):
                    codes[b] = self._sample_with_params(
                        logits[b : b + 1],
                        temperature=audio_temp,
                        top_p=audio_top_p,
                        top_k=audio_top_k,
                        generator=generator,
                    )[0]
            all_codes[:, channel] = codes
            if channel + 1 < self.n_vq:
                embedded = F.embedding(codes, head_weight).to(dtype=current.dtype)
                current = self.local_transformer._step_eager(embedded, channel + 1)
        if os.environ.get("MOSS_TTS_LOCAL_TRACE_STEPS") == "1":
            codes_cpu = all_codes.detach().to("cpu", dtype=torch.long)
            for b in range(B):
                info = infos[b] if b < len(infos) and isinstance(infos[b], dict) else {}
                state = info.get("audio_state", {}) or {}
                values = codes_cpu[b].reshape(-1).tolist()
                h = 1469598103934665603
                for value in values:
                    h ^= int(value) & 0xFFFFFFFF
                    h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
                logger.info(
                    "[moss-local-trace] req=%s step=%d stop=%d code_hash=%016x codes0=%s",
                    _request_label(info),
                    int(state.get("step", 0)),
                    int(stop_choices[b].item()),
                    h,
                    [int(x) for x in values[:4]],
                )
        return stop_choices, all_codes

    def _ensure_frame_graph_buffers(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if batch_size <= self._frame_graph_max_batch:
            return
        cap = max(batch_size, 8)
        self._frame_graph_input = torch.zeros(cap, self.hidden_size, device=device, dtype=dtype)
        self._frame_graph_rand = torch.zeros(1 + self.n_vq, cap, device=device, dtype=torch.float32)
        self._frame_graph_stop = torch.zeros(cap, device=device, dtype=torch.long)
        self._frame_graph_codes = torch.zeros(cap, self.n_vq, device=device, dtype=torch.long)
        self._frame_graph_max_batch = cap
        self._frame_graphs.clear()

    def _decode_frame_batched(
        self,
        hidden_batch: torch.Tensor,
        infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched local transformer decode: [B, H] -> (stop_choices[B], codes[B, n_vq])."""
        B = hidden_batch.shape[0]
        dtype = self.audio_embeddings[0].weight.dtype

        if hidden_batch.device.type != "cuda":
            self.local_transformer._ensure_kv_cache(B, hidden_batch.device, dtype)
            return self._decode_frame_eager(hidden_batch, infos)
        if os.environ.get("MOSS_TTS_LOCAL_ENABLE_FRAME_GRAPH") != "1":
            self.local_transformer._ensure_kv_cache(B, hidden_batch.device, dtype)
            return self._decode_frame_eager(hidden_batch, infos)

        first_params, graphable = self._frame_graph_params(infos, B)
        if not graphable:
            self.local_transformer._ensure_kv_cache(B, hidden_batch.device, dtype)
            return self._decode_frame_eager(hidden_batch, infos)

        try:
            bucket = self._frame_graph_bucket_for_batch(B)
            self.local_transformer._ensure_kv_cache(bucket, hidden_batch.device, dtype)
            self._ensure_frame_graph_buffers(bucket, hidden_batch.device, dtype)
            graph_key = (bucket, first_params)
            self._frame_graph_rand[:, :bucket].copy_(torch.rand(1 + self.n_vq, bucket, device=hidden_batch.device))
            if graph_key not in self._frame_graphs:
                self._frame_graph_input[:bucket].zero_()
                self._frame_graph_input[:B].copy_(hidden_batch)
                self._frame_graph_kernel(bucket, first_params)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._frame_graph_kernel(bucket, first_params)
                self._frame_graphs[graph_key] = g
                if self._profile_enabled:
                    self._profile_stats["frame_graph_capture"] += 1
                return self._frame_graph_stop[:B].clone(), self._frame_graph_codes[:B].clone()

            if bucket > B:
                self._frame_graph_input[:bucket].zero_()
            self._frame_graph_input[:B].copy_(hidden_batch)
            self._frame_graphs[graph_key].replay()
            if self._profile_enabled:
                self._profile_stats["frame_graph_replay"] += 1
            return self._frame_graph_stop[:B].clone(), self._frame_graph_codes[:B].clone()
        except Exception:
            logger.exception("MOSS-TTS Local native frame graph failed; falling back to eager local decode")
            self._frame_graphs.clear()
            if self._profile_enabled:
                self._profile_stats["frame_graph_fallback"] += 1
            self.local_transformer._ensure_kv_cache(B, hidden_batch.device, dtype)
            return self._decode_frame_eager(hidden_batch, infos)

    def decode_omni_frame_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            self._batch_state = None
            return model_outputs
        hidden = model_outputs
        info_dicts: list[dict[str, Any]] = (
            kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        )
        query_start_loc = kwargs.get("omni_query_start_loc")
        request_token_spans = kwargs.get("request_token_spans")
        sample_row_by_req = kwargs.get("omni_sample_row_by_req")
        span_lens: list[int] | None = kwargs.get("omni_span_lens")
        if span_lens is not None:
            span_lens = [int(x) for x in span_lens[: len(info_dicts)]]
        elif isinstance(request_token_spans, (list, tuple)) and len(request_token_spans) >= len(info_dicts):
            span_lens = [int(end) - int(start) for start, end in request_token_spans[: len(info_dicts)]]
        elif isinstance(query_start_loc, torch.Tensor) and query_start_loc.numel() >= len(info_dicts) + 1:
            qsl_cpu = query_start_loc[: len(info_dicts) + 1].detach().to("cpu").tolist()
            span_lens = [int(qsl_cpu[i + 1]) - int(qsl_cpu[i]) for i in range(len(info_dicts))]
            request_token_spans = [(int(qsl_cpu[i]), int(qsl_cpu[i + 1])) for i in range(len(info_dicts))]

        logits_index = kwargs.get("logits_index")
        sample_hidden = hidden
        logits_rows: list[int] | None = None
        logits_row_count = 0
        if isinstance(logits_index, torch.Tensor) and logits_index.numel() > 0:
            logits_index_gpu = logits_index.to(device=hidden.device, dtype=torch.long).reshape(-1)
            num_sample_rows = int(logits_index_gpu.numel())
            logits_row_count = num_sample_rows
            if int(hidden.shape[0]) == num_sample_rows:
                sample_hidden = hidden
            elif num_sample_rows <= int(hidden.shape[0]):
                sample_hidden = hidden.index_select(0, logits_index_gpu)
            else:
                sample_hidden = hidden[:num_sample_rows]
            if not isinstance(sample_row_by_req, (list, tuple)):
                logits_rows = logits_index.detach().to("cpu", dtype=torch.long).reshape(-1).tolist()
        elif isinstance(logits_index, int):
            logits_rows = [int(logits_index)]
            logits_row_count = 1
            sample_hidden = hidden[int(logits_index) : int(logits_index) + 1]
        elif isinstance(request_token_spans, (list, tuple)) and len(request_token_spans) >= len(info_dicts):
            sample_rows = [int(end) - 1 for _start, end in request_token_spans[: len(info_dicts)]]
            if sample_rows and all(0 <= row < int(hidden.shape[0]) for row in sample_rows):
                sample_rows_t = torch.tensor(sample_rows, device=hidden.device, dtype=torch.long)
                sample_hidden = hidden.index_select(0, sample_rows_t)
                logits_rows = sample_rows
        elif (
            span_lens is not None
            and len(span_lens) >= len(info_dicts)
            and all(span == 1 for span in span_lens[: len(info_dicts)])
            and len(info_dicts) > 0
            and int(hidden.shape[0]) > len(info_dicts)
        ):
            sample_hidden = hidden[: len(info_dicts)]
        self._record_shape_profile(
            hidden_rows=int(hidden.shape[0]) if hidden.dim() >= 1 else 1,
            sample_rows=int(sample_hidden.shape[0]) if sample_hidden.dim() >= 1 else 1,
            logits_rows=logits_row_count,
            info_rows=len(info_dicts),
        )

        for info in info_dicts:
            if isinstance(info, dict) and not isinstance(info.get("audio_state"), dict):
                state = {"step": 0, "is_stopping": False}
                _copy_sampling_overrides(state, info)
                default_seed = os.environ.get("MOSS_TTS_LOCAL_DEFAULT_SEED")
                sampling_generator = _make_sampling_generator(info.get("seed", default_seed), hidden.device)
                if sampling_generator is not None:
                    state["sampling_generator"] = sampling_generator
                info["audio_state"] = state

        pending_req_ids, pending_codes, pending_codec_streaming = self._consume_pending_stop_result(info_dicts)
        delta_req_ids: list[str] = list(pending_req_ids)
        delta_codes: list[torch.Tensor] = list(pending_codes)
        delta_codec_streaming: list[bool] = list(pending_codec_streaming)
        next_text_tensor: torch.Tensor | None = None
        batch_state_by_sample_row: list[dict[str, Any] | None] = []
        profile_local_ms = 0.0
        profile_start = self._profile_mark() if getattr(self, "_profile_enabled", False) else 0.0

        if sample_hidden.numel() > 0 and info_dicts:
            num_rows = int(sample_hidden.shape[0])
            rowmap_start = time.perf_counter() if getattr(self, "_profile_enabled", False) else 0.0
            next_text_tensor = torch.full(
                (num_rows,),
                self.audio_end_token_id,
                dtype=torch.long,
                device=hidden.device,
            )
            batch_state_by_sample_row = [None for _ in range(num_rows)]
            logits_row_to_sample_row = (
                {int(row): pos for pos, row in enumerate(logits_rows)} if logits_rows is not None else None
            )
            pure_decode_rows = (
                num_rows == len(info_dicts) and span_lens is not None and all(span == 1 for span in span_lens)
            )

            def sample_row_for_request(req_index: int) -> int | None:
                if isinstance(sample_row_by_req, (list, tuple)) and req_index < len(sample_row_by_req):
                    row = int(sample_row_by_req[req_index])
                    return row if row >= 0 else None
                if pure_decode_rows:
                    return req_index
                if isinstance(request_token_spans, (list, tuple)) and req_index < len(request_token_spans):
                    start, end = request_token_spans[req_index]
                    start_i = int(start)
                    end_i = int(end)
                    if end_i <= start_i:
                        return None
                    hidden_row = end_i - 1
                    if logits_row_to_sample_row is not None:
                        return logits_row_to_sample_row.get(hidden_row)
                    if 0 <= hidden_row < num_rows:
                        return hidden_row
                if num_rows == len(info_dicts) == 1:
                    return req_index
                return None

            decode_items: list[tuple[int, int]] = []
            for i, info in enumerate(info_dicts):
                if not isinstance(info, dict):
                    continue
                state = info.get("audio_state", {}) or {}
                row_idx = sample_row_for_request(i)
                if row_idx is None or row_idx < 0 or row_idx >= num_rows:
                    continue
                batch_state_by_sample_row[row_idx] = state
                if state.get("is_stopping"):
                    state["next_text"] = self.audio_end_token_id
                    next_text_tensor[row_idx] = self.audio_end_token_id
                    continue
                if span_lens is not None and i < len(span_lens) and span_lens[i] != 1:
                    state["next_text"] = self.audio_assistant_slot_token_id
                    next_text_tensor[row_idx] = self.audio_assistant_slot_token_id
                    continue
                decode_items.append((i, row_idx))
            if getattr(self, "_profile_enabled", False):
                self._profile_stats["rowmap_ms"] += (time.perf_counter() - rowmap_start) * 1000

            use_batched = (
                len(decode_items) > 1
                and os.environ.get("MOSS_TTS_LOCAL_DISABLE_LOCAL_BATCH") != "1"
                and all(isinstance(info_dicts[j], dict) for j, _ in decode_items)
            )

            if use_batched:
                decode_rows = [row_idx for _, row_idx in decode_items]
                n_active = len(decode_items)
                asm_buffers = self._ensure_output_asm_buffers(batch_size=n_active, device=hidden.device)
                rows_cpu = asm_buffers["rows_cpu"][:n_active]
                for bi, row_idx in enumerate(decode_rows):
                    rows_cpu[bi] = int(row_idx)
                decode_rows_t = asm_buffers["rows_gpu"][:n_active]
                decode_rows_t.copy_(rows_cpu, non_blocking=hidden.device.type == "cuda")
                hidden_batch = (
                    sample_hidden
                    if decode_rows == list(range(len(decode_rows))) and len(decode_rows) == int(sample_hidden.shape[0])
                    else sample_hidden.index_select(0, decode_rows_t)
                )

                local_start = self._profile_mark() if getattr(self, "_profile_enabled", False) else 0.0
                stop_choices, all_codes = self._decode_frame_batched(
                    hidden_batch, [info_dicts[j] for j, _ in decode_items]
                )
                if getattr(self, "_profile_enabled", False):
                    elapsed = self._profile_elapsed_ms(local_start)
                    self._profile_stats["local_ms"] += elapsed
                    profile_local_ms += elapsed
                    self._profile_stats["n_decode"] += 1
                    self._profile_stats["batch_sizes"].append(n_active)

                asm_start = time.perf_counter() if getattr(self, "_profile_enabled", False) else 0.0
                pool = self._get_request_state_pool()
                pool.ensure_frame_state_device(hidden.device)
                slot_cpu = asm_buffers["slot_cpu"][:n_active]
                active_states: list[dict[str, Any]] = []
                active_slot_ids: list[int] = []
                all_slots_valid = (
                    pool.steps_gpu is not None
                    and pool.min_frames_gpu is not None
                    and pool.max_frames_gpu is not None
                    and pool.is_stopping_gpu is not None
                    and hidden.device.type == "cuda"
                )
                for bi, (i, _) in enumerate(decode_items):
                    state = info_dicts[i].get("audio_state", {}) or {}
                    active_states.append(state)
                    try:
                        slot_id = int(info_dicts[i].get("_kv_slot_id"))
                    except (TypeError, ValueError):
                        slot_id = -1
                    active_slot_ids.append(slot_id)
                    slot_cpu[bi] = slot_id
                    if 0 <= slot_id < pool.max_slots:
                        if pool.is_stopping_py[slot_id]:
                            state["is_stopping"] = True
                    else:
                        all_slots_valid = False

                next_for_active = asm_buffers["next_gpu"][:n_active]
                next_for_active.fill_(self.audio_assistant_slot_token_id)
                slot_ids_t = asm_buffers["slot_gpu"][:n_active]
                slot_ids_t.copy_(slot_cpu, non_blocking=hidden.device.type == "cuda")
                steps_t = asm_buffers["steps_gpu"][:n_active]
                min_frames_t = asm_buffers["min_gpu"][:n_active]
                max_frames_t = asm_buffers["max_gpu"][:n_active]
                should_stop_t = asm_buffers["stop_gpu"][:n_active]
                tmp_stop_t = asm_buffers["tmp_stop_gpu"][:n_active]
                step_values: list[int] = []
                min_values: list[int] = []
                max_values: list[int] = []
                model_stop_poll: list[bool] = []
                max_stop_list: list[bool] = []
                active_req_ids: list[str] = []
                stop_interval = int(getattr(self, "_stop_check_interval", 1))
                for state, slot_id in zip(active_states, active_slot_ids, strict=False):
                    if 0 <= slot_id < pool.max_slots:
                        step_num = pool.steps_py[slot_id]
                        min_frames = pool.min_frames_py[slot_id]
                        max_frames = pool.max_frames_py[slot_id]
                    else:
                        step_num = int(state.get("step", 0))
                        min_frames = max(int(state.get("min_new_frames", 3)), self._default_min_frames)
                        max_frames = int(state.get("max_new_frames", 150))
                    step_values.append(step_num)
                    min_values.append(min_frames)
                    max_values.append(max_frames)
                    max_due = step_num >= max_frames
                    max_stop_list.append(max_due)
                    model_stop_poll.append(
                        (not max_due)
                        and step_num >= min_frames
                        and (stop_interval <= 1 or ((step_num - min_frames) % stop_interval) == 0)
                    )
                for i, _ in decode_items:
                    info = info_dicts[i]
                    active_req_ids.append(str(info.get("request_id") or info.get("global_request_id") or ""))
                stop_possible = any(model_stop_poll) or any(max_stop_list)

                if all_slots_valid:
                    torch.index_select(pool.steps_gpu, 0, slot_ids_t, out=steps_t)
                    torch.index_select(pool.min_frames_gpu, 0, slot_ids_t, out=min_frames_t)
                    torch.index_select(pool.max_frames_gpu, 0, slot_ids_t, out=max_frames_t)
                else:
                    steps_cpu = asm_buffers["steps_cpu"][:n_active]
                    min_cpu = asm_buffers["min_cpu"][:n_active]
                    max_cpu = asm_buffers["max_cpu"][:n_active]
                    for bi, (step_num, min_frames, max_frames) in enumerate(
                        zip(step_values, min_values, max_values, strict=False)
                    ):
                        steps_cpu[bi] = int(step_num)
                        min_cpu[bi] = int(min_frames)
                        max_cpu[bi] = int(max_frames)
                    steps_t.copy_(steps_cpu, non_blocking=hidden.device.type == "cuda")
                    min_frames_t.copy_(min_cpu, non_blocking=hidden.device.type == "cuda")
                    max_frames_t.copy_(max_cpu, non_blocking=hidden.device.type == "cuda")

                should_stop_list: list[bool] = [False for _ in range(n_active)]
                delay_model_stop = (
                    bool(getattr(self, "_delay_stop_sync_enabled", False))
                    and hidden.device.type == "cuda"
                    and stop_interval == 1
                    and any(model_stop_poll)
                )
                delay_delta_routing = bool(getattr(self, "_delay_delta_routing_enabled", False))
                stop_sync_start: float | None = None
                if stop_possible:
                    stop_cpu = asm_buffers["stop_cpu"][:n_active]
                    if delay_model_stop:
                        torch.eq(stop_choices.reshape(-1)[:n_active], 1, out=should_stop_t)
                        torch.ge(steps_t, min_frames_t, out=tmp_stop_t)
                        should_stop_t.logical_and_(tmp_stop_t)
                        stop_cpu.copy_(should_stop_t, non_blocking=True)
                        stop_copy_event = asm_buffers.get("stop_copy_event")
                        if isinstance(stop_copy_event, torch.cuda.Event):
                            stop_copy_event.record(torch.cuda.current_stream(hidden.device))
                            pending_stop_result: dict[str, Any] = {
                                "event": stop_copy_event,
                                "stop_cpu": stop_cpu,
                                "slot_ids": list(active_slot_ids),
                                "req_ids": list(active_req_ids),
                                "force_stop": list(max_stop_list),
                                "steps": list(step_values),
                            }
                            if delay_delta_routing:
                                pending_stop_result["codes"] = all_codes.detach().clone()
                                pending_stop_result["codec_streaming"] = [
                                    _stream_flag_from_info(info_dicts[i]) for i, _ in decode_items
                                ]
                            self._pending_stop_result = pending_stop_result
                        should_stop_list = [bool(x) for x in max_stop_list]
                        next_for_active.fill_(self.audio_assistant_slot_token_id)
                        should_stop_t.zero_()
                        if any(max_stop_list):
                            poll_cpu = asm_buffers["poll_cpu"][:n_active]
                            for bi, should_stop in enumerate(max_stop_list):
                                poll_cpu[bi] = bool(should_stop)
                            max_stop_t = asm_buffers["poll_gpu"][:n_active]
                            max_stop_t.copy_(poll_cpu, non_blocking=True)
                            should_stop_t.copy_(max_stop_t, non_blocking=True)
                            next_for_active.masked_fill_(max_stop_t, self.audio_end_token_id)
                    elif any(model_stop_poll):
                        stop_sync_start = time.perf_counter()
                        model_stop_mask = None
                        if stop_interval > 1:
                            poll_cpu = asm_buffers["poll_cpu"][:n_active]
                            for bi, allowed in enumerate(model_stop_poll):
                                poll_cpu[bi] = bool(allowed)
                            model_stop_mask = asm_buffers["poll_gpu"][:n_active]
                            model_stop_mask.copy_(poll_cpu, non_blocking=hidden.device.type == "cuda")
                        should_stop_t, next_for_active = self._compute_stop_and_next_tokens(
                            stop_choices=stop_choices.reshape(-1)[:n_active],
                            steps=steps_t,
                            min_frames=min_frames_t,
                            max_frames=max_frames_t,
                            audio_assistant_slot_token_id=self.audio_assistant_slot_token_id,
                            audio_end_token_id=self.audio_end_token_id,
                            should_stop_out=should_stop_t,
                            tmp_out=tmp_stop_t,
                            next_tokens_out=next_for_active,
                            model_stop_mask=model_stop_mask,
                        )
                        stop_cpu.copy_(should_stop_t, non_blocking=hidden.device.type == "cuda")
                        stop_copy_event = asm_buffers.get("stop_copy_event")
                        if isinstance(stop_copy_event, torch.cuda.Event):
                            stop_copy_event.record(torch.cuda.current_stream(hidden.device))
                    else:
                        for bi, should_stop in enumerate(max_stop_list):
                            stop_cpu[bi] = bool(should_stop)
                        should_stop_t.copy_(stop_cpu, non_blocking=hidden.device.type == "cuda")
                        if any(max_stop_list):
                            next_for_active.masked_fill_(should_stop_t, self.audio_end_token_id)
                    if (
                        (not delay_model_stop)
                        and any(model_stop_poll)
                        and isinstance(asm_buffers.get("stop_copy_event"), torch.cuda.Event)
                    ):
                        asm_buffers["stop_copy_event"].synchronize()
                    if stop_sync_start is not None and getattr(self, "_profile_enabled", False):
                        self._profile_stats["stop_sync_ms"] += (time.perf_counter() - stop_sync_start) * 1000
                    if not delay_model_stop:
                        should_stop_list = [bool(x) for x in stop_cpu.tolist()]
                else:
                    should_stop_t.zero_()
                next_text_tensor.index_copy_(0, decode_rows_t, next_for_active)

                if all_slots_valid:
                    pool.advance_frame_state_gpu(
                        slot_ids_t,
                        should_stop_t,
                        should_stop_list,
                        slot_ids_cpu=active_slot_ids,
                    )
                else:
                    pool.advance_frame_state(active_slot_ids, should_stop_list)

                update_start = time.perf_counter() if getattr(self, "_profile_enabled", False) else 0.0
                for bi, (i, _row_idx) in enumerate(decode_items):
                    info = info_dicts[i]
                    state = active_states[bi]
                    new_codes = all_codes[bi]
                    should_stop = should_stop_list[bi]
                    slot_id = active_slot_ids[bi]
                    if 0 <= slot_id < pool.max_slots:
                        step_num = pool.steps_py[slot_id]
                        max_frames = pool.max_frames_py[slot_id]
                    else:
                        step_num = int(state.get("step", 0))
                        max_frames = int(state.get("max_new_frames", 150))
                    if should_stop:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        state["stop_reason"] = "max_new_frames" if step_num >= max_frames else "model_stop"
                        info["audio_codes"] = {"current": new_codes}
                        self._set_slot_current_codes(slot_id, new_codes)
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = pool.steps_py[slot_id] if 0 <= slot_id < pool.max_slots else step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        self._set_slot_current_codes(slot_id, new_codes)
                        rid = str(info.get("request_id") or info.get("global_request_id") or "")
                        if rid or len(info_dicts) == 1:
                            if not (delay_model_stop and delay_delta_routing):
                                delta_req_ids.append(rid)
                                delta_codes.append(all_codes[bi : bi + 1])
                                delta_codec_streaming.append(_stream_flag_from_info(info))
                if getattr(self, "_profile_enabled", False):
                    self._profile_stats["update_ms"] += (time.perf_counter() - update_start) * 1000
                    self._profile_stats["output_asm_ms"] += (time.perf_counter() - asm_start) * 1000
            else:
                for i, row_idx in decode_items:
                    info = info_dicts[i]
                    if not isinstance(info, dict):
                        continue
                    state = info.get("audio_state", {}) or {}
                    if state.get("is_stopping"):
                        state["next_text"] = self.audio_end_token_id
                        next_text_tensor[row_idx] = self.audio_end_token_id
                        continue
                    local_start = self._profile_mark() if getattr(self, "_profile_enabled", False) else 0.0
                    stop_choices, all_codes = self._decode_frame_batched(sample_hidden[row_idx : row_idx + 1], [info])
                    if getattr(self, "_profile_enabled", False):
                        elapsed = self._profile_elapsed_ms(local_start)
                        self._profile_stats["local_ms"] += elapsed
                        profile_local_ms += elapsed
                        self._profile_stats["n_decode"] += 1
                        self._profile_stats["batch_sizes"].append(1)
                    new_codes = all_codes[0]
                    step_num = int(state.get("step", 0))
                    min_frames = max(int(state.get("min_new_frames", 3)), self._default_min_frames)
                    max_frames = int(state.get("max_new_frames", 150))
                    should_stop = (
                        int(stop_choices[0].item()) == 1 and step_num >= min_frames
                    ) or step_num >= max_frames
                    if should_stop:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        state["stop_reason"] = "max_new_frames" if step_num >= max_frames else "model_stop"
                        info["audio_codes"] = {"current": new_codes}
                        try:
                            self._set_slot_current_codes(int(info.get("_kv_slot_id")), new_codes)
                        except (TypeError, ValueError):
                            pass
                        next_text_tensor[row_idx] = self.audio_end_token_id
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        try:
                            self._set_slot_current_codes(int(info.get("_kv_slot_id")), new_codes)
                        except (TypeError, ValueError):
                            pass
                        next_text_tensor[row_idx] = self.audio_assistant_slot_token_id
                        rid = str(info.get("request_id") or info.get("global_request_id") or "")
                        if rid or len(info_dicts) == 1:
                            delta_req_ids.append(rid)
                            delta_codes.append(new_codes.unsqueeze(0))
                            delta_codec_streaming.append(_stream_flag_from_info(info))

        output_start = time.perf_counter() if getattr(self, "_profile_enabled", False) else 0.0
        self._batch_state = [
            (state if isinstance(state, dict) else {"next_text": self.audio_end_token_id})
            for state in batch_state_by_sample_row
        ] or [(info.get("audio_state", {}) if isinstance(info, dict) else {}) for info in info_dicts]
        if not delta_codes:
            output = OmniOutput(
                text_hidden_states=sample_hidden,
                multimodal_outputs={"meta": {"next_text": next_text_tensor}} if next_text_tensor is not None else {},
            )
            if getattr(self, "_profile_enabled", False):
                self._profile_stats["output_ms"] += (time.perf_counter() - output_start) * 1000
                self._profile_stats["make_output_ms"] += max(
                    0.0, self._profile_elapsed_ms(profile_start) - profile_local_ms
                )
                self._maybe_log_profile()
            return output

        step_counter = max(
            (int((info.get("audio_state", {}) or {}).get("step", 0)) for info in info_dicts if isinstance(info, dict)),
            default=0,
        )
        if len(info_dicts) == 1 and len(delta_codes) == 1:
            codec_streaming = _stream_flag_from_info(info_dicts[0]) if isinstance(info_dicts[0], dict) else True
            output = OmniOutput(
                text_hidden_states=sample_hidden,
                multimodal_outputs={
                    "codes": {"audio": delta_codes[0]},
                    "meta": {
                        "raw_rows": True,
                        "step": step_counter,
                        "next_text": next_text_tensor,
                        "codec_streaming": codec_streaming,
                    },
                },
            )
            if getattr(self, "_profile_enabled", False):
                self._profile_stats["output_ms"] += (time.perf_counter() - output_start) * 1000
                self._profile_stats["make_output_ms"] += max(
                    0.0, self._profile_elapsed_ms(profile_start) - profile_local_ms
                )
                self._maybe_log_profile()
            return output
        output = OmniOutput(
            text_hidden_states=sample_hidden,
            multimodal_outputs={
                "codes": {"audio": delta_codes},
                "meta": {
                    "sparse_audio": True,
                    "req_id": delta_req_ids,
                    "raw_rows": True,
                    "step": [
                        int((info.get("audio_state", {}) or {}).get("step", 0))
                        for info in info_dicts
                        if isinstance(info, dict)
                    ][: len(delta_codes)],
                    "next_text": next_text_tensor,
                    "codec_streaming": delta_codec_streaming,
                },
            },
        )
        if getattr(self, "_profile_enabled", False):
            self._profile_stats["output_ms"] += (time.perf_counter() - output_start) * 1000
            self._profile_stats["make_output_ms"] += max(
                0.0, self._profile_elapsed_ms(profile_start) - profile_local_ms
            )
            self._maybe_log_profile()
        return output

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if _env_enabled("MOSS_TTS_LOCAL_NATIVE_STATEPOOL_OUTPUT"):
            return self.decode_omni_frame_output(model_outputs, **kwargs)
        return self._make_omni_output_legacy(model_outputs, **kwargs)

    def _make_omni_output_legacy(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        profile_start = self._profile_mark() if self._profile_enabled else 0.0
        profile_local_ms = 0.0
        if isinstance(model_outputs, OmniOutput):
            self._batch_state = None
            return model_outputs
        hidden = model_outputs
        info_dicts: list[dict[str, Any]] = (
            kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        )
        logits_index = kwargs.get("logits_index")
        sample_hidden = hidden
        logits_rows: list[int] | None = None
        logits_row_count = 0
        if isinstance(logits_index, torch.Tensor) and logits_index.numel() > 0:
            logits_index_gpu = logits_index.to(device=hidden.device, dtype=torch.long).reshape(-1)
            logits_row_count = int(logits_index_gpu.numel())
            sample_hidden = (
                hidden
                if int(hidden.shape[0]) == int(logits_index_gpu.numel())
                else hidden.index_select(0, logits_index_gpu)
            )
        elif isinstance(logits_index, int):
            logits_rows = [int(logits_index)]
            logits_row_count = 1
            sample_hidden = hidden[logits_index : logits_index + 1]

        query_start_loc = kwargs.get("omni_query_start_loc")
        request_token_spans = kwargs.get("request_token_spans")
        sample_row_by_req = kwargs.get("omni_sample_row_by_req")
        span_lens: list[int] | None = kwargs.get("omni_span_lens")
        qsl_cpu: list[int] | None = None
        if span_lens is not None:
            span_lens = [int(x) for x in span_lens[: len(info_dicts)]]
        elif isinstance(request_token_spans, (list, tuple)) and len(request_token_spans) >= len(info_dicts):
            span_lens = [int(end) - int(start) for start, end in request_token_spans[: len(info_dicts)]]
        elif isinstance(query_start_loc, torch.Tensor) and query_start_loc.numel() >= len(info_dicts) + 1:
            qsl_cpu = query_start_loc[: len(info_dicts) + 1].detach().to("cpu").tolist()
            span_lens = [int(qsl_cpu[i + 1]) - int(qsl_cpu[i]) for i in range(len(info_dicts))]
            request_token_spans = [(int(qsl_cpu[i]), int(qsl_cpu[i + 1])) for i in range(len(info_dicts))]
        self._record_shape_profile(
            hidden_rows=int(hidden.shape[0]) if hidden.dim() >= 1 else 1,
            sample_rows=int(sample_hidden.shape[0]) if sample_hidden.dim() >= 1 else 1,
            logits_rows=logits_row_count,
            info_rows=len(info_dicts),
        )
        if _debug_state_enabled():
            if isinstance(logits_index, torch.Tensor):
                logits_index_dbg = logits_index.detach().to("cpu").reshape(-1).tolist()
            else:
                logits_index_dbg = logits_index
            logger.info(
                "[moss-local-state] make_output enter hidden=%s sample=%s infos=%d "
                "qsl=%s logits_index=%s spans=%s reqs=%s",
                tuple(hidden.shape),
                tuple(sample_hidden.shape),
                len(info_dicts),
                qsl_cpu,
                logits_index_dbg,
                span_lens,
                [_request_label(info) for info in info_dicts if isinstance(info, dict)],
            )
        for info in info_dicts:
            if isinstance(info, dict) and not isinstance(info.get("audio_state"), dict):
                state = {"step": 0, "is_stopping": False}
                _copy_sampling_overrides(state, info)
                default_seed = os.environ.get("MOSS_TTS_LOCAL_DEFAULT_SEED")
                sampling_generator = _make_sampling_generator(info.get("seed", default_seed), hidden.device)
                if sampling_generator is not None:
                    state["sampling_generator"] = sampling_generator
                info["audio_state"] = state
        self._consume_pending_stop_result(info_dicts)

        audio_codes_list: list[torch.Tensor | None] = [None for _ in info_dicts]
        batch_state_by_sample_row: list[dict[str, Any] | None] = []
        next_text_tensor: torch.Tensor | None = None
        delta_req_ids: list[str] = []
        delta_codes: list[torch.Tensor] = []
        delta_steps: list[int] = []
        if hidden.numel() > 0 and info_dicts:
            num_rows = int(sample_hidden.shape[0])
            rowmap_start = time.perf_counter() if self._profile_enabled else 0.0
            batch_state_by_sample_row = [None for _ in range(num_rows)]
            next_text_tensor = torch.full(
                (num_rows,),
                self.audio_end_token_id,
                dtype=torch.long,
                device=hidden.device,
            )
            logits_row_to_sample_row = (
                {int(row): pos for pos, row in enumerate(logits_rows)} if logits_rows is not None else None
            )
            decode_items: list[tuple[int, int]] = []
            pure_decode_rows = (
                num_rows == len(info_dicts) and span_lens is not None and all(span == 1 for span in span_lens)
            )

            def sample_row_for_request(req_index: int) -> int | None:
                if isinstance(sample_row_by_req, (list, tuple)) and req_index < len(sample_row_by_req):
                    row = int(sample_row_by_req[req_index])
                    return row if row >= 0 else None
                if pure_decode_rows:
                    return req_index
                if isinstance(request_token_spans, (list, tuple)) and req_index < len(request_token_spans):
                    start, end = request_token_spans[req_index]
                    start_i = int(start)
                    end_i = int(end)
                    if end_i <= start_i:
                        return None
                    hidden_row = end_i - 1
                    if logits_row_to_sample_row is not None:
                        return logits_row_to_sample_row.get(hidden_row)
                    if 0 <= hidden_row < num_rows:
                        return hidden_row
                    return None
                if num_rows == len(info_dicts) == 1:
                    return req_index
                return None

            for i, info in enumerate(info_dicts):
                if not isinstance(info, dict):
                    continue
                state = info.get("audio_state", {}) or {}
                if state.get("is_stopping"):
                    row_idx = sample_row_for_request(i)
                    if row_idx is not None and 0 <= row_idx < num_rows:
                        state["next_text"] = self.audio_end_token_id
                        batch_state_by_sample_row[row_idx] = state
                        next_text_tensor[row_idx] = self.audio_end_token_id
                    continue

                row_idx = sample_row_for_request(i)
                if row_idx is None or row_idx < 0 or row_idx >= num_rows:
                    continue
                if span_lens is not None and span_lens[i] != 1:
                    if _debug_state_enabled():
                        digest, norm, head = _debug_tensor_digest(sample_hidden[row_idx])
                        logger.info(
                            "[moss-local-state] prefill-hidden req=%s span=%s row=%d digest=%s norm=%.6f head=%s",
                            _request_label(info),
                            span_lens[i],
                            row_idx,
                            digest,
                            norm,
                            head,
                        )
                    state["next_text"] = self.audio_assistant_slot_token_id
                    batch_state_by_sample_row[row_idx] = state
                    next_text_tensor[row_idx] = self.audio_assistant_slot_token_id
                    continue
                batch_state_by_sample_row[row_idx] = state
                decode_items.append((i, row_idx))

            if self._profile_enabled:
                self._profile_stats["rowmap_ms"] += (time.perf_counter() - rowmap_start) * 1000
            use_batched = len(decode_items) > 0
            if _debug_state_enabled():
                row_debug = []
                for i, info in enumerate(info_dicts):
                    if not isinstance(info, dict):
                        continue
                    state = info.get("audio_state", {}) or {}
                    row_idx = sample_row_for_request(i)
                    row_debug.append(
                        {
                            "req": _request_label(info),
                            "info_idx": i,
                            "row": row_idx,
                            "span": span_lens[i] if span_lens is not None and i < len(span_lens) else None,
                            "step": int(state.get("step", 0)),
                            "next": state.get("next_text"),
                        }
                    )
                logger.info(
                    "[moss-local-state] make_output active=%s use_batched=%s rows=%d infos=%d rowmap=%s",
                    [i for i, _ in decode_items],
                    use_batched,
                    num_rows,
                    len(info_dicts),
                    row_debug,
                )

            if use_batched and len(decode_items) > 1 and os.environ.get("MOSS_TTS_LOCAL_DISABLE_LOCAL_BATCH") != "1":
                if _debug_state_enabled():
                    for i, row_idx in decode_items:
                        info = info_dicts[i]
                        state = info.get("audio_state", {}) if isinstance(info, dict) else {}
                        digest, norm, head = _debug_tensor_digest(sample_hidden[row_idx])
                        logger.info(
                            "[moss-local-state] hidden req=%s step=%d row=%d digest=%s norm=%.6f head=%s",
                            _request_label(info) if isinstance(info, dict) else "",
                            int(state.get("step", 0)) if isinstance(state, dict) else 0,
                            row_idx,
                            digest,
                            norm,
                            head,
                        )
                decode_rows = [row_idx for _, row_idx in decode_items]
                if len(decode_rows) == int(sample_hidden.shape[0]) and decode_rows == list(range(len(decode_rows))):
                    hidden_batch = sample_hidden
                    decode_rows_t = None
                else:
                    decode_rows_t = torch.tensor(decode_rows, device=sample_hidden.device, dtype=torch.long)
                    hidden_batch = sample_hidden.index_select(0, decode_rows_t)
                local_start = self._profile_mark() if self._profile_enabled else 0.0
                stop_choices, all_codes = self._decode_frame_batched(
                    hidden_batch,
                    [info_dicts[i] for i, _ in decode_items],
                )
                if self._profile_enabled:
                    elapsed = self._profile_elapsed_ms(local_start)
                    self._profile_stats["local_ms"] += elapsed
                    profile_local_ms += elapsed
                    self._profile_stats["n_decode"] += 1
                    self._profile_stats["batch_sizes"].append(len(decode_items))
                batch_min_frames = self._default_min_frames
                step_values = [
                    int((info_dicts[i].get("audio_state", {}) or {}).get("step", 0)) for i, _ in decode_items
                ]
                min_values = [
                    max(int((info_dicts[i].get("audio_state", {}) or {}).get("min_new_frames", 3)), batch_min_frames)
                    for i, _ in decode_items
                ]
                max_values = [
                    int((info_dicts[i].get("audio_state", {}) or {}).get("max_new_frames", 150))
                    for i, _ in decode_items
                ]
                step_nums = torch.tensor(step_values, dtype=torch.long, device=stop_choices.device)
                min_frames_t = torch.tensor(min_values, dtype=torch.long, device=stop_choices.device)
                max_frames_t = torch.tensor(max_values, dtype=torch.long, device=stop_choices.device)
                model_stop_tensor = (stop_choices == 1) & (step_nums >= min_frames_t)
                max_stop_tensor = step_nums >= max_frames_t
                delay_model_stop = (
                    bool(getattr(self, "_delay_stop_sync_enabled", False))
                    and stop_choices.device.type == "cuda"
                    and torch.cuda.is_available()
                    and any(step >= min_frame for step, min_frame in zip(step_values, min_values))
                )
                if delay_model_stop:
                    should_stop_tensor = max_stop_tensor
                    next_for_decode = torch.where(
                        max_stop_tensor,
                        torch.full_like(step_nums, self.audio_end_token_id),
                        torch.full_like(step_nums, self.audio_assistant_slot_token_id),
                    )
                    try:
                        stop_cpu = torch.empty(len(decode_items), dtype=torch.bool, pin_memory=True)
                    except RuntimeError:
                        stop_cpu = torch.empty(len(decode_items), dtype=torch.bool)
                    stop_cpu.copy_(model_stop_tensor.detach(), non_blocking=True)
                    stop_event = torch.cuda.Event()
                    stop_event.record(torch.cuda.current_stream(stop_choices.device))
                    self._pending_stop_result = {
                        "event": stop_event,
                        "stop_cpu": stop_cpu,
                        "req_ids": [
                            str(info_dicts[i].get("request_id") or info_dicts[i].get("global_request_id") or "")
                            for i, _ in decode_items
                        ],
                    }
                    should_stop_list = [bool(step >= max_frame) for step, max_frame in zip(step_values, max_values)]
                else:
                    should_stop_tensor = model_stop_tensor | max_stop_tensor
                    next_for_decode = torch.where(
                        should_stop_tensor,
                        torch.full_like(step_nums, self.audio_end_token_id),
                        torch.full_like(step_nums, self.audio_assistant_slot_token_id),
                    )
                    should_stop_list = should_stop_tensor.detach().cpu().tolist()
                if next_text_tensor is not None:
                    if decode_rows_t is None:
                        next_text_tensor[: len(decode_items)].copy_(next_for_decode)
                    else:
                        next_text_tensor.index_copy_(0, decode_rows_t, next_for_decode)
                update_start = time.perf_counter() if self._profile_enabled else 0.0
                if _debug_state_enabled():
                    logger.info(
                        "[moss-local-state] decode batched reqs=%s stops=%s should_stop=%s",
                        [_request_label(info_dicts[i]) for i, _ in decode_items],
                        stop_choices.detach().cpu().tolist(),
                        should_stop_list,
                    )
                for bi, (i, _) in enumerate(decode_items):
                    info = info_dicts[i]
                    if not isinstance(info, dict):
                        continue
                    state = info.get("audio_state", {}) or {}
                    if state.get("is_stopping"):
                        state["next_text"] = self.audio_end_token_id
                        continue
                    new_codes = all_codes[bi]
                    debug_hash = _update_debug_code_hash(state, new_codes) if _debug_state_enabled() else 0
                    step_num = int(step_values[bi])
                    max_frames = int(max_values[bi])
                    should_stop = bool(should_stop_list[bi])
                    if should_stop:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        state["stop_reason"] = "max_new_frames" if step_num >= max_frames else "model_stop"
                        info["audio_codes"] = {"current": new_codes}
                        try:
                            self._set_slot_current_codes(int(info.get("_kv_slot_id")), new_codes)
                        except (TypeError, ValueError):
                            pass
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        try:
                            self._set_slot_current_codes(int(info.get("_kv_slot_id")), new_codes)
                        except (TypeError, ValueError):
                            pass
                        audio_codes_list[i] = new_codes.unsqueeze(0)
                        rid = str(info.get("request_id") or info.get("global_request_id") or "")
                        if rid:
                            delta_req_ids.append(rid)
                            delta_codes.append(new_codes.unsqueeze(0))
                            delta_steps.append(step_num)
                    if _debug_state_enabled():
                        logger.info(
                            "[moss-local-state] update req=%s step=%d stop=%s next=%s code0=%d",
                            _request_label(info),
                            int(state.get("step", 0)),
                            bool(should_stop),
                            int(state.get("next_text", -1)),
                            int(new_codes[0].item()),
                        )
                        if should_stop:
                            logger.info(
                                "[moss-local-state] final req=%s frames=%d reason=%s hash=%016x "
                                "row=%d span=%s next=%s stop_choice=%d",
                                _request_label(info),
                                step_num + 1,
                                state.get("stop_reason"),
                                debug_hash,
                                int(decode_items[bi][1]),
                                span_lens[i] if span_lens is not None and i < len(span_lens) else None,
                                state.get("next_text"),
                                int(stop_choices[bi].item()),
                            )
                if self._profile_enabled:
                    self._profile_stats["update_ms"] += (time.perf_counter() - update_start) * 1000
            elif use_batched:
                for i, row_idx in decode_items:
                    info = info_dicts[i]
                    if not isinstance(info, dict):
                        continue
                    state = info.get("audio_state", {}) or {}
                    if state.get("is_stopping"):
                        state["next_text"] = self.audio_end_token_id
                        continue
                    if _debug_state_enabled():
                        digest, norm, head = _debug_tensor_digest(sample_hidden[row_idx])
                        logger.info(
                            "[moss-local-state] hidden req=%s step=%d row=%d digest=%s norm=%.6f head=%s",
                            _request_label(info),
                            int(state.get("step", 0)),
                            row_idx,
                            digest,
                            norm,
                            head,
                        )
                    local_start = self._profile_mark() if self._profile_enabled else 0.0
                    stop_choices, all_codes = self._decode_frame_batched(sample_hidden[row_idx : row_idx + 1], [info])
                    if self._profile_enabled:
                        elapsed = self._profile_elapsed_ms(local_start)
                        self._profile_stats["local_ms"] += elapsed
                        profile_local_ms += elapsed
                        self._profile_stats["n_decode"] += 1
                        self._profile_stats["batch_sizes"].append(1)
                    new_codes = all_codes[0]
                    debug_hash = _update_debug_code_hash(state, new_codes) if _debug_state_enabled() else 0
                    step_num = int(state.get("step", 0))
                    min_frames = max(int(state.get("min_new_frames", 3)), self._default_min_frames)
                    max_frames = int(state.get("max_new_frames", 150))
                    should_stop = (
                        int(stop_choices[0].item()) == 1 and step_num >= min_frames
                    ) or step_num >= max_frames
                    if should_stop:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        state["stop_reason"] = "max_new_frames" if step_num >= max_frames else "model_stop"
                        info["audio_codes"] = {"current": new_codes}
                        try:
                            self._set_slot_current_codes(int(info.get("_kv_slot_id")), new_codes)
                        except (TypeError, ValueError):
                            pass
                        if next_text_tensor is not None:
                            next_text_tensor[row_idx] = self.audio_end_token_id
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        try:
                            self._set_slot_current_codes(int(info.get("_kv_slot_id")), new_codes)
                        except (TypeError, ValueError):
                            pass
                        audio_codes_list[i] = new_codes.unsqueeze(0)
                        if next_text_tensor is not None:
                            next_text_tensor[row_idx] = self.audio_assistant_slot_token_id
                        rid = str(info.get("request_id") or info.get("global_request_id") or "")
                        if rid:
                            delta_req_ids.append(rid)
                            delta_codes.append(new_codes.unsqueeze(0))
                            delta_steps.append(step_num)
                    if _debug_state_enabled():
                        logger.info(
                            "[moss-local-state] update req=%s step=%d stop=%s next=%s code0=%d",
                            _request_label(info),
                            int(state.get("step", 0)),
                            bool(should_stop),
                            int(state.get("next_text", -1)),
                            int(new_codes[0].item()),
                        )
                        if should_stop:
                            logger.info(
                                "[moss-local-state] final req=%s frames=%d reason=%s hash=%016x "
                                "row=%d span=%s next=%s stop_choice=%d",
                                _request_label(info),
                                step_num + 1,
                                state.get("stop_reason"),
                                debug_hash,
                                int(row_idx),
                                span_lens[i] if span_lens is not None and i < len(span_lens) else None,
                                state.get("next_text"),
                                int(stop_choices[0].item()),
                            )

        output_start = time.perf_counter() if self._profile_enabled else 0.0
        self._batch_state = [
            (state if isinstance(state, dict) else {"next_text": self.audio_end_token_id})
            for state in batch_state_by_sample_row
        ]
        if next_text_tensor is None:
            next_text_values = [
                int(state.get("next_text", self.audio_end_token_id))
                if isinstance(state, dict)
                else self.audio_end_token_id
                for state in batch_state_by_sample_row
            ]
            next_text_tensor = torch.tensor(next_text_values, dtype=torch.long, device=hidden.device)
        active_codes = [c for c in audio_codes_list if c is not None]
        if not active_codes:
            output = OmniOutput(text_hidden_states=hidden, multimodal_outputs={"meta": {"next_text": next_text_tensor}})
            if self._profile_enabled:
                self._profile_stats["output_ms"] += (time.perf_counter() - output_start) * 1000
                self._profile_stats["make_output_ms"] += max(
                    0.0, self._profile_elapsed_ms(profile_start) - profile_local_ms
                )
                self._maybe_log_profile()
            return output
        step_counter = max(
            (int((info.get("audio_state", {}) or {}).get("step", 0)) for info in info_dicts if isinstance(info, dict)),
            default=0,
        )
        if len(info_dicts) == 1 and len(active_codes) == 1:
            output = OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "codes": {"audio": active_codes[0]},
                    "meta": {"raw_rows": True, "step": step_counter, "next_text": next_text_tensor},
                },
            )
            if self._profile_enabled:
                self._profile_stats["output_ms"] += (time.perf_counter() - output_start) * 1000
                self._profile_stats["make_output_ms"] += max(
                    0.0, self._profile_elapsed_ms(profile_start) - profile_local_ms
                )
                self._maybe_log_profile()
            return output
        if not delta_codes:
            output = OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
            if self._profile_enabled:
                self._profile_stats["output_ms"] += (time.perf_counter() - output_start) * 1000
                self._profile_stats["make_output_ms"] += max(
                    0.0, self._profile_elapsed_ms(profile_start) - profile_local_ms
                )
                self._maybe_log_profile()
            return output
        output = OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "codes": {"audio": delta_codes},
                "meta": {
                    "sparse_audio": True,
                    "req_id": delta_req_ids,
                    "raw_rows": True,
                    "step": delta_steps,
                    "next_text": next_text_tensor,
                },
            },
        )
        if self._profile_enabled:
            self._profile_stats["make_output_ms"] += max(
                0.0, self._profile_elapsed_ms(profile_start) - profile_local_ms
            )
            self._maybe_log_profile()
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._audio_embedding_weight_cache = None
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters())

        # vLLM Qwen3Model expects stacked qkv_proj and gate_up_proj
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        for original_name, tensor in weights:
            name = original_name
            if name.startswith("transformer."):
                name = "model." + name[len("transformer.") :]

            # Skip text/audio lm heads from checkpoint (we have our own)
            if name.startswith("text_lm_head.") or name.startswith("audio_lm_heads."):
                continue

            # Handle audio_embeddings separately
            if name.startswith("audio_embeddings.") and name.endswith(".weight"):
                param = params_dict.get(name)
                if param is not None:
                    rows = min(int(tensor.shape[0]), int(param.shape[0]))
                    with torch.no_grad():
                        param[:rows].copy_(tensor[:rows].to(device=param.device, dtype=param.dtype))
                    loaded.add(name)
                continue

            # local_transformer and local_text_lm_head
            if name.startswith("local_transformer.") or name.startswith("local_text_lm_head."):
                param = params_dict.get(name)
                if param is not None:
                    default_weight_loader(param, tensor)
                    loaded.add(name)
                continue

            # Backbone weights: try stacked mapping first
            is_stacked = False
            for packed_name, source_name, shard_id in stacked_params_mapping:
                if source_name in name:
                    packed_key = name.replace(source_name, packed_name)
                    param = params_dict.get(packed_key)
                    if param is not None:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, tensor, shard_id)
                        loaded.add(packed_key)
                        is_stacked = True
                    break

            if not is_stacked:
                param = params_dict.get(name)
                if param is not None:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
                    loaded.add(name)

        return loaded


__all__ = ["MossTTSLocalNativeModel"]
