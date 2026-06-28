# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Local Transformer v1.5 AR stage."""

from __future__ import annotations

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
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.moss_tts_local.configuration_moss_tts_local import (
    MossTTSLocalConfig,
)
from vllm_omni.model_executor.models.moss_tts_local.hf_compatible_qwen3 import (
    MossTTSLocalQwen3Backbone,
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


def _env_disabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


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


_SAMPLING_OVERRIDE_KEYS = (
    "text_temperature",
    "text_top_p",
    "text_top_k",
    "audio_temperature",
    "audio_top_p",
    "audio_top_k",
)


def _copy_sampling_overrides(state: dict[str, Any], info: dict[str, Any]) -> None:
    for key in _SAMPLING_OVERRIDE_KEYS:
        value = _first_scalar(info.get(key))
        if value is not None:
            state[key] = value


_BUCKET_SIZES = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def _next_bucket(value: int) -> int:
    for b in _BUCKET_SIZES:
        if b >= value:
            return b
    return value


def _debug_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, torch.Generator):
        return f"torch.Generator(initial_seed={value.initial_seed()})"
    if isinstance(value, dict):
        return {str(k): _debug_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class MossTTSLocalRequestStatePool:
    """Small request/slot state pool for the next model forward."""

    def __init__(self, max_slots: int) -> None:
        self.max_slots = max(1, int(max_slots))
        self._slot_to_request: list[str | None] = [None] * self.max_slots
        self._request_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self.max_slots))
        self._scheduled_slot_ids: list[int] = []
        self._scheduled_token_counts: list[int] = []
        self.steps_cpu = torch.zeros(self.max_slots, dtype=torch.long)
        self.min_frames_cpu = torch.full((self.max_slots,), 3, dtype=torch.long)
        self.max_frames_cpu = torch.full((self.max_slots,), 150, dtype=torch.long)
        self.is_stopping_cpu = torch.zeros(self.max_slots, dtype=torch.bool)
        # Keep Python mirrors for the per-frame hot path. Reading many
        # 0-d torch CPU tensors from Python costs milliseconds per decode step.
        self.steps_py = [0 for _ in range(self.max_slots)]
        self.min_frames_py = [3 for _ in range(self.max_slots)]
        self.max_frames_py = [150 for _ in range(self.max_slots)]
        self.is_stopping_py = [False for _ in range(self.max_slots)]
        self._frame_state_device: torch.device | None = None
        self.steps_gpu: torch.Tensor | None = None
        self.min_frames_gpu: torch.Tensor | None = None
        self.max_frames_gpu: torch.Tensor | None = None
        self.is_stopping_gpu: torch.Tensor | None = None

    def allocate_slot(self, request_id: str) -> int:
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(f"No free KV slots (max_num_seqs={self.max_slots})")
        slot_id = self._free_slots.pop(0)
        self._slot_to_request[slot_id] = request_id
        self._request_to_slot[request_id] = slot_id
        return slot_id

    def slot_for_request(self, request_id: str) -> int | None:
        return self._request_to_slot.get(request_id)

    def release_slot(self, request_id: str) -> None:
        slot_id = self._request_to_slot.pop(request_id, None)
        if slot_id is not None:
            self._slot_to_request[slot_id] = None
            self.steps_cpu[slot_id] = 0
            self.min_frames_cpu[slot_id] = 3
            self.max_frames_cpu[slot_id] = 150
            self.is_stopping_cpu[slot_id] = False
            self.steps_py[slot_id] = 0
            self.min_frames_py[slot_id] = 3
            self.max_frames_py[slot_id] = 150
            self.is_stopping_py[slot_id] = False
            if (
                self.steps_gpu is not None
                and self.min_frames_gpu is not None
                and self.max_frames_gpu is not None
                and self.is_stopping_gpu is not None
            ):
                self.steps_gpu[slot_id] = 0
                self.min_frames_gpu[slot_id] = 3
                self.max_frames_gpu[slot_id] = 150
                self.is_stopping_gpu[slot_id] = False
            self._free_slots.append(slot_id)

    def schedule_forward_slot(self, slot_id: int, token_count: int) -> None:
        self._scheduled_slot_ids.append(int(slot_id))
        self._scheduled_token_counts.append(int(token_count))

    def consume_scheduled_forward_slots(self) -> tuple[list[int], list[int]]:
        slot_ids = self._scheduled_slot_ids
        token_counts = self._scheduled_token_counts
        self._scheduled_slot_ids = []
        self._scheduled_token_counts = []
        return list(slot_ids), list(token_counts)

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
        if (
            self.steps_gpu is not None
            and self.min_frames_gpu is not None
            and self.max_frames_gpu is not None
            and self.is_stopping_gpu is not None
        ):
            self.steps_gpu[slot_id] = 0
            self.min_frames_gpu[slot_id] = int(min_new_frames)
            self.max_frames_gpu[slot_id] = int(max_new_frames)
            self.is_stopping_gpu[slot_id] = False

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

    def advance_frame_state(self, slot_ids: list[int], should_stop: list[bool]) -> None:
        for slot_id, stop in zip(slot_ids, should_stop):
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
        for slot_id, stop in zip(slot_ids_cpu, should_stop_cpu):
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

    def slot_request_id(self, slot_id: int) -> str | None:
        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self.max_slots:
            return None
        return self._slot_to_request[slot_id]


class MossTTSLocalForGeneration(nn.Module):
    """Stage-0 AR model: Qwen3 backbone plus per-frame local transformer."""

    input_modalities = "audio"
    have_multimodal_outputs: bool = True
    has_preprocess: bool = True
    has_postprocess: bool = True
    requires_raw_input_tokens: bool = True
    supports_omni_query_start_loc: bool = True
    disable_outer_cudagraph: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: MossTTSLocalConfig = vllm_config.model_config.hf_config
        lang_cfg = self.config.qwen3_config
        gpt2_cfg = self.config.gpt2_config

        self.hidden_size = int(lang_cfg.hidden_size)
        self.vocab_size = int(lang_cfg.vocab_size)
        self.n_vq = int(self.config.n_vq)
        self.audio_vocab_size = int(self.config.audio_vocab_size)
        self.audio_pad_code = int(self.config.audio_pad_code)
        self.audio_assistant_slot_token_id = int(self.config.audio_assistant_slot_token_id)
        self.audio_end_token_id = int(self.config.audio_end_token_id)

        del prefix
        self.model = MossTTSLocalQwen3Backbone(lang_cfg)

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
        self._batch_next_text: torch.Tensor | None = None
        self._audio_embedding_indices = torch.arange(self.n_vq, dtype=torch.long)
        self._audio_embedding_weight_cache: torch.Tensor | None = None
        self._graph_warmup_done: bool = False
        self._max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        self._request_state_pool = MossTTSLocalRequestStatePool(self._max_num_seqs)
        self._batch_stats: dict[str, Any] = {
            "batched": 0,
            "fallback_no_cache": 0,
            "fallback_overflow": 0,
            "fallback_prefill": 0,
            "single": 0,
            "log_interval": 100,
            "batch_sizes": [],
            "t_backbone_ms": 0.0,
            "t_local_tx_ms": 0.0,
            "t_output_asm_ms": 0.0,
            "t_stop_sync_ms": 0.0,
            "n_steps": 0,
            "completed_frames": [],
        }
        self._profile_detail_enabled = _env_enabled("MOSS_TTS_LOCAL_PROFILE_DETAIL")
        self._frame_graph_profile_enabled = _env_enabled("MOSS_TTS_LOCAL_FRAME_GRAPH_PROFILE")
        self._batch_stats.update(
            {
                "t_asm_active_state_ms": 0.0,
                "t_asm_state_gather_ms": 0.0,
                "t_asm_stop_compute_ms": 0.0,
                "t_asm_stop_wait_ms": 0.0,
                "t_asm_next_text_ms": 0.0,
                "t_asm_pool_advance_ms": 0.0,
                "t_asm_state_loop_ms": 0.0,
                "t_asm_tail_ms": 0.0,
                "frame_graph_replay": 0,
                "frame_graph_capture": 0,
                "frame_graph_buffer_reset": 0,
                "frame_graph_rand_ms": 0.0,
                "frame_graph_input_ms": 0.0,
                "frame_graph_replay_ms": 0.0,
                "frame_graph_capture_ms": 0.0,
                "frame_graph_clone_ms": 0.0,
                "frame_graph_keys": {},
            }
        )
        self._local_graph: torch.cuda.CUDAGraph | None = None
        self._local_graph_sampling: torch.cuda.CUDAGraph | None = None
        self._local_graph_input: torch.Tensor | None = None
        self._local_graph_stop: torch.Tensor | None = None
        self._local_graph_codes: torch.Tensor | None = None
        self._local_graph_rand: torch.Tensor | None = None
        self._batch_frame_graphs: dict[
            tuple[int, tuple[float, float, int, float, float, int]], torch.cuda.CUDAGraph
        ] = {}
        self._batch_frame_graph_input: torch.Tensor | None = None
        self._batch_frame_graph_rand: torch.Tensor | None = None
        self._batch_frame_graph_stop: torch.Tensor | None = None
        self._batch_frame_graph_codes: torch.Tensor | None = None
        self._batch_frame_graph_max_batch: int = 0
        self._batch_frame_graph_bucket_sizes = self._parse_frame_graph_bucket_sizes()
        self._output_asm_buffers: dict[str, Any] = {}
        self._local_graph_enabled = not _env_disabled("MOSS_TTS_LOCAL_DISABLE_FRAME_GRAPH")
        try:
            self._stop_check_interval = max(int(os.environ.get("MOSS_TTS_LOCAL_STOP_CHECK_INTERVAL", "1") or 1), 1)
        except ValueError:
            self._stop_check_interval = 1
        self._delay_stop_sync_enabled = not _env_disabled("MOSS_TTS_LOCAL_DISABLE_DELAY_STOP_SYNC")
        self._delay_delta_routing_enabled = _env_enabled("MOSS_TTS_LOCAL_DELAY_DELTA_ROUTING")
        self._pending_stop_result: dict[str, Any] | None = None
        if os.environ.get("MOSS_TTS_LOCAL_COMPILE_LOCAL_TX") == "1":
            self.local_transformer.step = torch.compile(self.local_transformer.step, dynamic=True)

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

    @staticmethod
    def _parse_frame_graph_bucket_sizes() -> tuple[int, ...]:
        raw = os.environ.get("MOSS_TTS_LOCAL_FRAME_GRAPH_BATCH_BUCKETS", "").strip()
        if not raw:
            return (8, 16, 32, 64)
        if raw.lower() in ("auto", "default"):
            return (8, 16, 32, 64)
        buckets: list[int] = []
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
                buckets.append(value)
        return tuple(sorted(set(buckets)))

    def _frame_graph_capture_batch_size(self, batch_size: int) -> int:
        for bucket in self._batch_frame_graph_bucket_sizes:
            if int(batch_size) <= bucket:
                return int(bucket)
        return int(batch_size)

    def _schedule_forward_slot(self, slot_id: int, token_count: int) -> None:
        self._get_request_state_pool().schedule_forward_slot(slot_id, token_count)

    def _consume_scheduled_forward_slots(self) -> tuple[list[int], list[int]]:
        return self._get_request_state_pool().consume_scheduled_forward_slots()

    def should_disable_outer_cudagraph(self) -> bool:
        return not _env_enabled("MOSS_TTS_LOCAL_ENABLE_OUTER_CUDAGRAPH")

    def _allocate_slot(self, request_id: str) -> int:
        return self._get_request_state_pool().allocate_slot(request_id)

    def _release_slot(self, request_id: str) -> None:
        self._get_request_state_pool().release_slot(request_id)

    def on_requests_finished(self, finished_req_ids: list[str]) -> None:
        for req_id in finished_req_ids:
            self._release_slot(req_id)

    def _ensure_output_asm_buffers(self, *, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
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
            buffers["rows_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["slot_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["steps_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["min_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["max_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            buffers["stop_gpu"] = torch.empty(bucket, device=device, dtype=torch.bool)
            buffers["tmp_stop_gpu"] = torch.empty(bucket, device=device, dtype=torch.bool)
            buffers["poll_cpu"] = cpu_empty(bucket, dtype=torch.bool)
            buffers["poll_gpu"] = torch.empty(bucket, device=device, dtype=torch.bool)
            buffers["next_gpu"] = torch.empty(bucket, device=device, dtype=torch.long)
            if device.type == "cuda":
                buffers["stop_copy_event"] = torch.cuda.Event()
        return buffers

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
            state = info.get("audio_state", {}) or {}
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
            if info_state is None:
                pool.mark_stopping(slot_id)
                continue
            _info, state = info_state
            pool.mark_stopping(slot_id)
            state["is_stopping"] = True
            state["next_text"] = self.audio_end_token_id
            state["stop_reason"] = "model_stop"
            state["step"] = stop_step
            self._batch_stats["completed_frames"].append(stop_step)
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

    def _log_batch_stats(self) -> None:
        s = self._batch_stats
        total = (
            s["batched"]
            + s.get("fallback_no_cache", 0)
            + s.get("fallback_overflow", 0)
            + s["fallback_prefill"]
            + s["single"]
        )
        if total > 0 and total % s["log_interval"] == 0:
            hit = s["batched"]
            multi = hit + s.get("fallback_no_cache", 0) + s.get("fallback_overflow", 0) + s["fallback_prefill"]
            rate = hit / max(1, multi) * 100
            n = s["n_steps"] or 1
            avg_backbone = s["t_backbone_ms"] / n
            avg_local = s["t_local_tx_ms"] / n
            avg_output_asm = s["t_output_asm_ms"] / n
            avg_stop_sync = s.get("t_stop_sync_ms", 0.0) / n
            bsizes = s["batch_sizes"][-20:]
            avg_bs = sum(bsizes) / max(1, len(bsizes))
            frames = s["completed_frames"]
            frames_info = ""
            if frames:
                frames_sorted = sorted(frames)
                p50 = frames_sorted[len(frames_sorted) // 2]
                p95 = frames_sorted[min(len(frames_sorted) - 1, int(len(frames_sorted) * 0.95))]
                frames_info = f" | frames: n={len(frames)} p50={p50} p95={p95} avg={sum(frames) / len(frames):.0f}"
            logger.info(
                "Profile[%d steps]: hit=%.0f%% avg_bs=%.1f | backbone=%.2fms local_tx=%.2fms "
                "output_asm=%.2fms stop_sync=%.2fms | "
                "batched=%d prefill=%d single=%d%s",
                n,
                rate,
                avg_bs,
                avg_backbone,
                avg_local,
                avg_output_asm,
                avg_stop_sync,
                hit,
                s["fallback_prefill"],
                s["single"],
                frames_info,
            )
            if bool(getattr(self, "_profile_detail_enabled", False)):
                logger.info(
                    "ProfileDetail[%d steps]: active_state=%.2fms state_gather=%.2fms "
                    "stop_compute=%.2fms stop_wait=%.2fms next_text=%.2fms "
                    "pool_advance=%.2fms state_loop=%.2fms tail=%.2fms",
                    n,
                    s.get("t_asm_active_state_ms", 0.0) / n,
                    s.get("t_asm_state_gather_ms", 0.0) / n,
                    s.get("t_asm_stop_compute_ms", 0.0) / n,
                    s.get("t_asm_stop_wait_ms", 0.0) / n,
                    s.get("t_asm_next_text_ms", 0.0) / n,
                    s.get("t_asm_pool_advance_ms", 0.0) / n,
                    s.get("t_asm_state_loop_ms", 0.0) / n,
                    s.get("t_asm_tail_ms", 0.0) / n,
                )
            if bool(getattr(self, "_frame_graph_profile_enabled", False)):
                replay = int(s.get("frame_graph_replay", 0))
                capture = int(s.get("frame_graph_capture", 0))
                graph_calls = max(1, replay + capture)
                key_counts = s.get("frame_graph_keys", {})
                top_keys = sorted(key_counts.items(), key=lambda item: item[1], reverse=True)[:5]
                top_key_text = ",".join(f"{key}:{count}" for key, count in top_keys)
                logger.info(
                    "FrameGraphProfile[%d steps]: replay=%d capture=%d resets=%d "
                    "input=%.3fms rand=%.3fms replay_wall=%.3fms capture_wall=%.3fms "
                    "clone=%.3fms keys=%s",
                    n,
                    replay,
                    capture,
                    int(s.get("frame_graph_buffer_reset", 0)),
                    s.get("frame_graph_input_ms", 0.0) / graph_calls,
                    s.get("frame_graph_rand_ms", 0.0) / graph_calls,
                    s.get("frame_graph_replay_ms", 0.0) / max(1, replay),
                    s.get("frame_graph_capture_ms", 0.0) / max(1, capture),
                    s.get("frame_graph_clone_ms", 0.0) / graph_calls,
                    top_key_text or "-",
                )

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
        slot_ids, token_counts = self._consume_scheduled_forward_slots()
        if len(slot_ids) > 1:
            if all(c == 1 for c in token_counts):
                if (
                    hasattr(self.model, "layers")
                    and self.model.layers[0].self_attn.static_key_cache is None
                    and hasattr(self.model, "init_static_kv_cache")
                ):
                    max_pos = int(getattr(self.config.qwen3_config, "max_position_embeddings", 4096))
                    device = input_ids.device if input_ids is not None else positions.device
                    dtype = self.model.embed_tokens.weight.dtype
                    self.model.init_static_kv_cache(
                        max_len=min(max_pos, 512),
                        max_batch=self._max_num_seqs,
                        device=device,
                        dtype=dtype,
                    )
                can_batch = (
                    hasattr(self.model, "layers") and self.model.layers[0].self_attn.static_key_cache is not None
                )
                fallback_reason = ""
                if not can_batch:
                    fallback_reason = "no_cache"
                if can_batch:
                    cap = int(self.model.layers[0].self_attn.static_key_cache.shape[1])
                    for sid in slot_ids:
                        if self.model.layers[0].self_attn._slot_cache_lens[sid] + 1 > cap:
                            can_batch = False
                            fallback_reason = "overflow"
                            break
                if can_batch:
                    self._batch_stats["batched"] += 1
                    self._batch_stats["batch_sizes"].append(len(slot_ids))
                    t0 = time.perf_counter()
                    result = self.model(
                        input_ids=input_ids,
                        positions=positions,
                        inputs_embeds=inputs_embeds,
                        slot_ids=slot_ids,
                    )
                    self._batch_stats["t_backbone_ms"] += (time.perf_counter() - t0) * 1000
                    self._batch_stats["n_steps"] += 1
                    self._log_batch_stats()
                    return result
                else:
                    self._batch_stats[f"fallback_{fallback_reason}"] = (
                        self._batch_stats.get(f"fallback_{fallback_reason}", 0) + 1
                    )
                    self._log_batch_stats()
            else:
                self._batch_stats["fallback_prefill"] += 1
                self._log_batch_stats()
            outputs = []
            offset = 0
            for slot_id, count in zip(slot_ids, token_counts):
                seg_embeds = inputs_embeds[offset : offset + count] if inputs_embeds is not None else None
                seg_ids = input_ids[offset : offset + count] if input_ids is not None else None
                seg_pos = positions[offset : offset + count]
                out = self.model(
                    input_ids=seg_ids,
                    positions=seg_pos,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=seg_embeds,
                    slot_id=slot_id,
                )
                outputs.append(out)
                offset += count
            return torch.cat(outputs, dim=0)
        self._batch_stats["single"] += 1
        self._log_batch_stats()
        slot_id = slot_ids[0] if slot_ids else 0
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            slot_id=slot_id,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        """Return one-hot text logits: continue slot or audio_end."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None or hidden_states.numel() == 0:
            return None
        num_rows = hidden_states.shape[0]
        next_text = self._batch_next_text
        if isinstance(next_text, torch.Tensor) and next_text.numel() > 0:
            row_tokens = next_text.to(device=hidden_states.device, dtype=torch.long).reshape(-1)
            num_rows = int(row_tokens.numel())
            row_tokens = row_tokens.clamp_(0, self.vocab_size - 1)
            logits = hidden_states.new_full((num_rows, self.vocab_size), float("-inf"))
            logits.scatter_(1, row_tokens.unsqueeze(1), 0.0)
            return logits
        states = self._batch_state or []
        if not states:
            logits = hidden_states.new_full((num_rows, self.vocab_size), float("-inf"))
            logits[:, self.audio_assistant_slot_token_id] = 0.0
            return logits
        rows_per_state = max(1, num_rows // max(1, len(states)))
        row_tokens = torch.full(
            (num_rows,),
            self.audio_end_token_id,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for i, state in enumerate(states):
            r0 = i * rows_per_state
            r1 = min(r0 + rows_per_state, num_rows)
            if r0 >= r1:
                continue
            token_id = int(state.get("next_text", self.audio_assistant_slot_token_id))
            if not 0 <= token_id < self.vocab_size:
                token_id = self.audio_end_token_id
            row_tokens[r0:r1] = token_id
        logits = hidden_states.new_full((num_rows, self.vocab_size), float("-inf"))
        logits.scatter_(1, row_tokens.unsqueeze(1), 0.0)
        if os.environ.get("MOSS_TTS_LOCAL_DEBUG_BATCH") and len(states) > 1:
            tokens = [int(s.get("next_text", self.audio_assistant_slot_token_id)) for s in states]
            steps = [int(s.get("step", 0)) for s in states]
            stopping = [bool(s.get("is_stopping")) for s in states]
            logger.info(
                "compute_logits batch=%d tokens=%s steps=%s stopping=%s",
                len(states),
                tokens,
                steps,
                stopping,
            )
        return logits

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
            reset_fn = getattr(self.model, "reset_slot", None) or getattr(self.model, "reset_cache", None)
            if callable(reset_fn):
                try:
                    reset_fn(slot_id)
                except TypeError:
                    reset_fn()
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
            max_positions = int(getattr(self.config.qwen3_config, "max_position_embeddings", 4096))
            needed_len = span_len + (max_new_frames if max_new_frames > 0 else 512) + 1
            bucket_len = min(max_positions, _next_bucket(needed_len))
            self.model.init_static_kv_cache(
                max_len=bucket_len,
                max_batch=self._max_num_seqs,
                device=device,
                dtype=self.model.embed_tokens.weight.dtype,
            )
            if not self._graph_warmup_done and os.environ.get("MOSS_TTS_LOCAL_WARMUP_GRAPHS") == "1":
                self._graph_warmup_done = True
                warmup_buckets = [b for b in _BUCKET_SIZES if b <= max_positions and b != bucket_len]
                if warmup_buckets:
                    self.model.warmup_decode_graphs(
                        warmup_buckets,
                        device=device,
                        dtype=self.model.embed_tokens.weight.dtype,
                        max_batch=self._max_num_seqs,
                    )
                    self.model.init_static_kv_cache(
                        max_len=bucket_len,
                        max_batch=self._max_num_seqs,
                        device=device,
                        dtype=self.model.embed_tokens.weight.dtype,
                    )
            sampling_generator = _make_sampling_generator(info_dict.get("seed"), device)
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
            _copy_sampling_overrides(state, info_dict)
            if sampling_generator is not None:
                state["sampling_generator"] = sampling_generator
            self._maybe_dump_request_debug(
                input_ids=input_ids,
                info_dict=info_dict,
                state=state,
            )
            self._schedule_forward_slot(slot_id, span_len)
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

        prev_codes = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(prev_codes, torch.Tensor) or prev_codes.numel() != self.n_vq:
            prev_codes = torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
        embeds = self._build_input_embeds(input_ids.reshape(-1), prev_codes.to(device=device).unsqueeze(0))
        self._schedule_forward_slot(slot_id, 1)
        return input_ids, embeds, {"_kv_slot_id": slot_id}

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Batch decode preprocess for single-frame MOSS Local decode rows.

        This is equivalent to calling ``preprocess(..., span_len=1)`` for each
        request, but it batches the text/audio embedding work into one GPU path.
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
                prev_codes = torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
            prev_codes_list.append(prev_codes.to(device=device, dtype=torch.long).reshape(-1)[: self.n_vq])

        if prev_codes_list:
            prev_codes_batch = torch.stack(prev_codes_list, dim=0)
        else:
            prev_codes_batch = torch.empty((0, self.n_vq), dtype=torch.long, device=device)
        embeds = self._build_input_embeds(text_ids, prev_codes_batch)
        for slot_id in slot_ids:
            self._schedule_forward_slot(slot_id, 1)
        updates = [{"_kv_slot_id": slot_id} for slot_id in slot_ids]
        return text_ids, embeds, updates

    def preprocess_decode_batch_fast(
        self,
        *,
        input_ids: torch.Tensor,
        req_ids: list[str],
        prev_codes: list[torch.Tensor | None],
        slot_ids: list[int | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Fast single-frame decode preprocess for the MOSS Local HF path.

        The generic path attaches a full req_info dictionary for every decode
        row before batching.  MOSS decode only needs the sampled text token, the
        previous RVQ codes and the state-pool slot, so this keeps the hot path
        out of the per-request metadata loop.
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
                codes = torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
            prev_codes_list.append(codes.to(device=device, dtype=torch.long).reshape(-1)[: self.n_vq])

        if prev_codes_list:
            prev_codes_batch = torch.stack(prev_codes_list, dim=0)
        else:
            prev_codes_batch = torch.empty((0, self.n_vq), dtype=torch.long, device=device)
        embeds = self._build_input_embeds(text_ids, prev_codes_batch)
        for slot_id in resolved_slot_ids:
            self._schedule_forward_slot(slot_id, 1)
        updates = [{"_kv_slot_id": slot_id} for slot_id in resolved_slot_ids]
        return text_ids, embeds, updates

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    def _maybe_dump_request_debug(
        self,
        *,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        dump_dir = os.environ.get("MOSS_TTS_LOCAL_DUMP_CODES_DIR")
        if not dump_dir:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"stage0_request_{os.getpid()}_{id(state)}.pt")
            torch.save(
                {
                    "input_ids": input_ids.detach().to("cpu"),
                    "prompt_rows": _debug_value(info_dict.get("prompt_rows")),
                    "max_new_frames": state.get("max_new_frames"),
                    "seed": _debug_value(info_dict.get("seed")),
                    "sampling_overrides": {key: state[key] for key in _SAMPLING_OVERRIDE_KEYS if key in state},
                    "info": {key: _debug_value(value) for key, value in info_dict.items() if key != "audio_state"},
                },
                path,
            )
        except Exception:
            logger.exception("Failed to dump MOSS-TTS Local debug request")

    @staticmethod
    def _sample_graph_safe(
        logits: torch.Tensor, top_k: int, top_p: float, temperature: float, rand_val: torch.Tensor
    ) -> torch.Tensor:
        scores = logits / max(temperature, 1e-6)
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
        return MossTTSLocalForGeneration._sample_graph_safe(
            logits,
            int(top_k),
            float(top_p),
            float(temperature),
            rand_val,
        )

    def _sampling_decode_frame_kernel(self) -> None:
        hidden = self._local_graph_input
        local_hidden = self.local_transformer.step(hidden.to(dtype=self.audio_embeddings[0].weight.dtype), 0)
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()
        stop = self._sample_graph_safe(text_logits, 50, 1.0, 1.0, self._local_graph_rand[0:1])
        self._local_graph_stop.copy_(stop)
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
            logits = F.linear(current, head_weight).float()
            code = self._sample_graph_safe(logits, 50, 0.95, 1.0, self._local_graph_rand[1 + channel : 2 + channel])
            self._local_graph_codes[channel] = code.reshape(())
            if channel + 1 < self.n_vq:
                current = self.local_transformer.step(
                    F.embedding(code, head_weight).to(dtype=current.dtype), channel + 1
                )

    def _greedy_decode_frame_kernel(self) -> None:
        hidden = self._local_graph_input
        local_hidden = self.local_transformer.step(hidden.to(dtype=self.audio_embeddings[0].weight.dtype), 0)
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()
        self._local_graph_stop.copy_(torch.argmax(text_logits, dim=-1))
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
            logits = F.linear(current, head_weight).float()
            code = torch.argmax(logits, dim=-1)
            self._local_graph_codes[channel] = code.reshape(())
            if channel + 1 < self.n_vq:
                current = self.local_transformer.step(
                    F.embedding(code, head_weight).to(dtype=current.dtype), channel + 1
                )

    def _ensure_local_graph(self, device: torch.device) -> None:
        if self._local_graph is not None:
            return
        dtype = self.audio_embeddings[0].weight.dtype
        self._local_graph_input = torch.zeros(1, self.hidden_size, device=device, dtype=dtype)
        self._local_graph_stop = torch.zeros(1, dtype=torch.long, device=device)
        self._local_graph_codes = torch.zeros(self.n_vq, dtype=torch.long, device=device)
        self._local_graph_rand = torch.zeros(1 + self.n_vq, device=device, dtype=torch.float32)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._greedy_decode_frame_kernel()
        torch.cuda.current_stream().wait_stream(s)
        self._local_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._local_graph):
            self._greedy_decode_frame_kernel()

    def _ensure_local_sampling_graph(self, device: torch.device) -> None:
        if self._local_graph_sampling is not None:
            return
        self._ensure_local_graph(device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._sampling_decode_frame_kernel()
        torch.cuda.current_stream().wait_stream(s)
        self._local_graph_sampling = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._local_graph_sampling):
            self._sampling_decode_frame_kernel()

    def _decode_frame_greedy_graph(self, hidden: torch.Tensor) -> tuple[int, torch.Tensor]:
        self._ensure_local_graph(hidden.device)
        self._local_graph_input.copy_(hidden.to(dtype=self._local_graph_input.dtype))
        self._local_graph.replay()
        return int(self._local_graph_stop.item()), self._local_graph_codes.clone()

    def _decode_frame_sampling_graph(
        self, hidden: torch.Tensor, generator: torch.Generator | None
    ) -> tuple[int, torch.Tensor]:
        self._ensure_local_sampling_graph(hidden.device)
        self._local_graph_input.copy_(hidden.to(dtype=self._local_graph_input.dtype))
        rand_vals = torch.rand(1 + self.n_vq, device=hidden.device, generator=generator)
        self._local_graph_rand.copy_(rand_vals)
        self._local_graph_sampling.replay()
        return int(self._local_graph_stop.item()), self._local_graph_codes.clone()

    def _ensure_batch_frame_graph_buffers(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            self._batch_frame_graph_input is not None
            and self._batch_frame_graph_input.device == device
            and self._batch_frame_graph_input.dtype == dtype
            and self._batch_frame_graph_max_batch >= batch_size
        ):
            return
        cap = max(batch_size, self._batch_frame_graph_max_batch, 8)
        self._batch_frame_graph_input = torch.zeros(cap, self.hidden_size, device=device, dtype=dtype)
        self._batch_frame_graph_rand = torch.zeros(1 + self.n_vq, cap, device=device, dtype=torch.float32)
        self._batch_frame_graph_stop = torch.zeros(cap, device=device, dtype=torch.long)
        self._batch_frame_graph_codes = torch.zeros(cap, self.n_vq, device=device, dtype=torch.long)
        self._batch_frame_graph_max_batch = cap
        self._batch_frame_graphs.clear()
        self._batch_stats["frame_graph_buffer_reset"] = self._batch_stats.get("frame_graph_buffer_reset", 0) + 1

    def _batch_frame_graph_kernel(
        self,
        batch_size: int,
        params: tuple[float, float, int, float, float, int],
    ) -> None:
        assert self._batch_frame_graph_input is not None
        assert self._batch_frame_graph_rand is not None
        assert self._batch_frame_graph_stop is not None
        assert self._batch_frame_graph_codes is not None
        text_temp, text_top_p, text_top_k, audio_temp, audio_top_p, audio_top_k = params
        hidden = self._batch_frame_graph_input[:batch_size]
        rand = self._batch_frame_graph_rand[:, :batch_size]
        local_hidden = self.local_transformer._step_eager(
            hidden.to(dtype=self.audio_embeddings[0].weight.dtype),
            0,
        )
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()
        stop = self._sample_graph_safe_param(
            text_logits,
            temperature=text_temp,
            top_p=text_top_p,
            top_k=text_top_k,
            rand_val=rand[0],
        )
        self._batch_frame_graph_stop[:batch_size].copy_(stop)
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
            self._batch_frame_graph_codes[:batch_size, channel].copy_(codes)
            if channel + 1 < self.n_vq:
                current = self.local_transformer._step_eager(
                    F.embedding(codes, head_weight).to(dtype=current.dtype),
                    channel + 1,
                )

    def _decode_frame_batched_graph(
        self,
        hidden_batch: torch.Tensor,
        params: tuple[float, float, int, float, float, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(hidden_batch.shape[0])
        capture_batch_size = self._frame_graph_capture_batch_size(batch_size)
        dtype = self.audio_embeddings[0].weight.dtype
        self.local_transformer._ensure_kv_cache(capture_batch_size, hidden_batch.device, dtype)
        self._ensure_batch_frame_graph_buffers(
            batch_size=capture_batch_size,
            device=hidden_batch.device,
            dtype=dtype,
        )
        assert self._batch_frame_graph_input is not None
        assert self._batch_frame_graph_rand is not None
        assert self._batch_frame_graph_stop is not None
        assert self._batch_frame_graph_codes is not None
        profile = bool(getattr(self, "_frame_graph_profile_enabled", False))
        _t = time.perf_counter()
        self._batch_frame_graph_input[:batch_size].copy_(hidden_batch.to(dtype=dtype))
        if capture_batch_size > batch_size:
            self._batch_frame_graph_input[batch_size:capture_batch_size].zero_()
        if profile:
            self._batch_stats["frame_graph_input_ms"] += (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()
        self._batch_frame_graph_rand[:, :capture_batch_size].copy_(
            torch.rand(1 + self.n_vq, capture_batch_size, device=hidden_batch.device)
        )
        if profile:
            self._batch_stats["frame_graph_rand_ms"] += (time.perf_counter() - _t) * 1000
        graph_key = (capture_batch_size, params)
        if profile:
            key_counts = self._batch_stats.setdefault("frame_graph_keys", {})
            key_counts[str(graph_key)] = int(key_counts.get(str(graph_key), 0)) + 1
        if graph_key not in self._batch_frame_graphs:
            _t = time.perf_counter()
            self._batch_frame_graph_kernel(capture_batch_size, params)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._batch_frame_graph_kernel(capture_batch_size, params)
            self._batch_frame_graphs[graph_key] = graph
            if profile:
                self._batch_stats["frame_graph_capture"] += 1
                self._batch_stats["frame_graph_capture_ms"] += (time.perf_counter() - _t) * 1000
                _t = time.perf_counter()
            stop = self._batch_frame_graph_stop[:batch_size].clone()
            codes = self._batch_frame_graph_codes[:batch_size].clone()
            if profile:
                self._batch_stats["frame_graph_clone_ms"] += (time.perf_counter() - _t) * 1000
            return stop, codes
        _t = time.perf_counter()
        self._batch_frame_graphs[graph_key].replay()
        if profile:
            self._batch_stats["frame_graph_replay"] += 1
            self._batch_stats["frame_graph_replay_ms"] += (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()
        stop = self._batch_frame_graph_stop[:batch_size].clone()
        codes = self._batch_frame_graph_codes[:batch_size].clone()
        if profile:
            self._batch_stats["frame_graph_clone_ms"] += (time.perf_counter() - _t) * 1000
        return stop, codes

    def _decode_frame(self, hidden: torch.Tensor, info: dict[str, Any]) -> tuple[int, torch.Tensor]:
        state = info.get("audio_state", {}) or {}
        if int(state.get("max_new_frames", -1)) > 0:
            emitted = int(state.get("step", 0))
            if emitted >= int(state["max_new_frames"]):
                return 1, torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long, device=hidden.device)

        text_temp = float(state.get("text_temperature", _first_scalar(info.get("text_temperature", 1.0))))
        audio_temp = float(state.get("audio_temperature", _first_scalar(info.get("audio_temperature", 1.0))))

        if getattr(self, "_local_graph_enabled", False) and hidden.device.type == "cuda":
            if text_temp <= 0 and audio_temp <= 0:
                return self._decode_frame_greedy_graph(hidden)
            text_top_k_v = int(state.get("text_top_k", _first_scalar(info.get("text_top_k", 50))))
            audio_top_k_v = int(state.get("audio_top_k", _first_scalar(info.get("audio_top_k", 50))))
            audio_top_p_v = float(state.get("audio_top_p", _first_scalar(info.get("audio_top_p", 0.95))))
            default_sampling = (
                text_top_k_v == 50
                and audio_top_k_v == 50
                and abs(audio_top_p_v - 0.95) < 0.01
                and abs(text_temp - 1.0) < 0.01
                and abs(audio_temp - 1.0) < 0.01
            )
            if default_sampling:
                generator = state.get("sampling_generator")
                if not isinstance(generator, torch.Generator):
                    generator = None
                return self._decode_frame_sampling_graph(hidden, generator)

        text_top_p = float(state.get("text_top_p", _first_scalar(info.get("text_top_p", 1.0))))
        text_top_k = int(state.get("text_top_k", _first_scalar(info.get("text_top_k", 50))))
        audio_top_p = float(state.get("audio_top_p", _first_scalar(info.get("audio_top_p", 0.95))))
        audio_top_k = int(state.get("audio_top_k", _first_scalar(info.get("audio_top_k", 50))))
        generator = state.get("sampling_generator")
        if not isinstance(generator, torch.Generator):
            generator = None

        local_hidden = self.local_transformer.step(hidden.to(dtype=self.audio_embeddings[0].weight.dtype), 0)
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()
        stop_choice = int(
            sample_top_k_top_p(
                text_logits,
                temperature=text_temp,
                top_p=text_top_p,
                top_k=text_top_k,
                generator=generator,
            )[0]
        )

        codes: list[torch.Tensor] = []
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
            logits = F.linear(current, head_weight).float()
            code = sample_top_k_top_p(
                logits,
                temperature=audio_temp,
                top_p=audio_top_p,
                top_k=audio_top_k,
                generator=generator,
            )
            codes.append(code.reshape(()))
            if channel + 1 < self.n_vq:
                current = self.local_transformer.step(
                    F.embedding(code, head_weight).to(dtype=current.dtype), channel + 1
                )
        return stop_choice, torch.stack(codes).to(dtype=torch.long)

    def _decode_frame_batched(
        self,
        hidden_batch: torch.Tensor,
        infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched local transformer decode: [B, H] -> (stop_choices[B], codes[B, n_vq])."""
        B = hidden_batch.shape[0]
        dtype = self.audio_embeddings[0].weight.dtype
        self.local_transformer._ensure_kv_cache(B, hidden_batch.device, dtype)

        # Extract per-request params once.  Seed the comparison from the first
        # request instead of hard-coded defaults so model-card sampling
        # (audio_temperature=1.7, top_k=25, top_p=0.8) can still use the
        # homogeneous batched path.
        def frame_params(info: dict[str, Any]) -> tuple[float, int, float, float, int, float]:
            if not isinstance(info, dict):
                info = {}
            state = info.get("audio_state", {}) or {}
            tt = float(state.get("text_temperature", _first_scalar(info.get("text_temperature", 1.0))))
            tk = int(state.get("text_top_k", _first_scalar(info.get("text_top_k", 50))))
            tp = float(state.get("text_top_p", _first_scalar(info.get("text_top_p", 1.0))))
            at = float(state.get("audio_temperature", _first_scalar(info.get("audio_temperature", 1.0))))
            ak = int(state.get("audio_top_k", _first_scalar(info.get("audio_top_k", 50))))
            ap = float(state.get("audio_top_p", _first_scalar(info.get("audio_top_p", 0.95))))
            return tt, tk, tp, at, ak, ap

        text_temp, text_top_k, text_top_p, audio_temp, audio_top_k, audio_top_p = (
            frame_params(infos[0]) if infos else (1.0, 50, 1.0, 1.0, 50, 0.95)
        )
        homogeneous = True
        for info in infos[1:]:
            tt, tk, tp, at, ak, ap = frame_params(info)
            if abs(tt - text_temp) > 0.01 or tk != text_top_k or abs(tp - text_top_p) > 0.01:
                homogeneous = False
                break
            if abs(at - audio_temp) > 0.01 or ak != audio_top_k or abs(ap - audio_top_p) > 0.01:
                homogeneous = False
                break

        graph_params = (
            float(text_temp),
            float(text_top_p),
            int(text_top_k),
            float(audio_temp),
            float(audio_top_p),
            int(audio_top_k),
        )
        if (
            homogeneous
            and self._local_graph_enabled
            and hidden_batch.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            try:
                return self._decode_frame_batched_graph(hidden_batch, graph_params)
            except Exception:
                logger.exception("MOSS-TTS Local batched frame graph failed; falling back to eager local decode")
                self._batch_frame_graphs.clear()

        if self._local_graph_enabled and hidden_batch.device.type == "cuda":
            bucket = _next_bucket(B)
            if not self.local_transformer._graph_enabled or self.local_transformer._graph_batch_size < bucket:
                self.local_transformer._ensure_kv_cache(bucket, hidden_batch.device, dtype)
                self.local_transformer.enable_graphs(bucket)
        local_hidden = self.local_transformer.step(hidden_batch.to(dtype=dtype), 0)

        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()

        if homogeneous:
            # Fast path: single batched call for all B rows (no per-request generator)
            if text_temp <= 0:
                stop_choices = torch.argmax(text_logits, dim=-1)
            else:
                stop_choices = sample_top_k_top_p(
                    text_logits,
                    temperature=text_temp,
                    top_p=text_top_p,
                    top_k=text_top_k,
                )
            all_codes = torch.zeros(B, self.n_vq, dtype=torch.long, device=hidden_batch.device)
            current = local_hidden
            for channel in range(self.n_vq):
                head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
                logits = F.linear(current, head_weight).float()
                if audio_temp <= 0:
                    codes = torch.argmax(logits, dim=-1)
                else:
                    codes = sample_top_k_top_p(
                        logits,
                        temperature=audio_temp,
                        top_p=audio_top_p,
                        top_k=audio_top_k,
                    )
                all_codes[:, channel] = codes
                if channel + 1 < self.n_vq:
                    embedded = F.embedding(codes, head_weight).to(dtype=current.dtype)
                    current = self.local_transformer.step(embedded, channel + 1)
            return stop_choices, all_codes

        # Slow path: per-request sampling (heterogeneous params or per-request generator)
        stop_choices = torch.zeros(B, dtype=torch.long, device=hidden_batch.device)
        for b in range(B):
            info = infos[b] if b < len(infos) else {}
            state = (info.get("audio_state", {}) or {}) if isinstance(info, dict) else {}
            gen = state.get("sampling_generator") if isinstance(state, dict) else None
            if not isinstance(gen, torch.Generator):
                gen = None
            stop_choices[b] = sample_top_k_top_p(
                text_logits[b : b + 1],
                temperature=float(state.get("text_temperature", 1.0)),
                top_p=float(state.get("text_top_p", 1.0)),
                top_k=int(state.get("text_top_k", 50)),
                generator=gen,
            )[0]

        all_codes = torch.zeros(B, self.n_vq, dtype=torch.long, device=hidden_batch.device)
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
            logits = F.linear(current, head_weight).float()
            codes = torch.zeros(B, dtype=torch.long, device=hidden_batch.device)
            for b in range(B):
                info = infos[b] if b < len(infos) else {}
                state = (info.get("audio_state", {}) or {}) if isinstance(info, dict) else {}
                gen = state.get("sampling_generator") if isinstance(state, dict) else None
                if not isinstance(gen, torch.Generator):
                    gen = None
                codes[b] = sample_top_k_top_p(
                    logits[b : b + 1],
                    temperature=float(state.get("audio_temperature", 1.0)),
                    top_p=float(state.get("audio_top_p", 0.95)),
                    top_k=int(state.get("audio_top_k", 50)),
                    generator=gen,
                )[0]
            all_codes[:, channel] = codes
            if channel + 1 < self.n_vq:
                embedded = F.embedding(codes, head_weight).to(dtype=current.dtype)
                current = self.local_transformer.step(embedded, channel + 1)
        return stop_choices, all_codes

    def _maybe_dump_codes_debug(
        self,
        *,
        hidden: torch.Tensor,
        new_codes: torch.Tensor,
        accumulated_codes: torch.Tensor,
        state: dict[str, Any],
    ) -> None:
        dump_dir = os.environ.get("MOSS_TTS_LOCAL_DUMP_CODES_DIR")
        if not dump_dir:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            step = int(state.get("step", 0))
            path = os.path.join(dump_dir, f"stage0_codes_{os.getpid()}_step{step}.pt")
            torch.save(
                {
                    "decode_hidden": hidden.detach().to("cpu"),
                    "new_codes": new_codes.detach().to("cpu"),
                    "codes": accumulated_codes.detach().to("cpu"),
                },
                path,
            )
        except Exception:
            logger.exception("Failed to dump MOSS-TTS Local debug codes")

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        return self.decode_omni_frame_output(model_outputs, **kwargs)

    def decode_omni_frame_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            self._batch_state = None
            self._batch_next_text = None
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
        if isinstance(logits_index, torch.Tensor) and logits_index.numel() > 0:
            logits_index_gpu = logits_index.to(device=hidden.device, dtype=torch.long).reshape(-1)
            num_sample_rows = int(logits_index_gpu.numel())
            if int(hidden.shape[0]) == num_sample_rows:
                sample_hidden = hidden
            elif int(logits_index_gpu.max().item()) < int(hidden.shape[0]):
                sample_hidden = hidden.index_select(0, logits_index_gpu)
            else:
                sample_hidden = hidden[:num_sample_rows]
            if not isinstance(sample_row_by_req, (list, tuple)):
                logits_rows = logits_index.detach().to("cpu", dtype=torch.long).reshape(-1).tolist()
        elif isinstance(logits_index, int):
            logits_rows = [int(logits_index)]
            sample_hidden = hidden[int(logits_index) : int(logits_index) + 1]
        elif isinstance(request_token_spans, (list, tuple)) and len(request_token_spans) >= len(info_dicts):
            sample_rows: list[int] = []
            for start, end in request_token_spans[: len(info_dicts)]:
                del start
                sample_rows.append(int(end) - 1)
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

        for info in info_dicts:
            if isinstance(info, dict) and not isinstance(info.get("audio_state"), dict):
                state = {"step": 0, "is_stopping": False}
                _copy_sampling_overrides(state, info)
                sampling_generator = _make_sampling_generator(info.get("seed"), hidden.device)
                if sampling_generator is not None:
                    state["sampling_generator"] = sampling_generator
                info["audio_state"] = state

        pending_req_ids, pending_codes, pending_codec_streaming = self._consume_pending_stop_result(info_dicts)

        delta_req_ids: list[str] = list(pending_req_ids)
        delta_codes: list[torch.Tensor] = list(pending_codes)
        delta_codec_streaming: list[bool] = list(pending_codec_streaming)
        next_text_tensor: torch.Tensor | None = None
        batch_state_by_sample_row: list[dict[str, Any] | None] = []
        if sample_hidden.numel() > 0 and info_dicts:
            num_rows = int(sample_hidden.shape[0])
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
                    return None
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

            use_batched = len(decode_items) > 1 and all(isinstance(info_dicts[j], dict) for j, _ in decode_items)

            if use_batched:
                profile_detail = bool(getattr(self, "_profile_detail_enabled", False))
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
                _t_local = time.perf_counter()
                stop_choices, all_codes = self._decode_frame_batched(
                    hidden_batch, [info_dicts[j] for j, _ in decode_items]
                )
                self._batch_stats["t_local_tx_ms"] += (time.perf_counter() - _t_local) * 1000

                _t_asm = time.perf_counter()
                _t_detail = _t_asm
                active_states: list[dict[str, Any]] = []
                active_slot_ids: list[int] = []
                pool = self._get_request_state_pool()
                pool.ensure_frame_state_device(hidden.device)
                slot_cpu = asm_buffers["slot_cpu"][:n_active]
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
                    slot_id_raw = info_dicts[i].get("_kv_slot_id")
                    try:
                        slot_id = int(slot_id_raw)
                    except (TypeError, ValueError):
                        slot_id = -1
                    active_slot_ids.append(slot_id)
                    slot_cpu[bi] = slot_id
                    if 0 <= slot_id < pool.max_slots:
                        if pool.is_stopping_py[slot_id]:
                            state["is_stopping"] = True
                    else:
                        all_slots_valid = False
                if profile_detail:
                    _now = time.perf_counter()
                    self._batch_stats["t_asm_active_state_ms"] += (_now - _t_detail) * 1000
                    _t_detail = _now

                next_for_active = asm_buffers["next_gpu"][:n_active]
                next_for_active.fill_(self.audio_assistant_slot_token_id)
                should_stop_list: list[bool] = [False for _ in range(n_active)]
                stop_sync_start: float | None = None
                delay_stop_pending = False
                slot_ids_t = asm_buffers["slot_gpu"][:n_active]
                slot_ids_t.copy_(slot_cpu, non_blocking=hidden.device.type == "cuda")
                steps_t = asm_buffers["steps_gpu"][:n_active]
                min_frames_t = asm_buffers["min_gpu"][:n_active]
                max_frames_t = asm_buffers["max_gpu"][:n_active]
                should_stop_t = asm_buffers["stop_gpu"][:n_active]
                tmp_stop_t = asm_buffers["tmp_stop_gpu"][:n_active]
                stop_interval = int(getattr(self, "_stop_check_interval", 1))
                step_values: list[int] = []
                min_values: list[int] = []
                max_values: list[int] = []
                model_stop_poll: list[bool] = []
                max_stop_list: list[bool] = []
                active_req_ids: list[str] = []
                for state, slot_id in zip(active_states, active_slot_ids):
                    if 0 <= slot_id < pool.max_slots:
                        step_num = pool.steps_py[slot_id]
                        min_frames = pool.min_frames_py[slot_id]
                        max_frames = pool.max_frames_py[slot_id]
                    else:
                        step_num = int(state.get("step", 0))
                        min_frames = int(state.get("min_new_frames", 3))
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
                    for bi, (step_num, min_frames, max_frames) in enumerate(zip(step_values, min_values, max_values)):
                        steps_cpu[bi] = step_num
                        min_cpu[bi] = min_frames
                        max_cpu[bi] = max_frames
                    steps_t.copy_(steps_cpu, non_blocking=hidden.device.type == "cuda")
                    min_frames_t.copy_(min_cpu, non_blocking=hidden.device.type == "cuda")
                    max_frames_t.copy_(max_cpu, non_blocking=hidden.device.type == "cuda")
                if profile_detail:
                    _now = time.perf_counter()
                    self._batch_stats["t_asm_state_gather_ms"] += (_now - _t_detail) * 1000
                    _t_detail = _now
                delay_model_stop = (
                    bool(getattr(self, "_delay_stop_sync_enabled", False))
                    and hidden.device.type == "cuda"
                    and stop_interval == 1
                    and not all_slots_valid
                    and any(model_stop_poll)
                )
                delay_delta_routing = bool(getattr(self, "_delay_delta_routing_enabled", False))
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
                            pending_stop_result = {
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
                            delay_stop_pending = True
                        should_stop_list = [bool(x) for x in max_stop_list]
                        next_for_active.fill_(self.audio_assistant_slot_token_id)
                        if any(max_stop_list):
                            poll_cpu = asm_buffers["poll_cpu"][:n_active]
                            for bi, should_stop in enumerate(max_stop_list):
                                poll_cpu[bi] = bool(should_stop)
                            max_stop_t = asm_buffers["poll_gpu"][:n_active]
                            max_stop_t.copy_(poll_cpu, non_blocking=True)
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
                        next_for_active.fill_(self.audio_assistant_slot_token_id)
                        if any(max_stop_list):
                            next_for_active.masked_fill_(should_stop_t, self.audio_end_token_id)
                    if profile_detail:
                        _now = time.perf_counter()
                        self._batch_stats["t_asm_stop_compute_ms"] += (_now - _t_detail) * 1000
                        _t_detail = _now
                    if delay_model_stop and delay_stop_pending:
                        stop_copy_event = asm_buffers.get("stop_copy_event")
                        if isinstance(stop_copy_event, torch.cuda.Event) and stop_copy_event.query():
                            model_stop_list = [bool(x) for x in stop_cpu.tolist()]
                            should_stop_list = [
                                bool(max_due or model_due) for max_due, model_due in zip(max_stop_list, model_stop_list)
                            ]
                            for bi, should_stop in enumerate(should_stop_list):
                                stop_cpu[bi] = bool(should_stop)
                            should_stop_t.copy_(stop_cpu, non_blocking=True)
                            next_for_active.fill_(self.audio_assistant_slot_token_id)
                            next_for_active.masked_fill_(should_stop_t, self.audio_end_token_id)
                            self._pending_stop_result = None
                            delay_stop_pending = False
                next_text_tensor.index_copy_(0, decode_rows_t, next_for_active)
                if profile_detail:
                    _now = time.perf_counter()
                    self._batch_stats["t_asm_next_text_ms"] += (_now - _t_detail) * 1000
                    _t_detail = _now
                if stop_possible:
                    stop_copy_event = asm_buffers.get("stop_copy_event")
                    if (
                        (not delay_model_stop)
                        and any(model_stop_poll)
                        and isinstance(stop_copy_event, torch.cuda.Event)
                    ):
                        stop_copy_event.synchronize()
                    if stop_sync_start is not None:
                        self._batch_stats["t_stop_sync_ms"] += (time.perf_counter() - stop_sync_start) * 1000
                    if not delay_model_stop:
                        should_stop_list = [bool(x) for x in stop_cpu.tolist()]
                    if profile_detail:
                        _now = time.perf_counter()
                        self._batch_stats["t_asm_stop_wait_ms"] += (_now - _t_detail) * 1000
                        _t_detail = _now
                else:
                    should_stop_t.zero_()
                if all_slots_valid:
                    pool.advance_frame_state_gpu(
                        slot_ids_t,
                        should_stop_t,
                        should_stop_list,
                        slot_ids_cpu=active_slot_ids,
                    )
                else:
                    pool.advance_frame_state(active_slot_ids, should_stop_list)
                if profile_detail:
                    _now = time.perf_counter()
                    self._batch_stats["t_asm_pool_advance_ms"] += (_now - _t_detail) * 1000
                    _t_detail = _now

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
                        self._batch_stats["completed_frames"].append(step_num)
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = pool.steps_py[slot_id] if 0 <= slot_id < pool.max_slots else step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        rid = str(info.get("request_id") or info.get("global_request_id") or "")
                        if rid or len(info_dicts) == 1:
                            if not (delay_model_stop and delay_stop_pending and delay_delta_routing):
                                delta_req_ids.append(rid)
                                delta_codes.append(all_codes[bi : bi + 1])
                                delta_codec_streaming.append(_stream_flag_from_info(info))
                if profile_detail:
                    _now = time.perf_counter()
                    self._batch_stats["t_asm_state_loop_ms"] += (_now - _t_detail) * 1000
                self._batch_stats["t_output_asm_ms"] += (time.perf_counter() - _t_asm) * 1000
            else:
                # Sequential fallback (prefill, single request, or mixed)
                for i, row_idx in decode_items:
                    info = info_dicts[i]
                    if not isinstance(info, dict):
                        continue
                    state = info.get("audio_state", {}) or {}
                    if state.get("is_stopping"):
                        state["next_text"] = self.audio_end_token_id
                        if next_text_tensor is not None:
                            next_text_tensor[row_idx] = self.audio_end_token_id
                        continue
                    stop_choice, new_codes = self._decode_frame(sample_hidden[row_idx].unsqueeze(0), info)
                    step_num = int(state.get("step", 0))
                    min_frames = int(state.get("min_new_frames", 3))
                    if stop_choice == 1 and step_num >= min_frames:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        if next_text_tensor is not None:
                            next_text_tensor[row_idx] = self.audio_end_token_id
                        info["audio_codes"] = {"current": new_codes}
                        continue
                    state["next_text"] = self.audio_assistant_slot_token_id
                    if next_text_tensor is not None:
                        next_text_tensor[row_idx] = self.audio_assistant_slot_token_id
                    state["step"] = int(state.get("step", 0)) + 1
                    info["audio_codes"] = {"current": new_codes}
                    rid = str(info.get("request_id") or info.get("global_request_id") or "")
                    if rid or len(info_dicts) == 1:
                        delta_req_ids.append(rid)
                        delta_codes.append(new_codes.unsqueeze(0))
                        delta_codec_streaming.append(_stream_flag_from_info(info))
                    self._maybe_dump_codes_debug(
                        hidden=sample_hidden[row_idx].unsqueeze(0),
                        new_codes=new_codes,
                        accumulated_codes=new_codes.unsqueeze(0),
                        state=state,
                    )

        _t_tail = time.perf_counter() if bool(getattr(self, "_profile_detail_enabled", False)) else 0.0
        self._batch_state = [
            (state if isinstance(state, dict) else {"next_text": self.audio_end_token_id})
            for state in batch_state_by_sample_row
        ] or [(info.get("audio_state", {}) if isinstance(info, dict) else {}) for info in info_dicts]
        self._batch_next_text = next_text_tensor
        if not delta_codes:
            if _t_tail:
                self._batch_stats["t_asm_tail_ms"] += (time.perf_counter() - _t_tail) * 1000
            return OmniOutput(
                text_hidden_states=sample_hidden,
                multimodal_outputs={"meta": {"next_text": next_text_tensor}} if next_text_tensor is not None else {},
            )
        step_counter = max(
            (int((info.get("audio_state", {}) or {}).get("step", 0)) for info in info_dicts if isinstance(info, dict)),
            default=0,
        )
        # Single request: emit codes directly (original behavior, proven correct).
        if len(info_dicts) == 1 and len(delta_codes) == 1:
            codec_streaming = _stream_flag_from_info(info_dicts[0]) if isinstance(info_dicts[0], dict) else True
            if _t_tail:
                self._batch_stats["t_asm_tail_ms"] += (time.perf_counter() - _t_tail) * 1000
            return OmniOutput(
                text_hidden_states=sample_hidden,
                multimodal_outputs={
                    "codes": {"audio": delta_codes[0]},
                    "meta": {
                        "raw_rows": True,
                        "step": step_counter,
                        "codec_streaming": codec_streaming,
                        "next_text": next_text_tensor,
                    },
                },
            )
        # Batch>1: sparse routing with delta rows (only this step's new_codes).
        # Each active request emits [1, n_vq]; stopped requests are skipped.
        if _t_tail:
            self._batch_stats["t_asm_tail_ms"] += (time.perf_counter() - _t_tail) * 1000
        return OmniOutput(
            text_hidden_states=sample_hidden,
            multimodal_outputs={
                "codes": {"audio": delta_codes},
                "meta": {
                    "sparse_audio": True,
                    "req_id": delta_req_ids,
                    "raw_rows": True,
                    "step": step_counter,
                    "codec_streaming": delta_codec_streaming,
                    "next_text": next_text_tensor,
                },
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._audio_embedding_weight_cache = None
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters())
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        skipped = 0

        for original_name, tensor in weights:
            name = original_name
            if name.startswith("transformer."):
                backbone_weights.append((name[len("transformer.") :], tensor))
                name = "model." + name[len("transformer.") :]
            elif name.startswith("model."):
                backbone_weights.append((name, tensor))

            if name.startswith("text_lm_head.") or name.startswith("audio_lm_heads."):
                continue
            if name.startswith("audio_embeddings.") and name.endswith(".weight"):
                param = params_dict.get(name)
                if param is not None:
                    rows = min(int(tensor.shape[0]), int(param.shape[0]))
                    with torch.no_grad():
                        param[:rows].copy_(tensor[:rows].to(device=param.device, dtype=param.dtype))
                    loaded.add(name)
                continue
            if name.startswith("local_transformer.") or name.startswith("local_text_lm_head."):
                param = params_dict.get(name)
                if param is not None:
                    default_weight_loader(param, tensor)
                    loaded.add(name)
                else:
                    skipped += 1
                continue

        backbone_loaded = self.model.load_weights(iter(backbone_weights))
        for name in backbone_loaded:
            loaded.add(f"model.{name}")

        with torch.no_grad():
            for emb in self.audio_embeddings:
                if emb.weight.shape[0] > self.audio_vocab_size:
                    emb.weight[self.audio_vocab_size :].zero_()
        if skipped:
            logger.warning("MOSS-TTS Local skipped %d unmatched local parameters", skipped)
        return loaded


__all__ = ["MossTTSLocalForGeneration"]
