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


class MossTTSLocalForGeneration(nn.Module):
    """Stage-0 AR model: Qwen3 backbone plus per-frame local transformer."""

    input_modalities = "audio"
    have_multimodal_outputs: bool = True
    has_preprocess: bool = True
    has_postprocess: bool = True
    requires_raw_input_tokens: bool = True

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
            attn_implementation=os.environ.get("MOSS_TTS_LOCAL_ATTN_IMPL") or getattr(self.config, "local_transformer_attn_implementation", None) or "eager",
        )
        self.local_text_lm_head = nn.Linear(self.hidden_size, 2, bias=False)
        self._batch_state: list[dict[str, Any]] | None = None
        self._graph_warmup_done: bool = False
        self._max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        self._slot_to_request: list[str | None] = [None] * self._max_num_seqs
        self._request_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self._max_num_seqs))
        self._pending_slot_ids: list[int] = []
        self._pending_token_counts: list[int] = []
        self._batch_stats: dict[str, Any] = {
            "batched": 0, "fallback_no_cache": 0, "fallback_overflow": 0,
            "fallback_prefill": 0, "single": 0, "log_interval": 100,
            "batch_sizes": [],
            "t_backbone_ms": 0.0, "t_local_tx_ms": 0.0, "t_output_asm_ms": 0.0,
            "n_steps": 0,
            "completed_frames": [],
        }
        self._local_graph: torch.cuda.CUDAGraph | None = None
        self._local_graph_sampling: torch.cuda.CUDAGraph | None = None
        self._local_graph_input: torch.Tensor | None = None
        self._local_graph_stop: torch.Tensor | None = None
        self._local_graph_codes: torch.Tensor | None = None
        self._local_graph_rand: torch.Tensor | None = None
        self._local_graph_enabled: bool = os.environ.get("MOSS_TTS_LOCAL_DECODE_GRAPH") == "1"
        if os.environ.get("MOSS_TTS_LOCAL_COMPILE_LOCAL_TX") == "1":
            self.local_transformer.step = torch.compile(self.local_transformer.step, dynamic=True)

        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("hidden_states", "last"),
        }

    def _allocate_slot(self, request_id: str) -> int:
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError(f"No free KV slots (max_num_seqs={self._max_num_seqs})")
        slot_id = self._free_slots.pop(0)
        self._slot_to_request[slot_id] = request_id
        self._request_to_slot[request_id] = slot_id
        return slot_id

    def _release_slot(self, request_id: str) -> None:
        slot_id = self._request_to_slot.pop(request_id, None)
        if slot_id is not None:
            self._slot_to_request[slot_id] = None
            self._free_slots.append(slot_id)

    def on_requests_finished(self, finished_req_ids: list[str]) -> None:
        for req_id in finished_req_ids:
            self._release_slot(req_id)

    def _log_batch_stats(self) -> None:
        s = self._batch_stats
        total = s["batched"] + s.get("fallback_no_cache", 0) + s.get("fallback_overflow", 0) + s["fallback_prefill"] + s["single"]
        if total > 0 and total % s["log_interval"] == 0:
            hit = s["batched"]
            multi = hit + s.get("fallback_no_cache", 0) + s.get("fallback_overflow", 0) + s["fallback_prefill"]
            rate = hit / max(1, multi) * 100
            n = s["n_steps"] or 1
            avg_backbone = s["t_backbone_ms"] / n
            avg_local = s["t_local_tx_ms"] / n
            avg_output_asm = s["t_output_asm_ms"] / n
            bsizes = s["batch_sizes"][-20:]
            avg_bs = sum(bsizes) / max(1, len(bsizes))
            frames = s["completed_frames"]
            frames_info = ""
            if frames:
                frames_sorted = sorted(frames)
                p50 = frames_sorted[len(frames_sorted) // 2]
                p95 = frames_sorted[min(len(frames_sorted) - 1, int(len(frames_sorted) * 0.95))]
                frames_info = f" | frames: n={len(frames)} p50={p50} p95={p95} avg={sum(frames)/len(frames):.0f}"
            logger.info(
                "Profile[%d steps]: hit=%.0f%% avg_bs=%.1f | backbone=%.2fms local_tx=%.2fms output_asm=%.2fms | "
                "batched=%d prefill=%d single=%d%s",
                n, rate, avg_bs, avg_backbone, avg_local, avg_output_asm,
                hit, s["fallback_prefill"], s["single"], frames_info,
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
        pending_slots = getattr(self, "_pending_slot_ids", None)
        pending_counts = getattr(self, "_pending_token_counts", None)
        if pending_slots and len(pending_slots) > 1:
            slot_ids = list(pending_slots)
            token_counts = list(pending_counts)
            pending_slots.clear()
            pending_counts.clear()
            if all(c == 1 for c in token_counts):
                if (
                    hasattr(self.model, 'layers')
                    and self.model.layers[0].self_attn.static_key_cache is None
                    and hasattr(self.model, 'init_static_kv_cache')
                ):
                    max_pos = int(getattr(self.config.qwen3_config, "max_position_embeddings", 4096))
                    device = input_ids.device if input_ids is not None else positions.device
                    dtype = self.model.embed_tokens.weight.dtype
                    self.model.init_static_kv_cache(
                        max_len=min(max_pos, 512), max_batch=self._max_num_seqs, device=device, dtype=dtype,
                    )
                can_batch = (
                    hasattr(self.model, 'layers')
                    and self.model.layers[0].self_attn.static_key_cache is not None
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
                    self._batch_stats[f"fallback_{fallback_reason}"] = self._batch_stats.get(f"fallback_{fallback_reason}", 0) + 1
                    self._log_batch_stats()
            else:
                self._batch_stats["fallback_prefill"] += 1
                self._log_batch_stats()
            outputs = []
            offset = 0
            for slot_id, count in zip(slot_ids, token_counts):
                seg_embeds = inputs_embeds[offset:offset + count] if inputs_embeds is not None else None
                seg_ids = input_ids[offset:offset + count] if input_ids is not None else None
                seg_pos = positions[offset:offset + count]
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
        slot_id = 0
        if pending_slots:
            slot_id = pending_slots.pop(0)
            if pending_counts:
                pending_counts.pop(0)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            slot_id=slot_id,
        )


    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor | None:
        """Return one-hot text logits: continue slot or audio_end."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None or hidden_states.numel() == 0:
            return None
        logits = hidden_states.new_full((hidden_states.shape[0], self.vocab_size), float("-inf"))
        states = self._batch_state or []
        if not states:
            logits[:, self.audio_assistant_slot_token_id] = 0.0
            return logits
        rows_per_state = max(1, hidden_states.shape[0] // max(1, len(states)))
        for i, state in enumerate(states):
            r0 = i * rows_per_state
            r1 = min(r0 + rows_per_state, hidden_states.shape[0])
            if r0 >= r1:
                continue
            token_id = int(state.get("next_text", self.audio_assistant_slot_token_id))
            if not 0 <= token_id < self.vocab_size:
                token_id = self.audio_end_token_id
            logits[r0:r1, token_id] = 0.0
        if os.environ.get("MOSS_TTS_LOCAL_DEBUG_BATCH") and len(states) > 1:
            tokens = [int(s.get("next_text", self.audio_assistant_slot_token_id)) for s in states]
            steps = [int(s.get("step", 0)) for s in states]
            stopping = [bool(s.get("is_stopping")) for s in states]
            logger.info(
                "compute_logits batch=%d tokens=%s steps=%s stopping=%s",
                len(states), tokens, steps, stopping,
            )
        return logits

    def _build_input_embeds(self, text_ids: torch.Tensor, audio_codes: torch.Tensor | None) -> torch.Tensor:
        embeds = self.model.embed_tokens(text_ids)
        if audio_codes is None:
            return embeds
        codes = audio_codes.to(device=text_ids.device, dtype=torch.long)
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        for idx, emb in enumerate(self.audio_embeddings):
            col = codes[:, idx].clamp(0, self.audio_vocab_size)
            valid = col.ne(self.audio_pad_code)
            safe_col = col.masked_fill(~valid, 0)
            embeds = embeds + emb(safe_col) * valid.unsqueeze(-1)
        return embeds

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

        request_id = str(
            info_dict.get("request_id")
            or info_dict.get("global_request_id")
            or ""
        )
        try:
            request_to_slot = self._request_to_slot
            slot_id = request_to_slot.get(request_id) if request_id else None
            if is_first_call and slot_id is None and request_id:
                slot_id = self._allocate_slot(request_id)
        except AttributeError:
            slot_id = None
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
                        warmup_buckets, device=device, dtype=self.model.embed_tokens.weight.dtype,
                        max_batch=self._max_num_seqs,
                    )
                    self.model.init_static_kv_cache(
                        max_len=bucket_len, max_batch=self._max_num_seqs, device=device, dtype=self.model.embed_tokens.weight.dtype,
                    )
            sampling_generator = _make_sampling_generator(info_dict.get("seed"), device)
            state = {
                "step": 0,
                "is_stopping": False,
                "next_text": self.audio_assistant_slot_token_id,
                "max_new_frames": max_new_frames,
            }
            _copy_sampling_overrides(state, info_dict)
            if sampling_generator is not None:
                state["sampling_generator"] = sampling_generator
            self._maybe_dump_request_debug(
                input_ids=input_ids,
                info_dict=info_dict,
                state=state,
            )
            try:
                self._pending_slot_ids.append(slot_id)
                self._pending_token_counts.append(span_len)
            except AttributeError:
                pass
            return input_ids, embeds, {
                "audio_state": state,
                "audio_codes": {"current": current_codes},
                "ref_offset": int(info_dict.get("ref_offset", 0)) + span_len,
                "_kv_slot_id": slot_id,
            }

        prev_codes = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(prev_codes, torch.Tensor) or prev_codes.numel() != self.n_vq:
            prev_codes = torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
        embeds = self._build_input_embeds(input_ids.reshape(-1), prev_codes.to(device=device).unsqueeze(0))
        try:
            self._pending_slot_ids.append(slot_id)
            self._pending_token_counts.append(1)
        except AttributeError:
            pass
        return input_ids, embeds, {"_kv_slot_id": slot_id}

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
                    "sampling_overrides": {
                        key: state[key] for key in _SAMPLING_OVERRIDE_KEYS if key in state
                    },
                    "info": {
                        key: _debug_value(value)
                        for key, value in info_dict.items()
                        if key != "audio_state"
                    },
                },
                path,
            )
        except Exception:
            logger.exception("Failed to dump MOSS-TTS Local debug request")

    @staticmethod
    def _sample_graph_safe(logits: torch.Tensor, top_k: int, top_p: float, temperature: float, rand_val: torch.Tensor) -> torch.Tensor:
        scores = logits / max(temperature, 1e-6)
        vocab = int(scores.shape[-1])
        if 0 < top_k < vocab:
            values, _ = torch.topk(scores, top_k, dim=-1)
            scores = scores.masked_fill(scores < values[..., -1:], float("-inf"))
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
            code = self._sample_graph_safe(logits, 50, 0.95, 1.0, self._local_graph_rand[1 + channel:2 + channel])
            self._local_graph_codes[channel] = code.reshape(())
            if channel + 1 < self.n_vq:
                current = self.local_transformer.step(F.embedding(code, head_weight).to(dtype=current.dtype), channel + 1)

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
                current = self.local_transformer.step(F.embedding(code, head_weight).to(dtype=current.dtype), channel + 1)

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

    def _decode_frame_sampling_graph(self, hidden: torch.Tensor, generator: torch.Generator | None) -> tuple[int, torch.Tensor]:
        self._ensure_local_sampling_graph(hidden.device)
        self._local_graph_input.copy_(hidden.to(dtype=self._local_graph_input.dtype))
        rand_vals = torch.rand(1 + self.n_vq, device=hidden.device, generator=generator)
        self._local_graph_rand.copy_(rand_vals)
        self._local_graph_sampling.replay()
        return int(self._local_graph_stop.item()), self._local_graph_codes.clone()

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
                text_top_k_v == 50 and audio_top_k_v == 50
                and abs(audio_top_p_v - 0.95) < 0.01
                and abs(text_temp - 1.0) < 0.01 and abs(audio_temp - 1.0) < 0.01
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
                current = self.local_transformer.step(F.embedding(code, head_weight).to(dtype=current.dtype), channel + 1)
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
        local_hidden = self.local_transformer.step(hidden_batch.to(dtype=dtype), 0)

        # Extract per-request params once
        text_temp = 1.0
        text_top_k = 50
        text_top_p = 1.0
        audio_temp = 1.0
        audio_top_k = 50
        audio_top_p = 0.95
        homogeneous = True
        for info in infos:
            if not isinstance(info, dict):
                continue
            state = info.get("audio_state", {}) or {}
            tt = float(state.get("text_temperature", _first_scalar(info.get("text_temperature", 1.0))))
            tk = int(state.get("text_top_k", _first_scalar(info.get("text_top_k", 50))))
            tp = float(state.get("text_top_p", _first_scalar(info.get("text_top_p", 1.0))))
            at = float(state.get("audio_temperature", _first_scalar(info.get("audio_temperature", 1.0))))
            ak = int(state.get("audio_top_k", _first_scalar(info.get("audio_top_k", 50))))
            ap = float(state.get("audio_top_p", _first_scalar(info.get("audio_top_p", 0.95))))
            if abs(tt - text_temp) > 0.01 or tk != text_top_k or abs(tp - text_top_p) > 0.01:
                homogeneous = False
                break
            if abs(at - audio_temp) > 0.01 or ak != audio_top_k or abs(ap - audio_top_p) > 0.01:
                homogeneous = False
                break

        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()

        if homogeneous:
            # Fast path: single batched call for all B rows (no per-request generator)
            stop_choices = sample_top_k_top_p(
                text_logits, temperature=text_temp, top_p=text_top_p, top_k=text_top_k,
            )
            all_codes = torch.zeros(B, self.n_vq, dtype=torch.long, device=hidden_batch.device)
            current = local_hidden
            for channel in range(self.n_vq):
                head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
                logits = F.linear(current, head_weight).float()
                codes = sample_top_k_top_p(
                    logits, temperature=audio_temp, top_p=audio_top_p, top_k=audio_top_k,
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
                text_logits[b:b+1],
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
                    logits[b:b+1],
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
        if isinstance(model_outputs, OmniOutput):
            self._batch_state = None
            return model_outputs
        hidden = model_outputs
        info_dicts: list[dict[str, Any]] = (
            kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        )
        for info in info_dicts:
            if isinstance(info, dict) and not isinstance(info.get("audio_state"), dict):
                state = {"step": 0, "is_stopping": False}
                _copy_sampling_overrides(state, info)
                sampling_generator = _make_sampling_generator(info.get("seed"), hidden.device)
                if sampling_generator is not None:
                    state["sampling_generator"] = sampling_generator
                info["audio_state"] = state

        audio_codes_list: list[torch.Tensor | None] = []
        if hidden.numel() > 0 and info_dicts:
            num_rows = int(hidden.shape[0])
            rows_per_req = max(1, num_rows // max(1, len(info_dicts)))

            # Try batched decode: all active (non-stopping) requests processed in one pass
            active_indices: list[int] = []
            for i, info in enumerate(info_dicts):
                if not isinstance(info, dict):
                    continue
                state = info.get("audio_state", {}) or {}
                if not state.get("is_stopping"):
                    active_indices.append(i)

            use_batched = (
                len(active_indices) > 1
                and num_rows == len(info_dicts)
                and all(isinstance(info_dicts[j], dict) for j in active_indices)
            )

            if use_batched:
                hidden_batch = torch.stack([hidden[j] for j in active_indices])
                _t_local = time.perf_counter()
                stop_choices, all_codes = self._decode_frame_batched(hidden_batch, [info_dicts[j] for j in active_indices])
                self._batch_stats["t_local_tx_ms"] += (time.perf_counter() - _t_local) * 1000

                _t_asm = time.perf_counter()
                stop_list = stop_choices.detach().cpu().tolist()
                active_set = set(active_indices)
                bi = 0
                for i, info in enumerate(info_dicts):
                    if not isinstance(info, dict):
                        audio_codes_list.append(None)
                        continue
                    state = info.get("audio_state", {}) or {}
                    if state.get("is_stopping"):
                        state["next_text"] = self.audio_end_token_id
                        audio_codes_list.append(None)
                        continue
                    if i not in active_set:
                        audio_codes_list.append(None)
                        continue
                    new_codes = all_codes[bi]
                    step_num = int(state.get("step", 0))
                    min_frames = int(state.get("min_new_frames", 3))
                    if stop_list[bi] == 1 and step_num >= min_frames:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list.append(None)
                        self._batch_stats["completed_frames"].append(step_num)
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = int(state.get("step", 0)) + 1
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list.append(new_codes.unsqueeze(0))
                    bi += 1
                self._batch_stats["t_output_asm_ms"] += (time.perf_counter() - _t_asm) * 1000
            else:
                # Sequential fallback (prefill, single request, or mixed)
                for i, info in enumerate(info_dicts):
                    if not isinstance(info, dict):
                        audio_codes_list.append(None)
                        continue
                    state = info.get("audio_state", {}) or {}
                    row_end = min((i + 1) * rows_per_req, num_rows)
                    if row_end <= i * rows_per_req:
                        audio_codes_list.append(None)
                        continue
                    if state.get("is_stopping"):
                        state["next_text"] = self.audio_end_token_id
                        audio_codes_list.append(None)
                        continue
                    stop_choice, new_codes = self._decode_frame(hidden[row_end - 1].unsqueeze(0), info)
                    step_num = int(state.get("step", 0))
                    min_frames = int(state.get("min_new_frames", 3))
                    if stop_choice == 1 and step_num >= min_frames:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list.append(None)
                        continue
                    state["next_text"] = self.audio_assistant_slot_token_id
                    state["step"] = int(state.get("step", 0)) + 1
                    info["audio_codes"] = {"current": new_codes}
                    self._maybe_dump_codes_debug(
                        hidden=hidden[row_end - 1].unsqueeze(0),
                        new_codes=new_codes,
                        accumulated_codes=new_codes.unsqueeze(0),
                        state=state,
                    )
                    audio_codes_list.append(new_codes.unsqueeze(0))

        self._batch_state = [(info.get("audio_state", {}) if isinstance(info, dict) else {}) for info in info_dicts]
        active_codes = [c for c in audio_codes_list if c is not None]
        if not active_codes:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        step_counter = max(
            (int((info.get("audio_state", {}) or {}).get("step", 0)) for info in info_dicts if isinstance(info, dict)),
            default=0,
        )
        # Single request: emit codes directly (original behavior, proven correct).
        if len(info_dicts) == 1 and len(active_codes) == 1:
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={"codes": {"audio": active_codes[0]}, "meta": {"raw_rows": True, "step": step_counter}},
            )
        # Batch>1: sparse routing with delta rows (only this step's new_codes).
        # Each active request emits [1, n_vq]; stopped requests are skipped.
        req_ids: list[str] = []
        delta_codes: list[torch.Tensor] = []
        for i, info in enumerate(info_dicts):
            code = audio_codes_list[i] if i < len(audio_codes_list) else None
            if code is None or not isinstance(info, dict):
                continue
            rid = str(info.get("request_id") or info.get("global_request_id") or "")
            if rid:
                req_ids.append(rid)
                delta_codes.append(code)
        if not delta_codes:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "codes": {"audio": delta_codes},
                "meta": {"sparse_audio": True, "req_id": req_ids, "raw_rows": True, "step": step_counter},
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
