# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Local Transformer v1.5 AR stage — native vLLM backbone.

Uses vLLM's compiled Qwen3Model with paged attention and CUDA graph
instead of the custom HF-compatible backbone, plus a full-frame CUDA
graph for the local transformer decode (13 sequential RVQ steps captured
as a single graph replay). Achieves ~3ms/token vs ~30ms/token with the
HF-compatible eager path (~9x speedup).

Activated via MOSS_TTS_LOCAL_NATIVE=1 environment variable.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any

import hashlib
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
    for key in ("text_temperature", "audio_temperature", "text_top_k", "audio_top_k",
                "text_top_p", "audio_top_p", "max_new_frames", "min_new_frames"):
        val = info.get(key)
        if val is not None:
            state[key] = _first_scalar(val)


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


def _debug_tensor_digest(tensor: torch.Tensor) -> tuple[str, float, list[float]]:
    sample = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:16]
    norm = float(sample.norm().item())
    head = [float(x) for x in sample.reshape(-1)[:4].tolist()]
    return digest, norm, head


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
            attn_implementation=os.environ.get("MOSS_TTS_LOCAL_ATTN_IMPL") or getattr(self.config, "local_transformer_attn_implementation", None) or "eager",
        )
        self.local_text_lm_head = nn.Linear(self.hidden_size, 2, bias=False)
        self._batch_state: list[dict[str, Any]] | None = None
        self._audio_embedding_indices = torch.arange(self.n_vq, dtype=torch.long)

        # Full-frame CUDA graph: captures the entire _decode_frame_eager as one graph
        self._frame_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._frame_graph_input: torch.Tensor = torch.empty(0)
        self._frame_graph_stop: torch.Tensor = torch.empty(0)
        self._frame_graph_codes: torch.Tensor = torch.empty(0)
        self._frame_graph_max_batch: int = 0
        self._profile_enabled = os.environ.get("MOSS_TTS_LOCAL_PROFILE") == "1"
        self._profile_sync = os.environ.get("MOSS_TTS_LOCAL_PROFILE_SYNC") == "1"
        self._profile_log_every = int(os.environ.get("MOSS_TTS_LOCAL_PROFILE_LOG_EVERY", "100") or 100)
        self._profile_stats: dict[str, Any] = {
            "n_forward": 0,
            "n_decode": 0,
            "qwen_ms": 0.0,
            "local_ms": 0.0,
            "make_output_ms": 0.0,
            "batch_sizes": [],
        }

        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("hidden_states", "last"),
        }

    def _can_profile_sync(self) -> bool:
        if not self._profile_sync or not torch.cuda.is_available():
            return False
        try:
            return not torch.cuda.is_current_stream_capturing()
        except Exception:
            return True

    def _profile_mark(self) -> float:
        if self._can_profile_sync():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _profile_elapsed_ms(self, start: float) -> float:
        if self._can_profile_sync():
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    def _maybe_log_profile(self) -> None:
        if not self._profile_enabled:
            return
        n_decode = int(self._profile_stats["n_decode"])
        n_forward = int(self._profile_stats["n_forward"])
        if n_decode == 0 or n_decode % self._profile_log_every != 0:
            return
        batch_sizes = self._profile_stats["batch_sizes"][-self._profile_log_every :]
        avg_bs = sum(batch_sizes) / max(1, len(batch_sizes))
        logger.info(
            "[moss-local-profile] forward=%d decode=%d avg_bs=%.2f "
            "qwen=%.3fms local=%.3fms make_output=%.3fms sync=%s",
            n_forward,
            n_decode,
            avg_bs,
            self._profile_stats["qwen_ms"] / max(1, n_forward),
            self._profile_stats["local_ms"] / max(1, n_decode),
            self._profile_stats["make_output_ms"] / max(1, n_decode),
            self._profile_sync,
        )

    def _build_input_embeds(self, text_ids: torch.Tensor, audio_codes: torch.Tensor | None) -> torch.Tensor:
        embeds = self.model.embed_tokens(text_ids)
        if audio_codes is None:
            return embeds
        codes = audio_codes.to(device=text_ids.device, dtype=torch.long)
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        valid = codes.ne(self.audio_pad_code)
        safe_codes = codes.clamp(0, self.audio_vocab_size).masked_fill(~valid, 0)
        weights = torch.stack([emb.weight for emb in self.audio_embeddings], dim=0)
        codebook_idx = self._audio_embedding_indices.to(device=text_ids.device)
        audio_embeds = weights[codebook_idx.unsqueeze(0), safe_codes]
        return embeds + (audio_embeds * valid.unsqueeze(-1)).sum(dim=1)

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

            state = {
                "step": 0,
                "is_stopping": False,
                "next_text": self.audio_assistant_slot_token_id,
            }
            _copy_sampling_overrides(state, info_dict)
            return input_ids, embeds, {
                "audio_state": state,
                "audio_codes": {"current": current_codes},
            }

        prev_codes = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(prev_codes, torch.Tensor) or prev_codes.numel() != self.n_vq:
            prev_codes = torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
        embeds = self._build_input_embeds(input_ids.reshape(-1), prev_codes.to(device=device).unsqueeze(0))
        return input_ids, embeds, {}

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

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor | None:
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
            rows = []
            for token in next_text_from_output.reshape(-1).tolist():
                one_hot = torch.full((self.vocab_size,), -1e9, device=hidden_states.device)
                one_hot[int(token)] = 0.0
                rows.append(one_hot)
            self._batch_state = None
            if _debug_state_enabled():
                logger.info(
                    "[moss-local-state] compute_logits source=meta rows=%d next=%s",
                    len(rows),
                    next_text_from_output.reshape(-1).detach().cpu().tolist(),
                )
            return torch.stack(rows)
        batch_state = self._batch_state
        if not batch_state:
            num_rows = hidden_states.shape[0] if hidden_states.dim() >= 1 else 1
            if _debug_state_enabled():
                logger.info("[moss-local-state] compute_logits source=zeros rows=%d", num_rows)
            return torch.zeros(num_rows, self.vocab_size, device=hidden_states.device)
        logit_rows: list[torch.Tensor] = []
        debug_next: list[int | None] = []
        for i, state in enumerate(batch_state):
            if not isinstance(state, dict):
                logit_rows.append(torch.zeros(self.vocab_size, device=hidden_states.device))
                debug_next.append(None)
                continue
            next_text = state.get("next_text")
            if next_text is not None:
                one_hot = torch.full((self.vocab_size,), -1e9, device=hidden_states.device)
                one_hot[int(next_text)] = 0.0
                logit_rows.append(one_hot)
                debug_next.append(int(next_text))
            else:
                logit_rows.append(torch.zeros(self.vocab_size, device=hidden_states.device))
                debug_next.append(None)
        self._batch_state = None
        if _debug_state_enabled():
            logger.info(
                "[moss-local-state] compute_logits source=batch_state rows=%d next=%s",
                len(logit_rows),
                debug_next,
            )
        return torch.stack(logit_rows)

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

        params = [self._frame_params(infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}) for i in range(B)]
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
                    text_logits[b:b + 1],
                    temperature=text_temp,
                    top_p=text_top_p,
                    top_k=text_top_k,
                    generator=generator,
                )[0]

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
                        logits[b:b + 1],
                        temperature=audio_temp,
                        top_p=audio_top_p,
                        top_k=audio_top_k,
                        generator=generator,
                    )[0]
            all_codes[:, channel] = codes
            if channel + 1 < self.n_vq:
                embedded = F.embedding(codes, head_weight).to(dtype=current.dtype)
                current = self.local_transformer._step_eager(embedded, channel + 1)
        return stop_choices, all_codes

    def _ensure_frame_graph_buffers(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if batch_size <= self._frame_graph_max_batch:
            return
        cap = max(batch_size, 8)
        self._frame_graph_input = torch.zeros(cap, self.hidden_size, device=device, dtype=dtype)
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
        self.local_transformer._ensure_kv_cache(B, hidden_batch.device, dtype)

        if hidden_batch.device.type != "cuda":
            return self._decode_frame_eager(hidden_batch, infos)
        if os.environ.get("MOSS_TTS_LOCAL_ENABLE_FRAME_GRAPH") != "1":
            return self._decode_frame_eager(hidden_batch, infos)

        params = [
            self._frame_params(info if isinstance(info, dict) else {})
            for info in infos[:B]
        ]
        if len(params) != B:
            params.extend([self._frame_params({}) for _ in range(B - len(params))])
        first_params = params[0][:6] if params else (1.0, 1.0, 50, 1.0, 0.95, 50)
        graphable = (
            float(first_params[0]) <= 0
            and float(first_params[3]) <= 0
            and all(item[:6] == first_params and item[6] is None for item in params)
        )
        if not graphable:
            return self._decode_frame_eager(hidden_batch, infos)

        self._ensure_frame_graph_buffers(B, hidden_batch.device, dtype)
        graph_key = (B, first_params)
        if graph_key not in self._frame_graphs:
            self._frame_graph_input[:B].copy_(hidden_batch)
            self._decode_frame_eager(self._frame_graph_input[:B], infos)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                stop, codes = self._decode_frame_eager(self._frame_graph_input[:B], infos)
                self._frame_graph_stop[:B].copy_(stop)
                self._frame_graph_codes[:B].copy_(codes)
            self._frame_graphs[graph_key] = g
            return self._frame_graph_stop[:B].clone(), self._frame_graph_codes[:B].clone()

        self._frame_graph_input[:B].copy_(hidden_batch)
        self._frame_graphs[graph_key].replay()
        return self._frame_graph_stop[:B].clone(), self._frame_graph_codes[:B].clone()

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
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
        if isinstance(logits_index, torch.Tensor) and logits_index.numel() > 0:
            logits_rows = logits_index.detach().to("cpu", dtype=torch.long).reshape(-1).tolist()
            sample_hidden = hidden[logits_index.to(device=hidden.device, dtype=torch.long)]
        elif isinstance(logits_index, int):
            logits_rows = [int(logits_index)]
            sample_hidden = hidden[logits_index:logits_index + 1]

        query_start_loc = kwargs.get("omni_query_start_loc")
        request_token_spans = kwargs.get("request_token_spans")
        span_lens: list[int] | None = None
        qsl_cpu: list[int] | None = None
        if isinstance(query_start_loc, torch.Tensor) and query_start_loc.numel() >= len(info_dicts) + 1:
            qsl_cpu = query_start_loc[:len(info_dicts) + 1].detach().to("cpu").tolist()
            span_lens = [int(qsl_cpu[i + 1]) - int(qsl_cpu[i]) for i in range(len(info_dicts))]
            if not isinstance(request_token_spans, (list, tuple)):
                request_token_spans = [(int(qsl_cpu[i]), int(qsl_cpu[i + 1])) for i in range(len(info_dicts))]
        elif (
            isinstance(request_token_spans, (list, tuple))
            and len(request_token_spans) >= len(info_dicts)
        ):
            span_lens = [
                int(end) - int(start)
                for start, end in request_token_spans[:len(info_dicts)]
            ]
        if _debug_state_enabled():
            if isinstance(logits_index, torch.Tensor):
                logits_index_dbg = logits_index.detach().to("cpu").reshape(-1).tolist()
            else:
                logits_index_dbg = logits_index
            logger.info(
                "[moss-local-state] make_output enter hidden=%s sample=%s infos=%d qsl=%s logits_index=%s spans=%s reqs=%s",
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
                sampling_generator = _make_sampling_generator(info.get("seed"), hidden.device)
                if sampling_generator is not None:
                    state["sampling_generator"] = sampling_generator
                info["audio_state"] = state

        audio_codes_list: list[torch.Tensor | None] = [None for _ in info_dicts]
        batch_state_by_sample_row: list[dict[str, Any] | None] = []
        if hidden.numel() > 0 and info_dicts:
            num_rows = int(sample_hidden.shape[0])
            batch_state_by_sample_row = [None for _ in range(num_rows)]
            logits_row_to_sample_row = (
                {int(row): pos for pos, row in enumerate(logits_rows)}
                if logits_rows is not None
                else None
            )
            decode_items: list[tuple[int, int]] = []

            def sample_row_for_request(req_index: int) -> int | None:
                if (
                    isinstance(request_token_spans, (list, tuple))
                    and req_index < len(request_token_spans)
                ):
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
                    continue

                row_idx = sample_row_for_request(i)
                if row_idx is None or row_idx < 0 or row_idx >= num_rows:
                    continue
                if span_lens is not None and span_lens[i] != 1:
                    # This request is still consuming a prefill span. The runner
                    # will sample one text token for it, but MOSS local should not
                    # generate an audio frame until the next decode-only step.
                    state["next_text"] = self.audio_assistant_slot_token_id
                    batch_state_by_sample_row[row_idx] = state
                    continue
                batch_state_by_sample_row[row_idx] = state
                decode_items.append((i, row_idx))

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

            if (
                use_batched
                and len(decode_items) > 1
                and os.environ.get("MOSS_TTS_LOCAL_DISABLE_LOCAL_BATCH") != "1"
            ):
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
                hidden_batch = torch.stack([sample_hidden[row_idx] for _, row_idx in decode_items])
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
                batch_min_frames = int(os.environ.get("MOSS_TTS_LOCAL_BATCH_MIN_FRAMES", "80") or 80)
                step_nums = torch.tensor(
                    [int((info_dicts[i].get("audio_state", {}) or {}).get("step", 0)) for i, _ in decode_items],
                    dtype=torch.long,
                    device=stop_choices.device,
                )
                min_frames_t = torch.tensor(
                    [max(int((info_dicts[i].get("audio_state", {}) or {}).get("min_new_frames", 3)), batch_min_frames) for i, _ in decode_items],
                    dtype=torch.long,
                    device=stop_choices.device,
                )
                max_frames_t = torch.tensor(
                    [int((info_dicts[i].get("audio_state", {}) or {}).get("max_new_frames", 150)) for i, _ in decode_items],
                    dtype=torch.long,
                    device=stop_choices.device,
                )
                should_stop_tensor = ((stop_choices == 1) & (step_nums >= min_frames_t)) | (step_nums >= max_frames_t)
                should_stop_list = should_stop_tensor.detach().cpu().tolist()
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
                    step_num = int(step_nums[bi].item())
                    max_frames = int(max_frames_t[bi].item())
                    should_stop = bool(should_stop_list[bi])
                    if should_stop:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        state["stop_reason"] = "max_new_frames" if step_num >= max_frames else "model_stop"
                        info["audio_codes"] = {"current": new_codes}
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list[i] = new_codes.unsqueeze(0)
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
                                "[moss-local-state] final req=%s frames=%d reason=%s hash=%016x",
                                _request_label(info),
                                step_num + 1,
                                state.get("stop_reason"),
                                debug_hash,
                            )
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
                    stop_choices, all_codes = self._decode_frame_batched(sample_hidden[row_idx:row_idx + 1], [info])
                    if self._profile_enabled:
                        elapsed = self._profile_elapsed_ms(local_start)
                        self._profile_stats["local_ms"] += elapsed
                        profile_local_ms += elapsed
                        self._profile_stats["n_decode"] += 1
                        self._profile_stats["batch_sizes"].append(1)
                    new_codes = all_codes[0]
                    debug_hash = _update_debug_code_hash(state, new_codes) if _debug_state_enabled() else 0
                    step_num = int(state.get("step", 0))
                    min_frames = int(state.get("min_new_frames", 3))
                    max_frames = int(state.get("max_new_frames", 150))
                    should_stop = (int(stop_choices[0].item()) == 1 and step_num >= min_frames) or step_num >= max_frames
                    if should_stop:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        state["stop_reason"] = "max_new_frames" if step_num >= max_frames else "model_stop"
                        info["audio_codes"] = {"current": new_codes}
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list[i] = new_codes.unsqueeze(0)
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
                                "[moss-local-state] final req=%s frames=%d reason=%s hash=%016x",
                                _request_label(info),
                                step_num + 1,
                                state.get("stop_reason"),
                                debug_hash,
                            )

        self._batch_state = [
            (state if isinstance(state, dict) else {"next_text": self.audio_end_token_id})
            for state in batch_state_by_sample_row
        ]
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
                self._profile_stats["make_output_ms"] += max(
                    0.0, self._profile_elapsed_ms(profile_start) - profile_local_ms
                )
                self._maybe_log_profile()
            return output
        req_ids: list[str] = []
        delta_codes: list[torch.Tensor] = []
        delta_steps: list[int] = []
        for i, info in enumerate(info_dicts):
            code = audio_codes_list[i] if i < len(audio_codes_list) else None
            if code is None or not isinstance(info, dict):
                continue
            rid = str(info.get("request_id") or info.get("global_request_id") or "")
            if rid:
                state = info.get("audio_state", {}) or {}
                req_ids.append(rid)
                delta_codes.append(code)
                delta_steps.append(max(0, int(state.get("step", 0)) - 1))
        if not delta_codes:
            output = OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
            if self._profile_enabled:
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
                    "req_id": req_ids,
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
                name = "model." + name[len("transformer."):]

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
