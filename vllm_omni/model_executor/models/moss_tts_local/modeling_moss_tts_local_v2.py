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
    for key in ("text_temperature", "audio_temperature", "text_top_k", "audio_top_k",
                "text_top_p", "audio_top_p", "max_new_frames", "min_new_frames"):
        val = info.get(key)
        if val is not None:
            state[key] = _first_scalar(val)


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

        # Full-frame CUDA graph: captures the entire _decode_frame_eager as one graph
        self._frame_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._frame_graph_input: torch.Tensor = torch.empty(0)
        self._frame_graph_stop: torch.Tensor = torch.empty(0)
        self._frame_graph_codes: torch.Tensor = torch.empty(0)
        self._frame_graph_max_batch: int = 0

        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("hidden_states", "last"),
        }

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
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states.numel() == 0:
            return torch.zeros(1, self.vocab_size, device=hidden_states.device)
        batch_state = self._batch_state
        if not batch_state:
            num_rows = hidden_states.shape[0] if hidden_states.dim() >= 1 else 1
            return torch.zeros(num_rows, self.vocab_size, device=hidden_states.device)
        logit_rows: list[torch.Tensor] = []
        for i, state in enumerate(batch_state):
            if not isinstance(state, dict):
                logit_rows.append(torch.zeros(self.vocab_size, device=hidden_states.device))
                continue
            next_text = state.get("next_text")
            if next_text is not None:
                one_hot = torch.full((self.vocab_size,), -1e9, device=hidden_states.device)
                one_hot[int(next_text)] = 0.0
                logit_rows.append(one_hot)
            else:
                logit_rows.append(torch.zeros(self.vocab_size, device=hidden_states.device))
        self._batch_state = None
        return torch.stack(logit_rows)

    def _decode_frame_eager(self, hidden_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure computation path (no graph logic) for capture or eager fallback."""
        B = hidden_batch.shape[0]
        dtype = self.audio_embeddings[0].weight.dtype
        local_hidden = self.local_transformer._step_eager(hidden_batch.to(dtype=dtype), 0)
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()
        stop_choices = sample_top_k_top_p(text_logits, temperature=1.0, top_p=1.0, top_k=50)
        all_codes = torch.zeros(B, self.n_vq, dtype=torch.long, device=hidden_batch.device)
        current = local_hidden
        for channel in range(self.n_vq):
            head_weight = self.audio_embeddings[channel].weight[: self.audio_vocab_size]
            logits = F.linear(current, head_weight).float()
            codes = sample_top_k_top_p(logits, temperature=1.0, top_p=0.95, top_k=50)
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
            return self._decode_frame_eager(hidden_batch)

        self._ensure_frame_graph_buffers(B, hidden_batch.device, dtype)
        graph_key = B
        if graph_key not in self._frame_graphs:
            self._frame_graph_input[:B].copy_(hidden_batch)
            self._decode_frame_eager(self._frame_graph_input[:B])
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                stop, codes = self._decode_frame_eager(self._frame_graph_input[:B])
                self._frame_graph_stop[:B].copy_(stop)
                self._frame_graph_codes[:B].copy_(codes)
            self._frame_graphs[graph_key] = g
            return self._frame_graph_stop[:B].clone(), self._frame_graph_codes[:B].clone()

        self._frame_graph_input[:B].copy_(hidden_batch)
        self._frame_graphs[graph_key].replay()
        return self._frame_graph_stop[:B].clone(), self._frame_graph_codes[:B].clone()

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
                info["audio_state"] = state

        audio_codes_list: list[torch.Tensor | None] = []
        if hidden.numel() > 0 and info_dicts:
            num_rows = int(hidden.shape[0])
            active_indices: list[int] = []
            for i, info in enumerate(info_dicts):
                if not isinstance(info, dict):
                    continue
                state = info.get("audio_state", {}) or {}
                if not state.get("is_stopping"):
                    active_indices.append(i)

            use_batched = len(active_indices) > 0 and num_rows == len(info_dicts)

            if use_batched and len(active_indices) > 1:
                hidden_batch = torch.stack([hidden[j] for j in active_indices])
                stop_choices, all_codes = self._decode_frame_batched(hidden_batch, [info_dicts[j] for j in active_indices])
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
                    max_frames = int(state.get("max_new_frames", 150))
                    should_stop = (stop_list[bi] == 1 and step_num >= min_frames) or step_num >= max_frames
                    if should_stop:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        state["stop_reason"] = "max_new_frames" if step_num >= max_frames else "model_stop"
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list.append(None)
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list.append(new_codes.unsqueeze(0))
                    bi += 1
            elif use_batched:
                # Single active request
                for i, info in enumerate(info_dicts):
                    if not isinstance(info, dict):
                        audio_codes_list.append(None)
                        continue
                    state = info.get("audio_state", {}) or {}
                    if state.get("is_stopping"):
                        state["next_text"] = self.audio_end_token_id
                        audio_codes_list.append(None)
                        continue
                    stop_choices, all_codes = self._decode_frame_batched(hidden[i:i+1], [info])
                    new_codes = all_codes[0]
                    step_num = int(state.get("step", 0))
                    min_frames = int(state.get("min_new_frames", 3))
                    max_frames = int(state.get("max_new_frames", 150))
                    should_stop = (int(stop_choices[0].item()) == 1 and step_num >= min_frames) or step_num >= max_frames
                    if should_stop:
                        state["is_stopping"] = True
                        state["next_text"] = self.audio_end_token_id
                        state["stop_reason"] = "max_new_frames" if step_num >= max_frames else "model_stop"
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list.append(None)
                    else:
                        state["next_text"] = self.audio_assistant_slot_token_id
                        state["step"] = step_num + 1
                        info["audio_codes"] = {"current": new_codes}
                        audio_codes_list.append(new_codes.unsqueeze(0))

        self._batch_state = [(info.get("audio_state", {}) if isinstance(info, dict) else {}) for info in info_dicts]
        active_codes = [c for c in audio_codes_list if c is not None]
        if not active_codes:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        step_counter = max(
            (int((info.get("audio_state", {}) or {}).get("step", 0)) for info in info_dicts if isinstance(info, dict)),
            default=0,
        )
        if len(info_dicts) == 1 and len(active_codes) == 1:
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={"codes": {"audio": active_codes[0]}, "meta": {"raw_rows": True, "step": step_counter}},
            )
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
