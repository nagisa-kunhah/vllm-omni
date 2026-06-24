# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""HF-compatible Qwen3 backbone for MOSS-TTS Local stage-0.

MOSS-TTS Local audio quality is sensitive to small global-backbone numerical
differences because the downstream local transformer autoregressively amplifies
argmax changes across codebooks.  This module mirrors the reference PyTorch
Qwen3 decoder while keeping the vLLM-Omni stage interface unchanged.  It is
owns its KV cache internally.  Static KV cache supports multiple request slots
for batched TTS decode while preserving the legacy slot-0 path.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class MossQwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MossQwen3RotaryEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict):
                rope_theta = rope_scaling.get("rope_theta")
        self.head_dim = head_dim
        self.rope_theta = float(rope_theta if rope_theta is not None else 1_000_000.0)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._compute_inv_freq(device=hidden_states.device)
        freqs = torch.einsum(
            "bs,d->bsd",
            position_ids.to(device=hidden_states.device, dtype=inv_freq.dtype),
            inv_freq,
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=hidden_states.dtype), emb.sin().to(dtype=hidden_states.dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, seq_len, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, seq_len, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, seq_len, num_key_value_heads * n_rep, head_dim)


class MossQwen3MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MossQwen3Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.static_key_cache: torch.Tensor | None = None
        self.static_value_cache: torch.Tensor | None = None
        self.static_cache_positions: torch.Tensor | None = None
        self._slot_cache_lens: list[int] = [0]
        self.active_slot = 0
        self.use_static_full_cache = False

        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)
        self.q_norm = MossQwen3RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.k_norm = MossQwen3RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))

    def forward_batched_decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        slot_ids: list[int],
        slot_idx_t: torch.Tensor | None = None,
        step_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        cache_capacity = int(self.static_key_cache.shape[1])
        bp = self._backbone_profile if getattr(self, "_backbone_profile_enabled", False) else None

        if bp is not None:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(batch_size, 1, self.num_heads, self.head_dim)
        )
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(batch_size, 1, self.num_key_value_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(batch_size, 1, self.num_key_value_heads, self.head_dim)

        if bp is not None:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            bp["qkv"] += (_t1 - _t0) * 1000

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if bp is not None:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()
            bp["rope"] += (_t2 - _t1) * 1000

        starts = [self._slot_cache_lens[sid] for sid in slot_ids]
        for i, (sid, start) in enumerate(zip(slot_ids, starts)):
            self.static_key_cache[sid, start] = key_states[i, 0]
            self.static_value_cache[sid, start] = value_states[i, 0]
            self._slot_cache_lens[sid] = start + 1

        if bp is not None:
            torch.cuda.synchronize()
            _t3 = time.perf_counter()
            bp["kv_write"] += (_t3 - _t2) * 1000

        if step_cache is not None and "mask" in step_cache:
            mask = step_cache["mask"]
            max_len = int(step_cache["max_len"])
        else:
            cache_lens = [self._slot_cache_lens[sid] for sid in slot_ids]
            max_len = max(cache_lens)
            lens_t = torch.tensor(cache_lens, device=hidden_states.device, dtype=torch.long)
            positions = torch.arange(max_len, device=hidden_states.device, dtype=torch.long)
            mask = (positions.unsqueeze(0) < lens_t.unsqueeze(1))[:, None, None, :]
            if step_cache is not None:
                step_cache["mask"] = mask
                step_cache["max_len"] = torch.tensor(max_len)

        if slot_idx_t is None:
            slot_idx_t = torch.tensor(slot_ids, device=hidden_states.device, dtype=torch.long)
        k_batch = self.static_key_cache[slot_idx_t, :max_len]
        v_batch = self.static_value_cache[slot_idx_t, :max_len]

        if bp is not None:
            torch.cuda.synchronize()
            _t4 = time.perf_counter()
            bp["kv_gather"] += (_t4 - _t3) * 1000

        k_batch = _repeat_kv(k_batch, self.num_key_value_groups).transpose(1, 2)
        v_batch = _repeat_kv(v_batch, self.num_key_value_groups).transpose(1, 2)
        query = query_states.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            query, k_batch, v_batch, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=self.scaling
        )

        if bp is not None:
            torch.cuda.synchronize()
            _t5 = time.perf_counter()
            bp["sdpa"] += (_t5 - _t4) * 1000

        output = output.transpose(1, 2).reshape(batch_size, 1, -1).contiguous()
        result = self.o_proj(output)

        if bp is not None:
            torch.cuda.synchronize()
            _t6 = time.perf_counter()
            bp["o_proj"] += (_t6 - _t5) * 1000

        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(*input_shape, self.num_heads, self.head_dim)
        )
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.static_key_cache is not None and self.static_value_cache is not None:
            slot_id = int(self.active_slot)
            if self.use_static_full_cache:
                assert self.static_cache_positions is not None
                self.static_key_cache[: key_states.shape[0]].index_copy_(1, self.static_cache_positions, key_states)
                self.static_value_cache[: value_states.shape[0]].index_copy_(1, self.static_cache_positions, value_states)
                key_states = self.static_key_cache[: key_states.shape[0]]
                value_states = self.static_value_cache[: value_states.shape[0]]
            else:
                if slot_id < 0 or slot_id >= len(self._slot_cache_lens):
                    raise IndexError(f"active_slot {slot_id} is outside static KV cache slots")
                start = int(self._slot_cache_lens[slot_id])
                end = start + int(key_states.shape[1])
                cache_capacity = int(self.static_key_cache.shape[1])
                if end > cache_capacity:
                    end = cache_capacity
                    key_states = key_states[:, : end - start]
                    value_states = value_states[:, : end - start]
                    if end <= start:
                        key_states = self.static_key_cache[slot_id : slot_id + 1, :start]
                        value_states = self.static_value_cache[slot_id : slot_id + 1, :start]
                        present = (key_states, value_states) if use_cache else None
                        key = _repeat_kv(key_states, self.num_key_value_groups).transpose(1, 2)
                        value = _repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2)
                        query = query_states.transpose(1, 2)
                        output = F.scaled_dot_product_attention(query, key, value, is_causal=False, scale=self.scaling)
                        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                        return self.o_proj(output), present
                else:
                    self.static_key_cache[slot_id : slot_id + 1, start:end].copy_(key_states)
                    self.static_value_cache[slot_id : slot_id + 1, start:end].copy_(value_states)
                    key_states = self.static_key_cache[slot_id : slot_id + 1, :end]
                    value_states = self.static_value_cache[slot_id : slot_id + 1, :end]
                    self._slot_cache_lens[slot_id] = end
        elif layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat([past_key.to(device=key_states.device, dtype=key_states.dtype), key_states], dim=1)
            value_states = torch.cat(
                [past_value.to(device=value_states.device, dtype=value_states.dtype), value_states],
                dim=1,
            )

        present = (key_states, value_states) if use_cache else None
        key = _repeat_kv(key_states, self.num_key_value_groups).transpose(1, 2)
        value = _repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2)
        query = query_states.transpose(1, 2)

        if self.use_static_full_cache:
            assert self.static_cache_positions is not None
            query_positions = self.static_cache_positions[-query.shape[-2] :]
            key_positions = torch.arange(key.shape[-2], device=query.device, dtype=torch.long)
        else:
            query_positions = torch.arange(query.shape[-2], device=query.device, dtype=torch.long)
            query_positions = query_positions + max(key.shape[-2] - query.shape[-2], 0)
            key_positions = torch.arange(key.shape[-2], device=query.device, dtype=torch.long)
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
        )
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), present


class MossQwen3DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = MossQwen3Attention(config)
        self.mlp = MossQwen3MLP(config)
        self.input_layernorm = MossQwen3RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))
        self.post_attention_layernorm = MossQwen3RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))

    def forward_batched_decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        slot_ids: list[int],
        slot_idx_t: torch.Tensor | None = None,
        step_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bp = getattr(self.self_attn, "_backbone_profile", None) if getattr(self.self_attn, "_backbone_profile_enabled", False) else None
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn.forward_batched_decode(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            slot_ids=slot_ids,
            slot_idx_t=slot_idx_t,
            step_cache=step_cache,
        )
        hidden_states = residual + attn_output

        if bp is not None:
            torch.cuda.synchronize()
            _tm0 = time.perf_counter()

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        if bp is not None:
            torch.cuda.synchronize()
            bp["norm_mlp"] += (time.perf_counter() - _tm0) * 1000

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class MossTTSLocalQwen3Backbone(nn.Module):
    """Reference-numerics Qwen3 backbone with an internal multi-slot KV cache."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MossQwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = MossQwen3RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))
        self.rotary_emb = MossQwen3RotaryEmbedding(config)
        self._past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
        self._static_cache_bucket_len: int = 0
        self._decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._decode_static_input: torch.Tensor | None = None
        self._decode_static_output: torch.Tensor | None = None
        self._decode_static_position: torch.Tensor | None = None
        self._decode_graph_enabled: bool = os.environ.get("MOSS_TTS_LOCAL_DECODE_GRAPH") == "1"
        self.active_slot: int = 0
        self._backbone_profile_enabled: bool = os.environ.get("MOSS_TTS_LOCAL_BACKBONE_PROFILE") == "1"
        self._backbone_profile: dict[str, float] = {"qkv": 0, "rope": 0, "kv_write": 0, "kv_gather": 0, "sdpa": 0, "o_proj": 0, "norm_mlp": 0, "n": 0}
        self._compile_enabled: bool = os.environ.get("MOSS_TTS_LOCAL_COMPILE") == "1"
        self._compiled_step: Any = None

    def reset_slot(self, slot_id: int) -> None:
        self._past_key_values = None
        self.active_slot = int(slot_id)
        for layer in self.layers:
            attn = layer.self_attn
            if attn.static_key_cache is not None:
                if slot_id < 0 or slot_id >= len(attn._slot_cache_lens):
                    raise IndexError(f"slot_id {slot_id} is outside static KV cache slots")
                attn._slot_cache_lens[slot_id] = 0
            attn.active_slot = int(slot_id)
            attn.use_static_full_cache = False

    def reset_cache(self) -> None:
        self._past_key_values = None
        self.active_slot = 0
        for layer in self.layers:
            attn = layer.self_attn
            attn._slot_cache_lens = [0 for _ in attn._slot_cache_lens]
            attn.active_slot = 0
            attn.use_static_full_cache = False

    def init_static_kv_cache(
        self,
        *,
        max_len: int,
        max_batch: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int | None = None,
    ) -> None:
        if max_batch is None:
            if batch_size is None:
                raise TypeError("init_static_kv_cache() missing required argument: 'max_batch'")
            max_batch = batch_size
        current_max_batch = 0
        if self.layers and self.layers[0].self_attn.static_key_cache is not None:
            current_max_batch = int(self.layers[0].self_attn.static_key_cache.shape[0])
        if self._static_cache_bucket_len >= max_len and current_max_batch >= int(max_batch):
            return
        self._decode_graphs.clear()
        self._decode_static_input = None
        self._decode_static_output = None
        self._decode_static_position = None
        positions = torch.empty(1, device=device, dtype=torch.long)
        for layer in self.layers:
            attn = layer.self_attn
            shape = (int(max_batch), int(max_len), attn.num_key_value_heads, attn.head_dim)
            attn.static_key_cache = torch.zeros(shape, device=device, dtype=dtype)
            attn.static_value_cache = torch.zeros(shape, device=device, dtype=dtype)
            attn.static_cache_positions = positions
            attn._slot_cache_lens = [0 for _ in range(int(max_batch))]
            attn.active_slot = 0
            attn.use_static_full_cache = False
            attn._backbone_profile_enabled = self._backbone_profile_enabled
            attn._backbone_profile = self._backbone_profile
        self._static_cache_bucket_len = max_len
        self._past_key_values = None

    def warmup_decode_graphs(self, bucket_sizes: list[int], device: torch.device, dtype: torch.dtype, max_batch: int = 1) -> None:
        if not self._decode_graph_enabled:
            return
        for bucket in bucket_sizes:
            self.init_static_kv_cache(max_len=bucket, max_batch=max_batch, device=device, dtype=dtype)
            dummy_ids = torch.zeros(2, dtype=torch.long, device=device)
            dummy_pos = torch.arange(2, dtype=torch.long, device=device)
            self.forward(input_ids=dummy_ids, positions=dummy_pos)
            dummy_token = torch.zeros(1, dtype=torch.long, device=device)
            dummy_step_pos = torch.tensor([2], dtype=torch.long, device=device)
            self.forward(input_ids=dummy_token, positions=dummy_step_pos)
            if bucket in self._decode_graphs:
                continue
        self.reset_cache()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _forward_eager_batched(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        slot_ids: list[int],
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        device = hidden_states.device
        slot_idx_t = torch.tensor(slot_ids, device=device, dtype=torch.long)
        step_cache: dict[str, torch.Tensor] = {}
        for decoder_layer in self.layers:
            hidden_states = decoder_layer.forward_batched_decode(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                slot_ids=slot_ids,
                slot_idx_t=slot_idx_t,
                step_cache=step_cache,
            )
        result = self.norm(hidden_states)
        bp = self._backbone_profile
        if self._backbone_profile_enabled:
            bp["n"] += 1
            if int(bp["n"]) % 50 == 0:
                n = int(bp["n"])
                nl = len(self.layers)
                total = sum(bp[k] for k in ["qkv", "rope", "kv_write", "kv_gather", "sdpa", "o_proj", "norm_mlp"]) / n
                import sys
                print(
                    f"BackboneProfile[{n} steps, {nl} layers]: qkv={bp['qkv']/n:.2f}ms rope={bp['rope']/n:.2f}ms "
                    f"kv_write={bp['kv_write']/n:.2f}ms kv_gather={bp['kv_gather']/n:.2f}ms "
                    f"sdpa={bp['sdpa']/n:.2f}ms o_proj={bp['o_proj']/n:.2f}ms norm_mlp={bp['norm_mlp']/n:.2f}ms | "
                    f"total={total:.2f}ms/step",
                    file=sys.stderr, flush=True,
                )
        return result


    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            layer.self_attn.active_slot = self.active_slot
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, decoder_layer in enumerate(self.layers):
            layer_past = None if self._past_key_values is None else self._past_key_values[layer_index]
            hidden_states, present = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                layer_past=layer_past,
                use_cache=True,
            )
            assert present is not None
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        if self.layers[0].self_attn.static_key_cache is None:
            self._past_key_values = tuple(presents)
        else:
            self._past_key_values = None
        return hidden_states

    def _ensure_decode_graph_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._decode_static_input is not None:
            return
        hidden_size = int(self.config.hidden_size)
        self._decode_static_input = torch.zeros(1, 1, hidden_size, device=device, dtype=dtype)
        self._decode_static_position = torch.zeros(1, 1, device=device, dtype=torch.long)
        self._decode_static_output = torch.zeros(1, 1, hidden_size, device=device, dtype=dtype)

    def _switch_to_full_cache_mode(self) -> None:
        pos_tensor = self._decode_static_position.reshape(-1)
        for layer in self.layers:
            attn = layer.self_attn
            attn.use_static_full_cache = True
            attn.static_cache_positions = pos_tensor

    def _clear_full_cache_mode(self) -> None:
        for layer in self.layers:
            layer.self_attn.use_static_full_cache = False

    def _capture_decode_graph(self, bucket: int) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self._forward_eager(self._decode_static_input, self._decode_static_position)
            self._decode_static_output.copy_(out)
        self._decode_graphs[bucket] = g
        return g

    def _ensure_batch_decode_buffers(self, max_batch: int, device: torch.device, dtype: torch.dtype) -> None:
        if getattr(self, "_batch_static_input", None) is not None and self._batch_static_input.shape[0] >= max_batch:
            return
        hidden_size = int(self.config.hidden_size)
        self._batch_static_input = torch.zeros(max_batch, 1, hidden_size, device=device, dtype=dtype)
        self._batch_static_pos = torch.zeros(max_batch, 1, device=device, dtype=torch.long)
        self._batch_static_output = torch.zeros(max_batch, 1, hidden_size, device=device, dtype=dtype)
        self._batch_static_slots = torch.zeros(max_batch, device=device, dtype=torch.long)
        self._batch_decode_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}

    def _decode_batched_with_graph(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        slot_ids: list[int],
    ) -> torch.Tensor:
        n = len(slot_ids)
        bucket = self._static_cache_bucket_len
        key = (n, bucket)
        self._ensure_batch_decode_buffers(n, hidden_states.device, hidden_states.dtype)
        self._batch_static_input[:n].copy_(hidden_states)
        self._batch_static_pos[:n].copy_(position_ids)
        self._batch_static_slots[:n].copy_(torch.tensor(slot_ids, device=hidden_states.device, dtype=torch.long))

        if not hasattr(self, "_batch_decode_graphs"):
            self._batch_decode_graphs = {}

        if key not in self._batch_decode_graphs:
            out = self._forward_eager_batched(
                self._batch_static_input[:n], self._batch_static_pos[:n], slot_ids
            )
            self._batch_static_output[:n].copy_(out)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._forward_eager_batched(
                    self._batch_static_input[:n], self._batch_static_pos[:n], slot_ids
                )
                self._batch_static_output[:n].copy_(out)
            self._batch_decode_graphs[key] = g
            return self._batch_static_output[:n].clone()

        self._batch_decode_graphs[key].replay()
        return self._batch_static_output[:n].clone()

    def _decode_with_graph(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bucket = self._static_cache_bucket_len
        self._ensure_decode_graph_buffers(hidden_states.device, hidden_states.dtype)
        self._switch_to_full_cache_mode()
        try:
            if bucket not in self._decode_graphs:
                self._decode_static_input.copy_(hidden_states)
                self._decode_static_position.copy_(position_ids)
                out = self._forward_eager(self._decode_static_input, self._decode_static_position)
                self._decode_static_output.copy_(out)
                self._capture_decode_graph(bucket)
                return self._decode_static_output.clone()

            self._decode_static_input.copy_(hidden_states)
            self._decode_static_position.copy_(position_ids)
            self._decode_graphs[bucket].replay()
            return self._decode_static_output.clone()
        finally:
            self._clear_full_cache_mode()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        slot_id: int | None = None,
        slot_ids: list[int] | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors

        # Batched decode path: N requests each with 1 token
        if slot_ids is not None and len(slot_ids) > 1:
            n = len(slot_ids)
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds must be provided")
                inputs_embeds = self.embed_input_ids(input_ids)
            hidden_states = inputs_embeds.view(n, 1, -1)
            position_ids = positions.view(n, 1).to(device=hidden_states.device, dtype=torch.long)

            # Batch CUDA graph is disabled: per-slot dynamic KV indexing and mask
            # computation cannot be captured in a static graph. Eager batched matmul
            # already provides ~2x throughput over sequential.
            hidden_states = self._forward_eager_batched(hidden_states, position_ids, slot_ids)
            return hidden_states.squeeze(1)

        # Single-slot path (prefill or batch=1 decode)
        self.active_slot = 0 if slot_id is None else int(slot_id)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_input_ids(input_ids)

        squeeze_batch = inputs_embeds.dim() == 2
        if squeeze_batch:
            hidden_states = inputs_embeds.unsqueeze(0)
        else:
            hidden_states = inputs_embeds
        position_ids = positions.reshape(1, -1).to(device=hidden_states.device, dtype=torch.long)
        if position_ids.numel() != hidden_states.shape[1]:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long).view(1, -1)

        is_decode = position_ids.numel() == 1
        if not is_decode:
            self.reset_slot(self.active_slot)

        use_graph = (
            self.active_slot == 0
            and hidden_states.shape[0] == 1
            and is_decode
            and self._decode_graph_enabled
            and self._static_cache_bucket_len > 0
            and self.layers[0].self_attn.static_key_cache is not None
        )

        if use_graph:
            try:
                hidden_states = self._decode_with_graph(hidden_states, position_ids)
            except Exception:
                self._decode_graph_enabled = False
                self._decode_graphs.clear()
                self._clear_full_cache_mode()
                hidden_states = self._forward_eager(hidden_states, position_ids)
        else:
            hidden_states = self._forward_eager(hidden_states, position_ids)

        return hidden_states.squeeze(0) if squeeze_batch else hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            target = name
            if target.startswith("model."):
                target = target[len("model.") :]
            if target.startswith("transformer."):
                target = target[len("transformer.") :]
            if target.startswith("model."):
                target = target[len("model.") :]
            if "rotary_emb.inv_freq" in target:
                continue
            param = params_dict.get(target)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(target)
        return loaded


__all__ = ["MossTTSLocalQwen3Backbone"]
