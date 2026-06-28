# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Frame-local transformer for MOSS-TTS Local Transformer v1.5."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(x)


def sample_top_k_top_p(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row with the model-card defaults."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

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
            fallback = probs.sum(dim=-1) <= 0
            sampled_rel = torch.multinomial(probs, num_samples=1, generator=generator)
            sampled = sorted_indices.gather(-1, sampled_rel).reshape(-1)
            return torch.where(fallback, torch.argmax(logits, dim=-1), sampled)

        probs = torch.softmax(topk_scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        fallback = probs.sum(dim=-1) <= 0
        sampled_rel = torch.multinomial(probs, num_samples=1, generator=generator)
        sampled = topk_indices.gather(-1, sampled_rel).reshape(-1)
        return torch.where(fallback, torch.argmax(logits, dim=-1), sampled)

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
    fallback = probs.sum(dim=-1) <= 0
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).reshape(-1)
    return torch.where(fallback, torch.argmax(logits, dim=-1), sampled)


class MossTTSLocalMLP(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, inner_size)
        self.fc_out = nn.Linear(inner_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.silu(self.fc_in(hidden_states)))


class MossTTSLocalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} not divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)


class MossTTSLocalBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, inner_size: int, layer_norm_eps: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = MossTTSLocalAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MossTTSLocalMLP(hidden_size, inner_size)


class MossTTSLocalTransformer(nn.Module):
    """Incremental local decoder over one frame's text+RVQ positions."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        inner_size: int,
        num_layers: int,
        max_positions: int,
        rope_base: float,
        layer_norm_eps: float = 1e-6,
        attn_implementation: str | None = "eager",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_positions = int(max_positions)
        self.attn_implementation = str(attn_implementation or "eager").lower()
        self.h = nn.ModuleList(
            [MossTTSLocalBlock(hidden_size, num_heads, inner_size, layer_norm_eps) for _ in range(int(num_layers))]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        inv_freq = 1.0 / (float(rope_base) ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        positions = torch.arange(self.max_positions, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("rope_cos", freqs.cos().repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("rope_sin", freqs.sin().repeat_interleave(2, dim=-1), persistent=False)
        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._kv_capacity = 0
        self._graph_enabled = False
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._graph_input: torch.Tensor | None = None
        self._graph_output: torch.Tensor | None = None
        self._graph_batch_size: int = 0
        self._rope_cache: dict[tuple[str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _rope_tables(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, dtype)
        cached = self._rope_cache.get(key)
        if cached is None:
            cached = (
                self.rope_cos.to(device=device, dtype=dtype),
                self.rope_sin.to(device=device, dtype=dtype),
            )
            self._rope_cache[key] = cached
        return cached

    def _ensure_kv_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._kv_capacity >= batch_size
            and self._kv_cache
            and self._kv_cache[0][0].device == device
            and self._kv_cache[0][0].dtype == dtype
        ):
            return
        capacity = max(batch_size, self._kv_capacity, 1)
        # Match the upstream static-cache layout [batch, seq, heads, dim].
        # The local GPT2 head is sensitive enough that the SDPA backend can
        # pick a different numerical path for [B, H, T, D] cache strides.
        shape = (capacity, self.max_positions, self.num_heads, self.head_dim)
        self._kv_cache = [
            (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in self.h
        ]
        self._kv_capacity = capacity

    def enable_graphs(self, batch_size: int) -> None:
        """Pre-allocate static buffers for graph capture."""
        self._graph_enabled = True
        self._graph_batch_size = batch_size
        device = self._kv_cache[0][0].device if self._kv_cache else torch.device("cuda")
        dtype = self._kv_cache[0][0].dtype if self._kv_cache else torch.bfloat16
        self._graph_input = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        self._graph_output = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        self._graphs.clear()

    def _step_eager(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        """The actual computation (no graph)."""
        batch_size = int(hidden_states.shape[0])
        rope_cos, rope_sin = self._rope_tables(hidden_states.device, hidden_states.dtype)
        cos = rope_cos[position]
        sin = rope_sin[position]
        x = hidden_states
        for layer_idx, block in enumerate(self.h):
            normed = block.ln_1(x)
            query, key, value = block.attn.c_attn(normed).split(self.hidden_size, dim=-1)
            query = query.view(batch_size, self.num_heads, self.head_dim)
            key = key.view(batch_size, self.num_heads, self.head_dim)
            value = value.view(batch_size, self.num_heads, self.head_dim)
            query = query * cos + _rotate_half_interleaved(query) * sin
            key = key * cos + _rotate_half_interleaved(key) * sin
            key_cache, value_cache = self._kv_cache[layer_idx]
            key_cache[:batch_size, position] = key
            value_cache[:batch_size, position] = value
            key_len = position + 1
            keys = key_cache[:batch_size, :key_len]
            values = value_cache[:batch_size, :key_len]
            scores = torch.einsum("bhd,bthd->bht", query, keys) * (self.head_dim**-0.5)
            probs = torch.softmax(scores, dim=-1)
            attn_out = torch.einsum("bht,bthd->bhd", probs, values).reshape(batch_size, self.hidden_size)
            x = x + block.attn.c_proj(attn_out)
            x = x + block.mlp(block.ln_2(x))
        return self.ln_f(x)

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        if not 0 <= position < self.max_positions:
            raise ValueError(f"local position {position} out of range [0, {self.max_positions})")
        batch_size = int(hidden_states.shape[0])
        self._ensure_kv_cache(batch_size, hidden_states.device, hidden_states.dtype)

        # Graph-accelerated path
        if self._graph_enabled and batch_size <= self._graph_batch_size and hidden_states.device.type == "cuda":
            key = (batch_size, position)
            if key not in self._graphs:
                # Warmup + capture
                self._graph_input[:batch_size].copy_(hidden_states)
                out = self._step_eager(self._graph_input[:batch_size], position)
                self._graph_output[:batch_size].copy_(out)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = self._step_eager(self._graph_input[:batch_size], position)
                    self._graph_output[:batch_size].copy_(out)
                self._graphs[key] = g
                return self._graph_output[:batch_size].clone()
            self._graph_input[:batch_size].copy_(hidden_states)
            self._graphs[key].replay()
            return self._graph_output[:batch_size].clone()

        # Eager path
        return self._step_eager(hidden_states, position)


__all__ = ["MossTTSLocalTransformer", "sample_top_k_top_p"]
