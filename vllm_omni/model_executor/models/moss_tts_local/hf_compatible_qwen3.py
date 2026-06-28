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
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rope_neox_inplace_kernel(
        x,
        cos,
        sin,
        x_stride_t: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        cos_stride_t: tl.constexpr,
        cos_stride_d: tl.constexpr,
        half_dim: tl.constexpr,
        block: tl.constexpr,
    ) -> None:
        token = tl.program_id(0)
        head = tl.program_id(1)
        offs = tl.arange(0, block)
        mask = offs < half_dim
        x_base = x + token * x_stride_t + head * x_stride_h
        cos_base = cos + token * cos_stride_t
        first = tl.load(x_base + offs * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        second = tl.load(x_base + (offs + half_dim) * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_base + offs * cos_stride_d, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin + token * cos_stride_t + offs * cos_stride_d, mask=mask, other=0.0).to(tl.float32)
        tl.store(x_base + offs * x_stride_d, first * c - second * s, mask=mask)
        tl.store(x_base + (offs + half_dim) * x_stride_d, second * c + first * s, mask=mask)


_fused_rope_runtime_disabled = False


def _disabled_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class MossQwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._fused_rmsnorm_enabled = not _disabled_env("MOSS_TTS_LOCAL_DISABLE_FUSED_RMSNORM")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fused_rmsnorm_enabled and hidden_states.is_cuda:
            try:
                from vllm import _custom_ops as ops

                out = torch.empty_like(hidden_states)
                ops.rms_norm(out, hidden_states, self.weight, self.variance_epsilon)
                return out
            except Exception:
                self._fused_rmsnorm_enabled = False
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
    if _fused_rotary_enabled() and _apply_rotary_pos_emb_triton(query, key, cos, sin):
        return query, key
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


def _fused_rotary_enabled() -> bool:
    return not _disabled_env("MOSS_TTS_LOCAL_DISABLE_FUSED_ROPE")


def _apply_rotary_pos_emb_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    global _fused_rope_runtime_disabled
    if _fused_rope_runtime_disabled:
        return False
    if triton is None or not query.is_cuda or not key.is_cuda:
        return False
    if query.dim() != 4 or key.dim() != 4 or query.shape[1] != 1 or key.shape[1] != 1:
        return False
    if query.shape[-1] != key.shape[-1] or query.shape[0] != key.shape[0]:
        return False
    head_dim = int(query.shape[-1])
    if head_dim <= 0 or head_dim % 2 != 0:
        return False
    try:
        q = query.reshape(-1, int(query.shape[-2]), head_dim)
        k = key.reshape(-1, int(key.shape[-2]), head_dim)
        cos_flat = cos.reshape(-1, int(cos.shape[-1]))
        sin_flat = sin.reshape(-1, int(sin.shape[-1]))
        if cos_flat.shape[0] != q.shape[0] or sin_flat.shape[0] != q.shape[0]:
            return False
        if cos_flat.shape[-1] < head_dim or sin_flat.shape[-1] < head_dim:
            return False
        half_dim = head_dim // 2
        block = 1 << max(0, (half_dim - 1).bit_length())
        grid_q = (int(q.shape[0]), int(q.shape[1]))
        _rope_neox_inplace_kernel[grid_q](
            q,
            cos_flat,
            sin_flat,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            cos_flat.stride(0),
            cos_flat.stride(1),
            half_dim,
            block=block,
        )
        grid_k = (int(k.shape[0]), int(k.shape[1]))
        _rope_neox_inplace_kernel[grid_k](
            k,
            cos_flat,
            sin_flat,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            cos_flat.stride(0),
            cos_flat.stride(1),
            half_dim,
            block=block,
        )
        return True
    except Exception:
        _fused_rope_runtime_disabled = True
        return False


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
        self._fused_gate_up_enabled = os.environ.get("MOSS_TTS_LOCAL_FUSED_MLP", "0").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._gate_up_weight_cache: torch.Tensor | None = None
        self._compiled_forward = None
        self._compile_enabled = os.environ.get("MOSS_TTS_LOCAL_COMPILE_MLP", "0").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fused_gate_up_enabled:
            weight = self._gate_up_weight_cache
            if weight is None or weight.device != hidden_states.device or weight.dtype != hidden_states.dtype:
                weight = (
                    torch.cat(
                        [
                            self.gate_proj.weight.detach(),
                            self.up_proj.weight.detach(),
                        ],
                        dim=0,
                    )
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                    .contiguous()
                )
                self._gate_up_weight_cache = weight
            gate, up = F.linear(hidden_states, weight).chunk(2, dim=-1)
            return self.down_proj(self.act_fn(gate) * up)
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._compile_enabled:
            compiled = self._compiled_forward
            if compiled is None:
                compile_fn = getattr(torch, "compile", None)
                if compile_fn is None:
                    self._compile_enabled = False
                else:
                    mode = os.environ.get("MOSS_TTS_LOCAL_COMPILE_MLP_MODE", "reduce-overhead")
                    try:
                        compiled = compile_fn(self._forward_impl, mode=mode, dynamic=True)
                        self._compiled_forward = compiled
                    except Exception:
                        self._compile_enabled = False
            if compiled is not None:
                try:
                    return compiled(hidden_states)
                except Exception:
                    self._compile_enabled = False
        return self._forward_impl(hidden_states)


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
        self._fused_qkv_enabled = not _disabled_env("MOSS_TTS_LOCAL_DISABLE_FUSED_QKV")
        self._sdpa_gqa_enabled = os.environ.get("MOSS_TTS_LOCAL_SDPA_GQA", "0").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._flashinfer_single_decode_enabled = os.environ.get(
            "MOSS_TTS_LOCAL_FLASHINFER_SINGLE_DECODE",
            "0",
        ).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._flashinfer_batch_decode_enabled = os.environ.get(
            "MOSS_TTS_LOCAL_FLASHINFER_BATCH_DECODE",
            "0",
        ).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._flashinfer_single_decode_warned = False
        self._flashinfer_batch_decode_warned = False
        self._flashinfer_batch_wrapper = None
        self._flashinfer_batch_workspace: torch.Tensor | None = None
        self._qkv_weight_cache: torch.Tensor | None = None
        self._qkv_bias_cache: torch.Tensor | None = None

    def _invalidate_qkv_cache(self) -> None:
        self._qkv_weight_cache = None
        self._qkv_bias_cache = None

    def _qkv_linear(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._fused_qkv_enabled:
            return (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            )
        weight = self._qkv_weight_cache
        if weight is None or weight.device != hidden_states.device or weight.dtype != hidden_states.dtype:
            weight = (
                torch.cat(
                    [
                        self.q_proj.weight.detach(),
                        self.k_proj.weight.detach(),
                        self.v_proj.weight.detach(),
                    ],
                    dim=0,
                )
                .to(device=hidden_states.device, dtype=hidden_states.dtype)
                .contiguous()
            )
            self._qkv_weight_cache = weight
            if self.q_proj.bias is None:
                self._qkv_bias_cache = None
            else:
                self._qkv_bias_cache = (
                    torch.cat(
                        [
                            self.q_proj.bias.detach(),
                            self.k_proj.bias.detach(),
                            self.v_proj.bias.detach(),
                        ],
                        dim=0,
                    )
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                    .contiguous()
                )
        qkv = F.linear(hidden_states, weight, self._qkv_bias_cache)
        q_end = self.num_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim
        return qkv.split((q_end, kv_width, kv_width), dim=-1)

    def _flashinfer_single_decode(
        self,
        *,
        query: torch.Tensor,
        k_batch: torch.Tensor,
        v_batch: torch.Tensor,
        cache_lens: list[int],
    ) -> torch.Tensor | None:
        if not self._flashinfer_single_decode_enabled:
            return None
        try:
            import flashinfer  # type: ignore[import-not-found]

            outs: list[torch.Tensor] = []
            for row, cache_len in enumerate(cache_lens):
                q_row = query[row, :, 0, :].contiguous()
                k_row = k_batch[row, : int(cache_len)].contiguous()
                v_row = v_batch[row, : int(cache_len)].contiguous()
                outs.append(
                    flashinfer.single_decode_with_kv_cache(
                        q_row,
                        k_row,
                        v_row,
                        kv_layout="NHD",
                        sm_scale=self.scaling,
                    )
                )
            return torch.stack(outs, dim=0).unsqueeze(2)
        except Exception:
            self._flashinfer_single_decode_enabled = False
            if not self._flashinfer_single_decode_warned:
                self._flashinfer_single_decode_warned = True
                import traceback

                traceback.print_exc()
            return None

    def _flashinfer_batch_decode(
        self,
        *,
        query: torch.Tensor,
        k_batch: torch.Tensor,
        v_batch: torch.Tensor,
        cache_lens: list[int],
        step_cache: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if not self._flashinfer_batch_decode_enabled:
            return None
        if not query.is_cuda or query.shape[2] != 1:
            return None
        try:
            from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper  # type: ignore[import-not-found]

            batch_size = int(query.shape[0])
            max_len = int(k_batch.shape[1])
            if batch_size <= 0 or max_len <= 0:
                return None

            workspace = self._flashinfer_batch_workspace
            if workspace is None or workspace.device != query.device:
                workspace = torch.empty(128 * 1024 * 1024, device=query.device, dtype=torch.uint8)
                self._flashinfer_batch_workspace = workspace
                self._flashinfer_batch_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
            wrapper = self._flashinfer_batch_wrapper

            if step_cache is not None and "flashinfer_indptr" in step_cache:
                indptr = step_cache["flashinfer_indptr"]
                indices = step_cache["flashinfer_indices"]
                last_page_len = step_cache["flashinfer_last_page_len"]
            else:
                indptr = torch.arange(batch_size + 1, device=query.device, dtype=torch.int32)
                indices = torch.arange(batch_size, device=query.device, dtype=torch.int32)
                last_page_len = torch.tensor(cache_lens, device=query.device, dtype=torch.int32)
                if step_cache is not None:
                    step_cache["flashinfer_indptr"] = indptr
                    step_cache["flashinfer_indices"] = indices
                    step_cache["flashinfer_last_page_len"] = last_page_len

            q = query[:, :, 0, :].contiguous()
            # FlashInfer's NHD paged cache expects [pages, page_size, kv_heads, head_dim].
            # A request row is one page here, with page_size equal to the current max KV length.
            k = k_batch.contiguous()
            v = v_batch.contiguous()
            wrapper.plan(
                indptr,
                indices,
                last_page_len,
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim,
                max_len,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                sm_scale=self.scaling,
            )
            return wrapper.run(q, (k, v)).unsqueeze(2)
        except Exception:
            self._flashinfer_batch_decode_enabled = False
            if not self._flashinfer_batch_decode_warned:
                self._flashinfer_batch_decode_warned = True
                import traceback

                traceback.print_exc()
            return None

    def forward_batched_decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        slot_ids: list[int],
        slot_idx_t: torch.Tensor | None = None,
        step_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        bp = self._backbone_profile if getattr(self, "_backbone_profile_enabled", False) else None

        if bp is not None:
            torch.accelerator.synchronize(hidden_states.device)
            _t0 = time.perf_counter()

        query_raw, key_raw, value_raw = self._qkv_linear(hidden_states)
        query_states = self.q_norm(query_raw.view(batch_size, 1, self.num_heads, self.head_dim))
        key_states = self.k_norm(key_raw.view(batch_size, 1, self.num_key_value_heads, self.head_dim))
        value_states = value_raw.view(batch_size, 1, self.num_key_value_heads, self.head_dim)

        if bp is not None:
            torch.accelerator.synchronize(hidden_states.device)
            _t1 = time.perf_counter()
            bp["qkv"] += (_t1 - _t0) * 1000

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if bp is not None:
            torch.accelerator.synchronize(hidden_states.device)
            _t2 = time.perf_counter()
            bp["rope"] += (_t2 - _t1) * 1000

        if step_cache is not None and "starts" in step_cache:
            starts = step_cache["starts"]
            starts_t = step_cache["starts_t"]
            uniform_start = step_cache["uniform_start"]
            slot_slice = step_cache["slot_slice"]
        else:
            starts = [self._slot_cache_lens[sid] for sid in slot_ids]
            starts_t = torch.tensor(starts, device=hidden_states.device, dtype=torch.long)
            uniform_start = starts[0] if starts and all(start == starts[0] for start in starts) else None
            slot_slice = None
            if slot_ids and all(slot_ids[i] + 1 == slot_ids[i + 1] for i in range(len(slot_ids) - 1)):
                slot_slice = (int(slot_ids[0]), int(slot_ids[0]) + len(slot_ids))
            if step_cache is not None:
                step_cache["starts"] = starts
                step_cache["starts_t"] = starts_t
                step_cache["uniform_start"] = uniform_start
                step_cache["slot_slice"] = slot_slice
        if slot_idx_t is None:
            slot_idx_t = torch.tensor(slot_ids, device=hidden_states.device, dtype=torch.long)
        if slot_slice is not None and uniform_start is not None:
            s0, s1 = slot_slice
            self.static_key_cache[s0:s1, int(uniform_start)] = key_states[:, 0]
            self.static_value_cache[s0:s1, int(uniform_start)] = value_states[:, 0]
        else:
            self.static_key_cache[slot_idx_t, starts_t] = key_states[:, 0]
            self.static_value_cache[slot_idx_t, starts_t] = value_states[:, 0]
        for sid, start in zip(slot_ids, starts):
            self._slot_cache_lens[sid] = int(start) + 1

        if bp is not None:
            torch.accelerator.synchronize(hidden_states.device)
            _t3 = time.perf_counter()
            bp["kv_write"] += (_t3 - _t2) * 1000

        if step_cache is not None and "mask" in step_cache:
            mask = step_cache["mask"]
            max_len = int(step_cache["max_len"])
        else:
            cache_lens = [self._slot_cache_lens[sid] for sid in slot_ids]
            max_len = max(cache_lens)
            if cache_lens and all(length == max_len for length in cache_lens):
                mask = None
            else:
                lens_t = torch.tensor(cache_lens, device=hidden_states.device, dtype=torch.long)
                positions = torch.arange(max_len, device=hidden_states.device, dtype=torch.long)
                mask = (positions.unsqueeze(0) < lens_t.unsqueeze(1))[:, None, None, :]
            if step_cache is not None:
                step_cache["mask"] = mask
                step_cache["max_len"] = torch.tensor(max_len)
                step_cache["cache_lens"] = cache_lens
        cache_lens = step_cache.get("cache_lens", None) if step_cache is not None else None
        if cache_lens is None:
            cache_lens = [self._slot_cache_lens[sid] for sid in slot_ids]

        if slot_slice is not None:
            s0, s1 = slot_slice
            k_batch = self.static_key_cache[s0:s1, :max_len]
            v_batch = self.static_value_cache[s0:s1, :max_len]
        else:
            k_batch = self.static_key_cache[slot_idx_t, :max_len]
            v_batch = self.static_value_cache[slot_idx_t, :max_len]

        if bp is not None:
            torch.accelerator.synchronize(hidden_states.device)
            _t4 = time.perf_counter()
            bp["kv_gather"] += (_t4 - _t3) * 1000

        query = query_states.transpose(1, 2)
        output = self._flashinfer_batch_decode(
            query=query,
            k_batch=k_batch,
            v_batch=v_batch,
            cache_lens=cache_lens,
            step_cache=step_cache,
        )
        if output is not None:
            pass
        else:
            output = self._flashinfer_single_decode(
                query=query,
                k_batch=k_batch,
                v_batch=v_batch,
                cache_lens=cache_lens,
            )
        if output is not None:
            pass
        elif self._sdpa_gqa_enabled and self.num_key_value_groups > 1:
            try:
                output = F.scaled_dot_product_attention(
                    query,
                    k_batch.transpose(1, 2),
                    v_batch.transpose(1, 2),
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scaling,
                    enable_gqa=True,
                )
            except TypeError:
                self._sdpa_gqa_enabled = False
                k_batch = _repeat_kv(k_batch, self.num_key_value_groups).transpose(1, 2)
                v_batch = _repeat_kv(v_batch, self.num_key_value_groups).transpose(1, 2)
                output = F.scaled_dot_product_attention(
                    query, k_batch, v_batch, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=self.scaling
                )
        else:
            k_batch = _repeat_kv(k_batch, self.num_key_value_groups).transpose(1, 2)
            v_batch = _repeat_kv(v_batch, self.num_key_value_groups).transpose(1, 2)
            output = F.scaled_dot_product_attention(
                query, k_batch, v_batch, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=self.scaling
            )

        if bp is not None:
            torch.accelerator.synchronize(hidden_states.device)
            _t5 = time.perf_counter()
            bp["sdpa"] += (_t5 - _t4) * 1000

        output = output.transpose(1, 2).reshape(batch_size, 1, -1).contiguous()
        result = self.o_proj(output)

        if bp is not None:
            torch.accelerator.synchronize(hidden_states.device)
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
        query_raw, key_raw, value_raw = self._qkv_linear(hidden_states)
        query_states = self.q_norm(query_raw.view(*input_shape, self.num_heads, self.head_dim))
        key_states = self.k_norm(key_raw.view(*input_shape, self.num_key_value_heads, self.head_dim))
        value_states = value_raw.view(*input_shape, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.static_key_cache is not None and self.static_value_cache is not None:
            slot_id = int(self.active_slot)
            if self.use_static_full_cache:
                assert self.static_cache_positions is not None
                self.static_key_cache[: key_states.shape[0]].index_copy_(1, self.static_cache_positions, key_states)
                self.static_value_cache[: value_states.shape[0]].index_copy_(
                    1,
                    self.static_cache_positions,
                    value_states,
                )
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
        bp = (
            getattr(self.self_attn, "_backbone_profile", None)
            if getattr(self.self_attn, "_backbone_profile_enabled", False)
            else None
        )
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
            torch.accelerator.synchronize(hidden_states.device)
            _tm0 = time.perf_counter()

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        if bp is not None:
            torch.accelerator.synchronize(hidden_states.device)
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
        self._backbone_profile: dict[str, float] = {
            "qkv": 0,
            "rope": 0,
            "kv_write": 0,
            "kv_gather": 0,
            "sdpa": 0,
            "o_proj": 0,
            "norm_mlp": 0,
            "n": 0,
        }
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

    def warmup_decode_graphs(
        self,
        bucket_sizes: list[int],
        device: torch.device,
        dtype: torch.dtype,
        max_batch: int = 1,
    ) -> None:
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
                    f"BackboneProfile[{n} steps, {nl} layers]: qkv={bp['qkv'] / n:.2f}ms rope={bp['rope'] / n:.2f}ms "
                    f"kv_write={bp['kv_write'] / n:.2f}ms kv_gather={bp['kv_gather'] / n:.2f}ms "
                    f"sdpa={bp['sdpa'] / n:.2f}ms o_proj={bp['o_proj'] / n:.2f}ms "
                    f"norm_mlp={bp['norm_mlp'] / n:.2f}ms | "
                    f"total={total:.2f}ms/step",
                    file=sys.stderr,
                    flush=True,
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
            out = self._forward_eager_batched(self._batch_static_input[:n], self._batch_static_pos[:n], slot_ids)
            self._batch_static_output[:n].copy_(out)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._forward_eager_batched(self._batch_static_input[:n], self._batch_static_pos[:n], slot_ids)
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
        if inputs_embeds is None and (input_ids is None or int(input_ids.numel()) == 0):
            hidden_size = int(self.config.hidden_size)
            return positions.new_empty((0, hidden_size), dtype=self.embed_tokens.weight.dtype)
        if inputs_embeds is not None and int(inputs_embeds.numel()) == 0:
            hidden_size = int(self.config.hidden_size)
            return inputs_embeds.new_empty((0, hidden_size))

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
            position_ids = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=torch.long,
            ).view(1, -1)

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
        for layer in self.layers:
            layer.self_attn._invalidate_qkv_cache()
        return loaded


__all__ = ["MossTTSLocalQwen3Backbone"]
