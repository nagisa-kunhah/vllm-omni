# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


def _metadata_has_lengths(attn_metadata: AttentionMetadata | None) -> bool:
    return attn_metadata is not None and (
        attn_metadata.query_lens is not None or attn_metadata.key_lens is not None
    )


def _check_no_attn_mask_with_lengths(attn_metadata: AttentionMetadata | None) -> None:
    if _metadata_has_lengths(attn_metadata) and attn_metadata.attn_mask is not None:
        raise ValueError("attn_mask cannot be used together with query_lens or key_lens.")


def _normalize_lengths(
    lens: torch.Tensor | None,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if lens is None:
        return torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    if lens.ndim != 1 or lens.numel() != batch:
        raise ValueError(f"lengths must have shape ({batch},), got {tuple(lens.shape)}.")
    if lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"lengths must use dtype torch.int32 or torch.int64, got {lens.dtype}.")

    lens = lens.to(device=device)
    if torch.any(lens < 0) or torch.any(lens > seq_len):
        raise ValueError(f"lengths must satisfy 0 <= length <= {seq_len}.")
    return lens.to(dtype=torch.int32).contiguous()


def _lengths_to_indices_cu_max(lens: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    positions = torch.arange(seq_len, device=lens.device).unsqueeze(0)
    keep = positions < lens.unsqueeze(1)
    indices = torch.nonzero(keep.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(lens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(lens.max().item()) if lens.numel() > 0 else 0
    return indices, cu_seqlens, max_seqlen


def _lengths_to_key_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    query_lens: torch.Tensor | None,
    key_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, query_len = query.shape[:2]
    key_len = key.shape[1]
    normalized_query_lens = _normalize_lengths(query_lens, batch, query_len, query.device)
    normalized_key_lens = _normalize_lengths(key_lens, batch, key_len, query.device)

    key_positions = torch.arange(key_len, device=query.device).unsqueeze(0)
    key_mask = key_positions < normalized_key_lens.unsqueeze(1)
    attention_mask = key_mask[:, None, None, :].expand(batch, 1, query_len, key_len).contiguous()
    return attention_mask, normalized_query_lens


def _zero_invalid_queries(output: torch.Tensor, query_lens: torch.Tensor) -> torch.Tensor:
    query_len = output.shape[1]
    query_positions = torch.arange(query_len, device=output.device).unsqueeze(0)
    query_mask = query_positions < query_lens.to(device=output.device).unsqueeze(1)
    return output * query_mask[:, :, None, None].to(dtype=output.dtype)


def _sliced_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: torch.Tensor | None,
    key_lens: torch.Tensor | None,
    *,
    causal: bool,
    softmax_scale: float | None,
    fallback_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if fallback_dtype is not None:
        half_dtypes = (torch.float16, torch.bfloat16)
        value = value if value.dtype in half_dtypes else value.to(fallback_dtype)
        query = (query if query.dtype in half_dtypes else query.to(fallback_dtype)).to(value.dtype)
        key = (key if key.dtype in half_dtypes else key.to(fallback_dtype)).to(value.dtype)

    batch, query_len = query.shape[:2]
    key_len = key.shape[1]
    normalized_query_lens = _normalize_lengths(query_lens, batch, query_len, query.device)
    normalized_key_lens = _normalize_lengths(key_lens, batch, key_len, key.device)

    outputs: list[torch.Tensor] = []
    for index in range(batch):
        cur_query_len = int(normalized_query_lens[index].item())
        cur_key_len = int(normalized_key_lens[index].item())
        query_item = query[index : index + 1, :cur_query_len].transpose(1, 2)
        key_item = key[index : index + 1, :cur_key_len].transpose(1, 2)
        value_item = value[index : index + 1, :cur_key_len].transpose(1, 2)
        if query_item.size(1) != key_item.size(1):
            if query_item.size(1) % key_item.size(1) != 0:
                raise ValueError(
                    "Attention query heads must be divisible by key heads: "
                    f"got {query_item.size(1)} and {key_item.size(1)}."
                )
            repeat = query_item.size(1) // key_item.size(1)
            key_item = key_item.repeat_interleave(repeat, dim=1)
            value_item = value_item.repeat_interleave(repeat, dim=1)
        output = F.scaled_dot_product_attention(
            query_item,
            key_item,
            value_item,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        ).transpose(1, 2)
        if cur_query_len < query_len:
            output = F.pad(output, (0, 0, 0, 0, 0, query_len - cur_query_len))
        outputs.append(output)
    return torch.cat(outputs, dim=0)
