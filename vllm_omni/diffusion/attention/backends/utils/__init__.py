# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utils for attention backends.
"""

from vllm_omni.diffusion.attention.backends.utils.fa import _pad_input, _unpad_input, _upad_input
from vllm_omni.diffusion.attention.backends.utils.lengths import (
    _check_no_attn_mask_with_lengths,
    _lengths_to_indices_cu_max,
    _lengths_to_key_mask,
    _metadata_has_lengths,
    _normalize_lengths,
    _zero_invalid_queries,
)

__all__ = [
    "_check_no_attn_mask_with_lengths",
    "_lengths_to_indices_cu_max",
    "_lengths_to_key_mask",
    "_metadata_has_lengths",
    "_normalize_lengths",
    "_pad_input",
    "_unpad_input",
    "_upad_input",
    "_zero_invalid_queries",
]
