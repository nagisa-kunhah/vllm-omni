# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Local Transformer v1.5 model stubs and config."""

from vllm_omni.model_executor.models.moss_tts_local.configuration_moss_tts_local import (
    MossTTSLocalConfig,
)
from vllm_omni.model_executor.models.moss_tts_local.modeling_moss_tts_local import (
    MossTTSLocalForGeneration,
)
from vllm_omni.model_executor.models.moss_tts_local.modeling_moss_tts_local_v2 import (
    MossTTSLocalNativeModel,
)
from vllm_omni.model_executor.models.moss_tts_local.modeling_moss_tts_local_vocoder import (
    MossTTSLocalVocoder,
)

__all__ = [
    "MossTTSLocalConfig",
    "MossTTSLocalForGeneration",
    "MossTTSLocalNativeModel",
    "MossTTSLocalVocoder",
]
