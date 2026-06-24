# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pipeline topology for MOSS-TTS Local Transformer v1.5."""

import os

from transformers import PretrainedConfig

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.moss_tts_local"

MOSS_TTS_LOCAL_PIPELINE = PipelineConfig(
    model_type="moss_tts_local",
    model_arch="MossTTSLocalModel",
    hf_architectures=("MossTTSLocalModel",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moss_tts_local",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="audio",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2vocoder_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.talker2vocoder",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="moss_tts_local_vocoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MossTTSLocalVocoder",
            sync_process_input_func=f"{_PROC}.vocoder_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

MOSS_TTS_LOCAL_NATIVE_PIPELINE = PipelineConfig(
    model_type="moss_tts_local",
    model_arch="MossTTSLocalNativeModel",
    hf_architectures=("MossTTSLocalNativeModel",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moss_tts_local",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="audio",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2vocoder_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.talker2vocoder",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="moss_tts_local_vocoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MossTTSLocalVocoder",
            sync_process_input_func=f"{_PROC}.vocoder_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


def resolve_moss_tts_local_pipeline(
    hf_config: PretrainedConfig | None = None,
) -> PipelineConfig:
    del hf_config

    native_flag = os.environ.get("MOSS_TTS_LOCAL_NATIVE", "0").strip().lower()
    if native_flag in ("1", "true", "yes", "on"):
        return MOSS_TTS_LOCAL_NATIVE_PIPELINE
    return MOSS_TTS_LOCAL_PIPELINE


__all__ = [
    "MOSS_TTS_LOCAL_PIPELINE",
    "MOSS_TTS_LOCAL_NATIVE_PIPELINE",
    "resolve_moss_tts_local_pipeline",
]
