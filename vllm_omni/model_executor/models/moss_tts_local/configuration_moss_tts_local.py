# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Configuration for MOSS-TTS Local Transformer v1.5."""

from __future__ import annotations

from transformers import AutoConfig, GPT2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3 import Qwen3Config


class MossTTSLocalConfig(PretrainedConfig):
    """Config for ``OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5``.

    This is a distinct architecture from the existing MOSS-TTS Delay and
    Realtime checkpoints.  It uses a Qwen3 backbone plus a one-layer GPT2 local
    frame transformer, emits 12 audio codebooks and decodes with
    ``MOSS-Audio-Tokenizer-v2`` at 48 kHz.
    """

    model_type = "moss_tts_local"

    def __init__(
        self,
        qwen3_config: dict | Qwen3Config | None = None,
        language_config: dict | Qwen3Config | None = None,
        gpt2_config: dict | GPT2Config | None = None,
        n_vq: int = 12,
        audio_vocab_size: int = 1024,
        audio_codebook_sizes: list[int] | None = None,
        audio_pad_token_id: int = 1024,
        audio_pad_code: int = 1024,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        audio_start_token_id: int = 151669,
        audio_end_token_id: int = 151670,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_slot_token_id: int = 151656,
        audio_assistant_gen_slot_token_id: int | None = None,
        sampling_rate: int = 48000,
        audio_tokenizer_name_or_path: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        attn_implementation: str | None = None,
        local_transformer_attn_implementation: str | None = None,
        local_transformer_layers: int = 1,
        local_text_head_mode: str = "binary",
        use_static_local_kv_cache: bool = True,
        initializer_range: float = 0.02,
        architectures: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        qwen3_source = qwen3_config if qwen3_config is not None else language_config
        self.qwen3_config = self._coerce_qwen3_config(qwen3_source)
        # Upstream keeps both names in config.json.  Mirror that layout so
        # remote configs round-trip and older call sites can read either name.
        self.language_config = self.qwen3_config
        self.gpt2_config = self._coerce_gpt2_config(gpt2_config)

        if audio_codebook_sizes is None:
            audio_codebook_sizes = [audio_vocab_size] * n_vq

        super().__init__(
            pad_token_id=pad_token_id,
            architectures=architectures or ["MossTTSLocalModel"],
            **kwargs,
        )

        self.n_vq = n_vq
        self.audio_vocab_size = audio_vocab_size
        self.audio_codebook_sizes = audio_codebook_sizes
        self.audio_pad_token_id = audio_pad_token_id
        self.audio_pad_code = audio_pad_code
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.audio_user_slot_token_id = audio_user_slot_token_id
        self.audio_assistant_slot_token_id = (
            audio_assistant_slot_token_id
            if audio_assistant_gen_slot_token_id is None
            else audio_assistant_gen_slot_token_id
        )
        self.audio_assistant_gen_slot_token_id = self.audio_assistant_slot_token_id
        self.sampling_rate = sampling_rate
        self.audio_tokenizer_name_or_path = audio_tokenizer_name_or_path
        self.attn_implementation = attn_implementation
        self.local_transformer_attn_implementation = local_transformer_attn_implementation or attn_implementation
        self.local_transformer_layers = local_transformer_layers
        self.local_text_head_mode = local_text_head_mode
        self.use_static_local_kv_cache = use_static_local_kv_cache
        self.initializer_range = initializer_range
        self.speculative_config = None

        self.hidden_size = int(self.qwen3_config.hidden_size)
        self.vocab_size = int(self.qwen3_config.vocab_size)
        self.channels = int(self.n_vq) + 1
        self.vocab_size_list = [self.vocab_size] + [self.audio_vocab_size + 1] * self.n_vq
        self.pad_token = [self.pad_token_id] + [self.audio_pad_code] * self.n_vq

    @staticmethod
    def _coerce_qwen3_config(config: dict | Qwen3Config | None) -> Qwen3Config:
        if config is None:
            config = {}
        if isinstance(config, Qwen3Config):
            return config
        config = dict(config)
        config.pop("model_type", None)
        return Qwen3Config(**config)

    @staticmethod
    def _coerce_gpt2_config(config: dict | GPT2Config | None) -> GPT2Config:
        if config is None:
            config = {}
        if isinstance(config, GPT2Config):
            return config
        config = dict(config)
        config.pop("model_type", None)
        return GPT2Config(**config)

    def get_text_config(self, **_: object) -> Qwen3Config:
        """Return the Qwen3 backbone config for vLLM KV-cache sizing."""
        return self.qwen3_config


AutoConfig.register(MossTTSLocalConfig.model_type, MossTTSLocalConfig)

__all__ = ["MossTTSLocalConfig"]
