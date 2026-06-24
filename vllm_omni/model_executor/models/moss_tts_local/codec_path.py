# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Path helpers for MOSS-TTS Local Transformer v1.5."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_moss_tts_local_codec_path(
    model_path: str,
    explicit_codec_path: str | None = None,
) -> str | None:
    """Resolve a local MOSS-Audio-Tokenizer-v2 path when one is colocated.

    ``AutoProcessor`` can load the codec from the repo id stored in config, but
    remote profiling hosts often run with restricted Hugging Face access.  When
    the main checkpoint is local, prefer a sibling codec directory so both
    prompt building and vocoder decode stay offline.
    """
    if explicit_codec_path:
        return explicit_codec_path

    env_path = os.environ.get("MOSS_TTS_LOCAL_CODEC_PATH")
    if env_path:
        return env_path

    path = Path(model_path).expanduser()
    if not path.exists():
        return None

    for candidate in (
        path.parent / "MOSS-Audio-Tokenizer-v2",
        path.parent / "OpenMOSS-Team" / "MOSS-Audio-Tokenizer-v2",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def moss_tts_local_processor_kwargs(
    model_path: str,
    explicit_codec_path: str | None = None,
) -> dict[str, object]:
    """Common kwargs for loading the upstream MOSS-TTS Local processor."""
    kwargs: dict[str, object] = {
        "trust_remote_code": True,
    }
    codec_path = resolve_moss_tts_local_codec_path(model_path, explicit_codec_path)
    if codec_path is not None:
        kwargs["codec_path"] = codec_path
    return kwargs


__all__ = ["moss_tts_local_processor_kwargs", "resolve_moss_tts_local_codec_path"]
