# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Stage input processors for MOSS-TTS Local Transformer v1.5."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import TokensPrompt as OmniTokensPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

logger = init_logger(__name__)


def _get_audio_from_payload(payload: Any) -> torch.Tensor | None:
    if not isinstance(payload, Mapping):
        return None
    codes = payload.get("codes")
    if isinstance(codes, Mapping):
        audio = codes.get("audio")
        if isinstance(audio, torch.Tensor):
            return audio
    audio = payload.get("codes.audio")
    if isinstance(audio, torch.Tensor):
        return audio
    return None


def _extract_audio_codes(stage_output: Any) -> torch.Tensor | None:
    outputs = getattr(stage_output, "outputs", None)
    if outputs:
        output = outputs[0]
        audio = _get_audio_from_payload(getattr(output, "multimodal_output", None))
        if audio is not None:
            return audio

    audio = _get_audio_from_payload(getattr(stage_output, "multimodal_output", None))
    if audio is not None:
        return audio

    audio = _get_audio_from_payload(getattr(stage_output, "multimodal_outputs", None))
    if audio is not None:
        return audio
    return None


def _flatten_codes(codes_t_nq: torch.Tensor) -> list[int]:
    if codes_t_nq.numel() == 0:
        return []
    if codes_t_nq.dim() != 2:
        raise ValueError(f"MOSS-TTS Local codes must be [T, NQ], got {tuple(codes_t_nq.shape)}")
    return codes_t_nq.to(torch.long).transpose(0, 1).contiguous().reshape(-1).tolist()


def _flatten_codes_tensor(codes_t_nq: torch.Tensor) -> torch.Tensor:
    if codes_t_nq.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if codes_t_nq.dim() != 2:
        raise ValueError(f"MOSS-TTS Local codes must be [T, NQ], got {tuple(codes_t_nq.shape)}")
    return codes_t_nq.to(torch.long).transpose(0, 1).contiguous().reshape(-1).cpu()


def _meta_step_for_req(meta: Mapping[str, Any], req_id: str) -> int:
    step = meta.get("step", -1)
    req_ids = meta.get("req_id")
    if isinstance(step, torch.Tensor):
        step = step.detach().cpu().reshape(-1).tolist()
    if isinstance(req_ids, torch.Tensor):
        req_ids = req_ids.detach().cpu().reshape(-1).tolist()
    if isinstance(step, (list, tuple)):
        if isinstance(req_ids, (list, tuple)):
            for idx, rid in enumerate(req_ids):
                if str(rid) == req_id and idx < len(step):
                    return int(step[idx])
        return -1
    try:
        return int(step)
    except (TypeError, ValueError):
        return -1


def _get_stage_output(stage_outputs: Any, source: Any) -> Any:
    if isinstance(stage_outputs, dict):
        if source in stage_outputs:
            return stage_outputs[source]
        source_str = str(source)
        if source_str in stage_outputs:
            return stage_outputs[source_str]
        try:
            source_int = int(source)
        except (TypeError, ValueError):
            return None
        return stage_outputs.get(source_int)

    try:
        source_int = int(source)
    except (TypeError, ValueError):
        return None
    if 0 <= source_int < len(stage_outputs):
        return stage_outputs[source_int]
    return None


def talker2vocoder(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[Any]:
    del prompt, requires_multimodal_data, streaming_context
    results: list[Any] = []
    for i, stage_out in enumerate(source_outputs):
        if hasattr(stage_out, "finished") and not stage_out.finished:
            continue
        audio_codes = _extract_audio_codes(stage_out)
        if audio_codes is None or audio_codes.numel() == 0:
            logger.warning("talker2vocoder: no MOSS-TTS Local audio codes in source output %s.", i)
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue
        flat = _flatten_codes(audio_codes)
        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data={"codes": {"audio": flat}},
            )
        )
    return results


def talker2vocoder_legacy(
    stage_list: Any,
    engine_input_source: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    del prompt, requires_multimodal_data
    results: list[Any] = []
    for source in engine_input_source:
        stage_out = _get_stage_output(stage_list, source)
        audio_codes = _extract_audio_codes(stage_out)
        if audio_codes is None or audio_codes.numel() == 0:
            logger.warning("talker2vocoder: no MOSS-TTS Local audio codes in stage output %s.", source)
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue
        flat = _flatten_codes(audio_codes)
        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data={"codes": {"audio": flat}},
            )
        )
    return results


def vocoder_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[Any]:
    return talker2vocoder(source_outputs, prompt, requires_multimodal_data, streaming_context)


def talker2vocoder_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | list[OmniPayloadStruct] | None:
    req_id = str(getattr(request, "external_req_id", None) or getattr(request, "request_id", id(request)))
    if not hasattr(transfer_manager, "_moss_tts_local_state"):
        transfer_manager._moss_tts_local_state = {}
    state = transfer_manager._moss_tts_local_state
    req_state = state.setdefault(req_id, {"pending": [], "pending_frames": 0, "emitted_frames": 0, "last_step": -1})

    if multimodal_output is not None:
        codes_dict = multimodal_output.get("codes", {}) or {}
        snapshot = codes_dict.get("audio")
        if isinstance(snapshot, torch.Tensor) and snapshot.numel() > 0:
            meta = multimodal_output.get("meta", {}) or {}
            step = _meta_step_for_req(meta, req_id) if isinstance(meta, Mapping) else -1
            if step >= 0 and step <= req_state.get("last_step", -1):
                pass
            else:
                if step >= 0:
                    req_state["last_step"] = step
                snapshot_cpu = snapshot.detach().cpu().to(torch.long)
                if bool(meta.get("raw_rows", False)):
                    new_rows = snapshot_cpu
                else:
                    total_frames = req_state["emitted_frames"] + req_state["pending_frames"]
                    new_rows = snapshot_cpu[total_frames:]
                if new_rows.numel() > 0:
                    req_state["pending"].append(new_rows)
                    req_state["pending_frames"] += int(new_rows.shape[0])

    pending_frames = req_state["pending_frames"]
    emitted_frames = int(req_state.get("emitted_frames", 0))

    if pending_frames == 0:
        if is_finished:
            state.pop(req_id, None)
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(
                    left_context_size=0,
                    finished=torch.tensor(True, dtype=torch.bool),
                    codec_streaming=True,
                    req_id=[req_id],
                ),
                request_id=req_id,
            )
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_frames = int(cfg.get("codec_chunk_frames", 25) or 25)
    initial_chunk_frames = int(cfg.get("initial_codec_chunk_frames", 0) or 0)
    if initial_chunk_frames > chunk_frames:
        initial_chunk_frames = chunk_frames

    threshold = initial_chunk_frames if emitted_frames == 0 and initial_chunk_frames > 0 else chunk_frames
    if not is_finished and pending_frames < threshold:
        return None

    # Only materialize (concat) when we're actually emitting.
    pending_list = req_state["pending"]
    acc = torch.cat(pending_list, dim=0) if len(pending_list) > 1 else pending_list[0]

    def make_payload(chunk_t_nq: torch.Tensor, *, finished: bool) -> OmniPayloadStruct:
        flat = _flatten_codes_tensor(chunk_t_nq)
        return OmniPayloadStruct(
            codes=CodesStruct(audio=flat),
            meta=MetaStruct(
                left_context_size=0,
                finished=torch.tensor(bool(finished), dtype=torch.bool),
                codec_streaming=True,
                req_id=[req_id],
            ),
            request_id=req_id,
        )

    if is_finished:
        payloads: list[OmniPayloadStruct] = []
        cursor = 0
        local_emitted = emitted_frames
        while cursor < pending_frames:
            local_threshold = (
                initial_chunk_frames
                if local_emitted == 0 and initial_chunk_frames > 0
                else chunk_frames
            )
            emit_frames = min(local_threshold, pending_frames - cursor)
            chunk = acc[cursor:cursor + emit_frames]
            cursor += emit_frames
            local_emitted += emit_frames
            payloads.append(make_payload(chunk, finished=cursor >= pending_frames))
        state.pop(req_id, None)
        return payloads

    emit_frames = threshold
    chunk = acc[:emit_frames]
    remaining = acc[emit_frames:] if emit_frames < pending_frames else None
    req_state["emitted_frames"] = emitted_frames + emit_frames
    req_state["pending"] = [remaining] if remaining is not None and remaining.numel() > 0 else []
    req_state["pending_frames"] = int(remaining.shape[0]) if remaining is not None and remaining.numel() > 0 else 0
    return make_payload(chunk, finished=False)


__all__ = ["talker2vocoder", "vocoder_token_only", "talker2vocoder_async_chunk"]
