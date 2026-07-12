# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    _MOSS_AUDIO_PAD_CODE,
    talker2codec_raw_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

N_VQ = 4


def _tm(*, chunk_frames: int = 3, initial_chunk_frames: int = 0):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "initial_codec_chunk_frames": initial_chunk_frames,
                }
            }
        ),
    )


def _req(rid: str):
    return SimpleNamespace(external_req_id=rid, request_id=rid)


def _payload(frames: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    return {"codes": {"audio": frames}}


def test_first_chunk_emits_codec_native_tensor_layout() -> None:
    tm = _tm(chunk_frames=3, initial_chunk_frames=1)
    frame = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)

    payload = talker2codec_raw_async_chunk(tm, _payload(frame), _req("r"))

    assert payload is not None
    assert isinstance(payload.codes.audio, torch.Tensor)
    assert payload.codes.audio.shape == (N_VQ, 1)
    assert payload.meta.codec_chunk_frames == 1
    assert payload.meta.code_flat_numel == N_VQ
    assert torch.equal(payload.codes.audio, frame.transpose(0, 1).contiguous())


def test_steady_chunk_preserves_frame_order_after_transpose() -> None:
    tm = _tm(chunk_frames=3)
    frames = torch.tensor(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.long,
    )

    payload = talker2codec_raw_async_chunk(tm, _payload(frames), _req("r"))

    assert payload is not None
    assert payload.codes.audio.shape == (N_VQ, 3)
    assert payload.meta.codec_chunk_frames == 3
    assert payload.meta.code_flat_numel == N_VQ * 3
    assert torch.equal(payload.codes.audio, frames.transpose(0, 1).contiguous())


def test_raw_processor_tracks_pending_and_emitted_frames() -> None:
    tm = _tm(chunk_frames=3)
    req = _req("r")

    first = talker2codec_raw_async_chunk(tm, _payload(torch.tensor([[0, 1, 2, 3]], dtype=torch.long)), req)
    second = talker2codec_raw_async_chunk(tm, _payload(torch.tensor([[10, 11, 12, 13]], dtype=torch.long)), req)

    assert first is None
    assert second is None
    state = tm._moss_tts_raw_chunk_states["r"]
    assert state.emitted_frame_count == 0
    assert len(state.pending_frames) == 2

    third_frame = torch.tensor([[20, 21, 22, 23]], dtype=torch.long)
    payload = talker2codec_raw_async_chunk(tm, _payload(third_frame), req)

    assert payload is not None
    assert payload.codes.audio.shape == (N_VQ, 3)
    assert state.emitted_frame_count == 3
    assert state.pending_frames == []


def test_initial_chunk_uses_initial_threshold_once_then_steady_threshold() -> None:
    tm = _tm(chunk_frames=3, initial_chunk_frames=1)
    req = _req("r")

    first_frame = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    first_payload = talker2codec_raw_async_chunk(tm, _payload(first_frame), req)

    assert first_payload is not None
    assert first_payload.codes.audio.shape == (N_VQ, 1)
    state = tm._moss_tts_raw_chunk_states["r"]
    assert state.emitted_frame_count == 1

    assert talker2codec_raw_async_chunk(tm, _payload(torch.tensor([[10, 11, 12, 13]], dtype=torch.long)), req) is None
    assert talker2codec_raw_async_chunk(tm, _payload(torch.tensor([[20, 21, 22, 23]], dtype=torch.long)), req) is None

    fourth_frame = torch.tensor([[30, 31, 32, 33]], dtype=torch.long)
    steady_payload = talker2codec_raw_async_chunk(tm, _payload(fourth_frame), req)

    assert steady_payload is not None
    assert steady_payload.codes.audio.shape == (N_VQ, 3)
    assert state.emitted_frame_count == 4
    assert state.pending_frames == []


def test_finish_with_pending_frames_flushes_tensor_chunk() -> None:
    tm = _tm(chunk_frames=3)
    frames = torch.tensor(
        [
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=torch.long,
    )

    payload = talker2codec_raw_async_chunk(tm, _payload(frames), _req("r"), is_finished=True)

    assert payload is not None
    assert payload.codes.audio.shape == (N_VQ, 2)
    assert torch.equal(payload.codes.audio, frames.transpose(0, 1).contiguous())
    assert payload.meta.stream_finished.item() is True
    assert payload.meta.finished.item() is True
    assert "r" not in tm.code_prompt_token_ids


def test_finish_flushes_remaining_frames_exactly_once() -> None:
    tm = _tm(chunk_frames=3)
    req = _req("r")
    frames = torch.tensor(
        [
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=torch.long,
    )

    assert talker2codec_raw_async_chunk(tm, _payload(frames), req) is None

    payload = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)
    repeat = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)

    assert payload is not None
    assert payload.codes.audio.shape == (N_VQ, 2)
    assert torch.equal(payload.codes.audio, frames.transpose(0, 1).contiguous())
    assert payload.meta.stream_finished.item() is True
    assert repeat is None
    state = tm._moss_tts_raw_chunk_states["r"]
    assert state.final_flushed is True
    assert state.sent_control_sentinel is False


def test_finish_with_no_pending_frames_emits_control_sentinel() -> None:
    tm = _tm(chunk_frames=3)

    payload = talker2codec_raw_async_chunk(tm, None, _req("r"), is_finished=True)

    assert payload is not None
    assert torch.equal(payload.codes.audio, torch.tensor([0], dtype=torch.long))
    assert payload.codes.audio.shape == (1,)
    assert payload.meta.codec_chunk_frames == 0
    assert payload.meta.code_flat_numel == 0
    assert payload.meta.stream_finished.item() is True
    assert payload.meta.finished.item() is True


def test_finish_without_pending_emits_control_sentinel_exactly_once() -> None:
    tm = _tm(chunk_frames=3)
    req = _req("r")

    payload = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)
    repeat = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)

    assert payload is not None
    assert torch.equal(payload.codes.audio, torch.tensor([0], dtype=torch.long))
    assert payload.codes.audio.shape == (1,)
    assert payload.meta.codec_chunk_frames == 0
    assert payload.meta.code_flat_numel == 0
    assert payload.meta.stream_finished.item() is True
    assert repeat is None
    state = tm._moss_tts_raw_chunk_states["r"]
    assert state.final_flushed is True
    assert state.sent_control_sentinel is True


def test_finish_after_regular_chunk_without_pending_emits_one_control_sentinel() -> None:
    tm = _tm(chunk_frames=3)
    req = _req("r")
    frames = torch.tensor(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.long,
    )

    chunk = talker2codec_raw_async_chunk(tm, _payload(frames), req)
    finish = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)
    repeat = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)

    assert chunk is not None
    assert chunk.meta.stream_finished.item() is False
    assert finish is not None
    assert torch.equal(finish.codes.audio, torch.tensor([0], dtype=torch.long))
    assert finish.codes.audio.shape == (1,)
    assert finish.meta.codec_chunk_frames == 0
    assert finish.meta.stream_finished.item() is True
    assert repeat is None


def test_all_pad_rows_do_not_enter_codec() -> None:
    tm = _tm(chunk_frames=3)
    req = _req("r")
    pad_rows = torch.full((3, N_VQ), _MOSS_AUDIO_PAD_CODE, dtype=torch.long)

    assert talker2codec_raw_async_chunk(tm, _payload(pad_rows), req) is None
    assert tm._moss_tts_raw_chunk_states["r"].pending_frames == []

    payload = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)

    assert payload is not None
    assert torch.equal(payload.codes.audio, torch.tensor([0], dtype=torch.long))
    assert payload.codes.audio.shape == (1,)
    assert payload.meta.codec_chunk_frames == 0
    assert payload.meta.code_flat_numel == 0
    assert payload.meta.stream_finished.item() is True


def test_new_non_finish_frames_after_finalized_state_reset_state() -> None:
    tm = _tm(chunk_frames=3)
    req = _req("r")

    sentinel = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)
    assert sentinel is not None
    old_state = tm._moss_tts_raw_chunk_states["r"]
    assert old_state.final_flushed is True

    first_reused_frame = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    payload = talker2codec_raw_async_chunk(tm, _payload(first_reused_frame), req)

    assert payload is None
    new_state = tm._moss_tts_raw_chunk_states["r"]
    assert new_state is not old_state
    assert new_state.final_flushed is False
    assert new_state.sent_control_sentinel is False
    assert new_state.emitted_frame_count == 0
    assert len(new_state.pending_frames) == 1

    remaining_reused_frames = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.long,
    )
    reused_chunk = talker2codec_raw_async_chunk(tm, _payload(remaining_reused_frames), req)

    assert reused_chunk is not None
    assert reused_chunk.codes.audio.shape == (N_VQ, 3)
    assert reused_chunk.meta.stream_finished.item() is False
    assert new_state.emitted_frame_count == 3
    assert new_state.pending_frames == []


def test_finalized_tombstones_are_pruned_to_bound() -> None:
    tm = _tm(chunk_frames=3)

    for i in range(4097):
        req = _req(f"r{i}")
        payload = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)
        assert payload is not None

    states = tm._moss_tts_raw_chunk_states
    assert len(states) == 4096
    assert "r0" not in states
    assert states["r1"].final_flushed is True
