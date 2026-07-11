# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import (
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


def test_finish_with_no_pending_frames_emits_control_sentinel() -> None:
    tm = _tm(chunk_frames=3)

    payload = talker2codec_raw_async_chunk(tm, None, _req("r"), is_finished=True)

    assert payload is not None
    assert torch.equal(payload.codes.audio, torch.tensor([0], dtype=torch.long))
    assert payload.meta.codec_chunk_frames == 0
    assert payload.meta.code_flat_numel == 0
    assert payload.meta.stream_finished.item() is True
    assert payload.meta.finished.item() is True
