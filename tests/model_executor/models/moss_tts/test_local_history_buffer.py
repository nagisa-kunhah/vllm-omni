# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MOSS-TTS Local Stage-0 audio history buffering."""

from __future__ import annotations

import functools
from typing import Any

import pytest

torch = pytest.importorskip("torch")

N_VQ = 4
HIDDEN = 8


@functools.lru_cache(maxsize=1)
def _local_talker_symbols():
    """Defer import because the talker pulls in vLLM model_executor modules."""
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSLocalTalkerForGeneration,
        _moss_local_materialize_history,
    )

    return MossTTSLocalTalkerForGeneration, _moss_local_materialize_history


class _FakeLocalTransformer:
    def __init__(self, frames: list[torch.Tensor], continues: list[bool] | None = None) -> None:
        self.frames = frames
        self.continues = continues if continues is not None else [True] * len(frames)
        self.histories: list[list[list[int]]] = []

    def generate_frame(
        self,
        last_talker_hidden: torch.Tensor,
        audio_lm_heads: Any,
        audio_embeddings: Any,
        local_text_lm_head: Any,
        *,
        n_vq: int,
        history_per_codebook: list[list[int]],
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del audio_lm_heads, audio_embeddings, local_text_lm_head, n_vq
        self.histories.append([list(values) for values in history_per_codebook])
        index = len(self.histories) - 1
        frame = self.frames[index].to(device=last_talker_hidden.device, dtype=torch.long).reshape(1, -1)
        should_continue = torch.tensor([self.continues[index]], device=last_talker_hidden.device)
        return should_continue, frame


def _make_bare_local_talker(frames: list[torch.Tensor], continues: list[bool] | None = None):
    cls, _ = _local_talker_symbols()
    talker = cls.__new__(cls)
    torch.nn.Module.__init__(talker)
    talker.n_vq = N_VQ
    talker.hidden_size = HIDDEN
    talker.local_transformer = _FakeLocalTransformer(frames, continues)
    talker.audio_lm_heads = torch.nn.ModuleList()
    talker.audio_embeddings = torch.nn.ModuleList()
    talker.local_text_lm_head = torch.nn.Linear(HIDDEN, 2, bias=False)
    talker._batch_state = None
    talker._batch_state_spans = None

    def _audio_embed(codes: torch.Tensor) -> torch.Tensor:
        return torch.zeros((codes.shape[0], HIDDEN), device=codes.device, dtype=torch.float32)

    talker._audio_embed = _audio_embed
    return talker


def _run_talker_step(talker, info: dict[str, Any]) -> None:
    input_embeds = torch.zeros((1, HIDDEN), dtype=torch.float32)
    last_hidden = torch.zeros((1, HIDDEN), dtype=torch.float32)
    input_ids = torch.zeros((1,), dtype=torch.long)
    text_step = torch.zeros((1,), dtype=torch.long)
    talker.talker_mtp(
        input_ids,
        input_embeds,
        last_hidden,
        text_step,
        req_infos=[info],
    )


def _history_tensor(info: dict[str, Any]) -> torch.Tensor:
    _, materialize = _local_talker_symbols()
    return materialize(info["audio_codes"]["accumulated"])


def test_local_talker_accumulates_into_cpu_history_buffer() -> None:
    frames = [torch.arange(N_VQ) + offset for offset in (0, 10, 20)]
    talker = _make_bare_local_talker(frames)
    info: dict[str, Any] = {"audio_state": {"step": 0, "max_new_frames": -1}}

    for _ in frames:
        _run_talker_step(talker, info)

    accumulated = info["audio_codes"]["accumulated"]
    assert isinstance(accumulated, dict)
    assert accumulated["length"] == 3
    assert accumulated["buffer"].device.type == "cpu"
    assert accumulated["buffer"].shape == (64, N_VQ)
    assert torch.equal(_history_tensor(info), torch.stack(frames))


def test_local_talker_hot_path_does_not_call_torch_cat(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [torch.arange(N_VQ), torch.arange(N_VQ) + 10]
    talker = _make_bare_local_talker(frames)
    info: dict[str, Any] = {"audio_state": {"step": 0, "max_new_frames": -1}}

    def _fail_cat(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("torch.cat must not be used for Local Stage-0 audio history")

    monkeypatch.setattr(torch, "cat", _fail_cat)

    _run_talker_step(talker, info)
    _run_talker_step(talker, info)

    assert torch.equal(_history_tensor(info), torch.stack(frames))


def test_local_talker_converts_legacy_tensor_history_and_preserves_order() -> None:
    legacy = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    frame = torch.tensor([9, 10, 11, 12], dtype=torch.long)
    talker = _make_bare_local_talker([frame])
    info: dict[str, Any] = {
        "audio_state": {"step": 0, "max_new_frames": -1},
        "audio_codes": {"accumulated": legacy},
    }

    _run_talker_step(talker, info)

    assert isinstance(info["audio_codes"]["accumulated"], dict)
    assert torch.equal(_history_tensor(info), torch.cat([legacy, frame.reshape(1, -1)], dim=0))


def test_local_talker_passes_only_last_50_frames_by_codebook() -> None:
    legacy = torch.arange(55 * N_VQ, dtype=torch.long).reshape(55, N_VQ)
    stop_frame = torch.full((N_VQ,), 999, dtype=torch.long)
    talker = _make_bare_local_talker([stop_frame], continues=[False])
    info: dict[str, Any] = {
        "audio_state": {"step": 0, "max_new_frames": -1},
        "audio_codes": {"accumulated": legacy},
    }

    _run_talker_step(talker, info)

    expected_tail = legacy[-50:]
    expected = [[int(row[cb].item()) for row in expected_tail] for cb in range(N_VQ)]
    assert talker.local_transformer.histories == [expected]
    assert torch.equal(_history_tensor(info), legacy)


def test_local_talker_does_not_append_stop_frame() -> None:
    legacy = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    stop_frame = torch.tensor([9, 9, 9, 9], dtype=torch.long)
    talker = _make_bare_local_talker([stop_frame], continues=[False])
    info: dict[str, Any] = {
        "audio_state": {"step": 0, "max_new_frames": -1},
        "audio_codes": {"accumulated": legacy},
    }

    _run_talker_step(talker, info)

    assert torch.equal(_history_tensor(info), legacy)
    assert info["audio_codes"]["emit"] is False
    assert info["audio_codes"]["current"].shape == (0, N_VQ)
    assert info["audio_state"]["is_stopping"] is True


def test_local_talker_does_not_append_when_max_new_frames_reached() -> None:
    legacy = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    next_frame = torch.tensor([9, 9, 9, 9], dtype=torch.long)
    talker = _make_bare_local_talker([next_frame])
    info: dict[str, Any] = {
        "audio_state": {"step": 1, "max_new_frames": 1},
        "audio_codes": {"accumulated": legacy},
    }

    _run_talker_step(talker, info)

    assert torch.equal(_history_tensor(info), legacy)
    assert info["audio_codes"]["emit"] is False
    assert info["audio_codes"]["current"].shape == (0, N_VQ)
    assert info["audio_state"]["is_stopping"] is True


def test_local_make_omni_output_emits_current_delta_only() -> None:
    talker = _make_bare_local_talker([])
    hidden = torch.zeros((1, HIDDEN), dtype=torch.float32)
    prior_history = {
        "buffer": torch.arange(3 * N_VQ, dtype=torch.long).reshape(3, N_VQ),
        "length": 3,
    }
    current = torch.tensor([[40, 41, 42, 43]], dtype=torch.long)
    info = {"audio_codes": {"accumulated": prior_history, "current": current, "emit": True}}

    result = talker.make_omni_output(
        hidden,
        runtime_additional_information=[info],
        request_token_spans=[(0, 1)],
    )

    audio = result.multimodal_outputs["codes"]["audio"]
    assert len(audio) == 1
    assert torch.equal(audio[0], current)
