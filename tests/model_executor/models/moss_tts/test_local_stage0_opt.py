# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for opt-in MOSS-TTS Local Stage-0 helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm")
pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def _local_talker_cls():
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSLocalTalkerForGeneration,
    )

    return MossTTSLocalTalkerForGeneration


class _DummyBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class _DeterministicLocalTransformer:
    def __init__(self, n_vq: int) -> None:
        self.n_vq = n_vq
        self.calls: list[int] = []

    def generate_frame(
        self,
        backbone_last_hidden,
        audio_lm_heads,
        audio_embeddings,
        local_text_lm_head,
        *,
        n_vq,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.0,
        history_per_codebook=None,
        generator=None,
    ):
        batch = int(backbone_last_hidden.shape[0])
        self.calls.append(batch)
        base = backbone_last_hidden[:, :1].round().to(torch.long).clamp_min(0)
        offsets = torch.arange(n_vq, device=backbone_last_hidden.device, dtype=torch.long).reshape(1, n_vq)
        codes = base + offsets
        return torch.ones(batch, dtype=torch.bool, device=backbone_last_hidden.device), codes


def _make_bare_local_talker(hidden_size: int = 4, n_vq: int = 3):
    cls = _local_talker_cls()
    talker = cls.__new__(cls)
    nn.Module.__init__(talker)
    talker.hidden_size = hidden_size
    talker.n_vq = n_vq
    talker.audio_vocab_size = 32
    talker.audio_pad_token_id = 32
    talker.text_vocab_size = 128
    talker.audio_assistant_slot_token_id = 7
    talker.im_end_token_id = 8
    talker.model = _DummyBackbone(talker.text_vocab_size, hidden_size)
    talker.audio_embeddings = nn.ModuleList([nn.Embedding(talker.audio_vocab_size, hidden_size) for _ in range(n_vq)])
    talker.audio_lm_heads = nn.ModuleList([nn.Linear(hidden_size, talker.audio_vocab_size) for _ in range(n_vq)])
    talker.local_text_lm_head = nn.Linear(hidden_size, 2)
    talker.local_transformer = _DeterministicLocalTransformer(n_vq)
    talker._stacked_audio_emb_w = None
    talker._batch_state = None
    talker._batch_state_spans = None
    talker._stage0_frame_graphs = {}
    talker._stage0_frame_graph_input = None
    talker._stage0_frame_graph_continue = None
    talker._stage0_frame_graph_codes = None
    talker._stage0_frame_graph_max_batch = 0
    talker._stage0_frame_graph_bucket_sizes = (1, 2, 4)
    return talker


def test_local_stage0_env_switches_default_off(monkeypatch):
    monkeypatch.delenv("MOSS_TTS_LOCAL_ENABLE_STAGE0_BATCH_PREPROCESS", raising=False)
    monkeypatch.delenv("MOSS_TTS_LOCAL_ENABLE_STAGE0_FRAME_GRAPH", raising=False)

    talker = _make_bare_local_talker()

    assert talker.should_enable_stage0_decode_batch_preprocess()
    assert not talker._decode_stage0_local_frames.__globals__["_env_enabled"](
        "MOSS_TTS_LOCAL_ENABLE_STAGE0_FRAME_GRAPH"
    )


def test_preprocess_decode_batch_returns_shape_and_updates():
    talker = _make_bare_local_talker(hidden_size=4, n_vq=3)
    req_infos = [
        {"hidden_states": {"last": torch.ones(4)}},
        {"hidden_states": {"last": torch.full((4,), 2.0)}},
    ]

    input_ids, embeds, last_hidden, text_step, updates = talker.preprocess_decode_batch(
        input_ids=torch.tensor([11, 12], dtype=torch.long),
        req_infos=req_infos,
    )

    assert input_ids.tolist() == [11, 12]
    assert embeds.shape == (2, 4)
    assert last_hidden.shape == (2, 4)
    assert torch.allclose(last_hidden[0], torch.ones(4))
    assert torch.allclose(last_hidden[1], torch.full((4,), 2.0))
    assert torch.equal(text_step, torch.zeros_like(last_hidden))
    assert updates == [{}, {}]


def test_batched_local_frame_matches_scalar_greedy(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_LOCAL_ENABLE_STAGE0_FRAME_GRAPH", "1")
    talker_batch = _make_bare_local_talker(hidden_size=4, n_vq=3)
    talker_scalar = _make_bare_local_talker(hidden_size=4, n_vq=3)
    hidden = torch.tensor([[1.0, 0, 0, 0], [4.0, 0, 0, 0]])
    input_embeds = torch.zeros((2, 4))
    infos_batch = [
        {"audio_state": {"is_stopping": False, "step": 0, "max_new_frames": -1}},
        {"audio_state": {"is_stopping": False, "step": 0, "max_new_frames": -1}},
    ]
    infos_scalar = [
        {"audio_state": {"is_stopping": False, "step": 0, "max_new_frames": -1}},
        {"audio_state": {"is_stopping": False, "step": 0, "max_new_frames": -1}},
    ]

    out_batch, _ = talker_batch.talker_mtp(
        torch.tensor([7, 7]),
        input_embeds,
        hidden,
        torch.zeros_like(hidden),
        do_sample=False,
        req_infos=infos_batch,
    )
    monkeypatch.delenv("MOSS_TTS_LOCAL_ENABLE_STAGE0_FRAME_GRAPH", raising=False)
    out_scalar, _ = talker_scalar.talker_mtp(
        torch.tensor([7, 7]),
        input_embeds,
        hidden,
        torch.zeros_like(hidden),
        do_sample=False,
        req_infos=infos_scalar,
    )

    assert torch.allclose(out_batch, out_scalar)
    assert talker_batch.local_transformer.calls == [2]
    assert talker_scalar.local_transformer.calls == [1, 1]
    for info_b, info_s in zip(infos_batch, infos_scalar, strict=True):
        assert torch.equal(info_b["audio_codes"]["current"], info_s["audio_codes"]["current"])


def test_frame_graph_failure_falls_back_to_eager(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_LOCAL_ENABLE_STAGE0_FRAME_GRAPH", "1")
    talker = _make_bare_local_talker(hidden_size=4, n_vq=3)

    def fail_graph(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(talker, "_decode_stage0_local_frames_graph", fail_graph)

    should_continue, codes = talker._decode_stage0_local_frames(
        torch.tensor([[2.0, 0, 0, 0]]),
        do_sample=False,
        temperature=1.7,
        top_k=25,
        top_p=0.8,
        generator=None,
    )

    assert should_continue.tolist() == [True]
    assert codes.tolist() == [[2, 3, 4]]
    assert talker.local_transformer.calls == [1]


def test_make_omni_output_emits_batch_aligned_finished_meta():
    talker = _make_bare_local_talker(hidden_size=4, n_vq=3)
    info_dicts = [
        {
            "audio_state": {"is_stopping": False},
            "audio_codes": {"current": torch.tensor([[1, 2, 3]]), "emit": True},
        },
        {
            "audio_state": {"is_stopping": True},
            "audio_codes": {"current": torch.empty((0, 3), dtype=torch.long), "emit": False},
        },
    ]

    output = talker.make_omni_output(
        torch.zeros((2, 4)),
        runtime_additional_information=info_dicts,
        request_token_spans=[(0, 1), (1, 2)],
    )

    assert len(output.multimodal_outputs["codes"]["audio"]) == 2
    assert torch.equal(output.multimodal_outputs["codes"]["audio"][0], torch.tensor([[1, 2, 3]]))
    assert output.multimodal_outputs["codes"]["audio"][1].numel() == 0
    finished = output.multimodal_outputs["meta"]["finished"]
    assert [bool(flag.item()) for flag in finished] == [False, True]
