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


def _sample_token_stable():
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
        _sample_token_stable,
    )

    return _sample_token_stable


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

    def generate_frame_graphable(
        self,
        backbone_last_hidden,
        audio_lm_heads,
        audio_embeddings,
        local_text_lm_head,
        *,
        n_vq,
        seed,
        frame_step,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        text_temperature=1.0,
        text_top_k=50,
        text_top_p=1.0,
    ):
        batch = int(backbone_last_hidden.shape[0])
        self.calls.append(batch)
        seed_base = seed.reshape(-1, 1).to(torch.long).remainder(1000)
        offsets = torch.arange(n_vq, device=backbone_last_hidden.device, dtype=torch.long).reshape(1, n_vq)
        codes = seed_base + frame_step.reshape(-1, 1).to(torch.long) + offsets
        audio_embed = torch.zeros(batch, backbone_last_hidden.shape[-1], device=backbone_last_hidden.device)
        audio_embed[:, 0] = codes[:, 0].to(audio_embed.dtype)
        return torch.ones(batch, dtype=torch.bool, device=backbone_last_hidden.device), codes, audio_embed


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
    talker._stage0_frame_graph_seed = None
    talker._stage0_frame_graph_frame_step = None
    talker._stage0_frame_graph_continue = None
    talker._stage0_frame_graph_codes = None
    talker._stage0_frame_graph_audio_embed = None
    talker._stage0_frame_graph_max_batch = 0
    talker._stage0_frame_graph_bucket_sizes = (1, 2, 4)
    talker._stage0_frame_graph_stats = {
        "capture_count": 0,
        "replay_count": 0,
        "eager_fallback_count": 0,
        "fallback_by_reason": {},
        "bucket_usage": {},
    }
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


def test_stable_sampler_is_batch_order_independent():
    sample_token = _sample_token_stable()
    logits_ab = torch.tensor([[0.1, 0.4, 0.8, 1.2], [1.2, 0.8, 0.4, 0.1]], dtype=torch.float32)
    seed_ab = torch.tensor([11, 22], dtype=torch.long)
    step_ab = torch.tensor([3, 5], dtype=torch.long)

    out_ab = sample_token(
        logits_ab,
        seed_ab,
        step_ab,
        codebook_index=0,
        sample_kind=1,
        temperature=1.7,
        top_k=4,
        top_p=1.0,
        do_sample=True,
    )
    out_ba = sample_token(
        logits_ab.flip(0),
        seed_ab.flip(0),
        step_ab.flip(0),
        codebook_index=0,
        sample_kind=1,
        temperature=1.7,
        top_k=4,
        top_p=1.0,
        do_sample=True,
    )

    assert torch.equal(out_ab, out_ba.flip(0))


def test_stable_sampler_greedy_and_mask_edges():
    sample_token = _sample_token_stable()
    seed = torch.tensor([1, 2], dtype=torch.long)
    step = torch.tensor([0, 0], dtype=torch.long)
    logits = torch.tensor([[0.1, 3.0, 2.0], [4.0, 1.0, 0.5]], dtype=torch.float32)

    greedy = sample_token(
        logits,
        seed,
        step,
        codebook_index=1,
        sample_kind=2,
        temperature=1.0,
        top_k=3,
        top_p=1.0,
        do_sample=False,
    )
    top_k_one = sample_token(
        logits,
        seed,
        step,
        codebook_index=1,
        sample_kind=2,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        do_sample=True,
    )
    all_masked = sample_token(
        torch.full((2, 3), float("-inf")),
        seed,
        step,
        codebook_index=1,
        sample_kind=2,
        temperature=1.0,
        top_k=3,
        top_p=0.5,
        do_sample=True,
    )

    assert greedy.tolist() == [1, 0]
    assert top_k_one.tolist() == [1, 0]
    assert all_masked.shape == (2,)
    assert torch.isfinite(all_masked.float()).all()
    assert all_masked.min().item() >= 0
    assert all_masked.max().item() < 3


def test_unseeded_graph_enabled_path_keeps_scalar_eager(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_LOCAL_ENABLE_STAGE0_FRAME_GRAPH", "1")
    talker = _make_bare_local_talker(hidden_size=4, n_vq=3)
    talker_scalar = _make_bare_local_talker(hidden_size=4, n_vq=3)
    talker_scalar.audio_embeddings.load_state_dict(talker.audio_embeddings.state_dict())
    hidden = torch.tensor([[1.0, 0, 0, 0], [4.0, 0, 0, 0]])
    input_embeds = torch.zeros((2, 4))
    infos = [
        {"audio_state": {"is_stopping": False, "step": 0, "max_new_frames": -1}},
        {"audio_state": {"is_stopping": False, "step": 0, "max_new_frames": -1}},
    ]
    infos_scalar = [
        {"audio_state": {"is_stopping": False, "step": 0, "max_new_frames": -1}},
        {"audio_state": {"is_stopping": False, "step": 0, "max_new_frames": -1}},
    ]

    out, _ = talker.talker_mtp(
        torch.tensor([7, 7]),
        input_embeds,
        hidden,
        torch.zeros_like(hidden),
        do_sample=False,
        req_infos=infos,
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

    assert torch.allclose(out, out_scalar)
    assert talker.local_transformer.calls == [1, 1]
    assert talker_scalar.local_transformer.calls == [1, 1]
    assert talker.get_stage0_frame_graph_stats()["fallback_by_reason"]["no_seed"] == 2
    for info_b, info_s in zip(infos, infos_scalar, strict=True):
        assert torch.equal(info_b["audio_codes"]["current"], info_s["audio_codes"]["current"])


def test_seeded_graphable_batch_is_stable_under_row_reordering(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_LOCAL_ENABLE_STAGE0_FRAME_GRAPH", "1")
    talker_ab = _make_bare_local_talker(hidden_size=4, n_vq=3)
    talker_ba = _make_bare_local_talker(hidden_size=4, n_vq=3)
    hidden_ab = torch.tensor([[1.0, 0, 0, 0], [4.0, 0, 0, 0]])
    hidden_ba = hidden_ab.flip(0)
    input_embeds = torch.zeros((2, 4))
    infos_ab = [
        {"tts_local_seed": 11, "audio_state": {"is_stopping": False, "step": 2, "max_new_frames": -1}},
        {"tts_local_seed": 22, "audio_state": {"is_stopping": False, "step": 5, "max_new_frames": -1}},
    ]
    infos_ba = [
        {"tts_local_seed": 22, "audio_state": {"is_stopping": False, "step": 5, "max_new_frames": -1}},
        {"tts_local_seed": 11, "audio_state": {"is_stopping": False, "step": 2, "max_new_frames": -1}},
    ]

    talker_ab.talker_mtp(
        torch.tensor([7, 7]),
        input_embeds,
        hidden_ab,
        torch.zeros_like(hidden_ab),
        req_infos=infos_ab,
    )
    talker_ba.talker_mtp(
        torch.tensor([7, 7]),
        input_embeds,
        hidden_ba,
        torch.zeros_like(hidden_ba),
        req_infos=infos_ba,
    )

    assert talker_ab.local_transformer.calls == [2]
    assert talker_ba.local_transformer.calls == [2]
    assert torch.equal(infos_ab[0]["audio_codes"]["current"], infos_ba[1]["audio_codes"]["current"])
    assert torch.equal(infos_ab[1]["audio_codes"]["current"], infos_ba[0]["audio_codes"]["current"])
    assert talker_ab.get_stage0_frame_graph_stats()["fallback_by_reason"]["cpu_device"] == 2


def test_frame_graph_failure_falls_back_to_eager(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_LOCAL_ENABLE_STAGE0_FRAME_GRAPH", "1")
    talker = _make_bare_local_talker(hidden_size=4, n_vq=3)

    def fail_graph(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(talker, "_decode_stage0_local_frames_graph", fail_graph)

    should_continue, codes, audio_embed = talker._decode_stage0_local_frames(
        torch.tensor([[2.0, 0, 0, 0]]),
        seed=torch.tensor([123], dtype=torch.long),
        frame_step=torch.tensor([0], dtype=torch.long),
        do_sample=False,
        temperature=1.7,
        top_k=25,
        top_p=0.8,
        generator=None,
    )

    assert should_continue.tolist() == [True]
    assert codes.tolist() == [[2, 3, 4]]
    assert audio_embed is not None
    assert talker.local_transformer.calls == [1]
    assert talker.get_stage0_frame_graph_stats()["fallback_by_reason"]["capture_failed"] == 1


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
