# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
    MossTTSCodecDecoder,
    _moss_codec_codes_from_payload_or_input,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

N_VQ = 4


class _FakeCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(codebook_size=1024)
        self.downsample_rate = 2
        self.decoded_codes: list[torch.Tensor] = []

    def batch_decode(self, codes_list: list[torch.Tensor], num_quantizers: int | None = None) -> SimpleNamespace:
        del num_quantizers
        self.decoded_codes.append(codes_list[0].detach().cpu().clone())
        t = int(codes_list[0].shape[1])
        audio = torch.arange(t * self.downsample_rate, dtype=torch.float32).reshape(1, 1, -1)
        lengths = torch.tensor([audio.shape[-1]], dtype=torch.long)
        return SimpleNamespace(audio=audio, audio_lengths=lengths)


def _decoder() -> MossTTSCodecDecoder:
    decoder = MossTTSCodecDecoder.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder.vllm_config = SimpleNamespace()
    decoder._n_vq = N_VQ
    decoder._codec = _FakeCodec()
    decoder._cuda_graph_wrapper = None
    decoder._n_channels = 1
    decoder._sr_tensor = torch.tensor(24_000, dtype=torch.int32)
    decoder._stream_session = None
    decoder._stream_slots = 0
    decoder._stream_max_step_frames = 100
    decoder._stream_req_slots = {}
    decoder._stream_pending_codes = {}
    decoder._stream_starved_reqs = set()
    return decoder


def _info(audio: Any, *, streaming: bool = False, finished: bool = False, code_flat_numel: int | None = None):
    meta: dict[str, Any] = {
        "codec_streaming": streaming,
        "stream_finished": torch.tensor(finished, dtype=torch.bool),
        "finished": torch.tensor(finished, dtype=torch.bool),
        "req_id": ["r"],
    }
    if code_flat_numel is not None:
        meta["code_flat_numel"] = code_flat_numel
    return {"codes": {"audio": audio}, "meta": meta}


def test_helper_accepts_codec_native_tensor_layout() -> None:
    codes = torch.arange(N_VQ * 3, dtype=torch.long).reshape(N_VQ, 3)

    parsed = _moss_codec_codes_from_payload_or_input(torch.tensor([0]), _info(codes), N_VQ, "cpu")

    assert parsed is not None
    assert parsed.dtype == torch.long
    assert parsed.device.type == "cpu"
    assert torch.equal(parsed, codes)


def test_helper_accepts_frame_major_tensor_layout() -> None:
    frame_major = torch.tensor(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
        ],
        dtype=torch.long,
    )

    parsed = _moss_codec_codes_from_payload_or_input(torch.tensor([0]), _info(frame_major), N_VQ, "cpu")

    assert parsed is not None
    assert torch.equal(parsed, frame_major.transpose(0, 1).contiguous())


@pytest.mark.parametrize("payload", [torch.arange(N_VQ * 2, dtype=torch.long), list(range(N_VQ * 2))])
def test_helper_accepts_legacy_flat_payloads(payload: Any) -> None:
    parsed = _moss_codec_codes_from_payload_or_input(torch.tensor([0]), _info(payload), N_VQ, "cpu")

    assert parsed is not None
    assert torch.equal(parsed, torch.arange(N_VQ * 2, dtype=torch.long).reshape(N_VQ, 2))


def test_helper_falls_back_to_input_segment() -> None:
    input_seg = torch.arange(N_VQ * 2, dtype=torch.long)

    parsed = _moss_codec_codes_from_payload_or_input(input_seg, {"meta": {}}, N_VQ, "cpu")

    assert parsed is not None
    assert torch.equal(parsed, input_seg.reshape(N_VQ, 2))


def test_forward_decodes_connector_tensor_payload_with_placeholder_input_ids() -> None:
    decoder = _decoder()
    codes = torch.arange(N_VQ * 3, dtype=torch.long).reshape(N_VQ, 3)

    out = decoder.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[_info(codes)],
        seq_token_counts=[1],
    )

    assert torch.equal(decoder._codec.decoded_codes[0], codes)
    wav = out.multimodal_outputs["model_outputs"][0]
    assert wav.shape == (6,)
    assert int(out.multimodal_outputs["sr"][0].item()) == 24_000


def test_streaming_path_passes_payload_as_codec_native_layout() -> None:
    decoder = _decoder()
    codes = torch.arange(N_VQ * 2, dtype=torch.long).reshape(N_VQ, 2)
    captured: dict[str, Any] = {}

    def _fake_decode_streaming_batch(self: MossTTSCodecDecoder, items: list[tuple[int, str, torch.Tensor, bool]]):
        del self
        captured["items"] = items
        return {0: torch.tensor([1.0, 2.0])}

    decoder._decode_streaming_batch = MethodType(_fake_decode_streaming_batch, decoder)

    out = decoder.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[_info(codes, streaming=True, finished=False)],
        seq_token_counts=[1],
    )

    [(index, req_key, passed_codes, finished)] = captured["items"]
    assert index == 0
    assert req_key == "r"
    assert finished is False
    assert torch.equal(passed_codes.cpu(), codes)
    assert torch.equal(out.multimodal_outputs["model_outputs"][0], torch.tensor([1.0, 2.0]))
    assert decoder._codec.decoded_codes == []


def test_control_finish_sentinel_does_not_decode_placeholder() -> None:
    decoder = _decoder()

    out = decoder.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[
            _info(torch.tensor([0], dtype=torch.long), streaming=True, finished=True, code_flat_numel=0)
        ],
        seq_token_counts=[1],
    )

    assert decoder._codec.decoded_codes == []
    assert out.multimodal_outputs["model_outputs"][0].numel() == 0
