# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online serving E2E smoke for a prepared local NAVA directory."""

from __future__ import annotations

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("NAVA_E2E_MODEL")
_SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=1)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not MODEL, reason="Set NAVA_E2E_MODEL to a prepared local NAVA directory."),
]


def _server_cases() -> list[pytest.ParamSpec]:
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL or "/tmp/nava-not-set",
                server_args=[
                    "--model-class-name",
                    "NAVAPipeline",
                    "--enforce-eager",
                    "--disable-log-stats",
                ],
                init_timeout=1200,
                stage_init_timeout=1200,
            ),
            id="text_to_audio_video",
            marks=_SINGLE_CARD_MARKS,
        )
    ]


@pytest.mark.parametrize("omni_server", _server_cases(), indirect=True)
def test_nava_text_to_audio_video_online(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": "清晨的海边，一名男子沿着湿润的沙滩慢跑，背景里有海浪声和微弱的风声。",
            "height": 704,
            "width": 1280,
            "num_frames": 37,
            "fps": 24,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }

    response = openai_client.send_video_diffusion_request(request_config)

    assert response[0].success
    assert response[0].videos
