# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hardware E2E smoke for a prepared local NAVA directory.

Set ``NAVA_E2E_MODEL=/path/to/NAVA`` after running
``examples/offline_inference/nava/download_nava.py`` and installing upstream
NAVA so ``import nava_src`` works.
"""

from __future__ import annotations

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

MODEL = os.environ.get("NAVA_E2E_MODEL")

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not MODEL, reason="Set NAVA_E2E_MODEL to a prepared local NAVA directory."),
    pytest.mark.parametrize(
        "omni_runner",
        [
            (
                MODEL or "/tmp/nava-not-set",
                None,
                {
                    "model_class_name": "NAVAPipeline",
                    "enforce_eager": True,
                },
            )
        ],
        indirect=True,
    ),
]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_nava_text_to_audio_video_offline(omni_runner_handler: OmniRunnerHandler) -> None:
    prompt = OmniTextPrompt(prompt="清晨的海边，一名男子沿着湿润的沙滩慢跑，背景里有海浪声和微弱的风声。")
    sampling_params = OmniDiffusionSamplingParams(
        height=704,
        width=1280,
        num_frames=37,
        fps=24,
        num_inference_steps=2,
        seed=42,
    )

    outputs = omni_runner_handler.runner.generate([prompt], [sampling_params])

    assert len(outputs) == 1
    output = outputs[0]
    assert output.images
    assert output.multimodal_output["audio"] is not None
