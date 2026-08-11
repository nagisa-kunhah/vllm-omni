# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.parallel,
]

MODEL_ENV = "VLLM_TEST_MINIMAX_H3_FL2VA_MODEL"


def _assert_joint_output(outputs):
    assert len(outputs) == 1
    frames = np.asarray(outputs[0].images[0])
    assert frames.shape == (107, 256, 448, 3)
    multimodal = outputs[0].multimodal_output
    assert multimodal is not None
    assert np.asarray(multimodal["audio"]).shape[1] == 2
    assert multimodal["audio_sample_rate"] == 32000
    assert multimodal["fps"] == 24


@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
def test_minimax_h3_t2va_ulysses8_smoke():
    import torch

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if torch.accelerator.device_count() < 8:
        pytest.skip("MiniMax H3 Ulysses smoke requires eight GPUs")

    engine = Omni(
        model=os.environ[MODEL_ENV],
        parallel_config=DiffusionParallelConfig(ulysses_degree=8),
        trust_remote_code=True,
        enable_cpu_offload=True,
        enforce_eager=True,
    )
    try:
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=97,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "duration": 4.0,
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )
    finally:
        engine.close()

    assert len(outputs) == 1
    frames = np.asarray(outputs[0].images[0])
    assert frames.shape == (107, 256, 448, 3)
    multimodal = outputs[0].multimodal_output
    assert multimodal is not None
    assert np.asarray(multimodal["audio"]).shape[1] == 2
    assert multimodal["audio_sample_rate"] == 32000
    assert multimodal["fps"] == 24


@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
def test_minimax_h3_t2va_fp8_single_gpu_smoke():
    import torch

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if torch.accelerator.device_count() < 1:
        pytest.skip("MiniMax H3 FP8 smoke requires a CUDA device")

    engine = Omni(
        model=os.environ[MODEL_ENV],
        quantization_config={
            "transformer": {"method": "fp8"},
            "text_encoder": None,
            "video_vae": None,
            "audio_vae": None,
        },
        parallel_config=DiffusionParallelConfig(ulysses_degree=1),
        trust_remote_code=True,
        enable_cpu_offload=True,
        enforce_eager=True,
    )
    try:
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=29,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )
    finally:
        engine.close()

    _assert_joint_output(outputs)


@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
@pytest.mark.parametrize("mode", ("w4a16", "w4a4"))
def test_minimax_h3_t2va_online_nvfp4_tp2_dlo_smoke(monkeypatch, mode: str):
    """Run each text-encoder NVFP4 mode through the TP2/DLO E2E topology."""
    import torch

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if torch.accelerator.device_count() < 2:
        pytest.skip("MiniMax H3 online NVFP4 DLO smoke requires two CUDA GPUs")
    pytest.importorskip("modelopt.torch.quantization.qtensor", reason="ModelOpt is required for online NVFP4")
    monkeypatch.setenv("VLLM_OMNI_H3_TEXT_ENCODER_QUANTIZATION", f"online_nvfp4_{mode}")

    engine = Omni(
        model=os.environ[MODEL_ENV],
        tensor_parallel_size=2,
        text_encoder_tp_size=2,
        vae_patch_parallel_size=2,
        vae_parallel_mode="tile",
        vae_use_tiling=True,
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=False,
        dlo_resident_layers=20,
        diffusion_attention_backend="CUDNN_ATTN",
        trust_remote_code=True,
        enforce_eager=True,
    )
    try:
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=97,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )
    finally:
        engine.close()

    _assert_joint_output(outputs)
