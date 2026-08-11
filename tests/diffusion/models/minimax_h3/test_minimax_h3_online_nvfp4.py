# SPDX-License-Identifier: Apache-2.0
"""CUDA/TP2 smoke coverage for MiniMax-H3 online NVFP4 text linears."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.distributed_cuda(num_cards=2),
]


def _online_nvfp4_tp2_worker(rank: int, init_method: str, mode: str) -> None:
    """Exercise one online NVFP4 execution method on both TP ranks."""
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    torch.cuda.set_device(rank)

    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLMergedColumnParallelLinear,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(tensor_model_parallel_size=2)
        try:
            linear = MiniMaxH3Qwen3VLMergedColumnParallelLinear(
                get_tp_group(),
                input_size=64,
                intermediate_size=128,
                dtype=torch.bfloat16,
                online_nvfp4=mode,
            ).to(f"cuda:{rank}")
            with torch.no_grad():
                linear.weight.uniform_(-0.02, 0.02)
            source_weight = linear.weight.detach().clone()

            source_nbytes, packed_state_nbytes = linear.quantize_for_inference()
            assert source_nbytes > packed_state_nbytes > 0
            assert "weight" not in linear._parameters

            packed_layer = linear._online_nvfp4_layer
            assert packed_layer is not None
            if mode == "w4a16":
                # Marlin repacks its initial uint8 FP4 representation to int32
                # words on supported hardware.
                assert packed_layer.weight.dtype in (torch.uint8, torch.int32)
            else:
                assert packed_layer.weight.dtype is torch.uint8
                assert packed_layer.input_global_scale_inv.dtype is torch.float32
            assert packed_layer.weight_scale.dtype is torch.float8_e4m3fn
            assert packed_layer.weight_global_scale.dtype is torch.float32

            inputs = torch.randn(8, 64, device=f"cuda:{rank}", dtype=torch.bfloat16)
            reference = F.linear(inputs, source_weight)
            outputs = linear(inputs)
            torch.accelerator.synchronize(rank)
            assert outputs.dtype is torch.bfloat16
            assert torch.isfinite(outputs).all()
            if mode == "w4a4":
                assert linear._online_nvfp4_input_calibrated
                assert (outputs.float() - reference.float()).abs().mean() < 0.02
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="online NVFP4 TP smoke requires two CUDA GPUs",
)
@pytest.mark.parametrize("mode", ("w4a16", "w4a4"))
def test_online_nvfp4_tp2_conversion_releases_bf16_weight_and_runs_bf16_output(mode: str):
    """Both explicit W4 modes create packed state and emit finite BF16 output."""
    pytest.importorskip("modelopt.torch.quantization.qtensor", reason="ModelOpt is required for online NVFP4")
    with tempfile.TemporaryDirectory() as directory:
        init_method = f"file://{Path(directory) / 'distributed_init'}"
        torch.multiprocessing.spawn(
            _online_nvfp4_tp2_worker,
            args=(init_method, mode),
            nprocs=2,
            join=True,
        )
