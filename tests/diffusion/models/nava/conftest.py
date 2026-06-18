# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight import shims for focused NAVA unit tests.

These tests exercise NAVA request parsing, postprocess, registration, and local
weight-path logic. They do not need vLLM's compiled CUDA extension, which may be
unavailable in CPU-only developer environments.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _module(name: str, *, package_path: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    if package_path is not None:
        module.__path__ = [str(package_path)]
    return module


def _build_import_shims() -> dict[str, object]:
    vllm_pkg = _module("vllm", package_path=Path("/tmp/vllm-test-shim"))
    diffusers = _module("diffusers")
    diffusers.DiffusionPipeline = type("DiffusionPipeline", (), {})

    vllm_config_pkg = _module("vllm.config", package_path=Path("/tmp/vllm-test-shim/config"))
    vllm_logger = _module("vllm.logger")
    vllm_logger.init_logger = lambda _name: MagicMock()

    vllm_config_utils = _module("vllm.config.utils")
    vllm_config_utils.config = lambda cls: cls

    vllm_model_executor = _module("vllm.model_executor", package_path=Path("/tmp/vllm-test-shim/model_executor"))
    vllm_layers = _module("vllm.model_executor.layers", package_path=Path("/tmp/vllm-test-shim/layers"))
    vllm_quant_pkg = _module(
        "vllm.model_executor.layers.quantization",
        package_path=Path("/tmp/vllm-test-shim/quantization"),
    )
    vllm_quant_base = _module("vllm.model_executor.layers.quantization.base_config")
    vllm_quant_base.QuantizationConfig = object

    vllm_model_loader = _module(
        "vllm.model_executor.model_loader",
        package_path=Path("/tmp/vllm-test-shim/model_loader"),
    )
    vllm_loader_utils = _module("vllm.model_executor.model_loader.utils")
    vllm_loader_utils.configure_quant_config = lambda *_args, **_kwargs: None

    class _LazyRegisteredModel:
        def __init__(self, *, module_name: str, class_name: str):
            self.module_name = module_name
            self.class_name = class_name

        def load_model_cls(self):
            module = __import__(self.module_name, fromlist=[self.class_name])
            return getattr(module, self.class_name)

    class _ModelRegistry:
        def __init__(self, models: dict[str, _LazyRegisteredModel]):
            self.models = dict(models)

        def _try_load_model_cls(self, model_arch: str | None):
            if model_arch not in self.models:
                return None
            return self.models[model_arch].load_model_cls()

        def register_model(self, model_arch: str, qualified_name: str):
            module_name, class_name = qualified_name.split(":", 1)
            self.models[model_arch] = _LazyRegisteredModel(module_name=module_name, class_name=class_name)

    vllm_models_pkg = _module("vllm.model_executor.models", package_path=Path("/tmp/vllm-test-shim/models"))
    vllm_model_registry = _module("vllm.model_executor.models.registry")
    vllm_model_registry._LazyRegisteredModel = _LazyRegisteredModel
    vllm_model_registry._ModelRegistry = _ModelRegistry

    vllm_inputs = _module("vllm.inputs")
    vllm_inputs.PromptType = object
    vllm_inputs.TextPrompt = dict
    vllm_inputs.TokensPrompt = dict
    vllm_inputs.EmbedsPrompt = dict

    vllm_inputs_engine = _module("vllm.inputs.engine")
    vllm_inputs_engine.TokensInput = dict

    vllm_engine_pkg = _module("vllm.engine", package_path=Path("/tmp/vllm-test-shim/engine"))
    vllm_engine_protocol = _module("vllm.engine.protocol")
    vllm_engine_protocol.EngineClient = object

    vllm_sampling_params = _module("vllm.sampling_params")
    vllm_sampling_params.SamplingParams = object

    vllm_outputs = _module("vllm.outputs")
    vllm_outputs.RequestOutput = object

    vllm_v1_pkg = _module("vllm.v1", package_path=Path("/tmp/vllm-test-shim/v1"))
    vllm_v1_outputs = _module("vllm.v1.outputs")
    vllm_v1_outputs.ModelRunnerOutput = object

    vllm_lora_pkg = _module("vllm.lora", package_path=Path("/tmp/vllm-test-shim/lora"))
    vllm_lora_request = _module("vllm.lora.request")
    vllm_lora_request.LoRARequest = object

    vllm_omni_platforms = _module("vllm_omni.platforms")
    vllm_omni_platforms.current_omni_platform = MagicMock()
    vllm_omni_platforms.current_omni_platform.is_available.return_value = False
    vllm_omni_platforms.current_omni_platform.is_initialized.return_value = False
    vllm_omni_platforms.current_omni_platform.get_torch_device.return_value = torch.device("cpu")
    vllm_omni_platforms.current_omni_platform.get_diffusion_packed_modules_mapping.return_value = None

    vllm_omni_quantization = _module("vllm_omni.quantization")
    vllm_omni_quantization.build_quant_config = lambda _raw: None

    vllm_omni_pkg = _module("vllm_omni", package_path=_REPO_ROOT / "vllm_omni")

    distributed_pkg = _module("vllm_omni.diffusion.distributed", package_path=Path("/tmp/omni-dist-shim"))
    distributed_utils = _module("vllm_omni.diffusion.distributed.utils")
    distributed_utils.get_local_device = lambda: torch.device("cpu")

    autoencoders_pkg = _module(
        "vllm_omni.diffusion.distributed.autoencoders",
        package_path=Path("/tmp/omni-dist-shim/autoencoders"),
    )
    vae_executor = _module("vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor")
    vae_executor.DistributedVaeMixin = type("DistributedVaeMixin", (), {})

    sp_plan = _module("vllm_omni.diffusion.distributed.sp_plan")
    sp_plan.SequenceParallelConfig = type("SequenceParallelConfig", (), {"__init__": lambda self, **_: None})
    sp_plan.get_sp_plan_from_model = lambda model: getattr(model, "_sp_plan", None)

    forward_context = _module("vllm_omni.diffusion.forward_context")
    forward_context.get_forward_context = lambda: SimpleNamespace(sp_plan_hooks_applied=False)

    sequence_parallel = _module("vllm_omni.diffusion.hooks.sequence_parallel")
    sequence_parallel.apply_sequence_parallel = lambda *_args, **_kwargs: None

    tf_utils = _module("vllm_omni.diffusion.utils.tf_utils")
    tf_utils.find_module_with_attr = lambda *_args, **_kwargs: None

    return {
        "diffusers": diffusers,
        "vllm": vllm_pkg,
        "vllm.config": vllm_config_pkg,
        "vllm.logger": vllm_logger,
        "vllm.config.utils": vllm_config_utils,
        "vllm.model_executor": vllm_model_executor,
        "vllm.model_executor.layers": vllm_layers,
        "vllm.model_executor.layers.quantization": vllm_quant_pkg,
        "vllm.model_executor.layers.quantization.base_config": vllm_quant_base,
        "vllm.model_executor.model_loader": vllm_model_loader,
        "vllm.model_executor.model_loader.utils": vllm_loader_utils,
        "vllm.model_executor.models": vllm_models_pkg,
        "vllm.model_executor.models.registry": vllm_model_registry,
        "vllm.inputs": vllm_inputs,
        "vllm.inputs.engine": vllm_inputs_engine,
        "vllm.engine": vllm_engine_pkg,
        "vllm.engine.protocol": vllm_engine_protocol,
        "vllm.sampling_params": vllm_sampling_params,
        "vllm.outputs": vllm_outputs,
        "vllm.v1": vllm_v1_pkg,
        "vllm.v1.outputs": vllm_v1_outputs,
        "vllm.lora": vllm_lora_pkg,
        "vllm.lora.request": vllm_lora_request,
        "vllm_omni": vllm_omni_pkg,
        "vllm_omni.platforms": vllm_omni_platforms,
        "vllm_omni.quantization": vllm_omni_quantization,
        "vllm_omni.diffusion.distributed": distributed_pkg,
        "vllm_omni.diffusion.distributed.utils": distributed_utils,
        "vllm_omni.diffusion.distributed.autoencoders": autoencoders_pkg,
        "vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor": vae_executor,
        "vllm_omni.diffusion.distributed.sp_plan": sp_plan,
        "vllm_omni.diffusion.forward_context": forward_context,
        "vllm_omni.diffusion.hooks.sequence_parallel": sequence_parallel,
        "vllm_omni.diffusion.utils.tf_utils": tf_utils,
    }


def pytest_configure(config: pytest.Config) -> None:
    try:
        import vllm_omni.diffusion.models.nava.config  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        message = str(exc)
        is_local_binary_issue = (
            "vllm/_C" in message
            or "vllm._C" in message
            or "torchvision::nms" in message
            or "DiffusionPipeline" in message
            or "AutoImageProcessor" in message
        )
        if not is_local_binary_issue:
            raise
    else:
        return

    patcher = pytest.MonkeyPatch()
    shims = _build_import_shims()
    for name, module in shims.items():
        patcher.setitem(sys.modules, name, module)
    config.add_cleanup(patcher.undo)
