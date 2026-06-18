# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.io_support import supports_audio_output, supports_multimodal_input
from vllm_omni.diffusion.models.nava import pipeline_nava as nava_pipeline_module
from vllm_omni.diffusion.models.nava.config import (
    NAVAConfig,
    count_speech_spans,
    inject_speaker_sentinel,
    parse_speech_spans,
)
from vllm_omni.diffusion.models.nava.pipeline_nava import (
    NAVAPipeline,
    _temporarily_disable_torch_compile,
    get_nava_post_process_func,
)
from vllm_omni.diffusion.output_formatter import (
    DiffusionStepTimings,
    format_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.registry import (
    _DIFFUSION_MODELS,
    _DIFFUSION_POST_PROCESS_FUNCS,
    _NO_CACHE_ACCELERATION,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOWNLOAD_SCRIPT_PATH = _REPO_ROOT / "examples" / "offline_inference" / "nava" / "download_nava.py"
_END2END_SCRIPT_PATH = _REPO_ROOT / "examples" / "offline_inference" / "nava" / "end2end.py"
_ONLINE_DIR = _REPO_ROOT / "examples" / "online_serving" / "nava"


def _make_nava_pipeline_shell(
    model_dir: str | Path = "/tmp/nava",
    *,
    model_config: dict | None = None,
    custom_pipeline_args: dict | None = None,
) -> NAVAPipeline:
    pipeline = object.__new__(NAVAPipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(
        model=str(model_dir),
        model_class_name="NAVAPipeline",
        model_config=model_config or {},
        custom_pipeline_args=custom_pipeline_args or {},
    )
    pipeline.nava_config = NAVAConfig()
    pipeline.audio_sample_rate = 16000
    pipeline.video_vae = SimpleNamespace()
    pipeline.audio_vae = SimpleNamespace()
    return pipeline


def _make_request(prompt, **sampling_kwargs) -> OmniDiffusionRequest:
    sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)
    return OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=sampling_params,
        request_id="nava-test",
    )


def _write_yaml_model_config(path: Path, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "model_type: NAVA",
                f"log_height: {height}",
                "log_width: 1280",
                "data:",
                "  video_fps: 24",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_model_index(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def _load_download_script():
    spec = importlib.util.spec_from_file_location("nava_download_script", _DOWNLOAD_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_download_model_dir(path: Path) -> None:
    (path / "configs").mkdir()
    (path / "configs" / "nava.yaml").write_text("model_type: NAVA\n", encoding="utf-8")
    (path / "NAVA.safetensors").write_bytes(b"")
    (path / "Wan2.2-TI2V-5B").mkdir()
    (path / "params").mkdir()


def _load_end2end_script():
    spec = importlib.util.spec_from_file_location("nava_end2end_script", _END2END_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_args(**kwargs) -> Namespace:
    values = {
        "audio_align_guidance_scale": 2.0,
        "audio_guidance_scale": 2.0,
        "disable_timbre_cfg": False,
        "fps": 24,
        "frames": 37,
        "height": 704,
        "image": None,
        "nava_weight_dtype": "auto",
        "num_inference_steps": 50,
        "prompt": "a person speaks",
        "seed": 100,
        "spk_wavs": None,
        "timbre_align_guidance_scale": 3.0,
        "video_align_guidance_scale": 3.0,
        "video_guidance_scale": 3.0,
        "width": 1280,
    }
    values.update(kwargs)
    return Namespace(**values)


class _FakeBackbone:
    def __init__(self) -> None:
        self.rope_set = False

    def set_rope_params(self) -> None:
        self.rope_set = True


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.backbone = _FakeBackbone()


class _FakeVideoVAE:
    def encode(self, image, **kwargs):
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: torch.zeros(1, 44, 80, 48)))


class _FakeAudioVAE:
    def encode(self, payload):
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: {"spk_embs": torch.ones(1, 192)}))


class _FakeAudioVideoPipeline:
    created_kwargs: dict | None = None

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.text_model = SimpleNamespace(model=object())
        self.video_vae = _FakeVideoVAE()
        self.audio_vae = _FakeAudioVAE()
        self.sample_kwargs: dict | None = None
        self.sample_batch: dict | None = None
        self.to_device = None

    @classmethod
    def create(cls, **kwargs):
        cls.created_kwargs = kwargs
        return cls()

    def to(self, device):
        self.to_device = device
        return self

    def sample(self, batch, **kwargs):
        self.sample_batch = batch
        assert batch["spk_embs"] is None or isinstance(batch["spk_embs"], list)
        self.sample_kwargs = kwargs
        video = torch.zeros(1, 2, 3, 4, 5)
        audio = [{"waveform": torch.ones(160), "sample_rate": 16000}]
        return video, audio


def _install_fake_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("nava_src")
    package.__path__ = []
    module = types.ModuleType("nava_src.pipeline_nava")
    module.AudioVideoPipeline = _FakeAudioVideoPipeline
    _FakeAudioVideoPipeline.created_kwargs = None
    monkeypatch.setitem(sys.modules, "nava_src", package)
    monkeypatch.setitem(sys.modules, "nava_src.pipeline_nava", module)


def _write_fake_upstream_model_dir(path: Path) -> None:
    (path / "configs").mkdir()
    (path / "configs" / "nava.yaml").write_text(
        "\n".join(
            [
                "model_type: NAVA",
                "modality: audio_video",
                "log_height: 704",
                "log_width: 1280",
                "data:",
                "  video_fps: 24",
                "model:",
                "  audio_vae_ckpt_dir: params",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "Wan2.2-TI2V-5B").mkdir()
    (path / "params").mkdir()
    save_file({"weight": torch.ones(1)}, path / "NAVA.safetensors")


def _make_forward_request() -> OmniDiffusionRequest:
    sampling_params = OmniDiffusionSamplingParams(
        height=704,
        width=1280,
        num_frames=37,
        fps=24,
        num_inference_steps=3,
        seed=11,
        extra_args={
            "video_guidance_scale": 4.0,
            "audio_guidance_scale": 2.5,
        },
    )
    return OmniDiffusionRequest(
        prompts=["cinematic beach sunrise"],
        sampling_params=sampling_params,
        request_id="nava-init-forward-test",
    )


def _make_image_forward_request() -> OmniDiffusionRequest:
    sampling_params = OmniDiffusionSamplingParams(
        height=704,
        width=1280,
        num_frames=37,
        fps=24,
        num_inference_steps=3,
        seed=11,
    )
    return OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "continue the reference frame",
                "multi_modal_data": {"image": Image.new("RGB", (8, 8), color=(255, 0, 0))},
            }
        ],
        sampling_params=sampling_params,
        request_id="nava-image-forward-test",
    )


def _make_speaker_forward_request() -> OmniDiffusionRequest:
    sampling_params = OmniDiffusionSamplingParams(
        height=704,
        width=1280,
        num_frames=37,
        fps=24,
        num_inference_steps=3,
        seed=11,
    )
    return OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "<S>Hello from a reference speaker<E>",
                "multi_modal_data": {"spk_wavs": ["speaker.wav"]},
            }
        ],
        sampling_params=sampling_params,
        request_id="nava-speaker-forward-test",
    )


def _load_serving_video_module(monkeypatch: pytest.MonkeyPatch):
    entrypoints_pkg = types.ModuleType("vllm_omni.entrypoints")
    entrypoints_pkg.__path__ = [str(_REPO_ROOT / "vllm_omni" / "entrypoints")]
    openai_pkg = types.ModuleType("vllm_omni.entrypoints.openai")
    openai_pkg.__path__ = [str(_REPO_ROOT / "vllm_omni" / "entrypoints" / "openai")]
    protocol_pkg = types.ModuleType("vllm_omni.entrypoints.openai.protocol")
    protocol_pkg.__path__ = [str(_REPO_ROOT / "vllm_omni" / "entrypoints" / "openai" / "protocol")]
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints", entrypoints_pkg)
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.openai", openai_pkg)
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.openai.protocol", protocol_pkg)

    async_omni = types.ModuleType("vllm_omni.entrypoints.async_omni")
    async_omni.AsyncOmni = object
    stage_params = types.ModuleType("vllm_omni.entrypoints.openai.stage_params")
    stage_params.build_stage_sampling_params_list = lambda *args, **kwargs: []
    stage_params.get_default_sampling_params_list = lambda *args, **kwargs: []
    openai_utils = types.ModuleType("vllm_omni.entrypoints.openai.utils")
    openai_utils.get_stage_type = lambda stage: getattr(stage, "stage_type", "diffusion")
    openai_utils.parse_lora_request = lambda lora_body: (None, None)
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.async_omni", async_omni)
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.openai.stage_params", stage_params)
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.openai.utils", openai_utils)

    spec = importlib.util.spec_from_file_location(
        "vllm_omni.entrypoints.openai.serving_video",
        _REPO_ROOT / "vllm_omni" / "entrypoints" / "openai" / "serving_video.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.openai.serving_video", module)
    spec.loader.exec_module(module)
    return module


def test_model_index_defaults_parse_without_diffusers_fields() -> None:
    cfg = NAVAConfig.from_dict({"_class_name": "NAVAPipeline", "nava_ckpt": "custom.safetensors"})

    assert cfg.model_type == "NAVA"
    assert cfg.ckpt_name == "custom.safetensors"
    assert cfg.audio_latent_ch == 128
    assert cfg.video_latent_ch == 48


def test_latent_shape_37_frames_1280x704() -> None:
    cfg = NAVAConfig()

    assert cfg.video_latent_hw() == (44, 80)
    assert cfg.audio_latent_length() == 152


def test_latent_shape_rejects_non_patch_aligned_resolution() -> None:
    cfg = NAVAConfig()

    with pytest.raises(ValueError, match="height and width must be divisible"):
        cfg.video_latent_hw(height=705, width=1280)


def test_config_accepts_public_request_aliases() -> None:
    cfg = NAVAConfig.from_dict({"height": 720, "width": 1280, "num_frames": 49, "num_inference_steps": 30})

    assert cfg.log_height == 720
    assert cfg.log_width == 1280
    assert cfg.frames == 49
    assert cfg.num_steps == 30


def test_yaml_video_tgt_frames_does_not_override_runtime_default() -> None:
    cfg = NAVAConfig.from_dict({"data": {"video_tgt_frames": 121, "video_fps": 24}})

    assert cfg.frames == 37
    assert cfg.fps == 24


def test_yaml_transformer_patch_size_does_not_override_latent_stride() -> None:
    cfg = NAVAConfig.from_dict({"patch_size": 2})

    assert cfg.video_latent_hw(height=704, width=1280) == (44, 80)


def test_speech_span_parser_single_speaker() -> None:
    assert parse_speech_spans("hello <S>line one<E> done") == ["line one"]


def test_speech_span_parser_multi_speaker() -> None:
    prompt = "<S>first<E> middle <S>second<E>"

    assert parse_speech_spans(prompt) == ["first", "second"]
    assert count_speech_spans(prompt) == 2


def test_inject_speaker_sentinel_matches_upstream_dataset() -> None:
    assert inject_speaker_sentinel("<S>Hello<E>") == "<S><extra_id_2>Hello<E>"


def test_package_exports_speaker_sentinel_helper() -> None:
    import vllm_omni.diffusion.models.nava as nava

    assert nava.inject_speaker_sentinel("<S>Hello<E>") == "<S><extra_id_2>Hello<E>"


def test_request_text_only_becomes_t2av_context() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request("cinematic beach sunrise", height=704, width=1280, num_frames=37, seed=7)

    ctx = pipeline._parse_request(request)

    assert ctx.prompt == "cinematic beach sunrise"
    assert ctx.image is None
    assert ctx.speaker_condition is None
    assert ctx.height == 704
    assert ctx.width == 1280
    assert ctx.frames == 37
    assert ctx.seed == 7
    assert not ctx.timbre_cfg


def test_request_accepts_omni_text_prompt_shape() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request(
        OmniTextPrompt(prompt="cinematic beach sunrise", modalities=["video"]),
        height=704,
        width=1280,
    )

    ctx = pipeline._parse_request(request)

    assert ctx.prompt == "cinematic beach sunrise"
    assert ctx.image is None
    assert ctx.speaker_condition is None
    assert not ctx.is_i2v


def test_request_rejects_multiple_prompts() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = OmniDiffusionRequest(
        prompts=["first prompt", "second prompt"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="nava-test",
    )

    with pytest.raises(ValueError, match="one prompt per request"):
        pipeline._parse_request(request)


def test_request_without_video_length_uses_nava_default_frames() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request("cinematic beach sunrise")

    ctx = pipeline._parse_request(request)

    assert ctx.frames == 37


def test_request_extra_args_accept_video_endpoint_aliases() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request("cinematic beach sunrise", extra_args={"num_frames": 49, "num_inference_steps": 30})

    ctx = pipeline._parse_request(request)

    assert ctx.frames == 49
    assert ctx.num_steps == 30


def test_request_extra_args_parse_string_booleans() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request(
        "cinematic beach sunrise",
        extra_args={
            "align_3d_cfg": "false",
            "negative_prompt_mode": "0",
            "offload_backbone": "true",
            "save_vid_latent": "yes",
            "tiled_vae": "on",
            "timbre_cfg": "false",
        },
    )

    ctx = pipeline._parse_request(request)

    assert not ctx.align_3d_cfg
    assert not ctx.negative_prompt_mode
    assert ctx.offload_backbone
    assert ctx.save_vid_latent
    assert ctx.tiled_vae
    assert not ctx.timbre_cfg


def test_request_rejects_timbre_cfg_without_speaker_reference() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request("cinematic beach sunrise", extra_args={"timbre_cfg": True})

    with pytest.raises(ValueError, match="timbre_cfg requires reference speaker WAVs"):
        pipeline._parse_request(request)


def test_request_rejects_custom_negative_prompt() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request("cinematic beach sunrise", extra_args={"video_negative_prompt": "blur"})

    with pytest.raises(ValueError, match="custom negative prompts are not supported"):
        pipeline._parse_request(request)


def test_request_rejects_openai_negative_prompt() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request({"prompt": "cinematic beach sunrise", "negative_prompt": "blur"})

    with pytest.raises(ValueError, match="custom negative prompts are not supported"):
        pipeline._parse_request(request)


def test_request_image_condition_enables_i2v() -> None:
    pipeline = _make_nava_pipeline_shell()
    image = object()
    request = _make_request({"prompt": "continue this frame", "multi_modal_data": {"image": image}})

    ctx = pipeline._parse_request(request)

    assert ctx.image is image
    assert ctx.is_i2v


def test_request_tensor_image_is_accepted_without_truthiness() -> None:
    pipeline = _make_nava_pipeline_shell()
    image = torch.zeros(3, 16, 32)
    request = _make_request({"prompt": "continue this frame", "multi_modal_data": {"image": image}})

    ctx = pipeline._parse_request(request)

    assert ctx.image is image


def test_request_speaker_wavs_match_speech_spans() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request(
        {
            "prompt": "<S>Hello<E><S>Hi<E>",
            "multi_modal_data": {"spk_wavs": ["a.wav", "b.wav"]},
        }
    )

    ctx = pipeline._parse_request(request)

    assert ctx.speaker_condition is not None
    assert ctx.speaker_condition.wavs == ["a.wav", "b.wav"]
    assert ctx.speaker_condition.spans == ["Hello", "Hi"]
    assert ctx.timbre_cfg


def test_request_audio_reference_path_matches_speech_span() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request(
        {
            "prompt": "<S>Hello from the online endpoint<E>",
            "multi_modal_data": {"audio": "/tmp/reference-speaker.wav"},
        }
    )

    ctx = pipeline._parse_request(request)

    assert ctx.speaker_condition is not None
    assert ctx.speaker_condition.wavs == ["/tmp/reference-speaker.wav"]
    assert ctx.speaker_condition.spans == ["Hello from the online endpoint"]
    assert ctx.timbre_cfg


def test_request_speaker_wavs_reject_mismatch() -> None:
    pipeline = _make_nava_pipeline_shell()
    request = _make_request(
        {
            "prompt": "<S>Hello<E><S>Hi<E>",
            "multi_modal_data": {"spk_wavs": ["only-one.wav"]},
        }
    )

    with pytest.raises(ValueError, match="speaker reference count"):
        pipeline._parse_request(request)


def test_build_sample_batch_without_image_or_speakers() -> None:
    pipeline = _make_nava_pipeline_shell()
    ctx = pipeline._parse_request(_make_request("plain prompt", height=704, width=1280, num_frames=37, fps=24))

    batch = pipeline._build_sample_batch(ctx)

    assert batch["captions"] == ["plain prompt"]
    assert batch["t_h_w_list"].tolist() == [[37, 44, 80]]
    assert batch["video_latents"].shape == (1, 37 * 44 * 80, 48)
    assert batch["audio_latents"][0].shape == (152, 128)
    assert batch["first_frames"] == [None]
    assert batch["spk_embs"] is None


def test_encode_speakers_uses_audio_vae_once_per_reference() -> None:
    pipeline = _make_nava_pipeline_shell()
    calls = []

    class FakeAudioVAE:
        def encode(self, payload):
            calls.append(payload)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: {"spk_embs": torch.ones(1, 192)}))

    pipeline.audio_vae = FakeAudioVAE()
    ctx = pipeline._parse_request(
        _make_request({"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}})
    )

    embeddings = pipeline._encode_speakers(ctx)

    assert calls == [{"data_path": "speaker.wav", "use_spk_emb": True}]
    assert len(embeddings) == 1
    assert embeddings[0].shape == (1, 192)


def test_encode_speakers_rejects_missing_redimnet_model() -> None:
    pipeline = _make_nava_pipeline_shell()

    class FakeAudioVAE:
        spk_model = None

        def encode(self, payload):
            raise AssertionError("encode should not run without a speaker model")

    pipeline.audio_vae = FakeAudioVAE()
    ctx = pipeline._parse_request(
        _make_request({"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}})
    )

    with pytest.raises(RuntimeError, match="ReDimNet speaker embedding"):
        pipeline._encode_speakers(ctx)


def test_encode_image_converts_pil_first_frame_for_video_vae() -> None:
    pipeline = _make_nava_pipeline_shell()
    captured = {}

    class FakeVideoVAE:
        def encode(self, image, **kwargs):
            captured["image"] = image
            captured["kwargs"] = kwargs
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: torch.zeros(1, 1, 2, 48)))

    pipeline.video_vae = FakeVideoVAE()
    image = Image.new("RGB", (4, 2), color=(255, 0, 0))
    ctx = pipeline._parse_request(
        _make_request({"prompt": "continue this frame", "multi_modal_data": {"image": image}}, height=16, width=32)
    )

    latents = pipeline._encode_image(ctx)

    assert captured["image"].shape == (1, 3, 16, 32)
    assert captured["image"].min().item() >= -1.0
    assert captured["image"].max().item() <= 1.0
    assert captured["kwargs"]["target_height"] == 16
    assert captured["kwargs"]["target_width"] == 32
    assert latents.shape == (1, 1, 2, 48)


def test_encode_image_rejects_multiple_first_frames() -> None:
    pipeline = _make_nava_pipeline_shell()
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    ctx = pipeline._parse_request(
        _make_request({"prompt": "continue this frame", "multi_modal_data": {"image": images}}, height=16, width=16)
    )

    with pytest.raises(ValueError, match="exactly one first-frame image"):
        pipeline._encode_image(ctx)


def test_post_process_audio_video_dict_keeps_metadata() -> None:
    postprocess = get_nava_post_process_func(SimpleNamespace())
    video = ["frame"]
    audio = torch.zeros(1, 16)

    result = postprocess({"video": video, "audio": audio, "audio_sample_rate": 16000, "fps": 24})

    assert result["video"] == video
    assert result["audio"] is audio
    assert result["audio_sample_rate"] == 16000
    assert result["fps"] == 24


def test_post_process_audio_video_tuple_adds_default_metadata() -> None:
    postprocess = get_nava_post_process_func(SimpleNamespace())
    video = ["frame"]
    audio = torch.zeros(1, 16)

    result = postprocess((video, audio))

    assert result["video"] == video
    assert result["audio"] is audio
    assert result["audio_sample_rate"] == 16000
    assert result["fps"] == 24


def test_post_process_tensor_video_keeps_batch_items() -> None:
    postprocess = get_nava_post_process_func(SimpleNamespace())
    video = torch.zeros(1, 2, 3, 4, 5)

    result = postprocess({"video": video, "audio": None}, output_type="pt")

    assert len(result["video"]) == 1
    assert result["video"][0].shape == (2, 3, 4, 5)


def test_component_discovery_lists_match_plan() -> None:
    assert NAVAPipeline._dit_modules == ["pipe.model"]
    assert NAVAPipeline._encoder_modules == ["pipe.text_model"]
    assert NAVAPipeline._vae_modules == ["pipe.video_vae", "pipe.audio_vae"]


def test_extra_body_params_cover_request_aliases() -> None:
    assert {
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "seed",
        "spk_wavs",
        "image_path",
        "video_guidance_scale",
        "audio_guidance_scale",
        "timbre_cfg",
    }.issubset(NAVAPipeline.EXTRA_BODY_PARAMS)
    assert "disable_text_encoder_compile" not in NAVAPipeline.EXTRA_BODY_PARAMS
    assert "nava_weight_dtype" not in NAVAPipeline.EXTRA_BODY_PARAMS
    assert "negative_prompt" not in NAVAPipeline.EXTRA_BODY_PARAMS
    assert "video_negative_prompt" not in NAVAPipeline.EXTRA_BODY_PARAMS
    assert "audio_negative_prompt" not in NAVAPipeline.EXTRA_BODY_PARAMS


def test_disable_torch_compile_context_restores_original() -> None:
    original_compile = torch.compile
    sentinel = object()

    with _temporarily_disable_torch_compile(True):
        assert torch.compile(sentinel) is sentinel

    assert torch.compile is original_compile


def test_weight_loader_rejects_missing_checkpoint(tmp_path: Path) -> None:
    pipeline = _make_nava_pipeline_shell(tmp_path)

    with pytest.raises(FileNotFoundError, match="NAVA checkpoint not found"):
        pipeline._load_upstream_checkpoint()


def test_load_nava_config_error_mentions_supported_config_layouts() -> None:
    pipeline = _make_nava_pipeline_shell("/missing/nava")

    with pytest.raises(ValueError) as exc_info:
        pipeline._load_nava_config(pipeline.od_config)

    message = str(exc_info.value)
    assert "nava.yaml or configs/nava.yaml" in message
    assert "Wan2.2-TI2V-5B/" in message


def test_weight_loader_rejects_checkpoint_with_no_matching_keys(tmp_path: Path) -> None:
    save_file({"other.weight": torch.ones(1)}, tmp_path / "NAVA.safetensors")
    pipeline = _make_nava_pipeline_shell(tmp_path)

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def load_state_dict(self, state_dict, strict=True):
            return [], list(state_dict)

    pipeline.pipe = SimpleNamespace(
        model=FakeModel(),
        to=lambda device: None,
    )
    pipeline.model = pipeline.pipe.model
    pipeline.device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="did not match the upstream model state dict"):
        pipeline._load_upstream_checkpoint()


def test_load_weights_noops_after_eager_checkpoint_load(tmp_path: Path) -> None:
    pipeline = _make_nava_pipeline_shell(tmp_path)

    loaded = pipeline.load_weights([("unused.weight", torch.zeros(1))])

    assert loaded == set()


def test_load_nava_config_reads_upstream_config_layout(tmp_path: Path) -> None:
    _write_yaml_model_config(tmp_path / "configs" / "nava.yaml", height=720)
    pipeline = _make_nava_pipeline_shell(tmp_path)

    cfg = pipeline._load_nava_config(pipeline.od_config)

    assert cfg.config_name == "configs/nava.yaml"
    assert cfg.log_height == 720


def test_load_nava_config_keeps_assembled_audio_vae_dir_over_yaml_training_path(tmp_path: Path) -> None:
    _write_model_index(tmp_path / "model_index.json", {"_class_name": "NAVAPipeline", "audio_vae_dir": "params"})
    yaml_path = tmp_path / "nava.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "model_type: NAVA",
                "model:",
                "  audio_vae_ckpt_dir: ./huggingface_upload/params",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pipeline = _make_nava_pipeline_shell(tmp_path)

    cfg = pipeline._load_nava_config(pipeline.od_config)

    assert cfg.audio_vae_ckpt_dir == "params"


def test_load_nava_config_accepts_legacy_root_config_layout(tmp_path: Path) -> None:
    _write_yaml_model_config(tmp_path / "nava.yaml", height=736)
    pipeline = _make_nava_pipeline_shell(tmp_path)

    cfg = pipeline._load_nava_config(pipeline.od_config)

    assert cfg.config_name == "nava.yaml"
    assert cfg.log_height == 736


def test_load_nava_config_keeps_legacy_config_when_model_index_points_to_default(tmp_path: Path) -> None:
    _write_yaml_model_config(tmp_path / "nava.yaml", height=736)
    _write_model_index(tmp_path / "model_index.json", {"_class_name": "NAVAPipeline", "config": "configs/nava.yaml"})
    pipeline = _make_nava_pipeline_shell(tmp_path)

    cfg = pipeline._load_nava_config(pipeline.od_config)

    assert cfg.config_name == "nava.yaml"
    assert cfg.log_height == 736


def test_load_nava_config_uses_explicit_config_path(tmp_path: Path) -> None:
    _write_yaml_model_config(tmp_path / "configs" / "nava.yaml", height=720)
    _write_yaml_model_config(tmp_path / "configs" / "custom.yaml", height=768)
    pipeline = _make_nava_pipeline_shell(tmp_path, model_config={"config": "configs/custom.yaml"})

    cfg = pipeline._load_nava_config(pipeline.od_config)

    assert cfg.config_name == "configs/custom.yaml"
    assert cfg.log_height == 768


def test_load_nava_config_prefers_explicit_model_config_over_repo_files(tmp_path: Path) -> None:
    _write_yaml_model_config(tmp_path / "configs" / "nava.yaml", height=720)
    _write_model_index(
        tmp_path / "model_index.json",
        {
            "_class_name": "NAVAPipeline",
            "nava_ckpt": "index.safetensors",
            "height": 704,
        },
    )
    pipeline = _make_nava_pipeline_shell(
        tmp_path,
        model_config={
            "nava_ckpt": "explicit.safetensors",
            "height": 736,
        },
    )

    cfg = pipeline._load_nava_config(pipeline.od_config)

    assert cfg.ckpt_name == "explicit.safetensors"
    assert cfg.log_height == 736


def test_checkpoint_resolution_prefers_explicit_model_config_path(tmp_path: Path) -> None:
    explicit_ckpt = tmp_path / "custom-nava.safetensors"
    default_ckpt = tmp_path / "NAVA.safetensors"
    explicit_ckpt.write_bytes(b"")
    default_ckpt.write_bytes(b"")
    pipeline = _make_nava_pipeline_shell(tmp_path, model_config={"nava_ckpt": "custom-nava.safetensors"})

    assert pipeline._resolve_checkpoint_path() == str(explicit_ckpt)


def test_checkpoint_resolution_prefers_fp8_when_requested(tmp_path: Path) -> None:
    bf16_ckpt = tmp_path / "NAVA.safetensors"
    fp8_ckpt = tmp_path / "NAVA_fp8.safetensors"
    bf16_ckpt.write_bytes(b"")
    fp8_ckpt.write_bytes(b"")
    pipeline = _make_nava_pipeline_shell(tmp_path, custom_pipeline_args={"nava_weight_dtype": "fp8_e4m3fn"})

    assert pipeline._resolve_checkpoint_path() == str(fp8_ckpt)


def test_checkpoint_resolution_auto_falls_back_to_fp8(tmp_path: Path) -> None:
    fp8_ckpt = tmp_path / "NAVA_fp8.safetensors"
    fp8_ckpt.write_bytes(b"")
    pipeline = _make_nava_pipeline_shell(tmp_path)

    assert pipeline._resolve_checkpoint_path() == str(fp8_ckpt)


def test_checkpoint_resolution_bf16_does_not_fallback_to_fp8(tmp_path: Path) -> None:
    fp8_ckpt = tmp_path / "NAVA_fp8.safetensors"
    fp8_ckpt.write_bytes(b"")
    pipeline = _make_nava_pipeline_shell(tmp_path, custom_pipeline_args={"nava_weight_dtype": "bf16"})

    assert pipeline._resolve_checkpoint_path() is None


def test_nava_registered_as_diffusion_pipeline() -> None:
    assert _DIFFUSION_MODELS["NAVAPipeline"] == ("nava", "pipeline_nava", "NAVAPipeline")


def test_nava_postprocess_registered() -> None:
    assert _DIFFUSION_POST_PROCESS_FUNCS["NAVAPipeline"] == "get_nava_post_process_func"


def test_nava_cache_acceleration_disabled_for_bridge_pipeline() -> None:
    assert "NAVAPipeline" in _NO_CACHE_ACCELERATION


def test_nava_rejects_vllm_omni_cpu_offload_for_bridge_pipeline(tmp_path: Path) -> None:
    _write_fake_upstream_model_dir(tmp_path)
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        enable_cpu_offload=True,
    )

    with pytest.raises(ValueError, match="does not support vLLM-Omni CPU/layerwise offload"):
        NAVAPipeline(od_config=od_config)


def test_nava_rejects_vllm_omni_layerwise_offload_for_bridge_pipeline(tmp_path: Path) -> None:
    _write_fake_upstream_model_dir(tmp_path)
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        enable_layerwise_offload=True,
    )

    with pytest.raises(ValueError, match="does not support vLLM-Omni CPU/layerwise offload"):
        NAVAPipeline(od_config=od_config)


@pytest.mark.parametrize(
    "parallel_config",
    [
        pytest.param(DiffusionParallelConfig(tensor_parallel_size=2), id="tp"),
        pytest.param(DiffusionParallelConfig(data_parallel_size=2), id="dp"),
        pytest.param(DiffusionParallelConfig(enable_expert_parallel=True), id="expert"),
    ],
)
def test_nava_rejects_native_parallelism_for_bridge_pipeline(
    tmp_path: Path,
    parallel_config: DiffusionParallelConfig,
) -> None:
    _write_fake_upstream_model_dir(tmp_path)
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        parallel_config=parallel_config,
    )

    with pytest.raises(ValueError, match="does not support native vLLM-Omni parallelism"):
        NAVAPipeline(od_config=od_config)


def test_nava_multimodal_capabilities_are_advertised() -> None:
    config = OmniDiffusionConfig(model="/tmp/nava", model_class_name="NAVAPipeline")

    assert supports_multimodal_input(config) == (True, True)
    assert supports_audio_output("NAVAPipeline")


def test_nava_config_metadata_advertises_single_image_input() -> None:
    config = OmniDiffusionConfig(model="/tmp/nava", model_class_name="NAVAPipeline")

    config.update_multimodal_support()

    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 1


def test_nava_video_audio_postprocess_becomes_omni_multimodal_output() -> None:
    video = torch.zeros(1, 2, 3, 4, 5)
    audio = torch.zeros(1, 16000)
    raw_output = {
        "video": video,
        "audio": audio,
        "audio_sample_rate": 16000,
        "fps": 24,
    }
    postprocess = get_nava_post_process_func(SimpleNamespace())
    postprocess_output = normalize_diffusion_postprocess_output(postprocess(raw_output, output_type="pt"), {})
    request = OmniDiffusionRequest(
        prompts=["a person speaks"],
        request_id="nava-formatter-test",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, resolution=512),
    )

    [result] = format_diffusion_outputs(
        request=request,
        od_config=SimpleNamespace(model_class_name="NAVAPipeline"),
        diffusion_output=DiffusionOutput(output=raw_output),
        output_data=raw_output,
        postprocess_output=postprocess_output,
        timings=DiffusionStepTimings(
            preprocess_time_s=0.0,
            exec_time_s=0.0,
            postprocess_time_s=0.0,
            total_time_ms=0.0,
        ),
    )

    assert result.images[0].shape == (2, 3, 4, 5)
    assert result.multimodal_output["audio"] is audio
    assert result.multimodal_output["audio_sample_rate"] == 16000
    assert result.multimodal_output["fps"] == 24


def test_nava_formatter_result_is_visible_to_video_serving_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    serving_video = _load_serving_video_module(monkeypatch)
    handler = serving_video.OmniOpenAIServingVideo(engine_client=SimpleNamespace())
    audio = torch.zeros(16000)
    video = torch.zeros(2, 3, 4, 5)
    result = SimpleNamespace(
        request_output=SimpleNamespace(
            images=[video],
            multimodal_output={
                "audio": audio,
                "audio_sample_rate": 16000,
                "fps": 24,
            },
        )
    )

    videos = handler._extract_video_outputs(result)
    audios = handler._extract_audio_outputs(result, expected_count=1)

    assert videos == [video]
    assert audios == [audio]
    assert handler._resolve_audio_sample_rate(result) == 16000
    assert handler._resolve_fps(result) == 24


def test_offline_example_text_only_prompt_stays_plain_string() -> None:
    module = _load_end2end_script()

    prompt = module._build_prompt(_base_args())

    assert prompt == "a person speaks"


def test_offline_example_help_does_not_import_runtime_stack() -> None:
    result = subprocess.run(
        [sys.executable, str(_END2END_SCRIPT_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--model" in result.stdout
    assert "--prompt" in result.stdout


def test_offline_example_image_and_speakers_build_multimodal_prompt(tmp_path: Path) -> None:
    module = _load_end2end_script()
    image_path = tmp_path / "first.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)

    prompt = module._build_prompt(
        _base_args(
            image=str(image_path),
            prompt="<S>Hello<E>",
            spk_wavs=["speaker.wav"],
        )
    )

    assert prompt["prompt"] == "<S>Hello<E>"
    assert isinstance(prompt["multi_modal_data"]["image"], Image.Image)
    assert prompt["multi_modal_data"]["spk_wavs"] == ["speaker.wav"]


def test_offline_example_sampling_params_disable_timbre_cfg_without_speaker() -> None:
    module = _load_end2end_script()

    sampling_params = module._build_sampling_params(_base_args(spk_wavs=None))

    assert sampling_params.extra_args["timbre_cfg"] is False


def test_offline_example_sampling_params_enable_timbre_cfg_with_speaker() -> None:
    module = _load_end2end_script()

    sampling_params = module._build_sampling_params(_base_args(spk_wavs=["speaker.wav"]))

    assert sampling_params.extra_args["timbre_cfg"] is True


def test_offline_example_sampling_params_can_disable_timbre_cfg_with_speaker() -> None:
    module = _load_end2end_script()

    sampling_params = module._build_sampling_params(_base_args(spk_wavs=["speaker.wav"], disable_timbre_cfg=True))

    assert sampling_params.extra_args["timbre_cfg"] is False


def test_online_run_server_prefers_vllm_omni_cli(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    called_path = tmp_path / "called.txt"
    fake_cli = bin_dir / "vllm-omni"
    fake_cli.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"printf '%s\\n' \"$@\" > {called_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "MODEL": "/models/nava",
        "HOST": "127.0.0.1",
        "PORT": "18000",
        "SERVED_MODEL_NAME": "nava-test",
    }

    subprocess.run(["bash", str(_ONLINE_DIR / "run_server.sh")], check=True, env=env)

    args = called_path.read_text(encoding="utf-8").splitlines()
    assert args[:3] == ["serve", "/models/nava", "--omni"]
    assert "--model-class-name" in args
    assert args[args.index("--model-class-name") + 1] == "NAVAPipeline"
    assert args[args.index("--served-model-name") + 1] == "nava-test"
    assert args[args.index("--port") + 1] == "18000"


def test_online_curl_script_uses_current_video_job_api() -> None:
    script = (_ONLINE_DIR / "run_curl_nava.sh").read_text(encoding="utf-8")

    assert "/v1/videos/generations" not in script
    assert 'curl -sS -X POST "${BASE_URL}/v1/videos"' in script
    assert '-F "prompt=' in script
    assert "-F 'extra_params=" in script
    assert 'curl -sS "${BASE_URL}/v1/videos/${video_id}"' in script
    assert 'curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content"' in script


def test_online_readme_uses_form_video_endpoint_examples() -> None:
    readme = (_ONLINE_DIR / "README.md").read_text(encoding="utf-8")

    assert "/v1/videos/generations" not in readme
    assert '/v1/videos" \\' in readme
    assert '-F \'image_reference={"image_url":"data:image/png;base64,<base64-png>"}\'' in readme
    assert '-F \'audio_reference={"audio_url":"data:audio/wav;base64,<base64-wav>"}\'' in readme
    assert "`extra_params` form field as a JSON string" in readme


def test_nava_pipeline_init_and_text_forward_with_fake_upstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_upstream(monkeypatch)
    _write_fake_upstream_model_dir(tmp_path)
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        enforce_eager=True,
    )

    pipeline = NAVAPipeline(od_config=od_config)
    output = pipeline.forward(_make_forward_request()).output

    assert _FakeAudioVideoPipeline.created_kwargs is not None
    assert _FakeAudioVideoPipeline.created_kwargs["cfg"]["model"]["ckpt_dir"] == str(tmp_path)
    assert _FakeAudioVideoPipeline.created_kwargs["cfg"]["model"]["audio_vae_ckpt_dir"] == str(tmp_path / "params")
    assert pipeline.transformer is pipeline.pipe.model
    assert pipeline.text_encoder is pipeline.pipe.text_model.model
    assert pipeline.pipe.model.weight.item() == 1.0
    assert pipeline.pipe.model.backbone.rope_set
    assert pipeline.pipe.to_device == torch.device("cpu")

    assert pipeline.pipe.sample_batch is not None
    assert pipeline.pipe.sample_batch["captions"] == ["cinematic beach sunrise"]
    assert pipeline.pipe.sample_batch["video_latents"].shape == (1, 37 * 44 * 80, 48)
    assert pipeline.pipe.sample_batch["spk_embs"] is None
    assert pipeline.pipe.sample_kwargs is not None
    assert pipeline.pipe.sample_kwargs["num_steps"] == 3
    assert pipeline.pipe.sample_kwargs["video_guidance_scale"] == 4.0
    assert pipeline.pipe.sample_kwargs["audio_guidance_scale"] == 2.5
    assert pipeline.pipe.sample_kwargs["is_i2v"] is False
    assert pipeline.pipe.sample_kwargs["timbre_cfg"] is False

    assert output["video"].shape == (1, 2, 3, 4, 5)
    assert torch.equal(output["audio"], torch.ones(160))
    assert output["audio_sample_rate"] == 16000
    assert output["fps"] == 24


def test_nava_pipeline_image_forward_marks_i2v_for_fake_upstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_upstream(monkeypatch)
    _write_fake_upstream_model_dir(tmp_path)
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        enforce_eager=True,
    )

    pipeline = NAVAPipeline(od_config=od_config)
    pipeline.forward(_make_image_forward_request())

    assert pipeline.pipe.sample_batch is not None
    assert isinstance(pipeline.pipe.sample_batch["first_frames"][0], torch.Tensor)
    assert pipeline.pipe.sample_batch["spk_embs"] is None
    assert pipeline.pipe.sample_kwargs is not None
    assert pipeline.pipe.sample_kwargs["is_i2v"] is True


def test_nava_pipeline_speaker_forward_passes_embeddings_to_fake_upstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_upstream(monkeypatch)
    _write_fake_upstream_model_dir(tmp_path)
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        enforce_eager=True,
    )

    pipeline = NAVAPipeline(od_config=od_config)
    pipeline.forward(_make_speaker_forward_request())

    assert pipeline.pipe.sample_batch is not None
    assert pipeline.pipe.sample_batch["captions"] == ["<S><extra_id_2>Hello from a reference speaker<E>"]
    assert pipeline.pipe.sample_batch["spk_embs"][0][0].shape == (1, 192)
    assert pipeline.pipe.sample_kwargs is not None
    assert pipeline.pipe.sample_kwargs["timbre_cfg"] is True


def test_nava_pipeline_create_parses_string_bool_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_upstream(monkeypatch)
    _write_fake_upstream_model_dir(tmp_path)
    compile_disabled_flags: list[bool] = []

    class _CompileGuard:
        def __init__(self, disabled: bool) -> None:
            compile_disabled_flags.append(disabled)

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        nava_pipeline_module,
        "_temporarily_disable_torch_compile",
        lambda disabled: _CompileGuard(disabled),
    )
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        additional_config={"use_bf16": "false"},
        custom_pipeline_args={
            "disable_text_encoder_compile": "false",
            "nava_weight_dtype": "bf16",
        },
    )

    NAVAPipeline(od_config=od_config)

    assert compile_disabled_flags == [False]
    assert _FakeAudioVideoPipeline.created_kwargs is not None
    assert _FakeAudioVideoPipeline.created_kwargs["use_bf16"] is False
    assert "disable_text_encoder_compile" not in _FakeAudioVideoPipeline.created_kwargs["cfg"]
    assert "nava_weight_dtype" not in _FakeAudioVideoPipeline.created_kwargs["cfg"]


def test_download_script_writes_model_index_and_excludes_fp8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_download_script()
    calls: list[list[str]] = []
    _write_download_model_dir(tmp_path)

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, check: calls.append(cmd),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["download_nava.py", "--local-dir", str(tmp_path), "--bf16-only"],
    )

    module.main()

    assert calls == [
        [
            "huggingface-cli",
            "download",
            "baidu/NAVA",
            "--local-dir",
            str(tmp_path.resolve()),
            "--exclude",
            "NAVA_fp8.safetensors",
        ]
    ]
    model_index = json.loads((tmp_path / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["_class_name"] == "NAVAPipeline"
    assert model_index["nava_ckpt"] == "NAVA.safetensors"
    assert model_index["fp8_ckpt"] == "NAVA_fp8.safetensors"
    assert model_index["config"] == "configs/nava.yaml"


def test_download_script_rejects_bf16_and_fp8_only_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["download_nava.py", "--local-dir", str(tmp_path), "--bf16-only", "--fp8-only"],
    )

    with pytest.raises(SystemExit, match="mutually exclusive"):
        module.main()


def test_download_script_can_prepare_redimnet_without_installing_upstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    redimnet_calls: list[str | None] = []
    _write_download_model_dir(tmp_path)

    monkeypatch.setattr(module.subprocess, "run", lambda cmd, check: None)
    monkeypatch.setattr(module, "prepare_redimnet", lambda torch_home: redimnet_calls.append(torch_home))
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "download_nava.py",
            "--local-dir",
            str(tmp_path),
            "--prepare-redimnet",
            "--torch-home",
            str(tmp_path / "torch-cache"),
        ],
    )

    module.main()

    assert redimnet_calls == [str(tmp_path / "torch-cache")]


def test_download_script_verify_only_can_prepare_redimnet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    redimnet_calls: list[str | None] = []
    _write_download_model_dir(tmp_path)

    monkeypatch.setattr(module, "prepare_redimnet", lambda torch_home: redimnet_calls.append(torch_home))
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "download_nava.py",
            "--local-dir",
            str(tmp_path),
            "--verify-only",
            "--prepare-redimnet",
            "--torch-home",
            str(tmp_path / "torch-cache"),
        ],
    )

    module.main()

    assert redimnet_calls == [str(tmp_path / "torch-cache")]
    assert not (tmp_path / "model_index.json").exists()


def test_download_script_verify_only_rejects_incomplete_model_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["download_nava.py", "--local-dir", str(tmp_path), "--verify-only"],
    )

    with pytest.raises(SystemExit, match="configs/nava.yaml or nava.yaml"):
        module.main()


def test_download_script_verify_only_accepts_minimal_model_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    _write_download_model_dir(tmp_path)
    model_index_path = tmp_path / "model_index.json"
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["download_nava.py", "--local-dir", str(tmp_path), "--verify-only"],
    )

    module.main()

    assert not model_index_path.exists()


def test_download_script_model_index_uses_legacy_config_when_only_root_yaml_exists(tmp_path: Path) -> None:
    module = _load_download_script()
    (tmp_path / "nava.yaml").write_text("model_type: NAVA\n", encoding="utf-8")

    model_index = module.build_model_index(tmp_path)

    assert model_index["config"] == "nava.yaml"


def test_download_script_verify_only_accepts_legacy_root_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    (tmp_path / "nava.yaml").write_text("model_type: NAVA\n", encoding="utf-8")
    (tmp_path / "NAVA.safetensors").write_bytes(b"")
    (tmp_path / "Wan2.2-TI2V-5B").mkdir()
    (tmp_path / "params").mkdir()
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["download_nava.py", "--local-dir", str(tmp_path), "--verify-only"],
    )

    module.main()

    assert not (tmp_path / "model_index.json").exists()
