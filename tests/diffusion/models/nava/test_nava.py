# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.io_support import supports_audio_output, supports_multimodal_input
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.nava import audio_vae as nava_audio_vae
from vllm_omni.diffusion.models.nava.audio_vae import NAVAAudioVAE
from vllm_omni.diffusion.models.nava.config import NAVAConfig, inject_speaker_sentinel, parse_speech_spans
from vllm_omni.diffusion.models.nava.nava_transformer import (
    NAVATransformer,
    WanSelfAttention,
    _rope_apply_1d,
    _rope_apply_3d,
    _rope_apply_3d_to_1d,
)
from vllm_omni.diffusion.models.nava.pipeline_nava import NAVAPipeline, _NAVATextEncoder, get_nava_post_process_func
from vllm_omni.diffusion.models.nava.scheduler import NAVAFlowMatchScheduler
from vllm_omni.diffusion.models.nava.speaker import NAVASpeakerEncoder
from vllm_omni.diffusion.models.nava.video_vae import NAVAVideoVAE
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _DIFFUSION_POST_PROCESS_FUNCS, _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOWNLOAD_SCRIPT = _REPO_ROOT / "examples" / "offline_inference" / "nava" / "download_nava.py"


class FakeTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.calls.append(texts)
        return torch.ones(len(texts), 2, 4, device=device, dtype=dtype)


class FakeTextEncoderWithSpeakerPositions(FakeTextEncoder):
    def encode(
        self,
        texts: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        return_speaker_positions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[int]]]:
        embeds = super().encode(texts, device=device, dtype=dtype)
        if return_speaker_positions:
            return embeds, [[1] for _ in texts]
        return embeds


class FakeVideoVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoded_images: list[torch.Tensor] = []
        self.decode_calls: list[dict[str, Any]] = []

    def encode_first_frame(self, image: torch.Tensor) -> torch.Tensor:
        self.encoded_images.append(image)
        return torch.ones(1, 1, 1, 1, device=image.device, dtype=image.dtype)

    def decode(self, video_latents: torch.Tensor, *, height: int, width: int, frames: int) -> torch.Tensor:
        self.decode_calls.append({"height": height, "width": width, "frames": frames, "shape": video_latents.shape})
        return torch.zeros(1, 3, frames, height, width, device=video_latents.device, dtype=video_latents.dtype)


class FakeAudioVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decode_shapes: list[torch.Size] = []

    def decode(self, audio_latents: torch.Tensor) -> torch.Tensor:
        self.decode_shapes.append(audio_latents.shape)
        return torch.zeros(audio_latents.shape[0], audio_latents.shape[1], device=audio_latents.device)


class FakeSpeakerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[Any]] = []

    def encode(self, wavs: list[Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.calls.append(wavs)
        return torch.ones(len(wavs), 3, device=device, dtype=dtype)


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        return {
            "video": torch.ones_like(kwargs["video_latents"]),
            "audio": torch.ones_like(kwargs["audio_latents"]) * 2,
        }

    def load_weights(self, weights):
        return {name.removeprefix("transformer.") for name, _ in weights}


def _make_pipeline(
    tmp_path: Path,
    custom_pipeline_args_extra: dict[str, Any] | None = None,
    **custom_components: Any,
) -> NAVAPipeline:
    model_config = {
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "num_inference_steps": 2,
        "audio_latent_ch": 4,
        "video_latent_ch": 3,
        "text_embed_dim": 4,
        "speaker_embed_dim": 3,
        "audio_tokens_per_sec": 4.0,
    }
    text_encoder = custom_components.get("text_encoder", FakeTextEncoder())
    video_vae = custom_components.get("video_vae", FakeVideoVAE())
    audio_vae = custom_components.get("audio_vae", FakeAudioVAE())
    speaker_encoder = custom_components.get("speaker_encoder", FakeSpeakerEncoder())
    transformer = custom_components.get("transformer", FakeTransformer())
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        model_config=model_config,
        custom_pipeline_args=custom_pipeline_args_extra or {},
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava._NAVATextEncoder",
            lambda *args, **kwargs: text_encoder,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.NAVAVideoVAE",
            lambda *args, **kwargs: video_vae,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.NAVAAudioVAE",
            lambda *args, **kwargs: audio_vae,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.NAVASpeakerEncoder",
            lambda *args, **kwargs: speaker_encoder,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.NAVATransformer",
            lambda *args, **kwargs: transformer,
        )
        return NAVAPipeline(od_config=od_config)


def _make_request(prompt: Any, **sampling_kwargs: Any) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=OmniDiffusionSamplingParams(**sampling_kwargs),
        request_id="nava-test",
    )


def _load_download_script():
    spec = importlib.util.spec_from_file_location("nava_download_script", _DOWNLOAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_minimal_model_dir(path: Path) -> None:
    (path / "configs").mkdir(parents=True)
    (path / "configs" / "nava.yaml").write_text("model_type: NAVA\n", encoding="utf-8")
    (path / "NAVA.safetensors").write_bytes(b"")
    wan_dir = path / "Wan2.2-TI2V-5B"
    (wan_dir / "google" / "umt5-xxl").mkdir(parents=True)
    (wan_dir / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"")
    (wan_dir / "Wan2.2_VAE.pth").write_bytes(b"")
    ltx_dir = path / "params" / "LTX2"
    ltx_dir.mkdir(parents=True)
    (ltx_dir / "ltx-2.3-22b-dev_audio_vae.safetensors").write_bytes(b"")
    speaker_dir = path / "speaker"
    speaker_dir.mkdir()
    (speaker_dir / "hubconf.py").write_text("def ReDimNet(**kwargs):\n    return None\n", encoding="utf-8")


def test_custom_pipeline_args_without_component_keys_do_not_inject_missing_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_minimal_model_dir(tmp_path)

    text_encoder = FakeTextEncoder()
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()
    speaker_encoder = FakeSpeakerEncoder()
    transformer = FakeTransformer()

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.nava.pipeline_nava._NAVATextEncoder",
        lambda *args, **kwargs: text_encoder,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.nava.pipeline_nava.NAVAVideoVAE",
        lambda *args, **kwargs: video_vae,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.nava.pipeline_nava.NAVAAudioVAE",
        lambda *args, **kwargs: audio_vae,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.nava.pipeline_nava.NAVASpeakerEncoder",
        lambda *args, **kwargs: speaker_encoder,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.nava.pipeline_nava.NAVATransformer",
        lambda *args, **kwargs: transformer,
    )

    pipeline = NAVAPipeline(
        od_config=OmniDiffusionConfig(
            model=str(tmp_path),
            model_class_name="NAVAPipeline",
            custom_pipeline_args={"nava_weight_dtype": "bf16"},
        )
    )

    assert pipeline.text_encoder is text_encoder
    assert pipeline.video_vae is video_vae
    assert pipeline.audio_vae is audio_vae
    assert pipeline.speaker_encoder is speaker_encoder
    assert pipeline.transformer is transformer


def test_default_text_encoder_compile_matches_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_minimal_model_dir(tmp_path)
    captured: list[bool] = []

    def make_text_encoder(*args: Any, **kwargs: Any) -> FakeTextEncoder:
        captured.append(bool(kwargs["compile_model"]))
        return FakeTextEncoder()

    monkeypatch.setattr("vllm_omni.diffusion.models.nava.pipeline_nava._NAVATextEncoder", make_text_encoder)
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.pipeline_nava.NAVAVideoVAE", lambda *args, **kwargs: FakeVideoVAE())
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.pipeline_nava.NAVAAudioVAE", lambda *args, **kwargs: FakeAudioVAE())
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.nava.pipeline_nava.NAVASpeakerEncoder",
        lambda *args, **kwargs: FakeSpeakerEncoder(),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.nava.pipeline_nava.NAVATransformer",
        lambda *args, **kwargs: FakeTransformer(),
    )

    NAVAPipeline(od_config=OmniDiffusionConfig(model=str(tmp_path), model_class_name="NAVAPipeline"))
    NAVAPipeline(
        od_config=OmniDiffusionConfig(
            model=str(tmp_path),
            model_class_name="NAVAPipeline",
            custom_pipeline_args={"nava_text_encoder_compile": False},
        )
    )

    assert captured == [True, False]


def test_config_accepts_model_index_aliases() -> None:
    cfg = NAVAConfig.from_dict({"nava_ckpt": "custom.safetensors", "height": 16, "width": 32})

    assert cfg.ckpt_name == "custom.safetensors"
    assert cfg.log_height == 16
    assert cfg.log_width == 32


def test_config_model_index_audio_vae_dir_wins_over_stale_yaml_model_block() -> None:
    cfg = NAVAConfig.from_dict(
        {
            "audio_vae_dir": "params",
            "model": {"audio_vae_ckpt_dir": "./huggingface_upload/params"},
        }
    )

    assert cfg.audio_vae_ckpt_dir == "params"


def test_config_joint_config_overrides_stale_yaml_architecture() -> None:
    cfg = NAVAConfig.from_dict(
        {
            "patch_size": 2,
            "hidden_size": 128,
            "model": {
                "joint_config_data": {
                    "patch_size": [1, 2, 2],
                    "dim": 3072,
                    "vid_in_dim": 48,
                    "audio_in_dim": 128,
                }
            },
        }
    )

    assert cfg.patch_size == (1, 2, 2)
    assert cfg.hidden_size == 3072
    assert cfg.video_latent_ch == 48
    assert cfg.audio_latent_ch == 128


def test_config_explicit_keys_override_joint_config() -> None:
    cfg = NAVAConfig.from_dict(
        {
            "_explicit_keys": {"patch_size", "hidden_size"},
            "patch_size": [1, 1, 1],
            "hidden_size": 256,
            "model": {
                "joint_config_data": {
                    "patch_size": [1, 2, 2],
                    "dim": 3072,
                }
            },
        }
    )

    assert cfg.patch_size == (1, 1, 1)
    assert cfg.hidden_size == 256


def test_config_rejects_unaligned_video_resolution() -> None:
    with pytest.raises(ValueError, match="divisible"):
        NAVAConfig().video_latent_hw(height=17, width=16)


def test_config_normalizes_output_frames_to_vae_stride() -> None:
    cfg = NAVAConfig(frames=10)

    assert cfg.normalize_output_frames() == 9
    assert cfg.normalize_output_frames(1) == 1
    assert cfg.video_latent_frames(9) == 3
    assert cfg.video_output_frames(3) == 9


def test_transformer_self_attention_uses_shared_attention_layer() -> None:
    attention = WanSelfAttention(dim=8, num_heads=2)

    assert attention.attn.__class__.__name__ == "Attention"


def test_speech_span_parser_extracts_ordered_spans() -> None:
    assert parse_speech_spans("<S>first<E> middle <S>second<E>") == ["first", "second"]


def test_speaker_sentinel_injection() -> None:
    assert inject_speaker_sentinel("<S>Hello<E>") == "<S><extra_id_2>Hello<E>"


def test_parse_text_only_request(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", seed=7))

    assert ctx.prompt == "plain prompt"
    assert ctx.image is None
    assert ctx.speaker_condition is None
    assert ctx.seed == 7
    assert ctx.frames == 5


def test_pipeline_weight_source_and_loaded_names_are_transformer_scoped(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)

    assert [source.prefix for source in pipeline.weights_sources] == ["transformer."]
    loaded = pipeline.load_weights([("transformer.backbone.weight", torch.ones(1))])

    assert loaded == {"transformer.backbone.weight"}


def test_pipeline_loads_explicit_config_file_before_building_nava_config(tmp_path: Path) -> None:
    runtime_config = tmp_path / "runtime.yaml"
    runtime_config.write_text(
        """
model_type: NAVA
data:
  video_fps: 8
  audio_tokens_per_sec: 25
model:
  audio_vae_ckpt_dir: params
""",
        encoding="utf-8",
    )
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        model_config={"config": str(runtime_config)},
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava._NAVATextEncoder",
            lambda *args, **kwargs: FakeTextEncoder(),
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.NAVAVideoVAE",
            lambda *args, **kwargs: FakeVideoVAE(),
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.NAVAAudioVAE",
            lambda *args, **kwargs: FakeAudioVAE(),
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.NAVASpeakerEncoder",
            lambda *args, **kwargs: FakeSpeakerEncoder(),
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.NAVATransformer",
            lambda *args, **kwargs: FakeTransformer(),
        )
        pipeline = NAVAPipeline(od_config=od_config)

    assert pipeline.nava_config.fps == 8
    assert pipeline.nava_config.audio_latent_length(frames=5) == 54


def test_parse_request_preserves_upstream_latent_frames(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", num_frames=10))

    assert ctx.frames == 10


def test_parse_image_condition_request(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    image = Image.new("RGB", (4, 4))
    ctx = pipeline._parse_request(_make_request({"prompt": "continue", "multi_modal_data": {"image": image}}))

    assert ctx.image is image
    assert ctx.is_i2v


def test_parse_speaker_wavs_matches_speech_spans(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(
        _make_request({"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}})
    )

    assert ctx.speaker_condition is not None
    assert ctx.speaker_condition.wavs == ["speaker.wav"]
    assert ctx.speaker_condition.spans == ["Hello"]
    assert ctx.timbre_cfg


def test_parse_speaker_wavs_rejects_span_mismatch(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    request = _make_request({"prompt": "<S>Hello<E><S>Hi<E>", "multi_modal_data": {"spk_wavs": ["one.wav"]}})

    with pytest.raises(ValueError, match="speaker reference count"):
        pipeline._parse_request(request)


def test_text_embedding_called_once_for_positive_prompt(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoder()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)

    embeds = pipeline._encode_text("<S>Hello<E>")

    assert embeds.shape == (1, 2, 4)
    assert text_encoder.calls == [["<S><extra_id_2>Hello<E>"]]


def test_text_embedding_returns_speaker_marker_positions(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoderWithSpeakerPositions()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)

    embeds, speaker_positions = pipeline._encode_text_with_speaker_positions("<S>Hello<E>")

    assert embeds.shape == (1, 2, 4)
    assert speaker_positions == [[1]]


def test_text_encoder_is_not_wrapped_in_pipeline_autocast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_autocast(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("NAVA text encoder must run without pipeline autocast")

    text_encoder = FakeTextEncoder()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)
    monkeypatch.setattr(torch, "autocast", fail_autocast)

    embeds = pipeline._encode_text("plain prompt")

    assert embeds.shape == (1, 2, 4)
    assert text_encoder.calls == [["plain prompt"]]


@pytest.mark.parametrize(
    ("eos_token", "unk_token", "expected_pad_token", "expected_pad_token_id"),
    [
        ("</s>", "<unk>", "</s>", None),
        (None, "<unk>", "<unk>", None),
        (None, None, None, 0),
    ],
)
def test_text_encoder_padding_token_matches_upstream_fallback(
    eos_token: str | None,
    unk_token: str | None,
    expected_pad_token: str | None,
    expected_pad_token_id: int | None,
) -> None:
    tokenizer = SimpleNamespace(
        pad_token=None,
        eos_token=eos_token,
        unk_token=unk_token,
        pad_token_id=None,
    )

    _NAVATextEncoder._ensure_tokenizer_padding(tokenizer)

    assert tokenizer.pad_token == expected_pad_token
    assert tokenizer.pad_token_id == expected_pad_token_id


def test_image_embedding_uses_video_vae(tmp_path: Path) -> None:
    video_vae = FakeVideoVAE()
    pipeline = _make_pipeline(tmp_path, video_vae=video_vae)
    ctx = pipeline._parse_request(
        _make_request({"prompt": "continue", "multi_modal_data": {"image": Image.new("RGB", (4, 4))}})
    )

    image_embeds = pipeline._encode_image(ctx)

    assert image_embeds is not None
    assert len(video_vae.encoded_images) == 1
    assert video_vae.encoded_images[0].shape == (1, 3, 16, 16)


def test_speaker_embedding_uses_speaker_encoder(tmp_path: Path) -> None:
    speaker_encoder = FakeSpeakerEncoder()
    pipeline = _make_pipeline(tmp_path, speaker_encoder=speaker_encoder)
    ctx = pipeline._parse_request(
        _make_request({"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}})
    )

    speaker_embeds = pipeline._encode_speakers(ctx)

    assert speaker_embeds is not None
    assert speaker_embeds.shape == (1, 3)
    assert speaker_encoder.calls == [["speaker.wav"]]


def test_default_speaker_encoder_requires_local_redimnet(tmp_path: Path) -> None:
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig())

    with pytest.raises(FileNotFoundError, match="local ReDimNet"):
        speaker_encoder.encode(["speaker.wav"], device=torch.device("cpu"), dtype=torch.float32)


def test_speaker_encoder_loads_wav_file_without_torchcodec(tmp_path: Path) -> None:
    sf = pytest.importorskip("soundfile")
    wav_path = tmp_path / "speaker.wav"
    sf.write(wav_path, torch.zeros(8000).numpy(), 8000)
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig(audio_sample_rate=16000))

    waveform = speaker_encoder._load_waveform(wav_path, torch.device("cpu"))

    assert waveform.shape == (1, 16000)
    assert waveform.dtype == torch.float32


def test_speaker_encoder_uses_torch_home_redimnet_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch_home = tmp_path / "torch"
    redimnet_dir = torch_home / "hub" / "IDRnD_ReDimNet_main"
    redimnet_dir.mkdir(parents=True)
    (redimnet_dir / "hubconf.py").write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("TORCH_HOME", str(torch_home))

    loaded: list[str] = []

    class TinySpeaker(nn.Module):
        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return torch.ones(192, device=waveform.device)

    def fake_load(path: str, model: str, source: str, **kwargs: Any) -> nn.Module:
        del model, source, kwargs
        loaded.append(path)
        return TinySpeaker()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig())

    embedding = speaker_encoder.encode([torch.zeros(16000)], device=torch.device("cpu"), dtype=torch.float32)

    assert loaded == [str(redimnet_dir)]
    assert embedding.shape == (1, 192)


def test_prepare_latents_uses_video_audio_shapes(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", seed=3))

    latents = pipeline._prepare_latents(ctx, pipeline._make_generator(ctx.seed))

    assert latents["video"].shape == (1, 5, 3)
    assert latents["audio"].shape == (1, 3, 4)
    assert latents["video"].dtype == torch.float32
    assert latents["audio"].dtype == torch.float32


def test_prepare_latents_uses_latent_video_frames(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", num_frames=9, seed=3))

    latents = pipeline._prepare_latents(ctx, pipeline._make_generator(ctx.seed))

    assert latents["video"].shape == (1, 9, 3)


def test_prepare_latents_audio_length_uses_model_fps_not_output_fps(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", frame_rate=8, seed=3))

    latents = pipeline._prepare_latents(ctx, pipeline._make_generator(ctx.seed))

    assert ctx.fps == 8
    assert latents["audio"].shape == (1, 3, 4)


def test_scheduler_uses_upstream_unipc_timestep_grid() -> None:
    scheduler = NAVAFlowMatchScheduler(shift=5.0)

    timesteps = scheduler.set_timesteps(1, device=torch.device("cpu"))

    assert timesteps.dtype == torch.int64
    assert timesteps.tolist() == [999]


def test_rng_restore_mode_uses_captured_global_rng_instead_of_request_generator(tmp_path: Path) -> None:
    pipeline = _make_pipeline(
        tmp_path,
        custom_pipeline_args_extra={
            "nava_init_seed": 47,
            "nava_restore_init_cuda_rng_before_sample": True,
        },
    )

    assert pipeline._rng_state_after_init is not None
    assert pipeline._make_generator(seed=999) is None


def test_skip_request_seed_uses_global_rng_without_generator(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path, custom_pipeline_args_extra={"skip_request_seed": True})

    assert pipeline._make_generator(seed=999) is None


def test_decode_video_passes_output_frames_with_latent_tokens(tmp_path: Path) -> None:
    video_vae = FakeVideoVAE()
    pipeline = _make_pipeline(tmp_path, video_vae=video_vae)
    ctx = pipeline._parse_request(_make_request("plain prompt", num_frames=9))

    video = pipeline._decode_video(torch.zeros(1, 9, 3), ctx)

    assert video.shape == (1, 3, 33, 16, 16)
    assert video_vae.decode_calls == [{"height": 16, "width": 16, "frames": 33, "shape": torch.Size([1, 9, 3])}]


def test_video_vae_decode_reshapes_latent_frames_and_trims_output() -> None:
    class TinyWanVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_shape: torch.Size | None = None

        def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
            self.seen_shape = latent.shape
            return torch.zeros(latent.shape[0], 3, 13, 16, 16, dtype=latent.dtype, device=latent.device)

    video_vae = NAVAVideoVAE.__new__(NAVAVideoVAE)
    nn.Module.__init__(video_vae)
    video_vae.config = NAVAConfig(video_latent_ch=3, log_height=16, log_width=16)
    video_vae.latent_channels = 3
    video_vae.vae = TinyWanVAE()

    video = video_vae.decode(torch.zeros(1, 3, 3), height=16, width=16, frames=9)

    assert video_vae.vae.seen_shape == torch.Size([1, 3, 3, 1, 1])
    assert video.shape == (1, 3, 9, 16, 16)


def test_audio_vae_checkpoint_key_mapping() -> None:
    assert nava_audio_vae._map_audio_vae_key("audio_vae.decoder.conv_in.conv.weight") == ("decoder.conv_in.conv.weight")
    assert nava_audio_vae._map_vocoder_key("vocoder.vocoder.conv_pre.weight", with_bwe=True) == (
        "vocoder.conv_in.weight"
    )
    assert nava_audio_vae._map_vocoder_key("vocoder.bwe_generator.ups.0.weight", with_bwe=True) == (
        "bwe_generator.upsamplers.0.weight"
    )


def test_audio_vae_decode_unpatchifies_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class TinyAudioDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(latent_channels=2)
            self.seen_shape: torch.Size | None = None

        def decode(self, latents: torch.Tensor, return_dict: bool = False):
            self.seen_shape = latents.shape
            return (torch.ones(latents.shape[0], 2, latents.shape[2], 2, device=latents.device),)

    class TinyVocoder(nn.Module):
        output_sampling_rate = 16000

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            return torch.ones(mel.shape[0], 2, mel.shape[2] * 4, device=mel.device)

    decoder = TinyAudioDecoder()
    monkeypatch.setattr(NAVAAudioVAE, "_load_components", staticmethod(lambda _: (decoder, TinyVocoder())))
    ckpt_dir = tmp_path / "params" / "LTX2"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ltx-2.3-22b-dev_audio_vae.safetensors").write_bytes(b"placeholder")

    audio_vae = NAVAAudioVAE(str(tmp_path), NAVAConfig(audio_vae_ckpt_dir="params"))
    waveform = audio_vae.decode(torch.zeros(1, 3, 4))

    assert decoder.seen_shape == (1, 2, 3, 2)
    assert waveform.shape == (1, 2, 12)


def test_cfg_combine_video_audio_without_align(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(
        _make_request(
            "plain prompt",
            extra_args={
                "negative_prompt": "bad",
                "video_guidance_scale": 3,
                "audio_guidance_scale": 2,
                "align_3d_cfg": False,
                "timbre_cfg": False,
            },
        )
    )

    out = pipeline._combine_guidance(
        ctx,
        {"video": torch.tensor([2.0]), "audio": torch.tensor([3.0])},
        {"video": torch.tensor([1.0]), "audio": torch.tensor([1.0])},
    )

    assert torch.equal(out["video"], torch.tensor([4.0]))
    assert torch.equal(out["audio"], torch.tensor([5.0]))


def test_guidance_with_missing_negative_still_applies_align_and_timbre(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt"))

    out = pipeline._combine_guidance(
        ctx,
        {"video": torch.tensor([2.0]), "audio": torch.tensor([3.0])},
        None,
        align={"video": torch.tensor([1.0]), "audio": torch.tensor([1.0])},
        timbre={"video": torch.tensor([2.0]), "audio": torch.tensor([2.0])},
    )

    assert torch.equal(out["video"], torch.tensor([5.0]))
    assert torch.equal(out["audio"], torch.tensor([10.0]))


def test_cfg_combine_with_align_and_timbre_guidance(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(
        _make_request(
            {"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}},
            extra_args={
                "negative_prompt": "bad",
                "video_guidance_scale": 3,
                "video_align_guidance_scale": 1,
                "audio_guidance_scale": 2,
                "audio_align_guidance_scale": 1,
                "timbre_align_guidance_scale": 1,
            },
        )
    )

    out = pipeline._combine_guidance(
        ctx,
        {"video": torch.tensor([2.0]), "audio": torch.tensor([3.0])},
        {"video": torch.tensor([1.0]), "audio": torch.tensor([1.0])},
        align={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        timbre={"video": torch.tensor([2.0]), "audio": torch.tensor([2.0])},
    )

    assert torch.equal(out["video"], torch.tensor([7.0]))
    assert torch.equal(out["audio"], torch.tensor([11.0]))


def test_forward_runs_native_generation_steps(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoder()
    transformer = FakeTransformer()
    audio_vae = FakeAudioVAE()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder, transformer=transformer, audio_vae=audio_vae)

    output = pipeline.forward(_make_request("plain prompt", seed=1)).output

    assert isinstance(output, dict)
    assert output["video"].shape == (1, 3, 17, 16, 16)
    assert output["audio"].shape == (1, 3)
    assert output["audio_sample_rate"] == 16000
    assert output["fps"] == 24
    assert len(transformer.calls) == 6
    assert {call["video_grid"] for call in transformer.calls} == {(5, 1, 1)}
    assert len(audio_vae.decode_shapes) == 1
    assert text_encoder.calls == [
        ["plain prompt"],
        [NAVAConfig.video_negative_prompt, NAVAConfig.audio_negative_prompt],
    ]
    negative_calls = [call for call in transformer.calls if call["speaker_embeds"] is None and "audio_text_embeds" in call]
    assert negative_calls
    assert all(call["audio_text_embeds"] is not call["text_embeds"] for call in negative_calls)
    assert all(call["slg_layer"] == 11 for call in negative_calls)


def test_negative_prompt_mode_false_uses_zero_unconditioned_embedding(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoder()
    positive = torch.ones(1, 2, 4)
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)
    ctx = pipeline._parse_request(_make_request("plain prompt", extra_args={"negative_prompt_mode": False}))

    video_neg, audio_neg = pipeline._encode_negative_texts(ctx, positive)

    assert torch.equal(video_neg, torch.zeros_like(positive))
    assert torch.equal(audio_neg, torch.zeros_like(positive))
    assert text_encoder.calls == []


def test_negative_prompt_mode_false_handles_trimmed_text_embedding_lists(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", extra_args={"negative_prompt_mode": False}))
    positive = [torch.ones(3, 4)]

    video_neg, audio_neg = pipeline._encode_negative_texts(ctx, positive)

    assert isinstance(video_neg, list)
    assert isinstance(audio_neg, list)
    assert torch.equal(video_neg[0], torch.zeros(3, 4))
    assert torch.equal(audio_neg[0], torch.zeros(3, 4))


def test_request_level_negative_prompts_override_defaults(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoder()
    positive = torch.ones(1, 2, 4)
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)
    ctx = pipeline._parse_request(
        _make_request(
            "plain prompt",
            extra_args={
                "video_negative_prompt": "bad video",
                "audio_negative_prompt": "bad audio",
            },
        )
    )

    pipeline._encode_negative_texts(ctx, positive)

    assert text_encoder.calls == [["bad video", "bad audio"]]


def test_forward_passes_speaker_positions_to_transformer(tmp_path: Path) -> None:
    transformer = FakeTransformer()
    pipeline = _make_pipeline(
        tmp_path,
        text_encoder=FakeTextEncoderWithSpeakerPositions(),
        speaker_encoder=FakeSpeakerEncoder(),
        transformer=transformer,
    )

    pipeline.forward(_make_request({"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}}))

    assert any(call["speaker_embeds"] is not None and call["speaker_positions"] == [[1]] for call in transformer.calls)
    assert any(call["speaker_embeds"] is None and call["speaker_positions"] == [[1]] for call in transformer.calls)


def test_postprocess_returns_video_audio_metadata() -> None:
    postprocess = get_nava_post_process_func(SimpleNamespace())
    video = torch.zeros(1, 3, 2, 4, 4)
    audio = torch.zeros(1, 8)

    output = postprocess({"video": video, "audio": audio, "audio_sample_rate": 24000, "fps": 12}, output_type="pt")

    assert torch.equal(output["video"], video)
    assert output["audio"] is audio
    assert output["audio_sample_rate"] == 24000
    assert output["fps"] == 12


def test_postprocess_converts_bfloat16_video_to_numpy_float32() -> None:
    postprocess = get_nava_post_process_func(SimpleNamespace())
    video = torch.zeros(1, 3, 2, 4, 4, dtype=torch.bfloat16)

    output = postprocess({"video": video}, output_type="np")

    assert len(output["video"]) == 1
    assert output["video"][0].dtype == np.float32


def test_nava_registered_as_diffusion_pipeline() -> None:
    assert _DIFFUSION_MODELS["NAVAPipeline"] == ("nava", "pipeline_nava", "NAVAPipeline")


def test_nava_postprocess_registered() -> None:
    assert _DIFFUSION_POST_PROCESS_FUNCS["NAVAPipeline"] == "get_nava_post_process_func"


def test_nava_cache_acceleration_disabled_until_verified() -> None:
    assert "NAVAPipeline" in _NO_CACHE_ACCELERATION


def test_nava_default_init_requires_local_model_assets(tmp_path: Path) -> None:
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        model_config={"height": 16, "width": 16, "num_frames": 5},
    )

    with pytest.raises(FileNotFoundError, match="text tokenizer"):
        NAVAPipeline(od_config=od_config)


def test_nava_multimodal_capabilities_are_advertised() -> None:
    config = OmniDiffusionConfig(model="/tmp/nava", model_class_name="NAVAPipeline")

    assert supports_multimodal_input(config) == (True, True)
    assert supports_audio_output("NAVAPipeline")
    assert get_diffusion_model_metadata("NAVAPipeline").max_multimodal_image_inputs == 1


def test_transformer_load_weights_rejects_empty_match() -> None:
    transformer = NAVATransformer.__new__(NAVATransformer)
    nn.Module.__init__(transformer)
    transformer.backbone = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="No NAVA transformer weights matched"):
        transformer.load_weights([("unexpected.weight", torch.zeros(2, 2))])


def test_transformer_load_weights_accepts_backbone_prefixless_key() -> None:
    transformer = NAVATransformer.__new__(NAVATransformer)
    nn.Module.__init__(transformer)
    transformer.backbone = nn.Linear(2, 2)

    loaded = transformer.load_weights([("weight", torch.ones(2, 2))])

    assert loaded == {"backbone.weight"}
    assert torch.equal(transformer.backbone.weight, torch.ones(2, 2))


def test_transformer_marks_first_frame_clean_for_image_conditioning() -> None:
    class TinyBackbone(nn.Module):
        patch_size = (1, 1, 1)

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        def forward(self, **kwargs: Any):
            self.calls.append(kwargs)
            return [torch.zeros(3, 2, 1, 1)], [torch.zeros(1, 4)]

    transformer = NAVATransformer.__new__(NAVATransformer)
    nn.Module.__init__(transformer)
    transformer.backbone = TinyBackbone()
    transformer.patch_size = 1
    transformer.video_latent_ch = 3
    transformer.audio_latent_ch = 4

    transformer.predict_eps(
        vid_context=[torch.zeros(2, 4)],
        audio_context=[torch.zeros(2, 4)],
        latents_vid=torch.zeros(2, 3),
        latents_audio=torch.zeros(1, 4),
        timesteps=torch.ones(1),
        t_h_w_list=torch.tensor([[2, 1, 1]]),
        audio_len_list=torch.tensor([[1]]),
        is_i2v=True,
        first_frames=[torch.ones(1, 1, 1, 3)],
    )

    assert transformer.backbone.calls[0]["first_frame_is_clean"]


def _rope_apply_1d_reference(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    c_rope = freqs.shape[1]
    output = []
    for i, (length,) in enumerate(grid_sizes.tolist()):
        seq_len = length
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]
        x_i_passthrough = x_i[:, :, c_rope:]
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


def _rope_apply_3d_reference(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    split_freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                split_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                split_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                split_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


def _rope_apply_3d_to_1d_reference(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    c_rope = freqs.shape[1]
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1).reshape(seq_len, 1, -1)
        x_i_rope = x_i[:, :, :c_rope] * freqs_i
        x_i_passthrough = x_i[:, :, c_rope:]
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


def test_rope_apply_chunked_matches_reference() -> None:
    torch.manual_seed(0)
    x_video = torch.randn(2, 17, 2, 12, dtype=torch.bfloat16)
    grid_video = torch.tensor([[2, 2, 3], [1, 3, 4]])
    freqs_video = torch.polar(torch.ones(8, 6, dtype=torch.float64), torch.randn(8, 6, dtype=torch.float64))

    assert torch.equal(_rope_apply_3d(x_video, grid_video, freqs_video), _rope_apply_3d_reference(x_video, grid_video, freqs_video))

    x_audio = torch.randn(2, 11, 2, 12, dtype=torch.bfloat16)
    grid_audio = torch.tensor([[7], [9]])
    freqs_audio = torch.polar(torch.ones(12, 4, dtype=torch.float64), torch.randn(12, 4, dtype=torch.float64))

    assert torch.equal(_rope_apply_1d(x_audio, grid_audio, freqs_audio), _rope_apply_1d_reference(x_audio, grid_audio, freqs_audio))
    assert torch.equal(
        _rope_apply_3d_to_1d(x_video, grid_video, freqs_audio),
        _rope_apply_3d_to_1d_reference(x_video, grid_video, freqs_audio),
    )


@pytest.mark.parametrize(
    "parallel_config",
    [
        pytest.param(DiffusionParallelConfig(tensor_parallel_size=2), id="tp"),
        pytest.param(DiffusionParallelConfig(data_parallel_size=2), id="dp"),
        pytest.param(DiffusionParallelConfig(cfg_parallel_size=2), id="cfg"),
    ],
)
def test_nava_rejects_unverified_parallelism(tmp_path: Path, parallel_config: DiffusionParallelConfig) -> None:
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        parallel_config=parallel_config,
        model_config={"height": 16, "width": 16, "num_frames": 5},
    )

    with pytest.raises(ValueError, match="not verified"):
        NAVAPipeline(od_config=od_config)


def test_download_script_writes_model_index_without_external_runtime_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    calls: list[list[str]] = []
    _write_minimal_model_dir(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monkeypatch.setattr(sys, "argv", ["download_nava.py", "--local-dir", str(tmp_path)])

    module.main()

    assert calls == [["huggingface-cli", "download", "baidu/NAVA", "--local-dir", str(tmp_path.resolve())]]
    model_index = json.loads((tmp_path / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["_class_name"] == "NAVAPipeline"
    assert "nava_ckpt" in model_index
    assert "speaker_dir" in model_index


def test_download_script_verify_only_rejects_incomplete_model_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    monkeypatch.setattr(sys, "argv", ["download_nava.py", "--local-dir", str(tmp_path), "--verify-only"])

    with pytest.raises(SystemExit, match="incomplete"):
        module.main()


def test_download_script_verify_only_rejects_missing_text_encoder_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    _write_minimal_model_dir(tmp_path)
    (tmp_path / "Wan2.2-TI2V-5B" / "models_t5_umt5-xxl-enc-bf16.pth").unlink()
    monkeypatch.setattr(sys, "argv", ["download_nava.py", "--local-dir", str(tmp_path), "--verify-only"])

    with pytest.raises(SystemExit, match="models_t5_umt5-xxl-enc-bf16"):
        module.main()


def test_download_script_help_has_no_external_runtime_options() -> None:
    result = subprocess.run(
        [sys.executable, str(_DOWNLOAD_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--local-dir" in result.stdout
    assert "install" not in result.stdout.lower()
    assert "upstream" not in result.stdout.lower()
