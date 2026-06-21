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
from vllm_omni.diffusion.models.nava.nava_transformer import NAVATransformer, WanSelfAttention
from vllm_omni.diffusion.models.nava.pipeline_nava import NAVAPipeline, get_nava_post_process_func
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
        return {name for name, _ in weights}


def _make_pipeline(tmp_path: Path, **custom_components: Any) -> NAVAPipeline:
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
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        model_config=model_config,
        custom_pipeline_args={
            "text_encoder": custom_components.get("text_encoder", FakeTextEncoder()),
            "video_vae": custom_components.get("video_vae", FakeVideoVAE()),
            "audio_vae": custom_components.get("audio_vae", FakeAudioVAE()),
            "speaker_encoder": custom_components.get("speaker_encoder", FakeSpeakerEncoder()),
            "transformer": custom_components.get("transformer", FakeTransformer()),
        },
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


def test_config_accepts_model_index_aliases() -> None:
    cfg = NAVAConfig.from_dict({"nava_ckpt": "custom.safetensors", "height": 16, "width": 32})

    assert cfg.ckpt_name == "custom.safetensors"
    assert cfg.log_height == 16
    assert cfg.log_width == 32


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


def test_parse_request_normalizes_unaligned_output_frames(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", num_frames=10))

    assert ctx.frames == 9


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


def test_prepare_latents_uses_video_audio_shapes(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", seed=3))

    latents = pipeline._prepare_latents(ctx, pipeline._make_generator(ctx.seed))

    assert latents["video"].shape == (1, 2, 3)
    assert latents["audio"].shape == (1, 1, 4)


def test_prepare_latents_uses_latent_video_frames(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", num_frames=9, seed=3))

    latents = pipeline._prepare_latents(ctx, pipeline._make_generator(ctx.seed))

    assert latents["video"].shape == (1, 3, 3)


def test_apply_first_frame_condition_keeps_image_latent_tokens(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    video_latents = torch.zeros(1, 3, 3)
    image_embeds = torch.ones(1, 1, 3) * 7

    conditioned = pipeline._apply_first_frame_condition(video_latents, image_embeds, (3, 1, 1))

    assert torch.equal(conditioned[:, :1], image_embeds)
    assert torch.equal(conditioned[:, 1:], torch.zeros(1, 2, 3))


def test_decode_video_passes_output_frames_with_latent_tokens(tmp_path: Path) -> None:
    video_vae = FakeVideoVAE()
    pipeline = _make_pipeline(tmp_path, video_vae=video_vae)
    ctx = pipeline._parse_request(_make_request("plain prompt", num_frames=9))

    video = pipeline._decode_video(torch.zeros(1, 3, 3), ctx)

    assert video.shape == (1, 3, 9, 16, 16)
    assert video_vae.decode_calls == [{"height": 16, "width": 16, "frames": 9, "shape": torch.Size([1, 3, 3])}]


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
    assert output["video"].shape == (1, 3, 5, 16, 16)
    assert output["audio"].shape == (1, 1)
    assert output["audio_sample_rate"] == 16000
    assert output["fps"] == 24
    assert len(transformer.calls) == 6
    assert {call["video_grid"] for call in transformer.calls} == {(2, 1, 1)}
    assert len(audio_vae.decode_shapes) == 1
    assert text_encoder.calls == [["plain prompt"], [""]]


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
