# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS serving adapters (Nano + full family).

Both variants share the same build/validate flow (``_build_moss_tts_params``
handles each); they are registered under distinct model-type names.
"""

from typing import TYPE_CHECKING

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, conditioning_cache_salt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


class _MossTTSAdapterBase(ARTTSAdapter):
    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        err = self.ctx.server._apply_uploaded_speaker(request)
        if err:
            return err
        return self.ctx.server._validate_moss_tts_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        tts_params = await server._build_moss_tts_params(request)
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                tts_params["voice_name"] = [voice_lower]
                tts_params["voice_created_at"] = [server._voice_created_at(voice_lower)]
        # MOSS reads the resolved seed at build time (it samples internally).
        if sampling_params_list and getattr(sampling_params_list[0], "seed", None) is not None:
            tts_params["seed"] = [sampling_params_list[0].seed]
        if isinstance(tts_params.get("prompt_token_ids"), list):
            prompt_token_ids = tts_params.pop("prompt_token_ids")
            prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        else:
            prompt = tokens_input(prompt_token_ids=[1])
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)


@register_tts_adapter
class MossTTSNanoAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts_nano"})
    name = "moss_tts_nano"


@register_tts_adapter
class MossTTSAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts", "moss_tts_codec", "moss_tts_local", "moss_tts_local_codec"})
    name = "moss_tts"


@register_tts_adapter
class MossTTSLocalAdapter(ARTTSAdapter):
    """Adapter for MOSS-TTS Local Transformer v1.5 default-voice serving."""

    stage_keys = frozenset({"moss_tts_local"})
    name = "moss_tts_local"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.voice is not None and request.voice.lower() != "default":
            return "MOSS-TTS Local currently supports only voice='default'"
        unsupported_fields = {
            "task_type": request.task_type,
            "instructions": request.instructions,
            "ref_audio": request.ref_audio,
            "ref_text": request.ref_text,
            "ref_audio_2": request.ref_audio_2,
            "ambient_sound": request.ambient_sound,
            "duration_seconds": request.duration_seconds,
            "x_vector_only_mode": request.x_vector_only_mode,
            "speaker_embedding": request.speaker_embedding,
        }
        for name, value in unsupported_fields.items():
            if value is not None:
                return f"'{name}' is not supported for MOSS-TTS Local; use seed-tts-text/default voice"
        if request.max_new_tokens is not None and request.max_new_tokens <= 0:
            return "'max_new_tokens' must be a positive integer"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        del has_inline_ref_audio
        tts_params = dict(request.extra_params or {})
        if request.seed is not None:
            tts_params["seed"] = [request.seed]
        elif sampling_params_list and getattr(sampling_params_list[0], "seed", None) is not None:
            tts_params["seed"] = [sampling_params_list[0].seed]

        prompt = {
            "prompt": request.input,
            "additional_information": tts_params,
            "cache_salt": conditioning_cache_salt(request, tts_params),
        }
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)
