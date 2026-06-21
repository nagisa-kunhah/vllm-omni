# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.nava.config import NAVAConfig


class NAVASpeakerEncoder(nn.Module):
    def __init__(self, model_root: str, config: NAVAConfig) -> None:
        super().__init__()
        self.embed_dim = config.speaker_embed_dim
        self.speaker_dir = os.path.join(model_root, config.speaker_dir)
        self.sample_rate = config.audio_sample_rate
        self._model: nn.Module | None = None

    def encode(self, wavs: list[Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not wavs:
            return torch.empty(0, self.embed_dim, device=device, dtype=dtype)
        model = self._load_model(device)
        embeddings = []
        for wav in wavs:
            waveform = self._load_waveform(wav, device)
            with torch.inference_mode():
                embedding = model(waveform)
            embeddings.append(embedding.reshape(1, -1).to(device=device, dtype=dtype))
        return torch.cat(embeddings, dim=0)

    def _load_model(self, device: torch.device) -> nn.Module:
        if self._model is not None:
            return self._model

        kwargs = {
            "model_name": "M",
            "train_type": "ft_mix",
            "dataset": "vb2+vox2+cnc",
        }
        hubconf = os.path.join(self.speaker_dir, "hubconf.py")
        if not os.path.exists(hubconf):
            raise FileNotFoundError(
                "NAVA speaker timbre conditioning requires a local ReDimNet speaker encoder under "
                f"{self.speaker_dir}. Prepare the speaker directory before using spk_wavs."
            )
        model = torch.hub.load(self.speaker_dir, "ReDimNet", source="local", **kwargs)
        self._model = model.eval().to(device)
        return self._model

    def _load_waveform(self, wav: Any, device: torch.device) -> torch.Tensor:
        import torchaudio

        if isinstance(wav, (str, os.PathLike)):
            waveform, sample_rate = torchaudio.load(os.fspath(wav))
        elif isinstance(wav, tuple) and len(wav) == 2:
            sample_rate, waveform = wav
            waveform = torch.as_tensor(waveform)
        else:
            waveform = torch.as_tensor(wav)
            sample_rate = self.sample_rate

        waveform = waveform.to(torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"NAVA speaker reference must be waveform [C,T] or [T], got {tuple(waveform.shape)}.")
        if int(sample_rate) != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=int(sample_rate), new_freq=self.sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        max_samples = int(30.0 * self.sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
        if waveform.shape[-1] == 0:
            raise ValueError("NAVA speaker reference waveform is empty.")
        if waveform.shape[-1] < self.sample_rate:
            waveform = F.pad(waveform, (0, self.sample_rate - waveform.shape[-1]))
        return waveform.to(device)
