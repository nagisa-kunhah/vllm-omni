# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import torch


class NAVAFlowMatchScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = True,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.sigmas = torch.empty(0)
        self.timesteps = torch.empty(0)

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device) -> torch.Tensor:
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        count = int(num_inference_steps) + 1 if self.extra_one_step else int(num_inference_steps)
        sigmas = torch.linspace(sigma_start, self.sigma_min, count, device=device)
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        return self.timesteps

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if self.timesteps.numel() == 0:
            raise RuntimeError("NAVAFlowMatchScheduler.set_timesteps() must be called before step().")
        timestep = timestep.to(device=self.timesteps.device, dtype=self.timesteps.dtype)
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sigma_next = torch.zeros_like(sigma) if timestep_id + 1 >= len(self.timesteps) else self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_next - sigma)

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.to(device=self.timesteps.device, dtype=self.timesteps.dtype)
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id].to(device=original.device, dtype=original.dtype)
        while sigma.ndim < original.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original + sigma * noise


def nava_audio_latent_length(frames: int, fps: float, audio_tokens_per_sec: float) -> int:
    video_duration_s = ((int(frames) - 1) * 4 + 1) / float(fps)
    return max(1, math.ceil(video_duration_s * audio_tokens_per_sec))
