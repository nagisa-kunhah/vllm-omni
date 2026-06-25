# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import numpy as np
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
        self.solver_order = 2
        self.model_outputs: list[torch.Tensor | None] = [None] * self.solver_order
        self.timestep_list: list[torch.Tensor | None] = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        self.step_index: int | None = None
        self.this_order = 1
        train_alphas = np.linspace(1.0, 1.0 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        train_sigmas = torch.from_numpy((1.0 - train_alphas).astype(np.float32))
        train_sigmas = self.shift * train_sigmas / (1 + (self.shift - 1) * train_sigmas)
        self._sigma_min = train_sigmas[-1].item()
        self._sigma_max = train_sigmas[0].item()

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device) -> torch.Tensor:
        count = int(num_inference_steps) + 1 if self.extra_one_step else int(num_inference_steps)
        sigmas = np.linspace(self._sigma_max, self._sigma_min, count).copy()
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.step_index = None
        self.this_order = 1
        return self.timesteps

    def _index_for_timestep(self, timestep: torch.Tensor) -> int:
        schedule_timesteps = self.timesteps
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(schedule_timesteps.device)
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos].item())

    def _init_step_index(self, timestep: torch.Tensor) -> None:
        self.step_index = self._index_for_timestep(timestep)

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma

    def _convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if self.step_index is None:
            raise RuntimeError("NAVA scheduler step index must be initialized before converting model output.")
        sigma_t = self.sigmas[self.step_index]
        return sample - sigma_t * model_output

    def _multistep_uni_p_bh_update(
        self,
        *,
        sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """UniPC predictor step using the B(h) update.

        The name mirrors FlowUniPCMultistepScheduler: ``uni_p`` is the
        predictor half and ``bh`` is the B(h) solver variant.
        """
        if self.step_index is None:
            raise RuntimeError("NAVA scheduler step index must be initialized before UniPC update.")
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        if m0 is None:
            raise RuntimeError("NAVA scheduler missing current converted model output.")
        x = sample

        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            if mi is None:
                raise RuntimeError("NAVA scheduler missing previous converted model output.")
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks_tensor = torch.tensor(rks, device=device)

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        b = []
        factorial_i = 1
        b_h = torch.expm1(hh)
        for i in range(1, order + 1):
            b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        if d1s:
            d1s_tensor = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                r = [torch.pow(rks_tensor, i - 1) for i in range(1, order + 1)]
                r_matrix = torch.stack(r)
                b_tensor = torch.tensor(b, device=device)
                rhos_p = torch.linalg.solve(r_matrix[:-1, :-1], b_tensor[:-1]).to(device).to(x.dtype)
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s_tensor)
        else:
            pred_res = 0

        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0 - alpha_t * b_h * pred_res
        return x_t.to(x.dtype)

    def _multistep_uni_c_bh_update(
        self,
        *,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """UniPC corrector step using the B(h) update.

        This is the companion ``uni_c`` update to the predictor above.
        """
        if self.step_index is None:
            raise RuntimeError("NAVA scheduler step index must be initialized before UniPC corrector.")
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        if m0 is None:
            raise RuntimeError("NAVA scheduler missing previous converted model output for corrector.")
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            if mi is None:
                raise RuntimeError("NAVA scheduler missing older converted model output for corrector.")
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks_tensor = torch.tensor(rks, device=device)

        r = []
        b = []
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        b_h = torch.expm1(hh)
        for i in range(1, order + 1):
            r.append(torch.pow(rks_tensor, i - 1))
            b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        if d1s:
            d1s_tensor = torch.stack(d1s, dim=1)
        else:
            d1s_tensor = None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            r_matrix = torch.stack(r)
            b_tensor = torch.tensor(b, device=device)
            rhos_c = torch.linalg.solve(r_matrix, b_tensor).to(device).to(x.dtype)

        if d1s_tensor is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s_tensor)
        else:
            corr_res = 0
        d1_t = model_t - m0
        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0 - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return x_t.to(x.dtype)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if self.timesteps.numel() == 0:
            raise RuntimeError("NAVAFlowMatchScheduler.set_timesteps() must be called before step().")
        if self.step_index is None:
            self._init_step_index(timestep)
        if self.step_index is None:
            raise RuntimeError("NAVA scheduler step index was not initialized.")

        use_corrector = self.step_index > 0 and self.last_sample is not None
        model_output_convert = self._convert_model_output(model_output, sample)
        if use_corrector:
            sample = self._multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
        self.this_order = min(this_order, self.lower_order_nums + 1)
        self.last_sample = sample
        prev_sample = self._multistep_uni_p_bh_update(sample=sample, order=self.this_order)
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self.step_index += 1
        return prev_sample

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
