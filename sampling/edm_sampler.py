"""
EDM Sampler
===========
Implements the **Heun 2nd-order ODE solver** (Algorithm 2) from:

    Karras et al., "Elucidating the Design Space of Diffusion-Based
    Generative Models", NeurIPS 2022.
    https://arxiv.org/abs/2206.00364

The sampler uses the EDMScheduler's stochastic churn mechanism to add a
controlled amount of noise at each step (gamma trick), which helps escape
local minima in the score function landscape.

Number of Function Evaluations (NFE):
--------------------------------------
- For all but the very last step, Heun performs two model evaluations
  (Euler predictor + 2nd-order corrector).
- The last step (sigma_next = 0) only uses one evaluation.
- Total: 2 * (steps - 1) + 1 = 2*steps - 1
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from tqdm import tqdm


class EDMSampler:
    """
    Heun 2nd-order sampler for EDM-preconditioned diffusion models.

    Parameters
    ----------
    model : EDMPrecond
        EDM-preconditioned denoiser.  Signature:
        ``model(x, sigma) -> denoised``
        where ``sigma`` is a **float** tensor of shape ``(B,)``.
    scheduler : EDMScheduler
        EDM noise scheduler.  Must expose ``sigma_min``, ``sigma_max``,
        ``sigma_data``, ``rho``, ``S_churn``, ``S_tmin``, ``S_tmax``,
        ``S_noise``, and the methods ``get_sampling_sigmas``,
        ``c_skip``, ``c_out``, ``c_in``, ``c_noise``.
    device : str | torch.device
        Computation device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scheduler,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Populated after sample() completes
        self.nfe: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _denoise(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Call the EDM-preconditioned model.

        Parameters
        ----------
        x : torch.Tensor  ``(B, C, H, W)``
        sigma : torch.Tensor  ``(B,)``  – noise level for each sample.

        Returns
        -------
        denoised : torch.Tensor  ``(B, C, H, W)``
        """
        return self.model(x, sigma)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        steps: int = 18,
        S_churn: Optional[float] = None,
        S_tmin: Optional[float] = None,
        S_tmax: Optional[float] = None,
        S_noise: Optional[float] = None,
        clip_denoised: bool = True,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using the Heun 2nd-order ODE solver.

        Parameters
        ----------
        shape : tuple
            ``(B, C, H, W)``.
        steps : int
            Number of discretisation steps (NFE ≈ 2*steps - 1).
        S_churn : float, optional
            Churn coefficient.  Defaults to ``scheduler.S_churn``.
        S_tmin : float, optional
            Lower sigma bound for churn.  Defaults to ``scheduler.S_tmin``.
        S_tmax : float, optional
            Upper sigma bound for churn.  Defaults to ``scheduler.S_tmax``.
        S_noise : float, optional
            Noise injection scale for churn.  Defaults to ``scheduler.S_noise``.
        clip_denoised : bool
            Clamp output to ``[-1, 1]`` after the **final** step.
        progress : bool
            Show a ``tqdm`` progress bar.

        Returns
        -------
        x : torch.Tensor
            Generated samples of shape ``shape``.
        """
        # ---- Resolve defaults from scheduler ---------------------------
        S_churn = S_churn if S_churn is not None else self.scheduler.S_churn
        S_tmin  = S_tmin  if S_tmin  is not None else self.scheduler.S_tmin
        S_tmax  = S_tmax  if S_tmax  is not None else self.scheduler.S_tmax
        S_noise = S_noise if S_noise is not None else self.scheduler.S_noise

        B = shape[0]

        # ---- Sigma schedule: length steps+1, last element is 0 --------
        sigmas = self.scheduler.get_sampling_sigmas(steps, self.device)
        # sigmas shape: (steps+1,)  e.g. [80.0, …, 0.002, 0.0]

        # ---- Sample initial noise scaled by the largest sigma ----------
        x = torch.randn(shape, device=self.device) * sigmas[0]

        nfe = 0  # function evaluation counter

        sqrt2m1 = math.sqrt(2) - 1  # ≈ 0.4142

        iterator = tqdm(
            range(steps),
            desc="EDM (Heun) sampling",
            disable=not progress,
            dynamic_ncols=True,
        )

        for i in iterator:
            sigma_i    = sigmas[i]       # current  σ   (scalar tensor)
            sigma_next = sigmas[i + 1]   # next     σ   (scalar tensor)

            # ----------------------------------------------------------------
            # Stochastic churn: inflate σ slightly and inject noise so the
            # sampler can escape local minima in the score landscape.
            # ----------------------------------------------------------------
            in_churn_range = (
                S_tmin <= sigma_i.item() <= S_tmax
            )
            gamma = (
                min(S_churn / steps, sqrt2m1) if in_churn_range else 0.0
            )

            sigma_hat = sigma_i * (1.0 + gamma)

            if gamma > 0.0:
                # x̂ = x + ε * sqrt(σ_hat² - σ_i²),  ε ~ N(0, I)
                noise_scale = (sigma_hat ** 2 - sigma_i ** 2).sqrt()
                x = x + torch.randn_like(x) * (noise_scale * S_noise)

            # ----------------------------------------------------------------
            # First derivative estimate (Euler step)
            # ----------------------------------------------------------------
            # Expand scalar sigma_hat to shape (B,) for the model
            sigma_hat_batch = sigma_hat.expand(B).to(self.device)

            denoised = self._denoise(x, sigma_hat_batch)
            nfe += 1

            # Score function direction: d = (x - D_θ(x, σ)) / σ
            d = (x - denoised) / sigma_hat

            dt = sigma_next - sigma_hat  # negative (sigma is decreasing)

            # ----------------------------------------------------------------
            # ODE step
            # ----------------------------------------------------------------
            is_last_step = (sigma_next.item() == 0.0)

            if is_last_step:
                # Pure Euler for the final step (Heun correction is undefined
                # at σ = 0 because we'd divide by zero in d_next).
                x = x + d * dt
            else:
                # Heun 2nd-order corrector
                # 1. Euler predictor
                x_euler = x + d * dt

                # 2. Evaluate model at predicted point
                sigma_next_batch = sigma_next.expand(B).to(self.device)
                denoised_next = self._denoise(x_euler, sigma_next_batch)
                nfe += 1

                d_next = (x_euler - denoised_next) / sigma_next

                # 3. Average derivatives and take corrected step
                d_avg = (d + d_next) / 2.0
                x = x + d_avg * dt

            if progress:
                iterator.set_postfix(
                    sigma=f"{sigma_i.item():.4f}",
                    sigma_next=f"{sigma_next.item():.4f}",
                    nfe=nfe,
                )

        # ---- Final clamp --------------------------------------------------
        if clip_denoised:
            x = x.clamp(-1.0, 1.0)

        self.nfe = nfe
        return x

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def theoretical_nfe(steps: int) -> int:
        """
        Return the theoretical NFE for ``steps`` Heun steps.

        NFE = 2 * (steps - 1) + 1 = 2*steps - 1
        """
        return 2 * steps - 1

    def __repr__(self) -> str:
        return (
            f"EDMSampler("
            f"sigma_min={self.scheduler.sigma_min}, "
            f"sigma_max={self.scheduler.sigma_max}, "
            f"device={self.device})"
        )
