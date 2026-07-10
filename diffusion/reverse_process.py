"""
diffusion/reverse_process.py
=============================
DDPM reverse (denoising) process — one-step denoising from x_t to x_{t-1}.

The reverse process parametrises the denoising distribution:

    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t^2 I)

with μ_θ derived from the model output, which can predict one of:

* **eps**  — the added noise ε  (most common, Ho et al. 2020)
* **x0**   — the clean sample x_0
* **v**    — the velocity v (Salimans & Ho 2022, https://arxiv.org/abs/2202.00512)

Classes
-------
ReverseProcess
    One-step DDPM denoising sampler.
"""

from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract values from a 1-D scheduler tensor at indices ``t`` and reshape
    for broadcasting against ``x_shape``.

    Parameters
    ----------
    arr : torch.Tensor
        1-D scheduler tensor of length T.
    t : torch.Tensor
        Integer indices, shape (B,).
    x_shape : torch.Size
        Shape of the data tensor for determining broadcast shape.

    Returns
    -------
    torch.Tensor
        Extracted values, shape (B, 1, 1, ...).
    """
    out = arr.to(device=t.device)[t]
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


# ---------------------------------------------------------------------------
# ReverseProcess
# ---------------------------------------------------------------------------

class ReverseProcess:
    """
    DDPM one-step reverse (denoising) process.

    Given the model's prediction ``model_out`` at timestep ``t``, computes
    the denoised sample at timestep ``t - 1`` using the posterior mean and
    variance.

    Supports three prediction parametrisations:

    * ``'eps'``  — model predicts the noise ε added to x_0
    * ``'x0'``   — model directly predicts the clean sample x_0
    * ``'v'``    — model predicts the velocity v (Salimans & Ho 2022)

    The velocity parametrisation is:
        v = sqrt(ᾱ_t) * ε − sqrt(1 − ᾱ_t) * x_0

    Parameters
    ----------
    scheduler : LinearScheduler | CosineScheduler
        A precomputed noise schedule exposing the required tensors.
    predict_type : {'eps', 'x0', 'v'}
        Which quantity the model predicts.  Default: ``'eps'``.
    """

    VALID_PREDICT_TYPES = frozenset({"eps", "x0", "v"})

    def __init__(self, scheduler, predict_type: str = "eps"):
        if predict_type not in self.VALID_PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {self.VALID_PREDICT_TYPES}, "
                f"got '{predict_type}'"
            )
        self.scheduler = scheduler
        self.predict_type = predict_type

    # ------------------------------------------------------------------
    # Internal: convert model output to (x0, eps)
    # ------------------------------------------------------------------

    def _get_x0_and_eps(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the model's raw output to (x̂_0, ε̂) regardless of prediction type.

        Parameters
        ----------
        model_out : torch.Tensor
            Raw network output, same shape as x_t: (B, C, H, W).
        x_t : torch.Tensor
            Noisy sample at timestep t, shape (B, C, H, W).
        t : torch.Tensor
            Integer timestep indices, shape (B,).

        Returns
        -------
        x0 : torch.Tensor
            Predicted clean sample, shape (B, C, H, W).
        eps : torch.Tensor
            Predicted noise, shape (B, C, H, W).

        Raises
        ------
        ValueError
            If an unrecognised ``predict_type`` is stored on the instance.
        """
        s = self.scheduler

        if self.predict_type == "eps":
            # Model predicts noise; recover x_0 from (x_t, ε̂)
            # x_0 = (1/sqrt(ᾱ_t)) * x_t  -  sqrt(1/ᾱ_t - 1) * ε̂
            eps = model_out
            recip = _extract(s.sqrt_recip_alphas_cumprod, t, x_t.shape)
            recip_m1 = _extract(s.sqrt_recip_alphas_cumprod_m1, t, x_t.shape)
            x0 = recip * x_t - recip_m1 * eps

        elif self.predict_type == "x0":
            # Model directly predicts clean sample; recover ε from (x_t, x̂_0)
            # ε = (x_t - sqrt(ᾱ_t) * x̂_0) / sqrt(1 - ᾱ_t)
            x0 = model_out
            sqrt_ab = _extract(s.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_1mab = _extract(s.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            eps = (x_t - sqrt_ab * x0) / sqrt_1mab

        elif self.predict_type == "v":
            # Model predicts velocity v = sqrt(ᾱ_t)*ε − sqrt(1−ᾱ_t)*x_0
            # Inverting:
            #   x_0 = sqrt(ᾱ_t) * x_t  − sqrt(1 − ᾱ_t) * v
            #   ε   = sqrt(ᾱ_t) * v    + sqrt(1 − ᾱ_t) * x_t
            v = model_out
            sqrt_ab = _extract(s.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_1mab = _extract(s.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            x0 = sqrt_ab * x_t - sqrt_1mab * v
            eps = sqrt_ab * v + sqrt_1mab * x_t

        else:
            raise ValueError(f"Unknown predict_type: '{self.predict_type}'")

        return x0, eps

    # ------------------------------------------------------------------
    # One-step reverse sample
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Draw one reverse-diffusion sample x_{t-1} ~ p_θ(x_{t-1} | x_t).

        Algorithm
        ---------
        1. Convert ``model_out`` to (x̂_0, ε̂) using :meth:`_get_x0_and_eps`.
        2. Optionally clip x̂_0 to [-1, 1] (stabilises training; Ho et al.).
        3. Compute the posterior mean:
               μ_t = coef1_t * x̂_0  +  coef2_t * x_t
        4. Sample:
               x_{t-1} = μ_t  +  exp(0.5 * log σ_t^2) * z,
           where z ~ N(0, I) for t > 0, and z = 0 for t = 0 (no noise at
           the final step to avoid adding unnecessary variance).

        Parameters
        ----------
        model_out : torch.Tensor
            Raw network prediction, shape (B, C, H, W).
        x_t : torch.Tensor
            Noisy sample at the current timestep, shape (B, C, H, W).
        t : torch.Tensor
            Integer timestep indices, shape (B,).  Values are in [0, T-1].
        clip_denoised : bool
            If ``True``, clip the predicted x_0 to [-1, 1] before computing
            the posterior mean.  Strongly recommended for pixel-space models.

        Returns
        -------
        torch.Tensor
            Denoised sample x_{t-1}, shape (B, C, H, W).
        """
        s = self.scheduler
        x0, _ = self._get_x0_and_eps(model_out, x_t, t)

        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)

        # Posterior mean: μ_t = coef1_t * x̂_0  +  coef2_t * x_t
        coef1 = _extract(s.posterior_mean_coef1, t, x_t.shape)
        coef2 = _extract(s.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0 + coef2 * x_t

        # Posterior log variance (clipped)
        log_var = _extract(s.posterior_log_variance_clipped, t, x_t.shape)

        # Sample noise; suppress it at the very last step (t == 0)
        noise = torch.randn_like(x_t)
        # Create a mask: 1 where t > 0, 0 where t == 0
        nonzero_mask = (t > 0).float().view(-1, *([1] * (x_t.ndim - 1)))

        # x_{t-1} = μ_t + σ_t * z  (z = 0 at t == 0)
        x_prev = mean + nonzero_mask * (0.5 * log_var).exp() * noise
        return x_prev

    # ------------------------------------------------------------------
    # Utility: get predicted x0 without sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_start(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Return the predicted clean sample x̂_0 without performing a reverse step.

        Useful for visualisation / debugging.

        Parameters
        ----------
        model_out : torch.Tensor
            Raw network prediction, shape (B, C, H, W).
        x_t : torch.Tensor
            Noisy sample at timestep t, shape (B, C, H, W).
        t : torch.Tensor
            Integer timestep indices, shape (B,).
        clip_denoised : bool
            If ``True``, clip x̂_0 to [-1, 1].

        Returns
        -------
        torch.Tensor
            Predicted x_0, shape (B, C, H, W).
        """
        x0, _ = self._get_x0_and_eps(model_out, x_t, t)
        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def __repr__(self) -> str:
        return (
            f"ReverseProcess(scheduler={self.scheduler!r}, "
            f"predict_type='{self.predict_type}')"
        )
