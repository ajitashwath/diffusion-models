"""
diffusion/forward_process.py
=============================
Forward (noising) processes for DDPM and EDM diffusion models.

The forward process takes a clean data sample x_0 and adds noise to produce
a noisy sample x_t at a given timestep or noise level.

Classes
-------
ForwardProcess
    DDPM-style forward process using a discrete beta schedule.
EDMForwardProcess
    EDM-style forward process operating in continuous sigma space.
"""

from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract values from a 1-D scheduler tensor at indices ``t`` and reshape
    so they can be broadcast against a tensor of shape ``x_shape``.

    Parameters
    ----------
    arr : torch.Tensor
        1-D tensor of length T (precomputed scheduler quantity).
    t : torch.Tensor
        Integer indices, shape (B,).
    x_shape : torch.Size
        Shape of the data tensor (B, C, H, W) or similar.

    Returns
    -------
    torch.Tensor
        Values at ``t``, reshaped to (B, 1, 1, …) for broadcasting.
    """
    batch_size = t.shape[0]
    out = arr.to(device=t.device)[t]
    # Reshape to (B, 1, 1, ...) for broadcasting against x_shape
    return out.view(batch_size, *([1] * (len(x_shape) - 1)))


# ---------------------------------------------------------------------------
# DDPM ForwardProcess
# ---------------------------------------------------------------------------

class ForwardProcess:
    """
    DDPM forward (noising) process.

    Implements the closed-form forward diffusion kernel:

        q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1 - ᾱ_t) I)

    which allows direct sampling at any timestep ``t`` without iterating
    through all intermediate steps.

    Parameters
    ----------
    scheduler : LinearScheduler | CosineScheduler
        A precomputed noise schedule that exposes the tensors:
        ``sqrt_alphas_cumprod``, ``sqrt_one_minus_alphas_cumprod``,
        ``sqrt_recip_alphas_cumprod``, ``sqrt_recip_alphas_cumprod_m1``,
        ``posterior_mean_coef1``, ``posterior_mean_coef2``,
        ``posterior_log_variance_clipped``.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Forward sample
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the forward kernel q(x_t | x_0).

        x_t = sqrt(ᾱ_t) * x_0  +  sqrt(1 - ᾱ_t) * ε,   ε ~ N(0, I)

        Parameters
        ----------
        x_start : torch.Tensor
            Clean data tensor, shape (B, C, H, W).
        t : torch.Tensor
            Integer timestep indices in [0, T-1], shape (B,).
        noise : torch.Tensor, optional
            Pre-generated Gaussian noise.  If ``None``, fresh noise is
            sampled from N(0, I).

        Returns
        -------
        x_t : torch.Tensor
            Noisy sample at timestep t, shape (B, C, H, W).
        noise : torch.Tensor
            The noise that was added (same shape as x_start).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        s = self.scheduler
        sqrt_ab = _extract(s.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1mab = _extract(s.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_t = sqrt_ab * x_start + sqrt_1mab * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # x_0 / eps prediction helpers
    # ------------------------------------------------------------------

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover the predicted clean sample x_0 from (x_t, ε̂).

        Using the rearranged forward formula:

            x_0 = (1/sqrt(ᾱ_t)) * x_t  -  sqrt(1/ᾱ_t - 1) * ε̂

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy sample at timestep t, shape (B, C, H, W).
        t : torch.Tensor
            Integer timestep indices, shape (B,).
        eps : torch.Tensor
            Predicted noise, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Predicted x_0, shape (B, C, H, W).
        """
        s = self.scheduler
        recip = _extract(s.sqrt_recip_alphas_cumprod, t, x_t.shape)
        recip_m1 = _extract(s.sqrt_recip_alphas_cumprod_m1, t, x_t.shape)
        return recip * x_t - recip_m1 * eps

    def predict_eps_from_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover the predicted noise ε from (x_t, x̂_0).

        Rearranging the forward formula:

            ε = (x_t - sqrt(ᾱ_t) * x_0) / sqrt(1 - ᾱ_t)

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy sample at timestep t, shape (B, C, H, W).
        t : torch.Tensor
            Integer timestep indices, shape (B,).
        x0 : torch.Tensor
            Predicted clean sample, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Predicted noise ε, shape (B, C, H, W).
        """
        s = self.scheduler
        sqrt_ab = _extract(s.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = _extract(s.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_ab * x0) / sqrt_1mab

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the posterior distribution q(x_{t-1} | x_t, x_0).

        Mean:
            μ_t = coef1_t * x_0  +  coef2_t * x_t

        where:
            coef1_t = sqrt(ᾱ_{t-1}) * β_t / (1 - ᾱ_t)
            coef2_t = sqrt(α_t) * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)

        Variance (log, clipped):
            log σ_t^2 = log max(β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t), 1e-20)

        Parameters
        ----------
        x_start : torch.Tensor
            Clean (or predicted clean) sample, shape (B, C, H, W).
        x_t : torch.Tensor
            Noisy sample at timestep t, shape (B, C, H, W).
        t : torch.Tensor
            Integer timestep indices, shape (B,).

        Returns
        -------
        mean : torch.Tensor
            Posterior mean μ_t, shape (B, C, H, W).
        log_variance_clipped : torch.Tensor
            Clipped log of posterior variance, shape (B, C, H, W).
        """
        s = self.scheduler
        coef1 = _extract(s.posterior_mean_coef1, t, x_t.shape)
        coef2 = _extract(s.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_start + coef2 * x_t

        log_var = _extract(s.posterior_log_variance_clipped, t, x_t.shape)
        return mean, log_var

    def __repr__(self) -> str:
        return f"ForwardProcess(scheduler={self.scheduler!r})"


# ---------------------------------------------------------------------------
# EDM ForwardProcess
# ---------------------------------------------------------------------------

class EDMForwardProcess:
    """
    EDM forward (noising) process in continuous sigma space.

    The EDM forward process is simply:

        x_noisy = x_start + σ * ε,   ε ~ N(0, I)

    with σ drawn from the log-normal distribution given by the scheduler.

    Parameters
    ----------
    scheduler : EDMScheduler
        EDM scheduler that provides ``sample_sigma``.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def q_sample(
        self,
        x_start: torch.Tensor,
        sigma: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the EDM forward process:

            x_noisy = x_start + σ * ε,   ε ~ N(0, I)

        Parameters
        ----------
        x_start : torch.Tensor
            Clean data tensor, shape (B, C, H, W).
        sigma : torch.Tensor
            Per-sample noise standard deviations, shape (B,).
            Typically produced by ``scheduler.sample_sigma(B)``.
        noise : torch.Tensor, optional
            Pre-generated Gaussian noise with the same shape as ``x_start``.
            If ``None``, fresh noise is sampled from N(0, I).

        Returns
        -------
        x_noisy : torch.Tensor
            Noisy sample, shape (B, C, H, W).
        noise : torch.Tensor
            The noise that was added (same shape as x_start).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Reshape sigma to (B, 1, 1, 1) for broadcasting
        sigma_ = sigma.view(-1, *([1] * (x_start.ndim - 1)))
        x_noisy = x_start + sigma_ * noise
        return x_noisy, noise

    def __repr__(self) -> str:
        return f"EDMForwardProcess(scheduler={self.scheduler!r})"
