"""
diffusion/schedulers.py
=======================
Noise schedulers for diffusion models.

Three schedulers are provided:

* ``LinearScheduler``  — DDPM linear beta schedule (Ho et al. 2020).
* ``CosineScheduler``  — Improved cosine schedule (Nichol & Dhariwal 2021).
* ``EDMScheduler``     — Elucidating Diffusion Models (Karras et al. 2022).
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(arr, dtype=torch.float32) -> torch.Tensor:
    """Convert a list / numpy array to a float32 CPU tensor."""
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype=dtype)
    return torch.tensor(arr, dtype=dtype)


# ---------------------------------------------------------------------------
# LinearScheduler
# ---------------------------------------------------------------------------

class LinearScheduler:
    """
    DDPM linear noise schedule (Ho et al. 2020, https://arxiv.org/abs/2006.11239).

    Precomputes all quantities needed by the forward and reverse processes so
    that they are only computed once and can be reused efficiently.

    Parameters
    ----------
    T : int
        Total number of diffusion timesteps.
    beta_start : float
        Value of beta at the first timestep.
    beta_end : float
        Value of beta at the final timestep.
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        self._precompute(betas)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute(self, betas: torch.Tensor):
        """
        Precompute all tensors derived from the beta schedule.
        Stored as float32 CPU tensors; use .to(device) when needed.
        """
        T = betas.shape[0]
        self.T = T

        self.betas: torch.Tensor = betas                            # β_t
        self.alphas: torch.Tensor = 1.0 - betas                    # α_t = 1 - β_t

        alphas_cumprod = torch.cumprod(self.alphas, dim=0)          # ᾱ_t
        self.alphas_cumprod: torch.Tensor = alphas_cumprod

        # ᾱ_{t-1}, with ᾱ_0 = 1 (convention: no noise at step 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod_prev: torch.Tensor = alphas_cumprod_prev

        # --- quantities used to compute q(x_t | x_0) ---
        self.sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - alphas_cumprod)

        # --- quantities used to recover x_0 from (x_t, ε) ---
        self.sqrt_recip_alphas_cumprod: torch.Tensor = torch.rsqrt(alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_m1: torch.Tensor = torch.sqrt(
            1.0 / alphas_cumprod - 1.0
        )

        # --- posterior q(x_{t-1} | x_t, x_0) ---
        # σ_t^2 = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance: torch.Tensor = posterior_variance

        # Clipped log variance so it is never -inf at t=0
        self.posterior_log_variance_clipped: torch.Tensor = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )

        # μ_t = coef1 * x_0  +  coef2 * x_t
        # coef1 = sqrt(ᾱ_{t-1}) * β_t / (1 - ᾱ_t)
        # coef2 = sqrt(α_t) * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_mean_coef1: torch.Tensor = (
            torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2: torch.Tensor = (
            torch.sqrt(self.alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-noise ratio at timestep t.

        SNR(t) = ᾱ_t / (1 - ᾱ_t)

        Parameters
        ----------
        t : torch.Tensor
            Integer timestep indices, shape (B,).

        Returns
        -------
        torch.Tensor
            SNR values, shape (B,).
        """
        ab = self.alphas_cumprod.to(t.device)[t]
        return ab / (1.0 - ab)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(T={self.T}, "
            f"beta_start={self.beta_start}, beta_end={self.beta_end})"
        )


# ---------------------------------------------------------------------------
# CosineScheduler
# ---------------------------------------------------------------------------

class CosineScheduler(LinearScheduler):
    """
    Improved cosine noise schedule (Nichol & Dhariwal 2021,
    https://arxiv.org/abs/2102.09672).

    Uses a cosine-based alpha-bar schedule:

        f(t) = cos((t/T + s) / (1 + s) * π/2)^2

        ᾱ_t = f(t) / f(0)

        β_t = 1 − ᾱ_t / ᾱ_{t−1}   (clipped to [0, 0.999])

    Parameters
    ----------
    T : int
        Total number of diffusion timesteps.
    s : float
        Small offset to prevent β_t from being too small near t=0.
        Recommended value: 0.008.
    """

    def __init__(self, T: int = 1000, s: float = 0.008):
        # Do NOT call LinearScheduler.__init__; build betas manually.
        self.T = T
        self.s = s
        self.beta_start = None  # not applicable
        self.beta_end = None    # not applicable

        steps = T + 1  # we need f(0)…f(T)
        t = torch.linspace(0, T, steps, dtype=torch.float64)
        f_t = torch.cos(((t / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_cumprod = (f_t / f_t[0]).float()

        # β_t = 1 − ᾱ_t / ᾱ_{t−1}, clipped to [0, 0.999]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, min=0.0, max=0.999)

        self._precompute(betas)

    def __repr__(self) -> str:
        return f"CosineScheduler(T={self.T}, s={self.s})"


# ---------------------------------------------------------------------------
# EDMScheduler
# ---------------------------------------------------------------------------

class EDMScheduler:
    """
    Elucidating the Design Space of Diffusion-Based Generative Models
    (Karras et al. 2022, https://arxiv.org/abs/2206.00364).

    This scheduler operates in *continuous sigma space* rather than the
    discrete timestep space used by DDPM / DDIM schedulers.

    Parameters
    ----------
    sigma_min : float
        Minimum noise standard deviation used during sampling.
    sigma_max : float
        Maximum noise standard deviation used during sampling.
    sigma_data : float
        Expected std of the training data (used in preconditioning).
    rho : float
        Exponent for the Karras sigma schedule (rho=7 in the paper).
    P_mean : float
        Mean of the log-normal distribution for training noise levels.
    P_std : float
        Std of the log-normal distribution for training noise levels.
    S_churn : float
        Stochasticity parameter for stochastic sampling (Alg. 2 in paper).
        Set to 0 for deterministic (DDIM-like) sampling.
    S_tmin : float
        Lower sigma bound for stochastic churn.
    S_tmax : float
        Upper sigma bound for stochastic churn.
    S_noise : float
        Extra noise multiplier used during stochastic churn.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        S_churn: float = 0.0,
        S_tmin: float = 0.05,
        S_tmax: float = 50.0,
        S_noise: float = 1.003,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def sample_sigma(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample noise levels for training using the log-normal distribution.

        σ ~ LogNormal(P_mean, P_std^2)   ↔   ln σ ~ Normal(P_mean, P_std^2)

        Parameters
        ----------
        n : int
            Batch size (number of sigma values to sample).
        device : torch.device, optional
            Target device for the returned tensor.

        Returns
        -------
        torch.Tensor
            Sampled sigma values, shape (n,).
        """
        log_sigma = torch.randn(n, device=device) * self.P_std + self.P_mean
        return log_sigma.exp()

    # ------------------------------------------------------------------
    # Preconditioning coefficients (Karras et al. Table 1)
    # ------------------------------------------------------------------

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Skip-connection scaling coefficient.

        c_skip(σ) = σ_data^2 / (σ^2 + σ_data^2)

        This weights how much of the input is passed directly (skip) vs
        predicted by the network.
        """
        sd2 = self.sigma_data ** 2
        return sd2 / (sigma ** 2 + sd2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Output scaling coefficient.

        c_out(σ) = σ * σ_data / sqrt(σ^2 + σ_data^2)

        Scales the network's raw output before adding to the skip connection.
        """
        sd2 = self.sigma_data ** 2
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + sd2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Input scaling coefficient.

        c_in(σ) = 1 / sqrt(σ^2 + σ_data^2)

        Normalises the noisy input so it has unit variance.
        """
        sd2 = self.sigma_data ** 2
        return 1.0 / torch.sqrt(sigma ** 2 + sd2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Noise conditioning input fed to the network's time-embedding MLP.

        c_noise(σ) = ln(σ) / 4

        Mapped to a compact range to be more network-friendly.
        """
        return sigma.log() / 4.0

    # ------------------------------------------------------------------
    # Loss weighting
    # ------------------------------------------------------------------

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Per-sample loss weight for training.

        λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2

        Derived from the optimal denoiser analysis in the paper.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise levels, arbitrary shape.

        Returns
        -------
        torch.Tensor
            Loss weights, same shape as ``sigma``.
        """
        sd2 = self.sigma_data ** 2
        return (sigma ** 2 + sd2) / ((sigma * self.sigma_data) ** 2)

    # ------------------------------------------------------------------
    # Inference sigma schedule (Karras et al. Eq. 5)
    # ------------------------------------------------------------------

    def get_sampling_sigmas(
        self, steps: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute the deterministic Karras sigma schedule for inference.

        σ_i = (σ_max^{1/ρ} + i/(steps-1) * (σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ
        with σ_{steps} = 0 (clean sample).

        The schedule decreases from σ_max to σ_min and then appends 0.

        Parameters
        ----------
        steps : int
            Number of denoising steps.  The returned tensor has length
            ``steps + 1`` (includes the final σ=0).
        device : torch.device, optional
            Target device.

        Returns
        -------
        torch.Tensor
            Sigma schedule, shape (steps + 1,).
        """
        inv_rho = 1.0 / self.rho
        ramp = torch.linspace(0, 1, steps, device=device)
        min_inv_rho = self.sigma_min ** inv_rho
        max_inv_rho = self.sigma_max ** inv_rho
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        # Append 0 to denote the fully denoised step
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        return sigmas

    # ------------------------------------------------------------------
    # Stochastic sampling helpers
    # ------------------------------------------------------------------

    def get_gamma(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute the gamma for stochastic sampling (Alg. 2 in the paper).

        γ(σ) = min(S_churn / steps, sqrt(2) - 1)  if S_tmin ≤ σ ≤ S_tmax
               0                                    otherwise

        Note: this is a per-element operation on sigma values; 'steps' is
        implicitly controlled through S_churn (set at construction time).

        Parameters
        ----------
        sigma : torch.Tensor
            Current noise level(s).

        Returns
        -------
        torch.Tensor
            Gamma values, same shape as ``sigma``.
        """
        gamma = torch.where(
            (sigma >= self.S_tmin) & (sigma <= self.S_tmax),
            torch.full_like(sigma, min(self.S_churn, math.sqrt(2) - 1)),
            torch.zeros_like(sigma),
        )
        return gamma

    def __repr__(self) -> str:
        return (
            f"EDMScheduler(sigma_min={self.sigma_min}, sigma_max={self.sigma_max}, "
            f"sigma_data={self.sigma_data}, rho={self.rho})"
        )
