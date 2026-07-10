"""
DDIM Sampler
============
Implements the Denoising Diffusion Implicit Models (DDIM) sampler described in:

    Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
    https://arxiv.org/abs/2010.02502

The key equation (Eq. 12 of the paper):

    x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred
             + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_pred
             + sigma_t * z

where:
    sigma_t = eta * sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t))
              * sqrt(1 - alpha_bar_t / alpha_bar_{t-1})
    z ~ N(0, I)  (or 0 when sigma_t = 0)

Setting eta = 0 gives the fully deterministic DDIM update.
Setting eta = 1 recovers DDPM (when using all T steps).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm


class DDIMSampler:
    """
    Accelerated deterministic (or semi-stochastic) sampler via DDIM.

    Parameters
    ----------
    model : nn.Module
        Noise-prediction network.  Signature: ``model(x, t) -> eps_pred``
        where ``t`` is a long tensor of shape ``(B,)``.
    scheduler : LinearScheduler | CosineScheduler
        Noise scheduler.  Must expose ``T`` and ``alphas_cumprod`` (a 1-D
        tensor of length T).
    forward_process : ForwardProcess
        Must implement ``predict_x0_from_eps(x_t, t, eps) -> x0``.
    device : str | torch.device
        Computation device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scheduler,
        forward_process,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.forward_process = forward_process
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_subsequence(self, steps: int) -> List[int]:
        """
        Build an integer subsequence of length ``steps`` from T-1 down to 0.

        Uses ``torch.linspace`` as specified, rounding to nearest integer.
        """
        T = self.scheduler.T
        # linspace from T-1 to 0 inclusive, steps points
        seq = torch.linspace(T - 1, 0, steps=steps)
        seq = seq.round().long().tolist()
        return seq  # descending: [T-1, …, 0]

    @staticmethod
    def _alpha_bar_prev(scheduler, t_prev: int) -> torch.Tensor:
        """Return alpha_bar for index t_prev, or 1.0 if t_prev < 0."""
        if t_prev < 0:
            return torch.tensor(1.0, dtype=torch.float32)
        return scheduler.alphas_cumprod[t_prev].float()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples via the DDIM reverse process.

        Parameters
        ----------
        shape : tuple
            ``(B, C, H, W)``.
        steps : int
            Number of denoising steps (``steps << T`` gives speedup).
        eta : float
            Stochasticity coefficient.  ``eta=0`` → deterministic DDIM;
            ``eta=1`` → variance matches DDPM.
        clip_denoised : bool
            Clamp ``x0_pred`` to ``[-1, 1]`` at each step.
        progress : bool
            Show a ``tqdm`` progress bar.

        Returns
        -------
        x : torch.Tensor
            Denoised samples, shape ``shape``.
        """
        B = shape[0]

        # ---- Build time subsequence ------------------------------------
        subseq = self._make_subsequence(steps)  # descending integers

        # ---- Initialise from Gaussian noise ----------------------------
        x = torch.randn(shape, device=self.device)

        # Pairs: (t_cur, t_prev) where t_prev is the *next* (smaller) index
        pairs: List[Tuple[int, int]] = []
        for i in range(len(subseq) - 1):
            pairs.append((subseq[i], subseq[i + 1]))
        # Final step goes to "before index 0"
        pairs.append((subseq[-1], -1))

        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        iterator = tqdm(
            pairs,
            desc="DDIM sampling",
            disable=not progress,
            dynamic_ncols=True,
        )

        for t_cur, t_prev in iterator:
            # ---- Build batch timestep tensor ---------------------------
            t_batch = torch.full(
                (B,), t_cur, dtype=torch.long, device=self.device
            )

            # ---- Predict noise -----------------------------------------
            pred_noise = self.model(x, t_batch)

            # ---- Predict x0 from eps and x_t ---------------------------
            x0_pred = self.forward_process.predict_x0_from_eps(
                x, t_batch, pred_noise
            )
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # ---- Retrieve alpha_bar values -----------------------------
            alpha_bar_cur = alphas_cumprod[t_cur]       # scalar tensor
            alpha_bar_prev = self._alpha_bar_prev(
                self.scheduler, t_prev
            ).to(self.device)                           # scalar tensor

            # ---- DDIM variance (Eq. 16 in the paper) ------------------
            # sigma_t = eta * sqrt( (1-ab_prev)/(1-ab_cur) * (1 - ab_cur/ab_prev) )
            # When t_prev == -1 the fraction is undefined; sigma_t = 0.
            if t_prev >= 0:
                ratio = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_cur)
                inner = ratio * (1.0 - alpha_bar_cur / alpha_bar_prev)
                # Clamp to avoid tiny negative values from numerical noise
                sigma_t = eta * inner.clamp(min=0.0).sqrt()
            else:
                sigma_t = torch.tensor(0.0, device=self.device)

            # ---- Direction pointing to x_t ----------------------------
            # sqrt(1 - alpha_bar_prev - sigma_t^2) * eps_pred
            coef_dir = (
                1.0 - alpha_bar_prev - sigma_t ** 2
            ).clamp(min=0.0).sqrt()
            direction = coef_dir * pred_noise

            # ---- DDIM update (Eq. 12) ----------------------------------
            x = (
                alpha_bar_prev.sqrt() * x0_pred
                + direction
            )

            if sigma_t.item() > 0.0:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise

            if progress:
                iterator.set_postfix(t_cur=t_cur, t_prev=t_prev)

        return x

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_function_evaluations(self) -> int:
        """NFE for DDIM equals ``steps`` (one forward pass per step)."""
        # Populated lazily; expose an informational method instead.
        raise AttributeError(
            "Call sample() first, or use count_nfe('ddim', steps) "
            "from evaluation.metrics."
        )

    def __repr__(self) -> str:
        return (
            f"DDIMSampler(T={self.scheduler.T}, device={self.device})"
        )
