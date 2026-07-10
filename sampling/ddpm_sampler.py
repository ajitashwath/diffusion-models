"""
DDPM Sampler
============
Implements ancestral sampling (Algorithm 2 in Ho et al., 2020).

Reference:
    Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    https://arxiv.org/abs/2006.11239
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchvision.utils as vutils
from tqdm import tqdm


class DDPMSampler:
    """
    Ancestral sampler for Denoising Diffusion Probabilistic Models (DDPM).

    Runs the full Markov chain from t = T-1 down to t = 0, calling the reverse
    process at each step to obtain x_{t-1} from x_t.

    Parameters
    ----------
    model : nn.Module
        Noise-prediction network.  Signature: ``model(x, t) -> eps_pred``
        where ``t`` is a long tensor of shape ``(B,)``.
    scheduler : LinearScheduler | CosineScheduler
        Noise scheduler that carries the pre-computed diffusion constants
        (``T``, ``alphas_cumprod``, ``posterior_variance``, …).
    forward_process : ForwardProcess
        Implements ``predict_x0_from_eps`` (not used in DDPM sampling itself,
        but kept for API symmetry and potential diagnostics).
    reverse_process : ReverseProcess
        Implements ``p_sample(model_out, x_t, t, clip_denoised) -> x_{t-1}``.
    device : str | torch.device
        Device on which tensors will be created.  Defaults to ``'cpu'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scheduler,
        forward_process,
        reverse_process,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.forward_process = forward_process
        self.reverse_process = reverse_process
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        clip_denoised: bool = True,
        return_intermediates: bool = False,
        progress: bool = True,
        intermediate_freq: int = 100,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate samples by running the full reverse Markov chain.

        Parameters
        ----------
        shape : tuple
            ``(B, C, H, W)`` – batch size and image dimensions.
        clip_denoised : bool
            If ``True`` the denoised prediction is clamped to ``[-1, 1]``
            inside :py:meth:`~ReverseProcess.p_sample`.
        return_intermediates : bool
            If ``True`` also return a list of intermediate ``x_t`` tensors
            sampled every ``intermediate_freq`` timesteps.
        progress : bool
            Show a ``tqdm`` progress bar over timesteps.
        intermediate_freq : int
            Store an intermediate every this many timesteps
            (only used when ``return_intermediates=True``).

        Returns
        -------
        x : torch.Tensor
            Final denoised samples of shape ``shape``.
        intermediates : list of torch.Tensor
            Present only when ``return_intermediates=True``.
            Each element has the same shape as ``x``.
        """
        B = shape[0]
        T = self.scheduler.T

        # ---- initialise from pure noise --------------------------------
        x = torch.randn(shape, device=self.device)

        intermediates: List[torch.Tensor] = []

        timesteps = list(range(T - 1, -1, -1))  # T-1, T-2, …, 0
        iterator = tqdm(timesteps, desc="DDPM sampling", disable=not progress, dynamic_ncols=True)

        for t in iterator:
            t_batch = torch.full(
                (B,), t, dtype=torch.long, device=self.device
            )

            # Model predicts noise (epsilon)
            model_out = self.model(x, t_batch)

            # Reverse diffusion step: x_t -> x_{t-1}
            x = self.reverse_process.p_sample(
                model_out, x, t_batch, clip_denoised
            )

            # Optionally record intermediate
            if return_intermediates and (t % intermediate_freq == 0):
                intermediates.append(x.cpu().clone())

            if progress:
                iterator.set_postfix(t=t)

        if return_intermediates:
            return x, intermediates
        return x

    # ------------------------------------------------------------------

    @staticmethod
    def save_samples(
        samples: torch.Tensor,
        path: Union[str, Path],
        nrow: Optional[int] = None,
        normalize: bool = True,
        value_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """
        Save a batch of image samples as a single grid PNG.

        Parameters
        ----------
        samples : torch.Tensor
            Shape ``(B, C, H, W)``.  Values are assumed to lie in
            ``value_range`` unless ``normalize=False``.
        path : str | Path
            Destination file path (PNG recommended).
        nrow : int, optional
            Number of images per row in the grid.  Defaults to
            ``ceil(sqrt(B))``.
        normalize : bool
            If ``True``, rescale pixel values to ``[0, 1]`` using
            ``value_range``.
        value_range : tuple of float
            ``(min, max)`` used for normalisation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        B = samples.shape[0]
        if nrow is None:
            import math
            nrow = math.ceil(math.sqrt(B))

        grid = vutils.make_grid(
            samples.cpu(),
            nrow=nrow,
            normalize=normalize,
            value_range=value_range,
        )
        # make_grid returns CHW; save_image expects CHW tensor or list
        vutils.save_image(grid, str(path))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_function_evaluations(self) -> int:
        """NFE for a full DDPM run equals T (one forward pass per step)."""
        return self.scheduler.T

    def __repr__(self) -> str:
        return (
            f"DDPMSampler(T={self.scheduler.T}, device={self.device})"
        )
