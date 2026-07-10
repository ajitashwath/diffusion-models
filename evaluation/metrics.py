"""
Evaluation Metrics
==================
Utility functions for evaluating diffusion model samples.

Functions
---------
count_nfe                : Theoretical NFE for each sampler type.
compute_sample_diversity : Mean pairwise L2 distance between samples.
compute_inception_score  : Inception Score (IS) mean and std over splits.
log_metrics_to_tensorboard : Write scalar metrics to a TensorBoard SummaryWriter.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. NFE counter
# ---------------------------------------------------------------------------

def count_nfe(sampler_type: str, steps: int) -> int:
    """
    Return the theoretical number of neural-network function evaluations (NFE)
    for a given sampler type and number of steps.

    Parameters
    ----------
    sampler_type : str
        One of ``'ddpm'``, ``'ddim'``, ``'edm'``.
        Case-insensitive.
    steps : int
        For DDPM this is T (the full chain length, typically 1000).
        For DDIM this is the number of DDIM steps (e.g., 50).
        For EDM this is the number of Heun steps (e.g., 18).

    Returns
    -------
    nfe : int

    Raises
    ------
    ValueError
        If ``sampler_type`` is not recognised.

    Notes
    -----
    - **DDPM**: one forward pass per timestep → NFE = T.
    - **DDIM**: one forward pass per step → NFE = steps.
    - **EDM (Heun)**: two evaluations per step except the last →
      NFE = 2 · (steps − 1) + 1 = 2 · steps − 1.
    """
    sampler_type = sampler_type.lower().strip()

    if sampler_type == "ddpm":
        return steps                   # NFE = T
    elif sampler_type == "ddim":
        return steps                   # NFE = steps
    elif sampler_type == "edm":
        # Heun: 2 evals per step except the very last step (1 eval)
        return 2 * steps - 1
    else:
        raise ValueError(
            f"Unknown sampler type '{sampler_type}'. "
            "Expected one of: 'ddpm', 'ddim', 'edm'."
        )


# ---------------------------------------------------------------------------
# 2. Sample diversity
# ---------------------------------------------------------------------------

def compute_sample_diversity(samples: torch.Tensor) -> float:
    """
    Estimate sample diversity as the mean pairwise L2 distance.

    Flattens each image to a vector and computes the mean of all
    off-diagonal pairwise Euclidean distances.

    Parameters
    ----------
    samples : torch.Tensor
        Shape ``(N, C, H, W)``, values in ``[-1, 1]``.
        If N > 1000 only the first 1000 samples are used.

    Returns
    -------
    diversity : float
        Mean pairwise L2 distance.

    Notes
    -----
    Memory complexity: O(N²·D) where D = C·H·W.  For N = 1000 and
    D = 3·32·32 ≈ 3072 this is about 9 GB of float32.  We therefore
    use a chunked approach and keep everything in float32.
    """
    N_max = 1000
    if samples.shape[0] > N_max:
        samples = samples[:N_max]

    N = samples.shape[0]
    if N < 2:
        return 0.0

    # Flatten: (N, D)
    flat = samples.float().view(N, -1).cpu()

    # Squared pairwise distances via the identity:
    # ||a - b||² = ||a||² + ||b||² - 2 a·b^T
    sq_norms = (flat ** 2).sum(dim=1, keepdim=True)   # (N, 1)
    # Gram matrix
    gram     = flat @ flat.t()                          # (N, N)
    sq_dists = sq_norms + sq_norms.t() - 2.0 * gram    # (N, N)

    # Clamp numerical negatives to zero, then take sqrt
    sq_dists = sq_dists.clamp(min=0.0)
    dists    = sq_dists.sqrt()                           # (N, N)

    # Collect upper triangle (excluding diagonal)
    idx    = torch.triu_indices(N, N, offset=1)
    upper  = dists[idx[0], idx[1]]                      # (N*(N-1)/2,)

    return float(upper.mean().item())


# ---------------------------------------------------------------------------
# 3. Inception Score
# ---------------------------------------------------------------------------

def compute_inception_score(
    samples: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 50,
    splits: int = 10,
) -> Tuple[float, float]:
    """
    Compute the Inception Score (IS) for a set of generated images.

    IS = exp(E_x [ KL( p(y|x) || p(y) ) ])

    The expectation is over generated images x, y is the InceptionV3
    class prediction (1000 ImageNet classes), and p(y) is the marginal
    class distribution over all images in the split.

    Parameters
    ----------
    samples : torch.Tensor
        ``(N, C, H, W)`` generated images in ``[-1, 1]``.
    device : str | torch.device
    batch_size : int
        Inception forward-pass batch size.
    splits : int
        Number of splits for mean/std estimation.

    Returns
    -------
    is_mean : float
    is_std  : float

    References
    ----------
    Salimans et al., "Improved Techniques for Training GANs", NeurIPS 2016.
    """
    import torchvision.models as tv_models

    device = torch.device(device)

    # Load full InceptionV3 (with softmax for class probabilities)
    try:
        from torchvision.models import Inception_V3_Weights
        inception = tv_models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1
        )
    except (ImportError, AttributeError):
        inception = tv_models.inception_v3(pretrained=True)  # type: ignore[call-arg]

    inception.eval()
    inception.to(device)

    # Normalise: [-1,1] → [0,1], then resize to 299
    def _preprocess(imgs: torch.Tensor) -> torch.Tensor:
        imgs = (imgs.clamp(-1.0, 1.0) + 1.0) / 2.0
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        if imgs.shape[-1] != 299 or imgs.shape[-2] != 299:
            imgs = F.interpolate(
                imgs, size=(299, 299), mode="bilinear", align_corners=False
            )
        return imgs

    # Collect conditional class probabilities p(y|x)
    all_probs = []

    with torch.no_grad():
        N = samples.shape[0]
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            batch = _preprocess(samples[start:end].float()).to(device)
            logits = inception(batch)
            # inception returns InceptionOutputs named tuple during training;
            # in eval mode it returns just the logits tensor.
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)            # (B, 1000)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()     # (N, 1000)

    # Compute IS over splits
    N          = all_probs.shape[0]
    split_size = N // splits
    scores     = []

    for k in range(splits):
        part = all_probs[k * split_size : (k + 1) * split_size]   # (M, 1000)
        p_y  = part.mean(axis=0, keepdims=True)                    # (1, 1000)
        # KL divergence for each image: sum_y p(y|x) log(p(y|x) / p(y))
        kl   = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))  # (M, 1000)
        kl   = kl.sum(axis=1)                                       # (M,)
        scores.append(np.exp(kl.mean()))

    scores    = np.array(scores)
    is_mean   = float(scores.mean())
    is_std    = float(scores.std())
    return is_mean, is_std


# ---------------------------------------------------------------------------
# 4. TensorBoard logging
# ---------------------------------------------------------------------------

def log_metrics_to_tensorboard(
    writer,
    metrics_dict: Dict[str, float],
    step: int,
) -> None:
    """
    Write scalar metrics to a TensorBoard ``SummaryWriter``.

    Parameters
    ----------
    writer : torch.utils.tensorboard.SummaryWriter
        An open TensorBoard writer instance.
    metrics_dict : dict
        Mapping from metric name (str) to scalar value (float).
        Example: ``{'FID': 12.3, 'IS_mean': 8.7, 'diversity': 245.1}``.
    step : int
        Global step or epoch number to associate with these metrics.

    Notes
    -----
    This function is a thin wrapper around ``writer.add_scalar`` so that
    callers do not need to loop explicitly over the metrics dict.  It will
    silently skip non-numeric values.
    """
    for name, value in metrics_dict.items():
        try:
            writer.add_scalar(name, float(value), global_step=step)
        except (TypeError, ValueError):
            # Skip non-numeric values gracefully
            pass
