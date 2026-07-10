"""
FID (Fréchet Inception Distance) Computation
=============================================
Implements the Fréchet Inception Distance metric introduced in:

    Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge
    to a Local Nash Equilibrium", NeurIPS 2017.
    https://arxiv.org/abs/1706.08500

The metric measures the distance between the distribution of generated
images and the distribution of real images in the feature space of an
InceptionV3 network (pool3 layer → 2048-dimensional features).

FID formula:
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 * sqrt(Σ_r · Σ_g))
"""

from __future__ import annotations

from typing import Tuple, Union, Iterator

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor
# ---------------------------------------------------------------------------

class _InceptionFeatureExtractor(nn.Module):
    """
    Truncated InceptionV3 that returns 2048-dim pool3 features.

    The original InceptionV3 is sliced so that only the layers up to and
    including the global average pooling (pool3) are retained.  The FC
    classification head and the auxiliary branch are removed.

    Input requirement: images of any spatial size; they are bilinearly
    resized to 299×299 before being fed into the network.
    """

    def __init__(self, model: tv_models.Inception3) -> None:
        super().__init__()

        # Keep all feature layers up to (and including) the adaptive pool
        self.Conv2d_1a_3x3      = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3      = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3      = model.Conv2d_2b_3x3
        self.maxpool1            = model.maxpool1
        self.Conv2d_3b_1x1      = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3      = model.Conv2d_4a_3x3
        self.maxpool2            = model.maxpool2
        self.Mixed_5b           = model.Mixed_5b
        self.Mixed_5c           = model.Mixed_5c
        self.Mixed_5d           = model.Mixed_5d
        self.Mixed_6a           = model.Mixed_6a
        self.Mixed_6b           = model.Mixed_6b
        self.Mixed_6c           = model.Mixed_6c
        self.Mixed_6d           = model.Mixed_6d
        self.Mixed_6e           = model.Mixed_6e
        self.Mixed_7a           = model.Mixed_7a
        self.Mixed_7b           = model.Mixed_7b
        self.Mixed_7c           = model.Mixed_7c
        self.avgpool            = model.avgpool  # AdaptiveAvgPool2d(1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images, shape ``(B, 3, H, W)``, values in ``[0, 1]``.
            Spatial size need not be 299×299; resizing is applied here.

        Returns
        -------
        features : torch.Tensor
            Shape ``(B, 2048)``.
        """
        # Resize to InceptionV3 expected input size
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(
                x, size=(299, 299), mode="bilinear", align_corners=False
            )

        # Inception normalisation: centre to [-1, 1]  (already done if
        # images were in [0,1]; we keep the forward consistent).
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)               # (B, 2048, 1, 1)
        x = x.flatten(1)                  # (B, 2048)
        return x


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_inception_model(device: Union[str, torch.device]) -> _InceptionFeatureExtractor:
    """
    Load an InceptionV3 model truncated at the pool3 layer.

    Parameters
    ----------
    device : str | torch.device

    Returns
    -------
    model : _InceptionFeatureExtractor
        In eval mode, on ``device``, outputs ``(B, 2048)`` features.
    """
    device = torch.device(device)

    # weights= argument preferred in newer torchvision; fall back to
    # pretrained=True for older versions.
    try:
        from torchvision.models import Inception_V3_Weights
        inception = tv_models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1
        )
    except (ImportError, AttributeError):
        inception = tv_models.inception_v3(pretrained=True)  # type: ignore[call-arg]

    model = _InceptionFeatureExtractor(inception)
    model.eval()
    model.to(device)
    return model


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Normalise images from the diffusion model's ``[-1, 1]`` range into the
    ``[0, 1]`` range expected by the InceptionV3 feature extractor.

    Parameters
    ----------
    images : torch.Tensor
        ``(B, C, H, W)`` with values in ``[-1, 1]``.

    Returns
    -------
    torch.Tensor
        ``(B, C, H, W)`` with values in ``[0, 1]``.
    """
    return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def _ensure_3channel(images: torch.Tensor) -> torch.Tensor:
    """Repeat grayscale channels to get 3-channel images."""
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.shape[1] != 3:
        raise ValueError(
            f"Expected 1 or 3 channels, got {images.shape[1]}."
        )
    return images


@torch.no_grad()
def extract_features(
    images: torch.Tensor,
    model: _InceptionFeatureExtractor,
    device: Union[str, torch.device],
    batch_size: int = 50,
) -> np.ndarray:
    """
    Extract InceptionV3 pool3 features from a tensor of images.

    Parameters
    ----------
    images : torch.Tensor
        ``(N, C, H, W)`` with values in ``[-1, 1]``.
    model : _InceptionFeatureExtractor
        Pre-loaded feature extractor (e.g. from :func:`get_inception_model`).
    device : str | torch.device
    batch_size : int
        Number of images processed per forward pass.

    Returns
    -------
    features : np.ndarray
        Shape ``(N, 2048)``.
    """
    device = torch.device(device)
    model.eval()

    images = _normalize_images(images)
    images = _ensure_3channel(images)

    all_features = []
    N = images.shape[0]

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        batch = images[start:end].to(device)
        feats = model(batch)                    # (B, 2048)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)  # (N, 2048)


@torch.no_grad()
def extract_features_from_loader(
    loader: DataLoader,
    model: _InceptionFeatureExtractor,
    device: Union[str, torch.device],
) -> np.ndarray:
    """
    Extract features when images are provided via a DataLoader.

    The loader is expected to yield ``(images, labels)`` or just ``images``.
    Images should be in ``[-1, 1]`` range.
    """
    device = torch.device(device)
    model.eval()
    all_features = []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = _normalize_images(images.float())
        images = _ensure_3channel(images).to(device)
        feats  = model(images)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_statistics(
    images_or_loader: Union[torch.Tensor, DataLoader],
    model: _InceptionFeatureExtractor,
    device: Union[str, torch.device],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of InceptionV3 features.

    Parameters
    ----------
    images_or_loader : torch.Tensor | DataLoader
        Either a tensor ``(N, C, H, W)`` in ``[-1, 1]``, or a DataLoader
        yielding ``(images, labels)`` / ``images`` batches.
    model : _InceptionFeatureExtractor
    device : str | torch.device

    Returns
    -------
    mu : np.ndarray   shape ``(2048,)``
    sigma : np.ndarray  shape ``(2048, 2048)``
    """
    if isinstance(images_or_loader, torch.Tensor):
        features = extract_features(images_or_loader, model, device)
    else:
        features = extract_features_from_loader(images_or_loader, model, device)

    mu    = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Fréchet Inception Distance between two Gaussian distributions.

    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2 * sqrtm(Σ₁ · Σ₂))

    Parameters
    ----------
    mu1, mu2 : np.ndarray  ``(2048,)``
        Feature distribution means.
    sigma1, sigma2 : np.ndarray  ``(2048, 2048)``
        Feature distribution covariances.
    eps : float
        Small value added to the diagonal of the product matrix to stabilise
        the matrix square root computation.

    Returns
    -------
    fid : float
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors must have the same shape."
    assert sigma1.shape == sigma2.shape, "Covariance matrices must have the same shape."

    diff = mu1 - mu2

    # Product of covariance matrices
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Handle numerical issues: if covmean has imaginary parts due to
    # floating-point errors, discard the imaginary component.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            # Significant imaginary part – add eps to diagonal and retry
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

        covmean = covmean.real

    # Tr(Σ₁ + Σ₂ - 2·sqrtm(Σ₁·Σ₂))
    tr_covmean = np.trace(covmean)

    fid = float(
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * tr_covmean
    )
    return fid


def compute_fid_from_samples(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 50,
) -> float:
    """
    Convenience wrapper: compute FID directly from image tensors.

    Loads InceptionV3, extracts statistics for both sets, and returns FID.

    Parameters
    ----------
    real_images : torch.Tensor
        ``(N, C, H, W)`` real images in ``[-1, 1]``.
    fake_images : torch.Tensor
        ``(M, C, H, W)`` generated images in ``[-1, 1]``.
    device : str | torch.device
    batch_size : int
        Inception forward-pass batch size.

    Returns
    -------
    fid : float
    """
    inception = get_inception_model(device)

    mu_real, sigma_real = compute_statistics(real_images, inception, device)
    mu_fake, sigma_fake = compute_statistics(fake_images, inception, device)

    return compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)
