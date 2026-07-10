"""
models/unet.py
==============
U-Net architecture for diffusion models.

This module provides a modern, efficient U-Net implementation suitable for
DDPM, DDIM and EDM-style diffusion models on images.

Architecture overview
---------------------
* **Time embedding**: Sinusoidal position embedding fed through an MLP.
* **Encoder**: A series of ResBlock + optional self-attention blocks,
  interleaved with strided-convolution downsampling.
* **Bottleneck**: ResBlock → AttentionBlock → ResBlock.
* **Decoder**: Mirrored encoder with skip connections, nearest-neighbour
  upsampling, and optional self-attention.
* **Output**: GroupNorm → SiLU → 3×3 convolution projecting back to
  input channels.

References
----------
* Ho et al. (2020) — DDPM: https://arxiv.org/abs/2006.11239
* Nichol & Dhariwal (2021) — Improved DDPM: https://arxiv.org/abs/2102.09672
* Karras et al. (2022) — EDM: https://arxiv.org/abs/2206.00364
"""

import math
from typing import List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: compute GroupNorm groups
# ---------------------------------------------------------------------------

def _gn_groups(channels: int, preferred: int = 32) -> int:
    """
    Find the largest divisor of ``channels`` that is ≤ ``preferred``.

    This prevents GroupNorm from receiving a ``num_groups`` that does not
    divide ``num_channels``, which would raise a RuntimeError.

    Parameters
    ----------
    channels : int
        Number of channels (must be ≥ 1).
    preferred : int
        Preferred number of groups (default: 32).

    Returns
    -------
    int
        A valid group count for ``nn.GroupNorm(groups, channels)``.
    """
    groups = preferred
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return max(1, groups)


# ---------------------------------------------------------------------------
# 1. Sinusoidal Embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embedding for scalar (timestep) inputs.

    Produces a ``(B, dim)`` embedding from a batch of scalar timestep values,
    identical in spirit to the transformer position encoding (Vaswani et al.
    2017) but applied to diffusion timesteps.

    For half-dimension index ``i`` in ``[0, half)`` and half = dim // 2:

        freqs_i = exp(−log(10000) * i / (half − 1))
        emb_i   = sin(t * freqs_i)    for i < half
        emb_i   = cos(t * freqs_{i−half})  for i ≥ half

    Parameters
    ----------
    dim : int
        Total embedding dimensionality.  Must be even.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalEmbedding dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor
            Scalar timestep values, shape (B,).  Can be integer indices or
            continuous floats (e.g. EDM's c_noise output).

        Returns
        -------
        torch.Tensor
            Sinusoidal embeddings, shape (B, dim).
        """
        half = self.dim // 2
        # Frequencies: (half,)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / (half - 1)
        )
        # Outer product: (B, half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# 2. TimeMLP
# ---------------------------------------------------------------------------

class TimeMLP(nn.Module):
    """
    Two-layer MLP that maps a sinusoidal timestep embedding to a richer
    latent representation used to condition each ResBlock via FiLM.

    Architecture:
        SinusoidalEmbedding(in_dim)
        → Linear(in_dim, out_dim * 4)
        → SiLU
        → Linear(out_dim * 4, out_dim)

    Parameters
    ----------
    in_dim : int
        Dimensionality of the raw sinusoidal embedding.
    out_dim : int
        Output dimensionality (typically ``base_channels * 4``).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmbedding(in_dim),
            nn.Linear(in_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor
            Timestep values, shape (B,).

        Returns
        -------
        torch.Tensor
            Time embeddings, shape (B, out_dim).
        """
        return self.net(t)


# ---------------------------------------------------------------------------
# 3. ResBlock
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with FiLM (Feature-wise Linear Modulation) conditioning.

    Architecture:
        input x  →  GroupNorm → SiLU → Conv2d(3×3)
                 →  FiLM(time_emb): scale + shift  [on channels]
                 →  GroupNorm → SiLU → Dropout → Conv2d(3×3)
                 →  + skip_connection(x)

    FiLM projection:
        t_emb  →  Linear(time_emb_dim, out_channels * 2)
    The output is split into (scale, shift) and applied as:
        out = out * (1 + scale) + shift          (after the first conv)

    The skip connection is a 1×1 convolution when ``in_channels ≠ out_channels``
    and an Identity otherwise.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    time_emb_dim : int
        Dimensionality of the time embedding vector.
    dropout : float
        Dropout probability applied before the second convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # --- first conv branch ---
        self.norm1 = nn.GroupNorm(_gn_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # --- FiLM conditioning ---
        # Produces (scale, shift) of shape (B, out_channels) each
        self.time_proj = nn.Linear(time_emb_dim, out_channels * 2)

        # --- second conv branch ---
        self.norm2 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # --- skip connection ---
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input feature map, shape (B, in_channels, H, W).
        t_emb : torch.Tensor
            Time embedding, shape (B, time_emb_dim).

        Returns
        -------
        torch.Tensor
            Output feature map, shape (B, out_channels, H, W).
        """
        # First conv
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM: compute scale & shift from time embedding
        film = self.time_proj(F.silu(t_emb))          # (B, out_channels * 2)
        film = film.unsqueeze(-1).unsqueeze(-1)        # (B, out_channels*2, 1, 1)
        scale, shift = film.chunk(2, dim=1)            # each (B, out_channels, 1, 1)
        h = h * (1.0 + scale) + shift                  # FiLM modulation

        # Second conv
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

        # Residual
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# 4. AttentionBlock
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """
    Self-attention block applied to 2-D feature maps.

    The spatial dimensions (H, W) are flattened into a sequence before
    computing multi-head self-attention, then unflattened back.

    Architecture:
        GroupNorm
        → flatten (B, C, H, W) to (B, H*W, C)
        → MultiheadAttention (batch_first=True)
        → project output: Linear(C, C)
        → unflatten to (B, C, H, W)
        → residual: x + attended

    Parameters
    ----------
    channels : int
        Number of feature channels (= embedding dimension for attention).
    num_heads : int
        Number of attention heads.  ``channels`` must be divisible by
        ``num_heads``.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        # Reduce num_heads if channels can't be divided evenly
        while num_heads > 1 and channels % num_heads != 0:
            num_heads //= 2

        self.norm = nn.GroupNorm(_gn_groups(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Attended feature map, shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        # Normalise and flatten spatial dims
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        # Self-attention (Q = K = V = h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.proj(h)                             # (B, HW, C)

        # Unflatten and residual
        h = h.permute(0, 2, 1).view(B, C, H, W)    # (B, C, H, W)
        return x + h


# ---------------------------------------------------------------------------
# 5. Downsample
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """
    Learned downsampling via strided convolution (factor 2).

    Uses a 3×3 convolution with stride 2 and padding 1 so that the spatial
    dimensions halve exactly: (H, W) → (H//2, W//2).

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# 6. Upsample
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    """
    Nearest-neighbour upsampling (factor 2) followed by a smoothing 3×3 conv.

    The convolution after interpolation reduces aliasing artefacts that can
    appear with pure nearest-neighbour upsampling.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# 7. UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net for diffusion models.

    This architecture closely follows the design used in improved DDPM
    (Nichol & Dhariwal 2021) and consists of:

    * An **encoder** that progressively downsamples the spatial resolution
      while increasing the channel count.
    * A **bottleneck** with residual blocks and global self-attention.
    * A **decoder** that upsamples and merges encoder skip connections.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (e.g. 1 for greyscale, 3 for RGB).
    image_size : int
        Spatial resolution of the input (assumed square).
    base_channels : int
        Number of channels in the first encoder level.
    channel_mults : tuple of int
        Channel multipliers per encoder level.  The channel count at level i
        is ``base_channels * channel_mults[i]``.
    num_res_blocks : int
        Number of residual blocks per encoder / decoder level.
    attention_resolutions : tuple of int
        Spatial resolutions at which to insert self-attention blocks.
    num_heads : int
        Number of attention heads in each AttentionBlock.
    dropout : float
        Dropout probability inside ResBlocks.

    Examples
    --------
    >>> model = UNet(in_channels=3, image_size=32, base_channels=64,
    ...              channel_mults=(1, 2, 4), num_res_blocks=2)
    >>> x = torch.randn(4, 3, 32, 32)
    >>> t = torch.randint(0, 1000, (4,))
    >>> out = model(x, t)
    >>> out.shape
    torch.Size([4, 3, 32, 32])
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (8,),
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size
        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.num_heads = num_heads
        self.dropout = dropout

        # --- Time embedding ---
        time_dim = base_channels * 4
        self.time_mlp = TimeMLP(in_dim=base_channels, out_dim=base_channels * 4)

        # --- Initial convolution ---
        # Maps raw input channels → base_channels
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ----------------------------------------------------------------
        # ENCODER
        # ----------------------------------------------------------------
        self.down_levels = nn.ModuleList()   # one entry (ModuleList) per level
        self.downsamples = nn.ModuleList()   # Downsample or Identity per level

        ch = base_channels       # current channel count
        res = image_size         # current spatial resolution

        # Keep track of output channels at each encoder level for the decoder
        # so we can build correctly-sized skip connections.
        self._enc_out_channels: List[int] = []

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = res in self.attention_resolutions
            is_last = i == len(channel_mults) - 1

            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch, num_heads))
                else:
                    level_blocks.append(nn.Identity())
                ch = out_ch

            self.down_levels.append(level_blocks)
            self._enc_out_channels.append(out_ch)

            # Downsample (or Identity at the last level)
            if not is_last:
                self.downsamples.append(Downsample(ch))
                res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # ----------------------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------------------
        self.mid_res1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_res2 = ResBlock(ch, ch, time_dim, dropout)

        # ----------------------------------------------------------------
        # DECODER
        # ----------------------------------------------------------------
        self.up_levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            # At this level the encoder pushed `out_ch` onto the skip stack
            skip_ch = self._enc_out_channels[i]
            use_attn = res in self.attention_resolutions
            is_first = i == len(channel_mults) - 1  # first decoded level (deepest)

            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                # Each res block receives (ch + skip_ch) because we concatenate
                # the skip from the encoder with the running feature map.
                level_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch, num_heads))
                else:
                    level_blocks.append(nn.Identity())
                ch = out_ch
                skip_ch = out_ch  # subsequent blocks in this level don't need extra skip

            self.up_levels.append(level_blocks)

            # Upsample (or Identity at the shallowest level i==0)
            if i != 0:
                self.upsamplers.append(Upsample(ch))
                res *= 2
            else:
                self.upsamplers.append(nn.Identity())

        # ----------------------------------------------------------------
        # FINAL BLOCK
        # ----------------------------------------------------------------
        # After all decoder levels, concatenate with the init_conv skip
        # (which has `base_channels` channels) before the final output conv.
        self.final_res = ResBlock(ch + base_channels, base_channels, time_dim, dropout)
        self.out_norm = nn.GroupNorm(_gn_groups(base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise convolution and linear weights following common practice."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-initialise the final output convolution so the model starts as
        # a near-identity and training is more stable.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Denoise a noisy image given the timestep (or noise level).

        Parameters
        ----------
        x : torch.Tensor
            Noisy image tensor, shape (B, in_channels, H, W).
        t : torch.Tensor
            Timestep indices or continuous noise levels, shape (B,).

        Returns
        -------
        torch.Tensor
            Model prediction (noise / x0 / velocity depending on training
            parametrisation), shape (B, in_channels, H, W).
        """
        # --- Time embedding ---
        t_emb = self.time_mlp(t)  # (B, time_dim)

        # --- Initial convolution ---
        h = self.init_conv(x)  # (B, base_channels, H, W)

        # Save the initial feature map as the very first skip connection
        # (consumed by final_res at the end of the decoder).
        skips: List[torch.Tensor] = [h]

        # ---- ENCODER ----
        for level_blocks, downsample in zip(self.down_levels, self.downsamples):
            # Process pairs of (ResBlock, AttentionBlock / Identity)
            for idx in range(0, len(level_blocks), 2):
                res_block = level_blocks[idx]
                attn_block = level_blocks[idx + 1]
                h = res_block(h, t_emb)
                h = attn_block(h)
                skips.append(h)
            h = downsample(h)

        # ---- BOTTLENECK ----
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # ---- DECODER ----
        for level_blocks, upsample in zip(self.up_levels, self.upsamplers):
            for idx in range(0, len(level_blocks), 2):
                res_block = level_blocks[idx]
                attn_block = level_blocks[idx + 1]
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = res_block(h, t_emb)
                h = attn_block(h)
            h = upsample(h)

        # ---- FINAL ----
        # Consume the init_conv skip (always the last remaining in the list)
        h = torch.cat([h, skips.pop()], dim=1)
        h = self.final_res(h, t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"UNet(in_channels={self.in_channels}, "
            f"image_size={self.image_size}, "
            f"base_channels={self.base_channels}, "
            f"channel_mults={self.channel_mults}, "
            f"num_res_blocks={self.num_res_blocks}, "
            f"params={self.num_parameters():,})"
        )


# ---------------------------------------------------------------------------
# 8. EDMPrecond
# ---------------------------------------------------------------------------

class EDMPrecond(nn.Module):
    """
    EDM preconditioning wrapper around a UNet.

    Wraps the raw U-Net with the input/output preconditioning described in
    Karras et al. 2022 (Table 1) so that the network always operates on
    roughly unit-variance activations regardless of the noise level σ.

    The preconditioned denoiser is:

        D(x, σ) = c_skip(σ) * x + c_out(σ) * F(c_in(σ) * x, c_noise(σ))

    where F is the underlying U-Net.

    Parameters
    ----------
    unet : UNet
        The underlying U-Net architecture.
    scheduler : EDMScheduler
        EDM scheduler providing the preconditioning coefficient methods:
        ``c_skip``, ``c_out``, ``c_in``, ``c_noise``.
    """

    def __init__(self, unet: "UNet", scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with EDM preconditioning.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input image, shape (B, C, H, W).
        sigma : torch.Tensor
            Per-sample noise standard deviations, shape (B,).

        Returns
        -------
        torch.Tensor
            Denoised image estimate D(x, σ), shape (B, C, H, W).
        """
        sigma_ = sigma.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting

        c_skip = self.scheduler.c_skip(sigma_)    # (B, 1, 1, 1)
        c_out = self.scheduler.c_out(sigma_)      # (B, 1, 1, 1)
        c_in = self.scheduler.c_in(sigma_)        # (B, 1, 1, 1)
        c_noise = self.scheduler.c_noise(sigma)   # (B,) — used as "timestep" input

        # Scale input, run network, scale output, add skip
        F_x = self.unet(c_in * x, c_noise)
        return c_skip * x + c_out * F_x

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return f"EDMPrecond(unet={self.unet!r}, scheduler={self.scheduler!r})"
