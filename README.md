# DDPM from Scratch
A from-scratch PyTorch implementation of DDPM, following [Ho et al., 2020](https://arxiv.org/abs/2006.11239). Built as a single Jupyter notebook (`ddpm.ipynb`) that trains a small U-Net to generate Fashion-MNIST images.

## What's implemented
- **Forward (diffusion) process**: Eq. (2) and Eq. (4): closed-form noising of an image to any timestep `t`
- **Simplified training objective**: Eq. (14), Algorithm 1: train a network to predict the noise added to an image
- **Reverse (sampling) process**: Algorithm 2: iteratively denoise from pure Gaussian noise back to an image
- **A small U-Net** with a sinusoidal timestep embedding (Section 4, Appendix B)

**Dataset:** Fashion-MNIST (28x28 grayscale)

### Not implemented
This notebook skips: EMA, learned variances, self-attention blocks, multi-GPU training, and the other scaling tricks from the paper's appendix.

## Notebook walkthrough
1. **Setup**: Imports and device selection (CUDA if available, else CPU)
2. **Data**: Loads Fashion-MNIST, scales pixel values from `[0, 255]` to `[-1, 1]` as in Section 3.3 of the paper
3. **Forward (noising) process**: Implements `q_sample` (Eq. 4) using a linear beta schedule (`β₁=1e-4 → β_T=0.02`, `T=1000`), plus a visualization of an image getting progressively noised to confirm it becomes indistinguishable from pure noise
4. **U-Net**: A compact U-Net (`SimpleUNet`) built from:
   - A sinusoidal timestep embedding (like Transformer positional embeddings)
   - Residual blocks with GroupNorm + SiLU activations
   - A down (28 → 14 → 7) → bottleneck → up (7 → 14 → 28) path with skip connections
   - Nearest-neighbor upsampling instead of transposed convolutions
   - No self-attention (unnecessary at this resolution/channel count)
5. **Training objective**: The simplified loss (Eq. 14): MSE between the true noise and the network's predicted noise
6. **Training loop**: Trains for 15 epochs (configurable) with Adam (`lr = 2e-4`), plotting the loss curve
7. **Sampling**: Implements Algorithm 2, generating new images by iteratively denoising from `x_T ~ N(0, I)` down to `x_0`
8. **Progressive generation**: Visualizes the reverse process at intermediate timesteps, showing the paper's coarse-to-fine generation pattern (Section 4.3, Figure 6)

## Requirements
```
torch
torchvision
matplotlib
numpy
```

A GPU is recommended but not required. Training on Fashion-MNIST at this model size is feasible on CPU, just slower.

## Usage
Open `ddpm.ipynb` and run all cells top to bottom. Fashion-MNIST downloads automatically to `./data` on first run. Key hyperparameters you can tweak:

| Variable | Location | Default | Effect |
|---|---|---|---|
| `T` | Forward process cell | 1000 | Number of diffusion steps |
| `BATCH_SIZE` | Data cell | 128 | Training batch size |
| `EPOCHS` | Training loop | 15 | Number of training epochs |
| `base_ch` | `SimpleUNet` | 32 | U-Net base channel width |

## Ideas for extending this
- Increase `EPOCHS` and/or `base_ch` for noticeably better samples
- Try `σ_t² = β̃_t` instead of `σ_t² = β_t` (Section 3.2 — the paper found both work similarly)
- Predict `μ̃_t` directly instead of `ε` and compare training stability (Table 2's baseline parameterization)
- Move to CIFAR-10 — will likely need a larger U-Net with self-attention at 16x16 resolution
- Reduce `T` (e.g. to 200) to explore the sampling speed/quality tradeoff

## Reference
Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
