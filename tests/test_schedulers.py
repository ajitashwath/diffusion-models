import pytest
import torch
from diffusion.schedulers import LinearScheduler, CosineScheduler, EDMScheduler

def test_linear_scheduler_shape():
    T = 1000
    scheduler = LinearScheduler(T=T)
    assert len(scheduler.betas) == T
    assert len(scheduler.alphas) == T
    assert len(scheduler.alphas_cumprod) == T
    assert len(scheduler.alphas_cumprod_prev) == T
    assert len(scheduler.sqrt_alphas_cumprod) == T
    assert len(scheduler.sqrt_one_minus_alphas_cumprod) == T
    assert len(scheduler.sqrt_recip_alphas_cumprod) == T
    assert len(scheduler.sqrt_recip_alphas_cumprod_m1) == T
    assert len(scheduler.posterior_variance) == T
    assert len(scheduler.posterior_log_variance_clipped) == T
    assert len(scheduler.posterior_mean_coef1) == T
    assert len(scheduler.posterior_mean_coef2) == T

def test_linear_scheduler_alphas_cumprod_decreasing():
    scheduler = LinearScheduler(T=100)
    alphas_cumprod = scheduler.alphas_cumprod
    # Check if strictly decreasing
    for i in range(1, len(alphas_cumprod)):
        assert alphas_cumprod[i] < alphas_cumprod[i-1]

def test_linear_scheduler_betas_range():
    scheduler = LinearScheduler(T=100, beta_start=1e-4, beta_end=0.02)
    assert (scheduler.betas > 0).all()
    assert (scheduler.betas < 1).all()

def test_linear_scheduler_sqrt_values():
    scheduler = LinearScheduler(T=100)
    # Check sqrt_alphas_cumprod^2 + sqrt_one_minus_alphas_cumprod^2 == 1.0 (approx)
    sum_sq = scheduler.sqrt_alphas_cumprod ** 2 + scheduler.sqrt_one_minus_alphas_cumprod ** 2
    assert torch.allclose(sum_sq, torch.ones_like(sum_sq), atol=1e-6)

def test_linear_scheduler_posterior_variance_positive():
    scheduler = LinearScheduler(T=100)
    assert (scheduler.posterior_variance >= 0).all()

def test_cosine_scheduler_shape():
    T = 100
    scheduler = CosineScheduler(T=T)
    assert len(scheduler.betas) == T
    assert len(scheduler.alphas) == T
    assert len(scheduler.alphas_cumprod) == T

def test_cosine_betas_clipped():
    scheduler = CosineScheduler(T=1000)
    assert (scheduler.betas <= 0.999).all()
    assert (scheduler.betas >= 0.0).all()

def test_edm_sample_sigma():
    scheduler = EDMScheduler()
    sigmas = scheduler.sample_sigma(10)
    assert sigmas.shape == (10,)
    assert (sigmas > 0).all()

def test_edm_preconditioning_coefficients():
    scheduler = EDMScheduler(sigma_data=0.5)
    sigma = torch.tensor([0.01, 0.1, 1.0, 10.0, 80.0])
    
    c_skip = scheduler.c_skip(sigma)
    c_out = scheduler.c_out(sigma)
    c_in = scheduler.c_in(sigma)
    
    # Check preconditioning rules (Karras Eq. 8/Table 1 properties)
    # E.g. at very small sigma, c_skip -> 1, c_out -> 0
    assert torch.allclose(c_skip[0], torch.tensor(1.0), atol=1e-2)
    assert torch.allclose(c_out[0], torch.tensor(0.0), atol=1e-2)

def test_edm_sampling_sigmas():
    scheduler = EDMScheduler(sigma_min=0.002, sigma_max=80.0, rho=7.0)
    sigmas = scheduler.get_sampling_sigmas(18)
    assert len(sigmas) == 19  # steps + 1
    assert sigmas[0].item() == pytest.approx(80.0)
    assert sigmas[-1].item() == pytest.approx(0.0)
    # check monotonically decreasing
    for i in range(1, len(sigmas)):
        assert sigmas[i] < sigmas[i-1]
