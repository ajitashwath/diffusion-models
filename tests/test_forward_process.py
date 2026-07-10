import pytest
import torch
from diffusion.schedulers import LinearScheduler, EDMScheduler
from diffusion.forward_process import ForwardProcess, EDMForwardProcess

def test_q_sample_shape():
    scheduler = LinearScheduler(T=1000)
    fp = ForwardProcess(scheduler)
    
    x_start = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    
    x_t, noise = fp.q_sample(x_start, t)
    assert x_t.shape == x_start.shape
    assert noise.shape == x_start.shape

def test_q_sample_noise_mean():
    scheduler = LinearScheduler(T=100)
    fp = ForwardProcess(scheduler)
    
    x_start = torch.zeros(100, 1, 32, 32)
    t = torch.full((100,), 50, dtype=torch.long)
    
    x_t, noise = fp.q_sample(x_start, t)
    # Mean of sampled noise should be close to 0, std close to 1
    assert torch.abs(noise.mean()) < 0.1
    assert torch.abs(noise.std() - 1.0) < 0.1

def test_q_sample_t0():
    # At t=0, beta is very small (1e-4), so alpha_cumprod is close to 1.
    scheduler = LinearScheduler(T=1000, beta_start=1e-4, beta_end=0.02)
    fp = ForwardProcess(scheduler)
    
    x_start = torch.randn(4, 3, 32, 32)
    t = torch.zeros(4, dtype=torch.long)
    
    x_t, _ = fp.q_sample(x_start, t)
    # Check if x_t is close to x_start
    assert torch.allclose(x_t, x_start, atol=1e-1)

def test_q_sample_tT():
    scheduler = LinearScheduler(T=1000)
    fp = ForwardProcess(scheduler)
    
    x_start = torch.randn(4, 3, 32, 32)
    t = torch.full((4,), 999, dtype=torch.long)
    
    x_t, noise = fp.q_sample(x_start, t)
    # At the end, alphas_cumprod is very small, so x_t is almost purely noise
    alpha_bar = scheduler.alphas_cumprod[999]
    expected_x_t = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1.0 - alpha_bar) * noise
    assert torch.allclose(x_t, expected_x_t, atol=1e-5)

def test_roundtrip():
    scheduler = LinearScheduler(T=1000)
    fp = ForwardProcess(scheduler)
    
    x_start = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    
    x_t, noise = fp.q_sample(x_start, t)
    x0_pred = fp.predict_x0_from_eps(x_t, t, noise)
    
    assert torch.allclose(x0_pred, x_start, atol=1e-4)

def test_posterior_mean_shape():
    scheduler = LinearScheduler(T=1000)
    fp = ForwardProcess(scheduler)
    
    x_start = torch.randn(4, 3, 32, 32)
    x_t = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    
    mean, log_var = fp.q_posterior(x_start, x_t, t)
    assert mean.shape == x_start.shape
    assert log_var.shape == (4, 1, 1, 1)

def test_edm_q_sample_shape():
    scheduler = EDMScheduler()
    fp = EDMForwardProcess(scheduler)
    
    x_start = torch.randn(4, 3, 32, 32)
    sigma = torch.tensor([0.1, 1.0, 5.0, 10.0])
    
    x_noisy, noise = fp.q_sample(x_start, sigma)
    assert x_noisy.shape == x_start.shape
    assert noise.shape == x_start.shape

def test_edm_noise_level():
    scheduler = EDMScheduler()
    fp = EDMForwardProcess(scheduler)
    
    x_start = torch.randn(4, 3, 32, 32)
    sigma = torch.zeros(4)
    
    x_noisy, _ = fp.q_sample(x_start, sigma)
    assert torch.allclose(x_noisy, x_start, atol=1e-5)
