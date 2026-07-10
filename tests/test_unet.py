import pytest
import torch
from models.unet import UNet, EDMPrecond
from diffusion.schedulers import EDMScheduler

def test_mnist_unet_forward():
    unet = UNet(
        in_channels=1,
        image_size=32,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(8,),
        num_heads=2,
        dropout=0.0
    )
    
    x = torch.randn(2, 1, 32, 32)
    t = torch.tensor([0, 500])
    
    out = unet(x, t)
    assert out.shape == (2, 1, 32, 32)

def test_cifar_unet_forward():
    unet = UNet(
        in_channels=3,
        image_size=32,
        base_channels=16,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(8,),
        num_heads=2,
        dropout=0.0
    )
    
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([10, 999])
    
    out = unet(x, t)
    assert out.shape == (2, 3, 32, 32)

def test_unet_parameter_count():
    unet = UNet(
        in_channels=3,
        image_size=32,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(8,),
        num_heads=2,
        dropout=0.0
    )
    params = unet.num_parameters()
    # Should be at least > 10k
    assert params > 10000

def test_edm_precond_forward():
    scheduler = EDMScheduler()
    unet = UNet(
        in_channels=3,
        image_size=32,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(8,),
        num_heads=2,
        dropout=0.0
    )
    precond = EDMPrecond(unet, scheduler)
    
    x = torch.randn(2, 3, 32, 32)
    sigma = torch.tensor([0.1, 1.0])
    
    out = precond(x, sigma)
    assert out.shape == x.shape

def test_edm_precond_zero_sigma_limit():
    scheduler = EDMScheduler(sigma_data=0.5)
    unet = UNet(
        in_channels=3,
        image_size=32,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(8,),
        num_heads=2,
        dropout=0.0
    )
    precond = EDMPrecond(unet, scheduler)
    
    x = torch.randn(2, 3, 32, 32)
    # At extremely small sigma (e.g. 1e-6), c_skip is almost 1.0, c_out is almost 0.0
    # Thus precond(x, sigma) should be almost exactly equal to x (raw identity skip)
    sigma = torch.tensor([1e-6, 1e-6])
    
    out = precond(x, sigma)
    assert torch.allclose(out, x, atol=1e-3)
