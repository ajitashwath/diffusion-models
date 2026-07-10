import argparse
import yaml
import os
import torch
import torchvision
from training.datasets import unnormalize
from models.unet import UNet, EDMPrecond
from diffusion import LinearScheduler, CosineScheduler, EDMScheduler, ForwardProcess, EDMForwardProcess, ReverseProcess
from sampling import DDPMSampler, DDIMSampler, EDMSampler

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def build_model(cfg, device):
    unet = UNet(
        in_channels=cfg['in_channels'],
        image_size=cfg['image_size'],
        base_channels=cfg['base_channels'],
        channel_mults=tuple(cfg['channel_mults']),
        num_res_blocks=cfg['num_res_blocks'],
        attention_resolutions=tuple(cfg['attention_resolutions']),
        num_heads=cfg['num_heads'],
        dropout=cfg['dropout'],
    ).to(device)
    return unet

def main():
    parser = argparse.ArgumentParser(description='Sample from Diffusion Model')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--sampler', required=True, choices=['ddpm', 'ddim', 'edm'], help='Which sampler to use')
    parser.add_argument('--n', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps (for DDIM and EDM)')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta parameter')
    parser.add_argument('--output', default='samples.png', help='Output PNG file path')
    parser.add_argument('--device', default='cpu', help='Device to run sampling on')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    # 1. Build Model
    if args.sampler == 'edm':
        edm_scheduler = EDMScheduler(
            sigma_min=cfg['sigma_min'], sigma_max=cfg['sigma_max'],
            sigma_data=cfg['sigma_data'], rho=cfg['rho']
        )
        unet = build_model(cfg, device)
        model = EDMPrecond(unet, edm_scheduler).to(device)
    else:
        model = build_model(cfg, device)

    # 2. Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Check if checkpoint contains 'model_state_dict' (wrapped) or state_dict directly
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. Build Scheduler/Forward/Reverse Processes
    if args.sampler in ['ddpm', 'ddim']:
        if cfg.get('schedule', 'cosine') == 'cosine':
            scheduler = CosineScheduler(T=cfg['T'])
        else:
            scheduler = LinearScheduler(T=cfg['T'], beta_start=cfg['beta_start'], beta_end=cfg['beta_end'])
        forward_process = ForwardProcess(scheduler)
        reverse_process = ReverseProcess(scheduler, predict_type=cfg.get('predict_type', 'eps'))

    # 4. Run Sampler
    shape = (args.n, cfg['in_channels'], cfg['image_size'], cfg['image_size'])
    print(f"Sampling {args.n} images using {args.sampler.upper()}...")
    
    if args.sampler == 'ddpm':
        sampler = DDPMSampler(model, scheduler, forward_process, reverse_process, device)
        samples = sampler.sample(shape, clip_denoised=True)
    elif args.sampler == 'ddim':
        sampler = DDIMSampler(model, scheduler, forward_process, device)
        samples = sampler.sample(shape, steps=args.steps, eta=args.eta, clip_denoised=True)
    else:  # edm
        sampler = EDMSampler(model, edm_scheduler, device)
        samples = sampler.sample(shape, steps=args.steps, clip_denoised=True)

    # 5. Save grid
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    import math
    grid = torchvision.utils.make_grid(unnormalize(samples), nrow=int(math.sqrt(args.n)))
    torchvision.utils.save_image(grid, args.output)
    print(f"Saved samples to {args.output}")

if __name__ == '__main__':
    main()
