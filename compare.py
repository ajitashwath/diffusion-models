import argparse
import yaml
import os
import time
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from models.unet import UNet, EDMPrecond
from diffusion import LinearScheduler, CosineScheduler, EDMScheduler, ForwardProcess, EDMForwardProcess, ReverseProcess
from sampling import DDPMSampler, DDIMSampler, EDMSampler
from training.datasets import unnormalize
from evaluation.metrics import count_nfe, compute_sample_diversity

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def build_unet(cfg, device):
    return UNet(
        in_channels=cfg['in_channels'],
        image_size=cfg['image_size'],
        base_channels=cfg['base_channels'],
        channel_mults=tuple(cfg['channel_mults']),
        num_res_blocks=cfg['num_res_blocks'],
        attention_resolutions=tuple(cfg['attention_resolutions']),
        num_heads=cfg['num_heads'],
        dropout=cfg['dropout'],
    ).to(device)

def load_state_dict_flexible(model, checkpoint_path, device, is_edm_precond=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Strip any potential 'module.' prefix (from DataParallel, though we don't use it here)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    # Now, handle the discrepancy between EDMPrecond (wrapped) and UNet (unwrapped)
    final_state_dict = {}
    if is_edm_precond:
        # We need keys starting with 'unet.'
        # If clean_state_dict keys already start with 'unet.', load directly
        has_unet_prefix = any(k.startswith('unet.') for k in clean_state_dict.keys())
        if has_unet_prefix:
            final_state_dict = clean_state_dict
        else:
            # Wrap them with 'unet.' prefix
            for k, v in clean_state_dict.items():
                final_state_dict[f'unet.{k}'] = v
    else:
        # We need raw UNet keys (no 'unet.' prefix)
        has_unet_prefix = any(k.startswith('unet.') for k in clean_state_dict.keys())
        if has_unet_prefix:
            # Strip 'unet.' prefix
            for k, v in clean_state_dict.items():
                if k.startswith('unet.'):
                    final_state_dict[k[5:]] = v
                else:
                    final_state_dict[k] = v
        else:
            final_state_dict = clean_state_dict
            
    model.load_state_dict(final_state_dict, strict=True)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='Compare DDPM, DDIM, and EDM samplers')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--n', type=int, default=8, help='Number of samples to generate per sampler')
    parser.add_argument('--ddim-steps', type=int, default=50, help='DDIM sampling steps')
    parser.add_argument('--edm-steps', type=int, default=18, help='EDM sampling steps')
    parser.add_argument('--output', default='comparison.png', help='Path to save comparison grid')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    
    # We will generate comparison samples using the same initial noise for fairness!
    shape = (args.n, cfg['in_channels'], cfg['image_size'], cfg['image_size'])
    torch.manual_seed(42)
    init_noise = torch.randn(shape, device=device)
    
    results = {}
    
    # ------------------ DDPM Sampler ------------------
    print("\n--- Running DDPM Sampler ---")
    try:
        unet = build_unet(cfg, device)
        load_state_dict_flexible(unet, args.checkpoint, device, is_edm_precond=False)
        
        if cfg.get('schedule', 'cosine') == 'cosine':
            scheduler = CosineScheduler(T=cfg['T'])
        else:
            scheduler = LinearScheduler(T=cfg['T'], beta_start=cfg['beta_start'], beta_end=cfg['beta_end'])
            
        forward_process = ForwardProcess(scheduler)
        reverse_process = ReverseProcess(scheduler, predict_type=cfg.get('predict_type', 'eps'))
        sampler = DDPMSampler(unet, scheduler, forward_process, reverse_process, device)
        
        # Override initial noise with our seed-noise
        t0 = time.time()
        # To force initial noise, we patch the sampler's random start
        # The sampler samples starting from pure noise, but we can temporarily mock torch.randn
        orig_randn = torch.randn
        torch.randn = lambda *s, **kw: init_noise.clone()
        
        samples = sampler.sample(shape, progress=True)
        
        torch.randn = orig_randn
        t1 = time.time()
        
        results['ddpm'] = {
            'samples': unnormalize(samples).cpu(),
            'time': t1 - t0,
            'nfe': count_nfe('ddpm', cfg['T']),
            'diversity': compute_sample_diversity(samples).item()
        }
    except Exception as e:
        print(f"Error running DDPM: {e}")
        import traceback
        traceback.print_exc()

    # ------------------ DDIM Sampler ------------------
    print("\n--- Running DDIM Sampler ---")
    try:
        unet = build_unet(cfg, device)
        load_state_dict_flexible(unet, args.checkpoint, device, is_edm_precond=False)
        
        if cfg.get('schedule', 'cosine') == 'cosine':
            scheduler = CosineScheduler(T=cfg['T'])
        else:
            scheduler = LinearScheduler(T=cfg['T'], beta_start=cfg['beta_start'], beta_end=cfg['beta_end'])
            
        forward_process = ForwardProcess(scheduler)
        sampler = DDIMSampler(unet, scheduler, forward_process, device)
        
        t0 = time.time()
        orig_randn = torch.randn
        torch.randn = lambda *s, **kw: init_noise.clone()
        
        samples = sampler.sample(shape, steps=args.ddim_steps, progress=True)
        
        torch.randn = orig_randn
        t1 = time.time()
        
        results['ddim'] = {
            'samples': unnormalize(samples).cpu(),
            'time': t1 - t0,
            'nfe': count_nfe('ddim', args.ddim_steps),
            'diversity': compute_sample_diversity(samples).item()
        }
    except Exception as e:
        print(f"Error running DDIM: {e}")

    # ------------------ EDM Sampler ------------------
    print("\n--- Running EDM Sampler ---")
    try:
        # For EDM, load configs for EDM or adapt from DDPM
        sigma_min = cfg.get('sigma_min', 0.002)
        sigma_max = cfg.get('sigma_max', 80.0)
        sigma_data = cfg.get('sigma_data', 0.5)
        rho = cfg.get('rho', 7.0)
        
        edm_scheduler = EDMScheduler(
            sigma_min=sigma_min, sigma_max=sigma_max,
            sigma_data=sigma_data, rho=rho
        )
        unet = build_unet(cfg, device)
        model = EDMPrecond(unet, edm_scheduler).to(device)
        load_state_dict_flexible(model, args.checkpoint, device, is_edm_precond=True)
        
        sampler = EDMSampler(model, edm_scheduler, device)
        
        t0 = time.time()
        orig_randn = torch.randn
        # For EDM, initial state is noise * sigmas[0]
        # In EDMSampler, it does: x = torch.randn(...) * sigmas[0]. So we mock torch.randn to return init_noise
        torch.randn = lambda *s, **kw: init_noise.clone()
        
        samples = sampler.sample(shape, steps=args.edm_steps, progress=True)
        
        torch.randn = orig_randn
        t1 = time.time()
        
        results['edm'] = {
            'samples': unnormalize(samples).cpu(),
            'time': t1 - t0,
            'nfe': count_nfe('edm', args.edm_steps),
            'diversity': compute_sample_diversity(samples).item()
        }
    except Exception as e:
        print(f"Error running EDM: {e}")
        import traceback
        traceback.print_exc()

    # ------------------ Plot and Compare ------------------
    if not results:
        print("No results to compare.")
        return
        
    print("\n" + "="*60)
    print(f"{'Sampler':<15} | {'Steps':<8} | {'NFE':<6} | {'Time (s)':<10} | {'Diversity (L2)':<12}")
    print("-" * 60)
    for name, res in results.items():
        steps = cfg['T'] if name == 'ddpm' else (args.ddim_steps if name == 'ddim' else args.edm_steps)
        print(f"{name.upper():<15} | {steps:<8} | {res['nfe']:<6} | {res['time']:<10.2f} | {res['diversity']:<12.4f}")
    print("="*60)

    # Plot visual side-by-side grid
    available_samplers = list(results.keys())
    num_cols = len(available_samplers)
    fig, axes = plt.subplots(args.n, num_cols, figsize=(3 * num_cols, 3 * args.n))
    
    # If n=1, axes is 1D, make it 2D
    if args.n == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for col_idx, name in enumerate(available_samplers):
        samples = results[name]['samples']  # (n, C, H, W) in [0, 1]
        nfe = results[name]['nfe']
        
        # Column title on top row
        axes[0, col_idx].set_title(f"{name.upper()}\n(NFE={nfe})", fontsize=14, fontweight='bold')
        
        for row_idx in range(args.n):
            img = samples[row_idx]
            if img.shape[0] == 1: # Greyscale
                img = img.squeeze(0).numpy()
                axes[row_idx, col_idx].imshow(img, cmap='gray')
            else: # RGB
                img = img.permute(1, 2, 0).numpy()
                axes[row_idx, col_idx].imshow(img)
                
            axes[row_idx, col_idx].axis('off')
            
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {args.output}")

if __name__ == '__main__':
    main()
