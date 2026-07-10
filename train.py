import argparse
import yaml
import os
import torch
import sys

from diffusion import LinearScheduler, CosineScheduler, EDMScheduler, ForwardProcess, EDMForwardProcess, ReverseProcess
from models import UNet, EDMPrecond, EMA
from training import Trainer, EDMTrainer, get_mnist_loaders, get_cifar10_loaders
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
    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--mode', default='ddpm', choices=['ddpm', 'edm'], help='Training mode')
    parser.add_argument('--resume', default=None, help='Checkpoint to resume from')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device(args.device)
    
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)
    
    # Data
    if cfg['dataset'] == 'mnist':
        train_loader, val_loader = get_mnist_loaders(cfg['data_dir'], cfg['batch_size'], cfg['num_workers'], cfg['image_size'])
    else:
        train_loader, val_loader = get_cifar10_loaders(cfg['data_dir'], cfg['batch_size'], cfg['num_workers'], cfg['image_size'])
    
    if args.mode == 'ddpm':
        # Scheduler
        if cfg.get('schedule', 'cosine') == 'cosine':
            scheduler = CosineScheduler(T=cfg['T'])
        else:
            scheduler = LinearScheduler(T=cfg['T'], beta_start=cfg['beta_start'], beta_end=cfg['beta_end'])
        
        forward_process = ForwardProcess(scheduler)
        reverse_process = ReverseProcess(scheduler, predict_type=cfg.get('predict_type', 'eps'))
        
        model = build_model(cfg, device)
        trainer = Trainer(model, scheduler, forward_process, cfg, device)
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Sampler for visualization
        # Note: We can pass DDIMSampler or DDPMSampler. DDIM is faster, let's use DDIM
        sampler = DDIMSampler(model, scheduler, forward_process, device)
        
        trainer.train(train_loader, val_loader, sampler=sampler)
    
    else:  # edm
        edm_scheduler = EDMScheduler(
            sigma_min=cfg['sigma_min'], sigma_max=cfg['sigma_max'],
            sigma_data=cfg['sigma_data'], rho=cfg['rho'],
            P_mean=cfg['P_mean'], P_std=cfg['P_std'],
            S_churn=cfg.get('S_churn', 0.0), S_tmin=cfg.get('S_tmin', 0.05),
            S_tmax=cfg.get('S_tmax', 50.0), S_noise=cfg.get('S_noise', 1.003)
        )
        forward_process = EDMForwardProcess(edm_scheduler)
        unet = build_model(cfg, device)
        model = EDMPrecond(unet, edm_scheduler).to(device)
        trainer = EDMTrainer(model, edm_scheduler, forward_process, cfg, device)
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        sampler = EDMSampler(model, edm_scheduler, device)
        trainer.train(train_loader, val_loader, sampler=sampler)

if __name__ == '__main__':
    main()
