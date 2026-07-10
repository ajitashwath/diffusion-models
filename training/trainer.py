import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.ema import EMA
from training.datasets import unnormalize
from torchvision.utils import make_grid

class Trainer:
    """
    DDPM/DDIM training with epsilon prediction.
    """
    def __init__(self, model, scheduler, forward_process, config: dict, device='cpu'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.forward_process = forward_process
        self.config = config
        self.device = device
        
        self.optimizer = Adam(self.model.parameters(), lr=config['lr'])
        self.ema = EMA(self.model, config.get('ema_decay', 0.9999))
        self.writer = SummaryWriter(config['log_dir'])
        
        self.step = 0
        self.epoch = 0
        
    def train_step(self, x_real) -> float:
        x_real = x_real.to(self.device)
        batch_size = x_real.shape[0]
        
        # Sample random t from [0, T) for each sample in batch
        t = torch.randint(0, self.scheduler.T, (batch_size,), device=self.device)
        
        # Forward process: x_t, noise
        x_t, noise = self.forward_process.q_sample(x_real, t)
        
        # Predict noise
        pred_noise = self.model(x_t, t)
        
        # Loss
        loss = F.mse_loss(pred_noise, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.get('grad_clip', None) is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
        self.optimizer.step()
        self.ema.update(self.model)
        
        self.step += 1
        return loss.item()

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {self.epoch+1}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                x_real = batch[0]
            else:
                x_real = batch
                
            loss = self.train_step(x_real)
            total_loss += loss
            
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            if self.step % 50 == 0:
                self.writer.add_scalar('train/loss', loss, self.step)
                
        avg_loss = total_loss / len(loader)
        self.writer.add_scalar('train/epoch_loss', avg_loss, self.epoch)
        return avg_loss

    def train(self, train_loader, val_loader=None, sampler=None):
        total_epochs = self.config['epochs']
        for epoch in range(self.epoch, total_epochs):
            self.epoch = epoch
            avg_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == total_epochs:
                ckpt_path = os.path.join(self.config['checkpoint_dir'], f"model_epoch_{epoch+1}.pt")
                self.save_checkpoint(ckpt_path)
                
            # Sample visualization
            if sampler is not None and (epoch + 1) % self.config['sample_every'] == 0:
                print(f"Generating samples at epoch {epoch+1}...")
                samples = self.generate_samples(sampler, n=self.config.get('sample_batch_size', 16))
                grid = make_grid(unnormalize(samples), nrow=int(math.sqrt(samples.shape[0])))
                self.writer.add_image('samples', grid, epoch + 1)
                
                # Also save sample locally
                sample_dir = os.path.join(self.config['checkpoint_dir'], "samples")
                os.makedirs(sample_dir, exist_ok=True)
                sampler.save_samples(samples, os.path.join(sample_dir, f"sample_epoch_{epoch+1}.png"))
                
    def save_checkpoint(self, path):
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint['epoch'] + 1  # start from next epoch
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path} (Epoch {self.epoch}, Step {self.step})")

    @torch.no_grad()
    def generate_samples(self, sampler, n=16) -> torch.Tensor:
        # Use EMA weights for sampling
        with self.ema.average_parameters(self.model):
            # The sampler expects (B, C, H, W)
            shape = (n, self.config['in_channels'], self.config['image_size'], self.config['image_size'])
            samples = sampler.sample(shape)
            return samples


class EDMTrainer:
    """
    EDM training with Karras 2022 preconditioning.
    """
    def __init__(self, model, edm_scheduler, forward_process, config, device='cpu'):
        self.model = model.to(device)  # model is EDMPrecond
        self.edm_scheduler = edm_scheduler
        self.forward_process = forward_process
        self.config = config
        self.device = device
        
        self.optimizer = Adam(self.model.parameters(), lr=config['lr'])
        # Wrap the preconditioning model parameters in EMA
        self.ema = EMA(self.model, config.get('ema_decay', 0.9999))
        self.writer = SummaryWriter(config['log_dir'])
        
        self.step = 0
        self.epoch = 0
        
    def train_step(self, x_real) -> float:
        x_real = x_real.to(self.device)
        batch_size = x_real.shape[0]
        
        # Sample random sigma level for each element in batch
        sigma = self.edm_scheduler.sample_sigma(batch_size, device=self.device)
        
        # Add noise
        x_noisy, noise = self.forward_process.q_sample(x_real, sigma)
        
        # Predict denoised target (EDMPrecond handles skip connection & scaling)
        denoised = self.model(x_noisy, sigma)
        
        # Weight loss according to sigma
        weight = self.edm_scheduler.loss_weight(sigma).view(-1, 1, 1, 1)
        loss = (weight * (denoised - x_real) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.get('grad_clip', None) is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
        self.optimizer.step()
        self.ema.update(self.model)
        
        self.step += 1
        return loss.item()

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {self.epoch+1}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                x_real = batch[0]
            else:
                x_real = batch
                
            loss = self.train_step(x_real)
            total_loss += loss
            
            pbar.set_postfix({"loss": f"{loss:.6f}"})
            
            if self.step % 50 == 0:
                self.writer.add_scalar('train/loss', loss, self.step)
                
        avg_loss = total_loss / len(loader)
        self.writer.add_scalar('train/epoch_loss', avg_loss, self.epoch)
        return avg_loss

    def train(self, train_loader, val_loader=None, sampler=None):
        total_epochs = self.config['epochs']
        for epoch in range(self.epoch, total_epochs):
            self.epoch = epoch
            avg_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == total_epochs:
                ckpt_path = os.path.join(self.config['checkpoint_dir'], f"model_epoch_{epoch+1}.pt")
                self.save_checkpoint(ckpt_path)
                
            # Sample visualization
            if sampler is not None and (epoch + 1) % self.config['sample_every'] == 0:
                print(f"Generating samples at epoch {epoch+1}...")
                samples = self.generate_samples(sampler, n=self.config.get('sample_batch_size', 16))
                grid = make_grid(unnormalize(samples), nrow=int(math.sqrt(samples.shape[0])))
                self.writer.add_image('samples', grid, epoch + 1)
                
                # Also save sample locally
                sample_dir = os.path.join(self.config['checkpoint_dir'], "samples")
                os.makedirs(sample_dir, exist_ok=True)
                # Since EDMSampler might not have save_samples, we implement/call standard grid saving
                import torchvision
                torchvision.utils.save_image(
                    unnormalize(samples),
                    os.path.join(sample_dir, f"sample_epoch_{epoch+1}.png"),
                    nrow=int(math.sqrt(samples.shape[0]))
                )

    def save_checkpoint(self, path):
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint['epoch'] + 1
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path} (Epoch {self.epoch}, Step {self.step})")

    @torch.no_grad()
    def generate_samples(self, sampler, n=16) -> torch.Tensor:
        with self.ema.average_parameters(self.model):
            shape = (n, self.config['in_channels'], self.config['image_size'], self.config['image_size'])
            # Pass Karras params if required
            samples = sampler.sample(shape)
            return samples
import math
