"""
Training script for DAW-FM and DAWN-FM on deblurring task

This script implements:
- DAW-FM: Data-Aware Flow Matching (data embedding only)
- DAWN-FM: Data-Aware Noise-Embedded Flow Matching (data + noise embedding)

Reference: https://www.aimsciences.org//article/doi/10.3934/fods.2026005
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import time

from dawnfm.forward_problems import blurFFT
from dawnfm.models import UNetFMG_DE, UNetFMG_DE_NE
from dawnfm.load_datasets import get_mnist_dataset, get_cifar10_dataset, get_stl10_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train DAW-FM/DAWN-FM for image deblurring')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'stl10'],
                        help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the datasets')
    
    # Model arguments
    parser.add_argument('--use_noise_embed', action='store_true',
                        help='Use noise embedding (DAWN-FM). If not set, uses only data embedding (DAW-FM)')
    parser.add_argument('--arch', type=int, nargs='+', default=None,
                        help='Network architecture (e.g., 1 16 32). If not specified, auto-determined from dataset')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=0,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Forward problem arguments
    parser.add_argument('--blur_sigma', type=float, nargs=2, default=[3.0, 3.0],
                        help='Blur kernel sigma values [sigma_x, sigma_y]')
    parser.add_argument('--noise_range', type=float, nargs=2, default=[0.0, 0.1],
                        help='Range of noise levels as fraction of data range (only for DAWN-FM)')
    
    # Interpolation arguments
    parser.add_argument('--interpolation_sigma', type=float, default=0.01,
                        help='Sigma for stochastic interpolation in flow matching')
    
    # Saving arguments
    parser.add_argument('--save_dir', type=str, default='./experiments',
                        help='Directory to save logs and models')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name. If not specified, auto-generated')
    parser.add_argument('--save_every', type=int, default=25,
                        help='Save model checkpoint every N epochs')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cpu). If not specified, uses cuda:0 if available')
    
    return parser.parse_args()


def pad_zeros_at_front(num, N):
    return str(num).zfill(N)


def get_gaussian_noise_std(data, percent_noise):
    """
    Calculate noise standard deviation for each image in batch
    
    Args:
        data: Batch of images [B, C, H, W]
        percent_noise: Noise level as fraction of data range for each image [B]
    
    Returns:
        std: Standard deviation for each image [B]
    """
    data_flat = data.view(data.size(0), -1)
    data_max, _ = torch.max(data_flat, dim=1)
    data_min, _ = torch.min(data_flat, dim=1)
    data_range = data_max - data_min
    std = percent_noise * data_range
    return std


def sample_conditional_pt(x0, x1, t, sigma):
    """
    Sample from conditional probability path
    
    Args:
        x0: Source samples (noise)
        x1: Target samples (data)
        t: Time values [0, 1]
        sigma: Diffusion coefficient
    
    Returns:
        xt: Sample at time t
        ut: Conditional velocity
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon, x1 - x0


def setup_experiment(args):
    """Setup experiment directories and configurations"""
    
    # Auto-determine architecture if not specified
    if args.arch is None:
        if args.dataset == 'mnist':
            args.arch = [1, 16, 32]
        else:  # cifar10 or stl10
            args.arch = [3, 16, 32]
    
    # Auto-generate experiment name if not specified
    if args.exp_name is None:
        mode = 'dawn-fm' if args.use_noise_embed else 'daw-fm'
        arch_str = 'x'.join(map(str, args.arch))
        args.exp_name = f'deblurring_{args.dataset}_{mode}_arch-{arch_str}'
    
    # Create directories
    logs_dir = os.path.join(args.save_dir, 'logs', args.exp_name)
    models_dir = os.path.join(args.save_dir, 'models', args.exp_name)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    train_logs_fpath = os.path.join(logs_dir, 'train_logs.csv')
    
    # Setup device
    if args.device is None:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)
    
    return models_dir, train_logs_fpath, args


def get_dataloader(args):
    """Get dataloader for specified dataset"""
    
    if args.dataset == 'mnist':
        dataset = get_mnist_dataset(split='train', data_dir=args.data_dir)
    elif args.dataset == 'cifar10':
        dataset = get_cifar10_dataset(split='train', data_dir=args.data_dir)
    elif args.dataset == 'stl10':
        dataset = get_stl10_dataset(split='train', data_dir=args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=args.num_workers
    )
    
    return train_loader, dataset


def create_model(args, image_size):
    """Create the flow matching model"""
    
    dims = torch.tensor([image_size, image_size])
    
    if args.use_noise_embed:
        model = UNetFMG_DE_NE(arch=args.arch, dims=dims)
    else:
        model = UNetFMG_DE(arch=args.arch, dims=dims)
    
    return model


def train_epoch(model, train_loader, optimizer, scaler, FP, args, epoch, tqdm_epoch):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    running_loss_velocity = 0.0
    running_loss_data = 0.0
    
    for inputs, labels in train_loader:
        x1 = inputs.to(args.device, non_blocking=True)
        x0 = torch.randn_like(x1)  # Gaussian random images
        
        # Antithetic sampling for variance reduction (always enabled)
        x1 = torch.cat((x1, x1), dim=0)
        x0 = torch.cat((x0, -x0), dim=0)
        
        # Generate blurred data
        data = FP(x1)
        
        # Add noise if using DAWN-FM
        if args.use_noise_embed:
            # Sample noise level uniformly from specified range
            percent_noise = torch.rand(x0.shape[0]).type_as(x0)
            percent_noise = percent_noise * (args.noise_range[1] - args.noise_range[0]) + args.noise_range[0]
            
            std = get_gaussian_noise_std(data, percent_noise)
            z = torch.randn_like(data)
            data = data + std[:, None, None, None] * z
        
        # Compute adjoint of data
        ATb = FP.adjoint(data)
        
        # Sample time uniformly
        t = torch.rand(x0.shape[0]).type_as(x0)
        
        # Sample from conditional probability path
        xt, ut = sample_conditional_pt(x0, x1, t, sigma=args.interpolation_sigma)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda' if args.device.type == 'cuda' else 'cpu'):
            if args.use_noise_embed:
                vt = model(xt, t, ATb, std)
            else:
                vt = model(xt, t, ATb)
            
            # Velocity matching loss
            loss_u = torch.mean((vt - ut) ** 2) / torch.mean((ut) ** 2)
            
            # Data consistency loss
            x1_hat = xt + (1 - t.view(xt.shape[0], 1, 1, 1)) * vt
            data_hat = FP(x1_hat)
            loss_d = torch.mean((data_hat - data) ** 2) / torch.mean((data) ** 2)
            
            loss = loss_u + loss_d
        
        # Backward pass
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        running_loss_velocity += loss_u.item() * inputs.size(0)
        running_loss_data += loss_d.item() * inputs.size(0)
        
        # Update progress bar
        tqdm_epoch.set_description(
            f'Epoch {epoch}, Loss={loss:.4e}, Velocity={loss_u:.4e}, Data={loss_d:.4e}'
        )
    
    # Calculate epoch losses
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_loss_velocity = running_loss_velocity / len(train_loader.dataset)
    epoch_loss_data = running_loss_data / len(train_loader.dataset)
    
    return epoch_loss, epoch_loss_velocity, epoch_loss_data


def main():
    args = parse_args()
    
    # Setup experiment
    models_dir, train_logs_fpath, args = setup_experiment(args)
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    for arg in vars(args):
        print(f"{arg:25s}: {getattr(args, arg)}")
    print("=" * 80)
    
    # Get dataloader
    train_loader, dataset = get_dataloader(args)
    image_size = dataset[0][0].shape[-1]
    print(f"Dataset: {args.dataset}, Image size: {image_size}x{image_size}")
    print(f"Number of training samples: {len(dataset)}")
    
    # Create model
    model = create_model(args, image_size)
    model.to(args.device)
    
    N = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {N:,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.lr_min)
    scaler = torch.amp.GradScaler()
    
    # Setup forward problem (blur operator)
    FP = blurFFT(image_size, sigma=args.blur_sigma, device=args.device)
    
    # Training loop
    TrainLossPerEpoch = []
    TrainLossVelocityPerEpoch = []
    TrainLossDataPerEpoch = []
    
    start_time = time.time()
    tqdm_epoch = tqdm(range(args.max_epochs), desc="Training progress")
    
    for epoch in tqdm_epoch:
        epoch_start_time = time.time()
        
        # Train for one epoch
        epoch_loss, epoch_loss_velocity, epoch_loss_data = train_epoch(
            model, train_loader, optimizer, scaler, FP, args, epoch, tqdm_epoch
        )
        
        # Save losses
        TrainLossPerEpoch.append(epoch_loss)
        TrainLossVelocityPerEpoch.append(epoch_loss_velocity)
        TrainLossDataPerEpoch.append(epoch_loss_data)
        
        # Save loss logs
        train_loss_data = np.column_stack((TrainLossPerEpoch, TrainLossVelocityPerEpoch, TrainLossDataPerEpoch))
        df = pd.DataFrame(data=train_loss_data, columns=['Loss', 'LossVelocity', 'LossData'])
        df.to_csv(train_logs_fpath, index=False)
        
        epoch_elapsed_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{args.max_epochs} | '
              f'Loss={epoch_loss:.4e} | Velocity={epoch_loss_velocity:.4e} | Data={epoch_loss_data:.4e} | '
              f'Time: {epoch_elapsed_time/60:.2f} min')
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            model_save_fpath = os.path.join(models_dir, f'model_ep={pad_zeros_at_front(epoch+1, 4)}.pth')
            torch.save(model.state_dict(), model_save_fpath)
    
    print('Training complete!')
    elapsed_time = time.time() - start_time
    print(f'Total time taken: {elapsed_time/(60*60):.2f} hours')


if __name__ == '__main__':
    main()
