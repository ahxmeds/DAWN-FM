"""
Training script for DAW-FM and DAWN-FM on tomography reconstruction task

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

from dawnfm.forward_problems import Tomography
from dawnfm.models import UNetFMG_DE, UNetFMG_DE_NE
from dawnfm.load_datasets import get_organcmnist_dataset, get_organamnist_dataset, get_organsmnist_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train DAW-FM/DAWN-FM for tomography reconstruction')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='organcmnist', 
                        choices=['organcmnist', 'organamnist', 'organsmnist'],
                        help='MedMNIST dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the datasets')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Image size (default: 64 for MedMNIST)')
    
    # Model arguments
    parser.add_argument('--use_noise_embed', action='store_true',
                        help='Use noise embedding (DAWN-FM). If not set, uses only data embedding (DAW-FM)')
    parser.add_argument('--arch', type=int, nargs='+', default=None,
                        help='Network architecture (e.g., 1 16 32 64 128). If not specified, uses [1, 16, 32, 64, 128]')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
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
    parser.add_argument('--num_angles', type=int, default=180,
                        help='Number of projection angles for tomography')
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
    parser.add_argument('--save_every', type=int, default=1,
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
        data: Batch of sinograms [B, 1, num_angles, num_detectors]
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
        args.arch = [1, 16, 32, 64, 128]  # Default for MedMNIST
    
    # Auto-generate experiment name if not specified
    if args.exp_name is None:
        mode = 'dawn-fm' if args.use_noise_embed else 'daw-fm'
        arch_str = 'x'.join(map(str, args.arch))
        args.exp_name = f'tomography_{args.dataset}_{mode}_arch-{arch_str}'
    
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
    
    # Combine train and validation splits for training
    if args.dataset == 'organcmnist':
        train_dataset = get_organcmnist_dataset(split='train', img_size=args.img_size, data_dir=args.data_dir)
        val_dataset = get_organcmnist_dataset(split='val', img_size=args.img_size, data_dir=args.data_dir)
    elif args.dataset == 'organamnist':
        train_dataset = get_organamnist_dataset(split='train', img_size=args.img_size, data_dir=args.data_dir)
        val_dataset = get_organamnist_dataset(split='val', img_size=args.img_size, data_dir=args.data_dir)
    elif args.dataset == 'organsmnist':
        train_dataset = get_organsmnist_dataset(split='train', img_size=args.img_size, data_dir=args.data_dir)
        val_dataset = get_organsmnist_dataset(split='val', img_size=args.img_size, data_dir=args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Combine train and validation datasets
    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset([train_dataset, val_dataset])
    
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
    
    for inputs, labels, image_ids in train_loader:
        x1 = inputs.to(args.device, non_blocking=True)
        x0 = torch.randn_like(x1)  # Gaussian random images
        
        # Antithetic sampling for variance reduction (always enabled)
        x1 = torch.cat((x1, x1), dim=0)
        x0 = torch.cat((x0, -x0), dim=0)
        
        # Generate sinogram (tomography data)
        data = FP(x1)
        
        # Add noise if using DAWN-FM
        if args.use_noise_embed:
            # Sample noise level uniformly from specified range
            percent_noise = torch.rand(x0.shape[0]).type_as(x0)
            percent_noise = percent_noise * (args.noise_range[1] - args.noise_range[0]) + args.noise_range[0]
            
            std = get_gaussian_noise_std(data, percent_noise)
            z = torch.randn_like(data)
            data = data + std[:, None, None, None] * z
        
        # Compute adjoint of data (backprojection)
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
            loss_velocity = torch.mean((vt - ut) ** 2) / torch.mean((ut) ** 2)
            
            # Data consistency loss
            x1_hat = xt + (1 - t.view(xt.shape[0], 1, 1, 1)) * vt
            data_hat = FP(x1_hat)
            loss_data = torch.mean((data_hat - data) ** 2) / torch.mean((data) ** 2)
            
            loss = loss_velocity + loss_data
        
        # Backward pass
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        running_loss_velocity += loss_velocity.item() * inputs.size(0)
        running_loss_data += loss_data.item() * inputs.size(0)
        
        # Update progress bar
        tqdm_epoch.set_description(
            f'Epoch {epoch}, Loss={loss:.4e}, Velocity={loss_velocity:.4e}, Data={loss_data:.4e}'
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
    image_size = args.img_size  # MedMNIST datasets use img_size parameter
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
    
    # Setup forward problem (tomography operator)
    FP = Tomography(dim=image_size, num_angles=args.num_angles, device=args.device)
    
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
