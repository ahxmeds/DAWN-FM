"""
Utility functions for DAWN-FM

This module contains helper functions for visualization, metrics computation,
and other utilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def visualize_reconstruction(original, blurred, reconstructions, save_path=None):
    """
    Visualize original image, blurred version, and reconstructions
    
    Args:
        original: Original image [C, H, W] or [H, W]
        blurred: Blurred image [C, H, W] or [H, W]
        reconstructions: List or array of reconstructions
        save_path: Path to save figure (optional)
    """
    n_recons = min(len(reconstructions), 5)
    fig, axes = plt.subplots(1, n_recons + 2, figsize=(3*(n_recons+2), 3))
    
    # Convert to numpy and handle channels
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(blurred, torch.Tensor):
        blurred = blurred.detach().cpu().numpy()
    
    # Handle single-channel images
    if original.ndim == 3 and original.shape[0] == 1:
        original = original[0]
        blurred = blurred[0]
        reconstructions = [r[0] if r.ndim == 3 else r for r in reconstructions]
    elif original.ndim == 3:  # Multi-channel
        original = original.transpose(1, 2, 0)
        blurred = blurred.transpose(1, 2, 0)
        reconstructions = [r.transpose(1, 2, 0) if r.ndim == 3 else r for r in reconstructions]
    
    # Plot original
    axes[0].imshow(original, cmap='gray' if original.ndim == 2 else None)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Plot blurred
    axes[1].imshow(blurred, cmap='gray' if blurred.ndim == 2 else None)
    axes[1].set_title('Blurred')
    axes[1].axis('off')
    
    # Plot reconstructions
    for i in range(n_recons):
        recon = reconstructions[i]
        if isinstance(recon, torch.Tensor):
            recon = recon.detach().cpu().numpy()
        axes[i+2].imshow(recon, cmap='gray' if recon.ndim == 2 else None)
        axes[i+2].set_title(f'Recon {i+1}')
        axes[i+2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(log_file, save_path=None):
    """
    Plot training curves from log file
    
    Args:
        log_file: Path to CSV file with training logs
        save_path: Path to save figure (optional)
    """
    import pandas as pd
    
    df = pd.read_csv(log_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(df['Loss'], label='Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Velocity loss
    axes[1].plot(df['LossVelocity'], label='Velocity Loss', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Velocity Matching Loss')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Data consistency loss
    axes[2].plot(df['LossData'], label='Data Loss', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Data Consistency Loss')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_statistics(metrics_file):
    """
    Compute and print summary statistics from metrics file
    
    Args:
        metrics_file: Path to CSV file with metrics
    
    Returns:
        dict: Dictionary of summary statistics
    """
    import pandas as pd
    
    df = pd.read_csv(metrics_file)
    
    stats = {
        'MSE': {
            'mean': df['MSE_mean'].mean(),
            'std': df['MSE_mean'].std(),
            'median': df['MSE_mean'].median()
        },
        'PSNR': {
            'mean': df['PSNR_mean'].mean(),
            'std': df['PSNR_mean'].std(),
            'median': df['PSNR_mean'].median()
        },
        'SSIM': {
            'mean': df['SSIM_mean'].mean(),
            'std': df['SSIM_mean'].std(),
            'median': df['SSIM_mean'].median()
        },
        'MISFIT': {
            'mean': df['MISFIT_mean'].mean(),
            'std': df['MISFIT_mean'].std(),
            'median': df['MISFIT_mean'].median()
        }
    }
    
    print("=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    for metric, values in stats.items():
        print(f"\n{metric}:")
        print(f"  Mean   : {values['mean']:.6f}")
        print(f"  Std    : {values['std']:.6f}")
        print(f"  Median : {values['median']:.6f}")
    print("=" * 60)
    
    return stats


def estimate_training_time(dataset_name: str, num_epochs: int, gpu_name: str = 'V100') -> float:
    """
    Estimate training time based on dataset and hardware
    
    Args:
        dataset_name: Name of dataset
        num_epochs: Number of epochs
        gpu_name: GPU model name
    
    Returns:
        Estimated time in hours
    """
    # Approximate time per epoch in minutes (on V100)
    time_per_epoch = {
        'mnist': 0.15,
        'cifar10': 0.35,
        'stl10': 0.60
    }
    
    if dataset_name not in time_per_epoch:
        return None
    
    # GPU speed factors relative to V100
    gpu_factors = {
        'V100': 1.0,
        'A100': 0.6,
        'RTX3090': 0.8,
        'T4': 1.5,
    }
    
    factor = gpu_factors.get(gpu_name, 1.0)
    total_time_minutes = time_per_epoch[dataset_name] * num_epochs * factor
    total_time_hours = total_time_minutes / 60
    
    return total_time_hours


def print_model_summary(model):
    """
    Print model summary including number of parameters
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("Model Summary:")
    print("=" * 60)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)
    
    # Print layer-wise parameters
    print("\nLayer-wise parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:40s} : {param.numel():>10,}")
    print("=" * 60)
