"""
Inference script for DAW-FM and DAWN-FM on tomography reconstruction task

This script performs inference and computes metrics (PSNR, SSIM, MSE, MISFIT)
for the trained models.

Reference: https://www.aimsciences.org//article/doi/10.3934/fods.2026005
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import time
from glob import glob
from joblib import Parallel, delayed
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import torch.multiprocessing as mp

from dawnfm.forward_problems import Tomography
from dawnfm.models import UNetFMG_DE, UNetFMG_DE_NE, odeSol_data, odeSol_data_noise
from dawnfm.load_datasets import (get_organcmnist_dataset, get_organamnist_dataset, 
                                  get_organsmnist_dataset, DatasetWithImageID)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for DAW-FM/DAWN-FM on tomography reconstruction')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='organcmnist', 
                        choices=['organcmnist', 'organamnist', 'organsmnist'],
                        help='MedMNIST dataset to use for inference')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the datasets')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Image size (default: 64 for MedMNIST)')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--use_noise_embed', action='store_true',
                        help='Use noise embedding (DAWN-FM). Must match training configuration')
    parser.add_argument('--arch', type=int, nargs='+', default=None,
                        help='Network architecture. If not specified, uses [1, 16, 32, 64, 128]')
    
    # Forward problem arguments
    parser.add_argument('--num_angles', type=int, default=180,
                        help='Number of projection angles for tomography')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='Noise level as fraction of data range')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (recommended: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--nsteps', type=int, default=100,
                        help='Number of ODE solver steps')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of reconstructions per image')
    
    # Saving arguments
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save generated images and metrics')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name. If not specified, auto-generated')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cpu). Ignored if --gpus is specified')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='GPU IDs to use for parallel inference (e.g., 0 1 2). If specified, dataset is split across GPUs')
    
    # Parallel processing
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Number of parallel jobs for metric computation')
    
    return parser.parse_args()


def pad_zeros_at_front(num, N):
    return str(num).zfill(N)


def get_gaussian_noise_std(data, percent_noise):
    """Calculate noise standard deviation for each image in batch"""
    data_flat = data.view(data.size(0), -1)
    data_max, _ = torch.max(data_flat, dim=1)
    data_min, _ = torch.min(data_flat, dim=1)
    data_range = data_max - data_min
    std = percent_noise * data_range
    return std


def save_images(image, data, recon_images, save_dir, image_id, dataset_name):
    """Save original image, sinogram, and reconstructions"""
    save_fpath = os.path.join(save_dir, f"{image_id}.npy")
    image = image.detach().cpu().numpy()  # [1, H, W]
    data = data.detach().cpu().numpy()  # [1, num_angles, num_detectors]
    recon_images = recon_images.detach().cpu().numpy()  # [num_runs, 1, H, W]
    
    # Un-normalize images based on dataset normalization parameters
    # MedMNIST datasets use Normalize((0.5,), (0.5,))
    mean = np.array([0.5])
    std = np.array([0.5])
    
    # Un-normalize: image_unnorm = image * std + mean
    mean = mean.reshape(1, 1, 1)
    std = std.reshape(1, 1, 1)
    
    image = image * std + mean
    # Note: data is sinogram, not image, so we don't un-normalize it the same way
    # We'll just clip it
    recon_images = recon_images * std + mean
    
    # Clip to [0, 1] range
    image = np.clip(image, 0, 1)
    recon_images = np.clip(recon_images, 0, 1)
    
    # Convert single-channel [1, H, W] to [H, W]
    image = image[0]  # [H, W]
    recon_images = recon_images[:, 0, :, :]  # [num_runs, H, W]
    
    # Stack: [original, recon_1, recon_2, ..., recon_N]
    # Note: For tomography, we don't save the sinogram data as it has different dimensions
    image_stack = np.zeros((1 + recon_images.shape[0], *image.shape))
    image_stack[0, ...] = image
    image_stack[1:, ...] = recon_images
    
    np.save(save_fpath, image_stack)


def generate_images_from_noise_and_data(model, image_size, ATb, sigma, nsteps, num_runs, device, use_noise_embed):
    """Generate multiple reconstructions from noise and data"""
    model.eval()
    batch_size = ATb.shape[0]
    num_channels = ATb.shape[1]
    xf_allruns = torch.zeros(num_runs, batch_size, num_channels, image_size, image_size)
    
    for i in range(num_runs):
        x0 = torch.randn(batch_size, num_channels, image_size, image_size).to(device)
        
        if use_noise_embed:
            traj = odeSol_data_noise(x0, ATb, sigma, model, nsteps=nsteps)
        else:
            traj = odeSol_data(x0, ATb, model, nsteps=nsteps)
        
        xf = traj[-1]
        xf_allruns[i, :, :, :, :] = xf
    
    return xf_allruns


def generate_and_save_one_batch(model, image_size, inputs, data, ATb, sigma, nsteps, num_runs, 
                                 save_dir, image_ids, device, use_noise_embed, dataset_name):
    """Generate and save reconstructions for one batch"""
    xf = generate_images_from_noise_and_data(
        model, image_size, ATb, sigma, nsteps, num_runs, device, use_noise_embed
    )
    
    for i in range(xf.shape[1]):  # Iterate over batch
        save_images(inputs[i], data[i], xf[:, i, ...], save_dir, image_ids[i], dataset_name)


def compute_misfit(FP, image, recon_image, device):
    """Compute data misfit between original and reconstructed image"""
    # Images are in [H, W] format for grayscale
    image_chw = np.expand_dims(image, axis=0)  # [1, H, W]
    recon_chw = np.expand_dims(recon_image, axis=0)  # [1, H, W]
    
    data = FP(torch.tensor(image_chw).unsqueeze(0).float().to(device))  # [1, 1, num_angles, num_detectors]
    recon_data = FP(torch.tensor(recon_chw).unsqueeze(0).float().to(device))  # [1, 1, num_angles, num_detectors]
    misfit = (data[0] - recon_data[0]).norm() / data[0].norm()  # Remove batch dimension
    return misfit.item()


def process_metrics(i, path, FP, device, num_runs):
    """Compute metrics for one image"""
    image_fname = os.path.basename(path)[:-4]
    image_stack = np.load(path)
    image = image_stack[0]
    recon_images = image_stack[1:]  # Note: different from deblurring (no data slice)
    
    mse_runs = np.zeros(num_runs)
    psnr_runs = np.zeros(num_runs)
    ssim_runs = np.zeros(num_runs)
    misfit_runs = np.zeros(num_runs)
    
    # Compute metrics for each reconstruction
    for j, recon_image in enumerate(recon_images):
        mse_runs[j] = np.mean((image - recon_image) ** 2)
        psnr_runs[j] = PSNR(image, recon_image, data_range=image.max() - image.min())
        ssim_runs[j] = SSIM(image, recon_image, data_range=image.max() - image.min())
        misfit_runs[j] = compute_misfit(FP, image, recon_image, device)
    
    # Compute metrics for mean reconstruction
    recon_image_mean = np.mean(recon_images, axis=0)
    mse_mean = np.mean((image - recon_image_mean) ** 2)
    psnr_mean = PSNR(image, recon_image_mean, data_range=image.max() - image.min())
    ssim_mean = SSIM(image, recon_image_mean, data_range=image.max() - image.min())
    misfit_mean = compute_misfit(FP, image, recon_image_mean, device)
    
    return (image_fname, mse_runs, psnr_runs, ssim_runs, misfit_runs, 
            mse_mean, psnr_mean, ssim_mean, misfit_mean)


def setup_experiment(args):
    """Setup experiment directories and configurations"""
    
    # Auto-determine architecture if not specified
    if args.arch is None:
        args.arch = [1, 16, 32, 64, 128]  # Default for MedMNIST
    
    # Auto-generate experiment name if not specified
    if args.exp_name is None:
        mode = 'dawn-fm' if args.use_noise_embed else 'daw-fm'
        noise_str = f'noise{int(args.noise_level*100):02d}'
        args.exp_name = f'tomography_{args.dataset}_{mode}_{noise_str}'
    
    # Create directories
    generated_images_dir = os.path.join(args.save_dir, 'generated_images', args.exp_name)
    metrics_dir = os.path.join(args.save_dir, 'metrics', args.exp_name)
    os.makedirs(generated_images_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Setup device
    if args.device is None:
        # If using multi-GPU, device will be set per-worker; use first GPU for temp operations
        if args.gpus is not None and len(args.gpus) > 0:
            args.device = torch.device(f'cuda:{args.gpus[0]}')
        else:
            args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)
    
    return generated_images_dir, metrics_dir, args


def get_dataloader(args):
    """Get dataloader for specified dataset"""
    
    if args.dataset == 'organcmnist':
        dataset = get_organcmnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
    elif args.dataset == 'organamnist':
        dataset = get_organamnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
    elif args.dataset == 'organsmnist':
        dataset = get_organSmnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    return test_loader, dataset


def load_model(args, image_size, device=None):
    """Load trained model"""
    if device is None:
        device = args.device
    
    dims = torch.tensor([image_size, image_size])
    
    if args.use_noise_embed:
        model = UNetFMG_DE_NE(arch=args.arch, dims=dims)
    else:
        model = UNetFMG_DE(arch=args.arch, dims=dims)
    
    # Load weights
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def inference_worker(gpu_id, args, dataset_indices, image_size, generated_images_dir, queue):
    """Worker function for multi-GPU inference"""
    try:
        print(f"[GPU {gpu_id}] Starting worker process...", flush=True)
        
        # Set this GPU as the current CUDA device for this process
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[GPU {gpu_id}] Set CUDA device to {gpu_id}", flush=True)
        
        # Load model on this GPU
        print(f"[GPU {gpu_id}] Loading model...", flush=True)
        model = load_model(args, image_size, device=device)
        
        # Setup forward problem
        print(f"[GPU {gpu_id}] Setting up tomography operator...", flush=True)
        FP = Tomography(dim=image_size, num_angles=args.num_angles, device=device)
        
        # Get test dataset with only assigned indices
        print(f"[GPU {gpu_id}] Loading dataset...", flush=True)
        if args.dataset == 'organcmnist':
            full_dataset = get_organcmnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
        elif args.dataset == 'organamnist':
            full_dataset = get_organamnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
        elif args.dataset == 'organsmnist':
            full_dataset = get_organSmnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
        
        subset_dataset = Subset(full_dataset, dataset_indices)
        
        test_loader = DataLoader(
            subset_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,  # Disable pin_memory for multi-GPU to avoid conflicts
            num_workers=0  # Use main process for data loading in multi-GPU mode
        )
        
        print(f"[GPU {gpu_id}] Processing {len(dataset_indices)} images...", flush=True)
        # Process this GPU's batch of images
        local_count = 0
        batch_count = 0
        for inputs, labels, image_ids in test_loader:
            x1 = inputs.to(device, non_blocking=True)
            
            # Generate sinogram with noise
            percent_noise = args.noise_level * torch.ones(x1.shape[0]).type_as(x1)
            data = FP(x1)
            
            if args.use_noise_embed:
                std = get_gaussian_noise_std(data, percent_noise)
                z = torch.randn_like(data).to(device)
                data = data + std[:, None, None, None] * z
            else:
                std = torch.zeros(x1.shape[0]).to(device)
            
            ATb = FP.adjoint(data)
            
            # Generate and save reconstructions
            with torch.no_grad():
                generate_and_save_one_batch(
                    model, image_size, x1, data, ATb, std, args.nsteps, args.num_runs,
                    generated_images_dir, image_ids, device, args.use_noise_embed, args.dataset
                )
            
            local_count += x1.shape[0]
            batch_count += 1
            
            # Send periodic progress updates (every 10 images)
            if local_count % 10 == 0:
                queue.put(('progress', gpu_id, local_count))
        
        # Report completion
        print(f"[GPU {gpu_id}] Completed: processed {local_count} images", flush=True)
        queue.put(('done', gpu_id, local_count))
    
    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        queue.put(('error', gpu_id, str(e)))


def run_multi_gpu_inference(args, dataset, image_size, generated_images_dir):
    """Run inference across multiple GPUs in parallel"""
    num_gpus = len(args.gpus)
    total_samples = len(dataset)
    
    # Check GPU availability
    print(f"\nGPU Status:")
    print(f"  Total GPUs available: {torch.cuda.device_count()}")
    for gpu_id in args.gpus:
        if gpu_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(gpu_id)
            mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            print(f"  GPU {gpu_id}: {props.name}, Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        else:
            print(f"  WARNING: GPU {gpu_id} not available!")
    
    # Pre-load dataset in main process to avoid conflicts
    print("\nPre-loading dataset in main process...")
    if args.dataset == 'organcmnist':
        _ = get_organcmnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
    elif args.dataset == 'organamnist':
        _ = get_organamnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
    elif args.dataset == 'organsmnist':
        _ = get_organSmnist_dataset(split='test', img_size=args.img_size, data_dir=args.data_dir)
    print("Dataset pre-loaded successfully")
    
    # Split dataset indices across GPUs
    indices_per_gpu = []
    samples_per_gpu = total_samples // num_gpus
    
    for i, gpu_id in enumerate(args.gpus):
        start_idx = i * samples_per_gpu
        if i == num_gpus - 1:  # Last GPU gets remaining samples
            end_idx = total_samples
        else:
            end_idx = (i + 1) * samples_per_gpu
        
        indices_per_gpu.append(list(range(start_idx, end_idx)))
    
    print(f"\nDistributing {total_samples} samples across {num_gpus} GPUs:")
    for i, gpu_id in enumerate(args.gpus):
        print(f"  GPU {gpu_id}: {len(indices_per_gpu[i])} samples (indices {indices_per_gpu[i][0]}-{indices_per_gpu[i][-1]})")
    
    # Create queue for collecting results
    queue = mp.Queue()
    
    # Spawn processes for each GPU
    processes = []
    print("\nSpawning worker processes...")
    for i, gpu_id in enumerate(args.gpus):
        p = mp.Process(
            target=inference_worker,
            args=(gpu_id, args, indices_per_gpu[i], image_size, generated_images_dir, queue)
        )
        p.start()
        processes.append(p)
        print(f"  Spawned worker for GPU {gpu_id} (PID: {p.pid})")
    
    # Monitor progress
    completed_gpus = 0
    total_processed = 0
    gpu_progress = {gpu_id: 0 for gpu_id in args.gpus}
    pbar = tqdm(total=total_samples, desc="Processing images (multi-GPU)")
    
    while completed_gpus < num_gpus:
        msg = queue.get()
        
        if msg[0] == 'progress':
            # Periodic progress update
            _, gpu_id, count = msg
            old_count = gpu_progress[gpu_id]
            gpu_progress[gpu_id] = count
            pbar.update(count - old_count)
            
        elif msg[0] == 'done':
            # GPU completed
            _, gpu_id, count = msg
            old_count = gpu_progress[gpu_id]
            gpu_progress[gpu_id] = count
            pbar.update(count - old_count)
            completed_gpus += 1
            total_processed += count
            print(f"\n  GPU {gpu_id} completed: {count} samples processed")
        
        elif msg[0] == 'error':
            # GPU encountered an error
            _, gpu_id, error_msg = msg
            print(f"\n  ERROR on GPU {gpu_id}: {error_msg}", flush=True)
            completed_gpus += 1
    
    pbar.close()
    
    # Wait for all processes to finish
    print("\nWaiting for all processes to terminate...")
    for p in processes:
        p.join()
    
    print(f"\nAll GPUs completed. Total samples processed: {total_processed}")


def main():
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    args = parse_args()
    
    # Setup experiment
    generated_images_dir, metrics_dir, args = setup_experiment(args)
    
    # Print configuration
    print("=" * 80)
    print("Inference Configuration:")
    print("=" * 80)
    for arg in vars(args):
        print(f"{arg:25s}: {getattr(args, arg)}")
    print("=" * 80)
    
    # Get dataloader
    test_loader, dataset = get_dataloader(args)
    image_size = args.img_size
    print(f"Dataset: {args.dataset}, Image size: {image_size}x{image_size}")
    print(f"Number of test samples: {len(dataset)}")
    
    # Check if multi-GPU inference is requested
    if args.gpus is not None and len(args.gpus) > 1:
        print(f"\nUsing multi-GPU inference on GPUs: {args.gpus}")
        
        # Load model on first GPU just to count parameters
        temp_model = load_model(args, image_size, device=torch.device(f'cuda:{args.gpus[0]}'))
        N = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        print(f"Number of model parameters: {N:,}")
        del temp_model
        torch.cuda.empty_cache()
        
        # Generate reconstructions using multiple GPUs
        print("\nGenerating reconstructions...")
        start_time = time.time()
        
        run_multi_gpu_inference(args, dataset, image_size, generated_images_dir)
        
        elapsed_time = (time.time() - start_time) / 60
        print(f'Generation complete. Time: {elapsed_time:.2f} minutes')
        
    else:
        # Single GPU inference (original code)
        if args.gpus is not None and len(args.gpus) == 1:
            args.device = torch.device(f'cuda:{args.gpus[0]}')
            print(f"\nUsing single GPU: {args.gpus[0]}")
        
        # Load model
        model = load_model(args, image_size)
        N = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of model parameters: {N:,}")
        
        # Setup forward problem (tomography operator)
        FP = Tomography(dim=image_size, num_angles=args.num_angles, device=args.device)
        
        # Generate reconstructions
        print("\nGenerating reconstructions...")
        start_time = time.time()
        
        for inputs, labels, image_ids in tqdm(test_loader, desc="Processing images"):
            x1 = inputs.to(args.device, non_blocking=True)
            
            # Generate sinogram with noise
            percent_noise = args.noise_level * torch.ones(x1.shape[0]).type_as(x1)
            data = FP(x1)
            
            if args.use_noise_embed:
                std = get_gaussian_noise_std(data, percent_noise)
                z = torch.randn_like(data).to(args.device)
                data = data + std[:, None, None, None] * z
            else:
                std = torch.zeros(x1.shape[0]).to(args.device)
            
            ATb = FP.adjoint(data)
            
            # Generate and save reconstructions
            with torch.no_grad():
                generate_and_save_one_batch(
                    model, image_size, x1, data, ATb, std, args.nsteps, args.num_runs,
                    generated_images_dir, image_ids, args.device, args.use_noise_embed, args.dataset
                )
        
        elapsed_time = (time.time() - start_time) / 60
        print(f'Generation complete. Time: {elapsed_time:.2f} minutes')
    
    # Compute metrics
    print("\nComputing metrics...")
    images_fpaths = sorted(glob(os.path.join(generated_images_dir, '*.npy')))
    print(f"Found {len(images_fpaths)} generated images")
    
    # Setup FP for metric computation (use first GPU or specified device)
    if args.gpus is not None and len(args.gpus) > 0:
        metric_device = torch.device(f'cuda:{args.gpus[0]}')
    else:
        metric_device = args.device
    FP = Tomography(dim=image_size, num_angles=args.num_angles, device=metric_device)
    
    # Parallel metric computation
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_metrics)(i, path, FP, metric_device, args.num_runs)
        for i, path in enumerate(images_fpaths)
    )
    
    # Unpack results
    ImageIDs = []
    MSE_runs = np.zeros((len(images_fpaths), args.num_runs))
    PSNR_runs = np.zeros((len(images_fpaths), args.num_runs))
    SSIM_runs = np.zeros((len(images_fpaths), args.num_runs))
    MISFIT_runs = np.zeros((len(images_fpaths), args.num_runs))
    MSE_means = []
    PSNR_means = []
    SSIM_means = []
    MISFIT_means = []
    
    for i, (image_fname, mse_runs, psnr_runs, ssim_runs, misfit_runs,
            mse_mean, psnr_mean, ssim_mean, misfit_mean) in enumerate(results):
        ImageIDs.append(image_fname)
        MSE_runs[i, :] = mse_runs
        PSNR_runs[i, :] = psnr_runs
        SSIM_runs[i, :] = ssim_runs
        MISFIT_runs[i, :] = misfit_runs
        MSE_means.append(mse_mean)
        PSNR_means.append(psnr_mean)
        SSIM_means.append(ssim_mean)
        MISFIT_means.append(misfit_mean)
    
    # Create DataFrame
    mse_col_names = [f'MSE_{pad_zeros_at_front(i, 2)}' for i in range(args.num_runs)]
    psnr_col_names = [f'PSNR_{pad_zeros_at_front(i, 2)}' for i in range(args.num_runs)]
    ssim_col_names = [f'SSIM_{pad_zeros_at_front(i, 2)}' for i in range(args.num_runs)]
    misfit_col_names = [f'MISFIT_{pad_zeros_at_front(i, 2)}' for i in range(args.num_runs)]
    
    all_col_names = (['ImageID'] + mse_col_names + psnr_col_names + ssim_col_names + 
                     misfit_col_names + ['MSE_mean', 'PSNR_mean', 'SSIM_mean', 'MISFIT_mean'])
    
    df = pd.DataFrame(columns=all_col_names)
    df['ImageID'] = ImageIDs
    df[mse_col_names] = MSE_runs
    df[psnr_col_names] = PSNR_runs
    df[ssim_col_names] = SSIM_runs
    df[misfit_col_names] = MISFIT_runs
    df['MSE_mean'] = MSE_means
    df['PSNR_mean'] = PSNR_means
    df['SSIM_mean'] = SSIM_means
    df['MISFIT_mean'] = MISFIT_means
    
    # Save metrics
    metrics_fpath = os.path.join(metrics_dir, 'metrics.csv')
    df.to_csv(metrics_fpath, index=False)
    print(f"\nMetrics saved to: {metrics_fpath}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics (Mean Reconstruction):")
    print("=" * 80)
    print(f"MSE:    {np.mean(MSE_means):.4f} ± {np.std(MSE_means):.4f}")
    print(f"PSNR:   {np.mean(PSNR_means):.4f} ± {np.std(PSNR_means):.4f} dB")
    print(f"SSIM:   {np.mean(SSIM_means):.4f} ± {np.std(SSIM_means):.4f}")
    print(f"MISFIT: {np.mean(MISFIT_means):.4f} ± {np.std(MISFIT_means):.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
