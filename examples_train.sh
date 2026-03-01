#!/bin/bash

# Example training script for DAWN-FM
# This script demonstrates training on MNIST with both DAW-FM and DAWN-FM modes

echo "=========================================="
echo "DAWN-FM Training Examples"
echo "=========================================="

# Example 1: DAW-FM on MNIST (data embedding only)
echo ""
echo "Example 1: Training DAW-FM on MNIST..."
echo "Command: python train_deblurring.py --dataset mnist --batch_size 512 --max_epochs 100 --save_dir ./experiments"
python train_deblurring.py \
    --dataset mnist \
    --batch_size 512 \
    --max_epochs 100 \
    --save_dir ./experiments \
    --save_every 25

# Example 2: DAWN-FM on MNIST (data + noise embedding)
echo ""
echo "Example 2: Training DAWN-FM on MNIST..."
echo "Command: python train_deblurring.py --dataset mnist --use_noise_embed --noise_range 0.0 0.1 --batch_size 512 --max_epochs 100 --save_dir ./experiments"
python train_deblurring.py \
    --dataset mnist \
    --use_noise_embed \
    --noise_range 0.0 0.1 \
    --batch_size 512 \
    --max_epochs 100 \
    --save_dir ./experiments \
    --save_every 25

# Example 3: DAW-FM on CIFAR10
echo ""
echo "Example 3: Training DAW-FM on CIFAR10..."
echo "Command: python train_deblurring.py --dataset cifar10 --batch_size 256 --max_epochs 100 --save_dir ./experiments"
python train_deblurring.py \
    --dataset cifar10 \
    --batch_size 256 \
    --max_epochs 100 \
    --save_dir ./experiments \
    --save_every 25

echo ""
echo "=========================================="
echo "Training examples completed!"
echo "Check ./experiments/logs for training logs"
echo "Check ./experiments/models for saved checkpoints"
echo "=========================================="
