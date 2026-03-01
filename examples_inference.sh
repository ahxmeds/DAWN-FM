#!/bin/bash

# Example inference script for DAWN-FM
# This script demonstrates inference for trained models

echo "=========================================="
echo "DAWN-FM Inference Examples"
echo "=========================================="

# Note: Update the model paths below to match your trained models

# Example 1: Inference with DAW-FM on MNIST (no noise)
echo ""
echo "Example 1: Inference with DAW-FM on MNIST (no noise)..."
MODEL_PATH="./experiments/models/mnist_daw-fm_arch-1x16x32/model_ep=0100.pth"
if [ -f "$MODEL_PATH" ]; then
    python inference_deblurring.py \
        --dataset mnist \
        --model_path $MODEL_PATH \
        --noise_level 0.0 \
        --nsteps 100 \
        --num_runs 32 \
        --save_dir ./results \
        --n_jobs 8
else
    echo "Model not found: $MODEL_PATH"
    echo "Please train the model first using examples_train.sh"
fi

# Example 2: Inference with DAWN-FM on MNIST (with noise)
echo ""
echo "Example 2: Inference with DAWN-FM on MNIST (5% noise)..."
MODEL_PATH="./experiments/models/mnist_dawn-fm_arch-1x16x32/model_ep=0100.pth"
if [ -f "$MODEL_PATH" ]; then
    python inference_deblurring.py \
        --dataset mnist \
        --use_noise_embed \
        --model_path $MODEL_PATH \
        --noise_level 0.05 \
        --nsteps 100 \
        --num_runs 32 \
        --save_dir ./results \
        --n_jobs 8
else
    echo "Model not found: $MODEL_PATH"
    echo "Please train the model first using examples_train.sh"
fi

# Example 3: Inference with DAW-FM on CIFAR10
echo ""
echo "Example 3: Inference with DAW-FM on CIFAR10..."
MODEL_PATH="./experiments/models/cifar10_daw-fm_arch-3x16x32/model_ep=0100.pth"
if [ -f "$MODEL_PATH" ]; then
    python inference_deblurring.py \
        --dataset cifar10 \
        --model_path $MODEL_PATH \
        --noise_level 0.0 \
        --nsteps 100 \
        --num_runs 32 \
        --save_dir ./results \
        --n_jobs 8
else
    echo "Model not found: $MODEL_PATH"
    echo "Please train the model first using examples_train.sh"
fi

# Example 4: Multi-GPU inference on CIFAR10 (using GPUs 0, 1, 2)
echo ""
echo "Example 4: Multi-GPU inference with DAW-FM on CIFAR10 (GPUs 0, 1, 2)..."
MODEL_PATH="./experiments/models/cifar10_daw-fm_arch-3x16x32/model_ep=0100.pth"
if [ -f "$MODEL_PATH" ]; then
    python inference_deblurring.py \
        --dataset cifar10 \
        --model_path $MODEL_PATH \
        --gpus 0 1 2 \
        --noise_level 0.0 \
        --nsteps 100 \
        --num_runs 32 \
        --save_dir ./results \
        --n_jobs 8
else
    echo "Model not found: $MODEL_PATH"
    echo "Please train the model first using examples_train.sh"
fi

echo ""
echo "=========================================="
echo "Inference examples completed!"
echo "Check ./results/generated_images for reconstructed images"
echo "Check ./results/metrics for evaluation metrics"
echo "=========================================="
