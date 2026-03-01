"""
Dataset loading utilities for MNIST, CIFAR10, and STL10

This module provides functions to load and preprocess datasets for training and testing.

Reference: https://www.aimsciences.org//article/doi/10.3934/fods.2026005
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Default data directory (can be overridden)
MAIN_DATA_DIR = './data'


def pad_zeros_at_front(num, N):
    """Pad number with leading zeros"""
    return str(num).zfill(N)


class RepeatChannelTransform:
    """Transform to repeat single channel to 3 channels"""
    def __call__(self, img):
        return img.repeat(3, 1, 1)


class DatasetWithImageID(Dataset):
    """
    Wrapper dataset that adds image IDs to samples
    
    Args:
        dataset: Base dataset
        dataset_name: Name of dataset for ID generation
    """
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_id = f'{self.dataset_name}_{pad_zeros_at_front(idx, 6)}'
        return image, label, image_id


def get_mnist_dataset(split='train', img_size=None, convert_to_three_channels=False, data_dir=None):
    """
    Load MNIST dataset
    
    Args:
        split: 'train' or 'test'
        img_size: Target image size (if None, uses default 28x28)
        convert_to_three_channels: If True, converts grayscale to 3-channel
        data_dir: Directory to store/load dataset (if None, uses MAIN_DATA_DIR)
    
    Returns:
        dataset: MNIST dataset
    """
    if data_dir is None:
        data_dir = MAIN_DATA_DIR
    
    if img_size is not None:
        if convert_to_three_channels:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.Normalize((0.1307,), (0.3081,)),
                RepeatChannelTransform(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if split == 'train':
        dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif split == 'test':
        dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f'Invalid split value: {split}')
    
    return dataset


def get_stl10_dataset(split='train', img_size=None, data_dir=None):
    """
    Load STL10 dataset
    
    Args:
        split: 'train' or 'test'
        img_size: Target image size (if None, uses default 64x64)
        data_dir: Directory to store/load dataset (if None, uses MAIN_DATA_DIR)
    
    Returns:
        dataset: STL10 dataset
    """
    if data_dir is None:
        data_dir = MAIN_DATA_DIR
    
    if img_size is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2240, 0.2215, 0.2239])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2240, 0.2215, 0.2239])
        ])
    
    if split == 'train':
        train_dataset = datasets.STL10(root=data_dir, split='train', download=True, transform=transform)
        unlabeled_dataset = datasets.STL10(root=data_dir, split='unlabeled', download=True, transform=transform)
        dataset = train_dataset + unlabeled_dataset
    elif split == 'test':
        dataset = datasets.STL10(root=data_dir, split='test', download=True, transform=transform)
    else:
        raise ValueError(f'Invalid split value: {split}')
    
    return dataset


def get_cifar10_dataset(split='train', img_size=None, data_dir=None):
    """
    Load CIFAR10 dataset
    
    Args:
        split: 'train' or 'test'
        img_size: Target image size (if None, uses default 32x32)
        data_dir: Directory to store/load dataset (if None, uses MAIN_DATA_DIR)
    
    Returns:
        dataset: CIFAR10 dataset
    """
    if data_dir is None:
        data_dir = MAIN_DATA_DIR
    
    mean_ = [0.49139968, 0.48215827, 0.44653124]
    std_ = [0.24703233, 0.24348505, 0.26158768]
    
    if img_size is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(mean=mean_, std=std_)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_, std=std_)
        ])
    
    if split == 'train':
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif split == 'test':
        dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f'Invalid split value: {split}')
    
    return dataset


def set_data_directory(data_dir):
    """
    Set the global data directory
    
    Args:
        data_dir: Path to data directory
    """
    global MAIN_DATA_DIR
    MAIN_DATA_DIR = data_dir


def get_organcmnist_dataset(split='train', img_size=64, data_dir=None):
    """
    Load OrganCMNIST dataset from MedMNIST collection
    
    OrganCMNIST is a grayscale dataset of abdominal CT organ scans.
    Dataset contains 11 classes of organs (liver, bladder, lungs, kidneys, etc.)
    
    Args:
        split: One of 'train', 'val', or 'test'
        img_size: Image resolution (default: 64)
        data_dir: Data directory (if None, uses MAIN_DATA_DIR)
        
    Returns:
        dataset: DatasetWithImageID instance
    """
    from medmnist import OrganCMNIST
    
    if data_dir is None:
        data_dir = MAIN_DATA_DIR
    
    # MedMNIST datasets are stored in medmnist subdirectory
    medmnist_dir = os.path.join(data_dir, 'medmnist')
    
    # Define transforms matching MNIST normalization approach
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = OrganCMNIST(split=split, root=medmnist_dir, download=True, 
                          size=img_size, transform=transform)
    dataset_name = f'organcmnist_{split}'
    
    return DatasetWithImageID(dataset, dataset_name)


def get_organamnist_dataset(split='train', img_size=64, data_dir=None):
    """
    Load OrganAMNIST dataset from MedMNIST collection
    
    OrganAMNIST is a grayscale dataset of abdominal CT organ scans
    (axial plane version of OrganCMNIST).
    
    Args:
        split: One of 'train', 'val', or 'test'
        img_size: Image resolution (default: 64)
        data_dir: Data directory (if None, uses MAIN_DATA_DIR)
        
    Returns:
        dataset: DatasetWithImageID instance
    """
    from medmnist import OrganAMNIST
    
    if data_dir is None:
        data_dir = MAIN_DATA_DIR
    
    # MedMNIST datasets are stored in medmnist subdirectory
    medmnist_dir = os.path.join(data_dir, 'medmnist')
    
    # Define transforms matching MNIST normalization approach
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = OrganAMNIST(split=split, root=medmnist_dir, download=True, 
                          size=img_size, transform=transform)
    dataset_name = f'organamnist_{split}'
    
    return DatasetWithImageID(dataset, dataset_name)


def get_organsmnist_dataset(split='train', img_size=64, data_dir=None):
    """
    Load OrganSMNIST dataset from MedMNIST collection
    
    OrganSMNIST is a grayscale dataset of abdominal CT organ scans
    (sagittal plane version of OrganCMNIST).
    
    Args:
        split: One of 'train', 'val', or 'test'
        img_size: Image resolution (default: 64)
        data_dir: Data directory (if None, uses MAIN_DATA_DIR)
        
    Returns:
        dataset: DatasetWithImageID instance
    """
    from medmnist import OrganSMNIST
    
    if data_dir is None:
        data_dir = MAIN_DATA_DIR
    
    # MedMNIST datasets are stored in medmnist subdirectory
    medmnist_dir = os.path.join(data_dir, 'medmnist')
    
    # Define transforms matching MNIST normalization approach
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = OrganSMNIST(split=split, root=medmnist_dir, download=True, 
                          size=img_size, transform=transform)
    dataset_name = f'organsmnist_{split}'
    
    return DatasetWithImageID(dataset, dataset_name)

