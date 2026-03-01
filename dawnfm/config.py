"""
Configuration file for DAWN-FM experiments

This file contains default configurations and preset experiment settings.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    image_size: int
    num_channels: int
    train_size: int
    test_size: int
    default_arch: List[int]


# Dataset configurations
MNIST_CONFIG = DatasetConfig(
    name='mnist',
    image_size=28,
    num_channels=1,
    train_size=60000,
    test_size=10000,
    default_arch=[1, 16, 32]
)

CIFAR10_CONFIG = DatasetConfig(
    name='cifar10',
    image_size=32,
    num_channels=3,
    train_size=50000,
    test_size=10000,
    default_arch=[3, 16, 32]
)

STL10_CONFIG = DatasetConfig(
    name='stl10',
    image_size=64,
    num_channels=3,
    train_size=105000,  # train + unlabeled
    test_size=8000,
    default_arch=[3, 16, 32]
)

DATASET_CONFIGS = {
    'mnist': MNIST_CONFIG,
    'cifar10': CIFAR10_CONFIG,
    'stl10': STL10_CONFIG
}


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Dataset
    dataset: str = 'mnist'
    data_dir: str = './data'
    
    # Model
    use_noise_embed: bool = False
    arch: Optional[List[int]] = None
    
    # Training
    batch_size: int = 512
    max_epochs: int = 1000
    lr: float = 1e-4
    lr_min: float = 0.0
    num_workers: int = 4
    
    # Forward problem
    blur_sigma: List[float] = None
    noise_range: List[float] = None
    
    # Interpolation
    interpolation_sigma: float = 0.01
    antithetic_sampling: bool = False
    
    # Saving
    save_dir: str = './experiments'
    exp_name: Optional[str] = None
    save_every: int = 25
    
    # Device
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.blur_sigma is None:
            self.blur_sigma = [3.0, 3.0]
        if self.noise_range is None:
            self.noise_range = [0.0, 0.1]
        if self.arch is None:
            self.arch = DATASET_CONFIGS[self.dataset].default_arch


@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Dataset
    dataset: str = 'mnist'
    data_dir: str = './data'
    
    # Model
    model_path: str = ''
    use_noise_embed: bool = False
    arch: Optional[List[int]] = None
    
    # Forward problem
    blur_sigma: List[float] = None
    noise_level: float = 0.0
    
    # Inference
    batch_size: int = 1
    num_workers: int = 4
    nsteps: int = 100
    num_runs: int = 32
    
    # Saving
    save_dir: str = './results'
    exp_name: Optional[str] = None
    
    # Performance
    device: Optional[str] = None
    n_jobs: int = 8
    
    def __post_init__(self):
        if self.blur_sigma is None:
            self.blur_sigma = [3.0, 3.0]
        if self.arch is None:
            self.arch = DATASET_CONFIGS[self.dataset].default_arch


# Preset experiment configurations
PRESET_EXPERIMENTS = {
    'mnist_daw': TrainingConfig(
        dataset='mnist',
        use_noise_embed=False,
        batch_size=512,
        max_epochs=1000,
        exp_name='mnist_daw-fm_baseline'
    ),
    'mnist_dawn': TrainingConfig(
        dataset='mnist',
        use_noise_embed=True,
        batch_size=512,
        max_epochs=1000,
        noise_range=[0.0, 0.1],
        exp_name='mnist_dawn-fm_baseline'
    ),
    'cifar10_daw': TrainingConfig(
        dataset='cifar10',
        use_noise_embed=False,
        batch_size=256,
        max_epochs=1000,
        exp_name='cifar10_daw-fm_baseline'
    ),
    'cifar10_dawn': TrainingConfig(
        dataset='cifar10',
        use_noise_embed=True,
        batch_size=256,
        max_epochs=1000,
        noise_range=[0.0, 0.1],
        exp_name='cifar10_dawn-fm_baseline'
    ),
    'stl10_daw': TrainingConfig(
        dataset='stl10',
        use_noise_embed=False,
        batch_size=128,
        max_epochs=1000,
        exp_name='stl10_daw-fm_baseline'
    ),
    'stl10_dawn': TrainingConfig(
        dataset='stl10',
        use_noise_embed=True,
        batch_size=128,
        max_epochs=1000,
        noise_range=[0.0, 0.1],
        exp_name='stl10_dawn-fm_baseline'
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


def get_preset_experiment(preset_name: str) -> TrainingConfig:
    """Get preset experiment configuration by name"""
    if preset_name not in PRESET_EXPERIMENTS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available: {list(PRESET_EXPERIMENTS.keys())}")
    return PRESET_EXPERIMENTS[preset_name]
