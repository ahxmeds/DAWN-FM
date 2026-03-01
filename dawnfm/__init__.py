"""
DAWN-FM: Data-Aware Noise-Embedded Flow Matching

This package contains the core modules for DAWN-FM/DAW-FM implementations.
"""

from .models import UNetFMG_DE, UNetFMG_DE_NE, odeSol_data, odeSol_data_noise
from .forward_problems import blurFFT, blurFFT_generator, Tomography
from .load_datasets import (
    get_mnist_dataset,
    get_cifar10_dataset,
    get_stl10_dataset,
    get_organcmnist_dataset,
    get_organamnist_dataset,
    get_organsmnist_dataset,
    DatasetWithImageID,
    set_data_directory
)
from .config import (
    DATASET_CONFIGS,
    TrainingConfig,
    InferenceConfig,
    get_dataset_config,
    get_preset_experiment
)

__all__ = [
    # Models
    'UNetFMG_DE',
    'UNetFMG_DE_NE',
    'odeSol_data',
    'odeSol_data_noise',
    
    # Forward operators
    'blurFFT',
    'blurFFT_generator',
    'Tomography',
    
    # Datasets
    'get_mnist_dataset',
    'get_cifar10_dataset',
    'get_stl10_dataset',
    'get_organcmnist_dataset',
    'get_organamnist_dataset',
    'get_organsmnist_dataset',
    'DatasetWithImageID',
    'set_data_directory',
    
    # Configuration
    'DATASET_CONFIGS',
    'TrainingConfig',
    'InferenceConfig',
    'get_dataset_config',
    'get_preset_experiment',
]

__version__ = '1.0.0'

