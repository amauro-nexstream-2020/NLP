"""Configuration module initialization."""

from .model_config import (
    ModelConfig,
    NANO_CONFIG,
    SMALL_CONFIG,
    BASE_CONFIG,
    LARGE_CONFIG,
    get_config,
)

from .training_config import (
    TrainingConfig,
    DataConfig,
    FAST_DEBUG_CONFIG,
    SINGLE_GPU_14DAY_CONFIG,
    MULTI_GPU_CONFIG,
    get_training_config,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig", 
    "DataConfig",
    "get_config",
    "get_training_config",
    "NANO_CONFIG",
    "SMALL_CONFIG",
    "BASE_CONFIG",
    "LARGE_CONFIG",
    "FAST_DEBUG_CONFIG",
    "SINGLE_GPU_14DAY_CONFIG",
    "MULTI_GPU_CONFIG",
]
