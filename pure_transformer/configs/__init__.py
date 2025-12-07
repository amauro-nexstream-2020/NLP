"""Configuration presets for Pure Transformer."""

from pure_transformer.configs.model_config import (
    TransformerConfig,
    TINY_CONFIG,
    SMALL_CONFIG,
    MEDIUM_CONFIG,
    get_model_config,
)
from pure_transformer.configs.training_config import (
    TrainingConfig,
    RLConfig,
    DataConfig,
    get_training_config,
    get_rl_config,
)

__all__ = [
    "TransformerConfig",
    "TINY_CONFIG",
    "SMALL_CONFIG", 
    "MEDIUM_CONFIG",
    "get_model_config",
    "TrainingConfig",
    "RLConfig",
    "DataConfig",
    "get_training_config",
    "get_rl_config",
]
