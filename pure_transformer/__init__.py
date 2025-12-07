"""
Pure Transformer LLM with SOTA RL Training (GRPO/ProRL)

A clean, efficient transformer architecture optimized for:
- 3-day training on single A100 (80GB)
- GRPO/ProRL reinforcement learning
- Easy K8s deployment

Architecture: ~350M parameters (trainable in <72 hours on A100)
"""

from pure_transformer.model import TransformerLM, TransformerConfig
from pure_transformer.configs import get_model_config, get_training_config

__all__ = [
    "TransformerLM",
    "TransformerConfig", 
    "get_model_config",
    "get_training_config",
]
