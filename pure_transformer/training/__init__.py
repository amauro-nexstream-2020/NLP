"""Training modules for Pure Transformer."""

from pure_transformer.training.pretrain import run_pretraining
from pure_transformer.training.grpo import run_grpo_training

__all__ = [
    "run_pretraining",
    "run_grpo_training",
]
