"""Training modules for Pure Transformer."""

from pure_transformer.training.pretrain import run_pretraining, get_lr
from pure_transformer.training.grpo import run_grpo_training, GRPOTrainer, medqa_reward, gsm8k_reward
from pure_transformer.training.enhanced_grpo import EnhancedGRPOTrainer, GRPOConfig

# Lightning module (requires lightning package)
try:
    from pure_transformer.training.lightning_module import LightningTransformer, create_lightning_module
    _has_lightning = True
except ImportError:
    LightningTransformer = None
    create_lightning_module = None
    _has_lightning = False

__all__ = [
    "run_pretraining",
    "run_grpo_training",
    "GRPOTrainer",
    "EnhancedGRPOTrainer",
    "GRPOConfig",
    "LightningTransformer",
    "create_lightning_module",
    "get_lr",
    "medqa_reward",
    "gsm8k_reward",
]
