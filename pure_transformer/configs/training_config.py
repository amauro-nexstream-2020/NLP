"""
Training Configuration for Pure Transformer

Includes:
- Pretraining config (SFT phase)
- GRPO/ProRL config (RL phase)
- Data configuration

Optimized for 3-day A100 (80GB) training.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Dataset configuration for streaming training."""
    
    # FineWeb-Edu (general pretraining)
    fineweb_subset: str = "sample-10BT"  # Smaller for 3-day training
    fineweb_probability: float = 0.85
    
    # MedQA (domain knowledge)
    medqa_probability: float = 0.15
    medqa_split: str = "train"
    
    # Streaming settings
    shuffle_buffer_size: int = 10_000
    seed: int = 42
    
    # Sequence packing
    max_seq_length: int = 2048


@dataclass
class TrainingConfig:
    """
    Pretraining/SFT configuration optimized for 3-day A100.
    
    Throughput target: ~500K tokens/sec on A100
    3 days = 259,200 seconds
    Total tokens: ~130B tokens (achievable)
    """
    
    # Training duration (3 days on A100)
    total_tokens: int = 50_000_000_000  # 50B tokens (conservative for 3 days)
    max_steps: int = -1  # Auto-compute from total_tokens
    
    # Batch configuration
    # A100 80GB can handle larger batches with gradient checkpointing
    global_batch_size: int = 524_288  # 512K tokens
    micro_batch_size: int = 16  # Per forward pass
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = -1  # Auto-compute
    
    # Optimizer (AdamW)
    optimizer_type: str = "adamw"
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_tokens: int = 1_000_000_000  # 1B tokens warmup
    lr_scheduler_type: str = "cosine"
    
    # Precision
    mixed_precision: str = "bf16"
    
    # Checkpointing
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    checkpoint_dir: str = "./checkpoints/pretrain"
    
    # Logging
    log_every_n_steps: int = 10
    project_name: str = "pure-transformer-pretrain"
    
    # Data
    data: DataConfig = field(default_factory=DataConfig)


@dataclass 
class RLConfig:
    """
    GRPO/ProRL configuration for reinforcement learning phase.
    
    Based on:
    - DeepSeekMath GRPO paper
    - RLHF best practices
    - nanochat implementation
    """
    
    # RL algorithm
    algorithm: str = "grpo"  # grpo | reinforce | ppo
    
    # Sampling
    num_samples_per_prompt: int = 16  # G in GRPO
    temperature: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 512
    
    # Batch sizes
    prompts_per_step: int = 8  # Number of prompts per training step
    device_batch_size: int = 4  # Max sequences per forward pass
    
    # Optimizer (lower LR for RL fine-tuning)
    learning_rate: float = 1e-5
    embedding_lr: float = 2e-4  # Higher for embeddings
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    
    # GRPO specific
    use_baseline: bool = True  # Subtract mean reward
    normalize_advantages: bool = False  # Don't divide by std
    
    # KL penalty (optional, for PPO-style)
    kl_coef: float = 0.0  # Set >0 to add KL penalty
    
    # Training duration
    num_epochs: int = 3
    save_every_n_steps: int = 100
    eval_every_n_steps: int = 50
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints/rl"
    pretrained_checkpoint: str = "./checkpoints/pretrain/latest.pt"
    
    # Logging
    log_every_n_steps: int = 1
    project_name: str = "pure-transformer-rl"
    
    # Tasks for reward
    tasks: List[str] = field(default_factory=lambda: ["gsm8k", "medqa"])


def get_training_config(name: str) -> TrainingConfig:
    """Get a training configuration preset."""
    
    presets = {
        "debug": TrainingConfig(
            total_tokens=10_000_000,  # 10M tokens
            global_batch_size=32_768,
            micro_batch_size=4,
            max_seq_length=512,
            save_every_n_steps=50,
            eval_every_n_steps=25,
            warmup_tokens=100_000,
        ),
        "a100_3day": TrainingConfig(
            total_tokens=50_000_000_000,  # 50B tokens
            global_batch_size=524_288,
            micro_batch_size=16,
            max_seq_length=2048,
            learning_rate=3e-4,
            warmup_tokens=1_000_000_000,
        ),
        "a100_1day": TrainingConfig(
            total_tokens=15_000_000_000,  # 15B tokens
            global_batch_size=524_288,
            micro_batch_size=16,
            max_seq_length=2048,
            learning_rate=3e-4,
            warmup_tokens=500_000_000,
        ),
    }
    
    if name not in presets:
        raise ValueError(f"Unknown training config: {name}. Choose from {list(presets.keys())}")
    return presets[name]


def get_rl_config(name: str) -> RLConfig:
    """Get an RL configuration preset."""
    
    presets = {
        "debug": RLConfig(
            num_samples_per_prompt=4,
            prompts_per_step=2,
            device_batch_size=2,
            num_epochs=1,
            save_every_n_steps=10,
            eval_every_n_steps=5,
        ),
        "grpo_gsm8k": RLConfig(
            algorithm="grpo",
            num_samples_per_prompt=16,
            prompts_per_step=8,
            device_batch_size=4,
            learning_rate=1e-5,
            num_epochs=3,
            tasks=["gsm8k"],
        ),
        "grpo_medqa": RLConfig(
            algorithm="grpo",
            num_samples_per_prompt=16,
            prompts_per_step=8,
            device_batch_size=4,
            learning_rate=1e-5,
            num_epochs=3,
            tasks=["medqa"],
        ),
        "grpo_full": RLConfig(
            algorithm="grpo",
            num_samples_per_prompt=16,
            prompts_per_step=16,
            device_batch_size=8,
            learning_rate=1e-5,
            num_epochs=5,
            tasks=["gsm8k", "medqa"],
        ),
    }
    
    if name not in presets:
        raise ValueError(f"Unknown RL config: {name}. Choose from {list(presets.keys())}")
    return presets[name]
