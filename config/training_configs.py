"""
Training Configuration

This module contains training hyperparameters and settings for different scenarios.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    
    # Training data
    batch_size: int = 32
    max_seq_length: int = 1024
    num_workers: int = 4
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 2000
    lr_decay_steps: Optional[int] = None  # None means no decay
    min_lr: float = 3e-5
    
    # Training duration
    max_epochs: int = 10
    max_steps: Optional[int] = None  # Override epochs if set
    
    # Validation
    eval_interval: int = 500
    eval_steps: int = 100
    
    # Checkpointing
    save_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Hardware
    device: str = "cuda"  # cuda, cpu, mps (Apple Silicon)
    mixed_precision: bool = True  # Use automatic mixed precision (fp16)
    compile_model: bool = False  # Use torch.compile (PyTorch 2.0+)
    
    # Reproducibility
    seed: int = 42
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1


class TrainingConfigs:
    """Collection of predefined training configurations."""
    
    @staticmethod
    def quick_test():
        """Configuration for quick testing/debugging."""
        return TrainingConfig(
            batch_size=4,
            max_seq_length=128,
            learning_rate=5e-4,
            max_steps=100,
            eval_interval=50,
            save_interval=50,
            log_interval=5,
            warmup_steps=10,
        )
    
    @staticmethod
    def small_scale():
        """Configuration for small-scale training (single GPU, limited time)."""
        return TrainingConfig(
            batch_size=16,
            max_seq_length=512,
            learning_rate=3e-4,
            max_epochs=5,
            eval_interval=200,
            save_interval=500,
            warmup_steps=500,
            gradient_accumulation_steps=2,
        )
    
    @staticmethod
    def full_training():
        """Configuration for full-scale training (multi-GPU, extended time)."""
        return TrainingConfig(
            batch_size=32,
            max_seq_length=1024,
            learning_rate=3e-4,
            max_epochs=20,
            eval_interval=500,
            save_interval=2000,
            warmup_steps=2000,
            gradient_accumulation_steps=4,
            mixed_precision=True,
        )
    
    @staticmethod
    def fine_tuning():
        """Configuration for fine-tuning a pre-trained model."""
        return TrainingConfig(
            batch_size=16,
            max_seq_length=512,
            learning_rate=1e-5,  # Lower LR for fine-tuning
            weight_decay=0.01,
            max_epochs=3,
            eval_interval=100,
            save_interval=200,
            warmup_steps=100,
            gradient_accumulation_steps=2,
        )
    
    @staticmethod
    def colab_gpu():
        """Configuration optimized for Google Colab GPU (limited VRAM)."""
        return TrainingConfig(
            batch_size=8,
            max_seq_length=512,
            learning_rate=3e-4,
            max_epochs=5,
            eval_interval=200,
            save_interval=500,
            warmup_steps=500,
            gradient_accumulation_steps=4,  # Simulate larger batch
            mixed_precision=True,
        )
    
    @staticmethod
    def nautilus_hpc():
        """Configuration for Nautilus HPC cluster (multi-GPU)."""
        return TrainingConfig(
            batch_size=64,
            max_seq_length=1024,
            learning_rate=6e-4,  # Scale with batch size
            max_epochs=30,
            eval_interval=1000,
            save_interval=5000,
            warmup_steps=4000,
            gradient_accumulation_steps=1,
            mixed_precision=True,
            compile_model=True,
        )


def print_training_config(config: TrainingConfig):
    """Print a summary of the training configuration."""
    print("=" * 60)
    print("Training Configuration Summary")
    print("=" * 60)
    print(f"Batch Size:           {config.batch_size}")
    print(f"Sequence Length:      {config.max_seq_length}")
    print(f"Learning Rate:        {config.learning_rate}")
    print(f"Weight Decay:         {config.weight_decay}")
    print(f"Gradient Clip:        {config.grad_clip}")
    print(f"Warmup Steps:         {config.warmup_steps}")
    print(f"Max Epochs:           {config.max_epochs}")
    print(f"Eval Interval:        {config.eval_interval}")
    print(f"Save Interval:        {config.save_interval}")
    print(f"Device:               {config.device}")
    print(f"Mixed Precision:      {config.mixed_precision}")
    print(f"Grad Accumulation:    {config.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    configs = [
        ("Quick Test", TrainingConfigs.quick_test()),
        ("Small Scale", TrainingConfigs.small_scale()),
        ("Google Colab", TrainingConfigs.colab_gpu()),
        ("Nautilus HPC", TrainingConfigs.nautilus_hpc()),
    ]
    
    for name, config in configs:
        print(f"\n{name} Configuration:")
        print_training_config(config)
