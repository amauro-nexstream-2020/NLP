"""
Training Configuration for Hybrid LLM

Optimized for:
- Single A100/A6000 GPU (40-80GB VRAM)
- 14-day training window
- ~100B tokens target (truncated from full FineWeb-Edu for feasibility)

Implements Chinchilla-optimal scaling with adjustments for hybrid architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class DataConfig:
    """
    Dataset configuration for streaming training.
    
    The defaults follow the requested FineWeb-Edu + MedQA mixture.
    """
    fineweb_subset: str = "sample-100BT"
    fineweb_min_quality: float = 0.99  # filter on language_score when available
    medqa_split: str = "train"
    fineweb_probability: float = 0.9
    medqa_probability: float = 0.1
    shuffle_buffer_size: int = 10_000
    seed: int = 42


@dataclass
class TrainingConfig:
    """
    Training hyperparameters following SOTA practices from:
    - Qwen-2.5 training protocols
    - Mamba-2 optimization guidelines
    - Chinchilla scaling laws
    """
    
    # ========================
    # Training Duration
    # ========================
    
    # Target tokens (truncated for 14-day feasibility)
    # Full FineWeb-Edu: 1.3T tokens
    # Our target: ~100B tokens (achievable in 14 days on single GPU)
    total_tokens: int = 100_000_000_000  # 100B tokens
    
    # Alternatively specify by iterations
    max_steps: int = -1  # -1 = compute from total_tokens
    
    # ========================
    # Batch Configuration
    # ========================
    
    # Global batch size in tokens (following Chinchilla)
    global_batch_size: int = 524_288  # 512K tokens
    
    # Per-device micro batch size (tune to fit VRAM)
    micro_batch_size: int = 8  # 8 sequences per forward pass
    
    # Sequence length
    max_seq_length: int = 4096
    
    # Gradient accumulation (computed automatically)
    gradient_accumulation_steps: int = -1  # -1 = auto-compute
    
    # ========================
    # Optimizer Configuration
    # ========================
    
    optimizer_type: str = "adamw_8bit"  # adamw | adamw_8bit
    
    # AdamW hyperparameters
    learning_rate: float = 3e-4  # Peak LR
    min_learning_rate: float = 3e-5  # 10% of peak
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # Embedding-specific LRs (scaled by model dimension)
    embedding_lr_scale: float = 1.0
    lm_head_lr_scale: float = 0.01  # Much lower for output projection
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # ========================
    # Learning Rate Schedule
    # ========================
    
    lr_scheduler_type: str = "cosine_with_warmup"
    
    # Warmup
    warmup_steps: int = 2000
    warmup_ratio: float = 0.0  # Alternative: ratio of total steps
    
    # Cooldown (linear decay at end)
    cooldown_ratio: float = 0.1  # Last 10% of training
    
    # ========================
    # Mixed Precision
    # ========================
    
    precision: str = "bf16-mixed"  # bf16-mixed | fp16-mixed | 32
    
    # ========================
    # Memory Optimization
    # ========================
    
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_fused_kernels: bool = True
    
    # Activation checkpointing granularity
    checkpoint_activations: str = "full"  # none | selective | full
    
    # ========================
    # Data Configuration
    # ========================
    
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B"
    data: DataConfig = field(default_factory=DataConfig)
    
    # ========================
    # Evaluation
    # ========================
    
    eval_every_steps: int = 500
    eval_tokens: int = 10_000_000  # 10M tokens for validation
    
    # ========================
    # Checkpointing
    # ========================
    
    save_every_steps: int = 1000
    save_total_limit: int = 5  # Keep last N checkpoints
    checkpoint_dir: str = "checkpoints"
    
    # ========================
    # Logging
    # ========================
    
    log_every_steps: int = 10
    use_clearml: bool = True
    clearml_project: str = "Hybrid-LLM-1.3B"
    clearml_task: str = "pretraining-v1"
    
    use_wandb: bool = False
    wandb_project: str = "hybrid-llm"
    
    # ========================
    # Reproducibility
    # ========================
    
    seed: int = 42
    deterministic: bool = False  # Slight perf hit if True
    
    # ========================
    # Hardware
    # ========================
    
    device: str = "cuda"
    num_gpus: int = 1
    
    def __post_init__(self):
        """Compute derived values."""
        # Tokens per micro-batch
        tokens_per_micro = self.micro_batch_size * self.max_seq_length
        
        # Auto-compute gradient accumulation
        if self.gradient_accumulation_steps == -1:
            self.gradient_accumulation_steps = self.global_batch_size // tokens_per_micro
        
        # Compute max_steps if not specified
        if self.max_steps == -1:
            self.max_steps = self.total_tokens // self.global_batch_size
        
        # Validate
        actual_batch_tokens = (
            self.micro_batch_size *
            self.max_seq_length *
            self.gradient_accumulation_steps
        )
        
        if actual_batch_tokens != self.global_batch_size:
            print(f"Warning: Adjusting global_batch_size from {self.global_batch_size} "
                  f"to {actual_batch_tokens} for exact divisibility")
            self.global_batch_size = actual_batch_tokens
    
    @property
    def estimated_training_time_hours(self) -> float:
        """
        Estimate training time on single A100.
        
        Assumptions (conservative):
        - ~40k tokens/sec throughput for 1.3B hybrid model
        - 14 days = 336 hours
        """
        tokens_per_second = 40_000  # Conservative for single GPU
        total_seconds = self.total_tokens / tokens_per_second
        return total_seconds / 3600
    
    @property
    def fits_in_14_days(self) -> bool:
        """Check if training fits in 14-day window."""
        return self.estimated_training_time_hours <= 336  # 14 * 24


# Predefined training configs
FAST_DEBUG_CONFIG = TrainingConfig(
    total_tokens=1_000_000,  # 1M tokens
    global_batch_size=32768,
    micro_batch_size=4,
    max_seq_length=512,
    max_steps=100,
    eval_every_steps=25,
    save_every_steps=50,
    warmup_steps=10,
    tokenizer_name="Qwen/Qwen2.5-1.5B",
    data=DataConfig(fineweb_subset="sample-10BT", fineweb_probability=0.8, medqa_probability=0.2),
)

# Quick training run - 1B tokens, ~6-8 hours on A100
QUICK_CONFIG = TrainingConfig(
    total_tokens=1_000_000_000,  # 1B tokens
    global_batch_size=262_144,  # 256K tokens per step
    micro_batch_size=8,
    max_seq_length=2048,
    max_steps=-1,  # Auto-compute from total_tokens
    warmup_steps=500,
    eval_every_steps=200,
    save_every_steps=500,
    learning_rate=3e-4,
    tokenizer_name="Qwen/Qwen2.5-1.5B",
    data=DataConfig(fineweb_subset="sample-10BT", fineweb_probability=0.9, medqa_probability=0.1),
    clearml_task="pretraining-quick",
)

SINGLE_GPU_14DAY_CONFIG = TrainingConfig(
    total_tokens=100_000_000_000,  # 100B tokens
    global_batch_size=524_288,
    micro_batch_size=8,
    max_seq_length=4096,
    warmup_steps=2000,
    eval_every_steps=500,
    save_every_steps=1000,
    tokenizer_name="Qwen/Qwen2.5-1.5B",
    data=DataConfig(),
)

MULTI_GPU_CONFIG = TrainingConfig(
    total_tokens=500_000_000_000,  # 500B tokens
    global_batch_size=2_097_152,  # 2M tokens
    micro_batch_size=8,
    max_seq_length=4096,
    num_gpus=8,
    tokenizer_name="Qwen/Qwen2.5-1.5B",
)


def get_training_config(name: str = "single_gpu") -> TrainingConfig:
    """Return a predefined training configuration by name."""
    presets: Dict[str, TrainingConfig] = {
        "debug": FAST_DEBUG_CONFIG,
        "quick": QUICK_CONFIG,
        "single_gpu": SINGLE_GPU_14DAY_CONFIG,
        "multi_gpu": MULTI_GPU_CONFIG,
    }
    if name not in presets:
        raise ValueError(f"Unknown training config: {name}. Choose from {list(presets.keys())}")
    return presets[name]
# Note: get_training_config is defined above (single definition)
