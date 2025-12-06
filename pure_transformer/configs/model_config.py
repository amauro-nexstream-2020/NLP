"""
Model Configuration for Pure Transformer LLM

Optimized for A100 training:
- tiny: ~85M params (debugging)
- small: ~178M params (quick experiments)  
- medium: ~374M params (baseline training)
- medium-large: ~401M params (0.4B - optimized for 2-day A100 training)
- large: ~763M params (0.76B - extended training)

**RECOMMENDED: medium-large for 2-day training on A100**
- Achieves ~2.6B tokens in 48 hours
- Optimal balance between model capacity and training data
- Tuned for USMLE QA with GRPO fine-tuning

Architecture follows SOTA practices:
- RoPE positional embeddings
- QK normalization
- GQA (Grouped Query Attention)
- SwiGLU MLP
- Pre-norm architecture
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Pure Transformer configuration."""
    
    # Model identity
    model_name: str = "pure-transformer-350m"
    
    # Core dimensions
    vocab_size: int = 50304  # Divisible by 64 for efficiency
    hidden_size: int = 1024
    intermediate_size: int = 2816  # ~2.75x hidden for SwiGLU
    num_layers: int = 24
    
    # Attention configuration
    num_heads: int = 16
    num_kv_heads: int = 4  # GQA: 4x fewer KV heads
    head_dim: int = 64
    
    # Sequence configuration
    max_seq_length: int = 2048
    
    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # RoPE configuration
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    
    # Training optimizations
    use_gradient_checkpointing: bool = True
    tie_word_embeddings: bool = False
    
    # Initialization
    initializer_range: float = 0.02
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads


# =============================================================================
# Model Presets (optimized for 3-day A100 training)
# =============================================================================

TINY_CONFIG = TransformerConfig(
    model_name="pure-transformer-45m",
    hidden_size=512,
    intermediate_size=1408,
    num_layers=12,
    num_heads=8,
    num_kv_heads=2,
    head_dim=64,
    max_seq_length=2048,
)

SMALL_CONFIG = TransformerConfig(
    model_name="pure-transformer-125m",
    hidden_size=768,
    intermediate_size=2048,
    num_layers=16,
    num_heads=12,
    num_kv_heads=4,
    head_dim=64,
    max_seq_length=2048,
)

# Target configuration for 3-day A100 training
MEDIUM_CONFIG = TransformerConfig(
    model_name="pure-transformer-350m",
    hidden_size=1024,
    intermediate_size=2816,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=64,
    max_seq_length=2048,
    use_gradient_checkpointing=True,
)

# Medium-large configuration: 0.4B parameters (optimal for 2-day A100 training)
# Balanced for maximum tokens (~2.6B) within time constraint
# Optimized for USMLE QA with GRPO fine-tuning
MEDIUM_LARGE_CONFIG = TransformerConfig(
    model_name="pure-transformer-400m",
    hidden_size=1152,
    intermediate_size=3168,  # ~2.75x hidden for SwiGLU
    num_layers=20,
    num_heads=16,
    num_kv_heads=4,
    head_dim=72,
    max_seq_length=2048,
    use_gradient_checkpointing=True,
)

# Large configuration: 0.76B parameters (optional - requires longer training)
# Similar scale to GPT-3.5, appropriate for domain-specific tasks
# Trainable on A100 in reasonable time with gradient checkpointing
LARGE_CONFIG = TransformerConfig(
    model_name="pure-transformer-760m",
    hidden_size=1536,
    intermediate_size=4224,  # ~2.75x hidden for SwiGLU
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=96,
    max_seq_length=2048,
    use_gradient_checkpointing=True,
)

# XLarge configuration: 1.3B parameters (OPTIMAL FOR 8x A100, 2-day training)
# Target: 11B tokens in 48 hours on 8x A100 GPUs
# Architecture: Chinchilla-optimal ratio (8.5 tokens/param)
# Use case: Maximum quality with multi-GPU training
XLARGE_CONFIG = TransformerConfig(
    model_name="pure-transformer-1.3b",
    hidden_size=1920,
    intermediate_size=5280,  # ~2.75x hidden for SwiGLU
    num_layers=32,
    num_heads=20,
    num_kv_heads=5,
    head_dim=96,
    max_seq_length=2048,
    use_gradient_checkpointing=True,
)


def get_model_config(name: str) -> TransformerConfig:
    """Get a model configuration by name."""
    configs = {
        "tiny": TINY_CONFIG,
        "small": SMALL_CONFIG,
        "medium": MEDIUM_CONFIG,
        "medium-large": MEDIUM_LARGE_CONFIG,  # 0.4B - optimized for 2-day training
        "large": LARGE_CONFIG,
        "xlarge": XLARGE_CONFIG,  # 1.3B - OPTIMAL for 8x A100
    }
    if name not in configs:
        raise ValueError(f"Unknown model config: {name}. Choose from {list(configs.keys())}")
    return configs[name]
