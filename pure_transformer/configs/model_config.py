"""
Model Configuration for Pure Transformer LLM

Optimized for 3-day A100 training:
- tiny: ~45M params (debugging)
- small: ~125M params (quick experiments)  
- medium: ~350M params (target for 3-day A100 training)

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


def get_model_config(name: str) -> TransformerConfig:
    """Get a model configuration by name."""
    configs = {
        "tiny": TINY_CONFIG,
        "small": SMALL_CONFIG,
        "medium": MEDIUM_CONFIG,
    }
    if name not in configs:
        raise ValueError(f"Unknown model config: {name}. Choose from {list(configs.keys())}")
    return configs[name]
