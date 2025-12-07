"""
Model Configuration for Hybrid Mamba-Attention LLM

Supports multiple model scales optimized for single-GPU training:
- nano: ~125M params (testing/debugging)
- small: ~350M params (quick experiments)
- base: ~770M params (balanced performance)
- large: ~1.3B params (full scale, 14-day training target)

Architecture follows SOTA protocols from Qwen, Mamba-2, and Gated Attention research.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import math


@dataclass
class ModelConfig:
    """
    Configuration for Hybrid Mamba-Gated Attention LLM.
    
    The architecture interleaves Mamba-2 SSM blocks with Gated Attention blocks
    for optimal efficiency and retrieval capability.
    """
    
    # Model scale
    model_name: str = "hybrid-llm-1.3b"
    
    # Core dimensions
    vocab_size: int = 50304  # Divisible by 64 for efficiency
    hidden_size: int = 2048  # d_model
    intermediate_size: int = 5632  # ~2.75x hidden for SwiGLU
    num_hidden_layers: int = 24  # Total layers (Mamba + Attention)
    
    # Attention configuration
    num_attention_heads: int = 16  # Query heads
    num_key_value_heads: int = 4   # GQA: 4x fewer KV heads
    head_dim: int = 128  # Per-head dimension
    
    # Mamba-2 SSM configuration
    mamba_d_state: int = 128  # SSM state dimension
    mamba_d_conv: int = 4     # Convolution kernel size
    mamba_expand: int = 2     # Expansion factor for Mamba
    
    # Architecture ratios
    mamba_ratio: int = 3  # 3:1 ratio of Mamba to Attention layers
    # With 24 layers: 18 Mamba + 6 Attention
    
    # Sequence configuration
    max_position_embeddings: int = 4096  # Context length for training
    rope_theta: float = 100000.0  # RoPE base frequency (extended context)
    
    # Regularization
    hidden_dropout: float = 0.0  # No dropout for pretraining
    attention_dropout: float = 0.0
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Initialization
    initializer_range: float = 0.02
    
    # Gated Attention specific
    use_gated_attention: bool = True
    gate_bias: bool = True  # Bias in gating projection
    
    # Memory efficiency
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Precision
    torch_dtype: str = "bfloat16"
    
    def __post_init__(self):
        """Validate and compute derived values."""
        # Ensure head_dim consistency
        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Compute layer distribution
        self.num_attention_layers = self.num_hidden_layers // (self.mamba_ratio + 1)
        self.num_mamba_layers = self.num_hidden_layers - self.num_attention_layers
        
        # Attention layer indices (evenly distributed)
        step = self.num_hidden_layers // self.num_attention_layers
        self.attention_layer_indices = set(
            (i + 1) * step - 1 for i in range(self.num_attention_layers)
        )
    
    @property
    def num_parameters(self) -> int:
        """Estimate total parameters."""
        # Embedding
        emb_params = self.vocab_size * self.hidden_size
        
        # Per Mamba layer
        mamba_params = (
            3 * self.hidden_size * self.mamba_expand * self.hidden_size +  # in_proj
            self.mamba_expand * self.hidden_size * self.mamba_d_conv +      # conv1d
            self.mamba_expand * self.hidden_size * self.mamba_d_state * 2 + # SSM
            self.mamba_expand * self.hidden_size * self.hidden_size          # out_proj
        )
        
        # Per Attention layer (with GQA)
        kv_dim = self.num_key_value_heads * self.head_dim
        attn_params = (
            self.hidden_size * self.hidden_size +  # Q
            self.hidden_size * kv_dim +             # K
            self.hidden_size * kv_dim +             # V
            self.hidden_size * self.hidden_size +  # out_proj
            self.head_dim * self.head_dim          # gate
        )
        
        # MLP per layer
        mlp_params = 3 * self.hidden_size * self.intermediate_size  # SwiGLU
        
        # LayerNorm (no learnable params with RMSNorm)
        
        # Total
        total = (
            emb_params +  # embeddings
            self.num_mamba_layers * (mamba_params + mlp_params) +
            self.num_attention_layers * (attn_params + mlp_params) +
            emb_params  # lm_head (tied or untied)
        )
        
        return total


# Predefined configurations for different scales
NANO_CONFIG = ModelConfig(
    model_name="hybrid-llm-nano",
    hidden_size=512,
    intermediate_size=1408,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    mamba_d_state=64,
    max_position_embeddings=2048,
)

SMALL_CONFIG = ModelConfig(
    model_name="hybrid-llm-small",
    hidden_size=1024,
    intermediate_size=2816,
    num_hidden_layers=16,
    num_attention_heads=8,
    num_key_value_heads=2,
    mamba_d_state=64,
    max_position_embeddings=2048,
)

BASE_CONFIG = ModelConfig(
    model_name="hybrid-llm-base",
    hidden_size=1536,
    intermediate_size=4096,
    num_hidden_layers=20,
    num_attention_heads=12,
    num_key_value_heads=4,
    mamba_d_state=96,
    max_position_embeddings=4096,
)

LARGE_CONFIG = ModelConfig(
    model_name="hybrid-llm-1.3b",
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=4,
    mamba_d_state=128,
    max_position_embeddings=4096,
)


def get_config(size: str = "large") -> ModelConfig:
    """Get model configuration by size name."""
    configs = {
        "nano": NANO_CONFIG,
        "small": SMALL_CONFIG,
        "base": BASE_CONFIG,
        "large": LARGE_CONFIG,
    }
    if size not in configs:
        raise ValueError(f"Unknown config size: {size}. Choose from {list(configs.keys())}")
    return configs[size]
