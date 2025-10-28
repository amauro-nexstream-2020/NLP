"""
Model Architecture Configurations

This module contains different model size configurations for the decoder-only transformer.
Configurations are based on common LLM architectures but scaled for educational purposes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for transformer model architecture."""
    
    # Model dimensions
    vocab_size: int = 50257  # GPT-2 vocabulary size (default)
    n_embd: int = 768        # Embedding dimension
    n_layer: int = 12        # Number of transformer blocks
    n_head: int = 12         # Number of attention heads
    n_positions: int = 1024  # Maximum sequence length
    
    # Architecture details
    dropout: float = 0.1
    bias: bool = True  # Use bias in Linear and LayerNorm layers
    
    # Model behavior
    use_cache: bool = False  # Cache key-value pairs for generation
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
    
    @property
    def n_params(self) -> int:
        """Estimate number of parameters (approximate)."""
        # Rough estimation
        emb_params = self.vocab_size * self.n_embd
        pos_params = self.n_positions * self.n_embd
        
        # Per transformer block
        attn_params = 4 * self.n_embd * self.n_embd  # QKV + output projection
        ffn_params = 8 * self.n_embd * self.n_embd   # Typically 4x expansion
        layer_params = (attn_params + ffn_params) * self.n_layer
        
        output_params = self.vocab_size * self.n_embd
        
        total = emb_params + pos_params + layer_params + output_params
        return total


# Predefined configurations
class ModelConfigs:
    """Collection of predefined model configurations."""
    
    @staticmethod
    def tiny():
        """Tiny model for quick experimentation (~10M parameters)."""
        return ModelConfig(
            n_embd=128,
            n_layer=4,
            n_head=4,
            n_positions=512,
            dropout=0.1
        )
    
    @staticmethod
    def small():
        """Small model for educational purposes (~50M parameters)."""
        return ModelConfig(
            n_embd=512,
            n_layer=8,
            n_head=8,
            n_positions=1024,
            dropout=0.1
        )
    
    @staticmethod
    def medium():
        """Medium model (~350M parameters, similar to GPT-2 Medium)."""
        return ModelConfig(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_positions=1024,
            dropout=0.1
        )
    
    @staticmethod
    def large():
        """Large model (~760M parameters, similar to GPT-2 Large)."""
        return ModelConfig(
            n_embd=1280,
            n_layer=36,
            n_head=20,
            n_positions=1024,
            dropout=0.1
        )
    
    @staticmethod
    def gpt2():
        """GPT-2 small configuration (124M parameters)."""
        return ModelConfig(
            vocab_size=50257,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_positions=1024,
            dropout=0.1
        )
    
    @staticmethod
    def custom(
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_positions: int = 1024,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """Create a custom configuration."""
        return ModelConfig(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=n_positions,
            dropout=dropout,
            bias=bias
        )


def print_config_summary(config: ModelConfig):
    """Print a summary of the model configuration."""
    print("=" * 60)
    print("Model Configuration Summary")
    print("=" * 60)
    print(f"Vocabulary Size:      {config.vocab_size:,}")
    print(f"Embedding Dimension:  {config.n_embd:,}")
    print(f"Number of Layers:     {config.n_layer}")
    print(f"Number of Heads:      {config.n_head}")
    print(f"Max Sequence Length:  {config.n_positions:,}")
    print(f"Dropout:              {config.dropout}")
    print(f"Use Bias:             {config.bias}")
    print("-" * 60)
    print(f"Estimated Parameters: ~{config.n_params / 1e6:.1f}M")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    configs = [
        ("Tiny", ModelConfigs.tiny()),
        ("Small", ModelConfigs.small()),
        ("Medium", ModelConfigs.medium()),
        ("GPT-2", ModelConfigs.gpt2()),
    ]
    
    for name, config in configs:
        print(f"\n{name} Model:")
        print_config_summary(config)
