"""
Pure Transformer Language Model

A clean, efficient implementation optimized for:
- Fast training on A100 GPUs
- GRPO/ProRL reinforcement learning
- Efficient inference with KV cache

Architecture:
- RoPE positional embeddings
- QK normalization (stabilizes training)
- GQA (Grouped Query Attention)
- SwiGLU MLP
- Pre-norm with RMSNorm
- Flash Attention (when available)
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# Compatibility for PyTorch < 2.4
if hasattr(F, "rms_norm"):
    rms_norm_func = F.rms_norm
else:
    # Custom autograd function to handle bf16 dtype explicitly for FlashAttention compatibility
    # while maintaining proper gradient flow for gradient checkpointing
    class RMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: Tensor, normalized_shape: Tuple[int, ...], weight: Optional[Tensor], eps: float):
            # Save for backward
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            ctx.normalized_shape = normalized_shape
            
            # Compute RMSNorm in input dtype (bf16)
            variance = x.pow(2).mean(-1, keepdim=True)
            rsqrt_var = torch.rsqrt(variance + eps)
            hidden_states = x * rsqrt_var
            if weight is not None:
                hidden_states = hidden_states * weight
            # Ensure output is same dtype as input (for FlashAttention)
            return hidden_states.to(x.dtype)
        
        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            eps = ctx.eps
            
            # Compute gradients
            variance = x.pow(2).mean(-1, keepdim=True)
            rsqrt_var = torch.rsqrt(variance + eps)
            normalized = x * rsqrt_var
            
            if weight is not None:
                grad_weight = (grad_output * normalized).sum(dim=tuple(range(grad_output.dim() - 1)))
                grad_normalized = grad_output * weight
            else:
                grad_weight = None
                grad_normalized = grad_output
            
            # Gradient of x through RMSNorm
            n = x.shape[-1]
            grad_x = grad_normalized * rsqrt_var - (grad_normalized * normalized).mean(-1, keepdim=True) * normalized * rsqrt_var
            
            return grad_x, None, grad_weight, None
    
    def rms_norm_func(x: Tensor, normalized_shape: Tuple[int, ...], weight: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
        """RMSNorm with custom autograd for bf16/FlashAttention + gradient checkpointing compatibility."""
        return RMSNormFunction.apply(x, normalized_shape, weight, eps)

from pure_transformer.configs.model_config import TransformerConfig

# Optional Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("flash_attn not installed. Using PyTorch SDPA.")


# =============================================================================
# Core Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable params for efficiency)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
    
    def forward(self, x: Tensor) -> Tensor:
        return rms_norm_func(x, (self.hidden_size,), eps=self.eps)


def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """Precompute RoPE cos/sin cache."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    # Shape: (1, seq_len, 1, head_dim//2) for broadcasting
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to input tensor."""
    # x: (B, T, H, D) or (B, H, T, D)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP (Shazeer 2020)."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Attention(nn.Module):
    """
    Multi-Head Attention with GQA support.
    
    Features:
    - Grouped Query Attention (GQA)
    - QK normalization
    - RoPE positional embeddings
    - Flash Attention (when available)
    - KV cache for inference
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        
        assert num_heads % num_kv_heads == 0
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.use_flash = FLASH_AVAILABLE
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass.
        
        Args:
            x: (B, T, D)
            cos, sin: RoPE cache
            attention_mask: Optional mask
            kv_cache: Optional (K, V) cache for inference
            
        Returns:
            output: (B, T, D)
            new_kv_cache: Updated cache
        """
        B, T, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rotary_emb(q, cos[:, :T], sin[:, :T])
        k = apply_rotary_emb(k, cos[:, :T], sin[:, :T])
        
        # QK normalization (improves training stability)
        q = rms_norm_func(q, (self.head_dim,))
        k = rms_norm_func(k, (self.head_dim,))
        
        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        new_kv_cache = (k.clone(), v.clone())
        
        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention
        if self.use_flash and q.is_cuda:
            # Flash Attention expects (B, T, H, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_out = flash_attn_func(q, k, v, causal=True)
            attn_out = attn_out.transpose(1, 2)  # Back to (B, H, T, D)
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(attn_out)
        
        return output, new_kv_cache


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""
    
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.input_norm = RMSNorm(config.hidden_size)
        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dropout=config.attention_dropout,
        )
        
        self.post_attn_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
        )
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Attention with residual
        h, new_kv = self.attention(
            self.input_norm(x), cos, sin, attention_mask, kv_cache
        )
        x = x + h
        
        # MLP with residual
        x = x + self.mlp(self.post_attn_norm(x))
        
        return x, new_kv


# =============================================================================
# Full Model
# =============================================================================

class TransformerLM(nn.Module):
    """
    Pure Transformer Language Model.
    
    Optimized for:
    - 3-day A100 training
    - GRPO/ProRL reinforcement learning
    - Fast inference with KV cache
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Output
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Optionally tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Precompute RoPE cache
        self._init_rope_cache(config.max_seq_length * 2, config.head_dim, config.rope_theta)
        
        # Initialize weights
        self.apply(self._init_weights)
        self._init_special_weights()
    
    def _init_rope_cache(self, max_len: int, head_dim: int, theta: float):
        """Initialize RoPE cache."""
        cos, sin = precompute_rope_cache(max_len, head_dim, theta)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
    
    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            std = self.config.initializer_range
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def _init_special_weights(self):
        """Zero-init output projections for residual stability."""
        # Zero out LM head
        torch.nn.init.zeros_(self.lm_head.weight)
        
        # Zero out output projections in each layer
        for layer in self.layers:
            torch.nn.init.zeros_(layer.attention.o_proj.weight)
            torch.nn.init.zeros_(layer.mlp.down_proj.weight)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ):
        """
        Forward pass.
        
        Args:
            input_ids: (B, T) token IDs
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            kv_cache: Optional KV cache list
            use_cache: Whether to return updated cache
            
        Returns:
            If labels: (loss, logits) or (loss, logits, cache)
            Else: logits or (logits, cache)
        """
        B, T = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Get RoPE cache
        cos = self.cos_cache[:, :T].to(hidden_states.dtype)
        sin = self.sin_cache[:, :T].to(hidden_states.dtype)
        
        # Initialize cache tracking
        new_kv_cache = [] if use_cache else None
        kv_cache = kv_cache or [None] * len(self.layers)
        
        # Use gradient checkpointing during training
        use_ckpt = self.training and self.config.use_gradient_checkpointing and not use_cache
        
        # Pass through layers
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache else None
            
            if use_ckpt:
                def layer_fn(h, _layer, _cos, _sin, _mask):
                    out, _ = _layer(h, _cos, _sin, _mask, None)
                    return out
                hidden_states = checkpoint(
                    layer_fn,
                    hidden_states,
                    layer,
                    cos,
                    sin,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states, layer_new_kv = layer(
                    hidden_states, cos, sin, attention_mask, layer_kv
                )
                if use_cache:
                    new_kv_cache.append(layer_new_kv)
        
        # Final norm and LM head
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # Return appropriate outputs
        if use_cache:
            return (loss, logits, new_kv_cache) if loss is not None else (logits, new_kv_cache)
        return (loss, logits) if loss is not None else logits
    
    def setup_optimizers(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        embedding_lr_scale: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        """
        Setup optimizer with proper parameter groups.
        
        Follows best practices:
        - Different LR for embeddings
        - No weight decay for norms and biases
        """
        # Separate parameters
        decay_params = []
        no_decay_params = []
        embed_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if "embed" in name or "lm_head" in name:
                embed_params.append(param)
            elif param.dim() < 2:  # Biases and norm parameters
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": learning_rate},
            {"params": embed_params, "weight_decay": weight_decay, "lr": learning_rate * embedding_lr_scale},
        ]
        
        return torch.optim.AdamW(param_groups, betas=betas)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: (B, T) prompt tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling (not implemented yet)
            
        Returns:
            Generated token IDs (B, T + max_new_tokens)
        """
        self.eval()
        
        # Initialize with prompt
        generated = input_ids
        kv_cache = None
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            if kv_cache is None:
                logits, kv_cache = self(generated, use_cache=True)
            else:
                logits, kv_cache = self(generated[:, -1:], kv_cache=kv_cache, use_cache=True)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: TransformerConfig) -> TransformerLM:
    """Create a model from config."""
    return TransformerLM(config)
