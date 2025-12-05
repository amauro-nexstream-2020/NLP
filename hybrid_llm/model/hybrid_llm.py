"""
Hybrid Mamba-Gated Attention LLM

This module implements a state-of-the-art hybrid architecture combining:
1. Mamba-2 SSM blocks for efficient sequence modeling (O(N) complexity)
2. Gated Attention blocks for precise retrieval (sparsified attention)
3. SwiGLU MLP for improved gradient flow
4. RoPE for relative positional encoding
5. GQA for memory-efficient attention

Architecture inspired by:
- Mamba-2 (State Space Duality)
- Qwen-2.5 (Gated Attention)
- Jamba/Zamba (Hybrid SSM-Attention)
- nanochat (training efficiency)
"""

import math
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# Conditional imports for optional dependencies
try:
    from mamba_ssm import Mamba2, Mamba
    MAMBA_AVAILABLE = True
    MAMBA2_AVAILABLE = True
except ImportError:
    try:
        from mamba_ssm import Mamba
        MAMBA_AVAILABLE = True
        MAMBA2_AVAILABLE = False
    except ImportError:
        MAMBA_AVAILABLE = False
        MAMBA2_AVAILABLE = False
        print("Warning: mamba_ssm not installed. Using fallback SSM implementation.")

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not installed. Using PyTorch SDPA.")


# =============================================================================
# Helper Functions
# =============================================================================

def _get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "gelu":
        return F.gelu
    elif activation == "silu" or activation == "swish":
        return F.silu
    elif activation == "relu":
        return F.relu
    elif activation == "relu2":
        return lambda x: F.relu(x).square()
    else:
        raise ValueError(f"Unknown activation: {activation}")


# =============================================================================
# RMSNorm (No learnable parameters for efficiency)
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Unlike LayerNorm, RMSNorm:
    - Does not center the mean (faster)
    - Has no learnable affine parameters in this implementation
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
    
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.hidden_size,), eps=self.eps)


class RMSNormWithWeight(nn.Module):
    """RMSNorm with learnable scale parameter."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: Tensor) -> Tensor:
        output = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        return output * self.weight


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================

def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """
    Precompute rotary position embeddings.
    
    Args:
        seq_len: Maximum sequence length
        head_dim: Dimension per attention head
        base: Base for frequency computation
        device: Target device
        dtype: Target dtype
    
    Returns:
        cos, sin tensors of shape (1, seq_len, 1, head_dim // 2)
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    
    # Position indices
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    # Outer product for frequencies
    freqs = torch.outer(t, inv_freq)
    
    # Compute cos and sin
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    
    # Reshape for broadcasting: (1, seq_len, 1, head_dim // 2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply rotary position embeddings to input tensor.
    
    Args:
        x: Input tensor of shape (B, T, H, D)
        cos: Cosine tensor of shape (1, T, 1, D//2)
        sin: Sine tensor of shape (1, T, 1, D//2)
    
    Returns:
        Rotated tensor of same shape as input
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    
    # Rotate pairs of dimensions
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


# =============================================================================
# Gated Attention Mechanism (NeurIPS 2025)
# =============================================================================

class GatedAttention(nn.Module):
    """
    Gated Grouped-Query Attention (G-GQA).
    
    Key innovations:
    1. Query-dependent sigmoid gate after attention output
    2. Grouped-Query Attention for KV memory efficiency
    3. QK normalization for training stability
    4. RoPE for relative positional encoding
    
    The gate allows the model to suppress irrelevant attention outputs,
    eliminating the "attention sink" problem.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash: bool = True,
        gate_bias: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE
        
        # Validate GQA configuration
        assert num_heads % num_kv_heads == 0
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # Gating projection: Q -> gate scores
        self.gate_proj = nn.Linear(self.head_dim, self.head_dim, bias=gate_bias)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass for gated attention.
        
        Args:
            x: Input tensor (B, T, D)
            cos, sin: RoPE embeddings
            attention_mask: Optional causal mask
            kv_cache: Optional (key, value) cache for inference
        
        Returns:
            output: Attention output (B, T, D)
            new_kv_cache: Updated KV cache if caching enabled
        """
        B, T, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rotary_emb(q, cos[:, :T], sin[:, :T])
        k = apply_rotary_emb(k, cos[:, :T], sin[:, :T])
        
        # QK normalization (following nanochat)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        
        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        
        # Always return new cache for incremental decoding
        new_kv_cache = (k.clone(), v.clone())
        
        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Save q in (B, T, H, D) form for gating before any transpose
        q_for_gate = q.transpose(1, 2)  # (B, T, H, D) - before SDPA reshape
        
        # Compute attention
        if self.use_flash and q.is_cuda:
            # FlashAttention expects (B, T, H, D)
            q_flash = q.transpose(1, 2)
            k_flash = k.transpose(1, 2)
            v_flash = v.transpose(1, 2)
            attn_out = flash_attn_func(q_flash, k_flash, v_flash, causal=True)
            attn_out = attn_out.transpose(1, 2)  # Back to (B, H, T, D)
        else:
            # Standard scaled dot-product attention
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        
        # Compute gate from query (using saved q_for_gate)
        # attn_out: (B, H, T, D) -> gate on per-head basis
        gate = torch.sigmoid(self.gate_proj(q_for_gate))  # (B, T, H, D)
        gate = gate.transpose(1, 2)  # (B, H, T, D)
        
        # Apply gate
        gated_out = attn_out * gate
        
        # Reshape and project output
        gated_out = gated_out.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(gated_out)
        
        return output, new_kv_cache


# =============================================================================
# Mamba-2 SSM Block (with fallback)
# =============================================================================

class MambaBlock(nn.Module):
    """
    Mamba-2 State Space Model block.
    
    Uses the official mamba_ssm implementation if available,
    otherwise falls back to a simplified SSM.
    """
    
    def __init__(
        self,
        hidden_size: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.d_state = d_state
        
        if MAMBA_AVAILABLE:
            # Use Mamba (v1) for better stability - Mamba2 has Triton compatibility issues
            self.mamba = Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: simplified SSM-like layer
            self.mamba = self._create_fallback_ssm(hidden_size, d_state, d_conv, expand)
    
    def _create_fallback_ssm(self, hidden_size, d_state, d_conv, expand):
        """Create a simplified SSM-like fallback using convolution + gating."""
        inner_dim = hidden_size * expand
        
        class FallbackSSM(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(hidden_size, inner_dim * 2, bias=False)
                self.conv = nn.Conv1d(inner_dim * 2, inner_dim * 2, d_conv, 
                                       padding=d_conv - 1, groups=inner_dim * 2)
                self.down_proj = nn.Linear(inner_dim, hidden_size, bias=False)
            
            def forward(self, x):
                B, T, D = x.shape
                out = self.up_proj(x)  # (B, T, inner*2)
                out = out.transpose(1, 2)  # (B, inner*2, T)
                out = self.conv(out)[..., :T]  # trim causal padding
                out = out.transpose(1, 2)  # (B, T, inner*2)
                gate, val = out.chunk(2, dim=-1)
                out = F.silu(gate) * val  # gated activation
                out = self.down_proj(out)
                return out
        
        return FallbackSSM()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through Mamba block.
        
        Args:
            x: Input tensor (B, T, D)
        
        Returns:
            Output tensor (B, T, D)
        """
        if MAMBA_AVAILABLE:
            return self.mamba(x)
        else:
            # Fallback path (uses the FallbackSSM module)
            return self.mamba(x)


# =============================================================================
# SwiGLU MLP
# =============================================================================

class SwiGLUMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    SwiGLU = Swish(x * W_gate) * (x * W_up)
    
    More parameter-efficient than standard FFN while 
    maintaining expressivity.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SwiGLU MLP."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out)


# =============================================================================
# Hybrid Block (Mamba or Attention + MLP)
# =============================================================================

class HybridBlock(nn.Module):
    """
    Single block of the Hybrid architecture.
    
    Can be either:
    - Mamba block (SSM for efficient sequence modeling)
    - Gated Attention block (for precise retrieval)
    
    Both followed by SwiGLU MLP.
    """
    
    def __init__(
        self,
        config,
        layer_idx: int,
        is_attention_layer: bool,
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.is_attention_layer = is_attention_layer
        self.hidden_size = config.hidden_size
        
        # Pre-normalization
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Core block: either Attention or Mamba
        if is_attention_layer:
            self.core = GatedAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dropout=config.attention_dropout,
                use_flash=config.use_flash_attention,
                gate_bias=config.gate_bias,
            )
        else:
            self.core = MambaBlock(
                hidden_size=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            )
        
        # MLP
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout,
        )
    
    def forward(
        self,
        x: Tensor,
        cos: Optional[Tensor] = None,
        sin: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass through hybrid block.
        
        Args:
            x: Input tensor (B, T, D)
            cos, sin: RoPE embeddings (only for attention layers)
            attention_mask: Optional attention mask
            kv_cache: Optional KV cache for attention layers
        
        Returns:
            output: Block output (B, T, D)
            new_kv_cache: Updated KV cache if attention layer
        """
        # Pre-norm + core block
        normed = self.input_norm(x)
        
        new_kv_cache = None
        if self.is_attention_layer:
            core_out, new_kv_cache = self.core(
                normed, cos, sin, attention_mask, kv_cache
            )
        else:
            core_out = self.core(normed)
        
        # Residual connection
        x = x + core_out
        
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.post_attention_norm(x))
        
        return x, new_kv_cache


# =============================================================================
# Complete Hybrid LLM
# =============================================================================

class HybridLLM(nn.Module):
    """
    Hybrid Mamba-Gated Attention Language Model.
    
    Architecture:
    - Token Embedding
    - N hybrid blocks (mix of Mamba and Attention)
    - Final RMSNorm
    - LM Head (untied from embeddings)
    
    The model interleaves Mamba blocks with Gated Attention blocks
    in a configurable ratio (default 3:1) for optimal efficiency
    and retrieval capability.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Embedding normalization (following nanochat)
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Build layers
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            is_attention = layer_idx in config.attention_layer_indices
            self.layers.append(HybridBlock(config, layer_idx, is_attention))
        
        # Final normalization
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head (untied from embeddings for better performance)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Precompute RoPE cache
        self._init_rope_cache()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output layers
        self._init_output_layers()
    
    def _init_rope_cache(self):
        """Initialize rotary position embedding cache."""
        max_seq = self.config.max_position_embeddings * 2  # 2x buffer
        cos, sin = precompute_rope_cache(
            max_seq,
            self.config.head_dim,
            base=self.config.rope_theta,
            dtype=torch.bfloat16,
        )
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
    
    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            # Fan-in/out scaling (from nanochat)
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
    
    def _init_output_layers(self):
        """Special initialization for output projections."""
        # Zero init for LM head (following nanochat)
        nn.init.zeros_(self.lm_head.weight)
        
        # Zero init for residual projections
        for layer in self.layers:
            if hasattr(layer.core, 'o_proj'):
                nn.init.zeros_(layer.core.o_proj.weight)
            nn.init.zeros_(layer.mlp.down_proj.weight)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs (B, T)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            kv_cache: Optional KV cache list
            use_cache: Whether to return updated KV cache
            return_dict: Whether to return a dict
        
        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits or (logits, kv_cache)
        """
        B, T = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_norm(hidden_states)
        
        # Get RoPE embeddings
        cos = self.cos_cache[:, :T].to(hidden_states.dtype)
        sin = self.sin_cache[:, :T].to(hidden_states.dtype)
        
        # Initialize KV cache tracking
        new_kv_cache = [] if use_cache else None
        kv_cache = kv_cache or [None] * len(self.layers)
        
        # Pass through layers
        attn_layer_idx = 0
        use_ckpt = self.training and self.config.use_gradient_checkpointing and not use_cache
        for layer_idx, layer in enumerate(self.layers):
            if layer.is_attention_layer:
                layer_kv = kv_cache[attn_layer_idx] if kv_cache else None
                if use_ckpt:
                    # Use functools.partial to properly capture variables for checkpoint
                    def attn_ckpt_fn(h, _layer, _cos, _sin, _mask):
                        out, _ = _layer(h, _cos, _sin, _mask, None)
                        return out
                    hidden_states = checkpoint(
                        attn_ckpt_fn,
                        hidden_states,
                        layer,
                        cos,
                        sin,
                        attention_mask,
                        use_reentrant=False,
                    )
                    layer_new_kv = None
                else:
                    hidden_states, layer_new_kv = layer(
                        hidden_states, cos, sin, attention_mask, layer_kv
                    )
                if use_cache and layer_new_kv is not None:
                    new_kv_cache.append(layer_new_kv)
                attn_layer_idx += 1
            else:
                if use_ckpt:
                    def mamba_ckpt_fn(h, _layer):
                        out, _ = _layer(h)
                        return out
                    hidden_states = checkpoint(
                        mamba_ckpt_fn,
                        hidden_states,
                        layer,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, _ = layer(hidden_states)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        if use_cache:
            return (loss, logits, new_kv_cache) if loss is not None else (logits, new_kv_cache)
        
        return (loss, logits) if loss is not None else logits
    
    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs per token for training."""
        n_params = sum(p.numel() for p in self.parameters())
        n_emb = self.embed_tokens.weight.numel()
        
        # Rough estimate: 6N for forward + backward
        # Plus attention: 12 * L * H * D * T
        L = self.config.num_hidden_layers
        H = self.config.num_attention_heads
        D = self.config.head_dim
        T = self.config.max_position_embeddings
        
        flops = 6 * (n_params - n_emb) + 12 * L * H * D * T
        return flops
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial tokens (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated token IDs (B, T + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Apply filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def create_model(config) -> HybridLLM:
    """Create a HybridLLM model from config."""
    return HybridLLM(config)
