"""
DeepSeek Sparse Attention (DSA) Transformer Implementation

Based on DeepSeek-V3.2 architecture, this implements:
1. Multi-head Latent Attention (MLA) with compressed KV cache
2. Lightning Indexer for O(L*k) sparse attention
3. Fine-grained top-k token selection

Key benefits:
- Reduces attention complexity from O(L^2) to O(L*k)
- Maintains model quality on long contexts  
- Compatible with GQA and QK normalization
- Efficient FP8-friendly indexer computation

Reference: DeepSeek-V3.2 Technical Report (2025)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from pure_transformer.configs.model_config import TransformerConfig


# =============================================================================
# DSA Configuration (extends TransformerConfig)
# =============================================================================

@dataclass  
class DSAConfig(TransformerConfig):
    """Configuration for DSA Transformer."""
    
    # Lightning Indexer settings
    index_n_heads: int = 4           # Number of indexer heads (small for efficiency)
    index_head_dim: int = 32         # Dimension per indexer head
    index_topk: int = 2048           # Top-k tokens to select
    
    # MLA settings (optional compression)
    use_mla: bool = False            # Use Multi-head Latent Attention
    kv_lora_rank: int = 512          # KV compression rank (if use_mla=True)
    q_lora_rank: int = 0             # Query compression rank (0 = no compression)
    
    # Sparse attention settings
    use_sparse_attention: bool = True
    sparse_threshold: int = 4096     # Only use sparse attention for seq_len > threshold
    
    # Indexer training
    train_indexer: bool = True       # Train indexer with KL loss
    indexer_loss_weight: float = 0.1 # Weight for indexer alignment loss


def get_dsa_config(base_config: TransformerConfig, **kwargs) -> DSAConfig:
    """Create DSA config from base TransformerConfig."""
    # Copy base config fields
    dsa_config = DSAConfig(
        model_name=base_config.model_name + "-dsa",
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_layers=base_config.num_layers,
        num_heads=base_config.num_heads,
        num_kv_heads=base_config.num_kv_heads,
        head_dim=base_config.head_dim,
        max_seq_length=base_config.max_seq_length,
        dropout=base_config.dropout,
        attention_dropout=base_config.attention_dropout,
        rope_theta=base_config.rope_theta,
        rope_scaling=base_config.rope_scaling,
        use_gradient_checkpointing=base_config.use_gradient_checkpointing,
        tie_word_embeddings=base_config.tie_word_embeddings,
        initializer_range=base_config.initializer_range,
    )
    # Override with DSA-specific settings
    for key, value in kwargs.items():
        if hasattr(dsa_config, key):
            setattr(dsa_config, key, value)
    return dsa_config


# =============================================================================
# Core Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_length: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin cache
        self._set_cos_sin_cache(max_seq_length)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_length:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to input tensor."""
    # x: (B, T, H, D) or (B, T, D)
    dim = x.shape[-1]
    x1, x2 = x[..., :dim//2], x[..., dim//2:]
    # Rotate
    cos = cos.unsqueeze(-2) if x.dim() == 4 else cos
    sin = sin.unsqueeze(-2) if x.dim() == 4 else sin
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# =============================================================================
# Lightning Indexer (DeepSeek-V3.2 Core Component)
# =============================================================================

class LightningIndexer(nn.Module):
    """
    Lightning Indexer for DeepSeek Sparse Attention.
    
    Computes index scores between query tokens and preceding tokens
    to determine which tokens to include in sparse attention.
    
    The index score is computed as:
        I_{t,s} = sum_j w_{t,j}^I * ReLU(q_{t,j}^I @ k_s^I)
    
    Features:
    - Uses fewer heads than main attention (efficient)
    - ReLU activation for hardware-friendly computation
    - Can be implemented in FP8 for maximum throughput
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_indexer_heads: int = 4,
        indexer_head_dim: int = 32,
        rope_head_dim: int = 64,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_indexer_heads
        self.head_dim = indexer_head_dim
        self.rope_head_dim = rope_head_dim
        
        # Query projection: produces q^I for each head
        self.q_proj = nn.Linear(hidden_size, num_indexer_heads * indexer_head_dim, bias=False)
        
        # Weight projection: produces per-head weights w^I
        self.w_proj = nn.Linear(hidden_size, num_indexer_heads, bias=False)
        
        # Key projection: shared across heads
        self.k_proj = nn.Linear(hidden_size, indexer_head_dim, bias=False)
        
        # Key normalization (stabilizes training)
        self.k_norm = RMSNorm(indexer_head_dim)
        
        # Scaling factor
        self.scale = indexer_head_dim ** -0.5
        
        # KV cache for inference
        self.k_cache: Optional[Tensor] = None
    
    def forward(
        self,
        hidden_states: Tensor,
        cos: Optional[Tensor] = None,
        sin: Optional[Tensor] = None,
        start_pos: int = 0,
    ) -> Tensor:
        """
        Compute index scores for sparse token selection.
        
        Args:
            hidden_states: (B, T, D) input hidden states
            cos, sin: RoPE embeddings (optional)
            start_pos: Starting position for KV cache
            
        Returns:
            index_scores: (B, T, T_kv) pairwise index scores
        """
        B, T, D = hidden_states.shape
        
        # Compute indexer queries: (B, T, num_heads, head_dim)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        
        # Compute head weights: (B, T, num_heads)
        w = self.w_proj(hidden_states) * (self.num_heads ** -0.5)
        
        # Compute indexer keys: (B, T, head_dim)
        k = self.k_proj(hidden_states)
        k = self.k_norm(k)
        
        # Apply RoPE to part of Q and K if provided
        # Note: cos/sin come from main model with shape (T, main_head_dim)
        # We need to slice to match indexer's rope_dim
        if cos is not None and sin is not None:
            rope_dim = min(self.rope_head_dim, self.head_dim)
            # Slice cos/sin to match our rope_dim (they are (T, main_head_dim))
            cos_sliced = cos[:T, :rope_dim]
            sin_sliced = sin[:T, :rope_dim]
            
            q_rope = q[..., :rope_dim]
            q_nope = q[..., rope_dim:]
            q_rope = apply_rotary_emb(q_rope, cos_sliced, sin_sliced)
            q = torch.cat([q_rope, q_nope], dim=-1)
            
            k_rope = k[..., :rope_dim]
            k_nope = k[..., rope_dim:]
            k_rope = apply_rotary_emb(k_rope.unsqueeze(2), cos_sliced, sin_sliced).squeeze(2)
            k = torch.cat([k_rope, k_nope], dim=-1)
        
        # Handle KV cache for inference
        if start_pos > 0 and self.k_cache is not None:
            k = torch.cat([self.k_cache[:B, :start_pos], k], dim=1)
        self.k_cache = k.clone()
        
        T_kv = k.shape[1]
        
        # Compute raw attention scores per head: (B, T, H, T_kv)
        # q: (B, T, H, D), k: (B, T_kv, D) -> (B, T, H, T_kv)
        raw_scores = torch.einsum('bthd,bsd->bths', q, k) * self.scale
        
        # Apply ReLU activation (as per DeepSeek paper, for efficiency)
        raw_scores = F.relu(raw_scores)
        
        # Weight by head weights and sum across heads
        # w: (B, T, H) -> (B, T, H, 1)
        # Result: (B, T, T_kv)
        index_scores = (raw_scores * w.unsqueeze(-1)).sum(dim=2)
        
        return index_scores
    
    def compute_alignment_loss(
        self,
        index_scores: Tensor,
        attention_probs: Tensor,
        causal_mask: Tensor,
    ) -> Tensor:
        """
        Compute KL divergence loss to align indexer with main attention.
        
        This trains the indexer to predict which tokens the main attention
        will focus on, enabling effective sparse selection at inference.
        """
        # Aggregate main attention across heads: (B, H, T, T) -> (B, T, T)
        if attention_probs.dim() == 4:
            target_probs = attention_probs.mean(dim=1)
        else:
            target_probs = attention_probs
        
        # L1 normalize target
        target_probs = target_probs / (target_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply causal mask and softmax to index scores
        masked_scores = index_scores.masked_fill(~causal_mask, float('-inf'))
        pred_probs = F.softmax(masked_scores, dim=-1)
        
        # KL divergence: KL(target || pred)
        kl_loss = F.kl_div(
            pred_probs.log().clamp(min=-100),
            target_probs.detach(),
            reduction='batchmean',
        )
        
        return kl_loss
    
    def clear_cache(self):
        """Clear KV cache."""
        self.k_cache = None


# =============================================================================
# DeepSeek Sparse Attention with Lightning Indexer
# =============================================================================

class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) with Lightning Indexer.
    
    Implements the full DSA mechanism from DeepSeek-V3.2:
    1. Lightning indexer computes token selection scores
    2. Top-k token selection per query position
    3. Sparse attention over selected tokens only
    
    Complexity: O(L * k) instead of O(L^2)
    """
    
    def __init__(self, config: DSAConfig, layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.top_k = config.index_topk
        self.sparse_threshold = config.sparse_threshold
        
        assert config.num_heads % config.num_kv_heads == 0
        self.num_kv_groups = config.num_heads // config.num_kv_heads
        
        # Main attention projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        
        # QK normalization (stabilizes training)
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)
        
        # Lightning Indexer
        self.indexer = LightningIndexer(
            hidden_size=config.hidden_size,
            num_indexer_heads=config.index_n_heads,
            indexer_head_dim=config.index_head_dim,
            rope_head_dim=config.head_dim // 2,  # Use half for RoPE
        )
        
        # Scaling
        self.scale = config.head_dim ** -0.5
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
        # KV cache for inference
        self.k_cache: Optional[Tensor] = None
        self.v_cache: Optional[Tensor] = None
    
    def _create_causal_mask(self, T_q: int, T_kv: int, device: torch.device) -> Tensor:
        """Create causal attention mask."""
        mask = torch.ones(T_q, T_kv, device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=T_kv - T_q + 1)
        return ~mask  # True = attend, False = mask
    
    def _select_top_k_indices(
        self,
        index_scores: Tensor,
        causal_mask: Tensor,
    ) -> Tensor:
        """
        Select top-k token indices based on index scores.
        
        Returns:
            topk_indices: (B, T, k) indices of selected tokens
        """
        B, T, T_kv = index_scores.shape
        actual_k = min(self.top_k, T_kv)
        
        # Apply causal mask
        masked_scores = index_scores.masked_fill(~causal_mask, float('-inf'))
        
        # Get top-k indices
        _, topk_indices = torch.topk(masked_scores, k=actual_k, dim=-1)
        
        return topk_indices
    
    def _sparse_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        topk_indices: Tensor,
        causal_mask: Tensor,
    ) -> Tensor:
        """
        Compute sparse attention using selected indices.
        
        Args:
            q: (B, H, T, D) queries
            k: (B, H_kv, T_kv, D) keys
            v: (B, H_kv, T_kv, D) values
            topk_indices: (B, T, k) selected indices
            causal_mask: (T, T_kv) causal mask
        """
        B, H, T, D = q.shape
        _, H_kv, T_kv, _ = k.shape
        actual_k = topk_indices.shape[-1]
        
        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Gather selected K, V for each query position
        # topk_indices: (B, T, k) -> expand for (B, H, T, k, D)
        idx = topk_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, -1, D)
        
        # k, v: (B, H, T_kv, D) -> (B, H, 1, T_kv, D) -> gather -> (B, H, T, k, D)
        k_expanded = k.unsqueeze(2).expand(-1, -1, T, -1, -1)
        v_expanded = v.unsqueeze(2).expand(-1, -1, T, -1, -1)
        
        selected_k = torch.gather(k_expanded, dim=3, index=idx)  # (B, H, T, k, D)
        selected_v = torch.gather(v_expanded, dim=3, index=idx)
        
        # Compute attention scores: (B, H, T, k)
        attn_scores = torch.einsum('bhtd,bhtkd->bhtk', q, selected_k) * self.scale
        
        # Create selection validity mask
        # A position is valid if idx <= query_position (causal)
        pos_range = torch.arange(T_kv, device=q.device)
        query_pos = torch.arange(T, device=q.device).unsqueeze(-1)
        valid_mask = torch.gather(
            (pos_range <= query_pos).unsqueeze(0).expand(B, -1, -1),
            dim=2,
            index=topk_indices,
        )  # (B, T, k)
        valid_mask = valid_mask.unsqueeze(1)  # (B, 1, T, k)
        
        attn_scores = attn_scores.masked_fill(~valid_mask, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply to values: (B, H, T, D)
        output = torch.einsum('bhtk,bhtkd->bhtd', attn_probs, selected_v)
        
        return output
    
    def _dense_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Standard dense attention (for short sequences or training).
        
        Returns both output and attention probs (for indexer training).
        """
        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Use PyTorch SDPA for efficiency
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        
        # Compute attention probs for indexer training (only during training)
        attn_probs = None
        if self.training and self.config.train_indexer:
            with torch.no_grad():
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                T = q.shape[2]
                T_kv = k.shape[2]
                causal_mask = self._create_causal_mask(T, T_kv, q.device)
                attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_probs = F.softmax(attn_scores, dim=-1)
        
        return output, attn_probs
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        start_pos: int = 0,
        use_sparse: Optional[bool] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with adaptive sparse/dense attention.
        
        For short sequences (< sparse_threshold), uses dense attention.
        For long sequences, uses sparse attention with top-k selection.
        
        Args:
            x: (B, T, D) input
            cos, sin: RoPE embeddings
            attention_mask: Optional attention mask
            start_pos: Position for KV cache (inference)
            use_sparse: Force sparse/dense mode (None = auto)
            
        Returns:
            output: (B, T, D) attention output
            indexer_loss: Optional loss for training indexer
        """
        B, T, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rotary_emb(q, cos[:T], sin[:T])
        k = apply_rotary_emb(k, cos[:T], sin[:T])
        
        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Handle KV cache for inference
        if start_pos > 0:
            if self.k_cache is not None:
                k = torch.cat([self.k_cache[:B], k], dim=1)
                v = torch.cat([self.v_cache[:B], v], dim=1)
        self.k_cache = k.clone()
        self.v_cache = v.clone()
        
        T_kv = k.shape[1]
        
        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Decide sparse vs dense
        if use_sparse is None:
            use_sparse = (T_kv > self.sparse_threshold) and self.config.use_sparse_attention
        
        indexer_loss = None
        
        if use_sparse:
            # Compute index scores
            index_scores = self.indexer(x, cos, sin, start_pos)
            
            # Create causal mask
            causal_mask = self._create_causal_mask(T, T_kv, x.device)
            
            # Select top-k indices
            topk_indices = self._select_top_k_indices(index_scores, causal_mask)
            
            # Sparse attention
            output = self._sparse_attention(q, k, v, topk_indices, causal_mask)
            
        else:
            # Dense attention (also computes attn probs for indexer training)
            output, attn_probs = self._dense_attention(q, k, v, attention_mask)
            
            # Train indexer during dense attention
            if self.training and self.config.train_indexer and attn_probs is not None:
                index_scores = self.indexer(x, cos, sin, 0)
                causal_mask = self._create_causal_mask(T, T_kv, x.device)
                indexer_loss = self.indexer.compute_alignment_loss(
                    index_scores, attn_probs, causal_mask
                )
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(output)
        
        return output, indexer_loss
    
    def clear_cache(self):
        """Clear KV cache."""
        self.k_cache = None
        self.v_cache = None
        self.indexer.clear_cache()


# =============================================================================
# SwiGLU MLP
# =============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP layer."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# =============================================================================
# DSA Transformer Block
# =============================================================================

class DSATransformerBlock(nn.Module):
    """Transformer block with DeepSeek Sparse Attention."""
    
    def __init__(self, config: DSAConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-norm
        self.input_norm = RMSNorm(config.hidden_size)
        self.post_attn_norm = RMSNorm(config.hidden_size)
        
        # Attention with DSA
        self.attention = DeepSeekSparseAttention(config, layer_idx)
        
        # MLP
        self.mlp = SwiGLUMLP(
            config.hidden_size,
            config.intermediate_size,
            config.dropout,
        )
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        start_pos: int = 0,
        use_sparse: Optional[bool] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Attention with residual
        h = self.input_norm(x)
        attn_out, indexer_loss = self.attention(h, cos, sin, attention_mask, start_pos, use_sparse)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.post_attn_norm(x))
        
        return x, indexer_loss


# =============================================================================
# Full DSA Transformer Model
# =============================================================================

class DSATransformer(nn.Module):
    """
    DeepSeek Sparse Attention Transformer.
    
    Full decoder-only transformer with:
    - Lightning Indexer for sparse attention
    - Top-k token selection (O(L*k) complexity)
    - GQA with QK normalization
    - SwiGLU MLP
    - RoPE positional embeddings
    """
    
    def __init__(self, config: DSAConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_seq_length,
            config.rope_theta,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DSATransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        start_pos: int = 0,
        use_sparse: Optional[bool] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: (B, T) token IDs
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            start_pos: Position for KV cache (inference)
            use_sparse: Force sparse/dense mode
            
        Returns:
            logits: (B, T, V) output logits
            loss: Optional cross-entropy loss
            indexer_loss: Optional indexer alignment loss
        """
        B, T = input_ids.shape
        
        # Embed tokens
        h = self.embed_tokens(input_ids)
        
        # Get RoPE embeddings
        cos, sin = self.rotary_emb(T + start_pos)
        cos = cos[start_pos:start_pos + T]
        sin = sin[start_pos:start_pos + T]
        
        # Accumulate indexer loss
        total_indexer_loss = 0.0
        num_losses = 0
        
        # Forward through layers
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                h, idx_loss = checkpoint(
                    layer, h, cos, sin, attention_mask, start_pos, use_sparse,
                    use_reentrant=False
                )
            else:
                h, idx_loss = layer(h, cos, sin, attention_mask, start_pos, use_sparse)
            
            if idx_loss is not None:
                total_indexer_loss = total_indexer_loss + idx_loss
                num_losses += 1
        
        # Final norm
        h = self.norm(h)
        
        # LM head
        logits = self.lm_head(h)
        
        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # Indexer loss
        indexer_loss = None
        if num_losses > 0:
            indexer_loss = total_indexer_loss / num_losses
        
        return logits, loss, indexer_loss
    
    def clear_cache(self):
        """Clear all KV caches."""
        for layer in self.layers:
            layer.attention.clear_cache()
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tensor:
        """Generate tokens autoregressively using sparse attention."""
        self.clear_cache()
        
        B, T = input_ids.shape
        generated = input_ids
        
        for i in range(max_new_tokens):
            # Forward pass
            if i == 0:
                logits, _, _ = self(generated, start_pos=0, use_sparse=False)
                next_logits = logits[:, -1, :]
            else:
                # Use only last token with KV cache
                logits, _, _ = self(generated[:, -1:], start_pos=T + i - 1, use_sparse=True)
                next_logits = logits[:, 0, :]
            
            # Temperature scaling
            next_logits = next_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# =============================================================================
# Factory Function
# =============================================================================

def create_dsa_transformer(
    base_config_name: str = "xlarge",
    use_sparse_attention: bool = True,
    index_topk: int = 2048,
    train_indexer: bool = True,
    **kwargs,
) -> DSATransformer:
    """
    Create a DSA Transformer from a base config.
    
    Args:
        base_config_name: One of "tiny", "small", "medium", "medium-large", "large", "xlarge"
        use_sparse_attention: Whether to use sparse attention
        index_topk: Top-k tokens to select
        train_indexer: Whether to train indexer with alignment loss
        **kwargs: Additional DSA config overrides
    """
    from pure_transformer.configs.model_config import get_model_config
    
    base_config = get_model_config(base_config_name)
    dsa_config = get_dsa_config(
        base_config,
        use_sparse_attention=use_sparse_attention,
        index_topk=index_topk,
        train_indexer=train_indexer,
        **kwargs,
    )
    
    return DSATransformer(dsa_config)


if __name__ == "__main__":
    # Test DSA Transformer
    print("Testing DSA Transformer...")
    
    from pure_transformer.configs.model_config import get_model_config
    
    base_config = get_model_config("small")
    dsa_config = get_dsa_config(base_config, index_topk=256)
    
    model = DSATransformer(dsa_config)
    print(f"Model: {dsa_config.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Test forward pass
    x = torch.randint(0, dsa_config.vocab_size, (2, 128))
    logits, loss, idx_loss = model(x, labels=x)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    if idx_loss is not None:
        print(f"Indexer Loss: {idx_loss.item():.4f}")
    
    print("DSA Transformer test passed!")
