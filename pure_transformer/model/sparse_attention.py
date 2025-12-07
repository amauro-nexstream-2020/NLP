"""
DeepSeek Sparse Attention (DSA) Implementation

Based on DeepSeek-V3.2 architecture:
- Lightning indexer for efficient token selection
- Fine-grained token selection mechanism
- Top-k sparse attention pattern

Key benefits:
- Reduces attention complexity from O(L^2) to O(L*k)
- Maintains model quality on long contexts
- Compatible with MLA (Multi-head Latent Attention)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LightningIndexer(nn.Module):
    """
    Lightning Indexer for DeepSeek Sparse Attention.
    
    Computes index scores between query tokens and preceding tokens
    to determine which tokens to select for attention.
    
    The index score is computed as:
        I_{t,s} = sum_j w_{t,j}^I * ReLU(q_{t,j}^I @ k_s^I)
    
    Uses fewer heads and can be implemented in FP8 for efficiency.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_indexer_heads: int = 4,
        indexer_head_dim: int = 32,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_indexer_heads
        self.head_dim = indexer_head_dim
        
        # Query projection for indexer (produces q^I and w^I)
        self.q_proj = nn.Linear(hidden_size, num_indexer_heads * indexer_head_dim, bias=False)
        self.w_proj = nn.Linear(hidden_size, num_indexer_heads, bias=False)  # Per-head weights
        
        # Key projection for indexer
        self.k_proj = nn.Linear(hidden_size, indexer_head_dim, bias=False)  # Shared across heads
    
    def forward(
        self,
        hidden_states: Tensor,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Compute index scores for token selection.
        
        Args:
            hidden_states: (B, T, D) input hidden states
            position_offset: Offset for cached positions
            
        Returns:
            index_scores: (B, T, T) pairwise index scores
        """
        B, T, D = hidden_states.shape
        
        # Compute indexer queries: (B, T, num_heads, head_dim)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        
        # Compute head weights: (B, T, num_heads)
        w = self.w_proj(hidden_states)
        
        # Compute indexer keys: (B, T, head_dim)
        k = self.k_proj(hidden_states)
        
        # Compute raw attention scores per head: (B, T, num_heads, T)
        # q: (B, T, H, D) @ k.T: (B, D, T) -> (B, T, H, T)
        raw_scores = torch.einsum('bthd,bsd->bths', q, k)
        
        # Apply ReLU activation (as per DeepSeek paper)
        raw_scores = F.relu(raw_scores)
        
        # Weight by head weights and sum across heads
        # w: (B, T, H) -> (B, T, H, 1)
        # raw_scores: (B, T, H, T)
        index_scores = (raw_scores * w.unsqueeze(-1)).sum(dim=2)  # (B, T, T)
        
        return index_scores
    
    def compute_indexer_loss(
        self,
        index_scores: Tensor,
        attention_probs: Tensor,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute KL divergence loss to align indexer with main attention.
        
        Args:
            index_scores: (B, T, T) raw index scores
            attention_probs: (B, H, T, T) main attention probabilities (aggregated)
            causal_mask: Optional causal mask
            
        Returns:
            kl_loss: Scalar KL divergence loss
        """
        B, T, _ = index_scores.shape
        
        # Aggregate main attention across heads
        if attention_probs.dim() == 4:
            # (B, H, T, T) -> (B, T, T)
            target_probs = attention_probs.mean(dim=1)
        else:
            target_probs = attention_probs
        
        # L1 normalize target along sequence dimension
        target_probs = target_probs / (target_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply causal mask
        if causal_mask is not None:
            index_scores = index_scores.masked_fill(~causal_mask, float('-inf'))
        
        # Softmax over index scores
        pred_probs = F.softmax(index_scores, dim=-1)
        
        # KL divergence: KL(target || pred)
        kl_loss = F.kl_div(
            pred_probs.log().clamp(min=-100),
            target_probs,
            reduction='batchmean',
        )
        
        return kl_loss


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) with Lightning Indexer.
    
    Implements the full DSA mechanism:
    1. Lightning indexer computes token selection scores
    2. Top-k token selection per query
    3. Sparse attention over selected tokens
    
    Features:
    - Reduces complexity from O(L^2) to O(L*k)
    - Maintains quality on long sequences
    - Compatible with GQA
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        top_k: int = 2048,
        num_indexer_heads: int = 4,
        indexer_head_dim: int = 32,
        dropout: float = 0.0,
        use_dense_fallback: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.dropout = dropout
        self.use_dense_fallback = use_dense_fallback
        
        assert num_heads % num_kv_heads == 0
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Main attention projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Lightning indexer
        self.indexer = LightningIndexer(
            hidden_size=hidden_size,
            num_indexer_heads=num_indexer_heads,
            indexer_head_dim=indexer_head_dim,
        )
        
        # Scaling factor
        self.scale = head_dim ** -0.5
    
    def _create_causal_mask(self, T: int, device: torch.device) -> Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return ~mask  # True = attend, False = mask
    
    def _select_top_k(
        self,
        index_scores: Tensor,
        k: Tensor,
        v: Tensor,
        causal_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Select top-k tokens based on index scores.
        
        Args:
            index_scores: (B, T, T) index scores
            k: (B, T, H_kv, D) key tensor
            v: (B, T, H_kv, D) value tensor
            causal_mask: (T, T) causal mask
            
        Returns:
            selected_k: (B, T, k, H_kv, D)
            selected_v: (B, T, k, H_kv, D)
            selection_mask: (B, T, k) bool mask for valid selections
        """
        B, T, _ = index_scores.shape
        actual_k = min(self.top_k, T)
        
        # Apply causal mask to scores
        masked_scores = index_scores.masked_fill(~causal_mask, float('-inf'))
        
        # Get top-k indices per query position
        _, top_indices = torch.topk(masked_scores, k=actual_k, dim=-1)  # (B, T, k)
        
        # Gather selected k, v
        # top_indices: (B, T, k) -> expand for gathering
        gather_idx = top_indices.unsqueeze(-1).unsqueeze(-1)  # (B, T, k, 1, 1)
        gather_idx = gather_idx.expand(-1, -1, -1, k.shape[2], k.shape[3])  # (B, T, k, H_kv, D)
        
        # Expand k, v for gathering
        k_expanded = k.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B, T, T, H_kv, D)
        v_expanded = v.unsqueeze(1).expand(-1, T, -1, -1, -1)
        
        selected_k = torch.gather(k_expanded, dim=2, index=gather_idx)  # (B, T, k, H_kv, D)
        selected_v = torch.gather(v_expanded, dim=2, index=gather_idx)
        
        # Create selection mask (valid = index is within causal range)
        position_range = torch.arange(T, device=index_scores.device)
        valid_positions = position_range.unsqueeze(0) <= position_range.unsqueeze(1)
        selection_mask = torch.gather(
            valid_positions.unsqueeze(0).expand(B, -1, -1),
            dim=2,
            index=top_indices,
        )
        
        return selected_k, selected_v, selection_mask
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        return_indexer_loss: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor]]:
        """
        Forward pass with sparse attention.
        
        For short sequences (< 2 * top_k), uses dense attention.
        For long sequences, uses sparse attention with top-k selection.
        """
        B, T, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        from pure_transformer.model.transformer import apply_rotary_emb
        q = apply_rotary_emb(q, cos[:, :T], sin[:, :T])
        k = apply_rotary_emb(k, cos[:, :T], sin[:, :T])
        
        # QK normalization
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        
        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        new_kv_cache = (k.clone(), v.clone())
        
        T_kv = k.shape[1]
        
        # Decide sparse vs dense
        use_sparse = (T_kv > 2 * self.top_k) and not self.use_dense_fallback
        
        indexer_loss = None
        
        if use_sparse:
            # Compute index scores
            index_scores = self.indexer(x)
            
            # Create causal mask
            causal_mask = self._create_causal_mask(T_kv, x.device)[:T, :T_kv]
            
            # Select top-k tokens
            selected_k, selected_v, selection_mask = self._select_top_k(
                index_scores, k, v, causal_mask
            )
            
            # Transpose for attention
            q = q.transpose(1, 2)  # (B, H, T, D)
            selected_k = selected_k.transpose(2, 3)  # (B, T, H_kv, k, D)
            selected_v = selected_v.transpose(2, 3)
            
            # Expand KV heads for GQA
            if self.num_kv_groups > 1:
                selected_k = selected_k.repeat_interleave(self.num_kv_groups, dim=2)
                selected_v = selected_v.repeat_interleave(self.num_kv_groups, dim=2)
            
            # Compute sparse attention
            # q: (B, H, T, D), selected_k: (B, T, H, k, D)
            # Need: q @ k.T for each query position
            attn_scores = torch.einsum('bhqd,bqhkd->bhqk', q, selected_k) * self.scale
            
            # Apply selection mask
            attn_scores = attn_scores.masked_fill(
                ~selection_mask.unsqueeze(1), float('-inf')
            )
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            
            # Apply attention to values
            # attn_probs: (B, H, T, k), selected_v: (B, T, H, k, D)
            attn_out = torch.einsum('bhqk,bqhkd->bhqd', attn_probs, selected_v)
            
        else:
            # Dense attention for short sequences
            q = q.transpose(1, 2)  # (B, H, T, D)
            k = k.transpose(1, 2)  # (B, H_kv, T, D)
            v = v.transpose(1, 2)
            
            # Expand KV heads for GQA
            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)
            
            # Standard scaled dot-product attention
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            
            # Compute indexer loss during training
            if return_indexer_loss and self.training:
                with torch.no_grad():
                    # Get attention probs for indexer alignment
                    attn_scores = (q @ k.transpose(-2, -1)) * self.scale
                    causal_mask = self._create_causal_mask(T_kv, x.device)[:T, :T_kv]
                    attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
                    attn_probs = F.softmax(attn_scores, dim=-1)
                
                index_scores = self.indexer(x)
                indexer_loss = self.indexer.compute_indexer_loss(
                    index_scores, attn_probs, causal_mask
                )
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(attn_out)
        
        return output, new_kv_cache, indexer_loss


class SparseTransformerBlock(nn.Module):
    """Transformer block with DeepSeek Sparse Attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        top_k: int = 2048,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        from pure_transformer.model.transformer import RMSNorm, SwiGLUMLP
        
        self.input_norm = RMSNorm(hidden_size)
        self.attention = DeepSeekSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            top_k=top_k,
            dropout=dropout,
        )
        
        self.post_attn_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLUMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        return_indexer_loss: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor]]:
        # Attention with residual
        h, new_kv, indexer_loss = self.attention(
            self.input_norm(x), cos, sin, attention_mask, kv_cache, return_indexer_loss
        )
        x = x + h
        
        # MLP with residual
        x = x + self.mlp(self.post_attn_norm(x))
        
        return x, new_kv, indexer_loss
