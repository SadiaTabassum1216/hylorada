"""
Direct Attention Adaptation (DAA) Module

Implements noise filtering through learnable attention modulation.
Addresses the "Lost-in-the-Middle" phenomenon in long-context models.
Based on: Li et al., 2025 - "LoRaDA: Low-Rank Direct Attention Adaptation"

Key features:
- Learnable scaling (α) and bias (β) for attention logits
- Per-head adaptation for fine-grained control
- Minimal parameter overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DirectAttentionAdapter(nn.Module):
    """
    Direct Attention Adaptation for noise suppression in long contexts.
    
    Modifies attention weights: attn' = α * attn + β
    
    This allows the model to learn to:
    - Downweight attention to irrelevant/noisy positions (α < 1)
    - Boost attention to important positions (α > 1) 
    - Add position-independent biases (β ≠ 0)
    
    Args:
        num_heads: Number of attention heads
        per_head: If True, learn separate α, β per head
        init_alpha: Initial value for scaling factor
        init_beta: Initial value for bias
        learnable_alpha: Whether α is learnable
        learnable_beta: Whether β is learnable
    """
    
    def __init__(
        self,
        num_heads: int,
        per_head: bool = True,
        init_alpha: float = 1.0,
        init_beta: float = 0.0,
        learnable_alpha: bool = True,
        learnable_beta: bool = True,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.per_head = per_head
        
        # Determine parameter shape
        param_shape = (num_heads,) if per_head else (1,)
        
        # Learnable scaling factor α
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.full(param_shape, init_alpha))
        else:
            self.register_buffer("alpha", torch.full(param_shape, init_alpha))
        
        # Learnable bias β
        if learnable_beta:
            self.beta = nn.Parameter(torch.full(param_shape, init_beta))
        else:
            self.register_buffer("beta", torch.full(param_shape, init_beta))
        
        self.learnable_alpha = learnable_alpha
        self.learnable_beta = learnable_beta
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply direct attention adaptation to attention scores.
        
        Args:
            attention_scores: Raw attention logits [batch, heads, seq_q, seq_k]
            attention_mask: Optional mask [batch, 1, seq_q, seq_k] or [batch, heads, seq_q, seq_k]
            
        Returns:
            Adapted attention scores with same shape
        """
        # Reshape α, β for broadcasting: [1, heads, 1, 1] or [1, 1, 1, 1]
        alpha = self.alpha.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        
        # Apply adaptation: attn' = α * attn + β
        adapted_scores = alpha * attention_scores + beta
        
        # Apply mask if provided (must happen after adaptation)
        if attention_mask is not None:
            adapted_scores = adapted_scores + attention_mask
        
        return adapted_scores
    
    def get_adaptation_stats(self) -> dict:
        """Return statistics about the learned adaptation."""
        alpha_np = self.alpha.detach().cpu().numpy()
        beta_np = self.beta.detach().cpu().numpy()
        
        return {
            "alpha_mean": float(alpha_np.mean()),
            "alpha_std": float(alpha_np.std()) if len(alpha_np) > 1 else 0.0,
            "alpha_min": float(alpha_np.min()),
            "alpha_max": float(alpha_np.max()),
            "beta_mean": float(beta_np.mean()),
            "beta_std": float(beta_np.std()) if len(beta_np) > 1 else 0.0,
            "beta_min": float(beta_np.min()),
            "beta_max": float(beta_np.max()),
        }
    
    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, per_head={self.per_head}, "
            f"learnable_alpha={self.learnable_alpha}, learnable_beta={self.learnable_beta}"
        )


class ContentAwareDAA(nn.Module):
    """
    Content-Aware Direct Attention Adaptation.
    
    Unlike static DAA which learns fixed α, β per head, this module
    computes input-dependent α, β based on the hidden states.
    
    This allows the model to dynamically adjust attention based on:
    - Content type (factual vs narrative)
    - Query complexity
    - Context relevance
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        init_alpha: Initial bias for alpha (default 1.0 for identity)
        init_beta: Initial bias for beta (default 0.0)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        init_alpha: float = 1.0,
        init_beta: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Project hidden states to per-head alpha and beta
        # Use small intermediate dim for efficiency
        intermediate_dim = max(64, num_heads * 2)
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, num_heads * 2),  # alpha + beta
        )
        
        # Initialize to output ~(1.0, 0.0) at start
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.constant_(self.proj[-1].bias[:num_heads], init_alpha)
        nn.init.constant_(self.proj[-1].bias[num_heads:], init_beta)
        
        # Fallback static parameters (for compatibility)
        self.register_buffer("_static_alpha", torch.full((num_heads,), init_alpha))
        self.register_buffer("_static_beta", torch.full((num_heads,), init_beta))
    
    @property
    def alpha(self):
        """Static alpha for compatibility with existing code."""
        return self._static_alpha
    
    @property
    def beta(self):
        """Static beta for compatibility with existing code."""
        return self._static_beta
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply content-aware attention adaptation.
        
        Args:
            attention_scores: [batch, heads, seq_q, seq_k]
            hidden_states: [batch, seq, hidden_size] - input for computing α, β
            attention_mask: Optional attention mask
            
        Returns:
            Adapted attention scores
        """
        batch_size, num_heads, seq_q, seq_k = attention_scores.shape
        
        if hidden_states is not None:
            # Ensure hidden_states matches input device/dtype
            if hidden_states.device != self.proj[0].weight.device:
                hidden_states = hidden_states.to(self.proj[0].weight.device)
            if hidden_states.dtype != self.proj[0].weight.dtype:
                hidden_states = hidden_states.to(self.proj[0].weight.dtype)
            
            # Pool over sequence dimension (mean pooling)
            pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
            
            # Compute content-aware alpha and beta
            ab = self.proj(pooled)  # [batch, num_heads * 2]
            alpha = ab[:, :self.num_heads].view(batch_size, num_heads, 1, 1)
            beta = ab[:, self.num_heads:].view(batch_size, num_heads, 1, 1)
        else:
            # Fallback to static parameters
            alpha = self._static_alpha.view(1, -1, 1, 1)
            beta = self._static_beta.view(1, -1, 1, 1)
        
        # Apply adaptation: attn' = α * attn + β
        adapted_scores = alpha * attention_scores + beta
        
        if attention_mask is not None:
            adapted_scores = adapted_scores + attention_mask
        
        return adapted_scores
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_heads={self.num_heads}"


class PositionalDAA(nn.Module):
    """
    Extended DAA with position-aware adaptation.
    
    Learns position-dependent biases to address the "Lost-in-the-Middle" 
    phenomenon by explicitly modulating attention based on position.
    
    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        num_buckets: Number of relative position buckets
        per_head: Learn separate parameters per head
    """
    
    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 32768,
        num_buckets: int = 64,
        per_head: bool = True,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.per_head = per_head
        
        # Base DAA (scalar adaptation)
        self.base_daa = DirectAttentionAdapter(
            num_heads=num_heads,
            per_head=per_head,
        )
        
        # Position-dependent bias
        head_dim = num_heads if per_head else 1
        self.position_bias = nn.Embedding(num_buckets, head_dim)
        nn.init.zeros_(self.position_bias.weight)
    
    @property
    def alpha(self):
        """Delegate alpha to base_daa for compatibility."""
        return self.base_daa.alpha
    
    @property
    def beta(self):
        """Delegate beta to base_daa for compatibility."""
        return self.base_daa.beta
    
    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert relative positions to bucket indices.
        Uses logarithmic bucketing for distant positions.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_seq_len
        
        # Half buckets for negative, half for positive
        half_buckets = num_buckets // 2
        
        # Separate positive and negative
        relative_buckets = torch.zeros_like(relative_position)
        negative_mask = relative_position < 0
        relative_position = relative_position.abs()
        
        # Linear buckets for close positions
        max_exact = half_buckets // 2
        is_small = relative_position < max_exact
        
        # Logarithmic buckets for distant positions
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (half_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, half_buckets - 1),
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        relative_buckets = torch.where(negative_mask, half_buckets + relative_buckets, relative_buckets)
        
        return relative_buckets.clamp(0, num_buckets - 1)
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply position-aware attention adaptation.
        
        Args:
            attention_scores: [batch, heads, seq_q, seq_k]
            attention_mask: Optional attention mask
            
        Returns:
            Adapted attention scores
        """
        batch_size, num_heads, seq_q, seq_k = attention_scores.shape
        
        # Compute relative positions
        positions_q = torch.arange(seq_q, device=attention_scores.device)
        positions_k = torch.arange(seq_k, device=attention_scores.device)
        relative_positions = positions_q.unsqueeze(1) - positions_k.unsqueeze(0)
        
        # Get bucket indices
        buckets = self._relative_position_bucket(relative_positions)  # [seq_q, seq_k]
        
        # Look up position biases
        pos_bias = self.position_bias(buckets)  # [seq_q, seq_k, head_dim]
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)  # [1, head_dim, seq_q, seq_k]
        
        # Apply base DAA
        adapted_scores = self.base_daa(attention_scores, attention_mask=None)
        
        # Add position bias
        adapted_scores = adapted_scores + pos_bias
        
        # Apply mask
        if attention_mask is not None:
            adapted_scores = adapted_scores + attention_mask
        
        return adapted_scores


def apply_daa_to_attention(
    attention_module: nn.Module,
    num_heads: int,
    config_per_head: bool = True,
    config_init_alpha: float = 1.0,
    config_init_beta: float = 0.0,
) -> Tuple[nn.Module, DirectAttentionAdapter]:
    """
    Apply DAA to an attention module by wrapping its forward method.
    
    This is a utility function to inject DAA into existing attention implementations.
    
    Args:
        attention_module: The attention module to wrap
        num_heads: Number of attention heads
        config_per_head: Whether to use per-head adaptation
        config_init_alpha: Initial alpha value
        config_init_beta: Initial beta value
        
    Returns:
        Tuple of (modified module, DAA adapter)
    """
    daa = DirectAttentionAdapter(
        num_heads=num_heads,
        per_head=config_per_head,
        init_alpha=config_init_alpha,
        init_beta=config_init_beta,
    )
    
    # Store DAA as a submodule
    attention_module.daa_adapter = daa
    
    return attention_module, daa


def get_daa_params(model: nn.Module) -> list:
    """Get all DAA parameters from a model."""
    params = []
    for module in model.modules():
        if isinstance(module, (DirectAttentionAdapter, PositionalDAA, ContentAwareDAA)):
            params.extend(module.parameters())
    return params


def count_daa_params(model: nn.Module) -> int:
    """Count total DAA parameters."""
    return sum(p.numel() for p in get_daa_params(model))
