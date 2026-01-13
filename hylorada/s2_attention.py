"""
Shifted Sparse Attention (S²-Attn) Module

Implements efficient long-context attention through group-wise computation.
Based on: Chen et al., 2024 - "LongLoRA: Efficient Fine-tuning of Long-Context LLMs"

Key features:
- Splits sequence into groups for memory efficiency
- Shifts groups to maintain cross-group information flow
- Reduces memory from O(n²) to O(n × group_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ShiftedSparseAttention(nn.Module):
    """
    Shifted Sparse Attention (S²-Attn) for efficient long-context processing.
    
    Instead of computing full O(n²) attention, splits the sequence into groups
    and computes attention within each group. To maintain information flow
    between groups, alternating layers shift the group boundaries.
    
    Memory: O(n × group_size) instead of O(n²)
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: hidden_size // num_heads)
        group_size: Number of tokens per attention group
        shift_ratio: Fraction of group to shift (0.5 = half group)
        dropout: Attention dropout probability
        layer_idx: Layer index (determines if this layer shifts)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        group_size: int = 2048,
        shift_ratio: float = 0.5,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.group_size = group_size
        self.shift_ratio = shift_ratio
        self.shift_amount = int(group_size * shift_ratio)
        self.layer_idx = layer_idx
        self.should_shift = (layer_idx % 2 == 1)  # Odd layers shift
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        daa_adapter: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute shifted sparse attention.
        
        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim]
            key: Key tensor [batch, num_heads, seq_len, head_dim]
            value: Value tensor [batch, num_heads, seq_len, head_dim]
            attention_mask: Optional mask [batch, 1, seq_len, seq_len]
            daa_adapter: Optional DirectAttentionAdapter for noise filtering
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [batch, num_heads, seq_len, head_dim]
            - attention_weights: Optional attention weights for analysis
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # For short sequences, use standard attention
        if seq_len <= self.group_size:
            return self._standard_attention(
                query, key, value, attention_mask, daa_adapter
            )
        
        # Apply shift for alternating layers
        if self.should_shift:
            query, key, value = self._shift_tensors(query, key, value)
            if attention_mask is not None:
                attention_mask = self._shift_mask(attention_mask)
        
        # Compute group-wise attention
        output = self._grouped_attention(
            query, key, value, attention_mask, daa_adapter
        )
        
        # Reverse shift
        if self.should_shift:
            output = self._unshift_tensor(output)
        
        return output, None
    
    def _shift_tensors(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shift tensors for cross-group information flow."""
        # Roll along sequence dimension
        query = torch.roll(query, shifts=-self.shift_amount, dims=2)
        key = torch.roll(key, shifts=-self.shift_amount, dims=2)
        value = torch.roll(value, shifts=-self.shift_amount, dims=2)
        return query, key, value
    
    def _shift_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Shift attention mask to match shifted sequences."""
        # Roll both query and key dimensions
        mask = torch.roll(mask, shifts=-self.shift_amount, dims=2)
        mask = torch.roll(mask, shifts=-self.shift_amount, dims=3)
        return mask
    
    def _unshift_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the shift operation."""
        return torch.roll(x, shifts=self.shift_amount, dims=2)
    
    def _grouped_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        daa_adapter: Optional[nn.Module],
    ) -> torch.Tensor:
        """Compute attention within groups."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Calculate number of groups (pad if necessary)
        num_groups = (seq_len + self.group_size - 1) // self.group_size
        padded_len = num_groups * self.group_size
        
        # Pad sequences if needed
        if padded_len > seq_len:
            pad_len = padded_len - seq_len
            query = F.pad(query, (0, 0, 0, pad_len))
            key = F.pad(key, (0, 0, 0, pad_len))
            value = F.pad(value, (0, 0, 0, pad_len))
            if attention_mask is not None:
                # Pad with large negative values to mask padding
                attention_mask = F.pad(
                    attention_mask, 
                    (0, pad_len, 0, pad_len), 
                    value=-1e9
                )
        
        # Reshape into groups: [batch, heads, num_groups, group_size, head_dim]
        query = query.view(batch_size, num_heads, num_groups, self.group_size, head_dim)
        key = key.view(batch_size, num_heads, num_groups, self.group_size, head_dim)
        value = value.view(batch_size, num_heads, num_groups, self.group_size, head_dim)
        
        # Compute attention within each group
        # [batch, heads, groups, group_size, group_size]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply group-wise mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(
                batch_size, 1, num_groups, self.group_size, self.group_size
            )
            attn_scores = attn_scores + mask
        
        # Apply DAA if provided (reshape for compatibility)
        if daa_adapter is not None:
            # Temporarily flatten groups for DAA
            orig_shape = attn_scores.shape
            attn_scores = attn_scores.view(
                batch_size, num_heads, -1, self.group_size
            )
            attn_scores = daa_adapter(attn_scores)
            attn_scores = attn_scores.view(orig_shape)
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, value)
        
        # Reshape back: [batch, heads, padded_len, head_dim]
        output = output.view(batch_size, num_heads, padded_len, head_dim)
        
        # Remove padding
        if padded_len > seq_len:
            output = output[:, :, :seq_len, :]
        
        return output
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        daa_adapter: Optional[nn.Module],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention for short sequences."""
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Apply DAA
        if daa_adapter is not None:
            attn_scores = daa_adapter(attn_scores)
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


class S2AttentionWrapper(nn.Module):
    """
    Wrapper to replace a model's attention with S²-Attn.
    
    This wrapper intercepts the attention computation and applies
    shifted sparse attention while preserving the original interface.
    
    Args:
        base_attention: Original attention module
        config: S²-Attn configuration
        layer_idx: Layer index for shift pattern
    """
    
    def __init__(
        self,
        base_attention: nn.Module,
        hidden_size: int,
        num_heads: int,
        group_size: int = 2048,
        shift_ratio: float = 0.5,
        layer_idx: int = 0,
    ):
        super().__init__()
        
        self.base_attention = base_attention
        
        # Create S²-Attn module
        self.s2_attn = ShiftedSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            group_size=group_size,
            shift_ratio=shift_ratio,
            layer_idx=layer_idx,
        )
        
        # Store reference for DAA integration
        self.daa_adapter = None
    
    def set_daa_adapter(self, daa_adapter: nn.Module):
        """Set the DAA adapter for attention modulation."""
        self.daa_adapter = daa_adapter
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass delegating to S²-Attn.
        
        This method attempts to work with common attention interfaces.
        May need customization for specific model architectures.
        """
        # Get Q, K, V projections from base attention
        # This is architecture-specific and may need adjustment
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Try to extract Q, K, V using common patterns
        if hasattr(self.base_attention, "q_proj"):
            query = self.base_attention.q_proj(hidden_states)
            key = self.base_attention.k_proj(hidden_states)
            value = self.base_attention.v_proj(hidden_states)
        elif hasattr(self.base_attention, "query"):
            query = self.base_attention.query(hidden_states)
            key = self.base_attention.key(hidden_states)
            value = self.base_attention.value(hidden_states)
        else:
            # Fallback to base attention
            return self.base_attention(hidden_states, attention_mask, **kwargs)
        
        # Reshape for multi-head attention
        num_heads = self.s2_attn.num_heads
        head_dim = self.s2_attn.head_dim
        
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Apply S²-Attn
        output, _ = self.s2_attn(
            query, key, value,
            attention_mask=attention_mask,
            daa_adapter=self.daa_adapter,
        )
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Apply output projection if available
        if hasattr(self.base_attention, "o_proj"):
            output = self.base_attention.o_proj(output)
        elif hasattr(self.base_attention, "out_proj"):
            output = self.base_attention.out_proj(output)
        
        return (output,)


def apply_s2_attention(
    model: nn.Module,
    hidden_size: int,
    num_heads: int,
    group_size: int = 2048,
    shift_ratio: float = 0.5,
    attention_pattern: str = "attn",
) -> Tuple[nn.Module, list]:
    """
    Apply S²-Attn to all attention layers in a model.
    
    Args:
        model: Model to modify
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        group_size: Tokens per attention group
        shift_ratio: Shift fraction
        attention_pattern: Name pattern to identify attention modules
        
    Returns:
        Tuple of (modified model, list of S²-Attn wrappers)
    """
    wrappers = []
    layer_idx = 0
    
    for name, module in list(model.named_modules()):
        # Skip if already wrapped
        if isinstance(module, (ShiftedSparseAttention, S2AttentionWrapper)):
            continue
        
        # Check if this looks like an attention module
        if attention_pattern in name.lower() and hasattr(module, "forward"):
            # Check for Q, K, V projections
            has_qkv = any(
                hasattr(module, attr) for attr in 
                ["q_proj", "k_proj", "v_proj", "query", "key", "value"]
            )
            
            if has_qkv:
                wrapper = S2AttentionWrapper(
                    base_attention=module,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    group_size=group_size,
                    shift_ratio=shift_ratio,
                    layer_idx=layer_idx,
                )
                
                # Replace in parent
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                
                setattr(parent, child_name, wrapper)
                wrappers.append(wrapper)
                layer_idx += 1
    
    return model, wrappers


def get_s2_memory_estimate(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    group_size: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # bf16
) -> dict:
    """
    Estimate memory usage for S²-Attn vs standard attention.
    
    Returns:
        Dictionary with memory estimates in bytes
    """
    head_dim = hidden_size // num_heads
    
    # Standard attention memory for attention weights
    standard_attn_mem = (
        batch_size * num_heads * seq_len * seq_len * dtype_bytes * num_layers
    )
    
    # S²-Attn memory (groups don't overlap)
    num_groups = (seq_len + group_size - 1) // group_size
    s2_attn_mem = (
        batch_size * num_heads * num_groups * group_size * group_size * dtype_bytes * num_layers
    )
    
    return {
        "standard_attention_bytes": standard_attn_mem,
        "s2_attention_bytes": s2_attn_mem,
        "memory_savings_ratio": standard_attn_mem / max(s2_attn_mem, 1),
        "standard_attention_gb": standard_attn_mem / (1024**3),
        "s2_attention_gb": s2_attn_mem / (1024**3),
    }
