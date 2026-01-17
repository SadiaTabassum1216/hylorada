"""
Structure Encoder Module

Unified structural encoding for HyLoRADA v2.
Supports positional, temporal, hierarchical, and topological signals.

Key features:
- Bucket-based positional encoding (from PositionalDAA)
- Learned frequency scaling (for time series)
- Optional structure ID embeddings (for code AST / graph nodes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class StructureEncoder(nn.Module):
    """
    Unified structure encoding for any sequential/structured data.
    
    Produces a structure prior that conditions LoRA adaptation strength.
    
    Args:
        hidden_size: Output dimension (matches model hidden size)
        max_seq_len: Maximum sequence length
        num_buckets: Number of relative position buckets
        encoding_dim: Internal encoding dimension (lightweight)
        max_structure_types: Number of distinct structure types (e.g., AST depths)
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int = 32768,
        num_buckets: int = 64,
        encoding_dim: int = 32,
        max_structure_types: int = 64,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.encoding_dim = encoding_dim
        
        # Learnable bucket embeddings for relative positions
        # Use 2*num_buckets+1 to handle both positive and negative distances
        self.bucket_embeddings = nn.Embedding(num_buckets * 2 + 1, encoding_dim)
        
        # Learnable frequency scales for sinusoidal encoding
        # Half the encoding for sin, half for cos
        self.freq_scale = nn.Parameter(torch.ones(encoding_dim // 2))
        
        # Optional: Structure type embeddings (for AST depth, graph node type, etc.)
        self.structure_embeddings = nn.Embedding(max_structure_types, encoding_dim)
        
        # Output projection to hidden size
        self.proj = nn.Linear(encoding_dim, hidden_size, bias=False)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(encoding_dim)
        
        # Gating to control structure influence
        self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for near-identity behavior at start."""
        nn.init.normal_(self.bucket_embeddings.weight, std=0.02)
        nn.init.normal_(self.structure_embeddings.weight, std=0.02)
        nn.init.zeros_(self.proj.weight)  # Start with zero output
    
    def _relative_position_bucket(
        self,
        positions: torch.Tensor,
        bidirectional: bool = True,
    ) -> torch.Tensor:
        """
        Convert absolute positions to bucket indices.
        Uses logarithmic bucketing for distant positions (from T5/PositionalDAA).
        
        Args:
            positions: [batch, seq] absolute positions
            bidirectional: Whether to use bidirectional buckets
            
        Returns:
            bucket_ids: [batch, seq] bucket indices
        """
        # Compute relative position from center
        seq_len = positions.size(-1)
        center = seq_len // 2
        relative_position = positions - center
        
        # Handle bidirectional
        if bidirectional:
            num_buckets = self.num_buckets
            # Offset for negative positions
            relative_buckets = (relative_position >= 0).long() * num_buckets
            relative_position = relative_position.abs()
        else:
            relative_buckets = torch.zeros_like(relative_position)
            relative_position = -torch.min(
                relative_position, 
                torch.zeros_like(relative_position)
            )
        
        # Half buckets for exact positions, half for log-spaced
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # Logarithmic bucketing for large distances
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-6)
            / math.log(self.max_seq_len / max_exact)
            * (num_buckets - max_exact)
        ).long()
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(
            is_small, 
            relative_position, 
            relative_position_if_large
        )
        
        return relative_buckets
    
    def _sinusoidal_encoding(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sinusoidal position encoding with learned frequencies.
        
        Args:
            positions: [batch, seq] absolute positions
            
        Returns:
            encoding: [batch, seq, encoding_dim]
        """
        # [batch, seq, encoding_dim//2]
        positions = positions.unsqueeze(-1).float()
        
        # Learned frequency scaling
        dim_indices = torch.arange(
            self.encoding_dim // 2, 
            device=positions.device, 
            dtype=torch.float
        )
        base_freqs = 1.0 / (10000 ** (dim_indices / (self.encoding_dim // 2)))
        freqs = base_freqs * self.freq_scale.abs()  # Learnable scaling
        
        # Compute sin and cos
        angles = positions * freqs
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        # Interleave sin and cos
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        
        return encoding
    
    def forward(
        self,
        positions: torch.Tensor,
        structure_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute structure prior for conditioning LoRA adaptation.
        
        Args:
            positions: [batch, seq] token positions (0 to seq_len-1)
            structure_ids: Optional [batch, seq] structure type IDs
                          (e.g., AST depth, graph node type)
                          
        Returns:
            structure_prior: [batch, seq, hidden_size]
        """
        batch_size, seq_len = positions.shape
        device = positions.device
        
        # 1. Bucket-based positional encoding
        bucket_ids = self._relative_position_bucket(positions)
        bucket_emb = self.bucket_embeddings(bucket_ids)  # [batch, seq, encoding_dim]
        
        # 2. Sinusoidal encoding with learned frequencies
        sin_emb = self._sinusoidal_encoding(positions)  # [batch, seq, encoding_dim]
        
        # 3. Combine bucket and sinusoidal
        structure_emb = bucket_emb + sin_emb
        
        # 4. Add structure-specific encoding if provided
        if structure_ids is not None:
            structure_ids = structure_ids.clamp(
                0, self.structure_embeddings.num_embeddings - 1
            )
            struct_type_emb = self.structure_embeddings(structure_ids)
            structure_emb = structure_emb + struct_type_emb
        
        # 5. Normalize
        structure_emb = self.norm(structure_emb)
        
        # 6. Project to hidden size with gating
        gate = torch.sigmoid(self.gate)
        output = self.proj(structure_emb) * gate
        
        return output
    
    def get_position_tensor(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create default position tensor for standard sequential input.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Device to create tensor on
            
        Returns:
            positions: [batch, seq] position indices
        """
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        return positions
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"encoding_dim={self.encoding_dim}, "
            f"num_buckets={self.num_buckets}"
        )


class TemporalStructureEncoder(StructureEncoder):
    """
    Specialized structure encoder for time series data.
    
    Adds:
    - Periodic encoding for capturing seasonality
    - Multi-scale temporal features
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int = 32768,
        num_buckets: int = 64,
        encoding_dim: int = 32,
        num_periods: int = 4,  # Number of learned period lengths
    ):
        super().__init__(
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_buckets=num_buckets,
            encoding_dim=encoding_dim,
        )
        
        # Learnable period lengths for capturing seasonality
        self.num_periods = num_periods
        self.period_lengths = nn.Parameter(
            torch.tensor([24.0, 168.0, 720.0, 8760.0])[:num_periods]  # hour, week, month, year
        )
        
        # Period embeddings
        self.period_proj = nn.Linear(num_periods * 2, encoding_dim, bias=False)
    
    def _periodic_encoding(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute periodic encoding for multiple learned periods.
        
        Args:
            positions: [batch, seq] time positions
            
        Returns:
            encoding: [batch, seq, num_periods * 2]
        """
        positions = positions.unsqueeze(-1).float()  # [batch, seq, 1]
        
        # Compute phase for each period
        phases = 2 * math.pi * positions / self.period_lengths.abs()  # [batch, seq, num_periods]
        
        # Sin and cos for each period
        sin_enc = torch.sin(phases)
        cos_enc = torch.cos(phases)
        
        return torch.cat([sin_enc, cos_enc], dim=-1)  # [batch, seq, num_periods * 2]
    
    def forward(
        self,
        positions: torch.Tensor,
        structure_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Override to add periodic encoding for time series."""
        # Get base structure encoding
        batch_size, seq_len = positions.shape
        
        # 1. Bucket + sinusoidal (from parent)
        bucket_ids = self._relative_position_bucket(positions)
        bucket_emb = self.bucket_embeddings(bucket_ids)
        sin_emb = self._sinusoidal_encoding(positions)
        
        # 2. Add periodic encoding (time series specific)
        periodic_emb = self._periodic_encoding(positions)
        periodic_emb = self.period_proj(periodic_emb)
        
        # 3. Combine
        structure_emb = bucket_emb + sin_emb + periodic_emb
        
        # 4. Structure type (if provided)
        if structure_ids is not None:
            structure_ids = structure_ids.clamp(
                0, self.structure_embeddings.num_embeddings - 1
            )
            struct_type_emb = self.structure_embeddings(structure_ids)
            structure_emb = structure_emb + struct_type_emb
        
        # 5. Normalize and project
        structure_emb = self.norm(structure_emb)
        gate = torch.sigmoid(self.gate)
        output = self.proj(structure_emb) * gate
        
        return output


def create_structure_encoder(
    encoder_type: str,
    hidden_size: int,
    **kwargs,
) -> StructureEncoder:
    """
    Factory function to create appropriate structure encoder.
    
    Args:
        encoder_type: "default", "temporal", or "graph"
        hidden_size: Model hidden size
        **kwargs: Additional encoder arguments
        
    Returns:
        Instantiated structure encoder
    """
    encoders = {
        "default": StructureEncoder,
        "temporal": TemporalStructureEncoder,
        # "graph": GraphStructureEncoder,  # Future extension
    }
    
    encoder_cls = encoders.get(encoder_type, StructureEncoder)
    return encoder_cls(hidden_size=hidden_size, **kwargs)
