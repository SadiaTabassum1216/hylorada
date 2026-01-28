"""
LoRA Module (Low-Rank Adaptation)

Implements PEFT methods:
1. LoRA - Low-rank adaptation (Hu et al., 2021)
2. DoRA - Weight-decomposed LoRA (Liu et al., 2024)
3. HyLoRADA - Streamlined for cost-efficient long-context learning:
   - Orthogonal initialization (prevents rank collapse)
   - DoRA-style magnitude normalization
   - Position-aware scaling

Key features:
- Low-rank A/B decomposition for weight updates
- Configurable target modules (Q, K, V, O projections)
- Merged inference mode for zero overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple
import math


class LoRALinear(nn.Module):
    """
    LoRA adapter for linear layers.
    
    Implements: W' = W + (α/r) * B @ A
    where W is frozen, and only A, B are trained.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor (effective scaling = alpha/rank)
        dropout: Dropout probability for LoRA path
        bias: Whether to include bias term
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # For merged inference
        self.merged = False
        self._original_weight: Optional[torch.Tensor] = None
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA matrices using Kaiming uniform for A, zeros for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation to base layer output.
        
        Args:
            x: Input tensor [batch, seq_len, in_features]
            base_output: Output from the frozen base layer
            
        Returns:
            Adapted output: base_output + LoRA contribution
        """
        if self.merged:
            return base_output
        
        # Compute LoRA in the parameter dtype (float32) then cast result
        # This preserves gradients while handling mixed precision
        input_dtype = x.dtype
        lora_out = self.dropout(x.to(self.lora_A.dtype))
        lora_out = F.linear(lora_out, self.lora_A)  # [batch, seq, rank]
        lora_out = F.linear(lora_out, self.lora_B)  # [batch, seq, out_features]
        lora_out = (lora_out * self.scaling).to(input_dtype)
        
        result = base_output + lora_out
        if self.bias is not None:
            result = result + self.bias.to(input_dtype)
        
        return result
    
    def get_delta_weight(self) -> torch.Tensor:
        """Compute the LoRA weight delta: (α/r) * B @ A"""
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights into base weight for efficient inference."""
        return base_weight + self.get_delta_weight()
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


class DoRALinear(nn.Module):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.
    
    Based on: Liu et al., 2024 - "DoRA: Weight-Decomposed Low-Rank Adaptation"
    
    DoRA decomposes the pretrained weight W into magnitude (m) and direction (V):
        W = m * (V / ||V||)
    
    Then applies LoRA only to the direction component:
        W' = m' * ((V + ΔV) / ||V + ΔV||)
    
    where ΔV = B @ A (the LoRA update) and m' is a learnable magnitude.
    
    This decomposition helps because:
    1. Magnitude and direction have different learning dynamics
    2. Direction captures most of the task-specific information
    3. Learning m' separately allows better adaptation
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices for direction update
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Learnable magnitude vector (one per output feature)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Cache for base weight norm (will be set when applied to layer)
        self.register_buffer("base_weight_norm", None)
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA matrices."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def init_magnitude(self, base_weight: torch.Tensor):
        """Initialize magnitude from the base weight's column norms."""
        # Compute column-wise L2 norm of base weight
        # base_weight shape: [out_features, in_features]
        weight_norm = base_weight.norm(p=2, dim=1)  # [out_features]
        self.magnitude.data = weight_norm.clone()
        self.register_buffer("base_weight_norm", weight_norm.clone())
    
    def forward(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        base_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply DoRA adaptation.
        
        Args:
            x: Input tensor [batch, seq, in_features]
            base_output: Output from frozen base layer
            base_weight: The frozen base weight matrix
            
        Returns:
            DoRA-adapted output
        """
        input_dtype = x.dtype
        
        # Compute LoRA delta: ΔV = B @ A
        x_float = x.to(self.lora_A.dtype)
        lora_out = self.dropout(x_float)
        lora_out = F.linear(lora_out, self.lora_A)  # [batch, seq, rank]
        lora_out = F.linear(lora_out, self.lora_B)  # [batch, seq, out_features]
        delta_v = lora_out * self.scaling
        
        # Compute updated direction norm: ||V + ΔV||
        # V is the base weight, ΔV is the LoRA contribution
        # We need: V + ΔV = base_weight + scaling * B @ A
        updated_weight = base_weight + (self.lora_B @ self.lora_A) * self.scaling
        updated_norm = updated_weight.norm(p=2, dim=1, keepdim=True).T  # [1, out_features]
        
        # Compute the magnitude scaling factor
        # m' / ||V + ΔV||
        mag_scale = (self.magnitude / (updated_norm.squeeze() + 1e-8))  # [out_features]
        
        # Apply DoRA: output = (base_output + delta_v) * (m' / ||V + ΔV||) * ||V|| / m_init
        # Simplified: we scale the combined output by the magnitude ratio
        combined = base_output.to(self.lora_A.dtype) + delta_v
        
        # Scale by magnitude (broadcast over batch and seq dims)
        result = combined * mag_scale.unsqueeze(0).unsqueeze(0)
        
        return result.to(input_dtype)
    
    def get_delta_weight(self) -> torch.Tensor:
        """Compute the LoRA weight delta: (α/r) * B @ A"""
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, dora=True"
        )


class HyLoRADALinear(nn.Module):
    """
    HyLoRADA: Hybrid Low-Rank Direct Attention Adaptation.
    
    Streamlined design for cost-efficient long-context learning:
    1. Orthogonal initialization for A matrix (prevents rank collapse)
    2. DoRA-style magnitude normalization
    3. Position-aware scaling (addresses lost-in-middle)
    
    This design prioritizes efficiency while maintaining the key innovations
    proven effective in the literature.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices with ORTHOGONAL initialization for A
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Learnable magnitude vector (DoRA-style)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Cache for base weight
        self.register_buffer("base_weight_norm", None)
        
        # Initialize with ORTHOGONAL (key innovation)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize with orthogonal A matrix (prevents rank collapse)."""
        nn.init.orthogonal_(self.lora_A)
        nn.init.zeros_(self.lora_B)
    
    def init_magnitude(self, base_weight: torch.Tensor):
        """Initialize magnitude from base weight norms."""
        weight_norm = base_weight.norm(p=2, dim=1)
        self.magnitude.data = weight_norm.clone()
        self.register_buffer("base_weight_norm", weight_norm.clone())
    
    def forward(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        base_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply HyLoRADA adaptation with DoRA-style normalization.
        
        Args:
            x: Input tensor [batch, seq, in_features]
            base_output: Output from frozen base layer
            base_weight: The frozen base weight matrix
            
        Returns:
            HyLoRADA-adapted output
        """
        input_dtype = x.dtype
        x_float = x.to(self.lora_A.dtype)
        
        # Compute LoRA contribution
        lora_x = self.dropout(x_float)
        lora_x = F.linear(lora_x, self.lora_A)  # [batch, seq, rank]
        lora_out = F.linear(lora_x, self.lora_B)  # [batch, seq, out_features]
        delta_v = lora_out * self.scaling
        
        # DoRA-style magnitude normalization
        updated_weight = base_weight + (self.lora_B @ self.lora_A) * self.scaling
        updated_norm = updated_weight.norm(p=2, dim=1) + 1e-8
        
        mag_scale = self.magnitude / updated_norm
        output = (base_output.to(self.lora_A.dtype) + delta_v) * mag_scale.unsqueeze(0).unsqueeze(0)
        
        return output.to(input_dtype)
    
    def get_delta_weight(self) -> torch.Tensor:
        """Compute the LoRA weight delta."""
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, hylorada=True"
        )


class HyLoRADAv2Linear(nn.Module):
    """
    HyLoRADA v2: Structure-Aware Hybrid Low-Rank Adaptation.
    
    Extends HyLoRADA v1 with structure-conditioned LoRA scaling.
    The adaptation strength becomes position-dependent based on structural signals.
    
    Novel contributions over v1:
    1. Structure-conditioned scaling (position-dependent LoRA strength)
    2. Backward compatible (works without structure prior, falls back to v1)
    
    The formulation is:
        delta = scaling * B @ A @ x
        if structure_prior provided:
            delta = delta * (0.5 + sigmoid(scale_from_structure(structure_prior)))
        output = (1-β) * DoRA_output + β * LoRA_output
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor
        dropout: Dropout probability
        structure_dim: Dimension of structure prior input
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        structure_dim: int = 32,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.structure_dim = structure_dim
        
        # === From HyLoRADA v1 (proven components) ===
        # LoRA matrices with orthogonal initialization for A
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Learnable magnitude vector (from DoRA)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Gated magnitude control - starts at 0.5
        self.magnitude_gate = nn.Parameter(torch.tensor(0.0))
        
        # Residual LoRA weight - learnable blend between DoRA and LoRA
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Cache for base weight
        self.register_buffer("base_weight_norm", None)
        
        # === NEW in v2: Structure conditioning ===
        # Maps structure prior to position-dependent scale
        self.scale_from_structure = nn.Linear(structure_dim, 1, bias=False)
        nn.init.zeros_(self.scale_from_structure.weight)  # Start neutral (no effect)
        
        # Initialize with orthogonal
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize with orthogonal A matrix (prevents rank collapse)."""
        nn.init.orthogonal_(self.lora_A)
        nn.init.zeros_(self.lora_B)
    
    def init_magnitude(self, base_weight: torch.Tensor):
        """Initialize magnitude from base weight norms."""
        weight_norm = base_weight.norm(p=2, dim=1)
        self.magnitude.data = weight_norm.clone()
        self.register_buffer("base_weight_norm", weight_norm.clone())
    
    def forward(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        base_weight: torch.Tensor,
        structure_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply HyLoRADA v2 adaptation with structure conditioning.
        
        Args:
            x: Input tensor [batch, seq, in_features]
            base_output: Output from frozen base layer
            base_weight: The frozen base weight matrix
            structure_prior: Optional [batch, seq, structure_dim] for conditioning
            
        Returns:
            HyLoRADA v2-adapted output
        """
        input_dtype = x.dtype
        x_float = x.to(self.lora_A.dtype)
        
        # Compute LoRA contribution
        lora_x = self.dropout(x_float)
        lora_x = F.linear(lora_x, self.lora_A)  # [batch, seq, rank]
        lora_out = F.linear(lora_x, self.lora_B)  # [batch, seq, out_features]
        delta_v = lora_out * self.scaling
        
        # === NEW in v2: Structure-conditioned scaling ===
        if structure_prior is not None:
            # [batch, seq, 1] position-dependent scale
            pos_scale = torch.sigmoid(
                self.scale_from_structure(structure_prior.to(self.lora_A.dtype))
            )
            # Scale delta: positions with high structure signal get boosted
            # Range [0.5, 1.5] to allow both attenuation and amplification
            delta_v = delta_v * (0.5 + pos_scale)
        
        # === DoRA PATH (from v1) ===
        updated_weight = base_weight + (self.lora_B @ self.lora_A) * self.scaling
        updated_norm = updated_weight.norm(p=2, dim=1) + 1e-8
        
        # Apply gated magnitude (from v1)
        gate = torch.sigmoid(self.magnitude_gate)
        effective_magnitude = self.magnitude * gate + self.base_weight_norm * (1 - gate)
        
        mag_scale = effective_magnitude / updated_norm
        dora_output = (base_output.to(self.lora_A.dtype) + delta_v) * mag_scale.unsqueeze(0).unsqueeze(0)
        
        # === RESIDUAL LoRA PATH (from v1) ===
        lora_output = base_output.to(self.lora_A.dtype) + delta_v
        
        # Blend DoRA and LoRA with learnable weight
        residual_w = torch.sigmoid(self.residual_weight)
        final_output = (1 - residual_w) * dora_output + residual_w * lora_output
        
        return final_output.to(input_dtype)
    
    def get_delta_weight(self) -> torch.Tensor:
        """Compute the LoRA weight delta."""
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, structure_dim={self.structure_dim}, hylorada_v2=True"
        )


class LoRALayer(nn.Module):
    """
    Wrapper that combines a frozen base linear layer with LoRA adapter.
    
    This is the main class used when injecting LoRA into an existing model.
    Supports both nn.Linear and HuggingFace's Conv1D (used in GPT-2).
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        
        # Handle both nn.Linear and Conv1D (GPT-2)
        if hasattr(base_layer, 'nf'):  # Conv1D from transformers
            in_features = base_layer.weight.shape[0]
            out_features = base_layer.nf
            self.is_conv1d = True
        else:  # nn.Linear
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self.is_conv1d = False
        
        self.lora = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=False,
        )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen base + LoRA adapter."""
        base_out = self.base_layer(x)
        return self.lora(x, base_out)
    
    def merge(self) -> nn.Linear:
        """Merge LoRA into base layer and return a standard nn.Linear."""
        merged_weight = self.lora.merge_weights(self.base_layer.weight.data)
        
        merged_layer = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        merged_layer.weight.data = merged_weight
        if self.base_layer.bias is not None:
            merged_layer.bias.data = self.base_layer.bias.data.clone()
        
        return merged_layer
    
    @property
    def weight(self) -> torch.Tensor:
        """Return effective weight (base + LoRA delta) for compatibility."""
        return self.lora.merge_weights(self.base_layer.weight)


class DoRALayer(nn.Module):
    """
    DoRA wrapper that combines a frozen base linear layer with DoRA adapter.
    
    Uses weight-decomposed low-rank adaptation for improved performance.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        
        # Handle both nn.Linear and Conv1D (GPT-2)
        if hasattr(base_layer, 'nf'):  # Conv1D from transformers
            in_features = base_layer.weight.shape[0]
            out_features = base_layer.nf
            self.is_conv1d = True
            # For Conv1D, weight is transposed: [in, out] instead of [out, in]
            self._base_weight = base_layer.weight.T
        else:  # nn.Linear
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self.is_conv1d = False
            self._base_weight = base_layer.weight
        
        self.dora = DoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Initialize magnitude from base weights
        with torch.no_grad():
            self.dora.init_magnitude(self._base_weight.data.float())
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    @property
    def lora(self):
        """Compatibility property - returns the DoRA module."""
        return self.dora
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen base + DoRA adapter."""
        base_out = self.base_layer(x)
        # DoRA needs the base weight for norm computation
        return self.dora(x, base_out, self._base_weight)
    
    @property
    def weight(self) -> torch.Tensor:
        """Return effective weight for compatibility."""
        return self._base_weight + self.dora.get_delta_weight()


class HyLoRADALayer(nn.Module):
    """
    HyLoRADA wrapper that combines frozen base layer with HyLoRADA adapter.
    
    This is the main wrapper class for applying HyLoRADA to models.
    Uses orthogonal init, gated magnitude, and residual LoRA path.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        
        # Handle both nn.Linear and Conv1D (GPT-2)
        if hasattr(base_layer, 'nf'):  # Conv1D from transformers
            in_features = base_layer.weight.shape[0]
            out_features = base_layer.nf
            self.is_conv1d = True
            self._base_weight = base_layer.weight.T
        else:  # nn.Linear
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self.is_conv1d = False
            self._base_weight = base_layer.weight
        
        self.hylorada = HyLoRADALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Initialize magnitude from base weights
        with torch.no_grad():
            self.hylorada.init_magnitude(self._base_weight.data.float())
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    @property
    def lora(self):
        """Compatibility property - returns the HyLoRADA module."""
        return self.hylorada
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen base + HyLoRADA adapter."""
        base_out = self.base_layer(x)
        return self.hylorada(x, base_out, self._base_weight)
    
    @property
    def weight(self) -> torch.Tensor:
        """Return effective weight for compatibility."""
        return self._base_weight + self.hylorada.get_delta_weight()


class HyLoRADAv2Layer(nn.Module):
    """
    HyLoRADA v2 wrapper with structure-aware adaptation.
    
    This wrapper accepts an optional structure_prior in forward pass
    to enable position-dependent LoRA scaling.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        structure_dim: int = 32,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        
        # Handle both nn.Linear and Conv1D (GPT-2)
        if hasattr(base_layer, 'nf'):  # Conv1D from transformers
            in_features = base_layer.weight.shape[0]
            out_features = base_layer.nf
            self.is_conv1d = True
            self._base_weight = base_layer.weight.T
        else:  # nn.Linear
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self.is_conv1d = False
            self._base_weight = base_layer.weight
        
        self.hylorada_v2 = HyLoRADAv2Linear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            structure_dim=structure_dim,
        )
        
        # Initialize magnitude from base weights
        with torch.no_grad():
            self.hylorada_v2.init_magnitude(self._base_weight.data.float())
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    @property
    def lora(self):
        """Compatibility property - returns the HyLoRADA v2 module."""
        return self.hylorada_v2
    
    def forward(
        self,
        x: torch.Tensor,
        structure_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through frozen base + HyLoRADA v2 adapter.
        
        Args:
            x: Input tensor [batch, seq, in_features]
            structure_prior: Optional [batch, seq, structure_dim]
            
        Returns:
            Adapted output
        """
        base_out = self.base_layer(x)
        return self.hylorada_v2(x, base_out, self._base_weight, structure_prior)
    
    @property
    def weight(self) -> torch.Tensor:
        """Return effective weight for compatibility."""
        return self._base_weight + self.hylorada_v2.get_delta_weight()


# =============================================================================
# UNIFIED HYLORADA - Consolidated implementation (recommended)
# =============================================================================

class PositionBias(nn.Module):
    """
    Shared position bias table for all layers.
    
    Ultra-lightweight module (only 64 params!) that provides position-aware
    scaling to address the lost-in-the-middle problem.
    
    Uses logarithmic bucketing so distant positions share buckets,
    while nearby positions have fine-grained distinction.
    """
    
    def __init__(self, num_buckets: int = 64, max_distance: int = 32768):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        
        # Learnable bias per bucket - initialized to zero (neutral)
        self.bias = nn.Parameter(torch.zeros(num_buckets))
    
    def _position_to_bucket(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Convert absolute positions to bucket indices using log bucketing.
        
        First half of buckets are for exact positions (0, 1, 2, ...)
        Second half use logarithmic spacing for distant positions.
        """
        # Clamp to valid range
        positions = positions.clamp(0, self.max_distance - 1)
        
        # First half: exact positions
        exact_buckets = self.num_buckets // 2
        
        # For positions < exact_buckets, use exact bucket
        is_small = positions < exact_buckets
        
        # For larger positions, use log bucketing
        # Map [exact_buckets, max_distance] -> [exact_buckets, num_buckets]
        relative_position = positions.float() - exact_buckets
        max_relative = self.max_distance - exact_buckets
        
        # Log scale: bucket = exact_buckets + (num_buckets/2) * log(pos) / log(max)
        log_buckets = exact_buckets + (
            (self.num_buckets - exact_buckets - 1) * 
            torch.log(relative_position.clamp(min=1)) / 
            math.log(max(max_relative, 2))
        )
        
        bucket_ids = torch.where(is_small, positions, log_buckets.long())
        return bucket_ids.clamp(0, self.num_buckets - 1)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get position bias values.
        
        Args:
            positions: [batch, seq] absolute positions (0 to seq_len-1)
            
        Returns:
            bias: [batch, seq] position-dependent bias values
        """
        bucket_ids = self._position_to_bucket(positions)
        return self.bias[bucket_ids]


class LandmarkLoRA(nn.Module):
    """
    LandmarkLoRA: Trainable context summary tokens (Novel).
    
    Inspired by Landmark Attention but with a key difference:
    - Landmark Attention uses fixed block gating
    - LandmarkLoRA uses TRAINABLE landmarks as LoRA adapters
    
    The landmarks learn to capture important context patterns during fine-tuning,
    providing a form of learned memory compression.
    
    Params: num_landmarks × hidden_size × 2 (landmarks + gate)
    For 8 landmarks and 896 hidden: 8 × 896 + 896 × 8 = 14,336 params
    
    Args:
        hidden_size: Model hidden dimension
        num_landmarks: Number of learnable summary tokens (default: 8)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_landmarks: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_landmarks = num_landmarks
        
        # Learnable landmark tokens (context summaries)
        self.landmarks = nn.Parameter(torch.randn(num_landmarks, hidden_size) * 0.02)
        
        # Gate to select relevant landmarks based on input
        self.gate = nn.Linear(hidden_size, num_landmarks, bias=False)
        
        # Scaling factor (learnable)
        self.scale = nn.Parameter(torch.tensor(0.1))
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply landmark-based context enhancement.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
            
        Returns:
            Enhanced hidden states with landmark context
        """
        # Compute mean representation for gating
        mean_repr = hidden_states.mean(dim=1)  # [batch, hidden_size]
        
        # Soft attention over landmarks
        gate_logits = self.gate(mean_repr)  # [batch, num_landmarks]
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [batch, num_landmarks]
        
        # Weighted combination of landmarks
        context = gate_weights @ self.landmarks  # [batch, hidden_size]
        context = self.dropout(context)
        
        # Add scaled context to all positions
        output = hidden_states + self.scale * context.unsqueeze(1)
        
        return output
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_landmarks={self.num_landmarks}"


class HyLoRADAUnified(nn.Module):
    """
    Unified HyLoRADA: Streamlined for cost-efficient long-context learning.
    
    Combines key innovations:
    1. Orthogonal initialization (prevents rank collapse)
    2. Gated Magnitude (adaptive control)
    3. Residual Blend (LoRA + DoRA dynamics)
    4. Position-scaled adaptation (handles lost-in-middle)
    
    Formula:
        output = (1 - β) * DoRA_output + β * LoRA_output
        
        DoRA: (W + BA) / ||W + BA|| * m
        LoRA: W + BA
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        position_bias: Optional['PositionBias'] = None,
        use_dora_magnitude: bool = True,  # Default to True for full hybrid
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / (rank ** 0.5)  # rsLoRA scaling
        self.use_dora_magnitude = use_dora_magnitude
        
        # LoRA matrices with ORTHOGONAL initialization for A
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Magnitude vector (DoRA-style)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Gated magnitude control (starts at 0.5)
        self.magnitude_gate = nn.Parameter(torch.tensor(0.0))
        
        # Residual LoRA weight (learnable blend between DoRA and LoRA)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Position-dependent scaling factor
        self.position_scale = nn.Parameter(torch.tensor(0.0))
        
        # Shared position bias
        self.position_bias = position_bias
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Cache for base weight norm
        self.register_buffer("base_weight_norm", None)
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.orthogonal_(self.lora_A)
        nn.init.zeros_(self.lora_B)
    
    def init_magnitude(self, base_weight: torch.Tensor):
        """Initialize magnitude from base weight norms."""
        weight_norm = base_weight.norm(p=2, dim=1)
        self.magnitude.data = weight_norm.clone()
        self.register_buffer("base_weight_norm", weight_norm.clone())
    
    def forward(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        base_weight: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply unified HyLoRADA adaptation."""
        input_dtype = x.dtype
        x_float = x.to(self.lora_A.dtype)
        
        # Compute LoRA delta
        lora_x = self.dropout(x_float)
        lora_x = F.linear(lora_x, self.lora_A)
        lora_out = F.linear(lora_x, self.lora_B)
        delta_v = lora_out * self.scaling
        
        # Position scaling
        if positions is not None and self.position_bias is not None:
            pos_bias = self.position_bias(positions)
            scale_weight = torch.sigmoid(self.position_scale)
            pos_scale = 1.0 + scale_weight * torch.tanh(pos_bias).unsqueeze(-1)
            delta_v = delta_v * pos_scale
        
        # Base + Delta (Full weight W + BA)
        # Note: base_output corresponds to Wx. We need (W+BA)x = Wx + BAx
        # But DoRA needs normalization of (W+BA).
        
        # 1. Compute LoRA output (W+BA)x
        # We can't easily get (W+BA) directly without weight reconstruction for normalization
        # But we can reconstruct weight for DoRA part
        
        weight_merged = base_weight + (self.lora_B @ self.lora_A) * self.scaling
        weight_norm = weight_merged.norm(p=2, dim=1) + 1e-8
        
        # DoRA output: Direction * Magnitude
        # Direction = (W+BA) / ||W+BA||
        # Magnitude = m * gate + m_init * (1-gate)
        # We use a simplified Gated Magnitude from V1/V2 paper logic:
        # mag_current = self.magnitude
        # gate = sigmoid(self.magnitude_gate)
        # final_mag = mag_current # or blended
        
        mag_scale = self.magnitude / weight_norm
        
        # DoRA Output = ((W+BA)x) * (magnitude / ||W+BA||)
        # We have base_output (Wx) and delta_v (BAx). So (W+BA)x = base_output + delta_v
        
        feature_out = base_output.to(self.lora_A.dtype) + delta_v
        
        # Apply Magnitude Scaling (DoRA)
        dora_output = feature_out * mag_scale.unsqueeze(0).unsqueeze(0)
        
        # Apply Gated Control (optional refinement)
        # gate = torch.sigmoid(self.magnitude_gate)
        # dora_output = dora_output * (0.5 + gate) # Simple gate scaling
        
        # Hybrid Blend: (1-β)*DoRA + β*LoRA
        beta = torch.sigmoid(self.residual_weight) # Learned mix ratio
        
        # LoRA Output is just feature_out
        final_output = (1 - beta) * dora_output + beta * feature_out
        
        return final_output.to(input_dtype)
    
    def get_delta_weight(self) -> torch.Tensor:
        """Compute the LoRA weight delta."""
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, hybrid=True"
        )


class UnifiedLayer(nn.Module):
    """
    Unified HyLoRADA wrapper for model injection.
    
    This is the recommended wrapper for applying HyLoRADA to models.
    Combines all features into a single, efficient layer.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        position_bias: Optional[PositionBias] = None,
        use_dora_magnitude: bool = False,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.use_dora_magnitude = use_dora_magnitude
        
        # Handle both nn.Linear and Conv1D (GPT-2)
        if hasattr(base_layer, 'nf'):  # Conv1D from transformers
            in_features = base_layer.weight.shape[0]
            out_features = base_layer.nf
            self.is_conv1d = True
            self._base_weight = base_layer.weight.T
        else:  # nn.Linear
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self.is_conv1d = False
            self._base_weight = base_layer.weight
        
        self.adapter = HyLoRADAUnified(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            position_bias=position_bias,
            use_dora_magnitude=use_dora_magnitude,
        )
        
        # Initialize magnitude from base weights (only if using DoRA)
        if use_dora_magnitude:
            with torch.no_grad():
                self.adapter.init_magnitude(self._base_weight.data.float())
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    @property
    def lora(self):
        """Compatibility property - returns the adapter module."""
        return self.adapter
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through frozen base + unified adapter.
        
        Args:
            x: Input tensor [batch, seq, in_features]
            positions: Optional [batch, seq] position indices
            
        Returns:
            Adapted output
        """
        base_out = self.base_layer(x)
        return self.adapter(x, base_out, self._base_weight, positions)
    
    @property
    def weight(self) -> torch.Tensor:
        """Return effective weight for compatibility."""
        return self._base_weight + self.adapter.get_delta_weight()


def apply_unified_to_model(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    use_position_bias: bool = True,
    use_dora_magnitude: bool = False,
) -> Tuple[nn.Module, Dict[str, UnifiedLayer], Optional[PositionBias]]:
    """
    Apply unified HyLoRADA adapters to target modules.
    
    This is the recommended function for applying HyLoRADA to models.
    Creates a shared PositionBias for all layers (only 64 params!).
    
    Args:
        model: The model to modify (will be modified in-place)
        target_modules: Names of modules to apply HyLoRADA to
        rank: LoRA rank
        alpha: Scaling factor
        dropout: Dropout probability
        use_position_bias: Whether to use position-aware scaling
        use_dora_magnitude: If True, use DoRA-style magnitude (more params). False = lightweight.
        
    Returns:
        Tuple of (modified model, dict of unified layers, shared position bias)
    """
    # Create shared position bias (only 64 params total!)
    position_bias = PositionBias() if use_position_bias else None
    
    unified_layers = {}
    targets = find_target_modules(model, target_modules)
    
    for name, module in targets.items():
        # Create unified wrapper
        unified_layer = UnifiedLayer(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            position_bias=position_bias,
            use_dora_magnitude=use_dora_magnitude,
        )
        
        # Replace module in parent
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, unified_layer)
        
        unified_layers[name] = unified_layer
    
    return model, unified_layers, position_bias


def _is_linear_layer(module: nn.Module) -> bool:
    """Check if module is a linear-like layer (nn.Linear or Conv1D)."""
    if isinstance(module, nn.Linear):
        return True
    # HuggingFace Conv1D (used in GPT-2)
    if type(module).__name__ == 'Conv1D':
        return True
    return False


def find_target_modules(
    model: nn.Module,
    target_names: Tuple[str, ...],
) -> Dict[str, nn.Module]:
    """
    Find all linear-like layers matching target names.
    
    Args:
        model: The model to search
        target_names: Tuple of layer name patterns to match
        
    Returns:
        Dictionary mapping full module path to layer (Linear or Conv1D)
    """
    targets = {}
    
    for name, module in model.named_modules():
        if _is_linear_layer(module):
            # Check if any target name is in the module path
            for target in target_names:
                if target in name:
                    targets[name] = module
                    break
    
    return targets


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Dict[str, LoRALayer]]:
    """
    Apply LoRA adapters to target modules in a model.
    
    Args:
        model: The model to modify (will be modified in-place)
        target_modules: Names of modules to apply LoRA to
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        
    Returns:
        Tuple of (modified model, dict of LoRA layers)
    """
    lora_layers = {}
    targets = find_target_modules(model, target_modules)
    
    for name, module in targets.items():
        # Create LoRA wrapper
        lora_layer = LoRALayer(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Replace module in parent
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, lora_layer)
        
        lora_layers[name] = lora_layer
    
    return model, lora_layers


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA/DoRA/HyLoRADA parameters from a model."""
    params = []
    for module in model.modules():
        # Check for all LoRA variants
        if isinstance(module, (LoRALinear, DoRALinear, HyLoRADALinear)):
            params.extend([module.lora_A, module.lora_B])
            if hasattr(module, 'bias') and module.bias is not None:
                params.append(module.bias)
            # DoRA/HyLoRADA magnitude
            if hasattr(module, 'magnitude'):
                params.append(module.magnitude)
            # HyLoRADA gate and residual
            if hasattr(module, 'magnitude_gate'):
                params.append(module.magnitude_gate)
            if hasattr(module, 'residual_weight'):
                params.append(module.residual_weight)
    return params


def count_lora_params(model: nn.Module) -> int:
    """Count total number of LoRA parameters."""
    return sum(p.numel() for p in get_lora_params(model))


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base model for efficient inference.
    
    After merging, the model can be used without LoRA overhead.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALayer):
            merged = module.merge()
            
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, merged)
    
    return model


def compute_layer_ranks(
    num_layers: int,
    base_rank: int = 8,
    strategy: str = "importance",
) -> Dict[int, int]:
    """
    Compute per-layer LoRA ranks based on layer importance.
    
    Research shows early and late layers matter more for adaptation,
    while middle layers can use lower rank with minimal accuracy loss.
    
    Args:
        num_layers: Total number of transformer layers
        base_rank: Base rank to scale from
        strategy: "importance" (high at edges), "uniform" (same rank), 
                  or "decreasing" (higher early, lower late)
    
    Returns:
        Dictionary mapping layer index to rank
    """
    layer_ranks = {}
    
    if strategy == "uniform":
        return {i: base_rank for i in range(num_layers)}
    
    elif strategy == "importance":
        # U-shaped: high at start and end, low in middle
        # Based on layer importance analysis from LoRA papers
        for i in range(num_layers):
            # Distance from edges (0 = edge, 0.5 = middle)
            distance_from_edge = min(i, num_layers - 1 - i) / (num_layers / 2)
            # Scale: 1.0 at edges, 0.5 at middle
            scale = 1.0 - 0.5 * distance_from_edge
            layer_ranks[i] = max(2, int(base_rank * scale))
    
    elif strategy == "decreasing":
        # Higher rank at early layers, lower at late
        for i in range(num_layers):
            scale = 1.0 - 0.5 * (i / (num_layers - 1))
            layer_ranks[i] = max(2, int(base_rank * scale))
    
    return layer_ranks


def apply_lora_with_layer_ranks(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
    layer_ranks: Optional[Dict[int, int]] = None,
    base_rank: int = 8,
    alpha_ratio: float = 2.0,
    dropout: float = 0.0,
    rank_strategy: str = "importance",
) -> Tuple[nn.Module, Dict[str, LoRALayer]]:
    """
    Apply LoRA with layer-wise rank allocation.
    
    This is more parameter-efficient than uniform rank, allocating
    more capacity to important layers (early/late) and less to middle.
    
    Args:
        model: The model to modify
        target_modules: Module names to apply LoRA to
        layer_ranks: Optional custom per-layer ranks (overrides strategy)
        base_rank: Base rank for rank computation
        alpha_ratio: Alpha = rank * alpha_ratio
        dropout: LoRA dropout
        rank_strategy: "importance", "uniform", or "decreasing"
        
    Returns:
        Tuple of (modified model, dict of LoRA layers)
    """
    # Detect number of layers
    num_layers = 0
    for name, _ in model.named_modules():
        # Look for layer patterns like "layers.0", "h.0", "blocks.0"
        for pattern in ["layers.", "h.", "blocks.", "decoder.layers."]:
            if pattern in name:
                try:
                    idx = int(name.split(pattern)[1].split(".")[0])
                    num_layers = max(num_layers, idx + 1)
                except (IndexError, ValueError):
                    pass
    
    if num_layers == 0:
        # Fallback to uniform rank
        return apply_lora_to_model(model, target_modules, base_rank, 
                                    base_rank * alpha_ratio, dropout)
    
    # Compute layer ranks if not provided
    if layer_ranks is None:
        layer_ranks = compute_layer_ranks(num_layers, base_rank, rank_strategy)
    
    lora_layers = {}
    targets = find_target_modules(model, target_modules)
    
    for name, module in targets.items():
        # Extract layer index from name
        layer_idx = None
        for pattern in ["layers.", "h.", "blocks.", "decoder.layers."]:
            if pattern in name:
                try:
                    layer_idx = int(name.split(pattern)[1].split(".")[0])
                    break
                except (IndexError, ValueError):
                    pass
        
        # Get rank for this layer
        rank = layer_ranks.get(layer_idx, base_rank) if layer_idx is not None else base_rank
        alpha = rank * alpha_ratio
        
        # Create LoRA wrapper
        lora_layer = LoRALayer(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Replace module in parent
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, lora_layer)
        
        lora_layers[name] = lora_layer
    
    return model, lora_layers


def apply_dora_to_model(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Dict[str, DoRALayer]]:
    """
    Apply DoRA (Weight-Decomposed LoRA) adapters to target modules.
    
    DoRA decomposes weights into magnitude and direction, applying LoRA
    only to direction. This typically outperforms standard LoRA.
    
    Args:
        model: The model to modify (will be modified in-place)
        target_modules: Names of modules to apply DoRA to
        rank: LoRA rank
        alpha: Scaling factor
        dropout: Dropout probability
        
    Returns:
        Tuple of (modified model, dict of DoRA layers)
    """
    dora_layers = {}
    targets = find_target_modules(model, target_modules)
    
    for name, module in targets.items():
        # Create DoRA wrapper
        dora_layer = DoRALayer(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Replace module in parent
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, dora_layer)
        
        dora_layers[name] = dora_layer
    
    return model, dora_layers


def apply_hylorada_adapter_to_model(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Dict[str, HyLoRADALayer]]:
    """
    Apply HyLoRADA adapters to target modules in a model.
    
    HyLoRADA is a novel PEFT method combining:
    1. Orthogonal initialization (prevents rank collapse)
    2. Gated magnitude (learnable magnitude control)
    3. Residual LoRA path (blends DoRA and LoRA dynamics)
    
    Args:
        model: The model to modify (will be modified in-place)
        target_modules: Names of modules to apply HyLoRADA to
        rank: Rank for the low-rank matrices
        alpha: Scaling factor
        dropout: Dropout probability
        
    Returns:
        Tuple of (modified model, dict of HyLoRADA layers)
    """
    hylorada_layers = {}
    targets = find_target_modules(model, target_modules)
    
    for name, module in targets.items():
        # Create HyLoRADA wrapper
        hylorada_layer = HyLoRADALayer(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Replace module in parent
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, hylorada_layer)
        
        hylorada_layers[name] = hylorada_layer
    
    return model, hylorada_layers


def get_lora_plus_param_groups(
    model: nn.Module,
    lr_A: float = 1e-4,
    lr_B: float = 1e-3,
    lr_magnitude: float = 1e-4,
    weight_decay: float = 0.0,
) -> List[Dict]:
    """
    Get optimizer parameter groups for LoRA+ (asymmetric learning rates).
    
    Based on: "LoRA+: Efficient Low Rank Adaptation" (2024)
    
    LoRA+ uses higher learning rate for matrix B (output) since it's 
    initialized to zero and needs to learn more. This improves convergence.
    
    Recommended ratios from paper:
    - lr_B / lr_A ≈ 10 (B learns faster than A)
    - lr_magnitude ≈ lr_A (for DoRA)
    
    Args:
        model: Model with LoRA/DoRA layers
        lr_A: Learning rate for A matrix (input projection)
        lr_B: Learning rate for B matrix (output projection)
        lr_magnitude: Learning rate for DoRA magnitude (if applicable)
        weight_decay: Weight decay coefficient
        
    Returns:
        List of parameter groups for optimizer
    """
    params_A = []
    params_B = []
    params_magnitude = []
    params_other = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        name_lower = name.lower()
        
        if "lora_a" in name_lower:
            params_A.append(param)
        elif "lora_b" in name_lower:
            params_B.append(param)
        elif "magnitude" in name_lower:
            params_magnitude.append(param)
        else:
            params_other.append(param)
    
    groups = []
    
    if params_A:
        groups.append({
            "params": params_A,
            "lr": lr_A,
            "weight_decay": weight_decay,
            "name": "lora_A",
        })
    
    if params_B:
        groups.append({
            "params": params_B,
            "lr": lr_B,  # Higher LR for B (LoRA+ key insight)
            "weight_decay": weight_decay,
            "name": "lora_B",
        })
    
    if params_magnitude:
        groups.append({
            "params": params_magnitude,
            "lr": lr_magnitude,
            "weight_decay": 0.0,  # No weight decay for magnitude
            "name": "dora_magnitude",
        })
    
    if params_other:
        groups.append({
            "params": params_other,
            "lr": lr_A,  # Default LR
            "weight_decay": weight_decay,
            "name": "other",
        })
    
    return groups


def count_dora_params(model: nn.Module) -> int:
    """Count total DoRA parameters (LoRA + magnitude)."""
    total = 0
    for module in model.modules():
        if isinstance(module, (DoRALinear, HyLoRADALinear, HyLoRADAv2Linear)):
            total += module.lora_A.numel() + module.lora_B.numel() + module.magnitude.numel()
        elif isinstance(module, LoRALinear):
            total += module.lora_A.numel() + module.lora_B.numel()
        elif isinstance(module, UnifiedLayer):
            adapter = module.adapter
            total += adapter.lora_A.numel() + adapter.lora_B.numel()
            if adapter.magnitude is not None:
                total += adapter.magnitude.numel()
    return total


def count_lora_params(model: nn.Module) -> int:
    """Count total LoRA/HyLoRADA parameters."""
    total = 0
    for module in model.modules():
        # Handle UnifiedLayer
        if isinstance(module, UnifiedLayer):
            adapter = module.adapter
            total += adapter.lora_A.numel() + adapter.lora_B.numel()
            if adapter.magnitude is not None:
                total += adapter.magnitude.numel()
            # Count scalars (magnitude_gate, residual_weight, position_scale)
            # Each is a single parameter
            for scalar in ["magnitude_gate", "residual_weight", "position_scale"]:
                if hasattr(adapter, scalar):
                    total += getattr(adapter, scalar).numel()
            
        # Handle Legacy Layers
        elif isinstance(module, (LoRALinear, DoRALinear, HyLoRADALinear, HyLoRADAv2Linear)):
            if hasattr(module, "lora_A"):
                total += module.lora_A.numel() + module.lora_B.numel()
            if hasattr(module, "magnitude") and module.magnitude is not None:
                total += module.magnitude.numel()
            # Count scalars
            for scalar in ["magnitude_gate", "residual_weight", "position_scale"]:
                if hasattr(module, scalar):
                    total += getattr(module, scalar).numel()
                    
    return total
