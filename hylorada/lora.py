"""
LoRA (Low-Rank Adaptation) Module

Implements global context adaptation through low-rank matrix decomposition.
Based on: Hu et al., 2021 - "LoRA: Low-Rank Adaptation of Large Language Models"

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
    """Get all LoRA parameters from a model."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
            if module.bias is not None:
                params.append(module.bias)
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
        if isinstance(module, DoRALinear):
            total += module.lora_A.numel() + module.lora_B.numel() + module.magnitude.numel()
        elif isinstance(module, LoRALinear):
            total += module.lora_A.numel() + module.lora_B.numel()
    return total
