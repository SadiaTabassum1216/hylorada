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
