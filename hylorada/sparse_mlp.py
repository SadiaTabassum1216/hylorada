"""
Sparse MLP Adapter Module

Implements local precision tuning through sparse neuron activation.
Based on: He et al., 2022 - "SparseAdapter" and Hao et al., 2025 - "MEFT"

Key features:
- Top-k neuron selection for memory efficiency
- Learnable gating mechanism for dynamic sparsity
- Bottleneck adapter architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TopKGate(nn.Module):
    """
    Differentiable Top-k gating mechanism for sparse activation.
    
    Uses straight-through estimator for gradient flow through discrete selection.
    
    Args:
        hidden_size: Dimension of input features
        num_neurons: Total number of neurons to gate
        topk_ratio: Fraction of neurons to activate (0 < ratio <= 1)
        temperature: Softmax temperature for soft gating during training
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_neurons: int,
        topk_ratio: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_neurons = num_neurons
        self.topk_ratio = topk_ratio
        self.k = max(1, int(num_neurons * topk_ratio))
        self.temperature = temperature
        
        # Learnable gate scores
        self.gate_scores = nn.Parameter(torch.zeros(num_neurons))
        nn.init.normal_(self.gate_scores, mean=0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        return_indices: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gating mask for input.
        
        Args:
            x: Input tensor [batch, seq, hidden_size]
            return_indices: Whether to return selected neuron indices
            
        Returns:
            Tuple of (gating_mask, optional_indices)
            - gating_mask: [num_neurons] binary mask
            - indices: [k] indices of selected neurons (if return_indices=True)
        """
        # Get top-k indices based on learned scores
        _, indices = torch.topk(self.gate_scores, self.k)
        
        # Create binary mask
        mask = torch.zeros(self.num_neurons, device=x.device, dtype=x.dtype)
        mask[indices] = 1.0
        
        if self.training:
            # Soft gating with straight-through estimator
            soft_scores = F.softmax(self.gate_scores / self.temperature, dim=0)
            # Scale soft scores to sum to k for gradient magnitude consistency
            soft_mask = soft_scores * self.num_neurons
            # Straight-through: use hard mask forward, soft mask backward
            mask = mask + soft_mask - soft_mask.detach()
        
        return mask, indices if return_indices else None
    
    def get_selected_neurons(self) -> torch.Tensor:
        """Return indices of currently selected neurons."""
        _, indices = torch.topk(self.gate_scores, self.k)
        return indices


class SparseAdapter(nn.Module):
    """
    Sparse bottleneck adapter for FFN layers.
    
    Applies a bottleneck transformation with sparse activation:
    output = x + sparse(down(x)) @ up
    
    Args:
        hidden_size: Model hidden dimension
        adapter_dim: Bottleneck dimension
        topk_ratio: Fraction of adapter neurons to activate
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_dim: int = 64,
        topk_ratio: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        
        # Down projection
        self.down_proj = nn.Linear(hidden_size, adapter_dim, bias=False)
        
        # Sparse gate
        self.gate = TopKGate(
            hidden_size=hidden_size,
            num_neurons=adapter_dim,
            topk_ratio=topk_ratio,
        )
        
        # Up projection
        self.up_proj = nn.Linear(adapter_dim, hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for near-identity at start."""
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)  # Start with zero contribution
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse adapter transformation.
        
        Args:
            x: Input tensor [batch, seq, hidden_size]
            
        Returns:
            Output with sparse adapter contribution added
        """
        # Ensure weights match input dtype (handles bfloat16 during mixed-precision)
        if x.dtype != self.down_proj.weight.dtype:
            self.down_proj.weight.data = self.down_proj.weight.data.to(x.dtype)
            self.up_proj.weight.data = self.up_proj.weight.data.to(x.dtype)
        
        # Down project
        down = self.down_proj(x)  # [batch, seq, adapter_dim]
        
        # Apply sparse gating
        mask, _ = self.gate(x)  # [adapter_dim]
        down = down * mask.unsqueeze(0).unsqueeze(0)  # Apply mask
        
        # Activation and dropout
        down = F.gelu(down)
        down = self.dropout(down)
        
        # Up project
        up = self.up_proj(down)  # [batch, seq, hidden_size]
        
        # Residual connection
        return x + up
    
    def get_sparsity_stats(self) -> dict:
        """Return statistics about the sparse activation."""
        selected = self.gate.get_selected_neurons()
        return {
            "num_selected": len(selected),
            "total_neurons": self.adapter_dim,
            "sparsity_ratio": 1.0 - (len(selected) / self.adapter_dim),
            "selected_indices": selected.cpu().tolist(),
        }


class SparseMLP(nn.Module):
    """
    Sparse MLP that wraps an existing FFN with Top-k neuron selection.
    
    Only activates the top-k most important neurons during forward pass,
    dramatically reducing memory usage while preserving local factual knowledge.
    
    Args:
        base_ffn: The original FFN module to wrap
        intermediate_size: Size of the FFN intermediate layer
        topk_ratio: Fraction of neurons to activate
        adapter_dim: Optional adapter bottleneck dimension
        use_adapter: Whether to use adapter-style or direct sparse tuning
    """
    
    def __init__(
        self,
        base_ffn: nn.Module,
        intermediate_size: int,
        topk_ratio: float = 0.1,
        adapter_dim: Optional[int] = None,
        use_adapter: bool = True,
    ):
        super().__init__()
        
        self.base_ffn = base_ffn
        self.intermediate_size = intermediate_size
        self.topk_ratio = topk_ratio
        self.use_adapter = use_adapter
        
        # Freeze base FFN
        for param in base_ffn.parameters():
            param.requires_grad = False
        
        if use_adapter:
            # Use sparse adapter (recommended for efficiency)
            self.sparse_adapter = SparseAdapter(
                hidden_size=self._get_hidden_size(),
                adapter_dim=adapter_dim or 64,
                topk_ratio=topk_ratio,
            )
        else:
            # Direct sparse tuning - learn sparse delta weights
            self.gate = TopKGate(
                hidden_size=intermediate_size,
                num_neurons=intermediate_size,
                topk_ratio=topk_ratio,
            )
            # Learnable delta for selected neurons
            self.neuron_delta = nn.Parameter(torch.zeros(intermediate_size))
    
    def _get_hidden_size(self) -> int:
        """Infer hidden size from base FFN (the input/output dim, not intermediate)."""
        # Try common attribute names
        for attr in ["hidden_size", "embed_dim", "d_model"]:
            if hasattr(self.base_ffn, attr):
                return getattr(self.base_ffn, attr)
        
        # Find all linear layer dimensions and take the SMALLEST in_features
        # (hidden_size is typically smaller than intermediate_size)
        min_in_features = float('inf')
        
        for module in self.base_ffn.modules():
            if isinstance(module, nn.Linear):
                min_in_features = min(min_in_features, module.in_features)
            # HuggingFace Conv1D (used in GPT-2)
            if type(module).__name__ == 'Conv1D':
                min_in_features = min(min_in_features, module.weight.shape[0])
        
        if min_in_features != float('inf'):
            return int(min_in_features)
        
        raise ValueError("Could not infer hidden_size from base FFN")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse activation.
        
        Args:
            x: Input tensor [batch, seq, hidden_size]
            
        Returns:
            Output with sparse FFN modifications
        """
        # Get base FFN output
        base_out = self.base_ffn(x)
        
        if self.use_adapter:
            # Apply sparse adapter
            return self.sparse_adapter(base_out)
        else:
            # Direct sparse tuning on intermediate activations
            # This requires access to intermediate activations which
            # we approximate by applying delta to output
            mask, _ = self.gate(x)
            delta = self.neuron_delta * mask
            return base_out + delta.unsqueeze(0).unsqueeze(0)


def apply_sparse_to_ffn(
    model: nn.Module,
    ffn_module_pattern: str = "mlp",
    topk_ratio: float = 0.1,
    adapter_dim: int = 64,
    target_layers: Optional[list] = None,
) -> Tuple[nn.Module, dict]:
    """
    Apply sparse adapters to FFN layers in a model.
    
    Args:
        model: The model to modify
        ffn_module_pattern: Name pattern to identify FFN modules
        topk_ratio: Fraction of neurons to activate
        adapter_dim: Bottleneck dimension for adapters
        target_layers: Specific layer indices to target (None = all)
        
    Returns:
        Tuple of (modified model, dict of sparse modules)
    """
    sparse_modules = {}
    layer_idx = 0
    
    for name, module in model.named_modules():
        if ffn_module_pattern in name and not any(
            isinstance(m, (SparseMLP, SparseAdapter)) for m in module.modules()
        ):
            # Check if this layer should be targeted
            if target_layers is not None and layer_idx not in target_layers:
                layer_idx += 1
                continue
            
            # Infer intermediate size
            intermediate_size = _infer_intermediate_size(module)
            if intermediate_size is None:
                layer_idx += 1
                continue
            
            # Create sparse wrapper
            sparse_mlp = SparseMLP(
                base_ffn=module,
                intermediate_size=intermediate_size,
                topk_ratio=topk_ratio,
                adapter_dim=adapter_dim,
            )
            
            # Replace in parent
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, sparse_mlp)
            
            sparse_modules[name] = sparse_mlp
            layer_idx += 1
    
    return model, sparse_modules


def _infer_intermediate_size(ffn_module: nn.Module) -> Optional[int]:
    """Infer the intermediate size of an FFN module."""
    # Look for common patterns
    for name, module in ffn_module.named_modules():
        # Check nn.Linear
        if isinstance(module, nn.Linear):
            if "up" in name or "gate" in name or "fc1" in name:
                return module.out_features
            if "down" in name or "fc2" in name:
                return module.in_features
        # Check Conv1D (GPT-2)
        if type(module).__name__ == 'Conv1D':
            if "c_fc" in name:  # GPT-2 uses c_fc for up projection
                return module.nf
            if "c_proj" in name:  # GPT-2 uses c_proj for down projection
                return module.weight.shape[0]
    
    # Fallback: find the largest linear layer output
    max_size = 0
    for module in ffn_module.modules():
        if isinstance(module, nn.Linear):
            max_size = max(max_size, module.out_features)
        if type(module).__name__ == 'Conv1D':
            max_size = max(max_size, module.nf)
    
    return max_size if max_size > 0 else None


def get_sparse_params(model: nn.Module) -> list:
    """Get all sparse adapter parameters from a model (not base FFN)."""
    params = []
    for module in model.modules():
        # Only count SparseAdapter and TopKGate, not SparseMLP (which includes base FFN)
        if isinstance(module, SparseAdapter):
            params.extend(module.parameters())
        elif isinstance(module, TopKGate):
            params.extend(module.parameters())
    return params


def count_sparse_params(model: nn.Module) -> int:
    """Count total sparse adapter parameters."""
    return sum(p.numel() for p in get_sparse_params(model))
