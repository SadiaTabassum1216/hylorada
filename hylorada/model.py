"""
HyLoRADA Model Wrapper

Main integration module for HyLoRADA (Hybrid Low-Rank Direct Attention Adaptation):
- Orthogonal initialization for LoRA A matrix (prevents rank collapse)
- Gated magnitude control (learnable DoRA magnitude blending)
- Residual LoRA path (combines DoRA and LoRA dynamics)

Provides a unified interface for applying HyLoRADA to any transformer model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .config import HyLoRADAConfig
from .lora import (
    UnifiedLayer, PositionBias, HyLoRADAUnified, LandmarkLoRA,
    apply_unified_to_model,
    count_lora_params, merge_lora_weights, get_lora_plus_param_groups,
    # Legacy imports for baselines
    apply_lora_to_model, apply_dora_to_model, apply_hylorada_adapter_to_model,
)
from .s2_attention import ShiftedSparseAttention, apply_s2_attention, get_s2_memory_estimate


@dataclass
class HyLoRADAState:
    """Tracks the state of HyLoRADA components in a model."""
    lora_layers: Dict[str, UnifiedLayer]
    position_bias: Optional[PositionBias]
    s2_wrappers: List[Any]
    landmark: Optional[LandmarkLoRA]
    config: HyLoRADAConfig


class HyLoRADAModel(nn.Module):
    """
    HyLoRADA wrapper for transformer models.
    
    This class wraps any HuggingFace transformer model and applies the
    HyLoRADA fine-tuning components according to the provided configuration.
    
    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from hylorada import HyLoRADAModel, HyLoRADAConfig
        
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        config = HyLoRADAConfig(lora_rank=8, landmark_enabled=True)
        
        model = HyLoRADAModel(base_model, config)
        model.print_trainable_params()
        ```
    
    Args:
        base_model: The HuggingFace transformer model to wrap
        config: HyLoRADA configuration
        model_config: Optional model config for architecture detection
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[HyLoRADAConfig] = None,
    ):
        super().__init__()
        
        self.config = config or HyLoRADAConfig()
        self.base_model = base_model
        
        # Detect model architecture
        self._detect_architecture()
        
        # Initialize component tracking
        self.state = HyLoRADAState(
            lora_layers={},
            position_bias=None,
            s2_wrappers=[],
            landmark=None,
            config=self.config,
        )
        
        # Apply HyLoRADA components
        self._apply_hylorada()
        
        # Freeze base model
        self._freeze_base_model()
    
    def _detect_architecture(self):
        """Detect model architecture for component injection."""
        model_config = getattr(self.base_model, "config", None)
        
        # Try to extract architecture info
        self.hidden_size = getattr(model_config, "hidden_size", 4096)
        self.num_heads = getattr(model_config, "num_attention_heads", 32)
        self.num_layers = getattr(model_config, "num_hidden_layers", 32)
        self.intermediate_size = getattr(model_config, "intermediate_size", 11008)
        
        # Detect module naming patterns
        self.attn_pattern = self._find_pattern(["attention", "attn", "self_attn"])
        self.ffn_pattern = self._find_pattern(["mlp", "feed_forward", "ffn"])
    
    def _find_pattern(self, candidates: List[str]) -> str:
        """Find which naming pattern is used in the model."""
        for name, _ in self.base_model.named_modules():
            for pattern in candidates:
                if pattern in name.lower():
                    return pattern
        return candidates[0]  # Default to first candidate
    
    def _apply_hylorada(self):
        """Apply all HyLoRADA components to the model."""
        # 1. Apply unified HyLoRADA adapters with shared position bias
        self.base_model, self.state.lora_layers, self.state.position_bias = apply_unified_to_model(
            model=self.base_model,
            target_modules=self.config.lora_target_modules,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            use_position_bias=self.config.position_bias_enabled,
            use_dora_magnitude=self.config.use_dora_magnitude,
        )
        
        # 2. Apply S²-Attn if enabled (long-context)
        if self.config.s2_attn_enabled:
            self.base_model, self.state.s2_wrappers = apply_s2_attention(
                model=self.base_model,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                group_size=self.config.s2_group_size,
                shift_ratio=self.config.s2_shift_ratio,
                sink_tokens=self.config.s2_sink_tokens,
                attention_pattern=self.attn_pattern,
            )
            
        # 3. Apply RoPE Scaling if configured (YaRN/LongRoPE)
        if self.config.rope_scaling_type and hasattr(self.base_model, "config"):
            self._apply_rope_scaling()
        
        # 4. Apply LandmarkLoRA if enabled (Novel context summarization)
        if self.config.landmark_enabled:
            self.state.landmark = LandmarkLoRA(
                hidden_size=self.hidden_size,
                num_landmarks=self.config.num_landmarks,
                dropout=self.config.lora_dropout,
            )
            # Register as submodule so it moves with model.to(device)
            self.add_module("landmark_module", self.state.landmark)
            # Register hook to apply landmark to hidden states
            # Register hook to apply landmark to hidden states
            self._register_landmark_hook()
            
        # 5. Enable Gradient Checkpointing if configured
        if self.config.gradient_checkpointing:
            if hasattr(self.base_model, "gradient_checkpointing_enable"):
                self.base_model.gradient_checkpointing_enable()
                # Required when using checkpointing with frozen weights
                self.base_model.enable_input_require_grads()
                print("Gradient checkpointing enabled (memory optimized)")
    
    def _register_landmark_hook(self):
        """Register forward hook to apply LandmarkLoRA to hidden states."""
        # Find the final layer norm (before LM head)
        norm_module = None
        for name, module in self.base_model.named_modules():
            if any(n in name.lower() for n in ["norm", "ln_f", "final_layer_norm"]):
                if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)) or "norm" in type(module).__name__.lower():
                    norm_module = module
        
        if norm_module is not None:
            def landmark_hook(module, input, output):
                # Apply LandmarkLoRA to hidden states
                if self.state.landmark is not None:
                    # Ensure dtype compatibility
                    original_dtype = output.dtype
                    # Convert landmarks to match output dtype
                    if self.state.landmark.landmarks.dtype != original_dtype:
                        self.state.landmark.to(original_dtype)
                    return self.state.landmark(output)
                return output
            
            norm_module.register_forward_hook(landmark_hook)
            print(f"Registered LandmarkLoRA hook on final norm layer")
    
    def _apply_rope_scaling(self):
        """Inject RoPE scaling configuration into base model."""
        rope_config = {
            "type": self.config.rope_scaling_type,
            "factor": self.config.rope_scaling_factor
        }
        
        # Determine attribute name (transformers versions vary)
        if hasattr(self.base_model.config, "rope_scaling"):
            # Update existing config
            print(f"Applying RoPE scaling: {rope_config}")
            self.base_model.config.rope_scaling = rope_config
        else:
            print(f"Warning: Base model does not support rope_scaling in config")

    def _freeze_base_model(self):
        """Freeze all non-HyLoRADA parameters."""
        # First, freeze everything
        # First, freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze embeddings if configured (LongLoRA)
        if self.config.train_embeddings:
            for name, module in self.base_model.named_modules():
                if isinstance(module, (nn.Embedding)):
                    for param in module.parameters():
                        param.requires_grad = True
        
        # Unfreeze norms if configured (LongLoRA)
        if self.config.train_norms:
            for name, module in self.base_model.named_modules():
                if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or "norm" in name.lower():
                    for param in module.parameters():
                        param.requires_grad = True
        
        # Then, unfreeze HyLoRADA components
        for lora_layer in self.state.lora_layers.values():
            for param in lora_layer.lora.parameters():
                param.requires_grad = True
        
        # Unfreeze position bias if present
        if self.state.position_bias is not None:
            for param in self.state.position_bias.parameters():
                param.requires_grad = True
        
        # Unfreeze landmark if present (Novel)
        if self.state.landmark is not None:
            for param in self.state.landmark.parameters():
                param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.base_model(*args, **kwargs)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable HyLoRADA parameters."""
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params
    
    def count_params(self) -> Dict[str, int]:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.get_trainable_params())
        lora_params = count_lora_params(self.base_model)
        landmark_params = sum(p.numel() for p in self.state.landmark.parameters()) if self.state.landmark else 0
        
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "lora_params": lora_params,
            "landmark_params": landmark_params,
        }
    
    def print_trainable_params(self):
        """Print detailed parameter counts."""
        counts = self.count_params()
        
        print("=" * 60)
        print("HyLoRADA Parameter Summary")
        print("=" * 60)
        print(f"Total parameters:     {counts['total_params']:,}")
        print(f"Trainable parameters: {counts['trainable_params']:,}")
        print(f"Trainable ratio:      {counts['trainable_ratio']:.4%}")
        print("-" * 60)
        print("Component Breakdown:")
        print(f"  LoRA:       {counts['lora_params']:,}")
        print(f"  Landmark:   {counts['landmark_params']:,}")
        print("=" * 60)
    
    def get_memory_estimate(self, seq_len: int, batch_size: int = 1) -> Dict[str, float]:
        """Estimate memory usage for a given sequence length."""
        return get_s2_memory_estimate(
            seq_len=seq_len,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            group_size=self.config.s2_group_size,
            batch_size=batch_size,
        )
    
    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights and return the base model."""
        merged = merge_lora_weights(self.base_model)
        return merged
    
    def save_hylorada(self, path: str):
        """Save only the HyLoRADA adapter weights."""
        state_dict = {}
        
        # Save LoRA weights
        for name, layer in self.state.lora_layers.items():
            state_dict[f"lora.{name}.A"] = layer.lora.lora_A.data
            state_dict[f"lora.{name}.B"] = layer.lora.lora_B.data
        
        # Save Landmark weights
        if self.state.landmark is not None:
            state_dict["landmark.landmarks"] = self.state.landmark.landmarks.data
            state_dict["landmark.gate"] = self.state.landmark.gate.weight.data
            state_dict["landmark.scale"] = self.state.landmark.scale.data
        
        # Save config
        state_dict["config"] = self.config.to_dict()
        
        torch.save(state_dict, path)
        print(f"Saved HyLoRADA weights to {path}")
    
    def load_hylorada(self, path: str):
        """Load HyLoRADA adapter weights."""
        state_dict = torch.load(path, map_location="cpu")
        
        # Load LoRA weights
        for name, layer in self.state.lora_layers.items():
            if f"lora.{name}.A" in state_dict:
                layer.lora.lora_A.data = state_dict[f"lora.{name}.A"]
                layer.lora.lora_B.data = state_dict[f"lora.{name}.B"]
        
        # Load Landmark weights
        if self.state.landmark is not None and "landmark.landmarks" in state_dict:
            self.state.landmark.landmarks.data = state_dict["landmark.landmarks"]
            self.state.landmark.gate.weight.data = state_dict["landmark.gate"]
            self.state.landmark.scale.data = state_dict["landmark.scale"]
        
        print(f"Loaded HyLoRADA weights from {path}")


def apply_hylorada(
    model: nn.Module,
    config: Optional[HyLoRADAConfig] = None,
) -> HyLoRADAModel:
    """
    Convenience function to apply HyLoRADA to a model.
    
    Args:
        model: The base transformer model
        config: HyLoRADA configuration (uses defaults if None)
        
    Returns:
        HyLoRADAModel wrapper
    """
    return HyLoRADAModel(model, config)


def get_hylorada_optimizer_groups(
    model: HyLoRADAModel,
    lr_lora: float = 2e-4,
    lr_daa: float = 1e-3,
    lr_sparse: float = 2e-4,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Get optimizer parameter groups with component-specific learning rates.
    
    This allows different learning rates for different HyLoRADA components,
    which can improve training stability.
    
    Args:
        model: HyLoRADAModel instance
        lr_lora: Learning rate for LoRA parameters
        lr_daa: Learning rate for DAA parameters
        lr_sparse: Learning rate for sparse MLP parameters
        weight_decay: Weight decay coefficient
        
    Returns:
        List of parameter groups for optimizer
    """
    lora_params = []
    daa_params = []
    sparse_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "lora" in name.lower():
            lora_params.append(param)
        elif "daa" in name.lower() or "alpha" in name or "beta" in name:
            daa_params.append(param)
        elif "sparse" in name.lower() or "gate" in name.lower():
            sparse_params.append(param)
        else:
            other_params.append(param)
    
    groups = []
    
    if lora_params:
        groups.append({
            "params": lora_params,
            "lr": lr_lora,
            "weight_decay": weight_decay,
            "name": "lora",
        })
    
    if daa_params:
        groups.append({
            "params": daa_params,
            "lr": lr_daa,
            "weight_decay": 0.0,  # No weight decay for attention scalars
            "name": "daa",
        })
    
    if sparse_params:
        groups.append({
            "params": sparse_params,
            "lr": lr_sparse,
            "weight_decay": weight_decay,
            "name": "sparse",
        })
    
    if other_params:
        groups.append({
            "params": other_params,
            "lr": lr_lora,  # Default LR
            "weight_decay": weight_decay,
            "name": "other",
        })
    
    return groups
