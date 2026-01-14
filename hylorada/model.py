"""
HyLoRADA Model Wrapper

Main integration module that combines all HyLoRADA components:
- LoRA adapters for global context
- Direct Attention Adaptation for noise filtering
- Sparse MLP for local precision
- Shifted Sparse Attention for efficiency

Provides a unified interface for applying HyLoRADA to any transformer model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .config import HyLoRADAConfig
from .lora import LoRALayer, apply_lora_to_model, count_lora_params, merge_lora_weights
from .daa import DirectAttentionAdapter, PositionalDAA, apply_daa_to_attention, count_daa_params
from .sparse_mlp import SparseMLP, SparseAdapter, apply_sparse_to_ffn, count_sparse_params
from .s2_attention import ShiftedSparseAttention, apply_s2_attention, get_s2_memory_estimate


@dataclass
class HyLoRADAState:
    """Tracks the state of HyLoRADA components in a model."""
    lora_layers: Dict[str, LoRALayer]
    daa_adapters: Dict[str, DirectAttentionAdapter]
    sparse_modules: Dict[str, SparseMLP]
    s2_wrappers: List[Any]
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
        config = HyLoRADAConfig(lora_rank=8, daa_enabled=True)
        
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
            daa_adapters={},
            sparse_modules={},
            s2_wrappers=[],
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
        # 1. Apply LoRA adapters
        self.base_model, self.state.lora_layers = apply_lora_to_model(
            model=self.base_model,
            target_modules=self.config.lora_target_modules,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        
        # 2. Apply DAA if enabled
        if self.config.daa_enabled:
            self._apply_daa()
        
        # 3. Apply Sparse MLP if enabled
        if self.config.sparse_enabled:
            self.base_model, self.state.sparse_modules = apply_sparse_to_ffn(
                model=self.base_model,
                ffn_module_pattern=self.ffn_pattern,
                topk_ratio=self.config.sparse_topk_ratio,
                adapter_dim=self.config.sparse_adapter_dim,
                target_layers=self.config.sparse_target_layers,
            )
        
        # 4. Apply S²-Attn if enabled
        if self.config.s2_attn_enabled:
            self.base_model, self.state.s2_wrappers = apply_s2_attention(
                model=self.base_model,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                group_size=self.config.s2_group_size,
                shift_ratio=self.config.s2_shift_ratio,
                attention_pattern=self.attn_pattern,
            )
            
            # Connect DAA adapters to S²-Attn wrappers
            self._connect_daa_to_s2()
    
    def _apply_daa(self):
        """Apply Direct Attention Adaptation to attention layers.
        
        Uses PositionalDAA when daa_use_positional is enabled (recommended for
        long-context tasks to address the Lost-in-the-Middle phenomenon).
        """
        layer_idx = 0
        
        for name, module in self.base_model.named_modules():
            if self.attn_pattern in name.lower():
                # Check if this is an attention layer (has projections)
                # Support LLaMA-style (q_proj, v_proj), GPT-2 (c_attn), and other patterns
                has_proj = any(
                    hasattr(module, attr) for attr in 
                    ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn", "c_proj"]
                )
                
                if has_proj:
                    # Use PositionalDAA for better long-context handling (Lost-in-the-Middle)
                    if self.config.daa_use_positional:
                        daa = PositionalDAA(
                            num_heads=self.num_heads,
                            max_seq_len=self.config.max_sequence_length,
                            num_buckets=self.config.daa_num_buckets,
                            per_head=self.config.daa_per_head,
                        )
                    else:
                        daa = DirectAttentionAdapter(
                            num_heads=self.num_heads,
                            per_head=self.config.daa_per_head,
                            init_alpha=self.config.daa_init_alpha,
                            init_beta=self.config.daa_init_beta,
                        )
                    module.daa_adapter = daa
                    self.state.daa_adapters[name] = daa
                    layer_idx += 1
    
    def _connect_daa_to_s2(self):
        """Connect DAA adapters to S²-Attn wrappers for integration."""
        for wrapper in self.state.s2_wrappers:
            # Find corresponding DAA adapter
            for name, daa in self.state.daa_adapters.items():
                if hasattr(wrapper, "base_attention"):
                    if hasattr(wrapper.base_attention, "daa_adapter"):
                        wrapper.set_daa_adapter(wrapper.base_attention.daa_adapter)
                        break
    
    def _freeze_base_model(self):
        """Freeze all non-HyLoRADA parameters."""
        # First, freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Then, unfreeze HyLoRADA components
        for lora_layer in self.state.lora_layers.values():
            for param in lora_layer.lora.parameters():
                param.requires_grad = True
        
        for daa in self.state.daa_adapters.values():
            for param in daa.parameters():
                param.requires_grad = True
        
        for sparse_mod in self.state.sparse_modules.values():
            if hasattr(sparse_mod, "sparse_adapter"):
                for param in sparse_mod.sparse_adapter.parameters():
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
        daa_params = count_daa_params(self.base_model)
        sparse_params = count_sparse_params(self.base_model)
        
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "lora_params": lora_params,
            "daa_params": daa_params,
            "sparse_params": sparse_params,
            "lora_ratio": lora_params / max(trainable, 1),
            "daa_ratio": daa_params / max(trainable, 1),
            "sparse_ratio": sparse_params / max(trainable, 1),
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
        print(f"  LoRA:       {counts['lora_params']:,} ({counts['lora_ratio']:.1%} of trainable)")
        print(f"  DAA:        {counts['daa_params']:,} ({counts['daa_ratio']:.1%} of trainable)")
        print(f"  Sparse MLP: {counts['sparse_params']:,} ({counts['sparse_ratio']:.1%} of trainable)")
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
        
        # Save DAA weights
        for name, daa in self.state.daa_adapters.items():
            state_dict[f"daa.{name}.alpha"] = daa.alpha.data
            state_dict[f"daa.{name}.beta"] = daa.beta.data
        
        # Save Sparse adapter weights
        for name, sparse in self.state.sparse_modules.items():
            if hasattr(sparse, "sparse_adapter"):
                adapter = sparse.sparse_adapter
                state_dict[f"sparse.{name}.down"] = adapter.down_proj.weight.data
                state_dict[f"sparse.{name}.up"] = adapter.up_proj.weight.data
                state_dict[f"sparse.{name}.gate"] = adapter.gate.gate_scores.data
        
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
        
        # Load DAA weights
        for name, daa in self.state.daa_adapters.items():
            if f"daa.{name}.alpha" in state_dict:
                daa.alpha.data = state_dict[f"daa.{name}.alpha"]
                daa.beta.data = state_dict[f"daa.{name}.beta"]
        
        # Load Sparse weights
        for name, sparse in self.state.sparse_modules.items():
            if hasattr(sparse, "sparse_adapter") and f"sparse.{name}.down" in state_dict:
                adapter = sparse.sparse_adapter
                adapter.down_proj.weight.data = state_dict[f"sparse.{name}.down"]
                adapter.up_proj.weight.data = state_dict[f"sparse.{name}.up"]
                adapter.gate.gate_scores.data = state_dict[f"sparse.{name}.gate"]
        
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
