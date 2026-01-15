"""
Baseline Comparison Methods

Implements different PEFT methods for fair comparison with HyLoRADA:
1. Standard LoRA - Basic low-rank adaptation
2. LoRaDA - LoRA + Direct Attention Adaptation  
3. LongLoRA - LoRA + trainable embeddings/norms
4. SparseAdapter - Only sparse MLP adapters

Each method uses similar parameter budgets for fair comparison.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from hylorada.config import HyLoRADAConfig
from hylorada.lora import apply_lora_to_model, count_lora_params
from hylorada.daa import DirectAttentionAdapter, PositionalDAA, count_daa_params
from hylorada.sparse_mlp import apply_sparse_to_ffn, count_sparse_params


@dataclass
class BaselineConfig:
    """Configuration for baseline methods."""
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    sparse_adapter_dim: int = 128
    sparse_topk_ratio: float = 0.05
    daa_per_head: bool = True
    max_sequence_length: int = 32768


class StandardLoRA(nn.Module):
    """
    Standard LoRA Implementation
    
    Based on: Hu et al., 2021 - "LoRA: Low-Rank Adaptation of Large Language Models"
    
    Only applies low-rank adapters to attention projections.
    This is the most basic PEFT baseline.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[BaselineConfig] = None,
    ):
        super().__init__()
        self.config = config or BaselineConfig()
        self.base_model = base_model
        
        # Detect architecture
        self._detect_architecture()
        
        # Apply LoRA only
        self.base_model, self.lora_layers = apply_lora_to_model(
            model=self.base_model,
            target_modules=("q_proj", "v_proj", "c_attn"),  # Original LoRA targets Q and V only
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        
        # Freeze base model
        self._freeze_base_model()
    
    def _detect_architecture(self):
        model_config = getattr(self.base_model, "config", None)
        self.num_heads = getattr(model_config, "num_attention_heads", 32)
    
    def _freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for lora_layer in self.lora_layers.values():
            for param in lora_layer.lora.parameters():
                param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def count_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "lora_params": count_lora_params(self.base_model),
        }
    
    def print_trainable_params(self):
        counts = self.count_params()
        print("=" * 60)
        print("Standard LoRA Parameter Summary")
        print("=" * 60)
        print(f"Total parameters:     {counts['total_params']:,}")
        print(f"Trainable parameters: {counts['trainable_params']:,}")
        print(f"Trainable ratio:      {counts['trainable_ratio']:.4%}")
        print("-" * 60)
        print(f"  LoRA: {counts['lora_params']:,}")
        print("=" * 60)


class LoRaDAModel(nn.Module):
    """
    LoRaDA Implementation
    
    Based on: "LoRaDA: Low-Rank Direct Attention Adaptation"
    
    Combines LoRA with Direct Attention Adaptation (DAA).
    DAA learns to modulate attention weights with α and β parameters.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[BaselineConfig] = None,
    ):
        super().__init__()
        self.config = config or BaselineConfig()
        self.base_model = base_model
        
        self._detect_architecture()
        
        # 1. Apply LoRA
        self.base_model, self.lora_layers = apply_lora_to_model(
            model=self.base_model,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"),
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        
        # 2. Apply DAA (the key addition in LoRaDA)
        self.daa_adapters = {}
        self._apply_daa()
        
        self._freeze_base_model()
    
    def _detect_architecture(self):
        model_config = getattr(self.base_model, "config", None)
        self.num_heads = getattr(model_config, "num_attention_heads", 32)
        self.attn_pattern = self._find_pattern(["attention", "attn", "self_attn"])
    
    def _find_pattern(self, candidates):
        for name, _ in self.base_model.named_modules():
            for pattern in candidates:
                if pattern in name.lower():
                    return pattern
        return candidates[0]
    
    def _apply_daa(self):
        for name, module in self.base_model.named_modules():
            if self.attn_pattern in name.lower():
                has_proj = any(
                    hasattr(module, attr) for attr in 
                    ["q_proj", "k_proj", "v_proj", "c_attn", "c_proj"]
                )
                if has_proj:
                    daa = DirectAttentionAdapter(
                        num_heads=self.num_heads,
                        per_head=self.config.daa_per_head,
                    )
                    module.daa_adapter = daa
                    self.daa_adapters[name] = daa
    
    def _freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for lora_layer in self.lora_layers.values():
            for param in lora_layer.lora.parameters():
                param.requires_grad = True
        for daa in self.daa_adapters.values():
            for param in daa.parameters():
                param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def count_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "lora_params": count_lora_params(self.base_model),
            "daa_params": count_daa_params(self.base_model),
        }
    
    def print_trainable_params(self):
        counts = self.count_params()
        print("=" * 60)
        print("LoRaDA Parameter Summary")
        print("=" * 60)
        print(f"Total parameters:     {counts['total_params']:,}")
        print(f"Trainable parameters: {counts['trainable_params']:,}")
        print(f"Trainable ratio:      {counts['trainable_ratio']:.4%}")
        print("-" * 60)
        print(f"  LoRA: {counts['lora_params']:,}")
        print(f"  DAA:  {counts['daa_params']:,}")
        print("=" * 60)


class LongLoRAModel(nn.Module):
    """
    LongLoRA Implementation
    
    Based on: Chen et al., 2023 - "LongLoRA: Efficient Fine-tuning of Long-Context LLMs"
    
    Key addition: Makes embedding and normalization layers trainable
    in addition to LoRA adapters. This significantly improves long-context adaptation.
    
    Note: S²-Attn (Shifted Sparse Attention) is part of LongLoRA but disabled here
    due to compatibility issues with GQA models.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[BaselineConfig] = None,
    ):
        super().__init__()
        self.config = config or BaselineConfig()
        self.base_model = base_model
        
        # 1. Apply LoRA to all attention projections
        self.base_model, self.lora_layers = apply_lora_to_model(
            model=self.base_model,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"),
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        
        # 2. Freeze base, then unfreeze LoRA + embeddings + norms
        self._freeze_and_unfreeze()
    
    def _freeze_and_unfreeze(self):
        # First freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA
        for lora_layer in self.lora_layers.values():
            for param in lora_layer.lora.parameters():
                param.requires_grad = True
        
        # LongLoRA's key finding: unfreeze embeddings
        for name, param in self.base_model.named_parameters():
            if "embed" in name.lower():
                param.requires_grad = True
        
        # LongLoRA's key finding: unfreeze layer norms
        for name, param in self.base_model.named_parameters():
            if "norm" in name.lower() or "ln" in name.lower():
                param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def count_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = count_lora_params(self.base_model)
        
        # Count embedding params
        embed_params = 0
        for name, param in self.base_model.named_parameters():
            if "embed" in name.lower() and param.requires_grad:
                embed_params += param.numel()
        
        # Count norm params
        norm_params = 0
        for name, param in self.base_model.named_parameters():
            if ("norm" in name.lower() or "ln" in name.lower()) and param.requires_grad:
                norm_params += param.numel()
        
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "lora_params": lora_params,
            "embed_params": embed_params,
            "norm_params": norm_params,
        }
    
    def print_trainable_params(self):
        counts = self.count_params()
        print("=" * 60)
        print("LongLoRA Parameter Summary")
        print("=" * 60)
        print(f"Total parameters:     {counts['total_params']:,}")
        print(f"Trainable parameters: {counts['trainable_params']:,}")
        print(f"Trainable ratio:      {counts['trainable_ratio']:.4%}")
        print("-" * 60)
        print(f"  LoRA:       {counts['lora_params']:,}")
        print(f"  Embeddings: {counts['embed_params']:,}")
        print(f"  LayerNorms: {counts['norm_params']:,}")
        print("=" * 60)


class SparseAdapterModel(nn.Module):
    """
    SparseAdapter Implementation
    
    Based on: He et al., 2022 - "SparseAdapter: An Easy Approach for Improving 
    the Parameter-Efficiency of Adapters"
    
    Uses sparse adapters on FFN layers only, without LoRA.
    Implements the "Large-Sparse" strategy: bigger adapters with higher sparsity.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[BaselineConfig] = None,
    ):
        super().__init__()
        self.config = config or BaselineConfig()
        self.base_model = base_model
        
        self._detect_architecture()
        
        # Apply Sparse adapters to FFN layers
        self.base_model, self.sparse_modules = apply_sparse_to_ffn(
            model=self.base_model,
            ffn_module_pattern=self.ffn_pattern,
            topk_ratio=self.config.sparse_topk_ratio,
            adapter_dim=self.config.sparse_adapter_dim,
        )
        
        self._freeze_base_model()
    
    def _detect_architecture(self):
        self.ffn_pattern = self._find_pattern(["mlp", "feed_forward", "ffn"])
    
    def _find_pattern(self, candidates):
        for name, _ in self.base_model.named_modules():
            for pattern in candidates:
                if pattern in name.lower():
                    return pattern
        return candidates[0]
    
    def _freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for sparse_mod in self.sparse_modules.values():
            if hasattr(sparse_mod, "sparse_adapter"):
                for param in sparse_mod.sparse_adapter.parameters():
                    param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def count_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / max(total, 1),
            "sparse_params": count_sparse_params(self.base_model),
        }
    
    def print_trainable_params(self):
        counts = self.count_params()
        print("=" * 60)
        print("SparseAdapter Parameter Summary")
        print("=" * 60)
        print(f"Total parameters:     {counts['total_params']:,}")
        print(f"Trainable parameters: {counts['trainable_params']:,}")
        print(f"Trainable ratio:      {counts['trainable_ratio']:.4%}")
        print("-" * 60)
        print(f"  Sparse MLP: {counts['sparse_params']:,}")
        print("=" * 60)


# ==================== Factory Function ====================

def get_baseline_model(
    method: str,
    base_model: nn.Module,
    config: Optional[BaselineConfig] = None,
) -> nn.Module:
    """
    Factory function to get different baseline models.
    
    Args:
        method: One of "lora", "lorada", "longlora", "sparse"
        base_model: The pretrained model to adapt
        config: Configuration options
    
    Returns:
        Wrapped model with the specified PEFT method applied
    """
    methods = {
        "lora": StandardLoRA,
        "lorada": LoRaDAModel,
        "longlora": LongLoRAModel,
        "sparse": SparseAdapterModel,
    }
    
    if method.lower() not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from: {list(methods.keys())}")
    
    return methods[method.lower()](base_model, config)
