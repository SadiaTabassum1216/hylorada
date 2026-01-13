"""
HyLoRADA Configuration Module

Defines hyperparameters for the hybrid fine-tuning framework:
- LoRA: Global context adaptation
- DAA: Direct Attention Adaptation for noise filtering
- Sparse MLP: Local precision tuning
- S²-Attn: Shifted Sparse Attention for efficiency
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class HyLoRADAConfig:
    """
    Configuration for HyLoRADA hybrid fine-tuning.
    
    The framework allocates a fixed parameter budget across three components:
    1. LoRA adapters for global context understanding
    2. Direct Attention Adaptation for noise suppression
    3. Sparse MLP adapters for local factual precision
    
    Args:
        lora_rank: Rank for LoRA decomposition (higher = more capacity)
        lora_alpha: Scaling factor for LoRA updates
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Which attention projections to apply LoRA
        
        daa_enabled: Whether to use Direct Attention Adaptation
        daa_init_alpha: Initial scaling factor for attention weights
        daa_init_beta: Initial bias for attention weights
        daa_per_head: Whether to learn separate α, β per attention head
        
        sparse_enabled: Whether to use Sparse MLP adapters
        sparse_topk_ratio: Fraction of neurons to activate (0.1 = top 10%)
        sparse_adapter_dim: Bottleneck dimension for sparse adapter
        sparse_target_layers: Which layers to apply sparse adapters (None = all)
        
        s2_attn_enabled: Whether to use Shifted Sparse Attention
        s2_group_size: Size of attention groups (smaller = more memory efficient)
        s2_shift_ratio: Fraction of group to shift (0.5 = half)
        
        budget_lora: Fraction of param budget for LoRA (0.0-1.0)
        budget_daa: Fraction of param budget for DAA (0.0-1.0)
        budget_sparse: Fraction of param budget for Sparse MLP (0.0-1.0)
    """
    
    # ============ LoRA Settings (Global Adaptation) ============
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    # Supports: LLaMA (q_proj, v_proj), GPT-2 (c_attn, c_proj), Falcon (query_key_value)
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "c_attn", "c_proj", "query_key_value")
    
    # ============ DAA Settings (Noise Filtering) ============
    daa_enabled: bool = True
    daa_init_alpha: float = 1.0
    daa_init_beta: float = 0.0
    daa_per_head: bool = True
    
    # ============ Sparse MLP Settings (Local Precision) ============
    sparse_enabled: bool = True
    sparse_topk_ratio: float = 0.1  # Activate top 10% of neurons
    sparse_adapter_dim: int = 64
    sparse_target_layers: Optional[List[int]] = None  # None = all layers
    
    # ============ S²-Attn Settings (Efficiency Backbone) ============
    # NOTE: S²-Attn disabled by default - requires architecture-specific handling
    # for models with Grouped Query Attention (GQA) like Qwen2, Llama 3, etc.
    s2_attn_enabled: bool = False
    s2_group_size: int = 2048  # Tokens per attention group
    s2_shift_ratio: float = 0.5  # Shift by half group size
    
    # ============ Parameter Budget Allocation ============
    budget_lora: float = 0.6   # 60% of trainable params to LoRA
    budget_daa: float = 0.1    # 10% to attention adaptation
    budget_sparse: float = 0.3  # 30% to sparse MLP
    
    # ============ Training Settings ============
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "fp16", "bf16", or "fp32"
    max_sequence_length: int = 32768
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure budget sums to 1.0
        total_budget = self.budget_lora + self.budget_daa + self.budget_sparse
        if abs(total_budget - 1.0) > 1e-6:
            raise ValueError(
                f"Budget allocation must sum to 1.0, got {total_budget:.4f}. "
                f"(LoRA: {self.budget_lora}, DAA: {self.budget_daa}, Sparse: {self.budget_sparse})"
            )
        
        # Validate ratios
        if not 0 < self.sparse_topk_ratio <= 1.0:
            raise ValueError(f"sparse_topk_ratio must be in (0, 1], got {self.sparse_topk_ratio}")
        
        if not 0 < self.s2_shift_ratio <= 1.0:
            raise ValueError(f"s2_shift_ratio must be in (0, 1], got {self.s2_shift_ratio}")
        
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}")
    
    def get_component_status(self) -> dict:
        """Return enabled/disabled status of each component."""
        return {
            "lora": True,  # LoRA is always enabled
            "daa": self.daa_enabled,
            "sparse_mlp": self.sparse_enabled,
            "s2_attn": self.s2_attn_enabled,
        }
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
            "daa_enabled": self.daa_enabled,
            "daa_init_alpha": self.daa_init_alpha,
            "daa_init_beta": self.daa_init_beta,
            "daa_per_head": self.daa_per_head,
            "sparse_enabled": self.sparse_enabled,
            "sparse_topk_ratio": self.sparse_topk_ratio,
            "sparse_adapter_dim": self.sparse_adapter_dim,
            "sparse_target_layers": self.sparse_target_layers,
            "s2_attn_enabled": self.s2_attn_enabled,
            "s2_group_size": self.s2_group_size,
            "s2_shift_ratio": self.s2_shift_ratio,
            "budget_lora": self.budget_lora,
            "budget_daa": self.budget_daa,
            "budget_sparse": self.budget_sparse,
            "gradient_checkpointing": self.gradient_checkpointing,
            "mixed_precision": self.mixed_precision,
            "max_sequence_length": self.max_sequence_length,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "HyLoRADAConfig":
        """Create config from dictionary."""
        # Convert list back to tuple for target_modules
        if "lora_target_modules" in config_dict:
            config_dict["lora_target_modules"] = tuple(config_dict["lora_target_modules"])
        return cls(**config_dict)


# Preset configurations for common use cases
class HyLoRADAPresets:
    """Pre-defined configurations for common scenarios."""
    
    @staticmethod
    def efficient() -> HyLoRADAConfig:
        """Memory-efficient config for limited GPU resources."""
        return HyLoRADAConfig(
            lora_rank=4,
            sparse_topk_ratio=0.05,
            s2_group_size=1024,
            gradient_checkpointing=True,
        )
    
    @staticmethod
    def balanced() -> HyLoRADAConfig:
        """Balanced config for typical fine-tuning."""
        return HyLoRADAConfig()  # Uses defaults
    
    @staticmethod
    def high_accuracy() -> HyLoRADAConfig:
        """Higher capacity config for maximum accuracy."""
        return HyLoRADAConfig(
            lora_rank=16,
            lora_alpha=32.0,
            sparse_topk_ratio=0.2,
            sparse_adapter_dim=128,
            budget_lora=0.5,
            budget_daa=0.15,
            budget_sparse=0.35,
        )
    
    @staticmethod
    def long_context_128k() -> HyLoRADAConfig:
        """Optimized for 128k+ context lengths."""
        return HyLoRADAConfig(
            s2_group_size=4096,
            max_sequence_length=131072,
            gradient_checkpointing=True,
            mixed_precision="bf16",
        )
