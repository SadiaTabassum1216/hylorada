"""
HyLoRADA Configuration Module

Defines hyperparameters for HyLoRADA (Hybrid Low-Rank Direct Attention Adaptation):
- Orthogonal initialization for LoRA A matrix (prevents rank collapse)
- Gated magnitude control (learnable DoRA magnitude blending)
- Residual LoRA path (combines DoRA and LoRA dynamics)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class HyLoRADAConfig:
    """
    Configuration for HyLoRADA hybrid fine-tuning.
    
    HyLoRADA is a novel PEFT method with three key innovations:
    1. Orthogonal initialization - Prevents rank collapse during training
    2. Gated magnitude - Learnable control over weight magnitude contribution
    3. Residual LoRA path - Blends DoRA and LoRA learning dynamics
    
    Args:
        lora_rank: Rank for LoRA decomposition (higher = more capacity)
        lora_alpha: Scaling factor for LoRA updates
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Which attention projections to apply LoRA
        
        use_hylorada: Enable HyLoRADA (orthogonal + gated + residual)
        use_dora: Enable DoRA (weight-decomposed LoRA)
        lora_plus_enabled: Enable LoRA+ asymmetric learning rates
        lora_plus_ratio: LR ratio for B matrix (10-20x recommended)
        
        daa_enabled: Whether to use Direct Attention Adaptation
        sparse_enabled: Whether to use Sparse MLP adapters (disabled by default)
    """
    
    # ============ LoRA Settings (Global Adaptation) ============
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    # Supports: LLaMA (q_proj, k_proj, v_proj, o_proj), GPT-2 (c_attn, c_proj), Falcon (query_key_value)
    # Added k_proj and o_proj for better coverage of attention mechanism
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query_key_value")
    # Layer-wise rank allocation: allocates higher rank to important layers (early/late)
    lora_layerwise_rank: bool = False  # Enable layer-wise rank allocation
    lora_rank_strategy: str = "importance"  # "importance", "uniform", or "decreasing"
    
    # DoRA: Weight-Decomposed LoRA (Liu et al., 2024)
    # Decomposes weight into magnitude + direction, applies LoRA to direction only
    use_dora: bool = False  # Enable DoRA instead of standard LoRA
    
    # HyLoRADA: Novel method with orthogonal init + gated magnitude + residual LoRA
    use_hylorada: bool = False  # Enable HyLoRADA (our novel method)
    
    # ============ HyLoRADA v2 Settings (Structure-Aware Adaptation) ============
    use_hylorada_v2: bool = False  # Enable HyLoRADA v2 with structure conditioning
    structure_dim: int = 32  # Dimension of structure prior
    structure_encoder_type: str = "default"  # "default", "temporal", or "graph"
    structure_num_buckets: int = 64  # Position buckets for structure encoder
    
    # LoRA+: Asymmetric learning rates (higher for B matrix)
    lora_plus_enabled: bool = False  # Enable LoRA+ learning rate scheduling
    lora_plus_ratio: float = 10.0  # lr_B / lr_A ratio (paper recommends 10)
    
    # ============ DAA Settings (Noise Filtering) ============
    daa_enabled: bool = True
    daa_init_alpha: float = 1.0
    daa_init_beta: float = 0.0
    daa_per_head: bool = True
    # PositionalDAA: Uses position-aware biases to address Lost-in-the-Middle phenomenon
    daa_use_positional: bool = True  # Enable PositionalDAA for better long-context handling
    daa_num_buckets: int = 64  # Number of relative position buckets
    # Content-aware DAA: learns input-dependent α, β instead of static values
    daa_content_aware: bool = False  # Enable content-aware adaptation
    
    # ============ Sparse MLP Settings (Local Precision) ============
    # Large-Sparse strategy (from SparseAdapter paper): larger adapter with higher sparsity
    # outperforms smaller dense adapters at the same parameter budget
    sparse_enabled: bool = True  # Enabled by default for full HyLoRADA
    sparse_topk_ratio: float = 0.05  # Activate top 5% of neurons (increased sparsity)
    sparse_adapter_dim: int = 128  # Larger bottleneck (Large-Sparse strategy)
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
    
    # ============ Advanced Accuracy Settings ============
    # Trainable LayerNorms: LongLoRA finding - norms matter for adaptation
    trainable_norms: bool = False  # Unfreeze layer norms (adds ~50K params)
    
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
            "hylorada_v2": self.use_hylorada_v2,
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
            "daa_use_positional": self.daa_use_positional,
            "daa_num_buckets": self.daa_num_buckets,
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

