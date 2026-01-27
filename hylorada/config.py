"""
HyLoRADA Configuration Module

Simplified configuration for the unified HyLoRADA architecture.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class HyLoRADAConfig:
    """
    Configuration for unified HyLoRADA fine-tuning.
    
    HyLoRADA combines proven techniques for cost-efficient long-context learning:
    1. Orthogonal initialization - Prevents rank collapse
    2. Position-aware scaling - Addresses lost-in-middle phenomenon
    3. S²-Attn (LongLoRA) - 16x training cost reduction
    4. Trainable embeddings & norms - Critical for >32k context
    5. RoPE scaling (YaRN) - Extends context up to 128k
    6. Sink tokens (SinkLoRA) - Stable attention patterns
    
    Args:
        lora_rank: Rank for LoRA decomposition
        lora_alpha: Scaling factor for LoRA updates
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Which projections to apply LoRA to
        
        daa_enabled: Use Direct Attention Adaptation for noise filtering
        daa_num_buckets: Position buckets for DAA
        
        sparse_enabled: Use Sparse MLP adapters
        sparse_adapter_dim: Bottleneck dimension for sparse adapter
        sparse_topk_ratio: Fraction of neurons to activate
        
        s2_attn_enabled: Use S²-Attn for long context efficiency
        s2_group_size: Tokens per attention group
    """
    
    # ============ Core LoRA Settings ============
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",  # LLaMA/Qwen
        "c_attn", "c_proj",  # GPT-2
        "query_key_value",  # Falcon
    )
    
    # ============ Position-Aware Scaling ============
    # Addresses lost-in-middle with only 64 shared params
    position_bias_enabled: bool = True
    position_num_buckets: int = 64
    
    # ============ DAA Settings (Noise Filtering) ============
    daa_enabled: bool = True
    daa_num_buckets: int = 64
    daa_per_head: bool = True
    
    # ============ Sparse MLP Settings ============
    sparse_enabled: bool = True
    sparse_adapter_dim: int = 64  # Smaller than before for efficiency
    sparse_topk_ratio: float = 0.1
    sparse_target_layers: Optional[List[int]] = None  # None = all layers
    
    # ============ S²-Attn Settings ============
    # Disabled by default - requires GQA handling
    s2_attn_enabled: bool = False
    s2_group_size: int = 2048
    s2_shift_ratio: float = 0.5
    
    # ============ Training Settings ============
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    max_sequence_length: int = 32768
    
    # ============ Long-Context Extensions (LongLoRA/SinkLoRA/YaRN) ============
    train_embeddings: bool = False  # Enable for >32k context (LongLoRA)
    train_norms: bool = False       # Enable for >32k context (LongLoRA)
    
    # Sink Token Support (SinkLoRA)
    s2_sink_tokens: int = 0         # Number of initial tokens to attend to globally
    
    # RoPE Scaling (YaRN/LongRoPE)
    rope_scaling_type: Optional[str] = None  # "linear", "dynamic", "yarn"
    rope_scaling_factor: float = 1.0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}")
        if not 0 < self.sparse_topk_ratio <= 1.0:
            raise ValueError(f"sparse_topk_ratio must be in (0, 1], got {self.sparse_topk_ratio}")
        if not 0 < self.s2_shift_ratio <= 1.0:
            raise ValueError(f"s2_shift_ratio must be in (0, 1], got {self.s2_shift_ratio}")
    
    def get_component_status(self) -> dict:
        """Return enabled/disabled status of each component."""
        return {
            "unified_lora": True,  # Always uses unified HyLoRADA
            "position_bias": self.position_bias_enabled,
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
            "position_bias_enabled": self.position_bias_enabled,
            "position_num_buckets": self.position_num_buckets,
            "daa_enabled": self.daa_enabled,
            "daa_num_buckets": self.daa_num_buckets,
            "daa_per_head": self.daa_per_head,
            "sparse_enabled": self.sparse_enabled,
            "sparse_adapter_dim": self.sparse_adapter_dim,
            "sparse_topk_ratio": self.sparse_topk_ratio,
            "sparse_target_layers": self.sparse_target_layers,
            "s2_attn_enabled": self.s2_attn_enabled,
            "s2_group_size": self.s2_group_size,
            "s2_shift_ratio": self.s2_shift_ratio,
            "gradient_checkpointing": self.gradient_checkpointing,
            "mixed_precision": self.mixed_precision,
            "max_sequence_length": self.max_sequence_length,
            "train_embeddings": self.train_embeddings,
            "train_norms": self.train_norms,
            "s2_sink_tokens": self.s2_sink_tokens,
            "rope_scaling_type": self.rope_scaling_type,
            "rope_scaling_factor": self.rope_scaling_factor,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "HyLoRADAConfig":
        """Create config from dictionary."""
        if "lora_target_modules" in config_dict:
            config_dict["lora_target_modules"] = tuple(config_dict["lora_target_modules"])
        return cls(**config_dict)


# Preset configurations
class HyLoRADAPresets:
    """Pre-defined configurations for common scenarios."""
    
    @staticmethod
    def efficient() -> HyLoRADAConfig:
        """Memory-efficient config for limited GPU resources."""
        return HyLoRADAConfig(
            lora_rank=4,
            sparse_enabled=False,  # Disable for minimum params
            sparse_adapter_dim=32,
            gradient_checkpointing=True,
        )
    
    @staticmethod
    def balanced() -> HyLoRADAConfig:
        """Balanced config for typical fine-tuning."""
        return HyLoRADAConfig()  # Uses defaults
    
    @staticmethod
    def quality() -> HyLoRADAConfig:
        """Higher capacity for best quality."""
        return HyLoRADAConfig(
            lora_rank=16,
            sparse_adapter_dim=128,
            sparse_topk_ratio=0.2,
        )
