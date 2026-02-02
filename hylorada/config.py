"""
HyLoRADA Configuration Module

Simplified configuration for the unified HyLoRADA architecture.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


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
    # DoRA magnitude: True = magnitude decomposition, False = standard LoRA
    # Note: Ablation shows DoRA can degrade performance (-5.62% in tests)
    # Enable only if validated on your specific task
    use_dora_magnitude: bool = False

    
    # ============ Position-Aware Scaling ============
    # Addresses lost-in-middle with only 64 shared params
    position_bias_enabled: bool = True
    position_num_buckets: int = 64
    
    # ============ S²-Attn Settings ============
    # Disabled by default - requires GQA handling
    s2_attn_enabled: bool = False
    s2_group_size: int = 2048
    s2_shift_ratio: float = 0.5
    
    # ============ Position-Adaptive LandmarkLoRA Settings ============
    # Enabled by default: 9.19% PPL improvement, 9.32% LIM-PPL improvement
    landmark_enabled: bool = True
    num_landmarks: int = 8  # Number of learnable context summary tokens
    num_position_buckets: int = 32  # Position bucketing granularity
    
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
        """Validate configuration parameters."""
        # Validate LoRA settings
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        
        # Validate landmark settings
        if self.landmark_enabled:
            if self.num_landmarks <= 0:
                raise ValueError(f"num_landmarks must be positive, got {self.num_landmarks}")
            if self.num_position_buckets <= 0:
                raise ValueError(f"num_position_buckets must be positive, got {self.num_position_buckets}")
        
        # Validate position bias settings
        if self.position_bias_enabled and self.position_num_buckets <= 0:
            raise ValueError(f"position_num_buckets must be positive, got {self.position_num_buckets}")
        
        # Validate S²-Attn settings
        if self.s2_attn_enabled:
            if self.s2_group_size <= 0:
                raise ValueError(f"s2_group_size must be positive, got {self.s2_group_size}")
            if not 0 < self.s2_shift_ratio <= 1.0:
                raise ValueError(f"s2_shift_ratio must be in (0, 1], got {self.s2_shift_ratio}")
            logger.warning(
                "S²-Attn is enabled. Ensure your model uses Grouped Query Attention (GQA) "
                "or Multi-Head Attention (MHA). Flash Attention variants may not be compatible."
            )
        
        # Validate RoPE scaling
        if self.rope_scaling_type is not None:
            valid_types = ["linear", "dynamic", "yarn"]
            if self.rope_scaling_type not in valid_types:
                raise ValueError(
                    f"rope_scaling_type must be one of {valid_types}, got {self.rope_scaling_type}"
                )
            if self.rope_scaling_factor <= 1.0:
                logger.warning(
                    f"rope_scaling_factor is {self.rope_scaling_factor}, which won't extend context. "
                    "Use values > 1.0 for context extension (e.g., 2.0 for 2x, 4.0 for 4x)."
                )
        
        # Validate mixed precision
        if self.mixed_precision not in ["fp16", "bf16", "fp32"]:
            raise ValueError(
                f"mixed_precision must be 'fp16', 'bf16', or 'fp32', got {self.mixed_precision}"
            )
        
        # Validate sequence length
        if self.max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {self.max_sequence_length}")
    
    def get_component_status(self) -> dict:
        """Return enabled/disabled status of each component."""
        return {
            "unified_lora": True,  # Always uses unified HyLoRADA
            "position_bias": self.position_bias_enabled,
            "landmark": self.landmark_enabled,
            "s2_attn": self.s2_attn_enabled,
        }
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
            "use_dora_magnitude": self.use_dora_magnitude,
            "position_bias_enabled": self.position_bias_enabled,
            "position_num_buckets": self.position_num_buckets,
            "landmark_enabled": self.landmark_enabled,
            "num_landmarks": self.num_landmarks,
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
