"""
HyLoRADA Configuration Module

Configuration for the unified HyLoRADA architecture with Position-Content Fusion.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class HyLoRADAConfig:
    """
    Configuration for unified HyLoRADA fine-tuning with Position-Content Fusion.
    
    HyLoRADA is a SINGLE unified method (no if-else thresholds on sequence length):
    
    Key Innovation - Position-Content Fusion (PCF):
    - Combines position awareness and content-based landmark selection
    - Model learns when to use position/landmark information
    - Same architecture handles all context lengths (512 to 32K+)
    
    Core Components:
    1. rsLoRA: Rank-stabilized LoRA with orthogonal initialization (α/√r scaling)
    2. PCF Module: Unified position-content gating over learnable landmarks
    3. Optional DoRA magnitude decomposition
    
    Extended Components (optional, for extreme contexts):
    - S²-Attn: Shifted sparse attention for >8K contexts
    - RoPE scaling: YaRN/Linear for position extrapolation
    - Trainable embeddings/norms: For >32K contexts
    
    Args:
        lora_rank: Rank for LoRA decomposition
        lora_alpha: Scaling factor for rsLoRA updates
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: Which projections to apply HyLoRADA to
        
        num_landmarks: Number of PCF landmarks (learnable context summaries)
        num_position_buckets: Position bucketing granularity for PCF
        
        s2_attn_enabled: Use S²-Attn for very long context efficiency
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
    # DoRA magnitude: True = magnitude decomposition, False = standard rsLoRA
    # Note: Ablation shows DoRA can degrade performance (-5.62% in tests)
    # Enable only if validated on your specific task
    use_dora_magnitude: bool = False

    
    # ============ Position-Content Fusion (PCF) Settings ============
    # Unified module that handles position and content awareness
    # No thresholds - learns when to use position/landmark information
    num_landmarks: int = 8           # Learnable context summary tokens
    num_position_buckets: int = 64   # Position bucketing granularity
    
    # ============ S²-Attn Settings (Optional) ============
    # Disabled by default - only needed for very long contexts (>8K)
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
        """Validate configuration parameters."""
        # Validate LoRA settings
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        
        # Validate PCF settings (num_landmarks=0 disables PCF, which is valid)
        if self.num_landmarks < 0:
            raise ValueError(f"num_landmarks must be >= 0, got {self.num_landmarks}")
        if self.num_landmarks > 0 and self.num_position_buckets <= 0:
            raise ValueError(f"num_position_buckets must be positive when PCF enabled, got {self.num_position_buckets}")
        
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
            "pcf": self.num_landmarks > 0,  # PCF enabled if landmarks > 0
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
            "num_landmarks": self.num_landmarks,
            "num_position_buckets": self.num_position_buckets,
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
