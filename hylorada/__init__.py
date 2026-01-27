# HyLoRADA: Hybrid Low-Rank Adaptation with Direct Attention
# Efficient Long-Context Fine-Tuning Framework

from hylorada.config import HyLoRADAConfig

# Main API (unified - recommended)
from hylorada.lora import (
    HyLoRADAUnified,
    UnifiedLayer,
    PositionBias,
    LandmarkLoRA,
    apply_unified_to_model,
)

# Legacy & baselines (for comparison)
from hylorada.lora import (
    LoRALinear,
    DoRALinear,
    HyLoRADALinear,
    HyLoRADAv2Linear,
    LoRALayer,
    DoRALayer,
    HyLoRADALayer,
    HyLoRADAv2Layer,
    apply_lora_to_model,
    apply_dora_to_model,
    apply_hylorada_adapter_to_model,
)

# Components
from hylorada.s2_attention import ShiftedSparseAttention

# Model wrapper (uses unified internally)
from hylorada.model import HyLoRADAModel, apply_hylorada

# Evaluation
from hylorada.evaluation import (
    evaluate_perplexity,
    evaluate_lost_in_the_middle,
    compare_models,
    run_full_evaluation,
)

# Baselines for comparison
from hylorada.baselines import (
    StandardLoRA,
    LoRaDAModel,
    LongLoRAModel,
    SparseAdapterModel,
    BaselineConfig,
    get_baseline_model,
)

__version__ = "0.3.0"  # Unified architecture

__all__ = [
    # Config
    "HyLoRADAConfig",
    # Main API (unified)
    "HyLoRADAUnified",
    "UnifiedLayer",
    "PositionBias",
    "apply_unified_to_model",
    # Model wrapper
    "HyLoRADAModel",
    "apply_hylorada",
    # Components
    "LandmarkLoRA",
    "ShiftedSparseAttention",
    # Evaluation
    "evaluate_perplexity",
    "evaluate_lost_in_the_middle",
    "compare_models",
    "run_full_evaluation",
    # Legacy (for comparison scripts)
    "LoRALinear",
    "DoRALinear",
    "HyLoRADALinear",
    "apply_lora_to_model",
    # Baselines
    "StandardLoRA",
    "LoRaDAModel",
    "LongLoRAModel",
    "SparseAdapterModel",
    "BaselineConfig",
    "get_baseline_model",
]
