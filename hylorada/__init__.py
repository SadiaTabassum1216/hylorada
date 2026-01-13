# HyLoRADA: Hybrid Low-Rank Adaptation with Direct Attention
# Efficient Long-Context Fine-Tuning Framework

from hylorada.config import HyLoRADAConfig
from hylorada.lora import LoRALinear, apply_lora_to_model
from hylorada.daa import DirectAttentionAdapter
from hylorada.sparse_mlp import SparseMLP, SparseAdapter
from hylorada.s2_attention import ShiftedSparseAttention
from hylorada.model import HyLoRADAModel, apply_hylorada
from hylorada.evaluation import (
    evaluate_perplexity,
    evaluate_lost_in_the_middle,
    compare_models,
    run_full_evaluation,
)

__version__ = "0.1.0"
__all__ = [
    "HyLoRADAConfig",
    "LoRALinear",
    "apply_lora_to_model",
    "DirectAttentionAdapter",
    "SparseMLP",
    "SparseAdapter",
    "ShiftedSparseAttention",
    "HyLoRADAModel",
    "apply_hylorada",
    "evaluate_perplexity",
    "evaluate_lost_in_the_middle",
    "compare_models",
    "run_full_evaluation",
]
