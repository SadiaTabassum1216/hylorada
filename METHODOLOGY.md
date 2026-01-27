# HyLoRADA: Hybrid Low-Rank Direct Attention Adaptation

## Overview

HyLoRADA is a streamlined PEFT method optimized for **cost-efficient long-context learning**. Based on findings from recent literature (2023-2025), it combines proven techniques that maximize efficiency gains.

| Feature | Solution | Params | Literature |
|---------|----------|--------|------------|
| **Rank Collapse Prevention** | Orthogonal init for A matrix | 0 | LongLoRA [1,2] |
| **Magnitude Normalization** | DoRA-style weight decomposition | d_out/layer | DoRA [Liu 2024] |
| **Lost-in-Middle** | PositionBias (log bucketing) | 64 shared | LIFT [6,8] |
| **Noise Filtering** | PositionalDAA | ~2K/layer | Lost-in-Middle |
| **Training Efficiency** | S²-Attn (Shifted Sparse) | 0 | LongLoRA [1,2] |
| **Context Extension** | Trainable Embeddings & Norms | ~10-20% | LongLoRA [1,2] |
| **Stable Attention** | Sink Tokens (Global) | 0 | SinkLoRA [9] |
| **Position Extension** | RoPE Scaling (YaRN/Linear) | 0 | YaRN [26] |

## Core Formula

```
output = (base + δ) * (m / ||V + ΔV||) * pos_scale

where:
  δ = (α/r) * x @ A^T @ B^T
  pos_scale = 1 + σ(position_scale) * tanh(position_bias[pos])
```

## Key Techniques

### 1. S²-Attn (Shifted Sparse Attention) - LongLoRA 
The primary efficiency technique. Reduces training complexity from O(n²) to O(n × group_size):
- Splits sequences into groups for memory efficiency
- Shifts groups on alternating layers for cross-group information flow
- **16x training cost reduction** with minimal accuracy impact
- Only applies during training; full attention during inference

### 2. Trainable Embeddings & Norms - LongLoRA 
Critical for context lengths >32k tokens. Unfreezes:
- Token embeddings (~10% of base params)
- Layer normalization parameters

### 3. RoPE Scaling - YaRN 
Extends positional encoding to longer contexts:
- **Linear**: Simple interpolation for moderate extension
- **Dynamic**: NTK-aware scaling for better resolution
- **YaRN**: Advanced entropy-based scaling for extreme lengths (>100k)

### 4. Sink Token Support - SinkLoRA 
Initial tokens receive global attention from all groups, preventing attention sink collapse.

### 5. PositionalDAA
Learns position-dependent attention biases to address the "Lost-in-the-Middle" phenomenon:

```python
attn' = α * attn + β + position_bias[bucket(q_pos, k_pos)]
```

## Usage

```python
from hylorada import HyLoRADAConfig, HyLoRADAModel

# Efficient long-context configuration
config = HyLoRADAConfig(
    lora_rank=8,
    train_embeddings=True,   # LongLoRA - for >32k context
    train_norms=True,        # LongLoRA normalization
    s2_sink_tokens=4,        # SinkLoRA - stable attention
    s2_attn_enabled=True,    # Enable S²-Attn for efficiency
    rope_scaling_type="linear",
    rope_scaling_factor=4.0,
)
model = HyLoRADAModel(base_model, config)
```

## Config Options

```python
HyLoRADAConfig(
    # Core LoRA
    lora_rank=8,              # LoRA rank
    lora_alpha=16.0,          # Scaling
    
    # Long-Context (LongLoRA)
    train_embeddings=False,   # Enable for >32k context
    train_norms=False,        # Enable for >32k context
    s2_attn_enabled=False,    # Shifted Sparse Attention
    s2_group_size=2048,       # Tokens per group
    s2_sink_tokens=0,         # SinkLoRA global tokens
    
    # Position Extension (YaRN)
    rope_scaling_type=None,   # "linear", "dynamic", "yarn"
    rope_scaling_factor=1.0, 
    
    # HyLoRADA Core
    position_bias_enabled=True,
    daa_enabled=True,         # PositionalDAA
    sparse_enabled=True,      # Sparse MLP adapters
)
```

## References

1. Chen et al., "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models" (2023)
2. Chen et al., "LongLoRA: Efficient Fine-tuning..." arXiv:2309.12307 (2023)
3. Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)
4. Mao et al., "LIFT: Improving Long Context Understanding..." (2025)
5. Zhang, "SinkLoRA: Enhanced Efficiency for Long-Context LLMs" (2024)
6. Peng et al., "YaRN: Efficient Context Window Extension of LLMs" (2023)
