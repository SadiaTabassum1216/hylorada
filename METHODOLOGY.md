# HyLoRADA: Hybrid Low-Rank Direct Attention Adaptation

## Overview

HyLoRADA is a unified PEFT method with eight key innovations for efficient long-context fine-tuning:

| Feature | Solution | Params |
|---------|----------|--------|
| **Rank Collapse Prevention** | Orthogonal init for A matrix | 0 |
| **Adaptive Magnitude** | Gated magnitude control | +1/layer |
| **Best of LoRA+DoRA** | Residual blend (learnable β) | +1/layer |
| **Lost-in-Middle** | PositionBias (log bucketing) | 64 shared |
| **Noise Filtering** | PositionalDAA | ~2K/layer |
| **Long Context Training** | Trainable Embeddings & Norms | ~10-20% of base |
| **Stable Attention** | Sink Tokens (Global Attention) | 0 |
| **Context Extension** | RoPE Scaling (YaRN/Linear) | 0 |

## Core Formula

```
output = (1 - β) * DoRA_output + β * LoRA_output
```

Where:
- `DoRA_output = (base + δ) * (gate * m' + (1-gate) * m_init) / ||V + ΔV||`
- `LoRA_output = base + δ`
- `δ = (α/r) * x @ A^T @ B^T * pos_scale`
- `pos_scale = 1 + σ(position_scale) * tanh(position_bias[pos])`

## Long-Context Extensions

### 1. Trainable Embeddings & Norms (LongLoRA)
To support context lengths >32k, HyLoRADA optionally unfreezes embedding layers and normalization layers. This is critical for adapting the model's internal representation space to longer sequences without full fine-tuning.

### 2. Sink Token Support (SinkLoRA)
Incorporates "sink tokens" (initial sequence tokens) into the Shifted Sparse Attention mechanism. These tokens are globally attended to by all groups, preventing attention sink collapse and ensuring stable long-term dependency modeling.

### 3. RoPE Scaling (YaRN)
Injects Rotary Position Embedding (RoPE) scaling factors directly into the base model configuration. Supports:
- **Linear**: Simple interpolation for moderate extension.
- **Dynamic**: NTK-aware scaling for better resolution.
- **YaRN**: Advanced entropy-based scaling for extreme context lengths (>100k).

## Components

### HyLoRADAUnified (lora.py)

```python
class HyLoRADAUnified(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0,
                 dropout=0.0, position_bias=None):
        self.lora_A = nn.Parameter(...)      # [rank, in_features]
        self.lora_B = nn.Parameter(...)      # [out_features, rank]
        self.magnitude = nn.Parameter(...)   # [out_features]
        self.magnitude_gate = nn.Parameter(...)  # scalar
        self.residual_weight = nn.Parameter(...) # scalar
        self.position_scale = nn.Parameter(...)  # scalar
```

### PositionBias (lora.py)

```python
class PositionBias(nn.Module):
    def __init__(self, num_buckets=64):
        self.bias = nn.Parameter(torch.zeros(num_buckets))
```

Uses logarithmic bucketing for distant positions.

### PositionalDAA (daa.py)

```python
attn' = α * attn + β + position_bias[bucket(q_pos, k_pos)]
```

## Usage

```python
from hylorada import HyLoRADAConfig, HyLoRADAModel

config = HyLoRADAConfig(
    lora_rank=8,
    train_embeddings=True,  # For >32k context
    s2_sink_tokens=4,       # Stable attention
    rope_scaling_type="linear",
    rope_scaling_factor=4.0
)
model = HyLoRADAModel(base_model, config)
```

## Config Options

```python
HyLoRADAConfig(
    lora_rank=8,              # LoRA rank
    lora_alpha=16.0,          # Scaling
    
    # Context Extension
    train_embeddings=False,   # LongLoRA (High Memory)
    train_norms=False,        # LongLoRA normalization
    s2_sink_tokens=0,         # SinkLoRA
    rope_scaling_type=None,   # "linear", "dynamic", "yarn"
    rope_scaling_factor=1.0, 
    
    # HyLoRADA Core
    position_bias_enabled=True,
    daa_enabled=True,         
    sparse_enabled=True,      
    s2_attn_enabled=False,    
)
```

## References

1. Hu et al., "LoRA: Low-Rank Adaptation" (2021)
2. Liu et al., "DoRA: Weight-Decomposed LoRA" (2024)
3. Chen et al., "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models" (2023)
4. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023)
5. Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2024)
