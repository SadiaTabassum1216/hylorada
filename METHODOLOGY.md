# HyLoRADA: Hybrid Low-Rank Direct Attention Adaptation

## Overview

HyLoRADA is a unified PEFT method with five key innovations:

| Feature | Solution | Params |
|---------|----------|--------|
| **Rank Collapse Prevention** | Orthogonal init for A matrix | 0 |
| **Adaptive Magnitude** | Gated magnitude control | +1/layer |
| **Best of LoRA+DoRA** | Residual blend (learnable β) | +1/layer |
| **Lost-in-Middle** | PositionBias (log bucketing) | 64 shared |
| **Noise Filtering** | PositionalDAA | ~2K/layer |

## Core Formula

```
output = (1 - β) * DoRA_output + β * LoRA_output
```

Where:
- `DoRA_output = (base + δ) * (gate * m' + (1-gate) * m_init) / ||V + ΔV||`
- `LoRA_output = base + δ`
- `δ = (α/r) * x @ A^T @ B^T * pos_scale`
- `pos_scale = 1 + σ(position_scale) * tanh(position_bias[pos])`

## Parameter Count

| Component | Params per Layer |
|-----------|------------------|
| lora_A | r × d_in |
| lora_B | d_out × r |
| magnitude | d_out |
| magnitude_gate | 1 |
| residual_weight | 1 |
| position_scale | 1 |
| **PositionBias (shared)** | **64 total** |

For r=8, d=896: **~15K per layer + 64 shared**

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

config = HyLoRADAConfig(lora_rank=8)
model = HyLoRADAModel(base_model, config)
```

## Config Options

```python
HyLoRADAConfig(
    lora_rank=8,              # LoRA rank
    lora_alpha=16.0,          # Scaling
    position_bias_enabled=True,  # Lost-in-middle
    daa_enabled=True,         # Noise filtering
    sparse_enabled=True,      # Sparse MLP
    s2_attn_enabled=False,    # Long context (optional)
)
```

## References

1. Hu et al., "LoRA: Low-Rank Adaptation" (2021)
2. Liu et al., "DoRA: Weight-Decomposed LoRA" (2024)
3. Hayou et al., "LoRA+: Efficient Adaptation" (2024)
