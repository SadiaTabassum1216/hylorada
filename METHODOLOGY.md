# HyLoRADA Methodology

## Abstract

HyLoRADA combines rank-stabilized LoRA (rsLoRA) with weight-decomposed adaptation (DoRA) for improved parameter-efficient fine-tuning. The method addresses gradient instability at high ranks while maintaining accuracy through magnitude-direction decomposition.

## 1. Core Components

### 1.1 rsLoRA: Rank-Stabilized Scaling

Standard LoRA uses scaling factor α/r, which causes gradient magnitude to decrease as rank increases. This leads to suboptimal performance at higher ranks.

**rsLoRA Solution** [Kalajdzievski 2024]:

$$\Delta W = \frac{\alpha}{\sqrt{r}} \cdot BA$$

Using α/√r instead of α/r stabilizes gradient magnitudes across different ranks, enabling effective training with r ∈ {8, 16, 32, 64}.

**Benefits**:
- Stable gradients at high ranks
- Better performance with larger rank values
- No additional parameters

### 1.2 DoRA: Weight-Decomposed Adaptation

DoRA [Liu et al. 2024] decomposes weight updates into magnitude and direction components, similar to how normalization techniques work.

**Formulation**:

$$W' = m \odot \frac{W + \Delta W}{\|W + \Delta W\|}$$

Where:
- m ∈ ℝ^(d_out): learnable magnitude vector
- ∆W = BA: LoRA update
- Division is column-wise normalization

**Benefits**:
- Matches full fine-tuning accuracy
- Separates learning of magnitude vs. direction
- Only d_out additional parameters per layer (~86K for typical attention layers)

### 1.3 Position-Aware Bias

To address lost-in-the-middle problems [Liu et al. 2023], we add learnable position biases:

$$\text{scale}(p) = 1 + \sigma(\text{w}) \cdot \tanh(\text{bias}[\text{bucket}(p)])$$

Using 64 logarithmic position buckets (only 64 shared parameters total).

## 2. Combined Architecture

**HyLoRADA** = rsLoRA + DoRA + PositionBias

```python
# LoRA update with rsLoRA scaling
lora_update = (alpha / sqrt(rank)) * (x @ A.T @ B.T)

# DoRA magnitude normalization
normalized = (W + lora_update) / ||W + lora_update||
output = magnitude * normalized

# Position-aware scaling
output = output * (1 + position_scale)
```

**Parameters per layer**:
- LoRA matrices: 2 × rank × d (~87K for rank=8, d=4096)
- DoRA magnitude: d_out (~4096)
- Position bias: 64 (shared across all layers)

**Total**: ~91K per layer (vs. 87K for standard LoRA)

## 3. Implementation

```python
from hylorada import HyLoRADAConfig, HyLoRADAModel

config = HyLoRADAConfig(
    lora_rank=8,               # Rank r
    lora_alpha=16.0,           # α (scaled by 1/√r)
    use_dora_magnitude=True,   # Enable DoRA
    position_bias_enabled=True, # Enable position bias
)

model = HyLoRADAModel(base_model, config)
```

## 4. Key Differences from Standard Methods

| Method | Scaling | Magnitude | Position | Params/Layer |
|--------|---------|-----------|----------|--------------|
| LoRA | α/r | - | - | ~87K |
| rsLoRA | α/√r | - | - | ~87K |
| DoRA | α/r | ✓ | - | ~91K |
| **HyLoRADA** | **α/√r** | **✓** | **✓** | **~91K** |

## 5. Expected Benefits

1. **Better high-rank performance**: rsLoRA enables effective use of r=16, 32, 64
2. **Improved accuracy**: DoRA magnitude decomposition approaches full fine-tuning
3. **Position awareness**: Mitigates lost-in-middle effects with minimal overhead

## References

1. Kalajdzievski, D. (2024). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.
2. Liu, S. et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation.
3. Liu, N. et al. (2023). Lost in the Middle: How Language Models Use Long Contexts.
4. Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
