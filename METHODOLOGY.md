# HyLoRADA Methodology

## Abstract

HyLoRADA combines rank-stabilized LoRA (rsLoRA) with weight-decomposed adaptation (DoRA) in a novel **Hybrid Blend** architecture. The method dynamically balances between directional updates (DoRA) and standard low-rank updates (LoRA) using a learnable residual gate, while incorporating **LandmarkLoRA** for compressed context memory.

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

### 1.4 LandmarkLoRA: Context Compression

To enhance long-context capabilities, we introduce **LandmarkLoRA**, which adds learnable "landmark" tokens (context summaries) to the model. Unlike fixed-gating approaches, these landmarks are optimized during fine-tuning to compress and retain critical global information, functioning as a trainable memory bank.

## 2. Combined Architecture

**HyLoRADA Unified** = Hybrid(rsLoRA, DoRA) + PositionBias + LandmarkLoRA

```python
# 1. Compute Base Components
lora_update = (alpha / sqrt(rank)) * (x @ A.T @ B.T)
dora_norm = (W + lora_update) / ||W + lora_update||

# 2. Hybrid Blending (Learnable \beta)
# Balances structural capacity (DoRA) with relaxation (LoRA)
beta = sigmoid(residual_weight)
output_dora = magnitude * dora_norm
output_lora = W + lora_update

output = (1 - beta) * output_dora + beta * output_lora

# 3. Position & Landmark Scaling
output = output * (1 + position_scale(pos)) + landmark_summary
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

| Method | Scaling | Magnitude | Position | Landmark | Params/Layer |
|--------|---------|-----------|----------|----------|--------------|
| LoRA | α/r | - | - | ~87K |
| rsLoRA | α/√r | - | - | ~87K |
| DoRA | α/r | ✓ | - | - | ~91K |
| **HyLoRADA** | **α/√r** | **✓ (Gated)** | **✓** | **✓** | **~105K** |

## 5. Expected Benefits

1. **Better high-rank performance**: rsLoRA enables effective use of r=16, 32, 64
2. **Improved accuracy**: DoRA magnitude decomposition approaches full fine-tuning
3. **Position awareness**: Mitigates lost-in-middle effects with minimal overhead

## 6. Time Complexity Analysis
    
| Method | Training Complexity (per token) | Inference Overhead (Merged) | Inference Overhead (Dynamic) |
| :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | $O(d_{in} \cdot d_{out})$ | **Zero** | - |
| **Sparse Adapter** | $O(r_{ad} \cdot (d_{in} + d_{out}))$ | **High** (Cannot merge) | - |
| **LongLoRA** | Same as LoRA | **Zero** | Low (S²-Attn overhead) |
| **LoRA** | $O(r \cdot (d_{in} + d_{out}))$ | **Zero** | - |
| **DoRA** | $O(r \cdot (d_{in} + d_{out})) + O(d_{out})$ | **Zero** | - |
| **HyLoRADA** | $O(r \cdot (d_{in} + d_{out})) + O(d_{out})$ | **Zero** (Linear components) | Low (Position/Landmark) |

### 6.1 Training
- **Full Fine-Tuning**: Most expensive. Updates all parameters ($d \times d$), requiring massive optimizer memory.
- **LoRA / LongLoRA**: Efficient. Updates only low-rank matrices ($r \ll d$). LongLoRA further reduces self-attention complexity from $O(L^2)$ to $O(L \cdot G)$ via S²-Attn.
- **Sparse Adapters**: Efficient updates, but introduces parameter inefficiency due to separate adapter modules.
- **DoRA**: Adds slight overhead for weight normalization (calculating column norms).
- **HyLoRADA**: Hybrid blend maintains efficient graph size roughly equal to DoRA.

### 6.2 Inference
- **Merged Weights (Zero Overhead)**: FFT, LoRA, LongLoRA, DoRA, and HyLoRADA (linear parts) can all represent the final model as a single weight matrix $W' = W + \Delta W$.
- **Non-Mergeable (High Overhead)**: **Sparse Adapters** contain non-linearities (ReLU/GELU) between down/up projections, preventing merging. This adds latency $O(r \cdot d)$ to *every* forward pass.
- **Dynamic Components (Low Overhead)**:
    - **Position Bias (HyLoRADA)**: $O(1)$ scalar op per token.
    - **LandmarkLoRA**: $O(K \cdot d)$ for $K$ landmarks.
    - **S²-Attn (LongLoRA)**: Requires shifting/masking operations, adding minor overhead vs standard FlashAttention.
    
**Conclusion**: HyLoRADA provides advanced capabilities (long-context handling, stability) with virtually no inference latency penalty over standard models.

## References

1. Kalajdzievski, D. (2024). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.
2. Liu, S. et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation.
3. Liu, N. et al. (2023). Lost in the Middle: How Language Models Use Long Contexts.
4. Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
