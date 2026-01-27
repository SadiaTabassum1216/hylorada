# HyLoRADA: Hybrid Low-Rank Adaptation with Landmark Memory

## Abstract

We present HyLoRADA, a parameter-efficient fine-tuning (PEFT) method that combines rank-stabilized LoRA (rsLoRA), weight-decomposed adaptation (DoRA), and a novel **LandmarkLoRA** module for efficient context summarization. HyLoRADA achieves competitive perplexity while maintaining parameter efficiency comparable to standard LoRA. Our key contribution is LandmarkLoRA—trainable "landmark" tokens that learn to summarize context patterns during fine-tuning, providing a form of learned memory compression.

## 1. Introduction

Parameter-efficient fine-tuning methods like LoRA have enabled adaptation of large language models with minimal additional parameters. However, challenges remain:

1. **Gradient instability at high ranks** — Standard LoRA scaling (α/r) causes gradient collapse
2. **Suboptimal learning dynamics** — Equal treatment of LoRA matrices A and B
3. **Context compression** — No mechanism for summarizing long-range context

HyLoRADA addresses these through a unified approach combining proven techniques with a novel landmark-based context module.

## 2. Method

### 2.1 rsLoRA Scaling

Following [Kalajdzievski 2024], we use rank-stabilized scaling:

$$\Delta W = \frac{\alpha}{\sqrt{r}} \cdot BA$$

This prevents gradient magnitude collapse at higher ranks, enabling stable training with r ∈ {8, 16, 32, 64}.

### 2.2 DoRA Magnitude Decomposition

Following [Liu et al. 2024], we decompose weight updates into magnitude and direction:

$$W' = m \cdot \frac{W + \Delta W}{\|W + \Delta W\|}$$

Where m is a learnable per-output-feature magnitude vector. This matches full fine-tuning accuracy while remaining parameter-efficient.

### 2.3 LandmarkLoRA (Novel Contribution)

Inspired by Landmark Attention [Mohtashami & Jaggi 2023], we introduce **trainable landmark tokens** as LoRA adapters:

$$\text{context} = \text{softmax}(g(h)) \cdot L$$
$$h' = h + \gamma \cdot \text{context}$$

Where:
- $L \in \mathbb{R}^{K \times d}$ — K learnable landmark tokens
- $g: \mathbb{R}^d \rightarrow \mathbb{R}^K$ — Gate projection
- $\gamma$ — Learnable scale factor

**Key difference from Landmark Attention**: While Landmark Attention uses fixed block gating for retrieval, LandmarkLoRA uses *trainable* landmarks that *learn* to capture important context patterns during fine-tuning.

**Parameters**: Only $2Kd$ additional parameters (∼14K for K=8, d=896).

### 2.4 Position-Aware Bias

To address the "Lost-in-the-Middle" phenomenon [Liu et al. 2023], we add learnable position biases:

$$\text{scale} = 1 + \sigma(\text{pos\_scale}) \cdot \tanh(\text{bias}[b(p)])$$

Where b(p) maps positions to 64 logarithmic buckets (only 64 shared parameters).

## 3. Architecture Summary

| Component | Parameters | Source |
|-----------|------------|--------|
| rsLoRA (α/√r) | 0 | [Kalajdzievski 2024] |
| DoRA magnitude | d_out per layer | [Liu et al. 2024] |
| LandmarkLoRA | 2Kd (∼14K) | **Novel** |
| Position bias | 64 shared | [Liu et al. 2023] |

**Total trainable**: ~100K per layer + 14K landmarks (for 8 landmarks)

## 4. Implementation

```python
from hylorada import HyLoRADAConfig, HyLoRADAModel

config = HyLoRADAConfig(
    lora_rank=8,
    lora_alpha=16.0,
    use_dora_magnitude=True,   # DoRA
    landmark_enabled=True,      # LandmarkLoRA (Novel)
    num_landmarks=8,
    position_bias_enabled=True,
)

model = HyLoRADAModel(base_model, config)
```

## 5. Conclusion

HyLoRADA provides a unified framework combining established PEFT techniques (rsLoRA, DoRA) with a novel LandmarkLoRA module for context summarization. The approach maintains parameter efficiency while providing mechanisms for improved long-context handling.

## References

1. Kalajdzievski, D. (2024). A Rank Stabilization Factor for Fine-Tuning with LoRA.
2. Liu, S. et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation.
3. Mohtashami, A. & Jaggi, M. (2023). Landmark Attention: Random-Access Infinite Context Length.
4. Liu, N. et al. (2023). Lost in the Middle: How Language Models Use Long Contexts.
5. Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
