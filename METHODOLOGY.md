# HyLoRADA: Hybrid Low-Rank Direct Attention Adaptation

## Overview

HyLoRADA is a novel Parameter-Efficient Fine-Tuning (PEFT) method that combines the best aspects of LoRA and DoRA with three key innovations:

1. **Orthogonal Initialization** - Prevents rank collapse during training
2. **Gated Magnitude** - Learnable control over weight magnitude contribution
3. **Residual LoRA Path** - Blends DoRA and LoRA learning dynamics

## Background

### LoRA (Low-Rank Adaptation)
Standard LoRA represents weight updates as:
```
ΔW = B @ A
W' = W + (α/r) * B @ A
```
Where:
- `A ∈ R^(r×d_in)` - Down-projection matrix
- `B ∈ R^(d_out×r)` - Up-projection matrix
- `r` - Rank (typically 4-16)
- `α` - Scaling factor

### DoRA (Weight-Decomposed LoRA)
DoRA decomposes the weight into magnitude and direction:
```
W = m * (V / ||V||)
```
Then applies LoRA only to the direction:
```
W' = m' * ((V + ΔV) / ||V + ΔV||)
```
Where:
- `m'` - Learnable magnitude vector
- `ΔV = B @ A` - LoRA update

## HyLoRADA Architecture

### Core Formula

```
output = (1 - β) * DoRA_output + β * LoRA_output
```

Where:
- `DoRA_output = (base + δ) * (gate * m' + (1-gate) * m_init) / ||V + ΔV||`
- `LoRA_output = base + δ`
- `δ = (α/r) * x @ A^T @ B^T`
- `gate = σ(g)` - Learnable gate (sigmoid of learnable parameter)
- `β = σ(b)` - Residual weight (sigmoid of learnable parameter)

### Innovation 1: Orthogonal Initialization

Standard LoRA uses Kaiming initialization for matrix A:
```python
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
```

**Problem**: Kaiming init can lead to rank collapse where the effective rank decreases during training.

**HyLoRADA Solution**: Use orthogonal initialization:
```python
nn.init.orthogonal_(self.lora_A)
```

**Benefits**:
- Preserves rank throughout training
- Better gradient flow through low-rank matrices
- More stable optimization

### Innovation 2: Gated Magnitude

DoRA uses a fixed learnable magnitude `m'`. 

**Problem**: The magnitude may not need to change equally for all inputs.

**HyLoRADA Solution**: Add a learnable gate that controls how much the learned magnitude contributes:
```python
gate = torch.sigmoid(self.magnitude_gate)
effective_magnitude = m' * gate + m_init * (1 - gate)
```

**Benefits**:
- If gate → 0: Uses original base weights (conservative)
- If gate → 1: Uses fully learned magnitude (aggressive)
- Model learns optimal balance per layer

### Innovation 3: Residual LoRA Path

DoRA and LoRA have different learning dynamics:
- **DoRA**: Better for direction changes, normalized outputs
- **LoRA**: Better for magnitude changes, simpler gradients

**HyLoRADA Solution**: Blend both with a learnable weight:
```python
residual_w = torch.sigmoid(self.residual_weight)
output = (1 - residual_w) * dora_output + residual_w * lora_output
```

**Benefits**:
- Combines strengths of both methods
- Model learns optimal blend per layer
- More expressive than either alone

## Implementation

### HyLoRADALinear Class

```python
class HyLoRADALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Magnitude (from DoRA)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Novel: Gated magnitude
        self.magnitude_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        
        # Novel: Residual LoRA weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Novel: Orthogonal init
        nn.init.orthogonal_(self.lora_A)
        nn.init.zeros_(self.lora_B)
```

### Parameter Count

| Component | Parameters |
|-----------|------------|
| lora_A | r × d_in |
| lora_B | d_out × r |
| magnitude | d_out |
| magnitude_gate | 1 |
| residual_weight | 1 |
| **Total per layer** | r(d_in + d_out) + d_out + 2 |

For Qwen2.5-0.5B (d=896, r=8):
- Per attention layer: 8×896 + 896×8 + 896 + 2 = **15,234 params**
- 24 layers × 4 projections × 15,234 = **~1.46M trainable params**

## Training

### LoRA+ Learning Rates (Optional)

HyLoRADA supports asymmetric learning rates from LoRA+:

```python
param_groups = [
    {"params": lora_A_params, "lr": base_lr},
    {"params": lora_B_params, "lr": base_lr * 10},  # 10x for B
    {"params": magnitude_params, "lr": base_lr},
    {"params": gate_params, "lr": base_lr},
]
```

### Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| lora_rank | 8 | [4, 16] | Higher = more capacity |
| alpha_ratio | 2.0 | [1.5, 3.0] | alpha = rank × ratio |
| lora_plus_ratio | 10.0 | [5, 20] | LR multiplier for B |
| gate_init | 0.0 | [-1, 1] | sigmoid input; 0 = 0.5 |
| residual_init | 0.1 | [0, 0.5] | sigmoid input |
| lora_dropout | 0.05 | [0, 0.1] | Regularization |

### Hyperparameter Optimization

Use Bayesian optimization with Optuna:
```bash
python optimize_hylorada.py --n_trials 15 --epochs 2
```

## Benchmark Results

| Method | Params | PPL | LiM PPL |
|--------|--------|-----|---------|
| LoRA | 540K | 32.11 | 25.58 |
| LoRaDA | 1.0M | 30.06 | 24.24 |
| DoRA | 1.1M | 29.91 | 24.24 |
| **HyLoRADA** | 1.1M | **29.24** | **23.37** |

**Improvement over DoRA**: -0.67 PPL (2.2% improvement)

## Usage

### Basic Usage

```python
from hylorada import HyLoRADAModel, HyLoRADAConfig

config = HyLoRADAConfig(
    lora_rank=8,
    lora_alpha=16.0,
    use_hylorada=True,
    lora_plus_enabled=True,
)

model = HyLoRADAModel(base_model, config)
```

### Benchmark

```bash
python run_benchmark.py --methods lora dora lorada hylorada --epochs 3
```

## File Structure

```
hylorada/
├── hylorada/
│   ├── __init__.py
│   ├── config.py         # HyLoRADAConfig dataclass
│   ├── lora.py           # HyLoRADALinear, HyLoRADALayer
│   ├── model.py          # HyLoRADAModel wrapper
│   └── trainer.py        # HyLoRADATrainer
├── run_benchmark.py      # Benchmark script
├── optimize_hylorada.py  # Bayesian HPO with Optuna
└── hylorada.ipynb        # Kaggle notebook
```

## References

1. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
2. Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)
3. Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models" (2024)

## Citation

```bibtex
@misc{hylorada2026,
  title={HyLoRADA: Hybrid Low-Rank Direct Attention Adaptation},
  author={Your Name},
  year={2026},
  url={https://github.com/SadiaTabassum1216/hylorada}
}
```
