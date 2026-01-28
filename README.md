# HyLoRADA: Hybrid Low-Rank Adaptation

**rsLoRA + DoRA + Hybrid Blend** — Parameter-efficient fine-tuning combining rank-stabilized scaling, DoRA magnitude normalization, and a learnable hybrid blend for optimal stability.

## Architecture

| Component | Description | Parameters | Source |
|-----------|-------------|------------|--------|
| **rsLoRA** | α/√r scaling for gradient stability | 0 | [Kalajdzievski 2024] |
| **DoRA** | Weight magnitude decomposition | ~86K/layer | [Liu et al. 2024] |
| **Position Bias** | Log-bucketed position awareness | 64 shared | [Liu et al. 2023] |
| **LandmarkLoRA** | Learnable context summary tokens | ~14K | [Novel] |

**Total**: ~105K trainable params per LoRA layer (extremely efficient)

## Quick Start

```python
from transformers import AutoModelForCausalLM
from hylorada import HyLoRADAConfig, HyLoRADAModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Apply HyLoRADA (rsLoRA + DoRA + Position Bias)
config = HyLoRADAConfig(
    lora_rank=8,
    lora_alpha=16.0,
    use_dora_magnitude=True,     # DoRA weight decomposition
    position_bias_enabled=True,   # Position-aware scaling
)

model = HyLoRADAModel(base_model, config)
model.print_trainable_params()
```

## Benchmark

```bash
# Compare methods
python run_benchmark.py --methods lora dora hylorada --epochs 3

# Long-context with S²-Attn
python run_benchmark.py --methods hylorada --s2_attn --max_length 4096
```

## Configuration

```python
config = HyLoRADAConfig(
    lora_rank=8,               # LoRA rank
    lora_alpha=16.0,           # rsLoRA: scaled by α/√r
    use_dora_magnitude=True,   # DoRA: magnitude normalization
    position_bias_enabled=True, # Position-aware adaptation
    
    # Optional: Long-context extensions
    s2_attn_enabled=False,     # Shifted-sparse attention
    train_embeddings=False,    # LongLoRA: trainable embeddings
    rope_scaling_type=None,    # "linear", "dynamic", "yarn"
)
```

## Architecture Details

```
HyLoRADA = rsLoRA + DoRA + PositionBias

W' = m × norm(W + (α/√r) × BA) × (1 + position_scale)

where:
- m: learnable magnitude vector (DoRA)
- α/√r: rank-stabilized scaling (rsLoRA)
- BA: low-rank decomposition
- position_scale: position-aware bias
```

## References

- [rsLoRA](https://arxiv.org/abs/2312.03732) - Rank Stabilization for LoRA
- [DoRA](https://arxiv.org/abs/2402.09353) - Weight-Decomposed Low-Rank Adaptation  
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
