# HyLoRADA

**Hybrid Low-Rank Adaptation with Landmark Memory** — Parameter-efficient fine-tuning combining rsLoRA, DoRA, and novel LandmarkLoRA for efficient context handling.

## Key Features

| Component | Description | Parameters |
|-----------|-------------|------------|
| **rsLoRA** | Rank-stabilized scaling (α/√r) | 0 |
| **DoRA** | Weight-decomposed magnitude | ~86K |
| **LandmarkLoRA** | Trainable context summary tokens | ~14K |
| **Position Bias** | Log-bucketed position awareness | 64 |

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from transformers import AutoModelForCausalLM
from hylorada import HyLoRADAConfig, HyLoRADAModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Apply HyLoRADA
config = HyLoRADAConfig(
    lora_rank=8,
    lora_alpha=16.0,
    use_dora_magnitude=True,    # DoRA magnitude normalization
    landmark_enabled=True,       # Novel: LandmarkLoRA
    num_landmarks=8,
    position_bias_enabled=True,
)

model = HyLoRADAModel(base_model, config)
model.print_trainable_params()
```

## Benchmark

```bash
# Compare methods
python run_benchmark.py --methods lora dora hylorada --epochs 3

# Long-context benchmark
python run_benchmark.py --methods hylorada --dataset longbench --max_length 4096
```

## Architecture

```
HyLoRADA = rsLoRA + DoRA + LandmarkLoRA + PositionBias

output = magnitude × norm(W + (α/√r)×BA) × pos_scale + landmark_context
```

**Novel Contribution**: LandmarkLoRA introduces trainable "landmark" tokens that learn to summarize context patterns during fine-tuning.

## Citation

```bibtex
@misc{hylorada2024,
  title={HyLoRADA: Hybrid Low-Rank Adaptation with Landmark Memory},
  year={2024},
}
```

## References

- [rsLoRA](https://arxiv.org/abs/2312.03732) - Rank Stabilization for LoRA
- [DoRA](https://arxiv.org/abs/2402.09353) - Weight-Decomposed Low-Rank Adaptation
- [Landmark Attention](https://arxiv.org/abs/2305.16300) - Random-Access Infinite Context
