# HyLoRADA

HyLoRADA is a novel parameter-efficient fine-tuning method for LLMs.

## Novel Contributions

HyLoRADA improves upon DoRA with three key innovations:

1. **Orthogonal Initialization** - Prevents rank collapse during training
2. **Gated Magnitude** - Learnable control over weight magnitude contribution
3. **Residual LoRA Path** - Blends DoRA and LoRA learning dynamics

## Benchmark Results

| Method | Params | PPL | LiM PPL |
|--------|--------|-----|---------|
| LoRA | 540K | 31.79 | 25.60 |
| DoRA | 1.1M | 30.42 | 24.45 |
| LoRaDA | 1.0M | 30.40 | 24.27 |
| **HyLoRADA** | 2.2M | **27.01** | **19.66** |

**Improvements over DoRA**: -3.41 PPL (11% better)  
**Improvements on LiM**: -4.61 PPL (19% better)

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from transformers import AutoModelForCausalLM
from hylorada import HyLoRADAConfig, HyLoRADAModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Apply HyLoRADA
config = HyLoRADAConfig(
    lora_rank=16,
    lora_alpha=47.2,
    use_hylorada=True,
    lora_plus_enabled=True,
    lora_plus_ratio=17.1,
)
hylorada_model = HyLoRADAModel(model, config)
hylorada_model.print_trainable_params()
```

## Benchmark All Methods

```bash
python run_benchmark.py --methods lora dora lorada hylorada --epochs 3
```

## Hyperparameter Optimization

```bash
pip install optuna
python optimize_hylorada.py --n_trials 15 --epochs 2
```

## Project Structure

```
hylorada/
├── hylorada/
│   ├── config.py       # Configuration
│   ├── lora.py         # LoRA / DoRA / HyLoRADA
│   ├── model.py        # HyLoRADAModel wrapper
│   ├── baselines.py    # Baseline methods
│   ├── trainer.py      # Training utilities
│   └── evaluation.py   # Metrics
├── run_benchmark.py    # Benchmark script
├── optimize_hylorada.py # Bayesian HPO
├── METHODOLOGY.md      # Technical documentation
└── tests/              # Unit tests
```

## Testing

```bash
python -m pytest tests/ -v
```
