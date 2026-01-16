# HyLoRADA

**Hybrid Low-Rank Direct Attention Adaptation** - A novel parameter-efficient fine-tuning method for LLMs.

## Novel Contributions

HyLoRADA improves upon DoRA with three key innovations:

1. **Orthogonal Initialization** - Prevents rank collapse during training
2. **Gated Magnitude** - Learnable control over weight magnitude contribution
3. **Residual LoRA Path** - Blends DoRA and LoRA learning dynamics

## Benchmark Results

| Method | Params | PPL | LiM PPL |
|--------|--------|-----|---------|
| LoRA | 540K | 32.11 | 25.58 |
| LoRaDA | 1.0M | 30.06 | 24.24 |
| DoRA | 1.1M | 29.91 | 24.24 |
| **HyLoRADA** | 1.1M | **29.24** | **23.37** |

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

## Baseline Methods

| Method | Description |
|--------|-------------|
| **LoRA** | Standard low-rank adaptation |
| **DoRA** | Weight-decomposed LoRA |
| **LoRaDA** | LoRA + Direct Attention Adaptation |
| **HyLoRADA** | Our method (orthogonal + gated + residual) |

```python
from hylorada import StandardLoRA, LoRaDAModel

# Use baselines for comparison
model = StandardLoRA(base_model)
model = LoRaDAModel(base_model)
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

## Citation

```bibtex
@misc{hylorada2026,
  title={HyLoRADA: Hybrid Low-Rank Direct Attention Adaptation},
  author={Sadia Tabassum},
  year={2026},
  url={https://github.com/SadiaTabassum1216/hylorada}
}
```
