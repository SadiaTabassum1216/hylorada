# HyLoRADA

**Unified HyLoRADA** - Parameter-efficient fine-tuning for long-context LLMs.

## Key Features

| Feature | Solution | Parameter Cost |
|---------|----------|----------------|
| **Long Context** | S²-Attn grouping | Optional |
| **Lost-in-Middle** | PositionBias | 64 params (shared) |
| **Noise Filtering** | PositionalDAA | ~2K per layer |
| **Rank Collapse** | Orthogonal init | 0 extra |
| **Adaptive Control** | Gated magnitude | +1 per layer |

## Quick Start

```python
from transformers import AutoModelForCausalLM
from hylorada import HyLoRADAConfig, HyLoRADAModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
config = HyLoRADAConfig(lora_rank=8)
wrapped = HyLoRADAModel(model, config)
wrapped.print_trainable_params()
```

## Benchmark Results

| Method | Params | PPL | LiM PPL |
|--------|--------|-----|---------|
| LoRA | 540K | 31.79 | 25.60 |
| DoRA | 1.1M | 30.42 | 24.45 |
| **HyLoRADA** | 1.5M | **27.01** | **19.66** |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run benchmark
python run_benchmark.py --methods hylorada --epochs 3

# Code task
python run_code_task.py --epochs 3

# Tests
python -m pytest tests/ -v
```

## Project Structure

```
hylorada/
├── hylorada/
│   ├── config.py         # Configuration
│   ├── lora.py           # HyLoRADAUnified, PositionBias
│   ├── daa.py            # PositionalDAA
│   ├── model.py          # HyLoRADAModel wrapper
│   ├── baselines.py      # Comparison methods
│   └── evaluation.py     # Metrics
├── run_benchmark.py      # Benchmark script
├── run_code_task.py      # Training script
└── tests/                # Unit tests
```

## Version History

- **v0.3.0** - Unified architecture (current)
- **v0.2.0** - Structure-aware v2
- **v0.1.0** - Initial release
