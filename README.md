# HyLoRADA

**Unified HyLoRADA** - Parameter-efficient fine-tuning for long-context LLMs.

## Key Features

| Feature | Solution | Params |
|---------|----------|--------|
| **Rank Collapse Prevention** | Orthogonal init | 0 |
| **Adaptive Magnitude** | Gated magnitude | +1/layer |
| **Best of LoRA+DoRA** | Residual blend | +1/layer |
| **Lost-in-Middle** | PositionBias | 64 shared |
| **Noise Filtering** | PositionalDAA | ~2K/layer |

## Quick Start

```python
from transformers import AutoModelForCausalLM
from hylorada import HyLoRADAConfig, HyLoRADAModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
config = HyLoRADAConfig(lora_rank=8)
wrapped = HyLoRADAModel(model, config)
wrapped.print_trainable_params()
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
├── config.py         # Configuration
├── lora.py           # HyLoRADAUnified, PositionBias
├── daa.py            # PositionalDAA
├── model.py          # HyLoRADAModel wrapper
├── baselines.py      # Comparison methods
└── evaluation.py     # Metrics
```

## Version

- **v0.3.0** - Unified architecture
