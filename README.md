# HyLoRADA

HyLoRADA is a novel parameter-efficient fine-tuning method for LLMs.

## Novel Contributions

### HyLoRADA v1
1. **Orthogonal Initialization** - Prevents rank collapse during training
2. **Gated Magnitude** - Learnable control over weight magnitude contribution
3. **Residual LoRA Path** - Blends DoRA and LoRA learning dynamics

### HyLoRADA v2 (New!)
4. **Structure-Conditioned Scaling** - Position-dependent LoRA adaptation for code, time series, and graph data

## Benchmark Results

| Method | Params | PPL | LiM PPL |
|--------|--------|-----|---------|
| LoRA | 540K | 31.79 | 25.60 |
| DoRA | 1.1M | 30.42 | 24.45 |
| LoRaDA | 1.0M | 30.40 | 24.27 |
| **HyLoRADA v1** | 2.2M | **27.01** | **19.66** |

**Improvements over DoRA**: -3.41 PPL (11% better)  
**Improvements on LiM**: -4.61 PPL (19% better)

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### HyLoRADA v1
```python
from transformers import AutoModelForCausalLM
from hylorada import HyLoRADAConfig, HyLoRADAModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
config = HyLoRADAConfig(
    lora_rank=16,
    use_hylorada=True,
    lora_plus_enabled=True,
)
hylorada_model = HyLoRADAModel(model, config)
```

### HyLoRADA v2 (Structure-Aware)
```python
from hylorada import HyLoRADAConfig, StructureEncoder, HyLoRADAv2Linear

config = HyLoRADAConfig(
    use_hylorada_v2=True,
    structure_dim=32,
)

# Create structure encoder for position-aware scaling
encoder = StructureEncoder(hidden_size=32)
```

## Run Benchmarks

```bash
# Standard benchmark
python run_benchmark.py --methods lora dora lorada hylorada --epochs 3

# Code task (v2)
python run_code_task.py --use_v2 --epochs 3
```

## Project Structure

```
hylorada/
├── hylorada/
│   ├── config.py            # Configuration
│   ├── lora.py              # LoRA / DoRA / HyLoRADA v1 & v2
│   ├── structure_encoder.py # Structure encoder (v2)
│   ├── model.py             # HyLoRADAModel wrapper
│   ├── daa.py               # Direct Attention Adaptation
│   ├── sparse_mlp.py        # Sparse MLP adapters
│   ├── trainer.py           # Training utilities
│   └── evaluation.py        # Metrics
├── run_benchmark.py         # Benchmark script
├── run_code_task.py         # Code task training
├── optimize_hylorada.py     # Bayesian HPO
├── METHODOLOGY.md           # Technical documentation
└── tests/                   # Unit tests
```

## Testing

```bash
python -m pytest tests/ -v
```

## Version

- **v0.2.0** - Added HyLoRADA v2 with structure-aware adaptation
- **v0.1.0** - Initial release with HyLoRADA v1
