# HyLoRADA

**Hybrid Low-Rank Adaptation with Direct Attention** - A parameter-efficient fine-tuning framework for long-context LLMs.

## Features

- **LoRA** - Global context adaptation via low-rank decomposition
- **DAA** - Direct Attention Adaptation for noise filtering (Lost-in-the-Middle)
- **Sparse MLP** - Top-k neuron selection for local precision
- **S²-Attn** - Shifted Sparse Attention for memory efficiency

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
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Apply HyLoRADA
config = HyLoRADAConfig(lora_rank=8, daa_enabled=True, sparse_enabled=True)
hylorada_model = HyLoRADAModel(model, config)
hylorada_model.print_trainable_params()
```

## Training

### Basic Training
```bash
python example_train.py --model_name Qwen/Qwen2-0.5B --num_epochs 1
```

### Full Options
```bash
python example_train.py \
    --model_name Qwen/Qwen2-0.5B \
    --max_length 2048 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --output_dir ./output
```

## Evaluation & Experiments

### Run Full Experiment (Baseline vs HyLoRADA)
```bash
python run_experiment.py \
    --model_name Qwen/Qwen2-0.5B \
    --max_length 1024 \
    --num_epochs 1 \
    --num_test_samples 50
```

Results are saved to `./experiments/exp_TIMESTAMP/results.json`

### Quick Test (CPU)
```bash
python run_experiment.py \
    --model_name sshleifer/tiny-gpt2 \
    --max_length 64 \
    --num_epochs 1 \
    --num_test_samples 10 \
    --gradient_accumulation 2
```

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
hylorada/
├── hylorada/
│   ├── config.py       # HyLoRADA configuration
│   ├── lora.py         # LoRA adapters
│   ├── daa.py          # Direct Attention Adaptation
│   ├── sparse_mlp.py   # Sparse MLP with Top-k gating
│   ├── s2_attention.py # Shifted Sparse Attention
│   ├── model.py        # Main HyLoRADAModel wrapper
│   ├── trainer.py      # Training utilities
│   └── evaluation.py   # Evaluation metrics
├── example_train.py    # Training script
├── run_experiment.py   # Full experiment runner
└── tests/              # Unit tests
```