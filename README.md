# HyLoRADA

**Hybrid Low-Rank Adaptation with Direct Attention** - A parameter-efficient fine-tuning framework for long-context LLMs.

## Features

- **LoRA** - Global context adaptation via low-rank decomposition (Q, K, V, O projections)
- **PositionalDAA** - Position-aware Direct Attention Adaptation for noise filtering (addresses Lost-in-the-Middle)
- **Sparse MLP** - Large-Sparse strategy with Top-k neuron selection for local precision
- **S²-Attn** - Shifted Sparse Attention for memory efficiency (optional)

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

# Apply HyLoRADA (all features enabled by default)
config = HyLoRADAConfig(lora_rank=8)
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
    --num_epochs 10 \
    --num_test_samples 50
```

### Compare Multiple PEFT Methods
```bash
# Compare all methods: Baseline, LoRA, LoRaDA, LongLoRA, SparseAdapter, HyLoRADA
python run_comparison.py \
    --model_name Qwen/Qwen2-0.5B \
    --max_length 1024 \
    --num_epochs 3

# Compare specific methods only
python run_comparison.py \
    --methods baseline lora lorada hylorada \
    --num_epochs 3
```

Results are saved to `./experiments/comparison_TIMESTAMP/comparison_results.json`

## Baseline Methods

HyLoRADA includes implementations of other PEFT methods for fair comparison:

| Method | Description | Components |
|--------|-------------|------------|
| **LoRA** | Standard low-rank adaptation | Q, V projections only |
| **LoRaDA** | LoRA + Direct Attention Adaptation | LoRA + DAA |
| **LongLoRA** | LoRA + trainable embeddings/norms | LoRA + embed + LayerNorm |
| **SparseAdapter** | Sparse FFN adapters only | Sparse MLP |
| **HyLoRADA** | Full hybrid approach | LoRA + PositionalDAA + Sparse MLP |

```python
from hylorada import StandardLoRA, LoRaDAModel, LongLoRAModel, SparseAdapterModel

# Use any baseline
model = StandardLoRA(base_model)
model = LoRaDAModel(base_model)
model = LongLoRAModel(base_model)
model = SparseAdapterModel(base_model)
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
│   ├── daa.py          # Direct Attention Adaptation (DAA + PositionalDAA)
│   ├── sparse_mlp.py   # Sparse MLP with Top-k gating
│   ├── s2_attention.py # Shifted Sparse Attention
│   ├── model.py        # Main HyLoRADAModel wrapper
│   ├── baselines.py    # Baseline methods (LoRA, LoRaDA, LongLoRA, SparseAdapter)
│   ├── trainer.py      # Training utilities
│   └── evaluation.py   # Evaluation metrics
├── example_train.py    # Training script
├── run_experiment.py   # Single experiment runner
├── run_comparison.py   # Multi-method comparison
└── tests/              # Unit tests
```

## Configuration Presets

```python
from hylorada.config import HyLoRADAPresets

# Memory-efficient config
config = HyLoRADAPresets.efficient()

# Balanced default config
config = HyLoRADAPresets.balanced()

# Maximum accuracy config
config = HyLoRADAPresets.high_accuracy()

# Long context (128k+) config
config = HyLoRADAPresets.long_context_128k()
```
