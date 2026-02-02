# HyLoRADA: Hybrid Low-Rank Adaptation with Position-Adaptive Landmarks

**Parameter-efficient fine-tuning framework** validated through comprehensive ablation studies. Achieves **+18.37% perplexity improvement** on WikiText-2 using rsLoRA + Position Bias + Position-Adaptive Landmarks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](./hylorada.ipynb)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hylorada.git
cd hylorada

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 🚀 Run on Kaggle

1. Open [hylorada.ipynb](./hylorada.ipynb) in Kaggle
2. Enable GPU (Settings → Accelerator → GPU)
3. Run cells in order to validate the ablation results

### Training Example

```python
from hylorada.trainer import HyLoRADATrainer, TrainingConfig

# Configure training
train_config = TrainingConfig(
    num_epochs=3,
    learning_rate=2e-4,
    per_device_batch_size=1,
    gradient_accumulation_steps=16,
    mixed_precision="bf16",
)

# Create trainer
trainer = HyLoRADATrainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    config=train_config,
)

# Train
trainer.train()
```

## Validation & Testing

### Comprehensive Ablation Study

```bash
# Run complete ablation test (validates all components)
python test_ablation_proper.py

# Expected results:
# - Baseline: 69.00 PPL
# - rsLoRA: 57.22 PPL (+17.07%)
# - +Position Bias: 59.16 PPL (+2.11%)
# - +Position-Adaptive: 56.33 PPL (+18.37% total) ✅ BEST
```

### Individual Component Tests

```bash
# Test landmark architectures
python test_landmarks.py --epochs 1 --num_train 500

# Unit tests
pytest tests/
```

## Benchmarking (Legacy)

For custom benchmarking, use `run_benchmark.py`:

```bash
# Basic comparison on WikiText-2
python run_benchmark.py \
    --methods lora hylorada \
    --epochs 3 \
    --max_length 1024
    --s2_attn \
    --max_length 4096 \
    --epochs 3

# Code domain (CodeSearchNet)
python run_benchmark.py \
    --dataset code \
    --methods lora hylorada \
    --epochs 3
```

### Supported Methods

- `lora`: Standard LoRA (baseline)
- `dora`: DoRA (weight decomposition)
- `lorada`: LoRA + Direct Attention Adaptation
- `longlora`: LongLoRA (trainable embeddings)
- `sparse`: SparseAdapter (MLP-only)
- `hylorada`: Full HyLoRADA (our method)

## Evaluation

### Perplexity Evaluation

```python
from hylorada.evaluation import evaluate_perplexity

perplexity, loss = evaluate_perplexity(
    model=model,
    dataset=test_dataset,
    tokenizer=tokenizer,
    max_length=4096,
)
print(f"Perplexity: {perplexity:.2f}")
```

### Lost-in-the-Middle Analysis

```python
from hylorada.evaluation import evaluate_lost_in_the_middle

results = evaluate_lost_in_the_middle(
    model=model,
    dataset=test_dataset,
    tokenizer=tokenizer,
    num_samples=100,
)
# Returns position-wise perplexity distribution
```

### Full Benchmark Suite

```python
from hylorada.evaluation import run_full_evaluation

metrics = run_full_evaluation(
    model=model,
    test_dataset=test_dataset,
    tokenizer=tokenizer,
    config=config,
)
```

## Key Features

✅ **Validated Components**:
- **rsLoRA**: Rank-stabilized LoRA (+17% improvement)
- **Position-Adaptive Landmarks**: Context-aware selection (+18.4% improvement, best result)
- **Position Bias**: Lightweight position scaling (+2% improvement)

⚠️ **Experimental Components**:
- **DoRA**: Magnitude decomposition (can degrade performance, use cautiously)
- **Learnable Bucketing**: Adaptive position boundaries (currently degrades vs fixed)

## Documentation

- **[METHODOLOGY.md](METHODOLOGY.md)**: Complete technical methodology and architecture
- **[RESEARCH.md](RESEARCH.md)**: Research contributions, validation results, and publication strategy
- **[ABLATION_FINDINGS.md](ABLATION_FINDINGS.md)**: Comprehensive ablation results and empirical findings
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Project organization and file structure
- **[requirements.txt](requirements.txt)**: Python dependencies

## Testing

```bash
# Validate all components
python test_validation.py

# Compare landmark architectures
python test_landmarks.py --epochs 1 --num_train 500

# Test learnable bucketing
python test_learnable_bucketing.py --epochs 1 --num_train 500

# Comprehensive ablation study
python test_ablation_proper.py --epochs 1 --num_train 500
```

## Acknowledgments

HyLoRADA builds upon excellent prior work including LoRA, DoRA, LongLoRA, and rsLoRA. We thank the authors of these methods for their foundational contributions to parameter-efficient fine-tuning.
