# HyLoRADA: Hybrid Low-Rank Adaptation

**Parameter-efficient fine-tuning framework** combining orthogonal initialization, DoRA magnitude decomposition, and position-aware scaling for efficient long-context learning.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hylorada.git
cd hylorada

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

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

## Benchmarking

Compare HyLoRADA against baseline methods:

```bash
# Basic comparison on WikiText-2
python run_benchmark.py \
    --methods lora dora hylorada \
    --epochs 3 \
    --max_length 1024

# Long-context with S²-Attn (4K context)
python run_benchmark.py \
    --methods hylorada \
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

## Acknowledgments

HyLoRADA builds upon excellent prior work including LoRA, DoRA, LongLoRA, and rsLoRA. We thank the authors of these methods for their foundational contributions to parameter-efficient fine-tuning.
