# HyLoRADA: Hybrid Low-Rank Adaptation with Position-Adaptive Landmarks

Context-adaptive PEFT framework for long-context fine-tuning. **+18.37% improvement** on WikiText-2 (short), optimized for 4K-8K token sequences.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

🚀 **Kaggle**: Open [hylorada.ipynb](./hylorada.ipynb), enable GPU, run cells

**Training**: See [METHODOLOGY.md](METHODOLOGY.md) for architecture details

## Running Experiments

### Ablation Study

```bash
# Validate individual components on WikiText-2 (512 tokens)
python test_ablation_proper.py

```

### Long-Context Benchmarks

```bash
# 2K tokens - position bias + landmarks enabled
python run_benchmark.py \
    --dataset longbench --max_length 2048 \
    --methods lora hylorada --train_embeddings \
    --rope_scaling_type linear --rope_scaling_factor 2.0 --epochs 3

# 4K tokens - S²-Attn enabled
python run_benchmark.py \
    --dataset longbench --max_length 4096 \
    --methods lora hylorada --s2_attn --train_embeddings --train_norms \
    --rope_scaling_type linear --rope_scaling_factor 4.0 --epochs 3

# 8K tokens - full configuration
python run_benchmark.py \
    --dataset pg19 --max_length 8192 \
    --methods lora hylorada --s2_attn --train_embeddings --train_norms \
    --rope_scaling_type yarn --rope_scaling_factor 8.0 \
    --batch_size 1 --grad_accum 32 --epochs 1
```

### Testing Components

```bash
# Test landmark architectures
python test_landmarks.py --epochs 1 --num_train 500

# Unit tests
pytest tests/
```

## Key Components

**Context-Adaptive Architecture** (≥2K tokens):
- rsLoRA (rank-stabilized, α/√r scaling)
- Position Bias (64 global params)
- Position-Adaptive Landmarks (8 tokens, ~12.5K params)

**Long-Context Extensions** (≥4K tokens):
- S²-Attn (shifted sparse attention, 16-64x memory reduction)
- RoPE scaling (linear/YaRN)
- Trainable embeddings/norms

## Documentation

- **[METHODOLOGY.md](METHODOLOGY.md)**: Complete architecture and research methodology
- **[PAPER.md](PAPER.md)**: Research paper (conference format)

## Citation

```bibtex
@article{hylorada2026,
  title={HyLoRADA: Context-Adaptive Parameter-Efficient Fine-Tuning},
  author={Your Name},
  year={2026}
}
```
