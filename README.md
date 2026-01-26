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
| **Long Context** | Trainable Embeddings & Norms | ~10% |
| **Stable Attention** | Sink Tokens | 0 |
| **Context Extension** | RoPE Scaling (YaRN) | 0 |

## Quick Start

```python
from transformers import AutoModelForCausalLM
from hylorada import HyLoRADAConfig, HyLoRADAModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Standard Config
config = HyLoRADAConfig(lora_rank=8)

# Long-Context Config (>32k tokens)
long_context_config = HyLoRADAConfig(
    lora_rank=8,
    train_embeddings=True,    # LongLoRA
    train_norms=True,         # LongLoRA
    s2_sink_tokens=4,         # SinkLoRA
    rope_scaling_type="linear", # RoPE Scaling
    rope_scaling_factor=2.0
)

wrapped = HyLoRADAModel(model, long_context_config)
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
├── s2_attention.py   # Shifted Sparse Attention + Sink Tokens
├── model.py          # HyLoRADAModel wrapper
├── baselines.py      # Comparison methods
└── evaluation.py     # Metrics
```

## Version

- **v0.4.0** - Long-context support (LongLoRA, SinkLoRA, YaRN)
- **v0.3.0** - Unified architecture
