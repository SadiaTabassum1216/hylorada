# HyLoRADA

**Unified HyLoRADA** - Parameter-efficient fine-tuning for cost-efficient long-context LLMs.

## Key Features

| Feature | Solution | Literature |
|---------|----------|------------|
| **Rank Collapse Prevention** | Orthogonal init | LongLoRA |
| **Magnitude Normalization** | DoRA-style | DoRA |
| **Lost-in-Middle** | PositionBias | LIFT |
| **Noise Filtering** | PositionalDAA | - |
| **Training Efficiency** | S²-Attn (16x faster) | LongLoRA |
| **Context Extension** | Embeddings & Norms | LongLoRA |
| **Stable Attention** | Sink Tokens | SinkLoRA |
| **Position Extension** | RoPE Scaling | YaRN |

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
    s2_attn_enabled=True,     # 16x training efficiency
    s2_sink_tokens=4,         # SinkLoRA
    rope_scaling_type="linear",
    rope_scaling_factor=4.0,
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
├── lora.py           # HyLoRADA adapters with orthogonal init
├── daa.py            # PositionalDAA
├── s2_attention.py   # Shifted Sparse Attention + Sink Tokens
├── model.py          # HyLoRADAModel wrapper
├── baselines.py      # Comparison methods
└── evaluation.py     # Metrics
```
