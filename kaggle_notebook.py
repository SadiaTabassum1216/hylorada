# HyLoRADA Kaggle Notebook
# Run this notebook on Kaggle with GPU accelerator enabled

# =============================================================================
# STEP 1: Clone the repository
# =============================================================================
!git clone https://github.com/SadiaTabassum1216/hylorada.git
%cd hylorada

# =============================================================================
# STEP 2: Install dependencies
# =============================================================================
!pip install -q transformers datasets accelerate tqdm

# =============================================================================
# STEP 3: Verify GPU is available
# =============================================================================
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# =============================================================================
# STEP 4: Quick test - verify HyLoRADA works
# =============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
from hylorada import HyLoRADAConfig, HyLoRADAModel

# Load a small model for testing
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer.pad_token = tokenizer.eos_token

# Apply HyLoRADA
config = HyLoRADAConfig(lora_rank=8, daa_enabled=True, sparse_enabled=True)
hylorada_model = HyLoRADAModel(model, config)
hylorada_model.print_trainable_params()

# =============================================================================
# STEP 5: Run the full experiment
# =============================================================================
!python run_experiment.py \
    --model_name Qwen/Qwen2-0.5B \
    --max_length 1024 \
    --num_epochs 1 \
    --num_test_samples 50 \
    --gradient_accumulation 8

# =============================================================================
# STEP 6: View results
# =============================================================================
import json
import glob

# Find the latest experiment
exp_dirs = sorted(glob.glob("./experiments/exp_*"))
if exp_dirs:
    latest = exp_dirs[-1]
    with open(f"{latest}/results.json") as f:
        results = json.load(f)
    
    print("=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Baseline Perplexity:    {results['baseline']['perplexity']:.2f}")
    print(f"HyLoRADA Perplexity:    {results['hylorada']['perplexity']:.2f}")
    print(f"Improvement:            {results['comparison']['perplexity_improvement_percent']:.2f}%")
    print(f"Middle Position Imp:    {results['comparison']['middle_position_improvement_percent']:.2f}%")
    print("=" * 60)

# =============================================================================
# OPTIONAL: Train with a larger model (requires more VRAM)
# =============================================================================
# !python run_experiment.py \
#     --model_name meta-llama/Llama-2-7b-hf \
#     --token YOUR_HF_TOKEN \
#     --max_length 2048 \
#     --num_epochs 1
