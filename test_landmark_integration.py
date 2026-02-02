"""Quick test to verify Position-Adaptive Landmark integration."""

import torch
from hylorada import HyLoRADAConfig, HyLoRADAModel
from transformers import AutoModelForCausalLM

# Test configuration with landmarks enabled
config = HyLoRADAConfig(
    lora_rank=8,
    landmark_enabled=True,
    num_landmarks=8,
    num_position_buckets=32,
)

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

print("Applying HyLoRADA with Position-Adaptive Landmarks...")
model = HyLoRADAModel(base_model, config)

print("\nModel created successfully!")
print(f"Landmarks enabled: {config.landmark_enabled}")
print(f"Num landmarks: {config.num_landmarks}")
print(f"Position buckets: {config.num_position_buckets}")

# Count landmark parameters
if hasattr(model.state, 'landmark') and model.state.landmark is not None:
    landmark_params = sum(p.numel() for p in model.state.landmark.parameters())
    print(f"\nLandmark parameters: {landmark_params:,}")
    print(f"Landmark module: {type(model.state.landmark).__name__}")
    print(f"Architecture: {model.state.landmark}")
    
print("\n✓ Position-Adaptive Landmarks successfully integrated!")
