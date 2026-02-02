"""
Test script to compare fixed bucketing vs learnable bucketing.

This script trains models with:
1. Baseline HyLoRADA (no landmarks)
2. Position-Adaptive Landmarks (fixed bucketing)
3. Learnable-Bucket Landmarks (learned bucketing)

Expected: Learnable bucketing should improve 1-3% over fixed bucketing.
"""

import argparse
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import time
from typing import Dict, List, Tuple

from hylorada import (
    HyLoRADAModel,
    HyLoRADAConfig,
)
from hylorada.landmark_redesigns import (
    PositionAdaptiveLandmark,
    LearnableBucketLandmark,
    count_landmark_params,
)


def evaluate_perplexity_simple(model, dataloader, device):
    """Simple perplexity evaluation on a dataloader."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def get_wikitext_dataloaders(tokenizer, max_length=512, num_train=1000, num_val=200):
    """Create WikiText-2 dataloaders."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    
    # Process datasets
    train_dataset = dataset["train"].select(range(min(num_train, len(dataset["train"]))))
    val_dataset = dataset["validation"].select(range(min(num_val, len(dataset["validation"]))))
    
    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False
    )
    
    return train_loader, val_loader


def create_baseline_model(device, dtype=torch.bfloat16):
    """Create baseline HyLoRADA model without landmarks."""
    config = HyLoRADAConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        landmark_enabled=False,  # No landmarks
    )
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = HyLoRADAModel(base_model, config)
    model.to(device=device, dtype=dtype)
    
    return model, config


def create_position_adaptive_model(device, num_landmarks=8, dtype=torch.bfloat16):
    """Create model with fixed position bucketing."""
    config = HyLoRADAConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        landmark_enabled=True,
        num_landmarks=num_landmarks,
        num_position_buckets=32,
    )
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = HyLoRADAModel(base_model, config)
    model.to(device=device, dtype=dtype)
    
    return model, config


def create_learnable_bucket_model(device, num_landmarks=8, dtype=torch.bfloat16):
    """Create model with learnable position bucketing."""
    # Start with baseline config (no built-in landmarks)
    config = HyLoRADAConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        landmark_enabled=False,  # We'll add manually
    )
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = HyLoRADAModel(base_model, config)
    
    # Manually add learnable bucket landmark
    hidden_size = base_model.config.hidden_size
    landmark = LearnableBucketLandmark(
        hidden_size=hidden_size,
        num_landmarks=num_landmarks,
        max_positions=base_model.config.n_positions,
        num_buckets=32,
        dropout=0.05,
    )
    
    # Register as hook on final layer norm (same as built-in landmarks)
    def landmark_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        return landmark(hidden_states)
    
    # Find final layer norm and register hook
    for name, module in model.base_model.named_modules():
        if name == "transformer.ln_f":
            module.register_forward_hook(landmark_hook)
            print(f"✓ Registered learnable bucket landmark at {name}")
            break
    
    model.to(device=device, dtype=dtype)
    
    return model, config, landmark


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=3,
    device="cuda",
    lr=5e-4,
    landmark=None,
):
    """Train a model and return final validation perplexity."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # Evaluate
    model.eval()
    val_ppl = evaluate_perplexity_simple(model.base_model, val_loader, device)
    
    # Print learned boundaries if applicable
    if landmark is not None and hasattr(landmark, 'get_learned_boundaries'):
        boundaries = landmark.get_learned_boundaries()
        print(f"  Learned boundaries: {boundaries.cpu().tolist()[:8]}...")
    
    return val_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--num_train", type=int, default=1000, help="Training samples")
    parser.add_argument("--num_val", type=int, default=200, help="Validation samples")
    parser.add_argument("--num_landmarks", type=int, default=8, help="Number of landmarks")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print("="*80)
    print("LEARNABLE BUCKETING COMPARISON")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Training samples: {args.num_train}")
    print(f"Validation samples: {args.num_val}")
    print(f"Epochs: {args.epochs}")
    print(f"Landmarks: {args.num_landmarks}")
    
    # Load data
    print("\nLoading WikiText-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_dataloaders(
        tokenizer, num_train=args.num_train, num_val=args.num_val
    )
    
    results = {}
    
    # Test 1: Baseline (no landmarks)
    print("\n" + "="*80)
    print("TEST 1: BASELINE (No Landmarks)")
    print("="*80)
    model, config = create_baseline_model(device, dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    
    start_time = time.time()
    ppl = train_model(model, train_loader, val_loader, args.epochs, device)
    elapsed = time.time() - start_time
    
    results["baseline"] = {
        "ppl": ppl,
        "params": trainable,
        "time": elapsed,
    }
    print(f"✓ Baseline PPL: {ppl:.2f}")
    
    del model
    torch.cuda.empty_cache()
    
    # Test 2: Position-Adaptive (fixed bucketing)
    print("\n" + "="*80)
    print("TEST 2: POSITION-ADAPTIVE (Fixed Bucketing)")
    print("="*80)
    model, config = create_position_adaptive_model(device, args.num_landmarks, dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    
    # Count landmark params (calculate expected since built-in landmarks aren't in separate module)
    if config.landmark_enabled:
        hidden_size = model.base_model.config.hidden_size
        landmark_params = (
            args.num_landmarks * hidden_size +  # landmarks
            config.num_position_buckets * args.num_landmarks +  # position_gates
            hidden_size * args.num_landmarks  # content_gate
        )
    else:
        landmark_params = 0
    print(f"Landmark parameters: {landmark_params:,}")
    
    start_time = time.time()
    ppl = train_model(model, train_loader, val_loader, args.epochs, device)
    elapsed = time.time() - start_time
    
    results["position_adaptive"] = {
        "ppl": ppl,
        "params": trainable,
        "landmark_params": landmark_params,
        "time": elapsed,
    }
    print(f"✓ Position-Adaptive PPL: {ppl:.2f}")
    
    del model
    torch.cuda.empty_cache()
    
    # Test 3: Learnable Bucketing
    print("\n" + "="*80)
    print("TEST 3: LEARNABLE BUCKETING (Learned Boundaries)")
    print("="*80)
    model, config, landmark = create_learnable_bucket_model(device, args.num_landmarks, dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    landmark_params = count_landmark_params(landmark)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Landmark parameters: {landmark_params:,}")
    print(f"  - Landmarks: {args.num_landmarks} × 768 = {args.num_landmarks * 768:,}")
    print(f"  - Bucket boundaries: {32-1} = {32-1:,}")
    print(f"  - Position gates: {32} × {args.num_landmarks} = {32 * args.num_landmarks:,}")
    print(f"  - Content gate: 768 × {args.num_landmarks} = {768 * args.num_landmarks:,}")
    
    start_time = time.time()
    ppl = train_model(model, train_loader, val_loader, args.epochs, device, landmark=landmark)
    elapsed = time.time() - start_time
    
    results["learnable_bucketing"] = {
        "ppl": ppl,
        "params": trainable,
        "landmark_params": landmark_params,
        "time": elapsed,
    }
    print(f"✓ Learnable Bucketing PPL: {ppl:.2f}")
    
    # Print comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    baseline_ppl = results["baseline"]["ppl"]
    
    print(f"\n{'Design':<25} {'PPL':>10} {'Δ PPL':>10} {'Params':>12} {'Time':>8}")
    print("-" * 80)
    
    for name, data in results.items():
        ppl = data["ppl"]
        params = data.get("landmark_params", 0)
        delta = ((baseline_ppl - ppl) / baseline_ppl) * 100
        time_str = f"{data['time']:.1f}s"
        
        display_name = name.replace("_", " ").title()
        print(f"{display_name:<25} {ppl:>10.2f} {delta:>9.1f}% {params:>12,} {time_str:>8}")
    
    # Calculate improvement of learnable over fixed
    pos_adaptive_ppl = results["position_adaptive"]["ppl"]
    learnable_ppl = results["learnable_bucketing"]["ppl"]
    additional_gain = ((pos_adaptive_ppl - learnable_ppl) / pos_adaptive_ppl) * 100
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"Position-Adaptive improvement: {((baseline_ppl - pos_adaptive_ppl) / baseline_ppl) * 100:.2f}%")
    print(f"Learnable Bucketing improvement: {((baseline_ppl - learnable_ppl) / baseline_ppl) * 100:.2f}%")
    print(f"Additional gain from learnable boundaries: {additional_gain:.2f}%")
    print(f"Additional parameters: {results['learnable_bucketing']['landmark_params'] - results['position_adaptive']['landmark_params']:,}")
    
    if additional_gain > 0:
        print(f"\n✓ SUCCESS: Learnable bucketing provides {additional_gain:.2f}% additional improvement!")
    else:
        print(f"\n⚠ Learnable bucketing did not improve over fixed bucketing.")
        print(f"  This may improve with more training or larger models.")


if __name__ == "__main__":
    main()
