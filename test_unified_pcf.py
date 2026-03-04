"""
Unified PCF Ablation Study

Tests the unified Position-Content Fusion (PCF) architecture.
Compares:
1. Baseline (no adaptation)
2. rsLoRA only
3. HyLoRADA-PCF (unified soft-gated architecture)

Usage:
    python test_unified_pcf.py --num_train 100 --num_test 50 --epochs 1
"""

import argparse
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from dataclasses import dataclass
from typing import List

from hylorada import (
    HyLoRADAModel,
    HyLoRADAConfig,
    evaluate_perplexity,
)


@dataclass
class Result:
    """Results for a single configuration."""
    name: str
    description: str
    ppl: float
    params: int
    time_s: float


def load_wikitext_texts(num_samples=200):
    """Load WikiText-2 as raw text samples."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    texts = [
        text for text in dataset["validation"]["text"]
        if text.strip() and len(text) > 100
    ]
    
    return texts[:num_samples]


def train_model(model, tokenizer, num_train=500, epochs=1, device="cuda"):
    """Train model on WikiText-2."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    
    train_data = dataset["train"].select(range(min(num_train, len(dataset["train"]))))
    train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])
    train_data.set_format("torch")
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()
    
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
        
        print(f"  Epoch {epoch+1}: loss={total_loss/num_batches:.4f}")


def test_baseline(device, dtype, tokenizer, test_texts):
    """Test 1: Baseline (no adaptation)."""
    print("\n" + "="*80)
    print("TEST 1: BASELINE")
    print("="*80)
    print("No LoRA adaptation, just pretrained GPT-2")
    
    start = time.time()
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device=device, dtype=dtype)
    model.eval()
    
    result = evaluate_perplexity(model, tokenizer, test_texts, max_length=512, show_progress=False)
    ppl = result.perplexity
    elapsed = time.time() - start
    
    print(f"✓ PPL: {ppl:.2f} (took {elapsed:.1f}s)")
    
    del model
    torch.cuda.empty_cache()
    
    return Result(
        name="Baseline",
        description="Pretrained GPT-2 (no adaptation)",
        ppl=ppl,
        params=0,
        time_s=elapsed,
    )


def test_rslora_only(device, dtype, tokenizer, test_texts, num_train, epochs):
    """Test 2: rsLoRA only (no PCF)."""
    print("\n" + "="*80)
    print("TEST 2: rsLoRA ONLY")
    print("="*80)
    print("Rank-stabilized LoRA without PCF modulation")
    
    start = time.time()
    
    # Create config with PCF disabled (num_landmarks=0)
    config = HyLoRADAConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_dora_magnitude=False,
        num_landmarks=0,  # Disables PCF
        num_position_buckets=32,
    )
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device=device, dtype=dtype)
    model = HyLoRADAModel(base_model, config)
    model.to(device=device, dtype=dtype)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")
    
    train_model(model, tokenizer, num_train, epochs, device)
    
    model.eval()
    result = evaluate_perplexity(model.base_model, tokenizer, test_texts, max_length=512, show_progress=False)
    ppl = result.perplexity
    elapsed = time.time() - start
    
    print(f"✓ PPL: {ppl:.2f} (took {elapsed:.1f}s)")
    
    del model
    torch.cuda.empty_cache()
    
    return Result(
        name="rsLoRA",
        description="Rank-8 rsLoRA (no PCF)",
        ppl=ppl,
        params=params,
        time_s=elapsed,
    )


def test_hylorada_pcf(device, dtype, tokenizer, test_texts, num_train, epochs, num_landmarks):
    """Test 3: Full HyLoRADA with PCF."""
    print("\n" + "="*80)
    print("TEST 3: HyLoRADA-PCF (UNIFIED)")
    print("="*80)
    print(f"rsLoRA + Position-Content Fusion ({num_landmarks} landmarks)")
    
    start = time.time()
    
    config = HyLoRADAConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_dora_magnitude=False,
        num_landmarks=num_landmarks,
        num_position_buckets=32,
    )
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device=device, dtype=dtype)
    model = HyLoRADAModel(base_model, config)
    model.to(device=device, dtype=dtype)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")
    
    train_model(model, tokenizer, num_train, epochs, device)
    
    model.eval()
    result = evaluate_perplexity(model.base_model, tokenizer, test_texts, max_length=512, show_progress=False)
    ppl = result.perplexity
    elapsed = time.time() - start
    
    print(f"✓ PPL: {ppl:.2f} (took {elapsed:.1f}s)")
    
    del model
    torch.cuda.empty_cache()
    
    return Result(
        name="HyLoRADA-PCF",
        description=f"rsLoRA + PCF ({num_landmarks} landmarks)",
        ppl=ppl,
        params=params,
        time_s=elapsed,
    )


def print_results_table(results: List[Result], baseline_ppl: float):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<20} {'PPL':>8} {'Δ vs Base':>12} {'Params':>12}")
    print("-" * 56)
    
    for r in results:
        delta = ((baseline_ppl - r.ppl) / baseline_ppl) * 100
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        params_str = f"{r.params:,}" if r.params > 0 else "-"
        print(f"{r.name:<20} {r.ppl:>8.2f} {delta_str:>12} {params_str:>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_train", type=int, default=100)
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--num_landmarks", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print("="*80)
    print("UNIFIED PCF ABLATION STUDY")
    print("="*80)
    print(f"Device: {device}")
    print(f"Training samples: {args.num_train}")
    print(f"Test samples: {args.num_test}")
    print(f"Epochs: {args.epochs}")
    print(f"Landmarks: {args.num_landmarks}")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nLoading WikiText-2...")
    test_texts = load_wikitext_texts(args.num_test)
    print(f"Loaded {len(test_texts)} test texts")
    
    results = []
    
    # Test 1: Baseline
    results.append(test_baseline(device, dtype, tokenizer, test_texts))
    baseline_ppl = results[0].ppl
    
    # Test 2: rsLoRA only
    results.append(test_rslora_only(device, dtype, tokenizer, test_texts, args.num_train, args.epochs))
    
    # Test 3: HyLoRADA-PCF
    results.append(test_hylorada_pcf(device, dtype, tokenizer, test_texts, args.num_train, args.epochs, args.num_landmarks))
    
    # Print summary
    print_results_table(results, baseline_ppl)
    
    print("\n" + "="*80)
    print("KEY FINDING")
    print("="*80)
    pcf_ppl = results[2].ppl
    rslora_ppl = results[1].ppl
    
    if pcf_ppl <= rslora_ppl:
        diff = ((rslora_ppl - pcf_ppl) / rslora_ppl) * 100
        print(f"✓ HyLoRADA-PCF achieves {diff:.1f}% lower PPL than rsLoRA alone")
        print("  The unified soft-gated architecture learns when to engage PCF")
    else:
        diff = ((pcf_ppl - rslora_ppl) / rslora_ppl) * 100
        print(f"⚠ PCF overhead: {diff:.1f}% higher PPL than rsLoRA")
        print("  On short contexts, γ should scale down PCF contribution")
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    main()
