"""
Test Script for LandmarkLoRA Redesigns

Compares different landmark architectures:
1. Original (single-point at final norm)
2. Per-Layer (applied at each transformer layer)
3. Attention-Integrated (injected as K/V pairs)
4. Position-Adaptive (context-aware selection)

Usage:
    python test_landmarks.py --model gpt2 --dataset wikitext --epochs 2
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.lora import LandmarkLoRA
from hylorada.landmark_redesigns import (
    PerLayerLandmark, 
    AttentionIntegratedLandmark,
    PositionAdaptiveLandmark,
    apply_per_layer_landmarks,
    count_landmark_params,
)
from hylorada.trainer import HyLoRADATrainer, TrainingConfig, create_long_context_dataloader
from hylorada.evaluation import evaluate_perplexity, evaluate_lost_in_the_middle


def test_original_landmark(base_model, tokenizer, train_texts, test_texts, args):
    """Test original single-point landmark design."""
    print("\n" + "="*60)
    print("Testing: Original LandmarkLoRA (single-point at final norm)")
    print("="*60)
    
    config = HyLoRADAConfig(
        lora_rank=args.rank,
        landmark_enabled=True,
        num_landmarks=args.num_landmarks,
        position_bias_enabled=False,  # Isolate landmark effect
        use_dora_magnitude=False,  # Minimal baseline
    )
    
    model = HyLoRADAModel(base_model, config)
    
    # Count parameters
    landmark_params = count_landmark_params(model.state.landmark)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Landmark params: {landmark_params:,}")
    print(f"Total trainable: {total_trainable:,}")
    
    # Train
    train_dataloader = create_long_context_dataloader(
        train_texts, tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )
    
    eval_dataloader = create_long_context_dataloader(
        test_texts[:50], tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )
    
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
    )
    
    trainer = HyLoRADATrainer(model, train_dataloader, train_config, eval_dataloader=eval_dataloader)
    trainer.train()
    
    # Evaluate
    ppl = evaluate_perplexity(model, tokenizer, test_texts, max_length=args.max_length)
    lim = evaluate_lost_in_the_middle(model, tokenizer, test_texts[:20], max_length=args.max_length)
    
    return {
        "perplexity": ppl.perplexity,
        "lim_perplexity": lim.perplexity,
        "landmark_params": landmark_params,
        "total_params": total_trainable,
    }


def test_per_layer_landmark(base_model, tokenizer, train_texts, test_texts, args):
    """Test per-layer landmark design."""
    print("\n" + "="*60)
    print("Testing: Per-Layer Landmarks (applied at each transformer layer)")
    print("="*60)
    
    # Create base HyLoRADA without original landmark
    config = HyLoRADAConfig(
        lora_rank=args.rank,
        landmark_enabled=False,
        position_bias_enabled=False,
        use_dora_magnitude=False,
    )
    
    model = HyLoRADAModel(base_model, config)
    
    # Apply per-layer landmarks manually
    landmarks = {}
    num_layers = 0
    
    for name, module in model.base_model.named_modules():
        # Find FFN output layers (varies by architecture)
        if any(pattern in name.lower() for pattern in ["mlp", "ffn", "feed_forward"]):
            # Only apply to the final output of FFN
            if "c_proj" in name or "dense" in name or "down_proj" in name:
                landmark = PerLayerLandmark(
                    hidden_size=model.hidden_size,
                    num_landmarks=args.num_landmarks,
                    dropout=0.0,
                )
                
                # Register as parameter
                param_name = name.replace(".", "_") + "_landmark"
                model.add_module(param_name, landmark)
                landmarks[name] = landmark
                num_layers += 1
                
                # Hook to apply landmark
                def make_hook(lm):
                    def hook(m, input, output):
                        return lm(output)
                    return hook
                
                module.register_forward_hook(make_hook(landmark))
    
    print(f"Applied landmarks to {num_layers} layers")
    
    # Count parameters
    landmark_params = sum(count_landmark_params(lm) for lm in landmarks.values())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Landmark params: {landmark_params:,} ({num_layers} layers × ~{landmark_params//num_layers:,})")
    print(f"Total trainable: {total_trainable:,}")
    
    # Train
    train_dataloader = create_long_context_dataloader(
        train_texts, tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )
    
    eval_dataloader = create_long_context_dataloader(
        test_texts[:50], tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )
    
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
    )
    
    trainer = HyLoRADATrainer(model, train_dataloader, train_config, eval_dataloader=eval_dataloader)
    trainer.train()
    
    # Evaluate
    ppl = evaluate_perplexity(model, tokenizer, test_texts, max_length=args.max_length)
    lim = evaluate_lost_in_the_middle(model, tokenizer, test_texts[:20], max_length=args.max_length)
    
    return {
        "perplexity": ppl.perplexity,
        "lim_perplexity": lim.perplexity,
        "landmark_params": landmark_params,
        "total_params": total_trainable,
        "num_layers": num_layers,
    }


def test_position_adaptive_landmark(base_model, tokenizer, train_texts, test_texts, args):
    """Test position-adaptive landmark design."""
    print("\n" + "="*60)
    print("Testing: Position-Adaptive Landmarks")
    print("="*60)
    
    config = HyLoRADAConfig(
        lora_rank=args.rank,
        landmark_enabled=False,
        position_bias_enabled=False,
        use_dora_magnitude=False,
    )
    
    model = HyLoRADAModel(base_model, config)
    
    # Apply position-adaptive landmark at final norm
    landmark = PositionAdaptiveLandmark(
        hidden_size=model.hidden_size,
        num_landmarks=args.num_landmarks,
        max_positions=args.max_length,
        num_buckets=32,
    )
    
    model.add_module("position_adaptive_landmark", landmark)
    
    # Register hook
    for name, module in model.base_model.named_modules():
        if any(n in name.lower() for n in ["norm", "ln_f", "final_layer_norm"]):
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or "norm" in type(module).__name__.lower():
                def landmark_hook(m, input, output):
                    return model.position_adaptive_landmark(output)
                
                module.register_forward_hook(landmark_hook)
                print(f"Registered hook on: {name}")
                break
    
    # Count parameters
    landmark_params = count_landmark_params(landmark)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Landmark params: {landmark_params:,}")
    print(f"Total trainable: {total_trainable:,}")
    
    # Train
    train_dataloader = create_long_context_dataloader(
        train_texts, tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )
    
    eval_dataloader = create_long_context_dataloader(
        test_texts[:50], tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )
    
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
    )
    
    trainer = HyLoRADATrainer(model, train_dataloader, train_config, eval_dataloader=eval_dataloader)
    trainer.train()
    
    # Evaluate
    ppl = evaluate_perplexity(model, tokenizer, test_texts, max_length=args.max_length)
    lim = evaluate_lost_in_the_middle(model, tokenizer, test_texts[:20], max_length=args.max_length)
    
    return {
        "perplexity": ppl.perplexity,
        "lim_perplexity": lim.perplexity,
        "landmark_params": landmark_params,
        "total_params": total_trainable,
    }


def test_baseline_no_landmark(base_model, tokenizer, train_texts, test_texts, args):
    """Test baseline without any landmarks."""
    print("\n" + "="*60)
    print("Testing: Baseline (No Landmarks)")
    print("="*60)
    
    config = HyLoRADAConfig(
        lora_rank=args.rank,
        landmark_enabled=False,
        position_bias_enabled=False,
        use_dora_magnitude=False,
    )
    
    model = HyLoRADAModel(base_model, config)
    
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable: {total_trainable:,}")
    
    # Train
    train_dataloader = create_long_context_dataloader(
        train_texts, tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )
    
    eval_dataloader = create_long_context_dataloader(
        test_texts[:50], tokenizer, max_length=args.max_length, batch_size=args.batch_size
    )
    
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
    )
    
    trainer = HyLoRADATrainer(model, train_dataloader, train_config, eval_dataloader=eval_dataloader)
    trainer.train()
    
    # Evaluate
    ppl = evaluate_perplexity(model, tokenizer, test_texts, max_length=args.max_length)
    lim = evaluate_lost_in_the_middle(model, tokenizer, test_texts[:20], max_length=args.max_length)
    
    return {
        "perplexity": ppl.perplexity,
        "lim_perplexity": lim.perplexity,
        "landmark_params": 0,
        "total_params": total_trainable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai-community/gpt2")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num_landmarks", type=int, default=8)
    parser.add_argument("--designs", nargs="+", 
                        default=["baseline", "original", "per_layer", "position_adaptive"])
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "wikitext":
        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
            train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50][:args.num_train]
            test_texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 50][:args.num_test]
        except:
            print("Using TinyStories as fallback")
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            texts = [t["text"] for t in dataset if len(t.get("text", "").strip()) > 50]
            train_texts = texts[:args.num_train]
            test_texts = texts[args.num_train:args.num_train + args.num_test]
    
    print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Results storage
    results = {}
    
    # Test each design
    for design in args.designs:
        # Load fresh model for each test
        print(f"\n{'='*60}")
        print(f"Loading fresh model: {args.model}")
        print(f"{'='*60}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        if design == "baseline":
            results["baseline"] = test_baseline_no_landmark(
                base_model, tokenizer, train_texts, test_texts, args
            )
        elif design == "original":
            results["original"] = test_original_landmark(
                base_model, tokenizer, train_texts, test_texts, args
            )
        elif design == "per_layer":
            results["per_layer"] = test_per_layer_landmark(
                base_model, tokenizer, train_texts, test_texts, args
            )
        elif design == "position_adaptive":
            results["position_adaptive"] = test_position_adaptive_landmark(
                base_model, tokenizer, train_texts, test_texts, args
            )
        
        # Clear GPU memory
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Design':<20} {'PPL':<10} {'LIM-PPL':<10} {'Landmark Params':<15} {'Total Params':<15}")
    print("-"*60)
    
    for design, result in results.items():
        print(f"{design:<20} {result['perplexity']:<10.2f} {result['lim_perplexity']:<10.2f} "
              f"{result['landmark_params']:<15,} {result['total_params']:<15,}")
    
    # Analysis
    if "baseline" in results:
        baseline_ppl = results["baseline"]["perplexity"]
        baseline_lim = results["baseline"]["lim_perplexity"]
        
        print("\n" + "="*60)
        print("IMPROVEMENT OVER BASELINE")
        print("="*60)
        
        for design, result in results.items():
            if design == "baseline":
                continue
            
            ppl_improvement = ((baseline_ppl - result["perplexity"]) / baseline_ppl) * 100
            lim_improvement = ((baseline_lim - result["lim_perplexity"]) / baseline_lim) * 100
            params_overhead = result["landmark_params"]
            
            print(f"\n{design}:")
            print(f"  PPL improvement:        {ppl_improvement:+.2f}%")
            print(f"  LIM-PPL improvement:    {lim_improvement:+.2f}%")
            print(f"  Parameter overhead:     {params_overhead:,}")
            print(f"  Params per 1% PPL gain: {params_overhead / max(0.01, abs(ppl_improvement)):.0f}")


if __name__ == "__main__":
    main()
