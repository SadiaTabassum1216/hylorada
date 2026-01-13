"""
HyLoRADA Full Experiment Script

Runs a complete experiment:
1. Load a base model
2. Evaluate baseline on long-context
3. Train with HyLoRADA on long documents
4. Evaluate HyLoRADA model
5. Compare results

Usage:
    python run_experiment.py --model_name Qwen/Qwen2-0.5B --max_length 2048
"""

import argparse
import torch
import json
import os
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.trainer import HyLoRADATrainer, TrainingConfig, create_long_context_dataloader
from hylorada.evaluation import (
    evaluate_perplexity,
    evaluate_lost_in_the_middle,
    compare_models,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run HyLoRADA experiment")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    
    # Data
    parser.add_argument("--train_dataset", type=str, default="wikitext")
    parser.add_argument("--train_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--test_dataset", type=str, default="wikitext")
    parser.add_argument("--test_config", type=str, default="wikitext-2-raw-v1")
    
    # Training
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    
    # HyLoRADA config
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--daa_enabled", action="store_true", default=True)
    parser.add_argument("--sparse_enabled", action="store_true", default=True)
    
    # Evaluation
    parser.add_argument("--num_test_samples", type=int, default=50)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./experiments")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print("=" * 70)
    print("HyLoRADA Experiment")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Max Length: {args.max_length}")
    print(f"Output: {exp_dir}")
    print("=" * 70)
    
    # ===== STEP 1: Load tokenizer =====
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== STEP 2: Load and evaluate baseline =====
    print("\n[2/5] Loading baseline model and evaluating...")
    
    has_cuda = torch.cuda.is_available()
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
        token=args.token,
    )
    
    # Load test data
    test_dataset = load_dataset(args.test_dataset, args.test_config, split="test")
    test_texts = [
        item["text"] for item in test_dataset 
        if len(item.get("text", "")) > 200
    ][:args.num_test_samples]
    
    print(f"Evaluating on {len(test_texts)} test samples...")
    
    baseline_result = evaluate_perplexity(
        baseline_model, tokenizer, test_texts, args.max_length
    )
    print(f"Baseline Perplexity: {baseline_result.perplexity:.2f}")
    
    baseline_litm = evaluate_lost_in_the_middle(
        baseline_model, tokenizer, test_texts, max_length=args.max_length
    )
    
    # ===== STEP 3: Apply HyLoRADA and train =====
    print("\n[3/5] Applying HyLoRADA and training...")
    
    hylorada_config = HyLoRADAConfig(
        lora_rank=args.lora_rank,
        daa_enabled=args.daa_enabled,
        sparse_enabled=args.sparse_enabled,
        max_sequence_length=args.max_length,
    )
    
    # Create a fresh model for HyLoRADA (don't reuse baseline to keep it clean)
    hylorada_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
        token=args.token,
    )
    
    hylorada_model = HyLoRADAModel(hylorada_base, hylorada_config)
    hylorada_model.print_trainable_params()
    
    # Load training data
    train_dataset = load_dataset(args.train_dataset, args.train_config, split="train")
    train_dataloader = create_long_context_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    
    # Training config (gradient checkpointing disabled to avoid issues with adapters)
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=os.path.join(exp_dir, "checkpoints"),
        mixed_precision="bf16" if has_cuda else "fp32",
        gradient_checkpointing=False,  # Disabled - can cause issues with adapters
    )
    
    # Train
    trainer = HyLoRADATrainer(hylorada_model, train_dataloader, config=training_config)
    train_results = trainer.train()
    
    # ===== STEP 4: Evaluate HyLoRADA =====
    print("\n[4/5] Evaluating HyLoRADA model...")
    
    hylorada_model.eval()
    hylorada_result = evaluate_perplexity(
        hylorada_model, tokenizer, test_texts, args.max_length
    )
    print(f"HyLoRADA Perplexity: {hylorada_result.perplexity:.2f}")
    
    hylorada_litm = evaluate_lost_in_the_middle(
        hylorada_model, tokenizer, test_texts, max_length=args.max_length
    )
    
    # ===== STEP 5: Compare and save results =====
    print("\n[5/5] Generating comparison report...")
    
    ppl_improvement = (baseline_result.perplexity - hylorada_result.perplexity) / baseline_result.perplexity * 100
    
    # Middle position improvement
    n = len(baseline_litm.position_perplexities)
    mid_start, mid_end = n // 3, 2 * n // 3
    baseline_mid = sum(baseline_litm.position_perplexities[mid_start:mid_end]) / max(mid_end - mid_start, 1)
    hylorada_mid = sum(hylorada_litm.position_perplexities[mid_start:mid_end]) / max(mid_end - mid_start, 1)
    
    if baseline_mid > 0:
        middle_improvement = (baseline_mid - hylorada_mid) / baseline_mid * 100
    else:
        middle_improvement = 0.0
    
    results = {
        "experiment_config": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "num_epochs": args.num_epochs,
            "lora_rank": args.lora_rank,
            "daa_enabled": args.daa_enabled,
            "sparse_enabled": args.sparse_enabled,
            "num_test_samples": len(test_texts),
        },
        "baseline": {
            "perplexity": baseline_result.perplexity,
            "loss": baseline_result.loss,
            "position_perplexities": baseline_litm.position_perplexities,
        },
        "hylorada": {
            "perplexity": hylorada_result.perplexity,
            "loss": hylorada_result.loss,
            "position_perplexities": hylorada_litm.position_perplexities,
            "training_loss": train_results["final_loss"],
        },
        "comparison": {
            "perplexity_improvement_percent": ppl_improvement,
            "middle_position_improvement_percent": middle_improvement,
        }
    }
    
    # Save results
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Context Length: {args.max_length}")
    print("-" * 70)
    print(f"Baseline Perplexity:      {baseline_result.perplexity:.2f}")
    print(f"HyLoRADA Perplexity:      {hylorada_result.perplexity:.2f}")
    print(f"Perplexity Improvement:   {ppl_improvement:+.2f}%")
    print("-" * 70)
    print(f"Middle Position Improvement: {middle_improvement:+.2f}%")
    print("(Positive = HyLoRADA better at remembering middle content)")
    print("-" * 70)
    print(f"Results saved to: {results_path}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
