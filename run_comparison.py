"""
Multi-Method Comparison Experiment

Compares HyLoRADA against multiple baseline PEFT methods:
1. Baseline (no adaptation)
2. Standard LoRA
3. LoRaDA (LoRA + DAA)
4. LongLoRA (LoRA + trainable embeddings/norms)
5. SparseAdapter (sparse FFN adapters only)
6. HyLoRADA (full hybrid approach)

Usage:
    python run_comparison.py --model_name Qwen/Qwen2-0.5B --max_length 1024
"""

import argparse
import json
import os
import time
from datetime import datetime
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.baselines import (
    StandardLoRA, 
    LoRaDAModel, 
    LongLoRAModel, 
    SparseAdapterModel,
    BaselineConfig,
    get_baseline_model,
)
from hylorada.trainer import HyLoRADATrainer, TrainingConfig
from hylorada.evaluation import evaluate_perplexity, evaluate_lost_in_middle


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PEFT methods")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=["baseline", "lora", "lorada", "longlora", "sparse", "hylorada"],
                        help="Methods to compare")
    parser.add_argument("--lora_rank", type=int, default=8)
    return parser.parse_args()


def load_model_fresh(model_name: str, device: str):
    """Load a fresh copy of the model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    return model.to(device)


def prepare_data(tokenizer, max_length: int, num_samples: int = 1000):
    """Prepare WikiText-2 dataset for training and evaluation."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    
    # Filter empty texts and tokenize
    train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50][:num_samples]
    test_texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 50][:num_samples // 10]
    
    return train_texts, test_texts


def train_model(model, tokenizer, train_texts, args, method_name: str):
    """Train a model using the HyLoRADA trainer."""
    train_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        warmup_ratio=0.03,
        logging_steps=50,
    )
    
    trainer = HyLoRADATrainer(
        model=model,
        tokenizer=tokenizer,
        config=train_config,
    )
    
    print(f"\n  Training {method_name}...")
    start_time = time.time()
    trainer.train(train_texts)
    train_time = time.time() - start_time
    
    return model, train_time


def evaluate_model(model, tokenizer, test_texts, max_length: int, device: str):
    """Evaluate a model on perplexity and lost-in-middle metrics."""
    results = {}
    
    # Perplexity
    print("    Evaluating perplexity...")
    ppl = evaluate_perplexity(model, tokenizer, test_texts, max_length, device)
    results["perplexity"] = ppl
    
    # Lost-in-Middle (if sequence is long enough)
    if max_length >= 512:
        print("    Evaluating lost-in-middle...")
        lim_results = evaluate_lost_in_middle(model, tokenizer, test_texts[:20], max_length, device)
        results["lost_in_middle"] = lim_results
    
    return results


def run_comparison(args):
    """Run the full comparison experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("PEFT Method Comparison Experiment")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Max Length: {args.max_length}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    print("\n[2] Preparing data...")
    train_texts, test_texts = prepare_data(tokenizer, args.max_length)
    print(f"    Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
    
    # Results storage
    all_results = {}
    
    # Baseline config for non-HyLoRADA methods
    baseline_config = BaselineConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        sparse_adapter_dim=128,
        sparse_topk_ratio=0.05,
    )
    
    # ==================== Run Each Method ====================
    
    for i, method in enumerate(args.methods):
        print(f"\n[{i+3}] Running {method.upper()}...")
        print("-" * 50)
        
        # Load fresh model
        print(f"  Loading fresh model...")
        base_model = load_model_fresh(args.model_name, device)
        
        try:
            if method == "baseline":
                # No adaptation, just evaluate
                model = base_model
                train_time = 0
                trainable_params = 0
                
            elif method == "lora":
                model = StandardLoRA(base_model, baseline_config)
                model.print_trainable_params()
                trainable_params = model.count_params()["trainable_params"]
                model, train_time = train_model(model, tokenizer, train_texts, args, "LoRA")
                
            elif method == "lorada":
                model = LoRaDAModel(base_model, baseline_config)
                model.print_trainable_params()
                trainable_params = model.count_params()["trainable_params"]
                model, train_time = train_model(model, tokenizer, train_texts, args, "LoRaDA")
                
            elif method == "longlora":
                model = LongLoRAModel(base_model, baseline_config)
                model.print_trainable_params()
                trainable_params = model.count_params()["trainable_params"]
                model, train_time = train_model(model, tokenizer, train_texts, args, "LongLoRA")
                
            elif method == "sparse":
                model = SparseAdapterModel(base_model, baseline_config)
                model.print_trainable_params()
                trainable_params = model.count_params()["trainable_params"]
                model, train_time = train_model(model, tokenizer, train_texts, args, "SparseAdapter")
                
            elif method == "hylorada":
                config = HyLoRADAConfig(
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_rank * 2,
                    daa_enabled=True,
                    daa_use_positional=True,
                    sparse_enabled=True,
                    sparse_adapter_dim=128,
                    sparse_topk_ratio=0.05,
                )
                model = HyLoRADAModel(base_model, config)
                model.print_trainable_params()
                trainable_params = model.count_params()["trainable_params"]
                model, train_time = train_model(model, tokenizer, train_texts, args, "HyLoRADA")
            
            else:
                print(f"  Unknown method: {method}, skipping...")
                continue
            
            # Evaluate
            print(f"  Evaluating {method}...")
            eval_results = evaluate_model(model, tokenizer, test_texts, args.max_length, device)
            
            all_results[method] = {
                "trainable_params": trainable_params,
                "train_time_seconds": train_time,
                "perplexity": eval_results.get("perplexity", None),
                "lost_in_middle": eval_results.get("lost_in_middle", None),
            }
            
            print(f"  ✓ {method}: PPL={eval_results.get('perplexity', 'N/A'):.2f}, "
                  f"Params={trainable_params:,}, Time={train_time:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Error with {method}: {e}")
            all_results[method] = {"error": str(e)}
        
        # Clean up GPU memory
        del base_model
        if method != "baseline":
            del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ==================== Summary ====================
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<15} {'Params':>12} {'Train Time':>12} {'Perplexity':>12}")
    print("-" * 70)
    
    for method, results in all_results.items():
        if "error" in results:
            print(f"{method:<15} {'ERROR':>12} {'-':>12} {'-':>12}")
        else:
            params = f"{results['trainable_params']:,}" if results['trainable_params'] else "0"
            time_str = f"{results['train_time_seconds']:.1f}s" if results['train_time_seconds'] else "-"
            ppl = f"{results['perplexity']:.2f}" if results['perplexity'] else "-"
            print(f"{method:<15} {params:>12} {time_str:>12} {ppl:>12}")
    
    print("=" * 70)
    
    # Save results
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
