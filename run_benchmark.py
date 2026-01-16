"""
PEFT Methods Benchmark Script

Compare HyLoRADA against other PEFT methods with identical hyperparameters.
Methods: Baseline, LoRA, LoRaDA, LongLoRA, SparseAdapter, HyLoRADA

Usage:
    python run_benchmark.py --model Qwen/Qwen2-0.5B --epochs 3
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.baselines import (
    StandardLoRA, LoRaDAModel, LongLoRAModel, SparseAdapterModel, BaselineConfig
)
from hylorada.trainer import HyLoRADATrainer, TrainingConfig, create_long_context_dataloader
from hylorada.evaluation import evaluate_perplexity, evaluate_lost_in_the_middle


def load_fresh_model(model_name, device, dtype):
    """Load a fresh model instance."""
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)


def train_model(model, tokenizer, train_texts, args, method_name):
    """Train a model and return training time."""
    train_dataloader = create_long_context_dataloader(
        dataset=train_texts, tokenizer=tokenizer,
        max_length=args.max_length, batch_size=args.batch_size,
    )
    
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        logging_steps=50,
    )
    
    trainer = HyLoRADATrainer(model=model, train_dataloader=train_dataloader, config=train_config)
    
    print(f"  Training {method_name}...")
    start = time.time()
    trainer.train()
    return time.time() - start


def evaluate_model(model, tokenizer, test_texts, max_length):
    """Evaluate model and return results."""
    print("    Evaluating perplexity...")
    ppl = evaluate_perplexity(model, tokenizer, test_texts, max_length=max_length)
    
    print("    Evaluating lost-in-middle...")
    lim = evaluate_lost_in_the_middle(model, tokenizer, test_texts[:20], max_length=max_length)
    
    return {
        "perplexity": ppl.perplexity,
        "loss": ppl.loss,
        "lim_perplexity": lim.perplexity,
        "lim_positions": lim.position_perplexities,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PEFT Methods")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--methods", nargs="+", 
                        default=["baseline", "lora", "lorada", "longlora", "sparse", "hylorada"])
    parser.add_argument("--sparse_dim", type=int, default=128,
                        help="Sparse adapter bottleneck dimension (default: 128, lite: 32)")
    parser.add_argument("--sparse_layers", type=str, default=None,
                        help="Comma-separated layer indices for sparse (e.g., '0,5,10,15,20,23')")
    args = parser.parse_args()
    
    # Parse sparse_layers if provided
    if args.sparse_layers:
        args.sparse_layers = [int(x.strip()) for x in args.sparse_layers.split(",")]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print("=" * 70)
    print("PEFT Methods Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Hyperparameters: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"Max Length: {args.max_length}, LoRA Rank: {args.lora_rank}")
    print("=" * 70)
    
    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("[2] Loading data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50][:args.num_train]
    test_texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 50][:args.num_test]
    print(f"    Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    # Baseline config for comparison methods
    baseline_config = BaselineConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        sparse_adapter_dim=args.sparse_dim,
        sparse_topk_ratio=0.05,
        sparse_target_layers=args.sparse_layers,
    )
    
    results = {}
    
    for i, method in enumerate(args.methods):
        print(f"\n[{i+3}] Running {method.upper()}...")
        print("-" * 50)
        
        try:
            # Load fresh model
            print("  Loading fresh model...")
            base_model = load_fresh_model(args.model, device, dtype)
            
            if method == "baseline":
                model = base_model
                train_time = 0
                params = 0
                
            elif method == "lora":
                model = StandardLoRA(base_model, baseline_config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time = train_model(model, tokenizer, train_texts, args, "LoRA")
                
            elif method == "lorada":
                model = LoRaDAModel(base_model, baseline_config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time = train_model(model, tokenizer, train_texts, args, "LoRaDA")
                
            elif method == "longlora":
                model = LongLoRAModel(base_model, baseline_config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time = train_model(model, tokenizer, train_texts, args, "LongLoRA")
                
            elif method == "sparse":
                model = SparseAdapterModel(base_model, baseline_config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time = train_model(model, tokenizer, train_texts, args, "SparseAdapter")
                
            elif method == "hylorada":
                config = HyLoRADAConfig(
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_rank * 2,
                    daa_enabled=True,
                    daa_use_positional=True,
                    sparse_enabled=True,
                    sparse_adapter_dim=args.sparse_dim,
                    sparse_topk_ratio=0.05,
                    sparse_target_layers=args.sparse_layers,
                )
                model = HyLoRADAModel(base_model, config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time = train_model(model, tokenizer, train_texts, args, "HyLoRADA")
            
            elif method == "hylorada-lite":
                from hylorada.config import HyLoRADAPresets
                config = HyLoRADAPresets.lightweight()
                config.lora_rank = args.lora_rank
                config.lora_alpha = args.lora_rank * 2
                model = HyLoRADAModel(base_model, config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time = train_model(model, tokenizer, train_texts, args, "HyLoRADA-Lite")
            
            elif method == "hylorada-pro":
                from hylorada.config import HyLoRADAPresets
                config = HyLoRADAPresets.pro()
                config.lora_rank = args.lora_rank
                config.lora_alpha = args.lora_rank * 2
                model = HyLoRADAModel(base_model, config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time = train_model(model, tokenizer, train_texts, args, "HyLoRADA-Pro")
            
            else:
                print(f"  Unknown method: {method}")
                continue
            
            # Evaluate
            print(f"  Evaluating {method}...")
            eval_results = evaluate_model(model, tokenizer, test_texts, args.max_length)
            
            results[method] = {
                "trainable_params": params,
                "train_time": train_time,
                **eval_results,
            }
            
            print(f"  ✓ {method}: PPL={eval_results['perplexity']:.2f}, Params={params:,}, Time={train_time:.1f}s")
            
            # Cleanup
            del base_model, model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[method] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Params':>12} {'Time':>10} {'PPL':>10} {'LiM PPL':>10}")
    print("-" * 70)
    
    for method, r in results.items():
        if "error" in r:
            print(f"{method:<12} {'ERROR':>12}")
        else:
            params = f"{r['trainable_params']:,}" if r['trainable_params'] else "0"
            time_s = f"{r['train_time']:.1f}s" if r['train_time'] else "-"
            ppl = f"{r['perplexity']:.2f}"
            lim = f"{r['lim_perplexity']:.2f}"
            print(f"{method:<12} {params:>12} {time_s:>10} {ppl:>10} {lim:>10}")
    
    print("=" * 70)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"benchmark_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
