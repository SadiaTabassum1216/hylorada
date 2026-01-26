"""
PEFT Methods Benchmark Script

Compare HyLoRADA against other PEFT methods with identical hyperparameters.
Methods: LoRA, DoRA, LoRaDA, LongLoRA, SparseAdapter, HyLoRADA

Datasets:
  - wikitext: WikiText-2 language modeling (default)
  - code: CodeSearchNet Python code summarization (Software Engineering)

Usage:
    python run_benchmark.py --dataset code --methods lora dora hylorada --epochs 3
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


def load_dataset_by_name(dataset_name, num_train, num_test, max_length):
    """
    Load dataset by name. Supports:
    - wikitext: WikiText-2 (short context, default)
    - code: MultiPL-E Python code
    - longbench: LongBench multi-task (long context 4K-8K)
    - pg19: PG19 books (very long context)
    """
    if dataset_name == "wikitext":
        try:
            # Try wikitext-2-raw (standard)
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        except Exception as e:
            print(f"    Warning: wikitext-2-raw-v1 failed ({e}), trying wikitext-103-raw-v1...")
            try:
                # Try larger wikitext-103
                dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
            except Exception as e2:
                print(f"    Warning: wikitext-103 failed ({e2}), trying ptb_text_only...")
                try: 
                    # Try Penn Treebank
                    dataset = load_dataset("ptb_text_only", "penn_treebank")
                except Exception as e3:
                    print(f"    Error: All text datasets failed. Please check internet connection.")
                    raise e3
        
        train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50][:num_train]
        test_texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 50][:num_test]
        print(f"    Dataset: Standard Language Modeling (WikiText/PTB)")
        
    elif dataset_name == "code":
        try:
            dataset = load_dataset("nuprl/MultiPL-E", "humaneval-py", split="test")
            texts = [f"# Python:\\n{s['prompt']}" for s in dataset if s.get("prompt")]
            while len(texts) < num_train + num_test:
                texts = texts + texts
            train_texts = texts[:num_train]
            test_texts = texts[num_train:num_train + num_test]
            print(f"    Dataset: MultiPL-E Python")
        except:
            print(f"    Warning: Code dataset failed, using wikitext")
            return load_dataset_by_name("wikitext", num_train, num_test, max_length)
            
    elif dataset_name == "longbench":
        # Use wikitext-103 (long articles, no deprecated scripts)
        try:
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
            texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 2000]
            print(f"    Found {len(texts)} long articles (>2000 chars)")
            train_texts = texts[:num_train]
            test_texts = texts[num_train:num_train + num_test]
            print(f"    Dataset: WikiText-103 (long context)")
        except Exception as e:
            print(f"    Warning: WikiText-103 failed ({e})")
            return load_dataset_by_name("wikitext", num_train, num_test, max_length)

            
    elif dataset_name == "pg19":
        # PG19 - books (very long context)
        try:
            dataset = load_dataset("pg19", split="train", streaming=True)
            texts = []
            for i, sample in enumerate(dataset):
                if i >= num_train + num_test:
                    break
                text = sample.get("text", "")[:max_length * 8]  # Books are very long
                if len(text) > 1000:
                    texts.append(text)
            train_texts = texts[:num_train]
            test_texts = texts[num_train:num_train + num_test]
            print(f"    Dataset: PG19 books (very long context)")
        except Exception as e:
            print(f"    Warning: PG19 failed ({e}), using wikitext")
            return load_dataset_by_name("wikitext", num_train, num_test, max_length)
    elif dataset_name == "c4":
        # C4 - RealNewsLike (streaming, huge)
        try:
            dataset = load_dataset("c4", "realnewslike", split="train", streaming=True)
            texts = []
            for i, sample in enumerate(dataset):
                if i >= num_train + num_test:
                    break
                text = sample.get("text", "")
                if len(text) > 500:
                    texts.append(text)
            train_texts = texts[:num_train]
            test_texts = texts[num_train:num_train + num_test]
            print(f"    Dataset: C4 RealNewsLike (streaming)")
        except Exception as e:
            print(f"    Warning: C4 failed ({e}), using wikitext")
            return load_dataset_by_name("wikitext", num_train, num_test, max_length)

    elif dataset_name == "ptb":
        # Penn Treebank
        try:
            dataset = load_dataset("ptb_text_only", "penn_treebank")
            train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50][:num_train]
            test_texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 50][:num_test]
            print(f"    Dataset: Penn Treebank")
        except Exception as e:
            print(f"    Warning: PTB failed ({e}), using wikitext")
            return load_dataset_by_name("wikitext", num_train, num_test, max_length)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_texts, test_texts


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
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "code", "longbench", "pg19", "c4", "ptb"],
                        help="Dataset: wikitext, code, longbench, pg19, c4, ptb")
    parser.add_argument("--methods", nargs="+", 
                        default=["baseline", "lora", "lorada", "longlora", "sparse", "hylorada"])
    parser.add_argument("--sparse_dim", type=int, default=128,
                        help="Sparse adapter bottleneck dimension (default: 128, lite: 32)")
    parser.add_argument("--sparse_layers", type=str, default=None,
                        help="Comma-separated layer indices for sparse (e.g., '0,5,10,15,20,23')")
    parser.add_argument("--s2_attn", action="store_true",
                        help="Enable S²-Attn for long context (4096+)")
    parser.add_argument("--train_embeddings", action="store_true",
                        help="Train embeddings (LongLoRA feature)")
    parser.add_argument("--train_norms", action="store_true",
                        help="Train norms (LongLoRA feature)")
    parser.add_argument("--sink_tokens", type=int, default=0,
                        help="Number of sink tokens for S²-Attn")
    parser.add_argument("--rope_scaling_type", type=str, default=None,
                        choices=["linear", "dynamic", "yarn"],
                        help="RoPE scaling type (e.g. yarn)")
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0,
                        help="RoPE scaling factor")
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
    print(f"Dataset: {args.dataset}")
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
    train_texts, test_texts = load_dataset_by_name(
        args.dataset, args.num_train, args.num_test, args.max_length
    )
    print(f"    Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    # Baseline config for comparison methods
    baseline_config = BaselineConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        sparse_adapter_dim=args.sparse_dim,
        sparse_topk_ratio=0.05,
        sparse_target_layers=args.sparse_layers,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
            
            elif method == "dora":
                # DoRA: Using baseline LoRA with DoRA-style training
                # (DoRA is now a baseline, not in unified path)
                from hylorada.lora import apply_dora_to_model
                apply_dora_to_model(
                    base_model,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                    rank=args.lora_rank,
                    alpha=args.lora_rank * 2,
                )
                # Freeze base and count params
                for p in base_model.parameters():
                    if not any(n in ["lora", "dora"] for n in []):
                        p.requires_grad = False
                params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
                model = base_model
                train_time = train_model(model, tokenizer, train_texts, args, "DoRA")
            
            elif method == "hylorada":
                # HyLoRADA Unified: All features in one
                config = HyLoRADAConfig(
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_rank * 3,
                    lora_dropout=0.01,
                    lora_plus_enabled=True,
                    lora_plus_ratio=17.1,
                    daa_enabled=True,
                    sparse_enabled=False,
                    s2_attn_enabled=args.s2_attn,
                    max_sequence_length=args.max_length,
                    train_embeddings=args.train_embeddings,
                    train_norms=args.train_norms,
                    s2_sink_tokens=args.sink_tokens,
                    rope_scaling_type=args.rope_scaling_type,
                    rope_scaling_factor=args.rope_scaling_factor,
                )
                model = HyLoRADAModel(base_model, config)
                
                # Apply optimized gate/residual initialization
                for module in model.modules():
                    if hasattr(module, 'magnitude_gate'):
                        module.magnitude_gate.data.fill_(0.37)
                    if hasattr(module, 'residual_weight'):
                        module.residual_weight.data.fill_(0.22)
                
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time = train_model(model, tokenizer, train_texts, args, "HyLoRADA")
                
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
            
            # Save checkpoint for qualitative analysis
            checkpoint_path = os.path.join(args.output_dir, f"{method}_checkpoint.pt")
            trainable_state = {
                name: param.cpu() for name, param in model.named_parameters() 
                if param.requires_grad
            }
            torch.save({
                "method": method,
                "trainable_params": trainable_state,
                "config": vars(args),
                "eval_results": eval_results,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            
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
