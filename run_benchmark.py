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
import torch.nn as nn
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
    # Ensure model is on correct device and dtype
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model.to(device=device, dtype=dtype)
    
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
        # Use working wikitext datasets
        fallbacks = [
            ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            ("Salesforce/wikitext", "wikitext-103-raw-v1"),
            ("roneneldan/TinyStories", None),
        ]
        dataset = None
        for ds_name, subset in fallbacks:
            try:
                if subset:
                    dataset = load_dataset(ds_name, subset)
                else:
                    dataset = load_dataset(ds_name, split="train")
                    # TinyStories format
                    texts = [t["text"] for t in dataset if len(t.get("text", "").strip()) > 50]
                    train_texts = texts[:num_train]
                    test_texts = texts[num_train:num_train + num_test]
                    print(f"    Dataset: {ds_name}")
                    return train_texts, test_texts
                break
            except Exception as e:
                print(f"    Warning: {ds_name} failed ({e})")
                continue
        
        if dataset is None:
            raise RuntimeError("All text datasets failed. Please check internet connection.")
        
        train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50][:num_train]
        test_texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 50][:num_test]
        print(f"    Dataset: Standard Language Modeling (WikiText)")
        
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
        # Use Salesforce wikitext-103 (long articles)
        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
            # Concatenate all text
            print("    Concatenating texts for long context...")
            all_text = "\n\n".join([t for t in dataset["train"]["text"] if len(t.strip()) > 0])
            
            # Create chunks (approximate by chars, assuming 4 chars/token)
            chunk_size = max_length * 4
            texts = [all_text[i:i + chunk_size] for i in range(0, len(all_text), chunk_size)]
            
            # Limit number of samples to avoid excessive memory
            texts = texts[:num_train + num_test]
            
            print(f"    Created {len(texts)} chunks of ~{chunk_size} chars")
            train_texts = texts[:num_train]
            test_texts = texts[num_train:num_train + num_test]
            print(f"    Dataset: WikiText-103 (concatenated & chunked)")
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
        # Penn Treebank - try alternative
        try:
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            texts = [t["text"] for t in dataset if len(t.get("text", "").strip()) > 50]
            train_texts = texts[:num_train]
            test_texts = texts[num_train:num_train + num_test]
            print(f"    Dataset: TinyStories (PTB alternative)")
        except Exception as e:
            print(f"    Warning: TinyStories failed ({e}), using wikitext")
            return load_dataset_by_name("wikitext", num_train, num_test, max_length)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_texts, test_texts


def extend_gpt2_context(model, new_length):
    """Resize GPT-2 position embeddings to support longer context."""
    # Check if this is GPT-2 architecture
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "wpe"):
        return
    
    old_wpe = model.transformer.wpe
    old_len, dim = old_wpe.weight.shape
    
    if new_length <= old_len:
        return

    print(f"  Extending GPT-2 WPE: {old_len} -> {new_length}")
    new_wpe = nn.Embedding(new_length, dim)
    
    # Copy existing weights and initialize new ones
    with torch.no_grad():
        new_wpe.weight[:old_len] = old_wpe.weight
        # Initialize rest with small noise
        new_wpe.weight[old_len:].normal_(mean=0.0, std=0.02)
        
    model.transformer.wpe = new_wpe
    model.config.n_positions = new_length
    
    # Ensure device matches
    if hasattr(model, "device"):
        model.transformer.wpe.to(model.device)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PEFT Methods")
    parser.add_argument("--model", type=str, default="openai-community/gpt2")
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

    # Define target modules based on model family
    is_gpt2 = "gpt2" in args.model.lower()
    if is_gpt2:
        target_modules = ("c_attn", "c_proj")
        print(f"Detected GPT-2 model. Using target modules: {target_modules}")
    else:
        target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
        print(f"Detected LLaMA/Qwen model. Using target modules: {target_modules}")
    
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
            
            # Extend context for GPT-2 if needed
            if "gpt2" in args.model.lower() and args.max_length > 1024:
                extend_gpt2_context(base_model, args.max_length)
                
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
                # DoRA: Using DoRA layers
                from hylorada.lora import apply_dora_to_model
                apply_dora_to_model(
                    base_model,
                    target_modules=target_modules,
                    rank=args.lora_rank,
                    alpha=args.lora_rank * 2,
                )
                # Freeze base params, keep DoRA params trainable
                for name, p in base_model.named_parameters():
                    if "lora" in name or "dora" in name or "magnitude" in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
                print(f"  DoRA trainable params: {params:,}")
                
                # Wrap model to add get_trainable_params method
                class DoRAWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    def forward(self, *args, **kwargs):
                        return self.model(*args, **kwargs)
                    def get_trainable_params(self):
                        return [p for p in self.model.parameters() if p.requires_grad]
                    def parameters(self):
                        return self.model.parameters()
                    def named_parameters(self):
                        return self.model.named_parameters()
                
                model = DoRAWrapper(base_model)
                train_time = train_model(model, tokenizer, train_texts, args, "DoRA")
            
            elif method == "hylorada":
                # HyLoRADA Unified: rsLoRA + DoRA + LandmarkLoRA
                config = HyLoRADAConfig(
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_rank * 3,
                    lora_dropout=0.01,
                    # daa_enabled merged into position_bias_enabled (default True)
                    # sparse_enabled removed (not implemented)
                    s2_attn_enabled=args.s2_attn,  # Allow enabling if flag is set (Works on GPT-2)
    
                    landmark_enabled=True,  # Enable LandmarkLoRA for better context summary
                    max_sequence_length=args.max_length,
                    train_embeddings=args.train_embeddings,
                    train_norms=args.train_norms,
                    s2_sink_tokens=args.sink_tokens,
                    rope_scaling_type=args.rope_scaling_type,
                    rope_scaling_factor=args.rope_scaling_factor,
                )
                model = HyLoRADAModel(base_model, config)
                
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
