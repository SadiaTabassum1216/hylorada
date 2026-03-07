"""
PEFT Methods Benchmark Script

Compare HyLoRADA against other PEFT methods with identical hyperparameters.
Methods: LoRA, DoRA, LoRaDA, LongLoRA, SparseAdapter, HyLoRADA

Datasets:
  - wikitext: WikiText-2 language modeling (default)
  - code: CodeSearchNet Python code summarization (Software Engineering)

Models (small, resource-efficient):
  - gpt2: GPT-2 Small (124M params)
  - distilgpt2: DistilGPT-2 (82M params)
  - gpt2-medium: GPT-2 Medium (355M params)
  - opt-125m: OPT 125M (Meta)
  - opt-350m: OPT 350M (Meta)
  - pythia-70m: Pythia 70M (EleutherAI)
  - pythia-160m: Pythia 160M (EleutherAI)
  - pythia-410m: Pythia 410M (EleutherAI)
  - qwen2-0.5b: Qwen2 0.5B (Alibaba)
  - tinyllama: TinyLlama 1.1B

Usage:
    python run_benchmark.py --dataset code --methods lora dora hylorada --epochs 3
    python run_benchmark.py --model pythia-160m --methods lora hylorada --epochs 3
    python run_benchmark.py --models gpt2 pythia-160m opt-125m --methods hylorada --epochs 3
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


# ============ Small Model Registry ============
# Maps short aliases to HuggingFace model IDs and their architecture info
# All models are small (<1B params unless noted) for resource-constrained validation
SMALL_MODELS = {
    # GPT-2 Family (OpenAI)
    "gpt2": {
        "hf_id": "openai-community/gpt2",
        "params": "124M",
        "architecture": "gpt2",
        "target_modules": ("c_attn", "c_proj"),
        "max_context": 1024,
    },
    "distilgpt2": {
        "hf_id": "distilbert/distilgpt2",
        "params": "82M",
        "architecture": "gpt2",
        "target_modules": ("c_attn", "c_proj"),
        "max_context": 1024,
    },
    "gpt2-medium": {
        "hf_id": "openai-community/gpt2-medium",
        "params": "355M",
        "architecture": "gpt2",
        "target_modules": ("c_attn", "c_proj"),
        "max_context": 1024,
    },
    # OPT Family (Meta)
    "opt-125m": {
        "hf_id": "facebook/opt-125m",
        "params": "125M",
        "architecture": "opt",
        "target_modules": ("q_proj", "k_proj", "v_proj", "out_proj"),
        "max_context": 2048,
    },
    "opt-350m": {
        "hf_id": "facebook/opt-350m",
        "params": "350M",
        "architecture": "opt",
        "target_modules": ("q_proj", "k_proj", "v_proj", "out_proj"),
        "max_context": 2048,
    },
    # Pythia Family (EleutherAI)
    "pythia-70m": {
        "hf_id": "EleutherAI/pythia-70m",
        "params": "70M",
        "architecture": "gpt_neox",
        "target_modules": ("query_key_value", "dense"),
        "max_context": 2048,
    },
    "pythia-160m": {
        "hf_id": "EleutherAI/pythia-160m",
        "params": "160M",
        "architecture": "gpt_neox",
        "target_modules": ("query_key_value", "dense"),
        "max_context": 2048,
    },
    "pythia-410m": {
        "hf_id": "EleutherAI/pythia-410m",
        "params": "410M",
        "architecture": "gpt_neox",
        "target_modules": ("query_key_value", "dense"),
        "max_context": 2048,
    },
    # Qwen Family (Alibaba) - smallest models
    "qwen2-0.5b": {
        "hf_id": "Qwen/Qwen2-0.5B",
        "params": "500M",
        "architecture": "qwen2",
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "max_context": 32768,
    },
    # TinyLlama (slightly larger but popular)
    "tinyllama": {
        "hf_id": "TinyLlama/TinyLlama_v1.1",
        "params": "1.1B",
        "architecture": "llama",
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "max_context": 2048,
    },
}


def get_model_info(model_name: str) -> dict:
    """Get model info from registry or infer from model name."""
    # Check if it's an alias
    if model_name.lower() in SMALL_MODELS:
        return SMALL_MODELS[model_name.lower()]
    
    # Otherwise, treat as a direct HuggingFace ID
    # Infer architecture from name
    lower_name = model_name.lower()
    if "gpt2" in lower_name:
        arch = "gpt2"
        target_modules = ("c_attn", "c_proj")
        max_ctx = 1024
    elif "opt" in lower_name:
        arch = "opt"
        target_modules = ("q_proj", "k_proj", "v_proj", "out_proj")
        max_ctx = 2048
    elif "pythia" in lower_name or "gpt-neox" in lower_name:
        arch = "gpt_neox"
        target_modules = ("query_key_value", "dense")
        max_ctx = 2048
    elif "qwen" in lower_name:
        arch = "qwen2"
        target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
        max_ctx = 32768
    elif "llama" in lower_name or "tinyllama" in lower_name:
        arch = "llama"
        target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
        max_ctx = 2048
    elif "falcon" in lower_name:
        arch = "falcon"
        target_modules = ("query_key_value", "dense")
        max_ctx = 2048
    else:
        # Default to LLaMA-style
        arch = "unknown"
        target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
        max_ctx = 2048
    
    return {
        "hf_id": model_name,
        "params": "unknown",
        "architecture": arch,
        "target_modules": target_modules,
        "max_context": max_ctx,
    }


def list_available_models():
    """Print available model presets."""
    print("\nAvailable Small Models:")
    print("-" * 70)
    print(f"{'Alias':<15} {'Params':<10} {'Architecture':<12} {'Max Context':<12}")
    print("-" * 70)
    for alias, info in SMALL_MODELS.items():
        print(f"{alias:<15} {info['params']:<10} {info['architecture']:<12} {info['max_context']:<12}")
    print("-" * 70)
    print("Use --model <alias> or --models <alias1> <alias2> ... for multi-model validation")
    print("You can also use any HuggingFace model ID directly.")
    print()


def load_fresh_model(model_name_or_alias, device, dtype):
    """Load a fresh model instance from alias or HuggingFace ID."""
    model_info = get_model_info(model_name_or_alias)
    hf_id = model_info["hf_id"]
    
    print(f"  Loading {hf_id} ({model_info['params']} params, {model_info['architecture']} arch)...")
    
    return AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)


def train_model(model, tokenizer, train_texts, args, method_name):
    """Train a model and return training time and memory usage."""
    # Ensure model is on correct device and dtype
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model.to(device=device, dtype=dtype)
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    train_dataloader = create_long_context_dataloader(
        dataset=train_texts, tokenizer=tokenizer,
        max_length=args.max_length, batch_size=args.batch_size,
    )
    
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.1,
        logging_steps=50,
    )
    
    trainer = HyLoRADATrainer(model=model, train_dataloader=train_dataloader, config=train_config)
    
    print(f"  Training {method_name}...")
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    
    # Get memory stats
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
        current_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
    else:
        peak_memory = 0.0
        current_memory = 0.0
    
    return train_time, peak_memory, current_memory


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
        # Try multiple code datasets in order of preference
        code_datasets = [
            ("mbpp", "test", lambda s: f"# Python:\n{s.get('text', s.get('prompt', ''))}", "MBPP Python"),
            ("google-research-datasets/mbpp", "test", lambda s: f"# Python:\n{s.get('text', s.get('prompt', ''))}", "MBPP Python"),
            ("nuprl/MultiPL-E", "py", lambda s: f"# Python:\n{s.get('prompt', '')}", "MultiPL-E Python"),
            ("codeparrot/codeparrot-clean-valid", None, lambda s: s.get("content", ""), "CodeParrot"),
        ]
        
        texts = None
        ds_name = None
        for ds_path, split_name, extractor, name in code_datasets:
            try:
                if split_name:
                    dataset = load_dataset(ds_path, split=split_name, trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_path, split="train", trust_remote_code=True)
                texts = [extractor(s) for s in dataset if extractor(s).strip()]
                texts = [t for t in texts if len(t) > 20]
                if len(texts) >= 10:
                    ds_name = name
                    print(f"    Dataset: {ds_name}")
                    break
            except Exception as e:
                print(f"    Warning: {name} failed ({e})")
                continue
        
        if texts is None or len(texts) < 10:
            print(f"    Warning: All code datasets failed, using wikitext")
            return load_dataset_by_name("wikitext", num_train, num_test, max_length)
        
        while len(texts) < num_train + num_test:
            texts = texts + texts
        train_texts = texts[:num_train]
        test_texts = texts[num_train:num_train + num_test]
            
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
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Single model alias or HuggingFace ID (default: gpt2)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Multiple models for cross-model validation (e.g., --models gpt2 pythia-160m opt-125m)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available small model presets")
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
    
    # List models and exit if requested
    if args.list_models:
        list_available_models()
        return
    
    # Parse sparse_layers if provided
    if args.sparse_layers:
        args.sparse_layers = [int(x.strip()) for x in args.sparse_layers.split(",")]
    
    # Determine models to benchmark
    if args.models:
        model_list = args.models
    else:
        model_list = [args.model]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Run benchmarks for each model
    all_results = {}
    for model_alias in model_list:
        print("\n" + "=" * 80)
        print(f"BENCHMARKING MODEL: {model_alias}")
        print("=" * 80)
        
        model_results = run_benchmark_for_model(model_alias, args, device, dtype)
        all_results[model_alias] = model_results
    
    # Print cross-model summary if multiple models
    if len(model_list) > 1:
        print_cross_model_summary(all_results, model_list)
    
    # Save combined results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"benchmark_multimodel_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def print_cross_model_summary(all_results, model_list):
    """Print a cross-model comparison summary."""
    print("\n" + "=" * 90)
    print("CROSS-MODEL COMPARISON")
    print("=" * 90)
    
    # Collect all methods used
    all_methods = set()
    for model_name in model_list:
        if model_name in all_results:
            all_methods.update(all_results[model_name].keys())
    
    # Print header
    header = f"{'Method':<12}"
    for model_name in model_list:
        model_info = get_model_info(model_name)
        short_name = model_name[:10]
        header += f" {short_name:>12}"
    print(header)
    print("-" * 90)
    
    # Print perplexity for each method
    for method in sorted(all_methods):
        if method == "error":
            continue
        row = f"{method:<12}"
        for model_name in model_list:
            if model_name in all_results and method in all_results[model_name]:
                r = all_results[model_name][method]
                if "error" in r:
                    row += f" {'ERROR':>12}"
                else:
                    ppl = r.get("perplexity", float("inf"))
                    row += f" {ppl:>12.2f}"
            else:
                row += f" {'-':>12}"
        print(row)
    
    print("=" * 90)


def run_benchmark_for_model(model_alias, args, device, dtype):
    """Run benchmark for a single model."""
    model_info = get_model_info(model_alias)
    hf_id = model_info["hf_id"]
    target_modules = model_info["target_modules"]
    architecture = model_info["architecture"]
    
    print(f"Model: {hf_id} ({model_info['params']} params)")
    print(f"Architecture: {architecture}")
    print(f"Target modules: {target_modules}")
    
    print("=" * 70)
    print("PEFT Methods Benchmark")
    print("=" * 70)
    print(f"Model: {hf_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Hyperparameters: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"Max Length: {args.max_length}, LoRA Rank: {args.lora_rank}")
    print("=" * 70)
    
    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
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
            base_model = load_fresh_model(model_alias, device, dtype)
            
            # Extend context for GPT-2 if needed
            if architecture == "gpt2" and args.max_length > 1024:
                extend_gpt2_context(base_model, args.max_length)
            
            # Ensure base model is in correct dtype (critical for evaluation stability)
            base_model.to(dtype=dtype)
                
            if method == "baseline":
                model = base_model
                train_time = 0
                params = 0
                peak_memory = 0
                current_memory = 0
                
            elif method == "lora":
                model = StandardLoRA(base_model, baseline_config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time, peak_memory, current_memory = train_model(model, tokenizer, train_texts, args, "LoRA")
            
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
                train_time, peak_memory, current_memory = train_model(model, tokenizer, train_texts, args, "DoRA")
            
            elif method == "hylorada":
                # HyLoRADA: Context-length adaptive configuration
                # Uses Position-Content Fusion (PCF) - unified approach
                # PCF learns when to use position/landmark information
                is_long_context = args.max_length >= 2048
                
                config = HyLoRADAConfig(
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_rank * 2,  # Standard rsLoRA scaling
                    lora_dropout=0.05,
                    use_dora_magnitude=False,  # Ablation: DoRA degrades on short context
                    num_landmarks=8,  # Always 8: shared PCF means no extra cost
                    num_position_buckets=64,  # Position bucketing granularity
                    s2_attn_enabled=args.s2_attn if args.max_length >= 4096 else False,
                    max_sequence_length=args.max_length,
                    train_embeddings=args.train_embeddings,
                    train_norms=args.train_norms,
                    s2_sink_tokens=args.sink_tokens if args.s2_attn else 0,
                    rope_scaling_type=args.rope_scaling_type,
                    rope_scaling_factor=args.rope_scaling_factor,
                )
                
                print(f"  Config: long_context={is_long_context}, landmarks={config.num_landmarks}, position_buckets={config.num_position_buckets}")
                model = HyLoRADAModel(base_model, config)
                
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time, peak_memory, current_memory = train_model(model, tokenizer, train_texts, args, "HyLoRADA")
                
            elif method == "lorada":
                model = LoRaDAModel(base_model, baseline_config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time, peak_memory, current_memory = train_model(model, tokenizer, train_texts, args, "LoRaDA")
                
            elif method == "longlora":
                model = LongLoRAModel(base_model, baseline_config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time, peak_memory, current_memory = train_model(model, tokenizer, train_texts, args, "LongLoRA")
                
            elif method == "sparse":
                model = SparseAdapterModel(base_model, baseline_config)
                model.print_trainable_params()
                params = model.count_params()["trainable_params"]
                train_time, peak_memory, current_memory = train_model(model, tokenizer, train_texts, args, "SparseAdapter")
            
            else:
                print(f"  Unknown method: {method}")
                continue
            
            # Evaluate
            print(f"  Evaluating {method}...")
            eval_results = evaluate_model(model, tokenizer, test_texts, args.max_length)
            
            results[method] = {
                "trainable_params": params,
                "train_time": train_time,
                "peak_memory_gb": peak_memory,
                "current_memory_gb": current_memory,
                **eval_results,
            }
            
            memory_str = f", Mem={peak_memory:.2f}GB" if torch.cuda.is_available() else ""
            print(f"  ✓ {method}: PPL={eval_results['perplexity']:.2f}, Params={params:,}, Time={train_time:.1f}s{memory_str}")
            
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
    
    # Summary for this model
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY for {model_alias}")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"{'Method':<12} {'Params':>12} {'Mem(GB)':>10} {'Time':>10} {'PPL':>10} {'LiM PPL':>10}")
    else:
        print(f"{'Method':<12} {'Params':>12} {'Time':>10} {'PPL':>10} {'LiM PPL':>10}")
    print("-" * 80)
    
    for method, r in results.items():
        if "error" in r:
            print(f"{method:<12} {'ERROR':>12}")
        else:
            params = f"{r['trainable_params']:,}" if r['trainable_params'] else "0"
            time_s = f"{r['train_time']:.1f}s" if r['train_time'] else "-"
            ppl = f"{r['perplexity']:.2f}"
            lim = f"{r['lim_perplexity']:.2f}"
            if torch.cuda.is_available():
                mem = f"{r.get('peak_memory_gb', 0):.2f}"
                print(f"{method:<12} {params:>12} {mem:>10} {time_s:>10} {ppl:>10} {lim:>10}")
            else:
                print(f"{method:<12} {params:>12} {time_s:>10} {ppl:>10} {lim:>10}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
