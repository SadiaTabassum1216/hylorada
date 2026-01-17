"""
HyLoRADA Code Benchmark

Compares all PEFT methods on code understanding tasks with larger datasets.
"""

import argparse
import os
import warnings
import json
warnings.filterwarnings('ignore')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.baselines import StandardLoRA, get_baseline_model, BaselineConfig
from hylorada.trainer import HyLoRADATrainer, TrainingConfig


def load_code_dataset(tokenizer, max_length=512, num_train=500, num_test=100):
    """Load code dataset with fallback options."""
    print("\n[1] Loading code dataset...")
    
    datasets_to_try = [
        ("openai/openai_humaneval", None, "prompt", "canonical_solution"),
        ("google-research-datasets/mbpp", "sanitized", "text", "code"),
    ]
    
    dataset = None
    for ds_info in datasets_to_try:
        ds_name, config, text_col, code_col = ds_info
        try:
            print(f"  Trying {ds_name}...")
            ds = load_dataset(ds_name, config) if config else load_dataset(ds_name)
            split_name = "train" if "train" in ds else "test"
            data = ds[split_name]
            
            texts = []
            for item in data:
                text = item.get(text_col, "")
                code = item.get(code_col, "")
                if isinstance(code, list) and len(code) > 0:
                    code = code[0]
                if isinstance(code, str):
                    combined = f"# Problem:\n{text}\n\n# Solution:\n{code}"
                    texts.append(combined)
            
            if len(texts) >= num_train + num_test:
                dataset = {"train": texts[:num_train], "test": texts[num_train:num_train+num_test]}
                print(f"  ✓ Loaded {ds_name}: {num_train} train, {num_test} test")
                break
            else:
                # Use all available data
                split = int(len(texts) * 0.8)
                dataset = {"train": texts[:split], "test": texts[split:]}
                print(f"  ✓ Loaded {ds_name}: {len(dataset['train'])} train, {len(dataset['test'])} test")
                break
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    if dataset is None:
        raise RuntimeError("Could not load any code dataset")
    
    class CodeDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.encodings = tokenizer(
                texts, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
            )
        
        def __len__(self):
            return self.encodings["input_ids"].shape[0]
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["input_ids"][idx],
            }
    
    return (
        CodeDataset(dataset["train"], tokenizer, max_length),
        CodeDataset(dataset["test"], tokenizer, max_length),
    )


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return {"loss": avg_loss, "perplexity": perplexity}


def run_method(method_name, base_model_name, train_loader, test_loader, args, device):
    """Train and evaluate a single method."""
    print(f"\n{'='*60}")
    print(f"Running: {method_name.upper()}")
    print(f"{'='*60}")
    
    # Load fresh model
    print("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Apply PEFT method
    if method_name == "lora":
        config = BaselineConfig(lora_rank=args.rank, lora_alpha=args.rank * 2)
        peft_model = StandardLoRA(model, config)
    elif method_name == "dora":
        config = HyLoRADAConfig(lora_rank=args.rank, use_dora=True, daa_enabled=False, sparse_enabled=False)
        peft_model = HyLoRADAModel(model, config)
    elif method_name == "hylorada":
        config = HyLoRADAConfig(lora_rank=args.rank, use_hylorada=True, daa_enabled=True, sparse_enabled=False)
        peft_model = HyLoRADAModel(model, config)
    elif method_name == "hylorada_v2":
        config = HyLoRADAConfig(lora_rank=args.rank, use_hylorada_v2=True, daa_enabled=True, sparse_enabled=False)
        peft_model = HyLoRADAModel(model, config)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Print params
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    
    # Training config
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=20,
        output_dir=f"./results_code/{method_name}",
    )
    
    # Train
    start_time = time.time()
    trainer = HyLoRADATrainer(
        model=peft_model,
        train_dataloader=train_loader,
        config=training_config,
    )
    trainer.train()
    train_time = time.time() - start_time
    
    # Evaluate
    results = evaluate_model(peft_model, test_loader, device)
    results["trainable_params"] = trainable
    results["train_time"] = train_time
    
    print(f"  ✓ {method_name}: PPL={results['perplexity']:.2f}, Time={train_time:.1f}s")
    
    # Cleanup
    del model, peft_model
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Code benchmark for PEFT methods")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--methods", nargs="+", default=["lora", "dora", "hylorada", "hylorada_v2"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./results_code")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("PEFT Methods Code Benchmark")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Epochs: {args.epochs}, Train: {args.num_train}, Test: {args.num_test}")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    train_dataset, test_dataset = load_code_dataset(
        tokenizer, args.max_length, args.num_train, args.num_test
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size
    )
    
    # Run each method
    all_results = {}
    for method in args.methods:
        try:
            results = run_method(method, args.model, train_loader, test_loader, args, device)
            all_results[method] = results
        except Exception as e:
            print(f"  ✗ {method} failed: {e}")
            all_results[method] = {"error": str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<15} {'Params':>12} {'PPL':>8} {'Time':>8}")
    print("-"*60)
    for method, res in all_results.items():
        if "error" in res:
            print(f"{method:<15} {'ERROR':>12}")
        else:
            print(f"{method:<15} {res['trainable_params']:>12,} {res['perplexity']:>8.2f} {res['train_time']:>7.1f}s")
    print("="*60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()
