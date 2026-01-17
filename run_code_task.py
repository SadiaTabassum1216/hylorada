"""
HyLoRADA v2 Code Task Runner

Trains and benchmarks HyLoRADA on code understanding/generation tasks.
Uses Python code datasets for training.
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hylorada import HyLoRADAConfig, HyLoRADAModel, StructureEncoder
from hylorada.trainer import HyLoRADATrainer, TrainingConfig


def load_code_dataset(tokenizer, max_length=1024, num_train=1000, num_test=100):
    """Load a code dataset for training."""
    print("[1] Loading code dataset...")
    
    # Try different code datasets in order of preference
    datasets_to_try = [
        ("bigcode/starcoderdata", "python", "content"),
        ("codeparrot/github-code", "Python", "code"),
        ("nuprl/MultiPL-E", "humaneval-py", "prompt"),
    ]
    
    dataset = None
    for ds_name, config_or_filter, text_col in datasets_to_try:
        try:
            print(f"  Trying {ds_name}...")
            if ds_name == "bigcode/starcoderdata":
                # StarCoderData - streaming dataset
                ds = load_dataset(ds_name, data_dir="python", split="train", streaming=True)
                # Take samples from streaming
                samples = []
                for i, sample in enumerate(ds):
                    if i >= num_train + num_test:
                        break
                    samples.append(sample[text_col])
                dataset = {"train": samples[:num_train], "test": samples[num_train:]}
                print(f"  ✓ Loaded {ds_name}")
                break
            elif ds_name == "nuprl/MultiPL-E":
                ds = load_dataset(ds_name, config_or_filter, trust_remote_code=True)
                texts = [s[text_col] for s in ds["test"]]
                # Split for train/test
                dataset = {"train": texts[:num_train], "test": texts[num_train:num_train+num_test]}
                print(f"  ✓ Loaded {ds_name}")
                break
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    # Fallback to a simple Python code generation approach
    if dataset is None:
        print("  Using simple code samples fallback...")
        code_samples = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
            "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid",
        ] * (num_train // 4 + 1)
        dataset = {"train": code_samples[:num_train], "test": code_samples[:num_test]}
    
    # Tokenize
    def tokenize(texts):
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
        )
    
    train_dataset = tokenize(dataset["train"])
    test_dataset = tokenize(dataset["test"])
    
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset


def train_hylorada_v2_code(args):
    """Train HyLoRADA v2 on code task."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("HyLoRADA v2 Code Task Training")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"V2 Mode: {args.use_v2}")
    print(f"{'='*60}\n")
    
    # Load tokenizer and model
    print("[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Load dataset
    train_dataset, test_dataset = load_code_dataset(
        tokenizer,
        max_length=args.max_length,
        num_train=args.num_train,
        num_test=args.num_test,
    )
    
    # Configure HyLoRADA
    print("\n[2] Applying HyLoRADA...")
    config = HyLoRADAConfig(
        lora_rank=args.rank,
        lora_alpha=args.rank * 2,
        use_hylorada=not args.use_v2,  # v1 if not v2
        use_hylorada_v2=args.use_v2,
        structure_dim=32,
        lora_plus_enabled=True,
        lora_plus_ratio=10.0,
        daa_enabled=True,
        sparse_enabled=False,  # Disable for simpler training
    )
    
    hylorada_model = HyLoRADAModel(model, config)
    hylorada_model.print_trainable_params()
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
    )
    
    # Training config
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=50,
        output_dir=args.save_dir or "./results_code",
    )
    
    # Train
    print(f"\n[3] Training for {args.epochs} epochs...")
    trainer = HyLoRADATrainer(
        model=hylorada_model,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        config=training_config,
    )
    trainer.train()

    
    # Evaluate
    print("\n[4] Evaluating...")
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"{'='*60}")
    
    # Save checkpoint
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "hylorada_v2_code.pt")
        torch.save({
            "config": config.to_dict(),
            "perplexity": perplexity,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k.lower()},
        }, save_path)
        print(f"Saved to: {save_path}")
    
    return perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HyLoRADA v2 on code task")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--use_v2", action="store_true", help="Use HyLoRADA v2")
    parser.add_argument("--save_dir", type=str, default="./results_code")
    
    args = parser.parse_args()
    train_hylorada_v2_code(args)
