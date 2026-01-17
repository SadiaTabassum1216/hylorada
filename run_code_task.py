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
        # OpenAI HumanEval - always available
        ("openai/openai_humaneval", None, "prompt", "canonical_solution"),
        # CodeParrot APPS - competitive programming
        ("codeparrot/apps", "introductory", "question", "solutions"),
        # MBPP - basic Python problems
        ("google-research-datasets/mbpp", "sanitized", "text", "code"),
    ]
    
    dataset = None
    for ds_info in datasets_to_try:
        ds_name, config, text_col, code_col = ds_info
        try:
            print(f"  Trying {ds_name}...")
            if config:
                ds = load_dataset(ds_name, config)
            else:
                ds = load_dataset(ds_name)
            
            # Get the split (usually 'test' for eval datasets)
            split_name = "train" if "train" in ds else "test"
            data = ds[split_name]
            
            # Combine prompt + solution for training
            texts = []
            for item in data:
                text = item.get(text_col, "")
                code = item.get(code_col, "")
                # Handle list of solutions
                if isinstance(code, list) and len(code) > 0:
                    code = code[0]
                if isinstance(code, str):
                    combined = f"# Problem:\n{text}\n\n# Solution:\n{code}"
                    texts.append(combined)
            
            if len(texts) >= 10:  # Need at least some samples
                # Split into train/test
                train_texts = texts[:min(num_train, len(texts)-num_test)]
                test_texts = texts[len(train_texts):len(train_texts)+num_test]
                dataset = {"train": train_texts, "test": test_texts}
                print(f"  ✓ Loaded {ds_name}: {len(train_texts)} train, {len(test_texts)} test")
                break
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    # Fallback to diverse Python code samples
    if dataset is None:
        print("  Using diverse code samples fallback...")
        code_samples = [
            "def fibonacci(n):\n    '''Calculate nth Fibonacci number'''\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
            "def factorial(n):\n    '''Calculate factorial recursively'''\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
            "def quicksort(arr):\n    '''QuickSort implementation'''\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            "def merge_sort(arr):\n    '''MergeSort implementation'''\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
            "def binary_search(arr, target):\n    '''Binary search in sorted array'''\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            "class LinkedList:\n    '''Simple linked list'''\n    def __init__(self):\n        self.head = None\n    def append(self, val):\n        if not self.head:\n            self.head = Node(val)\n        else:\n            curr = self.head\n            while curr.next:\n                curr = curr.next\n            curr.next = Node(val)",
            "def is_palindrome(s):\n    '''Check if string is palindrome'''\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]",
            "def two_sum(nums, target):\n    '''Find two numbers that add to target'''\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []",
            "def max_subarray(nums):\n    '''Kadane's algorithm for max subarray sum'''\n    max_sum = curr_sum = nums[0]\n    for num in nums[1:]:\n        curr_sum = max(num, curr_sum + num)\n        max_sum = max(max_sum, curr_sum)\n    return max_sum",
            "class BinaryTree:\n    '''Binary tree with traversal'''\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None\n    def inorder(self):\n        result = []\n        if self.left:\n            result += self.left.inorder()\n        result.append(self.val)\n        if self.right:\n            result += self.right.inorder()\n        return result",
        ]
        # Create more variety by combining samples
        extended = code_samples * (num_train // len(code_samples) + 1)
        dataset = {"train": extended[:num_train], "test": code_samples[:num_test]}

    
    # Custom dataset that returns dicts
    class CodeDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
        
        def __len__(self):
            return self.encodings["input_ids"].shape[0]
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["input_ids"][idx],  # Causal LM labels
            }
    
    train_dataset = CodeDataset(dataset["train"], tokenizer, max_length)
    test_dataset = CodeDataset(dataset["test"], tokenizer, max_length)
    
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
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
