"""
Qualitative Comparison Demo

Generate examples showing HyLoRADA's improvements:
1. Code summarization quality
2. Lost-in-the-middle problem handling

Usage:
    python generate_examples.py --checkpoint_dir ./checkpoints
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.baselines import StandardLoRA, BaselineConfig
from hylorada.lora import apply_dora_to_model


# ============ Code Summarization Examples ============

CODE_EXAMPLES = [
    {
        "name": "Binary Search",
        "code": '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
        "ground_truth": "Performs binary search on a sorted array to find target element index."
    },
    {
        "name": "Merge Sort",
        "code": '''def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result''',
        "ground_truth": "Implements merge sort algorithm using divide-and-conquer approach."
    },
    {
        "name": "LRU Cache",
        "code": '''class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)''',
        "ground_truth": "Implements LRU cache with O(n) get/put operations using dict and list."
    }
]


# ============ Lost-in-the-Middle Examples ============

def create_lim_example(key_position="middle"):
    """Create a long context where key fact is at start/middle/end."""
    
    filler = "The weather today is sunny with a chance of clouds. " * 20
    key_fact = "IMPORTANT: The secret code is ALPHA-7892."
    
    if key_position == "start":
        context = key_fact + " " + filler + filler
    elif key_position == "middle":
        context = filler + " " + key_fact + " " + filler
    else:  # end
        context = filler + filler + " " + key_fact
    
    question = "What is the secret code mentioned in the text?"
    expected = "ALPHA-7892"
    
    return {
        "context": context,
        "question": question,
        "expected": expected,
        "position": key_position
    }


def generate_summary(model, tokenizer, code, max_new_tokens=50):
    """Generate code summary from model."""
    prompt = f"# Code:\n{code}\n\n# Summary:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the new tokens
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def generate_lim_answer(model, tokenizer, context, question, max_new_tokens=30):
    """Generate answer for lost-in-the-middle test."""
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def print_code_comparison(models, tokenizer, example):
    """Print code summarization comparison."""
    print(f"\n{'='*60}")
    print(f"CODE EXAMPLE: {example['name']}")
    print(f"{'='*60}")
    print(f"\n```python\n{example['code']}\n```\n")
    print(f"GROUND TRUTH: {example['ground_truth']}\n")
    print("-" * 40)
    
    for name, model in models.items():
        summary = generate_summary(model, tokenizer, example["code"])
        print(f"{name:12}: {summary}")


def print_lim_comparison(models, tokenizer, example):
    """Print lost-in-the-middle comparison."""
    print(f"\n{'='*60}")
    print(f"LOST-IN-THE-MIDDLE: Key fact at {example['position'].upper()}")
    print(f"{'='*60}")
    print(f"\nQuestion: {example['question']}")
    print(f"Expected: {example['expected']}")
    print("-" * 40)
    
    for name, model in models.items():
        answer = generate_lim_answer(model, tokenizer, example["context"], example["question"])
        correct = example["expected"].lower() in answer.lower()
        status = "✓" if correct else "✗"
        print(f"{name:12}: {answer[:50]} {status}")


def load_checkpoint(model, checkpoint_path):
    """Load trainable parameters from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    trainable_state = checkpoint["trainable_params"]
    
    for name, param in model.named_parameters():
        if name in trainable_state:
            param.data.copy_(trainable_state[name])
    
    return checkpoint.get("eval_results", {})


def main():
    parser = argparse.ArgumentParser(description="Generate comparison examples")
    parser.add_argument("--model", type=str, default="openai-community/gpt2")
    # ... (skipping unchanged args)
    
    # ...

    for name, ckpt_path in checkpoint_files.items():
        if os.path.exists(ckpt_path):
            # ... 

            # Determine target modules
            if "gpt2" in args.model.lower():
                target_modules = ("c_attn", "c_proj")
            else:
                target_modules = ("q_proj", "v_proj")

            # Apply appropriate adapter
            if name == "LoRA":
                model = StandardLoRA(base_model, BaselineConfig(lora_rank=args.lora_rank))
            elif name == "DoRA":
                base_model, _ = apply_dora_to_model(
                    base_model, 
                    rank=args.lora_rank,
                    target_modules=target_modules
                )
                model = base_model
            else:  # HyLoRADA
                from hylorada import HyLoRADAConfig, HyLoRADAModel
                config = HyLoRADAConfig(lora_rank=args.lora_rank, use_hylorada=True)
                model = HyLoRADAModel(base_model, config)
            
            # Load trained weights
            load_checkpoint(model, ckpt_path)
            models[name] = model
            print(f"    ✓ Loaded {name}")
        else:
            print(f"  ⚠ Checkpoint not found: {ckpt_path}")
    
    if not models:
        print("\nNo checkpoints found! Run benchmark first:")
        print("  python run_benchmark.py --methods lora dora hylorada --epochs 3")
        print("\nUsing base model only for demo...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        models["Base"] = base_model
    
    # Generate markdown output
    output_lines = []
    output_lines.append("# Qualitative Comparison Examples\n")
    output_lines.append("## Code Summarization\n")
    
    print("\n" + "="*60)
    print("CODE SUMMARIZATION EXAMPLES")
    print("="*60)
    
    for example in CODE_EXAMPLES:
        print_code_comparison(models, tokenizer, example)
        
        # Add to markdown
        output_lines.append(f"### {example['name']}\n")
        output_lines.append(f"```python\n{example['code']}\n```\n")
        output_lines.append(f"**Ground Truth**: {example['ground_truth']}\n")
        output_lines.append("\n| Model | Generated Summary |\n|-------|-------------------|\n")
        
        for name, model in models.items():
            summary = generate_summary(model, tokenizer, example["code"])
            output_lines.append(f"| {name} | {summary[:80]}... |\n")
        
        output_lines.append("\n")
    
    print("\n" + "="*60)
    print("LOST-IN-THE-MIDDLE EXAMPLES")
    print("="*60)
    
    output_lines.append("## Lost-in-the-Middle Test\n")
    output_lines.append("Key information placed at different positions in long context.\n\n")
    
    for position in ["start", "middle", "end"]:
        example = create_lim_example(position)
        print_lim_comparison(models, tokenizer, example)
        
        output_lines.append(f"### Key at {position.upper()}\n")
        output_lines.append(f"**Question**: {example['question']}\n")
        output_lines.append(f"**Expected**: {example['expected']}\n\n")
        output_lines.append("| Model | Answer | Correct |\n|-------|--------|--------|\n")
        
        for name, model in models.items():
            answer = generate_lim_answer(model, tokenizer, example["context"], example["question"])
            correct = "✓" if example["expected"].lower() in answer.lower() else "✗"
            output_lines.append(f"| {name} | {answer[:40]}... | {correct} |\n")
        
        output_lines.append("\n")
    
    # Save markdown
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("".join(output_lines))
    
    print(f"\n\nResults saved to: {args.output}")
    print("\nNOTE: For real comparison, train models with:")
    print("  python run_benchmark.py --dataset code --methods lora dora hylorada --epochs 3")
    print("Then load the saved checkpoints in this script.")


if __name__ == "__main__":
    main()
