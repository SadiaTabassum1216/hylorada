"""
HyLoRADA Evaluation Script

Evaluate a trained HyLoRADA model on test data.

Usage:
    python evaluate_hylorada.py --model Qwen/Qwen2-0.5B --weights ./output/final/hylorada_weights.pt
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.evaluation import evaluate_perplexity, evaluate_lost_in_the_middle


def main():
    parser = argparse.ArgumentParser(description="Evaluate HyLoRADA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--weights", type=str, required=True, help="Path to HyLoRADA weights")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--lora_rank", type=int, default=8)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print("=" * 60)
    print("HyLoRADA Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
    print("=" * 60)
    
    # Load tokenizer and model
    print("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    
    # Apply HyLoRADA and load weights
    print("[2] Loading HyLoRADA weights...")
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
    model.load_hylorada(args.weights)
    model.eval()
    
    # Load test data
    print("\n[3] Loading test data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    test_texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 50][:args.num_samples]
    print(f"    Using {len(test_texts)} test samples")
    
    # Evaluate perplexity
    print("\n[4] Evaluating perplexity...")
    ppl_result = evaluate_perplexity(model, tokenizer, test_texts, max_length=args.max_length)
    
    # Evaluate lost-in-middle
    print("[5] Evaluating lost-in-middle...")
    lim_result = evaluate_lost_in_the_middle(model, tokenizer, test_texts[:20], max_length=args.max_length)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Perplexity: {ppl_result.perplexity:.2f}")
    print(f"Loss: {ppl_result.loss:.4f}")
    print(f"\nLost-in-Middle Analysis:")
    print(f"  Overall: {lim_result.perplexity:.2f}")
    if lim_result.position_perplexities:
        labels = lim_result.metadata.get("position_labels", [])
        for i, ppl in enumerate(lim_result.position_perplexities):
            label = labels[i] if i < len(labels) else f"Pos {i}"
            print(f"  {label}: {ppl:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
