"""
HyLoRADA Training Script

Train a model with HyLoRADA adaptation.

Usage:
    python train_hylorada.py --model Qwen/Qwen2-0.5B --epochs 3
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.trainer import HyLoRADATrainer, TrainingConfig, create_long_context_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train HyLoRADA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print("=" * 60)
    print("HyLoRADA Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Max Length: {args.max_length}, LoRA Rank: {args.lora_rank}")
    print("=" * 60)
    
    # Load tokenizer and model
    print("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    
    # Apply HyLoRADA
    print("[2] Applying HyLoRADA...")
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
    
    # Load data
    print("\n[3] Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50][:args.num_samples]
    print(f"    Using {len(train_texts)} samples")
    
    train_dataloader = create_long_context_dataloader(
        dataset=train_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    
    # Train
    print("\n[4] Training...")
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
        warmup_ratio=0.03,
        logging_steps=50,
        mixed_precision="bf16" if torch.cuda.is_available() else "fp32",
    )
    
    trainer = HyLoRADATrainer(model=model, train_dataloader=train_dataloader, config=train_config)
    results = trainer.train()
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Final loss: {results['final_loss']:.4f}")
    print(f"Weights saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
