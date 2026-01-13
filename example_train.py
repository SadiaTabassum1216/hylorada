"""
HyLoRADA Example Training Script

Demonstrates how to apply HyLoRADA to a language model and train it
on a long-context dataset.

Usage:
    python example_train.py --model_name meta-llama/Llama-2-7b-hf \
                            --dataset_name your-dataset \
                            --output_dir ./output
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada.trainer import HyLoRADATrainer, TrainingConfig, create_long_context_dataloader
from hylorada.config import HyLoRADAPresets


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with HyLoRADA")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model_name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (e.g., Llama-2)",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    
    # HyLoRADA arguments
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["efficient", "balanced", "high_accuracy", "long_context_128k"],
        help="HyLoRADA preset configuration",
    )
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--daa_enabled", action="store_true", help="Enable DAA")
    parser.add_argument("--sparse_enabled", action="store_true", help="Enable Sparse MLP")
    parser.add_argument("--s2_group_size", type=int, default=None, help="S²-Attn group size")
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--local_only", action="store_true", help="Use only local files")
    
    return parser.parse_args()


def get_hylorada_config(args) -> HyLoRADAConfig:
    """Create HyLoRADA config from arguments."""
    # Start with preset
    preset_map = {
        "efficient": HyLoRADAPresets.efficient,
        "balanced": HyLoRADAPresets.balanced,
        "high_accuracy": HyLoRADAPresets.high_accuracy,
        "long_context_128k": HyLoRADAPresets.long_context_128k,
    }
    config = preset_map[args.preset]()
    
    # Override with explicit arguments
    if args.lora_rank is not None:
        config.lora_rank = args.lora_rank
    if args.s2_group_size is not None:
        config.s2_group_size = args.s2_group_size
    if args.daa_enabled:
        config.daa_enabled = True
    if args.sparse_enabled:
        config.sparse_enabled = True
    
    config.max_sequence_length = args.max_length
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 60)
    print("HyLoRADA Training Script")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name,
        local_files_only=args.local_only,
        trust_remote_code=True,
        token=args.token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"Loading model from {args.model_name}...")
    
    # Determine device and dtype
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch_dtype = torch.bfloat16
        device_map = "auto"
    else:
        torch_dtype = torch.float32
        device_map = None  # Load on CPU
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        local_files_only=args.local_only,
        trust_remote_code=True,
        device_map=device_map,
        token=args.token,
    )
    
    # Create HyLoRADA config
    hylorada_config = get_hylorada_config(args)
    print(f"\nHyLoRADA Configuration:")
    print(f"  Preset: {args.preset}")
    print(f"  LoRA rank: {hylorada_config.lora_rank}")
    print(f"  DAA enabled: {hylorada_config.daa_enabled}")
    print(f"  Sparse MLP enabled: {hylorada_config.sparse_enabled}")
    print(f"  S²-Attn group size: {hylorada_config.s2_group_size}")
    
    # Apply HyLoRADA
    print("\nApplying HyLoRADA adaptations...")
    model = HyLoRADAModel(base_model, hylorada_config)
    model.print_trainable_params()
    
    # Load dataset
    print(f"\nLoading dataset {args.dataset_name}...")
    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split="train",
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using a dummy dataset for demonstration...")
        dataset = [{"text": "This is a sample text for demonstration. " * 100}] * 100
    
    # Create dataloaders
    train_dataloader = create_long_context_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    
    # Training config (disable checkpointing/bf16 on CPU)
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.output_dir,
        seed=args.seed,
        mixed_precision="bf16" if has_cuda else "fp32",
        gradient_checkpointing=has_cuda,  # Only enabled on GPU
    )
    
    # Create trainer
    trainer = HyLoRADATrainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config,
    )
    
    # Print memory estimate
    mem_estimate = model.get_memory_estimate(args.max_length)
    print(f"\nMemory Estimate for {args.max_length} tokens:")
    print(f"  Standard attention: {mem_estimate['standard_attention_gb']:.2f} GB")
    print(f"  S²-Attn: {mem_estimate['s2_attention_gb']:.2f} GB")
    print(f"  Savings: {mem_estimate['memory_savings_ratio']:.1f}x")
    
    # Train
    print("\nStarting training...")
    results = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
