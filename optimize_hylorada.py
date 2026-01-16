"""
HyLoRADA Hyperparameter Optimization using Optuna (Bayesian Optimization)

This script searches for the best hyperparameters for HyLoRADA using
Tree-structured Parzen Estimator (TPE) - a form of Bayesian optimization.

Usage:
    python optimize_hylorada.py --n_trials 20 --epochs 2
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import optuna
from optuna.trial import Trial

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hylorada import HyLoRADAModel, HyLoRADAConfig
from hylorada.trainer import HyLoRADATrainer, TrainingConfig


def load_data(max_samples: int = 500):
    """Load WikiText dataset for optimization."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t) > 100][:max_samples]
    
    # Split into train/val
    split_idx = int(len(texts) * 0.8)
    return texts[:split_idx], texts[split_idx:]


def create_model_and_train(
    trial: Trial,
    base_model_name: str,
    train_texts: list,
    val_texts: list,
    epochs: int,
    max_length: int,
) -> float:
    """Create model with trial hyperparameters and train."""
    
    # === HYPERPARAMETER SEARCH SPACE ===
    
    # LoRA rank (higher = more capacity, more params)
    lora_rank = trial.suggest_categorical("lora_rank", [4, 8, 12, 16])
    
    # Alpha ratio (alpha = rank * ratio)
    alpha_ratio = trial.suggest_float("alpha_ratio", 1.5, 3.0)
    
    # LoRA+ learning rate ratio (B learns faster than A)
    lora_plus_ratio = trial.suggest_float("lora_plus_ratio", 5.0, 20.0)
    
    # Base learning rate
    base_lr = trial.suggest_float("base_lr", 1e-5, 5e-4, log=True)
    
    # Dropout
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    
    # Gate initialization (sigmoid input, affects DoRA vs LoRA balance)
    gate_init = trial.suggest_float("gate_init", -1.0, 1.0)
    
    # Residual weight initialization
    residual_init = trial.suggest_float("residual_init", 0.0, 0.5)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Create HyLoRADA config
    config = HyLoRADAConfig(
        lora_rank=lora_rank,
        lora_alpha=lora_rank * alpha_ratio,
        lora_dropout=lora_dropout,
        use_hylorada=True,
        lora_plus_enabled=True,
        lora_plus_ratio=lora_plus_ratio,
        daa_enabled=False,
        sparse_enabled=False,
        budget_lora=1.0,
        budget_daa=0.0,
        budget_sparse=0.0,
    )
    
    # Create model
    model = HyLoRADAModel(base_model, config)
    
    # Adjust gate and residual initialization
    for module in model.modules():
        if hasattr(module, 'magnitude_gate'):
            module.magnitude_gate.data.fill_(gate_init)
        if hasattr(module, 'residual_weight'):
            module.residual_weight.data.fill_(residual_init)
    
    # Training config
    train_config = TrainingConfig(
        learning_rate=base_lr,
        num_epochs=epochs,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=max_length,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )
    
    # Train
    trainer = HyLoRADATrainer(model, train_config, tokenizer)
    
    # Create dataloaders
    train_loader = trainer.create_dataloader(train_texts)
    val_loader = trainer.create_dataloader(val_texts)
    
    # Train and get final loss
    try:
        trainer.train(train_loader, val_loader)
        final_loss = trainer.best_val_loss
    except Exception as e:
        print(f"Training failed: {e}")
        return float('inf')
    
    # Cleanup
    del model, base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return final_loss


def objective(trial: Trial, args) -> float:
    """Optuna objective function."""
    return create_model_and_train(
        trial=trial,
        base_model_name=args.model,
        train_texts=args.train_texts,
        val_texts=args.val_texts,
        epochs=args.epochs,
        max_length=args.max_length,
    )


def main():
    parser = argparse.ArgumentParser(description="HyLoRADA Hyperparameter Optimization")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Base model name")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of optimization trials")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs per trial")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max training samples")
    parser.add_argument("--output", type=str, default="optimization_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HyLoRADA Hyperparameter Optimization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    args.train_texts, args.val_texts = load_data(args.max_samples)
    print(f"Train samples: {len(args.train_texts)}, Val samples: {len(args.val_texts)}")
    
    # Create Optuna study with TPE sampler (Bayesian optimization)
    study = optuna.create_study(
        direction="minimize",  # Minimize validation loss
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    
    # Run optimization
    print("\nStarting optimization...")
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": args.n_trials,
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "all_trials": [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
            }
            for t in study.trials if t.value is not None
        ]
    }
    
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Print recommended config
    print("\n" + "=" * 60)
    print("RECOMMENDED HYLORADA CONFIG")
    print("=" * 60)
    bp = study.best_params
    print(f"""
config = HyLoRADAConfig(
    lora_rank={bp['lora_rank']},
    lora_alpha={bp['lora_rank'] * bp['alpha_ratio']:.1f},
    lora_dropout={bp['lora_dropout']:.3f},
    use_hylorada=True,
    lora_plus_enabled=True,
    lora_plus_ratio={bp['lora_plus_ratio']:.1f},
    daa_enabled=False,
    sparse_enabled=False,
)

# Gate init: {bp['gate_init']:.2f}
# Residual init: {bp['residual_init']:.2f}
# Base LR: {bp['base_lr']:.2e}
""")


if __name__ == "__main__":
    main()
