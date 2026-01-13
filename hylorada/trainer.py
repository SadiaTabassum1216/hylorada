"""
HyLoRADA Trainer Module

Training utilities for the HyLoRADA fine-tuning framework.
Supports long-context training with memory optimization techniques.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List, Any
import os
import json
from dataclasses import dataclass, field
from tqdm import tqdm
import math

from .config import HyLoRADAConfig
from .model import HyLoRADAModel, get_hylorada_optimizer_groups


@dataclass
class TrainingConfig:
    """Training configuration for HyLoRADA."""
    
    # Basic training params
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Component-specific learning rates
    lr_lora: float = 2e-4
    lr_daa: float = 1e-3
    lr_sparse: float = 2e-4
    
    # Batch settings
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    
    # Memory optimization
    gradient_checkpointing: bool = False  # Disabled by default; can break gradient flow in some models
    mixed_precision: str = "bf16"  # "fp16", "bf16", or "fp32"
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Paths
    output_dir: str = "./output"
    
    # Seed
    seed: int = 42


class HyLoRADATrainer:
    """
    Trainer for HyLoRADA models.
    
    Handles the training loop with support for:
    - Gradient accumulation for large effective batch sizes
    - Mixed precision training
    - Gradient checkpointing for memory efficiency
    - Component-specific learning rates
    - Warmup and scheduling
    
    Args:
        model: HyLoRADAModel to train
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        config: Training configuration
        compute_metrics: Optional function to compute metrics
    """
    
    def __init__(
        self,
        model: HyLoRADAModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        self.compute_metrics = compute_metrics
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup mixed precision
        self.scaler = None
        self.autocast_dtype = torch.float32
        self.use_autocast = False
        
        # Only use mixed precision on CUDA
        if self.device.type == 'cuda':
            if self.config.mixed_precision == "fp16":
                self.scaler = torch.amp.GradScaler('cuda')
                self.autocast_dtype = torch.float16
                self.use_autocast = True
            elif self.config.mixed_precision == "bf16":
                self.autocast_dtype = torch.bfloat16
                self.use_autocast = True
        
        # Setup gradient checkpointing
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.total_steps = self._calculate_total_steps()
        self.scheduler = self._create_scheduler()
        
        # Tracking
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.training_history: List[Dict] = []
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the base model."""
        if hasattr(self.model.base_model, "gradient_checkpointing_enable"):
            self.model.base_model.gradient_checkpointing_enable()
        else:
            print("Warning: Model does not support gradient checkpointing")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with component-specific learning rates."""
        param_groups = get_hylorada_optimizer_groups(
            model=self.model,
            lr_lora=self.config.lr_lora,
            lr_daa=self.config.lr_daa,
            lr_sparse=self.config.lr_sparse,
            weight_decay=self.config.weight_decay,
        )
        
        return torch.optim.AdamW(param_groups)
    
    def _calculate_total_steps(self) -> int:
        """Calculate total training steps."""
        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(self.total_steps - current_step) / 
                float(max(1, self.total_steps - warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Returns:
            Dictionary containing training results and history
        """
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Total steps: {self.total_steps}")
        print(f"Device: {self.device}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = self._train_epoch(epoch)
            
            # Evaluation
            if self.eval_dataloader is not None:
                eval_results = self.evaluate()
                print(f"Epoch {epoch + 1} - Eval Loss: {eval_results['eval_loss']:.4f}")
                
                # Save best model
                if eval_results["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_results["eval_loss"]
                    self._save_checkpoint("best")
            
            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}")
        
        # Save final model
        self._save_checkpoint("final")
        
        return {
            "final_loss": epoch_loss,
            "best_eval_loss": self.best_eval_loss,
            "history": self.training_history,
        }
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=False,
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self._optimization_step()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.optimizer.param_groups[0]["lr"]
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                    })
                    
                    self.training_history.append({
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": lr,
                    })
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / max(num_batches, 1)
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass (with autocast if on GPU)
        if self.use_autocast:
            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def _optimization_step(self):
        """Execute optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_params(),
            self.config.max_grad_norm,
        )
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the eval dataset."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.use_autocast:
                    with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                        outputs = self.model(**batch)
                        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / max(num_batches, 1)
        results = {"eval_loss": avg_loss}
        
        # Compute additional metrics if provided
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(self.model, self.eval_dataloader)
            results.update(metrics)
        
        return results
    
    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save HyLoRADA weights
        self.model.save_hylorada(os.path.join(checkpoint_dir, "hylorada_weights.pt"))
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump({
                "hylorada_config": self.model.config.to_dict(),
                "training_config": {
                    "num_epochs": self.config.num_epochs,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "warmup_ratio": self.config.warmup_ratio,
                    "max_grad_norm": self.config.max_grad_norm,
                },
            }, f, indent=2)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load a training checkpoint."""
        # Load HyLoRADA weights
        self.model.load_hylorada(os.path.join(checkpoint_dir, "hylorada_weights.pt"))
        
        # Load training state
        training_state = torch.load(
            os.path.join(checkpoint_dir, "training_state.pt"),
            map_location=self.device,
        )
        
        self.global_step = training_state["global_step"]
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        self.scheduler.load_state_dict(training_state["scheduler_state"])
        self.best_eval_loss = training_state["best_eval_loss"]
        
        print(f"Loaded checkpoint from {checkpoint_dir} at step {self.global_step}")


def create_long_context_dataloader(
    dataset,
    tokenizer,
    max_length: int = 32768,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader optimized for long-context training.
    
    Args:
        dataset: HuggingFace dataset or similar
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        batch_size: Batch size (typically 1 for long contexts)
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader configured for long-context training
    """
    def collate_fn(batch):
        # Tokenize if needed
        if isinstance(batch[0], str):
            tokenized = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
        elif isinstance(batch[0], dict):
            texts = [item.get("text", item.get("content", "")) for item in batch]
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            tokenized = batch[0]
        
        # Setup labels for causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
