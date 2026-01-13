"""
HyLoRADA Evaluation Module

Provides utilities for evaluating long-context performance:
1. Perplexity on long sequences
2. Lost-in-the-Middle analysis
3. Baseline vs HyLoRADA comparison
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import json
import os


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    perplexity: float
    loss: float
    sequence_length: int
    num_samples: int
    position_perplexities: Optional[List[float]] = None  # For Lost-in-the-Middle
    metadata: Optional[Dict] = None


def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    stride: int = 512,
) -> Tuple[float, float]:
    """
    Compute perplexity using sliding window approach for long sequences.
    
    Args:
        model: The language model
        input_ids: Token IDs [batch=1, seq_len]
        attention_mask: Optional attention mask
        stride: Sliding window stride (overlap = seq_len - stride)
        
    Returns:
        Tuple of (perplexity, average_loss)
    """
    device = next(model.parameters()).device
    max_length = getattr(model.config, 'max_position_embeddings', 2048)
    seq_len = input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_chunk = input_chunk.clone()
        
        # Only compute loss on new tokens (avoid double-counting in overlap)
        target_chunk[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood.item())
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    
    avg_loss = sum(nlls) / seq_len
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def evaluate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 2048,
    batch_size: int = 1,
    show_progress: bool = True,
) -> EvaluationResult:
    """
    Evaluate perplexity on a list of texts.
    
    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        texts: List of text samples
        max_length: Maximum sequence length
        batch_size: Number of samples per batch
        show_progress: Whether to show progress bar
        
    Returns:
        EvaluationResult with perplexity metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_perplexities = []
    all_losses = []
    
    iterator = tqdm(texts, desc="Evaluating") if show_progress else texts
    
    for text in iterator:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        
        if inputs.input_ids.size(1) < 10:  # Skip very short sequences
            continue
        
        ppl, loss = compute_perplexity(model, inputs.input_ids.to(device))
        all_perplexities.append(ppl)
        all_losses.append(loss)
    
    avg_perplexity = np.mean(all_perplexities)
    avg_loss = np.mean(all_losses)
    
    return EvaluationResult(
        perplexity=float(avg_perplexity),
        loss=float(avg_loss),
        sequence_length=max_length,
        num_samples=len(all_perplexities),
    )


def evaluate_lost_in_the_middle(
    model,
    tokenizer,
    texts: List[str],
    num_positions: int = 10,
    max_length: int = 2048,
) -> EvaluationResult:
    """
    Evaluate the "Lost-in-the-Middle" phenomenon.
    
    Measures perplexity at different positions in the sequence to see
    if the model attends better to beginning/end vs middle.
    
    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer
        texts: List of long text samples
        num_positions: Number of position buckets to analyze
        max_length: Maximum sequence length
        
    Returns:
        EvaluationResult with position-wise perplexities
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Collect losses per position bucket
    position_losses = [[] for _ in range(num_positions)]
    
    for text in tqdm(texts, desc="Lost-in-Middle Analysis"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        input_ids = inputs.input_ids.to(device)
        seq_len = input_ids.size(1)
        
        if seq_len < 100:  # Need reasonably long sequences
            continue
        
        # Get per-token losses
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            
            # Compute per-token loss manually
            logits = outputs.logits[:, :-1, :]  # [1, seq-1, vocab]
            labels = input_ids[:, 1:]  # [1, seq-1]
            
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction='none',
            )
        
        # Bucket by position
        bucket_size = (seq_len - 1) // num_positions
        for pos in range(num_positions):
            start = pos * bucket_size
            end = start + bucket_size if pos < num_positions - 1 else seq_len - 1
            bucket_loss = per_token_loss[start:end].mean().item()
            position_losses[pos].append(bucket_loss)
    
    # Compute average perplexity per position
    position_perplexities = [
        float(np.exp(np.mean(losses))) if losses else 0.0
        for losses in position_losses
    ]
    
    return EvaluationResult(
        perplexity=float(np.mean(position_perplexities)),
        loss=float(np.mean([np.mean(l) for l in position_losses if l])),
        sequence_length=max_length,
        num_samples=len(texts),
        position_perplexities=position_perplexities,
        metadata={
            "num_positions": num_positions,
            "position_labels": [f"{i*100//num_positions}%-{(i+1)*100//num_positions}%" 
                               for i in range(num_positions)],
        }
    )


def compare_models(
    baseline_model,
    hylorada_model,
    tokenizer,
    test_texts: List[str],
    max_length: int = 2048,
    run_lost_in_middle: bool = True,
) -> Dict[str, Any]:
    """
    Compare baseline model vs HyLoRADA-enhanced model.
    
    Args:
        baseline_model: Original model without HyLoRADA
        hylorada_model: Model with HyLoRADA adapters
        tokenizer: Shared tokenizer
        test_texts: List of test texts
        max_length: Maximum sequence length
        run_lost_in_middle: Whether to run Lost-in-the-Middle analysis
        
    Returns:
        Dictionary with comparison results
    """
    print("=" * 60)
    print("Model Comparison: Baseline vs HyLoRADA")
    print("=" * 60)
    
    results = {}
    
    # Perplexity comparison
    print("\nEvaluating baseline model...")
    baseline_ppl = evaluate_perplexity(
        baseline_model, tokenizer, test_texts, max_length
    )
    results["baseline_perplexity"] = baseline_ppl.perplexity
    results["baseline_loss"] = baseline_ppl.loss
    
    print(f"Baseline Perplexity: {baseline_ppl.perplexity:.2f}")
    
    print("\nEvaluating HyLoRADA model...")
    hylorada_ppl = evaluate_perplexity(
        hylorada_model, tokenizer, test_texts, max_length
    )
    results["hylorada_perplexity"] = hylorada_ppl.perplexity
    results["hylorada_loss"] = hylorada_ppl.loss
    
    print(f"HyLoRADA Perplexity: {hylorada_ppl.perplexity:.2f}")
    
    # Improvement
    ppl_improvement = (baseline_ppl.perplexity - hylorada_ppl.perplexity) / baseline_ppl.perplexity * 100
    results["perplexity_improvement_percent"] = ppl_improvement
    print(f"\nPerplexity Improvement: {ppl_improvement:.2f}%")
    
    # Lost-in-the-Middle analysis
    if run_lost_in_middle:
        print("\nRunning Lost-in-the-Middle analysis on baseline...")
        baseline_litm = evaluate_lost_in_the_middle(
            baseline_model, tokenizer, test_texts, max_length=max_length
        )
        results["baseline_position_perplexities"] = baseline_litm.position_perplexities
        
        print("Running Lost-in-the-Middle analysis on HyLoRADA...")
        hylorada_litm = evaluate_lost_in_the_middle(
            hylorada_model, tokenizer, test_texts, max_length=max_length
        )
        results["hylorada_position_perplexities"] = hylorada_litm.position_perplexities
        
        # Analyze middle positions
        n = len(baseline_litm.position_perplexities)
        middle_start, middle_end = n // 3, 2 * n // 3
        
        baseline_middle = np.mean(baseline_litm.position_perplexities[middle_start:middle_end])
        hylorada_middle = np.mean(hylorada_litm.position_perplexities[middle_start:middle_end])
        
        middle_improvement = (baseline_middle - hylorada_middle) / baseline_middle * 100
        results["middle_position_improvement_percent"] = middle_improvement
        
        print(f"\nMiddle Position Improvement: {middle_improvement:.2f}%")
        print("(Positive = HyLoRADA better at capturing middle context)")
    
    print("\n" + "=" * 60)
    
    return results


def run_full_evaluation(
    model_name: str,
    hylorada_weights_path: str,
    test_dataset_name: str = "wikitext",
    test_dataset_config: str = "wikitext-2-raw-v1",
    max_length: int = 1024,
    num_samples: int = 100,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.
    
    Args:
        model_name: HuggingFace model name
        hylorada_weights_path: Path to saved HyLoRADA weights
        test_dataset_name: Test dataset name
        test_dataset_config: Test dataset config
        max_length: Maximum sequence length
        num_samples: Number of test samples
        output_path: Where to save results
        
    Returns:
        Full evaluation results
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from hylorada import HyLoRADAConfig, HyLoRADAModel
    
    print("Loading models...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_model.eval()
    
    # Load HyLoRADA model
    hylorada_base = AutoModelForCausalLM.from_pretrained(model_name)
    hylorada_model = HyLoRADAModel(hylorada_base, HyLoRADAConfig())
    hylorada_model.load_hylorada(hylorada_weights_path)
    hylorada_model.eval()
    
    # Load test data
    print(f"Loading test data from {test_dataset_name}...")
    dataset = load_dataset(test_dataset_name, test_dataset_config, split="test")
    
    # Get long texts
    test_texts = []
    for item in dataset:
        text = item.get("text", "")
        if len(text) > 500:  # Only use reasonably long texts
            test_texts.append(text)
        if len(test_texts) >= num_samples:
            break
    
    print(f"Using {len(test_texts)} test samples")
    
    # Run comparison
    results = compare_models(
        baseline_model=baseline_model,
        hylorada_model=hylorada_model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        max_length=max_length,
    )
    
    results["model_name"] = model_name
    results["num_samples"] = len(test_texts)
    results["max_length"] = max_length
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results
