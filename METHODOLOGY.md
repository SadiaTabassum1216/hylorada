# HyLoRADA: Context‑Length Adaptive Parameter‑Efficient Fine‑Tuning

## Abstract
Long‑context fine‑tuning (4K–32K tokens) faces lost‑in‑the‑middle effects, quadratic attention cost, and position extrapolation failures. We introduce **HyLoRADA**, a context‑length adaptive PEFT framework that activates long‑context components only when sequence length justifies their overhead. Short‑context ablations (512 tokens) show that long‑context components degrade performance by 3–13%, motivating adaptive gating. HyLoRADA keeps rsLoRA always on and conditionally enables position bias, landmarks, sparse attention, and RoPE scaling. This design targets parameter efficiency across 512→8K+ contexts while preserving short‑context quality.

---

## 1. Introduction
Long‑context deployment is increasingly common in legal, scientific, and code settings. However, extending models beyond their pretraining context exposes three persistent issues: (i) **lost‑in‑the‑middle** recall degradation, (ii) **O(n²)** attention cost, and (iii) **position extrapolation** failures. Prior PEFT methods often enable long‑context modules uniformly, which we find harms short‑context performance.

**Contributions.**  
(1) We formalize a **context‑adaptive** PEFT design that gates long‑context components by sequence length.  
(2) We provide short‑context ablations showing consistent degradation from long‑context modules at 512 tokens.  
(3) We propose a practical configuration protocol for 512→8K+ scaling.

---

## 2. Methodology

### 2.1 rsLoRA Core
HyLoRADA builds on rank‑stabilized LoRA (rsLoRA):
$$W' = W_{\text{frozen}} + \frac{\alpha}{\sqrt{r}} B A$$
where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ (orthogonal init) and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ (zero init). The $\alpha/\sqrt{r}$ scaling stabilizes gradients across ranks.

### 2.2 Context‑Adaptive Components

**Position Bias (≥2K).**  
A shared, logarithmically bucketed position bias counteracts lost‑in‑the‑middle:
$$s(p) = 1 + \sigma(w)\tanh(b[\text{bucket}(p)])$$
$$\text{output}_p = \text{rsLoRA}(x_p)\cdot s(p)$$
(64 shared biases + 1 scale).

**Position‑Adaptive Landmarks (≥2K).**  
A small set of learnable landmarks summarizes long contexts:
$$g = \text{softmax}(W_g \cdot \text{mean}(h))$$
$$c = g^\top L,\quad \text{output} = h + \alpha_s c$$
(K=8 landmarks; ~12.5K params).

**Shifted Sparse Attention (≥4K, optional).**  
Group‑wise attention with alternating shifts reduces attention to O(n·g).

**RoPE Scaling (>1K).**  
Linear/dynamic/YaRN scaling extends positional encoding beyond the base context.

**Trainable Embeddings & Norms (≥2K/≥4K).**  
Position embeddings and layer norms are unfrozen to adapt to longer contexts.

### 2.3 Adaptive Configuration
```python
is_long = max_length >= 2048
config = HyLoRADAConfig(
    lora_rank=8, lora_alpha=16,
    position_bias_enabled=is_long,
    landmark_enabled=is_long,
    s2_attn_enabled=(max_length >= 4096 and args.s2_attn),
    rope_scaling_type="linear" if max_length > 1024 else None,
    rope_scaling_factor=max_length / 1024,
    train_embeddings=args.train_embeddings,
    train_norms=args.train_norms,
)
```

---

## 3. Experimental Setup

**Datasets.** WikiText‑2 (short‑context ablation), WikiText‑103 (2K–8K), MultiPL‑E Python (code).  
**Base model.** GPT‑2 (124M), extended via RoPE scaling + position embedding extension.  
**Training.** AdamW, lr 2e‑4 (5e‑4 ablations), batch 4 (1 long), grad acc 16–32, epochs 1–3.  
**Metrics.** Perplexity, lost‑in‑the‑middle accuracy, parameters per 1% PPL improvement.  
**Baselines.** Baseline, LoRA, DoRA, LoRA‑DA, LongLoRA, SparseAdapter, HyLoRADA.

---

## 4. Ablation Design
**Component isolation (512 tokens).** Baseline → rsLoRA → +Position Bias → +Landmarks → +Learnable Bucket → Bias+Landmarks.  
**Context sweep.** 512 / 2048 / 4096 / 8192 tokens: LoRA vs HyLoRADA.  
**Statistics.** Single seed (42); multi‑seed planned.

---

## 5. Results (Short‑Context)

| Configuration | PPL | Δ vs rsLoRA | Params |
|--------------|-----|-------------|--------|
| Baseline | 69.00 | – | 0 |
| rsLoRA | **57.40** | baseline | 811K |
| + Position Bias | 59.43 | −3.54% ❌ | 811K |
| + Landmarks | 60.38 | −5.19% ❌ | 824K |
| + Learnable Bucket | 64.69 | −12.69% ❌ | 811K |
| Bias + Landmarks | 59.06 | −2.89% ❌ | 824K |

**DoRA check.** LoRA + DoRA degrades (59.40 vs 56.11 PPL). We set `use_dora_magnitude=False`.

---

## 6. Discussion
Short‑context ablations show long‑context modules add noise or compression loss. For ≥2K tokens, position bias and landmarks are expected to counter lost‑in‑the‑middle and capacity limits, while S²‑Attn reduces memory for ≥4K. HyLoRADA’s gating avoids the short‑context penalty while preserving long‑context capability.

---

## 7. Limitations & Future Work
We report single‑seed results and only GPT‑2. Long‑context validation (2K–8K) is ongoing. Future work includes learned activation thresholds, multi‑seed confidence intervals, and larger models.

---

## 8. Reproducibility
```bash
python test_ablation_proper.py --num_train 500 --num_test 100 --epochs 1
python run_benchmark.py --dataset longbench --max_length 4096 \
  --methods lora hylorada --s2_attn --train_embeddings --train_norms \
  --rope_scaling_type linear --rope_scaling_factor 4.0 --epochs 3
```

---

## References
1. Hu et al., 2021 — LoRA  
2. Liu et al., 2023 — Lost in the Middle  
3. Liu et al., 2024 — DoRA  
4. Chen et al., 2024 — LongLoRA  
5. Merity et al., 2016 — WikiText  
6. Radford et al., 2019 — GPT‑2  
7. Cassano et al., 2022 — MultiPL‑E  

**Last Updated**: February 2, 2026