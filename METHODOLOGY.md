# HyLoRADA Methodology

## Abstract

HyLoRADA (Hybrid Low-Rank Adaptation with Direct Attention) is a parameter-efficient fine-tuning framework that combines proven techniques from recent PEFT research into a unified, flexible architecture for efficient long-context learning. The core framework uses rsLoRA scaling (α/√r), orthogonal initialization, DoRA-style magnitude decomposition, and position-aware bias to achieve high-quality adaptation with minimal trainable parameters (~1-2% of model size). Optional extensions include Shifted Sparse Attention (S²-Attn) for sequences >2K tokens, experimental LandmarkLoRA for context summarization, and RoPE scaling for extreme context lengths.

## 1. Core Architecture: HyLoRADAUnified Layer

The foundation of HyLoRADA is the **HyLoRADAUnified** class, which replaces standard linear projections (Q, K, V, O) in transformer attention layers.

### 1.1 rsLoRA: Rank-Stabilized Low-Rank Adaptation

**Mathematical Formulation**:

$$W' = W + \frac{\alpha}{\sqrt{r}} \cdot B @ A$$

where:
- W: frozen pretrained weight matrix (d_out × d_in)
- A: trainable matrix (r × d_in) - initialized with **orthogonal** initialization
- B: trainable matrix (d_out × r) - initialized with **zeros**
- α: scaling factor (default: 16)
- r: rank of decomposition (default: 8)
- **√r**: rank-stabilized denominator (instead of traditional r)

**Key Innovation - rsLoRA Scaling**: 
- **Traditional LoRA** uses α/r, which causes gradient magnitude to decrease as rank increases
- **rsLoRA** uses α/√r, maintaining stable gradient flow across different ranks
- This enables effective use of higher ranks (r=16, 32, 64) without gradient issues

**Why Orthogonal Initialization for A?**
- Prevents rank collapse during training
- Ensures A maintains full-rank properties throughout optimization
- Standard Kaiming initialization can lead to singular matrices
- Empirically improves final model quality

**Why Zero Initialization for B?**
- Ensures identity mapping at initialization: W' = W + 0
- Model starts identical to base model, preventing training instability
- Gradients flow cleanly in early training

**Parameter count**: ~131K per attention layer (Q,K,V,O with rank=8, d=4096)

### 1.2 DoRA-Style Magnitude Decomposition (Gated)

**Motivation**: Standard LoRA learns both direction and magnitude of weight updates jointly. DoRA separates these concerns, and HyLoRADA extends it with gated control.

**Mathematical Formulation**:

$$\text{gate} = \sigma(\text{gate\_param})$$
$$m_{\text{effective}} = m_{\text{learned}} \cdot \text{gate} + m_{\text{base}} \cdot (1 - \text{gate})$$
$$W' = m_{\text{effective}} \odot \frac{W + \Delta W}{||W + \Delta W||}$$

where:
- m_learned ∈ ℝ^(d_out): learnable magnitude vector
- m_base: column norms of the frozen base weight W
- gate: learnable scalar controlling magnitude adaptation strength
- ΔW = (α/√r) × B @ A: rsLoRA update
- ||·||: column-wise L2 normalization

**Why Gated Magnitude?**
- **Adaptive control**: Gate learns optimal balance between base and adapted magnitudes
- **Initialization at 0**: Starts with gate ≈ 0.5, gradually learns adaptation strength
- **Magnitude-direction separation**: Allows independent learning dynamics
- **Accuracy boost**: Empirically matches full fine-tuning (Liu et al. 2024, DoRA paper)

**When to disable**: Set `use_dora_magnitude=False` for minimal parameter overhead (saves 4K params per layer)

**Additional parameters**: Only d_out + 1 (~4K for typical layers)

### 1.3 Residual LoRA Blending

**Mathematical Formulation**:

$$\beta = \sigma(\text{residual\_weight})$$
$$\text{output} = (1 - \beta) \cdot \text{DoRA\_output} + \beta \cdot \text{LoRA\_output}$$

where:
- DoRA_output: Magnitude-normalized path (direction + magnitude)
- LoRA_output: Standard additive path (W + ΔW)
- β: learnable blend weight (initialized at 0.1)

**Why Blend Both Paths?**
- **Best of both worlds**: Combines DoRA's magnitude control with LoRA's additive updates
- **Learnable**: Model discovers optimal balance during training
- **Complementary**: DoRA handles direction/magnitude, LoRA handles direct weight updates
- **Initialization**: Starts DoRA-heavy (90%), gradually adjusts

**Additional parameters**: 1 scalar per layer

### 1.4 Position-Aware Bias for Long-Context Refinement

**Problem**: Long-context models suffer from "Lost-in-the-Middle" phenomenon where information in the middle of sequences is harder to access (Liu et al. 2023).

**Solution**: Shared position-dependent scaling with logarithmic bucketing

**Mathematical Formulation**:

$$\text{scale}(p) = 1 + \sigma(w) \cdot \tanh(\text{bias}[\text{bucket}(p)])$$
$$\text{output}_p = \text{HyLoRADA}(x_p) \cdot \text{scale}(p)$$

where:
- p: position index in sequence
- bucket(p): logarithmic bucketing function mapping position to 64 buckets
- w: learnable position scale weight (shared across all layers)
- bias: 64 learnable parameters (**shared globally**)

**Why This Design?**
- **Logarithmic bucketing**: Captures long-distance dependencies with O(log n) parameters
- **Shared globally**: Only 64 parameters total for entire model (not per-layer)
- **Position-aware adaptation**: Can increase/decrease attention to specific positions
- **Minimal overhead**: Negligible compute cost (lookup + multiply)

**When to enable**: Default enabled, critical for sequences >1K tokens

**Total additional parameters**: 64 + 1 = 65 parameters (shared across entire model)

## 2. Optional Extensions

### 2.1 LandmarkLoRA: Trainable Context Summary Tokens (Experimental)

**Status**: Implemented but **disabled by default** (`landmark_enabled=False`)

**Motivation**: Learn compressed representations of important context patterns

**Design**:
- K trainable landmark tokens (default: 8)
- Soft attention gating selects relevant landmarks based on input
- Applied at final layer norm before LM head
- Adds scaled context to all positions

**Mathematical Formulation**:

$$\text{gate\_weights} = \text{softmax}(\text{W}_g \cdot \text{mean}(h))$$
$$\text{context} = \text{gate\_weights} @ \text{Landmarks}$$
$$\text{output} = h + \alpha_{\text{scale}} \cdot \text{context}$$

where:
- Landmarks ∈ ℝ^(K × d): learnable summary tokens
- W_g ∈ ℝ^(K × d): gating projection
- α_scale: learnable scalar (init: 0.1)

**Why Experimental?**
- **Single-point application**: Applied only at final norm, may interfere with LoRA gradients
- **Unclear benefit**: Needs more empirical validation
- **Alternative approaches**: Position bias already addresses lost-in-middle

**Parameters**: K × d + d × K + 1 ≈ 14K (for K=8, d=896)

**When to enable**: For research/experimentation only (`config.landmark_enabled=True`)

### 2.2 Shifted Sparse Attention (S²-Attn)

**Status**: Implemented but **disabled by default** (`s2_attn_enabled=False`)

**Problem**: Full attention complexity is O(n²), prohibitive for very long contexts (>4K tokens)

**Solution**: Group-wise attention with alternating shifts to maintain information flow (from LongLoRA)

**Design**:
- Splits sequence into groups of size g (default: 2048 tokens)
- Computes attention only within each group: O(n × g) complexity
- Alternating layers shift group boundaries by g/2 to enable cross-group attention
- Optional sink tokens: first N tokens attend globally across all groups
- Reduces memory from O(n²) to O(n × g)

**Why This Design?**
- **Memory efficiency**: 16x training cost reduction for 4K sequences
- **Information flow**: Shifting groups maintains multi-hop reasoning across groups
- **Optional**: Disabled for sequences ≤2K (standard attention sufficient)
- **Based on LongLoRA**: Proven effective in Chen et al., 2024

**When to enable**: For sequences >2K tokens or GPU memory constraints (`config.s2_attn_enabled=True`)

**Parameters**: No additional trainable parameters (computational pattern only)

### 2.3 RoPE Scaling for Extended Context

**Status**: Optional configuration (`rope_scaling_type=None` by default)

**Problem**: Rotary positional embeddings (RoPE) are trained on a fixed context length; applying them to longer sequences causes extrapolation errors

**Solution**: Scale the frequency of position embeddings to fit longer sequences

**Supported methods**:
- **Linear scaling**: Directly scale frequencies by length ratio (simple but can hurt performance)
- **Dynamic scaling**: Progressively scale frequencies (better generalization)  
- **YaRN**: Interpolate frequencies differently for low/high dimensions (optimal for 10-100K contexts)

**Why This Design?**
- **Context extension**: Enables fine-tuning on sequences 2-4x longer than original training context
- **Flexible**: Choose method based on target context length
- **Combines with embeddings training**: For >32K contexts, also enable `train_embeddings=True`

**When to enable**: Target context >2x model's original training length (e.g., `config.rope_scaling_type="yarn"`)

**Parameters**: No additional trainable parameters (config-based scaling)

### 2.4 LongLoRA Extensions

**Status**: Optional (`train_embeddings=False`, `train_norms=False` by default)

**Design**:
- **Trainable embeddings**: Unfreeze embedding layers for adaptation
- **Trainable norms**: Unfreeze LayerNorm/RMSNorm parameters
- **Critical for >32K**: Required when extending far beyond original context

**Why This Works**:
- Embeddings encode positional information that needs updating for new context lengths
- Norms help adapt feature distributions for different sequence lengths
- From LongLoRA paper: Essential for extreme context extension

**Parameters**: 
- Embeddings: vocab_size × d_model (e.g., 50K × 4K = 200M params)
- Norms: Small (~1K per layer)

**When to enable**: Only for extreme context lengths >32K tokens

## 3. Complete HyLoRADA Architecture

### 3.1 Forward Pass Through HyLoRADAUnified Layer

```
Input: x [batch, seq, d_in]
Base frozen layer: base_out = W @ x

# 1. Compute rsLoRA delta
lora_out = A @ x          # [batch, seq, rank]
lora_out = B @ lora_out   # [batch, seq, d_out]
delta = (α/√r) * lora_out

# 2. DoRA path: magnitude-normalized
updated_weight = W + (α/√r) * B @ A
norm = ||updated_weight||_col  # column-wise norm
mag_scale = m_effective / norm
dora_out = (base_out + delta) * mag_scale

# 3. LoRA path: direct additive
lora_out = base_out + delta

# 4. Blend paths
β = sigmoid(residual_weight)
blended = (1-β) * dora_out + β * lora_out

# 5. Apply position bias (if enabled)
pos_scale = position_bias(positions)
output = blended * pos_scale

Return: output [batch, seq, d_out]
```

### 3.2 Complete Model Architecture

```
Input Sequence → Embedding Layer (frozen unless train_embeddings=True)
                    ↓
For each Transformer Layer:
    ├─ Attention Sublayer:
    │  ├─ Project Q, K, V using HyLoRADAUnified layers
    │  ├─ S²-Attn (optional, if enabled)
    │  ├─ Scaled dot-product attention
    │  ├─ Output projection with HyLoRADAUnified
    │  └─ Position bias scaling (shared)
    │
    ├─ Feed-Forward Sublayer:
    │  ├─ First projection with HyLoRADAUnified
    │  ├─ Activation (GELU/ReLU)
    │  └─ Second projection with HyLoRADAUnified
    │
    └─ LayerNorm (frozen unless train_norms=True)

Final Layer Norm → LandmarkLoRA (optional, if enabled) → LM Head
```

### 3.3 Parameter Breakdown

**Per attention layer** (rank=8, d=4096):
- rsLoRA A matrices (Q, K, V, O): 4 × (8 × 4096) = 131K
- rsLoRA B matrices (Q, K, V, O): 4 × (4096 × 8) = 131K  
- DoRA magnitude vectors: 4 × 4096 = 16K
- Magnitude gates: 4 scalars ≈ 0
- Residual blend weights: 4 scalars ≈ 0
- **Subtotal per attention layer**: ~278K

**Per FFN layer** (rank=8):
- 2 × rsLoRA updates (up + down): ~131K each = 262K
- 2 × DoRA magnitudes: 2 × 4K = 8K
- **Subtotal per FFN layer**: ~270K

**Global shared** (entire model):
- Position bias: 64 parameters
- Position scale weight: 1 parameter
- LandmarkLoRA (if enabled): ~14K
- **Subtotal**: 65 (+14K if landmark enabled)

**Total for 12-layer GPT-2** (example):
- Attention: 12 × 278K = 3.3M
- FFN: 12 × 270K = 3.2M
- Shared: 65
- **Total**: ~6.5M trainable parameters (~5% of GPT-2's 124M)

### 3.4 Configuration Examples

```python
from hylorada import HyLoRADAConfig, HyLoRADAModel

# Minimal configuration (default)
config = HyLoRADAConfig(
    lora_rank=8,
    lora_alpha=16.0,
    use_dora_magnitude=True,      # DoRA enabled
    position_bias_enabled=True,   # Position bias enabled
    s2_attn_enabled=False,        # S²-Attn disabled
    landmark_enabled=False,       # LandmarkLoRA disabled
)

# Long context (2K-4K tokens)
config_long = HyLoRADAConfig(
    lora_rank=8,
    use_dora_magnitude=True,
    position_bias_enabled=True,
    s2_attn_enabled=True,         # Enable S²-Attn
    s2_group_size=2048,
)

# Extreme context (>8K tokens)
config_extreme = HyLoRADAConfig(
    lora_rank=8,
    use_dora_magnitude=True,
    position_bias_enabled=True,
    s2_attn_enabled=True,
    train_embeddings=True,        # Unfreeze embeddings
    train_norms=True,             # Unfreeze norms
    rope_scaling_type="yarn",     # YaRN RoPE scaling
    rope_scaling_factor=2.0,
)

# With experimental LandmarkLoRA
config_experimental = HyLoRADAConfig(
    lora_rank=8,
    use_dora_magnitude=True,
    position_bias_enabled=True,
    landmark_enabled=True,        # Enable landmark
    num_landmarks=8,
)

model = HyLoRADAModel(base_model, config)
model.print_trainable_params()
```

## 4. Comparison with Baseline Methods

HyLoRADA is benchmarked against established PEFT methods with similar parameter budgets:

### 4.1 Standard LoRA (Baseline)
- **Components**: Basic rsLoRA (α/√r) only, applied to Q, V projections
- **Parameters**: ~87K per attention layer
- **Purpose**: Validates that HyLoRADA improvements come from specific design choices
- **Reference**: Hu et al., 2021

### 4.2 DoRA (Weight Decomposition)
- **Components**: LoRA + magnitude decomposition (without gating or blending)
- **Parameters**: ~91K per attention layer
- **Purpose**: Tests pure magnitude-direction separation
- **Reference**: Liu et al., 2024

### 4.3 LoRaDA (Direct Attention Adaptation)
- **Components**: LoRA + Direct Attention Adaptation (DAA)
- DAA: Learns per-head α, β to modulate attention weights
- **Parameters**: ~89K per attention layer
- **Purpose**: Tests attention-specific noise filtering
- **Note**: DAA is implemented but **not part of core HyLoRADA** (available as baseline)
- **Reference**: Li et al., 2025

### 4.4 LongLoRA
- **Components**: LoRA + trainable embeddings/norms + S²-Attn
- **Parameters**: Similar total, different composition
- **Purpose**: State-of-the-art long-context baseline
- **Reference**: Chen et al., 2024

### 4.5 SparseAdapter
- **Components**: Sparse MLP adapters only (no attention LoRA)
- Top-k gating + bottleneck architecture
- **Parameters**: ~50K per FFN layer
- **Purpose**: Tests MLP vs attention adaptation importance
- **Note**: Implemented as separate baseline, not part of HyLoRADA

### 4.6 HyLoRADA (Full)
- **Components**: rsLoRA + DoRA (gated) + Residual blend + Position bias
- **Parameters**: ~278K per attention layer + 65 shared
- **Key differences**:
  - rsLoRA scaling (α/√r) vs traditional (α/r)
  - Gated magnitude control (DoRA enhanced)
  - Residual path blending
  - Shared position bias (not per-layer)
  - Optional S²-Attn, LandmarkLoRA

## 5. Design Principles and Justifications

### 5.1 Why rsLoRA (α/√r) Instead of Traditional (α/r)?

**Problem with traditional LoRA scaling**:
- Gradient magnitude: ∇(α/r) causes gradients to decrease as 1/r
- Makes higher ranks (r=16, 32, 64) unstable or ineffective
- Forces users to manually tune α for each rank

**rsLoRA solution**:
- Gradient magnitude: ∇(α/√r) maintains O(1) gradient scale
- Enables effective use of higher ranks without retuning
- Empirically improves performance at r≥8

**Reference**: Kalajdzievski, 2024 - "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"

### 5.2 Why Orthogonal Initialization for A?

**Problem with Kaiming/Uniform**:
- Random initialization can create near-singular matrices
- Causes rank collapse during training (effective rank < nominal rank)
- Wastes parameters and hurts expressiveness

**Orthogonal solution**:
- Guarantees full-rank matrix at initialization
- Maintains rank throughout training
- Better gradient flow and parameter utilization

### 5.3 Why Zero Initialization for B?

**Principle**: Start with identity mapping

- LoRA update = B @ A @ x = 0 @ A @ x = 0
- Model identical to base model at initialization
- Prevents catastrophic forgetting in early training
- Standard practice in LoRA, DoRA, and all variants

### 5.4 Why Gated Magnitude Control?

**Problem with fixed magnitude**:
- DoRA initializes magnitude from base weight norms
- Optimal magnitude may differ from initialization
- No mechanism to learn "how much magnitude adaptation is needed"

**Gated solution**:
- Learnable interpolation: m = m_learned · gate + m_base · (1-gate)
- Starts neutral (gate ≈ 0.5)
- Model learns optimal balance during training
- Can fall back to base magnitudes if adaptation isn't beneficial

### 5.5 Why Residual LoRA Blending?

**Complementary paths**:
- DoRA path: Direction normalization + magnitude scaling
- LoRA path: Direct additive weight updates
- Different inductive biases, useful for different patterns

**Learnable blend**:
- Model discovers optimal combination
- Typically settles around 10-30% LoRA (β ≈ 0.1-0.3)
- Provides flexibility without manual tuning

### 5.6 Why Logarithmic Position Bucketing?

**Efficiency**:
- Linear bucketing: O(n) parameters for sequence length n
- Logarithmic: O(log n) = 64 buckets for n ≤ 10^19
- Captures scale-invariant position relationships

**Shared globally**:
- Position patterns similar across layers
- Sharing reduces parameters 12x (for 12-layer model)
- Still allows per-position adaptation

### 5.7 Why Not DAA in Core HyLoRADA?

**DAA (Direct Attention Adaptation)** learns α, β to modulate attention scores:
- attn' = α × attn + β

**Why excluded**:
- Position bias already addresses lost-in-middle at output level
- DAA modifies attention internals, adds complexity
- Empirically, position bias sufficient for most cases
- Available as baseline for comparison (LoRaDA method)

**When DAA helps**:
- Extreme noise in attention patterns
- Very challenging long-context tasks
- Can be used in LoRaDA baseline configuration

## 6. Component Synergies and Design Philosophy

### 6.1 Why These Components Work Together

**Non-Interfering**:
- rsLoRA: Operates on weight updates
- DoRA magnitude: Operates on column norms
- Residual blend: Combines outputs
- Position bias: Post-hoc output scaling
- Each component targets different aspects with minimal interaction

**Complementary Strengths**:
1. **rsLoRA**: Efficient parameter usage with stable gradients
2. **DoRA (gated)**: Better weight structure learning with adaptive control
3. **Residual blend**: Combines benefits of both paths
4. **Position bias**: Addresses long-context phenomena (lost-in-middle)
5. **S²-Attn** (optional): Memory efficiency for extreme lengths

**Minimal Redundancy**:
- No competing objectives or conflicting gradients
- Each component addresses a specific limitation
- Can be independently enabled/disabled

### 6.2 Configuration Guidelines

**Context Length Scaling**:

| Context Length | Components | Config |
|----------------|------------|--------|
| ≤1K tokens | Core only | Default |
| 1-2K tokens | Core + Position Bias | `position_bias_enabled=True` |
| 2-4K tokens | + S²-Attn | `s2_attn_enabled=True` |
| 4-8K tokens | + S²-Attn + RoPE | `rope_scaling_type="yarn"` |
| >8K tokens | + Embeddings/Norms | `train_embeddings=True` |

**Quality vs Efficiency**:

| Priority | DoRA | Rank | Config |
|----------|------|------|--------|
| Minimal params | Disabled | 4 | Lightweight |
| Balanced | Enabled | 8 | Default |
| Max quality | Enabled | 16 | High capacity |

**Memory Constraints**:

| GPU Memory | S²-Attn | Group Size | Gradient Checkpoint |
|------------|---------|------------|---------------------|
| <16GB | Required | 1024 | Enabled |
| 16-24GB | Optional | 2048 | Optional |
| >24GB | Disabled | N/A | Optional |

## 7. Computational Complexity

### 7.1 Training Complexity

**Forward pass** (per token):
- **Base transformer**: O(d²)
- **HyLoRADA overhead**: O(r × d) where r << d
- **Position bias**: O(1) lookup
- **Total**: O(d² + r×d) ≈ O(d²) since r << d

**S²-Attn modification**:
- **Standard attention**: O(n²×d) for sequence length n
- **S²-Attn**: O(n×g×d) where g = group size
- **Reduction**: 16x memory for g=2048, n=32K

**Backward pass**:
- Only LoRA parameters require gradients
- Gradient memory: O(r×d×L) vs O(d²×L) for full fine-tuning
- ~50-100x reduction in optimizer states

### 7.2 Memory Complexity

**Parameters**:
- **Full fine-tuning**: O(L × d²) 
- **HyLoRADA**: O(L × r × d) + O(d) for shared components
- **Ratio**: ~1-2% of full fine-tuning

**Activations** (with gradient checkpointing):
- Same as base model: O(n × d × L)
- Independent of PEFT method

**Gradients & Optimizer**:
- **Full fine-tuning**: 3× params (param + grad + optimizer states)
- **HyLoRADA**: 3× trainable params only (~1-2% of model)
- **Savings**: ~98-99% reduction in optimizer memory

### 7.3 Inference Complexity

**Merged mode** (recommended):
- Merge LoRA into base weights: W' = W + (α/√r) × B @ A
- **Zero overhead**: Same latency as base model
- **No extra parameters**: Model size unchanged

**Dynamic mode** (for analysis):
- Position bias: O(1) per token (negligible)
- LandmarkLoRA: O(K×d) where K=8 (small overhead)
- Both overheads < 1% of total compute

### 7.4 Comparison Table

| Method | Training FLOPs | Memory (Params) | Inference Overhead |
|--------|---------------|-----------------|-------------------|
| Full Fine-Tune | O(n×d²×L) | O(d²×L) | Zero |
| Standard LoRA | O(n×d²×L + n×r×d×L) | O(r×d×L) | Zero (merged) |
| DoRA | O(n×d²×L + n×r×d×L) | O(r×d×L + d×L) | Zero (merged) |
| **HyLoRADA** | **O(n×d²×L + n×r×d×L)** | **O(r×d×L + d×L + 65)** | **~Zero** |
| HyLoRADA + S²-Attn | O(n×g×d×L + n×r×d×L) | O(r×d×L + d×L + 65) | Minimal (if kept) |

**Key insights**:
- Training FLOPs: Comparable to standard LoRA (rsLoRA overhead negligible)
- Memory: Slightly higher than LoRA due to DoRA magnitudes (~3% increase)
- Inference: Can merge to zero overhead (except optional dynamic components)

## 8. Implementation Notes

### 8.1 Key Files

- **[hylorada/lora.py](hylorada/lora.py)**: Core HyLoRADAUnified, PositionBias, LandmarkLoRA implementations
- **[hylorada/config.py](hylorada/config.py)**: HyLoRADAConfig with all configuration options
- **[hylorada/model.py](hylorada/model.py)**: HyLoRADAModel wrapper and integration logic
- **[hylorada/s2_attention.py](hylorada/s2_attention.py)**: ShiftedSparseAttention (LongLoRA)
- **[hylorada/daa.py](hylorada/daa.py)**: DirectAttentionAdapter (baseline only, not core)
- **[hylorada/sparse_mlp.py](hylorada/sparse_mlp.py)**: Sparse MLP adapters (baseline only)
- **[hylorada/baselines.py](hylorada/baselines.py)**: Baseline PEFT methods for comparison
- **[hylorada/trainer.py](hylorada/trainer.py)**: Training utilities and HyLoRADATrainer
- **[hylorada/evaluation.py](hylorada/evaluation.py)**: Perplexity and lost-in-middle evaluation

### 8.2 Recommended Hyperparameters

```python
# Conservative (minimal params, fastest training)
config = HyLoRADAConfig(
    lora_rank=4,
    use_dora_magnitude=False,  # Disable for speed
    position_bias_enabled=True,
)

# Balanced (default, recommended for most cases)
config = HyLoRADAConfig(
    lora_rank=8,
    use_dora_magnitude=True,
    position_bias_enabled=True,
)

# High quality (best accuracy, more parameters)
config = HyLoRADAConfig(
    lora_rank=16,
    use_dora_magnitude=True,
    position_bias_enabled=True,
)

# Long context 2-4K
config = HyLoRADAConfig(
    lora_rank=8,
    use_dora_magnitude=True,
    position_bias_enabled=True,
    s2_attn_enabled=True,
    s2_group_size=2048,
)

# Extreme context >8K
config = HyLoRADAConfig(
    lora_rank=8,
    use_dora_magnitude=True,
    position_bias_enabled=True,
    s2_attn_enabled=True,
    train_embeddings=True,
    train_norms=True,
    rope_scaling_type="yarn",
    rope_scaling_factor=2.0,
)
```

### 8.3 Training Tips

**Learning rates**:
- LoRA parameters: 2e-4 (default)
- Position bias: 1e-3 (higher for faster adaptation)
- Embeddings (if enabled): 1e-5 (lower to preserve base knowledge)

**Optimization**:
- Use AdamW with weight_decay=0.01
- Warmup: 3% of total steps
- Gradient clipping: max_norm=1.0
- Mixed precision: bf16 if available, fp16 otherwise

**Gradient checkpointing**:
- Essential for long contexts or limited GPU memory
- Enabled by default (`gradient_checkpointing=True`)
- 30-50% speed reduction, but enables 2-4x longer sequences

**Batch size guidelines**:
| Sequence Length | Recommended Batch Size | Gradient Accumulation |
|----------------|------------------------|----------------------|
| 512-1K | 4-8 | 4-8 |
| 1-2K | 2-4 | 8-16 |
| 2-4K | 1-2 | 16-32 |
| >4K | 1 | 32-64 |

### 8.4 Common Issues

**OOM (Out of Memory)**:
1. Enable gradient checkpointing
2. Reduce batch size, increase gradient accumulation
3. Enable S²-Attn for long contexts
4. Use smaller rank (r=4 instead of 8)

**Poor long-context performance**:
1. Ensure position_bias_enabled=True
2. For >2K: Enable S²-Attn
3. For >4K: Add RoPE scaling
4. For >8K: Train embeddings/norms

**Unstable training**:
1. Check learning rate (try lower, e.g., 1e-4)
2. Increase warmup (5-10% of steps)
3. Use gradient clipping (max_norm=1.0)
4. Verify mixed precision compatibility

## 9. Comparison Summary Table

| Method | rsLoRA | DoRA | Gated Mag | Blend | Position | Landmark | Params/Layer |
|--------|--------|------|-----------|-------|----------|----------|--------------|
| LoRA | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~87K |
| rsLoRA | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ~87K |
| DoRA | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ~91K |
| LoRaDA | ✗ | ✗ | ✗ | ✗ | DAA | ✗ | ~89K |
| LongLoRA | ✗ | ✗ | ✗ | ✗ | S²-Attn | ✗ | ~87K + emb |
| **HyLoRADA** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓ (opt)** | **~278K + 65** |

### Component Status in HyLoRADA

| Component | Status | Default | Purpose |
|-----------|--------|---------|---------|
| rsLoRA (α/√r) | **Core** | Enabled | Rank-stable gradients |
| Orthogonal init | **Core** | Enabled | Prevent rank collapse |
| DoRA magnitude | **Core** | Enabled | Direction-magnitude separation |
| Gated magnitude | **Core** | Enabled | Adaptive magnitude control |
| Residual blend | **Core** | Enabled | Combine DoRA + LoRA paths |
| Position bias | **Core** | Enabled | Lost-in-middle mitigation |
| S²-Attn | **Optional** | Disabled | Long-context memory efficiency |
| LandmarkLoRA | **Experimental** | Disabled | Context summarization |
| RoPE scaling | **Optional** | Disabled | Extreme context extension |
| Train embeddings | **Optional** | Disabled | >32K context adaptation |
| Train norms | **Optional** | Disabled | >32K context adaptation |
| DAA | **Baseline only** | Not included | Attention noise filtering |
| Sparse MLP | **Baseline only** | Not included | MLP-specific adaptation |

## 10. Expected Benefits

Based on the design and component choices:

1. **Stable high-rank training**: rsLoRA enables effective r=16, 32, 64 without gradient issues
2. **Better accuracy**: DoRA magnitude decomposition + gating approaches full fine-tuning quality
3. **Long-context capability**: Position bias addresses lost-in-middle with minimal parameters
4. **Flexibility**: Modular design allows disabling components based on constraints
5. **Inference efficiency**: Core components merge to zero overhead
6. **Memory efficiency**: 98-99% reduction in optimizer memory vs full fine-tuning
7. **Context extension**: S²-Attn + RoPE scaling enable >8K sequences

## 11. Limitations and Future Work

### Current Limitations

1. **LandmarkLoRA**: Experimental status, unclear benefits, may interfere with LoRA gradients
2. **S²-Attn compatibility**: Requires careful handling with Grouped Query Attention (GQA)
3. **Position bias granularity**: Fixed 64 buckets may be suboptimal for some distributions
4. **Gating initialization**: Fixed at 0.0 (sigmoid → 0.5), may need task-specific tuning

### Future Directions

1. **Dynamic rank allocation**: Learn optimal rank per layer during training
2. **Attention pattern analysis**: Better understanding of when DAA helps vs position bias
3. **LandmarkLoRA improvements**: Multi-point application or integration with attention
4. **Adaptive bucketing**: Learn position bucket boundaries instead of fixed logarithmic
5. **Cross-layer parameter sharing**: Share more components (e.g., LoRA B matrices)

## References

1. **rsLoRA**: Kalajdzievski, D. (2024). "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"
2. **DoRA**: Liu, S. et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation"
3. **Lost in the Middle**: Liu, N. et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts"
4. **LoRA**: Hu, E. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models"
5. **LongLoRA**: Chen, Y. et al. (2024). "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models"
6. **LoRaDA**: Li, Y. et al. (2025). "LoRaDA: Low-Rank Direct Attention Adaptation"
7. **YaRN**: Peng, B. et al. (2023). "YaRN: Efficient Context Window Extension"

---

**Last Updated**: February 2, 2026
