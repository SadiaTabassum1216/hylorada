# HyLoRADA Methodology

## Abstract

HyLoRADA is a parameter-efficient fine-tuning framework combining orthogonal initialization, DoRA magnitude decomposition, and position-aware scaling for efficient long-context learning. The framework integrates complementary components—standard LoRA layers with learnable magnitude adaptation, position-aware bias for attention refinement, and optional extensions like Shifted Sparse Attention (S²-Attn) and RoPE scaling—all designed to work together as a unified system for cost-effective long-context adaptation.

## 1. Core Architecture: Unified LoRA Layer

The foundation of HyLoRADA is the **UnifiedLayer**, which replaces standard linear projections in transformer attention heads.

### 1.1 Base Low-Rank Adaptation (LoRA)

**Mathematical Formulation**:

$$W' = W + \frac{\alpha}{\sqrt{r}} \cdot B @ A$$

where:
- W: frozen pretrained weight matrix (d_out × d_in)
- A: trainable matrix (r × d_in) - initialized with orthogonal initialization
- B: trainable matrix (d_out × r) - initialized with zeros
- α: scaling factor (typically 16)
- r: rank of decomposition (typically 8)
- √r: rank-stabilized denominator (prevents gradient collapse at higher ranks)

**Why This Design?**
- **Low rank**: Exploits the observation that fine-tuning updates occupy a low-dimensional subspace
- **Orthogonal A initialization**: Prevents rank collapse by ensuring A maintains full-rank properties during training
- **√r scaling**: Maintains stable gradient magnitudes across different rank values (Kalajdzievski 2024), enabling flexible rank choices

**Parameter efficiency**: ~87K parameters per LoRA layer (rank=8, d=4096)

### 1.2 DoRA: Weight-Decomposed Magnitude Normalization

**Motivation**: Standard LoRA learns both direction and magnitude jointly. DoRA separates these concerns.

**Mathematical Formulation**:

$$W' = m \odot \text{norm}(W + \Delta W)$$

where:
- m ∈ ℝ^(d_out): learnable magnitude vector
- ΔW = (α/√r) × B @ A: LoRA update
- norm(·): column-wise normalization
- ⊙: element-wise multiplication

**Why This Design?**
- **Magnitude-direction separation**: Magnitude m and direction norm(W + ΔW) have different learning dynamics
- **Accuracy matching**: Empirically matches full fine-tuning performance (Liu et al. 2024)
- **Structured updates**: Maintains the structure of pretrained weights better than pure LoRA
- **Minimal overhead**: Only d_out additional parameters (~4K for typical layers)

**When to use**: Enable for high-quality adaptation when model accuracy is critical

### 1.3 Position-Aware Bias for Attention Refinement

**Problem**: Long-context models suffer from "Lost-in-the-Middle" phenomenon where information in the middle of long contexts is harder to access (Liu et al. 2023).

**Solution**: Learn position-dependent scaling factors with minimal parameters

**Mathematical Formulation**:

$$\text{scale}(p) = 1 + \sigma(w) \cdot \tanh(\text{bias}[\text{bucket}(p)])$$

where:
- p: position index in sequence
- bucket(p): logarithmic bucketing function mapping position to 64 buckets
- w: learnable scaling weight
- σ: sigmoid activation
- bias: 64 shared learnable parameters

**Why This Design?**
- **Logarithmic bucketing**: Captures long-distance dependencies while keeping parameter count minimal
- **Per-position scaling**: Allows the model to learn which positions need attention adjustment
- **Shared across layers**: 64 parameters total for the entire model
- **Position-aware**: Can increase or decrease attention to specific positions based on their utility

**Practical benefit**: Empirically improves performance on long-context benchmarks with negligible overhead

## 2. Extended Components

### 2.1 Shifted Sparse Attention (S²-Attn)

**Problem**: Full attention complexity is O(n²), prohibitive for very long contexts (>4K tokens)

**Solution**: Group-wise attention with alternating shifts to maintain information flow

**Design**:
- Splits sequence into groups of size g (e.g., 2048 tokens)
- Computes attention only within each group: O(n × g) complexity
- Alternating layers shift group boundaries to enable cross-group attention
- Reduces memory from O(n²) to O(n × g)

**Why This Design?**
- **Memory efficiency**: 16x training cost reduction for 4K sequences
- **Information flow**: Shifting groups maintains multi-hop reasoning across groups
- **Optional**: Can be disabled for sequences ≤2K (not needed for typical contexts)
- **Based on LongLoRA**: Proven effective in Chen et al., 2024

**When to use**: Enable for sequences >2K tokens or when GPU memory is constrained

### 2.2 RoPE Scaling for Extended Context

**Problem**: Rotary positional embeddings (RoPE) are trained on a fixed context length; applying them to longer sequences extrapolates poorly

**Solution**: Scale the frequency of position embeddings to fit longer sequences

**Supported methods**:
- **Linear scaling**: Directly scale frequencies by length ratio (simple but can hurt performance)
- **Dynamic scaling**: Progressively scale frequencies (better generalization)
- **YaRN**: Interpolate frequencies in high/low dimensions differently (optimal for 10-100K contexts)

**Why This Design?**
- **Context extension**: Enables fine-tuning on sequences 2-4x longer than original training context
- **Flexible**: Choose method based on target context length
- **Combines with embeddings training**: For >32K contexts, also enable train_embeddings

**When to use**: When target context >2x the model's original training length

### 2.3 Direct Attention Adaptation (DAA)

**Problem**: Attention distributions in long contexts can be noisy, with attention scattered across irrelevant positions

**Solution**: Learn position-independent attention modulation

**Design**:
- Learns per-head scaling (α) and bias (β): attn' = α × attn + β
- Allows filtering of irrelevant positions
- Only 2 parameters per attention head

**Why This Design?**
- **Noise filtering**: Can downweight attention to irrelevant positions
- **Minimal overhead**: 2 parameters per head vs. full attention relearning
- **Per-head control**: Different heads can learn different filtering strategies
- **Complementary to position bias**: Position bias scales output; DAA modulates attention

**When to use**: For very challenging long-context tasks where noise filtering is needed

### 2.4 Sparse MLP Adapters (Optional)

**Problem**: MLP layers in transformers are large and mostly task-agnostic

**Solution**: Learn sparse update patterns activating only k% of neurons

**Design**:
- Top-k gating mechanism selects most relevant neurons
- Bottleneck adapter architecture: d → m → d (where m = d × ratio)
- Straight-through estimator for gradient flow through discrete selection

**Why This Design?**
- **Selective adaptation**: Focus learning on neurons relevant to the task
- **Memory efficiency**: Reduces backward pass computation
- **Optional**: Can be disabled if efficiency isn't critical
- **Compatible**: Works alongside LoRA in attention layers

**When to use**: When parameter budget is extremely limited or for multi-task learning

## 3. Combined Architecture: HyLoRADA Unified

### 3.1 Complete Forward Pass

```
Input Sequence → Embedding Layer
                    ↓
For each Transformer Layer:
    ├─ Attention Sublayer:
    │  ├─ Project: Q, K, V using UnifiedLoRA (rsLoRA + DoRA)
    │  ├─ Position Bias scales outputs by position
    │  ├─ S²-Attn (optional) groups attention computation
    │  └─ Output projection with UnifiedLoRA
    │
    ├─ Feed-Forward Sublayer:
    │  ├─ First projection with LoRA
    │  ├─ Activation (GELU/ReLU)
    │  ├─ Optional: Sparse MLP adapter
    │  └─ Second projection with LoRA
    │
    └─ LayerNorm (frozen)

Output → LM Head (frozen)
```

### 3.2 Parameter Breakdown

**Per attention layer** (rank=8, d_in=4096, d_out=4096):
- LoRA A matrices (Q, K, V, O): 4 × (8 × 4096) = 131K
- LoRA B matrices (Q, K, V, O): 4 × (4096 × 8) = 131K  
- DoRA magnitude vectors: 4 × 4096 = 16K
- Position bias (shared): 64
- **Subtotal**: ~278K per attention layer

**Per FFN layer**:
- 2 × LoRA updates: ~131K
- Optional sparse adapter: ~50K
- **Subtotal**: ~131-181K per FFN layer

**Total for 12-layer GPT-2**: ~4.9M trainable parameters (~1.2% of model)

### 3.3 Configuration Example

```python
from hylorada import HyLoRADAConfig, HyLoRADAModel

config = HyLoRADAConfig(
    # Core LoRA settings
    lora_rank=8,
    lora_alpha=16.0,
    lora_dropout=0.05,
    
    # DoRA: Magnitude decomposition
    use_dora_magnitude=True,
    
    # Position-aware scaling
    position_bias_enabled=True,
    position_num_buckets=64,
    
    # Long-context extensions
    s2_attn_enabled=False,        # Enable for >2K sequences
    s2_group_size=2048,
    
    # Embeddings (for >32K context)
    train_embeddings=False,
    train_norms=False,
    
    # RoPE scaling
    rope_scaling_type=None,        # "yarn" for >4K
    rope_scaling_factor=1.0,
)

model = HyLoRADAModel(base_model, config)
print(model.count_trainable_params())  # ~5M for 12-layer model
```

## 4. Baseline Methods for Comparison

HyLoRADA is compared against proven methods with similar parameter budgets:

### 4.1 Standard LoRA
- **What**: Only LoRA in attention layers
- **Parameters**: ~87K per layer
- **Why baseline**: Validates that improvements come from specific HyLoRADA components

### 4.2 LoRaDA (LoRA + Direct Attention Adaptation)
- **What**: LoRA + per-head attention modulation
- **Parameters**: ~89K per layer
- **Why baseline**: Tests attention-specific adaptation

### 4.3 LongLoRA
- **What**: LoRA + trainable embeddings/norms + S²-Attn
- **Parameters**: Similar to HyLoRADA but different composition
- **Why baseline**: State-of-the-art for long context at small parameter budgets

### 4.4 SparseAdapter
- **What**: Only sparse MLP updates, no attention LoRA
- **Parameters**: ~50K per layer
- **Why baseline**: Tests if attention or MLP adaptation is more important

## 5. Design Principles and Justifications

### 5.1 Orthogonal Initialization of LoRA A
**Principle**: Prevent rank collapse during training
- Standard uniform initialization can lead to singular A matrix
- Orthogonal initialization maintains full-rank property throughout training
- Empirically improves final model quality

### 5.2 Zero Initialization of LoRA B  
**Principle**: Identity mapping at initialization
- LoRA update starts at 0, so model behaves like base model initially
- Gradients flow cleanly in early training
- Prevents training instability

### 5.3 √r Instead of r Scaling
**Principle**: Stable gradient magnitudes across ranks
- Without √r: ∇α/r causes gradient magnitude to decrease with rank
- With √r: ∇α/√r maintains consistent gradient scale
- Enables using higher ranks (r=32, 64) for better expressiveness

### 5.4 Logarithmic Position Bucketing  
**Principle**: Efficient long-distance encoding with minimal parameters
- Logarithmic bucketing captures scale-invariant position relationships
- 64 buckets sufficient for up to 10^19 token sequences
- Only 64 total parameters vs. O(n) for dense buckets

### 5.5 Per-Head Learning in DoRA
**Principle**: Different weights may need different magnitude learning
- All attention heads project to same output dimension
- But they represent different representation subspaces
- Per-weight magnitude learning allows fine-grained adaptation

## 6. Why These Components Together?

### Complementary Strengths:
1. **LoRA**: Efficient, proven, low parameter count
2. **DoRA**: Improves accuracy by separating magnitude learning
3. **Position Bias**: Fixes a specific long-context problem (lost-in-middle)
4. **S²-Attn**: Necessary for very long sequences (>2K)
5. **RoPE Scaling**: Extends positional embedding range without retraining

### Minimal Interference:
- Each component operates on different aspects (weights, magnitudes, positions, attention structure)
- No competing gradients or conflicting objectives
- Can be independently enabled/disabled based on context length

### Flexibility:
- **Small contexts (≤1K)**: Use core LoRA + DoRA
- **Long contexts (1-4K)**: Add Position Bias
- **Very long contexts (>4K)**: Add S²-Attn + RoPE scaling
- **High accuracy needed**: Invest in DoRA magnitude

## 7. Computational Complexity

### Training Complexity:
- **Forward pass**: O(n × d²) (same as standard transformer)
- **LoRA overhead**: O(n × r × d) (negligible, r << d)
- **Position bias**: O(n) (lookup table)
- **S²-Attn overhead**: O(n × g) where g = group size (major savings vs O(n²))

### Memory Complexity:
- **Parameters**: O(L × r × d) where L = layers (vs O(L × d²) for full fine-tuning)
- **Activations**: O(n × d) (unchanged)
- **Gradients**: O(L × r × d) (only stored for LoRA/DoRA parameters)

### Inference Complexity (after merging):
- **Zero overhead**: LoRA can be merged into base weights
- **No additional parameters**: Model size unchanged
- **Same latency**: Inference identical to base model

## 8. Implementation Notes

### Key Files:
- [hylorada/lora.py](hylorada/lora.py): Core UnifiedLayer, LoRA, DoRA implementation
- [hylorada/config.py](hylorada/config.py): HyLoRADAConfig with all parameters
- [hylorada/model.py](hylorada/model.py): HyLoRADAModel wrapper and integration
- [hylorada/s2_attention.py](hylorada/s2_attention.py): Shifted Sparse Attention
- [hylorada/daa.py](hylorada/daa.py): Direct Attention Adaptation
- [hylorada/sparse_mlp.py](hylorada/sparse_mlp.py): Sparse adapter for MLP layers
- [hylorada/evaluation.py](hylorada/evaluation.py): Evaluation metrics and benchmarking

### Recommended Hyperparameters:
```python
# Conservative (smallest parameter budget)
config = HyLoRADAConfig(lora_rank=4, use_dora_magnitude=False)

# Balanced (default)
config = HyLoRADAConfig(lora_rank=8, use_dora_magnitude=True)

# Aggressive (best quality)
config = HyLoRADAConfig(lora_rank=16, use_dora_magnitude=True)

# For long context (>2K)
config = HyLoRADAConfig(
    lora_rank=8, 
    use_dora_magnitude=True,
    s2_attn_enabled=True,
    position_bias_enabled=True,
)

# For very long context (>8K)
config = HyLoRADAConfig(
    lora_rank=8,
    use_dora_magnitude=True,
    s2_attn_enabled=True,
    position_bias_enabled=True,
    train_embeddings=True,
    rope_scaling_type="yarn",
)

| Method | Scaling | Magnitude | Position | Landmark | Params/Layer |
|--------|---------|-----------|----------|----------|--------------|
| LoRA | α/r | - | - | ~87K |
| rsLoRA | α/√r | - | - | ~87K |
| DoRA | α/r | ✓ | - | - | ~91K |
| **HyLoRADA** | **α/√r** | **✓ (Gated)** | **✓** | **✓** | **~105K** |

## 5. Expected Benefits

1. **Better high-rank performance**: rsLoRA enables effective use of r=16, 32, 64
2. **Improved accuracy**: DoRA magnitude decomposition approaches full fine-tuning
3. **Position awareness**: Mitigates lost-in-middle effects with minimal overhead

## 6. Time Complexity Analysis
    
| Method | Training Complexity (per token) | Inference Overhead (Merged) | Inference Overhead (Dynamic) |
| :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | $O(d_{in} \cdot d_{out})$ | **Zero** | - |
| **Sparse Adapter** | $O(r_{ad} \cdot (d_{in} + d_{out}))$ | **High** (Cannot merge) | - |
| **LongLoRA** | Same as LoRA | **Zero** | Low (S²-Attn overhead) |
| **LoRA** | $O(r \cdot (d_{in} + d_{out}))$ | **Zero** | - |
| **DoRA** | $O(r \cdot (d_{in} + d_{out})) + O(d_{out})$ | **Zero** | - |
| **HyLoRADA** | $O(r \cdot (d_{in} + d_{out})) + O(d_{out})$ | **Zero** (Linear components) | Low (Position/Landmark) |

### 6.1 Training
- **Full Fine-Tuning**: Most expensive. Updates all parameters ($d \times d$), requiring massive optimizer memory.
- **LoRA / LongLoRA**: Efficient. Updates only low-rank matrices ($r \ll d$). LongLoRA further reduces self-attention complexity from $O(L^2)$ to $O(L \cdot G)$ via S²-Attn.
- **Sparse Adapters**: Efficient updates, but introduces parameter inefficiency due to separate adapter modules.
- **DoRA**: Adds slight overhead for weight normalization (calculating column norms).
- **HyLoRADA**: Hybrid blend maintains efficient graph size roughly equal to DoRA.

### 6.2 Inference
- **Merged Weights (Zero Overhead)**: FFT, LoRA, LongLoRA, DoRA, and HyLoRADA (linear parts) can all represent the final model as a single weight matrix $W' = W + \Delta W$.
- **Non-Mergeable (High Overhead)**: **Sparse Adapters** contain non-linearities (ReLU/GELU) between down/up projections, preventing merging. This adds latency $O(r \cdot d)$ to *every* forward pass.
- **Dynamic Components (Low Overhead)**:
    - **Position Bias (HyLoRADA)**: $O(1)$ scalar op per token.
    - **LandmarkLoRA**: $O(K \cdot d)$ for $K$ landmarks.
    - **S²-Attn (LongLoRA)**: Requires shifting/masking operations, adding minor overhead vs standard FlashAttention.
    
**Conclusion**: HyLoRADA provides advanced capabilities (long-context handling, stability) with virtually no inference latency penalty over standard models.

## References

1. Kalajdzievski, D. (2024). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.
2. Liu, S. et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation.
3. Liu, N. et al. (2023). Lost in the Middle: How Language Models Use Long Contexts.
4. Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
