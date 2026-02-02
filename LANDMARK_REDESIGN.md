# LandmarkLoRA Redesign Guide

## Problem with Original Design

The original `LandmarkLoRA` has a fundamental architectural issue:

**Single-point application at final layer norm**
- Applied once at the very end, just before LM head
- Weak gradient signal to landmarks (passes through entire model)
- No hierarchical learning (can't capture layer-specific patterns)
- "Interferes with LoRA gradients" per code comments

## Three Improved Designs

### 1. Per-Layer Landmarks ⭐ **RECOMMENDED**

**Where**: Applied at each transformer layer's FFN output (before residual)

**How it works**:
```python
for each transformer layer:
    ffn_out = FFN(hidden_states)
    landmark_context = select_landmarks(ffn_out)  # Soft attention
    ffn_out = ffn_out + scale * landmark_context
    hidden_states = hidden_states + ffn_out  # Residual
```

**Advantages**:
- ✅ Better gradient flow (landmarks closer to each layer)
- ✅ Hierarchical abstractions (early = syntax, late = semantics)
- ✅ Proven pattern (similar to adapters, LoRA)
- ✅ Simple to implement

**Parameters**: `num_layers × num_landmarks × hidden_size`
- Example: 12 layers × 4 landmarks × 4096 = ~200K params

**When to use**: 
- Default choice for testing landmarks
- Works with any sequence length
- Good for heterogeneous tasks

---

### 2. Attention-Integrated Landmarks

**Where**: Injected as additional K/V pairs in attention mechanism

**How it works**:
```python
# In attention computation
K = [landmark_keys, regular_keys]  # Prepend landmarks
V = [landmark_values, regular_values]

attention = softmax(Q @ K^T) @ V
# Now queries can attend to landmarks!
```

**Advantages**:
- ✅ Direct influence on attention patterns
- ✅ Acts as "memory slots" for important context
- ✅ Similar to proven methods (prefix tuning, P-tuning)
- ✅ Naturally integrates with attention

**Parameters**: `num_layers × num_heads × num_landmarks × head_dim`
- Example: 12 layers × 12 heads × 4 landmarks × 64 = ~37K params

**Challenges**:
- ⚠️ Requires modifying attention forward pass
- ⚠️ May conflict with S²-Attn (grouped attention)
- ⚠️ More invasive integration

**When to use**:
- When attention patterns are known to be important
- For retrieval-style tasks
- If you need interpretability (can visualize attention to landmarks)

---

### 3. Position-Adaptive Landmarks

**Where**: Single-point at final norm, but with position-aware selection

**How it works**:
```python
# Instead of global mean pooling:
for each position in sequence:
    position_bias = position_gates[bucket(pos)]
    content_bias = gate_network(hidden[pos])
    landmark_weights = softmax(position_bias + content_bias)
    context[pos] = landmark_weights @ landmarks

output = hidden_states + scale * context
```

**Advantages**:
- ✅ Different positions access different landmarks
- ✅ Better for long contexts with heterogeneous info
- ✅ Combines position and content awareness
- ✅ Single-point application (minimal invasiveness)

**Parameters**: `num_buckets × num_landmarks + num_landmarks × hidden_size + hidden_size × num_landmarks`
- Example: 32 × 8 + 8 × 4096 + 4096 × 8 = ~66K params

**When to use**:
- Long context tasks (>1K tokens)
- When different regions need different abstractions
- As improved version of original design

---

## Experimental Protocol

### Quick Test (Recommended First)
```bash
# Test all designs on small dataset
python test_landmarks.py \
    --model openai-community/gpt2 \
    --dataset wikitext \
    --max_length 512 \
    --epochs 2 \
    --num_train 500 \
    --num_landmarks 4 \
    --designs baseline original per_layer position_adaptive
```

**Expected time**: ~30 minutes on single GPU

### Full Evaluation
```bash
# Per-layer landmarks (best candidate)
python test_landmarks.py \
    --model openai-community/gpt2 \
    --dataset wikitext \
    --max_length 1024 \
    --epochs 3 \
    --num_train 2000 \
    --num_landmarks 8 \
    --designs baseline per_layer
```

### What to Look For

**Success metrics**:
1. **Perplexity improvement**: >2% better than baseline
2. **Lost-in-middle improvement**: >5% better (more important)
3. **Parameter efficiency**: <50K params per 1% PPL gain
4. **Training stability**: No NaN losses, smooth convergence

**Red flags**:
- ❌ Worse than baseline (landmarks hurt performance)
- ❌ Unstable training (large loss spikes)
- ❌ No improvement on lost-in-middle (main use case)

---

## Recommendation Matrix

| Use Case | Recommended Design | Why |
|----------|-------------------|-----|
| **General fine-tuning** | Per-Layer | Best gradient flow, proven pattern |
| **Long context (>2K)** | Position-Adaptive | Position-aware selection crucial |
| **Retrieval tasks** | Attention-Integrated | Direct attention influence |
| **Limited compute** | Position-Adaptive | Fewer params than per-layer |
| **Research/novelty** | Per-Layer or Position-Adaptive | Cleaner to analyze |

---

## Integration Example

```python
from hylorada import HyLoRADAConfig, HyLoRADAModel
from hylorada import PerLayerLandmark, apply_per_layer_landmarks

# Option 1: Use config (once per-layer is validated)
config = HyLoRADAConfig(
    lora_rank=8,
    landmark_enabled=True,  # Will use per-layer by default
    landmark_design="per_layer",  # New field
    num_landmarks=4,
)

model = HyLoRADAModel(base_model, config)

# Option 2: Manual application
config = HyLoRADAConfig(lora_rank=8, landmark_enabled=False)
model = HyLoRADAModel(base_model, config)

landmarks = apply_per_layer_landmarks(
    model.base_model,
    hidden_size=model.hidden_size,
    num_landmarks=4,
)
```

---

## Next Steps

1. **Run quick test**: Validate all three designs (~30 min)
2. **Analyze results**: Which design(s) improve lost-in-middle?
3. **Full evaluation**: Best design on larger dataset
4. **Decision**:
   - If improvement >5%: Integrate into core (replace original)
   - If improvement 2-5%: Keep as optional feature
   - If improvement <2%: Mark as experimental or remove

5. **Update config**: Add `landmark_design` parameter
6. **Update docs**: Document when/how to use landmarks

---

## Expected Outcomes

**Optimistic**: Per-layer shows 5-10% improvement on lost-in-middle, becomes core feature

**Realistic**: One design shows 2-5% improvement, kept as optional for long-context tasks

**Pessimistic**: No significant improvement, remove landmark feature entirely and focus on proven components (position bias is already working)

The test script will give you clear data to make this decision!
