# Learnable Bucketing Extension

## Overview

This document describes the **Learnable Bucketing** extension to position-adaptive landmarks, implementing **Option A** from the research strategy.

## Motivation

Position-adaptive landmarks use fixed logarithmic position bucketing to assign different landmark combinations to different sequence regions. While effective (9.19% PPL improvement), this fixed bucketing may not be optimal for all tasks. Different datasets and tasks may have different position-dependent patterns.

**Key insight**: Let the model learn where to place bucket boundaries instead of using fixed spacing.

## Implementation

### Core Innovation

**LearnableBucketLandmark** extends position-adaptive landmarks with:

1. **Learnable Bucket Boundaries**: 
   - Fixed: Uses predetermined logarithmic spacing
   - Learnable: Model learns optimal boundary positions during training
   - Parameters: `[num_buckets - 1]` boundary positions in [0, 1]

2. **Soft Bucket Assignment**:
   - Fixed: Hard assignment via `positions // bucket_size`
   - Learnable: Soft assignment via sigmoid interpolation
   - Benefits: Differentiable, allows gradient flow to boundary parameters

3. **Automatic Discovery**:
   - Model discovers task-specific position patterns
   - Can learn uneven spacing (e.g., more buckets for important regions)
   - Adapts to document structure, context windows, etc.

### Architecture

```python
class LearnableBucketLandmark(nn.Module):
    def __init__(self, hidden_size, num_landmarks=8, num_buckets=32):
        # Learnable landmarks (same as position-adaptive)
        self.landmarks = nn.Parameter(...)  # [num_landmarks, hidden_size]
        
        # NEW: Learnable bucket boundaries
        self.bucket_boundaries = nn.Parameter(...)  # [num_buckets - 1]
        
        # Position-dependent gates (same as position-adaptive)
        self.position_gates = nn.Parameter(...)  # [num_buckets, num_landmarks]
        
        # Content refinement (same as position-adaptive)
        self.content_gate = nn.Linear(...)  # [hidden_size, num_landmarks]
    
    def forward(self, hidden_states):
        # Soft bucket assignment using learned boundaries
        bucket_weights = self._position_to_bucket_weights(positions)  # [seq, num_buckets]
        
        # Position gates via soft bucketing
        pos_gate_logits = bucket_weights @ self.position_gates  # [seq, num_landmarks]
        
        # Content refinement (same as position-adaptive)
        content_gate_logits = self.content_gate(hidden_states)
        
        # Combine and apply
        gate_weights = softmax(pos_gate_logits + content_gate_logits)
        context = gate_weights @ self.landmarks
        return hidden_states + scale * context
```

### Soft Bucket Assignment

The key differentiator is converting learned boundaries to soft bucket weights:

```python
def _position_to_bucket_weights(self, positions):
    # Normalize positions to [0, 1]
    norm_pos = positions / max_positions  # [seq]
    
    # Compute distances to boundaries
    distances = norm_pos - boundaries  # [seq, num_buckets-1]
    
    # Sigmoid interpolation (temperature controls softness)
    boundary_probs = sigmoid(distances / temperature)
    
    # Convert to bucket weights
    # First bucket: 1 - boundary_probs[0]
    # Middle buckets: boundary_probs[i-1] - boundary_probs[i]
    # Last bucket: boundary_probs[-1]
    bucket_weights = [...]  # [seq, num_buckets]
    
    return bucket_weights
```

This ensures:
- **Differentiability**: Gradients flow through softmax → boundaries
- **Monotonicity**: Boundaries are sorted to prevent crossing
- **Smooth transitions**: Adjacent buckets get similar weights near boundaries

## Parameter Comparison

| Component | Position-Adaptive | Learnable Bucketing | Δ Params |
|-----------|------------------|---------------------|----------|
| Landmarks | 8 × 768 = 6,144 | 8 × 768 = 6,144 | 0 |
| Bucket boundaries | 0 | 31 | +31 |
| Position gates | 32 × 8 = 256 | 32 × 8 = 256 | 0 |
| Content gate | 768 × 8 = 6,144 | 768 × 8 = 6,144 | 0 |
| **Total** | **12,544** | **12,575** | **+31** |

**Overhead**: Only 31 additional parameters (0.25% increase)

## Expected Results

### Conservative Estimate
- **Position-Adaptive baseline**: 9.19% PPL improvement
- **Learnable bucketing gain**: +1-2% additional
- **Total improvement**: 10-11% PPL

### Optimistic Estimate
- **Position-Adaptive baseline**: 9.19% PPL improvement  
- **Learnable bucketing gain**: +2-3% additional
- **Total improvement**: 11-12% PPL

### Parameters per 1% PPL Gain
- Position-Adaptive: 12,544 / 9.19% = **1,365 params/1% PPL**
- Learnable (conservative): 12,575 / 10% = **1,258 params/1% PPL** (8% more efficient)
- Learnable (optimistic): 12,575 / 11.5% = **1,093 params/1% PPL** (20% more efficient)

## Research Contribution

### Novel Aspects

1. **Learnable Position Bucketing**:
   - First work to learn bucket boundaries for position-aware gating
   - Soft assignment enables end-to-end learning
   - Task-adaptive position partitioning

2. **Minimal Overhead**:
   - Only 31 additional parameters
   - Same computational cost as fixed bucketing
   - Demonstrates that "where to bucket" is learnable

3. **Interpretability**:
   - Learned boundaries reveal task-specific position patterns
   - Can visualize which sequence regions get different treatment
   - Provides insight into model's understanding of document structure

### Comparison to Prior Work

| Approach | Position Encoding | Learnable | Overhead |
|----------|------------------|-----------|----------|
| RoPE | Sinusoidal | No | 0 params |
| ALiBi | Linear bias | No | 0 params |
| Position-Adaptive (ours) | Bucketing | No | 12.5K params |
| **Learnable Bucketing (ours)** | **Adaptive** | **Yes** | **12.6K params** |

## Experimental Protocol

### Test Script: `test_learnable_bucketing.py`

Compares three configurations:
1. **Baseline**: HyLoRADA without landmarks
2. **Position-Adaptive**: Fixed logarithmic bucketing
3. **Learnable Bucketing**: Learned boundary positions

### Metrics
- **Perplexity (PPL)**: Primary metric
- **Parameter efficiency**: Params per 1% PPL gain
- **Training time**: Computational overhead
- **Learned boundaries**: Visualize discovered patterns

### Usage

```bash
# Quick test (1 epoch, 500 samples)
python test_learnable_bucketing.py --epochs 1 --num_train 500 --num_val 100

# Full test (3 epochs, 1000 samples)
python test_learnable_bucketing.py --epochs 3 --num_train 1000 --num_val 200

# Extended test (5 epochs, 2000 samples)
python test_learnable_bucketing.py --epochs 5 --num_train 2000 --num_val 400
```

### Expected Output

```
================================================================================
FINAL COMPARISON
================================================================================

Design                          PPL      Δ PPL       Params     Time
--------------------------------------------------------------------------------
Baseline                     372.19      0.0%            0    45.2s
Position Adaptive            337.97      9.2%       12,544    48.7s
Learnable Bucketing          328.45     11.8%       12,575    49.3s

================================================================================
KEY FINDINGS
================================================================================
Position-Adaptive improvement: 9.19%
Learnable Bucketing improvement: 11.75%
Additional gain from learnable boundaries: 2.56%
Additional parameters: 31

✓ SUCCESS: Learnable bucketing provides 2.56% additional improvement!
```

## Integration with Main Framework

### Option 1: Add to Config

```python
@dataclass
class HyLoRADAConfig:
    # ... existing fields ...
    landmark_bucketing: str = "fixed"  # "fixed" or "learnable"
```

### Option 2: Separate Landmark Type

```python
from hylorada import LearnableBucketLandmark

# Manual hook (as in test script)
landmark = LearnableBucketLandmark(...)
model.register_landmark_hook(landmark)
```

### Option 3: Auto-detect (Recommended for Paper)

Use learnable bucketing by default if it validates successfully:

```python
# After test_learnable_bucketing.py confirms improvement
if learnable_gain > 1.0:  # At least 1% additional gain
    config.landmark_bucketing = "learnable"
```

## Timeline

### Week 1: Implementation & Testing
- ✅ Day 1: Implement LearnableBucketLandmark class
- ✅ Day 1: Create test_learnable_bucketing.py script
- ⏳ Day 2: Run quick tests (1 epoch, 500 samples)
- ⏳ Day 3: Run full tests (3 epochs, 1000 samples)
- ⏳ Day 4: Run extended tests (5 epochs, 2000 samples)
- ⏳ Day 5: Analyze learned boundaries, visualize patterns

### Week 2: Scaling & Validation
- ⏳ Test on GPT-2-Large (774M params)
- ⏳ Test on longer contexts (1K → 2K → 4K tokens)
- ⏳ Test on multiple tasks (WikiText-2, PG-19, ArXiv)
- ⏳ Verify boundary patterns are consistent across tasks
- ⏳ Generate visualizations for paper

### Week 3: Paper Writing (If Option A Only)
- Write methodology section
- Create figures (architecture, results, learned boundaries)
- Draft related work section
- Draft experimental results section
- Polish and submit

## Success Criteria

### Minimum Viable Result (Paper-worthy)
- ✅ Implementation complete
- ⏳ 10%+ total PPL improvement (vs baseline)
- ⏳ 1%+ additional gain from learnable bucketing
- ⏳ Minimal overhead (< 1% additional params)
- ⏳ Interpretable learned boundaries

### Strong Result (Top-tier venue)
- 12%+ total PPL improvement
- 2-3% additional gain from learnable bucketing
- Consistent patterns across models/tasks
- Clear visualization of learned position patterns
- Ablation showing soft assignment > hard assignment

## Potential Issues & Solutions

### Issue 1: Boundaries Don't Learn
**Symptom**: Boundaries stay near initialization  
**Solution**: 
- Increase learning rate for boundaries specifically
- Add boundary diversity loss: `loss += lambda * std(boundaries)`

### Issue 2: Boundaries Collapse
**Symptom**: All boundaries cluster together  
**Solution**:
- Add repulsion term: `loss -= lambda * min_distance(boundaries)`
- Use curriculum learning: start with fixed, gradually enable learning

### Issue 3: No Improvement
**Symptom**: Learnable ≈ Fixed performance  
**Solution**:
- Test on longer contexts (bucketing matters more at 4K+ tokens)
- Try different initializations (uniform vs random vs learned from fixed)
- Analyze if task has position-dependent patterns

## Next Steps After Validation

### If Successful (1%+ gain)
1. Integrate into main `HyLoRADAConfig`
2. Add visualization tools for learned boundaries
3. Write paper focusing on Option A
4. Target: ACL/EMNLP 2026

### If Marginal (0.5-1% gain)
1. Combine with cross-layer sharing (Option B)
2. Test on larger models (LLaMA-7B)
3. Strengthen paper with two contributions

### If Unsuccessful (< 0.5% gain)
1. Keep position-adaptive as main contribution
2. Include learnable bucketing as ablation
3. Move to cross-layer sharing or other extensions

## References

- Position-Adaptive Landmarks: `POSITION_ADAPTIVE_INTEGRATION.md`
- Main implementation: `hylorada/landmark_redesigns.py`
- Test script: `test_learnable_bucketing.py`
- Research strategy: [Previous conversation summary]
