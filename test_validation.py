"""Test input validation and error handling."""

import sys
import torch
from hylorada import HyLoRADAConfig, LandmarkLoRA, HyLoRADAUnified

def test_config_validation():
    """Test HyLoRADAConfig validation."""
    print("Testing HyLoRADAConfig validation...")
    
    # Test valid config
    try:
        config = HyLoRADAConfig(lora_rank=8, num_landmarks=8)
        print("✓ Valid config accepted")
    except Exception as e:
        print(f"✗ Valid config rejected: {e}")
        return False
    
    # Test invalid rank
    try:
        config = HyLoRADAConfig(lora_rank=0)
        print("✗ Invalid rank accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid rank rejected: {e}")
    
    # Test invalid landmarks
    try:
        config = HyLoRADAConfig(num_landmarks=-1)
        print("✗ Invalid num_landmarks accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid num_landmarks rejected: {e}")
    
    # Test invalid dropout
    try:
        config = HyLoRADAConfig(lora_dropout=1.5)
        print("✗ Invalid dropout accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid dropout rejected: {e}")
    
    # Test invalid RoPE scaling
    try:
        config = HyLoRADAConfig(rope_scaling_type="invalid")
        print("✗ Invalid rope_scaling_type accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid rope_scaling_type rejected: {e}")
    
    return True


def test_landmark_validation():
    """Test LandmarkLoRA validation."""
    print("\nTesting LandmarkLoRA validation...")
    
    # Test valid landmark
    try:
        lm = LandmarkLoRA(hidden_size=768, num_landmarks=8)
        print("✓ Valid landmark accepted")
    except Exception as e:
        print(f"✗ Valid landmark rejected: {e}")
        return False
    
    # Test invalid hidden_size
    try:
        lm = LandmarkLoRA(hidden_size=0, num_landmarks=8)
        print("✗ Invalid hidden_size accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid hidden_size rejected: {e}")
    
    # Test invalid num_landmarks
    try:
        lm = LandmarkLoRA(hidden_size=768, num_landmarks=0)
        print("✗ Invalid num_landmarks accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid num_landmarks rejected: {e}")
    
    # Test invalid dropout
    try:
        lm = LandmarkLoRA(hidden_size=768, num_landmarks=8, dropout=2.0)
        print("✗ Invalid dropout accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid dropout rejected: {e}")
    
    return True


def test_unified_validation():
    """Test HyLoRADAUnified validation."""
    print("\nTesting HyLoRADAUnified validation...")
    
    # Test valid unified
    try:
        unified = HyLoRADAUnified(in_features=768, out_features=768, rank=8)
        print("✓ Valid HyLoRADAUnified accepted")
    except Exception as e:
        print(f"✗ Valid HyLoRADAUnified rejected: {e}")
        return False
    
    # Test invalid rank (zero)
    try:
        unified = HyLoRADAUnified(in_features=768, out_features=768, rank=0)
        print("✗ Invalid rank=0 accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid rank=0 rejected: {e}")
    
    # Test invalid rank (exceeds dimensions)
    try:
        unified = HyLoRADAUnified(in_features=768, out_features=768, rank=1000)
        print("✗ Invalid rank > dimensions accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid rank > dimensions rejected: {e}")
    
    # Test invalid alpha
    try:
        unified = HyLoRADAUnified(in_features=768, out_features=768, rank=8, alpha=-1.0)
        print("✗ Invalid alpha accepted (should fail)")
        return False
    except ValueError as e:
        print(f"✓ Invalid alpha rejected: {e}")
    
    return True


def test_simplified_implementation():
    """Test that unvalidated components are removed."""
    print("\nTesting simplified implementation...")
    
    unified = HyLoRADAUnified(in_features=768, out_features=768, rank=8)
    
    # Check that residual_weight is removed
    if hasattr(unified, 'residual_weight'):
        print("✗ residual_weight still exists (should be removed)")
        return False
    else:
        print("✓ residual_weight removed")
    
    # Check that magnitude_gate is removed
    if hasattr(unified, 'magnitude_gate'):
        print("✗ magnitude_gate still exists (should be removed)")
        return False
    else:
        print("✓ magnitude_gate removed")
    
    # Check that DoRA magnitude is optional
    unified_no_dora = HyLoRADAUnified(
        in_features=768, out_features=768, rank=8, use_dora_magnitude=False
    )
    if unified_no_dora.magnitude is not None:
        print("✗ magnitude exists when use_dora_magnitude=False")
        return False
    else:
        print("✓ DoRA magnitude properly optional")
    
    return True


def main():
    """Run all validation tests."""
    print("="*60)
    print("HyLoRADA Validation Tests")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_config_validation()
    all_passed &= test_landmark_validation()
    all_passed &= test_unified_validation()
    all_passed &= test_simplified_implementation()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All validation tests PASSED")
        print("="*60)
        return 0
    else:
        print("❌ Some validation tests FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
