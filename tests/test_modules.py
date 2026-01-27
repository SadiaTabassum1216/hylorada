"""
HyLoRADA Unit Tests

Tests for core modules:
- LoRA / DoRA / HyLoRADA adapters
- Direct Attention Adaptation
- Model integration
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

# Import HyLoRADA modules
import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])

from hylorada.config import HyLoRADAConfig, HyLoRADAPresets
from hylorada.lora import LoRALinear, LoRALayer, apply_lora_to_model, count_lora_params
from hylorada.daa import DirectAttentionAdapter, PositionalDAA
from hylorada.sparse_mlp import TopKGate, SparseAdapter, SparseMLP
from hylorada.s2_attention import ShiftedSparseAttention, get_s2_memory_estimate


# ==================== Fixtures ====================

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture  
def seq_len():
    return 128

@pytest.fixture
def hidden_size():
    return 256

@pytest.fixture
def num_heads():
    return 8

@pytest.fixture
def sample_input(batch_size, seq_len, hidden_size):
    return torch.randn(batch_size, seq_len, hidden_size)


# ==================== Config Tests ====================

class TestConfig:
    """Tests for HyLoRADAConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HyLoRADAConfig()
        assert config.lora_rank == 8
        assert config.daa_enabled == True
        assert config.position_bias_enabled == True  # Unified position bias
        assert config.sparse_enabled == True  # Enabled by default
        assert config.s2_attn_enabled == False  # Disabled by default
        # New simplified defaults
        assert config.sparse_adapter_dim == 64  # Smaller for efficiency
        assert config.sparse_topk_ratio == 0.1
    
    def test_invalid_rank(self):
        """Test that rank must be >= 1."""
        with pytest.raises(ValueError):
            HyLoRADAConfig(lora_rank=0)
    
    def test_presets(self):
        """Test preset configurations."""
        efficient = HyLoRADAPresets.efficient()
        assert efficient.lora_rank == 4
        
        balanced = HyLoRADAPresets.balanced()
        assert balanced.lora_rank == 8  # Default
    
    def test_serialization(self):
        """Test config to/from dict."""
        config = HyLoRADAConfig(lora_rank=16)
        config_dict = config.to_dict()
        restored = HyLoRADAConfig.from_dict(config_dict)
        assert restored.lora_rank == 16


# ==================== LoRA Tests ====================

class TestLoRA:
    """Tests for LoRA implementation."""
    
    def test_lora_linear_shape(self, hidden_size):
        """Test LoRALinear output shapes."""
        in_features = hidden_size
        out_features = hidden_size * 4
        rank = 8
        
        lora = LoRALinear(in_features, out_features, rank=rank)
        
        # Check parameter shapes
        assert lora.lora_A.shape == (rank, in_features)
        assert lora.lora_B.shape == (out_features, rank)
    
    def test_lora_layer_forward(self, sample_input, hidden_size):
        """Test LoRALayer forward pass."""
        base_layer = nn.Linear(hidden_size, hidden_size)
        lora_layer = LoRALayer(base_layer, rank=8)
        
        output = lora_layer(sample_input)
        assert output.shape == sample_input.shape
    
    def test_lora_freezes_base(self, hidden_size):
        """Test that base layer is frozen after wrapping."""
        base_layer = nn.Linear(hidden_size, hidden_size)
        lora_layer = LoRALayer(base_layer, rank=8)
        
        for param in lora_layer.base_layer.parameters():
            assert not param.requires_grad
    
    def test_lora_delta_weight(self, hidden_size):
        """Test LoRA delta weight computation."""
        lora = LoRALinear(hidden_size, hidden_size, rank=8, alpha=16)
        delta = lora.get_delta_weight()
        
        assert delta.shape == (hidden_size, hidden_size)
    
    def test_apply_lora_to_model(self, hidden_size):
        """Test applying LoRA to a simple model."""
        model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Won't find any q_proj/v_proj but should not error
        model, lora_layers = apply_lora_to_model(model, target_modules=("0", "2"))
        assert len(lora_layers) >= 0


# ==================== Unified HyLoRADA Tests ====================

from hylorada.lora import HyLoRADAUnified, UnifiedLayer, PositionBias, apply_unified_to_model


class TestUnifiedHyLoRADA:
    """Tests for the new unified HyLoRADA implementation."""
    
    def test_position_bias_shape(self):
        """Test PositionBias outputs correct shape."""
        pos_bias = PositionBias(num_buckets=64)
        positions = torch.arange(128).unsqueeze(0).expand(2, -1)  # [2, 128]
        
        output = pos_bias(positions)
        assert output.shape == (2, 128)
    
    def test_unified_linear_param_count(self, hidden_size):
        """Test HyLoRADAUnified has expected number of parameters (lightweight mode)."""
        rank = 8
        # Lightweight mode: no magnitude
        adapter = HyLoRADAUnified(hidden_size, hidden_size, rank=rank, use_dora_magnitude=False)
        
        # Expected: lora_A (r*in) + lora_B (out*r) + 1 scalar (position_scale), NO magnitude
        expected = rank * hidden_size + hidden_size * rank + 1
        actual = sum(p.numel() for p in adapter.parameters())
        
        assert actual == expected
    
    def test_unified_linear_param_count_dora(self, hidden_size):
        """Test HyLoRADAUnified with DoRA magnitude has expected params."""
        rank = 8
        # DoRA mode: includes magnitude
        adapter = HyLoRADAUnified(hidden_size, hidden_size, rank=rank, use_dora_magnitude=True)
        
        # Expected: lora_A (r*in) + lora_B (out*r) + magnitude (out) + 1 scalar (position_scale)
        expected = rank * hidden_size + hidden_size * rank + hidden_size + 1
        actual = sum(p.numel() for p in adapter.parameters())
        
        assert actual == expected
    
    def test_unified_layer_forward(self, sample_input, hidden_size):
        """Test UnifiedLayer forward pass."""
        base_layer = nn.Linear(hidden_size, hidden_size)
        unified = UnifiedLayer(base_layer, rank=8)
        
        output = unified(sample_input)
        assert output.shape == sample_input.shape
    
    def test_unified_with_positions(self, sample_input, hidden_size, seq_len, batch_size):
        """Test UnifiedLayer with position input."""
        pos_bias = PositionBias()
        base_layer = nn.Linear(hidden_size, hidden_size)
        unified = UnifiedLayer(base_layer, rank=8, position_bias=pos_bias)
        
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        output = unified(sample_input, positions=positions)
        
        assert output.shape == sample_input.shape
    
    def test_apply_unified_to_model(self, hidden_size):
        """Test applying unified HyLoRADA to a model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
            
            def forward(self, x):
                return self.q_proj(x) + self.v_proj(x)
        
        model = SimpleModel()
        model, layers, pos_bias = apply_unified_to_model(
            model, target_modules=("q_proj", "v_proj"), rank=4
        )
        
        assert len(layers) == 2
        assert pos_bias is not None
        assert sum(p.numel() for p in pos_bias.parameters()) == 64


# ==================== DAA Tests ====================

class TestDAA:
    """Tests for Direct Attention Adaptation."""
    
    def test_daa_initialization(self, num_heads):
        """Test DAA parameter initialization."""
        daa = DirectAttentionAdapter(num_heads, per_head=True)
        
        assert daa.alpha.shape == (num_heads,)
        assert daa.beta.shape == (num_heads,)
        assert torch.allclose(daa.alpha, torch.ones(num_heads))
        assert torch.allclose(daa.beta, torch.zeros(num_heads))
    
    def test_daa_forward(self, batch_size, num_heads, seq_len):
        """Test DAA forward pass."""
        daa = DirectAttentionAdapter(num_heads, per_head=True)
        
        # Attention scores: [batch, heads, seq_q, seq_k]
        attn_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
        
        output = daa(attn_scores)
        assert output.shape == attn_scores.shape
    
    def test_daa_modulation(self, batch_size, num_heads, seq_len):
        """Test that DAA actually modulates attention."""
        daa = DirectAttentionAdapter(num_heads, per_head=True)
        
        # Modify alpha and beta
        daa.alpha.data = torch.full((num_heads,), 0.5)
        daa.beta.data = torch.full((num_heads,), 0.1)
        
        attn_scores = torch.ones(batch_size, num_heads, seq_len, seq_len)
        output = daa(attn_scores)
        
        # Should be 0.5 * 1.0 + 0.1 = 0.6
        expected = torch.full_like(attn_scores, 0.6)
        assert torch.allclose(output, expected)
    
    def test_positional_daa(self, num_heads, seq_len):
        """Test PositionalDAA initialization."""
        pdaa = PositionalDAA(num_heads=num_heads, max_seq_len=seq_len)
        
        assert pdaa.position_bias is not None


# ==================== Sparse MLP Tests ====================

class TestSparseMLP:
    """Tests for Sparse MLP components."""
    
    def test_topk_gate(self, hidden_size):
        """Test TopKGate neuron selection."""
        num_neurons = 128
        topk_ratio = 0.1
        
        gate = TopKGate(hidden_size, num_neurons, topk_ratio)
        x = torch.randn(2, 64, hidden_size)
        
        mask, indices = gate(x, return_indices=True)
        
        # Check that exactly k neurons are selected
        k = int(num_neurons * topk_ratio)
        assert mask.sum().item() == k
        assert len(indices) == k
    
    def test_sparse_adapter_shape(self, sample_input, hidden_size):
        """Test SparseAdapter output shape."""
        adapter = SparseAdapter(hidden_size, adapter_dim=64, topk_ratio=0.1)
        output = adapter(sample_input)
        
        assert output.shape == sample_input.shape
    
    def test_sparse_adapter_residual(self, hidden_size):
        """Test that sparse adapter uses residual connection."""
        adapter = SparseAdapter(hidden_size, adapter_dim=64, topk_ratio=0.1)
        
        # Zero out up projection to check residual
        adapter.up_proj.weight.data.zero_()
        
        x = torch.randn(1, 10, hidden_size)
        output = adapter(x)
        
        # Output should equal input when up_proj is zeroed
        assert torch.allclose(output, x, atol=1e-5)
    
    def test_sparsity_stats(self, hidden_size):
        """Test sparsity statistics."""
        adapter = SparseAdapter(hidden_size, adapter_dim=100, topk_ratio=0.2)
        x = torch.randn(1, 10, hidden_size)
        _ = adapter(x)
        
        stats = adapter.get_sparsity_stats()
        assert stats["num_selected"] == 20
        assert stats["sparsity_ratio"] == 0.8


# ==================== S²-Attn Tests ====================

class TestS2Attention:
    """Tests for Shifted Sparse Attention."""
    
    def test_s2_attn_short_sequence(self, batch_size, num_heads, hidden_size):
        """Test S²-Attn with sequence shorter than group size."""
        head_dim = hidden_size // num_heads
        s2_attn = ShiftedSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            group_size=256,  # Larger than seq_len
        )
        
        seq_len = 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        output, _ = s2_attn(q, k, v)
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    
    def test_s2_attn_long_sequence(self, batch_size, num_heads, hidden_size):
        """Test S²-Attn with sequence longer than group size."""
        head_dim = hidden_size // num_heads
        group_size = 32
        
        s2_attn = ShiftedSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            group_size=group_size,
            layer_idx=0,  # No shift
        )
        
        seq_len = 128
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        output, _ = s2_attn(q, k, v)
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    
    def test_s2_attn_shift_pattern(self, hidden_size, num_heads):
        """Test that odd layers shift and even layers don't."""
        s2_even = ShiftedSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_idx=0,
        )
        s2_odd = ShiftedSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_idx=1,
        )
        
        assert not s2_even.should_shift
        assert s2_odd.should_shift
    
    def test_memory_estimate(self):
        """Test memory estimation utility."""
        estimate = get_s2_memory_estimate(
            seq_len=32768,
            hidden_size=4096,
            num_heads=32,
            num_layers=32,
            group_size=2048,
        )
        
        assert estimate["memory_savings_ratio"] > 1.0
        assert estimate["s2_attention_gb"] < estimate["standard_attention_gb"]


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for the full HyLoRADA system."""
    
    def test_simple_model_integration(self, hidden_size):
        """Test HyLoRADA on a simple transformer-like model."""
        # Create a minimal model
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
            
            def forward(self, x):
                q = self.q_proj(x)
                v = self.v_proj(x)
                return self.mlp(q + v)
        
        model = SimpleTransformer()
        
        # Apply LoRA
        model, lora_layers = apply_lora_to_model(
            model, 
            target_modules=("q_proj", "v_proj"),
            rank=4,
        )
        
        # Verify LoRA was applied
        assert len(lora_layers) >= 2
        
        # Test forward pass
        x = torch.randn(2, 32, hidden_size)
        output = model(x)
        assert output.shape == x.shape
    
    def test_param_counting(self, hidden_size):
        """Test parameter counting."""
        lora = LoRALinear(hidden_size, hidden_size, rank=8)
        
        # Expected: A (8 x hidden_size) + B (hidden_size x 8)
        expected = 8 * hidden_size + hidden_size * 8
        
        actual = sum(p.numel() for p in lora.parameters())
        assert actual == expected


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
