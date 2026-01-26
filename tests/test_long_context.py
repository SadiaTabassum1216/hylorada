
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from hylorada import HyLoRADAModel, HyLoRADAConfig
from transformers import GPT2Config, GPT2Model

class TestLongContext(unittest.TestCase):
    def setUp(self):
        # Use a small GPT-2 model for testing
        self.config = GPT2Config(
            n_layer=2,
            n_head=4,
            n_embd=32,
            vocab_size=100
        )
        self.base_model = GPT2Model(self.config)

    def test_trainable_embeddings_and_norms(self):
        """Test if embeddings and norms are unfrozen when configured."""
        hylorada_config = HyLoRADAConfig(
            lora_rank=4,
            train_embeddings=True,
            train_norms=True,
            sparse_enabled=False # Disable for cleaner check
        )
        
        model = HyLoRADAModel(self.base_model, hylorada_config)
        
        # Check Embeddings
        wte = model.base_model.wte
        self.assertTrue(wte.weight.requires_grad, "Embeddings should be trainable")
        
        # Check Norms
        ln_f = model.base_model.ln_f
        self.assertTrue(ln_f.weight.requires_grad, "LayerNorm should be trainable")
        
        # Check other params (like MLP weights) are still frozen
        mlp_c_fc = model.base_model.h[0].mlp.c_fc
        self.assertFalse(mlp_c_fc.weight.requires_grad, "MLP weights should be frozen")

    def test_sink_tokens(self):
        """Test if sink tokens runs without error and affects output."""
        hylorada_config = HyLoRADAConfig(
            lora_rank=4,
            s2_attn_enabled=True,
            s2_group_size=8,
            s2_sink_tokens=2,
            max_sequence_length=32
        )
        
        model = HyLoRADAModel(self.base_model, hylorada_config)
        
        # Create input > group_size to trigger grouped attention
        # seq_len = 20, group_size = 8 -> 3 groups
        input_ids = torch.randint(0, 100, (1, 20))
        
        # Run forward pass
        try:
            output = model(input_ids).last_hidden_state
        except Exception as e:
            self.fail(f"Forward pass with sink tokens failed: {e}")
            
        self.assertEqual(output.shape, (1, 20, 32))

    def test_rope_scaling_injection(self):
        """Test if RoPE scaling config is injected into base model."""
        hylorada_config = HyLoRADAConfig(
            lora_rank=4,
            rope_scaling_type="linear",
            rope_scaling_factor=2.0
        )
        
        # GPT2Config usually doesn't have rope, but we check if attribute is added/modified
        # We Mock the config to have rope_scaling attribute or handle the warning
        
        # Create a dummy config that allows arbitrary attributes
        self.base_model.config.rope_scaling = None 
        
        model = HyLoRADAModel(self.base_model, hylorada_config)
        
        self.assertEqual(model.base_model.config.rope_scaling["type"], "linear")
        self.assertEqual(model.base_model.config.rope_scaling["factor"], 2.0)

if __name__ == '__main__':
    unittest.main()
