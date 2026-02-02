"""
LandmarkLoRA Redesign Experiments

Three improved architectures for LandmarkLoRA:
1. Per-Layer Landmarks: Apply at each transformer layer
2. Attention-Integrated: Inject as additional K/V pairs
3. Position-Adaptive: Context-aware landmark selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PerLayerLandmark(nn.Module):
    """
    Per-Layer Landmark: Apply landmarks at each transformer layer output.
    
    Instead of single-point application at final norm, this applies landmarks
    after each layer's FFN, before residual connection. This provides:
    - Better gradient flow to landmarks
    - Hierarchical abstractions (early = syntax, late = semantics)
    - More targeted adaptation per layer
    
    Args:
        hidden_size: Model hidden dimension
        num_landmarks: Number of learnable context summaries
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_landmarks: int = 4,  # Fewer per layer
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_landmarks = num_landmarks
        
        # Learnable landmark tokens
        self.landmarks = nn.Parameter(torch.randn(num_landmarks, hidden_size) * 0.02)
        
        # Gate: project hidden states to landmark weights
        self.gate = nn.Linear(hidden_size, num_landmarks, bias=False)
        
        # Learnable scale (starts small)
        self.scale = nn.Parameter(torch.tensor(0.05))
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply per-layer landmark adaptation.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
            
        Returns:
            Enhanced hidden states: h + scale * landmark_context
        """
        # Position-aware gating: use max pooling instead of mean
        # This focuses on salient positions
        max_repr = hidden_states.max(dim=1)[0]  # [batch, hidden_size]
        
        # Compute landmark attention weights
        gate_logits = self.gate(max_repr)  # [batch, num_landmarks]
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [batch, num_landmarks]
        
        # Weighted landmark context
        context = gate_weights @ self.landmarks  # [batch, hidden_size]
        context = self.dropout(context)
        
        # Add to all positions with small scale
        output = hidden_states + self.scale * context.unsqueeze(1)
        
        return output
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_landmarks={self.num_landmarks}, scale={self.scale.item():.4f}"


class AttentionIntegratedLandmark(nn.Module):
    """
    Attention-Integrated Landmark: Inject landmarks as additional K/V pairs.
    
    Similar to prefix tuning, landmarks are added as additional keys and values
    in the attention mechanism. This allows them to:
    - Directly influence attention patterns
    - Act as "memory slots" for important context
    - Be attended to by all query positions
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        num_landmarks: Number of landmark tokens
        head_dim: Dimension per head (default: hidden_size // num_heads)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_landmarks: int = 4,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.head_dim = head_dim or (hidden_size // num_heads)
        
        # Learnable landmark keys and values (one per head)
        self.landmark_keys = nn.Parameter(
            torch.randn(num_heads, num_landmarks, self.head_dim) * 0.02
        )
        self.landmark_values = nn.Parameter(
            torch.randn(num_heads, num_landmarks, self.head_dim) * 0.02
        )
        
        # Learnable scale for landmark contribution
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def augment_attention_inputs(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add landmarks to key and value tensors.
        
        Args:
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            
        Returns:
            Augmented (key, value) with landmarks prepended
        """
        batch_size = key.shape[0]
        
        # Expand landmarks to batch
        landmark_k = self.landmark_keys.unsqueeze(0).expand(batch_size, -1, -1, -1)
        landmark_v = self.landmark_values.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Scale landmarks
        landmark_v = landmark_v * self.scale
        
        # Concatenate landmarks before sequence
        # New shape: [batch, num_heads, num_landmarks + seq_len, head_dim]
        aug_key = torch.cat([landmark_k, key], dim=2)
        aug_value = torch.cat([landmark_v, value], dim=2)
        
        return aug_key, aug_value
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"num_landmarks={self.num_landmarks}, scale={self.scale.item():.4f}"
        )


class PositionAdaptiveLandmark(nn.Module):
    """
    Position-Adaptive Landmark: Select landmarks based on position in context.
    
    Instead of global selection, each position can select different landmarks.
    Useful for long contexts where different regions need different abstractions.
    
    Args:
        hidden_size: Model hidden dimension
        num_landmarks: Number of learnable landmarks
        max_positions: Maximum position for bucketing (default: 2048)
        num_buckets: Number of position buckets (default: 32)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_landmarks: int = 8,
        max_positions: int = 2048,
        num_buckets: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_landmarks = num_landmarks
        self.max_positions = max_positions
        self.num_buckets = num_buckets
        
        # Learnable landmark tokens
        self.landmarks = nn.Parameter(torch.randn(num_landmarks, hidden_size) * 0.02)
        
        # Position-dependent gating: each bucket gets different gate weights
        # Shape: [num_buckets, num_landmarks]
        self.position_gates = nn.Parameter(torch.randn(num_buckets, num_landmarks) * 0.02)
        
        # Content-dependent refinement
        self.content_gate = nn.Linear(hidden_size, num_landmarks, bias=False)
        
        # Learnable scale
        self.scale = nn.Parameter(torch.tensor(0.1))
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    
    def _position_to_bucket(self, positions: torch.Tensor) -> torch.Tensor:
        """Map positions to buckets using logarithmic spacing."""
        # Logarithmic bucketing similar to position bias
        bucket_size = self.max_positions / self.num_buckets
        buckets = (positions / bucket_size).long()
        buckets = torch.clamp(buckets, 0, self.num_buckets - 1)
        return buckets
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply position-adaptive landmark selection.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
            
        Returns:
            Enhanced hidden states with position-aware landmarks
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=hidden_states.device)
        buckets = self._position_to_bucket(positions)  # [seq]
        
        # Get position-dependent gate logits
        pos_gate_logits = self.position_gates[buckets]  # [seq, num_landmarks]
        
        # Compute content-dependent gate logits
        content_gate_logits = self.content_gate(hidden_states)  # [batch, seq, num_landmarks]
        
        # Combine position and content
        # Position provides base preference, content refines it
        combined_logits = pos_gate_logits.unsqueeze(0) + content_gate_logits
        gate_weights = torch.softmax(combined_logits, dim=-1)  # [batch, seq, num_landmarks]
        
        # Apply gates to landmarks
        # [batch, seq, num_landmarks] @ [num_landmarks, hidden_size] -> [batch, seq, hidden_size]
        context = torch.matmul(gate_weights, self.landmarks)
        context = self.dropout(context)
        
        # Add scaled context
        output = hidden_states + self.scale * context
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_landmarks={self.num_landmarks}, "
            f"num_buckets={self.num_buckets}, scale={self.scale.item():.4f}"
        )


def count_landmark_params(landmark_module: nn.Module) -> int:
    """Count trainable parameters in a landmark module."""
    return sum(p.numel() for p in landmark_module.parameters() if p.requires_grad)


# ============= Helper Functions for Integration =============

def apply_per_layer_landmarks(
    model: nn.Module,
    hidden_size: int,
    num_landmarks: int = 4,
    dropout: float = 0.0,
    ffn_pattern: str = "mlp",
) -> dict:
    """
    Apply PerLayerLandmark to all FFN outputs in a model.
    
    Args:
        model: Base transformer model
        hidden_size: Model hidden dimension
        num_landmarks: Landmarks per layer
        dropout: Dropout probability
        ffn_pattern: Pattern to identify FFN modules
        
    Returns:
        Dictionary of {layer_name: landmark_module}
    """
    landmarks = {}
    
    for name, module in model.named_modules():
        # Apply after FFN output (before residual)
        if ffn_pattern in name.lower() and "output" in name.lower():
            landmark = PerLayerLandmark(
                hidden_size=hidden_size,
                num_landmarks=num_landmarks,
                dropout=dropout,
            )
            
            # Register as submodule
            setattr(module, "landmark", landmark)
            landmarks[name] = landmark
            
            # Register forward hook to apply landmark
            def landmark_hook(m, input, output):
                if hasattr(m, "landmark"):
                    return m.landmark(output)
                return output
            
            module.register_forward_hook(landmark_hook)
    
    return landmarks


def apply_attention_landmarks(
    model: nn.Module,
    hidden_size: int,
    num_heads: int,
    num_landmarks: int = 4,
    attention_pattern: str = "attention",
) -> dict:
    """
    Apply AttentionIntegratedLandmark to all attention modules.
    
    Note: Requires modifying attention forward pass to call augment_attention_inputs.
    This is a more invasive integration.
    
    Args:
        model: Base transformer model
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        num_landmarks: Number of landmark tokens
        attention_pattern: Pattern to identify attention modules
        
    Returns:
        Dictionary of {layer_name: landmark_module}
    """
    landmarks = {}
    
    for name, module in model.named_modules():
        if attention_pattern in name.lower():
            # Check if this is the main attention module
            has_proj = any(
                hasattr(module, attr) 
                for attr in ["q_proj", "k_proj", "v_proj", "c_attn"]
            )
            
            if has_proj:
                landmark = AttentionIntegratedLandmark(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_landmarks=num_landmarks,
                )
                
                setattr(module, "attention_landmark", landmark)
                landmarks[name] = landmark
    
    return landmarks
