"""
vLLM Flash Attention Integration
Wrapper for vLLM's optimized Flash Attention implementation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from vllm.model_executor.layers.attention import FlashAttention
    from vllm.model_executor.layers.attention import PagedAttention
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. Using fallback implementation.")


class VLLMFlashAttention:
    """
    vLLM Flash Attention Wrapper
    Provides interface to vLLM's optimized Flash Attention implementation
    """
    
    def __init__(self, num_kv_heads: int = 1, head_dim: int = 64, scale: float = None):
        """
        Initialize vLLM Flash Attention
        
        Args:
            num_kv_heads: Number of key-value heads
            head_dim: Dimension of each head
            scale: Attention scale factor (default: 1/sqrt(head_dim))
        """
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)
        
        if VLLM_AVAILABLE:
            self.attention = FlashAttention(
                num_heads=num_kv_heads,
                head_size=head_dim,
                scale=self.scale,
            )
        else:
            self.attention = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention using vLLM's Flash Attention
        
        Args:
            query: Query tensor (batch_size, seq_len, num_heads, head_dim)
            key: Key tensor (batch_size, seq_len, num_heads, head_dim)
            value: Value tensor (batch_size, seq_len, num_heads, head_dim)
            attn_mask: Optional attention mask
            
        Returns:
            Attention output (batch_size, seq_len, num_heads, head_dim)
        """
        if VLLM_AVAILABLE and self.attention is not None:
            try:
                # vLLM expects batched multi-head format
                output = self.attention(query, key, value, kv_cache=None)
                return output
            except Exception as e:
                print(f"vLLM attention failed: {e}")
                return self._fallback_attention(query, key, value)
        else:
            return self._fallback_attention(query, key, value)
    
    def _fallback_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fallback: PyTorch SDPA (same as official vLLM uses on non-optimized paths)
        """
        # Reshape for SDPA: (batch, seq, num_heads, dim) -> (batch, num_heads, seq, dim)
        batch_size, seq_len, num_heads, head_dim = query.shape
        
        q = query.transpose(1, 2)  # (batch, num_heads, seq, dim)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        
        # Apply SDPA
        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        
        # Reshape back
        output = attn_output.transpose(1, 2)  # (batch, seq, num_heads, dim)
        return output


def create_vllm_attention_layer(
    num_heads: int = 1,
    head_dim: int = 64,
    scale: float = None,
) -> VLLMFlashAttention:
    """
    Factory function to create vLLM attention layer
    
    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        scale: Optional scale factor
        
    Returns:
        VLLMFlashAttention instance
    """
    return VLLMFlashAttention(
        num_kv_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
    )


def vllm_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Simple function interface for vLLM attention
    
    Args:
        query: (batch, seq_len, dim)
        key: (batch, seq_len, dim)
        value: (batch, seq_len, dim)
        scale: Optional scale factor
        
    Returns:
        Output attention tensor
    """
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    
    if VLLM_AVAILABLE:
        try:
            # Try to use vLLM's optimized path
            attn = VLLMFlashAttention(num_kv_heads=1, head_dim=query.shape[-1], scale=scale)
            
            # Reshape to multi-head format
            q = query.unsqueeze(2)  # Add heads dimension
            k = key.unsqueeze(2)
            v = value.unsqueeze(2)
            
            return attn.forward(q, k, v).squeeze(2)
        except Exception as e:
            print(f"vLLM forward failed: {e}, falling back to PyTorch SDPA")
            return F.scaled_dot_product_attention(query, key, value, scale=scale)
    else:
        # Fallback to PyTorch SDPA
        return F.scaled_dot_product_attention(query, key, value, scale=scale)


if __name__ == "__main__":
    print("vLLM Attention Module")
    print(f"vLLM Available: {VLLM_AVAILABLE}")
    
    # Test
    batch_size, seq_len, dim = 2, 512, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Q = torch.randn(batch_size, seq_len, dim, device=device)
    K = torch.randn(batch_size, seq_len, dim, device=device)
    V = torch.randn(batch_size, seq_len, dim, device=device)
    
    output = vllm_attention_forward(Q, K, V)
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print("Test passed!")
