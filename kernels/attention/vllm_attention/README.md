# vLLM Attention Module

Wrapper and integration for vLLM's Flash Attention implementation.

## Overview

This module provides:
1. **vLLM Attention Wrapper** - Integration with vLLM's optimized attention kernels
2. **Three-Way Comparison Benchmark** - Compare Custom FA2 vs PyTorch SDPA vs vLLM
3. **Fallback Support** - Graceful fallback to PyTorch SDPA if vLLM not available

## Installation

### Option 1: With vLLM (Recommended)
```bash
pip install vllm
```

### Option 2: Without vLLM (Fallback Mode)
The module will automatically use PyTorch SDPA if vLLM is not installed.

## Usage

### Basic Usage
```python
from vllm_attention import vllm_attention_forward

# Your attention tensors
Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)
V = torch.randn(batch_size, seq_len, dim)

# Compute attention
output = vllm_attention_forward(Q, K, V)
```

### Advanced Usage
```python
from vllm_attention import VLLMFlashAttention

# Create attention layer
attn = VLLMFlashAttention(
    num_kv_heads=8,
    head_dim=64,
    scale=1.0/8.0
)

# Forward pass
output = attn.forward(query, key, value)
```

## Benchmarking

### Run Three-Way Comparison
```bash
python compare_all_attention.py
```

Output includes:
- Latency comparison (ms)
- Throughput comparison (tokens/sec)
- Memory usage (MB)
- Relative speedups

### Example Output
```
==========================================================================================
THREE-WAY ATTENTION COMPARISON
Custom FA2 vs PyTorch SDPA vs vLLM vs Naive
==========================================================================================

GPU: NVIDIA GeForce RTX 3060
PyTorch: 2.8.0+cu128

[Loading Custom FA2 Benchmarks]
  ✓ Custom FA2 Latency: 3.3185 ms
  ✓ Custom FA2 Throughput: 308,574 T/s
  ✓ Custom FA2 Memory: 20.22 MB

[Benchmarking PyTorch SDPA]
  ✓ Latency: 0.2211 ms
  ✓ Throughput: 2,316,037 T/s
  ✓ Memory: 24.05 MB

[Benchmarking vLLM Attention]
  ✓ Latency: 0.2156 ms (from vLLM's optimized kernels)
  ✓ Throughput: 2,372,881 T/s
  ✓ Memory: 23.98 MB

[Benchmarking Naive Attention]
  ✓ Latency: 2.1543 ms
  ✓ Throughput: 237,930 T/s
  ✓ Memory: 142.50 MB
```

## Architecture

### Module Structure
```
vllm_attention/
├── vllm_attention.py          # Main vLLM wrapper module
├── compare_all_attention.py   # Three-way comparison benchmark
└── README.md                  # This file
```

### Implementation Details

#### VLLMFlashAttention Class
- Wraps vLLM's FlashAttention kernels
- Provides fallback to PyTorch SDPA
- Handles tensor reshaping and format conversion

#### Fallback Strategy
1. Try vLLM's optimized kernels (if installed)
2. Fall back to PyTorch SDPA (always available)
3. Both provide identical numerical results

## Performance Characteristics

### When vLLM is Installed
- Uses vLLM's highly optimized Flash Attention v2 kernels
- Custom CUDA kernels tuned for specific GPUs
- Best performance for production inference

### When vLLM is Not Installed
- Falls back to PyTorch SDPA (PyTorch 2.0+)
- Automatically selected efficient attention implementation
- Good performance, slightly slower than dedicated vLLM kernels

## Comparison Results

### Test Configuration
- **GPU**: NVIDIA GeForce RTX 3060
- **Batch Size**: 1
- **Sequence Length**: 512
- **Head Dimension**: 64

### Performance Ranking
1. **vLLM** - Fastest (best for production)
2. **PyTorch SDPA** - Very close to vLLM
3. **Custom FA2** - Good for educational purposes
4. **Naive** - Baseline (very slow)

## Known Limitations

1. **vLLM Dependency**: Large framework installation
2. **GPU Support**: Requires NVIDIA GPU (CUDA 11.0+)
3. **Memory Format**: Expects specific tensor layouts
4. **Multi-GPU**: Single GPU per process in this implementation

## Roadmap

- [ ] Multi-GPU support
- [ ] Custom CUDA kernel integration with vLLM
- [ ] Paged attention support
- [ ] KV cache optimization
- [ ] Quantization support (FP16, INT8)

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention v2 Paper](https://arxiv.org/abs/2307.08691)

## License

MIT License - Same as parent repository
