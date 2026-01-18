# WaveBoost - Flash Attention 2 CUDA Implementation

## Summary

WaveBoost is a custom CUDA implementation of Flash Attention 2, optimized for high-performance attention mechanisms in transformer models. This repository contains hand-crafted kernels benchmarked against PyTorch's production-ready baseline, demonstrating core algorithm correctness while identifying optimization opportunities. The implementation includes comprehensive performance analysis, memory profiling, and visualization tools for kernel development.

---

## 🎯 Project Overview

| ID | Kernel Name | Description |
|:--:|-------------|-------------|
| 1 | **Flash Attention 2** | Custom CUDA implementation of Flash Attention with IO-aware attention computation, reducing memory bandwidth while maintaining numerical stability |

---

## 📊 Performance Dashboard

### Benchmark Comparison
Your implementation benchmarked against PyTorch's optimized baseline:

![Performance Dashboard](kernels/attention/flash_attention/visualizations/4_dashboard.png)

### Key Metrics

| Metric | Custom FA2 | PyTorch FA2 | Status |
|--------|-----------|------------|--------|
| **Latency** | 3.32 ms | 0.58 ms | 5.7x faster (PyTorch) |
| **Throughput** | 308K T/s | 883K T/s | 2.9x higher (PyTorch) |
| **Memory** | 20.22 MB | ~24 MB | 16% less (Custom) ✓ |

---

## 🚀 Getting Started

### Prerequisites
- NVIDIA GPU with CUDA Compute Capability 7.0+ (Turing or newer)
- CUDA Toolkit 11.0 or higher
- PyTorch 2.0+
- CMake 3.18+

### Build Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/WaveBoost.git
cd WaveBoost
```

#### 2. Build the CUDA Kernel
```bash
cd kernels/attention/flash_attention
mkdir build
cd build
cmake ..
make -j$(nproc)
```

#### 3. Compile with PyTorch Bindings
```bash
# From the build directory
cd ../..
python setup.py build_ext --inplace
```

#### 4. Run Benchmarks
```bash
cd benchmarks/flash_attention

# Run latency benchmark
python ../../../kernels/attention/flash_attention/benchmark_comparison.py

# Generate visualizations
python ../../../kernels/attention/flash_attention/plot_benchmark.py

# Quick comparison against PyTorch
python ../../../kernels/attention/flash_attention/compare_baseline.py
```

### Quick Test
```python
import torch
from flash_attention import flash_attention_forward

# Create test tensors
batch_size, seq_len, d = 1, 512, 64
Q = torch.randn(batch_size, seq_len, d, device='cuda')
K = torch.randn(batch_size, seq_len, d, device='cuda')
V = torch.randn(batch_size, seq_len, d, device='cuda')

# Run custom implementation
output = flash_attention_forward(Q, K, V)
print(f"Output shape: {output.shape}")
```

---

## 📁 Project Structure

```
WaveBoost/
├── README.md                          # This file
├── kernels/
│   └── attention/
│       └── flash_attention/
│           ├── Flash.cu               # Main CUDA kernel implementation
│           ├── main.cpp               # C++ wrapper and PyTorch binding
│           ├── benchmark_comparison.py# Comprehensive benchmark suite
│           ├── compare_baseline.py    # Quick PyTorch comparison
│           ├── plot_benchmark.py      # Visualization generator
│           ├── PERFORMANCE_ANALYSIS.md# Detailed optimization analysis
│           ├── COMPARISON_ANALYSIS.md # Performance comparison report
│           ├── Readme.md              # Kernel-specific documentation
│           └── visualizations/        # Generated benchmark graphs
│               ├── 1_latency.png
│               ├── 2_throughput.png
│               ├── 3_memory.png
│               ├── 4_dashboard.png
│               └── 5_speedup.png
└── benchmarks/
    └── flash_attention/
        ├── latency/
        │   └── latency.csv
        ├── throughput/
        │   └── throughput.csv
        ├── memory/
        │   └── memory.csv
        └── profiling/
            └── profiler.txt
```

---

## 📈 Performance Analysis

### Optimization Opportunities

The current implementation demonstrates the core Flash Attention algorithm. Performance analysis identifies these optimization areas:

#### Priority 1: Quick Wins (Est. 2-3x speedup)
- Use `float4` vectorized memory loads
- Increase thread block size from 32 to 256
- Add pragma loop unrolling (`#pragma unroll`)

#### Priority 2: Medium Effort (Est. 3-5x additional)
- Implement double buffering for memory operations
- Vectorize exp/softmax computations
- Template specialization for common dimensions

#### Priority 3: Advanced (Est. 2x additional)
- Persistent kernel patterns
- Ring attention for multi-GPU
- Quantization support (FP16/BF16)

See `kernels/attention/flash_attention/PERFORMANCE_ANALYSIS.md` for detailed recommendations.

---

## 📊 Benchmark Results

### Test Configuration
- **GPU**: NVIDIA GeForce RTX 3060
- **Batch Size**: 1
- **Sequence Length**: 512
- **Head Dimension**: 64
- **PyTorch Version**: 2.8.0+cu128

### Results Summary
```
╔════════════════════════╦═══════════════╦═══════════════╦═════════╗
║ Metric                 ║ Custom FA2    ║ PyTorch FA2   ║ Ratio   ║
╠════════════════════════╬═══════════════╬═══════════════╬═════════╣
║ Latency (ms)           ║ 3.3185        ║ 0.2211        ║ 15.01x  ║
║ Throughput (T/s)       ║ 308,574       ║ 2,316,037     ║ 7.51x   ║
║ Peak Memory (MB)       ║ 20.22         ║ ~24           ║ 1.19x ✓ ║
║ Efficiency (T/s/MB)    ║ 15,262        ║ 96,502        ║ 6.32x   ║
╚════════════════════════╩═══════════════╩═══════════════╩═════════╝
```

### Visualization Artifacts

Generated benchmark comparison charts are available in `kernels/attention/flash_attention/visualizations/`:

- **1_latency.png** - Latency comparison bar chart
- **2_throughput.png** - Throughput comparison bar chart
- **3_memory.png** - Memory usage comparison
- **4_dashboard.png** - Complete performance dashboard
- **5_speedup.png** - Speedup analysis

---

## 🔧 Development

### Building from Source

```bash
# Install dependencies
pip install torch ninja

# Build CUDA extensions
python setup.py develop

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/flash_attention/run_all_benchmarks.py
```

### Profiling

Profile the kernel with NVIDIA Nsys:

```bash
nsys profile --stats=true \
  python kernels/attention/flash_attention/benchmark_comparison.py
```

### Contributing

Contributions are welcome! Areas of interest:
- Kernel optimizations
- FP16/BF16 support
- Multi-GPU implementations
- Additional attention variants

---

## 📚 References

### Flash Attention Papers
- **Flash Attention v1**: [Dao et al., 2022](https://arxiv.org/abs/2205.14135) - Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Flash Attention v2**: [Dao, 2023](https://arxiv.org/abs/2307.08691) - Faster Attention with Better Parallelism and Work Partitioning

### CUDA Optimization Resources
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Cutlass - CUDA Templates](https://github.com/NVIDIA/cutlass)
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-4)

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

### License Summary
- ✅ **Personal use** - Allowed
- ✅ **Commercial use** - Allowed
- ✅ **Modification** - Allowed
- ✅ **Distribution** - Allowed
- ⚠️ **Liability** - No warranty provided
- 📝 **Attribution** - Required (include license notice)

### Dependencies Licenses
- **PyTorch** - BSD License
- **CUDA Toolkit** - NVIDIA License
- **CMake** - BSD License

For detailed license information, see the full [MIT License](LICENSE).

---

## 🤝 Acknowledgments

- Flash Attention algorithm by [Tri Dao et al.](https://github.com/Dao-AILab/flash-attention)
- Benchmark methodology inspired by MLCommons MLPerf
- CUDA optimization techniques from NVIDIA's best practices

---

## 📞 Support & Contact

For issues, questions, or feature requests:
- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/WaveBoost/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/WaveBoost/discussions)

---

## 🗂️ Additional Documentation

- [Flash Attention Kernel Documentation](kernels/attention/flash_attention/Readme.md)
- [Performance Analysis Report](kernels/attention/flash_attention/PERFORMANCE_ANALYSIS.md)
- [Comparison Analysis](kernels/attention/flash_attention/COMPARISON_ANALYSIS.md)
- [Benchmark Data](benchmarks/flash_attention/)

---

**Last Updated**: January 18, 2026  
**Status**: Active Development  
**Maintained By**: Shlok Limbhare
