# WaveBoost - Minimal Flash Attention CUDA Implementation

## Summary

WaveBoost is my personal repository to experiment with inference-time optimizations.
I implemented individual CUDA kernels for LLM inference.

---

## 📊 Performance Dashboard

### Benchmark Comparison
Implementation benchmarked against PyTorch's optimized baseline:

![Performance Dashboard](kernels/attention/flash_attention/visualizations/4_dashboard.png)
```

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

## 📚 References

### Flash Attention Papers
- **Flash Attention v1**: [Dao et al., 2022](https://arxiv.org/abs/2205.14135) - Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Flash Attention v2**: [Dao, 2023](https://arxiv.org/abs/2307.08691) - Faster Attention with Better Parallelism and Work Partitioning

### CUDA Optimization Resources
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Cutlass - CUDA Templates](https://github.com/NVIDIA/cutlass)
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-4)
---