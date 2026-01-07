## Flash Attention (Forward Pass)

This experiment implements a minimal forward-only Flash Attention kernel.

### Files
- flash.cu: attention kernel
- main.cpp: kernel launcher
- bench.py: benchmarking harness

### Benchmarks
Results are stored under:
benchmarks/flash_attention/

Key observations:
- Kernel launch overhead dominates at small sequence lengths
- Memory access patterns limit throughput at larger heads

Minimal Beginner Flash Attention Forward Pass implementation on RTX 3060
Rebuilding from this : https://github.com/tspeterkim/flash-attention-minimal/blob/main/README.md