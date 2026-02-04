Foundations
[ ] Linux process, thread, async execution
[ ] CPU–GPU memory flow and synchronization
[ ] NUMA and PCIe effects on inference
[ ] Python GIL impact on serving
[ ] FLOPs vs memory bandwidth
[ ] Arithmetic intensity and roofline intuition

Transformer Inference Internals
[ ] Naive attention implementation (code: notes/Physics of LLM inference/Transformer mechanics/attention.py)
[ ] Causal attention
[ ] Multi-Head Attention (MHA)
[ ] Grouped Query Attention (GQA)
[ ] Multi-head Latent Attention (MLA)
[ ] Latency and tokens/sec benchmarking
[ ] Prefill vs decode separation

Core Kernel Ownership
[ ] Own kernel performance end-to-end (design → tune → benchmark → regress)
[ ] Custom GEMM kernels (Triton → CUDA → HIP)
[ ] GEMM kernel code proof (kernels/gemm/)
[ ] Attention kernel optimization (MHA, GQA, MLA, Flash-style)
[ ] MoE routing and matmul kernels
[ ] Decode kernels optimized for KV cache reads
[ ] Match or exceed vendor libraries (cuBLAS, rocBLAS, FlashAttention)
[ ] Kernel profiling with Nsight and rocprof
[ ] Bottleneck elimination across memory, registers, scheduling, occupancy

Attention Optimization
[ ] Triton scaled dot-product attention
[ ] SDPA kernel code proof (kernels/attention/triton_sdpa.py)
[ ] FlashAttention-style tiling and shared memory reuse
[ ] Decode-only attention kernels
[ ] Latent KV projection and fusion (MLA)
[ ] Comparison against PyTorch SDPA and FlashAttention
[ ] Roofline analysis and arithmetic intensity justification

Low-Level GPU Optimization
[ ] Shared memory / LDS optimization
[ ] Register pressure analysis
[ ] Instruction scheduling and pipelining
[ ] Occupancy versus ILP trade-offs
[ ] Warp and wave utilization analysis
[ ] Inline intrinsics (MMA / MFMA)
[ ] Intrinsics microkernel code proof (kernels/micro/mma_mfma/)
[ ] Reasoning about SM/WGP behavior
[ ] Cache hierarchy and memory system analysis

KV Cache and Memory Management
[ ] Block-based KV cache design
[ ] Logical to physical page mapping
[ ] KV allocator and free-list management
[ ] KV cache fragmentation handling
[ ] Prefix caching and reuse
[ ] Latent KV cache support (MLA)
[ ] CPU–GPU KV swap path (slow path)
[ ] KV cache benchmarking and capacity modeling

Inference Scheduling and Batching
[ ] Continuous batching for decode
[ ] Prefill and decode interleaving
[ ] Token-level scheduling
[ ] Priority and fairness policies
[ ] Preemption and eviction strategies
[ ] Memory-aware scheduling
[ ] Scheduler throughput and latency tuning

Inference Server Architecture
[ ] Separation of control plane and data plane
[ ] Async request lifecycle management
[ ] Streaming token generation
[ ] Request cancellation and timeout handling
[ ] Backpressure and admission control
[ ] OpenAI-compatible API surface
[ ] REST and gRPC serving paths

Runtime Performance Engineering
[ ] CUDA graph capture for decode
[ ] Kernel launch minimization per token
[ ] Overlap CPU scheduling with GPU execution
[ ] Persistent decode kernels
[ ] End-to-end inference flamegraph analysis
[ ] Nsight Systems trace analysis

Compiler and MLIR Integration
[ ] MLIR GPU, Linalg, and LLVM dialect understanding
[ ] Python / FX graph to MLIR lowering
[ ] GPU codegen lowering pipeline extensions
[ ] Compiler passes for tiling
[ ] Compiler passes for vectorization
[ ] Compiler passes for prefetching
[ ] Compiler passes for software pipelining
[ ] Compiler passes for layout transformations
[ ] ISA-aware transformations (warp and wave mapping)
[ ] Emission of efficient LLVM IR, PTX, and GCN

Performance Modeling and Tooling
[ ] Mental performance models (roofline, latency hiding)
[ ] Empirical performance modeling
[ ] Custom microbenchmarks for hypothesis testing
[ ] Microkernel code proof (kernels/micro/)
[ ] Profiler counter analysis (Nsight, rocprof)
[ ] Kernel disassembly validation
[ ] Performance regression testing

Reliability and Observability
[ ] Per-request latency breakdown
[ ] Tokens/sec and throughput metrics
[ ] GPU utilization monitoring
[ ] KV cache usage metrics
[ ] OOM detection and recovery
[ ] Health checks and graceful shutdown

Architecture Bring-Up
[ ] Kernel adaptation for new GPU architectures
[ ] Tuning for new SM and WGP characteristics
[ ] Compiler strategy updates per architecture
[ ] Correctness and performance validation
[ ] Documentation of architectural differences and lessons learned

Repo and Interview Readiness
[ ] Clean README with inference engine narrative
[ ] Scheduler, KV cache, and execution diagrams
[ ] Benchmark tables versus vLLM and TGI
[ ] Lessons learned and trade-off analysis
[ ] Ability to explain the repo without opening code