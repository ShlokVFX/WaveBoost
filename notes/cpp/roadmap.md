
# LLM Inference Systems (C++/Systems) — Roadmap Checklist

---

## C++ Language Foundations (Inference Infra Focus)

[ ] C++11 core features
[ ] C++14 enhancements
[ ] C++17 features
[ ] RAII for GPU / buffer lifetimes
[ ] Smart pointer memory ownership in tensor runtimes
[ ] Const correctness in kernel APIs
[ ] Compile-time vs run-time dispatch
[ ] Template metaprogramming basics

---

## Object-Oriented Design for Inference Runtimes

[ ] Tensor abstraction design
[ ] Operator interface design (OpKernel style)
[ ] Backend abstraction (CUDA / ROCm / CPU)
[ ] Execution provider architecture
[ ] Graph node class design
[ ] Memory planner classes
[ ] Composition vs inheritance in runtimes

---

## STL Internals (Performance Lens)

[ ] std::vector growth & realloc cost (tensor buffers)
[ ] Cache locality of contiguous containers
[ ] Iterator invalidation in async pipelines
[ ] std::deque vs ring buffers for token streaming
[ ] std::unordered_map for KV cache indexing
[ ] Custom allocators for tensor memory
[ ] Arena allocators / slab allocators

---

## Multithreading & Parallel Execution

[ ] std::thread & worker pools
[ ] Inference request batching threads
[ ] CPU parallelism for token sampling
[ ] Pipeline parallel execution
[ ] Context switching overhead
[ ] Thread affinity & NUMA awareness
[ ] Prefill vs decode threading models

---

## Synchronization & Locking

[ ] Mutex vs spinlock in hot paths
[ ] Lock contention in KV cache updates
[ ] Condition variables for batching queues
[ ] Deadlock prevention in graph schedulers
[ ] Atomic counters for token generation
[ ] Memory ordering semantics
[ ] Lock-free data structures

---

## Classical Concurrency Patterns (Inference Mapping)

[ ] Producer–Consumer → Request queue → GPU worker
[ ] Reader–Writer → KV cache reads/writes
[ ] Bounded buffer → Token streaming
[ ] Work stealing schedulers
[ ] Lock-free queues for serving

---

## Advanced C++ for Kernel & Runtime Design

[ ] Move semantics for tensor transfer
[ ] Rvalue refs in buffer passing
[ ] Perfect forwarding in kernel launch APIs
[ ] Lambda functions for graph execution
[ ] Callback systems for async compute
[ ] Smart pointers in device memory mgmt
[ ] Custom deleters for CUDA / HIP buffers
[ ] C++ casting in runtime polymorphism

---

## Low-Level Runtime / System Design

[ ] Operator registry design
[ ] Kernel dispatch tables
[ ] Backend plugin systems
[ ] Execution graph schedulers
[ ] Memory pool design
[ ] Tensor reuse planners
[ ] Static vs dynamic shape handling

---

## GPU Compute & GEMM Systems

[ ] GEMM tiling strategies
[ ] Tensor Core utilization
[ ] Shared memory staging
[ ] Register tiling
[ ] Double buffering
[ ] Warp-level MMA ops
[ ] CUTLASS / Triton kernel structure
[ ] FP16 / BF16 / FP8 matmul pipelines

---

## Attention Systems Engineering

[ ] Naive attention implementation
[ ] FlashAttention algorithm
[ ] IO-aware attention tiling
[ ] Causal masking optimization
[ ] KV cache layout design
[ ] Paged attention systems
[ ] Sliding window attention
[ ] Multi-query / grouped-query attention

---

## Mixture of Experts (MoE) Systems

[ ] Expert routing algorithms
[ ] Top-k gating implementation
[ ] Token dispatch & gather kernels
[ ] All-to-all communication
[ ] Load balancing strategies
[ ] Expert parallelism vs tensor parallelism
[ ] Capacity factor tuning
[ ] Sparse GEMM optimization

---

## Network & Serving Systems

[ ] gRPC / HTTP inference servers
[ ] Token streaming over sockets
[ ] RDMA / InfiniBand basics
[ ] Request batching over network
[ ] Latency vs throughput trade-offs
[ ] Serialization (protobuf / flatbuffers)

---

## Async Execution & Futures

[ ] std::future / std::promise
[ ] Async kernel launch tracking
[ ] CUDA streams & events
[ ] Overlapping compute & memcpy
[ ] Launch policies (async vs deferred)

---

## Memory & Cache Systems

[ ] KV cache allocation strategies
[ ] Paged KV cache
[ ] GPU vs CPU offload
[ ] Unified memory trade-offs
[ ] Memory fragmentation handling
[ ] Prefetching strategies

---

## Performance Engineering

[ ] Roofline model for attention/GEMM
[ ] FLOPs vs bandwidth analysis
[ ] Kernel fusion opportunities
[ ] Profiling with Nsight / rocprof
[ ] Warp occupancy tuning
[ ] Cache miss analysis
