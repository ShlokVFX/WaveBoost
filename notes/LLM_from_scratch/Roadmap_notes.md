Computing Methodologies → Artificial Intelligence

Study transformer inference pipeline (prefill vs decode)
Implement self-attention, MHA, GQA, MLA inference kernels
Learn KV cache design, layout, and paging
Study FlashAttention v1 v2 v3
Study PagedAttention (vLLM)
Implement speculative decoding
Study Medusa and lookahead decoding
Learn Mixture-of-Experts inference routing
Study quantization methods (INT8, FP8, AWQ, GPTQ)
Explore continuous batching and sequence packing
Read and profile vLLM, TensorRT-LLM, FasterTransformer, DeepSpeed-Inference

Theory of Computation → Design and Analysis of Algorithms

Implement beam search decoding
Implement top-k and top-p sampling
Study speculative decoding algorithms
Design prefix cache and prompt reuse systems
Implement dynamic batching scheduler
Study token scheduling and fairness policies
Learn block-sparse attention algorithms
Design MoE token routing algorithms
Analyze latency vs throughput tradeoffs
Study memory access complexity in attention
Implement O(1) KV cache lookup structures

Software and its Engineering → Parallel Programming Languages

Learn CUDA programming fundamentals
Learn HIP / ROCm programming fundamentals
Study warp and wavefront execution
Implement shared memory tiling kernels
Implement Tensor Core GEMM kernels
Learn async copy (cp.async / TMA)
Optimize attention kernels end-to-end
Study CUTLASS kernel design
Study Triton kernel programming
Explore ThunderKittens and FlashInfer kernels
Optimize register usage and occupancy
Implement fused GEMM + softmax + attention kernels

Software and its Engineering → Just-in-Time Compilers

Study TorchDynamo graph capture
Learn TorchInductor kernel generation
Study Triton compiler internals
Learn TVM auto-scheduler (Ansor)
Study TensorRT engine builder
Learn MLIR GPU dialect basics
Study LLVM lowering pipeline
Implement kernel fusion passes
Study CUDA Graphs for inference
Analyze operator fusion in PyTorch 2.x
Profile JIT vs ahead-of-time kernels

Computer Systems Organization → Cloud Computing

Study distributed inference architectures
Learn tensor parallelism
Learn pipeline parallelism
Learn expert parallelism (MoE)
Study KV cache sharding across GPUs
Implement multi-GPU decoding
Study NVLink vs PCIe bandwidth impact
Learn RDMA and GPU networking basics
Deploy models with Ray Serve
Deploy models with Kubernetes GPU scheduling
Study NVIDIA Triton Inference Server
Explore SkyPilot and SageMaker inference
Study FlexGen and offloading systems

General and Reference → Performance; Measurement

Learn Nsight Systems profiling
Learn Nsight Compute kernel analysis
Use torch.profiler for inference tracing
Study rocprof for AMD GPUs
Measure TTFT (time to first token)
Measure inter-token latency
Measure tokens per second throughput
Analyze SM occupancy
Analyze DRAM bandwidth utilization
Study arithmetic intensity and roofline model
Benchmark with MLPerf Inference
Benchmark with vLLM perf suite
Build custom latency and throughput dashboards