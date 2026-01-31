# WaveBoost – Inference Checklist

## Foundations

* [ ] Linux process / thread / async basics
* [ ] CPU ↔ GPU memory flow
* [ ] NUMA basics
* [ ] Python GIL impact on inference
* [ ] FLOPs vs memory bandwidth
* [ ] Arithmetic intensity

---

## Transformer Inference Internals

* [ ✅ ] Implement naive attention
* [ ✅ ] Implement causal attention
* [ ✅ ] Implement MHA
* [ ✅ ] Implement GQA
* [ ✅ ] Benchmark latency & tokens/sec
* [ ✅ ] Analyze MHA vs GQA performance
* [ ] Prefill vs decode separation

---

## KV Cache

* [ ✅ ] Implement KV cache
* [ ] Measure memory growth vs sequence length
* [ ] Decode-only attention using KV cache
* [ ] Visualize KV cache layout
* [ ✅ ] Benchmark with and without KV cache

---

## GPU Kernels & Performance

* [ ✅ ] Write Triton matmul kernel
* [ ✅ ] Tune tile sizes
* [ ✅ ] Use shared memory
* [ ✅ ] Measure occupancy
* [ ✅ ] Profile kernel execution

---

## Attention Optimization

* [ ] Triton scaled dot-product attention
* [ ] FlashAttention-style tiling
* [ ] Compare vs PyTorch SDPA
* [ ] Roofline analysis

---

## Batching & Serving

* [ ] Static batching
* [ ] Dynamic batching
* [ ] Continuous batching
* [ ] Measure padding waste
* [ ] Throughput vs latency tradeoff

---

## Load & Scale Testing

* [ ] Single-request latency
* [ ] Multi-request concurrency
* [ ] 10k token stress test
* [ ] 100k token stress test
* [ ] Latency percentiles (P50/P90/P99)

---

## Memory Optimization

* [ ] FP16 vs BF16 benchmarking
* [ ] INT8 inference
* [ ] KV cache quantization
* [ ] Paged KV cache concept

---

## Graph & Launch Optimization

* [ ] torch.compile
* [ ] CUDA graphs
* [ ] Reduce kernel launch count
* [ ] Identify CPU-side bottlenecks

---

## End-to-End LLM Serving

* [ ] Load LLM
* [ ] Prefill + decode pipeline
* [ ] KV cache reuse
* [ ] Continuous batching enabled
* [ ] Tokens/sec measurement
* [ ] GPU memory tracking

---

## Repo & Interview Readiness

* [ ] Clean README
* [ ] Architecture diagram
* [ ] Benchmark tables
* [ ] Lessons learned section
* [ ] Can explain repo without code
