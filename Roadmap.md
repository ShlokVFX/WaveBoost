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

* [ ✅ ] Implement naive attention ([attention.py](notes/Physics%20of%20LLM%20inference/Transformer%20mechanics/attention.py))
* [ ✅ ] Implement causal attention ([attention.py](notes/Physics%20of%20LLM%20inference/Transformer%20mechanics/attention.py))
* [ ✅ ] Implement MHA ([attention.py](notes/Physics%20of%20LLM%20inference/Transformer%20mechanics/attention.py))
* [ ✅ ] Implement GQA ([attention.py](notes/Physics%20of%20LLM%20inference/Transformer%20mechanics/attention.py))
* [ ✅ ] Benchmark latency & tokens/sec ([attention_comparison_all_tests.png](notes/Physics%20of%20LLM%20inference/Transformer%20mechanics/visualization/attention_comparison_all_tests.png))
* [ ✅ ] Analyze MHA vs GQA performance ([attention_comparison_all_tests.png](notes/Physics%20of%20LLM%20inference/Transformer%20mechanics/visualization/attention_comparison_all_tests.png))
* [ ] Prefill vs decode separation 

---

## KV Cache

* [ ✅ ] Implement KV cache ([text_generation_gpt2.py](notes/text_generation_gpt2.py))
* [ ] Measure memory growth vs sequence length
* [ ] Decode-only attention using KV cache
* [ ] Visualize KV cache layout
* [ ✅ ] Benchmark with and without KV cache ([kv_cache_comparison.png](notes/visualization/kv_cache_comparison.png))

---

## GPU Kernels & Performance

* [  ] Write Triton matmul kernel
* [  ] Tune tile sizes
* [  ] Use shared memory
* [  ] Measure occupancy
* [  ] Profile kernel execution

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
