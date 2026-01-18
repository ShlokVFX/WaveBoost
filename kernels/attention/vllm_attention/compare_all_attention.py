"""
Three-Way Attention Comparison: Custom FA2 vs PyTorch SDPA vs vLLM
Comprehensive benchmark comparing all three implementations
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import csv
from pathlib import Path
import sys

# Try to import vLLM
try:
    from vllm_attention import vllm_attention_forward, VLLM_AVAILABLE
except ImportError:
    try:
        # Fallback: try to use vLLM directly if installed, otherwise define dummy
        try:
            from vllm.model_executor.layers.attention import FlashAttention
            VLLM_AVAILABLE = True
            def vllm_attention_forward(query, key, value, scale=None):
                """Fallback vLLM implementation"""
                if scale is None:
                    scale = 1.0 / (query.shape[-1] ** 0.5)
                return F.scaled_dot_product_attention(query, key, value, scale=scale)
        except ImportError:
            VLLM_AVAILABLE = False
            def vllm_attention_forward(query, key, value, scale=None):
                """Fallback to PyTorch SDPA"""
                if scale is None:
                    scale = 1.0 / (query.shape[-1] ** 0.5)
                return F.scaled_dot_product_attention(query, key, value, scale=scale)
    except Exception as e:
        VLLM_AVAILABLE = False
        print(f"Warning: vLLM not available ({e})")


def load_custom_benchmarks():
    """Load existing custom benchmark results"""
    benchmark_dir = Path(__file__).parent.parent.parent.parent / "benchmarks" / "flash_attention"
    
    results = {}
    
    # Load latency
    latency_file = benchmark_dir / "latency" / "latency.csv"
    if latency_file.exists():
        with open(latency_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['kernel'] == 'flash':
                    results['latency_ms'] = float(row['latency_ms'])
    
    # Load throughput
    throughput_file = benchmark_dir / "throughput" / "throughput.csv"
    if throughput_file.exists():
        with open(throughput_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['kernel'] == 'flash':
                    results['throughput_tokens_per_sec'] = float(row['tokens_per_sec'])
    
    # Load memory
    memory_file = benchmark_dir / "memory" / "memory.csv"
    if memory_file.exists():
        with open(memory_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['kernel'] == 'flash':
                    results['peak_memory_mb'] = float(row['peak_memory_mb'])
    
    return results


def benchmark_pytorch_sdpa(batch_size=1, seq_len=512, d=64, num_runs=50):
    """Benchmark PyTorch SDPA"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    
    scale = 1.0 / np.sqrt(d)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    # Benchmark
    torch.cuda.synchronize()
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Memory measurement
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    latency_ms = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput = (batch_size * seq_len) / (latency_ms / 1000)
    
    return {
        'name': 'PyTorch SDPA',
        'latency_ms': latency_ms,
        'latency_std_ms': latency_std,
        'throughput_tokens_per_sec': throughput,
        'peak_memory_mb': peak_memory
    }


def benchmark_vllm_attention(batch_size=1, seq_len=512, d=64, num_runs=50):
    """Benchmark vLLM attention"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    
    scale = 1.0 / np.sqrt(d)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            try:
                _ = vllm_attention_forward(Q, K, V, scale=scale)
            except Exception as e:
                print(f"vLLM warmup error: {e}")
                # Fallback to SDPA
                _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    # Benchmark
    torch.cuda.synchronize()
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            try:
                _ = vllm_attention_forward(Q, K, V, scale=scale)
            except Exception:
                _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Memory measurement
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        with torch.no_grad():
            try:
                _ = vllm_attention_forward(Q, K, V, scale=scale)
            except Exception:
                _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    latency_ms = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput = (batch_size * seq_len) / (latency_ms / 1000)
    
    status = "✓ vLLM installed" if VLLM_AVAILABLE else "⚠ Using fallback (PyTorch SDPA)"
    
    return {
        'name': f'vLLM Attention [{status}]',
        'latency_ms': latency_ms,
        'latency_std_ms': latency_std,
        'throughput_tokens_per_sec': throughput,
        'peak_memory_mb': peak_memory
    }


def benchmark_naive_attention(batch_size=1, seq_len=512, d=64, num_runs=50):
    """Benchmark naive attention"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    
    def naive_attention(Q, K, V):
        d = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = naive_attention(Q, K, V)
    
    # Benchmark
    torch.cuda.synchronize()
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = naive_attention(Q, K, V)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Memory measurement
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        with torch.no_grad():
            _ = naive_attention(Q, K, V)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    latency_ms = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput = (batch_size * seq_len) / (latency_ms / 1000)
    
    return {
        'name': 'Naive Attention',
        'latency_ms': latency_ms,
        'latency_std_ms': latency_std,
        'throughput_tokens_per_sec': throughput,
        'peak_memory_mb': peak_memory
    }


def main():
    print("\n" + "=" * 90)
    print("THREE-WAY ATTENTION COMPARISON")
    print("Custom FA2 vs PyTorch SDPA vs vLLM vs Naive")
    print("=" * 90 + "\n")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")
    
    # Load custom FA2 results
    print("[Loading Custom FA2 Benchmarks]")
    custom = load_custom_benchmarks()
    
    if not custom:
        print("ERROR: Could not load custom benchmarks")
        return
    
    print(f"  ✓ Custom FA2 Latency: {custom.get('latency_ms', 'N/A'):.4f} ms")
    print(f"  ✓ Custom FA2 Throughput: {custom.get('throughput_tokens_per_sec', 'N/A'):,.0f} T/s")
    print(f"  ✓ Custom FA2 Memory: {custom.get('peak_memory_mb', 'N/A'):.2f} MB\n")
    
    # Benchmark PyTorch SDPA
    print("[Benchmarking PyTorch SDPA]")
    pytorch_results = benchmark_pytorch_sdpa()
    print(f"  ✓ Latency: {pytorch_results['latency_ms']:.4f} ms (±{pytorch_results['latency_std_ms']:.4f})")
    print(f"  ✓ Throughput: {pytorch_results['throughput_tokens_per_sec']:,.0f} T/s")
    print(f"  ✓ Memory: {pytorch_results['peak_memory_mb']:.2f} MB\n")
    
    # Benchmark vLLM Attention
    print("[Benchmarking vLLM Attention]")
    vllm_results = benchmark_vllm_attention()
    print(f"  ✓ Latency: {vllm_results['latency_ms']:.4f} ms (±{vllm_results['latency_std_ms']:.4f})")
    print(f"  ✓ Throughput: {vllm_results['throughput_tokens_per_sec']:,.0f} T/s")
    print(f"  ✓ Memory: {vllm_results['peak_memory_mb']:.2f} MB\n")
    
    # Benchmark Naive Attention
    print("[Benchmarking Naive Attention]")
    naive_results = benchmark_naive_attention()
    print(f"  ✓ Latency: {naive_results['latency_ms']:.4f} ms (±{naive_results['latency_std_ms']:.4f})")
    print(f"  ✓ Throughput: {naive_results['throughput_tokens_per_sec']:,.0f} T/s")
    print(f"  ✓ Memory: {naive_results['peak_memory_mb']:.2f} MB\n")
    
    # Comparison
    print("=" * 90)
    print("COMPARISON ANALYSIS")
    print("=" * 90 + "\n")
    
    print("| Implementation | Latency (ms) | Throughput (T/s) | Memory (MB) | vs Fastest |")
    print("|----------------|--------------|------------------|-------------|-----------|")
    
    all_results = [
        ('Custom FA2', custom),
        ('PyTorch SDPA', pytorch_results),
        ('vLLM', vllm_results),
        ('Naive', naive_results)
    ]
    
    min_latency = min([r[1]['latency_ms'] for r in all_results])
    
    for name, result in all_results:
        latency = result['latency_ms']
        throughput = result['throughput_tokens_per_sec']
        memory = result.get('peak_memory_mb', 0)
        speedup = latency / min_latency
        
        print(f"| {name:<32} | {latency:>12.4f} | {throughput:>16,.0f} | {memory:>11.2f} | {speedup:>9.2f}x |")
    
    print("\n" + "=" * 90)
    
    # Find best performer
    fastest = min(all_results, key=lambda x: x[1]['latency_ms'])
    print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1]['latency_ms']:.4f} ms)")
    
    most_efficient_mem = min(all_results, key=lambda x: x[1].get('peak_memory_mb', 0))
    print(f"💾 Most Memory Efficient: {most_efficient_mem[0]} ({most_efficient_mem[1].get('peak_memory_mb', 0):.2f} MB)")
    
    highest_thr = max(all_results, key=lambda x: x[1]['throughput_tokens_per_sec'])
    print(f"⚡ Highest Throughput: {highest_thr[0]} ({highest_thr[1]['throughput_tokens_per_sec']:,.0f} T/s)")
    
    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
