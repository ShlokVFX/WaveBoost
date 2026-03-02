"""
Flash Attention Benchmark Comparison
Compares custom Flash Attention 2 implementation with PyTorch baseline
Metrics: Latency, Throughput, Memory Usage
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import csv
from pathlib import Path
import json
import subprocess
import sys

# Ensure torch can find CUDA
torch.cuda.is_available()


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def reset_gpu_memory():
    """Reset GPU memory"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_pytorch_attention(Q, K, V, num_runs=100, warmup=10):
    """Benchmark PyTorch's scaled dot-product attention"""
    reset_gpu_memory()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=1.0/np.sqrt(Q.shape[-1]))
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Latency measurement
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = F.scaled_dot_product_attention(Q, K, V, scale=1.0/np.sqrt(Q.shape[-1]))
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    latency_ms = np.mean(latencies)
    latency_std = np.std(latencies)
    
    # Throughput: tokens per second
    batch_size, seq_len = Q.shape[0], Q.shape[1]
    total_tokens = batch_size * seq_len
    throughput = (total_tokens / (latency_ms / 1000)) / 1e6  # Million tokens/sec
    
    return {
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std,
        "peak_memory_mb": peak_memory,
        "throughput_mtoks_per_sec": throughput,
        "throughput_tokens_per_sec": total_tokens / (latency_ms / 1000)
    }


def benchmark_naive_attention(Q, K, V, num_runs=100, warmup=10):
    """Benchmark naive attention implementation (for comparison)"""
    reset_gpu_memory()
    
    def naive_attention(Q, K, V):
        d = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = naive_attention(Q, K, V)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Latency measurement
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = naive_attention(Q, K, V)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    latency_ms = np.mean(latencies)
    latency_std = np.std(latencies)
    
    # Throughput
    batch_size, seq_len = Q.shape[0], Q.shape[1]
    total_tokens = batch_size * seq_len
    throughput = (total_tokens / (latency_ms / 1000)) / 1e6
    
    return {
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std,
        "peak_memory_mb": peak_memory,
        "throughput_mtoks_per_sec": throughput,
        "throughput_tokens_per_sec": total_tokens / (latency_ms / 1000)
    }


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
                    results['throughput_mtoks_per_sec'] = float(row['tokens_per_sec']) / 1e6
    
    # Load memory
    memory_file = benchmark_dir / "memory" / "memory.csv"
    if memory_file.exists():
        with open(memory_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['kernel'] == 'flash':
                    results['peak_memory_mb'] = float(row['peak_memory_mb'])
    
    return results if results else None


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different sequence lengths and batch sizes"""
    
    print("=" * 80)
    print("FLASH ATTENTION 2 BENCHMARK COMPARISON")
    print("=" * 80)
    
    # Test configurations
    configs = [
        {"batch_size": 1, "seq_len": 128, "d": 64},
        {"batch_size": 1, "seq_len": 512, "d": 64},
        {"batch_size": 2, "seq_len": 512, "d": 64},
        {"batch_size": 4, "seq_len": 1024, "d": 64},
        {"batch_size": 8, "seq_len": 2048, "d": 64},
        {"batch_size": 1, "seq_len": 4096, "d": 64},
    ]
    
    results_summary = []
    
    # Load custom benchmark if available
    custom_results = load_custom_benchmarks()
    
    for config in configs:
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        d = config["d"]
        
        print(f"\n{'─' * 80}")
        print(f"Config: Batch={batch_size}, Seq_Len={seq_len}, Dim={d}")
        print(f"{'─' * 80}")
        
        # Create test tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Q = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
        K = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
        V = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
        
        # Benchmark PyTorch baseline
        print("\n[PyTorch F.scaled_dot_product_attention]")
        pytorch_results = benchmark_pytorch_attention(Q, K, V)
        print(f"  Latency: {pytorch_results['latency_ms']:.4f} ms (±{pytorch_results['latency_std_ms']:.4f} ms)")
        print(f"  Throughput: {pytorch_results['throughput_mtoks_per_sec']:.2f} MToks/s ({pytorch_results['throughput_tokens_per_sec']:.0f} Toks/s)")
        print(f"  Peak Memory: {pytorch_results['peak_memory_mb']:.2f} MB")
        
        # Benchmark Naive implementation
        print("\n[Naive Attention (Baseline)]")
        naive_results = benchmark_naive_attention(Q, K, V)
        print(f"  Latency: {naive_results['latency_ms']:.4f} ms (±{naive_results['latency_std_ms']:.4f} ms)")
        print(f"  Throughput: {naive_results['throughput_mtoks_per_sec']:.2f} MToks/s ({naive_results['throughput_tokens_per_sec']:.0f} Toks/s)")
        print(f"  Peak Memory: {naive_results['peak_memory_mb']:.2f} MB")
        
        # Compare with custom if available
        if custom_results:
            print("\n[Custom Flash Attention 2]")
            print(f"  Latency: {custom_results.get('latency_ms', 'N/A')} ms")
            print(f"  Throughput: {custom_results.get('throughput_mtoks_per_sec', 'N/A'):.2f} MToks/s ({custom_results.get('throughput_tokens_per_sec', 0):.0f} Toks/s)")
            print(f"  Peak Memory: {custom_results.get('peak_memory_mb', 'N/A')} MB")
            
            # Calculate speedups
            if custom_results.get('latency_ms'):
                speedup_vs_pytorch = pytorch_results['latency_ms'] / custom_results['latency_ms']
                speedup_vs_naive = naive_results['latency_ms'] / custom_results['latency_ms']
                print(f"\n[Speedup vs PyTorch]: {speedup_vs_pytorch:.2f}x")
                print(f"[Speedup vs Naive]: {speedup_vs_naive:.2f}x")
        
        results_summary.append({
            "config": config,
            "pytorch": pytorch_results,
            "naive": naive_results,
            "custom": custom_results
        })
    
    return results_summary


def generate_comparison_report(results_summary):
    """Generate a detailed comparison report"""
    
    report_path = Path(__file__).parent / "BENCHMARK_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# Flash Attention 2 Benchmark Comparison Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report compares the custom Flash Attention 2 implementation with:\n")
        f.write("- **PyTorch**: `F.scaled_dot_product_attention` (highly optimized)\n")
        f.write("- **Naive**: Standard PyTorch implementation (non-fused)\n\n")
        
        f.write("## Metrics\n")
        f.write("- **Latency**: Execution time in milliseconds (lower is better)\n")
        f.write("- **Throughput**: Tokens processed per second (higher is better)\n")
        f.write("- **Peak Memory**: Maximum GPU memory used in MB (lower is better)\n\n")
        
        f.write("## Results\n\n")
        
        for i, result in enumerate(results_summary):
            config = result['config']
            pytorch = result['pytorch']
            naive = result['naive']
            custom = result['custom']
            
            f.write(f"### Configuration {i+1}: B={config['batch_size']}, Seq={config['seq_len']}, D={config['d']}\n\n")
            
            f.write("| Metric | PyTorch | Naive | Custom | Status |\n")
            f.write("|--------|---------|-------|--------|--------|\n")
            
            # Latency
            f.write(f"| Latency (ms) | {pytorch['latency_ms']:.4f} | {naive['latency_ms']:.4f} | ")
            if custom and 'latency_ms' in custom:
                f.write(f"{custom['latency_ms']:.4f}")
            else:
                f.write("N/A")
            f.write(" | ")
            if custom and 'latency_ms' in custom:
                speedup = pytorch['latency_ms'] / custom['latency_ms']
                status = "✓ Faster" if speedup > 1 else "✗ Slower"
                f.write(f"{status} ({speedup:.2f}x)")
            else:
                f.write("—")
            f.write(" |\n")
            
            # Throughput
            f.write(f"| Throughput (MToks/s) | {pytorch['throughput_mtoks_per_sec']:.2f} | {naive['throughput_mtoks_per_sec']:.2f} | ")
            if custom and 'throughput_mtoks_per_sec' in custom:
                f.write(f"{custom['throughput_mtoks_per_sec']:.2f}")
            else:
                f.write("N/A")
            f.write(" | ")
            if custom and 'throughput_mtoks_per_sec' in custom:
                speedup = custom['throughput_mtoks_per_sec'] / pytorch['throughput_mtoks_per_sec']
                status = "✓ Higher" if speedup > 1 else "✗ Lower"
                f.write(f"{status} ({speedup:.2f}x)")
            else:
                f.write("—")
            f.write(" |\n")
            
            # Memory
            f.write(f"| Peak Memory (MB) | {pytorch['peak_memory_mb']:.2f} | {naive['peak_memory_mb']:.2f} | ")
            if custom and 'peak_memory_mb' in custom:
                f.write(f"{custom['peak_memory_mb']:.2f}")
            else:
                f.write("N/A")
            f.write(" | ")
            if custom and 'peak_memory_mb' in custom:
                ratio = custom['peak_memory_mb'] / pytorch['peak_memory_mb']
                status = "✓ Lower" if ratio < 1 else "✗ Higher"
                f.write(f"{status} ({ratio:.2f}x)")
            else:
                f.write("—")
            f.write(" |\n")
            
            f.write("\n")
        
        f.write("## Key Findings\n\n")
        f.write("### Performance Comparison\n")
        if results_summary and results_summary[0]['custom']:
            first_custom_latency = results_summary[0]['custom'].get('latency_ms')
            first_pytorch_latency = results_summary[0]['pytorch']['latency_ms']
            if first_custom_latency:
                overall_speedup = first_pytorch_latency / first_custom_latency
                if overall_speedup > 1:
                    f.write(f"- Custom implementation is **{overall_speedup:.2f}x faster** than PyTorch baseline\n")
                else:
                    f.write(f"- PyTorch baseline is **{1/overall_speedup:.2f}x faster** than custom implementation\n")
        
        f.write("\n### Memory Efficiency\n")
        if results_summary and results_summary[0]['custom']:
            first_custom_mem = results_summary[0]['custom'].get('peak_memory_mb')
            first_pytorch_mem = results_summary[0]['pytorch']['peak_memory_mb']
            if first_custom_mem:
                memory_ratio = first_custom_mem / first_pytorch_mem
                if memory_ratio < 1:
                    f.write(f"- Custom implementation uses **{1/memory_ratio:.2f}x less memory**\n")
                else:
                    f.write(f"- Custom implementation uses **{memory_ratio:.2f}x more memory**\n")
        
        f.write("\n### Throughput Analysis\n")
        if results_summary and results_summary[0]['custom']:
            first_custom_thr = results_summary[0]['custom'].get('throughput_mtoks_per_sec')
            first_pytorch_thr = results_summary[0]['pytorch']['throughput_mtoks_per_sec']
            if first_custom_thr:
                thr_ratio = first_custom_thr / first_pytorch_thr
                if thr_ratio > 1:
                    f.write(f"- Custom implementation achieves **{thr_ratio:.2f}x higher throughput**\n")
                else:
                    f.write(f"- PyTorch baseline achieves **{1/thr_ratio:.2f}x higher throughput**\n")
    
    print(f"\n✓ Report saved to: {report_path}")


if __name__ == "__main__":
    print("\nStarting Flash Attention Benchmark Comparison...\n")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. GPU required for benchmarking.")
        sys.exit(1)
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}\n")
    
    results_summary = run_comprehensive_benchmark()
    
    print("\n" + "=" * 80)
    print("GENERATING DETAILED REPORT")
    print("=" * 80)
    
    generate_comparison_report(results_summary)
    
    print("\n✓ Benchmark Complete!")
