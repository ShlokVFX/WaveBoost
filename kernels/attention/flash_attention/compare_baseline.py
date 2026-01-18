"""
Quick Comparison Analysis
Compares custom Flash Attention 2 results with PyTorch baseline
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from pathlib import Path
import csv

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


def benchmark_pytorch_quick(batch_size=1, seq_len=512, d=64, num_runs=50):
    """Quick benchmark of PyTorch attention"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, d, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=1.0/np.sqrt(d))
    
    # Benchmark
    torch.cuda.synchronize()
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=1.0/np.sqrt(d))
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latency_ms = np.mean(latencies)
    throughput = (batch_size * seq_len) / (latency_ms / 1000)
    
    return {
        'latency_ms': latency_ms,
        'throughput_tokens_per_sec': throughput
    }


def main():
    print("=" * 80)
    print("FLASH ATTENTION 2: CUSTOM vs PYTORCH BASELINE")
    print("=" * 80)
    
    # Load custom results
    custom = load_custom_benchmarks()
    
    if not custom:
        print("ERROR: Could not load custom benchmark results from benchmarks folder")
        return
    
    print("\n[CUSTOM IMPLEMENTATION RESULTS]")
    print(f"  Latency: {custom.get('latency_ms', 'N/A'):.4f} ms")
    print(f"  Throughput: {custom.get('throughput_tokens_per_sec', 'N/A'):,.0f} tokens/sec")
    print(f"  Peak Memory: {custom.get('peak_memory_mb', 'N/A'):.2f} MB")
    
    print("\n[BENCHMARKING PYTORCH BASELINE...]")
    pytorch = benchmark_pytorch_quick(batch_size=1, seq_len=512, d=64)
    
    print(f"\n[PYTORCH BASELINE RESULTS]")
    print(f"  Latency: {pytorch['latency_ms']:.4f} ms")
    print(f"  Throughput: {pytorch['throughput_tokens_per_sec']:,.0f} tokens/sec")
    
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Latency comparison
    if custom.get('latency_ms'):
        latency_ratio = pytorch['latency_ms'] / custom['latency_ms']
        if latency_ratio > 1:
            print(f"\n✓ SPEED: Custom is {latency_ratio:.2f}x FASTER than PyTorch")
            print(f"  Custom: {custom['latency_ms']:.4f} ms vs PyTorch: {pytorch['latency_ms']:.4f} ms")
        else:
            print(f"\n✗ SPEED: PyTorch is {1/latency_ratio:.2f}x faster than Custom")
            print(f"  PyTorch: {pytorch['latency_ms']:.4f} ms vs Custom: {custom['latency_ms']:.4f} ms")
    
    # Throughput comparison
    if custom.get('throughput_tokens_per_sec'):
        thr_ratio = custom['throughput_tokens_per_sec'] / pytorch['throughput_tokens_per_sec']
        if thr_ratio > 1:
            print(f"\n✓ THROUGHPUT: Custom is {thr_ratio:.2f}x HIGHER than PyTorch")
            print(f"  Custom: {custom['throughput_tokens_per_sec']:,.0f} T/s vs PyTorch: {pytorch['throughput_tokens_per_sec']:,.0f} T/s")
        else:
            print(f"\n✗ THROUGHPUT: PyTorch is {1/thr_ratio:.2f}x higher than Custom")
            print(f"  PyTorch: {pytorch['throughput_tokens_per_sec']:,.0f} T/s vs Custom: {custom['throughput_tokens_per_sec']:,.0f} T/s")
    
    # Efficiency score
    print(f"\n[EFFICIENCY ANALYSIS]")
    if custom.get('latency_ms') and custom.get('peak_memory_mb'):
        efficiency = custom['throughput_tokens_per_sec'] / custom['peak_memory_mb']
        pytorch_efficiency = pytorch['throughput_tokens_per_sec'] / 24  # Approximate PyTorch memory
        print(f"  Custom Efficiency: {efficiency:,.0f} tokens/sec per MB")
        print(f"  PyTorch Efficiency: {pytorch_efficiency:,.0f} tokens/sec per MB")
    
    print("\n" + "=" * 80)
    
    # Generate detailed report
    report_path = Path(__file__).parent / "COMPARISON_ANALYSIS.md"
    with open(report_path, 'w') as f:
        f.write("# Flash Attention 2: Performance Comparison Report\n\n")
        f.write("## Custom Implementation vs PyTorch Baseline\n\n")
        
        f.write("### Results Summary\n\n")
        f.write("| Metric | Custom | PyTorch | Ratio | Winner |\n")
        f.write("|--------|--------|---------|-------|--------|\n")
        
        if custom.get('latency_ms'):
            latency_ratio = pytorch['latency_ms'] / custom['latency_ms']
            winner = "✓ Custom" if latency_ratio > 1 else "✗ PyTorch"
            f.write(f"| Latency (ms) | {custom['latency_ms']:.4f} | {pytorch['latency_ms']:.4f} | {latency_ratio:.2f}x | {winner} |\n")
        
        if custom.get('throughput_tokens_per_sec'):
            thr_ratio = custom['throughput_tokens_per_sec'] / pytorch['throughput_tokens_per_sec']
            winner = "✓ Custom" if thr_ratio > 1 else "✗ PyTorch"
            f.write(f"| Throughput (T/s) | {custom['throughput_tokens_per_sec']:,.0f} | {pytorch['throughput_tokens_per_sec']:,.0f} | {thr_ratio:.2f}x | {winner} |\n")
        
        f.write("\n### Key Observations\n\n")
        
        if custom.get('latency_ms'):
            latency_ratio = pytorch['latency_ms'] / custom['latency_ms']
            if latency_ratio > 1:
                f.write(f"- **Speed Advantage**: Custom implementation is {latency_ratio:.2f}x faster\n")
            else:
                f.write(f"- **Speed Gap**: PyTorch baseline is {1/latency_ratio:.2f}x faster\n")
        
        if custom.get('throughput_tokens_per_sec'):
            thr_ratio = custom['throughput_tokens_per_sec'] / pytorch['throughput_tokens_per_sec']
            if thr_ratio > 1:
                f.write(f"- **Throughput Advantage**: Custom processes {thr_ratio:.2f}x more tokens/sec\n")
            else:
                f.write(f"- **Throughput Gap**: PyTorch achieves {1/thr_ratio:.2f}x higher throughput\n")
        
        f.write("- PyTorch uses highly optimized kernel libraries (cuBLAS, cuDNN)\n")
        f.write("- Custom implementation offers room for optimization\n")
    
    print(f"✓ Detailed report saved to: {report_path}\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")
    
    main()
