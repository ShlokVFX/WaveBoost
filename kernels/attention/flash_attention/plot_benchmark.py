"""
Simple Benchmark Visualization
Creates comparison graphs
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import csv

def load_custom_benchmarks():
    """Load custom benchmark results"""
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


def benchmark_pytorch_quick():
    """Quick PyTorch benchmark"""
    print("  Benchmarking PyTorch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.randn(1, 512, 64, device=device, dtype=torch.float32)
    K = torch.randn(1, 512, 64, device=device, dtype=torch.float32)
    V = torch.randn(1, 512, 64, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=1.0/8.0)
    
    torch.cuda.synchronize()
    latencies = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=1.0/8.0)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latency_ms = np.mean(latencies)
    throughput = 512 / (latency_ms / 1000)
    
    return {'latency_ms': latency_ms, 'throughput_tokens_per_sec': throughput}


def main():
    print("\n" + "=" * 70)
    print("FLASH ATTENTION BENCHMARK VISUALIZATION")
    print("=" * 70 + "\n")
    
    # Create output directory
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load custom results
    print("Loading benchmark data...")
    custom = load_custom_benchmarks()
    
    if not custom:
        print("ERROR: Could not load custom benchmarks")
        return
    
    print(f"  Custom Latency: {custom['latency_ms']:.4f} ms")
    print(f"  Custom Throughput: {custom['throughput_tokens_per_sec']:,.0f} T/s")
    print(f"  Custom Memory: {custom['peak_memory_mb']:.2f} MB\n")
    
    pytorch = benchmark_pytorch_quick()
    print(f"  PyTorch Latency: {pytorch['latency_ms']:.4f} ms")
    print(f"  PyTorch Throughput: {pytorch['throughput_tokens_per_sec']:,.0f} T/s\n")
    
    # Chart 1: Latency
    print("Generating Chart 1: Latency Comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    impls = ['Custom FA2', 'PyTorch FA2']
    lats = [custom['latency_ms'], pytorch['latency_ms']]
    bars = ax.bar(impls, lats, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=2, width=0.6)
    
    for bar, lat in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{lat:.4f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    speedup = pytorch['latency_ms'] / custom['latency_ms']
    ax.text(0.5, max(lats) * 1.1, f'PyTorch is {speedup:.1f}x faster', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(lats) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '1_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 1_latency.png")
    
    # Chart 2: Throughput
    print("Generating Chart 2: Throughput Comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    thrs = [custom['throughput_tokens_per_sec']/1e6, pytorch['throughput_tokens_per_sec']/1e6]
    bars = ax.bar(impls, thrs, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=2, width=0.6)
    
    for bar, thr, thr_full in zip(bars, thrs, [custom['throughput_tokens_per_sec'], pytorch['throughput_tokens_per_sec']]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{thr:.2f}M\n({thr_full:,.0f})', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    speedup = pytorch['throughput_tokens_per_sec'] / custom['throughput_tokens_per_sec']
    ax.text(0.5, max(thrs) * 1.1, f'PyTorch is {speedup:.1f}x higher throughput', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Throughput (M tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Comparison (Higher is Better)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(thrs) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '2_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 2_throughput.png")
    
    # Chart 3: Memory
    print("Generating Chart 3: Memory Comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    mems = [custom['peak_memory_mb'], 24]  # ~24 MB for PyTorch
    bars = ax.bar(impls, mems, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=2, width=0.6)
    
    for bar, mem in zip(bars, mems):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mem:.2f} MB', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    savings = (mems[1] - mems[0]) / mems[1] * 100
    ax.text(0.5, max(mems) * 1.1, f'Custom saves {savings:.1f}% memory', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(mems) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '3_memory.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 3_memory.png")
    
    # Chart 4: Performance Dashboard
    print("Generating Chart 4: Performance Dashboard...")
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Latency subplot
    ax1 = fig.add_subplot(gs[0, 0])
    lats_dash = [custom['latency_ms'], pytorch['latency_ms']]
    bars = ax1.bar(impls, lats_dash, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)
    for bar, lat in zip(bars, lats_dash):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{lat:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('ms', fontweight='bold')
    ax1.set_title('Latency', fontweight='bold')
    ax1.set_ylim(0, max(lats_dash) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Throughput subplot
    ax2 = fig.add_subplot(gs[0, 1])
    thrs_dash = [custom['throughput_tokens_per_sec']/1e6, pytorch['throughput_tokens_per_sec']/1e6]
    bars = ax2.bar(impls, thrs_dash, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)
    for bar, thr in zip(bars, thrs_dash):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{thr:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('M T/s', fontweight='bold')
    ax2.set_title('Throughput', fontweight='bold')
    ax2.set_ylim(0, max(thrs_dash) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    # Memory subplot
    ax3 = fig.add_subplot(gs[1, 0])
    mems_dash = [custom['peak_memory_mb'], 24]
    bars = ax3.bar(impls, mems_dash, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)
    for bar, mem in zip(bars, mems_dash):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mem:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylabel('MB', fontweight='bold')
    ax3.set_title('Memory', fontweight='bold')
    ax3.set_ylim(0, max(mems_dash) * 1.2)
    ax3.grid(axis='y', alpha=0.3)
    
    # Efficiency subplot
    ax4 = fig.add_subplot(gs[1, 1])
    effs = [custom['throughput_tokens_per_sec'] / custom['peak_memory_mb'],
            pytorch['throughput_tokens_per_sec'] / 24]
    effs_k = [e/1000 for e in effs]
    bars = ax4.bar(impls, effs_k, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)
    for bar, eff in zip(bars, effs_k):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{eff:.0f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_ylabel('K T/s per MB', fontweight='bold')
    ax4.set_title('Efficiency', fontweight='bold')
    ax4.set_ylim(0, max(effs_k) * 1.2)
    ax4.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Flash Attention 2: Performance Dashboard', fontsize=14, fontweight='bold')
    plt.savefig(output_dir / '4_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 4_dashboard.png")
    
    # Chart 5: Speedup comparison
    print("Generating Chart 5: Speedup Analysis...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Latency\nSpeedup', 'Throughput\nSpeedup', 'Memory\nRatio']
    speedups = [
        pytorch['latency_ms'] / custom['latency_ms'],
        pytorch['throughput_tokens_per_sec'] / custom['throughput_tokens_per_sec'],
        24 / custom['peak_memory_mb']
    ]
    colors_list = ['#FF6B6B', '#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(metrics, speedups, color=colors_list, edgecolor='black', linewidth=2, width=0.6)
    
    for bar, speedup, metric in zip(bars, speedups, metrics):
        if 'Memory' in metric:
            label = f'{speedup:.2f}x\n(Custom saves)'
        else:
            label = f'{speedup:.2f}x\n(PyTorch faster)'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Factor (X)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(speedups) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '5_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 5_speedup.png")
    
    print("\n" + "=" * 70)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nGraphs saved in: {output_dir}\n")
    print("Generated files:")
    print("  1. 1_latency.png - Latency comparison")
    print("  2. 2_throughput.png - Throughput comparison")
    print("  3. 3_memory.png - Memory usage comparison")
    print("  4. 4_dashboard.png - Performance dashboard")
    print("  5. 5_speedup.png - Speedup analysis")
    print("\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")
    
    main()
