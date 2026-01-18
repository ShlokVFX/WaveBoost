"""
Flash Attention Benchmark Visualization
Creates comprehensive graphs comparing custom vs PyTorch baseline
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import csv

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'custom': '#FF6B6B', 'pytorch': '#4ECDC4'}

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


def create_comparison_charts(custom, pytorch):
    """Create comprehensive comparison visualizations"""
    
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # ============ Chart 1: Latency Comparison (Bar Chart) ============
    fig, ax = plt.subplots(figsize=(10, 6))
    
    implementations = ['Custom FA2', 'PyTorch FA2']
    latencies = [custom['latency_ms'], pytorch['latency_ms']]
    
    bars = ax.bar(implementations, latencies, color=[colors['custom'], colors['pytorch']], 
                   width=0.6, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.4f} ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Flash Attention 2: Latency Comparison\n(Lower is Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(latencies) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add speedup annotation
    speedup = pytorch['latency_ms'] / custom['latency_ms']
    ax.text(0.5, max(latencies) * 1.1, f'PyTorch is {speedup:.1f}x faster', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: latency_comparison.png")
    plt.close()
    
    # ============ Chart 2: Throughput Comparison (Bar Chart) ============
    fig, ax = plt.subplots(figsize=(10, 6))
    
    throughputs = [custom['throughput_tokens_per_sec'], pytorch['throughput_tokens_per_sec']]
    throughputs_m = [x / 1e6 for x in throughputs]
    
    bars = ax.bar(implementations, throughputs_m, color=[colors['custom'], colors['pytorch']], 
                   width=0.6, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, thr in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{thr/1e6:.2f}M\n({thr:,.0f})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Throughput (M tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Flash Attention 2: Throughput Comparison\n(Higher is Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(throughputs_m) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add speedup annotation
    speedup = pytorch['throughput_tokens_per_sec'] / custom['throughput_tokens_per_sec']
    ax.text(0.5, max(throughputs_m) * 1.2, f'PyTorch is {speedup:.1f}x higher throughput', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: throughput_comparison.png")
    plt.close()
    
    # ============ Chart 3: Memory Usage Comparison (Bar Chart) ============
    fig, ax = plt.subplots(figsize=(10, 6))
    
    memory_vals = [custom['peak_memory_mb'], 24]  # ~24 MB for PyTorch
    
    bars = ax.bar(implementations, memory_vals, color=[colors['custom'], colors['pytorch']], 
                   width=0.6, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, mem in zip(bars, memory_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.2f} MB',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Flash Attention 2: Memory Usage Comparison\n(Lower is Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(memory_vals) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add savings annotation
    savings = (memory_vals[1] - memory_vals[0]) / memory_vals[1] * 100
    ax.text(0.5, max(memory_vals) * 1.1, f'Custom saves {savings:.1f}% memory', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: memory_comparison.png")
    plt.close()
    
    # ============ Chart 4: Performance Radar Chart ============
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics (0-100 scale, higher is better)
    latency_score_custom = (pytorch['latency_ms'] / custom['latency_ms']) / (pytorch['latency_ms'] / custom['latency_ms']) * 100
    latency_score_pytorch = 100
    
    thr_score_custom = (custom['throughput_tokens_per_sec'] / pytorch['throughput_tokens_per_sec']) * 100
    thr_score_pytorch = 100
    
    mem_score_custom = 100
    mem_score_pytorch = (custom['peak_memory_mb'] / 24) * 100
    
    categories = ['Speed\n(Latency)', 'Throughput', 'Memory\nEfficiency']
    custom_scores = [latency_score_custom, thr_score_custom, mem_score_custom]
    pytorch_scores = [latency_score_pytorch, thr_score_pytorch, mem_score_pytorch]
    
    # Complete the circle
    custom_scores += custom_scores[:1]
    pytorch_scores += pytorch_scores[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, custom_scores, 'o-', linewidth=2, label='Custom FA2', color=colors['custom'], markersize=8)
    ax.fill(angles, custom_scores, alpha=0.25, color=colors['custom'])
    
    ax.plot(angles, pytorch_scores, 'o-', linewidth=2, label='PyTorch FA2', color=colors['pytorch'], markersize=8)
    ax.fill(angles, pytorch_scores, alpha=0.25, color=colors['pytorch'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 120)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, linewidth=0.5)
    
    ax.set_title('Flash Attention 2: Multi-Metric Comparison\n(Normalized Performance)', 
                 fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: performance_radar.png")
    plt.close()
    
    # ============ Chart 5: Performance Summary Dashboard ============
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Latency
    ax1 = fig.add_subplot(gs[0, 0])
    impls = ['Custom', 'PyTorch']
    lats = [custom['latency_ms'], pytorch['latency_ms']]
    bars = ax1.bar(impls, lats, color=[colors['custom'], colors['pytorch']], edgecolor='black', linewidth=1.5)
    for bar, lat in zip(bars, lats):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{lat:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('ms', fontweight='bold')
    ax1.set_title('Latency', fontweight='bold', fontsize=11)
    ax1.set_ylim(0, max(lats) * 1.15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Throughput
    ax2 = fig.add_subplot(gs[0, 1])
    thrs = [custom['throughput_tokens_per_sec']/1e6, pytorch['throughput_tokens_per_sec']/1e6]
    bars = ax2.bar(impls, thrs, color=[colors['custom'], colors['pytorch']], edgecolor='black', linewidth=1.5)
    for bar, thr in zip(bars, thrs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{thr:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('M Tokens/sec', fontweight='bold')
    ax2.set_title('Throughput', fontweight='bold', fontsize=11)
    ax2.set_ylim(0, max(thrs) * 1.15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Memory
    ax3 = fig.add_subplot(gs[1, 0])
    mems = [custom['peak_memory_mb'], 24]
    bars = ax3.bar(impls, mems, color=[colors['custom'], colors['pytorch']], edgecolor='black', linewidth=1.5)
    for bar, mem in zip(bars, mems):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mem:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylabel('MB', fontweight='bold')
    ax3.set_title('Peak Memory', fontweight='bold', fontsize=11)
    ax3.set_ylim(0, max(mems) * 1.15)
    ax3.grid(axis='y', alpha=0.3)
    
    # Efficiency (Tokens per MB)
    ax4 = fig.add_subplot(gs[1, 1])
    effs = [custom['throughput_tokens_per_sec'] / custom['peak_memory_mb'],
            pytorch['throughput_tokens_per_sec'] / 24]
    effs_k = [e/1000 for e in effs]
    bars = ax4.bar(impls, effs_k, color=[colors['custom'], colors['pytorch']], edgecolor='black', linewidth=1.5)
    for bar, eff in zip(bars, effs):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{eff/1000:.1f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_ylabel('K Tokens/sec per MB', fontweight='bold')
    ax4.set_title('Efficiency', fontweight='bold', fontsize=11)
    ax4.set_ylim(0, max(effs_k) * 1.15)
    ax4.grid(axis='y', alpha=0.3)
    
    # Speedup factors
    ax5 = fig.add_subplot(gs[2, :])
    metrics = ['Latency Speedup', 'Throughput Speedup', 'Memory Ratio']
    speedups = [
        pytorch['latency_ms'] / custom['latency_ms'],
        pytorch['throughput_tokens_per_sec'] / custom['throughput_tokens_per_sec'],
        24 / custom['peak_memory_mb']
    ]
    colors_speedup = ['#FF6B6B', '#FF6B6B', '#4ECDC4']  # Red for slower, Green for better
    
    bars = ax5.barh(metrics, speedups, color=colors_speedup, edgecolor='black', linewidth=1.5)
    for bar, speedup, metric in zip(bars, speedups, metrics):
        width = bar.get_width()
        if metric == 'Memory Ratio':
            label = f'{speedup:.2f}x less memory'
        else:
            label = f'{speedup:.2f}x'
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f' {label}', ha='left', va='center', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Factor', fontweight='bold')
    ax5.set_title('Relative Performance', fontweight='bold', fontsize=11)
    ax5.grid(axis='x', alpha=0.3)
    
    fig.suptitle('Flash Attention 2 Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: performance_dashboard.png")
    plt.close()
    
    # ============ Chart 6: Comparison Table as Image ============
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Custom FA2', 'PyTorch FA2', 'Ratio', 'Winner'],
        ['Latency (ms)', f'{custom["latency_ms"]:.4f}', f'{pytorch["latency_ms"]:.4f}', 
         f'{pytorch["latency_ms"]/custom["latency_ms"]:.2f}x', 'PyTorch ✗'],
        ['Throughput (T/s)', f'{custom["throughput_tokens_per_sec"]:,.0f}', 
         f'{pytorch["throughput_tokens_per_sec"]:,.0f}', 
         f'{pytorch["throughput_tokens_per_sec"]/custom["throughput_tokens_per_sec"]:.2f}x', 'PyTorch ✗'],
        ['Peak Memory (MB)', f'{custom["peak_memory_mb"]:.2f}', '~24', 
         f'{24/custom["peak_memory_mb"]:.2f}x', 'Custom ✓'],
        ['Efficiency (T/s/MB)', f'{custom["throughput_tokens_per_sec"]/custom["peak_memory_mb"]:,.0f}', 
         f'{pytorch["throughput_tokens_per_sec"]/24:,.0f}', 
         f'{(pytorch["throughput_tokens_per_sec"]/24)/(custom["throughput_tokens_per_sec"]/custom["peak_memory_mb"]):.2f}x', 'PyTorch ✗']
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.2, 0.2, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
            table[(i, j)].set_text_props(weight='bold', fontsize=11)
    
    plt.title('Flash Attention 2: Detailed Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_table.png")
    plt.close()


def main():
    print("\n" + "=" * 80)
    print("FLASH ATTENTION BENCHMARK VISUALIZATION")
    print("=" * 80 + "\n")
    
    # Load custom results
    custom = load_custom_benchmarks()
    
    if not custom:
        print("ERROR: Could not load custom benchmark results")
        return
    
    print(f"[Custom Implementation]")
    print(f"  Latency: {custom['latency_ms']:.4f} ms")
    print(f"  Throughput: {custom['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"  Peak Memory: {custom['peak_memory_mb']:.2f} MB")
    
    print(f"\n[Benchmarking PyTorch...]")
    pytorch = benchmark_pytorch_quick()
    
    print(f"\n[PyTorch Baseline]")
    print(f"  Latency: {pytorch['latency_ms']:.4f} ms")
    print(f"  Throughput: {pytorch['throughput_tokens_per_sec']:,.0f} tokens/sec")
    
    print(f"\n[Creating Visualizations...]")
    create_comparison_charts(custom, pytorch)
    
    viz_dir = Path(__file__).parent / "visualizations"
    print(f"\n✓ All visualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    print("  1. latency_comparison.png")
    print("  2. throughput_comparison.png")
    print("  3. memory_comparison.png")
    print("  4. performance_radar.png")
    print("  5. performance_dashboard.png")
    print("  6. comparison_table.png")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")
    
    main()
