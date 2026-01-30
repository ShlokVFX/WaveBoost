import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
from attention import MultiHeadAttention, GroupedQueryAttention

def benchmark_attention():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    B = 4  # batch size
    S = 512  # sequence length
    H = 768  # hidden dim
    num_heads = 12  # for MHA
    num_kv_heads = 4  # for GQA, fewer KV heads

    # Create models
    mha = MultiHeadAttention(num_heads, H).to(device)
    gqa = GroupedQueryAttention(H, num_heads, num_kv_heads).to(device)
    # For correctness check, GQA with same num heads as MHA
    gqa_same = GroupedQueryAttention(H, num_heads, num_heads).to(device)
    # Copy weights to make them equivalent
    gqa_same.q_proj.weight.data = mha.q_proj.weight.data.clone()
    gqa_same.q_proj.bias.data = mha.q_proj.bias.data.clone()
    gqa_same.k_proj.weight.data = mha.k_proj.weight.data.clone()
    gqa_same.k_proj.bias.data = mha.k_proj.bias.data.clone()
    gqa_same.v_proj.weight.data = mha.v_proj.weight.data.clone()
    gqa_same.v_proj.bias.data = mha.v_proj.bias.data.clone()
    gqa_same.out_proj.weight.data = mha.out_proj.weight.data.clone()
    gqa_same.out_proj.bias.data = mha.out_proj.bias.data.clone()

    # Input
    x = torch.randn(B, S, H).to(device)

    # Correctness check
    mha.eval()
    gqa_same.eval()
    with torch.no_grad():
        out_mha = mha(x)
        out_gqa_same = gqa_same(x)
        correctness = torch.allclose(out_mha, out_gqa_same, atol=1e-6)
        print(f"Correctness (MHA vs GQA with same heads): {correctness}")

    # Benchmark function
    def benchmark(model, x, num_runs=100):
        model.eval()
        times = []
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            for _ in range(num_runs):
                if device.type == 'cuda':
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_evt.record()
                    _ = model(x)
                    end_evt.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start_evt.elapsed_time(end_evt)
                    times.append(elapsed_ms / 1000.0)  # to seconds
                else:
                    start = time.time()
                    _ = model(x)
                    end = time.time()
                    times.append(end - start)

            avg_time = sum(times) / len(times)
            variance = sum((t - avg_time)**2 for t in times) / len(times)
            std_dev = variance ** 0.5
            total_tokens = B * S * num_runs
            throughput = total_tokens / sum(times)
            return avg_time, throughput, std_dev

    # Benchmark MHA
    time_mha, thru_mha, std_mha = benchmark(mha, x)
    print(f"MHA: Avg time {time_mha*1000:.2f}ms (±{std_mha*1000:.2f}ms), Throughput {thru_mha:.2f} tokens/s")

    # Benchmark GQA
    time_gqa, thru_gqa, std_gqa = benchmark(gqa, x)
    print(f"GQA (kv_heads={num_kv_heads}): Avg time {time_gqa*1000:.2f}ms (±{std_gqa*1000:.2f}ms), Throughput {thru_gqa:.2f} tokens/s")

    # Memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = mha(x)
        mem_mha = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = gqa(x)
        mem_gqa = torch.cuda.max_memory_allocated()
        print(f"MHA Memory: {mem_mha / 1024**2:.2f} MB")
        print(f"GQA Memory: {mem_gqa / 1024**2:.2f} MB")

    # Declare who is faster
    if time_mha < time_gqa:
        print("MHA is faster than GQA.")
    elif time_gqa < time_mha:
        print("GQA is faster than MHA.")
    else:
        print("MHA and GQA have similar latency.")

    # Plot latency comparison
    plt.figure(figsize=(8, 5))
    attentions = ['MHA', f'GQA (kv_heads={num_kv_heads})']
    latencies = [time_mha * 1000, time_gqa * 1000]
    errors = [std_mha * 1000, std_gqa * 1000]
    plt.bar(attentions, latencies, yerr=errors, capsize=5, color=['blue', 'green'])
    plt.ylabel('Average Latency (ms)')
    plt.title('Attention Mechanisms Latency Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('visualization/attention_latency_comparison.png')
    print("Plot saved to visualization/attention_latency_comparison.png")

    # Simple diagram description (text-based)
    print("\nAttention Diagrams Description:")
    print("MHA (Multi-Head Attention): Each head has its own Q, K, V projections. All heads share the same KV for computation.")
    print("GQA (Grouped Query Attention): Queries are grouped, sharing KV heads. Reduces KV memory but may increase computation for repeating.")

if __name__ == "__main__":
    benchmark_attention()