# benchmark_attention.py (updated)
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from attention import MultiHeadAttention, GroupedQueryAttention, attention, causal_attention

def benchmark_attention():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test cases
    test_cases = [
        {"batch": 8, "heads": 16, "seq_len": 512, "embed_dim": 64},
        {"batch": 16, "heads": 32, "seq_len": 256, "embed_dim": 64},
        {"batch": 4, "heads": 32, "seq_len": 1024, "embed_dim": 32},
        {"batch": 32, "heads": 8, "seq_len": 512, "embed_dim": 128},
        {"batch": 8, "heads": 16, "seq_len": 2048, "embed_dim": 256},
        {"batch": 8, "heads": 16, "seq_len": 512, "embed_dim": 64},
    ]

    # Benchmark function for full models (simple wrapper that calls model(x))
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
            total_tokens = x.shape[0] * x.shape[1] * num_runs
            throughput = total_tokens / sum(times)
            return avg_time, throughput, std_dev

    # For fair comparison, benchmark complete pipeline including projections
    def benchmark_full_attention_fn(mha_model, x, attn_fn, num_runs=100):
        """Benchmark attention with projections included for both MHA and GQA-like classes.

        This function detects:
        - num_q_heads (or num_heads)
        - num_kv_heads (if present)
        and reshapes projections appropriately. If KV heads < Q heads (GQA case),
        it repeat_interleaves KV heads to match Q head groups before calling attn_fn.
        """
        B, S, H = x.shape

        # Determine q-heads and kv-heads for the model
        num_q_heads = getattr(mha_model, "num_heads", None)
        if num_q_heads is None:
            num_q_heads = getattr(mha_model, "num_q_heads", None)
        if num_q_heads is None:
            raise AttributeError("model must expose num_heads or num_q_heads")

        num_kv_heads = getattr(mha_model, "num_kv_heads", num_q_heads)

        # head dimension is derived from hidden / q_heads
        head_dim = H // num_q_heads

        times = []
        with torch.no_grad():
            # Warmup using projections + attn
            for _ in range(10):
                q = mha_model.q_proj(x).view(B, S, num_q_heads, head_dim).transpose(1, 2)
                k = mha_model.k_proj(x).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                v = mha_model.v_proj(x).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                # if KV heads are fewer, repeat_interleave to match Q heads (emulate GQA)
                if num_kv_heads != num_q_heads:
                    groups = num_q_heads // num_kv_heads
                    k = k.repeat_interleave(groups, dim=1)
                    v = v.repeat_interleave(groups, dim=1)
                out = attn_fn(q, k, v)
                _ = mha_model.out_proj(out.transpose(1, 2).contiguous().view(B, S, -1))

            if device.type == 'cuda':
                torch.cuda.synchronize()

            for _ in range(num_runs):
                if device.type == 'cuda':
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_evt.record()

                    q = mha_model.q_proj(x).view(B, S, num_q_heads, head_dim).transpose(1, 2)
                    k = mha_model.k_proj(x).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                    v = mha_model.v_proj(x).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                    if num_kv_heads != num_q_heads:
                        groups = num_q_heads // num_kv_heads
                        k = k.repeat_interleave(groups, dim=1)
                        v = v.repeat_interleave(groups, dim=1)

                    out = attn_fn(q, k, v)
                    _ = mha_model.out_proj(out.transpose(1, 2).contiguous().view(B, S, -1))

                    end_evt.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start_evt.elapsed_time(end_evt)
                    times.append(elapsed_ms / 1000.0)
                else:
                    start = time.time()
                    q = mha_model.q_proj(x).view(B, S, num_q_heads, head_dim).transpose(1, 2)
                    k = mha_model.k_proj(x).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                    v = mha_model.v_proj(x).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                    if num_kv_heads != num_q_heads:
                        groups = num_q_heads // num_kv_heads
                        k = k.repeat_interleave(groups, dim=1)
                        v = v.repeat_interleave(groups, dim=1)

                    out = attn_fn(q, k, v)
                    _ = mha_model.out_proj(out.transpose(1, 2).contiguous().view(B, S, -1))
                    end = time.time()
                    times.append(end - start)

            avg_time = sum(times) / len(times)
            variance = sum((t - avg_time)**2 for t in times) / len(times)
            std_dev = variance ** 0.5
            total_tokens = B * S * num_runs
            throughput = total_tokens / sum(times)
            return avg_time, throughput, std_dev

    # Store results
    results = []

    # Run benchmarks for each test case
    for idx, test_case in enumerate(test_cases, 1):
        B = test_case["batch"]
        num_heads = test_case["heads"]
        S = test_case["seq_len"]
        H = test_case["embed_dim"]
        num_kv_heads = max(1, num_heads // 4)  # GQA with 4x reduction

        print(f"\n{'='*60}")
        print(f"Test Case {idx}: Batch={B}, Heads={num_heads}, Seq_len={S}, Embed_dim={H}")
        print(f"{'='*60}")

        # Create models
        mha = MultiHeadAttention(num_heads, H).to(device)
        gqa = GroupedQueryAttention(H, num_heads, num_kv_heads).to(device)

        # Input
        x = torch.randn(B, S, H).to(device)

        # Precompute causal mask for this sequence length to avoid reallocating inside the inner loop
        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()

        def causal_with_mask(q, k, v, mask=mask):
            d_k = q.shape[-1]
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            scores = scores.masked_fill(mask, float('-inf'))
            weights = torch.softmax(scores, dim=-1)
            return torch.matmul(weights, v)

        # Benchmark all mechanisms (use the same attn_fn for a fair compare)
        # Standard (unmasked) attention (applied to projections)
        time_std, thru_std, std_std = benchmark_full_attention_fn(mha, x, attention)
        print(f"Standard Attention: {time_std*1000:.4f}ms (±{std_std*1000:.4f}ms), {thru_std:.2f} tokens/s")

        # Causal attention (using the precomputed mask)
        time_causal, thru_causal, std_causal = benchmark_full_attention_fn(mha, x, causal_with_mask)
        print(f"Causal Attention:   {time_causal*1000:.4f}ms (±{std_causal*1000:.4f}ms), {thru_causal:.2f} tokens/s")

        # MHA (benchmarked with the same unmasked standard attention to be comparable)
        time_mha, thru_mha, std_mha = benchmark_full_attention_fn(mha, x, attention)
        print(f"MHA:                {time_mha*1000:.4f}ms (±{std_mha*1000:.4f}ms), {thru_mha:.2f} tokens/s")

        # GQA (benchmarked with the same unmasked standard attention)
        time_gqa, thru_gqa, std_gqa = benchmark_full_attention_fn(gqa, x, attention)
        print(f"GQA (kv={num_kv_heads}):        {time_gqa*1000:.4f}ms (±{std_gqa*1000:.4f}ms), {thru_gqa:.2f} tokens/s")

        # Store results
        results.append({
            "Test Case": f"{idx}",
            "Batch": B,
            "Heads": num_heads,
            "Seq_len": S,
            "Embed_dim": H,
            "Standard_Latency": time_std * 1000,
            "Causal_Latency": time_causal * 1000,
            "MHA_Latency": time_mha * 1000,
            "GQA_Latency": time_gqa * 1000,
            "Standard_Throughput": thru_std,
            "Causal_Throughput": thru_causal,
            "MHA_Throughput": thru_mha,
            "GQA_Throughput": thru_gqa,
        })

    # Create results dataframe
    df_results = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("Summary Results - Latency (ms)")
    print(f"{'='*60}")
    print(df_results[["Test Case", "Standard_Latency", "Causal_Latency", "MHA_Latency", "GQA_Latency"]])
    
    print(f"\n{'='*60}")
    print("Summary Results - Throughput (tokens/s)")
    print(f"{'='*60}")
    print(df_results[["Test Case", "Standard_Throughput", "Causal_Throughput", "MHA_Throughput", "GQA_Throughput"]])

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Latency plot
    test_labels = [f"Test {i}" for i in range(1, len(test_cases) + 1)]
    x_pos = range(len(test_labels))
    width = 0.2
    
    axes[0].bar([p - 1.5*width for p in x_pos], df_results["Standard_Latency"], width, label="Standard", color='#1f77b4')
    axes[0].bar([p - 0.5*width for p in x_pos], df_results["Causal_Latency"], width, label="Causal", color='#ff7f0e')
    axes[0].bar([p + 0.5*width for p in x_pos], df_results["MHA_Latency"], width, label="MHA", color='#2ca02c')
    axes[0].bar([p + 1.5*width for p in x_pos], df_results["GQA_Latency"], width, label="GQA", color='#d62728')
    
    axes[0].set_xlabel("Test Case")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Attention Latency Comparison")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(test_labels)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Throughput plot
    axes[1].bar([p - 1.5*width for p in x_pos], df_results["Standard_Throughput"], width, label="Standard", color='#1f77b4')
    axes[1].bar([p - 0.5*width for p in x_pos], df_results["Causal_Throughput"], width, label="Causal", color='#ff7f0e')
    axes[1].bar([p + 0.5*width for p in x_pos], df_results["MHA_Throughput"], width, label="MHA", color='#2ca02c')
    axes[1].bar([p + 1.5*width for p in x_pos], df_results["GQA_Throughput"], width, label="GQA", color='#d62728')
    
    axes[1].set_xlabel("Test Case")
    axes[1].set_ylabel("Throughput (tokens/s)")
    axes[1].set_title("Attention Throughput Comparison")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(test_labels)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('visualization', exist_ok=True)
    plt.savefig('visualization/attention_comparison_all_tests.png', dpi=150)
    print(f"\nPlot saved to visualization/attention_comparison_all_tests.png")

if __name__ == "__main__":
    benchmark_attention()
