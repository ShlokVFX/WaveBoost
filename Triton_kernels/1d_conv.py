import torch
import triton
import triton.language as tl
import triton.testing
import torch.nn as nn
import os

# Import CUDA wrappe
try:
    from cuda_conv1d import cuda_conv1d_solution, get_cuda_conv1d
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Warning: CUDA kernel not available: {e}")
    CUDA_AVAILABLE = False

# 1. Define Device
DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=1),
        # Added larger configs for larger N/K
        triton.Config({"BLOCK_N": 64,  "BLOCK_K": 256}, num_warps=4, num_stages=1), 
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=1),
    ],
    key=["N", "K"],
)
@triton.jit
def conv1d_autotuned(
    A_ptr, B_ptr, C_ptr,
    N, K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    radius = (K - 1) // 2
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Large K means this loop runs many times (8191 / BLOCK_K)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        b = tl.load(B_ptr + offs_k, mask=mask_k, other=0.0)
        
        base = offs_n[:, None] - radius
        a_idx = base + offs_k[None, :]
        a_mask = (a_idx >= 0) & (a_idx < N) & mask_n[:, None]
        
        a = tl.load(A_ptr + a_idx, mask=a_mask, other=0.0)
        acc += tl.sum(a * b[None, :], axis=1)

    tl.store(C_ptr + offs_n, acc, mask=mask_n)


def solution(A, B, C, N: int, K: int):
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), )
    conv1d_autotuned[grid](A, B, C, N, K)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[32768, 65536, 131072, 524288], 
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch', 'cuda'] if CUDA_AVAILABLE else ['triton', 'torch'],
        line_names=['Triton', 'Torch', 'CUDA'] if CUDA_AVAILABLE else ['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')] if CUDA_AVAILABLE else [('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='conv1d',
        args={'K': 8191} 
    ))
def benchmark(N, K, provider):
    # Initialize Data
    A = torch.randn(N, device=DEVICE, dtype=torch.float32)
    C = torch.empty_like(A)

    # PyTorch Layer with K=8191
    conv_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=K, padding='same', bias=False).to(DEVICE)
    A_torch = A.view(1, 1, -1)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: conv_layer(A_torch), 
            quantiles=quantiles
        )
        
    elif provider == 'triton':
        weight_triton = conv_layer.weight.squeeze().contiguous()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solution(A, weight_triton, C, N, K), 
            quantiles=quantiles
        )
    
    elif provider == 'cuda':
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernel not available")
        weight_cuda = conv_layer.weight.squeeze().contiguous()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cuda_conv1d_solution(A, weight_cuda, C, N, K),
            quantiles=quantiles
        )

    gbps = lambda ms: (2 * N + K) * A.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# --- Execution ---
output_path = './conv1d_results'
if not os.path.exists(output_path):
    os.makedirs(output_path)

benchmark.run(print_data=True, show_plots=False, save_path=output_path)
print(f"Benchmark finished. Results saved to {os.path.abspath(output_path)}")