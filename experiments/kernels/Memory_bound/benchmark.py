import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"

import torch
import time
import csv
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline

# --------------------------------
# Output path
# --------------------------------
output_path = "./vecadd_benchmark"
os.makedirs(output_path, exist_ok=True)

# --------------------------------
# Inline CUDA kernel
# --------------------------------
cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vecadd_kernel(const float4* a,
                              const float4* b,
                              float4* c,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 av = a[idx];
        float4 bv = b[idx];
        c[idx] = make_float4(
            av.x + bv.x,
            av.y + bv.y,
            av.z + bv.z,
            av.w + bv.w
        );
    }
}

void vecadd_cuda(torch::Tensor a,
                 torch::Tensor b,
                 torch::Tensor c) {

    const int n = a.numel() / 4;
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    vecadd_kernel<<<blocks, threads>>>(
        reinterpret_cast<float4*>(a.data_ptr<float>()),
        reinterpret_cast<float4*>(b.data_ptr<float>()),
        reinterpret_cast<float4*>(c.data_ptr<float>()),
        n
    );
}
 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vecadd_cuda", &vecadd_cuda, "VecAdd CUDA");
}
'''

# --------------------------------
# Compile extension
# --------------------------------
vecadd = load_inline(
    name="vecadd_ext",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=None,
    extra_cuda_cflags=["--use_fast_math"],
    verbose=False,
)

# --------------------------------
# Test cases
# --------------------------------
sizes = [
    2**20,
    2**22,
    2**23,
    2**25,
    2**26,
    2**29,
    2**30,
]

results = []

print("\nRunning VecAdd Benchmark (RTX 3060, SM 8.6)\n")

for N in sizes:
    # ensure divisible by 4
    N = (N // 4) * 4

    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    c = torch.empty_like(a)

    # Warmup
    for _ in range(10):
        vecadd.vecadd_cuda(a, b, c)
        torch.add(a, b)

    torch.cuda.synchronize()

    iters = 50 if N >= 2**26 else 100

    # Custom kernel
    start = time.time()
    for _ in range(iters):
        vecadd.vecadd_cuda(a, b, c)
    torch.cuda.synchronize()
    custom_us = (time.time() - start) / iters * 1e6

    # torch.add
    start = time.time()
    for _ in range(iters):
        torch.add(a, b)
    torch.cuda.synchronize()
    torch_us = (time.time() - start) / iters * 1e6

    results.append((N, custom_us, torch_us))

    print(f"n = {N:,}")
    print(f"  Custom VecAdd : {custom_us:.2f} us")
    print(f"  torch.add    : {torch_us:.2f} us\n")

# --------------------------------
# Save CSV
# --------------------------------
csv_path = os.path.join(output_path, "results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "Custom_VecAdd_us", "Torch_Add_us"])
    writer.writerows(results)

# --------------------------------
# Save plot
# --------------------------------
sizes_plot = [r[0] for r in results]
custom_plot = [r[1] for r in results]
torch_plot = [r[2] for r in results]

plt.figure(figsize=(9, 6))
plt.plot(sizes_plot, custom_plot, marker="o", label="Custom CUDA VecAdd")
plt.plot(sizes_plot, torch_plot, marker="o", label="torch.add")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Vector Size")
plt.ylabel("Time per call (µs)")
plt.title("VecAdd Performance: Custom CUDA vs PyTorch (RTX 3060)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_path, "vecadd_benchmark.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("Benchmark finished.")
print(f"Results saved to {os.path.abspath(output_path)}")
