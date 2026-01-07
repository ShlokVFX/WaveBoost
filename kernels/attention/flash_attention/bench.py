import math
import os
import time
import csv
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"   # RTX 3060 (Ampere)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
# =========================
# Paths
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_ROOT = os.path.abspath(
    os.path.join(THIS_DIR, "..", "..", "..")
)

BENCH_ROOT = os.path.join(
    REPO_ROOT, "benchmarks", "flash_attention"
)

LAT_DIR  = os.path.join(BENCH_ROOT, "latency")
THR_DIR  = os.path.join(BENCH_ROOT, "throughput")
MEM_DIR  = os.path.join(BENCH_ROOT, "memory")
PROF_DIR = os.path.join(BENCH_ROOT, "profiling")

for d in [LAT_DIR, THR_DIR, MEM_DIR, PROF_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================
# Environment
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# =========================
# Load CUDA extension
# =========================
minimal_attn = load(
    name="minimal_attn",
    sources=["main.cpp", "flash.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# =========================
# Parameters
# =========================
batch_size = 16
n_head = 12
seq_len = 64
head_dim = 64
iters = 50

# =========================
# Inputs
# =========================
q = torch.randn(batch_size, n_head, seq_len, head_dim, device=device)
k = torch.randn(batch_size, n_head, seq_len, head_dim, device=device)
v = torch.randn(batch_size, n_head, seq_len, head_dim, device=device)

# =========================
# Reference attention
# =========================
def manual_attn(q, k, v):
    scale = 1.0 / math.sqrt(k.size(-1))
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

# =========================
# Warm-up
# =========================
for _ in range(10):
    manual_attn(q, k, v)
    minimal_attn.forward(q, k, v)
torch.cuda.synchronize()

# =========================
# Latency benchmark
# =========================
def benchmark(fn):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn(q, k, v)
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1e3  # ms

lat_manual = benchmark(manual_attn)
lat_flash = benchmark(minimal_attn.forward)

with open(os.path.join(LAT_DIR, "latency.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kernel", "latency_ms"])
    writer.writerow(["manual", lat_manual])
    writer.writerow(["flash", lat_flash])

# =========================
# Throughput
# =========================
tokens = batch_size * seq_len

thr_manual = tokens / (lat_manual / 1e3)
thr_flash = tokens / (lat_flash / 1e3)

with open(os.path.join(THR_DIR, "throughput.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kernel", "tokens_per_sec"])
    writer.writerow(["manual", thr_manual])
    writer.writerow(["flash", thr_flash])

# =========================
# Memory usage
# =========================
torch.cuda.reset_peak_memory_stats()
manual_attn(q, k, v)
torch.cuda.synchronize()
mem_manual = torch.cuda.max_memory_allocated() / (1024 ** 2)

torch.cuda.reset_peak_memory_stats()
minimal_attn.forward(q, k, v)
torch.cuda.synchronize()
mem_flash = torch.cuda.max_memory_allocated() / (1024 ** 2)

with open(os.path.join(MEM_DIR, "memory.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kernel", "peak_memory_mb"])
    writer.writerow(["manual", mem_manual])
    writer.writerow(["flash", mem_flash])

# =========================
# Profiling
# =========================
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_attn.forward(q, k, v)

with open(os.path.join(PROF_DIR, "profiler.txt"), "w") as f:
    f.write(
        prof.key_averages()
        .table(sort_by="cuda_time_total", row_limit=20)
    )

# =========================
# Correctness
# =========================
print(
    "correctness:",
    torch.allclose(
        minimal_attn.forward(q, k, v),
        manual_attn(q, k, v),
        rtol=0,
        atol=1e-2,
    ),
)
