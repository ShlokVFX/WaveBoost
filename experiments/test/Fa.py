import os
import math
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = '''
__global__ void flash_attn(const float* Q, const float* K, const float* V, const int N, const int d, const int Tc, const int Tr, const int Bc, const int Br, const float scale, float* l, float *m, float* O) {
    extern __shared__ float sram[]; float* Qi = sram, *Kj = &sram[Bc*d], *Vj = &sram[2*Bc*d], *S = &sram[3*Bc*d]; // SRAM Tiles for Q, K, V, & S
    int tx = threadIdx.x, qkv_idx = (blockIdx.x*gridDim.y + blockIdx.y)*N*d, lm_idx = qkv_idx/d; // Thread ID & Base Indices for Q/K/V & l/m
    for (int j = 0; j < Tc; j++) { // Loop over Column Tiles of K/V
        for (int x = 0; x < d; x++) Kj[tx * d + x] = K[qkv_idx + j * Bc * d + tx * d + x], Vj[tx * d + x] = V[qkv_idx + j * Bc * d + tx * d + x]; // Load K/V Tiles into SRAM
        __syncthreads(); // Ensure K/V Tile is Fully Loaded
        for (int i = 0; i < Tr; i++) { // Loop over Row Tiles of Q
            for (int x = 0; x < d; x++) Qi[tx * d + x] = Q[qkv_idx + i * Bc * d + tx * d + x];
            float mi = m[lm_idx + i*Br + tx], li = l[lm_idx + i*Br + tx], row_m = -INFINITY, row_l = 0; // Run Softmax on Row Tile
            #pragma unroll
            for (int y = 0; y < Bc; y++) { float sum = 0; for (int x = 0; x < d; x += 4) { float4 qv = reinterpret_cast<float4*>(&Qi[tx * d + x])[0], kv = reinterpret_cast<float4*>(&Kj[y * d + x])[0]; sum += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w; } row_m = max(row_m, S[Bc * tx + y] = sum * scale);}
            for (int y = 0; y < Bc; y++) row_l += S[Bc * tx + y] = __expf(S[Bc * tx + y] - row_m); // Exponentiate Shifted Logits & Accumulate Row Sum
            float mi_new = max(mi, row_m), li_new = __expf(mi - mi_new) * li + __expf(row_m - mi_new) * row_l; // Update Running Max & Sum for Numerically Stable Softmax
            for (int x = 0; x < d; x++) { float pv = 0; for (int y = 0; y < Bc; y++) pv += S[Bc * tx + y] * Vj[y * d + x]; O[qkv_idx + i * Bc * d + tx * d + x] = (1 / li_new) * (__expf(mi - mi_new) * li * O[qkv_idx + i * Bc * d + tx * d + x] + __expf(row_m - mi_new) * pv);} // Weighted Accumulation into Output
            m[lm_idx + i * Br + tx] = mi_new, l[lm_idx + i * Br + tx] = li_new; // Update Running Max & Sum for Next Iteration
        }
        __syncthreads(); // Ensure All Threads Have Finished
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    cudaDeviceProp props; cudaGetDeviceProperties(&props, 0); const int B = Q.size(0), nh = Q.size(1), N = Q.size(2), d = Q.size(3), Bc = ceil(props.sharedMemPerBlock/(4 * d)), Br = std::min(Bc, d), Tc = ceil((float)N/Bc), Tr = ceil((float)N/Br);
    auto O = torch::zeros_like(Q); auto l = torch::zeros({B, nh, N}).to(Q.device()), m = torch::full({B, nh, N}, -INFINITY).to(Q.device()); dim3 grid(B, nh), block(Bc);
    flash_attn<<<grid, block, (3*Bc*d + Bc*Br)*sizeof(float)>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc, Tr, Bc, Br, 1.0f/sqrt(d), l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>());
    return O; // Returns Final Attention Output
}
'''
cpp_src = 'torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);'

build_dir = 'cuda'
if not os.path.exists(build_dir):
    os.mkdir(build_dir)

flash_attn = load_inline(
    name='flash_attn',
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=['forward'],
    with_cuda=True,
    extra_cuda_cflags=['-O2'],
    build_directory=f'./{build_dir}'
)

batch_size = 32
n_head = 12
seq_len = 64
head_embd = 32

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== Profiling Manual Attention ===')

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=3))

print('=== Profiling Tiny Flash Attention === ')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = flash_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=3))
