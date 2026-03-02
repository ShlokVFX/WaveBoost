template <typename T , size_t BLOCK_SIZE>
__global__ void gemm_v01_1(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{

    // Compute the row and column of C that this thread is responsible for.
    const uint C_col_idx = blockIdx.x;
    const uint C_row_idx = blockIdx.y;

    __shared__ T As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE * BLOCK_SIZE];

    const uint threadCol = threadIdx.x % BLOCK_SIZE;
    const uint threadRow = threadIdx.x / BLOCK_SIZE;

    A += C_row_idx * BLOCK_SIZE * lda; //K is lda
    B += C_col_idx * BLOCK_SIZE;
    C += C_row_idx * BLOCK_SIZE * ldb + C_col_idx * BLOCK_SIZE; //N is ldb

    T sum{static_cast<T>(0)};

    for(int bkIdx = 0; bkIdx < lda; bkIdx += BLOCK_SIZE) {
        As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * lda + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * ldb + threadCol];

        __syncthreads();
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * ldb;

        for(int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            sum += As[threadRow * BLOCK_SIZE + dotIdx] * 
                   Bs[dotIdx * BLOCK_SIZE + threadCol];
        }
        __syncthreads();

        C[threadRow * ldb +  threadCol] = alpha + sum + beta * C[threadRow * ldb + threadCol];
    }
}

template <typename T>
void launch_gemm_kernel_v01_1(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    constexpr unsigned int BLOCK_SIZE{32U};
    dim3 const block_dim{BLOCK_SIZE, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v01_1<T, BLOCK_SIZE><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
}

// Explicit instantiation for float
template __global__ void gemm_v01_1<float, 32>(size_t, size_t, size_t, float, float const*,
                                             size_t, float const*, size_t, float, float*,
                                             size_t);

template void launch_gemm_kernel_v01_1<float>(
    size_t, size_t, size_t, float const*, float const*, size_t,
    float const*, size_t, float const*, float*, size_t, cudaStream_t);