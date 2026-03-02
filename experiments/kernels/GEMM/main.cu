// main.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
// #include "kernels/gemm_kernel.h"

template <typename T>
__global__ void gemm_v01_1(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc);

template <typename T>
void launch_gemm_kernel_v01_1(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

#define CHECK_LAST_CUDA_ERROR() {                                          \
    cudaError_t err = cudaGetLastError();                                 \
    if (err != cudaSuccess) {                                              \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__,           \
               cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

double get_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    size_t m = 1024, n = 1024, k = 1024;
    float alpha = 1.0f, beta = 0.0f;

    // Allocate and initialize host/device memory
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));

    for (size_t i = 0; i < m * k; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (size_t i = 0; i < k * n; ++i) h_B[i] = rand() / (float)RAND_MAX;
    for (size_t i = 0; i < m * n; ++i) h_C[i] = 0.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up (optional, to avoid initial overhead)
    launch_gemm_kernel_v01_1(m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, 0);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();

    // Time the kernel
    double start_time = get_time_seconds();

    launch_gemm_kernel_v01_1(m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, 0);

    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    double end_time = get_time_seconds();

    // Copy result back
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate TFLOPS
    double total_flops = 2.0 * m * n * k; // Each multiply-add counts as 2 FLOPs
    double time_seconds = end_time - start_time;
    double tflops = total_flops / (1e12 * time_seconds);

    printf("GEMM completed successfully!\n");
    printf("Time taken: %.6f seconds\n", time_seconds);
    printf("Performance: %.2f TFLOPS on RTX 3060\n", tflops);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
