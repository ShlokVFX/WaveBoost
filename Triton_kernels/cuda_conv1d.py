"""
CUDA 1D Convolution - All-in-one integration
Compiles and wraps the CUDA kernel in a single file.
"""

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
import torch


# CUDA kernel source code embedded
CUDA_KERNEL_SOURCE = r"""
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void conv1d_kernel_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t N,
    size_t K
) {
    extern __shared__ float sA[];

    int tid = threadIdx.x;
    int base = blockIdx.x * BLOCK_SIZE;
    int gid  = base + tid;
    int radius = K >> 1;

    for (int idx = tid; idx < BLOCK_SIZE + K - 1; idx += BLOCK_SIZE) {
        int a_idx = base + idx - radius;
        sA[idx] = (a_idx >= 0 && a_idx < N) ? A[a_idx] : 0.0f;
    }

    __syncthreads();

    // Convolution computation
    if (gid < N) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            sum += sA[tid + j] * B[j];
        }
        C[gid] = sum;
    }
}

extern "C" void solution(
    const float* A,
    const float* B,
    float* C,
    size_t N,
    size_t K
) {
    constexpr int BLOCK_SIZE = 1024/2;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem_bytes = (BLOCK_SIZE + K - 1) * sizeof(float);
    conv1d_kernel_shared<BLOCK_SIZE>
        <<<blocks, BLOCK_SIZE, smem_bytes>>>(A, B, C, N, K);
}
"""


class CUDAConv1D:
    """CUDA 1D Convolution kernel wrapper with on-demand compilation"""
    
    def __init__(self, compute_capability: str = "sm_86"):
        self.lib = None
        self.compute_capability = compute_capability
        self._compile_and_load()
    
    def _compile_cuda_kernel(self, cuda_file: Path, output_lib: Path) -> bool:
        """Compile CUDA kernel to shared library"""
        # Verify nvcc is available
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                print("ERROR: nvcc not found in PATH")
                return False
        except FileNotFoundError:
            print("ERROR: nvcc not found. Make sure CUDA Toolkit is installed and in PATH")
            return False
        
        print(f"Compiling CUDA kernel: {cuda_file.name}...")
        
        if sys.platform == "win32":
            cmd = [
                "nvcc",
                "-shared",
                "-Xcompiler", "/MD",
                str(cuda_file),
                "-o", str(output_lib),
                f"-arch={self.compute_capability}",
            ]
        else:
            cmd = [
                "nvcc",
                "-shared",
                "-Xcompiler", "-fPIC",
                str(cuda_file),
                "-o", str(output_lib),
                f"-arch={self.compute_capability}",
            ]
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Compilation failed:")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            return False
        
        print(f"✓ CUDA kernel compiled: {output_lib}")
        return True
    
    def _compile_and_load(self):
        """Compile CUDA kernel and load the library"""
        # Create temporary CUDA source file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(CUDA_KERNEL_SOURCE)
            cuda_temp_path = Path(f.name)
        
        try:
            # Determine output path
            if sys.platform == "win32":
                lib_name = "conv1d_cuda.dll"
            else:
                lib_name = "conv1d_cuda.so"
            
            # Use temp directory for the compiled library
            lib_path = Path(tempfile.gettempdir()) / lib_name
            
            # Compile
            if not self._compile_cuda_kernel(cuda_temp_path, lib_path):
                raise RuntimeError("Failed to compile CUDA kernel")
            
            # Load library
            try:
                self.lib = ctypes.CDLL(str(lib_path))
            except OSError as e:
                raise RuntimeError(f"Failed to load compiled CUDA library: {e}")
            
            # Get the solution function
            self.solution_func = self.lib.solution
            
            # Set function signature
            self.solution_func.argtypes = [
                ctypes.c_void_p,  # A_ptr
                ctypes.c_void_p,  # B_ptr
                ctypes.c_void_p,  # C_ptr
                ctypes.c_size_t,  # N
                ctypes.c_size_t,  # K
            ]
            self.solution_func.restype = None
            
            print("✓ CUDA kernel loaded successfully")
            
        finally:
            # Clean up temporary CUDA source file
            try:
                cuda_temp_path.unlink()
            except:
                pass
    
    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, K: int):
        """Execute CUDA kernel"""
        if not A.is_cuda:
            raise ValueError("Input tensor A must be on CUDA device")
        if not B.is_cuda:
            raise ValueError("Input tensor B must be on CUDA device")
        if not C.is_cuda:
            raise ValueError("Output tensor C must be on CUDA device")
        
        # Get data pointers
        A_ptr = A.data_ptr()
        B_ptr = B.data_ptr()
        C_ptr = C.data_ptr()
        
        # Call CUDA kernel
        self.solution_func(A_ptr, B_ptr, C_ptr, N, K)
        
        # Synchronize
        torch.cuda.synchronize()


# Global instance
_cuda_conv1d: Optional[CUDAConv1D] = None


def get_cuda_conv1d(compute_capability: str = "sm_86") -> CUDAConv1D:
    """Get or create the CUDA convolution instance"""
    global _cuda_conv1d
    if _cuda_conv1d is None:
        _cuda_conv1d = CUDAConv1D(compute_capability=compute_capability)
    return _cuda_conv1d


def cuda_conv1d_solution(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, K: int):
    """Convenience function to call CUDA convolution"""
    conv = get_cuda_conv1d()
    return conv(A, B, C, N, K)


if __name__ == "__main__":
    print("Testing CUDA 1D Convolution kernel...")
    try:
        conv = get_cuda_conv1d()
        print("✓ CUDA kernel ready for use")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
