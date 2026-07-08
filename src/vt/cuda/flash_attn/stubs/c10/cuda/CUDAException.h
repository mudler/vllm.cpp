#pragma once
#include <cuda_runtime.h>
#define C10_CUDA_CHECK(EXPR) do { cudaError_t _e = (EXPR); (void)_e; } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do { (void)cudaGetLastError(); } while (0)
