// #837 test/product accessor for file-local GetBlas.
// Defined in src/vt/rocm/rocm_matmul_hipblaslt.hip (HIP builds only).
#pragma once

#if defined(VLLM_CPP_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace vt::rocm {

// Executes production GetBlas: real HipBlasHooks + static thread_local tls_slots.
hipblasHandle_t ProductGetBlasHandle(int device, hipStream_t stream);

// Exact HipBlasHooks::StreamIsCapturing (not a parallel hipStreamIsCapturing).
bool ProductGetBlasStreamIsCapturing(hipStream_t stream);

}  // namespace vt::rocm
#endif
