// fp16 @ fp16 -> fp16|f32 row-major GEMM via cuBLASLt, for the EXL3
// reconstruct+cuBLAS prefill path (QUANT-EXL3 W6). The reference
// (exllamav3/exllamav3_ext/hgemm.cu) uses cublasGemmEx; this tree links
// cublasLt only, so the same GEMM goes through cublasLtMatmul with the
// heuristic cache that the bf16/f32 lanes already use (gemm_plan_cache.h).
//
// a is fp16 [M, K] row-major, b is fp16 [K, N] row-major, c is [M, N]
// row-major and may be fp16 or f32. Compute type is CUBLAS_COMPUTE_32F
// with f32 scale, matching the reference's CUBLAS_COMPUTE_32F /
// CUBLAS_GEMM_DEFAULT_TENSOR_OP.
#ifndef VT_CUDA_CUBLAS_LT_HGEMM_H_
#define VT_CUDA_CUBLAS_LT_HGEMM_H_

#include "vt/tensor.h"  // Tensor (includes vt/device.h for Queue)

namespace vt::cuda {

void CublasLtHgemm(Queue& q, Tensor& c, const Tensor& a, const Tensor& b);

}  // namespace vt::cuda

#endif  // VT_CUDA_CUBLAS_LT_HGEMM_H_
