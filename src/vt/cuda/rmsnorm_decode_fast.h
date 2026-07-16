// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// 1:1 port of upstream vLLM CUDA, see below.
//
// Env-flag plumbing for the decode-RMSNorm kernel-efficiency A/B, split into
// this pure-C++ (CUDA-free) header so the flag predicate is unit-testable on the
// CPU tier (tests/vt/test_rmsnorm_decode_fast.cpp). The kernel it selects is
// CUDA-only and lives in cuda_ops.cu (RmsNormRowFastKernel), a 1:1 port of
// vLLM's own vectorized CUDA rms-norm:
//   csrc/libtorch_stable/layernorm_kernels.cu:106-173 (fused_add_rms_norm_kernel
//   <scalar_t, width=8>) + launch :310-363 (block = min(hidden, 1024) when
//   num_tokens < 256; 16-byte _f16Vec loads; cub::BlockReduce<float,1024>),
//   with the _f16Vec packed bf16 add / f32 sum_squares of csrc/type_convert.cuh
//   :115-194, @ vLLM pin e24d1b24. This is the CUDA embodiment of the SAME
//   reduction/vectorization the production Inductor kernel runs
//   (triton_red_fused__to_copy_add..._rms_norm; the standalone-nsys-adjudicated
//   Phase-1 decode gap at the real 27B shape M x H=5120 is 3.18-3.56x, ours
//   8.44-8.53 us/launch vs vLLM 2.37-2.68 us — well over the 1.3x port bar).
//
// Why default OFF: the vectorized 1024-thread block reorders the f32 variance
// reduction vs the shipped 256-thread tree, so it is NOT bit-identical to
// RmsNormRowKernel (the isolated spike showed bf16-EXACT at c2-c16 but 2/163840
// elements 1-ULP at c32 on adversarial data — the same token-exactness hazard
// that keeps the fused attention preamble per-arch). It ships OFF until the
// 27B/35B token gates + an in-situ c16/c2 A/B prove the flip. VT_RMSNORM_DECODE_
// FAST=1 opts in. The launcher reads getenv per call (decode RMSNorm dispatch is
// coarse — 129 launches/step — so the getenv is negligible and in-process CUDA
// tests can flip the selection), mirroring the house VT_GDN_PACKED_REG_TILE /
// VT_GDN_CHUNKED convention; the parse itself is factored here so it is
// regression-covered on every platform, not just DGX.
#ifndef VT_CUDA_RMSNORM_DECODE_FAST_H_
#define VT_CUDA_RMSNORM_DECODE_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_RMSNORM_DECODE_FAST contract: the vectorized decode
// kernel is OFF by default (the shipped RmsNormRowKernel stays the default); it
// is ON only when the environment value is present AND its first character is
// '1'. nullptr (unset) and every non-'1'-leading value are OFF. Kept separate
// from the getenv call in the launcher so the parse is unit-testable without
// touching the environment.
inline bool RmsNormDecodeFastFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_RMSNORM_DECODE_FAST_H_
