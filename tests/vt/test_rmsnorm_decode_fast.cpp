// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// 1:1 port of vLLM CUDA fused_add_rms_norm_kernel<bf16,8>
// (csrc/libtorch_stable/layernorm_kernels.cu:106-173 @ e24d1b24).
//
// CPU-tier contract for the env-flag plumbing that selects the vectorized
// decode-RMSNorm kernel (src/vt/cuda/rmsnorm_decode_fast.h): the
// VT_RMSNORM_DECODE_FAST flag predicate. The vectorized kernel itself is
// CUDA-only and lives in cuda_ops.cu (RmsNormRowFastKernel); its bit-exactness
// vs RmsNormRowKernel and the aligned-bf16-only launch guard are DGX-gated CUDA
// checks. This suite pins the portable default-OFF / '1'-leading-on parse so the
// rollback contract is regression-covered on every platform, not just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/rmsnorm_decode_fast.h"

using vt::cuda::RmsNormDecodeFastFlagIsOn;

TEST_CASE(
    "VT_RMSNORM_DECODE_FAST defaults OFF; only a '1'-leading value opts in") {
  // Default (unset) is OFF: the shipped RmsNormRowKernel ships until the
  // 27B/35B token gates + an in-situ c16/c2 A/B prove the reordered-reduction
  // (token-exactness-hazard) flip.
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn(nullptr));
  // Non-'1'-leading values stay OFF.
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn(""));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("on"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("true"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("2"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("0"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn(" 1"));
  // Opt-in: FIRST character '1'.
  CHECK(RmsNormDecodeFastFlagIsOn("1"));
  CHECK(RmsNormDecodeFastFlagIsOn("10"));
  CHECK(RmsNormDecodeFastFlagIsOn("1abc"));
}
