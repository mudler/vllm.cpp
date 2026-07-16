// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// 1:1 port of vLLM CUDA fused_add_rms_norm_kernel<bf16,8>
// (csrc/libtorch_stable/layernorm_kernels.cu:106-173 @ e24d1b24).
//
// CPU-tier contract for the env-flag plumbing that selects the vectorized
// decode-RMSNorm kernel (src/vt/cuda/rmsnorm_decode_fast.h): the
// VT_RMSNORM_DECODE_FAST flag predicate. The vectorized kernel itself is
// CUDA-only and lives in cuda_ops.cu (RmsNormRowFastKernel); its one-bf16-ulp
// parity vs RmsNormRowKernel and the aligned-bf16-only launch guard are
// DGX-gated CUDA checks (tests/vt/test_cuda_ops.cpp). This suite pins the
// portable default-ON / '0'-rollback parse so the rollback contract is
// regression-covered on every platform, not just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/rmsnorm_decode_fast.h"

using vt::cuda::RmsNormDecodeFastFlagIsOn;

TEST_CASE(
    "VT_RMSNORM_DECODE_FAST defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON since the 2026-07-16 DGX proof @ 5a53fb5: token gates
  // PASS with the fast kernel on both models (27B 17/17+84/84, 35B 4/4+8/8) and
  // the corrected production-config c16 A/B shows a consistent TPOT/tput win.
  CHECK(RmsNormDecodeFastFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON.
  CHECK(RmsNormDecodeFastFlagIsOn(""));
  CHECK(RmsNormDecodeFastFlagIsOn("on"));
  CHECK(RmsNormDecodeFastFlagIsOn("true"));
  CHECK(RmsNormDecodeFastFlagIsOn("1"));
  CHECK(RmsNormDecodeFastFlagIsOn("2"));
  CHECK(RmsNormDecodeFastFlagIsOn(" 0"));
  // Rollback to the shipped RmsNormRowKernel: FIRST character '0'.
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("0"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("00"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("0abc"));
}
