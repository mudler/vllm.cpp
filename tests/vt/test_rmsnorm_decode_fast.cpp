// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// vectorized add+RMSNorm decode kernel whose output is BIT-IDENTICAL (0-ulp) to the
// shipped RmsNormRowKernel (cuda_ops.cu; the 235/235 through-stack bit-reference).
//
// CPU-tier contract for the env-flag plumbing that selects the vectorized
// decode-RMSNorm kernel (src/vt/cuda/rmsnorm_decode_fast.h): the
// VT_RMSNORM_DECODE_FAST flag predicate. The vectorized kernel itself is
// CUDA-only and lives in cuda_ops.cu (RmsNormRowFastKernel); its BIT-EXACT (0-ulp)
// parity vs RmsNormRowKernel and the aligned-bf16-only launch guard are
// DGX-gated CUDA checks (tests/vt/test_cuda_ops.cpp). This suite pins the
// portable default-ON / '0'-rollback parse so the contract is regression-covered
// on every platform, not just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/rmsnorm_decode_fast.h"

using vt::cuda::RmsNormDecodeFastFlagIsOn;

TEST_CASE(
    "VT_RMSNORM_DECODE_FAST defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON. The 2026-07-17 BIT-SAFETY rework (CLAIM-EW-NORM-ACT-3)
  // makes RmsNormRowFastKernel's output BIT-IDENTICAL (0-ulp) to the shipped
  // RmsNormRowKernel, so fast+cubin ≡ shipped+cubin ≡ 198 by construction
  // (test_qwen27_paged_engine 235/235 + qwen36 315/315 with the full production
  // default set; test_cuda_ops decode-fast 0-ulp) while staying 2.41× faster in
  // isolation — so it is the production default (mirroring vLLM's vectorized
  // add+RMSNorm). This inverts the prior opt-in parse.
  CHECK(RmsNormDecodeFastFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON.
  CHECK(RmsNormDecodeFastFlagIsOn(""));
  CHECK(RmsNormDecodeFastFlagIsOn("on"));
  CHECK(RmsNormDecodeFastFlagIsOn("true"));
  CHECK(RmsNormDecodeFastFlagIsOn("2"));
  CHECK(RmsNormDecodeFastFlagIsOn("1"));
  CHECK(RmsNormDecodeFastFlagIsOn(" 0"));
  // Roll back to the shipped RmsNormRowKernel: FIRST character '0'.
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("0"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("00"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("0abc"));
}
