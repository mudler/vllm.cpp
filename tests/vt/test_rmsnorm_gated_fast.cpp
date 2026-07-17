// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// gated RMSNorm decode kernel whose output is BIT-IDENTICAL (0-ulp) to the shipped
// RmsNormGatedRowKernel (cuda_gdn.cu).
//
// CPU-tier contract for the env-flag plumbing that selects the fast gated-RMSNorm
// decode kernel (src/vt/cuda/rmsnorm_gated_fast.h): the VT_RMSNORM_GATED_FAST flag
// predicate. The kernel itself is CUDA-only and lives in cuda_gdn.cu
// (RmsNormGatedRowFastKernel); its BIT-EXACT (0-ulp) parity vs RmsNormGatedRowKernel
// and the aligned-bf16-d128-only launch guard are DGX-gated CUDA checks
// (tests/vt/test_ops_gdn.cpp). This suite pins the portable default-ON /
// '0'-rollback parse so the contract is regression-covered on every platform, not
// just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/rmsnorm_gated_fast.h"

using vt::cuda::RmsNormGatedFastFlagIsOn;

TEST_CASE(
    "VT_RMSNORM_GATED_FAST defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON. RmsNormGatedRowFastKernel's output is BIT-IDENTICAL
  // (0-ulp) to the shipped RmsNormGatedRowKernel by construction (variance in
  // shipped's kBlock-partial + shared-tree order, 1.0f/sqrtf, same normalize/act
  // multiply order), so fast+cubin+fast-RMSNorm produces the SAME logits as the
  // passing baseline (test_qwen27_paged_engine 235/235 + qwen36 315/315) while
  // running strictly less GPU work (128 threads not 256, x loaded once) — so it is
  // the production default (mirroring vLLM's fused gated norm).
  CHECK(RmsNormGatedFastFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON.
  CHECK(RmsNormGatedFastFlagIsOn(""));
  CHECK(RmsNormGatedFastFlagIsOn("on"));
  CHECK(RmsNormGatedFastFlagIsOn("true"));
  CHECK(RmsNormGatedFastFlagIsOn("2"));
  CHECK(RmsNormGatedFastFlagIsOn("1"));
  CHECK(RmsNormGatedFastFlagIsOn(" 0"));
  // Roll back to the shipped RmsNormGatedRowKernel: FIRST character '0'.
  CHECK_FALSE(RmsNormGatedFastFlagIsOn("0"));
  CHECK_FALSE(RmsNormGatedFastFlagIsOn("00"));
  CHECK_FALSE(RmsNormGatedFastFlagIsOn("0abc"));
}
