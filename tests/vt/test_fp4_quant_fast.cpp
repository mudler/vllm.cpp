// vllm.cpp original (vt runtime env-flag plumbing; the kernels these flags
// select are BIT-IDENTICAL vectorized-load variants living in
// src/vt/cuda/cuda_matmul_nvfp4.cu).
// CPU-tier contract for the env-flag plumbing that selects the two
// NUMERICS-NEUTRAL FP4-quant fast-paths (src/vt/cuda/fp4_quant_fast.h):
//   VT_FP4_QUANT_FAST -> ScaledFp4QuantFastKernel
//   VT_SILU_FP4_FAST  -> SiluAndMulFp4QuantFastKernel
// The kernels themselves are CUDA-only; their BYTE-EXACT-vs-scalar bit-identity
// and the default-ON / "=0"-rollback launch selection are DGX-gated CUDA cases in
// tests/vt/test_ops_nvfp4_fp4.cpp. This suite pins the portable default-ON /
// '0'-rollback parse so the flip contract is regression-covered on every
// platform, not just DGX. (2026-07-18, CLAIM-CONV-UPDATE-FAST-1: flipped ON per
// the parity-enabler policy — bit-identical ⇒ never-slower + token-safe.)
#include <doctest/doctest.h>

#include "vt/cuda/fp4_quant_fast.h"

using vt::cuda::Fp4QuantFastFlagIsOn;
using vt::cuda::SiluFp4FastFlagIsOn;

TEST_CASE("VT_FP4_QUANT_FAST defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON: the bit-identical vectorized-load fast kernel is the
  // production default per the parity-enabler policy.
  CHECK(Fp4QuantFastFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON (the default).
  CHECK(Fp4QuantFastFlagIsOn(""));
  CHECK(Fp4QuantFastFlagIsOn("1"));
  CHECK(Fp4QuantFastFlagIsOn("off"));
  CHECK(Fp4QuantFastFlagIsOn("false"));
  CHECK(Fp4QuantFastFlagIsOn("2"));
  CHECK(Fp4QuantFastFlagIsOn(" 0"));  // leading space, not '0'
  // Rollback: FIRST character '0' selects the shipped scalar kernel in the SAME
  // binary.
  CHECK_FALSE(Fp4QuantFastFlagIsOn("0"));
  CHECK_FALSE(Fp4QuantFastFlagIsOn("0abc"));
  CHECK_FALSE(Fp4QuantFastFlagIsOn("00"));
}

TEST_CASE("VT_SILU_FP4_FAST defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON: the shipped scalar SiluAndMulFp4QuantKernel is the
  // rollback.
  CHECK(SiluFp4FastFlagIsOn(nullptr));
  CHECK(SiluFp4FastFlagIsOn(""));
  CHECK(SiluFp4FastFlagIsOn("1"));
  CHECK(SiluFp4FastFlagIsOn("off"));
  CHECK(SiluFp4FastFlagIsOn("false"));
  CHECK(SiluFp4FastFlagIsOn("2"));
  CHECK(SiluFp4FastFlagIsOn(" 0"));
  // Rollback: FIRST character '0'.
  CHECK_FALSE(SiluFp4FastFlagIsOn("0"));
  CHECK_FALSE(SiluFp4FastFlagIsOn("0abc"));
  CHECK_FALSE(SiluFp4FastFlagIsOn("00"));
}
