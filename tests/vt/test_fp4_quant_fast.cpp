// vllm.cpp original (vt runtime env-flag plumbing; the kernels these flags
// select are BIT-IDENTICAL vectorized-load variants living in
// src/vt/cuda/cuda_matmul_nvfp4.cu).
// CPU-tier contract for the env-flag plumbing that selects the two
// NUMERICS-NEUTRAL FP4-quant fast-paths (src/vt/cuda/fp4_quant_fast.h):
//   VT_FP4_QUANT_FAST -> ScaledFp4QuantFastKernel
//   VT_SILU_FP4_FAST  -> SiluAndMulFp4QuantFastKernel
// The kernels themselves are CUDA-only; their BYTE-EXACT-vs-scalar bit-identity
// and the default-off / "=1"-opt-in launch selection are DGX-gated CUDA cases in
// tests/vt/test_ops_nvfp4_fp4.cpp. This suite pins the portable default-OFF /
// '1'-opt-in parse so the opt-in contract is regression-covered on every
// platform, not just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/fp4_quant_fast.h"

using vt::cuda::Fp4QuantFastFlagIsOn;
using vt::cuda::SiluFp4FastFlagIsOn;

TEST_CASE("VT_FP4_QUANT_FAST defaults OFF; only a '1'-leading value opts in") {
  // Default (unset) is OFF: the shipped scalar ScaledFp4QuantKernel stays the
  // default; the orchestrator does the combined in-situ flip across the glue
  // levers together.
  CHECK_FALSE(Fp4QuantFastFlagIsOn(nullptr));
  // Non-'1'-leading values stay OFF (the default).
  CHECK_FALSE(Fp4QuantFastFlagIsOn(""));
  CHECK_FALSE(Fp4QuantFastFlagIsOn("0"));
  CHECK_FALSE(Fp4QuantFastFlagIsOn("off"));
  CHECK_FALSE(Fp4QuantFastFlagIsOn("false"));
  CHECK_FALSE(Fp4QuantFastFlagIsOn("2"));
  CHECK_FALSE(Fp4QuantFastFlagIsOn(" 1"));  // leading space, not '1'
  // Opt-in: FIRST character '1' selects the bit-identical vectorized-load fast
  // kernel in the SAME binary.
  CHECK(Fp4QuantFastFlagIsOn("1"));
  CHECK(Fp4QuantFastFlagIsOn("1abc"));
  CHECK(Fp4QuantFastFlagIsOn("11"));
}

TEST_CASE("VT_SILU_FP4_FAST defaults OFF; only a '1'-leading value opts in") {
  // Default (unset) is OFF: the shipped scalar SiluAndMulFp4QuantKernel stays the
  // default.
  CHECK_FALSE(SiluFp4FastFlagIsOn(nullptr));
  CHECK_FALSE(SiluFp4FastFlagIsOn(""));
  CHECK_FALSE(SiluFp4FastFlagIsOn("0"));
  CHECK_FALSE(SiluFp4FastFlagIsOn("off"));
  CHECK_FALSE(SiluFp4FastFlagIsOn("false"));
  CHECK_FALSE(SiluFp4FastFlagIsOn("2"));
  CHECK_FALSE(SiluFp4FastFlagIsOn(" 1"));
  // Opt-in: FIRST character '1'.
  CHECK(SiluFp4FastFlagIsOn("1"));
  CHECK(SiluFp4FastFlagIsOn("1abc"));
  CHECK(SiluFp4FastFlagIsOn("11"));
}
