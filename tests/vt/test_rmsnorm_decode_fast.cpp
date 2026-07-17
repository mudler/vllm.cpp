// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// TRUE 1:1 port of vLLM csrc fused_add_rms_norm_kernel<bf16,8> using the ACTUAL
// cub::BlockReduce<float,1024> (csrc/libtorch_stable/layernorm_kernels.cu:106-173
// @ e24d1b24).
//
// CPU-tier contract for the env-flag plumbing that selects the vectorized
// decode-RMSNorm kernel (src/vt/cuda/rmsnorm_decode_fast.h): the
// VT_RMSNORM_DECODE_FAST flag predicate. The vectorized kernel itself is
// CUDA-only and lives in cuda_ops.cu (RmsNormRowFastKernel); its one-bf16-ulp
// parity vs RmsNormRowKernel and the aligned-bf16-only launch guard are
// DGX-gated CUDA checks (tests/vt/test_cuda_ops.cpp). This suite pins the
// portable default-OFF / '1'-opt-in parse so the opt-in contract is
// regression-covered on every platform, not just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/rmsnorm_decode_fast.h"

using vt::cuda::RmsNormDecodeFastFlagIsOn;

TEST_CASE(
    "VT_RMSNORM_DECODE_FAST defaults OFF (opt-in); only a '1'-leading value opts in") {
  // Default (unset) is OFF. The 2026-07-17 numerics rework FIXED the token-exactness
  // bug (the fast kernel now uses the ACTUAL cub::BlockReduce<float,1024> and is
  // token-exact vs the pip-vLLM enforce_eager oracle — test_qwen27_paged_engine
  // 235/235 fast-ON, qwen36 315/315 — AND ~3.2x faster in isolation), but the c16
  // in-situ A/B showed no throughput win (NULL within noise), so it ships OPT-IN
  // pending an in-situ win; the shipped RmsNormRowKernel stays the default.
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn(nullptr));
  // Non-'1'-leading values stay OFF.
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn(""));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("on"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("true"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("2"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("0"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn(" 1"));
  // Opt-in to the cub fast kernel: FIRST character '1'.
  CHECK(RmsNormDecodeFastFlagIsOn("1"));
  CHECK(RmsNormDecodeFastFlagIsOn("10"));
  CHECK(RmsNormDecodeFastFlagIsOn("1abc"));
}
