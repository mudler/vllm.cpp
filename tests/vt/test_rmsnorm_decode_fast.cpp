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
// portable default-ON / '0'-rollback parse so the production-default contract is
// regression-covered on every platform, not just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/rmsnorm_decode_fast.h"

using vt::cuda::RmsNormDecodeFastFlagIsOn;

TEST_CASE(
    "VT_RMSNORM_DECODE_FAST defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON. The 2026-07-17 numerics rework made the fast kernel
  // token-exact vs the pip-vLLM enforce_eager oracle (real cub::BlockReduce<float,
  // 1024>; test_qwen27_paged_engine 235/235 + qwen36 315/315 fast-ON, e68c518) and
  // the 2026-07-17 c2 preflight A/B delivered the awaited in-situ WIN (interleaved
  // w0+3 pairs, binding c2 corpus: pooled per-request median TPOT -0.912 ms
  // (-0.887%), total throughput +1.446%, 3/3 pairs fast-better on BOTH axes), so
  // the true vLLM mirror ships as the default like the sibling GDN Triton kernels.
  CHECK(RmsNormDecodeFastFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON (the default).
  CHECK(RmsNormDecodeFastFlagIsOn(""));
  CHECK(RmsNormDecodeFastFlagIsOn("on"));
  CHECK(RmsNormDecodeFastFlagIsOn("true"));
  CHECK(RmsNormDecodeFastFlagIsOn("1"));
  CHECK(RmsNormDecodeFastFlagIsOn("2"));
  CHECK(RmsNormDecodeFastFlagIsOn(" 0"));  // leading space, not '0'
  // Rollback: FIRST character '0' restores the shipped one-block-per-row
  // RmsNormRowKernel in the SAME binary (bit-exact reproduction of the pre-flip
  // token streams; both kernels are oracle-exact since the cub rework).
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("0"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("0abc"));
  CHECK_FALSE(RmsNormDecodeFastFlagIsOn("00"));
}
