// vllm.cpp original (vt runtime env-flag plumbing; the kernel it selects is the
// VENDORED vLLM FLA packed-decode cubin, see src/vt/cuda/cuda_gdn.cu).
// CPU-tier contract for the env-flag plumbing that selects the vendored Triton
// AOT packed-decode fast-path (src/vt/cuda/gdn_packed_decode_triton.h): the
// VT_GDN_PACKED_DECODE_TRITON flag predicate. The cubin itself is CUDA-only and
// lives in cuda_gdn.cu (TryTritonPackedDecode -> gdn_decode_h48); its
// bit-exactness and the default-fires / "=0"-rollback launch check are DGX-gated
// CUDA cases in tests/vt/test_ops_gdn.cpp. This suite pins the portable
// default-ON / '0'-off parse so the rollback contract is regression-covered on
// every platform, not just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/gdn_packed_decode_triton.h"

using vt::cuda::GdnPackedDecodeTritonFlagIsOn;

TEST_CASE(
    "VT_GDN_PACKED_DECODE_TRITON defaults ON; only a '0'-leading value rolls "
    "back") {
  // Default (unset) is ON: the vendored FLA cubin IS vLLM's exact kernel and is
  // token-identical, so it ships as the default like the sibling GDN Triton
  // kernels (VT_GDN_DELTAH_TRITON / VT_GDN_CHUNKO_TRITON / VT_GDN_WU_TRITON).
  CHECK(GdnPackedDecodeTritonFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON (the default).
  CHECK(GdnPackedDecodeTritonFlagIsOn(""));
  CHECK(GdnPackedDecodeTritonFlagIsOn("on"));
  CHECK(GdnPackedDecodeTritonFlagIsOn("true"));
  CHECK(GdnPackedDecodeTritonFlagIsOn("1"));
  CHECK(GdnPackedDecodeTritonFlagIsOn("2"));
  CHECK(GdnPackedDecodeTritonFlagIsOn(" 0"));  // leading space, not '0'
  // Rollback: FIRST character '0' restores the hand GdnPackedDecodeKernel in the
  // SAME binary.
  CHECK_FALSE(GdnPackedDecodeTritonFlagIsOn("0"));
  CHECK_FALSE(GdnPackedDecodeTritonFlagIsOn("0abc"));
  CHECK_FALSE(GdnPackedDecodeTritonFlagIsOn("00"));
}
