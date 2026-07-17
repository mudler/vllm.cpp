// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is the
// VENDORED vLLM FLA packed-decode cubin (an exact 1:1 of upstream), see below.
//
// Env-flag plumbing for the vendored Triton AOT packed-decode fast-path, split
// into this pure-C++ (CUDA-free) header so the flag predicate is unit-testable on
// the CPU tier (tests/vt/test_gdn_packed_decode_triton.cpp). The cubin it selects
// is CUDA-only and lives in cuda_gdn.cu (TryTritonPackedDecode -> gdn_decode_h48),
// the VENDORED build of vLLM FLA's num_warps=1 / num_stages=3 register-resident
// fused_recurrent_gated_delta_rule_packed_decode_kernel
// (vllm/model_executor/layers/fla/ops/fused_recurrent.py:256-336, launch
// :439-478 @ 702f4814). MEASURED codegen-bound: the identical register-resident
// [BV=32,BK=128] fp32 state tile is REG:205/0-spill under Triton/ptxas but
// REG:255+STACK:48 (spills) as hand-CUDA (dgx phase1 2026-07-16), which is why
// this joins the sanctioned vendored-AOT set instead of a portable redesign.
//
// Why default ON: the vendored kernel IS vLLM's exact kernel, so it is
// token-identical to vLLM (27B model gate 235/235 token-exact with the path ON;
// AOT-vs-legacy-vs-CPU op test 28/28; oracle boundary 12/12; c16 A/B +5.48 tok/s
// / TPOT -1.26 ms, 3/3 pairs; memcheck 0/0 @ 9dd7d3f). MIRROR policy: vLLM runs
// this exact FLA kernel by default, so we do too — like the sibling GDN Triton
// kernels (VT_GDN_DELTAH_TRITON / VT_GDN_CHUNKO_TRITON / VT_GDN_WU_TRITON, all
// default ON via GdnTritonEnvOn). VT_GDN_PACKED_DECODE_TRITON=0 is the same-binary
// rollback to the hand GdnPackedDecodeKernel. The launcher reads getenv per call
// (coarse decode dispatch — negligible getenv, and in-process CUDA tests can flip
// the selection), mirroring the house VT_GDN_PACKED_REG_TILE / VT_GDN_CHUNKED
// convention; the parse itself is factored here so it is regression-covered on
// every platform, not just DGX. The kernel exists only in VLLM_CPP_TRITON builds
// (the vendored cubin is linked there — README "GB10 fast-GDN build"); non-Triton
// builds fall back to the hand kernel and this predicate is inert.
#ifndef VT_CUDA_GDN_PACKED_DECODE_TRITON_H_
#define VT_CUDA_GDN_PACKED_DECODE_TRITON_H_

namespace vt::cuda {

// Pure predicate for the VT_GDN_PACKED_DECODE_TRITON contract: the vendored FLA
// packed-decode cubin is ON by default (it is vLLM's exact token-identical
// kernel); it is OFF only when the environment value is present AND its first
// character is '0'. nullptr (unset) and every non-'0'-leading value are ON. Kept
// separate from the getenv call in the launcher so the parse is unit-testable
// without touching the environment. Mirrors GdnTritonEnvOn (the sibling GDN
// Triton kernels' default-ON parse) in a CPU-testable header form.
inline bool GdnPackedDecodeTritonFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_GDN_PACKED_DECODE_TRITON_H_
