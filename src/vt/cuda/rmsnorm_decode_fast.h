// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// 1:1 port of upstream vLLM CUDA, see below.
//
// Env-flag plumbing for the decode-RMSNorm kernel-efficiency A/B, split into
// this pure-C++ (CUDA-free) header so the flag predicate is unit-testable on the
// CPU tier (tests/vt/test_rmsnorm_decode_fast.cpp). The kernel it selects is
// CUDA-only and lives in cuda_ops.cu (RmsNormRowFastKernel), a 1:1 port of
// vLLM's own vectorized CUDA rms-norm:
//   csrc/libtorch_stable/layernorm_kernels.cu:106-173 (fused_add_rms_norm_kernel
//   <scalar_t, width=8>) + launch :310-363 (block = min(hidden, 1024) when
//   num_tokens < 256; 16-byte _f16Vec loads; cub::BlockReduce<float,1024>),
//   with the _f16Vec packed bf16 add / f32 sum_squares of csrc/type_convert.cuh
//   :115-194, @ vLLM pin e24d1b24. This is the CUDA embodiment of the SAME
//   reduction/vectorization the production Inductor kernel runs
//   (triton_red_fused__to_copy_add..._rms_norm; the standalone-nsys-adjudicated
//   Phase-1 decode gap at the real 27B shape M x H=5120 is 3.18-3.56x, ours
//   8.44-8.53 us/launch vs vLLM 2.37-2.68 us — well over the 1.3x port bar).
//
// DEFAULT ON since the 2026-07-16 DGX proof @ 5a53fb5 (gate3 token gates +
// gate4 corrected-build A/B, evidence dgx:~/work/vllm.cpp-ewnorm-act-src): the
// token gates PASS with the fast kernel on BOTH models (27B paged-forward 17/17
// cases + 84/84 asserts, 35B 4/4 + 8/8 — the reordered-reduction 1-ULP hazard
// did not surface in either greedy-token gate), and the corrected
// production-config c16 A/B (CUTLASS FP4 + FA2 verified in the configure log;
// w0 sanity 800.9 tok/s) shows fast strictly better: meanTPOT -1.68/-1.90 ms on
// the two clean pairs (3/3 by median; the legacy-r3 leg is void — a ~20%
// interference anomaly), tput +8.7/+9.2 tok/s (~801-802 vs ~793); c2 pooled
// medians are parity-to-slightly-better (small-sample lottery, not
// over-claimed). VT_RMSNORM_DECODE_FAST=0 is the rollback to the previous
// one-block-per-row RmsNormRowKernel (kept for bit-exact reproduction of
// pre-flip token streams). The launcher reads getenv per call (decode RMSNorm
// dispatch is coarse — 129 launches/step — so the getenv is negligible and
// in-process CUDA tests can flip the selection), mirroring the house
// VT_GDN_PACKED_DECODE default-ON / '0'-rollback convention; the parse itself
// is factored here so it is regression-covered on every platform, not just DGX.
#ifndef VT_CUDA_RMSNORM_DECODE_FAST_H_
#define VT_CUDA_RMSNORM_DECODE_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_RMSNORM_DECODE_FAST contract: the vectorized decode
// kernel is ON by default (DGX token gates + corrected A/B passed; see header
// comment); it is OFF only when the environment value is present AND its first
// character is '0' (rollback to the shipped RmsNormRowKernel). nullptr (unset)
// and every non-'0'-leading value are ON. Kept separate from the getenv call in
// the launcher so the parse is unit-testable without touching the environment.
inline bool RmsNormDecodeFastFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_RMSNORM_DECODE_FAST_H_
