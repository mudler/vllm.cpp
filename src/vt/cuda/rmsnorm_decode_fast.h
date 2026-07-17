// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// 1:1 port of upstream vLLM CUDA, see below.
//
// Env-flag plumbing for the decode-RMSNorm kernel-efficiency A/B, split into
// this pure-C++ (CUDA-free) header so the flag predicate is unit-testable on the
// CPU tier (tests/vt/test_rmsnorm_decode_fast.cpp). The kernel it selects is
// CUDA-only and lives in cuda_ops.cu (RmsNormRowFastKernel), a TRUE 1:1 port of
// vLLM's csrc fused_add_rms_norm_kernel<bf16,8>:
//   csrc/libtorch_stable/layernorm_kernels.cu:106-173 (fused_add_rms_norm_kernel
//   <scalar_t, width=8>), launch :310-331 (block = min(hidden,1024) for
//   num_tokens<256), reduction cub::BlockReduce<float,1024>.Reduce(variance,
//   CubAddOp{}, blockDim.x), with the _f16Vec packed bf16 `+=` / f32 `sum_squares`
//   of csrc/type_convert.cuh, @ vLLM pin e24d1b24.
//
// DEFAULT ON (rollback via VT_RMSNORM_DECODE_FAST=0). HISTORY: the 2026-07-16
// flip to ON (696a991) was ROLLED BACK (a0013a2) because that kernel APPROXIMATED
// cub::BlockReduce with a hand two-stage warp-shuffle whose reordered f32 sum
// flipped a 27B greedy near-tie at output token 7 (271 vs oracle 198;
// test_qwen27_paged_engine 234/235). The 2026-07-17 NUMERICS REWORK
// (CLAIM-EW-NORM-ACT-2, e68c518) FIXED that: swapping the hand reduction for the
// ACTUAL cub::BlockReduce reproduces the oracle's exact reduction order (the
// oracle golden is generated with LLM(..., enforce_eager=True) — the eager csrc
// CUDA op — per tools/parity/dump_qwen36.py:242), making the fast kernel
// TOKEN-EXACT vs the oracle (test_qwen27_paged_engine 235/235 + qwen36 315/315
// fast-ON; isolated nsys 2.66 µs vs shipped 8.66 µs, ~3.2x; evidence
// dgx:~/work/vllm.cpp-ewnorm-numerics). It landed OPT-IN because the c16 in-situ
// A/B was a NULL within noise; the spec named c2 as the target lane (the
// batch-independent 129-launch/step saving is a larger fraction of the smaller
// c2 step). The 2026-07-17 c2 preflight A/B (CLAIM-SERVE-GATE-2, one flock,
// hard-verified CUTLASS+FA2 a321d7c production build, interleaved w0-discard +
// 3 pairs on the binding c2 corpus, house pooled per-request-median convention)
// delivered that in-situ WIN: pooled-median TPOT -0.912 ms (-0.887%), total
// throughput +1.446% (fast HIGHER), 3/3 pairs fast-better on BOTH axes
// (evidence dgx:~/work/vllm.cpp-online-gate/preflight-rmsnorm-c2-a321d7c…).
// Flip acceptance met (measurable TPOT reduction, no throughput regression), so
// the true vLLM mirror ships as the default. `VT_RMSNORM_DECODE_FAST=0` is the
// same-binary rollback to the one-block-per-row RmsNormRowKernel (kept for
// bit-exact reproduction of pre-flip token streams; both kernels are
// oracle-exact since the cub rework). The launcher reads getenv per call (decode
// RMSNorm dispatch is coarse — 129 launches/step — so the getenv is negligible
// and in-process CUDA tests can flip the selection), mirroring the house
// default-ON / '0'-rollback convention of gdn_packed_decode_triton.h; the parse
// itself is factored here so it is regression-covered on every platform, not
// just DGX.
#ifndef VT_CUDA_RMSNORM_DECODE_FAST_H_
#define VT_CUDA_RMSNORM_DECODE_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_RMSNORM_DECODE_FAST contract: DEFAULT ON. Since the
// 2026-07-17 numerics rework the fast kernel (RmsNormRowFastKernel, real
// cub::BlockReduce) is token-exact vs the pip-vLLM oracle AND ~3.2x faster in
// isolation, and the 2026-07-17 c2 preflight A/B confirmed the in-situ win
// (pooled-median TPOT -0.887%, throughput +1.446%, 3/3 pairs), so it is the
// production default. OFF (rollback to RmsNormRowKernel) only when the
// environment value is present AND its first character is '0'. nullptr (unset)
// and every non-'0'-leading value are ON. Kept separate from the getenv call in
// the launcher so the parse is unit-testable without touching the environment.
inline bool RmsNormDecodeFastFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_RMSNORM_DECODE_FAST_H_
