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
// DEFAULT OFF (opt-in via VT_RMSNORM_DECODE_FAST=1). HISTORY: the 2026-07-16 flip
// to ON (696a991) was ROLLED BACK (a0013a2) because that kernel APPROXIMATED
// cub::BlockReduce with a hand two-stage warp-shuffle whose reordered f32 sum flipped
// a 27B greedy near-tie at output token 7 (271 vs oracle 198; test_qwen27_paged_engine
// 234/235). The 2026-07-17 NUMERICS REWORK (CLAIM-EW-NORM-ACT-2) FIXES that: the
// rollback note's "Inductor-Triton" premise was WRONG — the oracle golden is generated
// with LLM(..., enforce_eager=True) (tools/parity/dump_qwen36.py:242, source
// pip-vllm:0.24.0) => the eager csrc CUDA op with cub::BlockReduce<float,1024>, not
// Triton. Swapping the hand reduction for the ACTUAL cub::BlockReduce reproduces the
// oracle's exact reduction order, and the fast kernel is now TOKEN-EXACT vs the oracle
// (test_qwen27_paged_engine 235/235 + qwen36 315/315 with it ON; isolated nsys 2.66 µs
// vs shipped 8.66 µs, ~3.2x; evidence dgx:~/work/vllm.cpp-ewnorm-numerics). BUT the
// default stays OFF: the interleaved c16 in-situ A/B did NOT confirm a throughput WIN
// (this run: fast −0.60% tput / +0.34 ms TPOT, 3/3 pairs; contradicting the earlier
// gate4 +1.1% — the shipped-kernel arm's ~2% run-variation dominates, so the c16
// effect is a NULL within noise, as the spec anticipated). Per the flip acceptance
// (measurable TPOT reduction with no throughput regression) an unconfirmed/regressing
// c16 A/B does not clear the bar, so RmsNormRowFastKernel ships OPT-IN. It is now
// token-safe to enable (VT_RMSNORM_DECODE_FAST=1) and is the true vLLM mirror; the
// default flip awaits an in-situ WIN (the batch-independent ~0.77 ms/step RMSNorm
// saving is a larger fraction at c2 — the documented target per the c2/c8 attribution).
// `VT_RMSNORM_DECODE_FAST=1` opts in; unset/any-non-'1' keeps the shipped
// RmsNormRowKernel. The launcher reads getenv per call (decode RMSNorm dispatch is
// coarse — 129 launches/step — so the getenv is negligible and in-process CUDA tests can
// flip the selection), mirroring the house VT_GDN_PACKED_REG_TILE default-OFF / '1'-opt-in
// convention; the parse itself is factored here so it is regression-covered on every
// platform, not just DGX.
#ifndef VT_CUDA_RMSNORM_DECODE_FAST_H_
#define VT_CUDA_RMSNORM_DECODE_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_RMSNORM_DECODE_FAST contract: DEFAULT OFF (opt-in). Since
// the 2026-07-17 numerics rework the fast kernel (RmsNormRowFastKernel, real
// cub::BlockReduce) is token-exact vs the pip-vLLM oracle AND ~3.2x faster in isolation,
// but the c16 in-situ A/B showed no throughput win (NULL within noise), so it ships
// OPT-IN pending an in-situ win. ON only when the environment value is present AND its
// first character is '1'. nullptr (unset) and every non-'1'-leading value are OFF. Kept
// separate from the getenv call in the launcher so the parse is unit-testable without
// touching the environment.
inline bool RmsNormDecodeFastFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_RMSNORM_DECODE_FAST_H_
