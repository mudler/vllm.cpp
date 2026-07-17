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
// DEFAULT OFF again since the 2026-07-17 ROLLBACK. The 2026-07-16 flip to ON
// (@ 5a53fb5) was gated only on `test_qwen27_paged_FORWARD` (17/17) + 35B, which
// did not exercise the full production greedy stream. The async-default-flip DGX
// re-confirmation (evidence dgx:~/work/vllm.cpp-async-flip) ran the fuller
// `test_qwen27_paged_ENGINE` 16-token production stream and found fast-ON
// DIVERGES from the pip-vLLM oracle golden (greedy_ids.npy): tokens 1-6 match then
// token 7 flips (271 vs 198) and cascades — 234/235 with fast ON, 235/235 with
// fast OFF, IDENTICALLY across all async arms (async-independent). Root cause: the
// vectorized 1024-thread reduction reorders the f32 sum vs the shipped
// 256-thread-tree kernel, flipping a documented 27B whitespace/near-tie greedy
// argmax; and vLLM's real oracle runs an Inductor-generated Triton rmsnorm, not the
// csrc `fused_add_rms_norm_kernel` this port mirrors, so bit-parity vs the ORACLE
// stream was never guaranteed. Token-exactness vs vLLM is a sacrosanct precondition
// ⇒ the shipped RmsNormRowKernel (oracle-exact) is the default again;
// RmsNormRowFastKernel is an OPT-IN experiment (VT_RMSNORM_DECODE_FAST=1) pending a
// numerical fix that reproduces the oracle's Inductor-Triton stream. The launcher
// reads getenv per call (decode RMSNorm dispatch is coarse — 129 launches/step — so
// the getenv is negligible and in-process CUDA tests can flip the selection),
// mirroring the house VT_GDN_PACKED_REG_TILE default-OFF / '1'-opt-in convention;
// the parse itself is factored here so it is regression-covered on every platform,
// not just DGX.
#ifndef VT_CUDA_RMSNORM_DECODE_FAST_H_
#define VT_CUDA_RMSNORM_DECODE_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_RMSNORM_DECODE_FAST contract: DEFAULT OFF again since
// the 2026-07-17 rollback (see header comment) — the shipped RmsNormRowKernel is
// token-exact vs the pip-vLLM oracle; the vectorized RmsNormRowFastKernel is an
// OPT-IN experiment (VT_RMSNORM_DECODE_FAST=1) because its reordered reduction
// flips a 27B greedy near-tie away from the oracle stream. ON only when the
// environment value is present AND its first character is '1'. nullptr (unset) and
// every non-'1'-leading value are OFF. Kept separate from the getenv call in the
// launcher so the parse is unit-testable without touching the environment.
inline bool RmsNormDecodeFastFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_RMSNORM_DECODE_FAST_H_
