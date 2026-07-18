// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// bit-identical-to-shipped port, see below.
//
// Env-flag plumbing for the GDN decode causal-conv1d state-update kernel-efficiency
// selection, split into this pure-C++ (CUDA-free) header so the flag predicate is
// unit-testable on the CPU tier (tests/vt/test_conv_update_fast.cpp). The kernel it
// selects is CUDA-only and lives in cuda_gdn.cu (CausalConv1dUpdateFastKernel): a
// register-caching / div-mod-free variant of the per-step causal_conv1d_update
// state roll whose output (both the `out` activation AND the rolled conv_state) is
// BIT-IDENTICAL (0-ulp) to the shipped CausalConv1dUpdateKernel, mirroring the
// RMSNorm/gated decode-fast bit-safety technique (cuda_ops.cu RmsNormRowFastKernel
// @ 348d12d; cuda_gdn.cu RmsNormGatedRowFastKernel @ 9ecd9d0).
//
// WHY BIT-IDENTICAL: the conv-update feeds the SAME 27B forward as the fast
// plain/gated RMSNorm + GDN decode cubin, all converging on the greedy RAZOR
// near-tie at token 6 (198 prod vs 271 emu). Per the a875397 lesson a <=1-ulp-close
// kernel accumulated over the layers can tip that tie and fail the production
// stream. This kernel keeps the shipped float op order EXACTLY (same bias init,
// same left-to-right conv accumulation `acc += w[j]*state[j]`, same `w[width]*x`
// tail, same silu/identity epilogue, same round-to-store), so its output is the
// SAME BITS as shipped and cannot cross the tie by construction.
//
// The ONLY differences vs shipped are (both numerics-neutral):
//   (1) INDEX MATH. shipped derives (token bt, channel c) from a flat grid-stride
//       index via int64 idx/c_dim + idx%c_dim (two expensive 64-bit div/mod per
//       thread). The fast kernel uses a 2D grid — blockIdx.y == token bt,
//       (blockIdx.x*blockDim.x+threadIdx.x) == channel c — so no div/mod runs.
//       Index arithmetic is not part of the numeric output; the values loaded and
//       stored are identical.
//   (2) CACHED STATE ROW. shipped reads state[j] once in the conv accumulation
//       and then RE-READS state[j+1] from global memory in the roll-left pass. The
//       fast kernel loads the (k-1)-element state row ONCE into registers
//       (templated on the compile-time state width so it stays register-resident,
//       mirroring vLLM's per-KERNEL_WIDTH col0..col3 register specialization in
//       fla causal_conv1d.py _causal_conv1d_update_kernel:158-192) and reuses those
//       exact same bits for both the accumulation and the roll — strictly fewer
//       global loads, identical values.
// This is the same shape the shipped kernel already runs (one thread per
// (token,channel)); it does strictly less work with identical bits, so it cannot
// regress and cannot perturb the stream.
//
// DEFAULT ON (VT_CONV_UPDATE_FAST=0 rolls back to the shipped
// CausalConv1dUpdateKernel). The kernel is BIT-IDENTICAL (0-ulp) AND cleared the
// isolated microbench bar with margin: at the dominant 27B c16 decode shape
// (batch=16, conv_dim=10240, k=4, bf16) nsys pure-kernel median is 3,680 ns fast vs
// 7,072 ns shipped = 1.92x (avg 3,785 vs 7,229 = 1.91x; the win is dominated by
// eliminating the two int64 div/mod per thread on this tiny latency-bound kernel,
// plus the register-cached state row). Bit-identical ⇒ never-slower + token-safe,
// so the default is flipped ON per the parity-enabler policy; the binding grid
// re-measures the in-situ effect. OFF only when the environment value is present
// AND its first character is '0'; nullptr (unset) and every non-'0'-leading value
// select the fast kernel, mirroring the house VT_RMSNORM_GATED_FAST /
// VT_RMSNORM_DECODE_FAST default-ON '0'-rollback convention. The launcher reads
// getenv per call (decode conv-update dispatch is coarse — one launch/step — so the
// getenv is negligible and in-process CUDA tests can flip the selection); the parse
// is factored here so it is regression-covered on every platform, not just DGX.
#ifndef VT_CUDA_CONV_UPDATE_FAST_H_
#define VT_CUDA_CONV_UPDATE_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_CONV_UPDATE_FAST contract: DEFAULT ON. The fast kernel
// (CausalConv1dUpdateFastKernel) is BIT-IDENTICAL (0-ulp) to the shipped
// CausalConv1dUpdateKernel by construction AND 1.92x faster in isolation, so it is
// the production default (mirroring vLLM's register-caching FLA conv-update). OFF
// only when the environment value is present AND its first character is '0'.
// nullptr (unset) and every non-'0'-leading value select the fast kernel. Kept
// separate from the getenv call in the launcher so the parse is unit-testable
// without touching the environment.
inline bool ConvUpdateFastFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_CONV_UPDATE_FAST_H_
