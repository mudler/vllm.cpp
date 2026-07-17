// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// bit-identical-to-shipped vectorized port, see below.
//
// Env-flag plumbing for the decode-RMSNorm kernel-efficiency selection, split into
// this pure-C++ (CUDA-free) header so the flag predicate is unit-testable on the
// CPU tier (tests/vt/test_rmsnorm_decode_fast.cpp). The kernel it selects is
// CUDA-only and lives in cuda_ops.cu (RmsNormRowFastKernel): a vectorized
// add+RMSNorm decode kernel whose output is BIT-IDENTICAL (0-ulp) to the shipped
// RmsNormRowKernel — the through-stack 235/235 bit-reference that matches vLLM's
// production greedy stream — for the bf16 decode path (H%8==0, 1024<=H<=8192).
//
// DEFAULT ON (VT_RMSNORM_DECODE_FAST=0 rolls back to shipped). HISTORY: two prior
// flips (696a991, a321d7c) were ROLLED BACK (a0013a2, a875397) for the SAME cause —
// the fast kernel's reduction ORDER / rounding differed from shipped by ≤1 ulp, and
// combined with the GDN decode cubin's own ≤1-ulp perturbation that tipped a 27B
// greedy RAZOR near-tie at output token 6 to the emulation side (233/235; vLLM
// itself decides 198 in production vs 271 in emulation, ~zero logit gap). The
// 2026-07-17 BIT-SAFETY rework (CLAIM-EW-NORM-ACT-3) removes the cause by
// CONSTRUCTION: RmsNormRowFastKernel now reproduces shipped's variance path
// byte-for-byte (kBlock=256 strided partials + shared-memory tree + 1.0f/sqrtf;
// residual add bf16(f32(x)+f32(res))) and only vectorizes the element-independent
// normalize pass, so its output is the SAME BITS as shipped. Therefore
// fast+cubin ≡ shipped+cubin ≡ 198 always — the near-tie can never be crossed.
// GATE-PROVEN: test_cuda_ops decode-fast case fast==shipped BIT-EXACT (0-ulp);
// test_qwen27_paged_engine 235/235 + test_qwen36_paged_engine 315/315 with the full
// production default set (async ON + GDN cubin ON + RMSNorm-fast ON), and both
// rollback arms (=0) 235/235 + 315/315. Perf: the 1024-thread memory passes keep a
// large decode win despite bit-identity — isolated nsys 3.55 µs vs shipped 8.58 µs
// (2.41×) at M=16×H=5120, and in the real 27B engine forward RmsNorm median 4.38 µs
// vs 16.13 µs (3.68×) with the same 235/235 tokens; the kernel does strictly less
// GPU work with identical bits, so it cannot regress in-situ. The predecessor's c2
// serving preflight (+1.446% tput / −0.887% TPOT, a321d7c) established the in-situ
// acceptance; this bit-identical kernel retains ~84% of that per-step saving. Per
// the parity-enabler policy the default is flipped ON (gated) and the binding grid
// measures the production default. Evidence dgx:~/work/vllm.cpp-ewnorm-bitsafe.
// `VT_RMSNORM_DECODE_FAST=0` rolls back to shipped RmsNormRowKernel; unset/any-non-'0'
// keeps the fast kernel. The launcher reads getenv per call (decode RMSNorm dispatch
// is coarse — 129 launches/step — so the getenv is negligible and in-process CUDA
// tests can flip the selection), mirroring the house VT_GDN_PACKED_DECODE_TRITON
// default-ON / '0'-rollback convention; the parse is factored here so it is
// regression-covered on every platform, not just DGX.
#ifndef VT_CUDA_RMSNORM_DECODE_FAST_H_
#define VT_CUDA_RMSNORM_DECODE_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_RMSNORM_DECODE_FAST contract: DEFAULT ON. Since the
// 2026-07-17 bit-safety rework the fast kernel (RmsNormRowFastKernel) is BIT-IDENTICAL
// (0-ulp) to the shipped RmsNormRowKernel AND 2.41× faster in isolation, so it is the
// production default (mirroring vLLM's vectorized add+RMSNorm). OFF only when the
// environment value is present AND its first character is '0'. nullptr (unset) and
// every non-'0'-leading value select the fast kernel. Kept separate from the getenv
// call in the launcher so the parse is unit-testable without touching the environment.
inline bool RmsNormDecodeFastFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_RMSNORM_DECODE_FAST_H_
