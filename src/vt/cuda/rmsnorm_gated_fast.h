// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// bit-identical-to-shipped port, see below.
//
// Env-flag plumbing for the GDN gated-RMSNorm decode kernel-efficiency selection,
// split into this pure-C++ (CUDA-free) header so the flag predicate is
// unit-testable on the CPU tier (tests/vt/test_rmsnorm_gated_fast.cpp). The kernel
// it selects is CUDA-only and lives in cuda_gdn.cu (RmsNormGatedRowFastKernel): a
// one-block-per-row gated RMSNorm decode kernel (RMSNorm + silu/sigmoid gate) whose
// output is BIT-IDENTICAL (0-ulp) to the shipped RmsNormGatedRowKernel for the bf16
// gated-norm decode path (d==128 — the GDN value head dim Dv used by both gate
// models: 27B Hv=48, 35B Hv=32, Dv=128).
//
// WHY BIT-IDENTICAL (not <=1-ulp): the GDN gated norm sits in the same forward as
// the fast plain-RMSNorm + GDN decode cubin, all feeding the 27B greedy RAZOR
// near-tie at output token 6 (198 prod vs 271 emu, ~zero logit gap). The a875397
// lesson: a <=1-ulp-close kernel accumulated over the layers can tip that tie and
// fail the production stream. Making the fast gated output the SAME BITS as shipped
// removes the perturbation by construction, so {fast-RMSNorm + fast-gated + cubin}
// produces the SAME logits as the passing baseline and CANNOT cross the tie.
//
// The fast kernel reproduces the shipped RmsNormGatedRowKernel's exact float op
// sequence: variance = sum of f32(bf16 x)^2 in shipped's kBlock-partial + shared-
// memory binary-tree ORDER (for d==128 each thread owns exactly one element, so the
// per-thread partial is a single term and the 128-thread tree reproduces shipped's
// 256-thread tree byte-for-byte — the extra s=128 step in shipped only adds the
// provably-zero partials[128..255]); inv = 1.0f/sqrtf(var/d + eps) (NOT rsqrtf);
// normalize = bf16(((f32(x)*inv)*f32(w))*act(f32(z))) with the SAME left-to-right
// multiply order and the SAME act (silu z/(1+exp(-z)) or sigmoid 1/(1+exp(-z))).
// The ONLY differences vs shipped are launch geometry (128 threads, not 256, so the
// idle upper half of every block is gone) and loading x ONCE into a register
// (shipped reloads it in the normalize pass) — neither changes the arithmetic.
//
// HOUSE CONVENTION mirror of VT_RMSNORM_DECODE_FAST / VT_GDN_PACKED_DECODE_TRITON:
// DEFAULT ON, `VT_RMSNORM_GATED_FAST=0` rolls back to the shipped
// RmsNormGatedRowKernel; unset/any-non-'0'-leading value keeps the fast kernel. The
// launcher reads getenv per call (gated-norm dispatch is coarse — 48 launches/step
// — so the getenv is negligible and in-process CUDA tests can flip the selection).
// The parse is factored here so it is regression-covered on every platform, not
// just DGX.
#ifndef VT_CUDA_RMSNORM_GATED_FAST_H_
#define VT_CUDA_RMSNORM_GATED_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_RMSNORM_GATED_FAST contract: DEFAULT ON. The fast
// gated kernel (RmsNormGatedRowFastKernel) is BIT-IDENTICAL (0-ulp) to the shipped
// RmsNormGatedRowKernel by construction AND faster in isolation, so it is the
// production default (mirroring vLLM's fused gated norm). OFF only when the
// environment value is present AND its first character is '0'. nullptr (unset) and
// every non-'0'-leading value select the fast kernel. Kept separate from the getenv
// call in the launcher so the parse is unit-testable without touching the environment.
inline bool RmsNormGatedFastFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_RMSNORM_GATED_FAST_H_
