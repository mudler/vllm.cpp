// vllm.cpp original (vt runtime env-flag plumbing); the kernels these flags
// select are BIT-IDENTICAL vectorized-load variants of the existing scalar FP4
// quant kernels, whose 16-byte vectorized global load mirrors upstream vLLM, see
// below.
//
// Env-flag plumbing for the two NUMERICS-NEUTRAL decode-glue FP4-quant
// vectorization A/Bs, split into this pure-C++ (CUDA-free) header so the flag
// predicates are unit-testable on the CPU tier
// (tests/vt/test_fp4_quant_fast.cpp). The kernels they select are CUDA-only and
// live in cuda_matmul_nvfp4.cu:
//   VT_FP4_QUANT_FAST  -> ScaledFp4QuantFastKernel  (fast ScaledFp4Quant)
//   VT_SILU_FP4_FAST   -> SiluAndMulFp4QuantFastKernel (fast SiluAndMulFp4Quant)
//
// Each fast kernel keeps the EXACT per-element math of its scalar sibling
// (F32ToFp8Dev group scale, ReciprocalApproximateFtz/exact reciprocal choice,
// CastToFp4NibbleDev rounding, the fmaxf group-amax reduction, the bf16 SiLU
// round) and changes ONLY the MEMORY-ACCESS PATTERN: each thread loads its whole
// 16-element group with a single 16-byte (uint4) vectorized global load instead
// of 16 scalar loads, so the 32 threads of a warp coalesce their group spans into
// contiguous 128-byte transactions. This is exactly upstream vLLM's own quant
// load pattern — one thread loads CVT_FP4_ELTS_PER_THREAD=16 elements via
// ld256_cg / ld128_cg into a PackedVec:
//   csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu:56-80,126-149
//   (@ vLLM pin e24d1b24) — we use a plain aligned uint4 load (not the .cg
//   streaming variant) and keep our own bit-exact scalar cvt epilogue, so the
//   fp4 nibbles AND per-group fp8 scales are BYTE-EXACT vs the scalar kernel (the
//   bit-identity is proved by the old-vs-new adversarial parity cases in
//   tests/vt/test_ops_nvfp4_fp4.cpp). When a group span is not 16-byte aligned
//   the fast kernel falls back to the identical scalar load internally, so it is
//   bit-identical on every input.
//
// BOTH DEFAULT OFF (opt-in via VT_FP4_QUANT_FAST=1 / VT_SILU_FP4_FAST=1). The
// orchestrator does the combined in-situ default flip + A/B across all the glue
// levers together (the a875397 combined-set lesson); this header only makes the
// kernels selectable and CPU-regression-covers the parse. ON only when the
// environment value is present AND its first character is '1'; nullptr (unset)
// and every non-'1'-leading value are OFF, keeping the shipped scalar kernel. The
// launchers read getenv per call (decode FP4-quant dispatch is coarse — 144
// ScaledFp4Quant + 64 SiluAndMul launches/step — so the getenv is negligible and
// in-process CUDA tests can flip the selection), mirroring the house
// VT_RMSNORM_DECODE_FAST / VT_GDN_PACKED_REG_TILE default-OFF '1'-opt-in
// convention; the parse itself is factored here so it is regression-covered on
// every platform, not just DGX.
#ifndef VT_CUDA_FP4_QUANT_FAST_H_
#define VT_CUDA_FP4_QUANT_FAST_H_

namespace vt::cuda {

// Pure predicate for the VT_FP4_QUANT_FAST contract (fast ScaledFp4Quant):
// DEFAULT OFF (opt-in). ON only when the environment value is present AND its
// first character is '1'. nullptr (unset) and every non-'1'-leading value are
// OFF. Kept separate from the getenv call in the launcher so the parse is
// unit-testable without touching the environment.
inline bool Fp4QuantFastFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1';
}

// Pure predicate for the VT_SILU_FP4_FAST contract (fast SiluAndMulFp4Quant):
// DEFAULT OFF (opt-in), same '1'-leading parse as Fp4QuantFastFlagIsOn.
inline bool SiluFp4FastFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1';
}

}  // namespace vt::cuda

#endif  // VT_CUDA_FP4_QUANT_FAST_H_
