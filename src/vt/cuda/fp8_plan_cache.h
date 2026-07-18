// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Per-device plan cache key + flag plumbing for the cuBLASLt FP8 (e4m3) TN dense
// GEMM path in cuda_matmul.cu (MatmulFp8CublasLtKernelCuda). The heavy cuBLASLt
// call cublasLtMatmulAlgoGetHeuristic — plus the matmul descriptor + three matrix
// layouts it needs — was rebuilt on EVERY fp8 GEMM call, causing a recurring
// ~0.8 ms host gap before the sm_121 fp8 GEMM (nsys, dgx:~/work/prefill-attr-35b/)
// that hurts 35B prefill TTFT and c1-c4 decode. vLLM reuses an in-graph plan; we
// mirror that with a per-device {desc, A/B/C layouts, heuristic algo} cache keyed
// on the FULL shape/config that determines them, so a cache hit skips the
// descriptor/layout creation + heuristic and goes straight to cublasLtMatmul.
//
// This header holds ONLY the pure, CUDA-free pieces (the VT_FP8_PLAN_CACHE flag
// predicate and the plan KEY: fields + equality + hash) so they are unit-testable
// on the CPU tier (tests/vt/test_fp8_plan_cache.cpp), exactly like gemm_algo_log.h.
// The cache map itself holds cuBLASLt handles and therefore lives in cuda_matmul.cu.
//
// Bit-exactness: cuBLASLt algo selection is process-deterministic (the same shape
// selects the same algo per the algo-latching forensic record; see
// gemm_algo_log.h / .agents/state.md), so pinning the first-selected plan is
// numerically identical to rebuilding it — exactly what a captured graph does.
// Verified byte-exact vs a fresh-plan GEMM in test_ops_fp8_cutlass.cpp.
//
// DEFAULT OFF (opt-in, VT_FP8_PLAN_CACHE=1). The lever's premise — that the
// per-call cublasLtMatmulAlgoGetHeuristic + descriptor/layout rebuild is a
// removable ~0.8 ms host gap before the fp8 GEMM — was NOT reproduced on GB10
// (2026-07-18, CLAIM-FP8-PLAN-CACHE-1): a same-binary 35B A/B is wall-clock
// NEUTRAL on prefill TTFT (async on AND off) and c1/c4 decode TPOT, and nsys
// shows the pre-fp8-GEMM GPU-timeline gap is UNCHANGED by the cache (~210 µs
// with cache off vs ~204 µs on) — the heuristic host cost is negligible/hidden
// (prefill is GPU-bound so it overlaps GPU work; decode is CUDA-graph-captured
// so the heuristic runs once at capture, not per replay-step). The cache is a
// correct, bit-exact structural mirror of vLLM's in-graph plan reuse kept behind
// an opt-in flag for eager/non-graph regimes; it does not flip the default
// because the "faster" condition is unmet.
#ifndef VT_CUDA_FP8_PLAN_CACHE_H_
#define VT_CUDA_FP8_PLAN_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace vt::cuda {

// Pure predicate for the VT_FP8_PLAN_CACHE contract: the cache is OFF by default
// and ENABLED only for the exact value "1" (the opt-in). nullptr (unset) and
// every other value are OFF. Kept separate from the cached getter below so the
// parse is unit-testable without touching the process-global cache.
inline bool Fp8PlanCacheFlagIsOn(const char* env_value) {
  return env_value != nullptr && std::string_view(env_value) == "1";
}

// Process-cached gate, read from the environment exactly once (getenv on the
// first call only; every later hot-path call pays a single bool load). Kept out
// of the CPU unit test because the cache latches on first read; the parse itself
// is covered via Fp8PlanCacheFlagIsOn.
inline bool Fp8PlanCacheEnabled() {
  static const bool enabled = Fp8PlanCacheFlagIsOn(std::getenv("VT_FP8_PLAN_CACHE"));
  return enabled;
}

// The FULL key that determines the cuBLASLt descriptor + selected algo for the
// fp8 (e4m3) TN dense GEMM path in cuda_matmul.cu. EVERY input that changes the
// descriptor OR the heuristic-selected algo MUST appear here — a missed field =
// wrong algo/desc reuse = a correctness bug. Fields are plain ints (the cuBLASLt
// enums cast to int at the call site) so this stays CUDA-free and CPU-testable.
//
// Field rationale (all captured from MatmulFp8CublasLtKernelCuda):
//   device       — one cuBLASLt handle + cached plans per device index.
//   m, n, k      — the GEMM shape; drives all three layout extents/leading dims
//                  (A=[K,N] ld=K, B=[K,M] ld=K, C=D=[N,M] ld=N) AND the algo the
//                  heuristic selects. m=a_fp8.shape[0], n=b_fp8.shape[0],
//                  k=a_fp8.shape[1].
//   out_type     — cudaDataType_t of C/D (CUDA_R_32F for f32 out, CUDA_R_16BF for
//                  bf16 out); the ONLY dtype that varies (A/B are always e4m3).
//                  Changes the C/D layout AND can change the selected algo.
//   a_type       — cudaDataType_t of the A/B operands (always CUDA_R_8F_E4M3
//                  here); pinned in the key so a future dtype split can't alias.
//   compute_type — cublasComputeType_t on the descriptor (CUBLAS_COMPUTE_32F).
//   scale_type   — cudaDataType_t scale on the descriptor (CUDA_R_32F).
//   trans_a/b    — CUBLASLT_MATMUL_DESC_TRANSA/TRANSB (OP_T / OP_N for the TN form).
//   epilogue     — cublasLtEpilogue_t (DEFAULT here; no bias/act fusion).
//   scale_mode   — 0 = per-tensor scale folded into the host alpha (no device
//                  scale pointers on the descriptor). alpha is a per-call host
//                  scalar that does NOT affect the descriptor or the algo, so it
//                  is deliberately NOT part of the key; a future device-scale-
//                  pointer mode would be a distinct scale_mode value.
struct Fp8PlanKey {
  int device = 0;
  int64_t m = 0, n = 0, k = 0;
  int out_type = 0;
  int a_type = 0;
  int compute_type = 0;
  int scale_type = 0;
  int trans_a = 0;
  int trans_b = 0;
  int epilogue = 0;
  int scale_mode = 0;

  bool operator==(const Fp8PlanKey& o) const {
    return device == o.device && m == o.m && n == o.n && k == o.k &&
           out_type == o.out_type && a_type == o.a_type && compute_type == o.compute_type &&
           scale_type == o.scale_type && trans_a == o.trans_a && trans_b == o.trans_b &&
           epilogue == o.epilogue && scale_mode == o.scale_mode;
  }
};

// FNV-1a-style hash over every key field (order-independent correctness: the ==
// above is the authority; the hash only needs to spread). Mixing each field in
// keeps distinct shapes/dtypes/transposes in different buckets.
struct Fp8PlanKeyHash {
  std::size_t operator()(const Fp8PlanKey& kk) const {
    std::size_t h = 1469598103934665603ull;  // FNV offset basis
    auto mix = [&h](std::uint64_t v) {
      h ^= static_cast<std::size_t>(v);
      h *= 1099511628211ull;  // FNV prime
    };
    mix(static_cast<std::uint64_t>(kk.device));
    mix(static_cast<std::uint64_t>(kk.m));
    mix(static_cast<std::uint64_t>(kk.n));
    mix(static_cast<std::uint64_t>(kk.k));
    mix(static_cast<std::uint64_t>(kk.out_type));
    mix(static_cast<std::uint64_t>(kk.a_type));
    mix(static_cast<std::uint64_t>(kk.compute_type));
    mix(static_cast<std::uint64_t>(kk.scale_type));
    mix(static_cast<std::uint64_t>(kk.trans_a));
    mix(static_cast<std::uint64_t>(kk.trans_b));
    mix(static_cast<std::uint64_t>(kk.epilogue));
    mix(static_cast<std::uint64_t>(kk.scale_mode));
    return h;
  }
};

}  // namespace vt::cuda

#endif  // VT_CUDA_FP8_PLAN_CACHE_H_
