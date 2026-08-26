// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Per-call-key heuristic-result cache key + flag plumbing for the bf16/f32
// cuBLASLt GEMM lanes in cuda_matmul.cu (MatmulKernelCuda, MatmulBTKernelCuda,
// BatchedMatmulKernelCuda). Issue #1732: on CUDA 13.3 + sm_80,
// cublasLtMatmulAlgoGetHeuristic returns CUBLAS_STATUS_INTERNAL_ERROR (status
// 14; its own trace: "Could not obtain green context information") when it runs
// while a CREATED stream is in CUDA graph capture. The bf16/f32 lanes queried
// the heuristic on EVERY call, so the default (graphs-on) decode died on the
// first in-capture GEMM. The engine's eager warm step runs every decode slot's
// padded shapes eagerly BEFORE capture, so caching the heuristic result per
// full call key makes every in-capture call a map hit: cuBLASLt is never
// queried under capture. Upstream vLLM is immune for the same structural
// reason — torch caches the selected cuBLASLt algo per shape and vLLM runs
// eager warmup steps before capture.
//
// This header holds ONLY the pure, CUDA-free pieces (the VT_GEMM_PLAN_CACHE
// flag predicate and the plan KEY: fields + equality + hash) so they are
// unit-testable on the CPU tier (tests/vt/test_gemm_plan_cache.cpp), exactly
// like fp8_plan_cache.h. The cache map itself holds a cuBLASLt heuristic
// result and therefore lives in cuda_matmul.cu (GetOrQueryGemmHeuristic).
//
// Bit-exactness: cuBLASLt algo selection is process-deterministic per shape
// (the same premise the fp8 plan cache documents), so returning the first
// query's result is numerically identical to re-querying — the cache changes
// WHEN the heuristic runs, never WHAT it returns.
//
// DEFAULT ON (VT_GEMM_PLAN_CACHE=0 disables) — deliberately the OPPOSITE
// polarity from the fp8 cache's opt-in. The fp8 cache is a performance lever
// whose premise did not reproduce, so it ships off. This cache is not a
// performance knob: on CUDA 13.3 the uncached path is WRONG under capture, so
// ON is the only safe default. The variable exists as an escape hatch and for
// same-binary A/B measurement (the board gate runs VT_GEMM_PLAN_CACHE=0 to
// reproduce status 14), not as a supported configuration.
#ifndef VT_CUDA_GEMM_PLAN_CACHE_H_
#define VT_CUDA_GEMM_PLAN_CACHE_H_

#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace vt::cuda {

// Pure predicate for the VT_GEMM_PLAN_CACHE contract: the cache is ON by
// default and DISABLED only for the exact value "0" (the rollback / A/B arm).
// nullptr (unset) and every other value — including typos like "00" or "false"
// — leave it ON, so a misspelled escape hatch cannot resurrect the capture bug
// (#1732). Kept separate from the cached getter below so the parse is
// unit-testable without touching the process-global cache.
inline bool GemmPlanCacheFlagIsOn(const char* env_value) {
  return !(env_value != nullptr && std::string_view(env_value) == "0");
}

// Process-cached gate, read from the environment exactly once (getenv on the
// first call only; every later hot-path call pays a single bool load). Same
// shape as Fp8PlanCacheEnabled: the parse is what the unit test pins; this
// latching getter is deliberately not unit-tested.
inline bool GemmPlanCacheEnabled() {
  static const bool enabled = GemmPlanCacheFlagIsOn(std::getenv("VT_GEMM_PLAN_CACHE"));
  return enabled;
}

// Values for GemmPlanKey::op: which of the three bf16/f32 cuBLASLt lanes the
// call came from. The lane fixes the descriptor's TRANSA/TRANSB and the layout
// ORDER/batch attributes, so the op is part of plan identity — an NN-selected
// algo must never be handed to a TN call of the same (m,n,k), and vice versa.
enum : int {
  kNn = 0,         // MatmulKernelCuda: row-major NN, no transpose attributes
  kTn = 1,         // MatmulBTKernelCuda: col-major TN (TRANSA=T, TRANSB=N)
  kBatchedNn = 2,  // BatchedMatmulKernelCuda: row-major strided-batched NN
};

// The FULL key that determines the cublasLtMatmulHeuristicResult_t for the
// bf16/f32 cuBLASLt lanes. EVERY value that reaches a cublasLtMatrixLayoutCreate
// or the matmul descriptor on these paths MUST appear here — a missed field =
// wrong heuristic reuse = a correctness bug, the same defect class the fp8 key
// documents (a field that selects an algo but is absent from the key). Fields
// are plain ints (the cuBLASLt enums cast to int at the call site) so this
// stays CUDA-free and CPU-testable.
//
// Field rationale (captured from the three kernels in cuda_matmul.cu):
//   device     — one cuBLASLt handle per device index (GetContext), and algo
//                availability is per-device.
//   op         — the lane: fixes TRANSA/TRANSB (TN sets T/N; the other two set
//                none), layout ORDER (row vs column major) and batch presence.
//   m, n, k    — the GEMM shape; drives every layout's extents and the algo
//                the heuristic selects.
//   lda/ldb/ldc — every leading dimension passed to a layout create, per lane:
//                NN        A=[m,k] ld=k, B=[k,n] ld=n, C=[m,n] ld=n;
//                TN        A=[k,n] ld=k, B=[k,m] ld=a.stride[0] (the MLA
//                          chunked-prefill slice can be wider than k), C=[n,m]
//                          ld=n;
//                batched   A=[m,k] ld=a.stride[1], B=[k,n] ld=b.stride[1],
//                          C=[m,n] ld=out.stride[1].
//   batch, stride_a/b/c — the strided-batched lane's BATCH_COUNT and
//                STRIDED_BATCH_OFFSET per operand (a.stride[0], b.stride[0],
//                out.stride[0]); 0 on the dense lanes, where no batch
//                attribute is set. A stride changes the layout the heuristic
//                sees, so it changes the key.
//   ab_type    — cudaDataType_t of the A/B operands (CUDA_R_32F or CUDA_R_16BF;
//                both operands always share it on these lanes). Changes both
//                operand layouts and the selected algo.
//   out_type   — cudaDataType_t of C/D (CUDA_R_32F or CUDA_R_16BF). Changes the
//                C/D layout and can change the selected algo — the algo-latching
//                variable the GDN forensic record flagged.
//
// Not in the key because they are lane constants, not call variables: the
// descriptor's compute type (CUBLAS_COMPUTE_32F), scale type (CUDA_R_32F) and
// epilogue (DEFAULT, never set) are identical on all three lanes, and the
// transposes are fixed by `op`. The heuristic's requestedAlgoCount
// (kGemvHeuristicAlgos) and MAX_WORKSPACE_BYTES preference are file constants
// shared by every query through GetOrQueryGemmHeuristic, so they cannot vary
// per key.
struct GemmPlanKey {
  int device = 0;
  int op = kNn;
  int64_t m = 0, n = 0, k = 0;
  int64_t lda = 0, ldb = 0, ldc = 0;
  int64_t batch = 0;
  int64_t stride_a = 0, stride_b = 0, stride_c = 0;
  int ab_type = 0;
  int out_type = 0;

  bool operator==(const GemmPlanKey& o) const {
    return device == o.device && op == o.op && m == o.m && n == o.n && k == o.k &&
           lda == o.lda && ldb == o.ldb && ldc == o.ldc && batch == o.batch &&
           stride_a == o.stride_a && stride_b == o.stride_b && stride_c == o.stride_c &&
           ab_type == o.ab_type && out_type == o.out_type;
  }
};

// FNV-1a-style hash over every key field, mirroring Fp8PlanKeyHash
// (order-independent correctness: the == above is the authority; the hash only
// needs to spread). Mixing each field in keeps distinct shapes, strides, ops
// and dtypes in different buckets.
struct GemmPlanKeyHash {
  std::size_t operator()(const GemmPlanKey& kk) const {
    std::size_t h = 1469598103934665603ull;  // FNV offset basis
    auto mix = [&h](std::uint64_t v) {
      h ^= static_cast<std::size_t>(v);
      h *= 1099511628211ull;  // FNV prime
    };
    mix(static_cast<std::uint64_t>(kk.device));
    mix(static_cast<std::uint64_t>(kk.op));
    mix(static_cast<std::uint64_t>(kk.m));
    mix(static_cast<std::uint64_t>(kk.n));
    mix(static_cast<std::uint64_t>(kk.k));
    mix(static_cast<std::uint64_t>(kk.lda));
    mix(static_cast<std::uint64_t>(kk.ldb));
    mix(static_cast<std::uint64_t>(kk.ldc));
    mix(static_cast<std::uint64_t>(kk.batch));
    mix(static_cast<std::uint64_t>(kk.stride_a));
    mix(static_cast<std::uint64_t>(kk.stride_b));
    mix(static_cast<std::uint64_t>(kk.stride_c));
    mix(static_cast<std::uint64_t>(kk.ab_type));
    mix(static_cast<std::uint64_t>(kk.out_type));
    return h;
  }
};

}  // namespace vt::cuda

#endif  // VT_CUDA_GEMM_PLAN_CACHE_H_
