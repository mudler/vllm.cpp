// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CPU-tier contract for the cuBLASLt heuristic-result cache's pure plumbing
// (src/vt/cuda/gemm_plan_cache.h): the VT_GEMM_PLAN_CACHE flag predicate (DEFAULT
// ON, exact "0" rollback) and the GemmPlanKey equality + hash. The cache map
// itself holds a cuBLASLt heuristic result and is CUDA-only (lives in
// cuda_matmul.cu); the board gate (graphs-on decode, issue #1732) is the
// end-to-end proof that the cache removes the in-capture heuristic query.
// This suite pins the KEY completeness — every layout/descriptor-affecting field
// distinguishes plans, because a collision here would reuse the wrong algo's
// heuristic result, a silent wrong-result bug no token gate can localize.
#include <doctest/doctest.h>

#include <cstdlib>
#include <string>
#include <unordered_map>

#include "vt/cuda/gemm_plan_cache.h"

using vt::cuda::GemmPlanCacheFlagIsOn;
using vt::cuda::GemmPlanKey;
using vt::cuda::GemmPlanKeyHash;

// DEFAULT ON — the opposite polarity from VT_FP8_PLAN_CACHE's opt-in, and the
// polarity is the point: on CUDA 13.3 the uncached path is not slower, it is
// WRONG under CUDA graph capture (issue #1732: cublasLtMatmulAlgoGetHeuristic
// returns status 14 on a created stream in capture), so only the exact value
// "0" disables and every typo'd value ("false", "no", "00") keeps the safe
// default. A misspelled escape hatch must not resurrect the capture bug.
TEST_CASE("VT_GEMM_PLAN_CACHE is ON by default; OFF only for exactly \"0\"") {
  CHECK(GemmPlanCacheFlagIsOn(nullptr));  // unset -> ON (the shipped default)
  CHECK(GemmPlanCacheFlagIsOn("1"));      // explicit ON is accepted too
  CHECK_FALSE(GemmPlanCacheFlagIsOn("0"));  // the one rollback value
  CHECK(GemmPlanCacheFlagIsOn(""));
  CHECK(GemmPlanCacheFlagIsOn("2"));
  CHECK(GemmPlanCacheFlagIsOn("on"));
  CHECK(GemmPlanCacheFlagIsOn("true"));
  CHECK(GemmPlanCacheFlagIsOn("00"));  // only the exact "0" disables
  CHECK(GemmPlanCacheFlagIsOn("0 "));  // trailing space must not disable
  CHECK(GemmPlanCacheFlagIsOn(" 0"));  // leading space must not disable
}

// The parse the latching getter GemmPlanCacheEnabled() wraps, fed through the
// real getenv surface with setenv/unsetenv so the pointer the getter would pass
// is the one tested. The getter itself latches on first read (a process-cached
// bool, mirroring Fp8PlanCacheEnabled), so it is deliberately not called here —
// same documented choice as the fp8 suite; the parse is the contract.
TEST_CASE("VT_GEMM_PLAN_CACHE: getenv-fed parse is ON unset, OFF for \"0\", ON for \"1\"") {
  unsetenv("VT_GEMM_PLAN_CACHE");
  CHECK(GemmPlanCacheFlagIsOn(std::getenv("VT_GEMM_PLAN_CACHE")));  // unset -> ON
  setenv("VT_GEMM_PLAN_CACHE", "0", /*overwrite=*/1);
  CHECK_FALSE(GemmPlanCacheFlagIsOn(std::getenv("VT_GEMM_PLAN_CACHE")));
  setenv("VT_GEMM_PLAN_CACHE", "1", /*overwrite=*/1);
  CHECK(GemmPlanCacheFlagIsOn(std::getenv("VT_GEMM_PLAN_CACHE")));
  unsetenv("VT_GEMM_PLAN_CACHE");  // restore: later cases must not see our env
}

namespace {
// A canonical TN (bt) plan key — the lane issue #1732 actually killed: the
// bf16 decode in_proj GEMV, f32 out, contiguous activation (ldb == k). Values
// are opaque ints for ab_type/out_type (the real call sites pass the
// cudaDataType_t enum values; the key never interprets them).
GemmPlanKey Base() {
  GemmPlanKey k;
  k.device = 0;
  k.op = vt::cuda::kTn;
  k.m = 1;
  k.n = 6144;
  k.k = 2048;
  k.lda = 2048;  // TN: A=[K,N] col-major, ld = K
  k.ldb = 2048;  // TN: B=[K,M] col-major, ld = the activation row stride
  k.ldc = 6144;  // TN: C=D=[N,M] col-major, ld = N
  k.batch = 0;     // dense lanes carry no batch metadata
  k.stride_a = 0;  // (0 for the non-batched ops, by construction at the call
  k.stride_b = 0;  //  sites — pinned here so the fields stay load-bearing in ==)
  k.stride_c = 0;
  k.ab_type = 14;  // CUDA_R_16BF stand-in
  k.out_type = 0;  // CUDA_R_32F stand-in
  return k;
}

// A canonical strided-batched NN key: same (m,n,k) as Base but a different op,
// per-op leading dims and non-zero batch/strides — the shape the MLA weight
// absorption sends through vt::BatchedMatmul.
GemmPlanKey BatchedBase() {
  GemmPlanKey k;
  k.device = 0;
  k.op = vt::cuda::kBatchedNn;
  k.m = 1;
  k.n = 6144;
  k.k = 2048;
  k.lda = 2048;  // batched NN: A=[M,K] row-major, ld = a.stride[1]
  k.ldb = 6144;  // batched NN: B=[K,N] row-major, ld = b.stride[1]
  k.ldc = 6144;  // batched NN: C=[M,N] row-major, ld = out.stride[1]
  k.batch = 64;
  k.stride_a = 2048;  // batch stride in elements, per tensor
  k.stride_b = 2048 * 6144;
  k.stride_c = 1 * 6144;
  k.ab_type = 14;
  k.out_type = 0;
  return k;
}
}  // namespace

TEST_CASE("GemmPlanKey: two identical keys are equal, hash the same, and land in one bucket") {
  const GemmPlanKey a = Base(), b = Base();
  CHECK(a == b);
  CHECK(GemmPlanKeyHash{}(a) == GemmPlanKeyHash{}(b));
  // The point of hash/== consistency: an equal key constructed elsewhere must
  // FIND the first entry through the map (bucket lookup), not insert a second.
  std::unordered_map<GemmPlanKey, int, GemmPlanKeyHash> m;
  m[a] = 1;
  const auto it = m.find(b);
  REQUIRE(it != m.end());
  CHECK(it->second == 1);
  CHECK(m.size() == 1);
  CHECK(m.count(b) == 1);
}

TEST_CASE("GemmPlanKey: perturbing ANY layout/descriptor field makes a DISTINCT key") {
  const GemmPlanKey base = Base();
  // Every field below reaches a cublasLtMatrixLayoutCreate (or, for op, the
  // descriptor's transpose config) on its lane. A field missing from == would
  // let a different GEMM reuse the wrong heuristic result — a wrong-algo bug.
  auto differs = [&](GemmPlanKey k) {
    CHECK_FALSE(base == k);
    std::unordered_map<GemmPlanKey, int, GemmPlanKeyHash> m;
    m[base] = 1;
    m[k] = 2;
    CHECK(m.size() == 2);  // equality is the authority; no silent aliasing
  };
  { GemmPlanKey k = base; k.device = 1;    differs(k); }
  { GemmPlanKey k = base; k.op = vt::cuda::kNn; differs(k); }
  { GemmPlanKey k = base; k.op = vt::cuda::kBatchedNn; differs(k); }
  { GemmPlanKey k = base; k.m = 2;         differs(k); }
  { GemmPlanKey k = base; k.n = 4096;      differs(k); }
  { GemmPlanKey k = base; k.k = 1024;      differs(k); }
  { GemmPlanKey k = base; k.lda = 1024;    differs(k); }  // A ld != K would be a different view
  { GemmPlanKey k = base; k.ldb = 4096;    differs(k); }  // B ld = the MLA sliced stride
  { GemmPlanKey k = base; k.ldc = 4096;    differs(k); }
  { GemmPlanKey k = base; k.batch = 2;     differs(k); }
  { GemmPlanKey k = base; k.stride_a = 8;  differs(k); }
  { GemmPlanKey k = base; k.stride_b = 8;  differs(k); }
  { GemmPlanKey k = base; k.stride_c = 8;  differs(k); }
  { GemmPlanKey k = base; k.ab_type = 0;   differs(k); }  // f32 operands vs bf16
  { GemmPlanKey k = base; k.out_type = 2;  differs(k); }  // bf16 out vs f32 out
}

TEST_CASE("GemmPlanKey: an NN plan can NEVER alias a TN plan of the same GEMM") {
  // The three lanes fix different descriptor transpose configs and layout
  // orders for the SAME (m,n,k): op is in == precisely so a row-major NN algo
  // is never handed to a col-major TN call. Collapsing op would run the TN
  // GEMM on an NN-selected heuristic — silent wrong results, no shape field
  // would catch it.
  GemmPlanKey nn = Base();
  nn.op = vt::cuda::kNn;
  const GemmPlanKey tn = Base();  // kTn
  CHECK_FALSE(nn == tn);
  std::unordered_map<GemmPlanKey, int, GemmPlanKeyHash> m;
  m[nn] = 1;
  m[tn] = 2;
  CHECK(m.size() == 2);
}

TEST_CASE("GemmPlanKey: batched keys separate on batch and each batch stride") {
  // The strided-batched lane's layouts carry BATCH_COUNT and STRIDED_BATCH_
  // OFFSET per operand; a missed stride field would replay one matrix's algo
  // over a differently-strided batch. Same-shape dense TN key must not alias
  // the batched key even where the numbers coincide on the shared fields.
  const GemmPlanKey batched = BatchedBase();
  CHECK_FALSE(batched == Base());  // op alone separates them here
  auto differs = [&](GemmPlanKey k) {
    CHECK_FALSE(batched == k);
    std::unordered_map<GemmPlanKey, int, GemmPlanKeyHash> m;
    m[batched] = 1;
    m[k] = 2;
    CHECK(m.size() == 2);
  };
  { GemmPlanKey k = batched; k.batch = 128;     differs(k); }
  { GemmPlanKey k = batched; k.stride_a = 4096; differs(k); }
  { GemmPlanKey k = batched; k.stride_b = 1;    differs(k); }
  { GemmPlanKey k = batched; k.stride_c = 8;    differs(k); }
  { GemmPlanKey k = batched; k.lda = 4096;      differs(k); }  // per-batch-matrix ld
  // Two batched keys equal on every field are one entry, as on the dense lane.
  std::unordered_map<GemmPlanKey, int, GemmPlanKeyHash> m;
  m[batched] = 1;
  m[BatchedBase()] = 2;  // overwrite through the equal key
  CHECK(m.size() == 1);
  CHECK(m[batched] == 2);
}
