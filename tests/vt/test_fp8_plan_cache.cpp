// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CPU-tier contract for the cuBLASLt FP8 plan-cache pure plumbing
// (src/vt/cuda/fp8_plan_cache.h): the VT_FP8_PLAN_CACHE flag predicate (default
// ON, "0" rollback) and the Fp8PlanKey equality + hash. The cache map itself
// holds cuBLASLt handles and is CUDA-only (lives in cuda_matmul.cu); the byte-
// exact cached-vs-fresh GEMM proof is the CUDA-tier test_ops_fp8_cutlass.cpp.
// This suite pins the KEY completeness (every descriptor/algo-affecting field
// distinguishes plans — a collision here would be a wrong-algo correctness bug)
// on every platform, not just DGX.
#include <doctest/doctest.h>

#include <unordered_map>

#include "vt/cuda/fp8_plan_cache.h"

using vt::cuda::Fp8PlanCacheFlagIsOn;
using vt::cuda::Fp8PlanKey;
using vt::cuda::Fp8PlanKeyHash;

TEST_CASE("VT_FP8_PLAN_CACHE is OFF by default; ON only for exactly \"1\"") {
  CHECK_FALSE(Fp8PlanCacheFlagIsOn(nullptr));  // unset -> OFF (default: rebuild per call)
  CHECK(Fp8PlanCacheFlagIsOn("1"));            // the opt-in: cache the plan
  CHECK_FALSE(Fp8PlanCacheFlagIsOn(""));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("2"));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("on"));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("true"));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("0"));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("11"));  // only the exact "1" enables
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("1 "));  // trailing space must not enable
  CHECK_FALSE(Fp8PlanCacheFlagIsOn(" 1"));  // leading space must not enable
}

namespace {
// A canonical fp8 TN plan key (the 35B-shape family: e4m3 A/B, bf16 out, TN
// transposes, host-alpha scale). Each test perturbs ONE field.
Fp8PlanKey Base() {
  Fp8PlanKey k;
  k.device = 0;
  k.m = 8;
  k.n = 6144;
  k.k = 2048;
  k.out_type = 1;      // CUDA_R_16BF stand-in (values are opaque ints here)
  k.a_type = 28;       // CUDA_R_8F_E4M3 stand-in
  k.compute_type = 68; // CUBLAS_COMPUTE_32F stand-in
  k.scale_type = 0;    // CUDA_R_32F stand-in
  k.trans_a = 1;       // CUBLAS_OP_T
  k.trans_b = 0;       // CUBLAS_OP_N
  k.epilogue = 1;      // CUBLASLT_EPILOGUE_DEFAULT
  k.scale_mode = 0;    // host-alpha folded
  return k;
}
}  // namespace

TEST_CASE("Fp8PlanKey: two identical keys are equal and hash the same") {
  const Fp8PlanKey a = Base(), b = Base();
  CHECK(a == b);
  CHECK(Fp8PlanKeyHash{}(a) == Fp8PlanKeyHash{}(b));
}

TEST_CASE("Fp8PlanKey: perturbing ANY descriptor/algo field makes a DISTINCT key") {
  const Fp8PlanKey base = Base();
  // Every field is part of what determines the cuBLASLt descriptor or the
  // selected algo; a missed field would let a different shape/config reuse the
  // wrong plan. Each perturbation must break equality (and, being distinct keys,
  // must not silently alias in the map).
  auto differs = [&](Fp8PlanKey k) {
    CHECK_FALSE(base == k);
    // A hash collision is legal but not expected for these small perturbations;
    // equality is the authority, so we assert the map treats them as 2 entries.
    std::unordered_map<Fp8PlanKey, int, Fp8PlanKeyHash> m;
    m[base] = 1;
    m[k] = 2;
    CHECK(m.size() == 2);
  };
  { Fp8PlanKey k = base; k.device = 1;       differs(k); }
  { Fp8PlanKey k = base; k.m = 1;            differs(k); }
  { Fp8PlanKey k = base; k.n = 4096;         differs(k); }
  { Fp8PlanKey k = base; k.k = 1024;         differs(k); }
  { Fp8PlanKey k = base; k.out_type = 0;     differs(k); }  // bf16 out vs f32 out
  { Fp8PlanKey k = base; k.a_type = 29;      differs(k); }
  { Fp8PlanKey k = base; k.compute_type = 0; differs(k); }
  { Fp8PlanKey k = base; k.scale_type = 1;   differs(k); }
  { Fp8PlanKey k = base; k.trans_a = 0;      differs(k); }
  { Fp8PlanKey k = base; k.trans_b = 1;      differs(k); }
  { Fp8PlanKey k = base; k.epilogue = 2;     differs(k); }
  { Fp8PlanKey k = base; k.scale_mode = 1;   differs(k); }
}

TEST_CASE("Fp8PlanKey: same shape but different output dtype -> distinct plans") {
  // The f32-out and bf16-out fp8 GEMMs share (m,n,k) but select different C/D
  // layouts and can latch different algos; they must never share a cached plan.
  Fp8PlanKey bf16_out = Base();
  bf16_out.out_type = 1;
  Fp8PlanKey f32_out = Base();
  f32_out.out_type = 0;
  CHECK_FALSE(bf16_out == f32_out);
  std::unordered_map<Fp8PlanKey, int, Fp8PlanKeyHash> m;
  m[bf16_out] = 1;
  m[f32_out] = 2;
  CHECK(m.size() == 2);
}
