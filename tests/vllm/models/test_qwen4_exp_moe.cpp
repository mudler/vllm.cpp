// MODEL-MM-QWEN4-EXP W5d-4 — the MoE weight adapter, gated against an
// independent host reimplementation of the lane-pinned oracle's own
// `Qwen4ExpTextSparseMoeBlock.forward`.
//
// Issue #2249 item 4, wave issue #2031, campaign issue #1978, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT THIS SUITE HAS TO SEE, AND WHY A SHAPE CHECK CANNOT ────────────────
// #2249 says of this item: "the `_kq` arm's shapes match and `KqExpertSlice` is
// dtype-generic, so this looks small". `NaiveStackedKq` below is that sentence
// written out — the stacked towers handed to `MoeBlockWeights::expert_*_kq`
// verbatim — and `RunMoeBlock` on it throws
// "matmul_bt_quant_grouped: rank-2 out/act/weight required", because
// `LoadStackedExperts` records rank 3 and the seam's field is rank 2. That case
// is the RED this wave was written against and it stays in the suite: it is the
// only thing that keeps the next reader from re-deriving "the shapes match".
//
// THAT RED IS ROUTE-CONDITIONAL, and the suite says so rather than leaving a
// reader to discover it. `Qwen35GroupedMoeEnabled()` (`qwen3_5.cpp:6299-6302`)
// is ON by default; with `VT_QWEN35_GROUPED_MOE=0` the seam takes the
// per-expert `ExpertMlpKq` path, which reaches `KqResidentSlice`
// (`qwen3_5.cpp:5665-5678`), and that helper rebuilds a rank-2 view from its
// `N`/`K` ARGUMENTS by pointer arithmetic and never reads the declared rank. A
// rank-3 tower does not throw there, and since the tower is contiguous
// `[E, N, K]` the slice `row_off = e * N` is the right one, so it answers
// CORRECTLY. `GroupedRoute()` below reads the same environment variable the
// seam reads, and each case asserts the behaviour of the route it is on. Both
// routes are gated; neither is assumed.
//
// The value gate is `MoeReference`, a from-scratch double-precision
// reimplementation of the oracle at transformers **5.16.0**
// (`.agents/oracles/transformers.md`, this row's accepted lane pin):
// `Qwen4ExpTextSparseMoeBlock.forward` (:927-938),
// `Qwen4ExpTextTopKRouter.forward` (:907-916),
// `Qwen4ExpTextExperts.forward` (:869-894) and `Qwen4ExpTextMLP` (:842-855).
// It reads the SOURCE `Qwen4ExpMoeWeights` — the stacked `[E,I,H]` towers — not
// the `MoeBlockWeights` the adapter produced, so a mis-sliced expert, a swapped
// gate/up or a transposed slab has nothing on the other side agreeing with it.
//
// TWO PRECISION STATEMENTS, so the tolerance is a bound and not a hope:
//
//  * The GEMMs on the bf16 arm are EXACT here by construction. Every weight is
//    an integer multiple of 2^-4 with |k| <= 8 and every activation an integer
//    multiple of 2^-8 with |m| <= 127, so each product is an integer multiple of
//    2^-12 below 0.25 and a 64-term sum needs 16 significand bits — well inside
//    f32. The residual difference is therefore the bf16 STORES, at the two
//    points upstream stores bf16 too (the router logits at :909 before its f32
//    softmax at :910, and the per-expert output at :893), which the reference
//    reproduces, plus the f32 silu.
//  * The keep-quant arm's bound is MEASURED, not derived, and it is stated that
//    way because the derivation an earlier draft gave here was WRONG. All three
//    of its GEMMs quantize their activation to q8_0 through `kMatmulBTQuant`
//    (cpu_quant_gemm.cpp path 1, mirroring ggml), and none of the three is
//    exact. `QuantizeRowQ8_0` (`src/vt/cpu/cpu_quant_act.cpp:52-81`, its
//    per-block `amax` loop at `:58-69`) takes `amax` per 32-ELEMENT BLOCK,
//    not per row; `HiddenCodes()` forces
//    `|code| = 127` at element 0 of each ROW only, and `kH = 64` is TWO blocks,
//    so block 1 gets whatever amax its random codes happen to carry, its `d` is
//    not `2^-8`, and the gate and up projections carry quantization error too.
//    The down projection carries a further term the reference does not model at
//    all, since its activation is `silu(g)*u`, which nothing here controls. So
//    the number below is what the suite MEASURES: it is printed by a `MESSAGE`
//    on every run, and it is asserted an order of magnitude below every
//    mutation margin. Forcing `|code| = 127` in every 32-element block of every
//    row would make the derivation true; it is deliberately not done, because
//    it moves the router logits and therefore the routing, and the seven
//    mutation margins this wave was gated with have been independently
//    reproduced against THIS fixture. The spec's `## Owed` carries that.
//
// UPSTREAM PRECISION NOTE, recorded rather than mirrored: upstream casts the
// renormalized top-k weights back to the model dtype
// (`router_top_value.to(router_logits.dtype)`, :914) and our shared seam keeps
// them f32 (`vt::MoeRouterTopK` writes an f32 `dtw`, qwen3_5.cpp:7239-7242). The
// seam is therefore WIDER than the oracle by one bf16 rounding of a value in
// [0,1]. That is a divergence, not a feature, and it is NOT defended here as
// being "more precise": AGENTS.md §"Inherit vLLM defaults" is explicit that a
// token gate cannot see a dtype that is too wide, which is exactly the argument
// "more precise" would be. The width is INHERITED — it is what `MoeRouterTopK`
// writes and what every Qwen MoE in this tree reads, and narrowing it is a
// seam-level change, not an adapter-level one — so it is recorded under the
// spec's `## Owed` rather than resolved in this wave. The reference keeps the
// weights in double at that point so the suite measures the adapter and not
// this rounding.
//
// TWO LIMITS OF THE FIXTURE, so nobody reads it as wider than it is. The
// keep-quant towers here are hand-written Q8_0 blocks, so the keep-quant arm is
// value-proven at Q8_0 ONLY: the adapter is dtype-generic on that arm by
// construction (it re-declares rank and copies no bytes, and
// `MatmulBTQuantGrouped` accepts any `IsBlockQuant` dtype), but no k-quant tower
// is executed here and every released checkpoint of this model is a k-quant.
// And `BuildSource`'s bf16 towers are `nk = false` where `LoadStackedExperts`
// produces `ExpandBf16(..., /*nk=*/true)`; the adapter stamps `nk = true` on the
// per-expert views it hands the seam either way, so the gated orientation is the
// production one, but the SOURCE is not the loader's own output. Both are owed
// in the spec.
//
// CPU only: this is a weight adapter and a composition, there is no CUDA arm to
// compare, and a CUDA arm written on a CPU host could not be gated. The one
// GPU-only fact the adapter depends on is asserted STATICALLY instead — see
// "the dtype the CUDA GEMM refuses" below.
#include "vllm/model_executor/models/qwen4_exp_moe.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

// MOEDIV (#2552): the selected-expert-id tap prints on stderr, so this suite
// captures the descriptor around ONE `RunBlock` call. POSIX only; the tap case
// refuses to report on a platform without it rather than passing vacuously.
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#define VLLM_MOE_SEL_FP_CAPTURE 1
#endif

#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

#include "../../support/max_abs_diff.h"

namespace {

using vllm::MoeBlockWeights;
using vllm::OwnedTensor;
using vllm::Qwen4ExpMoeWeights;
using vllm::Qwen4ExpParams;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// --- geometry --------------------------------------------------------------
//
// E = 6 with top_k = 3 is the smallest pair that can carry a NON-TRIVIAL expert
// permutation: more than one expert active per token, a different set for
// different tokens, and at least one expert never selected. `AssertRoutingIsNonTrivial`
// gates all three, because an identity assignment makes a wrong expert index
// return the right answer and every mutation below would survive it.
constexpr int64_t kH = 64;   // whole q8_0 blocks: gate/up reduce over K = H
constexpr int64_t kE = 6;
constexpr int64_t kTopK = 3;
constexpr int64_t kI = 32;   // whole q8_0 blocks: down reduces over K = I
constexpr int64_t kIs = 16;
constexpr int64_t kT = 5;

// Weight unit 2^-4 and activation unit 2^-8: see the precision statement above.
constexpr float kWU = 0.0625F;
constexpr float kAU = 0.00390625F;
constexpr int kWMax = 8;    // |weight code|
constexpr int kAMax = 127;  // |activation code| — and exactly the q8_0 range

// Deterministic integer stream. A fixed seed per buffer keeps every case
// reproducible and keeps two buffers from accidentally sharing a pattern.
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  int Next(int lo, int hi) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint64_t r = (s >> 33) % static_cast<uint64_t>(hi - lo + 1);
    return lo + static_cast<int>(r);
  }
};

std::vector<int> Codes(size_t n, uint64_t seed, int lim) {
  Lcg g(seed);
  std::vector<int> c(n);
  for (size_t i = 0; i < n; ++i) c[i] = g.Next(-lim, lim);
  return c;
}

Qwen4ExpParams Params() {
  Qwen4ExpParams p;
  p.hidden_size = kH;
  p.num_experts = kE;
  p.num_experts_per_tok = kTopK;
  p.moe_intermediate_size = kI;
  p.shared_expert_intermediate_size = kIs;
  return p;
}

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

// WHICH EXPERT ROUTE THE SEAM WILL TAKE, read from the same environment variable
// with the same predicate `Qwen35GroupedMoeEnabled()` uses
// (`qwen3_5.cpp:6299-6302`). It is duplicated rather than called because the
// seam's copy is file-local to qwen3_5.cpp and is not declared in any header.
// The duplication is one `getenv` and one string compare, and it is the reason
// this suite can gate BOTH routes instead of silently asserting the default
// one's behaviour on whichever route it happens to run.
//
// The seam caches its answer in a function-local `static const`, which stops the
// value being flipped MID-PROCESS — not before launch. So
// `VT_QWEN35_GROUPED_MOE=0 ./tests/test_qwen4_exp_moe` really does exercise the
// per-expert `ExpertMlpKq` path, and that is a supported way to run this suite.
bool GroupedRoute() {
  const char* e = std::getenv("VT_QWEN35_GROUPED_MOE");
  return e == nullptr || std::string(e) != "0";
}

// `std::string`, NOT `const char*`. doctest stringifies a bare `char*` through
// its BOOL overload, so a `MESSAGE(... << RouteName())` returning a pointer
// prints "1" for every route and the line reports nothing at all. Measured on
// the first run of this helper.
std::string RouteName() {
  return GroupedRoute() ? std::string("grouped (default, VT_QWEN35_GROUPED_MOE unset or != 0)")
                        : std::string("per-expert ExpertMlpKq (VT_QWEN35_GROUPED_MOE=0)");
}

OwnedTensor Make(DType dt, const std::vector<int64_t>& shape, size_t bytes, bool nk) {
  OwnedTensor o;
  o.dtype = dt;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) o.shape[i] = shape[i];
  o.bytes.resize(bytes);
  return o;
}

OwnedTensor F32Tensor(const std::vector<int>& codes, const std::vector<int64_t>& shape) {
  OwnedTensor o = Make(DType::kF32, shape, codes.size() * sizeof(float), /*nk=*/false);
  auto* p = reinterpret_cast<float*>(o.bytes.data());
  for (size_t i = 0; i < codes.size(); ++i) p[i] = static_cast<float>(codes[i]) * kWU;
  return o;
}

OwnedTensor Bf16Tensor(const std::vector<int>& codes, const std::vector<int64_t>& shape,
                       bool nk) {
  OwnedTensor o = Make(DType::kBF16, shape, codes.size() * sizeof(uint16_t), nk);
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < codes.size(); ++i)
    p[i] = vt::F32ToBF16(static_cast<float>(codes[i]) * kWU);
  return o;
}

// One q8_0 tower holding EXACTLY the same values as `Bf16Tensor(codes, ...)`:
// block scale d = 2^-4 (exact in f16) and qs = the code itself, so
// `dequantize_row_q8_0`'s `d * qs[i]` reproduces `code * kWU` bit for bit. The
// blocks are written by hand rather than quantized so the two arms are gated
// against ONE reference instead of against each other.
OwnedTensor Q8_0Tensor(const std::vector<int>& codes, const std::vector<int64_t>& shape,
                       int64_t rows, int64_t k) {
  REQUIRE(k % 32 == 0);
  REQUIRE(static_cast<int64_t>(codes.size()) == rows * k);
  const int64_t nblk = k / 32;
  const size_t row_bytes = static_cast<size_t>(nblk) * 34;
  OwnedTensor o = Make(DType::kQ8_0, shape, static_cast<size_t>(rows) * row_bytes,
                       /*nk=*/false);
  uint8_t* p = o.bytes.data();
  const uint16_t d = vt::F32ToF16(kWU);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < nblk; ++b) {
      uint8_t* blk = p + static_cast<size_t>(r) * row_bytes + static_cast<size_t>(b) * 34;
      std::memcpy(blk, &d, sizeof(uint16_t));
      for (int64_t j = 0; j < 32; ++j) {
        blk[2 + j] = static_cast<uint8_t>(
            static_cast<int8_t>(codes[static_cast<size_t>(r * k + b * 32 + j)]));
      }
    }
  }
  return o;
}

// --- the source weight set -------------------------------------------------
//
// The codes are held alongside the tensors so the reference reads the SAME
// integers the tensors encode, in either residency, without decoding anything
// the code under test produced.
struct MoeSource {
  Qwen4ExpMoeWeights w;
  std::vector<int> router, shared_gate, gate, up, down, sg, su, sd;
};

MoeSource BuildSource(bool keep_quant) {
  MoeSource s;
  s.router = Codes(static_cast<size_t>(kE * kH), 1656, kWMax);
  s.shared_gate = Codes(static_cast<size_t>(kH), 12, kWMax);
  s.gate = Codes(static_cast<size_t>(kE * kI * kH), 13, kWMax);
  s.up = Codes(static_cast<size_t>(kE * kI * kH), 14, kWMax);
  s.down = Codes(static_cast<size_t>(kE * kH * kI), 15, kWMax);
  s.sg = Codes(static_cast<size_t>(kIs * kH), 16, kWMax);
  s.su = Codes(static_cast<size_t>(kIs * kH), 17, kWMax);
  s.sd = Codes(static_cast<size_t>(kH * kIs), 18, kWMax);

  // `LoadMoe` leaves both of these f32 on purpose (qwen4_exp_weights.cpp:437-447).
  s.w.router = F32Tensor(s.router, {kE, kH});
  s.w.shared_gate = F32Tensor(s.shared_gate, {kH});
  if (keep_quant) {
    s.w.gate_exps = Q8_0Tensor(s.gate, {kE, kI, kH}, kE * kI, kH);
    s.w.up_exps = Q8_0Tensor(s.up, {kE, kI, kH}, kE * kI, kH);
    s.w.down_exps = Q8_0Tensor(s.down, {kE, kH, kI}, kE * kH, kI);
  } else {
    s.w.gate_exps = Bf16Tensor(s.gate, {kE, kI, kH}, /*nk=*/true);
    s.w.up_exps = Bf16Tensor(s.up, {kE, kI, kH}, /*nk=*/true);
    s.w.down_exps = Bf16Tensor(s.down, {kE, kH, kI}, /*nk=*/true);
  }
  s.w.shared_gate_proj = Bf16Tensor(s.sg, {kIs, kH}, /*nk=*/true);
  s.w.shared_up_proj = Bf16Tensor(s.su, {kIs, kH}, /*nk=*/true);
  s.w.shared_down_proj = Bf16Tensor(s.sd, {kH, kIs}, /*nk=*/true);
  return s;
}

// Activation codes m in [-127,127] with |m| = 127 forced at element 0 of every
// row, so `quantize_row_q8_0` picks d = 127*kAU/127 = kAU exactly and recovers
// every code without loss (see the precision statement in the file header).
std::vector<int> HiddenCodes() {
  std::vector<int> c = Codes(static_cast<size_t>(kT * kH), 19, kAMax);
  for (int64_t t = 0; t < kT; ++t) c[static_cast<size_t>(t * kH)] = (t % 2 == 0) ? kAMax : -kAMax;
  return c;
}

// --- the oracle reimplementation -------------------------------------------

double Silu(double x) { return x / (1.0 + std::exp(-x)); }
double Bf16(double x) { return static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(x)))); }

struct RefOut {
  std::vector<double> y;                    // [T, H]
  std::vector<std::vector<int>> selected;   // [T][top_k], oracle order
  std::vector<double> margin;               // [T] p_k - p_{k+1}
  // MOEDIV (#2552): the bf16 BITS of the router logits. The tap's margin is in
  // logit space and its unit is bf16 ulps, so gating it needs the bit patterns
  // and not the doubles they were rounded from.
  std::vector<uint16_t> logit_bits;         // [T * E]
};

RefOut MoeReference(const MoeSource& s, const std::vector<int>& xc) {
  RefOut r;
  r.y.assign(static_cast<size_t>(kT) * kH, 0.0);
  r.selected.resize(static_cast<size_t>(kT));
  r.margin.assign(static_cast<size_t>(kT), 0.0);
  r.logit_bits.assign(static_cast<size_t>(kT) * kE, 0);
  for (int64_t t = 0; t < kT; ++t) {
    std::vector<double> x(static_cast<size_t>(kH));
    for (int64_t h = 0; h < kH; ++h)
      x[static_cast<size_t>(h)] = static_cast<double>(xc[static_cast<size_t>(t * kH + h)]) * kAU;

    // Router: F.linear(x, weight[E,H]) in the model dtype, then a float softmax
    // (modeling_qwen4_exp.py:909-910).
    std::vector<double> logit(static_cast<size_t>(kE));
    for (int64_t e = 0; e < kE; ++e) {
      double a = 0.0;
      for (int64_t h = 0; h < kH; ++h)
        a += x[static_cast<size_t>(h)] *
             (static_cast<double>(s.router[static_cast<size_t>(e * kH + h)]) * kWU);
      logit[static_cast<size_t>(e)] = Bf16(a);
      r.logit_bits[static_cast<size_t>(t * kE + e)] = vt::F32ToBF16(static_cast<float>(a));
    }
    const double mx = *std::max_element(logit.begin(), logit.end());
    std::vector<double> prob(static_cast<size_t>(kE));
    double den = 0.0;
    for (int64_t e = 0; e < kE; ++e) {
      prob[static_cast<size_t>(e)] = std::exp(logit[static_cast<size_t>(e)] - mx);
      den += prob[static_cast<size_t>(e)];
    }
    for (double& p : prob) p /= den;

    std::vector<int> order(static_cast<size_t>(kE));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return prob[static_cast<size_t>(a)] > prob[static_cast<size_t>(b)];
    });
    r.selected[static_cast<size_t>(t)].assign(order.begin(), order.begin() + kTopK);
    r.margin[static_cast<size_t>(t)] = prob[static_cast<size_t>(order[kTopK - 1])] -
                                       prob[static_cast<size_t>(order[kTopK])];

    // norm_topk_prob (:912-913). Upstream's default is True
    // (configuration_qwen4_exp.py:163) and the shared seam hardcodes it.
    double wsum = 0.0;
    for (int64_t j = 0; j < kTopK; ++j) wsum += prob[static_cast<size_t>(order[j])];

    // Routed experts (:869-894): gate/up over [I,H], SwiGLU in the model dtype,
    // down over [H,I], scaled by the routing weight and accumulated.
    for (int64_t j = 0; j < kTopK; ++j) {
      const int64_t e = order[static_cast<size_t>(j)];
      const double rw = prob[static_cast<size_t>(e)] / wsum;
      std::vector<double> act(static_cast<size_t>(kI));
      for (int64_t i = 0; i < kI; ++i) {
        double g = 0.0, u = 0.0;
        for (int64_t h = 0; h < kH; ++h) {
          const size_t o = static_cast<size_t>((e * kI + i) * kH + h);
          g += x[static_cast<size_t>(h)] * (static_cast<double>(s.gate[o]) * kWU);
          u += x[static_cast<size_t>(h)] * (static_cast<double>(s.up[o]) * kWU);
        }
        act[static_cast<size_t>(i)] = Bf16(Silu(g) * u);
      }
      for (int64_t h = 0; h < kH; ++h) {
        double d = 0.0;
        for (int64_t i = 0; i < kI; ++i)
          d += act[static_cast<size_t>(i)] *
               (static_cast<double>(s.down[static_cast<size_t>((e * kH + h) * kI + i)]) * kWU);
        r.y[static_cast<size_t>(t * kH + h)] += rw * Bf16(d);
      }
    }

    // Shared expert (:842-855) and its sigmoid gate (:934).
    std::vector<double> sact(static_cast<size_t>(kIs));
    for (int64_t i = 0; i < kIs; ++i) {
      double g = 0.0, u = 0.0;
      for (int64_t h = 0; h < kH; ++h) {
        const size_t o = static_cast<size_t>(i * kH + h);
        g += x[static_cast<size_t>(h)] * (static_cast<double>(s.sg[o]) * kWU);
        u += x[static_cast<size_t>(h)] * (static_cast<double>(s.su[o]) * kWU);
      }
      sact[static_cast<size_t>(i)] = Bf16(Silu(g) * u);
    }
    double gl = 0.0;
    for (int64_t h = 0; h < kH; ++h)
      gl += x[static_cast<size_t>(h)] *
            (static_cast<double>(s.shared_gate[static_cast<size_t>(h)]) * kWU);
    const double gate = 1.0 / (1.0 + std::exp(-gl));
    for (int64_t h = 0; h < kH; ++h) {
      double d = 0.0;
      for (int64_t i = 0; i < kIs; ++i)
        d += sact[static_cast<size_t>(i)] *
             (static_cast<double>(s.sd[static_cast<size_t>(h * kIs + i)]) * kWU);
      r.y[static_cast<size_t>(t * kH + h)] += gate * d;
    }
  }
  return r;
}

// The trap this row has already produced: an identity expert assignment makes a
// wrong index return the right answer. Gated, not assumed.
void AssertRoutingIsNonTrivial(const RefOut& r) {
  std::vector<std::string> sets;
  std::vector<bool> seen(static_cast<size_t>(kE), false);
  bool any_non_prefix = false;
  for (int64_t t = 0; t < kT; ++t) {
    std::string k;
    bool prefix = true;
    for (int64_t j = 0; j < kTopK; ++j) {
      const int e = r.selected[static_cast<size_t>(t)][static_cast<size_t>(j)];
      seen[static_cast<size_t>(e)] = true;
      k += std::to_string(e) + ",";
      if (e >= static_cast<int>(kTopK)) prefix = false;
    }
    if (!prefix) any_non_prefix = true;
    sets.push_back(k);
  }
  std::sort(sets.begin(), sets.end());
  sets.erase(std::unique(sets.begin(), sets.end()), sets.end());
  // GATE THE NUMBER THAT IS STATED, not a weaker one. The fixture's seeds were
  // searched for FIVE distinct selected sets across the five tokens, and that
  // is what the wave's prose claims; `>= 2` would let four of the five collapse
  // without a word. `kT` rather than a literal 5 so the two cannot drift apart.
  INFO("distinct selected sets = " << sets.size());
  CHECK(sets.size() == static_cast<size_t>(kT));
  CHECK(any_non_prefix);
  const int64_t unused = std::count(seen.begin(), seen.end(), false);
  INFO("experts never selected = " << unused);
  CHECK(unused >= 1);
  double worst = 1.0;
  for (double m : r.margin) worst = std::min(worst, m);
  MESSAGE("smallest top-k probability margin = " << worst);
  // A discrete selection gate has BIMODAL error, not a tolerance: a flipped
  // expert is an order-1 wrong answer, so what has to be bounded is the
  // SELECTION, and the margin is the only thing that bounds it. The seam rounds
  // the router logits to bf16 (~2^-9 relative on |logit| <= 4, so <= 0.008
  // absolute) and dp/dlogit <= 1/4, which caps the shift in this gap at ~0.004.
  // The fixture's seeds were SEARCHED for the widest gap that still keeps five
  // distinct selected sets, a set outside the {0,1,2} prefix and one expert
  // never chosen; it measures 0.074, so this bound sits an order of magnitude
  // above the perturbation and still reds if the fixture drifts.
  CHECK(worst > 0.03);
}

// Run the block through the production entry point and return [T,H] as f32.
std::vector<float> RunBlock(const MoeBlockWeights& mw, const Qwen4ExpParams& p,
                            const std::vector<int>& xc) {
  Queue q = CpuQ();
  std::vector<uint16_t> x(static_cast<size_t>(kT) * kH);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = vt::F32ToBF16(static_cast<float>(xc[i]) * kAU);
  Tensor dh = Tensor::Contiguous(x.data(), DType::kBF16, Cpu(), {kT, kH});
  vllm::MoeBlockOutput out = vllm::RunQwen4ExpMoeBlock(q, mw, p, dh, kT);
  const auto* y = static_cast<const uint16_t*>(out.tensor.data);
  std::vector<float> f(static_cast<size_t>(kT) * kH);
  for (size_t i = 0; i < f.size(); ++i) f[i] = vt::BF16ToF32(y[i]);
  return f;
}

double Scale(const std::vector<double>& y) {
  double m = 0.0;
  for (double v : y) m = std::max(m, std::abs(v));
  return m;
}

}  // namespace

// ─── THE RED THIS WAVE WAS WRITTEN AGAINST ───────────────────────────────────
TEST_CASE("Qwen4Exp MoE: the stacked towers do NOT drop into the seam's _kq fields") {
  MoeSource s = BuildSource(/*keep_quant=*/true);
  const Qwen4ExpParams p = Params();

  // #2249 item 4's "the shapes match", written out: the rank-3 [E,N,K] towers
  // assigned to the rank-2 [E*N,K] fields.
  MoeBlockWeights naive;
  naive.router_gate = s.w.router;
  naive.shared_gate = s.w.shared_gate;
  naive.expert_gate_kq = s.w.gate_exps;
  naive.expert_up_kq = s.w.up_exps;
  naive.expert_down_kq = s.w.down_exps;
  naive.shared_gate_proj = s.w.shared_gate_proj;
  naive.shared_up_proj = s.w.shared_up_proj;
  naive.shared_down_proj = s.w.shared_down_proj;
  REQUIRE(naive.expert_gate_kq.rank == 3);

  // Step one: it does not even reach the experts. `LoadMoe` leaves the router
  // f32 [E,H] with nk unset, so the seam's router GEMM is `vt::Matmul` over
  // [T,H] x [E,H] and the inner dims disagree. (On CUDA the same tensor reaches
  // `MatmulBTKernelCuda` and is refused for its DTYPE instead — see the last
  // case in this file.)
  const std::vector<int> xc = HiddenCodes();
  CHECK_THROWS_WITH_AS(RunBlock(naive, p, xc),
                       doctest::Contains("matmul: inner dims mismatch"), std::exception);

  // Step two, which is the claim itself: give it the adapter's router and shared
  // gate so the router GEMM is out of the way, and put the RANK-3 towers back in
  // the rank-2 `_kq` fields. `vt::MatmulBTQuantGrouped`, the DEFAULT keep-quant
  // route, refuses them by rank.
  MoeBlockWeights mixed = vllm::Qwen4ExpMoeBlockWeights(s.w, p);
  mixed.expert_gate_kq = s.w.gate_exps;
  mixed.expert_up_kq = s.w.up_exps;
  mixed.expert_down_kq = s.w.down_exps;
  REQUIRE(mixed.expert_gate_kq.rank == 3);
  MESSAGE("expert route = " << RouteName());
  if (GroupedRoute()) {
    CHECK_THROWS_WITH_AS(RunBlock(mixed, p, xc),
                         doctest::Contains("matmul_bt_quant_grouped: rank-2"),
                         std::exception);
  } else {
    // THE OTHER HALF OF #2249 ITEM 4, asserted rather than assumed away. On the
    // per-expert route `ExpertMlpKq` reaches `KqResidentSlice`
    // (`qwen3_5.cpp:5665-5678`), which rebuilds a rank-2 view from its `N`/`K`
    // ARGUMENTS and never reads the declared rank — so the rank-3 tower is
    // accepted, and because the tower is contiguous `[E, N, K]` the offset
    // `e * N` is the right slice and the ANSWER IS RIGHT. Item 4's "the shapes
    // match" is literally true here and false on the default route; a suite
    // that only ever ran the default one could state neither.
    const RefOut ref = MoeReference(s, xc);
    const std::vector<float> got = RunBlock(mixed, p, xc);
    const double worst = vllm_test::MaxAbsDiff(got, ref.y.data(), ref.y.size());
    MESSAGE("rank-3 towers on the per-expert route: max|diff| = " << worst);
    CHECK(std::isfinite(worst));
    CHECK(worst < 0.05 * Scale(ref.y));
  }

  // And the adapter's own output is what the field wants: rank 2, [E*I, H].
  MoeBlockWeights mw = vllm::Qwen4ExpMoeBlockWeights(s.w, p);
  CHECK(mw.expert_gate_kq.rank == 2);
  CHECK(mw.expert_gate_kq.shape[0] == kE * kI);
  CHECK(mw.expert_gate_kq.shape[1] == kH);
  CHECK(mw.expert_down_kq.rank == 2);
  CHECK(mw.expert_down_kq.shape[0] == kE * kH);
  CHECK(mw.expert_down_kq.shape[1] == kI);
  // Zero copy: the view must point INTO the source tower, not at a duplicate.
  CHECK(mw.expert_gate_kq.bytes.data() == s.w.gate_exps.bytes.data());
  CHECK(mw.expert_down_kq.bytes.data() == s.w.down_exps.bytes.data());
  // The bf16 per-expert vectors stay EMPTY on this arm — `MoeBlock` dispatches
  // on `expert_gate_kq` and a populated pair would be dead weight.
  CHECK(mw.expert_gate.empty());
  CHECK(mw.expert_up.empty());
  CHECK(mw.expert_down.empty());
}

TEST_CASE("Qwen4Exp MoE: the bf16 arm matches the lane-pinned oracle") {
  MoeSource s = BuildSource(/*keep_quant=*/false);
  const Qwen4ExpParams p = Params();
  const std::vector<int> xc = HiddenCodes();
  const RefOut ref = MoeReference(s, xc);
  AssertRoutingIsNonTrivial(ref);

  MoeBlockWeights mw = vllm::Qwen4ExpMoeBlockWeights(s.w, p);
  REQUIRE(mw.expert_gate.size() == static_cast<size_t>(kE));
  REQUIRE(mw.expert_up.size() == static_cast<size_t>(kE));
  REQUIRE(mw.expert_down.size() == static_cast<size_t>(kE));
  REQUIRE(mw.expert_gate_kq.Empty());
  for (int64_t e = 0; e < kE; ++e) {
    const OwnedTensor& g = mw.expert_gate[static_cast<size_t>(e)];
    const OwnedTensor& d = mw.expert_down[static_cast<size_t>(e)];
    CHECK(g.rank == 2);
    CHECK(g.shape[0] == kI);
    CHECK(g.shape[1] == kH);
    CHECK(g.nk);
    CHECK(d.shape[0] == kH);
    CHECK(d.shape[1] == kI);
    // Zero copy, at the expert's own offset in the stacked buffer.
    CHECK(g.bytes.data() == s.w.gate_exps.bytes.data() +
                                static_cast<size_t>(e) * kI * kH * sizeof(uint16_t));
  }

  const std::vector<float> got = RunBlock(mw, p, xc);
  const double scale = Scale(ref.y);
  INFO("|reference| max = " << scale);
  const double worst = vllm_test::MaxAbsDiff(got, ref.y.data(), ref.y.size());
  // The ROUTE is printed beside the number because the two are only known to be
  // equal by measurement: `qwen3_5.cpp:7261` claims the grouped path is
  // byte-identical to the per-expert scatter, and running this suite both ways
  // is what turns that claim into a reading.
  MESSAGE("bf16 arm max|diff| = " << worst << " over |reference| max " << scale
                                  << ", expert route = " << RouteName());
  CHECK(std::isfinite(worst));
  CHECK(worst < 0.02 * scale);
}

TEST_CASE("Qwen4Exp MoE: the keep-quant arm matches the same oracle") {
  MoeSource s = BuildSource(/*keep_quant=*/true);
  const Qwen4ExpParams p = Params();
  const std::vector<int> xc = HiddenCodes();
  const RefOut ref = MoeReference(s, xc);
  AssertRoutingIsNonTrivial(ref);

  MoeBlockWeights mw = vllm::Qwen4ExpMoeBlockWeights(s.w, p);
  REQUIRE(!mw.expert_gate_kq.Empty());
  const std::vector<float> got = RunBlock(mw, p, xc);
  const double scale = Scale(ref.y);
  const double worst = vllm_test::MaxAbsDiff(got, ref.y.data(), ref.y.size());
  // MESSAGE, not only INFO: a tolerance whose measured value is invisible is a
  // number nobody re-reads when the fixture drifts.
  MESSAGE("keep-quant arm max|diff| = " << worst << " over |reference| max " << scale
                                        << ", expert route = " << RouteName());
  CHECK(std::isfinite(worst));
  // Wider than the bf16 arm by exactly one term: the down projection's q8_0
  // activation quantization (see the file header). Still an order of magnitude
  // below every mutation margin this suite was gated with.
  CHECK(worst < 0.05 * scale);
}

TEST_CASE("Qwen4Exp MoE: the adapter refuses by name") {
  const Qwen4ExpParams p = Params();

  SUBCASE("a mixed expert residency, which the seam reads as keep-quant") {
    MoeSource s = BuildSource(/*keep_quant=*/true);
    MoeSource b = BuildSource(/*keep_quant=*/false);
    s.w.down_exps = b.w.down_exps;  // gate/up q8_0, down bf16
    CHECK_THROWS_WITH_AS(vllm::Qwen4ExpMoeBlockWeights(s.w, p),
                         doctest::Contains("disagree on residency"), std::exception);
  }
  SUBCASE("a router the loader no longer leaves f32") {
    MoeSource s = BuildSource(/*keep_quant=*/false);
    s.w.router = Bf16Tensor(s.router, {kE, kH}, /*nk=*/true);
    CHECK_THROWS_WITH_AS(vllm::Qwen4ExpMoeBlockWeights(s.w, p),
                         doctest::Contains("must be f32"), std::exception);
  }
  SUBCASE("a tower whose shape does not match the config") {
    MoeSource s = BuildSource(/*keep_quant=*/false);
    Qwen4ExpParams q = p;
    q.moe_intermediate_size = kI * 2;
    CHECK_THROWS_WITH_AS(vllm::Qwen4ExpMoeBlockWeights(s.w, q),
                         doctest::Contains("ffn_gate_exps axis 1 must be"), std::exception);
  }
  SUBCASE("a stacked dtype that is neither arm") {
    // THE FOURTH REFUSAL THE HEADER CONTRACTS, which nothing exercised before.
    // f32 towers of the RIGHT shape: `RequireShape` passes, `IsStackedKeepQuant`
    // is false because f32 is not block-quant, and the bf16 branch refuses by
    // name instead of borrowing f32 bytes into a field the seam will read as
    // bf16 — which would be a silent 2x misread of every expert weight.
    //
    // It is also the refusal that catches
    // [#2275](https://github.com/mudler/vllm.cpp/issues/2275) at this seam:
    // `LoadStackedExperts` today falls through to `ExpandBf16` for `kKeepF16`
    // and `kNvfp4Fp4`, so the adapter never sees a third dtype from the loader.
    // When that fall-through is replaced by the residency refusal the issue
    // asks for, THIS case is what says the adapter refuses rather than
    // reinterprets whatever arrives.
    MoeSource s = BuildSource(/*keep_quant=*/false);
    s.w.gate_exps = F32Tensor(s.gate, {kE, kI, kH});
    s.w.up_exps = F32Tensor(s.up, {kE, kI, kH});
    s.w.down_exps = F32Tensor(s.down, {kE, kH, kI});
    CHECK_THROWS_WITH_AS(vllm::Qwen4ExpMoeBlockWeights(s.w, p),
                         doctest::Contains("the seam has no arm for it"),
                         std::exception);
  }
  SUBCASE("a hidden state that is not bf16") {
    MoeSource s = BuildSource(/*keep_quant=*/false);
    MoeBlockWeights mw = vllm::Qwen4ExpMoeBlockWeights(s.w, p);
    Queue q = CpuQ();
    std::vector<float> x(static_cast<size_t>(kT) * kH, 0.0F);
    Tensor dh = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {kT, kH});
    CHECK_THROWS_WITH_AS(vllm::RunQwen4ExpMoeBlock(q, mw, p, dh, kT),
                         doctest::Contains("must be bf16"), std::exception);
  }
}

// The one GPU fact this adapter turns on, asserted where a CPU host can see it:
// the seam consumes the router and the shared gate against a BF16 activation,
// and `MatmulBTKernelCuda` accepts only (bf16,bf16) or (f32,f32)
// (cuda_matmul.cu:397-403). Passing `LoadMoe`'s f32 tensors through unconverted
// runs on CPU and throws on every CUDA device — so the conversion is gated on
// the dtype, which is what a CPU-only suite CAN see.
TEST_CASE("Qwen4Exp MoE: the router and shared gate reach the seam as bf16") {
  MoeSource s = BuildSource(/*keep_quant=*/false);
  const Qwen4ExpParams p = Params();
  REQUIRE(s.w.router.dtype == DType::kF32);
  REQUIRE(s.w.shared_gate.dtype == DType::kF32);
  MoeBlockWeights mw = vllm::Qwen4ExpMoeBlockWeights(s.w, p);
  CHECK(mw.router_gate.dtype == DType::kBF16);
  CHECK(mw.router_gate.rank == 2);
  CHECK(mw.router_gate.shape[0] == kE);
  CHECK(mw.router_gate.shape[1] == kH);
  CHECK(mw.router_gate.nk);
  CHECK(mw.shared_gate.dtype == DType::kBF16);
  CHECK(mw.shared_gate.rank == 2);
  CHECK(mw.shared_gate.shape[0] == kH);
  CHECK(mw.shared_gate.shape[1] == 1);
  CHECK_FALSE(mw.shared_gate.nk);

  // The values survive the conversion, EXACTLY and not to a tolerance: every source value is an integer multiple of 2^-4
  // below 1, so bf16 holds it without loss. (`doctest::Approx(...).epsilon(0)`
  // cannot express this — its comparison is a strict `<` against
  // `epsilon * (scale + max|operand|)`, which is 0, so identical values fail.)
  const auto* rg = reinterpret_cast<const uint16_t*>(mw.router_gate.bytes.data());
  size_t exact = 0;
  for (size_t i = 0; i < s.router.size(); ++i)
    if (vt::BF16ToF32(rg[i]) == static_cast<float>(s.router[i]) * kWU) ++exact;
  CHECK(exact == s.router.size());
}

// ─── MOEDIV (#2552): the SELECTED-EXPERT-ID tap ──────────────────────────────
//
// `VT_Q4EXP_LAYER_FP` taps values, and a value tap cannot separate an
// expert-selection FLIP from re-association inside the expert GEMM: a discrete
// selection has BIMODAL error and not a tolerance. `VT_MOE_SEL_FP` prints the
// selection instead. This case is what makes the printed line trustworthy —
// it gates the ids against `MoeReference`, this suite's independent
// double-precision reimplementation of the lane-pinned oracle, and it gates
// `ulps` against the reference's own bf16 logit BITS.
//
// THE CASE NAME CONTAINS NO COMMA. `-tc` splits its filter on commas, so a
// comma in a case name silently selects nothing and the run reports
// `assertions: 0 ... SUCCESS!` at rc 0 — a skip wearing a pass. The ctest entry
// `test_qwen4_exp_moe_sel_fp` re-runs this binary with the switch set and
// filters on this exact string.
//
// IT ASSERTS ITS OWN CAPTURE. A sentinel is written into the redirected
// descriptor and read back, so a capture that silently caught nothing fails
// here rather than reading as "the tap printed nothing", which is the polarity
// the OFF arm asserts and the one a broken instrument would fake.
namespace {

#ifdef VLLM_MOE_SEL_FP_CAPTURE
// Everything written to stderr while `body` runs, as a string.
template <typename F>
std::string CaptureStderr(F&& body) {
  std::fflush(stderr);
  int saved = ::dup(2);
  char path[] = "/tmp/vllm_moesel_XXXXXX";
  int fd = ::mkstemp(path);
  REQUIRE(saved >= 0);
  REQUIRE(fd >= 0);
  ::dup2(fd, 2);
  body();
  std::fflush(stderr);
  ::dup2(saved, 2);
  ::close(saved);
  ::lseek(fd, 0, SEEK_SET);
  std::string out;
  char buf[4096];
  ssize_t n = 0;
  while ((n = ::read(fd, buf, sizeof(buf))) > 0) out.append(buf, static_cast<size_t>(n));
  ::close(fd);
  ::unlink(path);
  return out;
}

std::vector<std::string> LinesWithPrefix(const std::string& s, const std::string& pfx) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    const size_t e = s.find('\n', i);
    const std::string line = s.substr(i, e == std::string::npos ? std::string::npos : e - i);
    if (line.rfind(pfx, 0) == 0) out.push_back(line);
    if (e == std::string::npos) break;
    i = e + 1;
  }
  return out;
}

// `key=` of a `moesel` line, as a string.
std::string Field(const std::string& line, const std::string& key) {
  const std::string k = " " + key + "=";
  const size_t i = line.find(k);
  if (i == std::string::npos) return std::string();
  const size_t b = i + k.size();
  const size_t e = line.find(' ', b);
  return line.substr(b, e == std::string::npos ? std::string::npos : e - b);
}

// bf16 under the sign-magnitude total order — the tap's own `Bf16Ordered`,
// re-derived here rather than shared, so a sign-handling defect in the tap has
// nothing on this side agreeing with it.
int32_t RefBf16Ordered(uint16_t b) {
  return (b & 0x8000U) != 0 ? -static_cast<int32_t>(b & 0x7fffU)
                            : static_cast<int32_t>(b & 0x7fffU);
}
#endif  // VLLM_MOE_SEL_FP_CAPTURE

}  // namespace

TEST_CASE("Qwen4Exp MoE: the selected-expert-id tap names the oracle's own set") {
#ifndef VLLM_MOE_SEL_FP_CAPTURE
  MESSAGE("no POSIX stderr capture on this platform; the tap is NOT gated here");
  return;
#else
  MoeSource s = BuildSource(/*keep_quant=*/false);
  const Qwen4ExpParams p = Params();
  MoeBlockWeights mw = vllm::Qwen4ExpMoeBlockWeights(s.w, p);
  const std::vector<int> xc = HiddenCodes();
  const RefOut ref = MoeReference(s, xc);
  AssertRoutingIsNonTrivial(ref);

  std::vector<float> got;
  const std::string cap = CaptureStderr([&] {
    std::fprintf(stderr, "moesel-capture-sentinel\n");
    got = RunBlock(mw, p, xc);
  });
  // The capture's OWN precondition. Without this line the OFF arm below is an
  // assertion of absence from a redirect that may never have happened.
  REQUIRE(cap.find("moesel-capture-sentinel") != std::string::npos);
  REQUIRE(got.size() == static_cast<size_t>(kT) * kH);

  const std::vector<std::string> lines = LinesWithPrefix(cap, "moesel call=");
  const char* budget = std::getenv("VT_MOE_SEL_FP");
  if (budget == nullptr || budget[0] == '\0' || std::string(budget) == "0") {
    // A DIAGNOSTIC MUST BE INERT ON THE DEFAULT PATH. The sentinel above proves
    // the capture ran, so this is a measurement and not a failed grep.
    MESSAGE("VT_MOE_SEL_FP unset: the tap must print nothing");
    CHECK(lines.empty());
    return;
  }

  // ON. One value line per token plus one digest line.
  REQUIRE(lines.size() == static_cast<size_t>(kT) + 1);
  const std::string& digest = lines.back();
  CHECK(Field(digest, "T") == std::to_string(kT));
  CHECK(Field(digest, "E") == std::to_string(kE));
  CHECK(Field(digest, "k") == std::to_string(kTopK));
  // `lines` is the PROCESS-WIDE running total, and this entry runs the whole
  // binary, so other cases have already spent some of it. The exact per-call
  // count is asserted above as `lines.size() == kT + 1` over the captured
  // window; this asserts the running total is at least what this call added,
  // which is the property the measuring job reads back.
  CHECK(std::stoll(Field(digest, "lines")) >= static_cast<long long>(kT));

  uint64_t want_sel = 1469598103934665603ULL;
  int64_t want_min_ulps = -1;
  int64_t want_min_tok = -1;
  for (int64_t t = 0; t < kT; ++t) {
    const std::string& ln = lines[static_cast<size_t>(t)];
    CAPTURE(ln);
    CHECK(Field(ln, "tok") == std::to_string(t));

    // (1) THE SET. Sorted, because two arms selecting the same experts in a
    // different order have made the same routing decision.
    std::vector<int> want = ref.selected[static_cast<size_t>(t)];
    std::sort(want.begin(), want.end());
    std::string want_ids;
    for (size_t j = 0; j < want.size(); ++j) {
      if (j != 0) want_ids += ",";
      want_ids += std::to_string(want[j]);
      const uint32_t v = static_cast<uint32_t>(want[j]);
      for (int b = 0; b < 4; ++b) {
        want_sel ^= static_cast<uint64_t>((v >> (8 * b)) & 0xffU);
        want_sel *= 1099511628211ULL;
      }
    }
    CHECK(Field(ln, "ids") == want_ids);

    // (2) THE BOUNDARY, from the reference's OWN bf16 logit bits.
    int lo_e = -1, hi_e = -1;
    for (int64_t e = 0; e < kE; ++e) {
      const bool picked = std::find(want.begin(), want.end(), static_cast<int>(e)) != want.end();
      const uint16_t b = ref.logit_bits[static_cast<size_t>(t * kE + e)];
      const int32_t o = RefBf16Ordered(b);
      if (picked) {
        if (lo_e < 0 || o < RefBf16Ordered(ref.logit_bits[static_cast<size_t>(t * kE + lo_e)]))
          lo_e = static_cast<int>(e);
      } else {
        if (hi_e < 0 || o > RefBf16Ordered(ref.logit_bits[static_cast<size_t>(t * kE + hi_e)]))
          hi_e = static_cast<int>(e);
      }
    }
    REQUIRE(lo_e >= 0);
    REQUIRE(hi_e >= 0);
    const int64_t want_ulps =
        static_cast<int64_t>(RefBf16Ordered(ref.logit_bits[static_cast<size_t>(t * kE + lo_e)])) -
        static_cast<int64_t>(RefBf16Ordered(ref.logit_bits[static_cast<size_t>(t * kE + hi_e)]));
    CHECK(Field(ln, "lo_e") == std::to_string(lo_e));
    CHECK(Field(ln, "hi_e") == std::to_string(hi_e));
    CHECK(Field(ln, "ulps") == std::to_string(want_ulps));
    // The margin has to be POSITIVE and the ulp count has to agree with it: a
    // selected logit below a rejected one is a routing defect, not a rounding.
    CHECK(want_ulps > 0);
    if (want_min_ulps < 0 || want_ulps < want_min_ulps) {
      want_min_ulps = want_ulps;
      want_min_tok = t;
    }
  }
  // (3) The digest hash IS the set-equality assertion two arms use. Recomputed
  // here from the oracle's sets, so a tap that hashed the SELECTION ORDER (and
  // would therefore report a false flip between two arms that agree) reds.
  char want_hex[32];
  std::snprintf(want_hex, sizeof(want_hex), "%016llx",
                static_cast<unsigned long long>(want_sel));
  CHECK(Field(digest, "sel") == std::string(want_hex));
  CHECK(Field(digest, "minulps") == std::to_string(want_min_ulps));
  CHECK(Field(digest, "mintok") == std::to_string(want_min_tok));

  // (4) The four decomposition axes are POPULATED and finite. `x` is the block
  // input and is checkable in closed form; the other three are only asserted
  // non-zero, because their value is the measurement and not the contract.
  double want_x = 0.0;
  for (size_t i = 0; i < xc.size(); ++i)
    want_x += std::fabs(static_cast<double>(
        vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(xc[i]) * kAU))));
  CHECK(std::stod(Field(digest, "x")) == doctest::Approx(want_x).epsilon(1e-9));
  CHECK(std::stod(Field(digest, "logit")) > 0.0);
  CHECK(std::stod(Field(digest, "exp")) > 0.0);
  CHECK(std::stod(Field(digest, "shr")) > 0.0);
#endif
}
