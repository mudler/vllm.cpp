// vllm.cpp original. Portable contract for the single-warp MoE router top-k
// (spec: .agents/specs/moe-router-topk-single-warp.md, issue #378).
//
// WHAT THIS TEST IS. The claim the row rests on is a REDUCTION-ORDER claim:
// that a 32-lane, register-resident router with the lane map
// `expert = lane + 32*slot` performs the bit-identical sequence of float
// additions and fmaxf calls as the incumbent 256-thread block kernel
// (src/vt/cuda/cuda_moe.cu:61-201). That claim is pure IEEE-754 single
// precision arithmetic, so it is checkable on a host with no GPU and no nvcc,
// which is exactly what this file does. Both models below are plain float, and
// every operation is an add or an fmaxf -- no multiply, hence no FMA
// contraction, hence the host evaluates the association as written.
//
// WHAT THIS TEST IS NOT. It does not prove the CUDA kernel compiles, launches,
// or that the device agrees with the host. The device-side proof is the
// parallel-vs-serial byte-exactness case in test_ops_moe_grouped.cpp, which
// needs a GPU. This file proves the ORDER is right; that one proves the KERNEL
// is right. Neither substitutes for the other.
//
// The warp model calls the SHIPPED map and the SHIPPED reduction trees from
// src/vt/cuda/moe_router_warp.h, so a defect introduced there fails here. The
// block model is a literal transcription of the incumbent kernel; if the
// incumbent ever changes, this test goes red, which is the point.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "moe_router_warp_env.h"
#include "vt/cuda/moe_router_warp.h"
#include "vt/dtype.h"

using vt::cuda::kMoeRouterWarpWidth;
using vt::cuda::MoeRouterWarpContiguousExpert;
using vt::cuda::MoeRouterWarpExpert;
using vt::cuda::MoeRouterWarpFlagIsOn;
using vt::cuda::MoeRouterWarpTreeMax;
using vt::cuda::MoeRouterWarpTreeSum;
using vt::cuda::MoeRouterWarpValuesPerThread;

namespace {

constexpr int kBlock = 256;  // cuda_moe.cu:20
constexpr int kWarp = 32;

struct RouterOut {
  float mx = 0.0f;
  float sum = 0.0f;
  std::vector<float> weights;
  std::vector<int32_t> indices;
};

// THE TWO COMPARATORS ARE TRANSCRIBED SEPARATELY, ONE PER KERNEL.
//
// Higher value wins; on an exact tie the lower expert index wins; -1 means
// "this lane holds nothing". The block kernel and the warp kernel spell that
// out independently, and an EARLIER revision of this file had both models call
// one shared helper — which makes a comparator divergence between the two
// KERNELS invisible by construction: the models would agree because they run
// the same code, not because the kernels agree. They are character-identical
// today; that is a fact to PIN, not to build the test on.
//
// Block kernel, cuda_moe.cu:169 (warp-shuffle argmax) and :186 (thread 0's scan
// over the eight warp leaders) — identical to each other there.
bool BlockArgmaxTakes(float other_v, int other_i, float cur_v, int cur_i) {
  return other_v > cur_v ||
         (other_v == cur_v && other_i >= 0 && (cur_i < 0 || other_i < cur_i));
}

// Warp kernel, cuda_moe.cu:302. Transcribed separately ON PURPOSE; the test case
// "the two kernels' argmax comparators agree ..." below is what holds them equal.
bool WarpArgmaxTakes(float other_v, int other_i, float cur_v, int cur_i) {
  return other_v > cur_v ||
         (other_v == cur_v && other_i >= 0 && (cur_i < 0 || other_i < cur_i));
}

// ---------------------------------------------------------------------------
// Model A: the incumbent block kernel, MoeRouterTopKKernel<Tin,false>, one
// block of 256 threads per token. Transcribed statement for statement from
// src/vt/cuda/cuda_moe.cu:61-201. `e <= kBlock` is the regime the warp kernel
// is dispatched in, and the strided seed loops then give each thread at most
// one expert.
RouterOut BlockModel(const std::vector<float>& logits, int e, int k, bool renormalize) {
  RouterOut out;
  out.weights.assign(static_cast<size_t>(k), 0.0f);
  out.indices.assign(static_cast<size_t>(k), 0);
  std::vector<float> sp(static_cast<size_t>(e), 0.0f);
  std::vector<float> red(kBlock, 0.0f);

  // cuda_moe.cu:70-79 -- max, seeded from -INFINITY (which ERASES NaN).
  for (int t = 0; t < kBlock; ++t) {
    float m = -INFINITY;
    for (int j = t; j < e; j += kBlock) m = fmaxf(m, logits[static_cast<size_t>(j)]);
    red[static_cast<size_t>(t)] = m;
  }
  for (int s = kBlock / 2; s > 0; s /= 2) {
    for (int t = 0; t < s; ++t) {
      red[static_cast<size_t>(t)] =
          fmaxf(red[static_cast<size_t>(t)], red[static_cast<size_t>(t + s)]);
    }
  }
  out.mx = red[0];

  // cuda_moe.cu:82-94 -- exp(logit - max) into shared, block-summed. This is
  // the SECOND full read of the logit row.
  for (int t = 0; t < kBlock; ++t) {
    float acc = 0.0f;
    for (int j = t; j < e; j += kBlock) {
      const float ex = expf(logits[static_cast<size_t>(j)] - out.mx);
      sp[static_cast<size_t>(j)] = ex;
      acc += ex;
    }
    red[static_cast<size_t>(t)] = acc;
  }
  for (int s = kBlock / 2; s > 0; s /= 2) {
    for (int t = 0; t < s; ++t) red[static_cast<size_t>(t)] += red[static_cast<size_t>(t + s)];
  }
  out.sum = red[0];

  // cuda_moe.cu:97-103 -- a DIVIDE (not a reciprocal multiply), the sum>0
  // guard, then the isfinite clamp AFTER normalize.
  for (int j = 0; j < e; ++j) {
    float pj = out.sum > 0.0f ? sp[static_cast<size_t>(j)] / out.sum : 0.0f;
    if (!std::isfinite(pj)) pj = 0.0f;
    sp[static_cast<size_t>(j)] = pj;
  }

  // cuda_moe.cu:137-199 -- parallel greedy top-k: per-thread local argmax, a
  // warp-shuffle argmax, then thread 0's serial scan over the 8 warp leaders.
  constexpr int kWarps = kBlock / kWarp;
  std::vector<float> lv(kBlock);
  std::vector<int> li(kBlock);
  float leader_v[kWarps];
  int leader_i[kWarps];
  float denom = 0.0f;
  for (int j = 0; j < k; ++j) {
    for (int t = 0; t < kBlock; ++t) {
      float v = -INFINITY;
      int i = -1;
      for (int idx = t; idx < e; idx += kBlock) {
        if (sp[static_cast<size_t>(idx)] > v) {
          v = sp[static_cast<size_t>(idx)];
          i = idx;
        }
      }
      lv[static_cast<size_t>(t)] = v;
      li[static_cast<size_t>(t)] = i;
    }
    for (int w = 0; w < kWarps; ++w) {
      const int base = w * kWarp;
      for (int off = kWarp / 2; off > 0; off >>= 1) {
        float nv[kWarp];
        int ni[kWarp];
        for (int lane = 0; lane < kWarp; ++lane) {
          const int src = lane + off < kWarp ? lane + off : lane;
          float cv = lv[static_cast<size_t>(base + lane)];
          int ci = li[static_cast<size_t>(base + lane)];
          const float ov = lv[static_cast<size_t>(base + src)];
          const int oi = li[static_cast<size_t>(base + src)];
          if (BlockArgmaxTakes(ov, oi, cv, ci)) {
            cv = ov;
            ci = oi;
          }
          nv[lane] = cv;
          ni[lane] = ci;
        }
        for (int lane = 0; lane < kWarp; ++lane) {
          lv[static_cast<size_t>(base + lane)] = nv[lane];
          li[static_cast<size_t>(base + lane)] = ni[lane];
        }
      }
      leader_v[w] = lv[static_cast<size_t>(base)];
      leader_i[w] = li[static_cast<size_t>(base)];
    }
    float best_v = leader_v[0];
    int best = leader_i[0];
    for (int w = 1; w < kWarps; ++w) {
      if (BlockArgmaxTakes(leader_v[w], leader_i[w], best_v, best)) {
        best_v = leader_v[w];
        best = leader_i[w];
      }
    }
    if (best >= 0) sp[static_cast<size_t>(best)] = -INFINITY;
    out.weights[static_cast<size_t>(j)] = best_v;
    out.indices[static_cast<size_t>(j)] = static_cast<int32_t>(best);
    denom += best_v;
  }
  if (renormalize) {
    if (!(denom > 0.0f)) denom = 1.0f;
    for (int j = 0; j < k; ++j) out.weights[static_cast<size_t>(j)] /= denom;
  }
  return out;
}

// __shfl_down_sync(0xffffffffu, v, off) semantics: lane `l` receives lane
// `l+off`, or its OWN value when `l+off >= 32`. Only lane 0 is ever read, and
// after the first level lane 0's dependency cone stays inside lanes < 16, so
// the self-returning lanes never enter it -- exactly as the block tree leaves
// red[t >= s] stale and reads only red[0].
template <typename Op>
void ShflDownReduce(float (&v)[kWarp], Op op) {
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    float n[kWarp];
    for (int lane = 0; lane < kWarp; ++lane) {
      n[lane] = op(v[lane], v[lane + off < kWarp ? lane + off : lane]);
    }
    for (int lane = 0; lane < kWarp; ++lane) v[lane] = n[lane];
  }
}

// ---------------------------------------------------------------------------
// Model B: the candidate warp kernel, one warp per token, the whole row in
// `p[lane][slot]` registers. Uses the SHIPPED reduction trees; `map` supplies
// the lane->expert map so the test can also instantiate the WRONG one.
template <int VPT, typename Map>
RouterOut WarpModel(const std::vector<float>& logits, int k, bool renormalize, Map map) {
  RouterOut out;
  out.weights.assign(static_cast<size_t>(k), 0.0f);
  out.indices.assign(static_cast<size_t>(k), 0);
  const int e = kWarp * VPT;

  float p[kWarp][VPT];
  for (int lane = 0; lane < kWarp; ++lane) {
    for (int q = 0; q < VPT; ++q) {
      p[lane][q] = logits[static_cast<size_t>(map(lane, q))];
    }
  }

  float m[kWarp];
  for (int lane = 0; lane < kWarp; ++lane) m[lane] = MoeRouterWarpTreeMax<VPT>(p[lane]);
  ShflDownReduce(m, [](float a, float b) { return fmaxf(a, b); });
  out.mx = m[0];

  float s[kWarp];
  for (int lane = 0; lane < kWarp; ++lane) {
    for (int q = 0; q < VPT; ++q) p[lane][q] = expf(p[lane][q] - out.mx);
    s[lane] = MoeRouterWarpTreeSum<VPT>(p[lane]);
  }
  ShflDownReduce(s, [](float a, float b) { return a + b; });
  out.sum = s[0];

  for (int lane = 0; lane < kWarp; ++lane) {
    for (int q = 0; q < VPT; ++q) {
      float pj = out.sum > 0.0f ? p[lane][q] / out.sum : 0.0f;
      if (!std::isfinite(pj)) pj = 0.0f;
      p[lane][q] = pj;
    }
  }

  float denom = 0.0f;
  for (int j = 0; j < k; ++j) {
    float lv[kWarp];
    int li[kWarp];
    for (int lane = 0; lane < kWarp; ++lane) {
      float v = -INFINITY;
      int i = -1;
      for (int q = 0; q < VPT; ++q) {  // ascending expert index within the lane
        if (p[lane][q] > v) {
          v = p[lane][q];
          i = map(lane, q);
        }
      }
      lv[lane] = v;
      li[lane] = i;
    }
    for (int off = kWarp / 2; off > 0; off >>= 1) {
      float nv[kWarp];
      int ni[kWarp];
      for (int lane = 0; lane < kWarp; ++lane) {
        const int src = lane + off < kWarp ? lane + off : lane;
        float cv = lv[lane];
        int ci = li[lane];
        if (WarpArgmaxTakes(lv[src], li[src], cv, ci)) {
          cv = lv[src];
          ci = li[src];
        }
        nv[lane] = cv;
        ni[lane] = ci;
      }
      for (int lane = 0; lane < kWarp; ++lane) {
        lv[lane] = nv[lane];
        li[lane] = ni[lane];
      }
    }
    const float best_v = lv[0];
    const int best = li[0];
    // Mask the winner in the owning lane's own register. The kernel uses the
    // closed form for the shipped map (lane == best & 31, slot == best >> 5);
    // the model searches so it stays correct for the mutation map too.
    if (best >= 0) {
      for (int lane = 0; lane < kWarp; ++lane) {
        for (int q = 0; q < VPT; ++q) {
          if (map(lane, q) == best) p[lane][q] = -INFINITY;
        }
      }
    }
    out.weights[static_cast<size_t>(j)] = best_v;
    out.indices[static_cast<size_t>(j)] = static_cast<int32_t>(best);
    denom += best_v;
  }
  if (renormalize) {
    if (!(denom > 0.0f)) denom = 1.0f;
    for (int j = 0; j < k; ++j) out.weights[static_cast<size_t>(j)] /= denom;
  }
  (void)e;
  return out;
}

auto StridedMap = [](int lane, int slot) { return MoeRouterWarpExpert(lane, slot); };

// Bitwise, not approximate. The whole row is a byte-exactness claim.
bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool BitEqual(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

// For an INTERNAL intermediate that can legitimately be NaN. The PAYLOAD of a
// NaN produced by a float addition is not a property of the algorithm:
// IEEE-754 leaves the payload of NaN+NaN unspecified, and a compiler may
// commute the operands of an add freely. Measured on this host (gcc, -O1):
// `red[0] = red[0] + red[1]` and `n[0] = v[0] + v[1]` over the identical
// operands 0x7fc00000 and 0xffc00000 -- same source-level operand order --
// produce 0x7fc00000 and 0xffc00000 respectively, because one loop got
// vectorized and the other did not.
//
// The SECOND exemption is the sign of a zero `mx`, and it is a HOST CODEGEN
// artifact, not a property of the algorithm. The algorithm has no freedom here:
// fmaxf(x, -INFINITY) == x BITWISE for every non-NaN x (all 2^32 of them,
// checked exhaustively during review), and past the seed both trees apply fmaxf
// to identical pairs in identical order — so whatever a conforming fmaxf does
// with the (+0.0f, -0.0f) tie, it does the same thing in both models. What
// actually differs is gcc VECTORIZING one transcription and not the other, and
// the vector max instruction does not implement fmaxf's zero handling; measured,
// this test's `signed-zeros` row passes at -O1 and fails at -O0 and -O2 on
// identical source.
//
// Neither exemption is a tolerance: one is a bit IEEE-754 leaves unspecified,
// the other is the host compiler's, both are proven below to be unobservable in
// the OUTPUT, and the weights and indices stay a strict, total comparison that
// holds at -O0, -O1 and -O2.
bool SameIntermediate(float a, float b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  if (a == 0.0f && b == 0.0f) return true;  // +0.0f and -0.0f: an fmaxf tie
  return BitEqual(a, b);
}

// Round through bf16 exactly as the kernel's Load(const __nv_bfloat16*) does.
std::vector<float> RoundToBf16(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = vt::BF16ToF32(vt::F32ToBF16(v[i]));
  return out;
}

// One (E, k, renormalize, logits) case, run through both models and compared
// bit for bit on every observable.
void CheckCase(const char* label, int e, int k, bool renormalize,
               const std::vector<float>& logits) {
  const RouterOut block = BlockModel(logits, e, k, renormalize);
  RouterOut warp;
  switch (MoeRouterWarpValuesPerThread(e)) {
    case 1: warp = WarpModel<1>(logits, k, renormalize, StridedMap); break;
    case 2: warp = WarpModel<2>(logits, k, renormalize, StridedMap); break;
    case 4: warp = WarpModel<4>(logits, k, renormalize, StridedMap); break;
    case 8: warp = WarpModel<8>(logits, k, renormalize, StridedMap); break;
    default: FAIL("E is not a dispatched width: " << e); return;
  }
  INFO("case: " << std::string(label));
  CAPTURE(e);
  CAPTURE(k);
  CAPTURE(renormalize);
  // `mx` is never NaN by construction: the fmaxf(-INFINITY, .) seed erases
  // every NaN before the tree ever runs, so a NaN here would itself be a
  // failure. Its zero SIGN is exempt (SameIntermediate) and provably cannot
  // reach the output -- see the dedicated test case below.
  CHECK(SameIntermediate(block.mx, warp.mx));
  CHECK_FALSE(std::isnan(block.mx));
  // `sum` may legitimately be NaN (an all-NaN or Inf-minus-Inf row), and then
  // only its NaN-ness is an algorithm property; see SameIntermediate. This
  // weakens NOTHING observable: `sum > 0.0f` is false for EVERY NaN payload, so
  // a NaN sum sends every probability through the `: 0.0f` arm of
  // cuda_moe.cu:99 identically in both models -- which is exactly why the
  // WEIGHTS below stay a strict, unconditional memcmp and still pass on those
  // rows. The row's claim is about weights and indices; `sum` is checked only
  // because pinning the intermediate localizes a defect faster.
  CHECK(SameIntermediate(block.sum, warp.sum));
  // The actual contract. Strict, unconditional, never widened. Weights can
  // never be NaN (probs are isfinite-clamped at cuda_moe.cu:100 and the
  // exhausted-expert sentinel is exactly -INFINITY), so this comparison is
  // total.
  CHECK(BitEqual(block.weights, warp.weights));
  CHECK(block.indices == warp.indices);
}

std::vector<float> RandomRow(int e, uint32_t seed) {
  std::vector<float> v(static_cast<size_t>(e));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-4.0f, 4.0f);
  for (auto& x : v) x = d(rng);
  return v;
}

}  // namespace

TEST_CASE("router warp lane map is lane + 32*slot, and covers the row exactly once") {
  // The map must be a BIJECTION onto [0, E) for every dispatched VPT --
  // a duplicated expert would be silently selectable twice, and a missing one
  // could never be selected at all.
  for (int vpt : {1, 2, 4, 8}) {
    const int e = kMoeRouterWarpWidth * vpt;
    std::vector<int> hits(static_cast<size_t>(e), 0);
    for (int lane = 0; lane < kMoeRouterWarpWidth; ++lane) {
      for (int slot = 0; slot < vpt; ++slot) {
        const int expert = MoeRouterWarpExpert(lane, slot);
        REQUIRE(expert >= 0);
        REQUIRE(expert < e);
        ++hits[static_cast<size_t>(expert)];
      }
    }
    CAPTURE(vpt);
    for (int j = 0; j < e; ++j) CHECK(hits[static_cast<size_t>(j)] == 1);
  }
  // Slot q covers the 32 CONSECUTIVE experts [32q, 32q+32) -- the coalescing
  // property, and the reason this is not vLLM's contiguous partition.
  static_assert(MoeRouterWarpExpert(0, 0) == 0);
  static_assert(MoeRouterWarpExpert(31, 0) == 31);
  static_assert(MoeRouterWarpExpert(0, 1) == 32);
  static_assert(MoeRouterWarpExpert(31, 7) == 255);
  static_assert(MoeRouterWarpContiguousExpert(1, 0, 8) == 8);  // vLLM's map, for contrast
}

// The two intermediate exemptions in SameIntermediate are claims about the
// OUTPUT, so they are asserted rather than merely asserted-in-a-comment. If
// either of these ever stops holding, the exemption is invalid and the
// intermediate comparisons must go back to a strict memcmp -- they must never
// be widened further to accommodate a failure.
TEST_CASE("the intermediate exemptions cannot reach the output") {
  // (1) SIGN OF A ZERO `mx`. mx is consumed only as `expf(logit - mx)`
  // (cuda_moe.cu:84). Subtracting +0.0f and -0.0f differ for exactly one input,
  // -0.0f, and expf maps both zeros to the same 1.0f -- so sp[] is bit-identical
  // whichever zero the fmaxf tie returned.
  for (float x : {-3.5f, -1.0f, -0.0f, 0.0f, 1.0f, 7.25f, INFINITY, -INFINITY}) {
    const float minus_pos = x - 0.0f;
    const float minus_neg = x - (-0.0f);
    CAPTURE(x);
    if (!BitEqual(minus_pos, minus_neg)) {
      // The one permitted disagreement, and only for x == -0.0f.
      CHECK(BitEqual(x, -0.0f));
      CHECK(BitEqual(minus_pos, -0.0f));
      CHECK(BitEqual(minus_neg, 0.0f));
    }
    CHECK(BitEqual(expf(minus_pos), expf(minus_neg)));
  }
  CHECK(BitEqual(expf(-0.0f), 1.0f));
  CHECK(BitEqual(expf(0.0f), 1.0f));

  // (2) NaN PAYLOAD of `sum`. sum is consumed only through the predicate
  // `sum > 0.0f` (cuda_moe.cu:99), which is false for EVERY NaN, so every
  // probability takes the `: 0.0f` arm regardless of payload.
  for (uint32_t payload : {0x7fc00000u, 0xffc00000u, 0x7fc0deadu}) {
    float nan_value;
    std::memcpy(&nan_value, &payload, sizeof(float));
    REQUIRE(std::isnan(nan_value));
    CAPTURE(payload);
    CHECK_FALSE(nan_value > 0.0f);
  }
}

TEST_CASE("only the derived widths dispatch to the warp kernel") {
  CHECK(MoeRouterWarpValuesPerThread(32) == 1);
  CHECK(MoeRouterWarpValuesPerThread(64) == 2);
  CHECK(MoeRouterWarpValuesPerThread(128) == 4);
  CHECK(MoeRouterWarpValuesPerThread(256) == 8);
  // Everything else falls through to the unchanged block kernel. 512 and 1024
  // are excluded DELIBERATELY: above the incumbent's 256-wide block the seed
  // loop accumulates several experts per thread, an association the spec does
  // not derive.
  for (int64_t e : {int64_t{0}, int64_t{1}, int64_t{8}, int64_t{16}, int64_t{31}, int64_t{33},
                    int64_t{48}, int64_t{96}, int64_t{160}, int64_t{192}, int64_t{255},
                    int64_t{257}, int64_t{384}, int64_t{512}, int64_t{1024}}) {
    CAPTURE(e);
    CHECK(MoeRouterWarpValuesPerThread(e) == 0);
  }
}

TEST_CASE("VT_MOE_ROUTER_WARP is on by default and '0' rolls back") {
  CHECK(MoeRouterWarpFlagIsOn(nullptr));
  CHECK(MoeRouterWarpFlagIsOn(""));
  CHECK(MoeRouterWarpFlagIsOn("1"));
  CHECK(MoeRouterWarpFlagIsOn("on"));
  CHECK_FALSE(MoeRouterWarpFlagIsOn("0"));
}

// The comparators are the ONE thing the two models are not allowed to differ
// on, because the row's claim is that the two KERNELS pick the same expert. They
// are transcribed separately (see BlockArgmaxTakes / WarpArgmaxTakes) so that a
// divergence between cuda_moe.cu:169/:186 and :302 shows up HERE instead of
// being absorbed by a shared helper. This case is what pins them equal, over the
// values the argmax can actually carry: -INFINITY (the seed and the mask value),
// signed zeros, a NaN (which the isfinite clamp forbids downstream but which a
// defect could reintroduce), and the -1 "this lane holds nothing" sentinel.
TEST_CASE("the two kernels' argmax comparators agree on every (value, index) pair") {
  const float values[] = {-INFINITY, -1.0f, -0.0f, 0.0f, 1.0f, INFINITY, std::nanf("")};
  const int idx[] = {-1, 0, 1, 7};
  for (float ov : values) {
    for (int oi : idx) {
      for (float cv : values) {
        for (int ci : idx) {
          CAPTURE(ov);
          CAPTURE(oi);
          CAPTURE(cv);
          CAPTURE(ci);
          CHECK(BlockArgmaxTakes(ov, oi, cv, ci) == WarpArgmaxTakes(ov, oi, cv, ci));
        }
      }
    }
  }
}

// THE SELECTOR PIN, executed. vt::cuda::MoeRouterWarpEnabled() (cuda_moe.cu) is
// a FRESH getenv per launch, so the AMBIENT environment decides which kernel
// vt::MoeRouterTopK runs. The on-device parallel-vs-serial gate in
// tests/vt/test_ops_moe_grouped.cpp used to neither set nor assert it, so an
// exported `VT_MOE_ROUTER_WARP=0` — exactly what spec §9 gates 6/7 tell the
// operator to export for the A/B — degenerated that sweep to block-vs-serial and
// reported the identical counts. It now runs its sweep TWICE under
// vt_test::ScopedMoeRouterWarp, and asserts the pinned state each time.
//
// That pin is shared code, not a copy, so this host case IS its contract test:
// the device file cannot be compiled without nvcc, this one runs anywhere. If
// the pin ever stops overriding the ambient value, this case goes red on a box
// whose environment disagrees with the arm — which is the whole failure mode.
TEST_CASE("pinning VT_MOE_ROUTER_WARP overrides the ambient value, then restores it") {
  using vt_test::ScopedMoeRouterWarp;
  const char* ambient = std::getenv(ScopedMoeRouterWarp::kName);
  const bool ambient_had = ambient != nullptr;
  const std::string ambient_saved = ambient_had ? std::string(ambient) : std::string();

  // Whatever the shell said, the ON arm runs the WARP kernel ...
  {
    ScopedMoeRouterWarp pin("1");
    CHECK(ScopedMoeRouterWarp::EffectiveFlag());
  }
  // ... and the OFF arm runs the incumbent block kernel.
  {
    ScopedMoeRouterWarp pin("0");
    CHECK_FALSE(ScopedMoeRouterWarp::EffectiveFlag());
  }
  // Nesting resolves inner-first and unwinds to the outer pin, so an arm cannot
  // leak into the arm that follows it.
  {
    ScopedMoeRouterWarp outer("0");
    {
      ScopedMoeRouterWarp inner("1");
      CHECK(ScopedMoeRouterWarp::EffectiveFlag());
    }
    CHECK_FALSE(ScopedMoeRouterWarp::EffectiveFlag());
  }
  // Unset is restored as ABSENT, not as the empty string (which reads as ON).
  {
    ScopedMoeRouterWarp pin(nullptr);
    CHECK(std::getenv(ScopedMoeRouterWarp::kName) == nullptr);
    CHECK(ScopedMoeRouterWarp::EffectiveFlag());  // default ON
  }

  const char* after = std::getenv(ScopedMoeRouterWarp::kName);
  CHECK((after != nullptr) == ambient_had);
  if (ambient_had && after != nullptr) CHECK(ambient_saved == std::string(after));
}

TEST_CASE("warp reduction trees reproduce the block tree bit for bit (random rows)") {
  for (int e : {32, 64, 128, 256}) {
    for (int k : {1, 2, 8}) {
      for (bool renormalize : {false, true}) {
        for (uint32_t seed = 0; seed < 8; ++seed) {
          const std::vector<float> f32 = RandomRow(e, 1234u + seed + 97u * static_cast<uint32_t>(e));
          CheckCase("random-f32", e, k, renormalize, f32);
          // bf16 LOGIT arm. The weights are f32 either way, but bf16 rounding
          // manufactures the exact ties the tie-break has to resolve, and this
          // project has a recorded case where a bf16 store hid a defect that
          // only an f32 arm caught -- so both arms run, always.
          CheckCase("random-bf16", e, k, renormalize, RoundToBf16(f32));
        }
      }
    }
  }
}

TEST_CASE("warp reduction trees reproduce the block tree on adversarial rows") {
  for (int e : {32, 64, 128, 256}) {
    const size_t n = static_cast<size_t>(e);
    for (bool renormalize : {false, true}) {
      // Exact-tie storm: blocks of identical logits, so many experts tie at the
      // max and the lowest-index tie-break decides every slot.
      {
        std::vector<float> v(n);
        for (size_t j = 0; j < n; ++j) v[j] = static_cast<float>((j / 4) % 5);
        CheckCase("tie-storm", e, 8, renormalize, v);
      }
      // Every expert identical: the whole row is one tie.
      {
        std::vector<float> v(n, 0.5f);
        CheckCase("all-equal", e, 8, renormalize, v);
      }
      // k > E: the -1 sentinel path (indices -1, weights -INFINITY). This is a
      // KERNEL-level input only — vt::MoeRouterTopK rejects k > E outright
      // ("top_k must be in [1, num_experts]", src/vt/ops.cpp), and within
      // 1 <= k <= E the sentinel cannot fire because every prob is finite and
      // >= 0, so only exhausting all E experts leaves best == -1. The two
      // kernels' `best >= 0` guards still have to AGREE, and this is the only
      // place that can compare them: the on-device parallel-vs-serial case in
      // test_ops_moe_grouped.cpp cannot, since the op throws first (and its
      // serial oracle would write sp[-1]). Keep this arm; do not add a device one.
      {
        CheckCase("k-gt-E", e, e + 3, renormalize, RandomRow(e, 4242u));
      }
      // k == 1.
      {
        CheckCase("k-eq-1", e, 1, renormalize, RandomRow(e, 777u));
      }
      // All NaN. The -INFINITY max seed erases every NaN, so mx is -INFINITY,
      // the sum is NaN, the clamp zeroes every prob, and the tie-break must
      // then hand back 0,1,2,...,k-1.
      {
        std::vector<float> v(n, std::nanf(""));
        CheckCase("all-nan", e, 8, renormalize, v);
      }
      // All +Inf, all -Inf, and a mix with NaN.
      {
        CheckCase("all-pos-inf", e, 8, renormalize, std::vector<float>(n, INFINITY));
        CheckCase("all-neg-inf", e, 8, renormalize, std::vector<float>(n, -INFINITY));
        std::vector<float> v(n, 1.0f);
        v[0] = std::nanf("");
        v[1] = INFINITY;
        v[n - 1] = -INFINITY;
        CheckCase("nan-inf-mix", e, 8, renormalize, v);
      }
      // Signed zeros: -0.0f is the one float for which x + 0.0f flips the sign
      // bit, which is what makes dropping the block's zero pad leaves legal.
      {
        std::vector<float> v(n, -0.0f);
        for (size_t j = 0; j < n; j += 2) v[j] = 0.0f;
        CheckCase("signed-zeros", e, 8, renormalize, v);
      }
      // Huge magnitudes: the max subtraction is what keeps expf finite.
      {
        std::vector<float> v(n);
        for (size_t j = 0; j < n; ++j) v[j] = 80.0f + static_cast<float>(j);
        CheckCase("huge-magnitude", e, 8, renormalize, v);
      }
    }
  }
}

// THE DISCRIMINATION TEST. A parallel-vs-serial comparison that cannot tell the
// two lane maps apart is not testing the reduction order, and would report
// GREEN for a kernel that quietly reassociates the softmax sum -- which is the
// exact defect this row exists to avoid. So pin the oracle's discriminating
// power in the gate rather than in a one-off mutation transcript.
//
// The contiguous map (lane L owns experts [VPT*L, VPT*L + VPT)) is vLLM's OWN
// partition, topk_softmax_kernels.cu:344-346. It is a correct softmax; it is
// simply a DIFFERENT association, so it lands on a different last ulp.
TEST_CASE("the contiguous (vLLM) lane map is detectably different, so the oracle bites") {
  for (int vpt : {2, 4, 8}) {  // VPT == 1 has one slot per lane: no association to differ
    const int e = kMoeRouterWarpWidth * vpt;
    auto contiguous = [vpt](int lane, int slot) {
      return MoeRouterWarpContiguousExpert(lane, slot, vpt);
    };
    int differing_rows = 0;
    for (uint32_t seed = 0; seed < 16; ++seed) {
      const std::vector<float> v = RandomRow(e, 555u + seed);
      const RouterOut block = BlockModel(v, e, 8, true);
      RouterOut wrong;
      switch (vpt) {
        case 2: wrong = WarpModel<2>(v, 8, true, contiguous); break;
        case 4: wrong = WarpModel<4>(v, 8, true, contiguous); break;
        default: wrong = WarpModel<8>(v, 8, true, contiguous); break;
      }
      // WEIGHTS ALONE. This case exists to prove the WEIGHTS memcmp — the
      // comparison the row's claim actually rests on, and the one the on-device
      // parallel-vs-serial gate makes — has discriminating power. An earlier
      // revision wrote `!BitEqual(sum) || !BitEqual(weights)`, which pins only
      // "sum-or-weights": a contiguous map that moved the denominator but never
      // reached the output would have satisfied it, and the weights memcmp would
      // still have been unproven.
      if (!BitEqual(block.weights, wrong.weights)) ++differing_rows;
    }
    CAPTURE(vpt);
    CHECK(differing_rows > 0);
  }
}
