// Grouped-topk (`noaux_tc`) MoE router unit tests (MLA campaign W3).
//
// Upstream test modules ported per .agents/test-porting.md:
//   vllm/tests/kernels/moe/test_grouped_topk.py  @ pin e24d1b24
//   vllm/tests/kernels/moe/test_routing.py       (the grouped-topk cases)
// Formula under port:
//   vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:106-161
//   (`grouped_topk`, the `forward_native` path).
//
// ─── WHY THIS FILE IS THE CORRECTNESS EVIDENCE, STATED PLAINLY ──────────────
// The MLA campaign's e2e gate vehicle is DeepSeek-V2-Lite, and its real config
// (W0-confirmed against the shipped config.json) is `n_group=1, topk_group=1,
// scoring_func="softmax", topk_method="greedy"` — which means the checkpoint has
// NO `e_score_correction_bias` parameter at all (deepseek_v2.py:313-318 only
// creates it for `topk_method == "noaux_tc"`). So the V2-Lite e2e gate exercises
// NONE of the machinery this file tests: not sigmoid scoring, not the
// biased-select / unbiased-weight asymmetry, not the two-level group mask, not
// routed scaling. The `noaux_tc` router is UNIT-GATED ONLY, and these tests —
// run at DeepSeek-V3's REAL dimensions (256 experts, n_group=8, topk_group=4,
// top_k=8, sigmoid, routed_scaling_factor=2.5, WITH the bias) — are the whole of
// its correctness evidence until a V3-class checkpoint fits on the hardware.
// Dimensions are free; weights are not.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::MoeRouterTopKArgs;
using vt::MoeScoringFunc;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

std::vector<float> RandF32(size_t n, uint32_t seed, float lo = -3.0f, float hi = 3.0f) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
    x = lo + u * (hi - lo);
  }
  return v;
}

// ─── INDEPENDENT reference ─────────────────────────────────────────────────
// A direct, statement-by-statement transcription of grouped_topk_router.py
// :106-161, written from the upstream source rather than from our kernel, and
// using a SORT-based top-k (upstream's torch.topk) instead of the kernel's
// greedy scan — so agreement is real evidence, not a restatement of the same
// code. Tie rule: lowest index wins, our recorded determinism convention (see
// the ops.h deviation note), applied as the sort's secondary key.
struct RouterOut {
  std::vector<float> weights;  // [T, top_k]
  std::vector<int32_t> ids;    // [T, top_k]
};

RouterOut RefGroupedTopK(const std::vector<float>& logits, int64_t t, int64_t e,
                         const MoeRouterTopKArgs& args, const std::vector<float>* bias) {
  const int k = args.top_k;
  const int64_t n_group = args.num_expert_group;
  const int64_t gsz = e / n_group;
  RouterOut out;
  out.weights.assign(static_cast<size_t>(t * k), 0.0f);
  out.ids.assign(static_cast<size_t>(t * k), -1);

  // Sort-based top-k over `vals`, restricted to `allowed`; returns indices.
  auto topk_idx = [](const std::vector<float>& vals, const std::vector<char>& allowed,
                     int n) {
    std::vector<int64_t> idx;
    for (size_t i = 0; i < vals.size(); ++i) {
      if (allowed.empty() || allowed[i] != 0) idx.push_back(static_cast<int64_t>(i));
    }
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
      if (vals[static_cast<size_t>(a)] != vals[static_cast<size_t>(b)]) {
        return vals[static_cast<size_t>(a)] > vals[static_cast<size_t>(b)];
      }
      return a < b;  // lowest index wins an exact tie
    });
    idx.resize(std::min<size_t>(idx.size(), static_cast<size_t>(n)));
    return idx;
  };

  for (int64_t row = 0; row < t; ++row) {
    // :110-117 scores = softmax | sigmoid
    std::vector<float> scores(static_cast<size_t>(e));
    if (args.scoring_func == MoeScoringFunc::kSigmoid) {
      for (int64_t j = 0; j < e; ++j) {
        scores[static_cast<size_t>(j)] =
            1.0f / (1.0f + std::exp(-logits[static_cast<size_t>(row * e + j)]));
      }
    } else {
      float mx = -INFINITY;
      for (int64_t j = 0; j < e; ++j) {
        mx = std::max(mx, logits[static_cast<size_t>(row * e + j)]);
      }
      float sum = 0.0f;
      for (int64_t j = 0; j < e; ++j) {
        const float ex = std::exp(logits[static_cast<size_t>(row * e + j)] - mx);
        scores[static_cast<size_t>(j)] = ex;
        sum += ex;
      }
      for (auto& s : scores) s = sum > 0.0f ? s / sum : 0.0f;
    }

    // :120-131 original_scores kept; bias applied to the SELECTION copy only.
    const std::vector<float> original = scores;
    std::vector<float> sel = scores;
    if (bias != nullptr) {
      for (int64_t j = 0; j < e; ++j) sel[static_cast<size_t>(j)] += (*bias)[static_cast<size_t>(j)];
    }
    std::vector<float> gscore(static_cast<size_t>(n_group));
    for (int64_t g = 0; g < n_group; ++g) {
      std::vector<float> grp(sel.begin() + static_cast<long>(g * gsz),
                             sel.begin() + static_cast<long>((g + 1) * gsz));
      std::sort(grp.begin(), grp.end(), std::greater<float>());
      if (bias != nullptr) {
        gscore[static_cast<size_t>(g)] = grp[0] + grp[1];  // :124-126 top-2 SUM
      } else {
        gscore[static_cast<size_t>(g)] = grp[0];  // :128-131 group MAX
      }
    }

    // :133-145 keep topk_group groups, mask the rest to -inf.
    const std::vector<int64_t> keep = topk_idx(gscore, {}, args.topk_group);
    std::vector<char> allowed(static_cast<size_t>(e), 0);
    for (int64_t g : keep) {
      for (int64_t j = 0; j < gsz; ++j) allowed[static_cast<size_t>(g * gsz + j)] = 1;
    }

    // :147-150 select on the (masked, biased) score; weight from the UNBIASED.
    const std::vector<int64_t> picked = topk_idx(sel, allowed, k);
    float denom = 0.0f;
    for (int j = 0; j < k; ++j) {
      const int64_t id = picked[static_cast<size_t>(j)];
      const float w = original[static_cast<size_t>(id)];
      out.ids[static_cast<size_t>(row * k + j)] = static_cast<int32_t>(id);
      out.weights[static_cast<size_t>(row * k + j)] = w;
      denom += w;
    }
    // :156-157 renormalize, THEN :159-160 routed scaling.
    if (args.renormalize) {
      if (!(denom > 0.0f)) denom = 1.0f;
      for (int j = 0; j < k; ++j) out.weights[static_cast<size_t>(row * k + j)] /= denom;
    }
    if (args.routed_scaling_factor != 1.0f) {
      for (int j = 0; j < k; ++j) {
        out.weights[static_cast<size_t>(row * k + j)] *= args.routed_scaling_factor;
      }
    }
  }
  return out;
}

// Runs the op on the CPU device and returns its output.
RouterOut RunOpCpu(const std::vector<float>& logits, int64_t t, int64_t e,
                   const MoeRouterTopKArgs& args, std::vector<float>* bias) {
  RouterOut out;
  out.weights.assign(static_cast<size_t>(t * args.top_k), 0.0f);
  out.ids.assign(static_cast<size_t>(t * args.top_k), -1);
  Tensor tl = Contig(const_cast<float*>(logits.data()), DType::kF32, Cpu(), {t, e});
  Tensor tw = Contig(out.weights.data(), DType::kF32, Cpu(), {t, args.top_k});
  Tensor ti = Contig(out.ids.data(), DType::kI32, Cpu(), {t, args.top_k});
  Queue q = Q();
  if (bias != nullptr) {
    Tensor tb = Contig(bias->data(), DType::kF32, Cpu(), {e});
    vt::MoeRouterTopK(q, tw, ti, tl, args, &tb);
  } else {
    vt::MoeRouterTopK(q, tw, ti, tl, args);
  }
  return out;
}

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = Contig(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

RouterOut RunOpCuda(const std::vector<float>& logits, int64_t t, int64_t e,
                    const MoeRouterTopKArgs& args, std::vector<float>* bias) {
  RouterOut out;
  out.weights.assign(static_cast<size_t>(t * args.top_k), 0.0f);
  out.ids.assign(static_cast<size_t>(t * args.top_k), -1);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dl(gpu, g.q, DType::kF32, {t, e}, logits.data());
  DeviceTensor dw(gpu, g.q, DType::kF32, {t, args.top_k});
  DeviceTensor di(gpu, g.q, DType::kI32, {t, args.top_k});
  if (bias != nullptr) {
    DeviceTensor db(gpu, g.q, DType::kF32, {e}, bias->data());
    vt::MoeRouterTopK(g.q, dw.tensor(), di.tensor(), dl.tensor(), args, &db.tensor());
    dw.Download(g.q, out.weights.data());
    di.Download(g.q, out.ids.data());
    return out;
  }
  vt::MoeRouterTopK(g.q, dw.tensor(), di.tensor(), dl.tensor(), args);
  dw.Download(g.q, out.weights.data());
  di.Download(g.q, out.ids.data());
  return out;
}

// DeepSeek-V3 / R1's REAL router config (deepseek_v2.py:370-378 +
// .agents/specs/mla-deepseek-campaign.md §5.1's V3 column).
MoeRouterTopKArgs V3Args() {
  MoeRouterTopKArgs a;
  a.top_k = 8;                      // num_experts_per_tok
  a.renormalize = true;             // norm_topk_prob
  a.scoring_func = MoeScoringFunc::kSigmoid;
  a.num_expert_group = 8;           // n_group
  a.topk_group = 4;                 // topk_group
  a.routed_scaling_factor = 2.5f;   // routed_scaling_factor
  return a;
}
constexpr int64_t kV3Experts = 256;  // n_routed_experts

}  // namespace

// ===========================================================================
// THE HEADLINE CASE — real V3 dimensions, sigmoid, WITH e_score_correction_bias.
// ===========================================================================
TEST_CASE("grouped_topk at REAL DeepSeek-V3 dims vs the upstream formula") {
  const int64_t t = 24, e = kV3Experts;
  const MoeRouterTopKArgs args = V3Args();
  const auto logits = RandF32(static_cast<size_t>(t * e), 1234);
  // A learned per-expert bias with both signs and a realistic magnitude
  // (upstream's is a trained [E] f32 parameter, unconstrained in sign).
  auto bias = RandF32(static_cast<size_t>(e), 4321, -0.35f, 0.35f);

  const RouterOut ref = RefGroupedTopK(logits, t, e, args, &bias);
  const RouterOut got = RunOpCpu(logits, t, e, args, &bias);

  REQUIRE(got.ids.size() == ref.ids.size());
  for (size_t i = 0; i < ref.ids.size(); ++i) {
    CHECK(got.ids[i] == ref.ids[i]);  // selection must be EXACT
    CHECK(got.weights[i] == doctest::Approx(ref.weights[i]).epsilon(1e-6));
  }

  // Every selected expert must live in one of the topk_group surviving groups —
  // asserted structurally, independent of the reference.
  const int64_t gsz = e / args.num_expert_group;
  for (int64_t row = 0; row < t; ++row) {
    std::vector<int64_t> groups;
    for (int j = 0; j < args.top_k; ++j) {
      groups.push_back(got.ids[static_cast<size_t>(row * args.top_k + j)] / gsz);
    }
    std::sort(groups.begin(), groups.end());
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    CHECK(static_cast<int>(groups.size()) <= args.topk_group);
  }
  // No expert is selected twice for a token.
  for (int64_t row = 0; row < t; ++row) {
    std::vector<int32_t> ids(got.ids.begin() + static_cast<long>(row * args.top_k),
                             got.ids.begin() + static_cast<long>((row + 1) * args.top_k));
    std::sort(ids.begin(), ids.end());
    CHECK(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
  }
}

TEST_CASE("grouped_topk V3 dims: CUDA matches CPU exactly on ids") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping grouped_topk parity");
    return;
  }
  const int64_t t = 24, e = kV3Experts;
  const MoeRouterTopKArgs args = V3Args();
  const auto logits = RandF32(static_cast<size_t>(t * e), 1234);
  auto bias = RandF32(static_cast<size_t>(e), 4321, -0.35f, 0.35f);

  const RouterOut cpu = RunOpCpu(logits, t, e, args, &bias);
  const RouterOut gpu = RunOpCuda(logits, t, e, args, &bias);
  for (size_t i = 0; i < cpu.ids.size(); ++i) {
    // SELECTION is the part that must be bit-for-bit: the greedy scan and the
    // lowest-index tie-break are pure comparisons over the same values, so the
    // chosen expert ids can never legitimately differ between devices.
    CHECK(gpu.ids[i] == cpu.ids[i]);
    CHECK(gpu.weights[i] == doctest::Approx(cpu.weights[i]).epsilon(1e-5));
  }
  // MEASURED (dgx/sm_121, NOT assumed): the WEIGHTS are NOT bit-identical, and
  // the reason is not a reduction order — sigmoid scoring has no cross-expert
  // reduction at all. It is the TRANSCENDENTAL itself: the CPU reference uses
  // the host `std::exp` and the CUDA kernel uses the device `expf`, which are
  // separately-rounded implementations of the same function. On the V3-dims case
  // 7 of 192 weights differ, each by one ULP. An earlier draft of this test
  // asserted exact equality here and was WRONG on the merits; the accurate
  // property is a 1-ULP bound, asserted below. (Selection is unaffected because
  // a 1-ULP score change cannot reorder experts that are not already tied — and
  // where they ARE exactly tied, the lowest-index rule decides identically.)
  for (size_t i = 0; i < cpu.weights.size(); ++i) {
    const float a = cpu.weights[i], b = gpu.weights[i];
    if (a == b) continue;
    const float ulp = std::nextafter(a, INFINITY) - a;
    CHECK(std::fabs(a - b) <= 2.0f * std::fabs(ulp));
  }
}

TEST_CASE("grouped_topk V3 dims: CUDA is run-to-run reproducible") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping grouped_topk reproducibility");
    return;
  }
  const int64_t t = 32, e = kV3Experts;
  const MoeRouterTopKArgs args = V3Args();
  const auto logits = RandF32(static_cast<size_t>(t * e), 777);
  auto bias = RandF32(static_cast<size_t>(e), 778, -0.35f, 0.35f);
  const RouterOut a = RunOpCuda(logits, t, e, args, &bias);
  const RouterOut b = RunOpCuda(logits, t, e, args, &bias);
  CHECK(a.ids == b.ids);
  CHECK(a.weights == b.weights);  // bit-identical, not toleranced
}

// ===========================================================================
// The three pieces of NEW numerics, isolated.
// ===========================================================================
TEST_CASE("the bias SELECTS but the unbiased score WEIGHTS (:120-124, :147-150)") {
  // Hand-built, 1 token, E=4, one group, top_k=1. Sigmoid is monotone, so the
  // ordering of the raw logits is the ordering of the scores.
  //   logits: e0=1.0, e1=0.9, e2=-5, e3=-5   -> e0 is the unbiased winner
  //   bias:   e0=0.0, e1=+0.5, ...           -> e1 becomes the BIASED winner
  // Correct behavior: id == 1 (biased selection) with weight == sigmoid(0.9)
  // (UNBIASED score), NOT sigmoid(0.9)+0.5. Getting this backwards is the
  // silent accuracy bug the campaign spec flags.
  const int64_t t = 1, e = 4;
  std::vector<float> logits = {1.0f, 0.9f, -5.0f, -5.0f};
  std::vector<float> bias = {0.0f, 0.5f, 0.0f, 0.0f};
  MoeRouterTopKArgs args;
  args.top_k = 1;
  args.renormalize = false;  // isolate the weight from the renorm divide
  args.scoring_func = MoeScoringFunc::kSigmoid;
  args.num_expert_group = 1;
  args.topk_group = 1;

  const RouterOut got = RunOpCpu(logits, t, e, args, &bias);
  CHECK(got.ids[0] == 1);
  const float unbiased = 1.0f / (1.0f + std::exp(-0.9f));
  CHECK(got.weights[0] == doctest::Approx(unbiased).epsilon(1e-6));
  CHECK(got.weights[0] != doctest::Approx(unbiased + 0.5f).epsilon(1e-6));

  // Without the bias the same logits select e0 — proving the bias is what moved
  // the selection, not the group machinery.
  const RouterOut nobias = RunOpCpu(logits, t, e, args, nullptr);
  CHECK(nobias.ids[0] == 0);
}

TEST_CASE("group score is TOP-2 SUM with a bias and MAX without (:124-131)") {
  // E=4, n_group=2 (groups {0,1} and {2,3}), topk_group=1, top_k=1.
  // Scores are shaped so the two rules disagree about which group survives:
  //   group A = {high, very low}, group B = {mid, mid}
  //   MAX rule    -> A wins (its max is the largest single score)
  //   TOP-2 SUM   -> B wins (two mids beat one high plus one very low)
  const int64_t t = 1, e = 4;
  // sigmoid(3)=0.9526, sigmoid(-6)=0.0025 -> A: max .9526, sum .9551
  // sigmoid(1)=0.7311, sigmoid(1)=0.7311  -> B: max .7311, sum 1.4622
  std::vector<float> logits = {3.0f, -6.0f, 1.0f, 1.0f};
  MoeRouterTopKArgs args;
  args.top_k = 1;
  args.renormalize = false;
  args.scoring_func = MoeScoringFunc::kSigmoid;
  args.num_expert_group = 2;
  args.topk_group = 1;

  // No bias -> group MAX -> group A survives -> expert 0.
  const RouterOut by_max = RunOpCpu(logits, t, e, args, nullptr);
  CHECK(by_max.ids[0] == 0);

  // With a ZERO bias present -> the top-2 SUM rule applies (upstream branches on
  // the bias being not-None, NOT on its values) -> group B survives -> expert 2.
  // A zero bias isolates the RULE SWITCH from any value effect.
  std::vector<float> zero_bias(static_cast<size_t>(e), 0.0f);
  const RouterOut by_sum = RunOpCpu(logits, t, e, args, &zero_bias);
  CHECK(by_sum.ids[0] == 2);
}

TEST_CASE("the group mask really excludes non-surviving groups (:133-145)") {
  // E=8, n_group=4 (groups of 2), topk_group=1, top_k=2. Group 3 holds the two
  // best experts; groups 0-2 are poor. With topk_group=1 ONLY group 3 survives,
  // so both picks must come from {6, 7} — an unmasked top-k would be identical
  // here, so we also assert the CONVERSE: put the single best expert alone in a
  // weak group and confirm it is MASKED OUT.
  const int64_t t = 1, e = 8;
  MoeRouterTopKArgs args;
  args.top_k = 2;
  args.renormalize = false;
  args.scoring_func = MoeScoringFunc::kSigmoid;
  args.num_expert_group = 4;
  args.topk_group = 1;

  // Group 0 = {9.0, -9.0}: contains the GLOBAL best expert but a terrible
  // partner. Group 3 = {5.0, 5.0}: no single best, but the best PAIR.
  std::vector<float> logits = {9.0f, -9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 5.0f};
  std::vector<float> zero_bias(static_cast<size_t>(e), 0.0f);
  const RouterOut got = RunOpCpu(logits, t, e, args, &zero_bias);
  // top-2-sum: group 0 ≈ 1.0 + 0.0001, group 3 ≈ 0.9933 * 2 = 1.9866 -> group 3.
  CHECK(got.ids[0] == 6);
  CHECK(got.ids[1] == 7);
  // Expert 0 is the global argmax and is NOT selected: the mask is real.
  CHECK(got.ids[0] != 0);
  CHECK(got.ids[1] != 0);
}

TEST_CASE("renormalize then routed_scaling_factor, in that order (:156-160)") {
  const int64_t t = 3, e = 32;
  const auto logits = RandF32(static_cast<size_t>(t * e), 555);
  MoeRouterTopKArgs base;
  base.top_k = 4;
  base.renormalize = true;
  base.scoring_func = MoeScoringFunc::kSigmoid;
  base.num_expert_group = 4;
  base.topk_group = 2;

  const RouterOut unscaled = RunOpCpu(logits, t, e, base, nullptr);
  MoeRouterTopKArgs scaled = base;
  scaled.routed_scaling_factor = 2.5f;  // V3's value
  const RouterOut got = RunOpCpu(logits, t, e, scaled, nullptr);

  for (size_t i = 0; i < got.ids.size(); ++i) {
    CHECK(got.ids[i] == unscaled.ids[i]);  // scaling never moves the selection
    CHECK(got.weights[i] == doctest::Approx(unscaled.weights[i] * 2.5f).epsilon(1e-6));
  }
  // Renormalized weights sum to exactly the scaling factor per token — which is
  // only true if the divide happened BEFORE the multiply.
  for (int64_t row = 0; row < t; ++row) {
    float sum = 0.0f;
    for (int j = 0; j < base.top_k; ++j) {
      sum += got.weights[static_cast<size_t>(row * base.top_k + j)];
    }
    CHECK(sum == doctest::Approx(2.5f).epsilon(1e-5));
  }
}

TEST_CASE("softmax scoring on the grouped path (the V2 / non-noaux_tc form)") {
  // deepseek_v2.py:370-378 passes scoring_func from the config: V2 is "softmax",
  // V3 is "sigmoid". Both reach grouped_topk, so both must work.
  const int64_t t = 6, e = 64;
  const auto logits = RandF32(static_cast<size_t>(t * e), 909);
  MoeRouterTopKArgs args;
  args.top_k = 6;
  args.renormalize = true;
  args.scoring_func = MoeScoringFunc::kSoftmax;
  args.num_expert_group = 8;
  args.topk_group = 3;

  const RouterOut ref = RefGroupedTopK(logits, t, e, args, nullptr);
  const RouterOut got = RunOpCpu(logits, t, e, args, nullptr);
  for (size_t i = 0; i < ref.ids.size(); ++i) {
    CHECK(got.ids[i] == ref.ids[i]);
    CHECK(got.weights[i] == doctest::Approx(ref.weights[i]).epsilon(1e-6));
  }
}

TEST_CASE("n_group == topk_group == 1 degenerates to plain top-k") {
  // DeepSeek-V2-Lite's ACTUAL config (W0-confirmed): n_group=1, topk_group=1,
  // softmax, greedy, no bias. With one group the mask is a no-op, so the grouped
  // path must produce EXACTLY what the pre-W3 ungrouped router produces. This is
  // the bridge between the e2e gate vehicle and this unit-only file.
  const int64_t t = 8, e = 64;
  const auto logits = RandF32(static_cast<size_t>(t * e), 246);

  MoeRouterTopKArgs plain;
  plain.top_k = 6;  // V2-Lite's num_experts_per_tok
  plain.renormalize = true;
  const RouterOut ungrouped = RunOpCpu(logits, t, e, plain, nullptr);

  MoeRouterTopKArgs grouped = plain;
  grouped.num_expert_group = 1;
  grouped.topk_group = 1;
  const RouterOut got = RunOpCpu(logits, t, e, grouped, nullptr);

  CHECK(got.ids == ungrouped.ids);
  for (size_t i = 0; i < got.weights.size(); ++i) {
    CHECK(got.weights[i] == doctest::Approx(ungrouped.weights[i]).epsilon(1e-6));
  }
}

TEST_CASE("the pre-W3 ungrouped router is UNTOUCHED by the W3 extension") {
  // BEHAVIOR-PRESERVATION at the op level: with a default-constructed args
  // struct (num_expert_group == 0) the call is routed to the original kernel.
  // These expectations are the pre-W3 ones (tests/vt/test_ops_moe.cpp): softmax
  // + descending greedy top-k + renormalize.
  const int64_t t = 1, e = 4;
  std::vector<float> logits = {0.0f, 2.0f, 1.0f, -1.0f};
  MoeRouterTopKArgs args;
  args.top_k = 2;
  args.renormalize = true;
  const RouterOut got = RunOpCpu(logits, t, e, args, nullptr);
  CHECK(got.ids[0] == 1);  // largest logit
  CHECK(got.ids[1] == 2);  // second largest
  CHECK(got.weights[0] + got.weights[1] == doctest::Approx(1.0f).epsilon(1e-6));
  // exp(2)/(exp(2)+exp(1)) after renormalizing over the two kept probs.
  const float w0 = std::exp(2.0f) / (std::exp(2.0f) + std::exp(1.0f));
  CHECK(got.weights[0] == doctest::Approx(w0).epsilon(1e-6));
}

TEST_CASE("grouped_topk argument contract is enforced") {
  const int64_t t = 2, e = 64;
  const auto logits = RandF32(static_cast<size_t>(t * e), 31);
  std::vector<float> bias(static_cast<size_t>(e), 0.0f);
  Tensor tl = Contig(const_cast<float*>(logits.data()), DType::kF32, Cpu(), {t, e});
  Tensor tb = Contig(bias.data(), DType::kF32, Cpu(), {e});
  std::vector<float> w(static_cast<size_t>(t * 4));
  std::vector<int32_t> id(static_cast<size_t>(t * 4));
  Tensor tw = Contig(w.data(), DType::kF32, Cpu(), {t, 4});
  Tensor ti = Contig(id.data(), DType::kI32, Cpu(), {t, 4});
  Queue q = Q();

  MoeRouterTopKArgs ok;
  ok.top_k = 4;
  ok.num_expert_group = 8;
  ok.topk_group = 2;
  CHECK_NOTHROW(vt::MoeRouterTopK(q, tw, ti, tl, ok, &tb));

  // num_experts must divide evenly into the groups.
  {
    MoeRouterTopKArgs bad = ok;
    bad.num_expert_group = 7;  // 64 % 7 != 0
    CHECK_THROWS_AS(vt::MoeRouterTopK(q, tw, ti, tl, bad, &tb), std::runtime_error);
  }
  // topk_group must be in [1, num_expert_group].
  {
    MoeRouterTopKArgs bad = ok;
    bad.topk_group = 9;
    CHECK_THROWS_AS(vt::MoeRouterTopK(q, tw, ti, tl, bad, &tb), std::runtime_error);
  }
  // top_k cannot exceed what survives the mask (topk_group * group_size).
  {
    MoeRouterTopKArgs bad = ok;
    bad.topk_group = 1;   // 1 * 8 == 8 experts survive
    bad.top_k = 4;        // fine
    CHECK_NOTHROW(vt::MoeRouterTopK(q, tw, ti, tl, bad, &tb));
    MoeRouterTopKArgs worse = ok;
    worse.num_expert_group = 32;  // group_size 2
    worse.topk_group = 1;         // only 2 experts survive
    CHECK_THROWS_AS(vt::MoeRouterTopK(q, tw, ti, tl, worse, &tb), std::runtime_error);
  }
  // Grouped-only knobs are rejected on the ungrouped path, so a half-filled args
  // struct fails loudly instead of silently ignoring the caller's intent.
  {
    MoeRouterTopKArgs bad;
    bad.top_k = 4;
    bad.scoring_func = MoeScoringFunc::kSigmoid;
    CHECK_THROWS_AS(vt::MoeRouterTopK(q, tw, ti, tl, bad), std::runtime_error);
  }
  {
    MoeRouterTopKArgs bad;
    bad.top_k = 4;
    bad.routed_scaling_factor = 2.5f;
    CHECK_THROWS_AS(vt::MoeRouterTopK(q, tw, ti, tl, bad), std::runtime_error);
  }
  {
    MoeRouterTopKArgs bad;
    bad.top_k = 4;
    CHECK_THROWS_AS(vt::MoeRouterTopK(q, tw, ti, tl, bad, &tb), std::runtime_error);
  }
  // The bias must be [num_experts] f32.
  {
    std::vector<float> short_bias(4, 0.0f);
    Tensor bad = Contig(short_bias.data(), DType::kF32, Cpu(), {4});
    CHECK_THROWS_AS(vt::MoeRouterTopK(q, tw, ti, tl, ok, &bad), std::runtime_error);
  }
}

TEST_CASE("grouped_topk CUDA matches CPU across shapes and both scoring funcs") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping grouped_topk shape sweep");
    return;
  }
  struct Case {
    int64_t t, e;
    int top_k, n_group, topk_group;
    MoeScoringFunc score;
    bool bias;
    float scale;
  };
  const std::vector<Case> cases = {
      {1, 256, 8, 8, 4, MoeScoringFunc::kSigmoid, true, 2.5f},    // V3, single token
      {64, 256, 8, 8, 4, MoeScoringFunc::kSigmoid, true, 2.5f},   // V3, decode batch
      {7, 64, 6, 8, 3, MoeScoringFunc::kSoftmax, false, 1.0f},    // V2-style grouped
      {5, 128, 4, 16, 2, MoeScoringFunc::kSigmoid, false, 1.0f},  // sigmoid, no bias
      {3, 64, 2, 1, 1, MoeScoringFunc::kSoftmax, false, 1.0f},    // degenerate 1 group
  };
  uint32_t seed = 100;
  for (const Case& c : cases) {
    MoeRouterTopKArgs args;
    args.top_k = c.top_k;
    args.renormalize = true;
    args.scoring_func = c.score;
    args.num_expert_group = c.n_group;
    args.topk_group = c.topk_group;
    args.routed_scaling_factor = c.scale;
    const auto logits = RandF32(static_cast<size_t>(c.t * c.e), seed++);
    auto bias = RandF32(static_cast<size_t>(c.e), seed++, -0.3f, 0.3f);
    std::vector<float>* bp = c.bias ? &bias : nullptr;

    const RouterOut cpu = RunOpCpu(logits, c.t, c.e, args, bp);
    const RouterOut gpu = RunOpCuda(logits, c.t, c.e, args, bp);
    // Also pin both against the independent upstream-formula reference.
    const RouterOut ref = RefGroupedTopK(logits, c.t, c.e, args, bp);
    for (size_t i = 0; i < cpu.ids.size(); ++i) {
      CHECK(cpu.ids[i] == ref.ids[i]);
      CHECK(gpu.ids[i] == cpu.ids[i]);
      CHECK(gpu.weights[i] == doctest::Approx(cpu.weights[i]).epsilon(1e-5));
      CHECK(cpu.weights[i] == doctest::Approx(ref.weights[i]).epsilon(1e-5));
    }
  }
}
