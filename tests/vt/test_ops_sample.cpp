// Ported from: vllm/v1/sample/ops/topk_topp_sampler.py + vllm/v1/sample/sampler.py @ e24d1b24.
// Core sampling-op unit tests (M1.7 Task 2). Composed-reference goldens: each op
// is validated against hand-computed expecteds (argmax tie-break, temperature +
// eps guard, sort-based top-k/top-p masking + the apply_top_k_only fast-path
// parity, log_softmax / softmax) exactly the way M0.8/M1.6 validated their ops.
// greedy_argmax is the bit-exact parity primitive (lowest-index tie-break vs
// torch.argmax). random_sample is validated by ALGORITHM (exponential-noise
// gumbel-max), DETERMINISM (fixed seed), and DISTRIBUTION (large-N frequency ≈
// softmax) — bit-exact-vs-torch-Philox is the documented M1.7 deferral.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/sample_common.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

Tensor F32_2(std::vector<float>& v, int64_t a, int64_t b) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b});
}
Tensor F32_1(std::vector<float>& v, int64_t a) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a});
}
Tensor I32_1(std::vector<int32_t>& v, int64_t a) {
  return Tensor::Contiguous(v.data(), DType::kI32, Cpu(), {a});
}
Tensor I64_1(std::vector<int64_t>& v, int64_t a) {
  return Tensor::Contiguous(v.data(), DType::kI64, Cpu(), {a});
}
}  // namespace

// ---------------------------------------------------------------------------
// apply_temperature (sampler.py::apply_temperature). In-place /= temp per row.
TEST_CASE("apply_temperature: divides each row by its temperature") {
  std::vector<float> logits = {2.0f, 4.0f, 8.0f,
                               3.0f, 6.0f, 9.0f};
  std::vector<float> temp = {2.0f, 3.0f};
  Tensor tl = F32_2(logits, 2, 3), tt = F32_1(temp, 2);
  Queue q = Q();
  vt::ApplyTemperature(q, tl, tt, /*all_random=*/true);
  CHECK(logits[0] == doctest::Approx(1.0f));
  CHECK(logits[1] == doctest::Approx(2.0f));
  CHECK(logits[2] == doctest::Approx(4.0f));
  CHECK(logits[3] == doctest::Approx(1.0f));
  CHECK(logits[4] == doctest::Approx(2.0f));
  CHECK(logits[5] == doctest::Approx(3.0f));
}

TEST_CASE("apply_temperature: !all_random guards temp<eps greedy rows (unchanged)") {
  // Row 0 is greedy (temp 0 < eps) -> temp replaced by 1.0, row unchanged.
  // Row 1 is random (temp 0.5) -> divided.
  std::vector<float> logits = {2.0f, 4.0f,
                               2.0f, 4.0f};
  std::vector<float> temp = {0.0f, 0.5f};
  Tensor tl = F32_2(logits, 2, 2), tt = F32_1(temp, 2);
  Queue q = Q();
  vt::ApplyTemperature(q, tl, tt, /*all_random=*/false);
  CHECK(logits[0] == doctest::Approx(2.0f));  // unchanged (divided by 1.0)
  CHECK(logits[1] == doctest::Approx(4.0f));
  CHECK(logits[2] == doctest::Approx(4.0f));  // 2 / 0.5
  CHECK(logits[3] == doctest::Approx(8.0f));  // 4 / 0.5
}

// ---------------------------------------------------------------------------
// greedy_argmax (sampler.py::greedy_sample). Lowest-index tie-break, bit-exact
// vs torch.argmax (first occurrence of the max via strict `>`).
TEST_CASE("greedy_argmax: picks the max index") {
  std::vector<float> logits = {0.1f, 5.0f, 0.2f, 0.3f,
                               9.0f, 1.0f, 1.0f, 2.0f};
  std::vector<int64_t> ids(2, -1);
  Tensor tl = F32_2(logits, 2, 4), ti = I64_1(ids, 2);
  Queue q = Q();
  vt::GreedyArgmax(q, ti, tl);
  CHECK(ids[0] == 1);
  CHECK(ids[1] == 0);
}

TEST_CASE("greedy_argmax: tie resolves to the LOWEST index (torch.argmax)") {
  // [1,3,3,2] -> the two 3s tie; torch.argmax returns the FIRST -> index 1.
  // A wrong (last-wins) tie-break would return 2, so this case flips.
  std::vector<float> logits = {1.0f, 3.0f, 3.0f, 2.0f};
  std::vector<int64_t> ids(1, -1);
  Tensor tl = F32_2(logits, 1, 4), ti = I64_1(ids, 1);
  Queue q = Q();
  vt::GreedyArgmax(q, ti, tl);
  CHECK(ids[0] == 1);
}

// ---------------------------------------------------------------------------
// apply_top_k_top_p — top-k only (apply_top_k_only fast path when p is null).
TEST_CASE("apply_top_k: keeps exactly k distinct values, masks the rest to -inf") {
  // logits [1,2,3,4,5], k=3 -> keep {3,4,5} (idx 2,3,4), mask idx 0,1.
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<int32_t> k = {3};
  Tensor tl = F32_2(logits, 1, 5), tk = I32_1(k, 1);
  Queue q = Q();
  vt::ApplyTopKTopP(q, tl, &tk, /*p=*/nullptr);
  CHECK(logits[0] == kNegInf);
  CHECK(logits[1] == kNegInf);
  CHECK(logits[2] == doctest::Approx(3.0f));
  CHECK(logits[3] == doctest::Approx(4.0f));
  CHECK(logits[4] == doctest::Approx(5.0f));
}

TEST_CASE("apply_top_k: ties AT the threshold are kept (can keep > k, torch semantics)") {
  // logits [3,3,3,1], k=1. The k-th largest value is 3; torch masks strictly
  // `< 3`, so ALL three 3s survive (>k kept), only the 1 is masked.
  std::vector<float> logits = {3.0f, 3.0f, 3.0f, 1.0f};
  std::vector<int32_t> k = {1};
  Tensor tl = F32_2(logits, 1, 4), tk = I32_1(k, 1);
  Queue q = Q();
  vt::ApplyTopKTopP(q, tl, &tk, nullptr);
  CHECK(logits[0] == doctest::Approx(3.0f));
  CHECK(logits[1] == doctest::Approx(3.0f));
  CHECK(logits[2] == doctest::Approx(3.0f));
  CHECK(logits[3] == kNegInf);
}

TEST_CASE("apply_top_k: k == vocab is a no-op") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int32_t> k = {4};
  Tensor tl = F32_2(logits, 1, 4), tk = I32_1(k, 1);
  Queue q = Q();
  vt::ApplyTopKTopP(q, tl, &tk, nullptr);
  CHECK(logits[0] == doctest::Approx(1.0f));
  CHECK(logits[1] == doctest::Approx(2.0f));
  CHECK(logits[2] == doctest::Approx(3.0f));
  CHECK(logits[3] == doctest::Approx(4.0f));
}

TEST_CASE("apply_top_k_only path == sort path on the same input (test_topk_impl_equivalence)") {
  // Ported from tests/v1/sample/test_topk_topp_sampler.py::test_topk_impl_equivalence:
  // apply_top_k_top_p(k, p=None) [fast path] must match apply_top_k_top_p(k, p=1.0)
  // [sort path with a no-op top-p].
  std::vector<float> base = {0.5f, -1.0f, 2.0f, 3.5f, -4.0f, 1.25f, 0.0f, 7.0f};
  std::vector<float> a = base, b = base;
  std::vector<int32_t> k = {3};
  std::vector<float> noop_p = {1.0f};
  Queue q = Q();

  Tensor ta = F32_2(a, 1, 8), tk1 = I32_1(k, 1);
  vt::ApplyTopKTopP(q, ta, &tk1, nullptr);  // fast path

  std::vector<int32_t> k2 = {3};
  Tensor tb = F32_2(b, 1, 8), tk2 = I32_1(k2, 1), tp = F32_1(noop_p, 1);
  vt::ApplyTopKTopP(q, tb, &tk2, &tp);  // sort path with no-op top-p

  for (size_t i = 0; i < base.size(); ++i) {
    if (std::isinf(a[i]) || std::isinf(b[i])) {
      CHECK(std::isinf(a[i]));
      CHECK(std::isinf(b[i]));
    } else {
      CHECK(a[i] == doctest::Approx(b[i]));
    }
  }
}

// ---------------------------------------------------------------------------
// apply_top_k_top_p — top-p (sort-based path).
TEST_CASE("apply_top_p: keeps the smallest tail whose cumulative prob >= p") {
  // logits chosen so softmax == [0.1,0.2,0.3,0.4]. p=0.7 -> 1-p=0.3.
  // Ascending cumsum [0.1,0.3,0.6,1.0]; mask cumsum<=0.3 -> idx0,idx1 masked
  // (keep last always). Kept {0.3,0.4} sum 0.7 >= p.
  std::vector<float> logits = {std::log(0.1f), std::log(0.2f), std::log(0.3f), std::log(0.4f)};
  std::vector<float> p = {0.7f};
  Tensor tl = F32_2(logits, 1, 4), tp = F32_1(p, 1);
  Queue q = Q();
  vt::ApplyTopKTopP(q, tl, /*k=*/nullptr, &tp);
  CHECK(logits[0] == kNegInf);
  CHECK(logits[1] == kNegInf);
  CHECK(logits[2] == doctest::Approx(std::log(0.3f)));
  CHECK(logits[3] == doctest::Approx(std::log(0.4f)));
}

TEST_CASE("apply_top_p: at-least-one — the argmax always survives") {
  // p=0.05 -> 1-p=0.95; every non-top position is masked, only the largest
  // prob (idx3) is force-kept by mask[:, -1] = false.
  std::vector<float> logits = {std::log(0.1f), std::log(0.2f), std::log(0.3f), std::log(0.4f)};
  std::vector<float> p = {0.05f};
  Tensor tl = F32_2(logits, 1, 4), tp = F32_1(p, 1);
  Queue q = Q();
  vt::ApplyTopKTopP(q, tl, nullptr, &tp);
  CHECK(logits[0] == kNegInf);
  CHECK(logits[1] == kNegInf);
  CHECK(logits[2] == kNegInf);
  CHECK(logits[3] == doctest::Approx(std::log(0.4f)));
}

TEST_CASE("apply_top_k_top_p: combined k then p") {
  // softmax == [0.1,0.2,0.3,0.4]; k=3 masks the smallest (idx0). Re-softmax over
  // {0.2,0.3,0.4}/0.9, cumsum [0,0.2222,0.5556,1.0], 1-p=0.3 masks idx1 too.
  std::vector<float> logits = {std::log(0.1f), std::log(0.2f), std::log(0.3f), std::log(0.4f)};
  std::vector<int32_t> k = {3};
  std::vector<float> p = {0.7f};
  Tensor tl = F32_2(logits, 1, 4), tk = I32_1(k, 1), tp = F32_1(p, 1);
  Queue q = Q();
  vt::ApplyTopKTopP(q, tl, &tk, &tp);
  CHECK(logits[0] == kNegInf);
  CHECK(logits[1] == kNegInf);
  CHECK(logits[2] == doctest::Approx(std::log(0.3f)));
  CHECK(logits[3] == doctest::Approx(std::log(0.4f)));
}

// ---------------------------------------------------------------------------
// Sort-FREE top-k/top-p masking — the exact scalar mirror of the CUDA
// `ApplyTopKTopPRowKernel` (src/vt/cuda/cuda_sample.cu). The CUDA kernel replaced
// the per-row full-vocab thrust sort + <<<n,1>>> serial cumsum + blocking sync
// with a block-cooperative pivot-bracket THRESHOLD search (mirroring flashinfer's
// TopK/TopPRenormProb; sampling.cuh). This scalar reference computes the SAME
// threshold math sequentially so the algorithm can be validated on CPU-only CI
// (the .cu itself needs nvcc). It is cross-checked below against the independent
// sort-based reference (vt::ApplyTopKTopP CPU path, `ApplyTopKTopPSortRow`).
//
// Distribution equivalence: for DISTINCT logits the kept set is identical to the
// sort path. The ONLY divergence is the measure-zero case of tokens with exactly
// equal probability straddling the top-p / top-k boundary: the sort path splits a
// tie group (keeping some, dropping others of equal prob, by stable index order),
// while the threshold path — like flashinfer — keeps/drops whole tie groups
// atomically. Real f32 model logits are effectively continuous, so this never
// fires; the cross-check below uses distinct logits.
namespace {
constexpr float kSFNegInf = -std::numeric_limits<float>::infinity();

// count(row[j] > x) over the whole row.
int SFCountGt(const float* r, int64_t v, float x) {
  int c = 0;
  for (int64_t j = 0; j < v; ++j)
    if (r[j] > x) ++c;
  return c;
}

// k-th largest value (torch top-k threshold): mask masks logits STRICTLY below it,
// keeping >= it (ties at the threshold survive). Found via a two-pivot bracket that
// pins `low` to the largest value with count(row>low) >= k; the k-th largest is then
// the smallest value strictly greater than `low` (min_gt_low at convergence).
float SFTopKThreshold(const float* r, int64_t v, int32_t k, float mn, float mx) {
  if (SFCountGt(r, v, mn) < k) return mn;  // k-th largest == global min => no-op
  float low = mn, high = mx, min_gt_low = mx, max_le_high = low;
  do {
    const float p0 = (2.0f * low + high) / 3.0f;
    const float p1 = (low + 2.0f * high) / 3.0f;
    const int c0 = SFCountGt(r, v, p0);
    const int c1 = SFCountGt(r, v, p1);
    min_gt_low = high;
    max_le_high = low;
    for (int64_t j = 0; j < v; ++j) {
      if (r[j] > low) min_gt_low = std::min(min_gt_low, r[j]);
      if (r[j] <= high) max_le_high = std::max(max_le_high, r[j]);
    }
    if (c1 >= k) {
      low = p1;
    } else if (c0 >= k) {
      low = p0;
      high = std::min(p1, max_le_high);
    } else {
      high = std::min(p0, max_le_high);
    }
  } while (min_gt_low != max_le_high);
  return min_gt_low;  // smallest value strictly > low == k-th largest
}

// The sort-free mask: identical result to vt::ApplyTopKTopP for distinct logits.
void SFApplyTopKTopPRow(float* r, int64_t v, const int32_t* kptr, const float* pptr) {
  float mx = kSFNegInf, mn = std::numeric_limits<float>::infinity();
  for (int64_t j = 0; j < v; ++j) {
    mx = std::max(mx, r[j]);
    if (r[j] != kSFNegInf) mn = std::min(mn, r[j]);  // min over FINITE logits
  }

  // --- top-k threshold (keep row[j] >= threshold_k) ---
  float threshold_k = kSFNegInf;  // -inf => keep all
  const bool topk = (kptr != nullptr);
  const int32_t k = topk ? *kptr : 0;
  if (topk && k >= 1 && k < v) threshold_k = SFTopKThreshold(r, v, k, mn, mx);
  auto survivor = [&](int64_t j) { return r[j] >= threshold_k; };

  // --- top-p threshold over the survivors (keep survivor with e_j > low) ---
  float low = -1.0f;  // on e=exp(row-mx) in (0,1]; -1 => keep every survivor
  const bool topp = (pptr != nullptr);
  const float p = topp ? *pptr : 1.0f;
  if (topp && p < 1.0f) {
    float denom = 0.0f;
    for (int64_t j = 0; j < v; ++j)
      if (survivor(j)) denom += std::exp(r[j] - mx);
    const float target = p * denom;  // cumulative survivor mass to keep
    float hi = 1.0f;                  // max survivor e == exp(mx-mx) == 1
    low = 0.0f;  // denom==0 (no survivors) => loop skipped; row masked fully anyway
    float min_gt_low = hi, max_le_high = 0.0f;
    for (int iter = 0; denom > 0.0f && iter < 128; ++iter) {
      const float p0 = (2.0f * low + hi) / 3.0f;
      const float p1 = (low + 2.0f * hi) / 3.0f;
      float a0 = 0.0f, a1 = 0.0f;
      min_gt_low = hi;
      max_le_high = low;
      for (int64_t j = 0; j < v; ++j) {
        if (!survivor(j)) continue;
        const float e = std::exp(r[j] - mx);
        if (e > p0) a0 += e;
        if (e > p1) a1 += e;
        if (e > low) min_gt_low = std::min(min_gt_low, e);
        if (e <= hi) max_le_high = std::max(max_le_high, e);
      }
      if (a1 >= target) {
        low = p1;
      } else if (a0 >= target) {
        low = p0;
        hi = std::min(p1, max_le_high);
      } else {
        hi = std::min(p0, max_le_high);
      }
      if (min_gt_low == max_le_high) break;
    }
  }

  for (int64_t j = 0; j < v; ++j) {
    const bool keep = survivor(j) && (low < 0.0f || std::exp(r[j] - mx) > low);
    if (!keep) r[j] = kSFNegInf;
  }
}

// xorshift, matching the RandomLogits generator style, but scaled to distinct
// values (the sort path and the threshold path agree exactly only off-tie).
std::vector<float> DistinctLogits(int64_t v, uint32_t seed) {
  std::vector<float> out(static_cast<size_t>(v));
  uint32_t s = seed ? seed : 1;
  for (auto& x : out) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = (static_cast<float>(s) / static_cast<float>(UINT32_MAX)) * 20.0f - 10.0f;
  }
  return out;
}
}  // namespace

TEST_CASE("sort-free top-k/top-p == sort-based reference (distinct logits, many rows)") {
  // Validate the exact algorithm the CUDA kernel runs against the independent
  // sort-based CPU reference over randomized (k, p) and vocab sizes.
  struct Cfg {
    int64_t v;
    int32_t k;   // 0 => no top-k
    float p;     // <=0 => no top-p
  };
  const std::vector<Cfg> cfgs = {
      {512, 32, 0.8f},  {512, 0, 0.9f},   {512, 40, -1.f}, {256, 1, 0.5f},
      {256, 200, 0.95f}, {1000, 50, 0.7f}, {129, 7, 0.99f}, {777, 300, 0.3f},
  };
  for (size_t c = 0; c < cfgs.size(); ++c) {
    const Cfg& cfg = cfgs[c];
    for (uint32_t seed = 1; seed <= 12; ++seed) {
      std::vector<float> ref = DistinctLogits(cfg.v, seed * 2654435761u + static_cast<uint32_t>(c));
      std::vector<float> got = ref;
      const bool has_k = cfg.k > 0;
      const bool has_p = cfg.p > 0.0f;
      std::vector<int32_t> kv = {cfg.k};
      std::vector<float> pv = {cfg.p};

      // sort-based reference (the current CPU parity target).
      Tensor tref = F32_2(ref, 1, cfg.v);
      Tensor tk = I32_1(kv, 1), tp = F32_1(pv, 1);
      Queue q = Q();
      vt::ApplyTopKTopP(q, tref, has_k ? &tk : nullptr, has_p ? &tp : nullptr);

      // sort-free algorithm (mirror of the CUDA kernel).
      SFApplyTopKTopPRow(got.data(), cfg.v, has_k ? &cfg.k : nullptr, has_p ? &pv[0] : nullptr);

      for (int64_t j = 0; j < cfg.v; ++j) {
        if (std::isinf(ref[static_cast<size_t>(j)])) {
          CHECK(std::isinf(got[static_cast<size_t>(j)]));  // masked in both
        } else {
          CHECK_FALSE(std::isinf(got[static_cast<size_t>(j)]));  // kept in both
          CHECK(got[static_cast<size_t>(j)] == doctest::Approx(ref[static_cast<size_t>(j)]));
        }
      }
    }
  }
}

TEST_CASE("sort-free top-k/top-p == sort-based reference (with pre-masked -inf logits)") {
  // Upstream procs (min_p / bad_words / min_tokens / allowed_token_ids) mask
  // logits to -inf BEFORE top_k_top_p runs. The threshold search must treat -inf
  // as bottom (its finite-min bracket) and reproduce the sort path's kept set.
  struct Cfg { int64_t v; int32_t k; float p; };
  const std::vector<Cfg> cfgs = {
      {512, 32, 0.8f}, {512, 0, 0.9f}, {600, 64, -1.f}, {300, 5, 0.6f}, {256, 250, 0.95f},
  };
  for (size_t c = 0; c < cfgs.size(); ++c) {
    const Cfg& cfg = cfgs[c];
    for (uint32_t seed = 1; seed <= 10; ++seed) {
      std::vector<float> ref = DistinctLogits(cfg.v, seed * 40503u + 7u + static_cast<uint32_t>(c));
      // Mask ~1/5 of the row to -inf (deterministic scatter), leaving enough finite.
      for (int64_t j = 0; j < cfg.v; ++j)
        if ((j * 2654435761u + seed) % 5u == 0) ref[static_cast<size_t>(j)] = kNegInf;
      std::vector<float> got = ref;
      const bool has_k = cfg.k > 0;
      const bool has_p = cfg.p > 0.0f;
      std::vector<int32_t> kv = {cfg.k};
      std::vector<float> pv = {cfg.p};

      Tensor tref = F32_2(ref, 1, cfg.v);
      Tensor tk = I32_1(kv, 1), tp = F32_1(pv, 1);
      Queue q = Q();
      vt::ApplyTopKTopP(q, tref, has_k ? &tk : nullptr, has_p ? &tp : nullptr);
      SFApplyTopKTopPRow(got.data(), cfg.v, has_k ? &cfg.k : nullptr, has_p ? &pv[0] : nullptr);

      for (int64_t j = 0; j < cfg.v; ++j) {
        if (std::isinf(ref[static_cast<size_t>(j)])) {
          CHECK(std::isinf(got[static_cast<size_t>(j)]));
        } else {
          CHECK_FALSE(std::isinf(got[static_cast<size_t>(j)]));
          CHECK(got[static_cast<size_t>(j)] == doctest::Approx(ref[static_cast<size_t>(j)]));
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// compute_logprobs / compute_probs — log_softmax / softmax (f32, max-subtracted).
TEST_CASE("compute_logprobs: matches hand-computed log_softmax") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f};
  std::vector<float> out(3, 0.0f);
  Tensor tl = F32_2(logits, 1, 3), to = F32_2(out, 1, 3);
  Queue q = Q();
  vt::ComputeLogprobs(q, to, tl);
  // lse = log(e^1+e^2+e^3) = 3 + log(e^-2+e^-1+1).
  const float lse = 3.0f + std::log(std::exp(-2.0f) + std::exp(-1.0f) + 1.0f);
  CHECK(out[0] == doctest::Approx(1.0f - lse));
  CHECK(out[1] == doctest::Approx(2.0f - lse));
  CHECK(out[2] == doctest::Approx(3.0f - lse));
  CHECK(std::exp(out[0]) + std::exp(out[1]) + std::exp(out[2]) == doctest::Approx(1.0f));
}

TEST_CASE("compute_probs: matches hand-computed softmax") {
  std::vector<float> logits = {std::log(1.0f), std::log(2.0f), std::log(3.0f), std::log(4.0f)};
  std::vector<float> out(4, 0.0f);
  Tensor tl = F32_2(logits, 1, 4), to = F32_2(out, 1, 4);
  Queue q = Q();
  vt::ComputeProbs(q, to, tl);
  CHECK(out[0] == doctest::Approx(0.1f));
  CHECK(out[1] == doctest::Approx(0.2f));
  CHECK(out[2] == doctest::Approx(0.3f));
  CHECK(out[3] == doctest::Approx(0.4f));
}

// ---------------------------------------------------------------------------
// random_sample — exponential-noise gumbel-max.
TEST_CASE("random_sample: deterministic under a fixed seed") {
  std::vector<float> probs = {0.1f, 0.2f, 0.3f, 0.4f,
                              0.4f, 0.3f, 0.2f, 0.1f};
  std::vector<int64_t> seeds = {123, 456};
  std::vector<int64_t> a(2, -1), b(2, -1);
  Queue q = Q();
  Tensor tp = F32_2(probs, 2, 4), ts = I64_1(seeds, 2);
  Tensor ta = I64_1(a, 2), tb = I64_1(b, 2);
  vt::RandomSample(q, ta, tp, ts);
  vt::RandomSample(q, tb, tp, ts);
  CHECK(a[0] == b[0]);
  CHECK(a[1] == b[1]);
  CHECK(a[0] >= 0);
  CHECK(a[0] < 4);
}

TEST_CASE("random_sample: a per-request seed override changes only that row") {
  std::vector<float> probs = {0.25f, 0.25f, 0.25f, 0.25f,
                              0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<int64_t> seeds_a = {111, 222};
  std::vector<int64_t> seeds_c = {111, 999};  // row 1 seed changed
  std::vector<int64_t> out_a(2, -1), out_c(2, -1);
  Queue q = Q();
  Tensor tp = F32_2(probs, 2, 4);
  Tensor tsa = I64_1(seeds_a, 2), tsc = I64_1(seeds_c, 2);
  Tensor toa = I64_1(out_a, 2), toc = I64_1(out_c, 2);
  vt::RandomSample(q, toa, tp, tsa);
  vt::RandomSample(q, toc, tp, tsc);
  CHECK(out_a[0] == out_c[0]);  // row 0 seed unchanged -> same draw
}

TEST_CASE("random_sample: large-N empirical frequency approximates softmax probs") {
  // The exponential race gives P(argmax(p/q)) == p exactly; check the empirical
  // frequency over N independent rows (each row's noise is seeded by its index)
  // matches [0.1,0.2,0.3,0.4] within a loose Monte-Carlo tolerance.
  const int64_t N = 100000;
  const float target[4] = {0.1f, 0.2f, 0.3f, 0.4f};
  std::vector<float> probs(static_cast<size_t>(N) * 4);
  for (int64_t i = 0; i < N; ++i)
    for (int j = 0; j < 4; ++j) probs[static_cast<size_t>(i * 4 + j)] = target[j];
  std::vector<int64_t> seeds(static_cast<size_t>(N), 20260703);  // shared batch seed
  std::vector<int64_t> out(static_cast<size_t>(N), -1);
  Queue q = Q();
  Tensor tp = F32_2(probs, N, 4), ts = I64_1(seeds, N), to = I64_1(out, N);
  vt::RandomSample(q, to, tp, ts);

  int64_t counts[4] = {0, 0, 0, 0};
  for (int64_t i = 0; i < N; ++i) {
    REQUIRE(out[static_cast<size_t>(i)] >= 0);
    REQUIRE(out[static_cast<size_t>(i)] < 4);
    counts[out[static_cast<size_t>(i)]]++;
  }
  for (int j = 0; j < 4; ++j) {
    const float freq = static_cast<float>(counts[j]) / static_cast<float>(N);
    CHECK(freq == doctest::Approx(target[j]).epsilon(0.03));  // ~3% Monte-Carlo tol
  }
}

// ===========================================================================
// CPU-vs-CUDA parity. Correctness-grade CUDA kernels must match the CPU
// reference: greedy_argmax / top-k / top-p indices EXACT; temperature /
// logprobs / probs to 1e-5; random_sample uses the SAME deterministic hash RNG,
// so token ids must match EXACTLY. Guarded by HasCuda (dgx-pending on CPU-only
// CI). Harness mirrors test_ops_moe.cpp.
namespace {

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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
    t_ = MakeT(p_, dt, Gpu(), shape);
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

std::vector<float> RandomLogits(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  // simple xorshift for reproducible pseudo-random logits in [-5, 5)
  uint32_t s = seed ? seed : 1;
  for (auto& x : v) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = (static_cast<float>(s) / static_cast<float>(UINT32_MAX)) * 10.0f - 5.0f;
  }
  return v;
}

}  // namespace

TEST_CASE("CUDA greedy_argmax / temperature / top-k-p / logprobs match CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping (dgx-pending)");
    return;
  }
  const int64_t N = 6, V = 512;
  const auto logits = RandomLogits(static_cast<size_t>(N * V), 4242);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  Queue cq = Q();

  // greedy_argmax: indices must be bit-exact.
  {
    std::vector<float> lc = logits;
    std::vector<int64_t> id_cpu(static_cast<size_t>(N), -1);
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor ti = MakeT(id_cpu.data(), DType::kI64, Cpu(), {N});
    vt::GreedyArgmax(cq, ti, tl);
    DeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, logits.data());
    DeviceTensor did(gpu, gq.q, DType::kI64, {N});
    vt::GreedyArgmax(gq.q, did.tensor(), dl.tensor());
    std::vector<int64_t> id_gpu(static_cast<size_t>(N));
    did.Download(gq.q, id_gpu.data());
    for (size_t i = 0; i < id_cpu.size(); ++i) CHECK(id_gpu[i] == id_cpu[i]);
  }

  // compute_logprobs: 1e-5.
  {
    std::vector<float> lp_cpu(static_cast<size_t>(N * V));
    std::vector<float> lc = logits;
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor to = MakeT(lp_cpu.data(), DType::kF32, Cpu(), {N, V});
    vt::ComputeLogprobs(cq, to, tl);
    DeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, logits.data());
    DeviceTensor dout(gpu, gq.q, DType::kF32, {N, V});
    vt::ComputeLogprobs(gq.q, dout.tensor(), dl.tensor());
    std::vector<float> lp_gpu(lp_cpu.size());
    dout.Download(gq.q, lp_gpu.data());
    for (size_t i = 0; i < lp_cpu.size(); ++i)
      CHECK(lp_gpu[i] == doctest::Approx(lp_cpu[i]).epsilon(1e-5));
  }

  // top-k + top-p: masked positions and kept values must match exactly.
  {
    std::vector<float> lc = logits, lg = logits;
    std::vector<int32_t> k(static_cast<size_t>(N), 32);
    std::vector<float> p(static_cast<size_t>(N), 0.8f);
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor tk = MakeT(k.data(), DType::kI32, Cpu(), {N});
    Tensor tp = MakeT(p.data(), DType::kF32, Cpu(), {N});
    vt::ApplyTopKTopP(cq, tl, &tk, &tp);
    DeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, lg.data());
    DeviceTensor dk(gpu, gq.q, DType::kI32, {N}, k.data());
    DeviceTensor dp(gpu, gq.q, DType::kF32, {N}, p.data());
    vt::ApplyTopKTopP(gq.q, dl.tensor(), &dk.tensor(), &dp.tensor());
    std::vector<float> out_gpu(lc.size());
    dl.Download(gq.q, out_gpu.data());
    for (size_t i = 0; i < lc.size(); ++i) {
      if (std::isinf(lc[i])) {
        CHECK(std::isinf(out_gpu[i]));
      } else {
        CHECK(out_gpu[i] == doctest::Approx(lc[i]).epsilon(1e-5));
      }
    }
  }
}

// Exercises the two-pass multi-block argmax over a realistic vocab (many blocks
// per row): random rows, cross-block ties (lowest index must win), and an
// all-(-inf) row (must return index 0) -- all bit-exact vs the CPU reference.
TEST_CASE("CUDA greedy_argmax: large vocab, cross-block ties, all-(-inf) row") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping (dgx-pending)");
    return;
  }
  const int64_t N = 5, V = 151936;  // Qwen3 vocab: spans ~594 blocks per row
  auto logits = RandomLogits(static_cast<size_t>(N * V), 9137);

  // Row 1: two equal maxima far apart and across block boundaries; the CPU/GPU
  // must both return the LOWER index. Place them well above the random range.
  logits[1 * V + 40000] = 100.0f;
  logits[1 * V + 130000] = 100.0f;  // tie -> expect 40000
  // Row 2: unique max at a high index (past many blocks).
  logits[2 * V + 151900] = 250.0f;
  // Row 3: all -inf -> expect index 0.
  for (int64_t j = 0; j < V; ++j) logits[3 * V + j] = -INFINITY;

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  Queue cq = Q();

  std::vector<float> lc = logits;
  std::vector<int64_t> id_cpu(static_cast<size_t>(N), -1);
  Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
  Tensor ti = MakeT(id_cpu.data(), DType::kI64, Cpu(), {N});
  vt::GreedyArgmax(cq, ti, tl);

  DeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, logits.data());
  DeviceTensor did(gpu, gq.q, DType::kI64, {N});
  vt::GreedyArgmax(gq.q, did.tensor(), dl.tensor());
  std::vector<int64_t> id_gpu(static_cast<size_t>(N));
  did.Download(gq.q, id_gpu.data());

  for (size_t i = 0; i < id_cpu.size(); ++i) CHECK(id_gpu[i] == id_cpu[i]);
  CHECK(id_gpu[1] == 40000);   // tie resolves to the lower index
  CHECK(id_gpu[2] == 151900);  // high-index unique max
  CHECK(id_gpu[3] == 0);       // all -inf -> index 0
}

TEST_CASE("CUDA random_sample agrees with CPU on the vast majority of rows") {
  // NOTE: host and device compute q = -log(U) in double via different libm
  // (host libm vs CUDA libdevice). IEEE-754 does NOT require correctly-rounded
  // transcendentals, so log() can differ by ~1 ULP; on a row whose top two p/q
  // scores are near-tied that ULP can flip the argmax. So this is a STATISTICAL
  // agreement test (>=98% of rows match), NOT a bit-exact one — greedy/top-k/
  // top-p are the exact parity primitives; random is distribution-correct only
  // (bit-exact-vs-torch-Philox is the documented T1 deferral).
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping (dgx-pending)");
    return;
  }
  const int64_t N = 64, V = 128;
  // Build a valid probability matrix (softmax of random logits).
  auto logits = RandomLogits(static_cast<size_t>(N * V), 909);
  std::vector<float> probs(static_cast<size_t>(N * V));
  for (int64_t i = 0; i < N; ++i) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < V; ++j) mx = std::max(mx, logits[static_cast<size_t>(i * V + j)]);
    float sum = 0.0f;
    for (int64_t j = 0; j < V; ++j) {
      const float e = std::exp(logits[static_cast<size_t>(i * V + j)] - mx);
      probs[static_cast<size_t>(i * V + j)] = e;
      sum += e;
    }
    for (int64_t j = 0; j < V; ++j) probs[static_cast<size_t>(i * V + j)] /= sum;
  }
  std::vector<int64_t> seeds(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; ++i) seeds[static_cast<size_t>(i)] = 700 + i;

  std::vector<int64_t> id_cpu(static_cast<size_t>(N), -1);
  Tensor tp = MakeT(probs.data(), DType::kF32, Cpu(), {N, V});
  Tensor ts = MakeT(seeds.data(), DType::kI64, Cpu(), {N});
  Tensor ti = MakeT(id_cpu.data(), DType::kI64, Cpu(), {N});
  Queue cq = Q();
  vt::RandomSample(cq, ti, tp, ts);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dp(gpu, gq.q, DType::kF32, {N, V}, probs.data());
  DeviceTensor ds(gpu, gq.q, DType::kI64, {N}, seeds.data());
  DeviceTensor did(gpu, gq.q, DType::kI64, {N});
  vt::RandomSample(gq.q, did.tensor(), dp.tensor(), ds.tensor());
  std::vector<int64_t> id_gpu(static_cast<size_t>(N));
  did.Download(gq.q, id_gpu.data());
  size_t agree = 0;
  for (size_t i = 0; i < id_cpu.size(); ++i)
    if (id_gpu[i] == id_cpu[i]) ++agree;
  // Allow a small number of ULP-driven argmax flips; require strong agreement.
  CHECK(agree >= static_cast<size_t>(0.98 * static_cast<double>(N)));
}

// ===========================================================================
// CPU-vs-ROCm parity (same contracts as the CUDA block above). Skips when no
// ROCm backend is linked/registered (CPU CI). Lab proof: dual R9700 gfx1201.
namespace {

bool HasRocm() {
  try {
    vt::GetBackend(DeviceType::kROCM);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device RocmDev() { return Device{DeviceType::kROCM, 0}; }

class RocmDeviceTensor {
 public:
  RocmDeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
                   const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeT(p_, dt, RocmDev(), shape);
  }
  ~RocmDeviceTensor() { b_.Free(p_); }
  RocmDeviceTensor(const RocmDeviceTensor&) = delete;
  RocmDeviceTensor& operator=(const RocmDeviceTensor&) = delete;
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

}  // namespace

TEST_CASE("ROCm greedy_argmax / temperature / top-k-p / allowed_ids / logprobs match CPU") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend registered; skipping");
    return;
  }
  const int64_t N = 4, V = 256;
  const auto logits = RandomLogits(static_cast<size_t>(N * V), 7777);
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  QueueGuard gq(gpu);
  Queue cq = Q();

  // greedy_argmax bit-exact
  {
    std::vector<float> lc = logits;
    std::vector<int64_t> id_cpu(static_cast<size_t>(N), -1);
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor ti = MakeT(id_cpu.data(), DType::kI64, Cpu(), {N});
    vt::GreedyArgmax(cq, ti, tl);
    RocmDeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, logits.data());
    RocmDeviceTensor did(gpu, gq.q, DType::kI64, {N});
    vt::GreedyArgmax(gq.q, did.tensor(), dl.tensor());
    std::vector<int64_t> id_gpu(static_cast<size_t>(N));
    did.Download(gq.q, id_gpu.data());
    for (size_t i = 0; i < id_cpu.size(); ++i) CHECK(id_gpu[i] == id_cpu[i]);
  }

  // apply_temperature
  {
    std::vector<float> lc = logits, lg = logits;
    std::vector<float> temp(static_cast<size_t>(N), 0.7f);
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor tt = MakeT(temp.data(), DType::kF32, Cpu(), {N});
    vt::ApplyTemperature(cq, tl, tt, /*all_random=*/true);
    RocmDeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, lg.data());
    RocmDeviceTensor dt(gpu, gq.q, DType::kF32, {N}, temp.data());
    vt::ApplyTemperature(gq.q, dl.tensor(), dt.tensor(), /*all_random=*/true);
    std::vector<float> out(lg.size());
    dl.Download(gq.q, out.data());
    for (size_t i = 0; i < lc.size(); ++i)
      CHECK(out[i] == doctest::Approx(lc[i]).epsilon(1e-5));
  }

  // top-k only
  {
    std::vector<float> lc = logits, lg = logits;
    std::vector<int32_t> k(static_cast<size_t>(N), 16);
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor tk = MakeT(k.data(), DType::kI32, Cpu(), {N});
    vt::ApplyTopKTopP(cq, tl, &tk, /*p=*/nullptr);
    RocmDeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, lg.data());
    RocmDeviceTensor dk(gpu, gq.q, DType::kI32, {N}, k.data());
    vt::ApplyTopKTopP(gq.q, dl.tensor(), &dk.tensor(), /*p=*/nullptr);
    std::vector<float> out(lc.size());
    dl.Download(gq.q, out.data());
    for (size_t i = 0; i < lc.size(); ++i) {
      if (std::isinf(lc[i])) CHECK(std::isinf(out[i]));
      else CHECK(out[i] == doctest::Approx(lc[i]).epsilon(1e-5));
    }
  }

  // allowed_token_ids mask (Hermes title-gen path).
  // Contract: mask[i]!=0 → logit becomes -inf (blocked). 0 keeps the value.
  {
    std::vector<float> lc = logits, lg = logits;
    std::vector<int8_t> mask(static_cast<size_t>(N * V), 0);
    // Block all but first 8 ids per row
    for (int64_t n = 0; n < N; ++n)
      for (int64_t v = 8; v < V; ++v) mask[static_cast<size_t>(n * V + v)] = 1;
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor tm = MakeT(mask.data(), DType::kI8, Cpu(), {N, V});
    vt::ApplyAllowedTokenIds(cq, tl, tm);
    RocmDeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, lg.data());
    RocmDeviceTensor dm(gpu, gq.q, DType::kI8, {N, V}, mask.data());
    vt::ApplyAllowedTokenIds(gq.q, dl.tensor(), dm.tensor());
    std::vector<float> out(lc.size());
    dl.Download(gq.q, out.data());
    for (size_t i = 0; i < lc.size(); ++i) {
      if (std::isinf(lc[i])) CHECK(std::isinf(out[i]));
      else CHECK(out[i] == doctest::Approx(lc[i]).epsilon(1e-5));
    }
  }

  // compute_logprobs
  {
    std::vector<float> lp_cpu(static_cast<size_t>(N * V));
    std::vector<float> lc = logits;
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor to = MakeT(lp_cpu.data(), DType::kF32, Cpu(), {N, V});
    vt::ComputeLogprobs(cq, to, tl);
    RocmDeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, logits.data());
    RocmDeviceTensor dout(gpu, gq.q, DType::kF32, {N, V});
    vt::ComputeLogprobs(gq.q, dout.tensor(), dl.tensor());
    std::vector<float> lp_gpu(lp_cpu.size());
    dout.Download(gq.q, lp_gpu.data());
    for (size_t i = 0; i < lp_cpu.size(); ++i)
      CHECK(lp_gpu[i] == doctest::Approx(lp_cpu[i]).epsilon(1e-4));
  }
}

TEST_CASE("ROCm apply_min_p / penalties surface matches CPU mask pattern") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend registered; skipping");
    return;
  }
  // min_p: keep values that are high enough vs row max
  std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f,
                               5.0f, 4.0f, 0.0f, -1.0f};
  std::vector<float> min_p = {0.5f, 0.1f};
  std::vector<float> lc = logits, lg = logits;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  QueueGuard gq(gpu);
  Queue cq = Q();
  Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {2, 4});
  Tensor tp = MakeT(min_p.data(), DType::kF32, Cpu(), {2});
  vt::ApplyMinP(cq, tl, tp);
  RocmDeviceTensor dl(gpu, gq.q, DType::kF32, {2, 4}, lg.data());
  RocmDeviceTensor dp(gpu, gq.q, DType::kF32, {2}, min_p.data());
  vt::ApplyMinP(gq.q, dl.tensor(), dp.tensor());
  std::vector<float> out(8);
  dl.Download(gq.q, out.data());
  for (size_t i = 0; i < 8; ++i) {
    if (std::isinf(lc[i])) CHECK(std::isinf(out[i]));
    else CHECK(out[i] == doctest::Approx(lc[i]).epsilon(1e-5));
  }
}

TEST_CASE("ROCm random_sample agrees with CPU on the vast majority of rows") {
  // Same contract as the CUDA case above: host and device compute q = -log(U)
  // in double via different libm (host libm vs ROCm device libm), so ~1 ULP
  // differences can flip a near-tied argmax. Statistical >=98% agreement, not
  // bit-exact. The parallel kernel uses the same GumbelScore and ArgReduce as
  // the CPU reference, so the agreement is a property of the shared header.
  if (!HasRocm()) {
    MESSAGE("no ROCm backend registered; skipping");
    return;
  }
  const int64_t N = 64, V = 128;
  auto logits = RandomLogits(static_cast<size_t>(N * V), 909);
  std::vector<float> probs(static_cast<size_t>(N * V));
  for (int64_t i = 0; i < N; ++i) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < V; ++j) mx = std::max(mx, logits[static_cast<size_t>(i * V + j)]);
    float sum = 0.0f;
    for (int64_t j = 0; j < V; ++j) {
      const float e = std::exp(logits[static_cast<size_t>(i * V + j)] - mx);
      probs[static_cast<size_t>(i * V + j)] = e;
      sum += e;
    }
    for (int64_t j = 0; j < V; ++j) probs[static_cast<size_t>(i * V + j)] /= sum;
  }
  std::vector<int64_t> seeds(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; ++i) seeds[static_cast<size_t>(i)] = 700 + i;

  std::vector<int64_t> id_cpu(static_cast<size_t>(N), -1);
  Tensor tp = MakeT(probs.data(), DType::kF32, Cpu(), {N, V});
  Tensor ts = MakeT(seeds.data(), DType::kI64, Cpu(), {N});
  Tensor ti = MakeT(id_cpu.data(), DType::kI64, Cpu(), {N});
  Queue cq = Q();
  vt::RandomSample(cq, ti, tp, ts);

  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  QueueGuard gq(gpu);
  RocmDeviceTensor dp(gpu, gq.q, DType::kF32, {N, V}, probs.data());
  RocmDeviceTensor ds(gpu, gq.q, DType::kI64, {N}, seeds.data());
  RocmDeviceTensor did(gpu, gq.q, DType::kI64, {N});
  vt::RandomSample(gq.q, did.tensor(), dp.tensor(), ds.tensor());
  std::vector<int64_t> id_gpu(static_cast<size_t>(N));
  did.Download(gq.q, id_gpu.data());
  size_t agree = 0;
  for (size_t i = 0; i < id_cpu.size(); ++i)
    if (id_gpu[i] == id_cpu[i]) ++agree;
  CHECK(agree >= static_cast<size_t>(0.98 * static_cast<double>(N)));
}

// ===========================================================================
// #1984 — the parallel Gumbel draw selects the SAME token as the serial scan.
//
// The CUDA kernel cannot be compiled, let alone run, on a CPU-only host, so the
// property it depends on is gated here instead, on the PRODUCTION code it
// depends on: `vt::sample::ArgReduce`, `vt::sample::GumbelScore` and
// `vt::sample::ArgBlocksPerRow` are the same inlines src/vt/cuda/cuda_sample.cu
// compiles. A test that re-typed the reduction would prove the transcription
// rather than the function, which is why they live in a shared header at all.
//
// What is NOT claimed here: that the kernel launches correctly, that its shared
// memory is sized right, or that its scratch is safe. Those need a device, and
// the CUDA equality case further down is what asks them.
namespace {

using vt::sample::ArgBlocksPerRow;
using vt::sample::ArgReduce;
using vt::sample::GumbelScore;
using vt::sample::kArgSentinel;

struct ArgPair {
  float v;
  int64_t i;
};

// The serial left-to-right scan the CPU reference performs, expressed over the
// same operator, so "order-independent" is checked against the order that
// actually defines the answer.
ArgPair SerialReduce(const std::vector<float>& scores) {
  ArgPair a{kNegInf, kArgSentinel};
  for (int64_t j = 0; j < static_cast<int64_t>(scores.size()); ++j) {
    ArgReduce(a.v, a.i, scores[static_cast<size_t>(j)], j);
  }
  return a;
}

// The EXACT two-pass decomposition src/vt/cuda/cuda_sample.cu launches: pass 1
// gives block `blk` the elements `blk*block + t` strided by `blocks_per_row *
// block` and reduces them within the block; pass 2 reduces the per-block
// partials. `block` mirrors the kernel's kBlock.
ArgPair TwoPassReduce(const std::vector<float>& scores, int block) {
  const int64_t v = static_cast<int64_t>(scores.size());
  const int bpr = ArgBlocksPerRow(v, block);
  std::vector<ArgPair> partials;
  partials.reserve(static_cast<size_t>(bpr));
  for (int blk = 0; blk < bpr; ++blk) {
    // Per-thread accumulation, then the in-block tree reduction.
    std::vector<ArgPair> lanes(static_cast<size_t>(block), ArgPair{kNegInf, kArgSentinel});
    const int64_t stride = static_cast<int64_t>(bpr) * block;
    for (int t = 0; t < block; ++t) {
      ArgPair& lane = lanes[static_cast<size_t>(t)];
      for (int64_t j = static_cast<int64_t>(blk) * block + t; j < v; j += stride) {
        ArgReduce(lane.v, lane.i, scores[static_cast<size_t>(j)], j);
      }
    }
    for (int s = block / 2; s > 0; s >>= 1) {
      for (int t = 0; t < s; ++t) {
        ArgReduce(lanes[static_cast<size_t>(t)].v, lanes[static_cast<size_t>(t)].i,
                  lanes[static_cast<size_t>(t + s)].v, lanes[static_cast<size_t>(t + s)].i);
      }
    }
    partials.push_back(lanes[0]);
  }
  std::vector<ArgPair> lanes(static_cast<size_t>(block), ArgPair{kNegInf, kArgSentinel});
  for (int t = 0; t < block; ++t) {
    for (int j = t; j < bpr; j += block) {
      ArgReduce(lanes[static_cast<size_t>(t)].v, lanes[static_cast<size_t>(t)].i,
                partials[static_cast<size_t>(j)].v, partials[static_cast<size_t>(j)].i);
    }
  }
  for (int s = block / 2; s > 0; s >>= 1) {
    for (int t = 0; t < s; ++t) {
      ArgReduce(lanes[static_cast<size_t>(t)].v, lanes[static_cast<size_t>(t)].i,
                lanes[static_cast<size_t>(t + s)].v, lanes[static_cast<size_t>(t + s)].i);
    }
  }
  return lanes[0];
}

// A deterministic 32-bit mixer, so the cases below need no <random> and repeat
// byte for byte on every platform.
uint32_t Mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// The row shapes that break a careless parallel argmax, plus one ordinary one.
enum class RowKind { kAllEqual, kOneHot, kTopKMasked, kSoftmax, kAllZero, kTailMax };

std::vector<float> MakeProbRow(RowKind kind, int64_t v, uint32_t salt) {
  std::vector<float> row(static_cast<size_t>(v), 0.0f);
  switch (kind) {
    case RowKind::kAllEqual:
      // Every prob identical => the score ordering is decided purely by the
      // noise, and equal probs are where a tie-break bug is most likely.
      for (auto& x : row) x = 1.0f / static_cast<float>(v);
      break;
    case RowKind::kOneHot:
      row[static_cast<size_t>(Mix32(salt) % static_cast<uint32_t>(v))] = 1.0f;
      break;
    case RowKind::kTopKMasked:
      // What #1985 makes the common case: top-k has zeroed all but 20 entries,
      // so almost every score is an exact 0.0f and the winner is far from
      // index 0.
      for (int k = 0; k < 20; ++k) {
        const auto idx = static_cast<size_t>(Mix32(salt + static_cast<uint32_t>(k)) %
                                             static_cast<uint32_t>(v));
        row[idx] = 0.05f;
      }
      break;
    case RowKind::kSoftmax: {
      float sum = 0.0f;
      for (int64_t j = 0; j < v; ++j) {
        const float e = std::exp(
            static_cast<float>(Mix32(salt + static_cast<uint32_t>(j)) % 1000u) / 250.0f);
        row[static_cast<size_t>(j)] = e;
        sum += e;
      }
      for (auto& x : row) x /= sum;
      break;
    }
    case RowKind::kAllZero:
      // Every score is 0/q == 0: a whole row of ties. The answer must be index
      // 0, exactly as the serial scan gives.
      break;
    case RowKind::kTailMax:
      // The winner is the LAST element, so a decomposition that drops the ragged
      // tail of a row cannot pass.
      row[static_cast<size_t>(v - 1)] = 1.0f;
      break;
  }
  return row;
}

}  // namespace

TEST_CASE("ArgReduce is order-independent, including on exact ties") {
  // The defect a careless parallelisation introduces: translate the serial
  // rule as `if (b.v > a.v) a = b;` and the answer becomes order-DEPENDENT,
  // because reducing right-to-left then keeps the HIGHEST tied index. Deleting
  // the `bi < ai` clause in vt/sample_common.h must turn this case red.
  for (uint32_t salt = 0; salt < 8; ++salt) {
    std::vector<float> scores(97);
    for (size_t j = 0; j < scores.size(); ++j) {
      // Deliberately coarse, so exact ties are the common case rather than a
      // measure-zero accident.
      scores[j] = static_cast<float>(Mix32(salt * 31u + static_cast<uint32_t>(j)) % 5u);
    }
    const ArgPair forward = SerialReduce(scores);

    std::vector<float> reversed(scores.rbegin(), scores.rend());
    ArgPair backward{kNegInf, kArgSentinel};
    for (int64_t j = static_cast<int64_t>(scores.size()) - 1; j >= 0; --j) {
      ArgReduce(backward.v, backward.i, scores[static_cast<size_t>(j)], j);
    }
    CAPTURE(salt);
    CHECK(backward.i == forward.i);
    CHECK(backward.v == forward.v);

    // ...and the same under the kernel's own decomposition, at several block
    // widths, so the answer cannot depend on how the row was cut up.
    for (const int block : {1, 2, 8, 32, 256}) {
      CAPTURE(block);
      CHECK(TwoPassReduce(scores, block).i == forward.i);
    }
  }

  // An unfilled lane never wins, even against an all -inf row: the sentinel is
  // what makes index 0 the answer for a fully masked row.
  const std::vector<float> all_neg_inf(300, kNegInf);
  CHECK(SerialReduce(all_neg_inf).i == 0);
  CHECK(TwoPassReduce(all_neg_inf, 256).i == 0);
}

TEST_CASE("random_sample: the two-pass decomposition equals the serial reference") {
  // Every row shape, every vocabulary width that straddles a block boundary,
  // several seeds. The reference is `vt::RandomSample` on the CPU backend --
  // the production op, not a re-typed scan.
  const std::vector<int64_t> widths = {1, 2, 3, 255, 256, 257, 511, 1000, 65536, 248320};
  const std::vector<RowKind> kinds = {RowKind::kAllEqual,    RowKind::kOneHot,
                                      RowKind::kTopKMasked,  RowKind::kSoftmax,
                                      RowKind::kAllZero,     RowKind::kTailMax};
  for (const int64_t v : widths) {
    for (size_t ki = 0; ki < kinds.size(); ++ki) {
      for (const int64_t seed : {int64_t{0}, int64_t{1}, int64_t{700}, int64_t{-9}}) {
        CAPTURE(v);
        CAPTURE(ki);
        CAPTURE(seed);
        const std::vector<float> row =
            MakeProbRow(kinds[ki], v, static_cast<uint32_t>(v * 7 + ki));

        // The reference: the CPU op, one row.
        std::vector<float> probs = row;
        std::vector<int64_t> seeds = {seed};
        std::vector<int64_t> out = {-1};
        Tensor tp = F32_2(probs, 1, v);
        Tensor ts = I64_1(seeds, 1);
        Tensor to = I64_1(out, 1);
        Queue q = Q();
        vt::RandomSample(q, to, tp, ts);

        // The decomposition, over the same production score expression.
        std::vector<float> scores(static_cast<size_t>(v));
        for (int64_t j = 0; j < v; ++j) {
          scores[static_cast<size_t>(j)] =
              GumbelScore(row[static_cast<size_t>(j)], static_cast<uint64_t>(seed), 0, j);
        }
        CHECK(TwoPassReduce(scores, 256).i == out[0]);
      }
    }
  }
}

TEST_CASE("random_sample: a batch reduces row by row, and rows do not interfere") {
  // The kernel gives each row its own blockIdx.y and its own scratch slice, so
  // the batched answer has to equal the per-row answers computed alone. A
  // decomposition that indexed the scratch by block alone would pass every
  // single-row case above and fail here.
  const int64_t n = 5, v = 1000;
  std::vector<float> probs(static_cast<size_t>(n * v));
  std::vector<int64_t> seeds(static_cast<size_t>(n));
  const std::vector<RowKind> kinds = {RowKind::kAllEqual, RowKind::kTopKMasked,
                                      RowKind::kSoftmax, RowKind::kAllZero,
                                      RowKind::kTailMax};
  for (int64_t i = 0; i < n; ++i) {
    seeds[static_cast<size_t>(i)] = 100 + i;
    const std::vector<float> row =
        MakeProbRow(kinds[static_cast<size_t>(i)], v, static_cast<uint32_t>(i));
    std::copy(row.begin(), row.end(), probs.begin() + static_cast<size_t>(i * v));
  }
  std::vector<int64_t> out(static_cast<size_t>(n), -1);
  Tensor tp = F32_2(probs, n, v);
  Tensor ts = I64_1(seeds, n);
  Tensor to = I64_1(out, n);
  Queue q = Q();
  vt::RandomSample(q, to, tp, ts);

  for (int64_t i = 0; i < n; ++i) {
    CAPTURE(i);
    std::vector<float> scores(static_cast<size_t>(v));
    for (int64_t j = 0; j < v; ++j) {
      scores[static_cast<size_t>(j)] =
          GumbelScore(probs[static_cast<size_t>(i * v + j)],
                      static_cast<uint64_t>(seeds[static_cast<size_t>(i)]), i, j);
    }
    CHECK(TwoPassReduce(scores, 256).i == out[static_cast<size_t>(i)]);
  }
}

// ---------------------------------------------------------------------------
// CUDA: the parallel path and the RETAINED serial path must agree EXACTLY.
//
// Why this is a subprocess A/B rather than two calls. `VT_FAST_RANDOM_SAMPLE`
// is latched in a function-local static on first use (as VT_FAST_ARGMAX is), so
// one process can only ever exercise one arm. The parent below therefore
// re-execs THIS binary twice, once per arm, and requires the printed token ids
// to be byte-identical. That is a same-binary A/B in the sense AGENTS.md means:
// one build, one input, two selections of the code under test.
//
// It has to be exact rather than statistical, and the distinction matters. The
// CPU-vs-CUDA case above can only be statistical, because host and device
// evaluate `-log(u)` through different libm and IEEE-754 does not require a
// correctly-rounded transcendental, so a near-tied row can flip on one ULP.
// Here both arms are the same device libm on the same inputs and differ only in
// the ORDER the identical floats are combined -- and ArgReduce is
// order-independent -- so any difference at all is a defect.
//
// Both cases print WHICH arm they measured, in words, in their own output.
namespace {

// The shapes the child reports on, in one place so both arms enumerate the same
// ones. 248320 is Qwen3.8-27B's vocabulary, the width #1984 is about.
const std::vector<int64_t> kAbWidths = {1, 2, 255, 256, 257, 1000, 65536, 248320};

std::string RunSelf(const char* arm) {
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';
  const std::string cmd = "VT_FAST_RANDOM_SAMPLE=" + std::string(arm) + " " +
                          std::string(exe) +
                          " --no-skip --test-case='random_sample_ab_child' 2>&1";
  FILE* pipe = ::popen(cmd.c_str(), "r");
  REQUIRE(pipe != nullptr);
  std::string out;
  std::array<char, 4096> buf{};
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) out += buf.data();
  REQUIRE(::pclose(pipe) != -1);
  // Keep only the payload lines, so a doctest banner or a device warning cannot
  // make the two arms differ for a reason that is not the kernel.
  std::string ids;
  size_t pos = 0;
  while (pos < out.size()) {
    const size_t eol = out.find('\n', pos);
    const std::string line = out.substr(pos, eol == std::string::npos ? eol : eol - pos);
    if (line.rfind("IDS ", 0) == 0) ids += line + "\n";
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  return ids;
}

}  // namespace

// The CHILD. Skipped in a normal run; the parent re-execs it by name.
TEST_CASE("random_sample_ab_child" * doctest::skip()) {
  if (!HasCuda()) {
    std::cout << "IDS no-cuda\n" << std::flush;
    std::exit(0);
  }
  const char* arm = std::getenv("VT_FAST_RANDOM_SAMPLE");
  std::cout << "IDS arm=" << (arm == nullptr ? "unset(fast)" : arm) << "\n";
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const std::vector<RowKind> kinds = {RowKind::kAllEqual, RowKind::kTopKMasked,
                                      RowKind::kSoftmax, RowKind::kAllZero};
  for (const int64_t v : kAbWidths) {
    const int64_t n = 4;
    std::vector<float> probs(static_cast<size_t>(n * v));
    std::vector<int64_t> seeds(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      seeds[static_cast<size_t>(i)] = 700 + i;
      const std::vector<float> row =
          MakeProbRow(kinds[static_cast<size_t>(i)], v, static_cast<uint32_t>(v + i));
      std::copy(row.begin(), row.end(), probs.begin() + static_cast<size_t>(i * v));
    }
    QueueGuard gq(gpu);
    DeviceTensor dp(gpu, gq.q, DType::kF32, {n, v}, probs.data());
    DeviceTensor ds(gpu, gq.q, DType::kI64, {n}, seeds.data());
    DeviceTensor did(gpu, gq.q, DType::kI64, {n});
    vt::RandomSample(gq.q, did.tensor(), dp.tensor(), ds.tensor());
    std::vector<int64_t> ids(static_cast<size_t>(n));
    did.Download(gq.q, ids.data());
    std::cout << "IDS v=" << v;
    for (const int64_t id : ids) std::cout << " " << id;
    std::cout << "\n";
  }
  std::cout << std::flush;
  std::exit(0);
}

TEST_CASE("CUDA random_sample: the parallel path is BIT-IDENTICAL to the serial one") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping (GPU-pending)");
    return;
  }
  const std::string parallel = RunSelf("1");
  const std::string serial = RunSelf("0");
  MESSAGE("compared VT_FAST_RANDOM_SAMPLE=1 (the two-pass reduction) against "
          "VT_FAST_RANDOM_SAMPLE=0 (the retained single-thread scan), same binary");
  INFO("parallel arm:\n" << parallel << "serial arm:\n" << serial);
  REQUIRE_FALSE(parallel.empty());
  REQUIRE(parallel.find("IDS v=248320") != std::string::npos);
  // Only the arm label may differ.
  std::string a = parallel, b = serial;
  const auto strip_arm = [](std::string& t) {
    const size_t p = t.find("IDS arm=");
    if (p == std::string::npos) return;
    t.erase(p, t.find('\n', p) - p + 1);
  };
  strip_arm(a);
  strip_arm(b);
  CHECK(a == b);
}

// ---------------------------------------------------------------------------
// ROCm: the same A/B contract as CUDA above. The child re-execs with
// VT_FAST_RANDOM_SAMPLE=0 and =1 and the token ids must be byte-identical,
// because both arms use the same device libm and ArgReduce is order-independent.
TEST_CASE("random_sample_ab_child_rocm" * doctest::skip()) {
  if (!HasRocm()) {
    std::cout << "IDS no-rocm\n" << std::flush;
    std::exit(0);
  }
  const char* arm = std::getenv("VT_FAST_RANDOM_SAMPLE");
  std::cout << "IDS arm=" << (arm == nullptr ? "unset(fast)" : arm) << "\n";
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  const std::vector<RowKind> kinds = {RowKind::kAllEqual, RowKind::kTopKMasked,
                                      RowKind::kSoftmax, RowKind::kAllZero};
  for (const int64_t v : kAbWidths) {
    const int64_t n = 4;
    std::vector<float> probs(static_cast<size_t>(n * v));
    std::vector<int64_t> seeds(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      seeds[static_cast<size_t>(i)] = 700 + i;
      const std::vector<float> row =
          MakeProbRow(kinds[static_cast<size_t>(i)], v, static_cast<uint32_t>(v + i));
      std::copy(row.begin(), row.end(), probs.begin() + static_cast<size_t>(i * v));
    }
    QueueGuard gq(gpu);
    RocmDeviceTensor dp(gpu, gq.q, DType::kF32, {n, v}, probs.data());
    RocmDeviceTensor ds(gpu, gq.q, DType::kI64, {n}, seeds.data());
    RocmDeviceTensor did(gpu, gq.q, DType::kI64, {n});
    vt::RandomSample(gq.q, did.tensor(), dp.tensor(), ds.tensor());
    std::vector<int64_t> ids(static_cast<size_t>(n));
    did.Download(gq.q, ids.data());
    std::cout << "IDS v=" << v;
    for (const int64_t id : ids) std::cout << " " << id;
    std::cout << "\n";
  }
  std::cout << std::flush;
  std::exit(0);
}

TEST_CASE("ROCm random_sample: the parallel path is BIT-IDENTICAL to the serial one") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend registered; skipping");
    return;
  }
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';
  auto run = [&](const char* arm) {
    const std::string cmd = "VT_FAST_RANDOM_SAMPLE=" + std::string(arm) + " " +
                            std::string(exe) +
                            " --no-skip --test-case='random_sample_ab_child_rocm' 2>&1";
    FILE* pipe = ::popen(cmd.c_str(), "r");
    REQUIRE(pipe != nullptr);
    std::string out;
    std::array<char, 4096> buf{};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) out += buf.data();
    REQUIRE(::pclose(pipe) != -1);
    std::string ids;
    size_t pos = 0;
    while (pos < out.size()) {
      const size_t eol = out.find('\n', pos);
      const std::string line = out.substr(pos, eol == std::string::npos ? eol : eol - pos);
      if (line.rfind("IDS ", 0) == 0) ids += line + "\n";
      if (eol == std::string::npos) break;
      pos = eol + 1;
    }
    return ids;
  };
  const std::string parallel = run("1");
  const std::string serial = run("0");
  MESSAGE("compared VT_FAST_RANDOM_SAMPLE=1 (the block-cooperative reduction) against "
          "VT_FAST_RANDOM_SAMPLE=0 (the retained single-thread scan), same binary");
  INFO("parallel arm:\n" << parallel << "serial arm:\n" << serial);
  REQUIRE_FALSE(parallel.empty());
  REQUIRE(parallel.find("IDS v=248320") != std::string::npos);
  std::string a = parallel, b = serial;
  const auto strip_arm = [](std::string& t) {
    const size_t p = t.find("IDS arm=");
    if (p == std::string::npos) return;
    t.erase(p, t.find('\n', p) - p + 1);
  };
  strip_arm(a);
  strip_arm(b);
  CHECK(a == b);
}
