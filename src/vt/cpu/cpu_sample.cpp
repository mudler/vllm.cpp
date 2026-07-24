// Ported from: vllm/v1/sample/ops/topk_topp_sampler.py + vllm/v1/sample/sampler.py @ e24d1b24.
//
// CPU reference kernels for the V1 sampling ops (M1.7 Task 2): temperature,
// greedy argmax (the bit-exact parity primitive), the sort-based top-k/top-p
// masking + the apply_top_k_only fast path, softmax/log_softmax, and the
// exponential-noise gumbel-max random sampler. All math is f32; the logits are
// the model's final `[num_reqs, vocab]` row-major layer output. Correctness-grade
// scalar loops (the FlashInfer/Triton fused samplers are M2.4).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "vt/ops.h"

namespace vt::cpu {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// apply_temperature (sampler.py:227-237). `logits.div_(temp.unsqueeze(1))`, with
// the per-row `where(temp<eps, 1.0, temp)` guard applied first when !all_random
// so greedy rows (temp≈0) do not divide by zero. In place.
void ApplyTemperatureKernel(Queue&, Tensor& logits, const Tensor& temp, bool all_random) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const float* tp = temp.Ptr<float>();
  float* lp = logits.Ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    float t = tp[i];
    if (!all_random && t < kSamplingEps) t = 1.0f;  // torch.where(temp<eps,1.0,temp)
    float* row = lp + i * v;
    for (int64_t j = 0; j < v; ++j) row[j] /= t;
  }
}

// greedy_sample (sampler.py:239-241) = argmax(dim=-1). Strict `>` keeps the FIRST
// (lowest-index) max — bit-exact vs torch.argmax on f32 logits.
void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const float* lp = logits.Ptr<float>();
  int64_t* out = token_ids.Ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const float* row = lp + i * v;
    int64_t best = 0;
    float best_v = row[0];
    for (int64_t j = 1; j < v; ++j) {
      if (row[j] > best_v) {  // strict `>` => first occurrence of the max wins
        best_v = row[j];
        best = j;
      }
    }
    out[i] = best;
  }
}

// apply_top_k_only (topk_topp_sampler.py:407-427): the no-sort k-only fast path.
// Per row, threshold = the k-th largest value (topk.values gathered at k-1);
// k == vocab is a no-op (threshold -inf). masked_fill_(logits < threshold, -inf).
void ApplyTopKOnlyRow(float* row, int64_t v, int32_t k) {
  if (k >= v || k < 1) return;  // no_top_k row (k>=v) or invalid k(<1): nothing to mask
  // k-th largest = element at index (k-1) of the descending order. nth_element on
  // a copy gives that pivot value without a full sort.
  std::vector<float> tmp(row, row + v);
  std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), std::greater<float>());
  const float threshold = tmp[static_cast<size_t>(k - 1)];
  for (int64_t j = 0; j < v; ++j)
    if (row[j] < threshold) row[j] = kNegInf;
}

// The sort-based path (topk_topp_sampler.py::apply_top_k_top_p_pytorch:363-404).
// One row: sort ascending (keep the permutation), apply the top-k threshold mask
// (`logits_sort < k-th largest`), then the top-p mask (softmax→cumsum, mask
// `cumsum <= 1-p`, force-keep the last/largest), then scatter back.
void ApplyTopKTopPSortRow(float* row, int64_t v, const int32_t* k_ptr, const float* p_ptr) {
  // logits_sort, logits_idx = logits.sort(dim=-1, descending=False).
  std::vector<int64_t> idx(static_cast<size_t>(v));
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&](int64_t a, int64_t b) { return row[a] < row[b]; });
  std::vector<float> sorted(static_cast<size_t>(v));
  for (int64_t s = 0; s < v; ++s) sorted[static_cast<size_t>(s)] = row[idx[static_cast<size_t>(s)]];

  if (k_ptr != nullptr) {
    const int64_t k = *k_ptr;
    // top_k_mask = logits_sort < logits_sort[vocab-k] (the k-th largest).
    // k >= v is a no-op (threshold = sorted[<=0], masks nothing); k < 1 is
    // invalid upstream (SamplingParams guarantees k in [1, vocab]) — guard both
    // to keep the sorted[v-k] index in-bounds.
    if (k >= 1 && k < v) {
      const float threshold = sorted[static_cast<size_t>(v - k)];
      for (int64_t s = 0; s < v; ++s)
        if (sorted[static_cast<size_t>(s)] < threshold) sorted[static_cast<size_t>(s)] = kNegInf;
    }
  }

  if (p_ptr != nullptr) {
    const float p = *p_ptr;
    // probs_sort = softmax(logits_sort); probs_sum = cumsum(probs_sort).
    float mx = kNegInf;
    for (int64_t s = 0; s < v; ++s) mx = std::max(mx, sorted[static_cast<size_t>(s)]);
    float denom = 0.0f;
    std::vector<float> probs(static_cast<size_t>(v));
    for (int64_t s = 0; s < v; ++s) {
      const float e = sorted[static_cast<size_t>(s)] == kNegInf
                          ? 0.0f
                          : std::exp(sorted[static_cast<size_t>(s)] - mx);
      probs[static_cast<size_t>(s)] = e;
      denom += e;
    }
    float cum = 0.0f;
    const float cutoff = 1.0f - p;
    for (int64_t s = 0; s < v; ++s) {
      cum += probs[static_cast<size_t>(s)] / denom;
      // top_p_mask = probs_sum <= 1-p, with mask[:, -1] = False (at least one).
      const bool keep_last = (s == v - 1);
      if (!keep_last && cum <= cutoff) sorted[static_cast<size_t>(s)] = kNegInf;
    }
  }

  // logits.scatter_(dim=-1, index=logits_idx, src=logits_sort).
  for (int64_t s = 0; s < v; ++s) row[idx[static_cast<size_t>(s)]] = sorted[static_cast<size_t>(s)];
}

// apply_top_k_top_p (topk_topp_sampler.py:345-360 + apply_top_k_top_p_pytorch,
// the CPU allow_cpu_sync path): dispatch to the k-only fast path when p is null,
// else the sort-based path. Per-row k/p (nullptr tensor => skip that mask). k/p
// tensors, when present, are [num_reqs].
void ApplyTopKTopPKernel(Queue&, Tensor& logits, const Tensor* k, const Tensor* p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  float* lp = logits.Ptr<float>();
  const int32_t* kp = k != nullptr ? k->Ptr<int32_t>() : nullptr;
  const float* pp = p != nullptr ? p->Ptr<float>() : nullptr;
  for (int64_t i = 0; i < n; ++i) {
    float* row = lp + i * v;
    if (pp == nullptr) {
      // p is None: k-only fast path (allow_cpu_sync=True). k must be present
      // (the both-None case is short-circuited by the vt::ApplyTopKTopP wrapper).
      ApplyTopKOnlyRow(row, v, kp[i]);
    } else {
      const int32_t* krow = kp != nullptr ? &kp[i] : nullptr;
      ApplyTopKTopPSortRow(row, v, krow, &pp[i]);
    }
  }
}

// Row-wise softmax (f32, max-subtracted). forward_native's
// `logits.softmax(dim=-1, dtype=torch.float32)`.
void ComputeProbsKernel(Queue&, Tensor& probs, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const float* lp = logits.Ptr<float>();
  float* op = probs.Ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    const float* row = lp + i * v;
    float* out = op + i * v;
    float mx = kNegInf;
    for (int64_t j = 0; j < v; ++j) mx = std::max(mx, row[j]);
    float sum = 0.0f;
    for (int64_t j = 0; j < v; ++j) {
      const float e = std::exp(row[j] - mx);
      out[j] = e;
      sum += e;
    }
    for (int64_t j = 0; j < v; ++j) out[j] /= sum;
  }
}

// compute_logprobs (sampler.py:304-306) = log_softmax(dim=-1). log-sum-exp with
// the row max subtracted: logprob = x - max - log(sum exp(x-max)).
void ComputeLogprobsKernel(Queue&, Tensor& logprobs, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const float* lp = logits.Ptr<float>();
  float* op = logprobs.Ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    const float* row = lp + i * v;
    float* out = op + i * v;
    float mx = kNegInf;
    for (int64_t j = 0; j < v; ++j) mx = std::max(mx, row[j]);
    float sum = 0.0f;
    for (int64_t j = 0; j < v; ++j) sum += std::exp(row[j] - mx);
    const float lse = mx + std::log(sum);
    for (int64_t j = 0; j < v; ++j) out[j] = row[j] - lse;
  }
}

// Deterministic seeded RNG for the exponential-noise gumbel-max sampler. splitmix64
// is a pure-uint64 mixer (bit-identical on any platform); the ONLY float op is the
// inverse-CDF `q = -log(U)` with U in (0,1). NOT bit-exact vs torch's Philox4x32
// `exponential_()` — that is the documented M1.7 deferral. Given (seed, row, col)
// it is fully reproducible and distribution-correct: U is a 53-bit uniform in
// (0,1), so q ~ Exponential(1) exactly, and the exponential race
// argmax(p_j / q_j) selects j with probability p_j (== softmax). The row index is
// mixed in so batch-default rows (shared seed) still get independent noise.
inline uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

inline double ExpNoise(uint64_t seed, int64_t row, int64_t col) {
  const uint64_t row_key = SplitMix64(seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(row));
  const uint64_t r = SplitMix64(row_key + static_cast<uint64_t>(col));
  // 53-bit uniform in (0,1): ((r>>11)+1) / (2^53 + 1). Strictly positive, < 1.
  const double u = static_cast<double>((r >> 11) + 1ULL) * (1.0 / 9007199254740993.0);
  return -std::log(u);  // Exponential(1) inverse-CDF
}

// random_sample (topk_topp_sampler.py::random_sample + sample_with_exponential_noise).
// scores = probs / q with q ~ Exp(1); argmax(scores, dim=-1) (lowest-index
// tie-break). `seeds[i]` is the per-row seed resolved by the Sampler (batch
// default or the per-request override from SamplingMetadata.generators).
void RandomSampleKernel(Queue&, Tensor& token_ids, const Tensor& probs, const Tensor& seeds) {
  const int64_t n = probs.shape[0], v = probs.shape[1];
  const float* pp = probs.Ptr<float>();
  const int64_t* sp = seeds.Ptr<int64_t>();
  int64_t* out = token_ids.Ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const float* row = pp + i * v;
    const uint64_t seed = static_cast<uint64_t>(sp[i]);
    int64_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < v; ++j) {
      const float q = static_cast<float>(ExpNoise(seed, i, j));
      const float score = row[j] / q;  // probs / q (higher => more likely)
      if (score > best_v) {            // strict `>` => lowest-index tie-break
        best_v = score;
        best = j;
      }
    }
    out[i] = best;
  }
}

// --- apply_penalties (utils.py::apply_penalties + the apply_repetition_penalties
// custom op, _custom_ops.py::apply_repetition_penalties_torch). Single fused
// elementwise pass matching the exact OpenAI-defined formula. In place.
void ApplyPenaltiesKernel(Queue&, Tensor& logits, const Tensor& prompt_mask,
                          const Tensor& output_bin_counts, const Tensor& output_mask,
                          const Tensor& frequency_penalties, const Tensor& presence_penalties,
                          const Tensor& repetition_penalties) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  float* lp = logits.Ptr<float>();
  const int8_t* pm = prompt_mask.Ptr<int8_t>();
  const int8_t* om = output_mask.Ptr<int8_t>();
  const int32_t* oc = output_bin_counts.Ptr<int32_t>();
  const float* freq = frequency_penalties.Ptr<float>();
  const float* pres = presence_penalties.Ptr<float>();
  const float* rep = repetition_penalties.Ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    float* row = lp + i * v;
    const int8_t* pmr = pm + i * v;
    const int8_t* omr = om + i * v;
    const int32_t* ocr = oc + i * v;
    const float r = rep[i], fr = freq[i], pr = pres[i];
    for (int64_t j = 0; j < v; ++j) {
      // repetition: present in prompt OR output -> divide/multiply by r by sign.
      if (pmr[j] || omr[j]) {
        row[j] = row[j] > 0.0f ? row[j] / r : row[j] * r;
      }
      // frequency (output count) then presence (output presence).
      row[j] -= fr * static_cast<float>(ocr[j]);
      row[j] -= pr * static_cast<float>(omr[j]);
    }
  }
}

// apply (builtin.py::MinPLogitsProcessor.apply). Per row softmax -> max prob ->
// mask probs < min_p*max to -inf. min_p==0 rows are untouched (threshold 0).
void ApplyMinPKernel(Queue&, Tensor& logits, const Tensor& min_p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  float* lp = logits.Ptr<float>();
  const float* mp = min_p.Ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    const float m = mp[i];
    if (m <= 0.0f) continue;  // no-op row (matches min_p_count / zero-threshold)
    float* row = lp + i * v;
    float mx = kNegInf;
    for (int64_t j = 0; j < v; ++j) mx = std::max(mx, row[j]);
    float sum = 0.0f;
    std::vector<float> probs(static_cast<size_t>(v));
    for (int64_t j = 0; j < v; ++j) {
      const float e = std::exp(row[j] - mx);
      probs[static_cast<size_t>(j)] = e;
      sum += e;
    }
    float pmax = 0.0f;
    for (int64_t j = 0; j < v; ++j) pmax = std::max(pmax, probs[static_cast<size_t>(j)] / sum);
    const float thr = m * pmax;
    for (int64_t j = 0; j < v; ++j)
      if (probs[static_cast<size_t>(j)] / sum < thr) row[j] = kNegInf;
  }
}

// apply (builtin.py::LogitBiasLogitsProcessor.apply) — logits[(rows,cols)] += bias.
void ApplyLogitBiasKernel(Queue&, Tensor& logits, const Tensor& rows, const Tensor& cols,
                          const Tensor& biases) {
  const int64_t v = logits.shape[1], m = rows.shape[0];
  float* lp = logits.Ptr<float>();
  const int32_t* rp = rows.Ptr<int32_t>();
  const int32_t* cp = cols.Ptr<int32_t>();
  const float* bp = biases.Ptr<float>();
  for (int64_t k = 0; k < m; ++k) lp[static_cast<int64_t>(rp[k]) * v + cp[k]] += bp[k];
}

// Sparse -inf scatter (min-tokens / bad-words). logits[(rows,cols)] = -inf.
void ApplyTokenMaskKernel(Queue&, Tensor& logits, const Tensor& rows, const Tensor& cols) {
  const int64_t v = logits.shape[1], m = rows.shape[0];
  float* lp = logits.Ptr<float>();
  const int32_t* rp = rows.Ptr<int32_t>();
  const int32_t* cp = cols.Ptr<int32_t>();
  for (int64_t k = 0; k < m; ++k) lp[static_cast<int64_t>(rp[k]) * v + cp[k]] = kNegInf;
}

// ─── Greedy spec-decode rejection sampling (SPEC-REJECTION I3) ──────────────
// The CPU REFERENCE for vt::GreedyRejectionSample. Ported from the `is_greedy`
// branch of `_rejection_kernel` (rejection_sampler_utils.py:564-585 accept/store,
// :628 accepted-length store) plus the greedy short-circuits of
// `_resample_kernel` (:846-849) and `_insert_resampled_kernel` (:828-841).
// The row argmax uses the SAME strict `>` lowest-index tie-break as
// GreedyArgmaxKernel above, so a k == 0 request reduces exactly to plain greedy.
void GreedyRejectionSampleKernel(Queue&, Tensor& sampled, Tensor& num_sampled,
                                 const Tensor& logits, const Tensor& draft_sampled,
                                 const Tensor& cu_num_logits) {
  const int64_t vocab = logits.shape[1];
  const int64_t num_reqs = cu_num_logits.shape[0] - 1;
  const int64_t width = sampled.shape[1];
  const float* lp = logits.Ptr<float>();
  const int32_t* dp = draft_sampled.Ptr<int32_t>();
  const int32_t* cu = cu_num_logits.Ptr<int32_t>();
  int32_t* sp = sampled.Ptr<int32_t>();
  int32_t* ns = num_sampled.Ptr<int32_t>();

  // argmax(logits[row]) with torch.argmax's lowest-index tie-break.
  const auto row_argmax = [&](int64_t row) -> int32_t {
    const float* r = lp + row * vocab;
    int64_t best = 0;
    float best_v = r[0];
    for (int64_t j = 1; j < vocab; ++j) {
      if (r[j] > best_v) {  // strict `>` => first occurrence of the max wins
        best_v = r[j];
        best = j;
      }
    }
    return static_cast<int32_t>(best);
  };

  for (int64_t req = 0; req < num_reqs; ++req) {
    const int64_t start = cu[req];
    const int64_t end = cu[req + 1];
    const int64_t num_draft_tokens = end - start - 1;
    int32_t* row = sp + req * width;
    for (int64_t j = 0; j < width; ++j) row[j] = -1;  // PLACEHOLDER_TOKEN_ID pad

    bool accepted = true;
    int64_t accepted_length = 0;
    for (int64_t i = 0; i < num_draft_tokens; ++i) {
      if (!accepted) break;  // upstream `elif accepted:` guard
      const int32_t target_argmax = row_argmax(start + i);
      // NOTE the +1: draft token i is the INPUT id at the NEXT expanded row
      // (rejection_sampler_utils.py:534). A -1 placeholder can never equal an
      // argmax (>= 0), so it is rejected with no out-of-bounds read.
      const int32_t draft = dp[start + i + 1];
      accepted = (target_argmax == draft);
      row[i] = accepted ? draft : target_argmax;
      accepted_length += accepted ? 1 : 0;
    }
    // Bonus token: every draft accepted -> resample (greedy: argmax) at the
    // bonus row. On a rejection the target argmax is already stored, so
    // _resample_kernel returns early (:846-849).
    if (accepted_length == num_draft_tokens) {
      row[accepted_length] = row_argmax(start + accepted_length);
    }
    // _insert_resampled_kernel:834 — num_sampled = accepted_length + 1.
    ns[req] = static_cast<int32_t>(accepted_length) + 1;
  }
}

// masked_fill_(mask, -inf) — mask TRUE means EXCLUDE (sampler.py:396-397).
void ApplyAllowedTokenIdsKernel(Queue&, Tensor& logits, const Tensor& mask) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  float* lp = logits.Ptr<float>();
  const int8_t* mk = mask.Ptr<int8_t>();
  for (int64_t idx = 0; idx < n * v; ++idx)
    if (mk[idx]) lp[idx] = kNegInf;
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kApplyTemperature, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ApplyTemperatureFn>(&ApplyTemperatureKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
    RegisterOp(OpId::kApplyTopKTopP, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ApplyTopKTopPFn>(&ApplyTopKTopPKernel)));
    RegisterOp(OpId::kComputeProbs, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ComputeProbsFn>(&ComputeProbsKernel)));
    RegisterOp(OpId::kComputeLogprobs, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ComputeLogprobsFn>(&ComputeLogprobsKernel)));
    RegisterOp(OpId::kRandomSample, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<RandomSampleFn>(&RandomSampleKernel)));
    RegisterOp(OpId::kGreedyRejectionSample, DeviceType::kCPU,
               reinterpret_cast<void*>(
                   static_cast<GreedyRejectionSampleFn>(&GreedyRejectionSampleKernel)));
    RegisterOp(OpId::kApplyPenalties, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ApplyPenaltiesFn>(&ApplyPenaltiesKernel)));
    RegisterOp(OpId::kApplyMinP, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ApplyMinPFn>(&ApplyMinPKernel)));
    RegisterOp(OpId::kApplyLogitBias, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ApplyLogitBiasFn>(&ApplyLogitBiasKernel)));
    RegisterOp(OpId::kApplyTokenMask, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ApplyTokenMaskFn>(&ApplyTokenMaskKernel)));
    RegisterOp(
        OpId::kApplyAllowedTokenIds, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<ApplyAllowedTokenIdsFn>(&ApplyAllowedTokenIdsKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
