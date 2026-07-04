// Ported from: vllm/v1/sample/sampler.py @ e24d1b24. See sampler.h for the full
// 9-step order + the deferred stubs. This file assembles the ordered pipeline by
// composing the Task 2/3 vt ops over the [num_reqs, vocab] f32 logits.
#include "vllm/v1/sample/sampler.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <optional>

#include "vllm/v1/sample/device_scratch.h"
#include "vllm/v1/sample/logits_processor/builtin.h"
#include "vllm/v1/sample/ops/bad_words.h"
#include "vllm/v1/sample/ops/penalties.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm::v1 {

namespace {

// Batch-default RNG seed for rows without a per-request generator override.
// Upstream uses a shared default torch.Generator; our RandomSample hashes
// (seed, row, vocab_index) so a constant default still gives independent per-row
// draws (the row index enters the hash). Exact torch-Philox parity is the
// documented M1.7 T1 carry.
constexpr uint64_t kDefaultSeed = 0;

// Owns a device-side output buffer of the given shape and downloads it to host.
// Mirrors the DeviceScratch materialization (unified vs discrete) but for OUTPUT
// tensors the vt ops write into: on unified backends Alloc is host-addressable
// and Copy is memcpy, so download() works uniformly on CPU and CUDA.
class DeviceBuffer {
 public:
  DeviceBuffer(vt::Device device, vt::Queue& q, vt::DType dtype,
               std::initializer_list<int64_t> shape)
      : backend_(&vt::GetBackend(device.type)), q_(q) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dtype);
    owned_ = backend_->Alloc(bytes_ == 0 ? 1 : bytes_);
    tensor_ = vt::Tensor::Contiguous(owned_, dtype, device, shape);
  }
  ~DeviceBuffer() { backend_->Free(owned_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  vt::Tensor& tensor() { return tensor_; }
  void download(void* dst) {
    if (bytes_ != 0) backend_->Copy(q_, dst, owned_, bytes_);
    backend_->Synchronize(q_);
  }

 private:
  vt::Backend* backend_ = nullptr;
  vt::Queue& q_;
  void* owned_ = nullptr;
  size_t bytes_ = 0;
  vt::Tensor tensor_;
};

// Materialize the per-req top_k / top_p optionals (SamplingMetadata) into the
// [num_reqs] tensors ApplyTopKTopP consumes; pass nullptr when a predicate says
// none. The DeviceScratch objects are locals whose lifetime spans the op call.
void ApplyTopKTopPFromMeta(vt::Queue& q, vt::Tensor& logits,
                           const std::optional<std::vector<int32_t>>& top_k,
                           const std::optional<std::vector<float>>& top_p) {
  const int64_t n = logits.shape[0];
  if (top_k.has_value())
    VT_CHECK(static_cast<int64_t>(top_k->size()) == n,
             "sampler: top_k must have num_reqs rows");
  if (top_p.has_value())
    VT_CHECK(static_cast<int64_t>(top_p->size()) == n,
             "sampler: top_p must have num_reqs rows");

  if (top_k.has_value() && top_p.has_value()) {
    DeviceScratch k(logits.device, q, top_k->data(), vt::DType::kI32, {n});
    DeviceScratch p(logits.device, q, top_p->data(), vt::DType::kF32, {n});
    vt::ApplyTopKTopP(q, logits, &k.tensor(), &p.tensor());
  } else if (top_k.has_value()) {
    DeviceScratch k(logits.device, q, top_k->data(), vt::DType::kI32, {n});
    vt::ApplyTopKTopP(q, logits, &k.tensor(), nullptr);
  } else if (top_p.has_value()) {
    DeviceScratch p(logits.device, q, top_p->data(), vt::DType::kF32, {n});
    vt::ApplyTopKTopP(q, logits, nullptr, &p.tensor());
  }
}

// GreedyArgmax over `logits`, returned as a [num_reqs] host vector (upstream's
// argmax returns int64; the lowest-index tie-break is bit-exact vs torch).
std::vector<int64_t> GreedyArgmaxHost(vt::Queue& q, const vt::Tensor& logits,
                                      int64_t n) {
  DeviceBuffer ids(logits.device, q, vt::DType::kI64, {n});
  vt::GreedyArgmax(q, ids.tensor(), logits);
  std::vector<int64_t> host(static_cast<size_t>(n));
  ids.download(host.data());
  return host;
}

// gather_logprobs (sampler.py::Sampler.gather_logprobs). Over the raw (pre-
// mutation) logprobs snapshot: top-k values+indices, the sampled token's
// logprob, ranks via batched_count_greater_than, concat [sampled | topk].
// batched_count_greater_than (ops/logprobs.py) is `(x >= values).sum(-1)`, so
// the rank COUNTS the sampled token itself (>=) — it is 1-BASED and the
// max-logprob token has rank 1.
LogprobsTensors GatherLogprobs(const std::vector<float>& raw_logprobs, int64_t n,
                               int64_t vocab, int num_logprobs,
                               const std::vector<int64_t>& sampled) {
  const int k = num_logprobs;
  const int width = k + 1;
  LogprobsTensors lt;
  lt.num_positions = static_cast<int>(n);
  lt.num_tokens_per_position = width;
  lt.logprob_token_ids.resize(static_cast<size_t>(n) * static_cast<size_t>(width));
  lt.logprobs.resize(static_cast<size_t>(n) * static_cast<size_t>(width));
  lt.selected_token_ranks.resize(static_cast<size_t>(n));

  std::vector<int32_t> idx(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < n; ++i) {
    const float* row = &raw_logprobs[static_cast<size_t>(i * vocab)];
    const int32_t tok = static_cast<int32_t>(sampled[static_cast<size_t>(i)]);
    const float tok_lp = row[tok];

    // Rank: count of logprobs >= the sampled token's logprob (1-based).
    int32_t rank = 0;
    for (int64_t j = 0; j < vocab; ++j)
      if (row[j] >= tok_lp) ++rank;
    lt.selected_token_ranks[static_cast<size_t>(i)] = rank;

    // Top-k by value desc, lowest index on ties (torch.topk sorted order).
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [row](int32_t a, int32_t b) {
                        if (row[a] != row[b]) return row[a] > row[b];
                        return a < b;
                      });

    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(width);
    lt.logprob_token_ids[base] = tok;  // column 0: the sampled token
    lt.logprobs[base] = tok_lp;
    for (int c = 0; c < k; ++c) {
      lt.logprob_token_ids[base + 1 + static_cast<size_t>(c)] = idx[static_cast<size_t>(c)];
      lt.logprobs[base + 1 + static_cast<size_t>(c)] = row[idx[static_cast<size_t>(c)]];
    }
  }
  return lt;
}

}  // namespace

std::vector<int64_t> Sampler::sample(vt::Queue& q, vt::Tensor& logits,
                                     const SamplingMetadata& sm) const {
  const int64_t n = logits.shape[0];
  const int64_t vocab = logits.shape[1];

  VT_CHECK(!(sm.all_greedy && sm.all_random),
           "sampler: all_greedy and all_random are mutually exclusive");

  // 7a. Greedy snapshot (unless all_random). For an all-greedy batch, return it.
  std::vector<int64_t> greedy_sampled;
  const bool have_greedy = !sm.all_random;
  if (have_greedy) {
    greedy_sampled = GreedyArgmaxHost(q, logits, n);
    if (sm.all_greedy) return greedy_sampled;
  }

  VT_CHECK(sm.temperature.has_value(),
           "sampler: temperature is required unless all_greedy");
  const std::vector<float>& temp = *sm.temperature;
  VT_CHECK(static_cast<int64_t>(temp.size()) == n,
           "sampler: temperature must have num_reqs rows");

  // 7b. Temperature (temp<eps->1.0 guard applied inside when !all_random).
  {
    DeviceScratch t(logits.device, q, temp.data(), vt::DType::kF32, {n});
    vt::ApplyTemperature(q, logits, t.tensor(), sm.all_random);
  }

  // 7c. Argmax-invariant procs: min_p (kMinPArgmaxInvariant == true).
  if (!sm.min_p.empty()) apply_min_p(q, logits, sm.min_p);

  // 7d. top_k and/or top_p (materialize the per-req optionals; nullptr => skip).
  ApplyTopKTopPFromMeta(q, logits, sm.top_k, sm.top_p);

  // 7e. probs = softmax(logits); random_sample (exponential-noise gumbel-max).
  DeviceBuffer probs(logits.device, q, vt::DType::kF32, {n, vocab});
  vt::ComputeProbs(q, probs.tensor(), logits);

  std::vector<int64_t> seeds(static_cast<size_t>(n), static_cast<int64_t>(kDefaultSeed));
  for (const auto& [i, seed] : sm.generators) {
    VT_CHECK(i >= 0 && static_cast<int64_t>(i) < n,
             "sampler: generator request index out of range");
    seeds[static_cast<size_t>(i)] = static_cast<int64_t>(seed);
  }
  std::vector<int64_t> random_sampled(static_cast<size_t>(n));
  {
    DeviceScratch s(logits.device, q, seeds.data(), vt::DType::kI64, {n});
    DeviceBuffer rs(logits.device, q, vt::DType::kI64, {n});
    vt::RandomSample(q, rs.tensor(), probs.tensor(), s.tensor());
    rs.download(random_sampled.data());
  }

  // 7f. Mixed batch: where(temp < eps, greedy, random) per row.
  if (!have_greedy) return random_sampled;
  std::vector<int64_t> sampled(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    sampled[static_cast<size_t>(i)] = (temp[static_cast<size_t>(i)] < vt::kSamplingEps)
                                          ? greedy_sampled[static_cast<size_t>(i)]
                                          : random_sampled[static_cast<size_t>(i)];
  }
  return sampled;
}

SamplerOutput Sampler::forward(vt::Queue& q, vt::Tensor& logits,
                               const SamplingMetadata& sm) const {
  VT_CHECK(logits.rank == 2, "sampler: logits must be [num_reqs, vocab]");
  VT_CHECK(logits.dtype == vt::DType::kF32, "sampler: logits must be f32");
  const int64_t n = logits.shape[0];
  const int64_t vocab = logits.shape[1];

  // 1. Raw-logprobs snapshot BEFORE any mutation (raw_logprobs mode; raw_logits
  //    and the processed_* modes are deferred stubs — see the header).
  //    NOTE: logprob_token_ids (generative-scoring) is a deferred stub, so the
  //    snapshot is driven solely by max_num_logprobs.
  const std::optional<int> num_logprobs = sm.max_num_logprobs;
  const bool want_logprobs = num_logprobs.has_value();
  std::vector<float> raw_logprobs;  // host [n*vocab] when want_logprobs
  if (want_logprobs) {
    VT_CHECK(logprobs_mode_ == LogprobsMode::kRawLogprobs,
             "sampler: only the raw_logprobs logprobs_mode is implemented at T0");
    DeviceBuffer rlp(logits.device, q, vt::DType::kF32, {n, vocab});
    vt::ComputeLogprobs(q, rlp.tensor(), logits);
    raw_logprobs.resize(static_cast<size_t>(n) * static_cast<size_t>(vocab));
    rlp.download(raw_logprobs.data());
  }

  // 2. float32 (already f32; checked above).
  // 3-6. Logits processors, in upstream order (apply_logits_processors):
  //   allowed_token_ids -> bad_words -> non-argmax-invariant (min_tokens,
  //   logit_bias) -> penalties.
  if (sm.allowed_token_ids_mask.has_value()) {
    apply_allowed_token_ids(q, logits, *sm.allowed_token_ids_mask);
  }
  if (!sm.bad_words_token_ids.empty()) {
    apply_bad_words(q, logits, sm.bad_words_token_ids, sm.output_token_ids);
  }
  // Non-argmax-invariant procs (kMinTokensArgmaxInvariant / kLogitBiasArgmax-
  // Invariant == false): min_tokens then logit_bias.
  apply_min_tokens(q, logits, sm.min_tokens, sm.output_token_ids);
  apply_logit_bias(q, logits, sm.logit_bias);
  // Penalties (repetition, frequency, presence).
  if (!sm.no_penalties) {
    VT_CHECK(sm.prompt_token_ids.has_value(),
             "sampler: prompt_token_ids required when penalties are active");
    apply_all_penalties(q, logits, *sm.prompt_token_ids, sm.presence_penalties,
                        sm.frequency_penalties, sm.repetition_penalties,
                        sm.output_token_ids);
  }

  // 7. Sample.
  const std::vector<int64_t> sampled = sample(q, logits, sm);

  // 8. Gather logprobs.
  std::optional<LogprobsTensors> logprobs_tensors;
  if (want_logprobs) {
    if (*num_logprobs == -1) {
      // Full unsorted/unranked raw logprobs (upstream LogprobsTensors(empty,
      // raw_logprobs, empty)). Minimal 1:1 form: the [n, vocab] raw logprobs
      // with empty ids/ranks. The OutputProcessor slicing is M1.8.
      LogprobsTensors lt;
      lt.num_positions = static_cast<int>(n);
      lt.num_tokens_per_position = static_cast<int>(vocab);
      lt.logprobs = raw_logprobs;
      logprobs_tensors = std::move(lt);
    } else {
      logprobs_tensors = GatherLogprobs(raw_logprobs, n, vocab, *num_logprobs, sampled);
    }
  }

  // 9. SamplerOutput: sampled_token_ids [num_reqs, 1] (list-of-lists form).
  SamplerOutput out;
  out.sampled_token_ids.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    out.sampled_token_ids[static_cast<size_t>(i)] = {
        static_cast<int32_t>(sampled[static_cast<size_t>(i)])};
  }
  out.logprobs_tensors = std::move(logprobs_tensors);
  return out;
}

}  // namespace vllm::v1
