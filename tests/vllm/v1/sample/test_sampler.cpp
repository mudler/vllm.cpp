// Ported from: vllm/v1/sample/sampler.py @ e24d1b24 (Sampler.forward / sample /
// gather_logprobs) + cases mirrored from tests/v1/sample/test_sampler.py
// (test_sampler_repetition_penalty, test_sampler_min_tokens, the greedy/random
// merge). The Sampler assembles the ordered pipeline over the [num_reqs, vocab]
// f32 logits (raw-logprobs snapshot BEFORE mutation -> allowed-ids -> bad-words
// -> non-argmax-invariant procs -> penalties -> sample{greedy snapshot;
// temperature; argmax-invariant min_p; top_k_top_p; random; where(temp<eps)}
// -> gather logprobs). Greedy is the bit-exact parity gate; random is
// distribution-correct (peaked distributions make the picked token deterministic
// for the assertions here). CPU is the correctness gate; a CUDA-Queue run is
// dgx-pending (the Sampler runs on whatever Queue the ops are registered for).
#include "vllm/v1/sample/sampler.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "vllm/v1/sample/metadata.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::v1::MinTokensState;
using vllm::v1::Sampler;
using vllm::v1::SamplingMetadata;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Logits(std::vector<float>& v, int64_t n, int64_t vocab) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {n, vocab});
}
}  // namespace

// ---------------------------------------------------------------------------
// All-greedy batch: argmax, bit-exact; no logprobs when not requested.
TEST_CASE("Sampler: all-greedy batch returns the argmax per row, no logprobs") {
  std::vector<float> logits = {0.1f, 5.0f, 0.2f, 0.3f,   // argmax 1
                               9.0f, 1.0f, 1.0f, 2.0f};  // argmax 0
  Tensor tl = Logits(logits, 2, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.sampled_token_ids.size() == 2);
  REQUIRE(out.sampled_token_ids[0].size() == 1);
  CHECK(out.sampled_token_ids[0][0] == 1);
  CHECK(out.sampled_token_ids[1][0] == 0);
  CHECK_FALSE(out.logprobs_tensors.has_value());
}

// ---------------------------------------------------------------------------
// Mixed greedy + random: the temp<eps where-merge picks greedy for the greedy
// row and random for the random row. Row 1 is a peaked distribution so the
// random draw is deterministic for the assertion.
TEST_CASE("Sampler: mixed batch merges greedy (temp<eps) and random per row") {
  // Row 0 (greedy, temp 0): argmax at index 3.
  // Row 1 (random, temp 1): logit 100 at index 2 -> softmax ~= one-hot(2).
  std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f,
                               0.0f, 0.0f, 100.0f, 0.0f};
  Tensor tl = Logits(logits, 2, 4);
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = false;
  sm.temperature = std::vector<float>{0.0f, 1.0f};
  sm.generators[1] = 20260704;  // per-request seed for the random row
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  CHECK(out.sampled_token_ids[0][0] == 3);  // greedy row -> argmax
  CHECK(out.sampled_token_ids[1][0] == 2);  // random row -> the dominant token
}

// ---------------------------------------------------------------------------
// Penalties change the argmax: a repetition penalty on the leading token pushes
// it below the runner-up so greedy flips. (test_sampler_repetition_penalty.)
TEST_CASE("Sampler: repetition penalty flips the greedy argmax") {
  // logits [1.0, 0.9]; token 0 is in the output -> rep 2.0 divides it (0.5),
  // now below token 1 (0.9) -> argmax flips 0 -> 1.
  std::vector<float> logits = {1.0f, 0.9f};
  Tensor tl = Logits(logits, 1, 2);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.no_penalties = false;
  sm.prompt_token_ids = std::vector<std::vector<int32_t>>{{}};
  sm.output_token_ids = {{0}};
  sm.presence_penalties = {0.0f};
  sm.frequency_penalties = {0.0f};
  sm.repetition_penalties = {2.0f};
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == 1);
}

// ---------------------------------------------------------------------------
// min_tokens masks eos before the floor so it is never sampled.
TEST_CASE("Sampler: min_tokens masks eos below the floor") {
  // eos (token 0) has the highest logit but is masked while output_len < floor.
  std::vector<float> logits = {5.0f, 1.0f, 2.0f};
  Tensor tl = Logits(logits, 1, 3);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  MinTokensState st;
  st.min_tokens = 5;
  st.stop_token_ids = {0};
  sm.min_tokens[0] = st;
  sm.output_token_ids = {{}};  // length 0 < 5 -> eos masked
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == 2);  // token 0 masked, next-highest is 2
}

// ---------------------------------------------------------------------------
// top_k restricts the support: with k=1 only the argmax survives, so even the
// random draw must return it.
TEST_CASE("Sampler: top_k=1 restricts random support to the argmax") {
  std::vector<float> logits = {0.0f, 1.0f, 3.0f, 2.0f};  // argmax index 2
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = true;
  sm.temperature = std::vector<float>{1.0f};
  sm.top_k = std::vector<int32_t>{1};
  sm.generators[0] = 7;
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == 2);
}

// ---------------------------------------------------------------------------
// Logprobs: top-k + sampled token logprob + ranks; the [n, k+1] concat shape.
// rank of the max-logprob token is 1 (batched_count_greater_than uses >=).
TEST_CASE("Sampler: logprobs gather (top-k, sampled col, ranks) all-greedy") {
  // logits [3,1,2,0] -> argmax token 0 (also the max logprob). num_logprobs=2.
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = 2;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.sampled_token_ids[0][0] == 0);
  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  CHECK(lt.num_positions == 1);
  CHECK(lt.num_tokens_per_position == 3);  // k + 1
  REQUIRE(lt.logprob_token_ids.size() == 3);
  // Column 0 is the sampled token; then the top-2 (token 0, token 2).
  CHECK(lt.logprob_token_ids[0] == 0);
  CHECK(lt.logprob_token_ids[1] == 0);
  CHECK(lt.logprob_token_ids[2] == 2);
  // log_softmax reference for [3,1,2,0].
  const float lse = 3.0f + std::log(std::exp(0.0f) + std::exp(-2.0f) +
                                    std::exp(-1.0f) + std::exp(-3.0f));
  CHECK(lt.logprobs[0] == doctest::Approx(3.0f - lse));  // sampled (token 0)
  CHECK(lt.logprobs[1] == doctest::Approx(3.0f - lse));  // top-1 (token 0)
  CHECK(lt.logprobs[2] == doctest::Approx(2.0f - lse));  // top-2 (token 2)
  // Sampled token is the max logprob -> rank 1.
  CHECK(lt.selected_token_ranks[0] == 1);
}

// ---------------------------------------------------------------------------
// Ranks are computed over the RAW (pre-penalty) logprobs snapshot, so when the
// penalty flips the sampled token off the raw max the rank is > 1.
TEST_CASE("Sampler: sampled-token rank over raw logprobs when penalty flips it") {
  // Raw logits [1.0, 0.9]: raw max is token 0. Rep penalty flips greedy to
  // token 1. rank(token1) = #{raw lp >= lp[1]} = 2 (token 0 and token 1).
  std::vector<float> logits = {1.0f, 0.9f};
  Tensor tl = Logits(logits, 1, 2);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.no_penalties = false;
  sm.prompt_token_ids = std::vector<std::vector<int32_t>>{{}};
  sm.output_token_ids = {{0}};
  sm.presence_penalties = {0.0f};
  sm.frequency_penalties = {0.0f};
  sm.repetition_penalties = {2.0f};
  sm.max_num_logprobs = 1;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.sampled_token_ids[0][0] == 1);
  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  CHECK(lt.num_tokens_per_position == 2);  // k + 1
  CHECK(lt.logprob_token_ids[0] == 1);     // sampled token in col 0
  CHECK(lt.logprob_token_ids[1] == 0);     // top-1 raw token is token 0
  CHECK(lt.selected_token_ranks[0] == 2);  // token 1 is the 2nd-highest raw lp
}

// ---------------------------------------------------------------------------
// Full ordered pipeline on a 2-req batch: req 0 greedy, req 1 random with a
// peaked distribution + logprobs, exercising the whole assembly at once.
TEST_CASE("Sampler: full pipeline on a 2-req batch (greedy + random) with logprobs") {
  std::vector<float> logits = {1.0f, 4.0f, 2.0f, 0.0f,     // greedy -> 1
                               0.0f, 0.0f, 0.0f, 50.0f};   // random -> 3 (peaked)
  Tensor tl = Logits(logits, 2, 4);
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = false;
  sm.temperature = std::vector<float>{0.0f, 1.0f};
  sm.generators[1] = 4242;
  sm.max_num_logprobs = 1;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  CHECK(out.sampled_token_ids[0][0] == 1);
  CHECK(out.sampled_token_ids[1][0] == 3);
  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  CHECK(lt.num_positions == 2);
  CHECK(lt.num_tokens_per_position == 2);
  // Col 0 of each row is the sampled token.
  CHECK(lt.logprob_token_ids[0] == 1);
  CHECK(lt.logprob_token_ids[2] == 3);
}
