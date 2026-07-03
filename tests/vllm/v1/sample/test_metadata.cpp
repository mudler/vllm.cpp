// Tests for SamplingMetadata + LogprobsTensors + InputBatch::make_sampling_metadata
// (M1.7 Task 1). Ported from vllm/v1/sample/metadata.py + vllm/v1/outputs.py +
// gpu_input_batch.py::_make_sampling_metadata @ e24d1b24.
//
// Oracle: the 2-req batch (one greedy temp=0, one random temp=0.8 top_p=0.9
// top_k=50) exercises the predicate + per-req field fill + None/[]-default gates
// exactly as upstream _make_sampling_metadata, and mirrors the intent of
// tests/v1/worker/test_gpu_input_batch.py::test_sampling_metadata_in_input_batch.
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/outputs.h"
#include "vllm/v1/sample/metadata.h"
#include "vllm/v1/worker/gpu/input_batch.h"

using vllm::SamplingParams;
using vllm::v1::CachedRequestState;
using vllm::v1::InputBatch;
using vllm::v1::LogprobsTensors;
using vllm::v1::SamplerOutput;
using vllm::v1::SamplingMetadata;

namespace {

InputBatch make_batch(int max_num_reqs = 8, int max_model_len = 64) {
  return InputBatch(/*max_num_reqs=*/max_num_reqs,
                    /*max_model_len=*/max_model_len,
                    /*max_num_batched_tokens=*/max_model_len,
                    /*vocab_size=*/1024, /*block_sizes=*/{16},
                    /*kernel_block_sizes=*/{16});
}

CachedRequestState make_req(const std::string& req_id,
                            std::vector<int32_t> prompt,
                            std::vector<int32_t> output, SamplingParams sp) {
  CachedRequestState state;
  state.req_id = req_id;
  state.prompt_token_ids = std::move(prompt);
  state.output_token_ids = std::move(output);
  state.sampling_params = sp;
  state.block_ids = {{0}};
  state.finalize();
  state.num_computed_tokens = state.num_prompt_tokens;
  return state;
}

}  // namespace

TEST_CASE("make_sampling_metadata: mixed greedy + random 2-req batch") {
  InputBatch batch = make_batch();

  SamplingParams greedy;
  greedy.temperature = 0.0;  // < _SAMPLING_EPS => greedy
  batch.add_request(make_req("r0", {10, 11, 12}, {}, greedy));

  SamplingParams random;
  random.temperature = 0.8;
  random.top_p = 0.9;
  random.top_k = 50;
  batch.add_request(make_req("r1", {20, 21}, {}, random));

  const SamplingMetadata md = batch.make_sampling_metadata();

  // Predicates: one greedy + one random => neither all_greedy nor all_random.
  CHECK_FALSE(md.all_greedy);
  CHECK_FALSE(md.all_random);
  CHECK(md.no_penalties);

  // temperature present (not all_greedy), sliced to num_reqs=2. Greedy row is
  // forced to 0.0 by add_request; random row carries 0.8.
  REQUIRE(md.temperature.has_value());
  REQUIRE(md.temperature->size() == 2);
  CHECK((*md.temperature)[0] == doctest::Approx(0.0f));
  CHECK((*md.temperature)[1] == doctest::Approx(0.8f));

  // top_p present because r1 has top_p < 1 (no_top_p == false).
  REQUIRE(md.top_p.has_value());
  REQUIRE(md.top_p->size() == 2);
  CHECK((*md.top_p)[0] == doctest::Approx(1.0f));  // r0 default
  CHECK((*md.top_p)[1] == doctest::Approx(0.9f));

  // top_k present because r1 has 0 < top_k < vocab (no_top_k == false).
  REQUIRE(md.top_k.has_value());
  REQUIRE(md.top_k->size() == 2);
  CHECK((*md.top_k)[0] == 1024);  // r0: clamped to vocab_size
  CHECK((*md.top_k)[1] == 50);

  // Penalties are always sliced to num_reqs; defaults (0/0/1) here.
  REQUIRE(md.repetition_penalties.size() == 2);
  CHECK(md.repetition_penalties[0] == doctest::Approx(1.0f));
  CHECK(md.frequency_penalties[1] == doctest::Approx(0.0f));

  // no_penalties => prompt_token_ids None, output_token_ids empty (upstream gate).
  CHECK_FALSE(md.prompt_token_ids.has_value());
  CHECK(md.output_token_ids.empty());

  // Not-yet-tracked fields keep faithful upstream defaults.
  CHECK(md.generators.empty());
  CHECK_FALSE(md.max_num_logprobs.has_value());
  CHECK_FALSE(md.allowed_token_ids_mask.has_value());
  CHECK(md.bad_words_token_ids.empty());
  CHECK(md.min_tokens.empty());
  CHECK(md.logit_bias.empty());
  CHECK(md.min_p.empty());

  // spec_token_ids: dense prefix of empty lists.
  REQUIRE(md.spec_token_ids.has_value());
  CHECK(md.spec_token_ids->size() == 2);
  CHECK((*md.spec_token_ids)[0].empty());
}

TEST_CASE("make_sampling_metadata: all-greedy => temperature None") {
  InputBatch batch = make_batch();
  SamplingParams greedy;
  greedy.temperature = 0.0;
  batch.add_request(make_req("g0", {1, 2}, {}, greedy));
  batch.add_request(make_req("g1", {3, 4}, {}, greedy));

  const SamplingMetadata md = batch.make_sampling_metadata();
  CHECK(md.all_greedy);
  CHECK_FALSE(md.all_random);
  CHECK_FALSE(md.temperature.has_value());  // upstream: None when all_greedy
  CHECK_FALSE(md.top_p.has_value());
  CHECK_FALSE(md.top_k.has_value());
}

TEST_CASE("make_sampling_metadata: penalties enable prompt+output token ids") {
  InputBatch batch = make_batch();
  SamplingParams sp;
  sp.temperature = 1.0;
  sp.frequency_penalty = 0.5;  // triggers no_penalties == false
  batch.add_request(make_req("p0", {7, 8, 9}, {100, 101}, sp));

  const SamplingMetadata md = batch.make_sampling_metadata();
  CHECK_FALSE(md.no_penalties);

  // prompt_token_ids populated, sliced to num_prompt_tokens per row.
  REQUIRE(md.prompt_token_ids.has_value());
  REQUIRE(md.prompt_token_ids->size() == 1);
  CHECK((*md.prompt_token_ids)[0] == std::vector<int32_t>{7, 8, 9});

  // output_token_ids populated from req_output_token_ids.
  REQUIRE(md.output_token_ids.size() == 1);
  CHECK(md.output_token_ids[0] == std::vector<int32_t>{100, 101});

  CHECK(md.frequency_penalties[0] == doctest::Approx(0.5f));
}

TEST_CASE("make_sampling_metadata: reads dense prefix after remove+condense") {
  InputBatch batch = make_batch();
  SamplingParams greedy;
  greedy.temperature = 0.0;
  SamplingParams random;
  random.temperature = 0.7;

  batch.add_request(make_req("a", {1}, {}, greedy));   // slot 0
  batch.add_request(make_req("b", {2}, {}, random));   // slot 1
  batch.add_request(make_req("c", {3}, {}, greedy));   // slot 2

  // Remove the middle request; condense slides "c" into slot 1.
  batch.remove_request("b");
  batch.condense();
  REQUIRE(batch.num_reqs() == 2);

  const SamplingMetadata md = batch.make_sampling_metadata();
  // After condense only greedy "a","c" remain => all_greedy.
  CHECK(md.all_greedy);
  CHECK_FALSE(md.temperature.has_value());
  // Dense slices are length 2 (not the max_num_reqs backing array).
  CHECK(md.repetition_penalties.size() == 2);
}

TEST_CASE("LogprobsTensors::empty_cpu has the right shapes") {
  // n positions, k+1 tokens per position.
  const LogprobsTensors lt = LogprobsTensors::empty_cpu(/*num_positions=*/3,
                                                        /*num_tokens=*/5);
  CHECK(lt.num_positions == 3);
  CHECK(lt.num_tokens_per_position == 5);
  CHECK(lt.logprob_token_ids.size() == 15);  // 3 * 5
  CHECK(lt.logprobs.size() == 15);
  CHECK(lt.selected_token_ranks.size() == 3);
}

TEST_CASE("SamplerOutput now carries an optional LogprobsTensors payload") {
  SamplerOutput out;
  out.sampled_token_ids = {{42}, {7}};
  CHECK_FALSE(out.logprobs_tensors.has_value());

  LogprobsTensors lt = LogprobsTensors::empty_cpu(2, 3);
  lt.logprob_token_ids[0] = 42;
  lt.selected_token_ranks[0] = 0;
  out.logprobs_tensors = std::move(lt);

  REQUIRE(out.logprobs_tensors.has_value());
  CHECK(out.logprobs_tensors->num_positions == 2);
  CHECK(out.logprobs_tensors->num_tokens_per_position == 3);
  CHECK(out.logprobs_tensors->logprob_token_ids[0] == 42);
}
