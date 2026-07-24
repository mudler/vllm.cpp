// Ported from (test-porting protocol .agents/test-porting.md):
//   * tests/v1/sample/test_rejection_sampler.py @ e24d1b24 —
//       test_perfect_match          :133
//       test_early_mismatch         :154
//       test_multiple_sequences     :179
//       test_single_token_sequence  :204
//       test_empty_sequence         :225
//       test_multiple_mismatches    :246
//       test_parametrized_cases     :288
//     (the legacy-sampler suite; its ASSERTIONS are realized here against our
//     MRV2-shaped RejectionSampler — test-porting rule 3.)
//   * tests/v1/spec_decode/test_rejection_sampler_utils.py @ e24d1b24 —
//       test_greedy_rejection_sample        :183 (k in {1, 3})
//       test_placeholder_draft_token_rejected :285
//     (the stochastic :141, synthetic :215 and block-verification :325/:372
//      cases are SKIPPED until M-mtp-3 — see the SKIPPED note at the bottom.)
//
// The upstream cases construct logits whose argmax at expanded row j is
// output_tokens[r][j], schedule spec_tokens[r] as the drafts, and assert the
// emitted token stream. Upstream's fixed-width [num_reqs, k+1] output pads
// rejected positions with PLACEHOLDER_TOKEN_ID (-1,
// vllm/v1/sample/rejection_sampler.py:30); our RejectionSamplerOutput returns
// ragged per-request vectors of exactly num_sampled tokens, so a padded
// upstream row [a, b, -1, -1] becomes {a, b} plus num_sampled == 2. Both forms
// are asserted.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/v1/spec_decode/rejection_sampler.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using vllm::v1::RejectionSampler;
using vllm::v1::RejectionSamplerOutput;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

constexpr int kVocab = 16;

// One verify-step fixture, built exactly the way upstream's
// `create_logits_tensor` + `create_spec_decode_metadata` do:
//   * `target_tokens[r]` are the tokens the TARGET argmaxes at request r's
//     expanded rows (length 1 + k_r);
//   * `spec_tokens[r]` are the k_r draft tokens the scheduler proposed.
// The draft_sampled array mirrors `input_ids[logits_indices]`: row cu[r] holds
// the previous step's token (never compared — upstream reads draft_sampled at
// logit_idx + 1, rejection_sampler_utils.py:534) and rows cu[r]+1.. hold the
// drafts.
struct VerifyStep {
  std::vector<float> logits;
  std::vector<int32_t> draft_sampled;
  std::vector<int32_t> cu_num_logits;
  int64_t num_logits = 0;
};

VerifyStep MakeStep(const std::vector<std::vector<int32_t>>& spec_tokens,
                    const std::vector<std::vector<int32_t>>& target_tokens) {
  REQUIRE(spec_tokens.size() == target_tokens.size());
  VerifyStep s;
  s.cu_num_logits.push_back(0);
  int32_t total = 0;
  for (size_t r = 0; r < spec_tokens.size(); ++r) {
    REQUIRE(target_tokens[r].size() == spec_tokens[r].size() + 1);
    total += static_cast<int32_t>(target_tokens[r].size());
    s.cu_num_logits.push_back(total);
  }
  s.num_logits = total;
  s.logits.assign(static_cast<size_t>(total) * kVocab, 0.0f);
  s.draft_sampled.assign(static_cast<size_t>(total), 0);
  int32_t row = 0;
  for (size_t r = 0; r < spec_tokens.size(); ++r) {
    // Row cu[r] input id: an arbitrary previously-sampled token (never read).
    s.draft_sampled[static_cast<size_t>(row)] = 0;
    for (size_t j = 0; j < target_tokens[r].size(); ++j) {
      // Make target_tokens[r][j] the strict argmax of expanded row cu[r]+j.
      s.logits[static_cast<size_t>(row + static_cast<int32_t>(j)) * kVocab +
               static_cast<size_t>(target_tokens[r][j])] = 10.0f;
      if (j < spec_tokens[r].size()) {
        s.draft_sampled[static_cast<size_t>(row) + j + 1] = spec_tokens[r][j];
      }
    }
    row += static_cast<int32_t>(target_tokens[r].size());
  }
  return s;
}

RejectionSamplerOutput Run(const std::vector<std::vector<int32_t>>& spec_tokens,
                           const std::vector<std::vector<int32_t>>& target_tokens,
                           int num_speculative_steps) {
  VerifyStep s = MakeStep(spec_tokens, target_tokens);
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(s.logits.data(), DType::kF32, Cpu(),
                                     {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(num_speculative_steps);
  return sampler.forward(q, logits, s.draft_sampled, s.cu_num_logits);
}

// The upstream fixed-width form: pad each request's emitted tokens to
// max_len with PLACEHOLDER_TOKEN_ID (-1).
std::vector<std::vector<int32_t>> Padded(const RejectionSamplerOutput& out, size_t width) {
  std::vector<std::vector<int32_t>> rows;
  for (const auto& toks : out.sampled_token_ids) {
    std::vector<int32_t> row = toks;
    row.resize(width, -1);
    rows.push_back(row);
  }
  return rows;
}

}  // namespace

// ---------------------------------------------------------------------------
// test_perfect_match (test_rejection_sampler.py:133): every draft matches, so
// the emitted stream is the drafts plus the bonus token.
TEST_CASE("rejection_sampler: perfect_match emits every draft plus the bonus token") {
  const RejectionSamplerOutput out = Run({{1, 2, 3}}, {{1, 2, 3, 4}}, /*k=*/3);
  CHECK(out.sampled_token_ids.size() == 1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 3, 4});
  CHECK(out.num_sampled[0] == 4);
  CHECK(out.num_rejected[0] == 0);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 2, 3, 4});
}

// test_early_mismatch (:154): mismatch at position 1 -> emit the target argmax
// there and STOP; nothing after it is accepted.
TEST_CASE("rejection_sampler: early_mismatch emits the target argmax and stops") {
  const RejectionSamplerOutput out = Run({{1, 2, 3}}, {{1, 5, 3, 4}}, /*k=*/3);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 5});
  CHECK(out.num_sampled[0] == 2);
  // 3 drafts scheduled, 1 accepted -> 2 rejected (the scheduler rolls back by 2).
  CHECK(out.num_rejected[0] == 2);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 5, -1, -1});
}

// test_multiple_sequences (:179): two requests with DIFFERENT k_i (2 and 1).
TEST_CASE("rejection_sampler: multiple_sequences with different per-request k") {
  const RejectionSamplerOutput out = Run({{1, 2}, {3}}, {{1, 2, 5}, {3, 4}}, /*k=*/2);
  CHECK(out.sampled_token_ids.size() == 2);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 5});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{3, 4});
  CHECK(out.num_sampled[0] == 3);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[0] == 0);
  CHECK(out.num_rejected[1] == 0);
  CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 2, 5});
  CHECK(Padded(out, 3)[1] == std::vector<int32_t>{3, 4, -1});
}

// test_single_token_sequence (:204): k == 1, accepted.
TEST_CASE("rejection_sampler: single_token_sequence (k=1) accepts and emits the bonus") {
  const RejectionSamplerOutput out = Run({{1}}, {{1, 2}}, /*k=*/1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2});
  CHECK(out.num_sampled[0] == 2);
  CHECK(out.num_rejected[0] == 0);
}

// test_empty_sequence (:225): NO drafts — the k == 0 reduction. This is the
// byte-identity anchor: the emitted token is exactly the plain greedy argmax.
TEST_CASE("rejection_sampler: empty_sequence (k=0) reduces to the plain greedy argmax") {
  const RejectionSamplerOutput out = Run({{}}, {{5}}, /*k=*/1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{5});
  CHECK(out.num_sampled[0] == 1);
  CHECK(out.num_rejected[0] == 0);
}

// test_multiple_mismatches (:246): both requests reject, at different positions.
TEST_CASE("rejection_sampler: multiple_mismatches reject independently per request") {
  const RejectionSamplerOutput out =
      Run({{1, 2, 3}, {4, 5, 6}}, {{1, 2, 7, 6}, {4, 8, 6, 9}}, /*k=*/3);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 7});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{4, 8});
  CHECK(out.num_sampled[0] == 3);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[0] == 1);
  CHECK(out.num_rejected[1] == 2);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 2, 7, -1});
  CHECK(Padded(out, 4)[1] == std::vector<int32_t>{4, 8, -1, -1});
}

// test_parametrized_cases (:288): the three upstream parametrizations.
TEST_CASE("rejection_sampler: parametrized cases (perfect / first-mismatch / mixed)") {
  SUBCASE("perfect match with bonus") {
    const RejectionSamplerOutput out = Run({{1, 2}}, {{1, 2, 3}}, /*k=*/2);
    CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 2, 3});
  }
  SUBCASE("first mismatch") {
    const RejectionSamplerOutput out = Run({{1}}, {{2, 3}}, /*k=*/1);
    CHECK(Padded(out, 2)[0] == std::vector<int32_t>{2, -1});
    CHECK(out.num_sampled[0] == 1);
    CHECK(out.num_rejected[0] == 1);
  }
  SUBCASE("mixed matches") {
    const RejectionSamplerOutput out = Run({{1, 2}, {3, 4}}, {{1, 5, 6}, {3, 4, 7}}, /*k=*/2);
    CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 5, -1});
    CHECK(Padded(out, 3)[1] == std::vector<int32_t>{3, 4, 7});
  }
}

// ---------------------------------------------------------------------------
// test_greedy_rejection_sample (test_rejection_sampler_utils.py:183, k in {1,3}):
// "Verify that greedy (temperature=0) always outputs the target argmax at every
// accepted position." Upstream drives one shared target distribution across many
// trials; we drive the same invariant over an exhaustive draft-pattern sweep,
// which additionally pins WHERE the run stops.
TEST_CASE("rejection_sampler: greedy_rejection_sample — every emitted token is the target argmax") {
  for (int k : {1, 3}) {
    // A fixed target argmax sequence; sweep every subset of matching drafts.
    const std::vector<int32_t> target_seq = {3, 7, 2, 9};  // length 4 >= k+1
    std::vector<int32_t> target(target_seq.begin(), target_seq.begin() + k + 1);
    const int num_patterns = 1 << k;
    for (int pattern = 0; pattern < num_patterns; ++pattern) {
      std::vector<int32_t> drafts(static_cast<size_t>(k));
      for (int i = 0; i < k; ++i) {
        // bit set => the draft matches the target argmax at position i.
        drafts[static_cast<size_t>(i)] =
            (pattern >> i) & 1 ? target[static_cast<size_t>(i)]
                               : (target[static_cast<size_t>(i)] + 1) % kVocab;
      }
      const RejectionSamplerOutput out = Run({drafts}, {target}, k);
      // Expected accepted length = the number of leading matching drafts.
      int expect_len = 0;
      while (expect_len < k && drafts[static_cast<size_t>(expect_len)] ==
                                   target[static_cast<size_t>(expect_len)]) {
        ++expect_len;
      }
      CAPTURE(k);
      CAPTURE(pattern);
      CHECK(out.num_sampled[0] == expect_len + 1);
      CHECK(out.num_rejected[0] == k - expect_len);
      REQUIRE(out.sampled_token_ids[0].size() == static_cast<size_t>(expect_len) + 1);
      // THE INVARIANT: every emitted token equals the target argmax at its row.
      for (int j = 0; j <= expect_len; ++j) {
        CHECK(out.sampled_token_ids[0][static_cast<size_t>(j)] ==
              target[static_cast<size_t>(j)]);
      }
    }
  }
}

// test_placeholder_draft_token_rejected (:285): a -1 placeholder draft id must
// be rejected without any out-of-bounds logits read.
TEST_CASE("rejection_sampler: placeholder draft token (-1) is rejected") {
  const RejectionSamplerOutput out = Run({{-1}}, {{5, 6}}, /*k=*/1);
  CHECK(out.num_sampled[0] == 1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{5});
  CHECK(out.num_rejected[0] == 1);
}

// A chunked-prefilling row samples and rejects nothing
// (_get_num_sampled_and_rejected_kernel, gpu/input_batch.py:421-433).
TEST_CASE("rejection_sampler: a chunked-prefilling row reports 0 sampled and 0 rejected") {
  VerifyStep s = MakeStep({{1, 2}, {3}}, {{1, 9, 5}, {3, 4}});
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(s.logits.data(), DType::kF32, Cpu(),
                                     {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(2);
  const RejectionSamplerOutput out =
      sampler.forward(q, logits, s.draft_sampled, s.cu_num_logits, {1, 0});
  CHECK(out.num_sampled[0] == 0);
  CHECK(out.num_rejected[0] == 0);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[1] == 0);
}

// SKIPPED (test-porting rule 6), tracked to M-mtp-3 (spec §5):
//   * test_stochastic_rejection_sample (test_rejection_sampler_utils.py:141) and
//     test_synthetic_rejection_sample (:215) — the Gumbel / probability-ratio
//     path is out of scope for I3 (greedy only).
//   * test_block_verification_rejection_sample (:325) and
//     test_block_verification_matches_standard (:372) — block verification.
//   * tests/v1/worker/test_gpu_rejection_sampler_i64.py:109 (>2^31 logits-buffer
//     indexing) — needs the block-verification buffer layout.
