// Ported from: vllm/v1/sample/ops/penalties.py + vllm/model_executor/layers/utils.py
// + vllm/v1/sample/ops/bad_words.py + vllm/v1/sample/logits_processor/builtin.py
// @ e24d1b24. Behavioral tests for the M1.7 Task 3 penalty / mask / builtin-proc
// entry points over the [num_reqs, vocab] f32 logits. Hand-computed expecteds
// (the exact OpenAI-defined penalty formula, the min-tokens floor, logit-bias
// shift, min-p mask + argmax invariance, the bad-words n-gram block, the
// allowed-ids whitelist). CPU is the correctness gate; the vt ops carry the
// CUDA counterparts (parity in tests/vt/test_ops_penalties.cpp, dgx-pending).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

#include "vllm/v1/sample/logits_processor/builtin.h"
#include "vllm/v1/sample/metadata.h"
#include "vllm/v1/sample/ops/bad_words.h"
#include "vllm/v1/sample/ops/penalties.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::v1::MinTokensState;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

Tensor Logits(std::vector<float>& v, int64_t n, int64_t vocab) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {n, vocab});
}
}  // namespace

// ---------------------------------------------------------------------------
// apply_penalties — the combined repetition / frequency / presence formula.
TEST_CASE("apply_penalties: repetition (sign) + frequency + presence, hand case") {
  // vocab 4, one request. logits = [2, -2, 1, -1].
  // prompt = {0, 1}, output = {2, 2}.  union mask = {0,1,2}; bincounts: tok2=2.
  // rep r=2:  tok0 2>0 -> /2 = 1.0 ; tok1 -2<0 -> *2 = -4.0 ; tok2 1>0 -> 0.5 ;
  //           tok3 not in union -> -1.0
  // freq fr=0.5: tok2 -= 0.5*2 -> 0.5 - 1.0 = -0.5
  // pres pr=1.0: tok2 -= 1.0 -> -0.5 - 1.0 = -1.5
  std::vector<float> logits = {2.0f, -2.0f, 1.0f, -1.0f};
  std::vector<std::vector<int32_t>> prompt = {{0, 1}};
  std::vector<std::vector<int32_t>> output = {{2, 2}};
  std::vector<float> pres = {1.0f}, freq = {0.5f}, rep = {2.0f};
  Tensor tl = Logits(logits, 1, 4);
  Queue q = Q();
  vllm::v1::apply_all_penalties(q, tl, prompt, pres, freq, rep, output);
  CHECK(logits[0] == doctest::Approx(1.0f));
  CHECK(logits[1] == doctest::Approx(-4.0f));
  CHECK(logits[2] == doctest::Approx(-1.5f));
  CHECK(logits[3] == doctest::Approx(-1.0f));  // token in neither -> unchanged
}

TEST_CASE("apply_penalties: no-op penalties (r=1, freq=pres=0) leave logits unchanged") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f};
  std::vector<std::vector<int32_t>> prompt = {{0, 1, 2}};
  std::vector<std::vector<int32_t>> output = {{0, 1, 2}};
  std::vector<float> pres = {0.0f}, freq = {0.0f}, rep = {1.0f};
  Tensor tl = Logits(logits, 1, 3);
  Queue q = Q();
  vllm::v1::apply_penalties(q, tl, prompt, output, pres, freq, rep);
  CHECK(logits[0] == doctest::Approx(1.0f));
  CHECK(logits[1] == doctest::Approx(2.0f));
  CHECK(logits[2] == doctest::Approx(3.0f));
}

TEST_CASE("apply_penalties: out-of-range / -1 placeholder token ids are ignored") {
  // vocab 3; output contains the vocab-size pad (3) and the -1 async placeholder.
  // Only token 1 is a valid output token (bincount 1, presence).
  std::vector<float> logits = {1.0f, 4.0f, 1.0f};
  std::vector<std::vector<int32_t>> prompt = {{}};
  std::vector<std::vector<int32_t>> output = {{3, -1, 1}};
  std::vector<float> pres = {0.0f}, freq = {2.0f}, rep = {1.0f};
  Tensor tl = Logits(logits, 1, 3);
  Queue q = Q();
  vllm::v1::apply_penalties(q, tl, prompt, output, pres, freq, rep);
  CHECK(logits[0] == doctest::Approx(1.0f));
  CHECK(logits[1] == doctest::Approx(2.0f));  // 4 - 2*1
  CHECK(logits[2] == doctest::Approx(1.0f));
}

// ---------------------------------------------------------------------------
// apply_min_tokens — stop tokens censored below the min_tokens floor.
TEST_CASE("apply_min_tokens: masks stop tokens while output_len < min_tokens") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
  std::map<int, MinTokensState> min_tokens;
  min_tokens[0] = MinTokensState{/*min_tokens=*/5, /*stop_token_ids=*/{1, 3}};
  std::vector<std::vector<int32_t>> output = {{10, 11}};  // len 2 < 5
  Tensor tl = Logits(logits, 1, 4);
  Queue q = Q();
  vllm::v1::apply_min_tokens(q, tl, min_tokens, output);
  CHECK(logits[0] == doctest::Approx(1.0f));
  CHECK(logits[1] == kNegInf);
  CHECK(logits[2] == doctest::Approx(3.0f));
  CHECK(logits[3] == kNegInf);
}

TEST_CASE("apply_min_tokens: no mask once output_len >= min_tokens") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
  std::map<int, MinTokensState> min_tokens;
  min_tokens[0] = MinTokensState{/*min_tokens=*/2, /*stop_token_ids=*/{1, 3}};
  std::vector<std::vector<int32_t>> output = {{10, 11}};  // len 2 >= 2 -> no mask
  Tensor tl = Logits(logits, 1, 4);
  Queue q = Q();
  vllm::v1::apply_min_tokens(q, tl, min_tokens, output);
  CHECK(logits[1] == doctest::Approx(2.0f));
  CHECK(logits[3] == doctest::Approx(4.0f));
}

// ---------------------------------------------------------------------------
// apply_logit_bias — per-(req, token) additive shift.
TEST_CASE("apply_logit_bias: shifts the specified tokens by their bias") {
  std::vector<float> logits = {0.0f, 0.0f, 0.0f, 0.0f};
  std::map<int, std::map<int32_t, float>> bias;
  bias[0] = {{1, 5.0f}, {3, -2.0f}};
  Tensor tl = Logits(logits, 1, 4);
  Queue q = Q();
  vllm::v1::apply_logit_bias(q, tl, bias);
  CHECK(logits[0] == doctest::Approx(0.0f));
  CHECK(logits[1] == doctest::Approx(5.0f));
  CHECK(logits[2] == doctest::Approx(0.0f));
  CHECK(logits[3] == doctest::Approx(-2.0f));
}

// ---------------------------------------------------------------------------
// apply_min_p — low-prob tokens masked; the argmax is invariant.
TEST_CASE("apply_min_p: masks probs below min_p * max_prob, argmax unchanged") {
  // softmax == [0.1, 0.2, 0.3, 0.4]; min_p=0.4 -> thr = 0.4*0.4 = 0.16.
  // tok0 (0.1 < 0.16) masked; tok1 (0.2 > 0.16) kept; tok2/tok3 kept. min_p is
  // kept off the 0.2 knife-edge so float rounding of softmax can't flip tok1.
  std::vector<float> logits = {std::log(0.1f), std::log(0.2f), std::log(0.3f), std::log(0.4f)};
  std::vector<float> min_p = {0.4f};
  Tensor tl = Logits(logits, 1, 4);
  Queue q = Q();
  vllm::v1::apply_min_p(q, tl, min_p);
  CHECK(logits[0] == kNegInf);
  CHECK(logits[1] == doctest::Approx(std::log(0.2f)));
  CHECK(logits[2] == doctest::Approx(std::log(0.3f)));
  CHECK(logits[3] == doctest::Approx(std::log(0.4f)));  // argmax (max prob) survives
}

TEST_CASE("apply_min_p: min_p == 0 row is a no-op") {
  std::vector<float> logits = {std::log(0.1f), std::log(0.2f), std::log(0.3f), std::log(0.4f)};
  std::vector<float> min_p = {0.0f};
  Tensor tl = Logits(logits, 1, 4);
  Queue q = Q();
  vllm::v1::apply_min_p(q, tl, min_p);
  CHECK(logits[0] == doctest::Approx(std::log(0.1f)));
  CHECK(logits[3] == doctest::Approx(std::log(0.4f)));
}

// ---------------------------------------------------------------------------
// apply_bad_words — n-gram suffix match blocks the final token.
TEST_CASE("apply_bad_words: 2-gram [2,3] blocks token 3 when last output token is 2") {
  std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::map<int, std::vector<std::vector<int32_t>>> bad;
  bad[0] = {{2, 3}};
  std::vector<std::vector<int32_t>> past = {{1, 2}};  // last token 2 -> prefix match
  Tensor tl = Logits(logits, 1, 5);
  Queue q = Q();
  vllm::v1::apply_bad_words(q, tl, bad, past);
  CHECK(logits[3] == kNegInf);
  CHECK(logits[2] == doctest::Approx(2.0f));  // only the final n-gram token blocked
  CHECK(logits[4] == doctest::Approx(4.0f));
}

TEST_CASE("apply_bad_words: non-matching prefix leaves logits unchanged") {
  std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::map<int, std::vector<std::vector<int32_t>>> bad;
  bad[0] = {{2, 3}};
  std::vector<std::vector<int32_t>> past = {{1, 1}};  // last token 1 != 2
  Tensor tl = Logits(logits, 1, 5);
  Queue q = Q();
  vllm::v1::apply_bad_words(q, tl, bad, past);
  CHECK(logits[3] == doctest::Approx(3.0f));
}

TEST_CASE("apply_bad_words: 1-gram unconditionally blocks its token") {
  std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::map<int, std::vector<std::vector<int32_t>>> bad;
  bad[0] = {{4}};                                 // prefix length 0 -> always matches
  std::vector<std::vector<int32_t>> past = {{0}};  // any context
  Tensor tl = Logits(logits, 1, 5);
  Queue q = Q();
  vllm::v1::apply_bad_words(q, tl, bad, past);
  CHECK(logits[4] == kNegInf);
}

// ---------------------------------------------------------------------------
// apply_allowed_token_ids — mask TRUE == exclude.
TEST_CASE("apply_allowed_token_ids: only whitelisted tokens survive") {
  // mask row [1,0,0,1] -> exclude tokens 0 and 3 (allowed = {1,2}).
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<std::vector<uint8_t>> mask = {{1, 0, 0, 1}};
  Tensor tl = Logits(logits, 1, 4);
  Queue q = Q();
  vllm::v1::apply_allowed_token_ids(q, tl, mask);
  CHECK(logits[0] == kNegInf);
  CHECK(logits[1] == doctest::Approx(2.0f));
  CHECK(logits[2] == doctest::Approx(3.0f));
  CHECK(logits[3] == kNegInf);
}
