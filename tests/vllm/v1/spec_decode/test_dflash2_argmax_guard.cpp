// SPEC-DFLASH2 W4 (#1314) — the guard that replaces three waves of refusals,
// pointing the other way.
//
// W1 refused a DFlash2 draft before any weight was read. W2 moved the refusal to
// the candidate selector, W3 to the PATH WALK, and each time the reason was the
// same: `SampleDflashBlockDrafts` would SUCCEED on a DFlash2 block. It returns
// well-formed tokens, the verify is lossless, the engine emits the TARGET's
// tokens either way, and only ACCEPTANCE falls. No token gate in this repository
// can see that.
//
// W4 lands the walk, so on the greedy arm a user can configure there is no
// missing mechanism left to refuse — and the hazard is unchanged. What this file
// gates is therefore the INVERSE guard: `RefuseDflash1ArgmaxOnDflash2Block`
// throws when a DFlash2 block reaches the DFlash1 per-slot argmax, and is a
// no-op on a DFlash1 draft. Both propose paths call it immediately before the
// argmax fallback they enter on emptiness, so DELETING the walk's call site --
// the mutation `.agents/reachability.md` requires a reviewer to make -- is a
// loud failure rather than a green run that drafts worse tokens.
//
// This file was `test_dflash2_walk_refusal.cpp` through W3 and
// `test_dflash2_selector_refusal.cpp` through W2. It is renamed rather than
// deleted because what it holds is the same obligation each time: the DFlash1
// argmax must never silently answer for a DFlash2 block.
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"

using vllm::Qwen3DFlashWeights;
using vllm::v1::RefuseDflash1ArgmaxOnDflash2Block;

namespace {

Qwen3DFlashWeights Dflash1() {
  Qwen3DFlashWeights w;
  w.num_taps = 5;
  w.mask_token_id = 248070;
  w.draft_vocab_size = 8;
  return w;  // conv_taps 0 -> IsDflash2() false
}

Qwen3DFlashWeights Dflash2() {
  Qwen3DFlashWeights w = Dflash1();
  w.conv_taps = 2;         // dflash_config.conv_kernel_size on both published drafts
  w.conv_group_size = 16;  // dflash_config.conv_group_size on both
  w.conv_block_size = 8;
  return w;
}

}  // namespace

TEST_CASE("dflash2: the DFlash1 argmax is REFUSED BY NAME for a DFlash2 block") {
  std::string what;
  try {
    RefuseDflash1ArgmaxOnDflash2Block(Dflash2());
    FAIL("expected a refusal for a DFlash2 draft");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  // WHAT went wrong, in the terms a reader needs: not "DFlash2 is unsupported"
  // (it is supported now) but "the walk's call site is gone".
  CHECK(what.find("per-slot argmax") != std::string::npos);
  CHECK(what.find("call site") != std::string::npos);
  // WHY it cannot be a fallback. This is the part a future agent needs, and it
  // is the same sentence W1, W2 and W3 each carried.
  CHECK(what.find("ACCEPTANCE falls") != std::string::npos);
  CHECK(what.find("no token gate") != std::string::npos);
  // WHAT must run instead, by symbol, on both sides of the port.
  CHECK(what.find("Dflash2WalkPath") != std::string::npos);
  CHECK(what.find("_selector_walk_kernel") != std::string::npos);
  CHECK(what.find("66e5414c6d75a8529473d977f7458c140bbab8a0") != std::string::npos);
  CHECK(what.find("SPEC-DFLASH2") != std::string::npos);
  CHECK(what.find("#1314") != std::string::npos);
}

TEST_CASE("dflash2: a DFlash1 draft passes the argmax guard untouched") {
  // The instrument's own precondition, and it is not a formality: this guard
  // runs on EVERY DFlash1 propose. A version that refused every dflash draft
  // would satisfy the case above while killing the lane that ships, and that
  // case's assertions could not tell the two apart.
  CHECK_NOTHROW(RefuseDflash1ArgmaxOnDflash2Block(Dflash1()));
}

TEST_CASE("dflash2: the DFlash1 per-slot argmax still answers for a DFlash1 block") {
  // The thing the selector replaces, unchanged: nothing in this wave moved
  // DFlash1's sampling. Two requests, k=2, draft_vocab=3; request 0's mask rows
  // peak at ids 2 and 0, request 1's at 1 and 2.
  const std::vector<float> logits = {
      0.0f, 0.0f, 0.0f,   // req 0 anchor (never sampled)
      0.1f, 0.2f, 0.9f,   // req 0 mask 0 -> 2
      0.7f, 0.3f, 0.1f,   // req 0 mask 1 -> 0
      0.0f, 0.0f, 0.0f,   // req 1 anchor
      0.2f, 0.8f, 0.4f,   // req 1 mask 0 -> 1
      0.1f, 0.2f, 0.6f,   // req 1 mask 1 -> 2
  };
  const std::vector<std::vector<int32_t>> drafts =
      vllm::v1::SampleDflashBlockDrafts(logits, /*num_reqs=*/2, /*k=*/2, /*draft_vocab=*/3);
  REQUIRE(drafts.size() == 2);
  REQUIRE(drafts[0].size() == 2);
  CHECK(drafts[0][0] == 2);
  CHECK(drafts[0][1] == 0);
  CHECK(drafts[1][0] == 1);
  CHECK(drafts[1][1] == 2);
}

TEST_CASE("dflash2: the WALK turns a scored lattice into the candidates it chose") {
  // `Dflash2WalkPath` at the speculator level, over a hand-built lattice: the
  // ops suite (tests/vt/test_ops_dflash2_path_walk.cpp) pins the walk's rule,
  // and this pins the LAYER above it -- that the result is per-request rows of
  // TARGET-vocabulary token ids gathered from `candidates.ids`, in the request
  // order the selector was given.
  vllm::v1::Dflash2ProposeState scored;
  scored.num_reqs = 2;
  scored.num_steps = 2;
  scored.top_k = 3;
  scored.candidates.rows = 4;
  scored.candidates.top_k = 3;
  //            req0 step0        req0 step1        req1 step0        req1 step1
  scored.candidates.ids = {101, 102, 103,  201, 202, 203,
                           301, 302, 303,  401, 402, 403};
  scored.candidates.values.assign(12, 0.0f);
  scored.edge_scores.assign(2 * 2 * 3 * 3, 0.0f);
  auto at = [&](int64_t b, int64_t l, int64_t p, int64_t c) -> float& {
    return scored.edge_scores[static_cast<size_t>(((b * 2 + l) * 3 + p) * 3 + c)];
  };
  // Request 0: step 0 (row 0) picks slot 1; step 1 must then read row 1, whose
  // best is slot 2. Row 0 of step 1 peaks at slot 0, so a dropped carry answers
  // 201 instead of 203.
  at(0, 0, 0, 1) = 5.0f;
  at(0, 1, 0, 0) = 5.0f;
  at(0, 1, 1, 2) = 7.0f;
  // Request 1: step 0 picks slot 2; step 1 reads row 2, whose best is slot 1.
  at(1, 0, 0, 2) = 5.0f;
  at(1, 1, 0, 0) = 5.0f;
  at(1, 1, 2, 1) = 9.0f;

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::v1::Dflash2WalkResult out = vllm::v1::Dflash2WalkPath(scored, q);
  REQUIRE(out.draft_token_ids.size() == 2);
  REQUIRE(out.draft_token_ids[0].size() == 2);
  CHECK(out.draft_token_ids[0][0] == 102);
  CHECK(out.draft_token_ids[0][1] == 203);
  CHECK(out.draft_token_ids[1][0] == 303);
  CHECK(out.draft_token_ids[1][1] == 402);
  // The two live mistakes, asserted as NOT the answer: a dropped carry reads
  // row 0 at step 1, and a per-request mix-up would cross the two rows.
  CHECK(out.draft_token_ids[0][1] != 201);
  CHECK(out.draft_token_ids[1][1] != 401);
}

namespace {

// A lattice `Dflash2WalkPath` accepts: 1 request, 2 steps, K 3, every transition
// scored 0, so the walk answers slot 0 at both steps.
vllm::v1::Dflash2ProposeState GoodState() {
  vllm::v1::Dflash2ProposeState scored;
  scored.num_reqs = 1;
  scored.num_steps = 2;
  scored.top_k = 3;
  scored.candidates.rows = 2;
  scored.candidates.top_k = 3;
  scored.candidates.ids.assign(6, 7);
  scored.candidates.values.assign(6, 0.0f);
  scored.edge_scores.assign(2 * 3 * 3, 0.0f);
  return scored;
}

}  // namespace

TEST_CASE("dflash2: the walk REFUSES a lattice that does not match its candidates") {
  // The postcondition `Dflash2SelectCandidates` asserts, re-asserted where the
  // walk consumes it: a lattice that is not [num_reqs, num_steps, K, K] would
  // index plausible scores from the wrong rows and move acceptance silently.
  //
  // Each refusal is matched on its MESSAGE. The FIVE blocks below drive THREE
  // guards -- the lattice-size check, the candidate-set shape check on each of
  // its two axes, and the i32-range check on each end of the range (#1518
  // corrects an earlier sentence here that counted them as three checks). They
  // sit next to each other over the same state, so a bare `CHECK_THROWS` would
  // be satisfied by whichever guard still stood after the others were deleted --
  // which is what W4's fresh review measured: deleting the candidate-set check
  // and the i32 refusal together left this suite green.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  CHECK_NOTHROW(vllm::v1::Dflash2WalkPath(GoodState(), q));

  // The LATTICE, one transition short.
  {
    vllm::v1::Dflash2ProposeState scored = GoodState();
    scored.edge_scores.assign(2 * 3 * 3 - 1, 0.0f);
    CHECK_THROWS_WITH_AS(vllm::v1::Dflash2WalkPath(scored, q),
                         doctest::Contains("must score every (step, predecessor"),
                         std::runtime_error);
  }
  // The CANDIDATE SET's own shape, both axes. `rows` is the flattened
  // (request, step) count and `top_k` the slot count; either one wrong reads the
  // ids of a different step, and the ids are well-formed token ids either way.
  {
    vllm::v1::Dflash2ProposeState scored = GoodState();
    scored.candidates.rows = 3;  // not num_reqs * num_steps
    CHECK_THROWS_WITH_AS(vllm::v1::Dflash2WalkPath(scored, q),
                         doctest::Contains("candidate set must be"),
                         std::runtime_error);
  }
  {
    vllm::v1::Dflash2ProposeState scored = GoodState();
    scored.candidates.top_k = 2;  // not the top_k the lattice was scored at
    CHECK_THROWS_WITH_AS(vllm::v1::Dflash2WalkPath(scored, q),
                         doctest::Contains("candidate set must be"),
                         std::runtime_error);
  }
  // The i32 RANGE the verify, the KV rollback and the input batch all carry.
  // The walk gathers an i64 id; one that does not fit would TRUNCATE into a
  // different, valid-looking token rather than fail, so it is named instead.
  {
    vllm::v1::Dflash2ProposeState scored = GoodState();
    scored.candidates.ids[0] = static_cast<int64_t>(INT32_MAX) + 1;
    CHECK_THROWS_WITH_AS(vllm::v1::Dflash2WalkPath(scored, q),
                         doctest::Contains("outside the i32 range"),
                         std::runtime_error);
  }
  {
    vllm::v1::Dflash2ProposeState scored = GoodState();
    scored.candidates.ids[0] = -1;
    CHECK_THROWS_WITH_AS(vllm::v1::Dflash2WalkPath(scored, q),
                         doctest::Contains("outside the i32 range"),
                         std::runtime_error);
  }
}
