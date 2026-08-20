// SPEC-DFLASH2 W2 (#1314) — the boundary W2 leaves the DFlash2 architecture at.
//
// A `DFlash2DraftModel` draft now LOADS and its block forward RUNS, grouped
// dynamic convolution and all. What it cannot do is CHOOSE: upstream replaces
// the DFlash1 per-slot argmax with a candidate selector -- keep the target
// head's top-K per slot, score adjacent transitions
// `<A[p] * project(h), B[c]> + unary[c]`, walk the best path from the verified
// anchor (`vllm/model_executor/models/qwen3_dflash2.py` `CandidateSelector` and
// `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` @
// vllm-project/vllm#52816 head `19c9351904df4c63042671bc67a866ca48dc7d6f`), and
// none of that is ported.
//
// WHY THIS IS A REFUSAL AND NOT A FALLBACK, which is the whole content of this
// file: `SampleDflashBlockDrafts` would SUCCEED on a DFlash2 block. It would
// return well-formed tokens, the verify would accept or reject them losslessly,
// the engine would emit the TARGET's tokens either way, and only ACCEPTANCE
// would fall. No token gate in this repository can see that. So the engine
// refuses by name instead, and this suite is what holds it to that.
//
// The refusal is placed AFTER the forward on purpose. The forward is implemented
// and gated (tests/vllm/models/test_qwen3_dflash2_draft.cpp,
// tests/vt/test_ops_dflash2_grouped_conv.cpp); the choice is not. Refusing
// before it would leave every line of W2 unreachable from any production entry
// point -- AGENTS.md `## Nothing lands dead`.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"

using vllm::Qwen3DFlashWeights;
using vllm::v1::RefuseDflash2CandidateSelector;

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

TEST_CASE("dflash2: the candidate selector is REFUSED BY NAME for a DFlash2 draft") {
  std::string what;
  try {
    RefuseDflash2CandidateSelector(Dflash2());
    FAIL("expected a refusal for a DFlash2 draft");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  // The mechanism that is missing, named -- not "DFlash2 is unsupported".
  CHECK(what.find("CANDIDATE SELECTOR") != std::string::npos);
  // The mechanism that is NOT missing, so a reader is not sent to reimplement it.
  CHECK(what.find("convolution IS implemented") != std::string::npos);
  // Why a fallback is inadmissible, which is the part a future agent needs.
  CHECK(what.find("only") != std::string::npos);
  CHECK(what.find("acceptance falls") != std::string::npos);
  // Who owns the wiring.
  CHECK(what.find("W3") != std::string::npos);
  CHECK(what.find("SPEC-DFLASH2") != std::string::npos);
  CHECK(what.find("#1314") != std::string::npos);
}

TEST_CASE("dflash2: a DFlash1 draft passes the selector check untouched") {
  // The instrument's own precondition. A check that refused EVERY dflash draft
  // would satisfy the case above while killing the lane that ships, and that
  // case's assertions could not tell the two apart.
  CHECK_NOTHROW(RefuseDflash2CandidateSelector(Dflash1()));
}

TEST_CASE("dflash2: the DFlash1 per-slot argmax still answers for a DFlash1 block") {
  // The thing the selector replaces, unchanged: the refusal above must not have
  // moved DFlash1's sampling. Two requests, k=2, draft_vocab=3; request 0's mask
  // rows peak at ids 2 and 0, request 1's at 1 and 2.
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
