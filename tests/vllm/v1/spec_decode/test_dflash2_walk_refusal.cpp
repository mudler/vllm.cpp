// SPEC-DFLASH2 W3 (#1314) — the boundary W3 leaves the DFlash2 architecture at.
//
// W2's boundary was the candidate SELECTOR: the draft loaded, its grouped
// dynamic convolution ran, and it was refused because nothing could choose among
// its logits. W3 implements the choosing up to but not including the walk -- the
// target head's top-K (`vt::TopKValuesIndices`), the two codebooks and the edge
// lattice (`vt::Dflash2SelectorEdges`) -- so the boundary MOVED one step and
// this file moved with it (it was `test_dflash2_selector_refusal.cpp`).
//
// What is still missing is the PATH WALK: upstream walks the best path through
// those edge scores from the verified anchor, by inverse CDF at T>0, and caches
// the realized q the lossless verify reads
// (`_selector_walk_kernel` and `_cache_draft_logits_kernel`,
// `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` @
// vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`).
//
// WHY THIS IS A REFUSAL AND NOT A FALLBACK, which is the whole content of this
// file: `SampleDflashBlockDrafts` would SUCCEED on a DFlash2 block. It would
// return well-formed tokens, the verify would accept or reject them losslessly,
// the engine would emit the TARGET's tokens either way, and only ACCEPTANCE
// would fall. No token gate in this repository can see that. So the engine
// refuses by name instead, and this suite is what holds it to that.
//
// The refusal is placed AFTER the selector on purpose. The forward and the
// selector are implemented and gated (tests/vllm/models/test_qwen3_dflash2_draft.cpp,
// tests/vt/test_ops_dflash2_grouped_conv.cpp,
// tests/vt/test_ops_dflash2_selector_edges.cpp,
// tests/vt/test_ops_topk_values_indices.cpp); the walk is not. Refusing before
// them would leave every line of W2 and W3 unreachable from any production entry
// point -- AGENTS.md `## Nothing lands dead`.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"

using vllm::Qwen3DFlashWeights;
using vllm::v1::RefuseDflash2PathWalk;

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

TEST_CASE("dflash2: the PATH WALK is REFUSED BY NAME for a DFlash2 draft") {
  std::string what;
  try {
    vllm::v1::Dflash2ProposeState scored;
    scored.num_reqs = 1;
    scored.num_steps = 7;
    scored.top_k = 16;
    scored.edge_scores.assign(1 * 7 * 16 * 16, 0.0f);
    RefuseDflash2PathWalk(Dflash2(), scored);
    FAIL("expected a refusal for a DFlash2 draft");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  // The mechanism that is missing, named -- not "DFlash2 is unsupported".
  CHECK(what.find("PATH WALK") != std::string::npos);
  // The mechanisms that are NOT missing, so a reader is not sent to reimplement
  // them. This is the assertion that MOVED in W3: the selector is now on the
  // implemented side of the sentence.
  CHECK(what.find("CANDIDATE SELECTOR are") != std::string::npos);
  CHECK(what.find("implemented and just ran") != std::string::npos);
  // Why a fallback is inadmissible, which is the part a future agent needs.
  CHECK(what.find("only") != std::string::npos);
  CHECK(what.find("acceptance falls") != std::string::npos);
  // Who owns the wiring.
  CHECK(what.find("W4") != std::string::npos);
  CHECK(what.find("SPEC-DFLASH2") != std::string::npos);
  CHECK(what.find("#1314") != std::string::npos);
  // The lattice it is declining to walk, named. This is also what makes the
  // SELECTOR's execution observable at the production call site: see the
  // header, and tests/vllm/models/test_qwen3_dflash2_draft.cpp for the case
  // that asserts these counts against a REAL block forward.
  CHECK(what.find("scored-transitions=1792") != std::string::npos);
  CHECK(what.find("steps=7") != std::string::npos);
  CHECK(what.find("top_k=16") != std::string::npos);
}

TEST_CASE("dflash2: a DFlash1 draft passes the walk check untouched") {
  // The instrument's own precondition. A check that refused EVERY dflash draft
  // would satisfy the case above while killing the lane that ships, and that
  // case's assertions could not tell the two apart.
  CHECK_NOTHROW(RefuseDflash2PathWalk(Dflash1(), vllm::v1::Dflash2ProposeState{}));
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
