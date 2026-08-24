// SPEC-DFLASH2 W3 (#1314) — the PRODUCTION reachability gate for the DFlash2
// candidate selector, and the discharge of spec `## Owed` O5 and O7.
//
// WHAT O5 AND O7 SAID, AND WHY THIS FILE EXISTS. W1 and W2 both landed with
// their production call site UNGATED and mutation-proven so: deleting
// `RefuseDflash2CandidateSelector` from `GPUModelRunner::propose_drafts_block`,
// or `draft->weights.conv_block_size = draft->k + 1;` from `LoadDflashDraft`,
// left every suite in this repository green. The stated reason was that a gate
// on the runner would need an on-disk target plus an on-disk draft driven
// through the loader, plus a populated per-request device KV store, and that
// this was W4's harness.
//
// It was not. What was missing was a way to hand a DFlash draft to a LoadedEngine
// built from IN-MEMORY weights -- the exact seam `mtp_weights` already had, and
// which its own header comment justifies in the same words: "a synthetic spec
// engine could only ever run with a NULL drafter, which is exactly the state a
// depth gate must not mistake for working speculation". W3 adds that overload,
// and everything downstream is the production path unchanged: ResolveSpecConfig,
// the aux-multi-tap refusal, `set_dflash_draft`, `propose_drafts` ->
// `propose_drafts_dflash` -> `propose_drafts_block`.
//
// WHAT THIS GATE ASSERTS AS OF W4, and why generation SUCCEEDING is not the
// assertion. Through W3 a DFlash2 draft could not generate at all: the path walk
// was missing and the engine refused by name, so the refusal's own text -- which
// carried the selector's counts -- was the whole gate. W4 lands the walk, so the
// engine generates, and "it generated" would be satisfied by a runner that
// dropped the DFlash2 arm entirely and drafted with the DFlash1 per-slot argmax.
// That is the exact silent-wrong this row exists to remove: the verify is
// lossless, the emitted tokens stay the target's, and only acceptance falls.
//
// So this file asserts the two things an argmax fallback cannot produce:
//
//   1. THE DRAFTS EXIST AND COME OUT OF THE PROPOSE, read off the production
//      `VT_SPEC_TRACE` line at real fd 2 rather than a test-only sink.
//   2. THE DRAFTS MOVE WITH THE SELECTOR'S VALUES. Two engines that differ ONLY
//      in `output_multiplier` and `final_logit_softcapping` -- D9's Muse Glimmer
//      scalars -- must draft DIFFERENT tokens. Those two scalars touch nothing
//      but the candidate VALUES the selector's unary term reads, so the block
//      forward, the convolution and the draft logits are byte-identical between
//      the two runs. The DFlash1 argmax reads the block LOGITS and would answer
//      identically for both; only a draft produced by walking the selector's
//      lattice can differ. The difference is MEASURED by this case, not asserted
//      about the fixture.
//
// And the third leg is structural rather than observational: both propose paths
// enter the DFlash1 argmax on EMPTINESS and call
// `RefuseDflash1ArgmaxOnDflash2Block` first, so deleting the walk's call site --
// the mutation `.agents/reachability.md` requires -- throws by name instead of
// quietly drafting worse tokens.
//
// The target is the same tiny synthetic Qwen3.5 DENSE model
// tests/vllm/v1/spec_decode/test_mtp_depth.cpp builds, because
// `supports_aux_multi_tap()` is true only for the Qwen3.5/3.6 dense and MoE
// forwards and the loader refuses any other target by name.
#include <doctest/doctest.h>

#include "dflash2_runner_fixture.h"

namespace {
// `VT_SPEC_TRACE` is latched by a function-local `static` on the FIRST propose in
// the process, so setting it inside a test case would be a race with whichever
// case ran first. This runs before main.
const bool kSpecTraceEnabled = [] {
  ::setenv("VT_SPEC_TRACE", "1", 1);
  return true;
}();

}  // namespace

TEST_CASE("dflash2 runner: a DFlash2 draft DRAFTS through the PATH WALK") {
  std::string threw;
  const std::vector<std::string> blocks = RunAndCollectDrafts(false, &threw);
  // W3 refused here by name. W4 must not.
  INFO("threw: ", threw);
  CHECK(threw.empty());
  // The propose ran, and it produced a whole block per step. `kSpecTokens` is 3,
  // so each traced block carries three ids.
  REQUIRE_FALSE(blocks.empty());
  for (const std::string& b : blocks) {
    INFO("block: [", b, "]");
    int ids = 0;
    for (size_t i = 0; i < b.size(); ++i)
      if (b[i] == ' ') ++ids;
    CHECK(ids == kSpecTokens);
  }
  // And every drafted id is a real token of this vocabulary rather than a
  // sentinel or an uninitialized slot -- the walk gathers `candidates.ids`, and
  // a walk that emitted the SLOT INDEX instead would still print three numbers.
  // kSelTopK is 3 and the vocabulary is 24, so "every id < top_k" would be the
  // signature of that mistake; this asserts at least one id is not.
  bool any_beyond_top_k = false;
  for (const std::string& b : blocks) {
    std::istringstream is(b);
    int id = 0;
    while (is >> id) {
      CHECK(id >= 0);
      CHECK(id < kVocab);
      if (id >= kSelTopK) any_beyond_top_k = true;
    }
  }
  CHECK(any_beyond_top_k);
}

TEST_CASE("dflash2 runner (W9): level 1 does NOT emit the device split") {
  // THE LEVEL BOUNDARY, PINNED FROM BELOW (#1851 F2). The level-2 binary pins
  // that `VT_SPEC_TRACE=2` emits `[spec-phase-dev]`; nothing pinned that
  // level 1 does NOT — mutating the runner's `propose_trace_level >= 2` to
  // `>= 1` left every suite green, so "the syncs run ONLY at level >= 2" (the
  // W9 claim that keeps every level-1 recipe overlap-preserving) was asserted,
  // not gated. This binary latches level 1 pre-main, so it is the one process
  // that can observe the boundary from this side: the level-1 line must be
  // PRESENT on the same real-fd-2 capture (proving the capture and the trace
  // both worked — absence alone would also be a dead instrument) and the
  // device split must be ABSENT.
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  std::string threw;
  const std::string captured = CaptureStderr([&] {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir), MakeDflash2Draft(target, false));
    threw = GenerateAndCatch(eng, "hello");
  });
  INFO("stderr: ", captured);
  CHECK(threw.empty());
  CHECK(captured.find("[spec-phase] ") != std::string::npos);
  CHECK(captured.find("[spec-propose]") != std::string::npos);
  CHECK(captured.find("[spec-phase-dev]") == std::string::npos);
}

TEST_CASE("dflash2 runner: the STARTUP notice names what runs, not what is owed") {
  // `## Risks/decisions` D10 pays for the moved refusal with a startup notice,
  // and every wave that moves the boundary has to move the notice with it. W3's
  // fresh review found the OTHER copy of this text still naming the wave that had
  // just shipped; the obligation is that the notice is TRUE at its own commit,
  // and this is what holds it.
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  std::string captured;
  {
    CerrCapture cap;
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir), MakeDflash2Draft(target, false));
    captured = cap.str();
  }
  INFO("notice: ", captured);
  CHECK(captured.find("DFlash2DraftModel") != std::string::npos);
  CHECK(captured.find("PATH WALK are all implemented") != std::string::npos);
  CHECK(captured.find("this draft DRAFTS") != std::string::npos);
  // What is still owed, named -- the GGUF drafter's bf16 residency and the
  // absent throughput number -- so the notice is not a claim that the row is
  // finished. W5 (#1314) LANDED the GGUF arm, so the notice must no longer name
  // it as owed; a text that kept saying so would tell a user running that arm
  // that it is refused, which is the staleness this case exists to catch.
  CHECK(captured.find("wave W5") == std::string::npos);
  CHECK(captured.find("GGUF drafter arm is refused") == std::string::npos);
  CHECK(captured.find("from safetensors and from GGUF alike") != std::string::npos);
  CHECK(captured.find("DEQUANTIZED") != std::string::npos);
  CHECK(captured.find("no throughput number") != std::string::npos);
  CHECK(captured.find("#1314") != std::string::npos);
  // And that the port is BEYOND-PIN, which is the one thing a user of a DFlash2
  // checkpoint cannot discover from the checkpoint.
  CHECK(captured.find("52816") != std::string::npos);
  // THE UPSTREAM STATE, PINNED AS A WORD AND NOT ONLY AS A NUMBER. vllm#52816
  // MERGED on 2026-08-21 at 05:27:22Z, and this notice went on telling every
  // user who loads a DFlash2 draft that it "is OPEN upstream" -- through W6 and
  // through W6's first repair wave, which corrected five statements in the spec
  // and never reached the two live surfaces. The line above matches only
  // "52816", so it could not see the difference. These four can: the merged
  // wording and the merged head must be PRESENT, and both spellings of the open
  // claim must be ABSENT. When #1561 reconciles the port onto the merged head,
  // the notice and this case move together.
  CHECK(captured.find("MERGED upstream") != std::string::npos);
  CHECK(captured.find("3406ec1dae9916f920b90f0dbf90dcf54923d042") !=
        std::string::npos);
  CHECK(captured.find("OPEN upstream") == std::string::npos);
  CHECK(captured.find("is OPEN") == std::string::npos);
}

TEST_CASE("dflash2 runner: D9's SCALARS move the drafted tokens, in production") {
  // THE REACHABILITY ASSERTION, and the reason it is this one. Both runs share
  // the same draft weights, the same target, the same prompt and the same block
  // forward; the ONLY difference is `output_multiplier` and
  // `final_logit_softcapping`, which `Qwen3DFlash2Model::ComputeCandidates`
  // applies to the candidate VALUES the selector's unary term reads. Nothing
  // else in the engine sees them.
  //
  // So the DFlash1 per-slot argmax -- which reads the block LOGITS -- answers
  // IDENTICALLY for both. A difference here can only come from a draft that was
  // produced by walking the selector's lattice, on the production path, at the
  // runner's own call site. That is what no exit status and no "it generated"
  // could show.
  std::string threw_default, threw_muse;
  const std::vector<std::string> plain = RunAndCollectDrafts(false, &threw_default);
  const std::vector<std::string> muse = RunAndCollectDrafts(true, &threw_muse);
  INFO("default threw: ", threw_default);
  INFO("muse threw: ", threw_muse);
  CHECK(threw_default.empty());
  CHECK(threw_muse.empty());
  REQUIRE_FALSE(plain.empty());
  REQUIRE(plain.size() == muse.size());
  int differing = 0;
  for (size_t i = 0; i < plain.size(); ++i) {
    INFO("step ", i, " default [", plain[i], "] muse [", muse[i], "]");
    if (plain[i] != muse[i]) ++differing;
  }
  // MEASURED, not assumed, and the MARGIN is written down: on 2026-08-20 this
  // fixture drafts 8 blocks and 2 of them differ between the two scalar arms.
  // A block only moves when the scalars flip the argmax at one of the `k` slots
  // the walk actually visits, so 2-of-8 is the shape of the synthetic ramp
  // rather than a weakness in the wiring -- the model suite measures the same
  // comparison at 5 of 6 blocks over a wider sweep of inputs
  // (tests/vllm/models/test_qwen3_dflash2_draft.cpp). The count is printed so a
  // fixture change that made the two arms agree is visible as a number rather
  // than as a silent pass.
  INFO("blocks: ", plain.size(), " differing: ", differing);
  CHECK(differing > 0);
}

// SPEC-DFLASH2 W8 (#1837, #1838): the two propose lanes agree at the ENGINE.
// Since W8 a DFlash2 propose takes the single-request PAGED lane by default and
// `VT_DFLASH_PAGED=0` selects the materialized fallback — before W8 both
// spellings took the fallback, so this case became meaningful with the wave.
// The drafted blocks must agree BIT-FOR-BIT across the lanes on the production
// path end to end (aux pre-phase, block forward, device selector, walk): the
// wave's whole claim is that residency moved and no float did.
TEST_CASE("dflash2 runner (W8): the paged lane and the materialized lane draft identically") {
  std::string threw_paged, threw_mat;
  const std::vector<std::string> paged = RunAndCollectDrafts(false, &threw_paged);
  setenv("VT_DFLASH_PAGED", "0", 1);
  const std::vector<std::string> mat = RunAndCollectDrafts(false, &threw_mat);
  unsetenv("VT_DFLASH_PAGED");
  INFO("paged threw: ", threw_paged);
  INFO("materialized threw: ", threw_mat);
  CHECK(threw_paged.empty());
  CHECK(threw_mat.empty());
  REQUIRE_FALSE(paged.empty());
  REQUIRE(paged.size() == mat.size());
  for (size_t i = 0; i < paged.size(); ++i) {
    INFO("step ", i, " paged [", paged[i], "] materialized [", mat[i], "]");
    CHECK(paged[i] == mat[i]);
  }
}
