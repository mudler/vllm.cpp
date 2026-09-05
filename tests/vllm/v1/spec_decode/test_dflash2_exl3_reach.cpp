// MODEL-DFLASH2-EXL3 (#2495 item 7) — the ENGINE reachability gate for the
// EXL3 DFlash2 draft.
//
// WHAT THE SIBLING SUITE CANNOT ANSWER.
// tests/vllm/models/test_qwen3_dflash2_exl3.cpp drives the production LOADER
// over a real on-disk EXL3 file and then the CONTEXT-FREE block forward. That
// covers `Qwen3DFlashModel::ForwardBlockLogits` and the context-aware host
// body. It does not cover the two bodies the runner actually reaches at decode:
// `ForwardBlockLogitsWithDeviceKV` and the paged one. `.agents/reachability.md`
// asks the other question — does anything a user arrives through ever call
// this? — and a forward called directly from a test cannot answer it.
//
// SO THIS BINARY ENTERS AT THE ENGINE, exactly as
// test_dflash2_runner_reach.cpp does and for the same reason its header gives.
// The `--speculative-config` DOCUMENT is parsed
// (`vllm::ParseSpeculativeConfigJson`, from `DflashSpecParams`), against a real
// on-disk draft directory whose `config.json` declares `DFlash2DraftModel`, so
// the production classification runs on this path. Everything downstream is the
// production chain unchanged: `ResolveSpecConfig`, `set_dflash_draft`,
// `propose_drafts` -> `propose_drafts_dflash` -> `propose_drafts_block`.
//
// WHAT THIS DOES NOT PROVE, said here rather than rounded up.
// `LoadedEngine::FromModelDir` opens the draft only after the TARGET's shards
// are mapped (`maybe_load_dflash`, src/vllm/entrypoints/model_loader.cpp), so a
// case driven from the command line needs a complete on-disk target checkpoint,
// which nothing in this tree builds. The WEIGHTS here therefore come from
// memory, through the same in-memory `LoadedEngine` overload every DFlash2
// runner gate uses — but they come out of `vllm::LoadQwen3DFlash` with the
// production presence predicate, which is the function `LoadDflashDraft` calls.
// The one hop that is verified in two pieces rather than one is
// `LoadDflashDraft` itself.
//
// THE ASSERTION IS THAT IT DRAFTS, and "it generated" is deliberately not
// enough. A runner that dropped the EXL3 weights and drafted from a zeroed
// operand would still generate, and the verify is lossless, so the emitted
// tokens would be the target's either way and only acceptance would fall. So
// this file asserts two things beside generation: that drafts came OUT of the
// propose, read off the production `VT_SPEC_TRACE` line at real fd 2, and that
// the object crossing the engine seam is the trellis arm rather than a bf16
// container. Whether those trellises decode to the right numbers is measured
// against a decoded twin in tests/vllm/models/test_qwen3_dflash2_exl3.cpp; the
// two suites answer different questions and neither replaces the other.
//
// THE REACHABILITY MUTATION for this row deletes the EXL3 arm of
// `ProjectDflashQkv` (src/vllm/model_executor/models/qwen3_dflash.cpp) or the
// `MakeMlpGateUpMethod` / `MakeLinearMethod` call sites beside it, and requires
// this suite RED.
#include <doctest/doctest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "dflash2_runner_fixture.h"

namespace {
// `VT_SPEC_TRACE` is latched by a function-local `static` on the FIRST propose
// in the process, so setting it inside a case would race whichever case ran
// first. This runs before main.
const bool kSpecTraceEnabled = [] {
  ::setenv("VT_SPEC_TRACE", "1", 1);
  return true;
}();

// The EXL3 target is WIDER than every other fixture in this header, and it has
// to be: the draft's hidden size is the target's, and `vt::Exl3HadR128` refuses
// a row length that is not a multiple of 128.
HfConfig Exl3Target() { return MakeDenseConfig(kMaxModelLen, kExl3Hidden); }

std::vector<std::string> RunExl3(std::string* threw, std::string* captured_out) {
  const HfConfig target = Exl3Target();
  const ScratchDraftDir dir;
  std::string what;
  const std::string captured = CaptureStderr([&] {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir),
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false,
                                      /*exl3=*/true));
    what = GenerateAndCatch(eng, "hello");
  });
  if (threw != nullptr) *threw = what;
  if (captured_out != nullptr) *captured_out = captured;
  return DraftedBlocks(captured);
}

}  // namespace

TEST_CASE("dflash2 exl3 reach: an EXL3 draft loads through the engine and drafts") {
  REQUIRE(kSpecTraceEnabled);
  std::string threw;
  std::string captured;
  const std::vector<std::string> drafts = RunExl3(&threw, &captured);
  INFO("stderr: ", captured);
  // Nothing refused. Before this row the load threw on `fc.weight`, which an
  // EXL3 module does not ship, and no engine could be built at all.
  CHECK(threw.empty());
  // The DFlash2 arm ran rather than an argmax fallback: the production
  // classification saw the draft directory and the propose emitted its trace.
  CHECK(captured.find("DFlash2DraftModel") != std::string::npos);
  CHECK(captured.find("[spec-propose]") != std::string::npos);
  // And drafts came OUT of it. An engine that proposed nothing would leave this
  // empty while still generating.
  CHECK_FALSE(drafts.empty());
  for (const std::string& b : drafts) CHECK_FALSE(b.empty());
}

TEST_CASE("dflash2 exl3 reach: the draft the engine is handed IS the trellis arm") {
  // What case 1 cannot separate on its own. An engine that generated while the
  // draft had quietly fallen back to bf16 owners would satisfy every assertion
  // above, so this pins the object that crosses the seam: the production loader
  // returned the EXL3 arm, at the published artifact's bits and codebook, with
  // all seven per-layer trellises populated and both merged bf16 owners empty.
  //
  // The NUMERIC half -- that those trellises are multiplied and decode to the
  // right weights -- is measured against a decoded twin in
  // tests/vllm/models/test_qwen3_dflash2_exl3.cpp. This case is about which
  // object the engine gets.
  const HfConfig target = Exl3Target();
  const std::unique_ptr<DflashDraft> draft =
      MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false, /*exl3=*/true);
  REQUIRE(draft != nullptr);
  const vllm::Qwen3DFlashWeights& w = draft->weights;
  CHECK(w.IsDflash2());
  CHECK(w.IsExl3());
  CHECK(w.fc.Empty());
  REQUIRE_FALSE(w.fc_exl3.Empty());
  CHECK(w.fc_exl3.Bits() == kExl3Bits);
  CHECK(w.fc_exl3.codebook == 2);  // mul1
  CHECK(w.fc_exl3.InFeatures() == target.hidden_size * 2);
  CHECK(w.fc_exl3.OutFeatures() == target.hidden_size);
  REQUIRE(static_cast<int>(w.layers.size()) == kDraftLayers);
  for (const vllm::Qwen3DFlashLayerWeights& l : w.layers) {
    CHECK(l.qkv_proj.Empty());
    CHECK(l.gate_up_proj.Empty());
    CHECK(l.o_proj.Empty());
    CHECK(l.down_proj.Empty());
    const vllm::Exl3Weight* all[] = {&l.q_proj_exl3,    &l.k_proj_exl3,  &l.v_proj_exl3,
                                     &l.o_proj_exl3,    &l.gate_proj_exl3,
                                     &l.up_proj_exl3,   &l.down_proj_exl3};
    for (const vllm::Exl3Weight* p : all) {
      REQUIRE_FALSE(p->Empty());
      CHECK(p->Bits() == kExl3Bits);
      CHECK(p->codebook == 2);
    }
  }
}

TEST_CASE("dflash2 exl3 reach: the bf16 arm is unchanged beside it") {
  // The other half of a scoped arm. `MakeDflash2Draft`'s default is still the
  // bf16 store at hidden 32, and it must keep drafting exactly as it did --
  // otherwise the EXL3 rung is not scoped, it is a rewrite of the DFlash2 lane
  // that happens to have an EXL3 case.
  REQUIRE(kSpecTraceEnabled);
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
  CHECK_FALSE(DraftedBlocks(captured).empty());
}
