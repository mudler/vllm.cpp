// SPEC-DFLASH2 (#1946) — the PRODUCTION reachability gate for the shared-embed
// rebind.
//
// WHAT THIS ANSWERS THAT THE BYTE-COUNT BINARY CANNOT.
// `test_dflash2_embed_dedup.cpp` proves that `BindDflashDraftSharedEmbed`
// dedups a device upload. It calls the function directly, so it measures the
// function. `.agents/reachability.md` asks the OTHER question: does anything a
// user arrives through ever call it? The `tp` handle threaded through four
// production files and was still dead.
//
// So this binary enters at the ENGINE. `LoadedEngine`'s private constructor is
// the one seam every draft loader crosses — the GGUF branch
// (the `method == "dflash"` arm of `FromModelDir`), the two safetensors branches
// (through `maybe_load_dflash`) and the in-memory W3 overload the DFlash2 gates drive —
// and it is the first point at which the target model and the draft both exist.
// The rebind is a member initialiser there. Delete that call and this file goes
// red; that mutation is the reachability evidence the row records.
//
// WHY A STDERR LINE IS THE OBSERVABLE. `LoadedEngine` exposes neither `model_`
// nor `dflash_draft_`, and adding an accessor for a test would be a surface that
// exists only for the gate. The engine already REPORTS its load facts on
// `std::cerr` — the load-phase timings, the auto-fit notice, the DFlash2 conv
// geometry, "DFlash draft loaded from ..." — and 2.5 GB of recovered unified
// memory is a load fact a GB10 user wants in the same place. The line carries
// the byte count it actually saved, and this file recomputes that number from
// the fixture's own vocabulary and hidden size rather than reading it back from
// the code, so a green here is not a tautology in the loader's own arithmetic.
//
// The target is the same synthetic Qwen3.5 dense model + DFlash2 draft the W4
// runner gate uses, for the reason that gate records: `supports_aux_multi_tap()`
// is true only for the Qwen3.5/3.6 dense and MoE forwards, and the loader
// refuses any other DFlash target by name.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>

#include "dflash2_runner_fixture.h"

namespace {

// The bytes the fixture's table occupies: BF16 [kVocab, hidden_size]. Derived
// HERE from the fixture's own geometry, never read back from the tensor the
// loader bound — which is what keeps this an independent measurement rather
// than a restatement of the loader's own number.
std::string ExpectedSavedBytes(const HfConfig& c) {
  return std::to_string(static_cast<int64_t>(c.vocab_size) * c.hidden_size * 2);
}

}  // namespace

TEST_CASE("#1946: the PRODUCTION engine rebinds the draft embed onto the target") {
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  CerrCapture cap;
  {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir),
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
  }
  const std::string out = cap.str();
  INFO("stderr: ", out);

  // The rebind ran, on the production path, with no test-only call.
  CHECK(out.find("DFlash draft embed SHARED with the target") != std::string::npos);
  // And it names the bytes it saved. A line without the number would still be a
  // line; this is what makes it a measurement.
  CHECK(out.find(ExpectedSavedBytes(target) + " B saved") != std::string::npos);
  // The refusal arm did NOT fire: the fixture's target and draft tables agree on
  // dtype and shape, so a "NOT shared" line here would mean the guard rejected
  // the case it exists to admit.
  CHECK(out.find("DFlash draft embed NOT shared") == std::string::npos);
}

TEST_CASE("#1946: a NON-speculative engine says nothing and is unchanged") {
  // The polarity check. Every production default load carries no draft, and the
  // rebind must be inert rather than merely harmless there — a line on every
  // load would be the first sign the wiring fires where it should not.
  const HfConfig target = MakeDenseConfig();
  CerrCapture cap;
  {
    vllm::entrypoints::EngineParams plain;
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(), plain);
  }
  CHECK(cap.str().find("DFlash draft embed") == std::string::npos);
}
