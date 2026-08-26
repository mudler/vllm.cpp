// SPEC-DFLASH2 W9 (#1849) — the `[spec-phase-dev]` device-segment trace of the
// DFlash2 draft phase, gated at the production propose path.
//
// WHY A SEPARATE BINARY. `VT_SPEC_TRACE` is read once by a function-local
// static on the FIRST propose in the process, so one process can only ever
// observe one level. `test_dflash2_runner_reach.cpp` latches "1"; this binary
// latches "2" and is the only place the level-2 lane can be observed.
//
// WHAT LEVEL 2 IS FOR. #1849 measures the draft phase at a flat ~23 ms at
// every K, and the level-1 `sample=` figure aggregates the pre-phase device
// work, the graph replay, the selector chain and the walk into one number.
// Level 2 brackets those four seams with queue synchronizes and prints each
// segment, so the owed on-box attribution run (`## Owed` O1 of
// .agents/specs/dflash2-draft-fixed-cost.md) reads four numbers instead of
// one. The syncs run ONLY at level >= 2 — level 1 keeps its
// overlap-preserving shape — and synchronization changes no value anywhere,
// which case 3 below measures rather than asserts.
//
// WHAT THIS FILE ASSERTS:
//   1. The `[spec-phase-dev]` line is EMITTED from the production propose
//      path, once per proposing step, beside (not instead of) the level-1
//      lines. Deleting the level-2 branch is the M3 mutation and reds this.
//   2. Every segment parses and is non-negative — mis-bracketed boundaries
//      (the M4 mutation: swapped seams read as a negative segment) red here.
//   3. TOKEN IDENTITY under the instrument: the drafted blocks satisfy the
//      same structural properties the level-1 reach gate holds them to, and
//      two engine runs under level 2 draft identical blocks. The instrument
//      may cost time; it may not move a draft.
#include <doctest/doctest.h>

#include "dflash2_runner_fixture.h"

namespace {

// This binary's OWN pre-main latch: level 2. See the fixture header's note.
const bool kSpecTraceLevel2 = [] {
  ::setenv("VT_SPEC_TRACE", "2", 1);
  return true;
}();

// Every `[spec-phase-dev]` payload, in step order.
std::vector<std::string> DevPhaseLines(const std::string& captured) {
  std::vector<std::string> out;
  const std::string key = "[spec-phase-dev] ";
  size_t at = 0;
  while ((at = captured.find(key, at)) != std::string::npos) {
    const size_t open = at + key.size();
    const size_t close = captured.find('\n', open);
    if (close == std::string::npos) break;
    out.push_back(captured.substr(open, close - open));
    at = close;
  }
  return out;
}

// Count of `[spec-phase] ` level-1 lines (the space excludes `-dev]`).
size_t Level1Lines(const std::string& captured) {
  size_t n = 0, at = 0;
  const std::string key = "[spec-phase] ";
  while ((at = captured.find(key, at)) != std::string::npos) {
    ++n;
    at += key.size();
  }
  return n;
}

// Parse `name=<float>ms` out of one payload; REQUIREs the field exists.
double Field(const std::string& line, const std::string& name) {
  const std::string key = name + "=";
  const size_t at = line.find(key);
  REQUIRE_MESSAGE(at != std::string::npos, "missing field ", name,
                  " in: ", line);
  return std::atof(line.c_str() + at + key.size());
}

}  // namespace

TEST_CASE("dflash2 level-2 trace: [spec-phase-dev] splits the draft phase, once per step") {
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  std::string threw;
  const std::string captured = CaptureStderr([&] {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir),
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
    threw = GenerateAndCatch(eng, "hello");
  });
  // NOTE: doctest stringifies a bare `const char*` as bool, so INFO gets the
  // std::string itself.
  INFO("stderr: ", captured);
  CHECK(threw.empty());

  // 1. Emitted, once per proposing step, ADDITIVE to the level-1 lane.
  const std::vector<std::string> dev = DevPhaseLines(captured);
  const size_t l1 = Level1Lines(captured);
  REQUIRE(dev.size() > 0);
  CHECK(l1 > 0);
  CHECK(dev.size() == l1);
  CHECK(DraftedBlocks(captured).size() == dev.size());

  // 2. Every segment parses and is non-negative. A swapped boundary produces
  //    a negative segment, which is the M4 mutation's signature.
  for (const std::string& line : dev) {
    INFO("line: ", line);
    for (const char* f : {"pre", "fwd", "select", "walk"}) {
      const double v = Field(line, f);
      CHECK(v >= 0.0);
    }
  }

  // 3. The drafts under the instrument are REAL walk output, held to the same
  //    structural properties the level-1 reach gate holds them to: a whole
  //    block of kSpecTokens ids per step, every id a token of this vocabulary,
  //    and at least one id beyond the top-k range (the slot-index-instead-of-
  //    candidate mistake's signature). The fixture is deterministic, so a
  //    level-2 lane that moved a draft would break these or the run-to-run
  //    identity case below.
  bool any_beyond_top_k = false;
  for (const std::string& b : DraftedBlocks(captured)) {
    INFO("block: [", b, "]");
    int ids = 0;
    for (size_t i = 0; i < b.size(); ++i)
      if (b[i] == ' ') ++ids;
    CHECK(ids == kSpecTokens);
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

TEST_CASE("dflash2 level-2 trace: `select` brackets the SELECTOR, not a label near it") {
  // THE PLACEMENT GATE (#1851 F1). Cases 1-3 pin presence, count, parse,
  // non-negativity and token neutrality — and the fresh review proved by a
  // surviving mutant that ALL of them stay green under a monotonic slide of
  // the select/walk seam (sync + `t_sel` stamp moved BEFORE
  // `Dflash2SelectCandidatesDevice`): `select` times nothing, `walk` absorbs
  // the selector, every segment stays >= 0, no draft moves. The one consumer
  // of these labels is the `## Owed` O1 lease run that picks the next kernel
  // hypothesis FROM them, so a label unbound from its work is the instrument
  // lying to the only reader it has.
  //
  // This case injects a known wall-clock floor INSIDE the selector's bracket
  // (`VT_SPEC_TEST_SELECT_SPIN_MS`, a steady_clock spin at the top of
  // `Dflash2SelectCandidatesDevice` — speculator.cpp) and asserts it lands in
  // `select` on EVERY traced step, and not in `walk`. Any seam slide, in
  // either direction, moves the floor out of `select` and reds here.
  // The spin is 200 ms; the floor asserted leaves margin for the %.2f print
  // and stamp skew while staying far above any real segment of this fixture.
  constexpr double kFloorMs = 150.0;
  ::setenv("VT_SPEC_TEST_SELECT_SPIN_MS", "200", 1);
  const HfConfig target = MakeDenseConfig();
  const ScratchDraftDir dir;
  std::string threw;
  const std::string captured = CaptureStderr([&] {
    LoadedEngine eng(target, MakeDenseWeights(target), BuildFixture(),
                     DflashSpecParams(dir),
                     MakeDflash2Draft(target, /*muse_glimmer_scalars=*/false));
    threw = GenerateAndCatch(eng, "hello");
  });
  ::unsetenv("VT_SPEC_TEST_SELECT_SPIN_MS");
  INFO("stderr: ", captured);
  CHECK(threw.empty());
  const std::vector<std::string> dev = DevPhaseLines(captured);
  REQUIRE(dev.size() > 0);
  for (const std::string& line : dev) {
    INFO("line: ", line);
    // The spin waits on the SAME clock the runner stamps, so under the
    // committed seam placement `select` carries the whole 200 ms spin by
    // construction, however loaded the box is. Under a slid seam `select` is
    // the stamp-to-stamp residue of a sub-millisecond host chain and cannot
    // reach the floor.
    CHECK(Field(line, "select") >= kFloorMs);
    // And the floor must NOT have drained into the walk segment, which is
    // where the reviewer's slide put the selector's time.
    CHECK(Field(line, "walk") < kFloorMs);
  }
  // The seam is a WAIT, not a computation: the drafts under the spin are
  // byte-identical to an undelayed run of the same deterministic fixture.
  std::string threw_plain;
  const std::vector<std::string> plain = RunAndCollectDrafts(false, &threw_plain);
  CHECK(threw_plain.empty());
  const std::vector<std::string> spun = DraftedBlocks(captured);
  REQUIRE(spun.size() == plain.size());
  for (size_t i = 0; i < spun.size(); ++i) CHECK(spun[i] == plain[i]);
}

TEST_CASE("dflash2 level-2 trace: the instrument does not move a single draft") {
  // The SAME fixture drafts the SAME blocks under the reach binary's level-1
  // run — pinned there by "a DFlash2 draft DRAFTS through the PATH WALK" and
  // the D9 scalar case. Here the whole engine runs again under level 2 (the
  // added syncs live inside the propose path), and the drafted blocks must be
  // identical run-to-run in THIS process too: two engines, same weights, same
  // prompt, one with nothing varied — a trace lane that moved a draft would
  // differ somewhere across these repetitions.
  std::string threw_a, threw_b;
  const std::vector<std::string> a = RunAndCollectDrafts(false, &threw_a);
  const std::vector<std::string> b = RunAndCollectDrafts(false, &threw_b);
  CHECK(threw_a.empty());
  CHECK(threw_b.empty());
  REQUIRE(a.size() > 0);
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) CHECK(a[i] == b[i]);
}
