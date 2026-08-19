// vllm.cpp original — the CPU-tier contract for the coarse-key knob ITSELF:
// `vt::GraphDedupCoarseKeyEnabled()` in src/vt/graph_dedup.h.
// Row ENG-CUDAGRAPH-DEDUP, issues #1226 and #1162.
//
// WHY THIS IS A SEPARATE BINARY, and it has to be. The accessor caches its answer in a
// function-local static, deliberately: the drivers capture lazily and repeatedly, and a
// knob that could change between two captures of one model would hand out a mixed set of
// handles for no stated reason. One process can therefore observe exactly ONE value of
// the variable. Pinning "1" turns it on and "10" does not needs more than one value, so
// it needs more than one process, and CTest is what supplies them: tests/CMakeLists.txt
// registers this binary once per (value, expectation) pair.
//
// WHAT WAS UNHELD BEFORE IT EXISTED, measured rather than asserted. The only coverage of
// this accessor was `CHECK_FALSE(GraphDedupCoarseKeyEnabled())` with the variable unset,
// inside tests/vt/test_graph_dedup_runtime.cpp. Two mutations of the accessor left that
// suite 22/22 green:
//
//   * accepting any non-null value -- so `=0`, `=true` and `=10` would all silently turn
//     the coarse key ON. That is the same defect class as the `"10"`-enables-dedup bug
//     the sibling knob's polarity test was written for, one knob over.
//   * reading `VT_TOTALLY_UNRELATED_NAME` instead. The suite could not prove WHICH
//     variable the accessor reads, which means it could not prove that the device A/B's
//     `VT_CUDA_GRAPH_DEDUP_COARSE_KEY=1` arm asked for anything at all.
//
// The polarity FUNCTION `GraphDedupEnabledFor` is string-tested next door and stays
// there. What is pinned here is that this accessor is wired to that function and to that
// variable name — the two facts a unit test on the pure function cannot reach.
//
// STATED LIMIT. This binary does not gate the `key mode = ` line the accessor prints on
// first call, only the value it returns. That line is the device A/B's proof that the
// flag reached the process, and a mutation that swapped its two strings would mislabel
// every cell; capturing it needs the print to happen inside a redirected stderr, which
// needs this file to control which call is the FIRST one in the process. It is left to
// the run's own log comparison rather than faked here.
#include <doctest/doctest.h>

#include <cstdlib>
#include <string>

#include "support/test_env.h"
#include "vt/graph_dedup.h"

namespace {

// What this process was registered to expect. Absent means the plain registration with
// no environment at all, which is the default arm: the flag is off.
bool ExpectOn() {
  const char* expect = std::getenv("VT_COARSE_KEY_EXPECT");
  const std::string want = expect == nullptr ? std::string("0") : std::string(expect);
  // A registration that sets neither "0" nor "1" is broken, and a broken registration
  // must not read as a pass: this is the "0 assertions is a skip wearing a pass" shape,
  // one level up.
  REQUIRE_MESSAGE((want == "0" || want == "1"),
                  "VT_COARSE_KEY_EXPECT must be \"0\" or \"1\", got: " << want);
  return want == "1";
}

std::string CoarseKeyValue() {
  const char* raw = std::getenv("VT_CUDA_GRAPH_DEDUP_COARSE_KEY");
  return raw == nullptr ? std::string("<unset>") : std::string(raw);
}

}  // namespace

TEST_CASE("the coarse key follows VT_CUDA_GRAPH_DEDUP_COARSE_KEY, and only \"1\" enables it") {
  const std::string value = CoarseKeyValue();
  INFO("VT_CUDA_GRAPH_DEDUP_COARSE_KEY=" << value);
  const bool expected = ExpectOn();
  CHECK(vt::GraphDedupCoarseKeyEnabled() == expected);
}

TEST_CASE("the coarse key is read once, so it cannot change between two captures") {
  // The cache is a contract, not an optimisation: the drivers capture lazily, one padded
  // bucket at a time, so a knob re-read per capture could give one model two key modes
  // and make its `captured N graphs, deduped to M execs` line unattributable. Flipping
  // the variable after the first read must not move the answer.
  const bool first = vt::GraphDedupCoarseKeyEnabled();
  const std::string original = CoarseKeyValue();

  vllm_test::SetEnv("VT_CUDA_GRAPH_DEDUP_COARSE_KEY", first ? "0" : "1");
  CHECK(vt::GraphDedupCoarseKeyEnabled() == first);

  // Restore, so the other case reads the environment this process was registered with
  // whichever order the two run in.
  if (original == "<unset>") {
    vllm_test::UnsetEnv("VT_CUDA_GRAPH_DEDUP_COARSE_KEY");
  } else {
    vllm_test::SetEnv("VT_CUDA_GRAPH_DEDUP_COARSE_KEY", original);
  }
}
