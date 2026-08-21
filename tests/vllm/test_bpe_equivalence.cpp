// SPEC-BPE-QUADRATIC-MERGE (#1365) — the equivalence corpus and the cost bound.
//
// `src/vllm/tokenizer/bpe.cpp::BpeMerge` is O(n^2) in the symbols of ONE
// pretoken, and nothing bounds a pretoken: five of the seven pretokenizer rules
// return an unbounded run, and on the SentencePiece family the whole prompt is
// one word. The committed 64-entry Qwen3.6 corpus has a longest pretoken of 54
// bytes, so no gate in the tree has ever entered the failing regime.
//
// This binary carries the two halves of the argument.
//
// 1. EQUIVALENCE (spec §Tests to port items 1 and 6). 80 corpus entries through
//    BOTH committed goldens — the byte-level Qwen3.6 arm reached by
//    `Tokenizer::EncodePlain`, and the SentencePiece Mistral arm reached by
//    `Tokenizer::EncodePlainSp`, whose `"split": false` pre_tokenizer makes the
//    whole prompt one word. The recorded ids come from HF `tokenizers` 0.22.2
//    reading THE SAME committed tokenizer.json bytes — an ORACLE, not a
//    snapshot of our own output, which would only be a change detector and
//    could go green on a faster wrong answer. Capture method, versions and
//    sha256s: `tools/parity/dump_bpe_equivalence.py` and the golden's `oracle`
//    block. This case PASSES on the code as it stands and must keep passing.
//
// 2. THE COST BOUND (spec §Tests to port item 2). One 65,536-byte input per
//    caller, each pretokenizing to ONE word, with an ABSOLUTE wall-time bound.
//    This case FAILS on the code as it stands. It is the row's only timing
//    assertion; spec §Tests to port item 3 records the growth-ratio assertion
//    that was considered and REJECTED, and why no form of it may come back.
//
// The cost case compares the full id vector BEFORE it looks at the clock, so it
// cannot go green on a faster wrong answer either.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/tokenizer.h"

using nlohmann::json;
using vllm::tok::Tokenizer;

namespace {

using Ids = std::vector<int32_t>;

const std::string kGoldenDir =
    std::string(PARITY_GOLDENS_DIR) + "/bpe_equivalence";

// The 1/5/15-minute load average, printed beside every timing this binary
// reports. A wall-clock figure with no load beside it is the defect that cost
// this row three rounds of irreproducible constants.
std::string LoadAverage() {
  std::ifstream in("/proc/loadavg");
  if (!in) return "UNKNOWN";
  std::string one, five, fifteen;
  if (!(in >> one >> five >> fifteen)) return "UNKNOWN";
  return one + "/" + five + "/" + fifteen;
}

const json& Golden() {
  static const json doc = [] {
    std::ifstream in(kGoldenDir + "/encodings.json", std::ios::binary);
    REQUIRE_MESSAGE(in.good(), "missing golden: " << kGoldenDir
                                                  << "/encodings.json");
    return json::parse(in);
  }();
  return doc;
}

// One load per arm for the whole binary. The Qwen3.6 tokenizer.json is ~20 MB /
// 248k vocab; parsing it per case would dominate the run.
const Tokenizer& Arm(const std::string& name) {
  static const Tokenizer qwen36 = Tokenizer::FromHfJson(
      std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36/tokenizer.json");
  static const Tokenizer mistral = Tokenizer::FromHfJson(
      std::string(PARITY_GOLDENS_DIR) + "/tokenizer_mistral/tokenizer.json");
  if (name == "qwen36") return qwen36;
  if (name == "mistral") return mistral;
  FAIL("unknown arm \"" << name << "\"");
  return qwen36;
}

Ids ToIds(const json& j) {
  Ids out;
  out.reserve(j.size());
  for (const auto& v : j) out.push_back(v.get<int32_t>());
  return out;
}

// Repeats `unit` and truncates to exactly `bytes`. Mirrors `repeat_to` in
// tools/parity/dump_bpe_equivalence.py. The two are never compared directly:
// if they disagree, the id vectors disagree and the case reds, which is the
// point of recording the ids rather than a hash of the text.
std::string RepeatTo(const std::string& unit, size_t bytes) {
  std::string out;
  out.reserve(bytes + unit.size());
  while (out.size() < bytes) out += unit;
  out.resize(bytes);
  return out;
}

// The absolute bound of spec §Tests to port item 2, in milliseconds.
//
// THE MARGIN, re-derived on 2026-08-20 on the 20-core AMD Ryzen 9 9950X3D this
// row was specified on, with the committed harness tools/bench/bpe_encode_cost.cpp:
//
//   defective, mistral english 65,536 B  25,327.8 ms  min-of-3, load 14.4/9.1/6.0
//   defective, qwen36 "a" 65,536 B       22,893.3 ms  min-of-3, load 12.4/9.9/6.6
//
// Both are SESSION READINGS, not constants: the same binary on the same input
// has moved 54% on load alone in this row's history. What survives the load is
// the SEPARATION. The bound sits an order of magnitude below the defective
// figures and two orders of magnitude above the fixed ones, so a runner would
// have to be ~100x slower than this box before a correct implementation crossed
// it, and ~11x faster than this box before the defective one stayed under it.
// Neither machine exists.
//
// It is deliberately not tighter. A bound whose margin approaches the noise of
// a shared runner measures the box rather than the code, and on a red day it
// reports the defect as fixed — which is exactly the failure spec item 3
// records for the growth-ratio assertion this replaces.
constexpr double kCostBoundMs = 2000.0;

}  // namespace

TEST_CASE("BPE ids match HF tokenizers 0.22.2 over the long-pretoken corpus") {
  const json& doc = Golden();
  const json& entries = doc.at("entries");
  REQUIRE_MESSAGE(entries.size() >= 80,
                  "corpus too small to be evidence: " << entries.size()
                                                      << " entries");
  MESSAGE("oracle: HF tokenizers "
          << doc.at("oracle").at("tokenizers").get<std::string>() << ", "
          << entries.size() << " entries x 2 arms x 2 special-token modes");

  size_t compared = 0;
  size_t longest_bytes = 0;
  for (const auto& e : entries) {
    const std::string name = e.at("name").get<std::string>();
    const std::string text = e.at("text").get<std::string>();
    longest_bytes = std::max(longest_bytes, text.size());
    for (const char* arm : {"qwen36", "mistral"}) {
      CAPTURE(name);
      CAPTURE(arm);
      const Tokenizer& tok = Arm(arm);
      CHECK(tok.Encode(text) == ToIds(e.at(std::string(arm) + "_ids")));
      // EncodeWithSpecialTokens is what the request path calls
      // (src/vllm/v1/engine/input_processor.cpp::process_inputs), so it is
      // checked as well as the bare Encode.
      CHECK(tok.EncodeWithSpecialTokens(text) ==
            ToIds(e.at(std::string(arm) + "_ids_special")));
      compared += 2;
    }
  }
  // Without this, a corpus that silently stopped loading would print
  // "assertions: 0 | 0 passed" and "Status: SUCCESS!".
  MESSAGE("compared " << compared << " id vectors; longest corpus entry "
                      << longest_bytes << " bytes");
  CHECK(compared == entries.size() * 4);
  CHECK(longest_bytes >= 8000);
}

TEST_CASE("one 65,536-byte pretoken encodes under an absolute wall-time bound") {
  const json& doc = Golden();
  const json& cost = doc.at("cost");
  REQUIRE_MESSAGE(cost.size() == 2,
                  "expected both callers, got " << cost.size());

  size_t checked = 0;
  for (const auto& c : cost) {
    const std::string name = c.at("name").get<std::string>();
    const std::string arm = c.at("arm").get<std::string>();
    CAPTURE(name);
    const std::string text =
        RepeatTo(c.at("repeat_unit").get<std::string>(),
                 c.at("bytes").get<size_t>());
    REQUIRE(text.size() == c.at("bytes").get<size_t>());
    const Tokenizer& tok = Arm(arm);  // load is not timed

    const std::string load_before = LoadAverage();
    const auto t0 = std::chrono::steady_clock::now();
    const Ids ids = tok.Encode(text);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0)
            .count();
    const std::string load_after = LoadAverage();

    // Correctness FIRST: a faster wrong answer is not a pass.
    const Ids expect = ToIds(c.at("ids"));
    CHECK(ids.size() == expect.size());
    CHECK(ids == expect);
    MESSAGE(name << ": elapsed_ms=" << elapsed_ms << " bound_ms="
                 << kCostBoundMs << " ids=" << ids.size()
                 << " load(1/5/15) before=" << load_before
                 << " after=" << load_after);
    CHECK_MESSAGE(elapsed_ms < kCostBoundMs,
                  name << ": elapsed_ms=" << elapsed_ms << " >= bound_ms="
                       << kCostBoundMs << " (load " << load_before << " -> "
                       << load_after << ")");
    ++checked;
  }
  CHECK(checked == 2);
}
