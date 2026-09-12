// The gate on the production token gate's reference-set predicate (#3094).
//
// vllm.cpp original (test harness); no upstream mirror. The production gate in
// `tests/vllm/models/test_rocm_moe_bf16.cpp` now accepts a native request
// sequence only as a whole-sequence member of the reference's captured
// configuration set. That predicate is pure host code in
// `support/rocm_moe_reference_set.h`, so its four required outcomes are gated
// here without a device:
//
//   1. the concurrency-1 member passes and the matched configuration is named;
//   2. the concurrency-2 member passes;
//   3. a change at a position where the reference configurations AGREE fails,
//      because those positions stay exact under every configuration;
//   4. a per-position mixture of the two members fails, because membership is
//      whole-sequence and never per-position.
//
// The values below are the captured length-33, length-3, and length-1 records
// from `oracle-selection-6/production.json`, whose pinned primary emits both
// `[66,1,70,57,33,81,63,69]` and `[66,1,70,57,33,81,118,66]` for request 0
// depending on the captured concurrency.
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

#include "support/rocm_moe_reference_set.h"

namespace {

using rocm_moe_reference_set::Collect;
using rocm_moe_reference_set::Compare;
using rocm_moe_reference_set::Comparison;
using rocm_moe_reference_set::Configuration;

const std::vector<int32_t> kConcurrencyOne{66, 1, 70, 57, 33, 81, 63, 69};
const std::vector<int32_t> kConcurrencyTwo{66, 1, 70, 57, 33, 81, 118, 66};
const std::vector<int32_t> kSecondRequest{63, 92, 85, 72, 62, 92, 85, 72};
const std::vector<int32_t> kLengthOne{81, 118, 105, 72, 105, 72, 72, 72};

// One record per (length, concurrency, repeat), each holding one token sequence
// per request. Request 1 exists only in the concurrency-2 records, and the
// length-33 repeat-1 record is a decoy for the repeat filter.
nlohmann::json CapturedRuns() {
  return nlohmann::json::parse(R"([
    {"length": 33, "concurrency": 1, "repeat": 0,
     "tokens": [[66,1,70,57,33,81,63,69]]},
    {"length": 33, "concurrency": 2, "repeat": 0,
     "tokens": [[66,1,70,57,33,81,118,66],[63,92,85,72,62,92,85,72]]},
    {"length": 33, "concurrency": 1, "repeat": 1, "tokens": [[1,2,3]]},
    {"length": 1, "concurrency": 1, "repeat": 0,
     "tokens": [[81,118,105,72,105,72,72,72]]},
    {"length": 1, "concurrency": 2, "repeat": 0,
     "tokens": [[81,118,105,72,105,72,72,72],[27,77,59,2,115,77,59,97]]},
    {"length": 3, "concurrency": 1, "repeat": 0,
     "tokens": [[72,84,66,66,66,66,66,25]]},
    {"length": 3, "concurrency": 2, "repeat": 0,
     "tokens": [[72,84,66,66,66,66,66,25],[64,78,19,26,2,35,1,64]]}
  ])");
}

// The name of the member a comparison reports, or -1 for no match.
int MatchedConcurrency(const std::vector<Configuration>& reference,
                       const Comparison& comparison) {
  if (!comparison.pass()) return -1;
  return reference[static_cast<size_t>(comparison.matched)].concurrency;
}

}  // namespace

TEST_CASE("reference set collects every captured configuration of one workload") {
  const auto runs = CapturedRuns();
  const auto length_33 = Collect(runs, 33, 0, 0);
  REQUIRE(length_33.size() == 2);
  CHECK(length_33[0].concurrency == 1);
  CHECK(length_33[0].tokens == kConcurrencyOne);
  CHECK(length_33[1].concurrency == 2);
  CHECK(length_33[1].tokens == kConcurrencyTwo);
  CHECK(rocm_moe_reference_set::Disagreements(length_33) == std::vector<int>{6, 7});

  // Request 1 exists only at concurrency 2, so its set is a singleton.
  const auto request_one = Collect(runs, 33, 0, 1);
  REQUIRE(request_one.size() == 1);
  CHECK(request_one[0].concurrency == 2);
  CHECK(request_one[0].tokens == kSecondRequest);
  CHECK(rocm_moe_reference_set::Disagreements(request_one).empty());

  // The repeat selects the record, so the decoy record cannot leak in.
  const auto other_repeat = Collect(runs, 33, 1, 0);
  REQUIRE(other_repeat.size() == 1);
  CHECK(other_repeat[0].tokens == std::vector<int32_t>{1, 2, 3});

  // Lengths 1 and 3 agree across concurrency, so their sets hold no
  // disagreement even though they hold two records each.
  const auto length_one = Collect(runs, 1, 0, 0);
  REQUIRE(length_one.size() == 2);
  CHECK(length_one[0].tokens == kLengthOne);
  CHECK(length_one[1].tokens == kLengthOne);
  CHECK(rocm_moe_reference_set::Disagreements(length_one).empty());

  const auto length_three = Collect(runs, 3, 0, 0);
  REQUIRE(length_three.size() == 2);
  CHECK(length_three[0].tokens == length_three[1].tokens);
  CHECK(rocm_moe_reference_set::Disagreements(length_three).empty());

  // A request index that no capture reaches collects nothing, which the
  // production gate refuses with REQUIRE(!reference_set.empty()).
  CHECK(Collect(runs, 33, 0, 2).empty());
}

TEST_CASE("the concurrency-1 member passes and reports its configuration") {
  const auto reference = Collect(CapturedRuns(), 33, 0, 0);

  const auto same = Compare(reference, kConcurrencyOne, 1);
  CHECK(same.pass());
  CHECK(MatchedConcurrency(reference, same) == 1);
  CHECK(same.same_configuration_match);

  // The recorded tie workload: the native run equals the concurrency-1 member
  // and the concurrency-2 record disagrees, so the gate passes while the
  // same-configuration comparison is reported false.
  const auto other = Compare(reference, kConcurrencyOne, 2);
  CHECK(other.pass());
  CHECK(MatchedConcurrency(reference, other) == 1);
  CHECK(other.same_configuration_match == false);
  CHECK(other.same_configuration == 1);
}

TEST_CASE("the concurrency-2 member passes") {
  const auto reference = Collect(CapturedRuns(), 33, 0, 0);
  const auto comparison = Compare(reference, kConcurrencyTwo, 2);
  CHECK(comparison.pass());
  CHECK(MatchedConcurrency(reference, comparison) == 2);
  CHECK(comparison.same_configuration_match);
}

TEST_CASE("a change at a position where the reference configurations agree fails") {
  const auto reference = Collect(CapturedRuns(), 33, 0, 0);

  // Position 3 is equal in both captured configurations, so no member explains
  // the change.
  auto changed_stable = kConcurrencyOne;
  changed_stable[3] = 34;
  const auto stable = Compare(reference, changed_stable, 1);
  CHECK(stable.pass() == false);
  CHECK(stable.same_configuration_match == false);
  CHECK(rocm_moe_reference_set::Disagreements(reference) == std::vector<int>{6, 7});

  // A change at a position where the configurations disagree fails too: the
  // position stays exact inside each member, and membership is whole-sequence.
  auto changed_tie = kConcurrencyTwo;
  changed_tie[7] = 69;
  const auto tie = Compare(reference, changed_tie, 2);
  CHECK(tie.pass() == false);
  CHECK(tie.same_configuration_match == false);

  const auto shorter = Compare(reference, std::vector<int32_t>{66, 1, 70}, 1);
  CHECK(shorter.pass() == false);
}

TEST_CASE("a per-position mix of two members fails") {
  const auto reference = Collect(CapturedRuns(), 33, 0, 0);

  // The first six tokens agree; the mix takes one member's token at position 6
  // and the other member's token at position 7, so it equals neither member.
  auto first_mix = kConcurrencyOne;
  first_mix[6] = kConcurrencyTwo[6];
  const auto mix_one = Compare(reference, first_mix, 2);
  CHECK(mix_one.pass() == false);
  CHECK(mix_one.same_configuration_match == false);
  CHECK(rocm_moe_reference_set::Disagreements(reference) == std::vector<int>{6, 7});

  auto second_mix = kConcurrencyTwo;
  second_mix[7] = kConcurrencyOne[7];
  const auto mix_two = Compare(reference, second_mix, 2);
  CHECK(mix_two.pass() == false);
  CHECK(mix_two.same_configuration_match == false);

  // Every position of a mix is individually present in some member, and the
  // whole sequence still matches none.
  CHECK((first_mix[6] == kConcurrencyTwo[6] && first_mix[7] == kConcurrencyOne[7]));
  CHECK(first_mix != kConcurrencyOne);
  CHECK(first_mix != kConcurrencyTwo);
}

TEST_CASE("a singleton reference set keeps its positions exact") {
  const auto runs = CapturedRuns();

  // Request 1 is captured once. Membership in a singleton is equality.
  const auto second_request = Collect(runs, 33, 0, 1);
  const auto exact = Compare(second_request, kSecondRequest, 2);
  CHECK(exact.pass());
  CHECK(exact.same_configuration_match);
  CHECK(exact.disagreements.empty());

  auto changed = kSecondRequest;
  changed[5] = 93;
  const auto inexact = Compare(second_request, changed, 2);
  CHECK(inexact.pass() == false);
  CHECK(inexact.same_configuration_match == false);

  // Lengths 1 and 3 hold two records that agree, so a change fails there too.
  const auto length_one = Collect(runs, 1, 0, 0);
  auto changed_length_one = kLengthOne;
  changed_length_one[7] = 73;
  const auto length_one_failure = Compare(length_one, changed_length_one, 2);
  CHECK(length_one_failure.pass() == false);
  CHECK(length_one_failure.disagreements.empty());

  const auto length_three = Collect(runs, 3, 0, 0);
  CHECK(Compare(length_three, length_three[1].tokens, 2).pass());
  auto changed_length_three = length_three[1].tokens;
  changed_length_three[0] = 73;
  const auto length_three_failure = Compare(length_three, changed_length_three, 2);
  CHECK(length_three_failure.pass() == false);
  CHECK(length_three_failure.disagreements.empty());

  const auto built = std::vector<Configuration>{{1, kConcurrencyOne}};
  CHECK(Compare(built, kConcurrencyOne, 1).pass());
  CHECK(Compare(built, kConcurrencyTwo, 1).pass() == false);
}

TEST_CASE("an empty reference set fails every run") {
  const auto empty = Compare({}, kConcurrencyOne, 1);
  CHECK(empty.pass() == false);
  CHECK(empty.disagreements.empty());
  CHECK(empty.same_configuration == -1);
}
