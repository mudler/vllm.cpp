// vllm.cpp original (test harness); no upstream mirror.
//
// BACKEND-ROCM-BF16-MOE (#3094) production token gate: the reference set.
//
// The pinned primary's greedy decode is not unique across its own legitimate
// configurations. At length 33, the concurrency-1 and concurrency-2 captures of
// the identical prompt emit `[66,1,70,57,33,81,63,69]` and
// `[66,1,70,57,33,81,118,66]` for request 0, because the two runs meet a BF16
// tie differently. A gate that compares the native run against one capture
// therefore fits one noise realization of the reference instead of the
// reference function.
//
// `Compare` implements the corrected rule: a request passes when its whole
// sequence equals one captured reference sequence (membership), never a
// position-by-position mix of two. Everything here is pure host code, so
// `tests/support/test_rocm_moe_reference_set.cpp` gates the predicate on the CPU
// and the production gate `tests/vllm/models/test_rocm_moe_bf16.cpp` applies it
// to the captured oracle records.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace rocm_moe_reference_set {

// One captured reference configuration: the concurrency of the oracle record
// and that record's generated tokens for one request.
struct Configuration {
  int concurrency = 0;
  std::vector<int32_t> tokens;
};

// The result of comparing one native request sequence with the reference set.
struct Comparison {
  // Index of the whole-sequence member the run equals, or -1 when the run
  // matches no captured configuration. The run's own configuration wins when it
  // matches, so the reported member is the strongest available statement.
  int matched = -1;
  // Index of the member captured at the run's own concurrency, or -1 when the
  // reference set holds no record for that concurrency.
  int same_configuration = -1;
  // Whether the run equals that same-configuration member. The gate reports this
  // value and never asserts it, because the reference disagrees with itself at
  // one workload.
  bool same_configuration_match = false;
  // Positions at which the captured configurations disagree with each other,
  // including a position that one configuration does not reach.
  std::vector<int> disagreements;

  bool pass() const { return matched >= 0; }
};

// Gather every captured sequence for one request from the records of the same
// length and repeat. Request 1 exists only at concurrency 2, so a record whose
// token list is shorter than the request index contributes nothing.
inline std::vector<Configuration> Collect(const nlohmann::json& runs, int length,
                                          int repeat, size_t request) {
  std::vector<Configuration> reference;
  for (const auto& record : runs) {
    if (record.at("length").get<int>() != length) continue;
    if (record.at("repeat").get<int>() != repeat) continue;
    const auto sequences = record.at("tokens").get<std::vector<std::vector<int32_t>>>();
    if (request >= sequences.size()) continue;
    reference.push_back({record.at("concurrency").get<int>(), sequences[request]});
  }
  return reference;
}

// Return every position at which the captured configurations disagree. A
// singleton set cannot disagree, so its positions stay exact.
inline std::vector<int> Disagreements(const std::vector<Configuration>& reference) {
  std::vector<int> positions;
  if (reference.size() < 2) return positions;
  size_t width = 0;
  for (const auto& configuration : reference)
    width = std::max(width, configuration.tokens.size());
  for (size_t position = 0; position < width; ++position) {
    const auto& first = reference.front().tokens;
    for (size_t i = 1; i < reference.size(); ++i) {
      const auto& other = reference[i].tokens;
      if (position >= first.size() || position >= other.size() ||
          first[position] != other[position]) {
        positions.push_back(static_cast<int>(position));
        break;
      }
    }
  }
  return positions;
}

// Compare one native request sequence with the reference set. The run passes
// only as a whole-sequence member; the per-position disagreement list never
// widens the accepted set.
inline Comparison Compare(const std::vector<Configuration>& reference,
                          const std::vector<int32_t>& run, int concurrency) {
  Comparison comparison;
  comparison.disagreements = Disagreements(reference);
  for (size_t i = 0; i < reference.size(); ++i) {
    if (reference[i].concurrency == concurrency)
      comparison.same_configuration = static_cast<int>(i);
    if (comparison.matched < 0 && reference[i].tokens == run)
      comparison.matched = static_cast<int>(i);
  }
  comparison.same_configuration_match =
      comparison.same_configuration >= 0 &&
      reference[static_cast<size_t>(comparison.same_configuration)].tokens == run;
  if (comparison.same_configuration_match)
    comparison.matched = comparison.same_configuration;
  return comparison;
}

// Render a token sequence for a gate report.
inline std::string Sequence(const std::vector<int32_t>& tokens) {
  std::string text = "[";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i != 0) text += ",";
    text += std::to_string(tokens[i]);
  }
  return text + "]";
}

// Render the reference set for a gate report.
inline std::string Describe(const std::vector<Configuration>& reference) {
  std::string text = "{";
  for (size_t i = 0; i < reference.size(); ++i) {
    if (i != 0) text += ", ";
    text += "concurrency " + std::to_string(reference[i].concurrency) + " " +
            Sequence(reference[i].tokens);
  }
  if (reference.empty()) text += "no captured configuration";
  return text + "}";
}

// Render a position list for a gate report.
inline std::string Describe(const std::vector<int>& positions) {
  std::string text = "[";
  for (size_t i = 0; i < positions.size(); ++i) {
    if (i != 0) text += ",";
    text += std::to_string(positions[i]);
  }
  return text + "]";
}

}  // namespace rocm_moe_reference_set
