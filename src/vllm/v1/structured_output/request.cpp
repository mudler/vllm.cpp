// Ported from: vllm/v1/structured_output/request.py @ e24d1b24
// See include/vllm/v1/structured_output/request.h for scope + deferrals.
#include "vllm/v1/structured_output/request.h"

#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace vllm::v1 {

StructuredOutputKey get_structured_output_key(
    const StructuredOutputsParams& params) {
  // request.py:77-98. Precedence + spec construction mirror upstream.
  if (params.json.has_value()) {
    // Upstream json.dumps(dict) is done caller-side (see header deviation): the
    // stored string is already the schema JSON, matching the str branch.
    return {StructuredOutputOptions::kJson, *params.json};
  }
  if (params.json_object.value_or(false)) {
    return {StructuredOutputOptions::kJsonObject, ""};
  }
  if (params.regex.has_value()) {
    return {StructuredOutputOptions::kRegex, *params.regex};
  }
  if (params.choice.has_value()) {
    // Upstream json.dumps(list[str]). DEVIATION: nlohmann dump() is compact
    // (`["a","b"]`, no post-comma space) vs Python's `["a", "b"]`. This is an
    // internal cache key / spec consumed only by our own choice_as_grammar
    // (Task 4), never compared against upstream, so the spacing is immaterial.
    const nlohmann::json arr = *params.choice;
    return {StructuredOutputOptions::kChoice, arr.dump()};
  }
  if (params.grammar.has_value()) {
    return {StructuredOutputOptions::kGrammar, *params.grammar};
  }
  if (params.structural_tag.has_value()) {
    return {StructuredOutputOptions::kStructuralTag, *params.structural_tag};
  }
  throw std::runtime_error("No valid structured output parameter found");
}

std::optional<StructuredOutputRequest> StructuredOutputRequest::from_sampling_params(
    const SamplingParams* sampling_params) {
  // request.py:31-40.
  if (sampling_params == nullptr) {
    return std::nullopt;
  }
  if (!sampling_params->structured_outputs.has_value()) {
    return std::nullopt;
  }
  const StructuredOutputsParams& params = *sampling_params->structured_outputs;
  if (params.all_constraints_none()) {
    return std::nullopt;
  }
  StructuredOutputRequest req;
  req.params = params;
  return req;
}

StructuredOutputKey StructuredOutputRequest::structured_output_key() const {
  // request.py:72-74.
  return get_structured_output_key(params);
}

}  // namespace vllm::v1
