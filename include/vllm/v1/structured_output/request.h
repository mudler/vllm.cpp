// Ported from: vllm/v1/structured_output/request.py @ e24d1b24
//
// Scope (M3.4 Task 1): the per-request structured-output handle
// (StructuredOutputRequest) + the (type, spec) key derivation
// (get_structured_output_key). NO manager / scheduler wiring (Task 2).
//
// DEFERRED (marked, cite upstream), slotted in later without reshaping:
//   - the async-compile Future path: upstream `_grammar` is
//     `Future[StructuredOutputGrammar] | StructuredOutputGrammar | None`
//     (request.py:24) with is_grammar_ready / _check_grammar_completion
//     (request.py:42-64). At T0 the manager compiles SYNCHRONOUSLY (plan Global
//     Constraints), so `grammar` is a plain owned handle set by the manager once
//     compiled; the Future/ThreadPoolExecutor path is deferred.
//   - reasoning/thinking gating fields reasoning_ended /
//     reasoning_parser_kwargs / reasoner (request.py:26-29): the
//     should_advance/should_fill_bitmask gating returns true at T0 (plan), so
//     these are omitted here.
#pragma once

#include <memory>
#include <optional>

#include "vllm/sampling_params.h"
#include "vllm/v1/structured_output/backend_types.h"

namespace vllm::v1 {

// get_structured_output_key (request.py:77-98): map the structured-output
// constraint fields to the (StructuredOutputOptions, spec) cache key. Throws if
// no constraint is set (upstream ValueError). Precedence mirrors upstream:
// json > json_object > regex > choice > grammar > structural_tag.
StructuredOutputKey get_structured_output_key(
    const StructuredOutputsParams& params);

// StructuredOutputRequest (request.py:21-74): the per-request structured-output
// state. Holds the constraint params, the compiled grammar (set later by the
// manager), and exposes the cache key.
struct StructuredOutputRequest {
  // params (request.py:23): the structured-output constraint spec.
  StructuredOutputsParams params;

  // _grammar (request.py:24): the compiled per-request grammar, owned. Null
  // until the manager compiles it (SYNCHRONOUSLY at T0 — see file header). The
  // Future async path is deferred.
  std::unique_ptr<StructuredOutputGrammar> grammar;

  // from_sampling_params (request.py:31-40): build a StructuredOutputRequest
  // from a request's SamplingParams, or nullopt when there is no structured
  // constraint. Mirrors upstream's null / all_constraints_none short-circuit.
  static std::optional<StructuredOutputRequest> from_sampling_params(
      const SamplingParams* sampling_params);

  // structured_output_key (request.py:72-74, a cached_property upstream): the
  // (type, spec) cache key derived from `params`.
  StructuredOutputKey structured_output_key() const;
};

}  // namespace vllm::v1
