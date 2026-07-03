// Ported from: vllm/outputs.py @ e24d1b24
//
// The public request-result types the offline `LLM` API and the OpenAI server
// return: CompletionOutput (one generated sequence) and RequestOutput (the
// per-request container). These are plain value carriers; field names, order,
// defaults and the one behavioral helper (finished / Finished) are mirrored 1:1
// with upstream for the T0 generate path.
//
// Scope = the generate path. The OutputProcessor (M1.8) fills these from the
// EngineCoreOutput deltas honoring RequestOutputKind (cumulative / delta /
// final-only); here we provide only the data type + a straightforward
// constructor. The upstream RequestOutput.from_seq_group / new() factory and
// RequestOutput.add() aggregation logic are OutputProcessor territory and are
// NOT ported here (noted so the porter of M1.8 knows where they land).
//
// DEFERRED upstream state, intentionally omitted — later units slot these in
// without reshaping the structs:
//   CompletionOutput: routed_experts (np.ndarray [seq_len,layer_num,topk]),
//     lora_request; logprobs detail (SampleLogprobs payload) is kept as an
//     opaque optional flag until the sampler/logprobs unit lands.
//   RequestOutput: prompt_logprobs (PromptLogprobs detail), metrics
//     (RequestStateStats), lora_request, encoder_prompt /
//     encoder_prompt_token_ids (encoder/decoder models), num_cached_tokens
//     (prefix-cache hit count), kv_transfer_params (P/D remote K/V), the
//     forward-compat **kwargs warn, and the STREAM_FINISHED sentinel.
//   Pooling/embedding result variants (PoolingOutput, PoolingRequestOutput,
//     EmbeddingOutput / EmbeddingRequestOutput, ClassificationOutput /
//     ClassificationRequestOutput, ScoringOutput / ScoringRequestOutput) are
//     the non-generate task heads and are NOT ported here.
//   __repr__ has no C++ analogue.
//
// DEVIATIONS, recorded:
//   - CompletionOutput.finish_reason is a STRING upstream (e.g. "stop" /
//     "length"), derived from the V1 FinishReason IntEnum via str() ==
//     FINISH_REASON_STRINGS[value]. We store the string form to match, and
//     provide FinishReasonToString + CompletionOutput::SetFinishReason using
//     that SAME mapping (vllm/v1/engine/__init__.py FINISH_REASON_STRINGS).
//   - CompletionOutput.stop_reason is int | str | None upstream; represented
//     here as std::optional<std::string> (same T0 deviation as EngineCoreOutput
//     in vllm/v1/engine/types.h — a stop_token_id match stringifies its id, a
//     stop-string match carries the string). A std::variant would be a closer
//     union but is heavier than a T0 value carrier warrants.
//   - RequestOutput.prompt_token_ids is list[int] | None upstream; the T0
//     pure-token generate path always has a list, so it is a plain vector here
//     (empty vector == the upstream None/[] case).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/v1/request.h"  // vllm::v1::FinishReason (for the string mapping)

namespace vllm {

// str(FinishReason): FINISH_REASON_STRINGS[value] upstream
// (vllm/v1/engine/__init__.py). These strings form part of the external API
// (RequestOutput.finish_reason). Mapping is load-bearing — match exactly.
std::string FinishReasonToString(vllm::v1::FinishReason reason);

// CompletionOutput (@dataclass): the output data of one completion of a
// request. Field order + defaults mirror the upstream dataclass.
struct CompletionOutput {
  // The index of the output in the request (0..n-1 for n sequences).
  int index = 0;
  // The generated output text.
  std::string text;
  // The token IDs of the generated output text.
  std::vector<int32_t> token_ids;
  // The cumulative log probability of the generated output text.
  std::optional<double> cumulative_logprob;
  // SampleLogprobs | None upstream. Payload deferred; the engaged/disengaged
  // state is preserved so downstream branching still ports (opaque flag).
  std::optional<bool> logprobs;
  // The reason the sequence finished, as the upstream STRING form ("stop" /
  // "length" / ...). None while still generating. Set via SetFinishReason to
  // apply the FinishReason -> string mapping upstream uses.
  std::optional<std::string> finish_reason;
  // The stop string or token id that caused the completion to stop; None for
  // any other reason (including EOS). int | str | None upstream — see header.
  std::optional<std::string> stop_reason;

  // finished (method upstream): finish_reason is not None.
  bool Finished() const { return finish_reason.has_value(); }

  // Set finish_reason from the V1 FinishReason using the upstream str() mapping
  // (FINISH_REASON_STRINGS). Mirrors how the OutputProcessor stringifies the
  // EngineCoreOutput.finish_reason when building CompletionOutput.
  void SetFinishReason(vllm::v1::FinishReason reason) {
    finish_reason = FinishReasonToString(reason);
  }
};

// RequestOutput: the output data of a completion request to the LLM. Carries
// the T0 generate-path fields of the upstream __init__.
struct RequestOutput {
  // The unique ID of the request.
  std::string request_id;
  // The prompt string of the request. None if not available (str | None).
  std::optional<std::string> prompt;
  // The token IDs of the prompt (list[int] | None upstream; empty == None/[]).
  std::vector<int32_t> prompt_token_ids;
  // The output sequences of the request (one per requested `n`).
  std::vector<CompletionOutput> outputs;
  // Whether the whole request is finished.
  bool finished = false;

  // Convenience accessor mirroring the `finished` attribute (upstream exposes
  // the plain attribute; provided here for parity with the *Output helpers).
  bool Finished() const { return finished; }
};

}  // namespace vllm
