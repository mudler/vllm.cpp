// Ported from: vllm/v1/engine/__init__.py + vllm/v1/outputs.py @ e24d1b24
//
// The EngineCore I/O message types the V1 engine passes across its process
// boundary: EngineCoreRequest (frontend -> core), SamplerOutput /
// ModelRunnerOutput (runner -> scheduler), EngineCoreOutput / EngineCoreOutputs
// (core -> frontend). These are plain value carriers; field names, defaults and
// the one behavioral helper (finished/Finished) are mirrored 1:1 with upstream.
//
// Scope = the T0 subset the scheduler's update_from_output (M1.4) and the
// OutputProcessor (M1.8) actually consume. FinishReason is REUSED from
// vllm/v1/request.h (upstream defines it in vllm/v1/engine/__init__.py and
// request.py imports it); it is NOT redefined here.
//
// DEFERRED upstream fields, intentionally omitted — later units slot these in
// without reshaping the structs:
//   EngineCoreRequest: mm_features (multimodal), pooling_params, lora_request,
//     cache_salt, data_parallel_rank, prompt_embeds, prompt_is_token_ids,
//     client_index, current_wave, trace_headers, resumable,
//     external_req_id, reasoning_ended / reasoning_parser_kwargs,
//     abort_immediately, and the params property. (eos_token_id is NOT a field
//     upstream either — it rides on sampling_params.eos_token_id.)
//   SamplerOutput: logprobs_tensors now carries the real LogprobsTensors payload
//     (vllm/v1/outputs.py, ported at M1.7); the sampler's gather_logprobs fills
//     it. It stays std::optional (None => no logprobs requested this step).
//   ModelRunnerOutput: logprobs (LogprobsLists), prompt_logprobs_dict,
//     pooler_output, kv_connector_output / ec_connector_output (P/D KV
//     transfer), num_nans_in_logits, cudagraph_stats, routed_experts, and the
//     with_kv_conn_output_only / EMPTY_MODEL_RUNNER_OUTPUT helpers.
//   EngineCoreOutput: new_logprobs / new_prompt_logprobs_tensors,
//     pooling_output, events (EngineCoreEvent), kv_transfer_params,
//     trace_headers, prefill_stats, routed_experts, num_nans_in_logits.
//   EngineCoreOutputs: scheduler_stats (SchedulerStats), utility_output,
//     finished_requests, wave_complete / start_wave (DP wave signalling), and
//     the __post_init__ monotonic-timestamp default (the frontend stamps it).
//
// DEVIATIONS, recorded:
//   - SamplerOutput.sampled_token_ids is a torch.Tensor [num_reqs,
//     max_num_generated_tokens] upstream; here it is the list-of-lists form
//     (vector<vector<int32_t>>) matching ModelRunnerOutput.sampled_token_ids,
//     which is how the scheduler reads it. T0 non-spec decode is [num_reqs, 1].
//     PLACEHOLDER_TOKEN_ID (-1) padding is a device-tensor concern that does
//     not apply to the ragged list form.
//   - EngineCoreOutput.stop_reason is int | str | None upstream; represented
//     here as std::optional<std::string> (the OutputProcessor passes it through
//     to CompletionOutput.stop_reason). A stop_token_id match stringifies its
//     id; a stop-string match carries the string. A std::variant would be a
//     closer union but is heavier than a T0 value carrier warrants.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/outputs.h"  // vllm::v1::LogprobsTensors (SamplerOutput payload)
#include "vllm/v1/request.h"  // vllm::v1::FinishReason (reused, not redefined)

namespace vllm::v1 {

// PLACEHOLDER_TOKEN_ID: upstream pads the SamplerOutput token tensor with -1.
// Kept for reference; the list-of-lists form here is ragged and unpadded.
inline constexpr int32_t kPlaceholderTokenId = -1;

// EngineCoreRequest: the frontend -> EngineCore message (T0 field subset).
// msgspec.Struct upstream; a plain value carrier here.
struct EngineCoreRequest {
  std::string request_id;
  // Upstream is list[int] | None; T0 pure-token path always has a list.
  std::vector<int32_t> prompt_token_ids;
  // Already PostInit'd / validated by the frontend (the InputProcessor runs
  // __post_init__ before building this message), so Request::FromEngineCoreRequest
  // does NOT re-validate. The model's EOS id rides on sampling_params.eos_token_id.
  SamplingParams sampling_params;
  double arrival_time = 0.0;
  // priority (engine/__init__.py EngineCoreRequest.priority): the scheduling
  // priority carried from the frontend request (OpenAI `priority` field) into
  // Request.priority. Default 0 keeps FCFS parity when unset.
  int priority = 0;
};

// SamplerOutput (vllm/v1/outputs.py): the raw sampler result for a step.
struct SamplerOutput {
  // [num_reqs, max_num_generated_tokens]; T0 non-spec decode is [num_reqs, 1].
  std::vector<std::vector<int32_t>> sampled_token_ids;
  // logprobs_tensors upstream (LogprobsTensors | None): the sampler's
  // gather_logprobs payload, None when no logprobs were requested this step.
  std::optional<LogprobsTensors> logprobs_tensors;
  // ENG-ASYNC-SCHED W3: set when Sampler::forward was given a device out-tensor
  // and produced the sampled ids DEVICE-RESIDENT (all-greedy fast path) WITHOUT a
  // host download — `sampled_token_ids` is then intentionally empty and the
  // async-output D2H (AsyncGPUModelRunnerOutput) owns the single copy to host.
  // Upstream keeps sampler_output.sampled_token_ids on the GPU unconditionally
  // (async_utils.py:31); this flag records that we took the same device-resident
  // route on our greedy gate path (sync path leaves it false, byte-identical).
  bool sampled_on_device = false;
};

// ModelRunnerOutput (vllm/v1/outputs.py): the runner -> scheduler result,
// serialized across the process boundary. update_from_output reads req_ids
// (ordering), req_id_to_index and sampled_token_ids at T0.
struct ModelRunnerOutput {
  // [num_reqs] — the per-request ordering the scheduler iterates.
  std::vector<std::string> req_ids;
  // req_id -> index into the num_reqs-major arrays.
  std::map<std::string, int> req_id_to_index;
  // num_reqs x num_generated_tokens (ragged; per-request length can differ
  // under speculative/jump decode). T0 non-spec decode is [num_reqs, 1].
  std::vector<std::vector<int32_t>> sampled_token_ids;
};

// EngineCoreOutput (vllm/v1/engine/__init__.py): the per-request core -> frontend
// delta. msgspec.Struct upstream; a plain value carrier here.
struct EngineCoreOutput {
  std::string request_id;
  std::vector<int32_t> new_token_ids;
  // finish_reason is None until the request finishes (reused FinishReason).
  std::optional<FinishReason> finish_reason;
  // int | str | None upstream — see DEVIATIONS in the file header.
  std::optional<std::string> stop_reason;

  // finished (property): a request is finished iff finish_reason is set.
  bool Finished() const { return finish_reason.has_value(); }
};

// EngineCoreOutputs (vllm/v1/engine/__init__.py): the batched core -> frontend
// message for one engine step.
struct EngineCoreOutputs {
  int engine_index = 0;
  // [num_reqs]
  std::vector<EngineCoreOutput> outputs;
  // Monotonic timestamp; upstream __post_init__ stamps time.monotonic() when
  // left at 0.0 — deferred (the frontend sets it), default preserved.
  double timestamp = 0.0;
};

}  // namespace vllm::v1
