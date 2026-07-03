// Ported from: vllm/v1/request.py @ e24d1b24
// (RequestStatus lives in request.py; FinishReason lives upstream in
// vllm/v1/engine/__init__.py @ e24d1b24 and is imported by request.py — it is
// defined here so Task 3's engine I/O types reuse vllm::v1::FinishReason from
// this header rather than redefining it.)
//
// Scope: the T0 subset the V1 engine passes around (see the M1.1
// engine-core-types plan). RequestStatus / FinishReason enum sets, the
// ordering-based is_finished detection, and the status->finish_reason mapping
// are mirrored 1:1 with upstream. Request itself carries only the T0 fields the
// scheduler / model runner / output processor read + mutate.
//
// DEFERRED upstream Request state, intentionally omitted here — later units
// slot these in without reshaping the struct:
//   - prompt_embeds / prompt_is_token_ids / _prompt_embeds_per_block_hashes,
//     mm_features (multimodal), pooling_params, structured_output_request,
//     lora_request, cache_salt (prefix caching salt), block_hashes /
//     _block_hasher / update_block_hashes (M1.2 BlockPool), events /
//     stop_reason / kv_transfer_params, spec_token_ids, priority /
//     client_index / __lt__ (priority scheduling), streaming / resumable
//     state, prefill_stats, async-scheduling counters
//     (num_output_placeholders, async_tokens_to_discard,
//     next_decode_eligible_step, last_sched_seq), num_nans_in_logits,
//     num_preemptions, max_tokens (derived from sampling_params).
//   - the read-only ConstantList views (output_token_ids / all_token_ids):
//     here output_token_ids is a plain vector mutated only via
//     AppendOutputToken; _all_token_ids is not materialized (NumTokens is
//     computed as prompt + output, which coincides in the no-prompt-embeds
//     T0 path).
//
// DEVIATIONS, recorded:
//   - The model's EOS token id is NOT stored on Request (upstream Request does
//     not store it either). It lives on SamplingParams::eos_token_id (upstream
//     SamplingParams._eos_token_id, engine-populated), and the M1.x stop check
//     reads request.sampling_params.eos_token_id exactly as upstream check_stop
//     (vllm/v1/core/sched/utils.py) does.
//   - Direct constructor (id, prompt_ids, sampling_params, arrival_time).
//     Upstream builds via Request.from_engine_core_request; Task 3 adds the
//     EngineCoreRequest factory once that type lands.
//   - sampling_params is stored by value (already PostInit'd / validated by the
//     frontend, exactly as upstream Request just stores already-validated
//     params — construction here does NOT re-validate).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"

namespace vllm::v1 {

// Forward decl: the frontend->core message (vllm/v1/engine/types.h). Declared
// here so Request::FromEngineCoreRequest can take it by const ref without a
// circular include (types.h includes this header for FinishReason).
struct EngineCoreRequest;

// FinishReason (IntEnum): reason a request finished. Int values are
// load-bearing (compact serialization upstream). Mirrors
// vllm/v1/engine/__init__.py FinishReason.
//   stop       - a stop string / EOS / stop_token_id was emitted
//   length     - max_tokens consumed, or max_model_len reached
//   abort      - aborted by client
//   error      - retryable request-level internal error (-> 500)
//   repetition - repetitive token pattern detected (hallucination)
enum class FinishReason : int {
  kStop = 0,
  kLength = 1,
  kAbort = 2,
  kError = 3,
  kRepetition = 4,
};

// RequestStatus (IntEnum): status of a request. enum.auto() starts at 1, and
// the *ordering* is load-bearing — is_finished is `status > PREEMPTED`, so
// every value after PREEMPTED must be a FINISHED_* status. Preserve this order.
enum class RequestStatus : int {
  kWaiting = 1,
  kWaitingForStructuredOutputGrammar = 2,
  kWaitingForRemoteKvs = 3,
  kWaitingForStreamingReq = 4,
  kRunning = 5,
  kPreempted = 6,
  // Note: anything after PREEMPTED is considered a finished status.
  kFinishedStopped = 7,
  kFinishedLengthCapped = 8,
  kFinishedAborted = 9,
  kFinishedIgnored = 10,
  kFinishedError = 11,
  kFinishedRepetition = 12,
};

// Free helpers mirroring RequestStatus.is_finished /
// RequestStatus.get_finished_reason (an enum class cannot carry the Python
// staticmethods, so these are namespace-scoped free functions).
//
// is_finished: finished iff status > PREEMPTED (relies on enum ordering).
bool IsFinished(RequestStatus status);
// get_finished_reason: _FINISHED_REASON_MAP.get(status). Returns nullopt for
// statuses not in the map. NOTE (1:1 with upstream): the map also contains
// WAITING_FOR_STREAMING_REQ -> STOP, which is NOT a finished status, so
// GetFinishedReason can return a reason for a non-finished status.
std::optional<FinishReason> GetFinishedReason(RequestStatus status);

// A generation request tracked by the V1 engine (T0 field subset). The
// scheduler / model runner mutate this in place exactly as upstream does.
struct Request {
  Request(std::string request_id, std::vector<int32_t> prompt_token_ids,
          SamplingParams sampling_params, double arrival_time);

  // from_engine_core_request: build a Request from the frontend->core message.
  // Mirrors upstream Request.from_engine_core_request for the T0 fields
  // (request_id, prompt_token_ids, sampling_params, arrival_time); status
  // starts kWaiting, num_computed_tokens 0, output empty. The params arrived
  // already PostInit'd / validated by the frontend, so this does NOT
  // re-validate (upstream's factory doesn't either).
  static Request FromEngineCoreRequest(const EngineCoreRequest& request);

  std::string request_id;
  std::vector<int32_t> prompt_token_ids;
  // Already PostInit'd / validated by the frontend (upstream stores the
  // already-validated params; construction here does not re-validate). The
  // model's EOS token id (for the stop check) rides on sampling_params, as
  // sampling_params.eos_token_id — read it there, matching upstream check_stop.
  SamplingParams sampling_params;
  // Appended to during decode via AppendOutputToken.
  std::vector<int32_t> output_token_ids;
  int num_computed_tokens = 0;
  RequestStatus status = RequestStatus::kWaiting;
  double arrival_time = 0.0;
  // Set at construction from prompt_token_ids.size() (upstream:
  // length_from_prompt_token_ids_or_embeds).
  int num_prompt_tokens = 0;

  // num_tokens: len(_all_token_ids) == prompt + output (no prompt_embeds in T0).
  int NumTokens() const;
  // num_output_tokens: len(_output_token_ids).
  int NumOutputTokens() const;

  // append_output_token_ids(int | list[int]): append the sampled token(s) to
  // output_token_ids. (Upstream also mirrors into _all_token_ids and updates
  // block hashes; NumTokens recomputes here and block hashing is deferred.)
  void AppendOutputToken(int32_t token_id);
  void AppendOutputToken(const std::vector<int32_t>& token_ids);

  // is_finished / get_finished_reason: delegate to RequestStatus.
  bool IsFinished() const;
  std::optional<FinishReason> GetFinishedReason() const;
};

}  // namespace vllm::v1
