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
//     mm_features (multimodal), pooling_params,
//     lora_request, cache_salt (prefix caching salt), events /
//     kv_transfer_params,
//     client_index, streaming / resumable
//     state, prefill_stats, last_sched_seq (defer_block_free fence),
//     num_nans_in_logits.
//     (num_preemptions is un-deferred below: PrefixCacheStats.record needs it
//     to route a lookup into the mutually-exclusive preempted_* counters —
//     vllm/v1/metrics/stats.py:130-142, called at
//     vllm/v1/core/kv_cache_manager.py:234-240.)
//     (The async-scheduling counters num_output_placeholders /
//     async_tokens_to_discard / next_decode_eligible_step and max_tokens are
//     un-deferred below for ENG-ASYNC-SCHED — spec async-serving.md W3.)
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
#include "vllm/v1/core/kv_cache_utils.h"       // BlockHash, BlockHasher
#include "vllm/v1/structured_output/request.h"  // StructuredOutputRequest

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
  // block_hasher mirrors upstream's `block_hasher=None` default: null => prefix
  // caching off (update_block_hashes() is a no-op, block_hashes stays empty).
  // `priority` mirrors upstream Request(priority=0): the scheduling priority
  // used by the priority policy (lower value = handled first). It is appended
  // after block_hasher (rather than kept in upstream's kwargs position) so the
  // existing positional call sites are unaffected; default 0 keeps FCFS parity.
  Request(std::string request_id, std::vector<int32_t> prompt_token_ids,
          SamplingParams sampling_params, double arrival_time,
          BlockHasher block_hasher = nullptr, int priority = 0);

  // from_engine_core_request: build a Request from the frontend->core message.
  // Mirrors upstream Request.from_engine_core_request for the T0 fields
  // (request_id, prompt_token_ids, sampling_params, arrival_time); status
  // starts kWaiting, num_computed_tokens 0, output empty. The params arrived
  // already PostInit'd / validated by the frontend, so this does NOT
  // re-validate (upstream's factory doesn't either). block_hasher is injected
  // by the engine exactly as upstream from_engine_core_request(request,
  // block_hasher).
  static Request FromEngineCoreRequest(const EngineCoreRequest& request,
                                       BlockHasher block_hasher = nullptr);

  std::string request_id;
  std::vector<int32_t> prompt_token_ids;
  // Already PostInit'd / validated by the frontend (upstream stores the
  // already-validated params; construction here does not re-validate). The
  // model's EOS token id (for the stop check) rides on sampling_params, as
  // sampling_params.eos_token_id — read it there, matching upstream check_stop.
  SamplingParams sampling_params;
  // structured_output_request (request.py:87-92): the per-request structured
  // output state (constraint params + the compiled grammar), or nullopt when the
  // request has no structured-output constraint. Populated at construction from
  // sampling_params.structured_outputs; the grammar is compiled later by the
  // StructuredOutputManager (grammar_init). Its unique_ptr grammar makes Request
  // move-only (it was never copied — the scheduler owns it via unique_ptr).
  std::optional<StructuredOutputRequest> structured_output_request;

  // Appended to during decode via AppendOutputToken.
  std::vector<int32_t> output_token_ids;
  // Draft (speculative) token ids proposed for the NEXT verify step (upstream
  // Request.spec_token_ids, request.py:152). Empty by default; set by
  // Scheduler::update_draft_token_ids after the drafter proposes, CONSUMED and
  // cleared in schedule() when the verify step is scheduled (mirror
  // scheduler.py:593-609). Un-deferred for SPEC-MTP (I2 scheduler plumbing).
  // With no speculator configured num_lookahead_tokens == 0 and this stays empty
  // on every request, so num_tokens_with_spec == num_tokens (inert / byte-
  // identical to the pre-spec scheduler).
  std::vector<int32_t> spec_token_ids;
  int num_computed_tokens = 0;
  // How many times this request has been preempted (upstream
  // Request.num_preemptions). Incremented by Scheduler::preempt_request; read
  // by KVCacheManager::get_computed_blocks to route the lookup into
  // PrefixCacheStats' preempted_* counters, so a preempted request's
  // guaranteed second-pass hit cannot inflate the headline hit rate.
  int num_preemptions = 0;
  RequestStatus status = RequestStatus::kWaiting;
  // stop_reason (upstream Request.stop_reason: int | str | None = None). Set by
  // check_stop (sched/utils) when a stop_token_ids match ends the request — it
  // carries the matched token id — and read by update_from_output when it builds
  // the request's EngineCoreOutput. Un-deferred from the Request deferred list
  // because M1.4 Task 4 needs it. Repetition detection's string reason
  // ("repetition_detected") is deferred with sampling_params.repetition_detection,
  // so at T0 only the int (stop-token) form ever occurs — hence std::optional<int>
  // rather than a variant<int, string>.
  std::optional<int> stop_reason;
  double arrival_time = 0.0;
  // Scheduling priority (upstream Request.priority, request.py:83). Lower value
  // = handled earlier by the priority policy; ties broken by arrival_time then
  // request_id (see RequestPriorityLess). Ignored under the default FCFS policy.
  int priority = 0;
  // Set at construction from prompt_token_ids.size() (upstream:
  // length_from_prompt_token_ids_or_embeds).
  int num_prompt_tokens = 0;

  // Whether this request is still in its prefill (context) phase, i.e. it has
  // scheduled tokens it has not yet computed. Written by the scheduler in
  // _update_after_schedule (num_computed_tokens < num_tokens + placeholders);
  // read by the DP-throttle / spec-pad paths (deferred at T0) and, from M1.4
  // Task 4 onward, by update_from_output. Un-deferred here (was in the Request
  // deferred list) because scheduler._update_after_schedule sets it 1:1.
  bool is_prefill_chunk = false;

  // --- Async-scheduling counters (request.py:141-146) --------------------------
  // Un-deferred for ENG-ASYNC-SCHED (spec async-serving.md W3). All three are
  // INERT under the default synchronous Scheduler (they stay 0 and every site
  // that reads them is a no-op), so the sync path is byte-identical; only the
  // AsyncScheduler subclass mutates them.
  //
  // num_output_placeholders (request.py:141): the count of in-flight sampled
  // tokens the scheduler has already reserved a slot for but not yet observed
  // (async scheduling schedules step N+1 before N's output returns). Added in
  // AsyncScheduler::update_after_schedule, drained in update_request_with_output.
  int num_output_placeholders = 0;
  // async_tokens_to_discard (request.py:142): output frames that were in flight
  // when the request was force-preempted (reset_prefix_cache) and are now stale;
  // each returning frame is dropped until this counter drains.
  int async_tokens_to_discard = 0;
  // next_decode_eligible_step (request.py:146): the earliest scheduler step in
  // which this request may be scheduled for decode again (PP microbatching
  // cadence). 0 at T0 (single GPU, pp_size fold is inert); read by the running
  // loop guard, which is a no-op while it stays 0.
  int next_decode_eligible_step = 0;

  // num_tokens: len(_all_token_ids) == prompt + output (no prompt_embeds in T0).
  int NumTokens() const;
  // num_tokens_with_spec (request.py:251-252): len(_all_token_ids) +
  // len(spec_token_ids) — the token count the scheduler's running loop targets so
  // the pending draft tokens are scheduled for verification alongside the sampled
  // token. Equals NumTokens() when spec_token_ids is empty (no speculator), so
  // the default scheduler path is unchanged. Un-deferred for SPEC-MTP (I2).
  int NumTokensWithSpec() const {
    return NumTokens() + static_cast<int>(spec_token_ids.size());
  }
  // num_output_tokens: len(_output_token_ids).
  int NumOutputTokens() const;
  // max_tokens (request.py:106-110): the request's output-token cap. Upstream
  // stores it as a field derived from sampling_params.max_tokens (or 1 for
  // pooling). Read by the AsyncScheduler running-loop guard (scheduler.py:452).
  int MaxTokens() const { return sampling_params.max_tokens.value_or(1); }
  // all_token_ids: the prompt token ids followed by the output token ids
  // (upstream _all_token_ids materialized; here concatenated on demand — no
  // prompt_embeds in T0). Used by scheduler._make_cached_request_data for the
  // MRV1 all_token_ids connector-propagation payload.
  std::vector<int32_t> AllTokenIds() const;

  // Per-block hashes of this request's full blocks, computed at
  // hash_block_size granularity and chained over the full prefix. Populated by
  // update_block_hashes() via _block_hasher (empty when caching is off).
  std::vector<BlockHash> block_hashes;

  // The block hasher (upstream _block_hasher). Stored without binding a
  // back-reference to this Request (upstream avoids the Request->partial->Request
  // reference cycle); here it is a plain std::function taking the request by
  // const ref. Null => prefix caching off.
  BlockHasher block_hasher_;

  // append_output_token_ids(int | list[int]): append the sampled token(s) to
  // output_token_ids, then update_block_hashes() (upstream mirrors into
  // _all_token_ids too; NumTokens recomputes prompt+output here).
  void AppendOutputToken(int32_t token_id);
  void AppendOutputToken(const std::vector<int32_t>& token_ids);

  // update_block_hashes: compute block hashes for any newly-complete full
  // blocks and append them to block_hashes. No-op when _block_hasher is null.
  // Mirrors upstream Request.update_block_hashes.
  void update_block_hashes();

  // use_structured_output (request.py:243-244): whether this request carries a
  // structured-output constraint (structured_output_request is set).
  bool use_structured_output() const {
    return structured_output_request.has_value();
  }

  // is_finished / get_finished_reason: delegate to RequestStatus.
  bool IsFinished() const;
  std::optional<FinishReason> GetFinishedReason() const;
};

// Request.__lt__ (request.py:309-320): the total order the priority scheduler
// uses. A request compares "less" (i.e. is scheduled earlier) when it has a
// smaller priority; ties break on the earlier arrival_time, then the smaller
// request_id, then object identity (upstream `id(self) < id(other)`). The
// request_id + pointer tie-breakers make this a STRICT TOTAL order (distinct
// requests never compare equal), so heap pop order is deterministic and matches
// upstream heapq. Returns true iff `a` should be handled before `b`.
bool RequestPriorityLess(const Request& a, const Request& b);
inline bool RequestPriorityLess(const Request* a, const Request* b) {
  return RequestPriorityLess(*a, *b);
}

}  // namespace vllm::v1
