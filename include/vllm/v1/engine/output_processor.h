// Ported from: vllm/v1/engine/output_processor.py @ e24d1b24
// (OutputProcessor + RequestState + OutputProcessorOutput — the T0 synchronous
// text path: incremental detokenize + string-level stop + RequestOutput
// assembly.)
//
// Scope (M1.8 Task 5): turn the per-step EngineCoreOutputs into RequestOutputs.
// This is the SYNCHRONOUS LLMEngine path (no queue): process_outputs loops the
// EngineCoreOutputs, drives our IncrementalDetokenizer.Update() (which does the
// STRING-level stop match the scheduler's token-level check_stop cannot), builds
// the streaming-delta vs full CompletionOutput/RequestOutput per RequestOutputKind,
// removes finished req states, and returns the reqs_to_abort feedback for
// requests the detokenizer stopped but EngineCore did not (output_processor.py
// :678). Mirrors OutputProcessor.process_outputs (:576-693),
// RequestState.from_new_request (:210-270), make_request_output (:272-331),
// _new_completion_output (:376-411), _new_request_output (:333-374),
// _finish_request (:695-707).
//
// DEVIATIONS vs the pinned API (recorded, use OUR names):
//   - __init__ takes a nullable tokenizer pointer + stream_interval (T0). The
//     upstream log_stats / tracing_enabled knobs are deferred (stats/tracing are
//     deferred below), so they are dropped.
//   - RequestState has no external_req_id field upstream-separate at T0: our
//     EngineCoreRequest deferred external_req_id (see v1/engine/types.h), so the
//     external id == request_id here (no parallel-sampling / streaming-input
//     remap). The external_req_ids map is kept for structural parity with
//     _finish_request even though it degenerates to a 1:1 mapping at T0.
//   - kv_transfer_params is dropped from make_request_output (our EngineCoreOutput
//     has no kv_transfer_params field — deferred in v1/engine/types.h).
//   - make_request_output returns std::optional<RequestOutput> (pooling deferred,
//     so PoolingRequestOutput never occurs; the None return still models
//     FINAL_ONLY / stream_interval hold-back).
//   - The upstream private _new_* helpers are PascalCase here (NewCompletionOutput
//     / NewRequestOutput); _finish_request -> FinishRequest.
//
// DEFERRED (marked; matches upstream so re-adding is mechanical):
//   LogprobsProcessor (sample + prompt logprobs), pooling outputs
//   (PoolingOutput / PoolingRequestOutput branch), routed_experts accumulation,
//   parallel sampling (ParentRequest / parent_requests / get_outputs),
//   streaming-input
//   chunk queue (StreamingUpdate / apply_streaming_update / resumable),
//   iteration + per-request stats (RequestStateStats / IterationStats /
//   LoRARequestStates), tracing (do_tracing), num_cached_tokens / prefill_stats,
//   LoRA and prompt_embeds. The T0 client-initiated abort path is present;
//   external/internal-id fan-out, pooling and parent-request breadth remain.
#pragma once

#include <cstdint>
#include <condition_variable>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/engine/detokenizer.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/request.h"

namespace vllm::tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h
}

namespace vllm::v1 {

// RequestOutputCollector (output_processor.py:45-105): a single-slot,
// thread-safe hand-off between AsyncLLM's output-handler thread and one
// consuming request. DELTA outputs coalesce when the producer outruns the
// consumer; cumulative outputs replace the same completion index.
class RequestOutputCollector {
 public:
  RequestOutputCollector(RequestOutputKind output_kind, std::string request_id);

  // Non-blocking producer operations (put / AsyncLLM propagate_error).
  void put(RequestOutput output);
  void put_error(std::exception_ptr error);

  // Blocking and non-blocking consumer operations (get / get_nowait).
  // Stored producer errors are rethrown on the consumer thread.
  RequestOutput get();
  std::optional<RequestOutput> get_nowait();

  bool has_output() const;
  const std::string& request_id() const { return request_id_; }

 private:
  void Merge(RequestOutput output);

  bool aggregate_ = false;
  std::string request_id_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::optional<RequestOutput> output_;
  std::exception_ptr error_;
};

// OutputProcessorOutput (@dataclass, output_processor.py:109-113). The
// synchronous return of process_outputs: the RequestOutputs to hand back to the
// caller + the request ids EngineCore must abort (detokenizer-detected stop that
// EngineCore did not itself signal). PoolingRequestOutput deferred.
struct OutputProcessorOutput {
  std::vector<RequestOutput> request_outputs;
  std::vector<std::string> reqs_to_abort;
};

// RequestState (output_processor.py:129): per-request accumulation state held by
// the OutputProcessor for the life of a request. Members are public, mirroring
// the upstream dataclass-like attributes; the deferred members are omitted (see
// the file header).
class RequestState {
 public:
  // from_new_request (:210): build a RequestState with our IncrementalDetokenizer
  // (the detokenizer branch). `tokenizer` may be nullptr (=> no detokenization);
  // if sampling_params.detokenize is false the tokenizer is dropped as upstream
  // (:223). LogprobsProcessor / pooling / parent_req are deferred; W2 supplies
  // the optional collector after this factory returns.
  static RequestState FromNewRequest(const tok::Tokenizer* tokenizer,
                                     const EngineCoreRequest& request,
                                     std::optional<std::string> prompt,
                                     int request_index, int stream_interval);

  // make_request_output (:272): assemble the streaming-delta vs full
  // CompletionOutput/RequestOutput honoring output_kind. Returns nullopt when
  // FINAL_ONLY-and-not-finished or a stream_interval hold-back suppresses this
  // step's output. kv_transfer_params deferred (see header).
  std::optional<RequestOutput> make_request_output(
      const std::vector<int32_t>& new_token_ids,
      std::optional<FinishReason> finish_reason,
      std::optional<std::string> stop_reason);

  std::string request_id;
  std::string external_req_id;  // == request_id at T0 (see header).
  int request_index = 0;
  RequestOutputKind output_kind = RequestOutputKind::kCumulative;
  std::optional<std::string> prompt;
  std::vector<int32_t> prompt_token_ids;
  size_t prompt_len = 0;
  std::unique_ptr<IncrementalDetokenizer> detokenizer;
  std::optional<int> max_tokens_param;
  bool is_prefilling = true;
  int num_cached_tokens = 0;  // deferred (no prefill_stats at T0); stays 0.
  int stream_interval = 1;
  size_t sent_tokens_offset = 0;
  // AsyncLLM only (output_processor.py RequestState.queue): null on the
  // synchronous LLMEngine path, otherwise the per-request collector.
  std::shared_ptr<RequestOutputCollector> queue;

 private:
  // _new_completion_output (:376): text/token_ids in delta vs cumulative mode.
  CompletionOutput NewCompletionOutput(std::vector<int32_t> token_ids,
                                       std::optional<FinishReason> finish_reason,
                                       std::optional<std::string> stop_reason);
  // _new_request_output (:333): wrap the CompletionOutput(s) in a RequestOutput.
  RequestOutput NewRequestOutput(const std::string& external_req_id,
                                 std::vector<CompletionOutput> outputs,
                                 bool finished);
};

// OutputProcessor (output_processor.py:417): process EngineCoreOutputs into
// synchronous RequestOutputs or W2 per-request collectors.
class OutputProcessor {
 public:
  // __init__ (:420). `tokenizer` may be nullptr (=> no detokenization). It must
  // outlive the OutputProcessor. log_stats / tracing_enabled deferred.
  explicit OutputProcessor(const tok::Tokenizer* tokenizer,
                           int stream_interval = 1);

  int get_num_unfinished_requests() const {
    return static_cast<int>(request_states_.size());
  }
  bool has_unfinished_requests() const { return !request_states_.empty(); }

  // add_request (:512): build + register a RequestState. parent_req / queue /
  // the streaming-update re-entry are deferred (T0: duplicate live ids throw).
  void add_request(
      const EngineCoreRequest& request, std::optional<std::string> prompt,
      int request_index = 0,
      std::shared_ptr<RequestOutputCollector> queue = nullptr);

  // process_outputs (:576): the per-EngineCoreOutput loop — detokenize + stop +
  // RequestOutput assembly + reqs_to_abort feedback. Stats/tracing/timestamp
  // args deferred.
  OutputProcessorOutput process_outputs(
      const EngineCoreOutputs& engine_core_outputs);

  // abort_requests (output_processor.py:534, the T0 subset): drop the tracked
  // RequestState for each id so a client-initiated abort (e.g. the C-ABI
  // streaming early-stop) fully tears the request down and it no longer counts
  // toward has_unfinished_requests(). Unknown ids are ignored. The upstream
  // parent_req / pooling / queue bookkeeping stays deferred (see the file
  // header); at T0 this is a map erase mirroring FinishRequest.
  std::vector<std::string> abort_requests(
      const std::vector<std::string>& request_ids,
      bool produce_final_output = false);

  // AsyncLLM teardown helper: abort every tracked internal request and return
  // the IDs that still need forwarding to EngineCore. The caller provides the
  // same external serialization as abort_requests/process_outputs.
  std::vector<std::string> abort_all_requests(
      bool produce_final_output = false);

  // propagate_error (output_processor.py:443-448): wake every live AsyncLLM
  // consumer with the output-handler failure. Sync states have no queue.
  void propagate_error(std::exception_ptr error);

 private:
  // _finish_request (:695): remove the finished req state from the maps.
  void FinishRequest(RequestState& req_state);

  const tok::Tokenizer* tokenizer_;
  int stream_interval_;
  std::map<std::string, std::unique_ptr<RequestState>> request_states_;
  // external_req_id -> [internal request_id, ...] (1:1 at T0, see header).
  std::map<std::string, std::vector<std::string>> external_req_ids_;
};

}  // namespace vllm::v1
