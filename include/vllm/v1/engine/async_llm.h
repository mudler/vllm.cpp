// Ported from: vllm/v1/engine/async_llm.py @ e24d1b24
// (AsyncLLM :70-745: add_request :280-410, generate :524-635,
// _run_output_handler :637-707, abort :709-745) and
// vllm/v1/engine/output_processor.py:45-105 (RequestOutputCollector).
//
// Scope (async-serving spec W2, row SERVE-ASYNC-LLM): the in-process C++
// asynchronous frontend over W1's InprocClient/EngineCoreProc queue split.
// A dedicated output-handler thread pulls EngineCoreOutputs, runs the shared
// OutputProcessor, and wakes one collector per request. Consumers may block on
// their own collector without serializing unrelated requests.
//
// DEVIATION (recorded in specs/async-serving.md D2): std::thread + condition
// variables replace asyncio tasks/queues and ZMQ. Queue ordering, per-request
// single-slot coalescing, abort-final-output behavior, and output-handler order
// remain the pinned upstream semantics. Parallel sampling, streaming input,
// pooling and DP are deferred.
//
// STATS (SERVE-METRICS, #277, specs/async-metrics.md): the output handler folds
// each step's SchedulerStats + IterationStats into an ATTACHED
// PrometheusStatLogger, mirroring async_llm.py:662-702. Opt-in: with no logger
// attached (the default) the handler takes the same no-stats path it always
// took. The config-gated metric families (spec-decode / kv-connector / mm /
// LoRA) remain deferred, as does update_scheduler_stats (LoRA-only upstream).
#ifndef VLLM_V1_ENGINE_ASYNC_LLM_H_
#define VLLM_V1_ENGINE_ASYNC_LLM_H_

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "vllm/multimodal/inputs.h"  // multimodal::MultiModalInputs (mm request)
#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/engine/core_client.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"

namespace vllm::v1 {

namespace metrics {
class PrometheusStatLogger;
}  // namespace metrics

// The value returned by add_request/generate-start. The collector is shared
// with OutputProcessor's RequestState until that request finishes or aborts.
struct AsyncRequest {
  std::string request_id;
  std::shared_ptr<RequestOutputCollector> collector;
};

// Ordered-wave admission inputs. Separate types make it impossible for one
// item to carry both prompt forms. The batch API preserves these objects'
// order in both core publication and returned collectors.
struct AsyncStringRequestInput {
  std::string request_id;
  std::string prompt;
  SamplingParams params;
  int priority = 0;
};

struct AsyncTokensRequestInput {
  std::string request_id;
  std::vector<int32_t> prompt_token_ids;
  SamplingParams params;
  int priority = 0;
};

// The second admission check lives on the lock-owned publish route, rather
// than only at the start of input preparation. This injectable seam lets a
// deterministic test model shutdown winning after the outer fast check and
// prove that no registration or core publish callback can run afterward.
template <typename Lockable, typename AlivePredicate, typename PublishCallback>
inline decltype(auto) PublishAsyncRequestWaveIfAlive(
    Lockable& admission_mutex, AlivePredicate&& alive_predicate,
    PublishCallback&& publish_callback) {
  std::lock_guard<Lockable> lock(admission_mutex);
  if (!std::forward<AlivePredicate>(alive_predicate)()) {
    throw EngineDeadError("request wave submitted to a stopped AsyncLLM");
  }
  return std::forward<PublishCallback>(publish_callback)();
}

class AsyncLLM {
 public:
  // Mirrors AsyncLLM.__init__ wiring, using caller-owned collaborators just as
  // LLMEngine does. InprocClient owns the EngineCoreProc + engine thread; this
  // object owns the output-handler thread. All referenced collaborators must
  // outlive AsyncLLM.
  //
  // max_concurrent_batches (VllmConfig.max_concurrent_batches,
  // vllm/config/vllm.py:490-501): the EngineCoreProc batch-queue depth. 1 keeps
  // the synchronous step(); 2 (async scheduling on a single-GPU MRV2, W3) selects
  // step_with_batch_queue for depth-2 overlap. LoadedEngine resolves it once from
  // SchedulerConfig::ResolveAsyncScheduling(runner_supports_async()) and passes it
  // here; a caller that constructs AsyncLLM directly keeps the sync default.
  //
  // structured_output_manager: the engine's StructuredOutputManager, threaded
  // through to the EngineCoreProc (grammar_init + the scheduler's bitmask).
  // Null keeps structured output a no-op (backward-compat with tests that
  // build a bare stack).
  //
  // `check_for_draft_tokens` is EngineCore's speculative-decode flag, threaded
  // down to the EngineCoreProc this owns. False (the default) leaves post_step
  // a no-op, which is byte-identical for a non-speculative engine; a
  // speculative engine MUST pass true or its drafts are proposed and dropped.
  AsyncLLM(InputProcessor& input_processor, Scheduler& scheduler,
           Executor& executor, OutputProcessor& output_processor,
           BlockHasher block_hasher = nullptr, int shutdown_timeout_s = 0,
           int max_concurrent_batches = 1,
           StructuredOutputManager* structured_output_manager = nullptr,
           bool check_for_draft_tokens = false);
  ~AsyncLLM();

  AsyncLLM(const AsyncLLM&) = delete;
  AsyncLLM& operator=(const AsyncLLM&) = delete;

  // add_request (async_llm.py:280-410): process inputs, register the collector
  // with OutputProcessor BEFORE enqueueing the Request into EngineCore.
  AsyncRequest add_request(const std::string& request_id,
                           const std::string& prompt, SamplingParams params,
                           int priority = 0);

  // add_request for a PRE-TOKENIZED prompt (vLLM TokensPrompt). Strictly
  // ADDITIVE overload mirroring LLMEngine::add_request(tokens): builds the
  // request from prompt_token_ids directly (InputProcessor::process_inputs_tokens),
  // skipping tokenization, and passes no prompt string (std::nullopt) to the
  // OutputProcessor. The string overload above is UNCHANGED. Used by the async
  // beam-search driver (BeamSearchAsync), which sources each per-beam decode from
  // the beam's growing token sequence, not a string.
  AsyncRequest add_request(const std::string& request_id,
                           std::vector<int32_t> prompt_token_ids,
                           SamplingParams params, int priority = 0);

  // Prepare and atomically publish one complete string or TokensPrompt wave.
  // Every input/core Request and collector is built before any frontend state
  // is registered. On duplicate, preparation/enqueue failure, or shutdown,
  // this call publishes no core prefix and removes only state created by this
  // call, without producing synthetic terminal outputs.
  std::vector<AsyncRequest> add_request_wave(
      std::vector<AsyncStringRequestInput> requests);
  std::vector<AsyncRequest> add_request_wave(
      std::vector<AsyncTokensRequestInput> requests);

  // add_request for a MULTIMODAL prompt (ROAD-V1-MM MM-SERVE-ENGINE). Strictly
  // ADDITIVE overload mirroring LLMEngine::add_request(MultiModalInputs): builds
  // the request from the placeholder-EXPANDED prompt ids + mm_features via
  // InputProcessor::process_inputs_mm, registers the collector, and enqueues the
  // Request (mm_features carried through FromEngineCoreRequest). The string /
  // tokens overloads above are UNCHANGED; an mm_inputs with empty mm_features is
  // byte-identical to the tokens overload.
  AsyncRequest add_request(const std::string& request_id,
                           multimodal::MultiModalInputs mm_inputs,
                           SamplingParams params, int priority = 0);

  // Consumer side of generate (:524-635). get_output blocks for this request
  // only; get_output_nowait is the fast path used before blocking.
  RequestOutput get_output(const AsyncRequest& request);
  std::optional<RequestOutput> get_output_nowait(
      const AsyncRequest& request);
  // Timed wait — nullopt on timeout (for SSE keepalives).
  std::optional<RequestOutput> get_output_for(
      const AsyncRequest& request, std::chrono::milliseconds timeout);

  // Blocking convenience for non-streaming callers: drain this request's
  // collector until its terminal RequestOutput. Other requests keep running.
  RequestOutput generate(const std::string& prompt, SamplingParams params,
                         const std::string& request_id = "0",
                         int priority = 0);

  // generate for a PRE-TOKENIZED prompt (vLLM TokensPrompt). Strictly ADDITIVE
  // overload of the blocking single-request driver (mirrors the string generate
  // loop, and LLMEngine::generate(tokens)): add the pre-tokenized request, then
  // drain its collector to the terminal RequestOutput. The async beam-search
  // driver issues one such single-token decode per beam per step.
  RequestOutput generate(std::vector<int32_t> prompt_token_ids,
                         SamplingParams params,
                         const std::string& request_id = "0",
                         int priority = 0);

  // generate for a MULTIMODAL prompt (ROAD-V1-MM MM-SERVE-ENGINE). Strictly
  // ADDITIVE blocking single-request driver mirroring the tokens loop: add the
  // mm request, then drain its collector to the terminal RequestOutput. The mm
  // forward consumes the carried mm_features on the GPU worker (MM-SERVE-E2E).
  RequestOutput generate(multimodal::MultiModalInputs mm_inputs,
                         SamplingParams params,
                         const std::string& request_id = "0",
                         int priority = 0);

  // abort (:709-745): OutputProcessor first (which emits the terminal ABORT
  // output), then EngineCore. Unknown/already-finished ids are no-ops.
  void abort(const std::string& request_id);
  void abort(const std::vector<std::string>& request_ids);

  int get_num_unfinished_requests() const;
  bool has_unfinished_requests() const {
    return get_num_unfinished_requests() != 0;
  }

  // The stat-logger attach point (async_llm.py:648-652 `logger_ref`, the
  // mutable one-element list holding self.logger_manager). Mirrors
  // LLMEngine::set_stat_logger: NON-OWNING. A non-null logger must remain alive
  // until it is detached or the engine is shut down. Detaching is a quiescence
  // barrier: after set_stat_logger(nullptr) returns, the output thread no longer
  // holds or uses the previous pointer. Null (the default) keeps RunOutputHandler
  // on its byte-identical no-stats path.
  //
  // DEVIATION: upstream's one-element list exists to avoid a circular ref from
  // the handler coroutine back to the AsyncLLM. The attachment mutex makes
  // pointer publication visible and gives detach its barrier semantics.
  void set_stat_logger(metrics::PrometheusStatLogger* logger);

  // Idempotent teardown. Active requests receive abort-final outputs before
  // the output handler is woken with the engine-dead sentinel and joined.
  void shutdown();

 private:
  struct PreparedRequest {
    EngineCoreRequest request;
    std::optional<std::string> prompt;
    std::shared_ptr<RequestOutputCollector> collector;
    std::unique_ptr<Request> core_request;
  };

  std::vector<AsyncRequest> PublishPreparedWave(
      std::vector<PreparedRequest> prepared);
  void RunOutputHandler();

  InputProcessor& input_processor_;
  OutputProcessor& output_processor_;
  BlockHasher block_hasher_;
  InprocClient engine_core_;

  mutable std::mutex output_processor_mutex_;
  // async_llm.py:650-652 logger_ref. Non-owning; null == log_stats off.
  mutable std::mutex stat_logger_mutex_;
  metrics::PrometheusStatLogger* stat_logger_ = nullptr;
  std::thread output_handler_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> shutdown_started_{false};
  std::atomic<bool> errored_{false};
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ENGINE_ASYNC_LLM_H_
