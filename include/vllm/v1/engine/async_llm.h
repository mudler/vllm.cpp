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
// pooling, DP, stats/tracing and W3 async scheduling are deferred.
#ifndef VLLM_V1_ENGINE_ASYNC_LLM_H_
#define VLLM_V1_ENGINE_ASYNC_LLM_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/engine/core_client.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"

namespace vllm::v1 {

// The value returned by add_request/generate-start. The collector is shared
// with OutputProcessor's RequestState until that request finishes or aborts.
struct AsyncRequest {
  std::string request_id;
  std::shared_ptr<RequestOutputCollector> collector;
};

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
  AsyncLLM(InputProcessor& input_processor, Scheduler& scheduler,
           Executor& executor, OutputProcessor& output_processor,
           BlockHasher block_hasher = nullptr, int shutdown_timeout_s = 0,
           int max_concurrent_batches = 1,
           StructuredOutputManager* structured_output_manager = nullptr);
  ~AsyncLLM();

  AsyncLLM(const AsyncLLM&) = delete;
  AsyncLLM& operator=(const AsyncLLM&) = delete;

  // add_request (async_llm.py:280-410): process inputs, register the collector
  // with OutputProcessor BEFORE enqueueing the Request into EngineCore.
  AsyncRequest add_request(const std::string& request_id,
                           const std::string& prompt, SamplingParams params,
                           int priority = 0);

  // Consumer side of generate (:524-635). get_output blocks for this request
  // only; get_output_nowait is the fast path used before blocking.
  RequestOutput get_output(const AsyncRequest& request);
  std::optional<RequestOutput> get_output_nowait(
      const AsyncRequest& request);

  // Blocking convenience for non-streaming callers: drain this request's
  // collector until its terminal RequestOutput. Other requests keep running.
  RequestOutput generate(const std::string& prompt, SamplingParams params,
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

  // Idempotent teardown. Active requests receive abort-final outputs before
  // the output handler is woken with the engine-dead sentinel and joined.
  void shutdown();

 private:
  void RunOutputHandler();

  InputProcessor& input_processor_;
  OutputProcessor& output_processor_;
  BlockHasher block_hasher_;
  InprocClient engine_core_;

  mutable std::mutex output_processor_mutex_;
  std::thread output_handler_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> shutdown_started_{false};
  std::atomic<bool> errored_{false};
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ENGINE_ASYNC_LLM_H_
