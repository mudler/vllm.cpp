// Ported from: vllm/v1/engine/llm_engine.py @ e24d1b24
// (LLMEngine.__init__ wiring + add_request + step) and the LLM.generate driver
// loop (vllm/entrypoints/offline_utils.py::_run_engine @ e24d1b24).
//
// Scope (M1.8 Task 6): the minimal SYNCHRONOUS LLMEngine that wires the M1.8
// pieces into a running V1 engine and proves the loop end-to-end:
//   InputProcessor (Task 2)  -> EngineCoreRequest
//   OutputProcessor (Task 5) -> RequestOutput   (registered per request)
//   EngineCore (Task 1)      -> the schedule/execute/sample/update loop over the
//                               Scheduler (M1.4) + the real GPUModelRunner (Task 4)
// This is the T0 offline `LLM`/`LLMEngine` path: add_request, step (get outputs ->
// process -> honor reqs_to_abort), and a single-request generate() convenience
// driver (add_request then loop step() to completion) for the e2e tests.
//
// WIRING (mirrors llm_engine.py __init__ :94-111):
//   self.input_processor  = InputProcessor(vllm_config, renderer)     -> ctor ref
//   self.output_processor = OutputProcessor(renderer.tokenizer, ...)  -> ctor ref
//   self.engine_core      = EngineCoreClient.make_client(...)         -> ctor ref
// At T0 (single process, in-process) the EngineCoreClient collapses to a direct
// reference to the EngineCore the caller built over {Scheduler, Executor(runner)}.
// The three collaborators are constructed by the caller and held by reference,
// mirroring upstream where __init__ constructs+owns them (our EngineCore already
// holds Scheduler& + Executor&; keeping the same by-reference seam here avoids
// reshaping Tasks 1-5). They must outlive the LLMEngine.
//
// DEVIATIONS vs the pinned API (recorded, use OUR names):
//   - add_request builds the Request (Request::FromEngineCoreRequest) and hands
//     it to EngineCore.add_request(unique_ptr<Request>). Upstream passes the
//     EngineCoreRequest to engine_core.add_request and core.py builds the Request
//     internally (from_engine_core_request(request, block_hasher)); our
//     EngineCore.add_request takes an already-built Request (Task 1), so the
//     factory + block_hasher injection move up here. block_hasher is engine-held
//     (default null => prefix caching off), exactly what core.py injects.
//   - step() returns std::vector<RequestOutput> (PoolingRequestOutput deferred).
//     Our EngineCore.step() returns the per-client outputs map (T0: at most one
//     entry); we process each entry's EngineCoreOutputs exactly as upstream's
//     single engine_core.get_output().outputs, then honor reqs_to_abort. The
//     should_execute_dummy_batch / stats / tracing / DP branches are dropped.
//   - generate(prompt, params) is the single-request _run_engine driver (add +
//     loop step to finish) returning the finished RequestOutput. The offline
//     LLM.generate batching/tqdm/sort over many prompts is out of the T0 scope
//     (the e2e tests drive multi-request concurrency via add_request+step
//     directly).
//
// DEFERRED (marked; matches upstream so re-adding is mechanical): the whole
// __init__ VllmConfig/renderer/DP-group/StatLoggerManager/tracing setup,
// EngineCoreClient (multiproc/async/ZMQ), parallel sampling (n>1 fan-out /
// ParentRequest), pooling (PoolingParams/PoolingRequestOutput), EngineCoreRequest
// pass-through + assign_request_id randomization, execute_dummy_batch, log_stats /
// IterationStats / scheduler-stats recording, profiling / sleep / wake_up / LoRA /
// reset_*_cache / collective_rpc utility surface, and the get_supported_tasks
// plumbing.
#ifndef VLLM_V1_ENGINE_LLM_ENGINE_H_
#define VLLM_V1_ENGINE_LLM_ENGINE_H_

#include <string>
#include <vector>

#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"  // BlockHasher
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"

namespace vllm::v1 {

// The synchronous offline LLMEngine (T0 subset). Holds non-owning references to
// the three collaborators (constructed + owned by the caller); they outlive it.
class LLMEngine {
 public:
  // __init__ (llm_engine.py:94-111): wire the processor + engine_core +
  // output_processor. `block_hasher` is injected into every Request built here
  // (upstream core.py from_engine_core_request(request, block_hasher)); null =>
  // prefix caching off.
  LLMEngine(InputProcessor& input_processor, EngineCore& engine_core,
            OutputProcessor& output_processor, BlockHasher block_hasher = nullptr);

  // add_request (llm_engine.py:218): process_inputs -> EngineCoreRequest, then
  // output_processor.add_request BEFORE engine_core.add_request (upstream order,
  // :274-276). Returns the request id.
  std::string add_request(const std::string& request_id,
                          const std::string& prompt, SamplingParams params,
                          int priority = 0);

  // step (llm_engine.py:296): get the EngineCore outputs -> process_outputs ->
  // abort any reqs the detokenizer stopped -> return the RequestOutputs.
  std::vector<RequestOutput> step();

  // abort_request (llm_engine.py:230 abort_request): client-initiated teardown of
  // an in-flight request. Mirrors upstream order — output_processor.abort_requests
  // THEN engine_core.abort_requests — so the request stops counting toward
  // has_unfinished_requests() AND is dropped from the scheduler. Unknown /
  // already-finished ids are a no-op. Used by the C-ABI streaming early-stop
  // (vllm_complete_stream) to tear down when the callback returns false.
  void abort_request(const std::string& request_id);

  // has_unfinished_requests (llm_engine.py:191): drives the generate loop. The DP
  // term is deferred (single engine at T0).
  bool has_unfinished_requests() const {
    return output_processor_.has_unfinished_requests();
  }
  // get_num_unfinished_requests (llm_engine.py:188).
  int get_num_unfinished_requests() const {
    return output_processor_.get_num_unfinished_requests();
  }

  // generate (the LLM.generate / _run_engine driver for one request): add the
  // request, then loop step() until it finishes, returning its finished
  // RequestOutput (the last one carrying finished == true). For DELTA output_kind
  // this is only the final delta — drive step() directly to accumulate streaming
  // deltas.
  RequestOutput generate(const std::string& prompt, SamplingParams params,
                         const std::string& request_id = "0", int priority = 0);

  // The rolling prefix-cache hit rate (queries/hits in TOKENS over the most
  // recent 1000 requests). See EngineCore::prefix_cache_metrics.
  const CachingMetrics& prefix_cache_metrics() const {
    return engine_core_.prefix_cache_metrics();
  }

 private:
  InputProcessor& input_processor_;
  EngineCore& engine_core_;
  OutputProcessor& output_processor_;
  BlockHasher block_hasher_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ENGINE_LLM_ENGINE_H_
