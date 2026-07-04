// Ported from: vllm/v1/engine/llm_engine.py @ e24d1b24 (LLMEngine.add_request /
// step) + vllm/entrypoints/offline_utils.py::_run_engine @ e24d1b24 (the
// generate driver loop). See llm_engine.h for scope, wiring and deviations.
#include "vllm/v1/engine/llm_engine.h"

#include <memory>
#include <utility>

#include "vllm/v1/engine/types.h"
#include "vllm/v1/request.h"

namespace vllm::v1 {

LLMEngine::LLMEngine(InputProcessor& input_processor, EngineCore& engine_core,
                     OutputProcessor& output_processor, BlockHasher block_hasher)
    : input_processor_(input_processor),
      engine_core_(engine_core),
      output_processor_(output_processor),
      block_hasher_(std::move(block_hasher)) {}

std::string LLMEngine::add_request(const std::string& request_id,
                                   const std::string& prompt,
                                   SamplingParams params) {
  // llm_engine.py:250 request = self.input_processor.process_inputs(...). The
  // text path: validate (SamplingParams PostInit) + tokenize + build the message.
  EngineCoreRequest request =
      input_processor_.process_inputs(request_id, prompt, std::move(params));
  const std::string req_id = request.request_id;

  // llm_engine.py:274 self.output_processor.add_request(request, prompt_text,
  // None, 0) — BEFORE engine_core.add_request. Register the RequestState (with
  // our incremental detokenizer) so process_outputs can detokenize its stream.
  output_processor_.add_request(request, prompt, /*request_index=*/0);

  // llm_engine.py:276 self.engine_core.add_request(request). Upstream hands the
  // EngineCoreRequest to core.py, which builds the Request via
  // from_engine_core_request(request, block_hasher); our EngineCore.add_request
  // takes the built Request, so construct it here with the engine's block_hasher.
  auto req = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));
  engine_core_.add_request(std::move(req));
  return req_id;
}

std::vector<RequestOutput> LLMEngine::step() {
  // llm_engine.py:303 outputs = self.engine_core.get_output(). Our EngineCore
  // fuses the core's schedule/execute/sample/update into step(), returning the
  // per-client outputs map (T0: at most one entry) + model_executed.
  std::pair<std::map<int, EngineCoreOutputs>, bool> stepped = engine_core_.step();
  std::map<int, EngineCoreOutputs>& outputs_by_client = stepped.first;

  std::vector<RequestOutput> request_outputs;
  for (auto& entry : outputs_by_client) {
    EngineCoreOutputs& engine_core_outputs = entry.second;

    // llm_engine.py:309 process_outputs — detokenize + string-stop + assemble.
    OutputProcessorOutput processed =
        output_processor_.process_outputs(engine_core_outputs);

    // llm_engine.py:318 abort any reqs that finished due to stop strings (the
    // detokenizer stopped them but EngineCore did not signal it itself).
    engine_core_.abort_requests(processed.reqs_to_abort);

    for (RequestOutput& ro : processed.request_outputs) {
      request_outputs.push_back(std::move(ro));
    }
  }
  return request_outputs;
}

void LLMEngine::abort_request(const std::string& request_id) {
  // llm_engine.py:230 abort_request: drop the request from the output processor
  // (so has_unfinished_requests() no longer counts it) and from the engine core
  // (scheduler finish_requests -> FINISHED_ABORTED). Upstream aborts the output
  // processor first, then the engine core; both are no-ops for unknown ids.
  const std::vector<std::string> ids = {request_id};
  output_processor_.abort_requests(ids);
  engine_core_.abort_requests(ids);
}

RequestOutput LLMEngine::generate(const std::string& prompt,
                                  SamplingParams params,
                                  const std::string& request_id) {
  // The LLM.generate / _run_engine driver for one request (offline_utils.py:591):
  //   while self.llm_engine.has_unfinished_requests():
  //       for output in self.llm_engine.step():
  //           if output.finished: outputs.append(output)
  add_request(request_id, prompt, std::move(params));
  RequestOutput result;
  while (has_unfinished_requests()) {
    std::vector<RequestOutput> step_outputs = step();
    for (RequestOutput& out : step_outputs) {
      if (out.finished) {
        result = std::move(out);
      }
    }
  }
  return result;
}

}  // namespace vllm::v1
