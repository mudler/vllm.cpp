// Ported from: vllm/v1/engine/async_llm.py @ e24d1b24
// (add_request :280-410, generate :524-635, _run_output_handler :637-707,
// abort :709-745). See async_llm.h for scope and the in-process deviation.
#include "vllm/v1/engine/async_llm.h"

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

#include "vllm/v1/request.h"

namespace vllm::v1 {

AsyncLLM::AsyncLLM(InputProcessor& input_processor, Scheduler& scheduler,
                   Executor& executor, OutputProcessor& output_processor,
                   BlockHasher block_hasher, int shutdown_timeout_s,
                   int max_concurrent_batches)
    : input_processor_(input_processor),
      output_processor_(output_processor),
      block_hasher_(std::move(block_hasher)),
      engine_core_(scheduler, executor,
                   /*structured_output_manager=*/nullptr,
                   max_concurrent_batches, shutdown_timeout_s),
      output_handler_() {
  // Start only after every member (including the stop flags) is constructed;
  // the thread blocks on get_output until the first request arrives.
  output_handler_ = std::thread(&AsyncLLM::RunOutputHandler, this);
}

AsyncLLM::~AsyncLLM() { shutdown(); }

AsyncRequest AsyncLLM::add_request(const std::string& request_id,
                                   const std::string& prompt,
                                   SamplingParams params, int priority) {
  if (shutdown_started_.load() || errored_.load() ||
      engine_core_.engine_dead()) {
    throw EngineDeadError("request submitted to a stopped AsyncLLM");
  }

  // async_llm.py:326-350: InputProcessor first, then one collector keyed by
  // the internal request id. T0 has no request-id remap/parallel sampling.
  EngineCoreRequest request = input_processor_.process_inputs(
      request_id, prompt, std::move(params), /*arrival_time=*/std::nullopt,
      priority);
  auto collector = std::make_shared<RequestOutputCollector>(
      request.sampling_params.output_kind, request.request_id);

  // Build the core Request before publishing the OutputProcessor state so an
  // allocation/factory failure cannot leave a consumer waiting forever.
  auto core_request = std::make_unique<Request>(
      Request::FromEngineCoreRequest(request, block_hasher_));

  // async_llm.py:_add_request :400-410: OutputProcessor registration MUST
  // precede the EngineCore enqueue; an immediately produced frame then always
  // finds its request state/collector.
  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    // Serialize the final admission check, frontend registration, and core
    // enqueue with shutdown's abort-all transition. A submitter may have
    // passed the fast check above while processing inputs; it must not publish
    // a new collector after shutdown has already swept the request table.
    if (shutdown_started_.load() || errored_.load() ||
        engine_core_.engine_dead()) {
      throw EngineDeadError("request submitted to a stopped AsyncLLM");
    }
    output_processor_.add_request(request, prompt, /*request_index=*/0,
                                  collector);
    try {
      engine_core_.add_request_async(std::move(core_request));
    } catch (...) {
      // Queue allocation is the only expected failure here. Roll back the
      // frontend state so the id remains reusable and no collector hangs.
      (void)output_processor_.abort_requests({request.request_id});
      throw;
    }
  }

  return AsyncRequest{request.request_id, std::move(collector)};
}

RequestOutput AsyncLLM::get_output(const AsyncRequest& request) {
  if (request.collector == nullptr) {
    throw std::invalid_argument("AsyncLLM request has no collector");
  }
  return request.collector->get();
}

std::optional<RequestOutput> AsyncLLM::get_output_nowait(
    const AsyncRequest& request) {
  if (request.collector == nullptr) {
    throw std::invalid_argument("AsyncLLM request has no collector");
  }
  return request.collector->get_nowait();
}

RequestOutput AsyncLLM::generate(const std::string& prompt,
                                 SamplingParams params,
                                 const std::string& request_id, int priority) {
  AsyncRequest request =
      add_request(request_id, prompt, std::move(params), priority);
  try {
    for (;;) {
      // async_llm.py:572 drains the single-slot collector without blocking
      // when the output handler has already produced a value.
      std::optional<RequestOutput> ready = get_output_nowait(request);
      RequestOutput output = ready.has_value()
                                 ? std::move(*ready)
                                 : get_output(request);
      if (output.finished) return output;
    }
  } catch (...) {
    abort(request.request_id);
    throw;
  }
}

void AsyncLLM::abort(const std::string& request_id) {
  abort(std::vector<std::string>{request_id});
}

void AsyncLLM::abort(const std::vector<std::string>& request_ids) {
  std::vector<std::string> core_request_ids;
  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    core_request_ids = output_processor_.abort_requests(
        request_ids, /*produce_final_output=*/true);
  }
  engine_core_.abort_requests_async(core_request_ids);
}

int AsyncLLM::get_num_unfinished_requests() const {
  std::lock_guard<std::mutex> lock(output_processor_mutex_);
  return output_processor_.get_num_unfinished_requests();
}

void AsyncLLM::RunOutputHandler() {
  try {
    for (;;) {
      // async_llm.py:651: one blocking EngineCoreOutputs pull.
      EngineCoreOutputs outputs = engine_core_.get_output();
      OutputProcessorOutput processed;
      {
        std::lock_guard<std::mutex> lock(output_processor_mutex_);
        // RequestOutputs are pushed to their collectors by OutputProcessor;
        // the synchronous return list must therefore stay empty.
        processed = output_processor_.process_outputs(outputs);
      }
      // Stop-string finishes detected by the detokenizer must be reflected in
      // EngineCore after leaving the OutputProcessor critical section.
      engine_core_.abort_requests_async(processed.reqs_to_abort);
    }
  } catch (...) {
    if (!stopping_.load()) {
      // Restore the upstream fatal log (vllm/v1/engine/async_llm.py:703-705:
      // logger.exception("AsyncLLM output_handler failed.") before
      // output_processor.propagate_error(e)). std::cerr only, so the witness
      // survives a SIGKILL escalation.
      try {
        std::rethrow_exception(std::current_exception());
      } catch (const std::exception& e) {
        std::cerr << "async-llm: output handler saw engine death: " << e.what()
                  << "\n";
      } catch (...) {
      }
      errored_.store(true);
      std::lock_guard<std::mutex> lock(output_processor_mutex_);
      output_processor_.propagate_error(std::current_exception());
    }
  }
}

void AsyncLLM::shutdown() {
  bool expected = false;
  if (!shutdown_started_.compare_exchange_strong(expected, true)) return;

  stopping_.store(true);
  // An ADD can still be queued when shutdown wins the race before the engine
  // thread's first busy-loop iteration. EngineCoreProc then observes shutdown
  // with an empty Scheduler and exits without consuming that ADD, so relying
  // only on core abort outputs would strand its collector. Abort the frontend
  // states first: every consumer is woken exactly once regardless of whether
  // the core has admitted its request yet. The queued core abort remains a
  // harmless no-op for not-yet-admitted/already-finished IDs.
  std::vector<std::string> core_request_ids;
  {
    std::lock_guard<std::mutex> lock(output_processor_mutex_);
    core_request_ids = output_processor_.abort_all_requests(
        /*produce_final_output=*/true);
  }
  engine_core_.abort_requests_async(core_request_ids);

  // EngineCoreProc abort/drain semantics enqueue all terminal frames before
  // its thread exits for already-admitted requests. Once joined, append a
  // sentinel to wake the output handler after it drains any residual frames.
  engine_core_.shutdown();
  engine_core_.proc().send_engine_dead("AsyncLLM shutdown");
  if (output_handler_.joinable()) output_handler_.join();
}

}  // namespace vllm::v1
