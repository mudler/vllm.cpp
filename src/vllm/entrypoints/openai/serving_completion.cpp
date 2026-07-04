// Ported from: vllm/entrypoints/openai/completion/serving.py @ e24d1b24
// See serving_completion.h for scope, the return-type design and deferrals.
#include "vllm/entrypoints/openai/serving_completion.h"

#include <ctime>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

OpenAIServingCompletion::OpenAIServingCompletion(v1::LLMEngine& engine,
                                                 std::string served_model_name)
    : engine_(engine), served_model_name_(std::move(served_model_name)) {}

CompletionResult OpenAIServingCompletion::create_completion(
    const CompletionRequest& request) {
  // request_id = f"cmpl-{...}" (completion/serving.py:143); created_time =
  // int(time.time()) (:144).
  const std::string request_id =
      "cmpl-" + std::to_string(request_counter_++);
  const auto created_time = static_cast<int64_t>(std::time(nullptr));
  const std::string model_name =
      request.model.has_value() ? *request.model : served_model_name_;

  // request → SamplingParams. to_sampling_params sets output_kind to kDelta
  // when stream, kFinalOnly otherwise (protocol.cpp) — matching upstream's
  // per-request RequestOutputKind (completion/serving.py:174).
  SamplingParams sampling_params = request.to_sampling_params();

  // T0: single prompt, single choice (n == 1). The engine sub-request id is
  // f"{request_id}-{i}" upstream (:179); here i == 0.
  const std::string engine_request_id = request_id + "-0";

  if (request.stream) {
    // ── Streaming (completion_stream_generator, :278) ─────────────────────
    // Drive the engine over DELTA RequestOutputs; format one
    // CompletionStreamResponse per non-empty delta, then `data: [DONE]\n\n`.
    CompletionResult result;
    result.streaming = true;

    int previous_num_tokens = 0;
    engine_.add_request(engine_request_id, request.prompt,
                        std::move(sampling_params));
    while (engine_.has_unfinished_requests()) {
      for (const RequestOutput& res : engine_.step()) {
        if (res.request_id != engine_request_id) continue;
        for (const CompletionOutput& output : res.outputs) {
          const std::string& delta_text = output.text;
          // :368-374 chunked-prefill: skip empty chunks (no text, no tokens,
          // and nothing emitted yet).
          if (delta_text.empty() && output.token_ids.empty() &&
              previous_num_tokens == 0) {
            continue;
          }
          previous_num_tokens += static_cast<int>(output.token_ids.size());

          CompletionResponseStreamChoice choice;
          choice.index = 0;  // output.index + prompt_idx * num_choices; T0 == 0
          choice.text = delta_text;
          choice.finish_reason = output.finish_reason;

          CompletionStreamResponse chunk;
          chunk.id = request_id;
          chunk.created = created_time;
          chunk.model = model_name;
          chunk.choices.push_back(std::move(choice));

          result.sse_chunks.push_back(
              "data: " + nlohmann::json(chunk).dump() + "\n\n");
        }
      }
    }
    result.sse_chunks.push_back("data: [DONE]\n\n");
    return result;
  }

  // ── Non-streaming (request_output_to_completion_response, :475) ──────────
  const RequestOutput final_res = engine_.generate(
      request.prompt, std::move(sampling_params), engine_request_id);

  CompletionResponse response;
  response.id = request_id;
  response.created = created_time;
  response.model = model_name;

  int num_prompt_tokens = static_cast<int>(final_res.prompt_token_ids.size());
  int num_generated_tokens = 0;
  for (const CompletionOutput& output : final_res.outputs) {
    CompletionResponseChoice choice;
    choice.index = static_cast<int>(response.choices.size());
    choice.text = output.text;  // echo deferred
    choice.finish_reason = output.finish_reason;
    response.choices.push_back(std::move(choice));
    num_generated_tokens += static_cast<int>(output.token_ids.size());
  }

  // UsageInfo (:576).
  response.usage.prompt_tokens = num_prompt_tokens;
  response.usage.completion_tokens = num_generated_tokens;
  response.usage.total_tokens = num_prompt_tokens + num_generated_tokens;

  CompletionResult result;
  result.streaming = false;
  result.response = std::move(response);
  return result;
}

}  // namespace vllm::entrypoints::openai
