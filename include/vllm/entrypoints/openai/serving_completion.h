// Ported from: vllm/entrypoints/openai/completion/serving.py @ e24d1b24
// (OpenAIServingCompletion.create_completion + completion_stream_generator +
// request_output_to_completion_response).
//
// SCOPE (M3.1 Task 2 / T0): the OpenAI /v1/completions serving logic DECOUPLED
// from HTTP. The handler holds an LLMEngine& and turns a CompletionRequest into
// either a full CompletionResponse (non-streaming) or a sequence of SSE
// `data: {json}\n\n` chunk strings ending with `data: [DONE]\n\n` (streaming).
// At T0 the engine runs to completion synchronously, so the streaming path
// collects every DELTA RequestOutput and formats the chunk sequence up front —
// a real async generator is a later concern; the chunk CONTENT/cadence matches
// what upstream's completion_stream_generator yields.
//
// DEFERRED (marked; matches upstream so re-adding is mechanical):
//   - tools / tool_choice / grammars (M3.3 / M3.4)
//   - logprobs *payload* (CompletionLogProbs); echo; best_of; suffix
//   - n > 1 (multiple choices) — T0 emits a single choice (index 0)
//   - stream_options / include_usage trailing usage chunk (default: no usage
//     chunk in the stream — only the deltas + [DONE])
//   - beam search, LoRA, data-parallel rank, trace headers, kv_transfer
#ifndef VLLM_ENTRYPOINTS_OPENAI_SERVING_COMPLETION_H_
#define VLLM_ENTRYPOINTS_OPENAI_SERVING_COMPLETION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/v1/engine/llm_engine.h"

namespace vllm::entrypoints::openai {

// The result of create_completion: EITHER a full non-streaming response OR the
// ordered SSE chunk strings (each already framed as `data: ...\n\n`, the last
// being `data: [DONE]\n\n`). The HTTP layer (Task 4) just writes whichever is
// populated. Mirrors upstream's
//   AsyncGenerator[str] | CompletionResponse
// return union (completion/serving.py:113).
struct CompletionResult {
  bool streaming = false;
  // Set when !streaming.
  std::optional<CompletionResponse> response;
  // Set when streaming: `data: {json}\n\n` lines, terminated by `data: [DONE]\n\n`.
  std::vector<std::string> sse_chunks;
};

class OpenAIServingCompletion {
 public:
  // `served_model_name` is the name echoed back in the response `model` field
  // (upstream: self.models.model_name(lora_request); T0 has one served model).
  OpenAIServingCompletion(v1::LLMEngine& engine, std::string served_model_name);

  // create_completion (completion/serving.py:109). request → SamplingParams →
  // engine → CompletionResponse (non-stream) or SSE chunk vector (stream).
  CompletionResult create_completion(const CompletionRequest& request);

 private:
  v1::LLMEngine& engine_;
  std::string served_model_name_;
  // Monotonic request counter — the request_id is "cmpl-<counter>". Upstream
  // uses random_uuid() (serving/engine/serving.py:_base_request_id); no
  // random/uuid is wired at T0, so a counter stands in (id uniqueness only).
  int64_t request_counter_ = 0;
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_COMPLETION_H_
