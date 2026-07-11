// Ported from: vllm/entrypoints/openai/completion/serving.py @ e24d1b24
// (OpenAIServingCompletion.create_completion + completion_stream_generator +
// request_output_to_completion_response).
//
// SCOPE (M3.1 Task 2 / T0): the OpenAI /v1/completions serving logic DECOUPLED
// from HTTP. The handler holds an LLMEngine& and turns a CompletionRequest into
// either a full CompletionResponse (non-streaming) or a sequence of SSE
// `data: {json}\n\n` chunk strings ending with `data: [DONE]\n\n` (streaming).
// W2 adds the live pull-based SseStream over AsyncLLM. The legacy LLMEngine
// constructor remains as a synchronous compatibility/test seam and still
// returns a precomputed vector; the production server uses AsyncLLM.
//
// DEFERRED (marked; matches upstream so re-adding is mechanical):
//   - tools / tool_choice / grammars (M3.3 / M3.4)
//   - logprobs *payload* (CompletionLogProbs); echo; best_of; suffix
//   - n > 1 (multiple choices) — T0 emits a single choice (index 0)
//   - beam search, LoRA, data-parallel rank, trace headers, kv_transfer
#ifndef VLLM_ENTRYPOINTS_OPENAI_SERVING_COMPLETION_H_
#define VLLM_ENTRYPOINTS_OPENAI_SERVING_COMPLETION_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/llm_engine.h"

namespace vllm::entrypoints::openai {

// Pull-based equivalent of upstream's AsyncGenerator[str]. next() blocks until
// one framed SSE chunk is ready and returns false after [DONE]. abort() is
// idempotent and tears down the underlying request on client disconnect.
class SseStream {
 public:
  virtual ~SseStream() = default;
  virtual bool next(std::string& chunk) = 0;
  virtual void abort() = 0;
};

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
  // Production AsyncLLM path: live source instead of precomputed chunks.
  std::shared_ptr<SseStream> sse_stream;
};

class OpenAIServingCompletion {
 public:
  // `served_model_name` is the name echoed back in the response `model` field
  // (upstream: self.models.model_name(lora_request); T0 has one served model).
  OpenAIServingCompletion(v1::LLMEngine& engine, std::string served_model_name,
                          bool enable_force_include_usage = false);
  OpenAIServingCompletion(v1::AsyncLLM& engine,
                          std::string served_model_name,
                          bool enable_force_include_usage = false);

  // create_completion (completion/serving.py:109). request → SamplingParams →
  // engine → CompletionResponse (non-stream) or SSE chunk vector (stream).
  CompletionResult create_completion(const CompletionRequest& request);

  // Transport compatibility seam: production handlers are backed by
  // AsyncLLM and need no request-level serialization. The retained LLMEngine
  // constructor is synchronous and must still be serialized by ApiServer.
  bool uses_async_engine() const { return async_engine_ != nullptr; }

 private:
  v1::LLMEngine* sync_engine_ = nullptr;
  v1::AsyncLLM* async_engine_ = nullptr;
  std::string served_model_name_;
  bool enable_force_include_usage_ = false;
  // Monotonic request counter — the request_id is "cmpl-<counter>". Upstream
  // uses random_uuid() (serving/engine/serving.py:_base_request_id); no
  // random/uuid is wired at T0, so a counter stands in (id uniqueness only).
  std::atomic<int64_t> request_counter_{0};
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_COMPLETION_H_
