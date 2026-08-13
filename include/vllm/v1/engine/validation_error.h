// The engine's "the caller asked for something we can never serve" refusal.
//
// This class was defined inline in vllm/v1/engine/input_processor.h until
// ENG-MM-INPUT-PIPELINE L1 (#607) needed to throw it from
// vllm/multimodal/processing/context.h as well — that is where upstream's
// `validate_num_items` raises `VLLMValidationError`
// (vllm/multimodal/processing/context.py:409-428 @ 5559679229bc). Including the
// whole input_processor.h from the multimodal layer would drag the tokenizer,
// the HF config and SamplingParams into every multimodal TU, and would put a
// header cycle one step away the moment the input processor calls into the
// multimodal validators (upstream's context.py:461 call site). So the type moved
// here, unchanged, in the same namespace and under the same name:
// input_processor.h includes this header, api_server.cpp keeps catching
// `vllm::v1::InputValidationError` through it, and nothing else moved.
//
// There is deliberately only ONE of these. Upstream has one
// VLLMValidationError, and the OpenAI server answers it with HTTP 400
// (src/vllm/entrypoints/openai/api_server.cpp:185,252 catch this type ahead of
// their generic std::exception -> HTTP 500 arm). A second, multimodal-only
// refusal class would compile, pass its own unit tests, and then surface a
// too-many-images request as a 500.
#ifndef VLLM_V1_ENGINE_VALIDATION_ERROR_H_
#define VLLM_V1_ENGINE_VALIDATION_ERROR_H_

#include <stdexcept>
#include <string>

namespace vllm::v1 {

// A request the engine REFUSES because the caller asked for something it can
// never serve — a prompt longer than the resolved max_model_len
// (_validate_prompt_len, input_processor.py:387-432 @ 5559679229bc), or more
// multimodal items of a modality than the configured limit allows
// (validate_num_items, multimodal/processing/context.py:409-428).
//
// Upstream raises a bare ValueError / VLLMValidationError here and the OpenAI
// server maps it to `BadRequestError` / HTTP 400 in create_error_response
// (vllm/entrypoints/serve/utils/error_response.py:62-65). C++ has no equivalent
// of "the user-input exception class", so the mapping needs a NAMED type: the
// api_server handlers catch this ahead of their generic `std::exception` ->
// HTTP 500 arm, which is what makes the status code 400 rather than 500.
class InputValidationError : public std::invalid_argument {
 public:
  explicit InputValidationError(const std::string& msg)
      : std::invalid_argument(msg) {}
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ENGINE_VALIDATION_ERROR_H_
