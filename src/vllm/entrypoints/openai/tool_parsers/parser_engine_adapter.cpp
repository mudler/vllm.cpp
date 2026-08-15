// See parser_engine_adapter.h. Ported from vllm/parser/engine/adapters.py:128 +
// registered_adapters.py:68-70 + vllm/tool_parsers/inkling_tool_parser.py:7
// @ 555967922.
#include "vllm/entrypoints/openai/tool_parsers/parser_engine_adapter.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "vllm/parser/parser_manager.h"

namespace vllm::entrypoints::openai {

namespace pe = vllm::parser::engine;

ParserEngineToolAdapter::ParserEngineToolAdapter(
    std::unique_ptr<pe::ParserEngine> engine)
    : engine_(std::move(engine)) {}

ParserEngineToolAdapter::~ParserEngineToolAdapter() = default;

// adapters.py:158.
ExtractedToolCallInformation ParserEngineToolAdapter::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  return engine_->extract_tool_calls_from_content(
      model_output, pe::ParserRequestFromChatCompletion(request));
}

// adapters.py:167.
std::optional<DeltaMessage>
ParserEngineToolAdapter::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  engine_->initialize_streaming(pe::ParserState::CONTENT);
  return engine_->extract_tool_calls_streaming(
      previous_text, current_text, delta_text,
      pe::ParserRequestFromChatCompletion(request));
}

// adapters.py:189.
std::optional<DeltaMessage> ParserEngineToolAdapter::finish_streaming() {
  return engine_->finish_streaming();
}

// inkling_tool_parser.py:7 over registered_adapters.py:68-70. The engine itself
// is built by parser::get_parser_engine("inkling") (parser_manager.cpp), which
// is the C++ analogue of make_adapters binding InklingParser to the adapter.
InklingEngineToolParser::InklingEngineToolParser()
    : ParserEngineToolAdapter(vllm::parser::get_parser_engine("inkling")) {}

}  // namespace vllm::entrypoints::openai
