// Ported from: vllm/tool_parsers/pythonic_tool_parser.py @ e24d1b24
// (shared pythonic helpers now live in pythonic_core.{h,cpp}:
//  vllm/tool_parsers/utils.py @ e24d1b24 - handle_single_tool,
//  get_parameter_value, make_valid_python, compute_tool_delta, find_common_prefix).
#include "vllm/entrypoints/openai/tool_parsers/pythonic.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/pythonic_core.h"

namespace vllm::entrypoints::openai {

namespace {

using pythonic_core::PyCall;

std::string ReplaceAll(std::string s, char from, char to) {
  for (char& c : s) {
    if (c == from) c = to;
  }
  return s;
}

}  // namespace

ExtractedToolCallInformation PythonicToolParser::ExtractClean(
    const std::string& clean_text) {
  // pythonic_tool_parser.py:75-115 - regex gate + ast parse collapsed into one
  // strict parse. On any failure: plain content, no tool calls.
  const std::optional<std::vector<PyCall>> calls =
      pythonic_core::parse_call_list(clean_text);
  if (!calls.has_value()) {
    return ExtractedToolCallInformation{false, {}, clean_text};
  }
  std::vector<ToolCall> tool_calls;
  tool_calls.reserve(calls->size());
  for (const PyCall& call : *calls) {
    ToolCall tc;
    tc.id = make_tool_call_id();
    tc.type = "function";
    tc.function.name = call.name;
    tc.function.arguments = call.arguments.dump();
    tool_calls.push_back(std::move(tc));
  }
  ExtractedToolCallInformation info;
  info.tools_called = true;
  info.tool_calls = std::move(tool_calls);
  // content is None when tools are called (upstream returns content=None).
  return info;
}

std::optional<DeltaMessage> PythonicToolParser::StreamClean(
    const std::string& current_clean) {
  // pythonic_tool_parser.py:130-202.
  try {
    const std::optional<std::pair<std::string, std::string>> mv =
        pythonic_core::make_valid_python(current_clean);
    if (!mv.has_value()) return std::nullopt;
    const std::string& valid_text = mv->first;
    const std::string& added_text = mv->second;

    const std::optional<std::vector<PyCall>> calls =
        pythonic_core::parse_call_list(valid_text);
    if (!calls.has_value()) return std::nullopt;

    std::vector<DeltaToolCall> tool_deltas;
    for (std::size_t index = 0; index < calls->size(); ++index) {
      if (static_cast<int>(index) < current_tool_id) continue;  // current_tool_index
      current_tool_id = static_cast<int>(index);
      if (streamed_args_for_tool.size() == index) {
        streamed_args_for_tool.emplace_back();
      }

      const bool new_call_complete = (index < calls->size() - 1) ||
                                     (added_text.find(")]") == std::string::npos);
      if (new_call_complete) ++current_tool_id;

      std::string withheld_suffix;
      if (!new_call_complete) {
        // added_text[:-2]
        withheld_suffix = added_text.substr(0, added_text.size() - 2);
        // added_text[-2] == ")" -> append "}"
        if (added_text[added_text.size() - 2] == ')') withheld_suffix += '}';
      }
      withheld_suffix = ReplaceAll(withheld_suffix, '\'', '"');

      const PyCall& call = (*calls)[index];
      const std::string call_args = call.arguments.dump();
      const std::string call_id = make_tool_call_id();

      const std::optional<DeltaToolCall> delta = pythonic_core::compute_tool_delta(
          streamed_args_for_tool[index], call_id, call.name, call_args,
          static_cast<int>(index), withheld_suffix);
      if (delta.has_value()) {
        tool_deltas.push_back(*delta);
        if (delta->function.arguments.has_value()) {
          streamed_args_for_tool[index] += *delta->function.arguments;
        }
      }
    }

    // HACK (pythonic_tool_parser.py:186-187): mark prev_tool_call_arr non-empty
    // so the serving layer sets finish_reason=tool_calls.
    if (!tool_deltas.empty() && prev_tool_call_arr.empty()) {
      prev_tool_call_arr.push_back(nlohmann::json{{"arguments", nlohmann::json::object()}});
    }

    if (!tool_deltas.empty()) {
      DeltaMessage msg;
      msg.tool_calls = std::move(tool_deltas);
      return msg;
    }
    if (added_text.empty() && current_tool_id > 0) {
      // Emit an empty content delta once all calls are done (finish_reason).
      DeltaMessage msg;
      msg.content = "";
      return msg;
    }
    return std::nullopt;
  } catch (const pythonic_core::ToolDeltaError&) {
    return std::nullopt;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

ExtractedToolCallInformation PythonicToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  return ExtractClean(model_output);
}

std::optional<DeltaMessage> PythonicToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // pythonic_tool_parser.py:127 - not a list -> stream delta as plain content.
  if (current_text.empty() || current_text.front() != '[') {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }
  return StreamClean(current_text);
}

}  // namespace vllm::entrypoints::openai
