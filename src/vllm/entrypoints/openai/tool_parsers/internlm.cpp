// Ported from: vllm/tool_parsers/internlm2_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/internlm.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/utils.h"

namespace vllm::entrypoints::openai {

namespace {

bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
std::string Strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && IsSpace(s[b])) ++b;
  while (e > b && IsSpace(s[e - 1])) --e;
  return s.substr(b, e - b);
}

// internlm2_tool_parser.py:50 (get_arguments). Returns the "parameters" value if
// present, else the "arguments" value, else nullopt (upstream returns None).
std::optional<nlohmann::ordered_json> GetArguments(
    const nlohmann::ordered_json& obj) {
  if (obj.is_object() && obj.contains("parameters")) return obj.at("parameters");
  if (obj.is_object() && obj.contains("arguments")) return obj.at("arguments");
  return std::nullopt;
}

// Python truthiness for a parsed JSON value (empty dict/list/string/0/false/null
// are falsy). Upstream guards arg streaming with `if cur_arguments:`.
bool JsonTruthy(const nlohmann::ordered_json& v) {
  if (v.is_null()) return false;
  if (v.is_string()) return !v.get<std::string>().empty();
  if (v.is_object() || v.is_array()) return !v.empty();
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number_float()) return v.get<double>() != 0.0;
  if (v.is_number()) return v.get<int64_t>() != 0;
  return true;
}

}  // namespace

ExtractedToolCallInformation Internlm2ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  const std::string text_in = model_output;
  const std::string plugin = kPluginStart;

  // internlm2_tool_parser.py:204-229.
  const std::size_t marker = text_in.find(plugin);
  if (marker == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, text_in};
  }

  // text, action = text.split("<|action_start|><|plugin|>") (first occurrence).
  const std::string text = text_in.substr(0, marker);
  std::string action = text_in.substr(marker + plugin.size());

  // action = action.split("<|action_end|>")[0].
  const std::size_t end = action.find(kActionEnd);
  if (end != std::string::npos) action = action.substr(0, end);

  // action = action[action.find("{"):]. Python's str.find returns -1 on miss,
  // and s[-1:] keeps the last char; replicate so a "{"-less body still reaches
  // json.loads and raises (upstream has no guard here).
  const std::size_t brace = action.find('{');
  if (brace != std::string::npos) {
    action = action.substr(brace);
  } else if (!action.empty()) {
    action = action.substr(action.size() - 1);
  }

  // action_dict = json.loads(action). No try/except upstream -> may throw.
  const nlohmann::ordered_json action_dict = nlohmann::ordered_json::parse(action);

  const std::string name = action_dict.at("name").get<std::string>();
  // parameters = json.dumps(action_dict.get("parameters",
  //   action_dict.get("arguments", {})), ensure_ascii=False).
  nlohmann::ordered_json params;
  if (action_dict.contains("parameters")) {
    params = action_dict.at("parameters");
  } else if (action_dict.contains("arguments")) {
    params = action_dict.at("arguments");
  } else {
    params = nlohmann::ordered_json::object();
  }

  // NOTE: upstream's `if not tools or name not in [...]` branch is DEAD CODE (it
  // constructs an ExtractedToolCallInformation but never returns it), so we omit
  // the tool-name validation entirely - see the header deviation note.

  ToolCall tc;
  tc.id = make_tool_call_id();
  tc.type = "function";
  tc.function.name = name;
  tc.function.arguments = params.dump();

  ExtractedToolCallInformation info;
  info.tools_called = true;
  info.tool_calls.push_back(std::move(tc));
  if (!text.empty()) info.content = text;
  return info;
}

std::optional<DeltaMessage> Internlm2ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  const std::string plugin = kPluginStart;

  // internlm2_tool_parser.py:67-69 - no action marker yet -> stream as content.
  if (current_text.find(kActionStart) == std::string::npos) {
    position_ = current_text.size();
    DeltaMessage m;
    m.content = delta_text;
    return m;
  }
  // internlm2_tool_parser.py:71-73 - tool already fully sent -> empty content so
  // the finish_reason is delivered.
  if (current_tool_id > 0) {
    DeltaMessage m;
    m.content = std::string();
    return m;
  }

  const std::size_t last_pos = position_;
  const std::string tail = current_text.substr(last_pos);
  // internlm2_tool_parser.py:76-77 - the plugin marker hasn't fully arrived yet.
  const std::size_t plugin_pos = tail.find(plugin);
  if (plugin_pos == std::string::npos) return std::nullopt;

  // internlm2_tool_parser.py:79-84 - text before the marker is leading content.
  const std::string leading = tail.substr(0, plugin_pos);
  std::string action = tail.substr(plugin_pos + plugin.size());
  if (!leading.empty()) {
    position_ += leading.size();
    DeltaMessage m;
    m.content = leading;
    return m;
  }

  // internlm2_tool_parser.py:86-87 - strip + cut at the end marker.
  action = Strip(action);
  const std::size_t end = action.find(kActionEnd);
  if (end != std::string::npos) action = action.substr(0, end);

  // internlm2_tool_parser.py:93 - Allow.ALL vs Allow.ALL & ~Allow.STR.
  const bool allow_partial_str = current_tool_name_sent;

  try {
    nlohmann::ordered_json tool_call_arr;
    try {
      // internlm2_tool_parser.py:101 - partial JSON parse of the OBJECT.
      tool_call_arr = partial_json_loads(action, allow_partial_str).first;
    } catch (const MalformedPartialJson&) {
      // internlm2_tool_parser.py:102-104 - not enough tokens yet.
      return std::nullopt;
    }

    std::optional<DeltaMessage> delta;  // None

    if (!current_tool_name_sent) {
      // internlm2_tool_parser.py:108-127 - send the tool name once, if present.
      if (tool_call_arr.is_object() && tool_call_arr.contains("name") &&
          JsonTruthy(tool_call_arr.at("name"))) {
        const std::string function_name =
            tool_call_arr.at("name").get<std::string>();
        current_tool_id += 1;
        DeltaToolCall d;
        d.index = current_tool_id;
        d.type = "function";
        d.id = make_tool_call_id();
        d.function.name = function_name;
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        delta = std::move(m);
        current_tool_name_sent = true;
        streamed_args_for_tool.emplace_back();
      }
    } else {
      // internlm2_tool_parser.py:130-182 - stream the argument diff.
      std::optional<nlohmann::ordered_json> prev_arguments;
      if (internlm_prev_tool_call_.has_value()) {
        prev_arguments = GetArguments(*internlm_prev_tool_call_);
      }
      const std::optional<nlohmann::ordered_json> cur_arguments =
          GetArguments(tool_call_arr);

      const bool cur_truthy =
          cur_arguments.has_value() && JsonTruthy(*cur_arguments);
      const bool prev_truthy =
          prev_arguments.has_value() && JsonTruthy(*prev_arguments);

      if (!cur_truthy && !prev_truthy) {
        // no arguments yet.
      } else if (!cur_truthy && prev_truthy) {
        // INVARIANT - impossible; drop.
      } else if (cur_truthy && !prev_truthy) {
        // internlm2_tool_parser.py:146-162 - first arguments fragment.
        const std::string cur_json = cur_arguments->dump();
        const std::size_t pos = cur_json.find(delta_text);
        if (pos == std::string::npos) {
          // Python str.index would raise -> caught -> None, prev NOT updated.
          return std::nullopt;
        }
        const std::string arguments_delta =
            cur_json.substr(0, pos + delta_text.size());
        DeltaToolCall d;
        d.index = current_tool_id;
        d.function.arguments = arguments_delta;
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        delta = std::move(m);
        streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] +=
            arguments_delta;
      } else {
        // internlm2_tool_parser.py:164-182 - subsequent diff.
        const std::string cur_json = cur_arguments->dump();
        const std::string prev_json = prev_arguments->dump();
        const std::string argument_diff =
            extract_intermediate_diff(cur_json, prev_json);
        DeltaToolCall d;
        d.index = current_tool_id;
        d.function.arguments = argument_diff;
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        delta = std::move(m);
        streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] +=
            argument_diff;
      }
    }

    // internlm2_tool_parser.py:187-189 - tool_call_arr["arguments"] =
    // get_arguments(tool_call_arr); prev = [tool_call_arr].
    nlohmann::ordered_json to_store = tool_call_arr;
    const std::optional<nlohmann::ordered_json> args = GetArguments(tool_call_arr);
    to_store["arguments"] = args.has_value() ? *args : nlohmann::ordered_json();
    internlm_prev_tool_call_ = std::move(to_store);
    return delta;

  } catch (const std::exception&) {
    // internlm2_tool_parser.py:190-195 - drop the chunk on any error.
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
