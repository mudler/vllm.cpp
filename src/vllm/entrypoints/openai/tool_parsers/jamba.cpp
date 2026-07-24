// Ported from: vllm/tool_parsers/jamba_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/jamba.h"

#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/utils.h"

namespace vllm::entrypoints::openai {

namespace {

// jamba_tool_parser.py:52 - re.compile(r"<tool_calls>(.*?)</tool_calls>",
// DOTALL). ECMAScript `.` is not DOTALL, so [\s\S] emulates it; `*?` non-greedy.
const std::regex& ToolCallsRegex() {
  static const std::regex re(R"(<tool_calls>([\s\S]*?)</tool_calls>)");
  return re;
}

// str.replace(old, new) - replace ALL occurrences (Python str.replace).
std::string ReplaceAll(std::string s, const std::string& from,
                       const std::string& to) {
  if (from.empty()) return s;
  std::size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

// Python truthiness for a parsed JSON value.
bool JsonTruthy(const nlohmann::ordered_json& v) {
  if (v.is_null()) return false;
  if (v.is_string()) return !v.get<std::string>().empty();
  if (v.is_object() || v.is_array()) return !v.empty();
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number_float()) return v.get<double>() != 0.0;
  if (v.is_number()) return v.get<int64_t>() != 0;
  return true;
}

// obj.get("arguments") -> value or nullopt.
std::optional<nlohmann::ordered_json> GetArgs(const nlohmann::ordered_json& obj) {
  if (obj.is_object() && obj.contains("arguments")) return obj.at("arguments");
  return std::nullopt;
}

}  // namespace

ExtractedToolCallInformation JambaToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  const std::string start = kToolCallsStart;

  // jamba_tool_parser.py:86-90 - sanity check: no start marker -> plain content.
  if (model_output.find(start) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    // jamba_tool_parser.py:95 - findall(...)[0]: first <tool_calls>...</> body.
    std::smatch m;
    if (!std::regex_search(model_output, m, ToolCallsRegex())) {
      throw std::runtime_error("no complete tool_calls block");
    }
    const std::string function_calls = m[1].str();

    // jamba_tool_parser.py:99-112 - json.loads + build ToolCalls.
    const nlohmann::ordered_json raw_function_calls =
        nlohmann::ordered_json::parse(function_calls);
    std::vector<ToolCall> tool_calls;
    for (const nlohmann::ordered_json& fc : raw_function_calls) {
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = fc.at("name").get<std::string>();
      // arguments are JSON but stored AS A STRING (json.dumps ensure_ascii=False).
      tc.function.arguments = fc.at("arguments").dump();
      tool_calls.push_back(std::move(tc));
    }

    // jamba_tool_parser.py:114-119 - content before the marker; None when blank
    // or exactly a single space.
    const std::string content =
        model_output.substr(0, model_output.find(start));
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    if (!content.empty() && content != " ") info.content = content;
    return info;

  } catch (const std::exception&) {
    // jamba_tool_parser.py:121-125 - any error -> whole output as content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> JambaToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  const std::string start = kToolCallsStart;
  const std::string end = kToolCallsEnd;

  // jamba_tool_parser.py:139-140 - no start marker -> stream as content.
  if (current_text.find(start) == std::string::npos) {
    DeltaMessage m;
    m.content = delta_text;
    return m;
  }

  // DEVIATION: upstream returns None here when the delta is JUST the lone start
  // token (`start_token_id in delta_token_ids and len==1`). We rely instead on
  // the empty/partial-array path below (tool_call_arr length 0 -> nullopt), which
  // is reached whenever nothing meaningful follows <tool_calls> yet.

  // jamba_tool_parser.py:159 - Allow.ALL vs Allow.ALL & ~Allow.STR.
  const bool allow_partial_str = current_tool_name_sent;

  try {
    // jamba_tool_parser.py:162-164 - text between the last start marker and the
    // (possibly absent) end marker.
    std::string parsable = current_text.substr(
        current_text.rfind(start) + start.size());
    const std::size_t end_pos = parsable.find(end);
    if (end_pos != std::string::npos) parsable = parsable.substr(0, end_pos);

    nlohmann::ordered_json tool_call_arr;
    try {
      // jamba_tool_parser.py:169 - partial JSON parse of the ARRAY.
      tool_call_arr = partial_json_loads(parsable, allow_partial_str).first;
    } catch (const MalformedPartialJson&) {
      // jamba_tool_parser.py:172-174 - not enough tokens yet.
      return std::nullopt;
    }
    if (!tool_call_arr.is_array()) return std::nullopt;

    const int arr_size = static_cast<int>(tool_call_arr.size());

    // jamba_tool_parser.py:178-180 - current_tool_call = arr[current_tool_id]
    // (Python negative index selects from the end when current_tool_id == -1).
    const auto current_tool_call = [&]() -> nlohmann::ordered_json {
      if (arr_size == 0) return nlohmann::ordered_json::object();
      int idx = current_tool_id;
      if (idx < 0) idx += arr_size;
      if (idx < 0 || idx >= arr_size) return nlohmann::ordered_json::object();
      return tool_call_arr[static_cast<std::size_t>(idx)];
    }();

    std::optional<DeltaMessage> delta;  // None

    // jamba_tool_parser.py:184-185 - only brackets so far -> stream nothing.
    if (arr_size == 0) {
      return std::nullopt;
    }

    // jamba_tool_parser.py:189-223 - a NEW tool has appeared in the array.
    if (arr_size > current_tool_id + 1) {
      if (current_tool_id >= 0) {
        const std::optional<nlohmann::ordered_json> diff_args =
            GetArgs(current_tool_call);
        if (diff_args.has_value() && JsonTruthy(*diff_args)) {
          // Flush any auto-completed args not yet streamed for the prev tool.
          std::string diff = diff_args->dump();
          diff = ReplaceAll(
              diff, streamed_args_for_tool[static_cast<std::size_t>(
                        current_tool_id)],
              "");
          DeltaToolCall d;
          d.index = current_tool_id;
          d.function.arguments = diff;
          DeltaMessage m;
          m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
          delta = std::move(m);
          streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] +=
              diff;
        }
      }
      // re-set progress for the new tool.
      current_tool_id = arr_size - 1;
      current_tool_name_sent = false;
      streamed_args_for_tool.emplace_back();
      jamba_prev_tool_call_arr_.clear();
      for (const nlohmann::ordered_json& e : tool_call_arr)
        jamba_prev_tool_call_arr_.push_back(e);
      return delta;
    }

    // jamba_tool_parser.py:229-246 - send the tool name once, if available.
    if (!current_tool_name_sent) {
      if (current_tool_call.is_object() && current_tool_call.contains("name") &&
          JsonTruthy(current_tool_call.at("name"))) {
        const std::string function_name =
            current_tool_call.at("name").get<std::string>();
        DeltaToolCall d;
        d.index = current_tool_id;
        d.type = "function";
        d.id = make_tool_call_id();
        d.function.name = function_name;
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        delta = std::move(m);
        current_tool_name_sent = true;
      }
    } else {
      // jamba_tool_parser.py:250-315 - stream the argument diff.
      std::optional<nlohmann::ordered_json> prev_arguments;
      if (static_cast<std::size_t>(current_tool_id) <
          jamba_prev_tool_call_arr_.size()) {
        prev_arguments = GetArgs(
            jamba_prev_tool_call_arr_[static_cast<std::size_t>(current_tool_id)]);
      }
      const std::optional<nlohmann::ordered_json> cur_arguments =
          GetArgs(current_tool_call);

      const std::string new_text = ReplaceAll(delta_text, "'", "\"");

      const bool cur_truthy =
          cur_arguments.has_value() && JsonTruthy(*cur_arguments);
      const bool prev_truthy =
          prev_arguments.has_value() && JsonTruthy(*prev_arguments);

      if (!cur_truthy && !prev_truthy) {
        // no args yet.
      } else if (!cur_truthy && prev_truthy) {
        // INVARIANT - impossible; drop.
      } else if (cur_truthy && !prev_truthy) {
        // jamba_tool_parser.py:265-285 - first arguments fragment.
        const std::string cur_json = cur_arguments->dump();
        const std::size_t pos = cur_json.find(new_text);
        if (pos == std::string::npos) {
          // Python str.index would raise -> caught -> None, prev NOT updated.
          return std::nullopt;
        }
        const std::string arguments_delta =
            cur_json.substr(0, pos + new_text.size());
        DeltaToolCall d;
        d.index = current_tool_id;
        d.function.arguments = arguments_delta;
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        delta = std::move(m);
        streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] +=
            arguments_delta;
      } else if (cur_truthy && prev_truthy) {
        // jamba_tool_parser.py:287-310 - subsequent diff.
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

    // jamba_tool_parser.py:320 - prev_tool_call_arr = tool_call_arr.
    jamba_prev_tool_call_arr_.clear();
    for (const nlohmann::ordered_json& e : tool_call_arr)
      jamba_prev_tool_call_arr_.push_back(e);
    return delta;

  } catch (const std::exception&) {
    // jamba_tool_parser.py:323-328 - drop the chunk on any error.
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
