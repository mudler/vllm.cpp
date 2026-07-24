// Ported from: vllm/tool_parsers/granite4_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/granite4.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

// tool_parsers/utils.py:42 (partial_tag_overlap): longest prefix of `tag` that
// matches a suffix of `text` (0 when none).
std::size_t PartialTagOverlap(const std::string& text, const std::string& tag) {
  const std::size_t max_check = std::min(tag.size() - 1, text.size());
  for (std::size_t k = max_check; k >= 1; --k) {
    if (text.compare(text.size() - k, k, tag, 0, k) == 0) return k;
  }
  return 0;
}

// granite4_tool_parser.py:33-37 (dump_args): a str passes through unchanged; a
// dict/list is json.dumps'd; None -> None (dropped by exclude_none downstream).
// ordered_json keeps the serialized arguments in author (insertion) order, like
// CPython json.dumps.
std::optional<std::string> DumpArgs(const nlohmann::ordered_json& args) {
  if (args.is_null()) return std::nullopt;
  if (args.is_string()) return args.get<std::string>();
  return args.dump();
}

struct ParsedFunc {
  std::string name;
  std::optional<std::string> arguments;
};

}  // namespace

// granite4_tool_parser.py:74-90 (_collect_results). json.loads each tool-call
// block, record it in prev_tool_call_arr, and pull out {name, arguments}.
namespace {
std::pair<std::string, std::vector<ParsedFunc>> CollectResults(
    std::vector<nlohmann::ordered_json>& prev_tool_call_arr,
    const std::vector<std::string>& text_segments,
    const std::vector<std::string>& tc_segments) {
  std::vector<ParsedFunc> funcs;
  for (const std::string& tc_text : tc_segments) {
    const nlohmann::ordered_json tc =
        nlohmann::ordered_json::parse(tc_text);  // throws if bad
    if (!tc.is_object()) throw std::runtime_error("tool call is not an object");
    prev_tool_call_arr.push_back(tc);
    ParsedFunc f;
    f.name = tc.at("name").get<std::string>();
    f.arguments = DumpArgs(tc.at("arguments"));
    funcs.push_back(std::move(f));
  }
  std::string content;
  for (const std::string& t : text_segments) content += t;
  return {content, funcs};
}
}  // namespace

ExtractedToolCallInformation Granite4ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // granite4_tool_parser.py:97-99 - default: no tools, content = full output.
  ExtractedToolCallInformation msg{false, {}, model_output};
  const std::string tc_start = kTcStart;
  const std::string tc_end = kTcEnd;
  try {
    // granite4_tool_parser.py:101-131 - walk the interleaved TC_START/TC_END
    // delimiters, collecting text vs tool-call segments.
    std::vector<std::string> text_segments;
    std::vector<std::string> tc_segments;
    std::size_t last_cut_loc = 0;
    std::size_t cursor = 0;
    bool in_tc = false;  // local (upstream uses self.in_tc); ends False.

    for (;;) {
      const std::size_t s_pos = model_output.find(tc_start, cursor);
      const std::size_t e_pos = model_output.find(tc_end, cursor);
      if (s_pos == std::string::npos && e_pos == std::string::npos) break;

      const bool is_start = s_pos != std::string::npos &&
                            (e_pos == std::string::npos || s_pos < e_pos);
      if (is_start) {
        if (in_tc) throw std::runtime_error("Two tool call start tokens in a row");
        const std::string preceding =
            model_output.substr(last_cut_loc, s_pos - last_cut_loc);
        if (!preceding.empty()) text_segments.push_back(preceding);
        in_tc = true;
        last_cut_loc = s_pos + tc_start.size();
        cursor = last_cut_loc;
      } else {
        if (!in_tc) throw std::runtime_error("Tool call end token without start");
        const std::string tool_text =
            model_output.substr(last_cut_loc, e_pos - last_cut_loc);
        if (tool_text.empty()) throw std::runtime_error("Empty tool call text");
        tc_segments.push_back(tool_text);
        in_tc = false;
        last_cut_loc = e_pos + tc_end.size();
        cursor = last_cut_loc;
      }
    }
    if (in_tc) throw std::runtime_error("Incomplete tool call");
    if (last_cut_loc < model_output.size()) {
      text_segments.push_back(model_output.substr(last_cut_loc));
    }

    auto [content, funcs] =
        CollectResults(granite4_prev_tool_call_arr_, text_segments, tc_segments);

    std::vector<ToolCall> tool_calls;
    for (const ParsedFunc& f : funcs) {
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = f.name;
      tc.function.arguments = f.arguments.value_or("");
      tool_calls.push_back(std::move(tc));
    }
    msg.tools_called = !tool_calls.empty();
    msg.tool_calls = std::move(tool_calls);
    msg.content = content.empty() ? std::optional<std::string>{}
                                  : std::optional<std::string>{content};
  } catch (const std::exception&) {
    // granite4_tool_parser.py:146-147 - on error keep the default (full content).
  }
  return msg;
}

std::tuple<bool, std::string, std::string> Granite4ToolParser::tool_extraction_step(
    const std::string& delta) {
  const std::string tc_start = kTcStart;
  const std::string tc_end = kTcEnd;

  // granite4_tool_parser.py:156-160 - locate a full or partial start token.
  int start_token_pos = -1;
  int start_token_end = -1;
  std::size_t partial_start = 0;
  const std::size_t full = delta.find(tc_start);
  if (full != std::string::npos) {
    start_token_pos = static_cast<int>(full);
    start_token_end = static_cast<int>(full + tc_start.size());
  } else {
    const std::size_t ov = PartialTagOverlap(delta, tc_start);
    if (ov > 0) {
      start_token_pos = -2;  // there IS a partial match
      partial_start = delta.size() - ov;
    }
  }

  // granite4_tool_parser.py:162-163 - locate the end token.
  int end_token_pos = -1;
  int end_token_end = -1;
  const std::size_t ep = delta.find(tc_end);
  if (ep != std::string::npos) {
    end_token_pos = static_cast<int>(ep);
    end_token_end = static_cast<int>(ep + tc_end.size());
  }

  bool done = true;
  std::string content;
  std::string tc_text;

  if (start_token_pos < 0) {
    // granite4_tool_parser.py:170-177 - just streaming text so far.
    if (start_token_pos == -2) {
      content = delta.substr(0, partial_start);
      look_ahead_ = delta.substr(partial_start);
    } else {
      content = delta;
    }
  } else if (!in_tc_) {
    // granite4_tool_parser.py:179-190 - entering a new tool call.
    in_tc_ = true;
    content = delta.substr(0, static_cast<std::size_t>(start_token_pos));
    if (end_token_pos > 0) {
      tc_text = delta.substr(static_cast<std::size_t>(start_token_end),
                             static_cast<std::size_t>(end_token_pos - start_token_end));
      look_ahead_ = delta.substr(static_cast<std::size_t>(end_token_end));
      done = false;  // there could be more content already buffered
    } else {
      look_ahead_ = delta.substr(static_cast<std::size_t>(start_token_pos));
    }
  } else if (end_token_pos < 0) {
    // granite4_tool_parser.py:192-195 - between start and end tokens.
    look_ahead_ = delta;
  } else {
    // granite4_tool_parser.py:196-202 - found the end.
    tc_text = delta.substr(static_cast<std::size_t>(start_token_end),
                           static_cast<std::size_t>(end_token_pos - start_token_end));
    in_tc_ = false;
    look_ahead_ = delta.substr(static_cast<std::size_t>(end_token_end));
    done = false;  // there could be more content already buffered
  }
  return {done, content, tc_text};
}

std::optional<DeltaMessage> Granite4ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  try {
    // granite4_tool_parser.py:216-228 - drain the buffered delta.
    bool done = false;
    std::vector<std::string> text_segments;
    std::vector<std::string> tc_segments;
    std::string delta = delta_text;
    while (!done) {
      delta = look_ahead_ + delta;
      look_ahead_.clear();
      auto [d, content, tc_text] = tool_extraction_step(delta);
      done = d;
      if (!content.empty()) text_segments.push_back(content);
      if (!tc_text.empty()) tc_segments.push_back(tc_text);
      delta.clear();
    }

    // granite4_tool_parser.py:230-232.
    auto [content, funcs] =
        CollectResults(granite4_prev_tool_call_arr_, text_segments, tc_segments);

    // granite4_tool_parser.py:234-248 - one DeltaToolCall per completed block.
    std::vector<DeltaToolCall> delta_tool_calls;
    for (const ParsedFunc& f : funcs) {
      ++current_tool_id;
      DeltaToolCall d;
      d.id = make_tool_call_id();
      d.type = "function";
      d.index = current_tool_id;
      d.function.name = f.name;
      if (f.arguments.has_value()) d.function.arguments = *f.arguments;
      delta_tool_calls.push_back(std::move(d));
      streamed_args_for_tool.push_back(f.arguments.value_or(""));
    }

    // granite4_tool_parser.py:250-252.
    DeltaMessage msg;
    if (!content.empty()) msg.content = content;
    if (!delta_tool_calls.empty()) msg.tool_calls = std::move(delta_tool_calls);
    if (msg.content.has_value() ||
        (msg.tool_calls.has_value() && !msg.tool_calls->empty())) {
      return msg;
    }
  } catch (const std::exception&) {
    // granite4_tool_parser.py:254-255 - swallow errors, emit nothing.
  }
  return std::nullopt;
}

}  // namespace vllm::entrypoints::openai
