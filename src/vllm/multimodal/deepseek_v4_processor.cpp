#include "vllm/multimodal/deepseek_v4_processor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vt/dtype.h"

namespace vllm::multimodal {
namespace {

using Json = nlohmann::ordered_json;

constexpr const char* kBos = "<｜begin▁of▁sentence｜>";
constexpr const char* kEos = "<｜end▁of▁sentence｜>";
constexpr const char* kUser = "<｜User｜>";
constexpr const char* kAssistant = "<｜Assistant｜>";
constexpr const char* kLatestReminder = "<｜latest_reminder｜>";
constexpr const char* kThinkStart = "<think>";
constexpr const char* kThinkEnd = "</think>";
constexpr const char* kDsml = "｜DSML｜";
constexpr int64_t kCompressPadTo = 4;

struct TaskTransition {
  const char* name;
  const char* token;
  bool opens_assistant;
};

constexpr TaskTransition kTaskTransitions[] = {
    {"action", "<｜action｜>", true},
    {"query", "<｜query｜>", false},
    {"authority", "<｜authority｜>", false},
    {"domain", "<｜domain｜>", false},
    {"title", "<｜title｜>", false},
    {"read_url", "<｜read_url｜>", false},
};

std::string StringValue(const Json& value) {
  return value.is_string() ? value.get<std::string>() : std::string();
}

std::string FieldString(const Json& object, const char* key) {
  const auto it = object.find(key);
  return it != object.end() ? StringValue(*it) : std::string();
}

bool FieldBool(const Json& object, const char* key, bool fallback = false) {
  const auto it = object.find(key);
  return it != object.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

std::string PythonJson(const Json& value) {
  if (value.is_array()) {
    std::string out = "[";
    for (size_t i = 0; i < value.size(); ++i) {
      if (i != 0) out += ", ";
      out += PythonJson(value[i]);
    }
    return out + "]";
  }
  if (value.is_object()) {
    std::string out = "{";
    size_t i = 0;
    for (auto it = value.begin(); it != value.end(); ++it, ++i) {
      if (i != 0) out += ", ";
      out += Json(it.key()).dump();
      out += ": ";
      out += PythonJson(it.value());
    }
    return out + "}";
  }
  return value.dump();
}

bool PythonTruthy(const Json& value) {
  if (value.is_null()) return false;
  if (value.is_boolean()) return value.get<bool>();
  if (value.is_number()) return value.get<double>() != 0.0;
  if (value.is_string()) {
    return !value.get_ref<const std::string&>().empty();
  }
  if (value.is_array() || value.is_object()) return !value.empty();
  if (value.is_binary()) return !value.get_binary().empty();
  return true;
}

Json ExtractImage(const Json& block) {
  Json record = Json::object();
  record["type"] = "image";
  if (FieldString(block, "type") == "image_url") {
    const auto it = block.find("image_url");
    if (it != block.end() && it->is_string()) {
      record["url"] = *it;
    } else if (it != block.end() && it->is_object()) {
      record["url"] = FieldString(*it, "url");
    } else {
      record["url"] = "";
    }
  } else {
    for (const char* key : {"source", "url", "data"}) {
      const auto it = block.find(key);
      if (it != block.end()) record[key] = *it;
    }
  }
  bool found = false;
  for (const char* key : {"source", "url", "data"}) {
    const auto it = record.find(key);
    found = found || (it != record.end() && PythonTruthy(*it));
  }
  if (!found) {
    throw std::invalid_argument("Image block does not contain a valid source");
  }
  return record;
}

std::pair<Json, std::vector<Json>> ProcessImageBlocks(const Json& blocks) {
  Json output = Json::array();
  std::vector<Json> images;
  for (const Json& original : blocks) {
    if (!original.is_object()) {
      output.push_back(original);
      continue;
    }
    const std::string type = FieldString(original, "type");
    if (type == "image" || type == "image_url") {
      output.push_back(Json{{"type", "text"},
                            {"text", kDeepSeekV4ImagePlaceholder}});
      images.push_back(ExtractImage(original));
    } else if (type == "tool_result") {
      Json block = original;
      auto content = block.find("content");
      if (content != block.end() && content->is_array()) {
        auto nested = ProcessImageBlocks(*content);
        block["content"] = std::move(nested.first);
        images.insert(images.end(),
                      std::make_move_iterator(nested.second.begin()),
                      std::make_move_iterator(nested.second.end()));
      }
      output.push_back(std::move(block));
    } else if (type == "text") {
      const std::string text = FieldString(original, "text");
      if (text.find(kDeepSeekV4ImagePlaceholder) != std::string::npos) {
        throw std::invalid_argument(
            "Text block contains image placeholder '" +
            std::string(kDeepSeekV4ImagePlaceholder) + "': '" +
            text.substr(0, 100) +
            "'. Images should be separate content blocks.");
      }
      output.push_back(original);
    } else {
      output.push_back(original);
    }
  }
  return {std::move(output), std::move(images)};
}

std::pair<Json, std::vector<Json>> ProcessImageMessages(const Json& messages) {
  if (!messages.is_array()) {
    throw std::invalid_argument("messages must be an array");
  }
  Json processed = Json::array();
  std::vector<Json> images;
  for (const Json& original : messages) {
    if (!original.is_object()) {
      throw std::invalid_argument("each message must be an object");
    }
    Json message = original;
    for (const char* field : {"content", "reasoning_content"}) {
      const auto it = message.find(field);
      if (it != message.end() && it->is_string() &&
          it->get_ref<const std::string&>().find(
              kDeepSeekV4ImagePlaceholder) != std::string::npos) {
        if (std::string(field) == "content") {
          throw std::invalid_argument(
              "Message content contains image special token '" +
              std::string(kDeepSeekV4ImagePlaceholder) +
              "'. Images should be provided as image content blocks.");
        }
        throw std::invalid_argument(
            "reasoning_content contains image special token '" +
            std::string(kDeepSeekV4ImagePlaceholder) + "'");
      }
    }
    auto content = message.find("content");
    if (content != message.end() && content->is_array() &&
        message.find("content_blocks") == message.end()) {
      message["content_blocks"] = *content;
      message.erase("content");
    }
    auto blocks = message.find("content_blocks");
    if (blocks != message.end() && blocks->is_array() && !blocks->empty()) {
      auto result = ProcessImageBlocks(*blocks);
      message["content_blocks"] = std::move(result.first);
      images.insert(images.end(),
                    std::make_move_iterator(result.second.begin()),
                    std::make_move_iterator(result.second.end()));
      const auto current_content = message.find("content");
      if (current_content == message.end() || !current_content->is_string()) {
        std::string joined;
        bool first = true;
        for (const Json& block : message["content_blocks"]) {
          if (!block.is_object() || FieldString(block, "type") != "text") {
            continue;
          }
          if (!first) joined += "\n\n";
          joined += FieldString(block, "text");
          first = false;
        }
        message["content"] = std::move(joined);
      }
    }
    processed.push_back(std::move(message));
  }
  return {std::move(processed), std::move(images)};
}

Json MergeToolMessages(const Json& messages) {
  Json merged = Json::array();
  for (const Json& original : messages) {
    Json message = original;
    const std::string role = FieldString(message, "role");
    if (role == "tool") {
      Json block{{"type", "tool_result"},
                 {"tool_use_id", FieldString(message, "tool_call_id")},
                 {"content", FieldString(message, "content")}};
      if (!merged.empty() && FieldString(merged.back(), "role") == "user" &&
          merged.back().find("content_blocks") != merged.back().end()) {
        merged.back()["content_blocks"].push_back(std::move(block));
      } else {
        merged.push_back(Json{{"role", "user"},
                              {"content_blocks", Json::array({block})}});
      }
    } else if (role == "user") {
      Json blocks;
      const auto it = message.find("content_blocks");
      if (it == message.end() || it->is_null()) {
        blocks = Json::array(
            {Json{{"type", "text"}, {"text", FieldString(message, "content")}}});
      } else {
        blocks = *it;
      }
      if (!merged.empty() && FieldString(merged.back(), "role") == "user" &&
          merged.back().find("content_blocks") != merged.back().end() &&
          merged.back().find("task") == merged.back().end()) {
        for (const Json& block : blocks) {
          merged.back()["content_blocks"].push_back(block);
        }
      } else {
        message["content_blocks"] = std::move(blocks);
        merged.push_back(std::move(message));
      }
    } else {
      merged.push_back(std::move(message));
    }
  }
  return merged;
}

void SortToolResults(Json* messages) {
  std::vector<std::string> order;
  for (Json& message : *messages) {
    const std::string role = FieldString(message, "role");
    auto calls = message.find("tool_calls");
    if (role == "assistant" && calls != message.end() && calls->is_array() &&
        !calls->empty()) {
      order.clear();
      for (const Json& call : *calls) {
        std::string id = FieldString(call, "id");
        if (id.empty()) {
          const auto function = call.find("function");
          if (function != call.end() && function->is_object()) {
            id = FieldString(*function, "id");
          }
        }
        order.push_back(std::move(id));
      }
    } else if (role == "user" && !order.empty()) {
      auto blocks = message.find("content_blocks");
      if (blocks == message.end() || !blocks->is_array()) continue;
      std::vector<Json> tools;
      for (const Json& block : *blocks) {
        if (FieldString(block, "type") == "tool_result") tools.push_back(block);
      }
      if (tools.size() <= 1) continue;
      auto rank = [&order](const Json& block) {
        const std::string id = FieldString(block, "tool_use_id");
        const auto it = std::find(order.begin(), order.end(), id);
        return it == order.end() ? size_t{0}
                                 : static_cast<size_t>(it - order.begin());
      };
      std::stable_sort(tools.begin(), tools.end(),
                       [&rank](const Json& a, const Json& b) {
                         return rank(a) < rank(b);
                       });
      size_t index = 0;
      for (Json& block : *blocks) {
        if (FieldString(block, "type") == "tool_result") {
          block = tools[index++];
        }
      }
    }
  }
}

int64_t LastUserIndex(const Json& messages) {
  for (int64_t i = static_cast<int64_t>(messages.size()) - 1; i >= 0; --i) {
    const std::string role = FieldString(messages[static_cast<size_t>(i)], "role");
    if (role == "user" || role == "developer") return i;
  }
  return -1;
}

Json DropThinkingMessages(const Json& messages) {
  const int64_t last_user = LastUserIndex(messages);
  Json output = Json::array();
  for (size_t i = 0; i < messages.size(); ++i) {
    Json message = messages[i];
    const std::string role = FieldString(message, "role");
    const bool keep_role = role == "user" || role == "system" ||
                           role == "tool" || role == "latest_reminder" ||
                           role == "direct_search_results";
    if (keep_role || static_cast<int64_t>(i) >= last_user) {
      output.push_back(std::move(message));
    } else if (role == "assistant") {
      message.erase("reasoning_content");
      output.push_back(std::move(message));
    }
  }
  return output;
}

std::string EncodeArguments(const Json& tool_call) {
  const auto function = tool_call.find("function");
  const Json* source = function != tool_call.end() && function->is_object()
                           ? &*function
                           : &tool_call;
  const std::string raw = FieldString(*source, "arguments");
  Json arguments;
  try {
    arguments = Json::parse(raw);
  } catch (const std::exception&) {
    arguments = Json{{"arguments", raw}};
  }
  if (!arguments.is_object()) arguments = Json{{"arguments", raw}};
  std::string output;
  bool first = true;
  for (auto it = arguments.begin(); it != arguments.end(); ++it) {
    if (!first) output += "\n";
    first = false;
    output += "<" + std::string(kDsml) + "parameter name=\"" + it.key() +
              "\" string=\"" + (it->is_string() ? "true" : "false") +
              "\">";
    output += it->is_string() ? it->get<std::string>() : PythonJson(*it);
    output += "</" + std::string(kDsml) + "parameter>";
  }
  return output;
}

std::string RenderToolCalls(const Json& calls) {
  std::string body;
  bool first = true;
  for (const Json& call : calls) {
    const auto function = call.find("function");
    const Json* source = function != call.end() && function->is_object()
                             ? &*function
                             : &call;
    if (!first) body += "\n";
    first = false;
    body += "<" + std::string(kDsml) + "invoke name=\"" +
            FieldString(*source, "name") + "\">\n" + EncodeArguments(call) +
            "\n</" + std::string(kDsml) + "invoke>";
  }
  return "\n\n<" + std::string(kDsml) + "tool_calls>\n" + body + "\n</" +
         std::string(kDsml) + "tool_calls>";
}

std::string RenderContentBlocks(const Json& blocks) {
  std::string output;
  bool first = true;
  for (const Json& block : blocks) {
    std::string part;
    const std::string type = FieldString(block, "type");
    if (type == "text") {
      part = FieldString(block, "text");
    } else if (type == "tool_result") {
      std::string content;
      const auto value = block.find("content");
      if (value != block.end() && value->is_array()) {
        bool first_nested = true;
        for (const Json& nested : *value) {
          if (!first_nested) content += "\n\n";
          first_nested = false;
          if (FieldString(nested, "type") == "text") {
            content += FieldString(nested, "text");
          } else {
            content += "[Unsupported " + FieldString(nested, "type") + "]";
          }
        }
      } else if (value != block.end() && value->is_string()) {
        content = value->get<std::string>();
      }
      part = "<tool_result>" + content + "</tool_result>";
    } else {
      part = "[Unsupported " + type + "]";
    }
    if (!first) output += "\n\n";
    first = false;
    output += part;
  }
  return output;
}

std::string RenderTools(const Json& tools) {
  std::string schemas;
  bool first = true;
  for (const Json& tool : tools) {
    const auto function = tool.find("function");
    const Json& schema = function != tool.end() && function->is_object()
                             ? *function
                             : tool;
    if (!first) schemas += "\n";
    first = false;
    schemas += PythonJson(schema);
  }
  return "## Tools\n\n"
         "You have access to a set of tools to help answer the user's question. "
         "You can invoke tools by writing a \"<" +
         std::string(kDsml) +
         "tool_calls>\" block like the following:\n\n<" +
         std::string(kDsml) + "tool_calls>\n<" + std::string(kDsml) +
         "invoke name=\"$TOOL_NAME\">\n<" + std::string(kDsml) +
         "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">"
         "$PARAMETER_VALUE</" + std::string(kDsml) +
         "parameter>\n...\n</" + std::string(kDsml) +
         "invoke>\n<" + std::string(kDsml) +
         "invoke name=\"$TOOL_NAME2\">\n...\n</" + std::string(kDsml) +
         "invoke>\n</" + std::string(kDsml) +
         "tool_calls>\n\nString parameters should be specified as is and set "
         "`string=\"true\"`. For all other types (numbers, booleans, arrays, "
         "objects), pass the value in JSON format and set "
         "`string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by " +
         std::string(kThinkStart) +
         "), you MUST output your complete reasoning inside " +
         std::string(kThinkStart) + "..." + std::string(kThinkEnd) +
         " BEFORE any tool calls or final response.\n\nOtherwise, output "
         "directly after " +
         std::string(kThinkEnd) +
         " with tool calls or final response.\n\n### Available Tool Schemas\n\n" +
         schemas + "\n\nYou MUST strictly follow the above defined tool name and "
                   "parameter schemas to invoke tool calls.\n";
}

void AppendToolsAndResponseFormat(const Json& message, std::string* output) {
  const auto tools = message.find("tools");
  if (tools != message.end() && tools->is_array() && !tools->empty()) {
    *output += "\n\n" + RenderTools(*tools);
  }
  const auto response = message.find("response_format");
  if (response != message.end() && !response->is_null() && !response->empty()) {
    *output += "\n\n## Response Format:\n\nYou MUST strictly adhere to the "
               "following schema to reply:\n" +
               PythonJson(*response);
  }
}

std::string RenderMessage(size_t index, const Json& messages,
                          const std::string& thinking_mode, bool drop_thinking,
                          const std::string& reasoning_effort) {
  const Json& message = messages[index];
  const std::string role = FieldString(message, "role");
  const TaskTransition* task_transition = nullptr;
  const auto task = message.find("task");
  if (task != message.end() && !task->is_null()) {
    if (!task->is_string()) {
      throw std::invalid_argument("Invalid task: " + PythonJson(*task));
    }
    const std::string& task_name = task->get_ref<const std::string&>();
    for (const auto& candidate : kTaskTransitions) {
      if (task_name == candidate.name) {
        task_transition = &candidate;
        break;
      }
    }
    if (task_transition == nullptr) {
      throw std::invalid_argument("Invalid task: '" + task_name + "'");
    }
  }
  std::string output;
  if (index == 0 && thinking_mode == "thinking") {
    if (reasoning_effort == "high") {
      output +=
          "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
          "You MUST be very thorough in your thinking and comprehensively "
          "decompose the problem to resolve the root cause, rigorously "
          "stress-testing your logic against all potential paths, edge cases, "
          "and adversarial scenarios.\n"
          "Explicitly write out your entire deliberation process, documenting "
          "every intermediate step, considered alternative, and rejected "
          "hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
    } else if (reasoning_effort == "max") {
      output +=
          "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
          "uncompromising.\n"
          "You MUST reason with the utmost depth and rigor, leaving absolutely "
          "nothing to chance: exhaustively decompose the problem into its most "
          "fundamental components, trace every causal chain to its root, and "
          "resolve the underlying cause rather than any surface symptom.\n"
          "Do not stop reasoning until you have independently verified the "
          "solution from multiple angles and are certain that no assumption "
          "remains unchecked and no error remains undiscovered.\n\n";
    }
  }

  if (role == "system") {
    output += FieldString(message, "content");
    AppendToolsAndResponseFormat(message, &output);
  } else if (role == "developer") {
    const std::string content = FieldString(message, "content");
    if (content.empty()) {
      throw std::invalid_argument("Invalid message for role `developer`");
    }
    output += std::string(kUser) + content;
    AppendToolsAndResponseFormat(message, &output);
  } else if (role == "user") {
    output += kUser;
    const auto blocks = message.find("content_blocks");
    output += blocks != message.end() && blocks->is_array() && !blocks->empty()
                  ? RenderContentBlocks(*blocks)
                  : FieldString(message, "content");
  } else if (role == "latest_reminder") {
    output += std::string(kLatestReminder) + FieldString(message, "content");
  } else if (role == "tool") {
    throw std::invalid_argument(
        "deepseek_v4 merges tool messages into user; please preprocess with "
        "merge_tool_messages()");
  } else if (role == "assistant") {
    std::string reasoning;
    bool previous_has_task = false;
    if (index > 0) {
      const auto previous_task = messages[index - 1].find("task");
      previous_has_task =
          previous_task != messages[index - 1].end() && !previous_task->is_null();
    }
    if (thinking_mode == "thinking" && !previous_has_task) {
      if (!drop_thinking || static_cast<int64_t>(index) > LastUserIndex(messages)) {
        reasoning = FieldString(message, "reasoning_content") + kThinkEnd;
      }
    }
    output += reasoning + FieldString(message, "content");
    const auto calls = message.find("tool_calls");
    if (calls != message.end() && calls->is_array() && !calls->empty()) {
      output += RenderToolCalls(*calls);
    }
    if (!FieldBool(message, "wo_eos")) output += kEos;
  } else {
    throw std::invalid_argument("Unknown role: " + role);
  }

  if (index + 1 < messages.size()) {
    const std::string next = FieldString(messages[index + 1], "role");
    if (next != "assistant" && next != "latest_reminder") return output;
  }

  if (task_transition != nullptr) {
    if (task_transition->opens_assistant) {
      output += kAssistant;
      output += thinking_mode == "thinking" ? kThinkStart : kThinkEnd;
    }
    output += task_transition->token;
  } else if (role == "user" || role == "developer") {
    output += kAssistant;
    output +=
        thinking_mode == "thinking" &&
                (!drop_thinking ||
                 static_cast<int64_t>(index) >= LastUserIndex(messages))
            ? kThinkStart
            : kThinkEnd;
  }
  return output;
}

std::string EncodeMessagesText(Json messages, const std::string& thinking_mode,
                               Json context, bool drop_thinking,
                               bool add_default_bos_token,
                               const std::string& reasoning_effort) {
  if (thinking_mode != "chat" && thinking_mode != "thinking") {
    throw std::invalid_argument("Invalid thinking_mode `" + thinking_mode + "`");
  }
  if (reasoning_effort != "low" && reasoning_effort != "high" &&
      reasoning_effort != "max") {
    throw std::invalid_argument("Invalid reasoning effort: " + reasoning_effort);
  }
  context = MergeToolMessages(context);
  messages = MergeToolMessages(messages);
  Json full = context;
  for (const Json& message : messages) full.push_back(message);
  SortToolResults(&full);

  bool effective_drop = drop_thinking;
  for (const Json& message : full) {
    const auto tools = message.find("tools");
    if (tools != message.end() && !tools->empty()) effective_drop = false;
  }
  size_t context_length = context.size();
  if (thinking_mode == "thinking" && effective_drop) {
    full = DropThinkingMessages(full);
    context_length = DropThinkingMessages(context).size();
  }

  std::string prompt =
      add_default_bos_token && context.empty() ? kBos : std::string();
  for (size_t index = context_length; index < full.size(); ++index) {
    prompt += RenderMessage(index, full, thinking_mode, effective_drop,
                            reasoning_effort);
  }
  return prompt;
}

[[noreturn]] void ThrowOverflow(const char* message) {
  throw std::overflow_error(message);
}

int64_t CheckedAdd(int64_t left, int64_t right, const char* message) {
  if (left < 0 || right < 0 ||
      left > std::numeric_limits<int64_t>::max() - right) {
    ThrowOverflow(message);
  }
  return left + right;
}

int64_t CheckedMul(int64_t left, int64_t right, const char* message) {
  if (left < 0 || right < 0 ||
      (right != 0 && left > std::numeric_limits<int64_t>::max() / right)) {
    ThrowOverflow(message);
  }
  return left * right;
}

size_t CheckedSize(int64_t value, const char* message) {
  if (value < 0 ||
      static_cast<uint64_t>(value) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    ThrowOverflow(message);
  }
  return static_cast<size_t>(value);
}

int CheckedInt(int64_t value, const char* message) {
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    ThrowOverflow(message);
  }
  return static_cast<int>(value);
}

int32_t CheckedInt32(int64_t value, const char* message) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    ThrowOverflow(message);
  }
  return static_cast<int32_t>(value);
}

int64_t CheckedDoubleToInt64(double value, const char* message) {
  if (!std::isfinite(value) ||
      value < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      value >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
    ThrowOverflow(message);
  }
  return static_cast<int64_t>(value);
}

int32_t CheckedDoubleToInt32(double value, const char* message) {
  if (!std::isfinite(value) ||
      value < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
      value > static_cast<double>(std::numeric_limits<int32_t>::max())) {
    ThrowOverflow(message);
  }
  return static_cast<int32_t>(value);
}

template <typename T>
size_t CheckedVectorSize(int64_t value, const char* message) {
  const size_t size = CheckedSize(value, message);
  if (size > std::vector<T>().max_size()) ThrowOverflow(message);
  return size;
}

int64_t CheckedCeilDiv(int64_t value, int64_t divisor,
                       const char* message) {
  if (value < 0 || divisor <= 0) {
    throw std::invalid_argument(
        "DeepSeek-V4 checked ceil-div requires non-negative operands");
  }
  const int64_t quotient = value / divisor;
  return value % divisor == 0 ? quotient : CheckedAdd(quotient, 1, message);
}

int64_t CheckedPixelBytes(int64_t height, int64_t width,
                          const char* message) {
  return CheckedMul(CheckedMul(height, width, message), 3, message);
}

void ValidateGeometry(int64_t height, int64_t width, int64_t patch_size,
                      int64_t downsample_ratio) {
  if (height <= 0 || width <= 0 || patch_size <= 0 ||
      downsample_ratio <= 0) {
    throw std::invalid_argument("DeepSeek-V4 image geometry must be positive");
  }
}

void ValidateProcessorConfig(const DeepSeekV4ProcessorConfig& config) {
  if (config.patch_size <= 0) {
    throw std::invalid_argument(
        "DeepSeek-V4 processor patch size must be positive");
  }
  if (config.downsample_ratio <= 0) {
    throw std::invalid_argument(
        "DeepSeek-V4 processor downsample ratio must be positive");
  }
  if (config.max_image_tokens < 9) {
    throw std::invalid_argument(
        "DeepSeek-V4 processor image token budget must be at least 9");
  }
  if (config.min_pixels < 0) {
    throw std::invalid_argument(
        "DeepSeek-V4 processor minimum pixels must not be negative");
  }
  if (config.max_width_height_ratio < 0) {
    throw std::invalid_argument(
        "DeepSeek-V4 processor width-height ratio must not be negative");
  }
  if (config.vocab_size <= 0 ||
      config.vocab_size > std::numeric_limits<int32_t>::max() - kImageEnd) {
    throw std::invalid_argument(
        "DeepSeek-V4 processor vocabulary size is invalid");
  }
  CheckedMul(config.patch_size, config.downsample_ratio,
             "DeepSeek-V4 processor patch geometry overflow");
  CheckedMul(CheckedMul(3, config.patch_size,
                        "DeepSeek-V4 processor patch geometry overflow"),
             config.patch_size,
             "DeepSeek-V4 processor patch geometry overflow");
}

// Pillow 12.1.1 src/libImaging/Resample.c uses a=-0.5.
double Bicubic(double x) {
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5;
  return 0.0;
}

constexpr int kPillowPrecisionBits = 22;
constexpr int64_t kPillowCoefficientScale = int64_t{1}
                                             << kPillowPrecisionBits;

struct ResampleAxis {
  int64_t kernel_size = 0;
  std::vector<int64_t> starts;
  std::vector<int64_t> counts;
  std::vector<int32_t> coefficients;
};

ResampleAxis BuildAxis(int64_t input, int64_t output) {
  ValidateGeometry(1, input, 1, 1);
  ValidateGeometry(1, output, 1, 1);
  const double scale = static_cast<double>(input) / output;
  const double filter_scale = std::max(1.0, scale);
  const double support = 2.0 * filter_scale;
  const double rounded_support = std::ceil(support);
  if (!std::isfinite(rounded_support) ||
      rounded_support >
          static_cast<double>((std::numeric_limits<int64_t>::max() - 1) / 2)) {
    ThrowOverflow("DeepSeek-V4 resize coefficient count overflow");
  }

  ResampleAxis axis;
  axis.kernel_size =
      CheckedAdd(CheckedMul(CheckedDoubleToInt64(
                                rounded_support,
                                "DeepSeek-V4 resize coefficient count overflow"),
                            2,
                            "DeepSeek-V4 resize coefficient count overflow"),
                 1, "DeepSeek-V4 resize coefficient count overflow");
  axis.starts.resize(CheckedVectorSize<int64_t>(
      output, "DeepSeek-V4 resize coefficient count overflow"));
  axis.counts.resize(axis.starts.size());
  const int64_t coefficient_count =
      CheckedMul(output, axis.kernel_size,
                 "DeepSeek-V4 resize coefficient count overflow");
  axis.coefficients.assign(
      CheckedVectorSize<int32_t>(
          coefficient_count,
          "DeepSeek-V4 resize coefficient count overflow"),
      0);

  std::vector<double> weights(CheckedVectorSize<double>(
      axis.kernel_size, "DeepSeek-V4 resize coefficient count overflow"));
  for (int64_t out = 0; out < output; ++out) {
    const double center = (static_cast<double>(out) + 0.5) * scale;
    int64_t first = CheckedDoubleToInt64(
        center - support + 0.5,
        "DeepSeek-V4 resize coefficient bound overflow");
    if (first < 0) first = 0;
    int64_t last = CheckedDoubleToInt64(
        center + support + 0.5,
        "DeepSeek-V4 resize coefficient bound overflow");
    if (last > input) last = input;
    const int64_t count = last - first;
    double total = 0.0;
    for (int64_t tap = 0; tap < count; ++tap) {
      const double weight =
          Bicubic((static_cast<double>(tap + first) - center + 0.5) /
                  filter_scale);
      weights[static_cast<size_t>(tap)] = weight;
      total += weight;
    }
    const size_t coefficient_base = CheckedSize(
        CheckedMul(out, axis.kernel_size,
                   "DeepSeek-V4 resize coefficient offset overflow"),
        "DeepSeek-V4 resize coefficient offset overflow");
    for (int64_t tap = 0; tap < count; ++tap) {
      double weight = weights[static_cast<size_t>(tap)];
      if (total != 0.0) weight /= total;
      const double scaled =
          weight * static_cast<double>(kPillowCoefficientScale) +
          (weight < 0.0 ? -0.5 : 0.5);
      axis.coefficients[coefficient_base + static_cast<size_t>(tap)] =
          CheckedDoubleToInt32(
              scaled, "DeepSeek-V4 resize coefficient narrowing overflow");
    }
    axis.starts[static_cast<size_t>(out)] = first;
    axis.counts[static_cast<size_t>(out)] = count;
  }
  return axis;
}

uint8_t PillowClip(int64_t accumulator) {
  const int64_t rounded =
      accumulator >= 0
          ? accumulator / kPillowCoefficientScale
          : -((-accumulator + kPillowCoefficientScale - 1) /
              kPillowCoefficientScale);
  return static_cast<uint8_t>(std::clamp<int64_t>(rounded, 0, 255));
}

std::vector<uint8_t> ResizeRgb(std::span<const uint8_t> rgb, int64_t input_h,
                               int64_t input_w, int64_t output_h,
                               int64_t output_w) {
  ValidateGeometry(input_h, input_w, 1, 1);
  ValidateGeometry(output_h, output_w, 1, 1);
  const size_t input_bytes = CheckedSize(
      CheckedPixelBytes(input_h, input_w,
                        "DeepSeek-V4 RGB byte extent overflow"),
      "DeepSeek-V4 RGB byte extent overflow");
  if (rgb.size() != input_bytes) {
    throw std::invalid_argument(
        "DeepSeek-V4 RGB byte extent does not equal height*width*3");
  }
  if (input_h == output_h && input_w == output_w) {
    return std::vector<uint8_t>(rgb.begin(), rgb.end());
  }

  std::vector<uint8_t> intermediate;
  std::span<const uint8_t> vertical_input = rgb;
  int64_t vertical_input_h = input_h;
  if (input_w != output_w) {
    const ResampleAxis horizontal = BuildAxis(input_w, output_w);
    intermediate.resize(CheckedSize(
        CheckedPixelBytes(input_h, output_w,
                          "DeepSeek-V4 resized image byte size overflow"),
        "DeepSeek-V4 resized image byte size overflow"));
    const size_t source_width = static_cast<size_t>(input_w);
    const size_t target_width = static_cast<size_t>(output_w);
    for (size_t y = 0; y < static_cast<size_t>(input_h); ++y) {
      for (size_t x = 0; x < target_width; ++x) {
        const size_t coefficient_base =
            x * static_cast<size_t>(horizontal.kernel_size);
        const size_t source_start =
            static_cast<size_t>(horizontal.starts[x]);
        const size_t count = static_cast<size_t>(horizontal.counts[x]);
        for (size_t channel = 0; channel < 3; ++channel) {
          int64_t accumulator = int64_t{1}
                                << (kPillowPrecisionBits - 1);
          for (size_t tap = 0; tap < count; ++tap) {
            const size_t source =
                ((y * source_width + source_start + tap) * 3) + channel;
            accumulator +=
                static_cast<int64_t>(rgb[source]) *
                horizontal.coefficients[coefficient_base + tap];
          }
          intermediate[(y * target_width + x) * 3 + channel] =
              PillowClip(accumulator);
        }
      }
    }
    vertical_input = intermediate;
  }

  if (input_h == output_h) return intermediate;

  const ResampleAxis vertical = BuildAxis(vertical_input_h, output_h);
  std::vector<uint8_t> output(CheckedSize(
      CheckedPixelBytes(output_h, output_w,
                        "DeepSeek-V4 resized image byte size overflow"),
      "DeepSeek-V4 resized image byte size overflow"));
  const size_t target_width = static_cast<size_t>(output_w);
  for (size_t y = 0; y < static_cast<size_t>(output_h); ++y) {
    const size_t coefficient_base =
        y * static_cast<size_t>(vertical.kernel_size);
    const size_t source_start = static_cast<size_t>(vertical.starts[y]);
    const size_t count = static_cast<size_t>(vertical.counts[y]);
    for (size_t x = 0; x < target_width; ++x) {
      for (size_t channel = 0; channel < 3; ++channel) {
        int64_t accumulator = int64_t{1} << (kPillowPrecisionBits - 1);
        for (size_t tap = 0; tap < count; ++tap) {
          const size_t source =
              (((source_start + tap) * target_width + x) * 3) + channel;
          accumulator +=
              static_cast<int64_t>(vertical_input[source]) *
              vertical.coefficients[coefficient_base + tap];
        }
        output[(y * target_width + x) * 3 + channel] =
            PillowClip(accumulator);
      }
    }
  }
  return output;
}

int64_t PythonRound(double value) {
  if (!std::isfinite(value)) {
    ThrowOverflow("DeepSeek-V4 image rounding overflow");
  }
  const double lower = std::floor(value);
  if (lower < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      lower >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
    ThrowOverflow("DeepSeek-V4 image rounding overflow");
  }
  int64_t result = static_cast<int64_t>(lower);
  const double fraction = value - lower;
  if (fraction > 0.5 ||
      (fraction == 0.5 && result % 2 != 0)) {
    result = CheckedAdd(result, 1, "DeepSeek-V4 image rounding overflow");
  }
  return result;
}

std::vector<uint8_t> PadRgb(std::span<const uint8_t> rgb, int64_t input_h,
                            int64_t input_w, int64_t target_h,
                            int64_t target_w) {
  int64_t resized_h = target_h;
  int64_t resized_w = target_w;
  const double input_ratio = static_cast<double>(input_w) / input_h;
  const double target_ratio = static_cast<double>(target_w) / target_h;
  if (input_ratio != target_ratio) {
    if (input_ratio > target_ratio) {
      const int64_t candidate =
          PythonRound(static_cast<double>(input_h) / input_w * target_w);
      if (candidate != target_h) resized_h = candidate;
    } else {
      const int64_t candidate =
          PythonRound(static_cast<double>(input_w) / input_h * target_h);
      if (candidate != target_w) resized_w = candidate;
    }
  }
  std::vector<uint8_t> resized =
      ResizeRgb(rgb, input_h, input_w, resized_h, resized_w);
  if (resized_h == target_h && resized_w == target_w) return resized;

  std::vector<uint8_t> output(
      CheckedSize(CheckedPixelBytes(
                      target_h, target_w,
                      "DeepSeek-V4 padded image byte size overflow"),
                  "DeepSeek-V4 padded image byte size overflow"),
      127);
  const int64_t offset_x =
      resized_w == target_w ? 0 : PythonRound((target_w - resized_w) * 0.5);
  const int64_t offset_y =
      resized_h == target_h ? 0 : PythonRound((target_h - resized_h) * 0.5);
  const size_t resized_row_bytes = CheckedSize(
      CheckedMul(resized_w, 3, "DeepSeek-V4 padded image row overflow"),
      "DeepSeek-V4 padded image row overflow");
  const size_t target_row_bytes = CheckedSize(
      CheckedMul(target_w, 3, "DeepSeek-V4 padded image row overflow"),
      "DeepSeek-V4 padded image row overflow");
  const size_t target_x_bytes = CheckedSize(
      CheckedMul(offset_x, 3, "DeepSeek-V4 padded image offset overflow"),
      "DeepSeek-V4 padded image offset overflow");
  for (size_t y = 0; y < static_cast<size_t>(resized_h); ++y) {
    const size_t target_y = y + static_cast<size_t>(offset_y);
    std::copy_n(resized.data() + y * resized_row_bytes, resized_row_bytes,
                output.data() + target_y * target_row_bytes + target_x_bytes);
  }
  return output;
}

int64_t CheckedFloorToInt64(double value, const char* message) {
  const double floored = std::floor(value);
  if (!std::isfinite(floored) || floored < 0.0 ||
      floored >=
          static_cast<double>(std::numeric_limits<int64_t>::max())) {
    ThrowOverflow(message);
  }
  return static_cast<int64_t>(floored);
}

int64_t CheckedTruncateToInt64(double value, const char* message) {
  if (!std::isfinite(value) || value < 0.0 ||
      value >=
          static_cast<double>(std::numeric_limits<int64_t>::max())) {
    ThrowOverflow(message);
  }
  return static_cast<int64_t>(value);
}

struct ImageBlockShape {
  int64_t compress_pad = 0;
  int64_t rows = 0;
  int64_t row_length = 0;
  int64_t cells = 0;
  int64_t pad_last = 0;
  int64_t total = 0;
  int64_t image_count = 0;
};

ImageBlockShape GetImageBlockShape(int64_t n_llm_h, int64_t n_llm_w,
                                   int64_t start_position) {
  if (n_llm_h <= 0 || n_llm_w <= 0 || start_position < 0) {
    throw std::invalid_argument(
        "DeepSeek-V4 image block dimensions are invalid");
  }
  constexpr const char* kError = "DeepSeek-V4 image block size overflow";
  ImageBlockShape shape;
  shape.compress_pad =
      kCompressPadTo - 1 - start_position % kCompressPadTo;
  shape.rows = CheckedAdd(n_llm_h, n_llm_h % 2, kError);
  shape.row_length = CheckedAdd(n_llm_w, 1, kError);
  shape.cells = CheckedMul(shape.rows, shape.row_length, kError);
  shape.pad_last =
      (CheckedMul(shape.rows / 2, shape.row_length, kError) % 2) * 2;
  shape.total = CheckedAdd(
      CheckedAdd(CheckedAdd(shape.compress_pad, 1, kError), shape.cells,
                 kError),
      CheckedAdd(shape.pad_last, 1, kError), kError);
  shape.image_count = CheckedMul(n_llm_h, n_llm_w, kError);
  return shape;
}

}  // namespace

Json ParseDeepSeekV4TaggedText(const std::string& text) {
  Json blocks = Json::array();
  size_t cursor = 0;
  bool matched = false;
  while (true) {
    const size_t start = text.find("<image>", cursor);
    const size_t stray_end = text.find("</image>", cursor);
    if (start == std::string::npos) {
      if (stray_end != std::string::npos) {
        throw std::invalid_argument("Malformed <image>path</image> tag");
      }
      break;
    }
    if (stray_end != std::string::npos && stray_end < start) {
      throw std::invalid_argument("Malformed <image>path</image> tag");
    }
    const size_t path_start = start + 7;
    const size_t end = text.find("</image>", path_start);
    if (end == std::string::npos ||
        text.find("<image>", path_start) < end) {
      throw std::invalid_argument("Malformed <image>path</image> tag");
    }
    if (start > cursor) {
      blocks.push_back(
          Json{{"type", "text"}, {"text", text.substr(cursor, start - cursor)}});
    }
    const std::string path = text.substr(path_start, end - path_start);
    if (path.empty()) {
      throw std::invalid_argument("Image path must not be empty");
    }
    blocks.push_back(Json{{"type", "image_url"},
                          {"image_url", Json{{"url", path}}}});
    matched = true;
    cursor = end + 8;
  }
  if (!matched) {
    if (text.find("<image>") != std::string::npos ||
        text.find("</image>") != std::string::npos) {
      throw std::invalid_argument("Malformed <image>path</image> tag");
    }
    return Json(text);
  }
  if (cursor < text.size()) {
    blocks.push_back(Json{{"type", "text"}, {"text", text.substr(cursor)}});
  }
  return blocks;
}

DeepSeekV4EncodedPrompt EncodeDeepSeekV4Messages(
    const Json& messages, const std::string& thinking_mode, const Json& context,
    bool drop_thinking, bool add_default_bos_token,
    const std::string& reasoning_effort) {
  auto processed_context = ProcessImageMessages(context);
  auto processed_messages = ProcessImageMessages(messages);
  DeepSeekV4EncodedPrompt result;
  result.prompt = EncodeMessagesText(
      std::move(processed_messages.first), thinking_mode,
      std::move(processed_context.first), drop_thinking, add_default_bos_token,
      reasoning_effort);
  result.images = std::move(processed_messages.second);
  return result;
}

DeepSeekV4GridTokens GridTokens(int64_t height, int64_t width,
                                int64_t patch_size,
                                int64_t downsample_ratio) {
  ValidateGeometry(height, width, patch_size, downsample_ratio);
  constexpr const char* kError = "DeepSeek-V4 grid token count overflow";
  const int64_t n_vit_h = height / patch_size;
  const int64_t n_vit_w = width / patch_size;
  const int64_t n_llm_h =
      CheckedCeilDiv(n_vit_h, downsample_ratio, kError);
  const int64_t n_llm_w =
      CheckedCeilDiv(n_vit_w, downsample_ratio, kError);
  const int64_t row_length = CheckedAdd(n_llm_w, 1, kError);
  int64_t tokens =
      CheckedAdd(CheckedMul(n_llm_h, row_length, kError), 2, kError);
  if (n_llm_h % 2 == 1) tokens = CheckedAdd(tokens, row_length, kError);
  const int64_t row_pairs =
      CheckedAdd(n_llm_h, 1, kError) / 2;
  const int64_t pad_last =
      (CheckedMul(row_pairs, row_length, kError) % 2) * 2;
  tokens = CheckedAdd(tokens, pad_last, kError);
  return {n_llm_h, n_llm_w, tokens};
}

DeepSeekV4Resize SolveResizeRatio(int64_t height, int64_t width,
                                  int64_t patch_size,
                                  int64_t downsample_ratio,
                                  int64_t max_image_tokens) {
  ValidateGeometry(height, width, patch_size, downsample_ratio);
  if (max_image_tokens <= 2) {
    throw std::invalid_argument("DeepSeek-V4 image token budget must exceed 2");
  }
  constexpr const char* kError = "DeepSeek-V4 resize geometry overflow";
  const int64_t available_tokens = max_image_tokens - 2;
  const double ratio = static_cast<double>(height) / width;
  const double max_w_float =
      std::sqrt(static_cast<double>(available_tokens) / ratio + 0.25) - 0.5;
  const double max_h_float = max_w_float * ratio;
  int64_t best_width;
  int64_t best_height;
  if (max_w_float < 1.0) {
    const int64_t max_w = 1;
    int64_t max_h = available_tokens / (max_w + 1);
    if (max_h % 2 == 1) --max_h;
    best_width = CheckedMul(
        CheckedMul(max_w, patch_size, kError), downsample_ratio, kError);
    best_height = CheckedMul(
        CheckedMul(max_h, patch_size, kError), downsample_ratio, kError);
  } else if (max_h_float < 2.0) {
    const int64_t max_h = 2;
    const int64_t max_w = available_tokens / max_h - 1;
    if (max_w <= 1) {
      throw std::invalid_argument("DeepSeek-V4 wide image budget is too small");
    }
    best_width = CheckedMul(
        CheckedMul(max_w, patch_size, kError), downsample_ratio, kError);
    best_height = CheckedMul(
        CheckedMul(max_h, patch_size, kError), downsample_ratio, kError);
  } else {
    const int64_t max_w =
        CheckedFloorToInt64(max_w_float, kError);
    int64_t max_h = CheckedFloorToInt64(max_h_float, kError);
    if (max_h % 2 == 1) --max_h;
    const int64_t width_limit = CheckedMul(
        CheckedMul(max_w, patch_size, kError), downsample_ratio, kError);
    const int64_t height_limit = CheckedMul(
        CheckedMul(max_h, patch_size, kError), downsample_ratio, kError);
    const double beta =
        std::min(static_cast<double>(width_limit) / width,
                 static_cast<double>(height_limit) / height);
    best_width = CheckedMul(
        CheckedFloorToInt64(
            static_cast<double>(width) * beta / patch_size, kError),
        patch_size, kError);
    best_height = CheckedMul(
        CheckedFloorToInt64(
            static_cast<double>(height) * beta / patch_size, kError),
        patch_size, kError);
  }
  const auto grid = GridTokens(best_height, best_width, patch_size,
                               downsample_ratio);
  return {grid.n_llm_h, grid.n_llm_w, best_height, best_width,
          grid.num_tokens};
}

DeepSeekV4Resize SafeResize(int64_t height, int64_t width,
                            int64_t best_height, int64_t best_width,
                            int64_t patch_size, int64_t downsample_ratio,
                            int64_t max_image_tokens) {
  ValidateGeometry(height, width, patch_size, downsample_ratio);
  ValidateGeometry(best_height, best_width, patch_size, downsample_ratio);
  if (max_image_tokens <= kCompressPadTo + 1) {
    throw std::invalid_argument("DeepSeek-V4 image token budget is too small");
  }
  const int64_t maximum = max_image_tokens - (kCompressPadTo - 1);
  auto grid = GridTokens(best_height, best_width, patch_size, downsample_ratio);
  DeepSeekV4Resize result{grid.n_llm_h, grid.n_llm_w, best_height, best_width,
                          grid.num_tokens};
  int64_t budget = maximum;
  while (result.num_tokens > maximum) {
    result = SolveResizeRatio(height, width, patch_size, downsample_ratio,
                              budget);
    --budget;
    if (budget <= 2 && result.num_tokens > maximum) {
      throw std::invalid_argument(
          "DeepSeek-V4 image token budget cannot fit one image block");
    }
  }
  return result;
}

DeepSeekV4ImageProcessor::DeepSeekV4ImageProcessor(
    DeepSeekV4ProcessorConfig config)
    : config_(std::move(config)) {
  ValidateProcessorConfig(config_);
}

ImageKwargs DeepSeekV4ImageProcessor::ProcessImage(
    std::span<const uint8_t> rgb, int64_t height, int64_t width) const {
  ValidateGeometry(height, width, config_.patch_size,
                   config_.downsample_ratio);
  const size_t required_bytes = CheckedSize(
      CheckedPixelBytes(height, width,
                        "DeepSeek-V4 RGB byte extent overflow"),
      "DeepSeek-V4 RGB byte extent overflow");
  if (rgb.size() != required_bytes) {
    throw std::invalid_argument(
        "DeepSeek-V4 RGB byte extent does not equal height*width*3");
  }

  int64_t sizing_width = width;
  int64_t sizing_height = height;
  if (config_.max_width_height_ratio > 0) {
    const int64_t maximum_width = CheckedMul(
        sizing_height, config_.max_width_height_ratio,
        "DeepSeek-V4 width-height limit overflow");
    if (sizing_width > maximum_width) sizing_width = maximum_width;
  }
  const int64_t sizing_area = CheckedMul(
      sizing_width, sizing_height, "DeepSeek-V4 image area overflow");
  if (config_.min_pixels > 0 && sizing_area < config_.min_pixels) {
    const double ratio = std::sqrt(
        static_cast<double>(config_.min_pixels) /
        static_cast<double>(sizing_area));
    sizing_width = CheckedTruncateToInt64(
        static_cast<double>(sizing_width) * ratio,
        "DeepSeek-V4 minimum-pixel resize overflow");
    sizing_height = CheckedTruncateToInt64(
        static_cast<double>(sizing_height) * ratio,
        "DeepSeek-V4 minimum-pixel resize overflow");
  }
  const int64_t patch = config_.patch_size;
  int64_t best_width = CheckedMul(
      CheckedCeilDiv(sizing_width, patch,
                     "DeepSeek-V4 padded width overflow"),
      patch, "DeepSeek-V4 padded width overflow");
  int64_t best_height = CheckedMul(
      CheckedCeilDiv(sizing_height, patch,
                     "DeepSeek-V4 padded height overflow"),
      patch, "DeepSeek-V4 padded height overflow");
  const auto resized = SafeResize(
      sizing_height, sizing_width, best_height, best_width, patch,
      config_.downsample_ratio, config_.max_image_tokens);
  best_height = resized.height;
  best_width = resized.width;
  const int64_t n_vit_h = best_height / patch;
  const int64_t n_vit_w = best_width / patch;

  std::vector<uint8_t> transformed_pixels;
  std::span<const uint8_t> pixels = rgb;
  if (height != best_height || width != best_width) {
    if (config_.max_width_height_ratio > 0 &&
        width >= CheckedMul(config_.max_width_height_ratio, height,
                            "DeepSeek-V4 width-height limit overflow")) {
      transformed_pixels =
          ResizeRgb(rgb, height, width, best_height, best_width);
    } else {
      transformed_pixels =
          PadRgb(rgb, height, width, best_height, best_width);
    }
    pixels = transformed_pixels;
  }

  ImageKwargs output;
  output.num_patches = CheckedMul(
      n_vit_h, n_vit_w, "DeepSeek-V4 patch count overflow");
  output.patch_feature_dim = CheckedMul(
      CheckedMul(3, patch, "DeepSeek-V4 patch feature size overflow"), patch,
      "DeepSeek-V4 patch feature size overflow");
  output.image_grid_thw = {1, n_vit_h, n_vit_w};
  const size_t values = CheckedSize(
      CheckedMul(output.num_patches, output.patch_feature_dim,
                 "DeepSeek-V4 patch value count overflow"),
      "DeepSeek-V4 patch value count overflow");
  output.pixel_values_bf16.resize(values);

  const size_t best_width_size = static_cast<size_t>(best_width);
  const size_t n_vit_w_size = static_cast<size_t>(n_vit_w);
  const size_t patch_size = static_cast<size_t>(patch);
  const size_t feature_dim =
      static_cast<size_t>(output.patch_feature_dim);
  for (size_t vit_h = 0; vit_h < static_cast<size_t>(n_vit_h); ++vit_h) {
    for (size_t vit_w = 0; vit_w < n_vit_w_size; ++vit_w) {
      const size_t row = vit_h * n_vit_w_size + vit_w;
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t patch_h = 0; patch_h < patch_size; ++patch_h) {
          const size_t source_h = vit_h * patch_size + patch_h;
          for (size_t patch_w = 0; patch_w < patch_size; ++patch_w) {
            const size_t source_w = vit_w * patch_size + patch_w;
            const uint8_t raw =
                pixels[(source_h * best_width_size + source_w) * 3 + channel];
            const float value =
                ((static_cast<float>(raw) / 255.0f) - 0.5f) / 0.5f;
            const size_t feature =
                (channel * patch_size + patch_h) * patch_size + patch_w;
            output.pixel_values_bf16[row * feature_dim + feature] =
                vt::F32ToBF16(value);
          }
        }
      }
    }
  }
  return output;
}

DeepSeekV4ImageBlock BuildDeepSeekV4ImageBlock(int64_t n_llm_h,
                                               int64_t n_llm_w,
                                               int64_t start_position) {
  const ImageBlockShape shape =
      GetImageBlockShape(n_llm_h, n_llm_w, start_position);
  const size_t cells =
      CheckedSize(shape.cells, "DeepSeek-V4 image block size overflow");
  std::vector<int64_t> natural_types;
  natural_types.reserve(cells);
  std::vector<int64_t> image_index(cells, -1);
  int64_t image = 0;
  for (int64_t row = 0; row < shape.rows; ++row) {
    for (int64_t column = 0; column < shape.row_length; ++column) {
      const size_t index =
          static_cast<size_t>(row * shape.row_length + column);
      if (row < n_llm_h && column < n_llm_w) {
        natural_types.push_back(kImage);
        image_index[index] = image++;
      } else if (row < n_llm_h) {
        natural_types.push_back(kImageNewLine);
      } else {
        natural_types.push_back(kImagePad);
      }
    }
  }

  DeepSeekV4ImageBlock block;
  block.types.reserve(
      CheckedSize(shape.total, "DeepSeek-V4 image block size overflow"));
  block.permutation.reserve(
      CheckedSize(shape.image_count,
                  "DeepSeek-V4 image block size overflow"));
  block.types.insert(
      block.types.end(),
      CheckedSize(shape.compress_pad,
                  "DeepSeek-V4 image block size overflow"),
      kImagePad);
  block.types.push_back(kImageStart);
  for (int64_t pair = 0; pair < shape.rows / 2; ++pair) {
    for (int64_t column = 0; column < shape.row_length; ++column) {
      for (int64_t row_in_pair = 0; row_in_pair < 2; ++row_in_pair) {
        const int64_t row = pair * 2 + row_in_pair;
        const size_t index =
            static_cast<size_t>(row * shape.row_length + column);
        block.types.push_back(natural_types[index]);
        const int64_t source = image_index[index];
        if (source >= 0) block.permutation.push_back(source);
      }
    }
  }
  block.types.insert(
      block.types.end(),
      CheckedSize(shape.pad_last,
                  "DeepSeek-V4 image block size overflow"),
      kImagePad);
  block.types.push_back(kImageEnd);
  return block;
}

MultiModalInputs PrepareDeepSeekV4Inputs(
    const std::vector<int32_t>& prompt_token_ids, int32_t image_token_id,
    const std::vector<std::shared_ptr<ImageKwargs>>& images,
    const DeepSeekV4ProcessorConfig& config) {
  ValidateProcessorConfig(config);
  const int64_t expected_feature_dim = CheckedMul(
      CheckedMul(3, config.patch_size,
                 "DeepSeek-V4 patch feature size overflow"),
      config.patch_size, "DeepSeek-V4 patch feature size overflow");
  const size_t placeholders = static_cast<size_t>(std::count(
      prompt_token_ids.begin(), prompt_token_ids.end(), image_token_id));
  if (placeholders != images.size()) {
    throw std::invalid_argument(
        "Found " + std::to_string(placeholders) + " image tokens but got " +
        std::to_string(images.size()) + " images");
  }

  constexpr const char* kPromptError =
      "DeepSeek-V4 expanded prompt size overflow";
  int64_t expanded_size = 0;
  size_t image_index = 0;
  for (const int32_t token : prompt_token_ids) {
    if (token != image_token_id) {
      expanded_size = CheckedAdd(expanded_size, 1, kPromptError);
      continue;
    }
    const auto& image = images[image_index++];
    if (!image || image->empty()) {
      throw std::invalid_argument("DeepSeek-V4 image input must not be empty");
    }
    const int64_t n_vit_h = image->image_grid_thw[1];
    const int64_t n_vit_w = image->image_grid_thw[2];
    if (image->image_grid_thw[0] != 1 || n_vit_h <= 0 || n_vit_w <= 0 ||
        image->num_patches !=
            CheckedMul(n_vit_h, n_vit_w,
                       "DeepSeek-V4 image input shape overflow")) {
      throw std::invalid_argument("DeepSeek-V4 image input shape is invalid");
    }
    if (image->patch_feature_dim != expected_feature_dim) {
      throw std::invalid_argument(
          "DeepSeek-V4 image feature width does not match patch size");
    }
    const int64_t n_llm_h = CheckedCeilDiv(
        n_vit_h, config.downsample_ratio,
        "DeepSeek-V4 image block dimensions overflow");
    const int64_t n_llm_w = CheckedCeilDiv(
        n_vit_w, config.downsample_ratio,
        "DeepSeek-V4 image block dimensions overflow");
    CheckedInt(expanded_size, "DeepSeek-V4 image offset exceeds int range");
    const ImageBlockShape shape =
        GetImageBlockShape(n_llm_h, n_llm_w, expanded_size);
    CheckedInt(shape.total,
               "DeepSeek-V4 image feature length exceeds int range");
    const int64_t image_values = CheckedMul(
        image->num_patches, image->patch_feature_dim,
        "DeepSeek-V4 image input shape overflow");
    if (image->pixel_values_bf16.size() !=
        CheckedSize(image_values,
                    "DeepSeek-V4 image input shape overflow")) {
      throw std::invalid_argument(
          "DeepSeek-V4 image BF16 extent does not match shape");
    }
    expanded_size = CheckedAdd(expanded_size, shape.total, kPromptError);
  }
  CheckedInt(expanded_size,
             "DeepSeek-V4 expanded prompt size exceeds int range");

  MultiModalInputs result;
  result.prompt_token_ids.reserve(CheckedSize(expanded_size, kPromptError));
  result.mm_features.reserve(images.size());
  image_index = 0;
  for (const int32_t token : prompt_token_ids) {
    if (token != image_token_id) {
      result.prompt_token_ids.push_back(token);
      continue;
    }
    const auto& image = images[image_index++];
    const int64_t n_llm_h = CheckedCeilDiv(
        image->image_grid_thw[1], config.downsample_ratio,
        "DeepSeek-V4 image block dimensions overflow");
    const int64_t n_llm_w = CheckedCeilDiv(
        image->image_grid_thw[2], config.downsample_ratio,
        "DeepSeek-V4 image block dimensions overflow");
    const int offset = CheckedInt(
        static_cast<int64_t>(result.prompt_token_ids.size()),
        "DeepSeek-V4 image offset exceeds int range");
    const auto block =
        BuildDeepSeekV4ImageBlock(n_llm_h, n_llm_w, offset);
    for (const int64_t type : block.types) {
      result.prompt_token_ids.push_back(CheckedInt32(
          CheckedAdd(config.vocab_size, type,
                     "DeepSeek-V4 image sentinel token overflow"),
          "DeepSeek-V4 image sentinel token exceeds int32 range"));
    }
    MultiModalFeatureSpec feature;
    feature.modality = "image";
    feature.offset = offset;
    feature.length = CheckedInt(
        static_cast<int64_t>(block.types.size()),
        "DeepSeek-V4 image feature length exceeds int range");
    feature.data = image;
    result.mm_features.push_back(std::move(feature));
  }
  return result;
}

}  // namespace vllm::multimodal
