// Ported from: vllm/tool_parsers/abstract_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>

#include "vllm/entrypoints/openai/tool_parsers/hermes.h"
#include "vllm/entrypoints/openai/tool_parsers/qwen3.h"

namespace vllm::entrypoints::openai {

std::string make_tool_call_id() {
  // Upstream: f"chatcmpl-tool-{random_uuid()}" with random_uuid() == the hex of
  // a uuid4 (32 hex chars). We emit 32 random hex chars (uniqueness only).
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist;
  const uint64_t hi = dist(rng);
  const uint64_t lo = dist(rng);
  std::ostringstream oss;
  oss << "chatcmpl-tool-" << std::hex << std::setfill('0') << std::setw(16)
      << hi << std::setw(16) << lo;
  return oss.str();
}

std::unique_ptr<ToolParser> get_tool_parser(const std::string& name) {
  // Ported from ToolParserManager.get_tool_parser (abstract_tool_parser.py:235).
  // T0 registry: "hermes" -> Hermes2ProToolParser; "qwen3" -> the gate model's
  // Hermes-format parser. Unknown name -> nullptr (upstream raises KeyError).
  if (name == "hermes") {
    return std::make_unique<HermesToolParser>();
  }
  if (name == "qwen3") {
    return std::make_unique<Qwen3ToolParser>();
  }
  return nullptr;
}

}  // namespace vllm::entrypoints::openai
