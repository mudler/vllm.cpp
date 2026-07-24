// Ported from: vllm/reasoning/olmo3_reasoning_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/reasoning_parsers/olmo3.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace vllm::entrypoints::openai {

namespace {

struct Indices {
  std::size_t start = 0;
  std::size_t end = 0;
  std::size_t len() const { return end - start; }
};

// olmo3:34 (string_overlap). Longest overlap where the end of `A` matches the
// start of `B`, or `A` fully contained in `B` (either direction). Returns the
// overlapping index spans in (A, B). We only consume the B-side span downstream.
std::pair<std::optional<Indices>, std::optional<Indices>> StringOverlap(
    const std::string& A, const std::string& B) {
  // Swap so `a` is the shorter string (olmo3:49). swap==true means A was longer.
  const bool swap = !(A.size() < B.size());
  const std::string& a = swap ? B : A;
  const std::string& b = swap ? A : B;

  auto ret = [&](Indices ind_a, Indices ind_b)
      -> std::pair<std::optional<Indices>, std::optional<Indices>> {
    if (swap) return {ind_b, ind_a};
    return {ind_a, ind_b};
  };

  // check 1: a fully contained in b (olmo3:52-55).
  const std::size_t pos = b.find(a);
  if (pos != std::string::npos) {
    return ret(Indices{0, a.size()}, Indices{pos, pos + a.size()});
  }
  // check 2: end of a overlaps start of b (olmo3:59-63).
  for (std::size_t i = a.size() - 1; i >= 1; --i) {
    if (a.compare(a.size() - i, i, b, 0, i) == 0) {
      return ret(Indices{a.size() - i, a.size()}, Indices{0, i});
    }
  }
  // check 3: start of a overlaps end of b (olmo3:66-71).
  for (std::size_t i = a.size() - 1; i >= 1; --i) {
    if (b.compare(b.size() - i, i, a, 0, i) == 0) {
      return ret(Indices{0, i}, Indices{b.size() - i, b.size()});
    }
  }
  return {std::nullopt, std::nullopt};
}

DeltaMessage ReasoningDelta(std::string text) {
  DeltaMessage msg;
  msg.reasoning = std::move(text);
  return msg;
}
DeltaMessage ContentDelta(std::string text) {
  DeltaMessage msg;
  msg.content = std::move(text);
  return msg;
}

std::optional<std::string> OrNullopt(std::string s) {
  if (s.empty()) return std::nullopt;
  return s;
}

}  // namespace

std::optional<DeltaMessage> Olmo3ReasoningBuffer::process_buffer() {
  const std::size_t start_idx = buffer_.find(think_start);
  if (start_idx != std::string::npos) {
    state_ = State::kReasoning;
    const std::string pretext = buffer_.substr(0, start_idx);
    buffer_ = buffer_.substr(start_idx + think_start.size());
    if (start_idx > 0) {
      return ContentDelta(pretext);
    }
  }

  const std::size_t end_idx = buffer_.rfind(think_end);
  if (end_idx != std::string::npos) {
    state_ = State::kContent;
    const std::string pretext = buffer_.substr(0, end_idx);
    buffer_ = buffer_.substr(end_idx + think_end.size());
    if (end_idx > 0) {
      return ReasoningDelta(pretext);
    }
  }

  if (state_ == State::kReasoning) {
    std::string text = buffer_;
    buffer_.clear();
    return ReasoningDelta(std::move(text));
  }
  // state_ == kContent
  std::string text = buffer_;
  buffer_.clear();
  return ContentDelta(std::move(text));
}

std::optional<DeltaMessage> Olmo3ReasoningBuffer::add_text(
    const std::string& delta_text) {
  buffer_ += delta_text;

  const auto ov_start = StringOverlap(delta_text, think_start).second;
  const auto ov_end = StringOverlap(delta_text, think_end).second;

  const bool partial_overlap_start =
      ov_start.has_value() && ov_start->len() < think_start.size();
  const bool partial_overlap_end =
      ov_end.has_value() && ov_end->len() < think_end.size();

  const bool has_start = buffer_.find(think_start) != std::string::npos;
  const bool has_end = buffer_.find(think_end) != std::string::npos;

  if (partial_overlap_start && has_start && !partial_overlap_end) {
    return process_buffer();
  }
  if (partial_overlap_end && has_end) {
    return process_buffer();
  }
  if (partial_overlap_start || partial_overlap_end) {
    // Wait for the (split) marker to complete (olmo3:178-182).
    return std::nullopt;
  }
  return process_buffer();
}

ExtractedReasoning Olmo3ReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // olmo3:227 regex: ^(?:<think>)?(reasoning.*?)</think>(content.*)$ DOTALL.
  ExtractedReasoning out;
  std::size_t body_start = 0;
  if (model_output.compare(0, think_start_.size(), think_start_) == 0) {
    body_start = think_start_.size();
  }
  const std::size_t ep = model_output.find(think_end_, body_start);
  if (ep == std::string::npos) {
    // No </think>: the regex fails to match, all output is content (olmo3:296-297).
    out.reasoning = std::nullopt;
    out.content = model_output;
    return out;
  }
  out.reasoning = OrNullopt(model_output.substr(body_start, ep - body_start));
  out.content = OrNullopt(model_output.substr(ep + think_end_.size()));
  return out;
}

std::optional<DeltaMessage> Olmo3ReasoningParser::extract_reasoning_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  std::optional<DeltaMessage> delta_message = buffer_.add_text(delta_text);
  if (!delta_message.has_value() &&
      buffer_.buffer().find(buffer_.think_end) != std::string::npos) {
    // Terminal delta carried the end marker but yielded nothing; drain the
    // buffer once more so the final content is not stranded (olmo3:311-318).
    delta_message = buffer_.process_buffer();
  }
  return delta_message;
}

bool Olmo3ReasoningParser::is_reasoning_end(const std::string& text) const {
  return text.find(think_end_) != std::string::npos;
}

}  // namespace vllm::entrypoints::openai
