// Tests for the OutputProcessor ported from
// vllm/v1/engine/output_processor.py @ e24d1b24 (M1.8 Task 5). Exercises the
// synchronous text path against the REAL IncrementalDetokenizer + a tiny
// oracle-verified BPE fixture (shared with tests/vllm/test_detokenizer.cpp):
//   (a) streaming DELTA deltas concatenate to the cumulative full text;
//   (b) a stop STRING terminates -> finish_reason=STOP, stop_reason=string,
//       and reqs_to_abort carries the req (EngineCore did not signal STOP);
//   (c) finish-reason mapping (LENGTH/STOP) -> RequestOutput strings;
//   (d) a normal EngineCore-signaled EOS finish -> finished, no reqs_to_abort.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/request.h"

using nlohmann::json;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCoreOutput;
using vllm::v1::EngineCoreOutputs;
using vllm::v1::EngineCoreRequest;
using vllm::v1::FinishReason;
using vllm::v1::OutputProcessor;

namespace {

using Ids = std::vector<int32_t>;

// Writes `body` to a unique temp file; removed in the destructor.
class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_outproc_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// The oracle-verified tiny fixture from test_detokenizer.cpp: ids 0..18 spell
// out "hello world12" pieces (hello=13, Ġworld=17, "1"=8, "2"=9), plus the
// byte-fragment tokens (harmless here). Enough to Encode a prompt and drive
// generation.
Tokenizer BuildFixture() {
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},    {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},    {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11},  {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15},  {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  vocab[MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 23;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  TempJson f(doc.dump());
  return Tokenizer::FromHfJson(f.path());
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

EngineCoreOutputs MakeStep(const std::string& rid, Ids toks,
                           std::optional<FinishReason> fr = std::nullopt,
                           std::optional<std::string> sr = std::nullopt) {
  EngineCoreOutput o;
  o.request_id = rid;
  o.new_token_ids = std::move(toks);
  o.finish_reason = fr;
  o.stop_reason = std::move(sr);
  EngineCoreOutputs outs;
  outs.outputs.push_back(std::move(o));
  return outs;
}

EngineCoreRequest MakeRequest(const std::string& rid, const Ids& prompt_ids,
                              RequestOutputKind kind,
                              std::vector<std::string> stop = {}) {
  EngineCoreRequest req;
  req.request_id = rid;
  req.prompt_token_ids = prompt_ids;
  SamplingParams sp;
  sp.output_kind = kind;
  sp.max_tokens = 100;
  sp.stop = std::move(stop);
  req.sampling_params = sp;
  return req;
}

}  // namespace

TEST_CASE("streaming DELTA deltas concatenate to the cumulative full text") {
  const Tokenizer& tok = Fixture();
  const Ids prompt = tok.Encode("hello");  // seeds detok prefix, not output
  // Generated tokens: " world" (17), "1" (8), "2" (9) -> output " world12".
  const std::vector<Ids> gen = {{17}, {8}, {9}};

  // DELTA: each step returns only the new text.
  std::string streamed;
  {
    OutputProcessor op(&tok);
    op.add_request(MakeRequest("r0", prompt, RequestOutputKind::kDelta), "hello");
    for (size_t i = 0; i < gen.size(); ++i) {
      const bool last = i + 1 == gen.size();
      auto out = op.process_outputs(MakeStep(
          "r0", gen[i], last ? std::optional<FinishReason>(FinishReason::kLength)
                             : std::nullopt));
      REQUIRE(out.request_outputs.size() == 1);
      REQUIRE(out.request_outputs[0].outputs.size() == 1);
      streamed += out.request_outputs[0].outputs[0].text;
    }
    CHECK(op.has_unfinished_requests() == false);
  }
  CHECK(streamed == " world12");

  // CUMULATIVE: the finished step carries the full text + all generated ids.
  {
    OutputProcessor op(&tok);
    op.add_request(MakeRequest("r0", prompt, RequestOutputKind::kCumulative),
                   "hello");
    std::string last_text;
    Ids last_ids;
    for (size_t i = 0; i < gen.size(); ++i) {
      const bool last = i + 1 == gen.size();
      auto out = op.process_outputs(MakeStep(
          "r0", gen[i], last ? std::optional<FinishReason>(FinishReason::kLength)
                             : std::nullopt));
      REQUIRE(out.request_outputs.size() == 1);
      const auto& co = out.request_outputs[0].outputs[0];
      last_text = co.text;
      last_ids = co.token_ids;
    }
    CHECK(last_text == " world12");           // full, not a delta
    CHECK(last_ids == Ids({17, 8, 9}));        // all generated ids
  }
}

TEST_CASE("stop STRING terminates the request and feeds reqs_to_abort") {
  const Tokenizer& tok = Fixture();
  const Ids prompt = tok.Encode("hello");
  OutputProcessor op(&tok);
  // stop "wo": appears inside " world"; EngineCore does NOT signal finish.
  op.add_request(
      MakeRequest("r0", prompt, RequestOutputKind::kCumulative, {"wo"}), "hello");

  auto out = op.process_outputs(MakeStep("r0", {17}));  // " world"
  REQUIRE(out.request_outputs.size() == 1);
  const auto& ro = out.request_outputs[0];
  CHECK(ro.finished == true);
  REQUIRE(ro.outputs.size() == 1);
  const auto& co = ro.outputs[0];
  REQUIRE(co.finish_reason.has_value());
  CHECK(*co.finish_reason == "stop");
  REQUIRE(co.stop_reason.has_value());
  CHECK(*co.stop_reason == "wo");
  // include_stop_str_in_output defaults false -> text truncated before "wo".
  CHECK(co.text == " ");
  // EngineCore did not signal STOP -> abort feedback.
  REQUIRE(out.reqs_to_abort.size() == 1);
  CHECK(out.reqs_to_abort[0] == "r0");
  CHECK(op.has_unfinished_requests() == false);
}

TEST_CASE("finish-reason mapping (LENGTH / STOP)") {
  const Tokenizer& tok = Fixture();
  const Ids prompt = tok.Encode("hello");

  {  // LENGTH
    OutputProcessor op(&tok);
    op.add_request(MakeRequest("a", prompt, RequestOutputKind::kCumulative),
                   "hello");
    auto out = op.process_outputs(MakeStep("a", {17}, FinishReason::kLength));
    REQUIRE(out.request_outputs.size() == 1);
    CHECK(*out.request_outputs[0].outputs[0].finish_reason == "length");
  }
  {  // STOP (EngineCore-signaled EOS maps to "stop")
    OutputProcessor op(&tok);
    op.add_request(MakeRequest("b", prompt, RequestOutputKind::kCumulative),
                   "hello");
    auto out = op.process_outputs(MakeStep("b", {17}, FinishReason::kStop));
    REQUIRE(out.request_outputs.size() == 1);
    CHECK(*out.request_outputs[0].outputs[0].finish_reason == "stop");
  }
}

TEST_CASE("EngineCore-signaled EOS finish: finished, no reqs_to_abort") {
  const Tokenizer& tok = Fixture();
  const Ids prompt = tok.Encode("hello");
  OutputProcessor op(&tok);
  op.add_request(MakeRequest("r0", prompt, RequestOutputKind::kCumulative),
                 "hello");

  // A couple of unfinished steps, then EngineCore signals STOP itself.
  auto s0 = op.process_outputs(MakeStep("r0", {17}));
  CHECK(s0.request_outputs[0].finished == false);
  CHECK(s0.reqs_to_abort.empty());

  auto s1 = op.process_outputs(MakeStep("r0", {8}, FinishReason::kStop));
  REQUIRE(s1.request_outputs.size() == 1);
  CHECK(s1.request_outputs[0].finished == true);
  CHECK(*s1.request_outputs[0].outputs[0].finish_reason == "stop");
  // EngineCore signaled finish -> NO abort feedback.
  CHECK(s1.reqs_to_abort.empty());
  CHECK(op.has_unfinished_requests() == false);
}

TEST_CASE("output for an unknown / already-finished request is ignored") {
  const Tokenizer& tok = Fixture();
  OutputProcessor op(&tok);
  auto out = op.process_outputs(MakeStep("ghost", {17}, FinishReason::kStop));
  CHECK(out.request_outputs.empty());
  CHECK(out.reqs_to_abort.empty());
}
