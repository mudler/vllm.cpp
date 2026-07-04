// Tests for the M3.4 Task 1 structured-output type/contract skeleton
// (vllm/v1/structured_output/{backend_types,request}.py @ e24d1b24).
#include <doctest/doctest.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/request.h"

using vllm::SamplingParams;
using vllm::StructuredOutputsParams;
using vllm::v1::BitmaskWordsForVocab;
using vllm::v1::get_structured_output_key;
using vllm::v1::StructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputKey;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::StructuredOutputRequest;
using vllm::v1::TokenBitmask;

namespace {

// A trivial concrete grammar proving the ABC is usable as an interface: it
// accepts everything and fills every allowed bit.
class MockGrammar : public StructuredOutputGrammar {
 public:
  bool accept_tokens(const std::string&,
                     const std::vector<int32_t>&) override {
    return true;
  }
  std::vector<int32_t> validate_tokens(
      const std::vector<int32_t>& tokens) override {
    return tokens;
  }
  void rollback(int num_tokens) override { rolled_back_ += num_tokens; }
  void fill_bitmask(TokenBitmask& bitmask, int batch_index) override {
    const int base = batch_index * bitmask.num_words;
    for (int w = 0; w < bitmask.num_words; ++w) {
      bitmask.data[base + w] = ~0;  // all tokens allowed
    }
  }
  bool is_terminated() override { return terminated_; }
  void reset() override { terminated_ = false; }
  int rolled_back_ = 0;
  bool terminated_ = false;
};

// A trivial concrete backend proving the engine-level ABC is usable.
class MockBackend : public StructuredOutputBackend {
 public:
  explicit MockBackend(int vocab_size) : vocab_size_(vocab_size) {}
  std::unique_ptr<StructuredOutputGrammar> compile_grammar(
      StructuredOutputOptions, const std::string&) override {
    return std::make_unique<MockGrammar>();
  }
  TokenBitmask allocate_token_bitmask(int max_num_seqs) override {
    TokenBitmask m;
    m.num_seqs = max_num_seqs;
    m.num_words = BitmaskWordsForVocab(vocab_size_);
    m.data.assign(static_cast<std::size_t>(max_num_seqs) * m.num_words, 0);
    return m;
  }
  void destroy() override {}

 private:
  int vocab_size_;
};

}  // namespace

TEST_CASE("get_structured_output_key maps each option to its (type, spec) key") {
  SUBCASE("json") {
    StructuredOutputsParams p;
    p.json = R"({"type":"object"})";
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kJson);
    CHECK(key.second == R"({"type":"object"})");
  }
  SUBCASE("json_object") {
    StructuredOutputsParams p;
    p.json_object = true;
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kJsonObject);
    CHECK(key.second.empty());
  }
  SUBCASE("regex") {
    StructuredOutputsParams p;
    p.regex = "[0-9]+";
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kRegex);
    CHECK(key.second == "[0-9]+");
  }
  SUBCASE("choice serialized as a JSON array") {
    StructuredOutputsParams p;
    p.choice = std::vector<std::string>{"yes", "no"};
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kChoice);
    CHECK(key.second == R"(["yes","no"])");
  }
  SUBCASE("grammar") {
    StructuredOutputsParams p;
    p.grammar = R"(root ::= "a")";
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kGrammar);
    CHECK(key.second == R"(root ::= "a")");
  }
  SUBCASE("structural_tag (deferred backend, key still maps)") {
    StructuredOutputsParams p;
    p.structural_tag = "<tag/>";
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kStructuralTag);
    CHECK(key.second == "<tag/>");
  }
  SUBCASE("no constraint set throws") {
    StructuredOutputsParams p;
    CHECK_THROWS_AS(get_structured_output_key(p), std::runtime_error);
  }
}

TEST_CASE("json_object=false is not-set (upstream `if params.json_object`)") {
  // json_object is optional<bool>; only a truthy value selects JSON_OBJECT.
  StructuredOutputsParams p;
  p.json_object = false;
  // has_value() => counts for mutual exclusion, but the key falls through.
  CHECK_THROWS_AS(get_structured_output_key(p), std::runtime_error);
}

TEST_CASE("StructuredOutputsParams carries the spec through SamplingParams") {
  SamplingParams sp;
  StructuredOutputsParams so;
  so.grammar = R"(root ::= "hi")";
  sp.structured_outputs = so;
  sp.PostInit();  // must not throw with exactly one constraint set

  auto req = StructuredOutputRequest::from_sampling_params(&sp);
  REQUIRE(req.has_value());
  REQUIRE(req->params.grammar.has_value());
  CHECK(*req->params.grammar == R"(root ::= "hi")");

  const StructuredOutputKey key = req->structured_output_key();
  CHECK(key.first == StructuredOutputOptions::kGrammar);
  CHECK(key.second == R"(root ::= "hi")");
}

TEST_CASE("from_sampling_params returns nullopt without a constraint") {
  CHECK_FALSE(StructuredOutputRequest::from_sampling_params(nullptr).has_value());

  SamplingParams sp;  // no structured_outputs
  CHECK_FALSE(StructuredOutputRequest::from_sampling_params(&sp).has_value());
}

TEST_CASE("mutually-exclusive constraint validation") {
  SUBCASE("more than one set throws") {
    StructuredOutputsParams p;
    p.json = "{}";
    p.regex = "x";
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("none set throws") {
    StructuredOutputsParams p;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("exactly one set is valid") {
    StructuredOutputsParams p;
    p.choice = std::vector<std::string>{"a", "b"};
    CHECK_NOTHROW(p.Verify());
  }
  SUBCASE("PostInit propagates the constraint validation") {
    SamplingParams sp;
    StructuredOutputsParams so;
    so.json = "{}";
    so.grammar = "root ::= \"x\"";  // two constraints
    sp.structured_outputs = so;
    CHECK_THROWS_AS(sp.PostInit(), std::runtime_error);
  }
}

TEST_CASE("all_constraints_none / all_non_structural_tag_constraints_none") {
  StructuredOutputsParams p;
  CHECK(p.all_constraints_none());
  CHECK(p.all_non_structural_tag_constraints_none());

  p.structural_tag = "<t/>";
  CHECK_FALSE(p.all_constraints_none());
  CHECK(p.all_non_structural_tag_constraints_none());  // structural_tag excluded
}

TEST_CASE("BitmaskWordsForVocab is ceil(vocab/32)") {
  CHECK(BitmaskWordsForVocab(32) == 1);
  CHECK(BitmaskWordsForVocab(33) == 2);
  CHECK(BitmaskWordsForVocab(1) == 1);
  CHECK(BitmaskWordsForVocab(64) == 2);
  CHECK(BitmaskWordsForVocab(65) == 3);
}

TEST_CASE("the ABCs are usable as interfaces (mock backend + grammar)") {
  const int kVocab = 100;   // -> ceil(100/32) = 4 words
  const int kMaxSeqs = 3;
  MockBackend backend(kVocab);

  TokenBitmask mask = backend.allocate_token_bitmask(kMaxSeqs);
  CHECK(mask.num_seqs == kMaxSeqs);
  CHECK(mask.num_words == 4);
  CHECK(mask.data.size() == static_cast<std::size_t>(kMaxSeqs) * 4);

  auto grammar = backend.compile_grammar(StructuredOutputOptions::kGrammar, "g");
  REQUIRE(grammar != nullptr);
  CHECK(grammar->accept_tokens("req-0", {1, 2, 3}));
  CHECK(grammar->validate_tokens({7, 8}) == std::vector<int32_t>{7, 8});
  CHECK_FALSE(grammar->is_terminated());

  grammar->fill_bitmask(mask, /*batch_index=*/1);
  // Row 1 all-allowed; rows 0 and 2 untouched (all-forbidden).
  for (int w = 0; w < mask.num_words; ++w) {
    CHECK(mask.data[0 * mask.num_words + w] == 0);
    CHECK(mask.data[1 * mask.num_words + w] == ~0);
    CHECK(mask.data[2 * mask.num_words + w] == 0);
  }

  grammar->rollback(2);
  grammar->reset();
  backend.destroy();
}
