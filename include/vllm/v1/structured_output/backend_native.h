// vllm.cpp ORIGINAL component (§9 deviation) — NOT a 1:1 upstream port.
//
// The NATIVE grammar engine that plugs into the 1:1-ported structured-output
// seam (StructuredOutputBackend / StructuredOutputGrammar ABCs, Task 1). Upstream
// vLLM delegates to xgrammar/guidance/outlines/lm-format-enforcer; we ship a
// from-scratch, correctness-grade GBNF/EBNF + regex + choice engine at T0 and
// vendor xgrammar in a LATER milestone (a second backend behind this same proven
// seam). This mirrors upstream's own multi-backend design (see the plan's
// ARCHITECTURE DECISION) rather than deviating from it.
//
// The engine:
//   - parses an EBNF/GBNF grammar (llama.cpp GBNF-style) into a byte-level rule
//     table (every terminal matches exactly ONE byte; multi-byte codepoints are
//     lowered to byte sequences),
//   - runs a stack-based push-down FSM over grammar positions per request,
//   - decodes each vocab token to its RAW bytes via the tokenizer's inverse
//     GPT-2 bytes_to_unicode map and advances the FSM byte-by-byte,
//   - fills the per-step token bitmask WITHOUT re-running every token: a
//     TOKEN-BYTE TRIE is built ONCE at construction, and fill_bitmask is a single
//     DFS over (trie x grammar-state) so its cost is ~ (reachable trie nodes),
//     roughly independent of vocab size (THE BYTE-ALIGNMENT CORE).
//
// Coverage at T0: GRAMMAR (GBNF/EBNF), REGEX (regex->GBNF lowering,
// correctness-grade for common constructs), CHOICE (choice->GBNF, mirrors
// utils.py::choice_as_grammar). JSON / JSON_OBJECT are M3.4 Task 5 (JSON-schema
// -> GBNF) and throw here until then.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/v1/structured_output/backend_types.h"

namespace vllm::tok {
class Tokenizer;
}

namespace vllm::v1 {

// Opaque engine internals (defined in backend_native.cpp):
//   NativeCompiledGrammar — the parsed byte-level rule table.
//   NativeBackendShared    — the once-built token-byte trie + stop/EOS token ids
//                            + vocab size, shared by every grammar the backend
//                            compiles.
struct NativeCompiledGrammar;
struct NativeBackendShared;

// The per-request grammar + FSM state (a concrete StructuredOutputGrammar).
// Declared here (not just in the .cpp) so tests can down-cast to inspect the
// byte-trie perf counter.
class NativeGrammar : public StructuredOutputGrammar {
 public:
  NativeGrammar(std::shared_ptr<const NativeBackendShared> shared,
                std::shared_ptr<const NativeCompiledGrammar> grammar);
  ~NativeGrammar() override;

  bool accept_tokens(const std::string& request_id,
                     const std::vector<int32_t>& tokens) override;
  std::vector<int32_t> validate_tokens(
      const std::vector<int32_t>& tokens) override;
  void rollback(int num_tokens) override;
  void fill_bitmask(TokenBitmask& bitmask, int batch_index) override;
  bool is_terminated() override;
  void reset() override;

  // The number of (trie node) visits the LAST fill_bitmask performed. Used by
  // the perf test to assert the fill is sub-O(vocab): a restrictive grammar
  // visits far fewer nodes than there are vocab tokens.
  int64_t last_fill_visited_nodes() const { return last_fill_visited_nodes_; }

 private:
  // One FSM snapshot per accepted token (the front is the initial state); the
  // back is the current state. `done` marks a state reached by consuming the
  // EOS/stop token (nothing may follow).
  struct Snapshot;
  std::shared_ptr<const NativeBackendShared> shared_;
  std::shared_ptr<const NativeCompiledGrammar> grammar_;
  std::vector<Snapshot> history_;
  int64_t last_fill_visited_nodes_ = 0;
};

// The engine-level native backend. Constructed with the tokenizer + vocab size;
// builds the token-byte trie ONCE (over all regular vocab tokens, decoded to raw
// bytes). `stop_token_ids` are the tokens allowed only at an accepting state
// (typically just EOS); if empty, the tokenizer's EosId() is used when >= 0.
class NativeStructuredOutputBackend : public StructuredOutputBackend {
 public:
  NativeStructuredOutputBackend(const tok::Tokenizer& tokenizer, int vocab_size,
                                std::vector<int32_t> stop_token_ids = {});
  ~NativeStructuredOutputBackend() override;

  std::unique_ptr<StructuredOutputGrammar> compile_grammar(
      StructuredOutputOptions request_type,
      const std::string& grammar_spec) override;
  TokenBitmask allocate_token_bitmask(int max_num_seqs) override;
  void destroy() override;

  int vocab_size() const;

 private:
  std::shared_ptr<const NativeBackendShared> shared_;
};

// Factory helper for wiring the StructuredOutputManager's BackendFactory to the
// native engine. The manager builds its single backend lazily on the first
// grammar; this returns a std::function that does so with the given tokenizer +
// vocab size. The tokenizer must outlive the manager (the backend keeps a
// reference only during construction — the trie is a value copy of the bytes).
std::function<std::unique_ptr<StructuredOutputBackend>()>
MakeNativeBackendFactory(const tok::Tokenizer& tokenizer, int vocab_size,
                         std::vector<int32_t> stop_token_ids = {});

}  // namespace vllm::v1
