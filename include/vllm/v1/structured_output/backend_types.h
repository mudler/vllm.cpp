// Ported from: vllm/v1/structured_output/backend_types.py @ e24d1b24
//
// Scope (M3.4 Task 1): the backend-agnostic INTEGRATION contract for structured
// output — the request-type enum, the two abstract backends (engine-level
// StructuredOutputBackend + request-level StructuredOutputGrammar), the bitmask
// value type they exchange, and the (type, spec) cache key. Name-for-name with
// upstream: this is the parity-critical seam future PRs touch. NO engine, NO
// grammar logic here (the native engine is M3.4 Task 4, behind these ABCs).
//
// THE BITMASK INTERFACE (backend-invariant, per the architecture decision): a
// packed [num_seqs, ceil(vocab_size/32)] int32 mask — bit j of row i set => token
// j is grammar-valid for sequence i; a cleared bit masks that logit to -inf
// (reusing the M1.7 sampler's allowed-token -inf masking). Chosen here as a small
// TokenBitmask struct (data + shape) rather than a bare vector so fill_bitmask /
// allocate_token_bitmask carry their shape explicitly. Upstream this is a
// torch.Tensor of dtype int32 shape [max_num_seqs, ceil(vocab/32)].
//
// DEFERRED (marked, cite upstream), slotted in later without reshaping:
//   - StructuredOutputOptions::kStructuralTag — upstream member STRUCTURAL_TAG
//     (backend_types.py:25); the structural-tag path is deferred at T0.
//   - StructuredOutputBackend's dataclass fields vllm_config / tokenizer
//     (backend_types.py:102-103): the CONCRETE backend (Task 4) takes
//     (vllm_config, tokenizer, vocab_size) at construction; the ABC stays pure so
//     Task 1 pulls in no VllmConfig / tokenizer dependency. Only vocab_size is
//     load-bearing here and the concrete backend owns it (it sizes the bitmask).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vllm::v1 {

// StructuredOutputOptions (backend_types.py:19-25). Members mirror upstream in
// order; the integer values are not load-bearing (used only as cache-key /
// dispatch discriminants, never serialized to a tensor).
enum class StructuredOutputOptions {
  kJson,
  kJsonObject,
  kRegex,
  kGrammar,
  kChoice,
  // DEFERRED at T0 (upstream STRUCTURAL_TAG). Present so the enum matches
  // upstream 1:1; get_structured_output_key maps to it, but no backend compiles
  // it yet (Task 4+).
  kStructuralTag,
};

// StructuredOutputKey (backend_types.py:28): tuple[StructuredOutputOptions, str]
// — the (request_type, grammar_spec) pair used to key the compiled-grammar
// cache. Modeled as std::pair to mirror the tuple; StructuredOutputKeyHash lets
// it index an unordered_map (the Task 2 manager's grammar cache).
using StructuredOutputKey = std::pair<StructuredOutputOptions, std::string>;

struct StructuredOutputKeyHash {
  std::size_t operator()(const StructuredOutputKey& key) const {
    const std::size_t h1 =
        std::hash<int>{}(static_cast<int>(key.first));
    const std::size_t h2 = std::hash<std::string>{}(key.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

// The packed per-step token bitmask exchanged between grammar and sampler.
// Row-major [num_seqs, num_words], num_words == ceil(vocab_size / 32). Bit j of
// row i (word j/32, bit j%32) set => token j allowed for sequence i.
struct TokenBitmask {
  std::vector<int32_t> data;  // size == num_seqs * num_words
  int num_seqs = 0;
  int num_words = 0;
};

// ceil(vocab_size / 32): the number of int32 words per bitmask row.
inline int BitmaskWordsForVocab(int vocab_size) {
  return (vocab_size + 31) / 32;
}

// StructuredOutputGrammar (backend_types.py:31-95): the request-level backend.
// A per-request compiled grammar + FSM state. Abstract; the native engine
// (Task 4) implements it. Method names 1:1 with upstream.
class StructuredOutputGrammar {
 public:
  virtual ~StructuredOutputGrammar() = default;

  // accept_tokens (backend_types.py:34-46): advance the FSM by `tokens` for the
  // given request. Returns true iff every token was accepted.
  virtual bool accept_tokens(const std::string& request_id,
                             const std::vector<int32_t>& tokens) = 0;

  // validate_tokens (backend_types.py:48-60): the accepted PREFIX of `tokens`
  // WITHOUT advancing the FSM (empty if none accepted).
  virtual std::vector<int32_t> validate_tokens(
      const std::vector<int32_t>& tokens) = 0;

  // rollback (backend_types.py:62-70): undo the last `num_tokens` accepted
  // tokens, reverting the processed-token counters too.
  virtual void rollback(int num_tokens) = 0;

  // fill_bitmask (backend_types.py:72-80): set the allowed-token bits for the
  // current FSM state into row `batch_index` of `bitmask`.
  virtual void fill_bitmask(TokenBitmask& bitmask, int batch_index) = 0;

  // is_terminated (backend_types.py:82-89): true once the grammar has reached an
  // accepting/final state.
  virtual bool is_terminated() = 0;

  // reset (backend_types.py:91-95): reset the FSM to its initial state.
  virtual void reset() = 0;
};

// StructuredOutputBackend (backend_types.py:98-137): the engine-level backend.
// Abstract; the concrete native backend (Task 4) is constructed with
// (vllm_config, tokenizer, vocab_size) — see the DEFERRED note in the file
// header. Method names 1:1 with upstream.
class StructuredOutputBackend {
 public:
  virtual ~StructuredOutputBackend() = default;

  // compile_grammar (backend_types.py:106-120): compile a spec of the given
  // request_type into a per-request grammar.
  virtual std::unique_ptr<StructuredOutputGrammar> compile_grammar(
      StructuredOutputOptions request_type,
      const std::string& grammar_spec) = 0;

  // allocate_token_bitmask (backend_types.py:122-130): allocate a zeroed
  // [max_num_seqs, ceil(vocab_size/32)] int32 bitmask (vocab_size owned by the
  // concrete backend).
  virtual TokenBitmask allocate_token_bitmask(int max_num_seqs) = 0;

  // destroy (backend_types.py:132-136): backend-specific cleanup.
  virtual void destroy() = 0;
};

}  // namespace vllm::v1
