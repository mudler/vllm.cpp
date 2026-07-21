// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE. Loads HF tokenizer.json (FromHfJson) and GGUF vocab (FromGguf) into
// one engine: added-token longest-match pre-pass -> pretokenize ->
// merge-ranked BPE -> vocab ids. Anything unsupported throws loudly.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/pretokenizer.h"

namespace vllm {
class GgufFile;  // model_executor/model_loader/gguf_reader.h
}

namespace vllm::tok {

// One entry of tokenizer.json "added_tokens" (or a GGUF special/user-defined
// token). `special` mirrors HF's flag: it does NOT change encoding (all added
// tokens split the text the same way); it drives detokenizer skip logic.
struct SpecialToken {
  std::string text;
  int32_t id = -1;
  bool special = false;
};

class Tokenizer {
 public:
  // Loads an HF tokenizer.json. Throws std::runtime_error on anything that is
  // not byte-level BPE with a recognized Split pre-tokenizer regex (no silent
  // wrong tokenization).
  static Tokenizer FromHfJson(const std::string& tokenizer_json_path);
  // Loads a GGUF byte-level BPE vocab (tokenizer.ggml.* kvs). Throws
  // std::runtime_error unless tokenizer.ggml.model == "gpt2" and
  // tokenizer.ggml.pre is a recognized pre-tokenizer name.
  static Tokenizer FromGguf(const GgufFile& f);

  // Encodes UTF-8 text. No BOS/EOS added (caller policy). Added tokens are
  // matched leftmost-longest against the raw text before pretokenization.
  std::vector<int32_t> Encode(std::string_view text) const;

  // Encodes UTF-8 text and APPLIES the post_processor's special tokens —
  // HF tokenizers' `encode(text, add_special_tokens=True)`, which is what vLLM
  // tokenizes prompts with. Concretely it realizes the TemplateProcessing
  // "single" template that ExtractBosEos already parses: prepend BosId() when
  // >= 0, append EosId() when >= 0.
  //
  // ADDED by the OPT (`OPTForCausalLM`) bring-up. Every Qwen tokenizer in the
  // tree has a ByteLevel post_processor carrying NEITHER id, so both are -1 and
  // this is byte-identical to Encode() for them — but OPT's post_processor DOES
  // declare `[SpecialToken </s>, Sequence A]`, i.e. a prepended BOS (id 2). We
  // parsed that field but never applied it, so OPT prompts came out one token
  // short and every continuation diverged (spec "seams that leaked" L4).
  std::vector<int32_t> EncodeWithSpecialTokens(std::string_view text) const;

  // Full (non-incremental) decode: ids -> raw UTF-8 bytes. Added tokens
  // contribute their literal content. Throws on unknown ids.
  std::string Decode(const std::vector<int32_t>& ids) const;

  // Raw token string as stored: byte-level (mapped) alphabet for vocab
  // tokens, literal content for added tokens. Throws on unknown ids.
  const std::string& TokenText(int32_t id) const;

  // True when `id` has an assigned token (in range and not a vocab hole).
  bool HasToken(int32_t id) const;
  // True when `id` is an added token carrying the special flag. Drives the
  // detokenizer's skip_special_tokens; encoding is unaffected.
  bool IsSpecial(int32_t id) const;

  // Number of id slots (max id + 1); ids in [0, VocabSize) may still be
  // unassigned for vocabs with holes.
  int32_t VocabSize() const { return static_cast<int32_t>(token_text_.size()); }
  // From the post_processor when trivially extractable (TemplateProcessing),
  // else -1. NOTE: callers should prefer the model config's eos_token_id;
  // e.g. Qwen's ByteLevel post_processor carries no eos/bos.
  int32_t EosId() const { return eos_id_; }
  int32_t BosId() const { return bos_id_; }
  SplitPattern Pattern() const { return pattern_; }
  const std::vector<SpecialToken>& AddedTokens() const { return added_tokens_; }

 private:
  Tokenizer() = default;

  // Builds token_text_/is_added_ from vocab_ + added_tokens_ (shared by the
  // HF and GGUF loaders). Throws on id collisions or out-of-range ids.
  void FinalizeTables();
  // Pretokenize + BPE for a text segment with no added tokens inside.
  void EncodePlain(std::string_view text, std::vector<int32_t>& out) const;

  std::unordered_map<std::string, int32_t> vocab_;  // mapped symbol -> id
  MergeRanks merge_ranks_;
  std::vector<SpecialToken> added_tokens_;
  std::vector<std::string> token_text_;  // id -> stored text ("" = unassigned)
  // id -> 0 plain vocab, 1 added, 2 added+special. Nonzero decodes literally.
  std::vector<uint8_t> is_added_;
  SplitPattern pattern_ = SplitPattern::kQwen2;
  int32_t eos_id_ = -1;
  int32_t bos_id_ = -1;
  // The tokens the POST-PROCESSOR itself prepends/appends, which is a strictly
  // narrower thing than "this model's bos/eos ids". Set ONLY from an HF
  // TemplateProcessing "single" template (FromHfJson); left -1 by FromGguf,
  // whose `tokenizer.ggml.{bos,eos}_token_id` merely NAME the special tokens
  // and do NOT imply the tokenizer adds them to a prompt. Conflating the two
  // would append an EOS to every GGUF prompt.
  int32_t template_bos_ = -1;
  int32_t template_eos_ = -1;
  bool ignore_merges_ = false;  // BPE option: whole-pretoken vocab hit wins
};

}  // namespace vllm::tok
