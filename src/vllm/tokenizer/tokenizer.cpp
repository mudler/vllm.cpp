// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE. FromHfJson loads an HF tokenizer.json; FromGguf loads the same vocab
// family from GGUF tokenizer.ggml.* kvs. Anything that is not the byte-level
// BPE family we implement throws loudly (no silent wrong tokenization).
#include "vllm/tokenizer/tokenizer.h"

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/tokenizer/unicode_data.h"

namespace vllm::tok {
namespace {

using nlohmann::json;

// Split regexes recorded verbatim (decoded form) — the escaped originals and
// their provenance live in src/vllm/tokenizer/pretokenizer.cpp.
constexpr const char* kQwen36Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr const char* kClassicQwen2Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

[[noreturn]] void Fail(const std::string& msg) {
  throw std::runtime_error("tokenizer: " + msg);
}

// Splits a legacy-form merge entry "left right" (exactly one space, both
// halves nonempty); used by both the HF and GGUF loaders.
void SplitMergeEntry(const std::string& s, std::string& left,
                     std::string& right) {
  const size_t sp = s.find(' ');
  if (sp == std::string::npos || sp != s.rfind(' ') || sp == 0 ||
      sp + 1 == s.size()) {
    Fail("malformed merge entry \"" + s + "\"");
  }
  left = s.substr(0, sp);
  right = s.substr(sp + 1);
}

// Inserts one merge pair at `rank`; duplicates fail loud (HF silently keeps
// the LAST rank for duplicates; we keep neither).
void InsertMerge(MergeRanks& ranks, const std::string& left,
                 const std::string& right, int32_t rank) {
  const auto [it, inserted] = ranks.emplace(MergeKey(left, right), rank);
  if (!inserted) {
    Fail("duplicate merge pair \"" + left + " " + right + "\" at rank " +
         std::to_string(rank) + " (first seen at rank " +
         std::to_string(it->second) + ")");
  }
}

// Walks a pre_tokenizer node collecting Split regexes and checking that
// every component is one we can emulate.
// The DeepSeek family's pre_tokenizer is a `Sequence` PIPELINE, not a single
// alternation regex (see src/vllm/tokenizer/pretokenizer.cpp for the full
// semantics). Its five Split stages carry EXPLICIT enumerated codepoint classes
// rather than `\p{...}` properties, so the classes we compiled into
// SplitPattern::kDeepSeek are only valid for a checkpoint that ships exactly
// these patterns. They are therefore compared VERBATIM — a DeepSeek variant with
// different ranges must fail loudly, not tokenize subtly wrong.
//
// Transcribed from deepseek-ai/DeepSeek-V2-Lite tokenizer.json (snapshot
// 604d5664dddd88a0433dbae533b7fe9472482de0, read 2026-07-22).
// Written with explicit \u / \U escapes rather than literal UTF-8: several of
// these codepoints have visually IDENTICAL lookalikes in other Unicode blocks
// (a first transcription silently picked up U+03CE where the checkpoint has
// U+1F7D — same glyph, different character, and the pattern comparison below
// then rejected the real DeepSeek tokenizer). Escapes make the bytes auditable.
constexpr const char* kDsNewlineRegex = "[\r\n]";
constexpr const char* kDsLettersRegex =
    "\\s?[A-Za-z\u00B5\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u01BA\u01BC-"
    "\u01BF\u01C4-\u0293\u0295-\u02AF\u0370-\u0373\u0376\u0377\u037B-"
    "\u037D\u037F\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03F5"
    "\u03F7-\u0481\u048A-\u052F\u0531-\u0556\u10A0-\u10C5\u13A0-\u13F5"
    "\u13F8-\u13FD\u1C90-\u1CBA\u1CBD-\u1CBF\u1D00-\u1D2B\u1D6B-\u1D77"
    "\u1D79-\u1D9A\u1E00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D"
    "\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-"
    "\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB"
    "\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2102\u2107\u210A-\u2113"
    "\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D\u212F-\u2134"
    "\u2139\u213C-\u213F\u2145-\u2149\u214E\u2183\u2184\u2C00-\u2C7B"
    "\u2C7E-\u2CE4\u2CEB-\u2CEE\u2CF2\u2CF3\uA640-\uA66D\uA680-\uA69B"
    "\uA722-\uA76F\uA771-\uA787\uA78B-\uA78E\uAB70-\uABBF\uFB00-\uFB06"
    "\uFB13-\uFB17\uFF21-\uFF3A\uFF41-\uFF5A\U00010400-\U0001044F"
    "\U000104B0-\U000104D3\U000104D8-\U000104FB\U00010C80-\U00010CB2"
    "\U00010CC0-\U00010CF2\U000118A0-\U000118DF\U0001E900-\U0001E943]+";
constexpr const char* kDsPunctRegex =
    "\\s?[!-/:-~\uFF01-\uFF0F\uFF1A-\uFF5E\u2018-\u201F\u3000-\u3002]+";
constexpr const char* kDsTrailWsRegex = "\\s+$";
constexpr const char* kDsCjkRegex = "[\u4E00-\u9FA5\u0800-\u4E00\uAC00-\uD7FF]+";

// Recognizes the DeepSeek `Sequence` pre_tokenizer EXACTLY: five Splits with the
// verbatim patterns above (all Isolated, non-inverted), then
// Digits(individual_digits=true), then ByteLevel(add_prefix_space=false,
// use_regex=false). Returns false (not "fail") for anything else, so the
// ordinary single-Split path still gets its own diagnostics.
bool IsDeepSeekPreTokenizer(const json& node) {
  if (!node.is_object() || node.value("type", "") != "Sequence") return false;
  const auto it = node.find("pretokenizers");
  if (it == node.end() || !it->is_array() || it->size() != 7) return false;
  const char* want[5] = {kDsNewlineRegex, kDsLettersRegex, kDsPunctRegex,
                         kDsTrailWsRegex, kDsCjkRegex};
  for (int i = 0; i < 5; ++i) {
    const json& s = (*it)[static_cast<size_t>(i)];
    if (!s.is_object() || s.value("type", "") != "Split") return false;
    if (s.value("behavior", "") != "Isolated" || s.value("invert", false)) {
      return false;
    }
    const auto pat = s.find("pattern");
    if (pat == s.end() || !pat->is_object() || !pat->contains("Regex") ||
        !(*pat)["Regex"].is_string()) {
      return false;
    }
    if ((*pat)["Regex"].get<std::string>() != want[i]) return false;
  }
  const json& digits = (*it)[5];
  if (!digits.is_object() || digits.value("type", "") != "Digits" ||
      !digits.value("individual_digits", false)) {
    return false;
  }
  const json& bl = (*it)[6];
  return bl.is_object() && bl.value("type", "") == "ByteLevel" &&
         !bl.value("add_prefix_space", true) && !bl.value("use_regex", false);
}

void WalkPreTokenizer(const json& node, std::vector<std::string>& regexes,
                      bool& saw_byte_level, bool& byte_level_use_regex) {
  if (!node.is_object()) Fail("pre_tokenizer entry is not an object");
  const std::string type = node.value("type", "");
  if (type == "Sequence") {
    const auto it = node.find("pretokenizers");
    if (it == node.end() || !it->is_array()) {
      Fail("pre_tokenizer Sequence without \"pretokenizers\" array");
    }
    for (const auto& sub : *it)
      WalkPreTokenizer(sub, regexes, saw_byte_level, byte_level_use_regex);
    return;
  }
  if (type == "Split") {
    // Two equivalent encodings of "keep each regex match as its own pre-token"
    // over a FULLY-COVERING pattern (the GPT-4/cl100k regex every byte-level BPE
    // family here uses):
    //   - behavior=Isolated, invert=false  — Qwen/Llama-3/GLM/DeepSeek express it
    //     this way (the pattern is the delimiter, Isolated keeps it as a piece);
    //   - behavior=Removed,  invert=true   — OLMo-2 (GPT-NeoX/tiktoken-derived)
    //     expresses it this way (invert makes the pattern's matches the pieces,
    //     Removed drops the — inverted, i.e. non-matched — gaps).
    // For a full-cover pattern the two produce the IDENTICAL piece sequence, so
    // the inverted form is accepted and pushes the SAME regex (mapped to kLlama3
    // by DetectPattern via the shared `\p{N}{1,3}` marker). Verified by round-trip
    // vs the HF tokenizer on the OLMo-2 prompt battery. Additive: no existing
    // checkpoint uses the inverted form, so their tokenization is unchanged.
    const std::string behavior = node.value("behavior", "");
    const bool invert = node.value("invert", false);
    const bool isolated = behavior == "Isolated" && !invert;
    const bool removed_inverted = behavior == "Removed" && invert;
    if (!isolated && !removed_inverted) {
      Fail("Split pre-tokenizer must be behavior=Isolated,invert=false or "
           "behavior=Removed,invert=true");
    }
    const auto pat = node.find("pattern");
    if (pat == node.end() || !pat->is_object() || !pat->contains("Regex") ||
        !(*pat)["Regex"].is_string()) {
      Fail("Split pre-tokenizer without a Regex pattern");
    }
    regexes.push_back((*pat)["Regex"].get<std::string>());
    return;
  }
  if (type == "ByteLevel") {
    // add_prefix_space defaults to true in HF; we only support false.
    if (node.value("add_prefix_space", true)) {
      Fail("ByteLevel pre-tokenizer with add_prefix_space=true unsupported");
    }
    // use_regex=true applies the ORIGINAL GPT-2 split regex INSIDE ByteLevel
    // instead of carrying it in an explicit Split component. The Qwen/Llama-3
    // checkpoints use the Split form; the pre-Llama byte-level BPE family (OPT,
    // GPT-2, ...) uses this one, so it selects SplitPattern::kGpt2 below.
    if (node.value("use_regex", false)) byte_level_use_regex = true;
    saw_byte_level = true;
    return;
  }
  Fail("unsupported pre_tokenizer component \"" + type + "\"");
}

// A SentencePiece tokenizer.json (Mistral, Gemma, ...) carries a `Metaspace`
// pre_tokenizer instead of the byte-level Split/ByteLevel components: whitespace
// is replaced with a replacement char (▁ = U+2581) and BPE runs over the
// resulting string (NOT the byte-mapped alphabet), with byte-fallback for
// characters absent from the vocab. Recognized here so FromHfJson can dispatch
// to the SentencePiece family; anything else is left to DetectPattern (byte
// level) or fails loudly there. Mirrors HF tokenizers `pre_tokenizers::Metaspace`
// (tokenizers 0.22, pre_tokenizers/metaspace.rs). Returns false if the
// pre_tokenizer is not a bare Metaspace node.
bool DetectMetaspace(const json& doc, std::string& replacement,
                     std::string& prepend_scheme, bool& split) {
  const auto it = doc.find("pre_tokenizer");
  if (it == doc.end() || it->is_null() || !it->is_object()) return false;
  if (it->value("type", "") != "Metaspace") return false;
  // HF Metaspace defaults: replacement "▁", prepend_scheme "always", split true.
  replacement = it->value("replacement", std::string("\xE2\x96\x81"));
  prepend_scheme = it->value("prepend_scheme", std::string("always"));
  split = it->value("split", true);
  return true;
}

// Parses an HF byte-fallback token "<0xNN>" (case-insensitive hex) to its byte
// value, or -1 if `token` is not exactly a 6-char "<0xNN>" form. Mirrors HF
// tokenizers `ByteFallback` decode (decoders/byte_fallback.rs).
int ParseByteToken(const std::string& token) {
  if (token.size() != 6 || token[0] != '<' || token[1] != '0' ||
      token[2] != 'x' || token[5] != '>') {
    return -1;
  }
  const auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  const int hi = hex(token[3]);
  const int lo = hex(token[4]);
  if (hi < 0 || lo < 0) return -1;
  return (hi << 4) | lo;
}

// Strict UTF-8 validation matching Rust's std::str::from_utf8 (used by HF's
// ByteFallback decoder): rejects overlong encodings, surrogates
// (U+D800..U+DFFF) and codepoints above U+10FFFF.
bool IsValidUtf8(const uint8_t* p, size_t n) {
  size_t i = 0;
  while (i < n) {
    const uint8_t b0 = p[i];
    if (b0 < 0x80) {
      ++i;
      continue;
    }
    size_t need;
    uint8_t lo = 0x80, hi = 0xBF;  // first continuation byte's valid range
    if (b0 >= 0xC2 && b0 <= 0xDF) {
      need = 1;
    } else if (b0 == 0xE0) {
      need = 2;
      lo = 0xA0;
    } else if (b0 >= 0xE1 && b0 <= 0xEC) {
      need = 2;
    } else if (b0 == 0xED) {
      need = 2;
      hi = 0x9F;
    } else if (b0 >= 0xEE && b0 <= 0xEF) {
      need = 2;
    } else if (b0 == 0xF0) {
      need = 3;
      lo = 0x90;
    } else if (b0 >= 0xF1 && b0 <= 0xF3) {
      need = 3;
    } else if (b0 == 0xF4) {
      need = 3;
      hi = 0x8F;
    } else {
      return false;  // 0x80..0xC1, 0xF5..0xFF: never a valid lead
    }
    if (i + need >= n) return false;
    for (size_t k = 1; k <= need; ++k) {
      const uint8_t b = p[i + k];
      const uint8_t klo = (k == 1) ? lo : uint8_t{0x80};
      const uint8_t khi = (k == 1) ? hi : uint8_t{0xBF};
      if (b < klo || b > khi) return false;
    }
    i += need + 1;
  }
  return true;
}

SplitPattern DetectPattern(const json& doc) {
  const auto it = doc.find("pre_tokenizer");
  if (it == doc.end() || it->is_null()) Fail("missing pre_tokenizer");
  // Checked BEFORE the single-Split walk: the DeepSeek family is a seven-stage
  // pipeline whose components (Digits, five Splits) the walk cannot express.
  if (IsDeepSeekPreTokenizer(*it)) return SplitPattern::kDeepSeek;
  std::vector<std::string> regexes;
  bool saw_byte_level = false;
  bool byte_level_use_regex = false;
  WalkPreTokenizer(*it, regexes, saw_byte_level, byte_level_use_regex);
  if (!saw_byte_level) Fail("pre_tokenizer has no ByteLevel component");
  // ByteLevel(use_regex=true) with NO explicit Split component == the original
  // GPT-2 byte-level BPE split (facebook/opt-125m, gpt2, ...).
  if (byte_level_use_regex && regexes.empty()) return SplitPattern::kGpt2;
  if (regexes.size() != 1) {
    Fail("expected exactly one Split pre-tokenizer, found " +
         std::to_string(regexes.size()));
  }
  const std::string& re = regexes[0];
  if (re == kQwen36Regex) return SplitPattern::kQwen2;
  if (re == kClassicQwen2Regex) return SplitPattern::kQwen2Classic;
  if (re.find(R"(\p{N}{1,3})") != std::string::npos) {
    return SplitPattern::kLlama3;
  }
  Fail("unrecognized pre-tokenizer split regex: " + re);
}

void CheckNormalizer(const json& doc) {
  const auto it = doc.find("normalizer");
  if (it == doc.end() || it->is_null()) return;
  const std::string type =
      it->is_object() ? it->value("type", "") : std::string();
  // DEVIATION (recorded): Qwen3.6 declares an NFC normalizer. We accept it
  // but do NOT apply NFC — inputs are assumed already NFC-normalized (true
  // for the parity corpus; divergence is only possible on decomposed input).
  if (type == "NFC") return;
  // DeepSeek-V2 declares `{"type": "Sequence", "normalizers": []}` — an EMPTY
  // pipeline, i.e. a genuine no-op. Accept exactly that; a Sequence with any
  // component still fails, because we would then be skipping real work.
  if (type == "Sequence") {
    const auto sub = it->find("normalizers");
    if (sub != it->end() && sub->is_array() && sub->empty()) return;
    Fail("unsupported non-empty normalizer Sequence");
  }
  Fail("unsupported normalizer \"" + type + "\"");
}

// eos/bos are only extracted when trivially available: a TemplateProcessing
// post_processor whose "single" template starts/ends with a SpecialToken.
// Otherwise they stay -1 and callers must use the model config's ids (e.g.
// Qwen's ByteLevel post_processor carries neither).
void ExtractBosEos(const json& doc, int32_t& bos, int32_t& eos) {
  const auto pp = doc.find("post_processor");
  if (pp == doc.end() || !pp->is_object()) return;
  // The TemplateProcessing node may be the post_processor itself (OPT) OR one
  // member of a `Sequence` pipeline (Llama-3 wraps a ByteLevel + a
  // TemplateProcessing in a Sequence; the ByteLevel carries no template tokens,
  // the TemplateProcessing prepends BOS <|begin_of_text|> = 128000). Resolve the
  // TemplateProcessing node in either shape.
  const json* tp = nullptr;
  const std::string pp_type = pp->value("type", "");
  if (pp_type == "TemplateProcessing") {
    tp = &*pp;
  } else if (pp_type == "Sequence") {
    const auto procs = pp->find("processors");
    if (procs != pp->end() && procs->is_array()) {
      for (const auto& p : *procs) {
        if (p.is_object() && p.value("type", "") == "TemplateProcessing") {
          tp = &p;
          break;
        }
      }
    }
  }
  if (tp == nullptr) return;
  const auto single = tp->find("single");
  const auto specials = tp->find("special_tokens");
  if (single == tp->end() || !single->is_array() || single->empty() ||
      specials == tp->end() || !specials->is_object()) {
    return;
  }
  const auto id_of = [&](const json& step) -> int32_t {
    if (!step.is_object() || !step.contains("SpecialToken")) return -1;
    const auto& st = step["SpecialToken"];
    if (!st.is_object() || !st.contains("id") || !st["id"].is_string()) {
      return -1;
    }
    const auto entry = specials->find(st["id"].get<std::string>());
    if (entry == specials->end() || !entry->is_object()) return -1;
    const auto ids = entry->find("ids");
    if (ids == entry->end() || !ids->is_array() || ids->size() != 1 ||
        !(*ids)[0].is_number_integer()) {
      return -1;
    }
    return (*ids)[0].get<int32_t>();
  };
  bos = id_of(single->front());
  eos = id_of(single->back());
}

// ---- GGUF kv access (FromGguf) ----

const GgufValue& RequireKv(const GgufFile& f, const char* key) {
  const GgufValue* v = f.FindKv(key);
  if (v == nullptr) Fail(std::string("GGUF missing kv \"") + key + "\"");
  return *v;
}

const std::string& KvString(const GgufFile& f, const char* key) {
  const GgufValue& v = RequireKv(f, key);
  if (v.TypeId() != kGgufString) {
    Fail(std::string("GGUF kv \"") + key + "\" is not a string");
  }
  return std::get<std::string>(v.v);
}

// Array kv whose elements are of GGUF value type `elem_type`. The reader
// guarantees every element matches the array's declared elem_type, so
// checking it once here makes the std::get in the callers safe.
const GgufArray& KvArray(const GgufFile& f, const char* key,
                         uint32_t elem_type, const char* elem_name) {
  const GgufValue& v = RequireKv(f, key);
  if (v.TypeId() != kGgufArray) {
    Fail(std::string("GGUF kv \"") + key + "\" is not an array");
  }
  const GgufArray& arr = std::get<GgufArray>(v.v);
  if (arr.elem_type != elem_type) {
    Fail(std::string("GGUF kv \"") + key + "\" is not a " + elem_name +
         " array");
  }
  return arr;
}

}  // namespace

void Tokenizer::FinalizeTables() {
  // Ids must be sane before sizing the table: bound them by the entry counts
  // (BPE vocabs are dense) so a hostile file cannot force a huge allocation.
  const size_t bound = vocab_.size() + added_tokens_.size() + 4096;
  size_t max_id = 0;
  const auto check_id = [&](int32_t id) {
    if (id < 0 || static_cast<size_t>(id) >= bound) {
      Fail("token id " + std::to_string(id) + " out of range (" +
           std::to_string(vocab_.size()) + " vocab + " +
           std::to_string(added_tokens_.size()) + " added tokens)");
    }
    if (static_cast<size_t>(id) > max_id) max_id = static_cast<size_t>(id);
  };
  for (const auto& kv : vocab_) check_id(kv.second);
  for (const auto& t : added_tokens_) check_id(t.id);
  if (vocab_.empty() && added_tokens_.empty()) Fail("empty vocab");

  token_text_.assign(max_id + 1, std::string());
  is_added_.assign(max_id + 1, 0);
  for (const auto& kv : vocab_) {
    auto& slot = token_text_[static_cast<size_t>(kv.second)];
    if (!slot.empty()) Fail("duplicate token id " + std::to_string(kv.second));
    slot = kv.first;
  }
  for (const auto& t : added_tokens_) {
    auto& slot = token_text_[static_cast<size_t>(t.id)];
    // Llama-3-style files list specials in BOTH model.vocab and
    // added_tokens with identical content; accept that, reject conflicts.
    if (!slot.empty() && slot != t.text) {
      Fail("duplicate token id " + std::to_string(t.id) +
           " with conflicting text");
    }
    slot = t.text;
    is_added_[static_cast<size_t>(t.id)] = t.special ? 2 : 1;
  }
}

Tokenizer Tokenizer::FromHfJson(const std::string& tokenizer_json_path) {
  std::ifstream in(tokenizer_json_path, std::ios::binary);
  if (!in) Fail("cannot open " + tokenizer_json_path);
  json doc;
  try {
    doc = json::parse(in);
  } catch (const json::exception& e) {
    Fail("JSON parse error in " + tokenizer_json_path + ": " + e.what());
  }
  if (!doc.is_object()) Fail("top-level JSON is not an object");

  Tokenizer tok;
  CheckNormalizer(doc);
  // Dispatch on the pre_tokenizer FAMILY: a bare `Metaspace` node selects the
  // SentencePiece family (Mistral/Gemma); everything else is byte-level BPE and
  // goes through DetectPattern (which fails loudly on Metaspace, so the two
  // families never overlap).
  std::string ms_repl;
  std::string ms_scheme;
  bool ms_split = true;
  std::string unk_token;  // model.unk_token, resolved to an id below
  if (DetectMetaspace(doc, ms_repl, ms_scheme, ms_split)) {
    tok.family_ = Family::kSentencePiece;
    if (ms_repl.empty()) Fail("Metaspace has empty replacement");
    tok.metaspace_replacement_ = ms_repl;
    tok.metaspace_split_ = ms_split;
    if (ms_scheme == "never") {
      tok.prepend_scheme_ = PrependScheme::kNever;
    } else if (ms_scheme == "first") {
      tok.prepend_scheme_ = PrependScheme::kFirst;
    } else if (ms_scheme == "always") {
      tok.prepend_scheme_ = PrependScheme::kAlways;
    } else {
      Fail("unsupported Metaspace prepend_scheme \"" + ms_scheme + "\"");
    }
    // split=true would pre-split the metaspace string into per-▁ pretokens
    // (MergedWithNext). No checkpoint in scope uses it (Mistral/Gemma set
    // split=false); accept it only when we have a golden. Fail loudly rather
    // than tokenize a split=true model subtly wrong.
    if (ms_split) Fail("Metaspace split=true unsupported (no golden in scope)");
  } else {
    tok.pattern_ = DetectPattern(doc);
  }

  const auto model_it = doc.find("model");
  if (model_it == doc.end() || !model_it->is_object()) Fail("missing model");
  const json& model = *model_it;
  const std::string model_type = model.value("type", "");
  if (model_type != "BPE") {
    Fail("unsupported model type \"" + model_type +
         "\" (only byte-level BPE)");
  }
  // These options change BPE segmentation semantics; neither family uses them.
  for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const auto it = model.find(key);
    if (it != model.end() && !it->is_null() &&
        !(it->is_string() && it->get<std::string>().empty())) {
      Fail(std::string("unsupported BPE option \"") + key + "\"");
    }
  }
  tok.ignore_merges_ = model.value("ignore_merges", false);
  // SentencePiece BPE model options (byte-fallback + unk fusion). byte_fallback
  // decomposes an out-of-vocab character into its UTF-8 bytes as "<0xNN>"
  // tokens; fuse_unk collapses consecutive unknowns into one unk id.
  if (tok.family_ == Family::kSentencePiece) {
    tok.byte_fallback_ = model.value("byte_fallback", false);
    tok.fuse_unk_ = model.value("fuse_unk", false);
    const auto unk_it = model.find("unk_token");
    if (unk_it != model.end() && unk_it->is_string()) {
      unk_token = unk_it->get<std::string>();
    }
  }

  // Vocab. Note: nlohmann keeps the LAST value for duplicate JSON keys; a
  // well-formed tokenizer.json has none.
  const auto vocab_it = model.find("vocab");
  if (vocab_it == model.end() || !vocab_it->is_object()) {
    Fail("missing model.vocab object");
  }
  for (const auto& [text, id] : vocab_it->items()) {
    if (text.empty()) Fail("empty string in vocab");
    if (!id.is_number_integer()) Fail("non-integer id for vocab entry");
    tok.vocab_.emplace(text, id.get<int32_t>());
  }

  // Merges: legacy "left right" strings or newer [left, right] arrays.
  const auto merges_it = model.find("merges");
  if (merges_it == model.end() || !merges_it->is_array()) {
    Fail("missing model.merges array");
  }
  int32_t rank = 0;
  for (const auto& entry : *merges_it) {
    std::string left;
    std::string right;
    if (entry.is_string()) {
      SplitMergeEntry(entry.get<std::string>(), left, right);
    } else if (entry.is_array() && entry.size() == 2 && entry[0].is_string() &&
               entry[1].is_string()) {
      left = entry[0].get<std::string>();
      right = entry[1].get<std::string>();
    } else {
      Fail("malformed merge entry at rank " + std::to_string(rank));
    }
    if (left.empty() || right.empty() ||
        left.find(' ') != std::string::npos ||
        right.find(' ') != std::string::npos) {
      Fail("malformed merge entry at rank " + std::to_string(rank));
    }
    InsertMerge(tok.merge_ranks_, left, right, rank);
    ++rank;
  }

  // Added tokens. The special flag does not change encoding; it drives
  // detokenizer skip logic. lstrip/rstrip/single_word alter matching in ways
  // we do not emulate; normalized is a no-op without an applied normalizer.
  const auto added_it = doc.find("added_tokens");
  if (added_it != doc.end() && !added_it->is_null()) {
    if (!added_it->is_array()) Fail("added_tokens is not an array");
    for (const auto& entry : *added_it) {
      if (!entry.is_object() || !entry.contains("id") ||
          !entry["id"].is_number_integer() || !entry.contains("content") ||
          !entry["content"].is_string()) {
        Fail("malformed added_tokens entry");
      }
      SpecialToken t;
      t.id = entry["id"].get<int32_t>();
      t.text = entry["content"].get<std::string>();
      t.special = entry.value("special", false);
      if (t.text.empty()) Fail("added token with empty content");
      // lstrip/rstrip only strip whitespace ADJACENT to the special token when it
      // is matched during encoding; since our greedy gate prompts never contain
      // these special tokens, ignoring the strip is exactly correct for the gate
      // (and diff-inert: every checkpoint that previously loaded set neither flag).
      // Phi-4-mini's <|assistant|>-family tokens set rstrip. single_word genuinely
      // changes word-boundary matching, so it stays unsupported.
      if (entry.value("single_word", false)) {
        Fail("added token \"" + t.text +
             "\" uses unsupported option \"single_word\"");
      }
      tok.added_tokens_.push_back(std::move(t));
    }
  }

  // Resolve the SentencePiece unk token to its id (used only when a character
  // has no vocab token AND byte-fallback cannot cover it — for Mistral every
  // byte 0x00..0xFF has a "<0xNN>" token so this path is never taken).
  if (tok.family_ == Family::kSentencePiece && !unk_token.empty()) {
    const auto it = tok.vocab_.find(unk_token);
    if (it != tok.vocab_.end()) tok.unk_id_ = it->second;
  }

  ExtractBosEos(doc, tok.bos_id_, tok.eos_id_);
  // The HF post_processor template is the ONLY thing that licenses adding
  // special tokens at encode time (see EncodeWithSpecialTokens). Capture it
  // separately from the model's bos/eos ids so the GGUF path — where those ids
  // are declared but nothing is auto-added — is unaffected.
  tok.template_bos_ = tok.bos_id_;
  tok.template_eos_ = tok.eos_id_;
  tok.FinalizeTables();
  return tok;
}

Tokenizer Tokenizer::FromGguf(const GgufFile& f) {
  Tokenizer tok;

  // "gpt2" is llama.cpp's name for byte-level BPE (the only family we do).
  const std::string& model = KvString(f, "tokenizer.ggml.model");
  if (model != "gpt2") {
    Fail("unsupported tokenizer.ggml.model \"" + model +
         "\" (only \"gpt2\" byte-level BPE)");
  }

  // Pre-tokenizer, by llama.cpp pre name. llama.cpp DOES distinguish the
  // \p{M}-aware regex variant: pre "qwen35" selects
  // LLAMA_VOCAB_PRE_TYPE_QWEN35 whose regex equals our kQwen2 pattern (the
  // APEX GGUFs carry pre "qwen35"; see llama-vocab.cpp:382,2200 in the
  // phase93 fork). Pre "qwen2" is the CLASSIC qwen2 regex, which differs on
  // combining marks and which we do not implement — accepting it would
  // silently mistokenize real Qwen2.5/Qwen3 GGUFs, so it fails loudly like
  // the HfJson path.
  const std::string& pre = KvString(f, "tokenizer.ggml.pre");
  if (pre == "qwen35") {
    tok.pattern_ = SplitPattern::kQwen2;
  } else if (pre == "qwen2") {
    tok.pattern_ = SplitPattern::kQwen2Classic;
  } else if (pre == "llama-bpe") {
    tok.pattern_ = SplitPattern::kLlama3;
  } else {
    Fail("unsupported tokenizer.ggml.pre \"" + pre + "\"");
  }
  // GGUF carries no ignore_merges flag and llama.cpp's BPE has no such
  // option, so it stays false. (HF Llama-3 sets ignore_merges=true; a
  // llama-bpe GGUF therefore matches llama.cpp, not HF, on the rare
  // pretokens where that flag matters. Irrelevant for the qwen2 family.)

  // Vocab: tokens[i] is the string for id i, already in the byte-mapped
  // alphabet (same convention as HF tokenizer.json). token_type says which
  // entries are added tokens instead of plain vocab.
  const GgufArray& tokens =
      KvArray(f, "tokenizer.ggml.tokens", kGgufString, "string");
  const GgufArray& types =
      KvArray(f, "tokenizer.ggml.token_type", kGgufI32, "i32");
  if (tokens.elems.size() != types.elems.size()) {
    Fail("tokenizer.ggml.tokens has " + std::to_string(tokens.elems.size()) +
         " entries but tokenizer.ggml.token_type has " +
         std::to_string(types.elems.size()));
  }

  for (size_t i = 0; i < tokens.elems.size(); ++i) {
    const std::string& text = std::get<std::string>(tokens.elems[i].v);
    const int32_t type = std::get<int32_t>(types.elems[i].v);
    const int32_t id = static_cast<int32_t>(i);
    if (text.empty()) Fail("empty token string at id " + std::to_string(id));
    switch (type) {
      case 1:    // normal
      case 2:    // unknown — still a plain vocab slot for byte-level BPE
      case 5:    // unused — inert PAD-row padding (243x in the APEX GGUF);
                 // plain vocab entry, unreachable via merges
      case 6: {  // byte
        const auto [it, inserted] = tok.vocab_.emplace(text, id);
        if (!inserted) {
          Fail("duplicate token text \"" + text + "\" at ids " +
               std::to_string(it->second) + " and " + std::to_string(id));
        }
        break;
      }
      case 3:  // control -> added token, special (detokenizer may skip)
        // NOTE: real Qwen GGUFs tag FIM/tool tokens control(3) while the HF
        // json has special=false — skip_special_tokens=true detokenization
        // diverges between loaders (faithful to each file).
        tok.added_tokens_.push_back({text, id, /*special=*/true});
        break;
      case 4:  // user-defined -> added token, kept on decode
        tok.added_tokens_.push_back({text, id, /*special=*/false});
        break;
      default:
        Fail("unsupported tokenizer.ggml.token_type " + std::to_string(type) +
             " for token id " + std::to_string(id));
    }
  }

  // Merges are legacy-form "left right" strings.
  const GgufArray& merges =
      KvArray(f, "tokenizer.ggml.merges", kGgufString, "string");
  int32_t rank = 0;
  for (const auto& entry : merges.elems) {
    std::string left;
    std::string right;
    SplitMergeEntry(std::get<std::string>(entry.v), left, right);
    InsertMerge(tok.merge_ranks_, left, right, rank);
    ++rank;
  }

  // bos/eos ids are optional u32 kvs; absent -> -1 (callers fall back to the
  // model config, as with FromHfJson).
  const auto token_id_kv = [&](const char* key) -> int32_t {
    const GgufValue* v = f.FindKv(key);
    if (v == nullptr) return -1;
    if (v->TypeId() != kGgufU32) {
      Fail(std::string("GGUF kv \"") + key + "\" is not u32");
    }
    const uint32_t id = std::get<uint32_t>(v->v);
    if (id >= tokens.elems.size()) {
      Fail(std::string(key) + " = " + std::to_string(id) +
           " out of range (vocab size " + std::to_string(tokens.elems.size()) +
           ")");
    }
    return static_cast<int32_t>(id);
  };
  tok.bos_id_ = token_id_kv("tokenizer.ggml.bos_token_id");
  tok.eos_id_ = token_id_kv("tokenizer.ggml.eos_token_id");

  tok.FinalizeTables();
  return tok;
}

void Tokenizer::EncodePlain(std::string_view text,
                            std::vector<int32_t>& out) const {
  for (const auto& [begin, end] : Pretokenize(text, pattern_)) {
    const std::string mapped =
        MapBytesToUnicode(text.substr(begin, end - begin));
    if (ignore_merges_) {
      const auto it = vocab_.find(mapped);
      if (it != vocab_.end()) {
        out.push_back(it->second);
        continue;
      }
    }
    for (const std::string& sym : BpeSplit(mapped, merge_ranks_)) {
      const auto it = vocab_.find(sym);
      if (it == vocab_.end()) {
        // Cannot happen for byte-level vocabs with a complete byte alphabet.
        Fail("symbol \"" + sym +
             "\" not in vocab (incomplete byte-level alphabet?)");
      }
      out.push_back(it->second);
    }
  }
}

void Tokenizer::EncodePlainSp(std::string_view text, bool at_input_start,
                             std::vector<int32_t>& out) const {
  // 1) Metaspace normalization: replace every ASCII space (0x20) with the
  //    replacement (▁), then prepend one replacement per prepend_scheme when
  //    the result does not already start with it (HF Metaspace::pre_tokenize,
  //    tokenizers 0.22 pre_tokenizers/metaspace.rs). Only 0x20 is replaced;
  //    \t/\n are left for byte-fallback below (matches HF).
  std::string s;
  s.reserve(text.size() + metaspace_replacement_.size());
  for (const char c : text) {
    if (c == ' ') {
      s += metaspace_replacement_;
    } else {
      s.push_back(c);
    }
  }
  const bool prepend =
      prepend_scheme_ == PrependScheme::kAlways ||
      (prepend_scheme_ == PrependScheme::kFirst && at_input_start);
  if (prepend && !s.empty() &&
      s.compare(0, metaspace_replacement_.size(), metaspace_replacement_) != 0) {
    s.insert(0, metaspace_replacement_);
  }
  if (s.empty()) return;

  // 2) Build the initial BPE symbols. HF constructs the Word BEFORE merging:
  //    each character maps to itself when present in the vocab, else (with
  //    byte_fallback) decomposes into its UTF-8 bytes as "<0xNN>" tokens, else
  //    becomes unk. Merges then run over these symbols.
  static const std::string kUnk("\x01\x01unk\x01\x01");  // never a real symbol
  std::vector<std::string> symbols;
  size_t pos = 0;
  while (pos < s.size()) {
    const size_t begin = pos;
    (void)DecodeUtf8(s, pos);
    std::string ch = s.substr(begin, pos - begin);
    if (vocab_.find(ch) != vocab_.end()) {
      symbols.push_back(std::move(ch));
      continue;
    }
    if (byte_fallback_) {
      std::vector<std::string> bytes;
      bool all = true;
      for (const unsigned char b : ch) {
        char buf[7];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned>(b));
        std::string bt(buf);
        if (vocab_.find(bt) == vocab_.end()) {
          all = false;
          break;
        }
        bytes.push_back(std::move(bt));
      }
      if (all) {
        for (auto& b : bytes) symbols.push_back(std::move(b));
        continue;
      }
    }
    if (unk_id_ < 0) {
      Fail("SentencePiece: character \"" + ch +
           "\" has no vocab token, byte-fallback unavailable, and no unk_token");
    }
    symbols.push_back(kUnk);
  }

  BpeMerge(symbols, merge_ranks_);

  // 3) Map merged symbols to ids; fuse consecutive unk ids when fuse_unk.
  int32_t prev = -1;
  for (const std::string& sym : symbols) {
    int32_t id;
    if (sym == kUnk) {
      id = unk_id_;
    } else {
      const auto it = vocab_.find(sym);
      if (it != vocab_.end()) {
        id = it->second;
      } else if (unk_id_ >= 0) {
        id = unk_id_;
      } else {
        Fail("SentencePiece: merged symbol \"" + sym + "\" not in vocab");
      }
    }
    if (fuse_unk_ && id == unk_id_ && prev == unk_id_) continue;
    out.push_back(id);
    prev = id;
  }
}

std::vector<int32_t> Tokenizer::Encode(std::string_view text) const {
  std::vector<int32_t> out;
  size_t pos = 0;
  while (pos < text.size()) {
    // Leftmost-longest added-token match over the raw text (HF added_tokens
    // semantics; special and non-special split identically).
    size_t best_pos = std::string_view::npos;
    const SpecialToken* best = nullptr;
    for (const auto& t : added_tokens_) {
      const size_t found = text.find(t.text, pos);
      if (found == std::string_view::npos) continue;
      if (best == nullptr || found < best_pos ||
          (found == best_pos && t.text.size() > best->text.size())) {
        best_pos = found;
        best = &t;
      }
    }
    // Metaspace prepend_scheme="first" prepends ▁ ONLY to the segment that
    // begins at byte 0 of the whole input (a segment following a special token
    // is not "first"); at_input_start captures exactly that. Inert for the
    // byte-level family.
    const bool at_input_start = pos == 0;
    if (best == nullptr) {
      if (family_ == Family::kSentencePiece) {
        EncodePlainSp(text.substr(pos), at_input_start, out);
      } else {
        EncodePlain(text.substr(pos), out);
      }
      break;
    }
    if (family_ == Family::kSentencePiece) {
      EncodePlainSp(text.substr(pos, best_pos - pos), at_input_start, out);
    } else {
      EncodePlain(text.substr(pos, best_pos - pos), out);
    }
    out.push_back(best->id);
    pos = best_pos + best->text.size();
  }
  return out;
}

std::vector<int32_t> Tokenizer::EncodeWithSpecialTokens(
    std::string_view text) const {
  // Realize the TemplateProcessing "single" template ExtractBosEos parsed: a
  // leading SpecialToken is prepended, a trailing one appended; -1 means the
  // template has no such step. Both are -1 for every Qwen tokenizer (their
  // post_processor is ByteLevel, not TemplateProcessing) AND for every GGUF
  // tokenizer (no HF post_processor exists there at all), so this reduces to
  // Encode() for every model that predates the OPT bring-up.
  std::vector<int32_t> out;
  if (template_bos_ >= 0) out.push_back(template_bos_);
  const std::vector<int32_t> body = Encode(text);
  out.insert(out.end(), body.begin(), body.end());
  if (template_eos_ >= 0) out.push_back(template_eos_);
  return out;
}

std::string Tokenizer::SpDecodeTokens(const std::vector<std::string>& tokens,
                                     size_t begin, size_t end) const {
  // HF tokenizers Sequence decoder for the SentencePiece family:
  //   Replace(▁->" ") -> ByteFallback -> Fuse -> Strip(1 leading space).
  // Replace only affects ▁ (never inside a "<0xNN>" byte token), so it is
  // applied inline when a non-byte token is emitted. ByteFallback accumulates a
  // run of consecutive "<0xNN>" tokens and, on the next non-byte token or at
  // the end, decodes the run: valid UTF-8 -> the decoded text, else one U+FFFD
  // per byte (decoders/byte_fallback.rs). Fuse is the implicit concatenation.
  std::string out;
  std::vector<uint8_t> byte_run;
  const auto flush = [&]() {
    if (byte_run.empty()) return;
    if (IsValidUtf8(byte_run.data(), byte_run.size())) {
      out.append(reinterpret_cast<const char*>(byte_run.data()),
                 byte_run.size());
    } else {
      for (size_t k = 0; k < byte_run.size(); ++k) out += "\xEF\xBF\xBD";
    }
    byte_run.clear();
  };
  for (size_t t = begin; t < end; ++t) {
    const std::string& tk = tokens[t];
    const int byte = ParseByteToken(tk);
    if (byte >= 0) {
      byte_run.push_back(static_cast<uint8_t>(byte));
      continue;
    }
    flush();
    // Replace every ▁ with a space while appending.
    size_t p = 0;
    while (p < tk.size()) {
      const size_t hit = tk.find(metaspace_replacement_, p);
      if (hit == std::string::npos) {
        out.append(tk, p, tk.size() - p);
        break;
      }
      out.append(tk, p, hit - p);
      out.push_back(' ');
      p = hit + metaspace_replacement_.size();
    }
  }
  flush();
  // Strip(content=" ", start=1): remove at most one leading space.
  if (!out.empty() && out.front() == ' ') out.erase(out.begin());
  return out;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids) const {
  if (family_ == Family::kSentencePiece) {
    // ids -> stored token strings (vocab: raw "▁Hello"/"<0xNN>"; added: literal
    // content), then the SentencePiece decoder chain. Includes special tokens
    // literally (full decode), matching the byte-level branch below.
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (const int32_t id : ids) tokens.push_back(TokenText(id));
    return SpDecodeTokens(tokens, 0, tokens.size());
  }
  std::string out;
  for (const int32_t id : ids) {
    const std::string& text = TokenText(id);
    if (is_added_[static_cast<size_t>(id)]) {
      out += text;  // literal content
    } else {
      out += UnmapUnicodeToBytes(text);
    }
  }
  return out;
}

const std::string& Tokenizer::TokenText(int32_t id) const {
  if (!HasToken(id)) Fail("unknown token id " + std::to_string(id));
  return token_text_[static_cast<size_t>(id)];
}

bool Tokenizer::HasToken(int32_t id) const {
  return id >= 0 && static_cast<size_t>(id) < token_text_.size() &&
         !token_text_[static_cast<size_t>(id)].empty();
}

bool Tokenizer::IsSpecial(int32_t id) const {
  return id >= 0 && static_cast<size_t>(id) < is_added_.size() &&
         is_added_[static_cast<size_t>(id)] == 2;
}

}  // namespace vllm::tok
