// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE. FromHfJson loads an HF tokenizer.json; FromGguf loads the same vocab
// family from GGUF tokenizer.ggml.* kvs. Anything that is not the byte-level
// BPE family we implement throws loudly (no silent wrong tokenization).
#include "vllm/tokenizer/tokenizer.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_reader.h"

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
void WalkPreTokenizer(const json& node, std::vector<std::string>& regexes,
                      bool& saw_byte_level) {
  if (!node.is_object()) Fail("pre_tokenizer entry is not an object");
  const std::string type = node.value("type", "");
  if (type == "Sequence") {
    const auto it = node.find("pretokenizers");
    if (it == node.end() || !it->is_array()) {
      Fail("pre_tokenizer Sequence without \"pretokenizers\" array");
    }
    for (const auto& sub : *it) WalkPreTokenizer(sub, regexes, saw_byte_level);
    return;
  }
  if (type == "Split") {
    if (node.value("behavior", "") != "Isolated" ||
        node.value("invert", false)) {
      Fail("Split pre-tokenizer must be behavior=Isolated, invert=false");
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
    // use_regex=true would apply the GPT-2 split regex INSIDE ByteLevel; the
    // models we support carry the split in an explicit Split component.
    if (node.value("use_regex", false)) {
      Fail("ByteLevel pre-tokenizer with use_regex=true unsupported");
    }
    saw_byte_level = true;
    return;
  }
  Fail("unsupported pre_tokenizer component \"" + type + "\"");
}

SplitPattern DetectPattern(const json& doc) {
  const auto it = doc.find("pre_tokenizer");
  if (it == doc.end() || it->is_null()) Fail("missing pre_tokenizer");
  std::vector<std::string> regexes;
  bool saw_byte_level = false;
  WalkPreTokenizer(*it, regexes, saw_byte_level);
  if (!saw_byte_level) Fail("pre_tokenizer has no ByteLevel component");
  if (regexes.size() != 1) {
    Fail("expected exactly one Split pre-tokenizer, found " +
         std::to_string(regexes.size()));
  }
  const std::string& re = regexes[0];
  if (re == kQwen36Regex) return SplitPattern::kQwen2;
  if (re == kClassicQwen2Regex) {
    Fail("classic qwen2 pre-tokenizer differs on combining marks; add "
         "kQwen2Classic before accepting");
  }
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
  Fail("unsupported normalizer \"" + type + "\"");
}

// eos/bos are only extracted when trivially available: a TemplateProcessing
// post_processor whose "single" template starts/ends with a SpecialToken.
// Otherwise they stay -1 and callers must use the model config's ids (e.g.
// Qwen's ByteLevel post_processor carries neither).
void ExtractBosEos(const json& doc, int32_t& bos, int32_t& eos) {
  const auto pp = doc.find("post_processor");
  if (pp == doc.end() || !pp->is_object()) return;
  if (pp->value("type", "") != "TemplateProcessing") return;
  const auto single = pp->find("single");
  const auto specials = pp->find("special_tokens");
  if (single == pp->end() || !single->is_array() || single->empty() ||
      specials == pp->end() || !specials->is_object()) {
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
    is_added_[static_cast<size_t>(t.id)] = 1;
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
  tok.pattern_ = DetectPattern(doc);

  const auto model_it = doc.find("model");
  if (model_it == doc.end() || !model_it->is_object()) Fail("missing model");
  const json& model = *model_it;
  const std::string model_type = model.value("type", "");
  if (model_type != "BPE") {
    Fail("unsupported model type \"" + model_type +
         "\" (only byte-level BPE)");
  }
  // These options change BPE segmentation semantics; the byte-level family
  // never uses them.
  for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const auto it = model.find(key);
    if (it != model.end() && !it->is_null() &&
        !(it->is_string() && it->get<std::string>().empty())) {
      Fail(std::string("unsupported BPE option \"") + key + "\"");
    }
  }
  tok.ignore_merges_ = model.value("ignore_merges", false);

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
      for (const char* key : {"lstrip", "rstrip", "single_word"}) {
        if (entry.value(key, false)) {
          Fail("added token \"" + t.text + "\" uses unsupported option \"" +
               key + "\"");
        }
      }
      tok.added_tokens_.push_back(std::move(t));
    }
  }

  ExtractBosEos(doc, tok.bos_id_, tok.eos_id_);
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

  // Pre-tokenizer, by llama.cpp pre name. NOTE: llama.cpp's "qwen2" pre
  // applies its qwen2 splitting to Qwen2/2.5/3/3.5/3.6 GGUFs alike — its pre
  // names do not distinguish the \p{M}-aware Qwen3.6 regex variant (the APEX
  // GGUFs carry arch qwen35moe with pre "qwen2"), so "qwen2" maps onto our
  // kQwen2, which implements the Qwen3.6 \p{M}-aware regex.
  const std::string& pre = KvString(f, "tokenizer.ggml.pre");
  if (pre == "qwen2") {
    tok.pattern_ = SplitPattern::kQwen2;
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
      case 6: {  // byte
        const auto [it, inserted] = tok.vocab_.emplace(text, id);
        if (!inserted) {
          Fail("duplicate token text \"" + text + "\" at ids " +
               std::to_string(it->second) + " and " + std::to_string(id));
        }
        break;
      }
      case 3:  // control -> added token, special (detokenizer may skip)
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
    if (best == nullptr) {
      EncodePlain(text.substr(pos), out);
      break;
    }
    EncodePlain(text.substr(pos, best_pos - pos), out);
    out.push_back(best->id);
    pos = best_pos + best->text.size();
  }
  return out;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids) const {
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
  if (id < 0 || static_cast<size_t>(id) >= token_text_.size() ||
      token_text_[static_cast<size_t>(id)].empty()) {
    Fail("unknown token id " + std::to_string(id));
  }
  return token_text_[static_cast<size_t>(id)];
}

}  // namespace vllm::tok
