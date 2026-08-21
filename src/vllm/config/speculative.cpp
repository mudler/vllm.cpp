// SPEC-MTP I5d: `--speculative-config` JSON parsing. Mirrors the CLI subset of
// vllm/engine/arg_utils.py + vllm/config/speculative.py that the entrypoint
// needs; the full method auto-detection / draft-model resolution stays in the
// loader (see SpeculativeConfig::ResolveMtp).
#include "vllm/config/speculative.h"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {
namespace {

// #1160 — key ADMISSION for `--speculative-config`.
//
// Upstream deserializes this JSON into the `SpeculativeConfig` dataclass, which
// carries `extra="forbid"` through its `@config` decorator
// (vllm/config/speculative.py:81-83 @ the pinned oracle `555967922`), so vLLM
// refuses a name it does not declare. This hand-written parser read five keys
// and dropped everything else, which turned `"draft_sample_method":
// "probabilistic"` into a silent GREEDY run and a misspelled
// `"num_speculatve_tokens"` into the resolved default. A dropped key does not
// merely do nothing: it changes what configuration a measurement was taken
// under, which is the one thing a gate cannot recover afterwards.
//
// The three classes below restore that refusal. The split matters because a key
// vLLM genuinely declares deserves a different message from a typo: telling a
// user that `quantization` is "unknown" sends them hunting for a spelling
// mistake that is not there.

// Class 1: parsed and used below.
constexpr std::array<std::string_view, 5> kHonouredKeys = {
    "method", "num_speculative_tokens", "model", "prompt_lookup_min",
    "prompt_lookup_max",
};

// Class 3: declared by `SpeculativeConfig` at the pin
// (vllm/config/speculative.py:85-283 @ `555967922`) and NOT implemented here.
// Mirrored field-for-field so a pin bump that adds a field is reconciled here
// rather than silently widening what this parser drops. The internal
// `SkipValidation` fields (`target_model_config`, `draft_model_config`,
// `target_parallel_config`, `draft_parallel_config`, `draft_load_config`) are
// listed too: upstream builds them rather than reading them from this JSON, so a
// user who names one is asking for something no spelling of this flag delivers.
constexpr std::array<std::string_view, 26> kUpstreamUnimplementedKeys = {
    "enforce_eager",                          // :85
    "draft_tensor_parallel_size",             // :102
    "tensor_parallel_size",                   // :105
    "quantization",                           // :110
    "moe_backend",                            // :114
    "attention_backend",                      // :119
    "kv_cache_dtype",                         // :123
    "max_model_len",                          // :126
    "revision",                               // :129
    "code_revision",                          // :133
    "disable_padded_drafter_batch",           // :139
    "use_local_argmax_reduction",             // :144
    "use_heterogeneous_vocab",                // :150
    "parallel_drafting",                      // :165
    "target_model_config",                    // :172
    "target_parallel_config",                 // :174
    "num_speculative_tokens_per_batch_size",  // :178
    "draft_model_config",                     // :186
    "draft_parallel_config",                  // :188
    "suffix_decoding_max_tree_depth",         // :192
    "suffix_decoding_max_cached_requests",    // :196
    "suffix_decoding_max_spec_factor",        // :202
    "suffix_decoding_min_token_prob",         // :207
    "draft_load_config",                      // :212
    "synthetic_acceptance_rates",             // :224
    "synthetic_acceptance_length",            // :232
};

// Class 2: an upstream key whose value space we implement exactly ONE point of.
// Named as a set (#1598) rather than spelled inline, because the chain's entry
// admission has to make the SAME three-way split and a second hand-written
// comparison is how the split was lost there in the first place. `CheckEntryKeys`
// reads this array; so does the top-level admission loop.
//
// Both are ENGINE-WIDE. `draft_sample_method` describes how a draft is sampled
// and `rejection_sample_method` describes the verify, and the verify is one
// verify however many speculators propose into it — which is why the top level
// honours them BESIDE a chain, and why an ENTRY is the one place they cannot be
// spelled.
constexpr std::array<std::string_view, 2> kEngineWideValueGatedKeys = {
    "draft_sample_method",     // :77 alias, :283 field @ 555967922
    "rejection_sample_method"  // :78 alias, :216 field @ 555967922
};

// The tail every refusal carries, so the message closes the user's search
// instead of only ending it. Class 2 is spelled with its one accepted value
// because "supported" would otherwise overstate what those keys accept.
constexpr std::string_view kSupportedKeys =
    "supported keys: method, num_speculative_tokens, model, prompt_lookup_min, "
    "prompt_lookup_max, draft_sample_method (only \"greedy\"), "
    "rejection_sample_method (only \"standard\"), and this engine's own "
    "vllm_cpp.drafter_chain (SPEC-DRAFTER-CHAIN)";

bool Contains(const std::array<std::string_view, 5>& set, const std::string& key) {
  for (std::string_view k : set) {
    if (k == key) return true;
  }
  return false;
}

bool Contains(const std::array<std::string_view, 2>& set, const std::string& key) {
  for (std::string_view k : set) {
    if (k == key) return true;
  }
  return false;
}

bool Contains(const std::array<std::string_view, 26>& set, const std::string& key) {
  for (std::string_view k : set) {
    if (k == key) return true;
  }
  return false;
}

// Class 2: an upstream key whose value space we implement exactly one point of.
// The accepted value is upstream's own DEFAULT in both cases, so a vLLM config
// that spells the default explicitly keeps running, and only a request for
// behavior we do not have is refused. `owed_by` names the missing part, which is
// what AGENTS.md requires of a refused arm.
void CheckValueGatedKey(const nlohmann::json& doc, const char* key,
                        const char* accepted, const std::string& reason) {
  if (!doc.contains(key)) return;
  const nlohmann::json& v = doc.at(key);
  if (v.is_string() && v.get<std::string>() == accepted) return;
  const std::string got = v.is_string() ? ("\"" + v.get<std::string>() + "\"") : v.dump();
  throw std::invalid_argument(std::string("speculative-config: ") + key + " " + got +
                              " is not implemented. " + reason);
}


// SPEC-DRAFTER-CHAIN W1 (#1522) — the `vllm_cpp.drafter_chain` extension.
//
// WHY THE FIELD IS NESTED UNDER `vllm_cpp` (.agents/specs/drafter-chain.md D6).
// The developer's constraint is that the new name must not collide with a key
// vLLM's `SpeculativeConfig` declares "now or plausibly". A bare `drafter_chain`
// satisfies only the first half: nothing stops upstream from later choosing that
// spelling, and if it did, the identical document would mean two different
// things in the two engines with no error anywhere. One `vllm_cpp` object
// reduces the collision surface to exactly ONE name, and a collision on THAT
// name would be loud rather than silent. It is also the landed convention here:
// `--offload-config` carries this engine's non-vLLM residency tier under exactly
// the same key (src/vllm/config/weight_residency.cpp:451-478), so a reader who
// has met one flag already knows what `vllm_cpp` means on the other.
constexpr std::string_view kExtKey = "vllm_cpp";
constexpr std::string_view kChainKey = "drafter_chain";
constexpr std::string_view kChainPath = "vllm_cpp.drafter_chain";

// The methods a chain entry may name. The four this engine implements AND the
// four the row scopes (.agents/specs/drafter-chain.md `## Scope`). `draft_model`
// is deliberately absent and is refused with that fact said out loud below: it
// IS an implemented top-level method, so a message calling it unknown would send
// the user looking for a typo that is not there — the same distinction #1160
// draws for keys, applied to methods.
constexpr std::array<std::string_view, 4> kChainMethods = {"mtp", "dflash",
                                                           "dspark", "ngram"};

constexpr std::string_view kChainMethodList =
    "\"mtp\", \"dflash\", \"dspark\", \"ngram\"";

std::string EntryPath(std::size_t index) {
  return std::string(kChainPath) + "[" + std::to_string(index) + "]";
}

// The chain's own entry-key admission. It applies the SAME three classes #1160
// established for the top-level document, and it applies class 2 DIFFERENTLY,
// which is the whole of the difference between the two admissions:
//
//   class 1, honoured        — the five per-speculator keys, read below
//   class 2, value-gated     — engine-wide, so honoured at the TOP LEVEL only
//                              and refused here BY NAME with that said out loud
//   class 3, unimplemented   — refused as a vLLM field we do not have
//   anything else            — refused as unknown, with the accepted list
//
// An entry is not a weaker place to spell a config than the field it replaces,
// so a name is honoured, refused, or refused — never dropped.
//
// #1598: class 2 was MISSING here, and the two keys in it fell through to the
// unknown branch. `draft_sample_method` is a real `SpeculativeConfig` field
// (`speculative.py:77,283` @ 555967922) that this engine HONOURS at the top
// level of this very document, and it was answered with a message
// byte-for-byte of the shape a misspelling gets — the exact failure #1160 split
// the classes to prevent, arriving from the other direction. Neither of the two
// existing refusals is right for it: "unknown" sends the user hunting for a typo
// that is not there, and "this engine does not implement it" is simply false.
// So class 2 gets its own message, and that message says WHERE the key goes.
void CheckEntryKeys(const nlohmann::json& entry, const std::string& path) {
  for (auto it = entry.begin(); it != entry.end(); ++it) {
    const std::string& key = it.key();
    if (Contains(kHonouredKeys, key)) continue;
    if (Contains(kEngineWideValueGatedKeys, key)) {
      throw std::invalid_argument(
          "speculative-config: \"" + path + "." + key +
          "\" is ENGINE-WIDE, not per drafter, so it cannot sit in a chain "
          "entry. This engine honours it, at the TOP LEVEL of this same "
          "document, beside \"vllm_cpp.drafter_chain\" — move it there. "
          "\"draft_sample_method\" describes how a draft is sampled and "
          "\"rejection_sample_method\" describes the verify, and the verify is "
          "ONE verify however many speculators propose into it "
          "(vllm/config/speculative.py:77,283 and :78,216 @ 555967922). A chain "
          "entry accepts: method, num_speculative_tokens, model, "
          "prompt_lookup_min, prompt_lookup_max");
    }
    if (Contains(kUpstreamUnimplementedKeys, key)) {
      throw std::invalid_argument(
          "speculative-config: \"" + path + "." + key +
          "\" is a vLLM SpeculativeConfig field "
          "(vllm/config/speculative.py:85-283 @ 555967922) that this engine does "
          "not implement at this pin, so it is refused rather than dropped. A "
          "chain entry accepts: method, num_speculative_tokens, model, "
          "prompt_lookup_min, prompt_lookup_max");
    }
    throw std::invalid_argument(
        "speculative-config: unknown key \"" + path + "." + key +
        "\" (a chain entry accepts: method, num_speculative_tokens, model, "
        "prompt_lookup_min, prompt_lookup_max)");
  }
}

// A positive integer, read the way the top-level `num_speculative_tokens` and
// `prompt_lookup_*` are read, including the "explicit null means absent" rule
// the landed contract already has for both.
std::optional<int> EntryInt(const nlohmann::json& entry, const char* key,
                            const std::string& path, int minimum) {
  if (!entry.contains(key) || entry.at(key).is_null()) return std::nullopt;
  const nlohmann::json& v = entry.at(key);
  if (!v.is_number_integer() || v.get<int>() < minimum) {
    throw std::invalid_argument("speculative-config: \"" + path + "." + key +
                                "\" must be an integer >= " +
                                std::to_string(minimum));
  }
  return v.get<int>();
}

// Parse and validate one entry. Every refusal names the entry BY POSITION and
// the offending value BY NAME, because a chain has several entries and "one of
// them is wrong" closes nothing.
SpeculativeChainEntry ParseChainEntry(const nlohmann::json& entry,
                                      std::size_t index) {
  const std::string path = EntryPath(index);
  if (!entry.is_object()) {
    throw std::invalid_argument(
        "speculative-config: \"" + path +
        "\" must be a JSON object naming one speculator, e.g. "
        "{\"method\":\"dflash\",\"model\":\"z-lab/...\","
        "\"num_speculative_tokens\":16}");
  }
  if (!entry.contains("method") || !entry.at("method").is_string()) {
    throw std::invalid_argument("speculative-config: \"" + path +
                                "\" requires a string \"method\" (one of " +
                                std::string(kChainMethodList) + ")");
  }
  SpeculativeChainEntry out;
  out.method = entry.at("method").get<std::string>();
  bool known = false;
  for (std::string_view m : kChainMethods) {
    if (m == out.method) known = true;
  }
  if (!known) {
    // `draft_model` is the one refusal that must not read as "no such method":
    // it IS implemented as a top-level method, and only its CHAIN arm is owed.
    const std::string owed =
        out.method == "draft_model"
            ? " \"draft_model\" IS an accepted top-level method here, but not "
              "yet a chain entry: the chain arm is owed by row "
              "SPEC-DRAFTER-CHAIN (.agents/specs/drafter-chain.md), issue "
              "https://github.com/mudler/vllm.cpp/issues/1522."
            : " A method this engine does not implement is refused by name "
              "rather than skipped, because a silently shortened chain is a "
              "different feature running under your document. Owed by row "
              "SPEC-DRAFTER-CHAIN (.agents/specs/drafter-chain.md).";
    throw std::invalid_argument("speculative-config: \"" + path +
                                ".method\" \"" + out.method +
                                "\" is not a chain entry method; accepted: " +
                                std::string(kChainMethodList) + "." + owed);
  }
  CheckEntryKeys(entry, path);
  out.num_speculative_tokens = EntryInt(entry, "num_speculative_tokens", path, 1);
  out.prompt_lookup_min = EntryInt(entry, "prompt_lookup_min", path, 1);
  out.prompt_lookup_max = EntryInt(entry, "prompt_lookup_max", path, 1);
  if (entry.contains("model") && entry.at("model").is_string()) {
    out.draft_model_path = entry.at("model").get<std::string>();
  }
  // The SAME required-key rules the top-level document applies to the same
  // method. "dflash" and "dspark" each name a SEPARATE draft checkpoint
  // (speculative.py:875), and "ngram" has no head depth to default a k from
  // (speculative.py:1224-1234). "mtp" requires neither: its draft lives inside
  // the target and its k defaults to the head depth.
  if ((out.method == "dflash" || out.method == "dspark") &&
      !out.draft_model_path.has_value()) {
    throw std::invalid_argument(
        "speculative-config: \"" + path + "\" method \"" + out.method +
        "\" requires a \"model\" key naming the draft checkpoint (path or HF "
        "repo id), exactly as the top-level method does");
  }
  if (out.method == "ngram" && !out.num_speculative_tokens.has_value()) {
    throw std::invalid_argument(
        "speculative-config: \"" + path +
        "\" method \"ngram\" requires \"num_speculative_tokens\" "
        "(speculative.py:1224-1234), exactly as the top-level method does");
  }
  return out;
}

// Parse and validate the whole extension object, and return the chain.
std::vector<SpeculativeChainEntry> ParseDrafterChain(const nlohmann::json& doc) {
  const nlohmann::json& ext = doc.at(std::string(kExtKey));
  // PRESENCE is judged, including an explicit `null` — the #1160 polarity. A
  // null on a name we DO implement must not read as "absent", or the user who
  // deliberately spelled it gets told a string "method" is required and goes
  // looking for a key they never wanted.
  if (!ext.is_object()) {
    throw std::invalid_argument(
        "speculative-config: \"vllm_cpp\" must be a JSON object carrying this "
        "engine's non-vLLM extensions; its only key at this pin is "
        "\"drafter_chain\", a JSON array of speculator entries "
        "(SPEC-DRAFTER-CHAIN, .agents/specs/drafter-chain.md)");
  }
  for (auto it = ext.begin(); it != ext.end(); ++it) {
    if (it.key() != std::string(kChainKey)) {
      throw std::invalid_argument(
          "speculative-config: unknown key \"vllm_cpp." + it.key() +
          "\" (the \"vllm_cpp\" extension object accepts only "
          "\"drafter_chain\")");
    }
  }
  if (!ext.contains(std::string(kChainKey))) {
    // An extension object that configures nothing on the extension surface. It
    // cannot mean "no chain", because omitting `vllm_cpp` already means that;
    // accepting it would run a one-drafter engine under a document whose author
    // believes it says otherwise.
    throw std::invalid_argument(
        "speculative-config: \"vllm_cpp\" carries no \"drafter_chain\". Omit "
        "\"vllm_cpp\" entirely to run without a chain "
        "(SPEC-DRAFTER-CHAIN, .agents/specs/drafter-chain.md)");
  }
  const nlohmann::json& arr = ext.at(std::string(kChainKey));
  if (!arr.is_array()) {
    throw std::invalid_argument(
        "speculative-config: \"vllm_cpp.drafter_chain\" must be a JSON array of "
        "speculator entries, in preference order — a list of OBJECTS and not a "
        "list of names, because each entry carries its own \"model\" and its own "
        "\"num_speculative_tokens\"");
  }
  if (arr.empty()) {
    throw std::invalid_argument(
        "speculative-config: \"vllm_cpp.drafter_chain\" is empty; a chain names "
        "at least one speculator (" + std::string(kChainMethodList) +
        "). Omit \"vllm_cpp\" entirely to run without a chain");
  }
  std::vector<SpeculativeChainEntry> chain;
  chain.reserve(arr.size());
  for (std::size_t i = 0; i < arr.size(); ++i) {
    SpeculativeChainEntry entry = ParseChainEntry(arr[i], i);
    // A method named twice makes the per-drafter attribution the row's gates
    // depend on ambiguous: W2 keys the counters and the per-sequence winner on
    // the method name, so two entries with the same one cannot be told apart in
    // the numbers that decide whether a chain helps at all. Refusing now is
    // reversible; shipping ambiguous counters is not
    // (.agents/specs/drafter-chain.md D11).
    for (std::size_t j = 0; j < chain.size(); ++j) {
      if (chain[j].method == entry.method) {
        throw std::invalid_argument(
            "speculative-config: \"vllm_cpp.drafter_chain\" names method \"" +
            entry.method + "\" twice, at entries " + std::to_string(j) +
            " and " + std::to_string(i) +
            ". A chain is a preference ORDER over DISTINCT methods at this pin, "
            "because per-drafter attribution keys on the method name. Two "
            "checkpoints of one method are owed by row SPEC-DRAFTER-CHAIN "
            "(.agents/specs/drafter-chain.md)");
      }
    }
    chain.push_back(std::move(entry));
  }
  return chain;
}

}  // namespace

SpeculativeConfig ParseSpeculativeConfigJson(const std::string& json_text) {
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("speculative-config: invalid JSON: ") +
                                e.what());
  }
  if (!doc.is_object()) {
    throw std::invalid_argument("speculative-config: expected a JSON object");
  }

  SpeculativeConfig cfg;
  // SPEC-DRAFTER-CHAIN W1 (#1522): presence of this engine's extension object,
  // judged BEFORE the method requirement because the two are mutually exclusive
  // and the method requirement would otherwise fire first with the wrong advice.
  // Presence only — the object itself is validated further down, after the
  // top-level key admission, so a typo'd top-level key still errors first and
  // the landed error ORDERING is unchanged for every chain-free document.
  const bool has_chain = doc.contains(std::string(kExtKey));
  // method (vllm/config/speculative.py `method`). Required for this CLI path;
  // upstream can auto-detect it from the draft checkpoint, but the entrypoint
  // that reaches here always passes an explicit method.
  //
  // D7: `method` names ONE speculator and a chain names an ordered list. No rule
  // here makes a top-level `method` the head of the chain, its tail, or a
  // default, and inventing one would silently reinterpret a document. So exactly
  // one of the two must be given, and with a chain present `method` is not
  // merely optional — it is refused.
  if (has_chain) {
    if (doc.contains("method")) {
      const nlohmann::json& m = doc.at("method");
      throw std::invalid_argument(
          "speculative-config: \"method\" and \"vllm_cpp.drafter_chain\" are "
          "mutually exclusive. \"method\" names ONE speculator and the chain "
          "names an ordered list of them, and nothing here makes a top-level "
          "\"method\" the head, the tail or a default of the chain. Name every "
          "speculator as a chain entry instead (got method " +
          (m.is_string() ? "\"" + m.get<std::string>() + "\"" : m.dump()) + ")");
    }
  } else {
    if (!doc.contains("method") || !doc.at("method").is_string()) {
      throw std::invalid_argument(
          "speculative-config: a string \"method\" is required");
    }
    cfg.method = doc.at("method").get<std::string>();
  }
  // SPEC-DFLASH D4: accept "dflash" alongside "mtp". SPEC-NGRAM (ROAD-V1-D3):
  // accept "ngram" — the draft-free proposer. SPEC-DRAFT-MODEL: accept
  // "draft_model" — the classic model-agnostic SEPARATE draft model
  // (speculative.py:684 default; uses_draft_model :1195; runner
  // gpu_model_runner.py:604-609). "mtp"/"dflash" are draft-hidden-state methods;
  // the loader resolves the concrete draft (MTP head vs the z-lab DFlash
  // checkpoint) and the block-derived k from the model config. "ngram" needs no
  // draft model; "draft_model" needs a separate `model` checkpoint (below).
  // Any other method is still rejected at this pin.
  // SPEC-DSPARK W1: accept "dspark" — the semi-autoregressive BLOCK drafter that
  // extends DFlash with a Markov logit-bias head (speculative.py:62,310;
  // dspark/speculator.py:37). Like DFlash it names a SEPARATE draft checkpoint,
  // so it requires the `model` key below.
  if (!has_chain && cfg.method != "mtp" && cfg.method != "dflash" &&
      cfg.method != "ngram" && cfg.method != "draft_model" &&
      cfg.method != "dspark") {
    throw std::invalid_argument(
        "speculative-config: only methods \"mtp\", \"dflash\", \"dspark\", "
        "\"ngram\" and \"draft_model\" are supported at this pin (got \"" +
        cfg.method + "\")");
  }

  // #1160: key admission, AFTER the method check so an unsupported method is
  // still the error a user sees first (it is the field that selects everything
  // else), and BEFORE every other read so no key can be dropped on the way past.
  // Presence is what is judged for classes 1 and 3, including an explicit
  // `null`: a null on a key we do not implement still names a capability we do
  // not have, and answering it with silence is the defect this restores.
  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& key = it.key();
    if (Contains(kHonouredKeys, key)) continue;
    // Class 2 (#1598): honoured here, value-gated below by `CheckValueGatedKey`.
    if (Contains(kEngineWideValueGatedKeys, key)) continue;
    // SPEC-DRAFTER-CHAIN W1: the ONE name this engine adds to vLLM's document.
    // It is admitted here DELIBERATELY, which is the mechanism that kept the
    // field inert until this wave: every other new name still falls through to
    // one of the two refusals below.
    if (key == std::string(kExtKey)) continue;
    if (Contains(kUpstreamUnimplementedKeys, key)) {
      throw std::invalid_argument(
          "speculative-config: \"" + key +
          "\" is a vLLM SpeculativeConfig field (vllm/config/speculative.py:85-283 "
          "@ 555967922) that this engine does not implement at this pin, so it is "
          "refused rather than dropped (" + std::string(kSupportedKeys) + ")");
    }
    throw std::invalid_argument(
        "speculative-config: unknown key \"" + key +
        "\" (vLLM's SpeculativeConfig declares no such field at the pinned oracle "
        "555967922; " + std::string(kSupportedKeys) + ")");
  }
  // draft_sample_method (speculative.py:77,283). The draft is GREEDY by
  // construction here — `speculator.h` takes the argmax and the verify accepts
  // iff equal — so "probabilistic" would change the acceptance rule, and with it
  // whether the run is adjudicable by the token-exact greedy gate at all.
  CheckValueGatedKey(
      doc, "draft_sample_method", "greedy",
      "This engine drafts greedy only (include/vllm/v1/worker/gpu/spec_decode/"
      "dspark/speculator.h) and verifies accept-iff-equal (include/vllm/v1/"
      "spec_decode/rejection_sampler.h), so \"greedy\", which is upstream's own "
      "default, is the only accepted value at this pin. Probabilistic draft "
      "sampling is owed by row SPEC-ACCEPT-VARIANTS (.agents/engine-matrix.md), "
      "vllm/config/speculative.py:77,283 @ 555967922.");
  // rejection_sample_method (speculative.py:78,216). "standard" is upstream's
  // default and the semantics the landed verify implements. "synthetic" and
  // "block" are the two branches at
  // vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:82-91, both listed as
  // deferred in rejection_sampler.h.
  CheckValueGatedKey(
      doc, "rejection_sample_method", "standard",
      "This engine implements upstream's default \"standard\" acceptance only, "
      "as accept-iff-equal under greedy decode (include/vllm/v1/spec_decode/"
      "rejection_sampler.h). The other two are owed by row SPEC-ACCEPT-VARIANTS "
      "(.agents/engine-matrix.md), vllm/config/speculative.py:78,216 and "
      "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:82-91 @ 555967922.");

  // SPEC-DRAFTER-CHAIN W1 (#1522): the chain branch. Placed AFTER the two
  // value-gated keys above, which are ENGINE-WIDE rather than per drafter —
  // `draft_sample_method` describes how a draft is sampled and
  // `rejection_sample_method` describes the verify, and the verify is one verify
  // however many speculators propose into it. So those two may sit beside a
  // chain and keep their gate, while every key below is per speculator and
  // cannot.
  if (has_chain) {
    cfg.drafter_chain = ParseDrafterChain(doc);
    // D7 again, one key at a time. `model` beside a two-entry chain names a
    // checkpoint with no entry to belong to, and applying it to the first entry,
    // to all of them, or to none are three different engines. There is no
    // reading that is safe to guess, so the document is refused and the message
    // says which key to move.
    for (const char* key : {"model", "num_speculative_tokens",
                            "prompt_lookup_min", "prompt_lookup_max"}) {
      if (!doc.contains(key)) continue;
      throw std::invalid_argument(
          "speculative-config: \"" + std::string(key) +
          "\" cannot be given beside \"vllm_cpp.drafter_chain\": it configures "
          "ONE speculator and the chain has " +
          std::to_string(cfg.drafter_chain.size()) +
          ", so there is no entry it belongs to. Move it into the chain entry "
          "that needs it");
    }
    // `cfg.method` stays EMPTY. Promoting an entry into it would make a chain
    // document parse into a config a one-drafter engine runs without complaint,
    // which is the failure this wave exists to prevent; the loader refuses the
    // chain by name instead (LoadedEngine::ResolveSpecConfig).
    return cfg;
  }

  // num_speculative_tokens (k). Optional; the loader defaults it to n_predict
  // (mtp_num_hidden_layers) via ResolveMtp when absent (speculative.py:865-875).
  if (doc.contains("num_speculative_tokens") &&
      !doc.at("num_speculative_tokens").is_null()) {
    const nlohmann::json& k = doc.at("num_speculative_tokens");
    if (!k.is_number_integer() || k.get<int>() <= 0) {
      throw std::invalid_argument(
          "speculative-config: num_speculative_tokens must be a positive integer");
    }
    cfg.num_speculative_tokens = k.get<int>();
  }
  // SPEC-NGRAM (ROAD-V1-D3): the n-gram proposer window (speculative.py:157-161).
  // Optional; ResolveNgram defaults both to 5 when absent. Ignored for mtp/dflash.
  auto parse_lookup = [&](const char* key) -> std::optional<int> {
    if (doc.contains(key) && !doc.at(key).is_null()) {
      const nlohmann::json& v = doc.at(key);
      if (!v.is_number_integer() || v.get<int>() < 1) {
        throw std::invalid_argument(std::string("speculative-config: ") + key +
                                    " must be an integer >= 1");
      }
      return v.get<int>();
    }
    return std::nullopt;
  };
  cfg.prompt_lookup_min = parse_lookup("prompt_lookup_min");
  cfg.prompt_lookup_max = parse_lookup("prompt_lookup_max");
  if (cfg.method == "ngram" && !cfg.num_speculative_tokens.has_value()) {
    throw std::invalid_argument(
        "speculative-config: method \"ngram\" requires \"num_speculative_tokens\" "
        "(speculative.py:1224-1234)");
  }

  // SPEC-DFLASH D5: the DFlash draft is a SEPARATE checkpoint (unlike MTP's
  // in-target mtp.* tensors), so `--speculative-config` carries a `model` key
  // (vllm/config/speculative.py `model`) pointing at the z-lab draft. Required
  // for dflash; ignored (and not required) for mtp.
  if (doc.contains("model") && doc.at("model").is_string()) {
    cfg.draft_model_path = doc.at("model").get<std::string>();
  }
  if (cfg.method == "dflash" && !cfg.draft_model_path.has_value()) {
    throw std::invalid_argument(
        "speculative-config: method \"dflash\" requires a \"model\" key naming "
        "the DFlash draft checkpoint (path or HF repo id)");
  }
  // SPEC-DSPARK W1: the in-scope DSpark drafts (Qwen3 + Gemma4 families) are
  // SEPARATE checkpoints exactly like DFlash — e.g.
  // deepseek-ai/dspark_qwen3_4b_block7 (speculative.py:875). The DeepSeek-V4
  // variant that ships its draft inside the target checkpoint
  // (speculative.py:706-709) is out of scope for this row (hardware-blocked), so
  // the key is required here.
  if (cfg.method == "dspark" && !cfg.draft_model_path.has_value()) {
    throw std::invalid_argument(
        "speculative-config: method \"dspark\" requires a \"model\" key naming "
        "the DSpark draft checkpoint (path or HF repo id)");
  }
  // SPEC-DRAFT-MODEL: the generic draft model is a SEPARATE standalone
  // checkpoint, so it likewise requires the `model` key (speculative.py:692-701;
  // a separate `draft_model_config` is built from it). num_speculative_tokens (k)
  // is also required for a draft model — there is no head depth to default from
  // (speculative.py has no n_predict for "draft_model").
  if (cfg.method == "draft_model") {
    if (!cfg.draft_model_path.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"draft_model\" requires a \"model\" key "
          "naming the separate draft checkpoint (path or HF repo id)");
    }
    if (!cfg.num_speculative_tokens.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"draft_model\" requires "
          "\"num_speculative_tokens\"");
    }
  }
  // n_predict stays 0 here: the model loader resolves it from the checkpoint's
  // mtp_num_hidden_layers and re-runs ResolveMtp with this user k.
  return cfg;
}

}  // namespace vllm
