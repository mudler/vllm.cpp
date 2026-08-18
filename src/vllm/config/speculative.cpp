// SPEC-MTP I5d: `--speculative-config` JSON parsing. Mirrors the CLI subset of
// vllm/engine/arg_utils.py + vllm/config/speculative.py that the entrypoint
// needs; the full method auto-detection / draft-model resolution stays in the
// loader (see SpeculativeConfig::ResolveMtp).
#include "vllm/config/speculative.h"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

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

// The tail every refusal carries, so the message closes the user's search
// instead of only ending it. Class 2 is spelled with its one accepted value
// because "supported" would otherwise overstate what those keys accept.
constexpr std::string_view kSupportedKeys =
    "supported keys: method, num_speculative_tokens, model, prompt_lookup_min, "
    "prompt_lookup_max, draft_sample_method (only \"greedy\"), "
    "rejection_sample_method (only \"standard\")";

bool Contains(const std::array<std::string_view, 5>& set, const std::string& key) {
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
  // method (vllm/config/speculative.py `method`). Required for this CLI path;
  // upstream can auto-detect it from the draft checkpoint, but the entrypoint
  // that reaches here always passes an explicit method.
  if (!doc.contains("method") || !doc.at("method").is_string()) {
    throw std::invalid_argument(
        "speculative-config: a string \"method\" is required");
  }
  cfg.method = doc.at("method").get<std::string>();
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
  if (cfg.method != "mtp" && cfg.method != "dflash" && cfg.method != "ngram" &&
      cfg.method != "draft_model" && cfg.method != "dspark") {
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
    if (key == "draft_sample_method" || key == "rejection_sample_method") continue;
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
