// GLM-5.3-Flash config resolution + validation (W1 of MODEL-MM-GLM53-FLASH,
// #2067).
//
// The resolve IS the validation: every refusal below mirrors one in upstream
// `Glm5NextTextConfig.__post_init__` / `validate_architecture` at transformers
// **v5.16.1**, the only revision of any admissible oracle that implements
// `glm5_next` at all. vLLM implements it at no revision, so there is nothing to
// mirror on this surface and transformers is the only source; see
// `.agents/specs/glm5-next-flash.md` `## Oracles`.
//
// ONE PARSER, TWO SOURCES. This function is reached from a `config.json`
// (through `ModelFactory::parse_config`) and from a converter-written GGUF
// (through `Glm5NextHfConfigFromGguf`, which synthesizes an HF-shaped
// `text_config`/`vision_config` under the same key spellings). That is
// deliberate: a second, parallel notion of what a `glm5_next` config is would
// be a surface that can drift, and this model has no token gate anywhere on
// this fleet that would catch the drift.
#include "vllm/model_executor/models/glm5_next.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vllm {
namespace {

constexpr const char* kSlug = "glm5_next";

[[noreturn]] void Refuse(const std::string& what) {
  throw std::runtime_error(std::string(kSlug) + ": " + what +
                           " See .agents/specs/glm5-next-flash.md and "
                           "issue #2067.");
}

// The model's own sub-object. Upstream nests everything except the wrapper's
// `architectures`/`model_type`, the six placeholder ids, `quantization_config`
// and the vision block under `text_config`; a flat config is accepted by
// falling back to the top level, which is upstream's own BC path
// (`Glm5NextConfig.__post_init__` forwards the wrapper kwargs to
// `Glm5NextTextConfig` when `text_config` is absent).
//
// This has to resolve to the SAME object `HfConfig`'s own `ResolveTextConfig`
// picked, or one parse answers "what is the text config?" twice and the typed
// fields describe a different object from the untyped ones.
const nlohmann::json& TextOf(const nlohmann::json& raw) {
  auto it = raw.find("text_config");
  if (it != raw.end() && it->is_object()) return *it;
  it = raw.find("llm_config");
  if (it != raw.end() && it->is_object()) return *it;
  return raw;
}

int64_t OptInt(const nlohmann::json& j, const char* key, int64_t fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  if (!it->is_number_integer() && !it->is_number_unsigned()) {
    Refuse(std::string("`") + key + "` must be an integer, got " + it->dump() +
           ".");
  }
  return it->get<int64_t>();
}

double OptDouble(const nlohmann::json& j, const char* key, double fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  if (!it->is_number()) {
    Refuse(std::string("`") + key + "` must be a number, got " + it->dump() +
           ".");
  }
  return it->get<double>();
}

bool OptBool(const nlohmann::json& j, const char* key, bool fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  if (!it->is_boolean()) {
    Refuse(std::string("`") + key + "` must be a boolean, got " + it->dump() +
           ".");
  }
  return it->get<bool>();
}

std::vector<std::string> OptStringArray(const nlohmann::json& j,
                                        const char* key) {
  std::vector<std::string> out;
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return out;
  if (!it->is_array()) {
    Refuse(std::string("`") + key + "` must be an array, got " + it->dump() +
           ".");
  }
  for (const auto& e : *it) {
    if (!e.is_string()) {
      Refuse(std::string("`") + key + "` must contain only strings, found " +
             e.dump() + ".");
    }
    out.push_back(e.get<std::string>());
  }
  return out;
}

std::vector<int64_t> OptIntArray(const nlohmann::json& j, const char* key) {
  std::vector<int64_t> out;
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return out;
  if (!it->is_array()) {
    Refuse(std::string("`") + key + "` must be an array, got " + it->dump() +
           ".");
  }
  for (const auto& e : *it) {
    if (!e.is_number_integer() && !e.is_number_unsigned()) {
      Refuse(std::string("`") + key + "` must contain only integers, found " +
             e.dump() + ".");
    }
    out.push_back(e.get<int64_t>());
  }
  return out;
}

// `full_attention` -> `deepseek_sparse_attention`. Upstream rewrites the
// published list in `__post_init__`; accepting the pre-rewrite spelling here is
// what makes the rewrite OURS rather than a thing the caller has to remember.
Glm5NextLayerKind LayerKindFromString(const std::string& s) {
  if (s == "linear_attention") return Glm5NextLayerKind::kLinearAttention;
  if (s == "deepseek_sparse_attention" || s == "full_attention") {
    return Glm5NextLayerKind::kDeepseekSparseAttention;
  }
  Refuse("unsupported layer type '" + s +
         "'; expected `linear_attention` or `deepseek_sparse_attention` (the "
         "checkpoint's `full_attention` is rewritten to the latter).");
}

Glm5NextMlpKind MlpKindFromString(const std::string& s) {
  if (s == "dense") return Glm5NextMlpKind::kDense;
  if (s == "sparse") return Glm5NextMlpKind::kSparse;
  Refuse("unsupported mlp layer type '" + s +
         "'; expected `dense` or `sparse`.");
}

Glm5NextIndexerKind IndexerKindFromString(const std::string& s) {
  if (s == "full") return Glm5NextIndexerKind::kFull;
  if (s == "shared") return Glm5NextIndexerKind::kShared;
  Refuse("unsupported indexer type '" + s + "'; expected `full` or `shared`.");
}

}  // namespace

const char* Glm5NextLayerKindName(Glm5NextLayerKind kind) {
  return kind == Glm5NextLayerKind::kLinearAttention
             ? "linear_attention"
             : "deepseek_sparse_attention";
}

const char* Glm5NextMlpKindName(Glm5NextMlpKind kind) {
  return kind == Glm5NextMlpKind::kDense ? "dense" : "sparse";
}

const char* Glm5NextIndexerKindName(Glm5NextIndexerKind kind) {
  return kind == Glm5NextIndexerKind::kFull ? "full" : "shared";
}

int64_t Glm5NextParams::num_kda_layers() const {
  return static_cast<int64_t>(
      std::count(layer_types.begin(), layer_types.end(),
                 Glm5NextLayerKind::kLinearAttention));
}

int64_t Glm5NextParams::num_dsa_layers() const {
  return static_cast<int64_t>(
      std::count(layer_types.begin(), layer_types.end(),
                 Glm5NextLayerKind::kDeepseekSparseAttention));
}

Glm5NextParams ParseGlm5NextParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  const nlohmann::json& text = TextOf(raw);
  Glm5NextParams p;

  p.hidden_size = config.hidden_size;
  p.num_hidden_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.rms_norm_eps = config.rms_norm_eps;
  p.max_position_embeddings = config.max_position_embeddings;
  p.intermediate_size = OptInt(text, "intermediate_size", 12288);
  p.swiglu_limit = OptDouble(text, "swiglu_limit", 10.0);
  p.tie_word_embeddings = OptBool(raw, "tie_word_embeddings",
                                  OptBool(text, "tie_word_embeddings", false));

  if (p.hidden_size <= 0) {
    Refuse("`hidden_size` must be > 0, got " + std::to_string(p.hidden_size) +
           ".");
  }
  if (p.num_hidden_layers <= 0) {
    Refuse("`num_hidden_layers` must be > 0, got " +
           std::to_string(p.num_hidden_layers) + ".");
  }
  if (p.vocab_size <= 0) {
    Refuse("`vocab_size` must be > 0, got " + std::to_string(p.vocab_size) +
           ".");
  }
  if (p.intermediate_size <= 0) {
    Refuse("`intermediate_size` must be > 0, got " +
           std::to_string(p.intermediate_size) + ".");
  }
  if (!(p.swiglu_limit > 0.0)) {
    Refuse("`swiglu_limit` must be > 0 (it is a symmetric clamp on the SwiGLU "
           "up projection and a one-sided clamp on the gate), got " +
           std::to_string(p.swiglu_limit) + ".");
  }

  // --- heads. Upstream's FIRST validate_architecture rejection ---------------
  p.num_attention_heads = config.num_attention_heads;
  // `num_key_value_heads is None -> num_attention_heads`, which the shared
  // reader already applies; the guard below is upstream's, over the resolved
  // pair, and it is the reason this model has no GQA arm at all.
  p.num_key_value_heads = config.num_key_value_heads;
  if (p.num_attention_heads <= 0) {
    Refuse("`num_attention_heads` must be > 0, got " +
           std::to_string(p.num_attention_heads) + ".");
  }
  if (p.num_attention_heads != p.num_key_value_heads) {
    Refuse("num_attention_heads (" + std::to_string(p.num_attention_heads) +
           ") must be the same as num_key_value_heads (" +
           std::to_string(p.num_key_value_heads) + ").");
  }

  // --- the three per-layer schedules ----------------------------------------
  //
  // `layer_types`, defaulting to upstream's `idx % 4 != 3` KDA pattern, then
  // rewritten so no `full_attention` survives.
  //
  // READ FROM `text` AND NEVER FROM `config.layer_types`, and that is #2070
  // rather than a style choice. When `layer_types` is absent the shared reader
  // SYNTHESIZES one from `linear_attn_config.kda_layers` using Kimi-Linear's
  // convention, in which that list is ONE-INDEXED (`hf_config.cpp`, the
  // `is_kda[one_indexed - 1]` loop). GLM-5.3-Flash's list is ZERO-INDEXED --
  // it contains `0`, which a one-indexed list of 45 layers cannot, and its
  // maximum is 44 on 45 layers -- so read through that rule the schedule comes
  // out shifted by one and a third of the stack gets the wrong attention kind,
  // silently. Worse, the transformers reference IGNORES `kda_layers` entirely
  // for this model. Taking the schedule from this model's own config is what
  // keeps another family's heuristic out of it.
  {
    std::vector<std::string> names = OptStringArray(text, "layer_types");
    if (names.empty()) {
      for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
        names.push_back(i % 4 != 3 ? "linear_attention"
                                   : "deepseek_sparse_attention");
      }
    }
    if (static_cast<int64_t>(names.size()) != p.num_hidden_layers) {
      Refuse("`layer_types` has " + std::to_string(names.size()) +
             " entries but `num_hidden_layers` is " +
             std::to_string(p.num_hidden_layers) + ".");
    }
    for (const std::string& s : names) {
      p.layer_types.push_back(LayerKindFromString(s));
    }
  }

  // `mlp_layer_types`, defaulting to `["dense"] * min(3, L) + ["sparse"] *
  // (L - 3)`. NOT read from `first_k_dense_replace`: the runtime class declares
  // no such field and `__post_init__` never mentions it (see `glm5_next.h` for
  // where the inherited attribute is removed), so the checkpoint's copy of it
  // is an inert kwarg that merely agrees with the literal 3 below. A port that
  // reads the key instead works on this checkpoint and silently follows a value
  // upstream ignores on any other.
  {
    std::vector<std::string> names = OptStringArray(text, "mlp_layer_types");
    if (names.empty()) {
      const int64_t dense = std::min<int64_t>(3, p.num_hidden_layers);
      for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
        names.push_back(i < dense ? "dense" : "sparse");
      }
    }
    if (static_cast<int64_t>(names.size()) != p.num_hidden_layers) {
      Refuse("`mlp_layer_types` has " + std::to_string(names.size()) +
             " entries but `num_hidden_layers` is " +
             std::to_string(p.num_hidden_layers) + ".");
    }
    for (const std::string& s : names) {
      p.mlp_layer_types.push_back(MlpKindFromString(s));
    }
  }

  // `indexer_types`, and BOTH of upstream's fallbacks for an absent one.
  // `__post_init__` reads `index_topk_pattern` FIRST and only reaches the
  // freq/offset schedule when that key is absent too, so a port that
  // implements only the schedule resolves a pattern config to a different
  // stack -- and silently, because nothing forwards `indexer_types` yet and
  // this model has no reachable token gate anywhere on this fleet. A 45-char
  // `"FSSFSS..."` would come out 45 layers of `full` where the reference
  // alternates, which is two thirds of the DSA layers running the indexer
  // where the reference reuses a prior selection.
  //
  // A pattern is a STRING of `F`/`S` codes, or any other sequence, in which
  // case upstream takes `list(pattern)` verbatim and the entries are already
  // `full`/`shared` names. Both spellings land in `names` and meet the same
  // length check and the same `IndexerKindFromString` below, so an unknown
  // code is refused rather than defaulted -- upstream's own `{"F":..,"S":..}[c]`
  // raises `KeyError` on one.
  {
    std::vector<std::string> names = OptStringArray(text, "indexer_types");
    const auto pattern = text.find("index_topk_pattern");
    const bool has_pattern =
        names.empty() && pattern != text.end() && !pattern->is_null();
    if (has_pattern && pattern->is_string()) {
      for (const char c : pattern->get<std::string>()) {
        if (c == 'F') {
          names.emplace_back("full");
        } else if (c == 'S') {
          names.emplace_back("shared");
        } else {
          Refuse(std::string("`index_topk_pattern` is a string of `F` and `S` "
                             "codes, found '") +
                 c + "'.");
        }
      }
    } else if (has_pattern) {
      names = OptStringArray(text, "index_topk_pattern");
    } else if (names.empty()) {
      const int64_t freq = std::max<int64_t>(OptInt(text, "index_topk_freq", 1), 1);
      const int64_t offset = OptInt(text, "index_skip_topk_offset", 2);
      for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
        const int64_t shifted = std::max<int64_t>(i - offset + 1, 0);
        names.push_back(shifted % freq == 0 ? "full" : "shared");
      }
    }
    if (static_cast<int64_t>(names.size()) != p.num_hidden_layers) {
      Refuse("`indexer_types` has " + std::to_string(names.size()) +
             " entries but `num_hidden_layers` is " +
             std::to_string(p.num_hidden_layers) + ".");
    }
    for (const std::string& s : names) {
      p.indexer_types.push_back(IndexerKindFromString(s));
    }
  }

  // The FIRST indexer layer cannot be `shared`: `shared` reuses the previous
  // full layer's top-k selection and there is no previous one. Upstream never
  // synthesizes such a schedule (its offset rule always makes layer 0 full at
  // freq 1), but a hand-written or converter-written list can, and the failure
  // it produces downstream is an attention over an empty selection rather than
  // an error.
  for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
    if (p.layer_types[static_cast<size_t>(i)] !=
        Glm5NextLayerKind::kDeepseekSparseAttention) {
      continue;
    }
    if (p.indexer_types[static_cast<size_t>(i)] == Glm5NextIndexerKind::kFull) {
      break;
    }
    Refuse("layer " + std::to_string(i) +
           " is the first `deepseek_sparse_attention` layer and its "
           "`indexer_types` entry is `shared`, which reuses a previous full "
           "layer's top-k selection that does not exist.");
  }

  // --- NoPE MLA. Upstream's FOURTH and FIFTH rejections ---------------------
  // 1536 is the CLASS DEFAULT, so an absent key is a config upstream ACCEPTS
  // and this must accept too. `validate_architecture` refuses only an explicit
  // `None`, which in JSON is an explicit null -- a distinction a plain
  // "read with default 0, refuse 0" would collapse into a false refusal.
  p.mla.q_lora_rank = OptInt(text, "q_lora_rank", 1536);
  p.mla.kv_lora_rank = OptInt(text, "kv_lora_rank", 512);
  p.mla.qk_nope_head_dim = OptInt(text, "qk_nope_head_dim", 256);
  p.mla.qk_rope_head_dim = OptInt(text, "qk_rope_head_dim", 0);
  p.mla.v_head_dim = OptInt(text, "v_head_dim", 256);
  // Upstream's forced overrides. `head_dim` and `qk_head_dim` are COMPUTED and
  // the config's own copies are popped/ignored, so reading them is reading a
  // value nothing upstream consults.
  p.mla.head_dim = p.mla.qk_rope_head_dim;
  p.mla.qk_head_dim = p.mla.qk_rope_head_dim + p.mla.qk_nope_head_dim;

  {
    // Upstream's `if self.q_lora_rank is None`, and its message verbatim. Only
    // an explicit null reaches it; an absent key took the class default above.
    auto it = text.find("q_lora_rank");
    if (it != text.end() && it->is_null()) {
      Refuse("For DSA usage in the attention layers, the `q_lora_rank` is "
             "strictly required!");
    }
  }
  // A LOCAL guard, stricter than upstream, and named as such: upstream does not
  // range-check the rank, but a non-positive one sizes the q-compression
  // projection to nothing and there is no downstream gate on this fleet that
  // would notice.
  if (p.mla.q_lora_rank <= 0) {
    Refuse("`q_lora_rank` must be > 0, got " +
           std::to_string(p.mla.q_lora_rank) + ".");
  }
  // NOT a typo and NOT the inverse of our own MLA guard by accident. Upstream
  // REQUIRES NoPE here, and `MlaBlockDims::Validate`
  // (layers/attention/mla_attention.cpp:90-93) requires every dimension to be
  // > 0 — the two are exact complements over this field and no value satisfies
  // both. W1 mirrors upstream; relaxing ours is W3's, tracked as O11.
  if (p.mla.qk_rope_head_dim > 0) {
    Refuse("Expecting NoPE for the DSA attention layers, but got " +
           std::to_string(p.mla.qk_rope_head_dim) + " as RoPE dim.");
  }
  // A LOCAL guard, stricter than upstream, and named as such for the same
  // reason the `q_lora_rank` one above is: `validate_architecture` refuses only
  // a POSITIVE `qk_rope_head_dim`, so upstream accepts a negative one and
  // carries it into `head_dim = qk_rope_head_dim` and
  // `qk_head_dim = qk_rope_head_dim + qk_nope_head_dim`. A negative width there
  // sizes the rope slice and the query head to nonsense, and no gate on this
  // fleet would notice.
  if (p.mla.qk_rope_head_dim < 0) {
    Refuse("`qk_rope_head_dim` must be >= 0, got " +
           std::to_string(p.mla.qk_rope_head_dim) + ".");
  }
  if (p.mla.qk_nope_head_dim <= 0 || p.mla.v_head_dim <= 0 ||
      p.mla.kv_lora_rank <= 0) {
    Refuse("`qk_nope_head_dim`, `v_head_dim` and `kv_lora_rank` must be > 0, "
           "got " + std::to_string(p.mla.qk_nope_head_dim) + ", " +
           std::to_string(p.mla.v_head_dim) + " and " +
           std::to_string(p.mla.kv_lora_rank) + ".");
  }

  // --- the DSA indexer. Upstream's SECOND and THIRD rejections --------------
  p.indexer.head_dim = OptInt(text, "index_head_dim", 128);
  p.indexer.n_heads = OptInt(text, "index_n_heads", 32);
  p.indexer.topk = OptInt(text, "index_topk", 2048);
  // The class default is 16; the published checkpoint says 4. Read, never
  // defaulted-past.
  p.indexer.kpool = OptInt(text, "index_kpool", 16);
  p.indexer.kpool_always_select_tail =
      OptBool(text, "index_kpool_always_select_tail", true);

  if (p.indexer.kpool < 1) {
    Refuse("index_kpool must be positive, got " +
           std::to_string(p.indexer.kpool) + ".");
  }
  if (p.indexer.topk % p.indexer.kpool != 0) {
    Refuse("index_topk (" + std::to_string(p.indexer.topk) +
           ") must be divisible by index_kpool (" +
           std::to_string(p.indexer.kpool) + ").");
  }
  if (p.indexer.topk <= 0 || p.indexer.head_dim <= 0 ||
      p.indexer.n_heads <= 0) {
    Refuse("`index_topk`, `index_head_dim` and `index_n_heads` must be > 0, "
           "got " + std::to_string(p.indexer.topk) + ", " +
           std::to_string(p.indexer.head_dim) + " and " +
           std::to_string(p.indexer.n_heads) + ".");
  }

  // --- KDA, through the `linear_attn_config` SUB-OBJECT ---------------------
  //
  // Four keys under four DIFFERENT spellings, remapped by `__post_init__`. The
  // dict's `kda_layers` and `full_attn_layers` index lists are read here ONLY
  // to check them against `layer_types`; upstream IGNORES both, and the
  // top-level `layer_types` is the authority. Checking rather than ignoring is
  // the one place this port is stricter than upstream, and deliberately so: a
  // checkpoint whose two descriptions of the same schedule disagree is a
  // checkpoint one of whose readers is wrong.
  p.kda.head_dim = OptInt(text, "linear_head_dim", 128);
  p.kda.num_heads = OptInt(text, "linear_num_heads", 64);
  p.kda.conv_kernel_dim = OptInt(text, "linear_conv_kernel_dim", 4);
  // `linear_lower_bound` is the FLAT spelling of the bound after upstream's
  // remap. Its three states are distinct and all three are reachable: ABSENT
  // takes the class default -5.0, an explicit NULL is upstream's `None` and
  // selects the softplus branch, and a number is used as it stands.
  {
    auto it = text.find("linear_lower_bound");
    if (it == text.end()) {
      p.kda.lower_bound = -5.0;
    } else if (!it->is_null()) {
      p.kda.lower_bound = OptDouble(text, "linear_lower_bound", -5.0);
    }
  }

  auto lin_it = text.find("linear_attn_config");
  if (lin_it != text.end() && lin_it->is_object()) {
    const nlohmann::json& lin = *lin_it;
    p.kda.head_dim = OptInt(lin, "head_dim", p.kda.head_dim);
    p.kda.num_heads = OptInt(lin, "num_heads", p.kda.num_heads);
    p.kda.conv_kernel_dim =
        OptInt(lin, "short_conv_kernel_size", p.kda.conv_kernel_dim);
    if (lin.find("gate_lower_bound") != lin.end()) {
      if (lin["gate_lower_bound"].is_null()) {
        p.kda.lower_bound.reset();
      } else {
        p.kda.lower_bound = OptDouble(lin, "gate_lower_bound", -5.0);
      }
    }
    // Upstream's `safe_gate` rule, verbatim: `safe_gate` DEFAULTS TO TRUE, and
    // a true `safe_gate` over an absent bound installs -5.0. So the softplus
    // branch is reachable only with an explicit `"safe_gate": false`.
    if (OptBool(lin, "safe_gate", true) && !p.kda.lower_bound.has_value()) {
      p.kda.lower_bound = -5.0;
    }

    const std::vector<int64_t> kda_layers = OptIntArray(lin, "kda_layers");
    const std::vector<int64_t> full_layers =
        OptIntArray(lin, "full_attn_layers");
    for (int64_t idx : kda_layers) {
      if (idx < 0 || idx >= p.num_hidden_layers) {
        Refuse("`linear_attn_config.kda_layers` names layer " +
               std::to_string(idx) + ", which is outside [0, " +
               std::to_string(p.num_hidden_layers) + ").");
      }
      if (p.layer_types[static_cast<size_t>(idx)] !=
          Glm5NextLayerKind::kLinearAttention) {
        Refuse("`linear_attn_config.kda_layers` names layer " +
               std::to_string(idx) +
               " but `layer_types` calls it `deepseek_sparse_attention`; "
               "`layer_types` is the authority upstream reads and the two must "
               "agree.");
      }
    }
    for (int64_t idx : full_layers) {
      if (idx < 0 || idx >= p.num_hidden_layers) {
        Refuse("`linear_attn_config.full_attn_layers` names layer " +
               std::to_string(idx) + ", which is outside [0, " +
               std::to_string(p.num_hidden_layers) + ").");
      }
      if (p.layer_types[static_cast<size_t>(idx)] !=
          Glm5NextLayerKind::kDeepseekSparseAttention) {
        Refuse("`linear_attn_config.full_attn_layers` names layer " +
               std::to_string(idx) +
               " but `layer_types` calls it `linear_attention`; `layer_types` "
               "is the authority upstream reads and the two must agree.");
      }
    }
  }

  if (p.kda.num_heads <= 0 || p.kda.head_dim <= 0) {
    Refuse("`linear_num_heads` and `linear_head_dim` must be > 0, got " +
           std::to_string(p.kda.num_heads) + " and " +
           std::to_string(p.kda.head_dim) + ".");
  }
  if (p.kda.conv_kernel_dim <= 0) {
    Refuse("`linear_conv_kernel_dim` must be > 0, got " +
           std::to_string(p.kda.conv_kernel_dim) + ".");
  }
  // A POSITIVE lower bound would flip the forget gate's sign: the gate is
  // `-bound * sigmoid(...)`, so a positive bound makes `decay_rate` negative
  // and turns decay into growth. Upstream types the field `float | None` and
  // does not range-check it; this refuses rather than producing a divergent
  // model that still generates text.
  if (p.kda.lower_bound.has_value() && !(*p.kda.lower_bound < 0.0)) {
    Refuse("`gate_lower_bound` must be < 0 (the forget gate is `-bound * "
           "sigmoid(...)`, so a non-negative bound turns decay into growth), "
           "got " + std::to_string(*p.kda.lower_bound) + ".");
  }

  // --- mHC -----------------------------------------------------------------
  p.mhc.mult = OptInt(text, "hc_mult", 4);
  p.mhc.sinkhorn_iters = OptInt(text, "hc_sinkhorn_iters", 20);
  p.mhc.eps = OptDouble(text, "hc_eps", 1e-6);
  if (p.mhc.mult <= 1) {
    Refuse("`hc_mult` must be > 1 (it is the number of residual streams in the "
           "hyper-connection manifold), got " + std::to_string(p.mhc.mult) +
           ".");
  }
  if (p.mhc.sinkhorn_iters <= 0) {
    Refuse("`hc_sinkhorn_iters` must be > 0, got " +
           std::to_string(p.mhc.sinkhorn_iters) + ".");
  }
  if (!(p.mhc.eps > 0.0)) {
    Refuse("`hc_eps` must be > 0 (it is added to every Sinkhorn denominator, "
           "not used as a floor), got " + std::to_string(p.mhc.eps) + ".");
  }

  // --- MoE -----------------------------------------------------------------
  p.moe.n_routed_experts = OptInt(text, "n_routed_experts", 288);
  p.moe.n_shared_experts = OptInt(text, "n_shared_experts", 1);
  p.moe.num_experts_per_tok = OptInt(text, "num_experts_per_tok", 8);
  p.moe.moe_intermediate_size = OptInt(text, "moe_intermediate_size", 2048);
  p.moe.n_group = OptInt(text, "n_group", 1);
  p.moe.topk_group = OptInt(text, "topk_group", 1);
  p.moe.routed_scaling_factor = OptDouble(text, "routed_scaling_factor", 2.5);
  p.moe.norm_topk_prob = OptBool(text, "norm_topk_prob", true);

  const bool any_sparse =
      std::find(p.mlp_layer_types.begin(), p.mlp_layer_types.end(),
                Glm5NextMlpKind::kSparse) != p.mlp_layer_types.end();
  if (any_sparse) {
    if (p.moe.n_routed_experts <= 0) {
      Refuse("`n_routed_experts` must be > 0 on a config with `sparse` mlp "
             "layers, got " + std::to_string(p.moe.n_routed_experts) + ".");
    }
    if (p.moe.num_experts_per_tok <= 0 ||
        p.moe.num_experts_per_tok > p.moe.n_routed_experts) {
      Refuse("`num_experts_per_tok` must be in [1, n_routed_experts], got " +
             std::to_string(p.moe.num_experts_per_tok) + " and " +
             std::to_string(p.moe.n_routed_experts) + ".");
    }
    if (p.moe.moe_intermediate_size <= 0) {
      Refuse("`moe_intermediate_size` must be > 0, got " +
             std::to_string(p.moe.moe_intermediate_size) + ".");
    }
    if (p.moe.n_group <= 0 || p.moe.topk_group <= 0 ||
        p.moe.topk_group > p.moe.n_group) {
      Refuse("`topk_group` must be in [1, n_group], got " +
             std::to_string(p.moe.topk_group) + " and " +
             std::to_string(p.moe.n_group) + ".");
    }
    if (p.moe.n_routed_experts % p.moe.n_group != 0) {
      Refuse("`n_routed_experts` (" + std::to_string(p.moe.n_routed_experts) +
             ") must be divisible by `n_group` (" +
             std::to_string(p.moe.n_group) + ").");
    }
  }

  // --- vision --------------------------------------------------------------
  auto vis_it = raw.find("vision_config");
  if (vis_it != raw.end() && vis_it->is_object()) {
    const nlohmann::json& v = *vis_it;
    p.has_vision = true;
    p.vision.depth = OptInt(v, "depth", 24);
    p.vision.hidden_size = OptInt(v, "hidden_size", 1024);
    p.vision.intermediate_size = OptInt(v, "intermediate_size", 4096);
    p.vision.num_heads = OptInt(v, "num_heads", 16);
    p.vision.in_channels = OptInt(v, "in_channels", 3);
    p.vision.patch_size = OptInt(v, "patch_size", 14);
    // The class defaults are 336 and 1536; the checkpoint says 448 and 4096.
    p.vision.image_size = OptInt(v, "image_size", 336);
    p.vision.spatial_merge_size = OptInt(v, "spatial_merge_size", 2);
    p.vision.temporal_patch_size = OptInt(v, "temporal_patch_size", 2);
    p.vision.out_hidden_size = OptInt(v, "out_hidden_size", 1536);
    p.vision.projection_intermediate_size =
        OptInt(v, "projection_intermediate_size", 10240);
    p.vision.rms_norm_eps = OptDouble(v, "rms_norm_eps", 1e-5);
    p.vision.swiglu_limit = OptDouble(v, "swiglu_limit", 10.0);
    p.vision.attention_bias = OptBool(v, "attention_bias", true);

    if (p.vision.depth <= 0 || p.vision.hidden_size <= 0 ||
        p.vision.num_heads <= 0 || p.vision.patch_size <= 0 ||
        p.vision.spatial_merge_size <= 0 || p.vision.temporal_patch_size <= 0) {
      Refuse("every vision geometry field must be > 0; got depth " +
             std::to_string(p.vision.depth) + ", hidden_size " +
             std::to_string(p.vision.hidden_size) + ", num_heads " +
             std::to_string(p.vision.num_heads) + ", patch_size " +
             std::to_string(p.vision.patch_size) + ", spatial_merge_size " +
             std::to_string(p.vision.spatial_merge_size) +
             ", temporal_patch_size " +
             std::to_string(p.vision.temporal_patch_size) + ".");
    }
    if (p.vision.hidden_size % p.vision.num_heads != 0) {
      Refuse("vision `hidden_size` (" + std::to_string(p.vision.hidden_size) +
             ") must be divisible by `num_heads` (" +
             std::to_string(p.vision.num_heads) + ").");
    }
    // The merger consumes `projection_intermediate_size` as its context dim,
    // NOT the `out_hidden_size * in_channels` rule the GlmOcr parent uses;
    // 10240 against 12288 here, so a port that inherits the parent's rule sizes
    // the merger wrong on a checkpoint that never says so.
    if (p.vision.projection_intermediate_size <= 0) {
      Refuse("vision `projection_intermediate_size` must be > 0 (it is the "
             "merger's context dim, not `out_hidden_size * in_channels`), got " +
             std::to_string(p.vision.projection_intermediate_size) + ".");
    }
  }

  // The six placeholder ids live on the WRAPPER, not in `text_config`. Image
  // and video share one id in the emitted sequence, so all six travel together
  // or a reader classifies every video frame as an image.
  p.mm_tokens.image = OptInt(raw, "image_token_id", 154854);
  p.mm_tokens.video = OptInt(raw, "video_token_id", 154855);
  p.mm_tokens.image_start = OptInt(raw, "image_start_token_id", 154830);
  p.mm_tokens.image_end = OptInt(raw, "image_end_token_id", 154831);
  p.mm_tokens.video_start = OptInt(raw, "video_start_token_id", 154832);
  p.mm_tokens.video_end = OptInt(raw, "video_end_token_id", 154833);

  // --- quantization --------------------------------------------------------
  auto q_it = raw.find("quantization_config");
  if (q_it != raw.end() && q_it->is_object()) {
    const nlohmann::json& q = *q_it;
    auto method = q.find("quant_method");
    if (method != q.end() && method->is_string()) {
      p.quant_method = method->get<std::string>();
    }
    auto fmt = q.find("fmt");
    if (fmt != q.end() && fmt->is_string()) p.quant_fmt = fmt->get<std::string>();
    p.weight_block_size = OptIntArray(q, "weight_block_size");
    p.modules_to_not_convert = OptStringArray(q, "modules_to_not_convert");
    if (!p.weight_block_size.empty() && p.weight_block_size.size() != 2) {
      Refuse("`quantization_config.weight_block_size` must have exactly 2 "
             "entries, got " + std::to_string(p.weight_block_size.size()) +
             ".");
    }
    for (int64_t b : p.weight_block_size) {
      if (b <= 0) {
        Refuse("`quantization_config.weight_block_size` entries must be > 0, "
               "got " + std::to_string(b) + ".");
      }
    }
  }

  return p;
}

void ParseGlm5NextConfig(const HfConfig& config) {
  (void)ParseGlm5NextParams(config);
}

}  // namespace vllm
