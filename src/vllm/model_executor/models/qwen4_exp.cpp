// Qwen4-Exp config resolution + validation (W1 of MODEL-MM-QWEN4-EXP, #1981).
//
// The resolve IS the validation: every refusal below mirrors one in upstream
// `Qwen4ExpTextConfig.validate_architecture` / `__post_init__` at the accepted
// lane pin, transformers **5.16.0**. vLLM implements `qwen4_exp` at NO
// revision, so there is nothing to mirror on this surface and transformers is
// the only source; see `.agents/specs/qwen4-exp-flash-next.md` `## Oracles`.
#include "vllm/model_executor/models/qwen4_exp.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

namespace vllm {
namespace {

constexpr const char* kSlug = "qwen4_exp";

[[noreturn]] void Refuse(const std::string& what) {
  throw std::runtime_error(std::string(kSlug) + ": " + what +
                           " See .agents/specs/qwen4-exp-flash-next.md and "
                           "issue #1981.");
}

// The model's own sub-object. Upstream nests everything except the wrapper's
// `architectures`/`model_type`/vision block under `text_config`; a flat config
// (what a GGUF-derived or hand-written config looks like) is accepted by
// falling back to the top level, exactly as the shared HfConfig reader does.
const nlohmann::json& TextOf(const nlohmann::json& raw) {
  auto it = raw.find("text_config");
  if (it != raw.end() && it->is_object()) return *it;
  return raw;
}

int64_t OptInt(const nlohmann::json& j, const char* key, int64_t fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  if (!it->is_number_integer() && !it->is_number_unsigned()) {
    Refuse(std::string("`") + key + "` must be an integer.");
  }
  return it->get<int64_t>();
}

double OptDouble(const nlohmann::json& j, const char* key, double fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  if (!it->is_number()) Refuse(std::string("`") + key + "` must be a number.");
  return it->get<double>();
}

// Upstream treats an ABSENT QSA field and a null one alike, and the group is
// all-or-nothing, so presence has to be observable separately from value.
bool HasKey(const nlohmann::json& j, const char* key) {
  auto it = j.find(key);
  return it != j.end() && !it->is_null();
}

std::vector<int64_t> OptIntArray(const nlohmann::json& j, const char* key) {
  std::vector<int64_t> out;
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return out;
  if (!it->is_array()) Refuse(std::string("`") + key + "` must be an array.");
  for (const auto& e : *it) {
    if (!e.is_number_integer() && !e.is_number_unsigned()) {
      Refuse(std::string("`") + key + "` must contain only integers.");
    }
    out.push_back(e.get<int64_t>());
  }
  return out;
}

// `full_attention` -> `qwen_sparse_attention`. Upstream rewrites the published
// list in `__post_init__` with the comment that the checkpoint "contains
// full_attention entries for layers that are actually using an indexer".
Qwen4ExpLayerKind KindFromString(const std::string& s) {
  if (s == "linear_attention") return Qwen4ExpLayerKind::kLinearAttention;
  if (s == "qwen_sparse_attention" || s == "full_attention") {
    return Qwen4ExpLayerKind::kQwenSparseAttention;
  }
  Refuse("unsupported layer type '" + s +
         "'; expected `linear_attention` or `qwen_sparse_attention` (the "
         "checkpoint's `full_attention` is rewritten to the latter).");
}

}  // namespace

Qwen4ExpParams ParseQwen4ExpParams(const HfConfig& config) {
  const nlohmann::json& text = TextOf(config.raw);
  Qwen4ExpParams p;

  p.hidden_size = config.hidden_size;
  p.num_hidden_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.rms_norm_eps = config.rms_norm_eps;
  p.num_attention_heads = config.num_attention_heads;
  p.num_key_value_heads = config.num_key_value_heads;
  p.head_dim = config.head_dim;

  if (p.num_hidden_layers <= 0) {
    Refuse("`num_hidden_layers` must be > 0, got " +
           std::to_string(p.num_hidden_layers) + ".");
  }

  // --- layer_types: rewrite a published list, or synthesize from the interval.
  // Both paths must agree on the real checkpoint, and the test asserts that
  // they do: the published 48-entry list rewritten is byte-identical to the
  // list synthesized from `full_attention_interval` = 4.
  if (!config.layer_types.empty()) {
    if (static_cast<int64_t>(config.layer_types.size()) != p.num_hidden_layers) {
      Refuse("`layer_types` has " + std::to_string(config.layer_types.size()) +
             " entries but `num_hidden_layers` is " +
             std::to_string(p.num_hidden_layers) + ".");
    }
    p.layer_types.reserve(config.layer_types.size());
    for (const std::string& s : config.layer_types) {
      p.layer_types.push_back(KindFromString(s));
    }
  } else {
    const int64_t interval = OptInt(text, "full_attention_interval", 4);
    if (interval <= 0) {
      Refuse("`full_attention_interval` must be > 0, got " +
             std::to_string(interval) + ".");
    }
    p.layer_types.reserve(static_cast<size_t>(p.num_hidden_layers));
    for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
      p.layer_types.push_back((i + 1) % interval != 0
                                  ? Qwen4ExpLayerKind::kLinearAttention
                                  : Qwen4ExpLayerKind::kQwenSparseAttention);
    }
  }

  // --- gated residual ---
  p.hc_count = OptInt(text, "hc_count", 4);
  p.hc_lowrank = OptInt(text, "hc_lowrank", 320);
  if (p.hc_count <= 1) {
    Refuse("requires `hc_count` > 1, got " + std::to_string(p.hc_count) + ".");
  }
  if (p.hc_lowrank <= 0) {
    Refuse("`hc_lowrank` must be > 0, got " + std::to_string(p.hc_lowrank) +
           ".");
  }

  // --- MoE ---
  p.num_experts = config.num_experts;
  p.num_experts_per_tok = config.num_experts_per_tok;
  p.moe_intermediate_size = config.moe_intermediate_size;
  p.shared_expert_intermediate_size = config.shared_expert_intermediate_size;
  if (p.num_experts <= 0) {
    Refuse("`num_experts` must be > 0, got " + std::to_string(p.num_experts) +
           ".");
  }
  if (p.num_experts_per_tok <= 0 || p.num_experts_per_tok > p.num_experts) {
    Refuse("`num_experts_per_tok` must be in [1, num_experts], got " +
           std::to_string(p.num_experts_per_tok) + " and " +
           std::to_string(p.num_experts) + ".");
  }
  if (p.moe_intermediate_size <= 0 || p.shared_expert_intermediate_size <= 0) {
    Refuse("`moe_intermediate_size` and `shared_expert_intermediate_size` must "
           "be > 0, got " + std::to_string(p.moe_intermediate_size) + " and " +
           std::to_string(p.shared_expert_intermediate_size) + ".");
  }

  // --- output gate. The shared reader already canonicalizes `swish` to `silu`
  // and refuses anything outside {silu, swish, sigmoid}; upstream Qwen4-Exp
  // accepts {sigmoid, silu}, so the two sets agree AFTER canonicalization.
  if (config.output_gate_type != "silu" && config.output_gate_type != "sigmoid") {
    Refuse("unsupported output gate activation '" + config.output_gate_type +
           "'; expected `sigmoid` or `silu`.");
  }

  // --- rotary. `partial_rotary_factor` is read HERE with upstream's inherited
  // default of 0.25 rather than taken from `config.rotary_dim`, and that is
  // deliberate. `IsQwen35Family` in the shared reader does not list
  // `qwen4_exp`, so an ABSENT key defaults there to 1.0 (full rotary) where
  // upstream `Qwen4ExpTextConfig`, subclassing `Qwen3_5MoeTextConfig`, inherits
  // 0.25. On the published checkpoint the key is present and both agree at 64;
  // on a config that omits it they would disagree 256 vs 64, and because
  // upstream's own guard is `rotary_dim > indexer_head_dim`, the shared
  // reader's value would make us REFUSE a config upstream ACCEPTS. Mirroring
  // the inheritance is what keeps the refusal set identical.
  p.partial_rotary_factor = OptDouble(text, "partial_rotary_factor", 0.25);
  if (!(p.partial_rotary_factor > 0.0)) {
    Refuse("`partial_rotary_factor` must be > 0, got " +
           std::to_string(p.partial_rotary_factor) + ".");
  }
  p.rotary_dim =
      static_cast<int64_t>(static_cast<double>(p.head_dim) *
                           p.partial_rotary_factor);

  // --- QSA: all-or-nothing, then per-field.
  static constexpr const char* kQsaFields[] = {
      "indexer_n_heads", "indexer_kv_heads", "indexer_head_dim",
      "indexer_budget", "indexer_compress_ratio"};
  int present = 0;
  std::string missing;
  for (const char* f : kQsaFields) {
    if (HasKey(text, f)) {
      ++present;
    } else {
      if (!missing.empty()) missing += ", ";
      missing += f;
    }
  }
  if (present > 0) {
    if (present != 5) {
      Refuse("QSA config is missing required fields: " + missing + ".");
    }
    p.qsa.n_heads = OptInt(text, "indexer_n_heads", 0);
    p.qsa.kv_heads = OptInt(text, "indexer_kv_heads", 0);
    p.qsa.head_dim = OptInt(text, "indexer_head_dim", 0);
    p.qsa.budget = OptInt(text, "indexer_budget", 0);
    p.qsa.compress_ratio = OptInt(text, "indexer_compress_ratio", 0);
    if (p.qsa.n_heads <= 0 || p.qsa.kv_heads <= 0 || p.qsa.head_dim <= 0 ||
        p.qsa.budget <= 0 || p.qsa.compress_ratio <= 0) {
      Refuse("QSA config values must be positive.");
    }
    if (p.qsa.kv_heads != 1) {
      Refuse("QSA requires `indexer_kv_heads` = 1, got " +
             std::to_string(p.qsa.kv_heads) + ".");
    }
    if (p.qsa.budget % p.qsa.compress_ratio != 0) {
      Refuse("`indexer_budget` (" + std::to_string(p.qsa.budget) +
             ") must be divisible by `indexer_compress_ratio` (" +
             std::to_string(p.qsa.compress_ratio) + ").");
    }
    if (p.rotary_dim > p.qsa.head_dim) {
      Refuse("rotary dim " + std::to_string(p.rotary_dim) +
             " exceeds `indexer_head_dim` " + std::to_string(p.qsa.head_dim) +
             ".");
    }
  }

  // --- PLE. One-indexed on the way in, 0-based on the way out.
  const std::vector<int64_t> raw_ple = OptIntArray(text, "ple_layer_ids");
  std::set<int64_t> sorted_unique(raw_ple.begin(), raw_ple.end());
  for (int64_t one_based : sorted_unique) {
    if (one_based < 1 || one_based > p.num_hidden_layers) {
      Refuse("`ple_layer_ids` must contain one-indexed ids in [1, " +
             std::to_string(p.num_hidden_layers) + "], got " +
             std::to_string(one_based) + ".");
    }
    const int64_t zero_based = one_based - 1;
    if (p.layer_types[static_cast<size_t>(zero_based)] !=
        Qwen4ExpLayerKind::kLinearAttention) {
      Refuse("PLE is only supported on `linear_attention` layers; "
             "`ple_layer_ids` names one-indexed layer " +
             std::to_string(one_based) + " (0-based " +
             std::to_string(zero_based) + "), which is a sparse-attention "
             "layer.");
    }
    p.ple.layer_ids_zero_based.push_back(zero_based);
  }
  if (!p.ple.layer_ids_zero_based.empty()) {
    p.ple.embed_dim = OptInt(text, "ple_embed_dim", p.hidden_size);
    p.ple.conv_kernel_size = OptInt(text, "ple_conv_kernel_size", 4);
    p.ple.ngram_size = OptInt(text, "ngram_size", 0);
    p.ple.heads_per_ngram = OptInt(text, "heads_per_ngram", 0);
    p.ple.ngram_vocab_size_base = OptInt(text, "ngram_vocab_size_base", 0);
    p.ple.make_ngram_vocab_size_divisible_by =
        OptInt(text, "make_ngram_vocab_size_divisible_by", 0);
    p.ple.split_ngram_parts = OptInt(text, "split_ngram_parts", 512);
    p.ple.seed = OptInt(text, "seed", 1234);
    if (p.ple.ngram_size < 2) {
      Refuse("`ngram_size` must be >= 2 when PLE is enabled, got " +
             std::to_string(p.ple.ngram_size) + ".");
    }
    if (p.ple.heads_per_ngram <= 0) {
      Refuse("`heads_per_ngram` must be > 0 when PLE is enabled, got " +
             std::to_string(p.ple.heads_per_ngram) + ".");
    }
    if (p.ple.conv_kernel_size <= 0) {
      Refuse("`ple_conv_kernel_size` must be > 0, got " +
             std::to_string(p.ple.conv_kernel_size) + ".");
    }
    // 2560 / 16 = 160. A ragged split would silently mis-slice every gathered
    // row, so it is refused rather than truncated.
    const int64_t heads = p.ple.ngram_heads();
    if (heads <= 0 || p.ple.embed_dim % heads != 0) {
      Refuse("`ple_embed_dim` (" + std::to_string(p.ple.embed_dim) +
             ") must be divisible by the n-gram head count (" +
             std::to_string(heads) + ").");
    }
  }

  // --- MTP. `mtp_num_hidden_layers` is a sibling of the `mtp` sub-object; the
  // head is a block inside THIS text config, not a separately registered
  // architecture, which is why this row adds ONE model-matrix row and not two.
  p.mtp_num_hidden_layers = OptInt(text, "mtp_num_hidden_layers", 0);
  if (p.mtp_num_hidden_layers < 0) {
    Refuse("`mtp_num_hidden_layers` must be >= 0, got " +
           std::to_string(p.mtp_num_hidden_layers) + ".");
  }

  return p;
}

void ParseQwen4ExpConfig(const HfConfig& config) {
  (void)ParseQwen4ExpParams(config);
}

}  // namespace vllm
