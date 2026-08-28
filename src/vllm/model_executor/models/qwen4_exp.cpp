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
// falling back to the top level.
//
// The alternatives BELOW `text_config` are not decoration. This has to resolve
// to the same object `HfConfig`'s own `ResolveTextConfig` picked
// (hf_config.cpp:113-121), or one parse answers "what is the text config?"
// twice. A config nested under `llm_config` resolved `hidden_size`,
// `num_hidden_layers` and `layer_types` through the shared reader while this
// function fell back to the WRAPPER and found no `hc_*`, QSA, PLE, MTP or
// interval key at all -- accepted, silently half-parsed, and reporting one conv
// state on a model that needs three.
const nlohmann::json& TextOf(const nlohmann::json& raw) {
  auto it = raw.find("text_config");
  if (it != raw.end() && it->is_object()) return *it;
  it = raw.find("llm_config");
  if (it != raw.end() && it->is_object()) return *it;
  it = raw.find("thinker_config");
  if (it != raw.end() && it->is_object()) {
    auto text = it->find("text_config");
    if (text != it->end() && text->is_object()) return *text;
  }
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

// Upstream's `x or y` treats an absent key, a null and an EMPTY STRING alike,
// so all three have to fall through to the alternative.
std::string OptNonEmptyString(const nlohmann::json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return std::string();
  // A non-string value is dumped verbatim (`3`, `[]`) so the refusal names what
  // was actually found rather than reporting it as absent.
  return it->is_string() ? it->get<std::string>() : it->dump();
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

  // --- linear attention. Taken from the SHARED reader, which already resolves
  // this group for the whole Gated DeltaNet family, rather than re-read from
  // `text` here: a second reading of the same keys is a second thing to keep in
  // agreement, and this row has already paid for one of those
  // (`partial_rotary_factor`, below).
  //
  // W5 (#2031) added these because it is the first wave that consumes them —
  // the GGUF loader cannot size one Gated DeltaNet tensor without them. W1
  // recorded the omission under `## Owed` rather than guessing at the time.
  p.linear_num_key_heads = config.linear_num_key_heads;
  p.linear_num_value_heads = config.linear_num_value_heads;
  p.linear_key_head_dim = config.linear_key_head_dim;
  p.linear_value_head_dim = config.linear_value_head_dim;
  p.linear_conv_kernel_dim = config.linear_conv_kernel_dim;

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

  // A `linear_attention` layer without the Gated DeltaNet geometry cannot be
  // built, and the failure without this check is a shape complaint several
  // layers down naming a tensor the reader did not ask about. LOCAL and tighter
  // than upstream, which lets the dataclass defaults stand in — it can, because
  // its layer object constructs lazily from whatever is there; ours has to size
  // a buffer at load. W5 (#2031).
  const bool any_linear =
      std::find(p.layer_types.begin(), p.layer_types.end(),
                Qwen4ExpLayerKind::kLinearAttention) != p.layer_types.end();
  if (any_linear) {
    if (p.linear_num_key_heads <= 0 || p.linear_num_value_heads <= 0 ||
        p.linear_key_head_dim <= 0 || p.linear_value_head_dim <= 0 ||
        p.linear_conv_kernel_dim <= 0) {
      Refuse("`layer_types` names a `linear_attention` layer, so the Gated "
             "DeltaNet geometry must be positive, got linear_num_key_heads " +
             std::to_string(p.linear_num_key_heads) +
             ", linear_num_value_heads " +
             std::to_string(p.linear_num_value_heads) +
             ", linear_key_head_dim " + std::to_string(p.linear_key_head_dim) +
             ", linear_value_head_dim " +
             std::to_string(p.linear_value_head_dim) +
             ", linear_conv_kernel_dim " +
             std::to_string(p.linear_conv_kernel_dim) + ".");
    }
    // The V heads are grouped BY key head, so a ragged split has no grouping at
    // all — and it is the same divisor the convert-time V-head reorder uses, so
    // a ragged config silently mis-permutes every Gated DeltaNet tensor rather
    // than failing. `num_v == num_k` is legal and simply means no reorder.
    if (p.linear_num_value_heads % p.linear_num_key_heads != 0) {
      Refuse("`linear_num_value_heads` (" +
             std::to_string(p.linear_num_value_heads) +
             ") must be divisible by `linear_num_key_heads` (" +
             std::to_string(p.linear_num_key_heads) + ").");
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

  // --- output gate. `output_gate_type = self.output_gate_type or
  // self.hidden_act`, then refuse anything outside {sigmoid, silu}
  // (configuration_qwen4_exp.py:193-195).
  //
  // Taken from the RAW text config rather than from `config.output_gate_type`,
  // and the two differences are both real refusals. The shared reader defaults
  // an ABSENT key to "silu" unconditionally (hf_config.cpp:458-460), with no
  // `hidden_act` fallback, so a checkpoint whose `hidden_act` is `gelu` and
  // whose gate key is missing was accepted here and would have run a silu gate
  // on a model trained with something else. And the shared reader collapses
  // `swish` to `silu` before this line runs, where upstream compares the raw
  // string and raises on `swish` -- so the local check as written was a
  // constant false that no config could ever trip.
  std::string gate = OptNonEmptyString(text, "output_gate_type");
  if (gate.empty()) {
    gate = OptNonEmptyString(text, "hidden_act");
    if (gate.empty()) gate = "silu";  // the dataclass default, :114
  }
  if (gate != "silu" && gate != "sigmoid") {
    Refuse("unsupported output gate activation '" + gate +
           "'; expected `sigmoid` or `silu`.");
  }

  // --- rotary. TAKEN FROM THE SHARED READER, which already is the mirror.
  //
  // The previous shape read `partial_rotary_factor` out of the text config here
  // with a default of 0.25, on the stated ground that `Qwen4ExpTextConfig`
  // subclasses `Qwen3_5MoeTextConfig` and inherits that value. It does not, and
  // three facts at the pin say so. The generated -- executed -- class is
  // `class Qwen4ExpTextConfig(PreTrainedConfig)`
  // (configuration_qwen4_exp.py:29). `partial_rotary_factor` is not among its
  // declared fields (:109-164) and the string `0.25` does not occur anywhere in
  // that file; its only two mentions of the name are the validator's own
  // `partial_rotary_factor = (self.rope_parameters or {}).get(
  //     "partial_rotary_factor", 1.0)` (:225) and the `rotary_dim` it feeds
  // (:226). And the modular source shows the bypass is deliberate:
  // `__post_init__` calls `PreTrainedConfig.__post_init__(self, **kwargs)`
  // DIRECTLY (modular_qwen4_exp.py:194), skipping the
  // `kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC`
  // that is the sole source of 0.25 (configuration_qwen3_5_moe.py:124).
  //
  // So the default is 1.0 and the value lives in `rope_parameters`, which is
  // exactly `ParseRopeParameters` (hf_config.cpp:143-205): top level first, the
  // rope dict overriding. That ordering is upstream's too --
  // `convert_rope_params_to_dict` does
  // `self.rope_parameters.setdefault("partial_rotary_factor", <kwarg>)`
  // (modeling_rope_utils.py:755-757), and it runs BEFORE the generic `setattr`
  // loop that would let `standardize_rope_params`:788 overwrite the dict
  // (configuration_utils.py:314 vs :339), so `setdefault` is the whole
  // precedence. `IsQwen35Family` correctly does NOT list `qwen4_exp`, so its
  // default there is 1.0.
  //
  // The local read therefore diverged in BOTH directions: it ACCEPTED a config
  // with no factor at all (upstream: 1.0 -> rotary_dim 256 > indexer_head_dim
  // 128 -> raise) and handed W4 a 64-of-256 slice, and it REFUSED a config with
  // top-level 1.0 and `rope_parameters.partial_rotary_factor` 0.25, which
  // upstream accepts -- the very failure its comment claimed to prevent.
  //
  // NO LOCAL POSITIVITY GUARD, and its absence is measured rather than assumed.
  // The shared reader already refuses anything outside (0, 1]
  // (`hf_config: partial_rotary_factor must be in (0, 1]`), so a local
  // `> 0` check here could never be reached — it would be the same constant
  // false the output-gate check used to be. That bound is itself tighter than
  // upstream, which validates the factor not at all; it belongs to the shared
  // seam, and the row's spec records it there with the sweep row that shows it.
  p.partial_rotary_factor = config.rope_parameters.partial_rotary_factor;
  p.rotary_dim = config.rotary_dim;

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
  //
  // The n-gram fields resolve UNCONDITIONALLY, with upstream's own defaults.
  // They are declared dataclass fields (configuration_qwen4_exp.py:149-157), so
  // they hold those values whether or not any layer uses PLE, and a config that
  // omits them is legal upstream. Defaulting them to 0 inside the PLE branch
  // did two wrong things at once: it REFUSED such a config ("`ngram_size` must
  // be >= 2 ... got 0"), and it left `ngram_heads()` at zero on a PLE-free
  // config, which makes the `head_dim_per_ngram()` this header advertises a
  // division by zero.
  //
  // `ple_embed_dim` defaults to `hidden_size`, set in `__post_init__` (:168).
  p.ple.embed_dim = OptInt(text, "ple_embed_dim", p.hidden_size);
  p.ple.conv_kernel_size = OptInt(text, "ple_conv_kernel_size", 4);
  p.ple.ngram_size = OptInt(text, "ngram_size", 3);
  p.ple.heads_per_ngram = OptInt(text, "heads_per_ngram", 8);
  p.ple.ngram_vocab_size_base = OptInt(text, "ngram_vocab_size_base", 20000000);
  p.ple.make_ngram_vocab_size_divisible_by =
      OptInt(text, "make_ngram_vocab_size_divisible_by", 128);
  p.ple.split_ngram_parts = OptInt(text, "split_ngram_parts", 512);
  p.ple.seed = OptInt(text, "seed", 1234);
  // Stated only by a GGUF-derived config; see the field comment. Read
  // unconditionally alongside the other n-gram fields, so a PLE-free config
  // that happens to carry them is not treated differently from one that does
  // not. W5 (#2031).
  p.ple.head_vocab_sizes = OptIntArray(text, "ple_head_vocab_sizes");

  const std::vector<int64_t> raw_ple = OptIntArray(text, "ple_layer_ids");
  const std::set<int64_t> sorted_unique(raw_ple.begin(), raw_ple.end());
  if (!sorted_unique.empty()) {
    // ORDER MIRRORS UPSTREAM (:233-257): head-count and embedding width first,
    // then the layer-id range, then the layer kind, then EOS. Two violations at
    // once must report the same one upstream reports, or a reader comparing the
    // two runtimes is told to fix a different field.
    //
    // Upstream folds the first into one condition,
    // `ngram_heads <= 0 or self.ple_embed_dim <= 0 or
    //  self.ple_embed_dim % ngram_heads != 0` (:235). The `ngram_size` and
    // `heads_per_ngram` refusals below split that first term so the message
    // names the field the reader has to edit; the accept/reject boundary is
    // identical, because a sub-2 n-gram or a non-positive head count makes
    // `ngram_heads` non-positive either way.
    if (p.ple.ngram_size < 2) {
      Refuse("`ngram_size` must be >= 2 when PLE is enabled, got " +
             std::to_string(p.ple.ngram_size) + ".");
    }
    if (p.ple.heads_per_ngram <= 0) {
      Refuse("`heads_per_ngram` must be > 0 when PLE is enabled, got " +
             std::to_string(p.ple.heads_per_ngram) + ".");
    }
    // 2560 / 16 = 160. A ragged split would silently mis-slice every gathered
    // row, so it is refused rather than truncated. The `<= 0` term is upstream's
    // and is not redundant in C++: `-2560 % 16 == 0`, so a negative width
    // satisfies the divisibility test and `head_dim_per_ngram()` then returns
    // -160.
    const int64_t heads = p.ple.ngram_heads();
    if (heads <= 0 || p.ple.embed_dim <= 0 || p.ple.embed_dim % heads != 0) {
      Refuse("`ple_embed_dim` (" + std::to_string(p.ple.embed_dim) +
             ") must be > 0 and divisible by the n-gram head count (" +
             std::to_string(heads) + ").");
    }
    // LOCAL, tighter than upstream, which does not validate the PLE conv.
    if (p.ple.conv_kernel_size <= 0) {
      Refuse("`ple_conv_kernel_size` must be > 0, got " +
             std::to_string(p.ple.conv_kernel_size) + ".");
    }
    // A STATED head-size list has to have one entry per n-gram head and no
    // non-positive entry, or the row count derived from it is wrong and the
    // n-gram gather reads outside the table it sizes. Silence is legal (a
    // config.json states none), a WRONG list is not. W5 (#2031).
    if (!p.ple.head_vocab_sizes.empty()) {
      if (static_cast<int64_t>(p.ple.head_vocab_sizes.size()) != heads) {
        Refuse("`ple_head_vocab_sizes` has " +
               std::to_string(p.ple.head_vocab_sizes.size()) +
               " entries but the n-gram head count is " +
               std::to_string(heads) + ".");
      }
      for (int64_t sz : p.ple.head_vocab_sizes) {
        if (sz <= 0) {
          Refuse("`ple_head_vocab_sizes` must be all positive, got " +
                 std::to_string(sz) + ".");
        }
      }
      if (p.ple.make_ngram_vocab_size_divisible_by <= 0) {
        Refuse("`make_ngram_vocab_size_divisible_by` must be > 0, got " +
               std::to_string(p.ple.make_ngram_vocab_size_divisible_by) + ".");
      }
    }
    for (int64_t one_based : sorted_unique) {
      if (one_based < 1 || one_based > p.num_hidden_layers) {
        Refuse("`ple_layer_ids` must contain one-indexed ids in [1, " +
               std::to_string(p.num_hidden_layers) + "], got " +
               std::to_string(one_based) + ".");
      }
    }
    // Safe to index only because the range check above already ran over the
    // whole set, exactly as upstream orders it.
    for (int64_t one_based : sorted_unique) {
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
    // `if self.eos_token_id is None or isinstance(self.eos_token_id, list) and
    //  not self.eos_token_id` (:256-257). This is load-bearing rather than
    // hygiene: the n-gram history is built with `_shift_right_ignore_eos`
    // (modeling_qwen4_exp.py:1095), so EOS is a SEGMENT BOUNDARY in the hashed
    // n-gram construction, and the published GGUF carries it as a first-class
    // PLE key (`qwen4exp.ple.eos_token_id`). A config without one cannot have
    // its n-gram ids constructed at all.
    const auto eos = text.find("eos_token_id");
    if (eos == text.end() || eos->is_null() ||
        (eos->is_array() && eos->empty())) {
      Refuse("`eos_token_id` must be set when PLE layers are enabled.");
    }
    // W5 (#2031) RESOLVES it rather than only asserting its presence, because
    // the loader and the n-gram hash both need the VALUE and neither may pick
    // its own element of a list. Upstream takes element [0]
    // (`eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id`,
    // modeling_qwen4_exp.py, inside `Qwen4ExpTextModel.forward`), so [0] it is.
    //
    // llama.cpp #27742's converter writes `int(eos[-1])` into
    // `qwen4exp.ple.eos_token_id` — the LAST element, not the first. On a
    // checkpoint whose list has one entry the two coincide and nothing shows;
    // on a list of several they disagree, and the disagreement is invisible
    // because both runtimes produce fluent text from different n-gram segment
    // boundaries. This resolver follows the ALGORITHM oracle, which is the
    // polarity AGENTS.md requires, and the spec's `## Owed` records the
    // divergence against the container so it is not rediscovered.
    if (eos->is_array()) {
      const auto& first = eos->front();
      if (!first.is_number_integer() && !first.is_number_unsigned()) {
        Refuse("`eos_token_id[0]` must be an integer.");
      }
      p.eos_token_id = first.get<int64_t>();
    } else if (eos->is_number_integer() || eos->is_number_unsigned()) {
      p.eos_token_id = eos->get<int64_t>();
    } else {
      Refuse("`eos_token_id` must be an integer or a non-empty list of them.");
    }
    // The bound is the whole reason this value is resolved here. It reaches a
    // `uint64_t` multiply against `layer_multipliers[i]`, and the product is
    // bounded below 2^63 only while every id in the mix is under `vocab_size`.
    // Out of range it overflows and diverges from the oracle in SILENCE.
    if (p.eos_token_id < 0 || p.eos_token_id >= p.vocab_size) {
      Refuse("`eos_token_id` must be in [0, vocab_size) because it seeds the "
             "n-gram hash on the first token of every sequence, got " +
             std::to_string(p.eos_token_id) + " against a vocab_size of " +
             std::to_string(p.vocab_size) + ".");
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
