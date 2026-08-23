// dots3-note (`Dots3NoteForCausalLM`) W1 — config resolution + the on-disk name
// map + the refusing forward. Issue #699, spec `.agents/specs/dots3-note.md`.
//
// Upstream anchors are in `dots3_note.h`, all read at vLLM `origin/main` =
// `c205726108df54bb6fbf15b19e725a4a3add2b18`. dots3-note does NOT exist at our
// parity pin `555967922`.
//
// WHY THE CONFIG LIVES HERE AND NOT IN `hf_config.cpp`. Spec §3.2 item 6 named
// `hf_config.cpp` as the home for "dots3_note config parsing". It is not, and
// the reason is a rule rather than a preference: AGENTS.md forbids "a surface
// that every PR must write", and `hf_config.cpp` is the shared container reader
// every architecture would otherwise have to edit. dots3-note contributes ~30
// architecture scalars, none of which any other model reads, and every other
// model in this tree keeps exactly that kind of field in its own TU
// (`DeepseekV4Params` in `deepseek_v4_weights.cpp`, `NemotronHParams` in
// `nemotron_h_weights.cpp`, `MuseGlimmerParams`). `hf_config.cpp` needs NO edit
// for this checkpoint — it parses `dots-studio/dots3-note-prev`'s `config.json`
// unchanged, and it must NOT normalize `sliding_window_size` into the typed
// `sliding_window`, because upstream does not either: the window is handed to
// one MLAAttention per sliding layer (`model.py`::Dots3NoteSlidingAttention
// :457), not to the model-level config. Recorded in the spec's §4/W1.
#include "vllm/model_executor/models/dots3_note.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

// --- raw config.json readers (dots3 keys are not typed on HfConfig) ---
const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &(*it);
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<int64_t>() : fallback;
}
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<double>() : fallback;
}
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_boolean()) ? f->get<bool>() : fallback;
}
std::string RawString(const nlohmann::json& doc, const char* key,
                      const std::string& fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_string()) ? f->get<std::string>() : fallback;
}

std::string LayerPrefix(int64_t layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

}  // namespace

Dots3NoteParams ParseDots3NoteParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  Dots3NoteParams p;

  // --- shared geometry ---
  p.hidden_size = config.hidden_size > 0 ? config.hidden_size
                                         : RawInt(raw, "hidden_size", 0);
  p.num_hidden_layers = config.num_hidden_layers > 0
                            ? config.num_hidden_layers
                            : RawInt(raw, "num_hidden_layers", 0);
  p.vocab_size =
      config.vocab_size > 0 ? config.vocab_size : RawInt(raw, "vocab_size", 0);
  p.intermediate_size = config.intermediate_size > 0
                            ? config.intermediate_size
                            : RawInt(raw, "intermediate_size", 0);
  p.rms_norm_eps = RawDouble(raw, "rms_norm_eps", 1e-5);
  p.max_position_embeddings = RawInt(raw, "max_position_embeddings", 0);
  p.tie_word_embeddings = RawBool(raw, "tie_word_embeddings", false);

  for (const std::string& kind : config.layer_types) {
    if (kind == "full_attention") {
      p.layer_types.push_back(Dots3NoteLayerKind::kFullAttention);
    } else if (kind == "sliding_attention") {
      p.layer_types.push_back(Dots3NoteLayerKind::kSlidingAttention);
    } else {
      VT_CHECK(false,
               "dots3-note: layer_types carries \"" + kind +
                   "\", but this architecture has exactly two attention "
                   "classes (full_attention, sliding_attention) — "
                   "model.py:503");
    }
  }

  // --- MoE ---
  p.n_routed_experts = RawInt(raw, "n_routed_experts", 0);
  p.num_experts_per_tok = RawInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = RawInt(raw, "moe_intermediate_size", 0);
  p.n_shared_experts = RawInt(raw, "n_shared_experts", 0);
  p.first_k_dense_replace = RawInt(raw, "first_k_dense_replace", 3);
  p.moe_layer_freq = RawInt(raw, "moe_layer_freq", 1);
  p.norm_topk_prob = RawBool(raw, "norm_topk_prob", true);
  p.routed_scaling_factor = RawDouble(raw, "routed_scaling_factor", 1.0);
  p.scoring_func = RawString(raw, "scoring_func", "sigmoid");
  p.topk_method = RawString(raw, "topk_method", "noaux_tc");

  // ★ §4 TRAP 1+2 — `n_group` / `topk_group`, ABSENT from the released
  // config.json. `Dots3NoteConfig.__init__` (configs/dots3_note.py:18-19)
  // does `kwargs.setdefault("n_group", 1)` / `("topk_group", 1)` BEFORE
  // `super().__init__`, so DeepseekV3Config's own defaults of 8 and 4
  // (transformers configuration_deepseek_v3.py:168-169) never apply. The
  // fallback here is therefore 1, NOT 8/4 and NOT 0: reading the parent's
  // default would regroup the noaux_tc router at every MoE layer and change
  // which experts are selected, with no shape change and no error.
  p.n_group = RawInt(raw, "n_group", 1);
  p.topk_group = RawInt(raw, "topk_group", 1);

  // --- DSA lightning indexer ---
  p.index_n_heads = RawInt(raw, "index_n_heads", 0);
  p.index_head_dim = RawInt(raw, "index_head_dim", 0);
  p.index_topk = RawInt(raw, "index_topk", 0);
  // ★ §4 TRAP 3 — `indexer_rope_interleave`, ABSENT from the released
  // config.json, defaulted TRUE by configs/dots3_note.py:23. Consumed as
  // `is_neox_style = not indexer_rope_interleave`
  // (deepseek_v2.py::DeepseekV2MLAAttention:1148), whose own `getattr` default
  // is False — i.e. DeepSeek-V3.2's split-half NeoX. Taking THAT default here
  // would rotate a different set of learned coordinates.
  p.indexer_rope_interleave = RawBool(raw, "indexer_rope_interleave", true);

  // ★ §4 TRAP 4 — `num_nextn_predict_layers`, ABSENT from the released
  // config.json, defaulted 1 by configs/dots3_note.py:24. DeepseekV3Config has
  // no such field, so an absent key would otherwise read 0 and the entire
  // nextn tail (`model.layers.46.*`) would go unclaimed by the loader.
  p.num_nextn_predict_layers = RawInt(raw, "num_nextn_predict_layers", 1);

  // ★ §4 TRAP 5 — present in the config.json, but it is NOT what our DeepSeek
  // MLA assumes. model.py:305-307 (full) and :438-443 (sliding).
  p.apply_mla_qkv_lora_rescale = RawBool(raw, "apply_mla_qkv_lora_rescale", false);

  // --- the two attention geometries ---
  p.full.num_attention_heads = config.num_attention_heads > 0
                                   ? config.num_attention_heads
                                   : RawInt(raw, "num_attention_heads", 0);
  p.full.q_lora_rank = RawInt(raw, "q_lora_rank", 0);
  p.full.kv_lora_rank = RawInt(raw, "kv_lora_rank", 0);
  p.full.qk_nope_head_dim = RawInt(raw, "qk_nope_head_dim", 0);
  p.full.qk_rope_head_dim = RawInt(raw, "qk_rope_head_dim", 0);
  p.full.v_head_dim = RawInt(raw, "v_head_dim", 0);
  p.full.rope_theta = RawDouble(raw, "rope_theta", 10000.0);
  p.full.sliding_window = 0;
  p.full.attention_gate_type = RawString(raw, "attention_gate_type", "headwise");
  p.full.has_indexer = true;

  p.swa.num_attention_heads = RawInt(raw, "swa_num_attention_heads", 0);
  p.swa.q_lora_rank = RawInt(raw, "swa_q_lora_rank", 0);
  p.swa.kv_lora_rank = RawInt(raw, "swa_kv_lora_rank", 0);
  p.swa.qk_nope_head_dim = RawInt(raw, "swa_qk_nope_head_dim", 0);
  p.swa.qk_rope_head_dim = RawInt(raw, "swa_qk_rope_head_dim", 0);
  p.swa.v_head_dim = RawInt(raw, "swa_v_head_dim", 0);
  // ★ §4 TRAP 6a — the sliding layers carry their OWN theta (5e4), NOT the
  // model-level 8e7 (model.py:404-407).
  p.swa.rope_theta = RawDouble(raw, "swa_rope_theta", 10000.0);
  p.swa.sliding_window = RawInt(raw, "sliding_window_size", 0);
  p.swa.attention_gate_type =
      RawString(raw, "swa_attention_gate_type", "headwise");
  p.swa.has_indexer = false;

  // ★ §4 TRAP 6b — the RoPE LAYOUT. BOTH geometries are GPT-J (interleaved
  // adjacent pairs), i.e. `is_neox_style=False`: the sliding layers say so
  // literally (model.py:408) and the full layers inherit it from
  // `deepseek_v2.py`::DeepseekV2MLAAttention:1093-1097, which hard-codes
  // `is_neox_style=False`. This CORRECTS the spec's W0 §4 item 6, which read
  // it as sliding-only. Note the polarity clash with the indexer, which is the
  // whole point of trap 3: with `indexer_rope_interleave=True` the indexer rope
  // is ALSO non-NeoX, where DeepSeek-V3.2 leaves it NeoX and disagreeing with
  // its own MLA rope.
  p.full.rope_is_neox_style = false;
  p.swa.rope_is_neox_style = false;

  const auto rescale = [&](Dots3NoteAttnParams& a) {
    if (!p.apply_mla_qkv_lora_rescale) {
      a.q_lora_scale = 1.0;
      a.kv_lora_scale = 1.0;
      return;
    }
    VT_CHECK(a.q_lora_rank > 0 && a.kv_lora_rank > 0,
             "dots3-note: apply_mla_qkv_lora_rescale needs positive LoRA ranks");
    a.q_lora_scale = std::sqrt(static_cast<double>(p.hidden_size) /
                               static_cast<double>(a.q_lora_rank));
    a.kv_lora_scale = std::sqrt(static_cast<double>(p.hidden_size) /
                                static_cast<double>(a.kv_lora_rank));
  };
  rescale(p.full);
  rescale(p.swa);

  // --- validation: refuse by name, never silently reshape ---
  VT_CHECK(p.hidden_size > 0, "dots3-note: hidden_size must be positive");
  VT_CHECK(p.num_hidden_layers > 0,
           "dots3-note: num_hidden_layers must be positive");
  VT_CHECK(p.vocab_size > 0, "dots3-note: vocab_size must be positive");
  VT_CHECK(static_cast<int64_t>(p.layer_types.size()) == p.num_hidden_layers,
           "dots3-note: layer_types has " +
               std::to_string(p.layer_types.size()) + " entries but " +
               std::to_string(p.num_hidden_layers) +
               " layers — the hybrid schedule is structural, not optional");
  VT_CHECK(p.n_routed_experts > 0,
           "dots3-note: n_routed_experts must be positive (this is a MoE arch)");
  VT_CHECK(p.scoring_func == "sigmoid",
           "dots3-note: only scoring_func='sigmoid' is scoped; got '" +
               p.scoring_func + "'");
  VT_CHECK(p.topk_method == "noaux_tc",
           "dots3-note: only topk_method='noaux_tc' is scoped; got '" +
               p.topk_method + "'");
  // The GROUPED router arm is refused rather than approximated. Our noaux_tc
  // router is gated at DeepSeek-V3's grouped dims, and dots3-note is trained
  // UNGROUPED (§4 trap 1): a checkpoint that really did carry n_group>1 would
  // need the grouped path wired and gated, which is W5's work, not a silent
  // fall-through here.
  VT_CHECK(p.n_group == 1 && p.topk_group == 1,
           "dots3-note: only the UNGROUPED noaux_tc router (n_group=1, "
           "topk_group=1) is scoped — configs/dots3_note.py:18-19; got n_group=" +
               std::to_string(p.n_group) +
               ", topk_group=" + std::to_string(p.topk_group));
  VT_CHECK(p.full.attention_gate_type == "headwise" &&
               p.swa.attention_gate_type == "headwise",
           "dots3-note: only the headwise attention gate is scoped "
           "(model.py:191); got attention_gate_type='" +
               p.full.attention_gate_type + "', swa_attention_gate_type='" +
               p.swa.attention_gate_type + "'");
  VT_CHECK(p.swa.sliding_window > 0,
           "dots3-note: sliding_window_size must be positive — 33 of the 46 "
           "layers are windowed (model.py:457)");
  VT_CHECK(p.swa.qk_rope_head_dim == p.full.qk_rope_head_dim,
           "dots3-note: the two geometries must share qk_rope_head_dim; got " +
               std::to_string(p.full.qk_rope_head_dim) + " and " +
               std::to_string(p.swa.qk_rope_head_dim));
  VT_CHECK(p.physical_latent_row() >= p.full.latent_row(),
           "dots3-note: the padded physical MLA row (" +
               std::to_string(p.physical_latent_row()) +
               ") must cover the full layers' logical row (" +
               std::to_string(p.full.latent_row()) + ") — model.py:283, :213");
  VT_CHECK(p.index_topk > 0 && p.index_n_heads > 0 && p.index_head_dim > 0,
           "dots3-note: the DSA lightning indexer geometry is structural on the "
           "full layers (index_topk / index_n_heads / index_head_dim)");
  return p;
}

void ParseDots3NoteConfig(const HfConfig& config) {
  // The resolve IS the validation: it throws on everything unrepresentable.
  (void)ParseDots3NoteParams(config);
}

std::vector<Dots3NoteTensor> EnumerateDots3NoteTensors(
    const Dots3NoteParams& p, const std::vector<int64_t>& layers,
    bool include_root, bool include_nextn) {
  std::vector<Dots3NoteTensor> out;
  const auto add = [&out](std::string name, const char* consumer) {
    out.push_back(Dots3NoteTensor{std::move(name), consumer});
  };

  if (include_root) {
    add("model.embed_tokens.weight", "embed_tokens");
    add("model.norm.weight", "final_norm");
    if (!p.tie_word_embeddings) add("lm_head.weight", "lm_head");
  }

  // One backbone layer. `kind` decides the attention tensor set; `moe` decides
  // the MLP tensor set. Both are read from the schedule, never guessed.
  const auto emit_layer = [&](int64_t layer, Dots3NoteLayerKind kind, bool moe) {
    const std::string pre = LayerPrefix(layer);
    add(pre + "input_layernorm.weight", "layer_norm");
    add(pre + "post_attention_layernorm.weight", "layer_norm");

    // MLA. `q_a_proj` and `kv_a_proj_with_mqa` ship SEPARATELY; upstream fuses
    // them into `fused_qkv_a_proj` at load time (model.py:150,
    // DeepSeekV2FusedQkvAProjLinear), so the module-level view hides them.
    const char* attn = kind == Dots3NoteLayerKind::kFullAttention
                           ? "mla_full"
                           : "mla_sliding";
    add(pre + "self_attn.q_a_proj.weight", attn);
    add(pre + "self_attn.q_a_layernorm.weight", attn);
    add(pre + "self_attn.q_b_proj.weight", attn);
    add(pre + "self_attn.kv_a_proj_with_mqa.weight", attn);
    add(pre + "self_attn.kv_a_layernorm.weight", attn);
    add(pre + "self_attn.kv_b_proj.weight", attn);
    add(pre + "self_attn.o_proj.weight", attn);
    // The two dots3-only attention tensors: the headwise gate (model.py:294)
    // and the extra RMSNorm over the 64-wide rope-only k slice (model.py:299,
    // consumed at :160). DeepSeek has neither.
    add(pre + "self_attn.g_proj.weight", "attn_gate_headwise");
    add(pre + "self_attn.k_rope_only_layernorm.weight", "k_rope_only_norm");

    // The DSA lightning indexer exists ONLY on the full layers
    // (model.py:432-434 sets it to None on the sliding class).
    if (kind == Dots3NoteLayerKind::kFullAttention) {
      add(pre + "self_attn.indexer.wq_b.weight", "dsa_indexer");
      add(pre + "self_attn.indexer.wk.weight", "dsa_indexer");
      add(pre + "self_attn.indexer.k_norm.weight", "dsa_indexer");
      add(pre + "self_attn.indexer.k_norm.bias", "dsa_indexer");
      add(pre + "self_attn.indexer.weights_proj.weight", "dsa_indexer");
    }

    if (!moe) {
      add(pre + "mlp.gate_proj.weight", "dense_mlp");
      add(pre + "mlp.up_proj.weight", "dense_mlp");
      add(pre + "mlp.down_proj.weight", "dense_mlp");
      return;
    }
    add(pre + "mlp.gate.weight", "moe_router");
    // noaux_tc's per-expert bias. It ships F32 while every other tensor in the
    // language tower is BF16 — recorded here because a loader that assumed one
    // dtype for the whole checkpoint would read it wrong (porting.md, "mirror
    // the memory format").
    add(pre + "mlp.gate.e_score_correction_bias", "moe_router");
    for (int64_t e = 0; e < p.n_routed_experts; ++e) {
      const std::string ep = pre + "mlp.experts." + std::to_string(e) + ".";
      add(ep + "gate_proj.weight", "moe_expert");
      add(ep + "up_proj.weight", "moe_expert");
      add(ep + "down_proj.weight", "moe_expert");
    }
    if (p.n_shared_experts > 0) {
      add(pre + "mlp.shared_experts.gate_proj.weight", "moe_shared_expert");
      add(pre + "mlp.shared_experts.up_proj.weight", "moe_shared_expert");
      add(pre + "mlp.shared_experts.down_proj.weight", "moe_shared_expert");
    }
  };

  for (const int64_t layer : layers) {
    VT_CHECK(layer >= 0 && layer < p.num_hidden_layers,
             "dots3-note: backbone layer " + std::to_string(layer) +
                 " is outside [0, " + std::to_string(p.num_hidden_layers) + ")");
    emit_layer(layer, p.kind_of(layer), p.is_moe_layer(layer));
  }

  if (include_nextn && p.num_nextn_predict_layers > 0) {
    // The nextn (MTP) tail. `Dots3NoteMultiTokenPredictor` builds
    // `num_nextn_predict_layers` layers starting at `num_hidden_layers`
    // (mtp.py:91-101), each an enorm/hnorm/eh_proj/shared_head wrapper around a
    // whole `Dots3NoteDecoderLayer` (mtp.py:38-56).
    //
    // TWO facts about that block are taken from the RELEASED CHECKPOINT rather
    // than from upstream, because upstream's `config.layer_types[layer_idx]`
    // (model.py:503) has no entry at index 46 and so does not answer either:
    //   * it carries the SLIDING attention tensor set — no `indexer.*`, and
    //     `q_b_proj` is [16384, 1024] = 64 heads x (192+64), the swa geometry;
    //   * its MLP is DENSE, which upstream's `layer_idx < num_hidden_layers`
    //     MoE predicate (model.py:512) does agree with.
    // `shared_head.head.weight` is absent, matching `has_own_lm_head = False`
    // (mtp.py:142); `model.mtp.embed_tokens.weight` is present, matching
    // `has_own_embed_tokens = True` (mtp.py:141).
    add("model.mtp.embed_tokens.weight", "mtp_embed_tokens");
    for (int64_t i = 0; i < p.num_nextn_predict_layers; ++i) {
      const int64_t layer = p.num_hidden_layers + i;
      const std::string pre = LayerPrefix(layer);
      add(pre + "enorm.weight", "mtp_norm");
      add(pre + "hnorm.weight", "mtp_norm");
      add(pre + "eh_proj.weight", "mtp_eh_proj");
      emit_layer(layer, Dots3NoteLayerKind::kSlidingAttention, /*moe=*/false);
      add(pre + "shared_head.norm.weight", "mtp_shared_head");
    }
  }
  return out;
}

std::vector<Dots3NoteTensor> EnumerateDots3NoteTensors(
    const Dots3NoteParams& p) {
  std::vector<int64_t> all;
  all.reserve(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) all.push_back(l);
  return EnumerateDots3NoteTensors(p, all, /*include_root=*/true,
                                   /*include_nextn=*/true);
}

Dots3NoteAccounting AccountDots3NoteTensors(
    const Dots3NoteParams& p, const std::vector<std::string>& present,
    const std::vector<int64_t>& expected_layers) {
  Dots3NoteAccounting acc;
  const std::vector<Dots3NoteTensor> claimed = EnumerateDots3NoteTensors(
      p, expected_layers, /*include_root=*/true, /*include_nextn=*/true);

  std::unordered_set<std::string> claimed_names;
  for (const Dots3NoteTensor& t : claimed) {
    VT_CHECK(!t.consumer.empty(),
             "dots3-note: enumerated " + t.name + " with no named consumer");
    if (!claimed_names.insert(t.name).second) acc.duplicated.push_back(t.name);
  }

  const std::unordered_set<std::string> on_disk(present.begin(), present.end());
  for (const std::string& name : claimed_names) {
    if (on_disk.count(name) == 0) acc.missing.push_back(name);
  }

  const auto starts_with = [](const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
  };
  for (const std::string& name : present) {
    if (claimed_names.count(name) != 0) {
      ++acc.language;
    } else if (starts_with(name, "vision_encoder.")) {
      ++acc.vision;  // W6, named deferral
    } else if (starts_with(name, "audio_encoder.")) {
      ++acc.audio;  // W7, named deferral
    } else {
      acc.unaccounted.push_back(name);
    }
  }
  // Deterministic order, so a refusal names the same tensor on every run.
  std::sort(acc.missing.begin(), acc.missing.end());
  std::sort(acc.unaccounted.begin(), acc.unaccounted.end());
  std::sort(acc.duplicated.begin(), acc.duplicated.end());
  return acc;
}

Dots3NoteWeights LoadDots3NoteWeights(const std::vector<SafetensorsFile>& shards,
                                       const HfConfig& config) {
  Dots3NoteWeights w;
  w.params = ParseDots3NoteParams(config);
  VT_CHECK(!shards.empty(), "dots3-note: safetensors model source has no shards");

  std::vector<std::string> present;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) present.push_back(name);
  }
  std::vector<int64_t> backbone;
  backbone.reserve(static_cast<size_t>(w.params.num_hidden_layers));
  for (int64_t l = 0; l < w.params.num_hidden_layers; ++l) backbone.push_back(l);
  w.accounting = AccountDots3NoteTensors(w.params, present, backbone);

  VT_CHECK(w.accounting.duplicated.empty(),
           "dots3-note: the name map claims " + w.accounting.duplicated.front() +
               " twice (" + std::to_string(w.accounting.duplicated.size()) +
               " total)");
  VT_CHECK(w.accounting.missing.empty(),
           "dots3-note: the checkpoint is missing " +
               w.accounting.missing.front() + " (" +
               std::to_string(w.accounting.missing.size()) +
               " enumerated tensors are absent) — a weight nobody loads reads "
               "as zeros");
  VT_CHECK(w.accounting.unaccounted.empty(),
           "dots3-note: no consumer claims " + w.accounting.unaccounted.front() +
               " (" + std::to_string(w.accounting.unaccounted.size()) +
               " unaccounted tensors) — see .agents/specs/dots3-note.md W2 and "
               "issue #699");

  // W2 owns the materialization. Returning an UNMATERIALIZED model rather than
  // throwing is deliberate: the accounting above is a real production result
  // worth having, and the refusal belongs at the forward, where it names the
  // brick that owes the maths. `materialized` stays false, and every path that
  // would read a weight goes through `Dots3NoteForward`, which refuses.
  w.materialized = false;
  return w;
}

ForwardLogits Dots3NoteModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Dots3NoteWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)token_ids;
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)queue;
  (void)logits_indices;
  VT_CHECK(false,
           std::string(
               "Dots3NoteForCausalLM forward: not ported — the language tower "
               "needs the sliding-window MLA over 33 of 46 layers (W4), the "
               "headwise-gated MLA with the lora rescales and "
               "k_rope_only_layernorm (W3), the ungrouped noaux_tc MoE (W5), "
               "and the vision/audio towers (W6/W7). Host weights are ") +
               (weights.materialized ? "materialized" : "not materialized") +
               ". See .agents/specs/dots3-note.md and issue #699.");
}

v1::KVCacheConfig MakeDots3NoteKVCache(const HfConfig& config, int block_size,
                                        int num_blocks) {
  // ONE MLA group at the PADDED physical row, mirroring
  // `model.py`::Dots3NotePaddedMLAAttention.get_kv_cache_spec (:213-217): the
  // full layers report `physical_head_size = swa_kv_lora_rank +
  // swa_qk_rope_head_dim` (1088) rather than their own logical 576, so both
  // attention classes allocate one block shape and the full layers narrow on
  // read (`attention.py`::Dots3NotePaddedSparseImpl._logical_cache). Reporting
  // 576 here would under-allocate every sliding layer by 512 rows.
  //
  // The heterogeneous per-layer group split, the DSA index cache and the
  // windowed metadata are W4's, and are NOT represented here.
  const Dots3NoteParams p = ParseDots3NoteParams(config);
  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<v1::MLAAttentionSpec>(
          block_size, static_cast<int>(p.physical_latent_row()),
          v1::ResolveKvCacheDType()));
  return kv;
}

}  // namespace vllm
