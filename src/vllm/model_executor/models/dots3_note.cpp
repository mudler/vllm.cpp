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
//
// ★ THE POLARITY OF THESE READERS IS THE POINT (#1805 review finding F1).
// W1 shipped them as SILENT FALLBACKS: an absent `swa_rope_theta` read 10000.0
// and an absent `apply_mla_qkv_lora_rescale` read false, so a config missing
// either key parsed clean and then rotated 33 of 46 layers at the wrong theta,
// or dropped both LoRA rescales, with no shape change and no error. A wrong
// JSON TYPE fell into the same fallback. Spec §4 item 4 asks for exactly the
// opposite — "check each field we read rather than assuming the JSON is
// complete" — and §6.4 records that no oracle for this model runs on any host
// we own, so nothing downstream could ever catch the substitution.
//
// So a field this port READS is now one of exactly two things, and never a
// third:
//
//   Req*  REQUIRED. Absent or wrong-typed REFUSES BY NAME. Every one of these
//         is a key the released `config.json` carries.
//   Opt*  OPTIONAL, with a default taken from a NAMED upstream site — the four
//         `Dots3NoteConfig.__init__` setdefaults and the two `getattr`s in
//         `model.py`. A wrong TYPE still refuses: upstream would carry a string
//         into arithmetic, not substitute its default.
//
// For the subset of required fields that ARE `DeepseekV3Config` dataclass
// fields, refusing is STRICTER than upstream, which would silently substitute
// V3's default. That is deliberate and it is the same call `hf_config.cpp`
// already makes for `output_gate_type`: a silent default is a numerics change
// no gate we have can see. The refusal message says which upstream behaviour it
// is standing in for, so a reader can tell the two cases apart.
const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &(*it);
}

// `upstream` names what upstream would have done with the key absent, so the
// refusal explains itself rather than only complaining.
[[noreturn]] void RefuseMissing(const char* key, const char* upstream) {
  VT_CHECK(false, std::string("dots3-note: config.json has no \"") + key +
                      "\" and this port reads it — " + upstream +
                      ". A substituted default here is a silent numerics change "
                      "and there is no oracle for this model on any host we own "
                      "(spec §4 item 4, §6.4). See .agents/specs/dots3-note.md "
                      "and issue #699.");
  throw std::runtime_error("unreachable");
}

[[noreturn]] void RefuseType(const char* key, const nlohmann::json& value,
                             const char* want) {
  VT_CHECK(false, std::string("dots3-note: config.json field \"") + key +
                      "\" is " + value.type_name() + " (" + value.dump() +
                      "), expected " + want +
                      " — a wrong type is never silently defaulted here. See "
                      ".agents/specs/dots3-note.md and issue #699.");
  throw std::runtime_error("unreachable");
}

int64_t ReqInt(const nlohmann::json& doc, const char* key, const char* upstream) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr) RefuseMissing(key, upstream);
  if (!f->is_number_integer()) RefuseType(key, *f, "an integer");
  return f->get<int64_t>();
}
double ReqDouble(const nlohmann::json& doc, const char* key,
                 const char* upstream) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr) RefuseMissing(key, upstream);
  if (!f->is_number()) RefuseType(key, *f, "a number");
  return f->get<double>();
}
bool ReqBool(const nlohmann::json& doc, const char* key, const char* upstream) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr) RefuseMissing(key, upstream);
  if (!f->is_boolean()) RefuseType(key, *f, "a boolean");
  return f->get<bool>();
}
std::string ReqString(const nlohmann::json& doc, const char* key,
                      const char* upstream) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr) RefuseMissing(key, upstream);
  if (!f->is_string()) RefuseType(key, *f, "a string");
  return f->get<std::string>();
}

// Optional, with an upstream-anchored default. Absent takes the default;
// present-but-wrong-typed still refuses.
int64_t OptInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr) return fallback;
  if (!f->is_number_integer()) RefuseType(key, *f, "an integer");
  return f->get<int64_t>();
}
double OptDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr) return fallback;
  if (!f->is_number()) RefuseType(key, *f, "a number");
  return f->get<double>();
}
bool OptBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr) return fallback;
  if (!f->is_boolean()) RefuseType(key, *f, "a boolean");
  return f->get<bool>();
}

// The upstream behaviour each REQUIRED key's refusal stands in for. Three
// distinct cases, and the message says which, because they are not equally
// surprising: an AttributeError is loud upstream too, a substituted
// DeepseekV3Config default is not, and an absent `index_topk` silently switches
// upstream off the V3.2 sparse path entirely
// (`deepseek_v2.py`::DeepseekV2MLAAttention, `self.is_v32 = hasattr(config,
// "index_topk")`).
constexpr const char* kUpstreamRaises =
    "upstream reads it as a plain attribute and raises AttributeError";
constexpr const char* kUpstreamV3Default =
    "upstream would silently substitute DeepseekV3Config's own default, which "
    "this checkpoint does not use";
constexpr const char* kUpstreamDisablesDsa =
    "upstream tests `hasattr(config, \"index_topk\")` and would silently take "
    "the NON-sparse path instead";

std::string LayerPrefix(int64_t layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

}  // namespace

Dots3NoteParams ParseDots3NoteParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  Dots3NoteParams p;

  // Every field below is read from `config.raw` rather than from HfConfig's
  // typed mirrors, so that ABSENT and PRESENT-BUT-ZERO stay distinguishable:
  // the typed fields collapse both to 0, which is the state finding F1 is
  // about. dots3-note's released `config.json` is FLAT — no `text_config`,
  // `llm_config` or `thinker_config` wrapper — so the raw document IS the text
  // config here. A wrapped layout is refused rather than read at the wrong
  // level, because a config that silently deserializes to defaults is a
  // wrong-shaped model with no error (porting-a-model.md §1).
  VT_CHECK(!raw.contains("text_config") && !raw.contains("llm_config") &&
               !raw.contains("thinker_config"),
           "dots3-note: this config nests its text fields under a wrapper key, "
           "which the released dots-studio/dots3-note-prev layout does not. "
           "Reading it at the top level would silently produce an all-defaults "
           "model. See .agents/specs/dots3-note.md and issue #699.");

  // --- shared geometry ---
  p.hidden_size = ReqInt(raw, "hidden_size", kUpstreamV3Default);
  p.num_hidden_layers = ReqInt(raw, "num_hidden_layers", kUpstreamV3Default);
  p.vocab_size = ReqInt(raw, "vocab_size", kUpstreamV3Default);
  p.intermediate_size = ReqInt(raw, "intermediate_size", kUpstreamV3Default);
  p.rms_norm_eps = ReqDouble(raw, "rms_norm_eps", kUpstreamV3Default);
  p.max_position_embeddings =
      ReqInt(raw, "max_position_embeddings", kUpstreamV3Default);
  // PretrainedConfig's own default is False and the checkpoint agrees, so an
  // absent key changes nothing about which tensors are read.
  p.tie_word_embeddings = OptBool(raw, "tie_word_embeddings", false);

  // `layer_types` is read as `config.layer_types[layer_idx]` (model.py:503),
  // so an absent schedule is an upstream AttributeError, and a schedule of the
  // wrong length is caught below.
  const nlohmann::json* layer_types = Field(raw, "layer_types");
  if (layer_types == nullptr) RefuseMissing("layer_types", kUpstreamRaises);
  if (!layer_types->is_array()) {
    RefuseType("layer_types", *layer_types, "an array of strings");
  }
  for (const nlohmann::json& entry : *layer_types) {
    if (!entry.is_string()) RefuseType("layer_types", entry, "a string");
    const std::string kind = entry.get<std::string>();
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
  p.n_routed_experts = ReqInt(raw, "n_routed_experts", kUpstreamV3Default);
  p.num_experts_per_tok = ReqInt(raw, "num_experts_per_tok", kUpstreamV3Default);
  p.moe_intermediate_size =
      ReqInt(raw, "moe_intermediate_size", kUpstreamV3Default);
  p.n_shared_experts = ReqInt(raw, "n_shared_experts", kUpstreamV3Default);
  // DeepseekV3Config defaults this to 3 and the checkpoint ships 1, so an
  // absent key would make layers 1 and 2 dense that are really MoE — 512
  // unclaimed expert tensors per layer, and a different network.
  p.first_k_dense_replace =
      ReqInt(raw, "first_k_dense_replace", kUpstreamV3Default);
  p.norm_topk_prob = ReqBool(raw, "norm_topk_prob", kUpstreamV3Default);
  p.scoring_func = ReqString(raw, "scoring_func", kUpstreamV3Default);
  p.topk_method = ReqString(raw, "topk_method", kUpstreamV3Default);
  // The two genuine `getattr` defaults on this path, and the only two:
  // model.py:513 `getattr(config, "moe_layer_freq", 1)` and model.py:546
  // `getattr(config, "routed_scaling_factor", 1.0)`. Absent takes the default
  // exactly as upstream does; a wrong type still refuses.
  p.moe_layer_freq = OptInt(raw, "moe_layer_freq", 1);
  p.routed_scaling_factor = OptDouble(raw, "routed_scaling_factor", 1.0);

  // ★ §4 TRAP 1 — `n_group` / `topk_group`, ABSENT from the released
  // config.json. `Dots3NoteConfig.__init__` (configs/dots3_note.py:18-19)
  // does `kwargs.setdefault("n_group", 1)` / `("topk_group", 1)` BEFORE
  // `super().__init__`, so DeepseekV3Config's own defaults of 8 and 4
  // (transformers configuration_deepseek_v3.py:168-169) never apply. The
  // fallback here is therefore 1, NOT 8/4 and NOT 0: reading the parent's
  // default would regroup the noaux_tc router at every MoE layer and change
  // which experts are selected, with no shape change and no error.
  p.n_group = OptInt(raw, "n_group", 1);
  p.topk_group = OptInt(raw, "topk_group", 1);

  // --- DSA lightning indexer ---
  // All three are plain reads on the dots3 config, and `index_topk` is the one
  // upstream probes with `hasattr` to decide whether the model is V3.2-sparse
  // at all — so its absence is a silent path switch, not an error.
  p.index_n_heads = ReqInt(raw, "index_n_heads", kUpstreamRaises);
  p.index_head_dim = ReqInt(raw, "index_head_dim", kUpstreamRaises);
  p.index_topk = ReqInt(raw, "index_topk", kUpstreamDisablesDsa);
  // ★ §4 TRAP 2 — `indexer_rope_interleave`, ABSENT from the released
  // config.json, defaulted TRUE by configs/dots3_note.py:23. Consumed as
  // `is_neox_style = not indexer_rope_interleave`
  // (deepseek_v2.py::DeepseekV2MLAAttention:1148), whose own `getattr` default
  // is False — i.e. DeepSeek-V3.2's split-half NeoX. Taking THAT default here
  // would rotate a different set of learned coordinates.
  p.indexer_rope_interleave = OptBool(raw, "indexer_rope_interleave", true);

  // ★ §4 TRAP 3 — `num_nextn_predict_layers`, ABSENT from the released
  // config.json, defaulted 1 by configs/dots3_note.py:24. DeepseekV3Config has
  // no such field, so an absent key would otherwise read 0 and the entire
  // nextn tail (`model.layers.46.*`) would go unclaimed by the loader.
  p.num_nextn_predict_layers = OptInt(raw, "num_nextn_predict_layers", 1);

  // ★ §4 TRAP 5 — present in the config.json, but it is NOT what our DeepSeek
  // MLA assumes. model.py:305-307 (full) and :438-443 (sliding). REQUIRED, and
  // that is finding F1: it is not one of the four setdefaults, upstream reads
  // it plainly, and W1's `false` fallback silently dropped all four scales.
  p.apply_mla_qkv_lora_rescale =
      ReqBool(raw, "apply_mla_qkv_lora_rescale", kUpstreamRaises);

  // --- the two attention geometries ---
  p.full.num_attention_heads =
      ReqInt(raw, "num_attention_heads", kUpstreamV3Default);
  p.full.q_lora_rank = ReqInt(raw, "q_lora_rank", kUpstreamV3Default);
  p.full.kv_lora_rank = ReqInt(raw, "kv_lora_rank", kUpstreamV3Default);
  p.full.qk_nope_head_dim = ReqInt(raw, "qk_nope_head_dim", kUpstreamV3Default);
  p.full.qk_rope_head_dim = ReqInt(raw, "qk_rope_head_dim", kUpstreamV3Default);
  p.full.v_head_dim = ReqInt(raw, "v_head_dim", kUpstreamV3Default);
  p.full.rope_theta = ReqDouble(raw, "rope_theta", kUpstreamV3Default);
  p.full.sliding_window = 0;
  p.full.attention_gate_type =
      ReqString(raw, "attention_gate_type", kUpstreamRaises);
  p.full.has_indexer = true;

  // Every `swa_*` key is a plain attribute read in
  // `Dots3NoteSlidingAttention.__init__` (model.py:341-346, :390, :406) — none
  // is a DeepseekV3Config field and none is a setdefault, so upstream raises
  // AttributeError on each. This port refuses by name instead of substituting.
  p.swa.num_attention_heads =
      ReqInt(raw, "swa_num_attention_heads", kUpstreamRaises);
  p.swa.q_lora_rank = ReqInt(raw, "swa_q_lora_rank", kUpstreamRaises);
  p.swa.kv_lora_rank = ReqInt(raw, "swa_kv_lora_rank", kUpstreamRaises);
  p.swa.qk_nope_head_dim = ReqInt(raw, "swa_qk_nope_head_dim", kUpstreamRaises);
  p.swa.qk_rope_head_dim = ReqInt(raw, "swa_qk_rope_head_dim", kUpstreamRaises);
  p.swa.v_head_dim = ReqInt(raw, "swa_v_head_dim", kUpstreamRaises);
  // ★ §4 TRAP 6a — the sliding layers carry their OWN theta (5e4), NOT the
  // model-level 8e7 (model.py:404-407). REQUIRED, and the other half of
  // finding F1: W1's 10000.0 fallback rotated 33 of the 46 layers three orders
  // of magnitude away from where the model was trained.
  p.swa.rope_theta = ReqDouble(raw, "swa_rope_theta", kUpstreamRaises);
  p.swa.sliding_window = ReqInt(raw, "sliding_window_size", kUpstreamRaises);
  p.swa.attention_gate_type =
      ReqString(raw, "swa_attention_gate_type", kUpstreamRaises);
  p.swa.has_indexer = false;

  // ★ §4 TRAP 6b — the RoPE LAYOUT. BOTH geometries are GPT-J (interleaved
  // adjacent pairs), i.e. `is_neox_style=False`: the sliding layers say so
  // literally (model.py:408) and the full layers inherit it from
  // `deepseek_v2.py`::DeepseekV2MLAAttention:1093-1097, which hard-codes
  // `is_neox_style=False`. This CORRECTS the spec's W0 §4 item 6, which read
  // it as sliding-only (#1804). Note the polarity clash with the indexer,
  // which is the whole point of trap 2: with `indexer_rope_interleave=True`
  // the indexer rope is ALSO non-NeoX, where DeepSeek-V3.2 leaves it NeoX and
  // disagreeing with its own MLA rope.
  //
  // These two assignments are the RESOLUTION STEP, and the struct default is
  // deliberately the OPPOSITE value so that deleting them cannot go unnoticed
  // (review finding F9): a field whose default equals its production value is
  // a field no gate can prove is being set.
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

const std::vector<Dots3NoteDeferredTower>& Dots3NoteDeferredTowers() {
  // The prefixes are upstream's own, read from the hf_to_vllm_mapper at
  // `nvidia/multimodal.py:70-78` (the two prefixes at `:75-76`):
  // "vision_encoder." -> "visual." and "audio_encoder." -> "audio_tower.".
  // RE-DERIVED at vLLM `origin/main` = `185cada36b`, which is where W2 read it;
  // the same mapper sits at `:54-62` at W1's `c205726108`, and citing that from
  // here was an inherited anchor rather than a re-read one (review F4 on
  // #1847). The FILE beside each one is the
  // released checkpoint's, from `model.safetensors.index.json`'s weight_map at
  // revision 1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b: each tower ships whole
  // in one standalone file rather than across the 131 numbered language shards.
  static const std::vector<Dots3NoteDeferredTower> kTowers{
      {"vision_encoder.", "model-vision.safetensors", "W6",
       "the MoE ViT vision tower (nvidia/vision.py, nvidia/vision_moe.py)"},
      {"audio_encoder.", "model-audio.safetensors", "W7",
       "the `dots` Whisper-variant audio tower (nvidia/audio_encoder.py)"},
  };
  return kTowers;
}

const Dots3NoteDeferredTower* Dots3NoteDeferralFor(const std::string& name) {
  for (const Dots3NoteDeferredTower& t : Dots3NoteDeferredTowers()) {
    if (name.rfind(t.prefix, 0) == 0) return &t;
  }
  return nullptr;
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
    // NO RUNTIME GUARD HERE, deliberately. A name cannot be both a language
    // weight and a deferred tower weight — the language branch below wins, so
    // the tower count would silently drop while the total still read 100%
    // accounted. W2 first wrote that invariant as a VT_CHECK on this line and
    // MEASURED it dead: no config makes `EnumerateDots3NoteTensors` emit a
    // `vision_encoder.` or `audio_encoder.` name, every name it emits is
    // `model.`- or `lm_head`-prefixed by construction, and deleting the check
    // left the whole gate green (spec §4.4, mutation M12). The invariant is
    // real, so it is asserted over the real map in the test instead, where
    // adding such a name to this function is what fires it.
    if (!claimed_names.insert(t.name).second) acc.duplicated.push_back(t.name);
  }

  const std::unordered_set<std::string> on_disk(present.begin(), present.end());
  for (const std::string& name : claimed_names) {
    if (on_disk.count(name) == 0) acc.missing.push_back(name);
  }

  for (const std::string& name : present) {
    if (claimed_names.count(name) != 0) {
      ++acc.language;
      continue;
    }
    // NOT an else-branch on a prefix literal: the deferral TABLE decides, so a
    // tower this port forgot to register cannot quietly pass as language.
    const Dots3NoteDeferredTower* tower = Dots3NoteDeferralFor(name);
    if (tower == nullptr) {
      acc.unaccounted.push_back(name);
      continue;
    }
    // Dispatch on the table INDEX. `else ++acc.audio` would count a THIRD
    // registered tower as audio — the table would decide language-versus-
    // deferred correctly and then silently inflate the wrong bucket (review F3
    // on #1847). A tower with no counter is reported UNACCOUNTED instead, so
    // the load refuses naming it, and the refusal prints the table beside it so
    // a reader can see that it IS registered and only the counter is missing.
    // The branch is unreachable while the table has two entries, and it is a
    // safe degradation rather than a guard this gate can prove.
    const size_t which = static_cast<size_t>(tower - Dots3NoteDeferredTowers().data());
    if (which == 0) {
      ++acc.vision;
    } else if (which == 1) {
      ++acc.audio;
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
  std::string towers;
  for (const Dots3NoteDeferredTower& t : Dots3NoteDeferredTowers()) {
    if (!towers.empty()) towers += ", ";
    towers += std::string(t.prefix) + "* (" + t.brick + ")";
  }
  VT_CHECK(w.accounting.unaccounted.empty(),
           "dots3-note: no consumer claims " + w.accounting.unaccounted.front() +
               " (" + std::to_string(w.accounting.unaccounted.size()) +
               " unaccounted tensors), and it is not one of the DEFERRED "
               "towers " + towers +
               " — see .agents/specs/dots3-note.md and issue #699");

  // MATERIALIZATION IS CONDITIONAL, and the condition is the DEVICE FORWARD's
  // own scope rather than a preference (W4a, #699).
  //
  // `Dots3NoteDeviceRefusal` is empty only for a config whose every layer is
  // FULL attention with a DENSE MLP — the shape W4a put on the decode path.
  // For that shape the tower is read for real, with every tensor's shape
  // checked BY NAME, and `materialized` becomes true. For everything else, the
  // released `dots-studio/dots3-note-prev` config included, nothing changes:
  // the accounting above is still a real production result, no tensor byte is
  // read, and the refusal stays at the forward where it names the brick that
  // owes the maths.
  //
  // The alternative — materialize unconditionally — was rejected on a
  // measurement rather than on taste. The released config's `embed_tokens`
  // alone is 152064 x 5120 bf16 = 1.5 GiB, and W1/W2's gate drives the WHOLE
  // 38006-name index through this loader from a synthetic checkpoint of
  // one-element tensors. Demanding real shapes there would either red the
  // accounting gate or force a fixture nothing can hold in memory.
  if (Dots3NoteDeviceRefusal(w.params).empty()) {
    w.device = MaterializeDots3NoteDevice(shards, w.params);
    w.materialized = w.device.present;
  }
  return w;
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
  // Three things are still NOT represented here, and W4b-2 changed which brick
  // owes the first of them. The comment used to say all three were "W4's",
  // which stopped being true the moment the sliding layers ran.
  //
  //   the heterogeneous per-layer  W4b-3. Upstream gives a sliding layer a
  //   KV-cache GROUP SPLIT         `SlidingWindowMLASpec`
  //                                (`mla_attention.py:1215-1219` @
  //                                `bc2d63e650`, fed
  //                                `sliding_window=config.sliding_window_size`
  //                                at `model.py:457`), which is a SECOND spec
  //                                kind and therefore a second group. We emit
  //                                one uniform `MLAAttentionSpec` for all 46
  //                                layers, so 33 of them hold a full-length
  //                                latent cache where upstream caps them near
  //                                the window. No correctness consequence — the
  //                                window is applied on READ, and W4b-2's gate
  //                                proves it — but it is the largest memory
  //                                property of this architecture and a token
  //                                gate cannot see it. `## Owed` carries it.
  //   the DSA index cache          W4b-3, with the indexer's SELECTION
  //   the windowed metadata        never, here: `_build_sliding_window_metadata`
  //                                is upstream's Triton gather bound, and the
  //                                port walks the paged block table instead
  //                                (spec §4.8), so there is no metadata to
  //                                allocate.
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
