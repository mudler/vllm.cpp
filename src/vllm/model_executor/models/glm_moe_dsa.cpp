// GLM-5.3 config resolve, `glm-dsa` GGUF config builder and refuse-by-name
// forward. See `glm_moe_dsa.h` for why this family owns its own params struct
// and its own translation unit.
//
// Spec `.agents/specs/glm-dsa-latest-deepseek.md` §3.7 W2, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214). vLLM parity pin
// `5559679229bc961848b121ccdeaa8fa5d79bec98`; llama.cpp secondary pin `b10451`.
#include "vllm/model_executor/models/glm_moe_dsa.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "vllm/v1/kv_cache_dtype.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

const std::string kGgufPrefix = "glm-dsa.";

[[noreturn]] void Refuse(const std::string& what) {
  throw std::runtime_error("GlmMoeDsaForCausalLM config: " + what);
}

const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  if (!doc.is_object()) return nullptr;
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &*it;
}

int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr || !f->is_number()) return fallback;
  return f->get<int64_t>();
}

double RawFloat(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr || !f->is_number()) return fallback;
  return f->get<double>();
}

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr || !f->is_boolean()) return fallback;
  return f->get<bool>();
}

std::string RawStr(const nlohmann::json& doc, const char* key,
                   const std::string& fallback) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr || !f->is_string()) return fallback;
  return f->get<std::string>();
}

// An OPTIONAL string array. Empty for an absent key, a null, a non-array and an
// array carrying a non-string, so the caller's LENGTH check is what refuses a
// malformed list rather than a silent truncation here.
std::vector<std::string> OptStringArray(const nlohmann::json& doc,
                                        const char* key) {
  const nlohmann::json* f = Field(doc, key);
  if (f == nullptr || !f->is_array()) return {};
  std::vector<std::string> out;
  for (const nlohmann::json& e : *f) {
    if (!e.is_string()) return {};
    out.push_back(e.get<std::string>());
  }
  return out;
}

const char* MlpKindName(GlmMoeDsaMlpKind k) {
  return k == GlmMoeDsaMlpKind::kDense ? "dense" : "sparse";
}

GlmMoeDsaIndexerKind IndexerKindFromString(const std::string& s) {
  if (s == "full") return GlmMoeDsaIndexerKind::kFull;
  if (s == "shared") return GlmMoeDsaIndexerKind::kShared;
  Refuse("`indexer_types` entry `" + s +
         "` is neither `full` nor `shared`; upstream's schedule is boolean "
         "(`_skip_topk`, deepseek_v2.py:1097-1101) and llama.cpp writes it as "
         "one bool per block (b10451:conversion/glm.py:337-339)");
}

GlmMoeDsaMlpKind MlpKindFromString(const std::string& s) {
  if (s == "dense") return GlmMoeDsaMlpKind::kDense;
  if (s == "sparse") return GlmMoeDsaMlpKind::kSparse;
  Refuse("`mlp_layer_types` entry `" + s +
         "` is neither `dense` nor `sparse`");
}

// ── GGUF scalar readers ─────────────────────────────────────────────────────

int64_t KvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    default:
      throw std::runtime_error("glm-dsa gguf: key " + key +
                               " is not an integer");
  }
}

double KvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvInt(v, key));
}

int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "glm-dsa gguf: missing metadata key " + key);
  return KvInt(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvInt(*v, key) : dflt;
}

double OptFloat(const GgufFile& g, const std::string& key, double dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvFloat(*v, key) : dflt;
}

bool OptBool(const GgufFile& g, const std::string& key, bool dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvInt(*v, key) != 0 : dflt;
}

// `%s.attention.indexer.types`, one bool per BACKBONE layer. Returns false when
// the key is absent; the caller refuses rather than substituting a table.
bool OptIndexerTypes(const GgufFile& g, const std::string& key,
                     std::vector<std::string>* out) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr) return false;
  VT_CHECK(v->TypeId() == kGgufArray,
           "glm-dsa gguf: key " + key +
               " is the per-block indexer schedule and llama.cpp writes it as "
               "an array of bool (b10451:gguf_writer.py:806-808)");
  out->clear();
  for (const GgufValue& e : std::get<GgufArray>(v->v).elems) {
    out->push_back(KvInt(e, key) != 0 ? "full" : "shared");
  }
  return true;
}

// Every primitive this model needs that this tree does not have, each with the
// wave that owes it and the record that tracks it. A refusal that says only
// "not implemented" makes the reader go looking; this one hands over the list.
//
// ─── THIS LIST SHRANK, AND THE ENTRIES THAT LEFT ARE NAMED HERE ──────────────
// W2 wrote seven entries. Four of them have since landed and the text is
// corrected in the same change that made the fourth of them false, rather than
// left to accumulate: (a) the expert-streaming seam is
// `expert_stream_seam.{h,cpp}` since W3 (spec O8, DISCHARGED); (b) the
// per-layer heterogeneous `MlaBlockDims` and the `skip_topk` reuse are
// `GlmMoeDsaMlaSchedule` since W4; (c) the `IQ4_XS` keep-quant `vec_dot` landed
// as `VecDotIQ4_XSQ8_K` in `2e9f4d88d` (spec O2, DISCHARGED), so all six
// encodings of the target arm keep their blocks; (d) the fp32 router GEMM is
// `deepseek_v2.cpp:363` sizing its output from `router_dtype_is_f32`, which is
// no longer a hardcoded bf16. A refusal that keeps naming work that is done
// sends its reader to look for something they will not find.
//
// ─── O20: THE `dots3_note_device.cpp` ANCHOR IS CITED BY PREDICATE ───────────
// This string used to carry `dots3_note_device.cpp:1147-1180`, which is the
// explanatory COMMENT above the guard and not the guard, so a user meeting this
// refusal was handed a line range containing no refusal. The range moved again
// inside the branch that found it (`:1204-1227` on `origin/main` `03e0dcd19`,
// `:1205-1228` after W4's own three-line edit to the same file), which is the
// reason the durable citation is the PREDICATE. `!elig.prunes || elig.Active()`
// is unique repo-wide, verified over `src`, `include` and `tests`. Spec O20,
// repaired in flow by the wave that next touched this file, as O20's
// DISPOSITION directs.
constexpr const char* kForwardRefusal =
    "The GlmMoeDsaForCausalLM forward IS implemented (W9) and this STEP cannot "
    "be served, because of the two primitives this build does NOT have. (1) The "
    "indexer KV side cache, upstream's `DeepseekV32IndexerCache` "
    "(deepseek_v2.py:696-701): without it a sparse step refuses every RESUMED "
    "request, for the same reason and at the same predicate as the `VT_CHECK` "
    "spelled `!elig.prunes || elig.Active()` in dots3_note_device.cpp (cited by "
    "predicate rather than by line, spec O20: the range moved twice while this "
    "message was first written) — owned by KV-DSV4-MULTICACHE, spec O4, issues "
    "#1925 and #2323. A FIRST token on a fresh prompt is reachable and a SECOND "
    "is not. (2) sparse prefill: `MlaPrefillAttentionArgs` carries no topk "
    "member and `MlaPrefillAttention` has no selection arm, so a sparse step "
    "must route EVERY token through MQA and only a request with no previously "
    "computed context can do that — W6, spec O6. Multi-token prediction is "
    "skipped, not implemented: block 78 is read, counted and dropped (spec O5); "
    "the loader that materializes these weights is W7's. No end-to-end token "
    "gate against vLLM is reachable on this fleet at all (spec O1, §3.6): vLLM "
    "at the pin implements this architecture and cannot fit its 703.74 GiB on "
    "any device this project reaches. See "
    ".agents/specs/glm-dsa-latest-deepseek.md §3.7.";

constexpr const char* kSafetensorsRefusal =
    "Model architecture GlmMoeDsaForCausalLM does not support safetensors "
    "weights, and this is deferred rather than merely unwritten: the published "
    "zai-org/GLM-5.3 checkpoint is 703.74 GiB across 141 shards in bf16/fp8, "
    "the DeepSeek-V2 loader holds host `OwnedTensor` bytes with no streaming "
    "path (57,600 host tensors for the routed experts alone), and there is no "
    "MoE-expert block-fp8 rung anywhere in this tree. The GGUF arm is the only "
    "one that can be fed on this fleet. See "
    ".agents/specs/glm-dsa-latest-deepseek.md §3.8 D1.";

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

std::vector<GlmMoeDsaIndexerKind> DeriveGlmMoeDsaIndexerSchedule(
    int64_t num_hidden_layers, int64_t index_topk_freq,
    int64_t index_skip_topk_offset) {
  // `freq` divides. Upstream would raise `ZeroDivisionError` on 0, which is a
  // crash rather than a schedule, so it is refused by name here.
  if (index_topk_freq <= 0) {
    Refuse("`index_topk_freq` is " + std::to_string(index_topk_freq) +
           "; upstream divides by it (deepseek_v2.py:1098-1101) and a "
           "non-positive period states no schedule");
  }
  std::vector<GlmMoeDsaIndexerKind> out;
  out.reserve(static_cast<size_t>(std::max<int64_t>(num_hidden_layers, 0)));
  for (int64_t i = 0; i < num_hidden_layers; ++i) {
    const int64_t shifted = std::max<int64_t>(i - index_skip_topk_offset + 1, 0);
    out.push_back(shifted % index_topk_freq != 0 ? GlmMoeDsaIndexerKind::kShared
                                                 : GlmMoeDsaIndexerKind::kFull);
  }
  return out;
}

std::vector<GlmMoeDsaMlpKind> DeriveGlmMoeDsaMlpSchedule(
    int64_t num_hidden_layers, int64_t first_k_dense_replace,
    int64_t moe_layer_freq, int64_t n_routed_experts) {
  std::vector<GlmMoeDsaMlpKind> out;
  out.reserve(static_cast<size_t>(std::max<int64_t>(num_hidden_layers, 0)));
  for (int64_t i = 0; i < num_hidden_layers; ++i) {
    const bool moe = n_routed_experts > 0 && i >= first_k_dense_replace &&
                     (moe_layer_freq <= 1 || i % moe_layer_freq == 0);
    out.push_back(moe ? GlmMoeDsaMlpKind::kSparse : GlmMoeDsaMlpKind::kDense);
  }
  return out;
}

GlmMoeDsaParams ParseGlmMoeDsaParams(const HfConfig& config) {
  const nlohmann::json& doc = config.raw;
  GlmMoeDsaParams p;

  if (config.model_type != "glm_moe_dsa") {
    Refuse("model_type is `" + config.model_type +
           "`, and this architecture is registered for `glm_moe_dsa` "
           "(registry.py:117 at the pin routes `GlmMoeDsaForCausalLM` into "
           "deepseek_v2, and `_get_moe_router_dtype` keys the one behavioural "
           "special case off this exact string, deepseek_v2.py:127)");
  }

  p.hidden_size = config.hidden_size;
  p.num_hidden_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.num_attention_heads = config.num_attention_heads;
  p.intermediate_size = config.intermediate_size;
  p.rms_norm_eps = config.rms_norm_eps;
  p.max_position_embeddings = config.max_position_embeddings;
  p.rope_theta = config.rope_theta;
  // W4 (#2214) closes a hole W2 left and W4 is the first wave that would fall
  // into it. `MlaAttentionScale` (`mla_attention.cpp:320-325`) multiplies the
  // softmax scale by `yarn_get_mscale(factor, mscale_all_dim)` SQUARED whenever
  // a YaRN factor is present, and `ParseGlmMoeDsaParams` reads no factor at all
  // — so a YaRN checkpoint resolving through here would silently get the plain
  // `qk_head_dim ** -0.5`, which is a WRONG attention scale that no structural
  // gate would notice. GLM-5.3 ships `rope_parameters: {"rope_type": "default"}`
  // (revision `935644c05e76fc198714f4cca449fd8b970ff6d7`), so the value this
  // port needs is the plain one; refusing anything else keeps that a measured
  // fact about this checkpoint rather than an assumption about the family.
  if (config.rope_parameters.rope_type != "default") {
    Refuse("rope_parameters.rope_type is '" + config.rope_parameters.rope_type +
           "'; this port resolves the MLA softmax scale as `qk_head_dim ** -0.5` "
           "and reads no YaRN factor, so a scaled rope would silently get the "
           "unscaled mscale^2 correction (deepseek_v2.py:1053-1058, "
           "MlaAttentionScale). GLM-5.3 ships 'default'");
  }
  if (p.num_hidden_layers <= 0) {
    Refuse("num_hidden_layers is " + std::to_string(p.num_hidden_layers) +
           "; every per-layer schedule below is sized from it");
  }
  if (p.hidden_size <= 0 || p.num_attention_heads <= 0 || p.vocab_size <= 0) {
    Refuse("hidden_size / num_attention_heads / vocab_size are " +
           std::to_string(p.hidden_size) + " / " +
           std::to_string(p.num_attention_heads) + " / " +
           std::to_string(p.vocab_size) + "; each must be positive");
  }

  // --- MLA geometry ---
  p.q_lora_rank = RawInt(doc, "q_lora_rank", 0);
  p.kv_lora_rank = RawInt(doc, "kv_lora_rank", 0);
  p.qk_nope_head_dim = RawInt(doc, "qk_nope_head_dim", 0);
  p.qk_rope_head_dim = RawInt(doc, "qk_rope_head_dim", 0);
  p.v_head_dim = RawInt(doc, "v_head_dim", 0);
  if (p.kv_lora_rank <= 0 || p.qk_rope_head_dim <= 0 ||
      p.qk_nope_head_dim <= 0 || p.v_head_dim <= 0) {
    Refuse("the MLA geometry is incomplete: kv_lora_rank " +
           std::to_string(p.kv_lora_rank) + ", qk_nope_head_dim " +
           std::to_string(p.qk_nope_head_dim) + ", qk_rope_head_dim " +
           std::to_string(p.qk_rope_head_dim) + ", v_head_dim " +
           std::to_string(p.v_head_dim) +
           "; this architecture is MLA-only (deepseek_v2.py:1040-1074) and has "
           "no MHA fallback");
  }
  // WRITTEN, not defaulted. Upstream passes `is_neox_style=False`
  // unconditionally (deepseek_v2.py:1073) and consults no config key, so a
  // checkpoint that declares `rope_interleave` cannot move it either way. The
  // line exists so the value is a READ of upstream rather than the coincidence
  // of `MlaBlockDims::is_neox_style` also defaulting false.
  p.is_neox_style = false;
  // deepseek_v2.py:1120, and this one IS config-driven.
  p.indexer_rope_is_neox_style = !RawBool(doc, "indexer_rope_interleave", false);

  // --- MoE ---
  p.n_routed_experts = RawInt(doc, "n_routed_experts", 0);
  p.num_experts_per_tok = RawInt(doc, "num_experts_per_tok", 0);
  p.moe_intermediate_size = RawInt(doc, "moe_intermediate_size", 0);
  p.n_shared_experts = RawInt(doc, "n_shared_experts", 0);
  p.first_k_dense_replace = RawInt(doc, "first_k_dense_replace", 0);
  p.moe_layer_freq = RawInt(doc, "moe_layer_freq", 1);
  p.n_group = RawInt(doc, "n_group", 1);
  p.topk_group = RawInt(doc, "topk_group", 1);
  p.norm_topk_prob = RawBool(doc, "norm_topk_prob", false);
  p.routed_scaling_factor = RawFloat(doc, "routed_scaling_factor", 1.0);
  {
    const std::string scoring = RawStr(doc, "scoring_func", "softmax");
    if (scoring != "sigmoid") {
      Refuse("scoring_func is `" + scoring +
             "`; GLM-5.3 is a sigmoid `noaux_tc` router "
             "(deepseek_v2.py:313-318) and no other scoring function is "
             "reachable on this architecture");
    }
    const std::string topk_method = RawStr(doc, "topk_method", "noaux_tc");
    if (topk_method != "noaux_tc") {
      Refuse("topk_method is `" + topk_method +
             "`; `noaux_tc` is the only setting that creates the learned "
             "`e_score_correction_bias` gate this checkpoint ships "
             "(deepseek_v2.py:313-318)");
    }
    p.has_e_score_correction_bias = true;
  }
  // Forced f32, ahead of the generic branch. See the field's comment.
  p.router_dtype_is_f32 = true;

  // --- the indexer ---
  p.index_topk = RawInt(doc, "index_topk", 0);
  p.index_n_heads = RawInt(doc, "index_n_heads", 0);
  p.index_head_dim = RawInt(doc, "index_head_dim", 0);
  if (p.index_topk <= 0 || p.index_n_heads <= 0 || p.index_head_dim <= 0) {
    Refuse("the DSA indexer geometry is incomplete: index_topk " +
           std::to_string(p.index_topk) + ", index_n_heads " +
           std::to_string(p.index_n_heads) + ", index_head_dim " +
           std::to_string(p.index_head_dim) +
           "; upstream selects the sparse path on the mere PRESENCE of "
           "`index_topk` (deepseek_v2.py:1068), so a `glm_moe_dsa` config "
           "without it describes no model this class builds");
  }

  // --- MTP ---
  p.num_nextn_predict_layers = RawInt(doc, "num_nextn_predict_layers", 0);
  p.index_share_for_mtp_iteration =
      RawBool(doc, "index_share_for_mtp_iteration", false);

  // --- the indexer schedule ---
  //
  // DERIVED is the source; the explicit list is an OVERRIDE, and it is the
  // spelling the GGUF arm has to use because llama.cpp's converter writes the
  // list and writes neither `index_topk_freq` nor `index_skip_topk_offset`
  // (b10451:conversion/glm.py:333-339).
  //
  // A config that states NEITHER is refused rather than defaulted. Upstream's
  // `getattr(config, "index_topk_freq", 1)` would make every layer `full` on
  // such a file, which is a schedule that runs 78 indexers where the reference
  // runs 21 and reports nothing; and llama.cpp's answer — a hardcoded 78-entry
  // table (b10451:src/models/glm-dsa.cpp:6-27) — is right for exactly the
  // checkpoints that exist today. Spec D3.
  {
    const bool has_freq = Field(doc, "index_topk_freq") != nullptr ||
                          Field(doc, "index_skip_topk_offset") != nullptr;
    p.index_topk_freq = RawInt(doc, "index_topk_freq", 1);
    p.index_skip_topk_offset = RawInt(doc, "index_skip_topk_offset", 2);
    if (Field(doc, "index_topk_pattern") != nullptr) {
      Refuse(
          "`index_topk_pattern` is set. Upstream reads it AHEAD of the "
          "freq/offset schedule (deepseek_v2.py:1097-1103) and no published "
          "`glm_moe_dsa` checkpoint uses it, so this port refuses it by name "
          "rather than resolving it to a stack the reference would not build");
    }

    std::vector<std::string> declared = OptStringArray(doc, "indexer_types");
    if (declared.empty() && !has_freq) {
      Refuse(
          "the indexer schedule is stated nowhere: there is no `indexer_types` "
          "list and no `index_topk_freq` / `index_skip_topk_offset` to derive "
          "one from. A GGUF written by "
          "`b10451:conversion/glm.py:337-339` carries the list as "
          "`glm-dsa.attention.indexer.types` and carries no freq/offset keys "
          "at all, so a file missing both is one llama.cpp itself only survives "
          "by falling back to a hardcoded 78-entry table "
          "(b10451:src/models/glm-dsa.cpp:6-27). That table is bit-identical "
          "to GLM-5.3's own list and would silently become wrong on the next "
          "checkpoint, so it is deliberately not copied "
          "(.agents/specs/glm-dsa-latest-deepseek.md §3.8 D3)");
    }

    const std::vector<GlmMoeDsaIndexerKind> derived =
        DeriveGlmMoeDsaIndexerSchedule(p.num_hidden_layers, p.index_topk_freq,
                                       p.index_skip_topk_offset);
    if (declared.empty()) {
      p.indexer_types = derived;
    } else {
      if (static_cast<int64_t>(declared.size()) != p.num_hidden_layers) {
        Refuse("`indexer_types` has " + std::to_string(declared.size()) +
               " entries but num_hidden_layers is " +
               std::to_string(p.num_hidden_layers));
      }
      for (const std::string& s : declared) {
        p.indexer_types.push_back(IndexerKindFromString(s));
      }
    }
    // The FIRST layer cannot be `shared`: `shared` reuses the previous full
    // layer's selection and there is no previous one. Upstream's offset rule
    // never synthesizes such a schedule; a converter-written list can.
    if (p.indexer_types[0] == GlmMoeDsaIndexerKind::kShared) {
      Refuse(
          "the first layer's `indexer_types` entry is `shared`, which reuses "
          "the preceding full layer's top-k selection and there is no "
          "preceding layer");
    }
  }

  // --- the dense/MoE schedule ---
  //
  // DERIVED from `first_k_dense_replace` / `moe_layer_freq`, which is what
  // upstream does and the ONLY thing it does: `grep -c mlp_layer_types` over
  // `deepseek_v2.py` at the pin is 0, and the key exists upstream only in
  // `cohere2_moe.py` and `mellum.py`. GLM-5.3 nevertheless ships a 78-entry
  // `mlp_layer_types`, so the list is READ and CROSS-CHECKED rather than
  // followed. That polarity is deliberate: following it would make this port
  // obey a value the reference ignores, while ignoring it silently would let a
  // checkpoint whose two descriptions disagree load as whichever one we happen
  // to read. A disagreement is one of the two being wrong, and refusing says so.
  {
    const std::vector<GlmMoeDsaMlpKind> derived = DeriveGlmMoeDsaMlpSchedule(
        p.num_hidden_layers, p.first_k_dense_replace, p.moe_layer_freq,
        p.n_routed_experts);
    const std::vector<std::string> declared =
        OptStringArray(doc, "mlp_layer_types");
    if (!declared.empty()) {
      if (static_cast<int64_t>(declared.size()) != p.num_hidden_layers) {
        Refuse("`mlp_layer_types` has " + std::to_string(declared.size()) +
               " entries but num_hidden_layers is " +
               std::to_string(p.num_hidden_layers));
      }
      for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
        const size_t u = static_cast<size_t>(i);
        const GlmMoeDsaMlpKind stated = MlpKindFromString(declared[u]);
        if (stated != derived[u]) {
          Refuse(
              "the config states its dense/MoE layout twice and the two "
              "disagree at layer " +
              std::to_string(i) + ": `mlp_layer_types` says `" +
              std::string(MlpKindName(stated)) +
              "` while first_k_dense_replace=" +
              std::to_string(p.first_k_dense_replace) + " / moe_layer_freq=" +
              std::to_string(p.moe_layer_freq) + " derives `" +
              std::string(MlpKindName(derived[u])) +
              "` (deepseek_v2.py:1214-1218). Upstream reads only the second, so "
              "the first would be silently ignored");
        }
      }
    }
    p.mlp_layer_types = derived;
  }

  return p;
}

// ─── W4: the heterogeneous per-layer MLA schedule ────────────────────────────
// Upstream never builds a schedule OBJECT. It constructs one
// `DeepseekV2MLAAttention` per layer and decides that layer's shape inside the
// constructor (`deepseek_v2.py:1092-1103` for `_skip_topk`, `:1115` for whether
// an indexer exists, `:1175` for the flag the MLA wrapper keeps). Materializing
// the same decisions as a vector is a HARNESS adaptation, not a semantic one:
// this tree's layers are `mla::MlaBlockDims` values rather than modules, so the
// per-layer decision has to live somewhere a caller can hold.
mla::MlaBlockDims GlmMoeDsaMlaBlockDims(const GlmMoeDsaParams& p, int64_t layer) {
  if (layer < 0 || layer >= static_cast<int64_t>(p.indexer_types.size())) {
    throw std::out_of_range(
        "GlmMoeDsaMlaBlockDims: layer " + std::to_string(layer) +
        " is outside [0, " + std::to_string(p.indexer_types.size()) +
        ") — the indexer schedule is num_hidden_layers long");
  }
  mla::MlaBlockDims d{};
  d.hidden_size = p.hidden_size;
  d.num_heads = p.num_attention_heads;
  d.q_lora_rank = p.q_lora_rank;
  d.kv_lora_rank = p.kv_lora_rank;
  d.qk_nope_head_dim = p.qk_nope_head_dim;
  d.qk_rope_head_dim = p.qk_rope_head_dim;
  d.v_head_dim = p.v_head_dim;
  // `is_neox_style=False` unconditionally at `deepseek_v2.py:1073`. W2 made this
  // an explicit parsed field precisely so it is not two defaults agreeing.
  d.is_neox_style = p.is_neox_style;
  // `rope_type: "default"` on this checkpoint — no YaRN, so `MlaAttentionScale`
  // reduces to `qk_head_dim ** -0.5` and the mscale^2 term is absent. Computed
  // through the shared helper rather than written as a literal so a checkpoint
  // that DOES carry a YaRN factor gets the correction instead of a wrong number.
  mla::DeepseekYarnRopeParams rope{};
  rope.base = p.rope_theta;
  rope.rotary_dim = p.qk_rope_head_dim;
  // `rope_parameters["rope_type"] != "default"` is upstream's `yarn` predicate
  // (`deepseek_v2.py:1053-1058`), and `ParseGlmMoeDsaParams` refuses every value
  // but "default", so this is false by a CHECKED fact rather than by a default.
  rope.yarn = false;
  d.scale = mla::MlaAttentionScale(d, rope);

  // THE SPLIT, and it is the whole of this function. A `kFull` layer carries the
  // indexer geometry and `skip_topk` false; a `kShared` layer carries NEITHER
  // piece of geometry and `skip_topk` true, which is exactly upstream's
  // `self.indexer = None` (`:1134-1135`) plus `skip_topk=True` (`:1175`).
  //
  // The geometry is CLEARED rather than left set on a shared layer, and
  // `MlaBlockDims::Validate` refuses the combination, because a shared layer
  // whose `index_topk` survived would run an indexer over weights the checkpoint
  // does not ship for it — GLM-5.3 stores `self_attn.indexer.*` on 22 of 79
  // blocks and on no other.
  if (p.indexer_types[static_cast<size_t>(layer)] == GlmMoeDsaIndexerKind::kFull) {
    d.index_n_heads = p.index_n_heads;
    d.index_head_dim = p.index_head_dim;
    d.index_topk = p.index_topk;
    d.indexer_rope_is_neox_style = p.indexer_rope_is_neox_style;
    d.skip_topk = false;
  } else {
    d.index_n_heads = 0;
    d.index_head_dim = 0;
    d.index_topk = 0;
    d.indexer_rope_is_neox_style = false;
    d.skip_topk = true;
  }
  d.Validate();
  return d;
}

std::vector<mla::MlaBlockDims> GlmMoeDsaMlaSchedule(const GlmMoeDsaParams& p) {
  std::vector<mla::MlaBlockDims> out;
  out.reserve(p.indexer_types.size());
  for (int64_t i = 0; i < static_cast<int64_t>(p.indexer_types.size()); ++i) {
    out.push_back(GlmMoeDsaMlaBlockDims(p, i));
  }
  // LAYER 0 CANNOT BE SHARED, and this is the one ordering property the reuse
  // depends on that the derivation does not state. `mla.py:180` reuses whatever
  // is in the shared buffer, and on the first layer of a forward pass that is
  // the buffer's UNINITIALIZED contents — a selection of arbitrary positions
  // that attention would accept without complaint. Upstream's rule cannot
  // produce it (`max(0 - offset + 1, 0) % freq == 0` for every non-negative
  // offset, because `max(...)` clamps to 0 and 0 % freq is 0), so a schedule
  // that starts shared came from an explicit `indexer_types` list or an
  // `index_topk_pattern`, and it is a checkpoint this port must refuse rather
  // than read past.
  if (!out.empty() && out.front().skip_topk) {
    throw std::runtime_error(
        "GlmMoeDsaMlaSchedule: layer 0 is `shared`, so there is no preceding "
        "full layer whose selection it could attend through (mla.py:180 reuses "
        "the shared topk_indices_buffer, which at layer 0 holds nothing). "
        "Upstream's derived rule at deepseek_v2.py:1097-1101 always makes layer "
        "0 full; this schedule did not come from it");
  }
  return out;
}

int64_t GlmMoeDsaFullIndexerLayerCount(const GlmMoeDsaParams& p) {
  return static_cast<int64_t>(std::count(p.indexer_types.begin(), p.indexer_types.end(),
                                         GlmMoeDsaIndexerKind::kFull));
}

void ParseGlmMoeDsaConfig(const HfConfig& config) {
  // The resolve IS the validation: `ParseGlmMoeDsaParams` throws with a precise
  // message on every field this port cannot serve.
  const GlmMoeDsaParams p = ParseGlmMoeDsaParams(config);
  // W4: and the per-layer MLA geometry is part of what "this port cannot serve"
  // means. `MlaBlockDims::Validate` is the block's own refusal set — the
  // `v_head_dim <= qk_head_dim` rule, the even rope width, the
  // skip_topk/indexer mutual exclusion — and running it HERE means a config
  // whose attention the block would refuse is refused where the user meets it,
  // at resolve, rather than on the first forward of a 201 GiB load.
  //
  // This is also the only production call site the schedule has until W7 builds
  // a forward, and it is a real one: `ModelRegistry::Resolve` reaches it for
  // every `GlmMoeDsaForCausalLM` config, from a `config.json` and from a
  // `glm-dsa` GGUF header alike.
  (void)GlmMoeDsaMlaSchedule(p);
}

bool IsGlmMoeDsaGguf(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("general.architecture");
  return v != nullptr && v->TypeId() == kGgufString &&
         std::get<std::string>(v->v) == kGlmMoeDsaGgufArch;
}

HfConfig GlmMoeDsaHfConfigFromGguf(const GgufFile& gguf) {
  VT_CHECK(IsGlmMoeDsaGguf(gguf),
           "glm-dsa gguf: general.architecture must be '" +
               std::string(kGlmMoeDsaGgufArch) + "'");
  const std::string p = kGgufPrefix;

  HfConfig c;
  // The HF `model_type` and architecture string, verbatim from the released
  // `zai-org/GLM-5.3` config.json at revision
  // `935644c05e76fc198714f4cca449fd8b970ff6d7`. NOT derived from the GGUF key:
  // `glm-dsa` is llama.cpp's family name and the registry keys on the HF class.
  c.model_type = "glm_moe_dsa";
  c.architectures = {"GlmMoeDsaForCausalLM"};

  // BLOCKS ARE NOT LAYERS. `b10451:conversion/glm.py:287-289` writes
  // `block_count = num_hidden_layers + num_nextn_predict_layers`, so the
  // published 79-block file describes a 78-layer backbone plus one MTP block.
  // Reading `block_count` into `num_hidden_layers` would build one extra
  // decoder layer out of the MTP block, and nothing downstream would refuse it.
  const int64_t n_blocks = ReqInt(gguf, p + "block_count");
  VT_CHECK(n_blocks > 0, "glm-dsa gguf: block_count must be > 0");
  const int64_t n_mtp = OptInt(gguf, p + "nextn_predict_layers", 0);
  VT_CHECK(n_mtp >= 0,
           "glm-dsa gguf: nextn_predict_layers is " + std::to_string(n_mtp) +
               " and a count of multi-token-prediction blocks cannot be "
               "negative");
  const int64_t n_layers = n_blocks - n_mtp;
  VT_CHECK(n_layers > 0,
           "glm-dsa gguf: block_count is " + std::to_string(n_blocks) +
               " and nextn_predict_layers is " + std::to_string(n_mtp) +
               ", so the backbone would be " + std::to_string(n_layers) +
               " layers deep; llama.cpp writes block_count as "
               "num_hidden_layers + nextn_predict_layers "
               "(b10451:conversion/glm.py:287-289)");

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = n_layers;
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  // llama.cpp's DeepSeek converter FORCES `num_key_value_heads = 1` because MLA
  // converts into MQA (`b10451:conversion/deepseek.py`, the
  // `self.hparams["num_key_value_heads"] = 1` line). The released config.json
  // says 64. The file's own value is kept rather than the checkpoint's, because
  // this field describes the CACHE this file's tensors were written for, and no
  // consumer of an MLA config reads it for anything else.
  c.num_key_value_heads = OptInt(gguf, p + "attention.head_count_kv", 1);
  c.vocab_size = OptInt(gguf, p + "vocab_size", 0);
  if (c.vocab_size == 0) {
    const GgufValue* toks = gguf.FindKv("tokenizer.ggml.tokens");
    VT_CHECK(toks != nullptr && toks->TypeId() == kGgufArray,
             "glm-dsa gguf: neither " + p +
                 "vocab_size nor tokenizer.ggml.tokens states the vocabulary "
                 "size");
    c.vocab_size =
        static_cast<int64_t>(std::get<GgufArray>(toks->v).elems.size());
  }
  c.intermediate_size = ReqInt(gguf, p + "feed_forward_length");
  c.rms_norm_eps = OptFloat(gguf, p + "attention.layer_norm_rms_epsilon", 1e-5);
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);
  c.rope_theta = OptFloat(gguf, p + "rope.freq_base", 10000.0);

  nlohmann::json raw = nlohmann::json::object();
  raw["model_type"] = c.model_type;
  raw["architectures"] = c.architectures;

  // MLA. `attention.key_length_mla` is `qk_nope_head_dim + qk_rope_head_dim`
  // and `rope.dimension_count` is `qk_rope_head_dim`
  // (b10451:conversion/deepseek.py, the `add_key_length_mla` /
  // `add_rope_dimension_count` pair), so the no-rope half is a SUBTRACTION and
  // is checked for positivity rather than assumed.
  const int64_t kv_lora_rank = ReqInt(gguf, p + "attention.kv_lora_rank");
  const int64_t qk_rope = ReqInt(gguf, p + "rope.dimension_count");
  const int64_t qk_head_dim = ReqInt(gguf, p + "attention.key_length_mla");
  const int64_t qk_nope = qk_head_dim - qk_rope;
  VT_CHECK(qk_nope > 0,
           "glm-dsa gguf: attention.key_length_mla is " +
               std::to_string(qk_head_dim) + " and rope.dimension_count is " +
               std::to_string(qk_rope) + ", so qk_nope_head_dim would be " +
               std::to_string(qk_nope) + "; llama.cpp writes key_length_mla as "
               "`qk_nope_head_dim + qk_rope_head_dim`");
  raw["kv_lora_rank"] = kv_lora_rank;
  raw["qk_rope_head_dim"] = qk_rope;
  raw["qk_nope_head_dim"] = qk_nope;
  raw["v_head_dim"] = ReqInt(gguf, p + "attention.value_length_mla");
  raw["q_lora_rank"] = OptInt(gguf, p + "attention.q_lora_rank", 0);
  // `attention.key_length` is the THIRD statement of the same latent,
  // `kv_lora_rank + qk_rope_head_dim`. A file that states it and disagrees is
  // contradicting itself about the width of its own cache.
  {
    const int64_t key_length =
        OptInt(gguf, p + "attention.key_length", kv_lora_rank + qk_rope);
    VT_CHECK(key_length == kv_lora_rank + qk_rope,
             "glm-dsa gguf: attention.key_length is " +
                 std::to_string(key_length) + " but kv_lora_rank + "
                 "rope.dimension_count is " +
                 std::to_string(kv_lora_rank + qk_rope) +
                 "; the file states the latent width twice and the two "
                 "disagree");
  }
  // The indexer's rope. llama.cpp writes no key for `indexer_rope_interleave`,
  // and every published `glm_moe_dsa` checkpoint sets it true, so the GGUF arm
  // states the checkpoint's value rather than inheriting upstream's
  // `getattr(..., False)` default — which would flip the indexer onto NeoX rope
  // and produce plausible tokens with a wrong selection.
  raw["indexer_rope_interleave"] = true;

  // MoE.
  raw["n_routed_experts"] = ReqInt(gguf, p + "expert_count");
  raw["num_experts_per_tok"] = ReqInt(gguf, p + "expert_used_count");
  raw["moe_intermediate_size"] = ReqInt(gguf, p + "expert_feed_forward_length");
  raw["n_shared_experts"] = OptInt(gguf, p + "expert_shared_count", 0);
  raw["first_k_dense_replace"] = OptInt(gguf, p + "leading_dense_block_count", 0);
  raw["moe_layer_freq"] = 1;
  raw["n_group"] = OptInt(gguf, p + "expert_group_count", 1);
  raw["topk_group"] = OptInt(gguf, p + "expert_group_used_count", 1);
  raw["routed_scaling_factor"] = OptFloat(gguf, p + "expert_weights_scale", 1.0);
  raw["norm_topk_prob"] = OptBool(gguf, p + "expert_weights_norm", false);
  raw["scoring_func"] = "sigmoid";
  raw["topk_method"] = "noaux_tc";

  // The indexer geometry and its schedule.
  raw["index_n_heads"] = ReqInt(gguf, p + "attention.indexer.head_count");
  raw["index_head_dim"] = ReqInt(gguf, p + "attention.indexer.key_length");
  raw["index_topk"] = ReqInt(gguf, p + "attention.indexer.top_k");
  raw["num_nextn_predict_layers"] = n_mtp;
  {
    std::vector<std::string> types;
    if (OptIndexerTypes(gguf, p + "attention.indexer.types", &types)) {
      // The converter writes the TRUNK list (78 entries) into a file whose
      // `block_count` is 79, so this is checked against the backbone depth.
      VT_CHECK(static_cast<int64_t>(types.size()) == n_layers,
               "glm-dsa gguf: " + p + "attention.indexer.types has " +
                   std::to_string(types.size()) +
                   " entries but the backbone is " + std::to_string(n_layers) +
                   " layers deep");
      raw["indexer_types"] = types;
    }
    // Nothing is synthesized when the key is absent, and no freq/offset key is
    // written into `raw` either. `ParseGlmMoeDsaParams` then refuses by name
    // (spec D3), which is the refusal the published artifact meets: it declares
    // indexer weights on all 79 blocks and writes no `indexer.types`.
  }

  c.raw = raw;
  // Refuse HERE rather than at first forward, through the SAME validator a
  // config.json meets. That is the whole reason `raw` is HF-shaped.
  (void)ParseGlmMoeDsaParams(c);
  return c;
}

v1::KVCacheConfig MakeGlmMoeDsaKVCache(const HfConfig& config, int block_size,
                                       int num_blocks) {
  const GlmMoeDsaParams p = ParseGlmMoeDsaParams(config);
  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<v1::MLAAttentionSpec>(
          block_size, static_cast<int>(p.mla_kv_head_size()),
          v1::ResolveKvCacheDType()));
  return kv;
}

const char* GlmMoeDsaForwardRefusal() { return kForwardRefusal; }
const char* GlmMoeDsaSafetensorsRefusal() { return kSafetensorsRefusal; }

// `GlmMoeDsaModel::Forward` and `::ForwardDevice` live in
// `glm_moe_dsa_forward.cpp` (W9, #2214). They used to refuse here; what remains
// in this translation unit is the CONFIG surface and the refusal TEXT, which the
// forward still raises for the one step shape it cannot serve.

}  // namespace vllm
