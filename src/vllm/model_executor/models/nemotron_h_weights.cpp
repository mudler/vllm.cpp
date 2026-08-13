// Nemotron-H W3: config descent + the on-disk weight name map. See nemotron_h.h
// for the port anchors and the scope boundary against W1/W4.
#include "vllm/model_executor/models/nemotron_h.h"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vllm {
namespace {

using nlohmann::json;

[[noreturn]] void Refuse(const std::string& detail) {
  throw std::runtime_error("NemotronHForCausalLM: " + detail);
}

// `raw` is the whole config document; every NemotronH-specific key is read from
// it because HfConfig types only the shared subset.
const json& Raw(const HfConfig& config) { return config.raw; }

bool Has(const json& doc, const char* key) {
  return doc.contains(key) && !doc.at(key).is_null();
}

int64_t GetInt(const json& doc, const char* key, int64_t fallback) {
  if (!Has(doc, key)) return fallback;
  const json& v = doc.at(key);
  if (!v.is_number_integer() && !v.is_number_unsigned()) {
    Refuse(std::string("config key '") + key + "' must be an integer");
  }
  return v.get<int64_t>();
}

double GetDouble(const json& doc, const char* key, double fallback) {
  if (!Has(doc, key)) return fallback;
  const json& v = doc.at(key);
  if (!v.is_number()) {
    Refuse(std::string("config key '") + key + "' must be a number");
  }
  return v.get<double>();
}

bool GetBool(const json& doc, const char* key, bool fallback) {
  if (!Has(doc, key)) return fallback;
  const json& v = doc.at(key);
  if (!v.is_boolean()) {
    Refuse(std::string("config key '") + key + "' must be a boolean");
  }
  return v.get<bool>();
}

std::string GetString(const json& doc, const char* key,
                      const std::string& fallback) {
  if (!Has(doc, key)) return fallback;
  const json& v = doc.at(key);
  if (!v.is_string()) {
    Refuse(std::string("config key '") + key + "' must be a string");
  }
  return v.get<std::string>();
}

// The legacy-alias reads of the `mamba_*` SCALARS
// (configuration_nemotron_h.py:145-155). Upstream is:
//
//   self.n_groups = kwargs.pop("mamba_n_groups") if "mamba_n_groups" in kwargs
//                   else self.n_groups
//
// The dataclass field already holds the modern value (or the class default) by
// the time `__post_init__` runs, and the legacy alias OVERWRITES it whenever it
// is present. So the precedence is LEGACY > modern > class default — the
// opposite of what reads naturally, and the opposite of the SCHEDULE pair below.
//
// Verified by RUNNING transformers @ 7d06b1a5 rather than by reading it:
//   NemotronHConfig(n_groups=8, mamba_n_groups=4,
//                   conv_kernel=4, mamba_d_conv=7) -> n_groups=4, conv_kernel=7
//
// A checkpoint carrying only the alias must likewise not silently deserialize
// to the class default. No released checkpoint ships both spellings of one
// field, so this is a mirroring obligation, not a live defect — pinned by
// "when BOTH spellings ship, the precedence is upstream's and it is PER-FAMILY"
// so it cannot drift back.
int64_t GetIntAliased(const json& doc, const char* key, const char* legacy,
                      int64_t fallback) {
  if (Has(doc, legacy)) return GetInt(doc, legacy, fallback);
  return GetInt(doc, key, fallback);
}

double GetDoubleAliased(const json& doc, const char* key, const char* legacy,
                        double fallback) {
  if (Has(doc, legacy)) return GetDouble(doc, legacy, fallback);
  return GetDouble(doc, key, fallback);
}

bool GetBoolAliased(const json& doc, const char* key, const char* legacy,
                    bool fallback) {
  if (Has(doc, legacy)) return GetBool(doc, legacy, fallback);
  return GetBool(doc, key, fallback);
}

// The four block spellings, in enum order. The single source of truth for both
// directions of the name<->enum map, so a fifth block kind cannot be added with
// a refusal message that still lists four.
constexpr NemotronHBlock kAllBlocks[] = {
    NemotronHBlock::kMamba, NemotronHBlock::kAttention, NemotronHBlock::kMoe,
    NemotronHBlock::kMlp};

NemotronHBlock BlockFromName(const std::string& name) {
  for (NemotronHBlock block : kAllBlocks) {
    if (name == NemotronHBlockName(block)) return block;
  }
  // Mirror of validate_layer_type (configuration_nemotron_h.py:195-204).
  std::string expected;
  for (NemotronHBlock block : kAllBlocks) {
    if (!expected.empty()) expected += ", ";
    expected += NemotronHBlockName(block);
  }
  Refuse("layers_block_type contains the unsupported block type '" + name +
         "' (expected one of " + expected + ")");
}

NemotronHBlock BlockFromPatternChar(char c) {
  // configuration_nemotron_h.py:265-268.
  switch (c) {
    case 'M':
      return NemotronHBlock::kMamba;
    case 'E':
      return NemotronHBlock::kMoe;
    case '*':
      return NemotronHBlock::kAttention;
    case '-':
      return NemotronHBlock::kMlp;
    default:
      Refuse(std::string("hybrid_override_pattern contains '") + c +
             "' (expected one of M, E, *, -)");
  }
}

// One layer schedule, resolved with upstream's precedence: the explicit list,
// else the legacy pattern string, else the class default. `list_key` /
// `pattern_key` are the modern/legacy pair, `fallback` the class default.
//
// NOTE the polarity, which is the OPPOSITE of the `mamba_*` scalars above and
// is deliberate on both sides. configuration_nemotron_h.py:158-165:
//
//   if "hybrid_override_pattern" in kwargs:
//       pattern = kwargs.pop("hybrid_override_pattern")
//       if self.layer_types is None:
//           self.layer_types = self._pattern_to_list(pattern)
//
// the legacy pattern is consulted ONLY when the modern list is absent, so here
// MODERN wins; :176-184 does the same for the MTP pair. Verified by running
// transformers @ 7d06b1a5: `NemotronHConfig(layer_types=['mamba','mamba'],
// hybrid_override_pattern='*-')` -> `['mamba','mamba']`. Do not "unify" the two
// families — upstream genuinely disagrees with itself here.
std::vector<NemotronHBlock> ResolveSchedule(
    const json& doc, const char* list_key, const char* pattern_key,
    const std::vector<NemotronHBlock>& fallback) {
  if (Has(doc, list_key)) {
    const json& v = doc.at(list_key);
    if (!v.is_array()) {
      Refuse(std::string("config key '") + list_key + "' must be a list");
    }
    std::vector<NemotronHBlock> out;
    out.reserve(v.size());
    for (const json& entry : v) {
      if (!entry.is_string()) {
        Refuse(std::string("config key '") + list_key +
               "' must be a list of strings");
      }
      out.push_back(BlockFromName(entry.get<std::string>()));
    }
    return out;
  }
  if (Has(doc, pattern_key)) {
    std::vector<NemotronHBlock> out;
    for (char c : GetString(doc, pattern_key, "")) {
      out.push_back(BlockFromPatternChar(c));
    }
    return out;
  }
  return fallback;
}

NemotronHQuantSurface ResolveQuantSurface(const json& doc) {
  NemotronHQuantSurface q;
  if (!Has(doc, "quantization_config")) return q;
  const json& qc = doc.at("quantization_config");
  if (!qc.is_object()) Refuse("quantization_config must be an object");
  q.present = true;
  q.quant_method = GetString(qc, "quant_method", "");
  q.quant_algo = GetString(qc, "quant_algo", "");
  // W3 resolves NO per-module algorithm — that is W1 (#517 W1). It only refuses
  // a producer whose on-disk companion-tensor layout we have not read, because
  // enumerating the wrong companions is exactly the silent-wrong-bytes failure
  // a token gate cannot see.
  if (q.quant_method != "modelopt") {
    Refuse("quantization_config.quant_method '" + q.quant_method +
           "' is not implemented (this row ports ModelOpt MIXED_PRECISION; see "
           ".agents/specs/nemotron-h-model.md W1)");
  }
  if (Has(qc, "kv_cache_scheme")) {
    const json& kv = qc.at("kv_cache_scheme");
    const bool fp8 = kv.is_object() && GetInt(kv, "num_bits", 0) == 8 &&
                     GetString(kv, "type", "") == "float";
    if (!fp8) {
      Refuse("quantization_config.kv_cache_scheme is not the fp8 scheme this "
             "row implements (expected num_bits 8, type float)");
    }
    q.fp8_kv_cache = true;
  }
  if (Has(qc, "ignore")) {
    const json& ignore = qc.at("ignore");
    if (!ignore.is_array()) Refuse("quantization_config.ignore must be a list");
    for (const json& entry : ignore) {
      if (!entry.is_string()) continue;
      const std::string name = entry.get<std::string>();
      // The released list carries the wildcard `mtp*`, which is what leaves the
      // whole MTP tower bf16 with no scale companions.
      if (name.rfind("mtp", 0) == 0) q.mtp_ignored = true;
    }
  }
  return q;
}

// ─── enumeration helpers ─────────────────────────────────────────────────────

void Claim(std::vector<NemotronHTensor>& out, std::string name,
           std::string consumer) {
  out.push_back(NemotronHTensor{std::move(name), std::move(consumer)});
}

// An NVFP4 W4A16 group-16 weight: the packed nibbles, the per-16-block e4m3
// scale, and the fp32 global scale (spec §1; config_groups group_1 on the
// released checkpoint). WHICH kernel consumes them is W1/W4, not W3.
void ClaimNvfp4(std::vector<NemotronHTensor>& out, const std::string& prefix,
                const std::string& consumer, bool quantized) {
  Claim(out, prefix + ".weight", consumer);
  if (!quantized) return;
  Claim(out, prefix + ".weight_scale", consumer + ".weight_scale[nvfp4]");
  Claim(out, prefix + ".weight_scale_2", consumer + ".weight_scale_2[nvfp4]");
}

// An FP8 W8A8 static-scaled projection: the e4m3 weight, its fp32 weight scale
// and the fp32 static input scale (config_groups group_0, 46 targets).
// `quantized` gates the companions exactly as `ClaimNvfp4` does: an UNQUANTIZED
// producer ships the bare bf16 weight and no scales at all.
void ClaimFp8(std::vector<NemotronHTensor>& out, const std::string& prefix,
              const std::string& consumer, bool quantized) {
  Claim(out, prefix + ".weight", consumer);
  if (!quantized) return;
  Claim(out, prefix + ".weight_scale", consumer + ".weight_scale[fp8]");
  Claim(out, prefix + ".input_scale", consumer + ".input_scale[fp8]");
}

// One Mamba2 mixer (MambaMixer2, nemotron_h.py:373-389). `use_conv_bias` and
// `mamba_proj_bias` gate the two optional biases exactly as upstream does, and
// `quantized` gates the in/out projection scale companions. Released bf16
// NemotronH safetensors checkpoints ship NO `quantization_config` and no
// `mixer.{in,out}_proj.{weight_scale,input_scale}`; hard-coding the FP8 pair
// here enumerated 92 tensors such a checkpoint does not have (23 mamba blocks x
// 2 projections x 2 companions).
void ClaimMamba(std::vector<NemotronHTensor>& out, const NemotronHParams& p,
                const std::string& mixer, bool quantized) {
  ClaimFp8(out, mixer + ".in_proj", "mamba2.in_proj", quantized);
  ClaimFp8(out, mixer + ".out_proj", "mamba2.out_proj", quantized);
  if (p.mamba_proj_bias) {
    Claim(out, mixer + ".in_proj.bias", "mamba2.in_proj.bias");
    Claim(out, mixer + ".out_proj.bias", "mamba2.out_proj.bias");
  }
  Claim(out, mixer + ".conv1d.weight", "mamba2.conv1d");
  if (p.use_conv_bias) Claim(out, mixer + ".conv1d.bias", "mamba2.conv1d.bias");
  // `A_log` on disk; the mapper renames it to `A` in-module
  // (nemotron_h.py:719).
  Claim(out, mixer + ".A_log", "mamba2.A_log");
  Claim(out, mixer + ".D", "mamba2.D");
  Claim(out, mixer + ".dt_bias", "mamba2.dt_bias");
  Claim(out, mixer + ".norm.weight", "mamba2.gated_rmsnorm");
}

// One GQA attention mixer (NemotronHAttention, nemotron_h.py:503). q/k/v ship
// SEPARATE on disk; upstream stacks them into `qkv_proj` at load
// (hf_to_vllm_mapper orig_to_new_stacked, :719-723).
void ClaimAttention(std::vector<NemotronHTensor>& out,
                    const NemotronHParams& p, const std::string& mixer,
                    bool fp8_kv) {
  for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    Claim(out, mixer + "." + proj + ".weight", std::string("attn.") + proj);
    if (p.attention_bias) {
      Claim(out, mixer + "." + proj + ".bias",
            std::string("attn.") + proj + ".bias");
    }
  }
  if (fp8_kv) {
    Claim(out, mixer + ".k_proj.k_scale", "attn.k_scale[fp8-kv]");
    Claim(out, mixer + ".v_proj.v_scale", "attn.v_scale[fp8-kv]");
  }
}

// One non-gated relu² MoE block (NemotronHMoE, nemotron_h.py:126-256). There is
// NO gate_proj anywhere: FusedMoE is built with
// ckpt_names=("up_proj","down_proj","") (:220).
void ClaimMoe(std::vector<NemotronHTensor>& out, const NemotronHParams& p,
              const std::string& mixer, bool quantized) {
  Claim(out, mixer + ".gate.weight", "moe.router");
  // The noaux_tc score-correction bias registered on the gate (:158).
  Claim(out, mixer + ".gate.e_score_correction_bias", "moe.router.bias");
  for (int64_t e = 0; e < p.n_routed_experts; ++e) {
    const std::string expert = mixer + ".experts." + std::to_string(e);
    ClaimNvfp4(out, expert + ".up_proj", "moe.expert.up_proj", quantized);
    ClaimNvfp4(out, expert + ".down_proj", "moe.expert.down_proj", quantized);
  }
  if (p.n_shared_experts > 0) {
    const std::string shared = mixer + ".shared_experts";
    ClaimNvfp4(out, shared + ".up_proj", "moe.shared.up_proj", quantized);
    ClaimNvfp4(out, shared + ".down_proj", "moe.shared.down_proj", quantized);
  }
}

// One dense relu² MLP block (NemotronHMLP, nemotron_h.py:86-123). Absent from
// the driver checkpoint (no `-` in its schedule) but reachable through the
// class default schedule, so it is enumerated rather than left to be discovered.
//
// HONEST DEBT: the quantized companion layout here is DERIVED, not verified.
// `up_proj`/`down_proj` are the same ColumnParallelLinear/RowParallelLinear
// pair the experts use under the same ModelOpt config, so the NVFP4 triple is
// the only consistent reading — but no in-scope released NemotronH checkpoint
// ships an `mlp` block, so the enumeration gate cannot confirm it. It is
// recorded here rather than discovered later.
void ClaimMlp(std::vector<NemotronHTensor>& out, const NemotronHParams& p,
              const std::string& mixer, bool quantized) {
  ClaimNvfp4(out, mixer + ".up_proj", "mlp.up_proj", quantized);
  ClaimNvfp4(out, mixer + ".down_proj", "mlp.down_proj", quantized);
  if (p.mlp_bias) {
    Claim(out, mixer + ".up_proj.bias", "mlp.up_proj.bias");
    Claim(out, mixer + ".down_proj.bias", "mlp.down_proj.bias");
  }
}

}  // namespace

std::string_view NemotronHBlockName(NemotronHBlock block) {
  switch (block) {
    case NemotronHBlock::kMamba:
      return "mamba";
    case NemotronHBlock::kAttention:
      return "attention";
    case NemotronHBlock::kMoe:
      return "moe";
    case NemotronHBlock::kMlp:
      return "mlp";
  }
  return "unknown";
}

std::vector<int64_t> NemotronHParams::LayerIndices(NemotronHBlock block) const {
  std::vector<int64_t> out;
  for (size_t i = 0; i < layers_block_type.size(); ++i) {
    if (layers_block_type[i] == block) out.push_back(static_cast<int64_t>(i));
  }
  return out;
}

NemotronHParams ParseNemotronHParams(const HfConfig& config) {
  const json& doc = Raw(config);
  NemotronHParams p;

  // --- the schedule, which IS the depth ---
  p.layers_block_type = ResolveSchedule(
      doc, "layers_block_type", "hybrid_override_pattern",
      // configuration_nemotron_h.py:165.
      {NemotronHBlock::kMamba, NemotronHBlock::kMoe, NemotronHBlock::kAttention,
       NemotronHBlock::kMlp});
  if (p.layers_block_type.empty()) {
    Refuse("layers_block_type resolved to an empty schedule");
  }
  p.num_nextn_predict_layers = GetInt(doc, "num_nextn_predict_layers", 0);
  p.mtp_layers_block_type =
      ResolveSchedule(doc, "mtp_layers_block_type",
                      "mtp_hybrid_override_pattern",
                      // configuration_nemotron_h.py:180.
                      {NemotronHBlock::kAttention, NemotronHBlock::kMoe});
  if (p.num_nextn_predict_layers > 0 && p.mtp_layers_block_type.empty()) {
    // Mirror of validate_layer_type (configuration_nemotron_h.py:206-212).
    Refuse(
        "mtp_layers_block_type is required when num_nextn_predict_layers > 0");
  }

  // --- shared geometry ---
  p.hidden_size = GetInt(doc, "hidden_size", 4096);
  p.vocab_size = GetInt(doc, "vocab_size", 131072);
  p.max_position_embeddings = GetInt(doc, "max_position_embeddings", 4096);
  p.layer_norm_epsilon = GetDouble(doc, "layer_norm_epsilon", 1e-5);
  p.tie_word_embeddings = GetBool(doc, "tie_word_embeddings", false);

  // --- attention ---
  p.num_attention_heads = GetInt(doc, "num_attention_heads", 32);
  // Three states, not two (configuration_nemotron_h.py:97 default 8, :188-189
  // `if self.num_key_value_heads is None: = num_attention_heads`): ABSENT takes
  // the class default 8, an explicit `null` takes the query-head count, a value
  // is taken as-is.
  if (Has(doc, "num_key_value_heads")) {
    p.num_key_value_heads = GetInt(doc, "num_key_value_heads", 8);
  } else if (doc.contains("num_key_value_heads")) {
    p.num_key_value_heads = p.num_attention_heads;
  } else {
    p.num_key_value_heads = 8;
  }
  p.head_dim = GetInt(doc, "head_dim", 128);
  p.rope_theta = GetDouble(doc, "rope_theta", 10000.0);
  p.partial_rotary_factor = GetDouble(doc, "partial_rotary_factor", 1.0);
  p.attention_bias = GetBool(doc, "attention_bias", false);
  if (Has(doc, "sliding_window")) {
    p.sliding_window = GetInt(doc, "sliding_window", 0);
  }

  // --- Mamba2 (legacy aliases WIN, configuration_nemotron_h.py:145-155) ---
  p.mamba_num_heads = GetInt(doc, "mamba_num_heads", 128);
  p.mamba_head_dim = GetInt(doc, "mamba_head_dim", 64);
  p.n_groups = GetIntAliased(doc, "n_groups", "mamba_n_groups", 8);
  p.ssm_state_size = GetInt(doc, "ssm_state_size", 128);
  p.conv_kernel = GetIntAliased(doc, "conv_kernel", "mamba_d_conv", 4);
  p.chunk_size = GetIntAliased(doc, "chunk_size", "mamba_chunk_size", 128);
  p.expand = GetIntAliased(doc, "expand", "mamba_expand", 2);
  p.mamba_hidden_act = GetString(doc, "mamba_hidden_act", "silu");
  p.mamba_ssm_cache_dtype = GetString(doc, "mamba_ssm_cache_dtype", "float32");
  p.use_conv_bias = GetBoolAliased(doc, "use_conv_bias", "mamba_conv_bias", true);
  p.use_bias = GetBool(doc, "use_bias", false);
  p.mamba_proj_bias = GetBool(doc, "mamba_proj_bias", false);
  p.time_step_min = GetDoubleAliased(doc, "time_step_min", "mamba_dt_min", 1e-3);
  p.time_step_max = GetDoubleAliased(doc, "time_step_max", "mamba_dt_max", 1e-1);
  p.time_step_floor =
      GetDoubleAliased(doc, "time_step_floor", "mamba_dt_init_floor", 1e-4);
  if (p.mamba_num_heads <= 0 || p.mamba_head_dim <= 0 || p.n_groups <= 0 ||
      p.ssm_state_size <= 0 || p.conv_kernel <= 1) {
    Refuse("the Mamba2 geometry is degenerate (mamba_num_heads, mamba_head_dim, "
           "n_groups, ssm_state_size must be positive and conv_kernel > 1)");
  }

  // --- MoE ---
  p.n_routed_experts = GetInt(doc, "n_routed_experts", 8);
  p.num_experts_per_tok = GetInt(doc, "num_experts_per_tok", 2);
  p.moe_intermediate_size = GetInt(doc, "moe_intermediate_size", 7688);
  p.n_shared_experts = GetInt(doc, "n_shared_experts", 1);
  p.moe_shared_expert_intermediate_size =
      GetInt(doc, "moe_shared_expert_intermediate_size", 7688);
  p.n_group = GetInt(doc, "n_group", 1);
  p.topk_group = GetInt(doc, "topk_group", 1);
  p.routed_scaling_factor = GetDouble(doc, "routed_scaling_factor", 1.0);
  p.norm_topk_prob = GetBool(doc, "norm_topk_prob", true);
  p.moe_shared_expert_overlap = GetBool(doc, "moe_shared_expert_overlap", true);
  p.mlp_hidden_act = GetString(doc, "mlp_hidden_act", "relu2");
  if (Has(doc, "moe_latent_size")) {
    p.moe_latent_size = GetInt(doc, "moe_latent_size", 0);
    // The `fc1_latent_proj`/`fc2_latent_proj` pair (nemotron_h.py:191-207) is
    // out of scope for this row (spec §0), and silently ignoring the key would
    // build a differently-shaped MoE with no error.
    Refuse("moe_latent_size is set, but the latent MoE "
           "(fc1_latent_proj/fc2_latent_proj) is out of scope for this row "
           "(see .agents/specs/nemotron-h-model.md §0)");
  }
  if (p.mlp_hidden_act != "relu2") {
    Refuse("mlp_hidden_act '" + p.mlp_hidden_act +
           "' is not implemented (this architecture is the non-gated relu2 "
           "expert; see .agents/specs/nemotron-h-model.md W2)");
  }

  // --- dense MLP block ---
  if (doc.contains("intermediate_size") && doc.at("intermediate_size").is_array()) {
    // `get_nemotron_h_config_for_layer` / NemotronHPuzzleForCausalLM
    // (nemotron_h.py:283-288) is explicitly out of scope (spec §0).
    Refuse("a per-layer intermediate_size list (NemotronHPuzzleForCausalLM) is "
           "out of scope for this row (see .agents/specs/nemotron-h-model.md §0)");
  }
  p.intermediate_size = GetInt(doc, "intermediate_size", 21504);
  p.mlp_bias = GetBool(doc, "mlp_bias", false);

  // The recurrent-cache dtype is resolved independently of the conv dtype
  // (mamba_utils.py:99-104); refuse an alias we cannot represent HERE rather
  // than at cache-allocation time.
  const std::string& ssm = p.mamba_ssm_cache_dtype;
  if (!(ssm.empty() || ssm == "auto" || ssm == "float32" || ssm == "float" ||
        ssm == "float16" || ssm == "half" || ssm == "bfloat16")) {
    Refuse("mamba_ssm_cache_dtype '" + ssm +
           "' is not supported (expected auto, float32/float, float16/half or "
           "bfloat16)");
  }

  p.quant = ResolveQuantSurface(doc);

  // `num_hidden_layers` is DEPRECATED, and its setter ignores whatever the
  // checkpoint says (configuration_nemotron_h.py:233-238) — depth is the
  // schedule's LENGTH. Upstream only WARNS on a conflicting scalar
  // (:167-175), so this mirrors that rather than refusing: the released
  // config.json ships `num_hidden_layers: 52` alongside a 52-entry
  // `layers_block_type`, and a checkpoint that ships only the scalar becomes a
  // 4-block model upstream too. The behavior is pinned by a test so it stays a
  // deliberate mirror rather than an accident.
  (void)config.num_hidden_layers;
  return p;
}

void ParseNemotronHConfig(const HfConfig& config) {
  (void)ParseNemotronHParams(config);
}

vt::DType NemotronHSsmCacheDType(const NemotronHParams& params,
                                 vt::DType conv_dtype) {
  const std::string& dtype = params.mamba_ssm_cache_dtype;
  if (dtype.empty() || dtype == "auto") return conv_dtype;
  if (dtype == "float32" || dtype == "float") return vt::DType::kF32;
  if (dtype == "float16" || dtype == "half") return vt::DType::kF16;
  if (dtype == "bfloat16") return vt::DType::kBF16;
  // ParseNemotronHParams already refused anything else by name; this is the
  // unreachable arm kept so the mapping cannot silently widen.
  Refuse("mamba_ssm_cache_dtype '" + dtype + "' is not supported");
}

std::vector<NemotronHTensor> EnumerateNemotronHTensors(
    const NemotronHParams& params) {
  const NemotronHParams& p = params;
  // The backbone is quantized where the config groups say so; the MTP tower is
  // covered by the `mtp*` entry in `ignore` and ships bf16 with no companions.
  const bool quantized = p.quant.present;
  const bool mtp_quantized = quantized && !p.quant.mtp_ignored;

  std::vector<NemotronHTensor> out;
  out.reserve(4096);

  // --- root ---
  Claim(out, "backbone.embeddings.weight", "embed_tokens");
  Claim(out, "backbone.norm_f.weight", "final_norm");
  if (!p.tie_word_embeddings) {
    ClaimNvfp4(out, "lm_head", "lm_head", quantized);
  }

  // --- the 52 backbone layers ---
  for (size_t i = 0; i < p.layers_block_type.size(); ++i) {
    const std::string layer = "backbone.layers." + std::to_string(i);
    const std::string mixer = layer + ".mixer";
    // Every block kind carries the pre-mixer RMSNorm (nemotron_h.py:299, :339,
    // :391, :521).
    Claim(out, layer + ".norm.weight", "layer_norm");
    switch (p.layers_block_type[i]) {
      case NemotronHBlock::kMamba:
        ClaimMamba(out, p, mixer, quantized);
        break;
      case NemotronHBlock::kAttention:
        ClaimAttention(out, p, mixer, quantized && p.quant.fp8_kv_cache);
        break;
      case NemotronHBlock::kMoe:
        ClaimMoe(out, p, mixer, quantized);
        break;
      case NemotronHBlock::kMlp:
        ClaimMlp(out, p, mixer, quantized);
        break;
    }
  }

  // --- the MTP tower (nemotron_h_mtp.py:241-275) ---
  // total_layers = num_nextn_predict_layers * pattern_len; the FIRST block of
  // every step carries enorm/hnorm/eh_proj and the LAST carries
  // final_layernorm (:65-86, :152-173).
  const int64_t pattern_len =
      static_cast<int64_t>(p.mtp_layers_block_type.size());
  const int64_t mtp_layers = p.num_nextn_predict_layers * pattern_len;
  for (int64_t i = 0; i < mtp_layers; ++i) {
    const int64_t rel = i % pattern_len;
    const std::string layer = "mtp.layers." + std::to_string(i);
    const std::string mixer = layer + ".mixer";
    Claim(out, layer + ".norm.weight", "mtp.layer_norm");
    if (rel == 0) {
      Claim(out, layer + ".enorm.weight", "mtp.enorm");
      Claim(out, layer + ".hnorm.weight", "mtp.hnorm");
      Claim(out, layer + ".eh_proj.weight", "mtp.eh_proj");
    }
    if (rel == pattern_len - 1) {
      Claim(out, layer + ".final_layernorm.weight", "mtp.final_layernorm");
    }
    switch (p.mtp_layers_block_type[static_cast<size_t>(rel)]) {
      case NemotronHBlock::kAttention:
        // The MTP attention block is unquantized: no k_scale/v_scale ship for
        // it, which is what `ignore: [... "mtp*"]` buys.
        ClaimAttention(out, p, mixer, mtp_quantized && p.quant.fp8_kv_cache);
        break;
      case NemotronHBlock::kMoe:
        ClaimMoe(out, p, mixer, mtp_quantized);
        break;
      case NemotronHBlock::kMamba:
        ClaimMamba(out, p, mixer, mtp_quantized);
        break;
      case NemotronHBlock::kMlp:
        ClaimMlp(out, p, mixer, mtp_quantized);
        break;
    }
  }
  return out;
}

}  // namespace vllm
