// vllm.cpp ORIGINAL — the `glm5next` GGUF config builder and tensor name map.
// See the header for why this family owns its own translation unit and where
// each spelling comes from.
#include "vllm/model_executor/models/glm5_next_weights.h"

#include <algorithm>
#include <string>
#include <vector>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

const std::string kPrefix = "glm5next.";

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
      throw std::runtime_error("glm5_next gguf: key " + key +
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
  VT_CHECK(v != nullptr, "glm5_next gguf: missing metadata key " + key);
  return KvInt(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvInt(*v, key) : dflt;
}

double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "glm5_next gguf: missing metadata key " + key);
  return KvFloat(*v, key);
}

double OptFloat(const GgufFile& g, const std::string& key, double dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvFloat(*v, key) : dflt;
}

bool OptBool(const GgufFile& g, const std::string& key, bool dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvInt(*v, key) != 0 : dflt;
}

// A required STRING array. The per-layer schedules this architecture is built
// out of travel as string arrays, and reading one wrong splits the model into
// layers of the wrong kind — so a missing or wrong-typed array fails here
// rather than defaulting to a plausible pattern.
std::vector<std::string> ReqStrArray(const GgufFile& g,
                                     const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "glm5_next gguf: missing metadata key " + key);
  VT_CHECK(v->TypeId() == kGgufArray,
           "glm5_next gguf: key " + key + " must be an array");
  std::vector<std::string> out;
  for (const GgufValue& e : std::get<GgufArray>(v->v).elems) {
    VT_CHECK(e.TypeId() == kGgufString,
             "glm5_next gguf: key " + key + " must contain only strings");
    out.push_back(std::get<std::string>(e.v));
  }
  return out;
}

std::vector<std::string> OptStrArray(const GgufFile& g,
                                     const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufArray) return {};
  std::vector<std::string> out;
  for (const GgufValue& e : std::get<GgufArray>(v->v).elems) {
    if (e.TypeId() != kGgufString) return {};
    out.push_back(std::get<std::string>(e.v));
  }
  return out;
}

// Does this file carry `blk.<layer>.<suffix>`?
bool HasLayerTensor(const GgufFile& g, int64_t layer, const char* suffix) {
  const std::string name = "blk." + std::to_string(layer) + "." + suffix;
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

}  // namespace

bool IsGlm5NextGguf(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("general.architecture");
  return v != nullptr && v->TypeId() == kGgufString &&
         std::get<std::string>(v->v) == kGlm5NextGgufArch;
}

// ---------------------------------------------------------------------------
// The name maps. Spellings as in `scripts/convert-glm5-next-gguf.py`, which is
// the only writer of this container and whose own anchors are in this file's
// header.

std::vector<Glm5NextTensorName> Glm5NextCommonTensorMap() {
  return {
      {"input_layernorm.weight", "attn_norm.weight"},
      {"post_attention_layernorm.weight", "ffn_norm.weight"},
      // mHC, FLAT on the layer in the checkpoint rather than under `attn_hc.*`.
      // There is no `hc_head.*` at any layer; nothing here allocates one.
      {"hc_attn_fn", "hc_attn_fn.weight"},
      {"hc_attn_base", "hc_attn_base.weight"},
      {"hc_attn_scale", "hc_attn_scale.weight"},
      {"hc_ffn_fn", "hc_ffn_fn.weight"},
      {"hc_ffn_base", "hc_ffn_base.weight"},
      {"hc_ffn_scale", "hc_ffn_scale.weight"},
  };
}

std::vector<Glm5NextTensorName> Glm5NextKdaTensorMap() {
  return {
      {"self_attn.q_proj.weight", "attn_q.weight"},
      {"self_attn.k_proj.weight", "attn_k.weight"},
      {"self_attn.v_proj.weight", "attn_v.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      // THREE separate depthwise convs. The reference uses one grouped conv
      // over the concatenated [q; k; v] channel axis, so a loader either
      // concatenates in q, k, v order or runs three.
      {"self_attn.q_conv1d.weight", "ssm_conv1d_q.weight"},
      {"self_attn.k_conv1d.weight", "ssm_conv1d_k.weight"},
      {"self_attn.v_conv1d.weight", "ssm_conv1d_v.weight"},
      {"self_attn.f_a_proj.weight", "ssm_f_a.weight"},
      {"self_attn.f_b_proj.weight", "ssm_f_b.weight"},
      {"self_attn.g_a_proj.weight", "ssm_g_a.weight"},
      {"self_attn.g_b_proj.weight", "ssm_g_b.weight"},
      {"self_attn.b_proj.weight", "ssm_beta.weight"},
      {"self_attn.A_log", "ssm_a"},
      {"self_attn.dt_bias", "ssm_dt"},
      {"self_attn.o_norm.weight", "ssm_norm.weight"},
  };
}

std::vector<Glm5NextTensorName> Glm5NextDsaTensorMap() {
  return {
      {"self_attn.q_a_proj.weight", "attn_q_a.weight"},
      {"self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight"},
      {"self_attn.q_b_proj.weight", "attn_q_b.weight"},
      {"self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight"},
      {"self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight"},
      {"self_attn.kv_b_proj.weight", "attn_kv_b.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      {"self_attn.indexer.wq_b.weight", "indexer.attn_q_b.weight"},
      {"self_attn.indexer.wk.weight", "indexer.attn_k.weight"},
      // A LayerNorm WITH bias, not an RMSNorm: the checkpoint carries
      // `indexer.k_norm.bias`, which settles it.
      {"self_attn.indexer.k_norm.weight", "indexer.k_norm.weight"},
      {"self_attn.indexer.k_norm.bias", "indexer.k_norm.bias"},
      {"self_attn.indexer.weights_proj.weight", "indexer.proj.weight"},
      // The k-pool compression stage. Unconditional in the reference: the
      // config's `index_kpool_compress: true` is an inert kwarg the config
      // class does not declare, and the parameters exist on every DSA layer.
      {"self_attn.indexer.index_kpool_compress_ape",
       "indexer_compressor_ape.weight"},
      {"self_attn.indexer.index_kpool_compress_gate",
       "indexer_compressor_gate.weight"},
  };
}

std::vector<Glm5NextTensorName> Glm5NextDenseMlpTensorMap() {
  return {
      {"mlp.gate_proj.weight", "ffn_gate.weight"},
      {"mlp.up_proj.weight", "ffn_up.weight"},
      {"mlp.down_proj.weight", "ffn_down.weight"},
  };
}

std::vector<Glm5NextTensorName> Glm5NextSparseMlpTensorMap() {
  return {
      {"mlp.gate.weight", "ffn_gate_inp.weight"},
      {"mlp.gate.e_score_correction_bias", "exp_probs_b.bias"},
      {"mlp.shared_experts.gate_proj.weight", "ffn_gate_shexp.weight"},
      {"mlp.shared_experts.up_proj.weight", "ffn_up_shexp.weight"},
      {"mlp.shared_experts.down_proj.weight", "ffn_down_shexp.weight"},
  };
}

std::vector<Glm5NextTensorName> Glm5NextStackedExpertTensorMap() {
  return {
      {"mlp.experts.{e}.gate_proj.weight", "ffn_gate_exps.weight"},
      {"mlp.experts.{e}.up_proj.weight", "ffn_up_exps.weight"},
      {"mlp.experts.{e}.down_proj.weight", "ffn_down_exps.weight"},
  };
}

std::vector<Glm5NextTensorName> Glm5NextVisionTensorMap() {
  return {
      {"patch_embed.proj.weight", "v.patch_embd.weight"},
      {"patch_embed.proj.bias", "v.patch_embd.bias"},
      {"post_layernorm.weight", "v.post_ln.weight"},
      {"downsample.weight", "v.downsample.weight"},
      {"downsample.bias", "v.downsample.bias"},
      {"merger.proj.weight", "v.merger.proj.weight"},
      {"merger.gate_proj.weight", "v.merger.gate.weight"},
      {"merger.up_proj.weight", "v.merger.up.weight"},
      {"merger.down_proj.weight", "v.merger.down.weight"},
      {"merger.post_projection_norm.weight", "v.merger.norm.weight"},
      {"merger.post_projection_norm.bias", "v.merger.norm.bias"},
  };
}

std::vector<Glm5NextTensorName> Glm5NextVisionBlockTensorMap() {
  return {
      {"norm1.weight", "ln1.weight"},
      {"norm2.weight", "ln2.weight"},
      {"attn.qkv.weight", "attn_qkv.weight"},
      {"attn.qkv.bias", "attn_qkv.bias"},
      {"attn.proj.weight", "attn_out.weight"},
      {"attn.proj.bias", "attn_out.bias"},
      {"attn.q_norm.weight", "attn_q_norm.weight"},
      {"attn.k_norm.weight", "attn_k_norm.weight"},
      {"mlp.gate_proj.weight", "ffn_gate.weight"},
      {"mlp.gate_proj.bias", "ffn_gate.bias"},
      {"mlp.up_proj.weight", "ffn_up.weight"},
      {"mlp.up_proj.bias", "ffn_up.bias"},
      {"mlp.down_proj.weight", "ffn_down.weight"},
      {"mlp.down_proj.bias", "ffn_down.bias"},
  };
}

std::vector<std::string> Glm5NextExpectedGgufTensors(
    const Glm5NextParams& params) {
  // `ParseGlm5NextParams` always sizes both schedules to `num_hidden_layers`,
  // but this is a public entry point and a hand-built `Glm5NextParams` is a
  // legal argument. Refuse rather than index past the end: an out-of-bounds
  // read here would be undefined behaviour that no gate would attribute to the
  // caller who built the struct.
  // Named `layer_count` rather than `n`: the seven range-`for` loops below each
  // bind a `Glm5NextTensorName`, and a one-letter local at function scope is
  // the declaration every one of them hid. MSVC C4456 is fatal under /W4 /WX.
  const size_t layer_count = static_cast<size_t>(std::max<int64_t>(
      params.num_hidden_layers, 0));
  VT_CHECK(params.layer_types.size() == layer_count &&
               params.mlp_layer_types.size() == layer_count,
           "glm5_next: Glm5NextExpectedGgufTensors needs both per-layer "
           "schedules sized to num_hidden_layers (" +
               std::to_string(params.num_hidden_layers) + "), got " +
               std::to_string(params.layer_types.size()) + " and " +
               std::to_string(params.mlp_layer_types.size()));
  std::vector<std::string> out;
  out.emplace_back("token_embd.weight");
  out.emplace_back("output_norm.weight");
  // `lm_head` is a separate tensor: `tie_word_embeddings` is FALSE here, so the
  // output projection is not the gather table transposed.
  if (!params.tie_word_embeddings) out.emplace_back("output.weight");

  const auto common = Glm5NextCommonTensorMap();
  const auto kda = Glm5NextKdaTensorMap();
  const auto dsa = Glm5NextDsaTensorMap();
  const auto dense = Glm5NextDenseMlpTensorMap();
  const auto sparse = Glm5NextSparseMlpTensorMap();
  const auto experts = Glm5NextStackedExpertTensorMap();

  for (int64_t il = 0; il < params.num_hidden_layers; ++il) {
    const std::string blk = "blk." + std::to_string(il) + ".";
    for (const Glm5NextTensorName& tn : common) out.push_back(blk + tn.gguf);
    const bool is_kda = params.layer_types[static_cast<size_t>(il)] ==
                        Glm5NextLayerKind::kLinearAttention;
    for (const Glm5NextTensorName& tn : (is_kda ? kda : dsa)) {
      out.push_back(blk + tn.gguf);
    }
    const bool is_dense = params.mlp_layer_types[static_cast<size_t>(il)] ==
                          Glm5NextMlpKind::kDense;
    if (is_dense) {
      for (const Glm5NextTensorName& tn : dense) out.push_back(blk + tn.gguf);
    } else {
      for (const Glm5NextTensorName& tn : sparse) out.push_back(blk + tn.gguf);
      for (const Glm5NextTensorName& tn : experts) out.push_back(blk + tn.gguf);
    }
  }

  if (params.has_vision) {
    for (const Glm5NextTensorName& tn : Glm5NextVisionTensorMap()) {
      out.emplace_back(tn.gguf);
    }
    const auto vblock = Glm5NextVisionBlockTensorMap();
    for (int64_t ib = 0; ib < params.vision.depth; ++ib) {
      const std::string blk = "v.blk." + std::to_string(ib) + ".";
      for (const Glm5NextTensorName& tn : vblock) out.push_back(blk + tn.gguf);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------

HfConfig Glm5NextHfConfigFromGguf(const GgufFile& gguf) {
  VT_CHECK(IsGlm5NextGguf(gguf),
           "glm5_next gguf: general.architecture must be '" +
               std::string(kGlm5NextGgufArch) + "'");
  const std::string p = kPrefix;

  HfConfig c;
  // The HF `model_type` and architecture string, verbatim from the released
  // `zai-org/GLM-5.3-Flash` config.json (read 2026-08-27). They are NOT derived
  // from the GGUF architecture key: `glm5next` is llama.cpp's family key, not a
  // model class, and the registry keys on the HF class.
  c.model_type = "glm5_next";
  c.architectures = {"Glm5NextForConditionalGeneration"};

  const int64_t n_layers = ReqInt(gguf, p + "block_count");
  VT_CHECK(n_layers > 0, "glm5_next gguf: block_count must be > 0");

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = n_layers;
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      OptInt(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  c.max_position_embeddings = ReqInt(gguf, p + "context_length");
  c.rms_norm_eps = ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  c.vocab_size = OptInt(gguf, p + "vocab_size", 0);
  if (c.vocab_size == 0) {
    c.vocab_size = gguf.Get("token_embd.weight").shape[0];
  }
  c.torch_dtype = "bfloat16";

  // MLA. `key_length_mla` is `qk_head_dim` and `key_length` is
  // `qk_nope_head_dim`, so the rope slice is their DIFFERENCE — derived, not
  // transcribed, and then cross-checked against the `rope.dimension_count` the
  // converter writes independently. A file where the two disagree is a file
  // one of whose two descriptions of the same geometry is wrong, and on a NoPE
  // model that difference is exactly the thing a token gate could not see.
  const int64_t qk_head_dim = ReqInt(gguf, p + "attention.key_length_mla");
  const int64_t qk_nope_head_dim = ReqInt(gguf, p + "attention.key_length");
  const int64_t qk_rope_head_dim = qk_head_dim - qk_nope_head_dim;
  const int64_t rope_dim = OptInt(gguf, p + "rope.dimension_count", 0);
  VT_CHECK(qk_rope_head_dim == rope_dim,
           "glm5_next gguf: attention.key_length_mla - attention.key_length is " +
               std::to_string(qk_rope_head_dim) +
               " but rope.dimension_count is " + std::to_string(rope_dim) +
               "; the file states this model's rotary width twice and the two "
               "disagree");
  // The one field `head_dim` is NOT: upstream forces `head_dim =
  // qk_rope_head_dim`, so it is 0 on this model. Setting the shared reader's
  // `head_dim` from `hidden_size / num_attention_heads` instead would give 64,
  // a number nothing upstream computes.
  c.head_dim = qk_rope_head_dim;

  // The per-layer schedules. Required, not defaulted: the whole model is the
  // interleave, and a file that does not state it is a file we cannot place a
  // single layer of.
  const std::vector<std::string> layer_types =
      ReqStrArray(gguf, p + "layer_types");
  VT_CHECK(static_cast<int64_t>(layer_types.size()) == n_layers,
           "glm5_next gguf: layer_types has " +
               std::to_string(layer_types.size()) +
               " entries but block_count is " + std::to_string(n_layers));
  c.layer_types = layer_types;

  std::vector<std::string> mlp_layer_types =
      OptStrArray(gguf, p + "mlp_layer_types");
  if (!mlp_layer_types.empty()) {
    VT_CHECK(static_cast<int64_t>(mlp_layer_types.size()) == n_layers,
             "glm5_next gguf: mlp_layer_types has " +
                 std::to_string(mlp_layer_types.size()) +
                 " entries but block_count is " + std::to_string(n_layers));
  }

  // THE INVENTORY CONTRADICTION CHECK, and it is a contradiction check rather
  // than a completeness check on purpose. A file can legitimately be missing
  // tensors (a partial artifact, a dry run), so ABSENCE proves nothing. But a
  // `blk.N` that carries KDA tensors while the metadata calls layer N
  // `deepseek_sparse_attention` — or the converse — is a file whose two
  // descriptions of the same layer disagree, and taking the metadata's word for
  // it loads a wrong model quietly. `ssm_a` and `attn_kv_a_mqa.weight` are the
  // discriminators: each exists on exactly one kind and on no other.
  for (int64_t il = 0; il < n_layers; ++il) {
    const std::string& kind = layer_types[static_cast<size_t>(il)];
    const bool says_kda = kind == "linear_attention";
    const bool has_kda = HasLayerTensor(gguf, il, "ssm_a");
    const bool has_mla = HasLayerTensor(gguf, il, "attn_kv_a_mqa.weight");
    VT_CHECK(!(says_kda && has_mla),
             "glm5_next gguf: layer_types calls blk." + std::to_string(il) +
                 " `linear_attention` but it carries "
                 "`attn_kv_a_mqa.weight`, which only a "
                 "`deepseek_sparse_attention` layer has");
    VT_CHECK(!(!says_kda && has_kda),
             "glm5_next gguf: layer_types calls blk." + std::to_string(il) +
                 " `" + kind +
                 "` but it carries `ssm_a`, which only a `linear_attention` "
                 "layer has");
  }

  // MoE.
  c.num_experts = ReqInt(gguf, p + "expert_count");
  c.num_experts_per_tok = ReqInt(gguf, p + "expert_used_count");
  c.moe_intermediate_size = ReqInt(gguf, p + "expert_feed_forward_length");
  c.shared_expert_intermediate_size = OptInt(
      gguf, p + "expert_shared_feed_forward_length", c.moe_intermediate_size);
  c.intermediate_size = ReqInt(gguf, p + "feed_forward_length");

  // KDA, under llama.cpp's `kda.*` / `ssm.*` namespaces.
  const int64_t kda_head_dim = ReqInt(gguf, p + "kda.head_dim");
  const int64_t kda_num_heads = ReqInt(gguf, p + "attention.linear_head_count");
  const int64_t kda_conv = ReqInt(gguf, p + "ssm.conv_kernel");
  c.linear_num_key_heads = kda_num_heads;
  c.linear_num_value_heads = kda_num_heads;
  c.linear_key_head_dim = kda_head_dim;
  c.linear_value_head_dim = kda_head_dim;
  c.linear_conv_kernel_dim = kda_conv;

  // --- the HF-shaped `raw`, which is what makes this ONE parser -------------
  //
  // Every name below is the released `zai-org/GLM-5.3-Flash` config.json's, not
  // a name invented here, so an implementer of a later wave reads the same
  // spellings whether the config came from a GGUF or from a config.json — and
  // `ParseGlm5NextParams` is reached identically from both.
  nlohmann::json raw = nlohmann::json::object();
  raw["model_type"] = c.model_type;
  raw["architectures"] = c.architectures;
  raw["tie_word_embeddings"] = false;

  nlohmann::json text = nlohmann::json::object();
  text["model_type"] = "glm5_next_text";
  text["hidden_size"] = c.hidden_size;
  text["num_hidden_layers"] = n_layers;
  text["vocab_size"] = c.vocab_size;
  text["num_attention_heads"] = c.num_attention_heads;
  text["num_key_value_heads"] = c.num_key_value_heads;
  text["max_position_embeddings"] = c.max_position_embeddings;
  text["rms_norm_eps"] = c.rms_norm_eps;
  text["intermediate_size"] = c.intermediate_size;
  text["layer_types"] = layer_types;
  if (!mlp_layer_types.empty()) text["mlp_layer_types"] = mlp_layer_types;
  const std::vector<std::string> indexer_types =
      OptStrArray(gguf, p + "attention.indexer.types");
  if (!indexer_types.empty()) text["indexer_types"] = indexer_types;

  // MLA.
  text["q_lora_rank"] = ReqInt(gguf, p + "attention.q_lora_rank");
  text["kv_lora_rank"] = ReqInt(gguf, p + "attention.kv_lora_rank");
  text["qk_nope_head_dim"] = qk_nope_head_dim;
  text["qk_rope_head_dim"] = qk_rope_head_dim;
  text["v_head_dim"] = ReqInt(gguf, p + "attention.value_length_mla");

  // The DSA indexer, including the three k-pool keys that have no upstream
  // spelling because no upstream implements this indexer.
  text["index_head_dim"] = ReqInt(gguf, p + "attention.indexer.key_length");
  text["index_n_heads"] = ReqInt(gguf, p + "attention.indexer.head_count");
  text["index_topk"] = ReqInt(gguf, p + "attention.indexer.top_k");
  text["index_kpool"] = ReqInt(gguf, p + "attention.indexer.kpool");
  text["index_kpool_always_select_tail"] =
      OptBool(gguf, p + "attention.indexer.kpool_always_select_tail", true);

  // KDA. `gate_lower_bound` is the parameter the whole port hinges on: -5.0
  // selects `-bound * sigmoid(exp(A_log) * (g + dt_bias))`, and its ABSENCE
  // selects the `-exp(A_log) * softplus(g + dt_bias)` our Kimi-Linear KDA
  // implements. Two different functions, and the sign of `decay_rate` differs.
  // It is carried through `linear_attn_config` rather than as a flat
  // `linear_lower_bound` so that the GGUF path exercises the same sub-object
  // descent and the same `safe_gate` rule a config.json does.
  nlohmann::json lin = nlohmann::json::object();
  lin["head_dim"] = kda_head_dim;
  lin["num_heads"] = kda_num_heads;
  lin["short_conv_kernel_size"] = kda_conv;
  const GgufValue* bound = gguf.FindKv(p + "kda.gate_lower_bound");
  if (bound != nullptr) {
    lin["gate_lower_bound"] = KvFloat(*bound, p + "kda.gate_lower_bound");
  } else {
    // ABSENT is not the same as -5.0 and must not be silently promoted to it:
    // absent means the softplus branch. `safe_gate` false is what says so, and
    // it is stated explicitly rather than left to a default that means the
    // opposite.
    lin["gate_lower_bound"] = nullptr;
    lin["safe_gate"] = false;
  }
  text["linear_attn_config"] = lin;

  // mHC.
  text["hc_mult"] = ReqInt(gguf, p + "hyper_connection.count");
  text["hc_sinkhorn_iters"] =
      ReqInt(gguf, p + "hyper_connection.sinkhorn_iterations");
  text["hc_eps"] = ReqFloat(gguf, p + "hyper_connection.epsilon");

  // MoE.
  text["n_routed_experts"] = c.num_experts;
  text["n_shared_experts"] = OptInt(gguf, p + "expert_shared_count", 1);
  text["num_experts_per_tok"] = c.num_experts_per_tok;
  text["moe_intermediate_size"] = c.moe_intermediate_size;
  text["n_group"] = OptInt(gguf, p + "expert_group_count", 1);
  text["topk_group"] = OptInt(gguf, p + "expert_group_used_count", 1);
  text["routed_scaling_factor"] = OptFloat(gguf, p + "expert_weights_scale", 2.5);
  text["norm_topk_prob"] = OptBool(gguf, p + "expert_weights_norm", true);
  text["swiglu_limit"] = OptFloat(gguf, p + "swiglu_clamp_exp", 10.0);
  raw["text_config"] = text;

  // The six placeholder ids, on the wrapper. Image and video share one id in
  // the emitted sequence, so all six travel together or a reader classifies
  // every video frame as an image.
  for (const char* key : {"image_token_id", "video_token_id",
                          "image_start_token_id", "image_end_token_id",
                          "video_start_token_id", "video_end_token_id"}) {
    const GgufValue* v = gguf.FindKv(p + key);
    if (v != nullptr) raw[key] = KvInt(*v, p + key);
  }

  if (gguf.FindKv(p + "vision.block_count") != nullptr) {
    nlohmann::json vis = nlohmann::json::object();
    vis["model_type"] = "glm5_next_vision";
    vis["depth"] = ReqInt(gguf, p + "vision.block_count");
    vis["hidden_size"] = ReqInt(gguf, p + "vision.embedding_length");
    vis["intermediate_size"] = ReqInt(gguf, p + "vision.feed_forward_length");
    vis["num_heads"] = ReqInt(gguf, p + "vision.head_count");
    vis["patch_size"] = ReqInt(gguf, p + "vision.patch_size");
    vis["image_size"] = ReqInt(gguf, p + "vision.image_size");
    vis["spatial_merge_size"] = ReqInt(gguf, p + "vision.spatial_merge_size");
    vis["temporal_patch_size"] =
        ReqInt(gguf, p + "vision.temporal_patch_size");
    vis["out_hidden_size"] = ReqInt(gguf, p + "vision.out_embedding_length");
    vis["projection_intermediate_size"] =
        ReqInt(gguf, p + "vision.projection_intermediate_size");
    vis["rms_norm_eps"] =
        ReqFloat(gguf, p + "vision.attention.layer_norm_rms_epsilon");
    vis["swiglu_limit"] = OptFloat(gguf, p + "vision.swiglu_clamp", 10.0);
    raw["vision_config"] = vis;
  }

  c.raw = raw;
  // Refuse HERE rather than at first forward. A GGUF whose metadata does not
  // describe a `glm5_next` is refused by the SAME validator a malformed
  // config.json meets, which is the whole point of synthesizing an HF-shaped
  // `raw` instead of a private struct.
  (void)ParseGlm5NextParams(c);
  return c;
}

}  // namespace vllm
