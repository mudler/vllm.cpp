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

// An OPTIONAL string array. `layer_types` used to be read through a `Req`
// sibling of this, because the per-layer schedule is what the whole model is
// and a synthesized one splits the stack into layers of the wrong kind. That
// obligation has not been weakened, it has MOVED: the schedule is now required
// in either of its two on-disk spellings, and the refusal that stands in for
// the removed `ReqStrArray` is the one at the bottom of the schedule block
// below, which names both keys.
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

// A GGUF key that llama.cpp writes as EITHER one scalar or one value per
// block. `llama-model.cpp:1177` at the pinned `llama-cpp` release `b10451`
// reads `%s.attention.head_count_kv` through `get_key_or_arr(..., n_layer,
// false)`, so a per-layer array is as legal a spelling of that key as a scalar
// is, and a reader that accepts only the scalar refuses the published file
// (#2243). Returns false when the key is absent or is not an array; the caller
// then reads it as the scalar it is.
bool OptIntArray(const GgufFile& g, const std::string& key,
                 std::vector<int64_t>* out) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufArray) return false;
  out->clear();
  for (const GgufValue& e : std::get<GgufArray>(v->v).elems) {
    out->push_back(KvInt(e, key));
  }
  return true;
}

// A per-block array has ONE entry per block or the file is describing a stack
// this reader cannot place. Refuses by name, with the key and the shape found,
// rather than indexing a schedule of the wrong length.
void CheckPerLayerLength(const std::string& key, size_t found,
                         int64_t n_layers) {
  VT_CHECK(static_cast<int64_t>(found) == n_layers,
           "glm5_next gguf: key " + key + " is a per-layer array of " +
               std::to_string(found) + " entries but block_count is " +
               std::to_string(n_layers) +
               "; a per-layer array states exactly one value per block");
}

// A float key in the same scalar-or-array shape. `swiglu_clamp_exp` and
// `swiglu_clamp_shexp` arrive as `array[f32]` of `block_count` entries in the
// published artifact and as scalars from our own converter.
//
// UPSTREAM HAS ONE `swiglu_limit`, not one per layer. So a NON-UNIFORM array
// is a file whose clamp this config cannot represent, and taking element 0 of
// it would build a fluent model that clamps every layer with a number the file
// states for one. That is refused by name instead. Returns false when the key
// is absent.
bool OptPerLayerFloat(const GgufFile& g, const std::string& key,
                      int64_t n_layers, double* out) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr) return false;
  if (v->TypeId() != kGgufArray) {
    *out = KvFloat(*v, key);
    return true;
  }
  const std::vector<GgufValue>& elems = std::get<GgufArray>(v->v).elems;
  CheckPerLayerLength(key, elems.size(), n_layers);
  const double first = KvFloat(elems[0], key);
  for (size_t i = 1; i < elems.size(); ++i) {
    const double here = KvFloat(elems[i], key);
    VT_CHECK(here == first,
             "glm5_next gguf: key " + key + " states " + std::to_string(first) +
                 " for block 0 and " + std::to_string(here) + " for block " +
                 std::to_string(i) +
                 ", but this architecture has ONE `swiglu_limit` and no "
                 "per-layer clamp to put the second value in");
  }
  *out = first;
  return true;
}

// Does this file carry `blk.<layer>.<suffix>`?
bool HasLayerTensor(const GgufFile& g, int64_t layer, const char* suffix) {
  const std::string name = "blk." + std::to_string(layer) + "." + suffix;
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

// The FIRST dimension of a per-layer tensor, when the file carries it. `shape`
// is the on-disk ggml dims REVERSED into row-major order; every tensor asked
// about here is 1-D, so the reversal cannot change the answer.
bool LayerTensorDim0(const GgufFile& g, int64_t layer, const char* suffix,
                     int64_t* out) {
  const std::string name = "blk." + std::to_string(layer) + "." + suffix;
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name != name) continue;
    if (t.shape.empty()) return false;
    *out = t.shape[0];
    return true;
  }
  return false;
}

// THE NUMBER OF KDA HEADS, which a file can state in four places and the only
// published artifact of this model states in NONE of the first three.
//
// `%s.attention.linear_head_count` is OURS. llama.cpp spells it nowhere at the
// pinned release — `git grep linear_head_count b10451` is rc=1, tree-wide — so
// reading it with `ReqInt` refused the published
// `unsloth/GLM-5.3-Flash-GGUF` artifact on a key no llama.cpp writer emits
// (#2268). llama.cpp's own `glm5next` branch writes the same number under the
// `ssm.*` names its Kimi-Linear parent uses:
//
//   :78  add_ssm_inner_size(linear["num_heads"] * linear["head_dim"])
//   :79  add_ssm_state_size(linear["head_dim"])
//   :80  add_ssm_group_count(linear["num_heads"])
//     -- conversion/glm5next.py at ggml-org/llama.cpp refs/pull/27752/head
//        8a8d0bcc4, the revision `.agents/oracles/llama-cpp-glm5next.md` pins
//
// and the published file carries none of those three either: its 72 keys hold
// `kda.head_dim = 128` and `ssm.conv_kernel = 4` and no head count at all.
//
// So the last resort is the MODEL, which states the width whether or not the
// metadata does. `blk.<L>.ssm_a` is the per-head decay and is 1-D of exactly
// `num_heads` entries — `A_log` has that shape in the checkpoint, the reference
// converter only negates and exponentiates it (`conversion/glm5next.py:97-98`),
// and `scripts/convert-glm5-next-gguf.py:680` maps `self_attn.A_log -> ssm_a`
// unchanged. On the published artifact `blk.0.ssm_a` is `[64]` beside
// `kda.head_dim = 128`, and `blk.0.attn_q.weight` is `[4096, 8192] = 64 * 128`,
// so the file is self-consistent about a number it never names.
//
// It is DERIVED, never DEFAULTED, and this is where we are deliberately
// STRICTER than the secondary oracle. llama.cpp does not read a head count for
// this architecture at all: it sizes the recurrent state with `n_head() *
// n_embd_head_kda` and says why in the file — "note: n_embd_r()/n_embd_s()
// size the recurrent state with n_head()*n_embd_head_kda, which works only
// because linear_attn_config.num_heads == num_attention_heads"
// (`src/models/glm5next.cpp:121-122` at 8a8d0bcc4). That invariant holds on
// this checkpoint — `attention.head_count` is 64 and `blk.0.ssm_a` is `[64]` —
// and it is a coincidence of the checkpoint rather than a property of the
// architecture, which is the exact shape of the defect #2177 already cost this
// row: a value that is right here and silently wrong on the next file, with no
// gate able to see it. So the count is read from what the file STATES, and a
// file that states it nowhere is refused by name, listing every place that
// would have answered.
int64_t KdaHeadCount(const GgufFile& g, const std::string& p,
                     const std::vector<std::string>& layer_types,
                     int64_t kda_head_dim) {
  int64_t from_meta = 0;
  std::string meta_key;
  if (const GgufValue* v = g.FindKv(p + "attention.linear_head_count")) {
    meta_key = p + "attention.linear_head_count";
    from_meta = KvInt(*v, meta_key);
  } else if (const GgufValue* v_group = g.FindKv(p + "ssm.group_count")) {
    meta_key = p + "ssm.group_count";
    from_meta = KvInt(*v_group, meta_key);
  } else if (const GgufValue* v_inner = g.FindKv(p + "ssm.inner_size")) {
    const std::string inner_key = p + "ssm.inner_size";
    const int64_t inner = KvInt(*v_inner, inner_key);
    VT_CHECK(inner > 0 && inner % kda_head_dim == 0,
             "glm5_next gguf: " + inner_key + " is " + std::to_string(inner) +
                 " and " + p + "kda.head_dim is " +
                 std::to_string(kda_head_dim) +
                 "; llama.cpp writes the inner size as `num_heads * head_dim` "
                 "(conversion/glm5next.py:78), so the one must divide the "
                 "other");
    meta_key = inner_key + " / " + p + "kda.head_dim";
    from_meta = inner / kda_head_dim;
  }

  // The first `linear_attention` block, which is the one whose `ssm_a` states
  // the count. Every KDA layer of this model carries the same width, so one is
  // enough and the schedule says which one to look at.
  int64_t from_tensor = 0;
  bool have_tensor = false;
  for (size_t il = 0; il < layer_types.size(); ++il) {
    if (layer_types[il] != "linear_attention") continue;
    have_tensor =
        LayerTensorDim0(g, static_cast<int64_t>(il), "ssm_a", &from_tensor);
    break;
  }

  if (!meta_key.empty()) {
    VT_CHECK(from_meta > 0,
             "glm5_next gguf: " + meta_key + " is " +
                 std::to_string(from_meta) +
                 " and a linear-attention head count must be positive");
    // A CONTRADICTION check, not a completeness one. Absence proves nothing —
    // a metadata-only shard carries no `ssm_a` at all — but a file whose
    // metadata and whose weights state different KDA widths would build a
    // stack whose per-head reshape is wrong on every linear layer, and a token
    // gate would only see fluent garbage.
    VT_CHECK(!have_tensor || from_meta == from_tensor,
             "glm5_next gguf: " + meta_key + " states " +
                 std::to_string(from_meta) +
                 " linear-attention heads but the first `linear_attention` "
                 "block's `ssm_a` has " +
                 std::to_string(from_tensor) +
                 " entries, which is one entry per KDA head; the file states "
                 "this model's linear width twice and the two disagree");
    return from_meta;
  }

  VT_CHECK(have_tensor,
           "glm5_next gguf: this file states no linear-attention head count. "
           "It is not defaultable: it sets the per-head reshape of every KDA "
           "layer. Write one of " +
               p + "attention.linear_head_count, " + p + "ssm.group_count or " +
               p +
               "ssm.inner_size (`num_heads * head_dim`), or carry the "
               "`blk.<L>.ssm_a` tensor of a `linear_attention` block, whose "
               "length is the head count");
  VT_CHECK(from_tensor > 0,
           "glm5_next gguf: the first `linear_attention` block's `ssm_a` has " +
               std::to_string(from_tensor) +
               " entries and a linear-attention head count must be positive");
  return from_tensor;
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
      // `ssm_dt.bias`, NOT `ssm_dt`. The converter RENAMES the parameter before
      // the generic map sees it -- `if name.endswith(".dt_bias"): name =
      // name.rpartition(".dt_bias")[0] + ".dt_proj.bias"`
      // (llama.cpp #27752 @ 8a8d0bcc4, `conversion/glm5next.py`) -- so it
      // resolves through `MODEL_TENSOR.SSM_DT`'s `self_attn.dt_proj` row and
      // lands as `blk.N.ssm_dt.bias`. The converter's own comment says why:
      // "the time-step bias to be named like a bias so it is not loaded as a
      // MUL_MAT weight." The published `unsloth/GLM-5.3-Flash-GGUF` artifact
      // carries `ssm_dt.bias` on all 34 KDA blocks and no `ssm_dt` anywhere.
      {"self_attn.dt_bias", "ssm_dt.bias"},
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
      // `kv_b_proj` IS NOT HERE. It is SPLIT into two GGUF tensors, so one HF
      // name maps to two and it cannot live in a 1:1 table; see
      // `Glm5NextMlaKvBSplitTensorMap` below.
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

std::vector<Glm5NextTensorName> Glm5NextMlaKvBSplitTensorMap() {
  return {
      // ONE HF parameter, TWO GGUF tensors, and the second one is transposed.
      // `DeepseekV2Model.modify_tensors` (llama.cpp #27752 @ `8a8d0bcc4`,
      // `conversion/deepseek.py`, "note: MLA with the absorption optimization,
      // needs these two split and k_b_proj transposed"):
      //
      //     kv_b = W.view(n_head_kv, v_head_dim + qk_nope_head_dim, -1)
      //     k_b, v_b = split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
      //     k_b = k_b.transpose(1, 2)
      //
      // So `attn_k_b` is `[heads, kv_lora_rank, qk_nope_head_dim]` and
      // `attn_v_b` is `[heads, v_head_dim, kv_lora_rank]` — DIFFERENT shapes
      // even on this model, where `qk_nope_head_dim == v_head_dim == 256`:
      // ne [256, 512, 64] against ne [512, 256, 64] in the published
      // `unsloth/GLM-5.3-Flash-GGUF` artifact, which carries no
      // `attn_kv_b.weight` on any block.
      //
      // The HF side is spelled with a `[k]` / `[v]` selector rather than the
      // bare parameter name so this stays a 1:1 table a reader (and the
      // converter interop gate) can compare key for key. The selector is not a
      // tensor name and nothing looks it up.
      {"self_attn.kv_b_proj.weight[k]", "attn_k_b.weight"},
      {"self_attn.kv_b_proj.weight[v]", "attn_v_b.weight"},
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
  const auto kvb = Glm5NextMlaKvBSplitTensorMap();
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
    if (!is_kda) {
      for (const Glm5NextTensorName& tn : kvb) out.push_back(blk + tn.gguf);
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

  // BLOCKS ARE NOT LAYERS, and the difference is one whole decoder layer.
  //
  // llama.cpp's `block_count` counts the multi-token-prediction blocks on top
  // of the backbone, and it states the relationship in its own converters:
  // `self.block_count = self.hparams["num_hidden_layers"] +
  // self.hparams.get("num_nextn_predict_layers", 0)`
  // (`b10451:conversion/exaone.py:134`, and the same `+=` at
  // `b10451:conversion/deepseek.py:470` and `:545`). The published artifact is
  // that formula exactly: `block_count = 46`, `nextn_predict_layers = 1`, and
  // the released `config.json` declares `num_hidden_layers = 45`.
  //
  // So `num_hidden_layers` here is the BACKBONE depth, which is what the field
  // means on the `config.json` path and what `glm5_next.h` annotates it as. It
  // is resolved by subtraction rather than transcribed, because reading
  // `block_count` into it makes ONE model resolve to a 45-layer stack from its
  // config.json and a 46-layer stack from its GGUF, and the extra entry is the
  // MTP block. Nothing downstream would refuse that: `ParseGlm5NextParams`
  // sizes every schedule from `num_hidden_layers`, and a decoder stack built
  // from 46 would carry an extra layer made out of the MTP block, run, and
  // produce plausible tokens. A token gate cannot see that, which is the whole
  // reason this is subtracted here and asserted as a relationship rather than
  // as a number.
  //
  // The MTP block itself is READ, COUNTED and then DROPPED. W5b
  // (https://github.com/mudler/vllm.cpp/issues/2241) owns the head that would
  // consume it; until then the per-block schedules below are truncated to the
  // backbone and no layer is built for it, which is what the reference does
  // too.
  const int64_t n_blocks = ReqInt(gguf, p + "block_count");
  VT_CHECK(n_blocks > 0, "glm5_next gguf: block_count must be > 0");
  const int64_t n_mtp = OptInt(gguf, p + "nextn_predict_layers", 0);
  VT_CHECK(n_mtp >= 0,
           "glm5_next gguf: nextn_predict_layers is " + std::to_string(n_mtp) +
               " and a count of multi-token-prediction blocks cannot be "
               "negative");
  const int64_t n_layers = n_blocks - n_mtp;
  VT_CHECK(n_layers > 0,
           "glm5_next gguf: block_count is " + std::to_string(n_blocks) +
               " and nextn_predict_layers is " + std::to_string(n_mtp) +
               ", so the backbone would be " + std::to_string(n_layers) +
               " layers deep; llama.cpp writes block_count as "
               "num_hidden_layers + nextn_predict_layers, so this file states "
               "more MTP blocks than it has blocks");

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = n_layers;
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");

  // `attention.head_count_kv`, in llama.cpp's SCALAR-OR-ARRAY form, and the
  // array form is not a mis-typed scalar: it is THE per-layer schedule, and on
  // the only published artifact of this model it is the only schedule the file
  // carries (#2243, #2177).
  //
  // llama.cpp reads the key with `get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV,
  // hparams.n_head_kv_arr, hparams.n_layer(), false)`
  // (`b10451:src/llama-model.cpp:1177`) and then every hybrid family decides
  // which layers are linear from the SAME predicate — `is_recr_impl[i] =
  // hparams.n_head_kv(i) == 0`, spelled for this model's own KDA parent at
  // `b10451:src/models/kimi-linear.cpp:18` with the comment "KDA layers are
  // recurrent". So `0` means a KDA layer and any non-zero means an attention
  // layer, which for this architecture is DSA/MLA.
  //
  // THE ARRAY IS NOT A KV-HEAD COUNT AND MUST NOT BE READ AS ONE. Its non-zero
  // entries are `1`, the single latent KV head MLA has, while upstream's
  // `Glm5NextTextConfig` requires `num_attention_heads == num_key_value_heads`
  // and the released `config.json` states 64 for both. Assigning `1` here would
  // refuse the published file with a message about GQA, which is a true
  // statement about a number this file never made. The array form therefore
  // leaves `num_key_value_heads` at upstream's own `None -> num_attention_heads`
  // default and spends the values on the schedule below, which is the only
  // thing they mean.
  std::vector<int64_t> head_count_kv_arr;
  const bool kv_is_per_layer =
      OptIntArray(gguf, p + "attention.head_count_kv", &head_count_kv_arr);
  if (kv_is_per_layer) {
    CheckPerLayerLength(p + "attention.head_count_kv",
                        head_count_kv_arr.size(), n_blocks);
    c.num_key_value_heads = c.num_attention_heads;
  } else {
    c.num_key_value_heads =
        OptInt(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  }
  c.max_position_embeddings = ReqInt(gguf, p + "context_length");
  c.rms_norm_eps = ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  c.vocab_size = OptInt(gguf, p + "vocab_size", 0);
  if (c.vocab_size == 0) {
    c.vocab_size = gguf.Get("token_embd.weight").shape[0];
  }
  c.torch_dtype = "bfloat16";

  // MLA, in llama.cpp's OWN vocabulary and not in a private one.
  //
  // `%s.attention.key_length` is NOT this model's `qk_nope_head_dim`. For an
  // MLA model llama.cpp caches the LATENT, so the key it writes under that
  // name is the width of one cached K row — the latent plus the rope slice —
  // and the per-head query geometry is spelled by the two `_mla` keys beside
  // it. `b10451:conversion/deepseek.py`, `DeepseekModel.set_gguf_parameters`:
  //
  //   :345  add_key_length(kv_lora_rank + qk_rope_head_dim)
  //   :346  add_value_length(kv_lora_rank)
  //   :347  add_key_length_mla(qk_nope_head_dim + qk_rope_head_dim)
  //   :348  add_value_length_mla(v_head_dim)
  //   :369  add_rope_dimension_count(qk_rope_head_dim)
  //
  // On GLM-5.3-Flash — `kv_lora_rank 512`, `qk_nope_head_dim 256`,
  // `qk_rope_head_dim 0`, `v_head_dim 256` — that is `key_length 512`,
  // `value_length 512`, `key_length_mla 256`, `value_length_mla 256`, which is
  // the published `unsloth/GLM-5.3-Flash-GGUF` artifact's KV block exactly.
  //
  // This reader used to read `key_length` as `qk_nope_head_dim`, which is our
  // own converter's former spelling and nobody else's, so the published file
  // derived `256 - 512 = -256` as its rotary width and was refused as
  // self-contradictory (#2268). `scripts/convert-glm5-next-gguf.py` moved onto
  // llama.cpp's meaning in the same change, so ONE convention is written and
  // ONE is read, and a file in the old private spelling is REFUSED by the
  // `key_length < kv_lora_rank` check below rather than silently misread.
  const int64_t kv_lora_rank = ReqInt(gguf, p + "attention.kv_lora_rank");
  const int64_t qk_head_dim = ReqInt(gguf, p + "attention.key_length_mla");
  const int64_t key_length = ReqInt(gguf, p + "attention.key_length");
  const int64_t qk_rope_head_dim = key_length - kv_lora_rank;
  VT_CHECK(qk_rope_head_dim >= 0,
           "glm5_next gguf: attention.key_length is " +
               std::to_string(key_length) + " and attention.kv_lora_rank is " +
               std::to_string(kv_lora_rank) +
               ", so the rotary width would be " +
               std::to_string(qk_rope_head_dim) +
               "; llama.cpp writes attention.key_length as `kv_lora_rank + "
               "qk_rope_head_dim` (b10451:conversion/deepseek.py:345), which is "
               "never below kv_lora_rank, so this file spells key_length by "
               "some other convention");
  // The rotary width is stated TWICE by the same producer — once as
  // `key_length - kv_lora_rank` (deepseek.py:345) and once as
  // `rope.dimension_count` (deepseek.py:369) — so a disagreement between them
  // is a file one of whose two descriptions of the same geometry is wrong. On
  // a NoPE model that difference is exactly the thing a token gate could not
  // see, so it is a hard refusal rather than a first-wins.
  const int64_t rope_dim = OptInt(gguf, p + "rope.dimension_count", 0);
  VT_CHECK(qk_rope_head_dim == rope_dim,
           "glm5_next gguf: attention.key_length - attention.kv_lora_rank is " +
               std::to_string(qk_rope_head_dim) +
               " but rope.dimension_count is " + std::to_string(rope_dim) +
               "; the file states this model's rotary width twice and the two "
               "disagree");
  const int64_t qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;
  VT_CHECK(qk_nope_head_dim > 0,
           "glm5_next gguf: attention.key_length_mla is " +
               std::to_string(qk_head_dim) + " and the rotary width is " +
               std::to_string(qk_rope_head_dim) +
               ", so qk_nope_head_dim would be " +
               std::to_string(qk_nope_head_dim) +
               "; llama.cpp writes attention.key_length_mla as "
               "`qk_nope_head_dim + qk_rope_head_dim` "
               "(b10451:conversion/deepseek.py:347), and a non-positive "
               "no-rope width is not a geometry this model has");
  // `value_length` is the THIRD statement of the same latent, and llama.cpp
  // writes it as `kv_lora_rank` (deepseek.py:346). Checked when present rather
  // than required, because a file may legitimately omit it — but a file that
  // states it and disagrees with its own `kv_lora_rank` is contradicting
  // itself about the size of the cache.
  const int64_t value_length =
      OptInt(gguf, p + "attention.value_length", kv_lora_rank);
  VT_CHECK(value_length == kv_lora_rank,
           "glm5_next gguf: attention.value_length is " +
               std::to_string(value_length) + " but attention.kv_lora_rank is " +
               std::to_string(kv_lora_rank) +
               "; llama.cpp writes attention.value_length as the latent rank "
               "itself (b10451:conversion/deepseek.py:346), so the file states "
               "the latent width twice and the two disagree");
  // The one field `head_dim` is NOT: upstream forces `head_dim =
  // qk_rope_head_dim`, so it is 0 on this model. Setting the shared reader's
  // `head_dim` from `hidden_size / num_attention_heads` instead would give 64,
  // a number nothing upstream computes.
  c.head_dim = qk_rope_head_dim;

  // The per-layer schedules. Required, not defaulted: the whole model is the
  // interleave, and a file that does not state it is a file we cannot place a
  // single layer of.
  // TWO SPELLINGS, and the schedule is READ from whichever the file carries
  // rather than synthesized from a stride.
  //
  // `glm5next.layer_types` is a string array that only
  // `scripts/convert-glm5-next-gguf.py` writes, so our own output round-trips
  // through it. No other tool emits it, and the published
  // `unsloth/GLM-5.3-Flash-GGUF` artifacts carry the schedule ONLY as the
  // per-layer `attention.head_count_kv` array read above. Requiring
  // `layer_types` refused every one of those files, and falling back to
  // upstream's `idx % 4 != 3` pattern instead would be worse than refusing:
  // over the published checkpoint's 45 model layers that stride happens to be
  // right, so the wrong reader and the right reader agree on THIS file and
  // disagree silently on a fine-tune that moves one layer (#2177). The values
  // are on disk; they get read.
  //
  // The published 46-entry array is NOT `idx % 4 == 3` over its whole length.
  // `block_count` is 46 because it counts the multi-token-prediction block,
  // `nextn_predict_layers = 1`; entries 0..44 are the model's layers and entry
  // 45 is the MTP block, which is MLA-shaped and sits at `45 % 4 == 1`. A
  // consumer that re-derives the stride selects eleven MLA blocks where the
  // file states twelve, and reports nothing.
  const std::vector<std::string> declared =
      OptStrArray(gguf, p + "layer_types");
  if (!declared.empty()) {
    VT_CHECK(static_cast<int64_t>(declared.size()) == n_blocks,
             "glm5_next gguf: layer_types has " +
                 std::to_string(declared.size()) +
                 " entries but block_count is " + std::to_string(n_blocks));
  }

  // CROSS-CHECK, not a preference. A file that states the schedule twice and
  // disagrees with itself is a file one of whose two descriptions is wrong, and
  // silently taking either one loads a model whose attention kind is wrong on
  // the layers where they differ. Compared on the KIND and not on the string,
  // because `full_attention` is a legal `layer_types` spelling that
  // `LayerKindFromString` rewrites to `deepseek_sparse_attention`, and both are
  // attention layers with a non-zero KV head count.
  if (!declared.empty() && kv_is_per_layer) {
    for (int64_t il = 0; il < n_blocks; ++il) {
      const size_t i = static_cast<size_t>(il);
      const bool declared_kda = declared[i] == "linear_attention";
      const bool derived_kda = head_count_kv_arr[i] == 0;
      VT_CHECK(declared_kda == derived_kda,
               "glm5_next gguf: the file states its layer schedule twice and "
               "the two disagree at block " +
                   std::to_string(il) + ": layer_types says `" + declared[i] +
                   "` while attention.head_count_kv says " +
                   std::to_string(head_count_kv_arr[i]) +
                   " KV heads (0 means a `linear_attention` layer, non-zero an "
                   "attention layer)");
    }
  }

  std::vector<std::string> layer_types;
  if (!declared.empty()) {
    layer_types = declared;
  } else if (kv_is_per_layer) {
    for (int64_t v : head_count_kv_arr) {
      layer_types.push_back(v == 0 ? "linear_attention"
                                   : "deepseek_sparse_attention");
    }
  } else {
    VT_CHECK(false,
             "glm5_next gguf: this file states no per-layer attention "
             "schedule. The whole model is the interleave, so it cannot be "
             "defaulted: write either the string array " +
                 p + "layer_types or a per-layer " + p +
                 "attention.head_count_kv array of " +
                 std::to_string(n_blocks) +
                 " entries, 0 on each `linear_attention` block");
  }
  // TRUNCATE the per-block schedules to the backbone. Every array above is
  // `block_count` long because it describes BLOCKS; `layer_types` and the two
  // below describe LAYERS, and the trailing `n_mtp` entries are the MTP blocks
  // this port does not build (O2, and W5b owns the head). Dropping them here,
  // once, is what keeps `ParseGlm5NextParams` — which sizes all three schedules
  // from `num_hidden_layers` — from meeting a length it would have to refuse,
  // and what makes a GGUF and a config.json of the SAME model resolve to the
  // same stack.
  layer_types.resize(static_cast<size_t>(n_layers));
  c.layer_types = layer_types;

  std::vector<std::string> mlp_layer_types =
      OptStrArray(gguf, p + "mlp_layer_types");
  if (!mlp_layer_types.empty()) {
    VT_CHECK(static_cast<int64_t>(mlp_layer_types.size()) == n_blocks,
             "glm5_next gguf: mlp_layer_types has " +
                 std::to_string(mlp_layer_types.size()) +
                 " entries but block_count is " + std::to_string(n_blocks));
    mlp_layer_types.resize(static_cast<size_t>(n_layers));
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
  // `kda.head_dim`, with llama.cpp's OWN fallback. `src/models/glm5next.cpp`
  // at the pinned PR head 8a8d0bcc4 reads it optionally and falls back to
  // `ssm.state_size` — ":110  if (!ml.get_key(LLM_KV_KDA_HEAD_DIM,
  // hparams.n_embd_head_kda, false)) { :112 ml.get_key(LLM_KV_SSM_STATE_SIZE,
  // hparams.n_embd_head_kda); }", above the comment "older GGUFs store the KDA
  // head dim as ssm.state_size". That revision's converter writes only
  // `ssm.state_size` (`conversion/glm5next.py:79`), so the fallback is the
  // live arm for a file it produced, not a legacy path.
  int64_t kda_head_dim = OptInt(gguf, p + "kda.head_dim", 0);
  if (kda_head_dim == 0) kda_head_dim = ReqInt(gguf, p + "ssm.state_size");
  VT_CHECK(kda_head_dim > 0,
           "glm5_next gguf: the KDA head width is " +
               std::to_string(kda_head_dim) +
               " and a linear-attention head width must be positive");
  const int64_t kda_num_heads =
      KdaHeadCount(gguf, p, layer_types, kda_head_dim);
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
  std::vector<std::string> indexer_types =
      OptStrArray(gguf, p + "attention.indexer.types");
  if (!indexer_types.empty()) {
    VT_CHECK(static_cast<int64_t>(indexer_types.size()) == n_blocks,
             "glm5_next gguf: attention.indexer.types has " +
                 std::to_string(indexer_types.size()) +
                 " entries but block_count is " + std::to_string(n_blocks));
    indexer_types.resize(static_cast<size_t>(n_layers));
    text["indexer_types"] = indexer_types;
  }

  // MLA.
  text["q_lora_rank"] = ReqInt(gguf, p + "attention.q_lora_rank");
  text["kv_lora_rank"] = kv_lora_rank;
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
  // The clamped-SwiGLU limit, from EITHER of the two keys the writers use and
  // in either shape. Our converter writes both `swiglu_clamp_exp` and
  // `swiglu_clamp_shexp` as scalars from the one `text["swiglu_limit"]`
  // (`scripts/convert-glm5-next-gguf.py:1022-1023`); the published artifact
  // writes both as per-layer `array[f32]` of `block_count` entries. Reading
  // only the scalar form refused the published file one key after
  // `head_count_kv` did (#2243).
  //
  // BOTH are read, and a disagreement between them is refused. Upstream has ONE
  // `swiglu_limit` covering the routed and the shared expert alike, so a file
  // that states two different clamps is describing a model this config cannot
  // hold, and first-wins would pick one of them by the order of these lines.
  double swiglu_limit = 10.0;
  double clamp_exp = 0.0;
  double clamp_shexp = 0.0;
  const bool has_exp =
      OptPerLayerFloat(gguf, p + "swiglu_clamp_exp", n_blocks, &clamp_exp);
  const bool has_shexp =
      OptPerLayerFloat(gguf, p + "swiglu_clamp_shexp", n_blocks, &clamp_shexp);
  VT_CHECK(!(has_exp && has_shexp) || clamp_exp == clamp_shexp,
           "glm5_next gguf: " + p + "swiglu_clamp_exp is " +
               std::to_string(clamp_exp) + " and " + p +
               "swiglu_clamp_shexp is " + std::to_string(clamp_shexp) +
               ", but this architecture has ONE `swiglu_limit` for the routed "
               "and the shared expert alike");
  if (has_exp) {
    swiglu_limit = clamp_exp;
  } else if (has_shexp) {
    swiglu_limit = clamp_shexp;
  }
  text["swiglu_limit"] = swiglu_limit;
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
