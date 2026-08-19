// DFlash draft loading from a `dflash`-arch GGUF (`SPEC-DFLASH-GGUF`, GD1-GD2).
//
// Unlike the MTP head (`SPEC-MTP-GGUF`), this one DOES go through the
// TensorResolver seam, and the difference is principled rather than stylistic:
//
//   * the MTP head sits inside a Qwen3.5 TRUNK GGUF, whose RMSNorm weights are
//     stored (w + 1) and whose matmul weights carry the trunk's keep-quant
//     residency routing. Those conventions live in the trunk loader's helpers,
//     so the head had to reuse them directly;
//   * a DFlash draft is its OWN file written by `DFlashModel(Qwen3Model)`, which
//     does NOT inherit the Qwen3Next `+1` norm shift (verified in llama.cpp
//     `conversion/qwen.py`: the shift is in `Qwen3NextModel.modify_tensors`, and
//     `DFlashModel` overrides only set_vocab / set_gguf_parameters /
//     filter_tensors). Its norms are therefore RAW, exactly as the safetensors
//     draft stores them.
//
// So a resolver that dequantizes to bf16 and hands back torch-[N, K] views
// reproduces the safetensors layout faithfully, and the whole existing
// `LoadQwen3DFlash(get, ...)` body - including its qkv / gate_up row
// concatenation - is reused unchanged. Dequantizing wholesale is affordable here
// because the draft is a handful of layers (~1 GB at Q4_K_M), not a full model.
#include "vllm/model_executor/models/qwen3_dflash_gguf.h"

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

int64_t KvI64(const GgufValue& v, const std::string& key) {
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
      VT_CHECK(false, "dflash gguf: key " + key + " is not an integer");
      return 0;
  }
}

double KvF64(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvI64(v, key));
}

int64_t ReqI64(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "dflash gguf: missing required key '" + key + "'");
  return KvI64(*v, key);
}

int64_t OptI64(const GgufFile& g, const std::string& key, int64_t fallback) {
  const GgufValue* v = g.FindKv(key);
  return v == nullptr ? fallback : KvI64(*v, key);
}

double ReqF64(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "dflash gguf: missing required key '" + key + "'");
  return KvF64(*v, key);
}

// Returns a POINTER, not a reference: -Wdangling-reference cannot see through
// std::get on a variant member and flags the reference form.
const GgufArray* KvArray(const GgufValue& v, const std::string& key) {
  VT_CHECK(v.TypeId() == kGgufArray, "dflash gguf: key " + key + " is not an array");
  return &std::get<GgufArray>(v.v);
}

}  // namespace

HfConfig MakeDflashGgufConfig(const GgufFile& gguf) {
  const GgufValue* arch = gguf.FindKv("general.architecture");
  VT_CHECK(arch != nullptr && arch->TypeId() == kGgufString &&
               std::get<std::string>(arch->v) == "dflash",
           "dflash gguf: general.architecture must be 'dflash'");
  const std::string p = "dflash.";

  HfConfig c;
  c.model_type = "dflash";
  c.architectures = {"DFlashQwen3Model"};
  c.hidden_size = ReqI64(gguf, p + "embedding_length");
  c.num_hidden_layers = ReqI64(gguf, p + "block_count");
  c.num_attention_heads = ReqI64(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      OptI64(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  c.head_dim = ReqI64(gguf, p + "attention.key_length");
  c.rotary_dim = c.head_dim;
  c.rope_theta = ReqF64(gguf, p + "rope.freq_base");
  c.intermediate_size = ReqI64(gguf, p + "feed_forward_length");
  c.rms_norm_eps = ReqF64(gguf, p + "attention.layer_norm_rms_epsilon");
  c.sliding_window = OptI64(gguf, p + "attention.sliding_window", 0);
  // vocab_size is deliberately NOT read: a DFlash draft GGUF carries no vocab
  // key and no embed/lm_head tensors, because it SHARES the target's. The
  // caller fills it from the target (LoadDflashDraft already does).
  c.vocab_size = 0;

  // layer_types from the sliding-window pattern: true => sliding, false => full.
  // Absent pattern means every layer is full attention.
  c.layer_types.assign(static_cast<size_t>(c.num_hidden_layers),
                       "full_attention");
  if (const GgufValue* swa = gguf.FindKv(p + "attention.sliding_window_pattern")) {
    const GgufArray* pat_a = KvArray(*swa, p + "attention.sliding_window_pattern");
    VT_CHECK(static_cast<int64_t>(pat_a->elems.size()) == c.num_hidden_layers,
             "dflash gguf: sliding_window_pattern length must equal block_count");
    for (size_t i = 0; i < pat_a->elems.size(); ++i) {
      if (KvI64(pat_a->elems[i], "sliding_window_pattern") != 0) {
        c.layer_types[i] = "sliding_attention";
      }
    }
  }

  c.raw = nlohmann::json::object();
  c.raw["block_size"] = ReqI64(gguf, p + "block_size");

  // SPEC-DFLASH2 W1 (#1314): `dflash.attention.causal` is the GGUF spelling of
  // the HF top-level `is_causal`, and it resolves in the same precedence --
  // ahead of `dflash_config.causal` and ahead of the pattern-derived rule
  // above. The published DFlash2 GGUF declares it FALSE beside an all-true
  // sliding-window pattern, so the pattern alone answers CAUSAL for every layer
  // and the drafter loses acceptance with nothing to see. Optional: a DFlash1
  // GGUF declares no such key and keeps the pattern-derived answer.
  if (const GgufValue* causal = gguf.FindKv(p + "attention.causal")) {
    c.raw["is_causal"] = KvI64(*causal, p + "attention.causal") != 0;
  }

  // target_layer_ids: llama.cpp's DFlashModel::set_gguf_parameters writes
  // `[i + 1 for i in target_layer_ids]`, so the stored list is OFFSET BY ONE
  // from the HF value the engine expects. Undo it here. Getting this wrong is
  // invisible to every shape check - num_taps stays correct and each tap is a
  // valid layer index - and would surface only as degraded acceptance.
  const GgufValue* tl = gguf.FindKv(p + "target_layers");
  VT_CHECK(tl != nullptr, "dflash gguf: missing required key 'dflash.target_layers'");
  const GgufArray* stored = KvArray(*tl, p + "target_layers");
  VT_CHECK(!stored->elems.empty(), "dflash gguf: dflash.target_layers must be non-empty");
  nlohmann::json ids = nlohmann::json::array();
  for (const GgufValue& e : stored->elems) {
    const int64_t v = KvI64(e, p + "target_layers");
    VT_CHECK(v >= 1,
             "dflash gguf: dflash.target_layers entries are stored +1-offset and "
             "must therefore be >= 1");
    ids.push_back(v - 1);
  }
  nlohmann::json dcfg = nlohmann::json::object();
  dcfg["target_layer_ids"] = ids;
  // The mask token rides the STANDARD tokenizer KV, not a dflash-specific one
  // (llama.cpp conversion/qwen.py uses add_mask_token_id).
  const GgufValue* mask = gguf.FindKv("tokenizer.ggml.mask_token_id");
  VT_CHECK(mask != nullptr,
           "dflash gguf: missing 'tokenizer.ggml.mask_token_id' (the block "
           "drafter masks nothing without it)");
  dcfg["mask_token_id"] = KvI64(*mask, "tokenizer.ggml.mask_token_id");
  c.raw["dflash_config"] = std::move(dcfg);
  return c;
}

bool IsDflash2Gguf(const GgufFile& gguf, std::string* matched_key) {
  // Ordered so the message names the most specific key a reader can grep for.
  static const char* kDflash2Keys[] = {
      "dflash.selector_rank",
      "dflash.selector_top_k",
      "dflash.conv_kernel_size",
  };
  for (const char* key : kDflash2Keys) {
    if (gguf.FindKv(key) != nullptr) {
      if (matched_key != nullptr) *matched_key = key;
      return true;
    }
  }
  return false;
}

// Owns the dequantized bf16 buffers the resolver hands out views over. The
// views must outlive the LoadQwen3DFlash call, which is why this is a separate
// object the caller keeps alive rather than a function-local cache.
class DflashGgufTensors {
 public:
  explicit DflashGgufTensors(const GgufFile& gguf) : gguf_(gguf) {}

  const StTensor& Get(const std::string& hf_name) {
    const auto it = views_.find(hf_name);
    if (it != views_.end()) return it->second;
    const std::string gguf_name = MapName(hf_name);
    const GgufTensorInfo& t = gguf_.Get(gguf_name);
    int64_t numel = 1;
    for (const int64_t d : t.shape) numel *= d;
    std::vector<uint16_t> bf16 =
        DequantGgufRowToBf16(t.ggml_type, t.data, numel);
    StTensor view;
    view.dtype = "BF16";
    view.shape = t.shape;  // already torch [N, K]
    const size_t bytes = bf16.size() * sizeof(uint16_t);
    auto& owned = buffers_[hf_name];
    owned = std::move(bf16);
    view.data = reinterpret_cast<const uint8_t*>(owned.data());
    view.nbytes = bytes;
    return views_.emplace(hf_name, std::move(view)).first->second;
  }

 private:
  // HF name -> GGUF name. `model.` prefixes are stripped by the caller's naming
  // (the safetensors loader asks for "layers.N...", "fc.weight", ...).
  static std::string MapName(const std::string& n) {
    if (n == "fc.weight") return "fc.weight";
    if (n == "hidden_norm.weight") return "enc.output_norm.weight";
    if (n == "norm.weight") return "output_norm.weight";
    VT_CHECK(n.rfind("layers.", 0) == 0,
             "dflash gguf: unexpected tensor name '" + n + "'");
    const size_t dot = n.find('.', 7);
    VT_CHECK(dot != std::string::npos, "dflash gguf: malformed name '" + n + "'");
    const std::string idx = n.substr(7, dot - 7);
    const std::string rest = n.substr(dot + 1);
    const std::string blk = "blk." + idx + ".";
    static const std::unordered_map<std::string, std::string> kSuffix = {
        {"self_attn.q_proj.weight", "attn_q.weight"},
        {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.v_proj.weight", "attn_v.weight"},
        {"self_attn.o_proj.weight", "attn_output.weight"},
        {"self_attn.q_norm.weight", "attn_q_norm.weight"},
        {"self_attn.k_norm.weight", "attn_k_norm.weight"},
        {"input_layernorm.weight", "attn_norm.weight"},
        {"post_attention_layernorm.weight", "ffn_norm.weight"},
        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"},
    };
    const auto it = kSuffix.find(rest);
    VT_CHECK(it != kSuffix.end(),
             "dflash gguf: no GGUF name for '" + rest + "'");
    return blk + it->second;
  }

  const GgufFile& gguf_;
  std::unordered_map<std::string, std::vector<uint16_t>> buffers_;
  std::unordered_map<std::string, StTensor> views_;
};

Qwen3DFlashWeights LoadQwen3DFlashFromGguf(const GgufFile& gguf,
                                           const HfConfig& config,
                                           int64_t num_taps,
                                           int32_t mask_token_id) {
  DflashGgufTensors tensors(gguf);
  const TensorResolver get = [&tensors](const std::string& name) -> const StTensor& {
    return tensors.Get(name);
  };
  return LoadQwen3DFlash(get, config, num_taps, mask_token_id);
}

}  // namespace vllm
