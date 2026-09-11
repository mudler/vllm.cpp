// Maple (`maple`) GGUF loading — SELF-CONTAINED on PUBLIC APIs. The heavy
// lifting (block dequant, residency routing, keep-quant stacked-expert towers)
// goes through the public seams:
//   - DequantGgufRowToBf16 / DequantGgufRowToF32  (gguf_dequant.h)
//   - RouteGgufTensor + GgufLoadPolicy            (gguf_keep_quant.h)
//   - OwnGgufQuantBlocks / OwnGgufF16             (qwen3_5_gguf_weights.h —
//     declared PUBLIC there for cross-family keep-quant loads)
//   - dense_loaders::MakeOwned                    (dense_weight_loaders.h)
//
// Structure mirrors qwen3_5_gguf_weights.cpp's loader shape but with only the
// maple-specific logic: arch gate -> HfConfigFromGguf -> per-layer loaders.
//
// Grounding:
//   - deepgrove-ai/llama.cpp src/models/maple.cpp load_arch_hparams /
//     load_arch_tensors @ 7e30f3a — which KV keys and tensor names exist.
//   - llama-arch.cpp tensor-name table: token_embd, output(_norm),
//     blk.%d.attn_{norm,q,k,v,output,q_norm,k_norm}, blk.%d.ffn_{norm,
//     gate_inp,gate_exps,up_exps,down_exps}.
#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"  // MakeOwned
#include "vllm/model_executor/models/maple.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks/OwnGgufF16

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {

namespace {

using dense_loaders::MakeOwned;

std::string Blk(int64_t il, const std::string& suffix) {
  return "blk." + std::to_string(il) + "." + suffix;
}

int64_t KvInt(const GgufValue& v, const std::string& key) {
  VT_CHECK(v.TypeId() == kGgufI64 || v.TypeId() == kGgufU64 ||
               v.TypeId() == kGgufI32 || v.TypeId() == kGgufU32 ||
               v.TypeId() == kGgufBool,
           key + ": expected an integer");
  switch (v.TypeId()) {
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI32: return static_cast<int64_t>(std::get<int32_t>(v.v));
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    case kGgufU32: return static_cast<int64_t>(std::get<uint32_t>(v.v));
    default: return std::get<int64_t>(v.v);
  }
}

double KvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvInt(v, key));
}

int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, key + ": missing required gguf kv");
  return KvInt(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v ? KvInt(*v, key) : dflt;
}

double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, key + ": missing required gguf kv");
  return KvFloat(*v, key);
}

// Dequant a whole gguf row to bf16 (sidecar-scale aware via the same global
// scale resolution DequantGgufRowToF32 uses; maple GGUFs carry none today).
std::vector<uint16_t> DqRowBf16(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo& ti = g.Get(name);
  const GgmlTypeTraits& tt = GgmlTraits(ti.ggml_type);
  const int64_t numel = tt.block_bytes > 0
                            ? ti.nbytes / tt.block_bytes * tt.block_elems
                            : ti.nbytes;
  return DequantGgufRowToBf16(ti.ggml_type, ti.data, numel);
}

// bf16 tensor dequantized then owned with `shape` (the qwen3_5 OwnBf16
// contract; error text names this arch).
OwnedTensor MapleOwnBf16(const GgufFile& g, const std::string& name,
                         const std::vector<int64_t>& shape) {
  std::vector<uint16_t> dq = DqRowBf16(g, name);
  int64_t numel = 1;
  for (int64_t s : shape) numel *= s;
  VT_CHECK(numel == static_cast<int64_t>(dq.size()),
           "maple gguf: element-count mismatch for " + name);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
  return o;
}

// Maple norm weights are stored RAW (w as trained): measured blk.0.attn_q_norm
// first8 ≈ [1.156, 0.703, 0.535, ...] and output_norm ≈ 1.3 — a (w+1) store
// would make attn_norm read ≈1.0 everywhere. llama-maple consumes these bytes
// unshifted (build_norm reads model.layers[il].attn_norm directly), so unlike
// qwen3_5's OwnNormMinus1 contract there is NO -1 rewrite for this arch.
OwnedTensor MapleOwnNormMinus1(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo& ti = g.Get(name);
  const GgmlTypeTraits& tt = GgmlTraits(ti.ggml_type);
  const int64_t numel = tt.block_bytes > 0
                            ? ti.nbytes / tt.block_bytes * tt.block_elems
                            : ti.nbytes;
  std::vector<float> dq =
      DequantGgufRowToF32(ti.ggml_type, ti.data, numel, 1.0F);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {static_cast<int64_t>(dq.size())});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < dq.size(); ++i) dst[i] = vt::F32ToBF16(dq[i]);
  return o;
}

// Route a matmul/expert weight through the policy. Keep-quant/keep-f16 keeps
// the file's own [N,K] blocks (MatmulBT reads them directly); expand-bf16
// dequantizes to a bf16 [N,K] nk=true owner.
OwnedTensor MapleOwnWeight(const GgufFile& g, const std::string& name,
                           const GgufLoadPolicy& pol, GgufTensorRole role) {
  const GgufTensorInfo& t = g.Get(name);
  const GgufResidency r = pol.Route(t, role);
  if (r == GgufResidency::kKeepQuant) {
    const int64_t n = t.shape[0];
    const int64_t k = t.shape[1];
    return OwnGgufQuantBlocks(t, n, k, /*row_offset=*/0, &g);
  }
  if (r == GgufResidency::kKeepF16) {
    return OwnGgufF16(t, t.shape[0], t.shape[1], /*row_offset=*/0, &g);
  }
  VT_CHECK(r == GgufResidency::kExpandBf16,
           "maple gguf: unexpected residency for " + name);
  // Expand to bf16 in the file's OWN [N,K] orientation (nk=true — MatmulBT).
  OwnedTensor o = MapleOwnBf16(g, name, {t.shape[0], t.shape[1]});
  o.nk = true;  // MakeOwned defaults nk=false; this IS a matmul weight ([N,K]).
  return o;
}

// Stacked routed-expert tower [E, out, in]: exactly one of
//   * bf16-expand  -> E transposed per-expert owners (reference path), or
//   * keep-quant   -> ONE whole [E*out, in] block tower (grouped path).
void MapleLoadExperts(const GgufFile& g, int64_t il, const std::string& stem,
                      int64_t num_experts, const GgufLoadPolicy& pol,
                      MoeBlockWeights& m, const std::string& which) {
  const std::string name = Blk(il, stem);
  const GgufTensorInfo& ti = g.Get(name);
  VT_CHECK(ti.shape.size() == 3 && ti.shape[0] == num_experts,
           "maple gguf: expected [E,out,in] expert tensor " + stem);
  const int64_t out_dim = ti.shape[1];
  const int64_t in_dim = ti.shape[2];

  const GgufResidency r =
      pol.Route(ti, GgufTensorRole::kStackedExpertWeight);  // ONE audit event
  if (r == GgufResidency::kKeepQuant || r == GgufResidency::kKeepF16) {
    // Whole stacked tower, E*out whole rows starting at 0 (never cuts a block).
    OwnedTensor tower = OwnGgufQuantBlocks(ti, num_experts * out_dim, in_dim,
                                           /*row_offset=*/0, &g);
    if (which == "gate") m.expert_gate_kq = std::move(tower);
    else if (which == "up") m.expert_up_kq = std::move(tower);
    else m.expert_down_kq = std::move(tower);
    return;
  }
  VT_CHECK(r == GgufResidency::kExpandBf16,
           "maple gguf: unexpected expert residency for " + stem);
  // Reference arm: E transposed bf16 [in,out] slabs (ExpertMlp layout).
  std::vector<uint16_t> dq = DqRowBf16(g, name);
  const int64_t per = out_dim * in_dim;
  auto& vec = which == "gate"   ? m.expert_gate
              : which == "up"   ? m.expert_up
                                : m.expert_down;
  vec.reserve(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
    auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
    const uint16_t* src = dq.data() + e * per;
    // TransposeBf16 semantics: [out,in] file order -> [in,out] Matmul-B.
    for (int64_t i = 0; i < in_dim; ++i)
      for (int64_t j = 0; j < out_dim; ++j)
        dst[i * out_dim + j] = src[j * in_dim + i];
    vec.push_back(std::move(o));
  }
}

}  // namespace

bool IsMapleGguf(const GgufFile& gguf) {
  const GgufValue* arch_v = gguf.FindKv("general.architecture");
  if (arch_v == nullptr || arch_v->TypeId() != kGgufString) return false;
  return std::get<std::string>(arch_v->v) == "maple";
}

HfConfig MapleHfConfigFromGguf(const GgufFile& gguf) {
  const GgufValue* arch_v = gguf.FindKv("general.architecture");
  VT_CHECK(arch_v != nullptr && arch_v->TypeId() == kGgufString,
           "maple gguf: general.architecture must be a string");
  const std::string arch = std::get<std::string>(arch_v->v);
  VT_CHECK(arch == "maple", "maple gguf: unexpected architecture '" + arch + "'");
  const std::string p = arch + ".";

  HfConfig c;
  c.model_type = arch;
  // general.architecture is llama.cpp's family key; the HF class name is what
  // ModelRegistry resolves against.
  c.architectures = {"MapleForCausalLM"};

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      OptInt(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  c.head_dim =
      OptInt(gguf, p + "attention.key_length",
             c.num_attention_heads > 0 ? c.hidden_size / c.num_attention_heads
                                       : 0);

  const GgufValue* vocab_kv = gguf.FindKv(p + "vocab_size");
  c.vocab_size = vocab_kv ? KvInt(*vocab_kv, p + "vocab_size")
                          : gguf.Get("token_embd.weight").shape[0];

  // MoE (deepgrove key spellings).
  c.num_experts = OptInt(gguf, p + "expert_count", 0);
  c.num_experts_per_tok = OptInt(gguf, p + "expert_used_count", 0);
  c.moe_intermediate_size =
      OptInt(gguf, p + "expert_feed_forward_length", 0);
  c.shared_expert_intermediate_size =
      OptInt(gguf, p + "expert_shared_feed_forward_length", 0);  // maple: none
  // NOTE: HF `intermediate_size` (4096 on maple-preview) has NO dense FFN in
  // this arch — deliberately not mapped onto c.intermediate_size.

  // RoPE / norm / context.
  const GgufValue* freq = gguf.FindKv(p + "rope.freq_base");
  c.rope_theta = freq ? KvFloat(*freq, p + "rope.freq_base") : 10000.0;
  c.rotary_dim = OptInt(gguf, p + "rope.dimension_count", 0);
  c.rms_norm_eps = ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 131072);
  c.torch_dtype = "bfloat16";

  // Sliding window (config.json sliding_window=512).
  c.sliding_window = OptInt(gguf, p + "attention.sliding_window", 0);

  // Layer pattern: the fork stores an explicit per-layer SWA bool array under
  // attention.sliding_window_pattern (maple.cpp:10 get_key_or_arr onto
  // hparams.is_swa_impl); fall back to the trained [S,S,S,F] repeat.
  c.layer_types.reserve(static_cast<size_t>(c.num_hidden_layers));
  const GgufValue* swa = gguf.FindKv(p + "attention.sliding_window_pattern");
  if (swa != nullptr && swa->TypeId() == kGgufArray) {
    const GgufArray& arr = std::get<GgufArray>(swa->v);
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      const bool is_swa =
          l < static_cast<int64_t>(arr.elems.size()) &&
          KvInt(arr.elems[static_cast<size_t>(l)],
                "sliding_window_pattern") != 0;
      c.layer_types.emplace_back(is_swa ? "sliding_attention"
                                        : "full_attention");
    }
  } else {
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      c.layer_types.emplace_back((l % 4 == 3) ? "full_attention"
                                              : "sliding_attention");
  }
  return c;
}

MapleWeights LoadMapleFromGguf(const GgufFile& gguf, const HfConfig& config,
                               const GgufLoadPolicy* policy,
                               vt::DeviceType device) {
  const GgufLoadPolicy env_policy = GgufLoadPolicy::FromEnv(device);
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env_policy;

  MapleWeights w;
  w.tie_word_embeddings = false;

  // Embedding gather table [vocab, H], untied head [H, vocab].
  w.embed_tokens = MapleOwnBf16(gguf, "token_embd.weight",
                                {config.vocab_size, config.hidden_size});
  w.embed_tokens.nk = false;  // gather table, never a GEMM
  w.lm_head = MapleOwnWeight(gguf, "output.weight", pol,
                             GgufTensorRole::kEmbeddingTable);
  // output.weight's GGUF bytes ARE [vocab, H] row-major (ne=[H,vocab]) — the
  // MatmulBT [N,K] orientation that OwnGgufQuantBlocks/MapleOwnBf16 already
  // mark nk=true. The old explicit `nk=false` here contradicted the bytes and
  // made the head run as a silently transposed GEMM.
  w.final_norm = MapleOwnNormMinus1(gguf, "output_norm.weight");

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t il = 0; il < config.num_hidden_layers; ++il) {
    MapleLayerWeights layer;
    layer.is_swa =
        config.layer_types[static_cast<size_t>(il)] == "sliding_attention";

    const int64_t H = config.hidden_size;
    const int64_t Hq = config.num_attention_heads;
    const int64_t Hkv = config.num_key_value_heads;
    const int64_t Dh = config.head_dim;
    const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;

    layer.input_layernorm =
        MapleOwnNormMinus1(gguf, Blk(il, "attn_norm.weight"));
    layer.post_attention_layernorm =
        MapleOwnNormMinus1(gguf, Blk(il, "ffn_norm.weight"));

    // Attention projections ship SPLIT (attn_q/k/v): build the merged
    // [qdim+2kdim, H] raw-NK owner AttnBlock consumes by row-concatenating
    // q|k|v expanded slabs.
    std::vector<uint16_t> wq = DqRowBf16(gguf, Blk(il, "attn_q.weight"));
    std::vector<uint16_t> wk = DqRowBf16(gguf, Blk(il, "attn_k.weight"));
    std::vector<uint16_t> wv = DqRowBf16(gguf, Blk(il, "attn_v.weight"));
    VT_CHECK(static_cast<int64_t>(wq.size()) == qdim * H &&
                 static_cast<int64_t>(wk.size()) == kdim * H &&
                 static_cast<int64_t>(wv.size()) == kdim * H,
             "maple gguf: attention projection shape mismatch at blk." +
                 std::to_string(il));
    OwnedTensor qkv = MakeOwned(vt::DType::kBF16, {qdim + 2 * kdim, H});
    qkv.nk = true;
    auto* dst = reinterpret_cast<uint16_t*>(qkv.bytes.data());
    std::memcpy(dst, wq.data(), static_cast<size_t>(wq.size()) * 2);
    std::memcpy(dst + wq.size(), wk.data(),
                static_cast<size_t>(wk.size()) * 2);
    std::memcpy(dst + wq.size() + wk.size(), wv.data(),
                static_cast<size_t>(wv.size()) * 2);
    layer.attn.qkv_proj = std::move(qkv);

    layer.attn.o_proj = MapleOwnWeight(gguf, Blk(il, "attn_output.weight"),
                                       pol, GgufTensorRole::kMatmulWeight);

    // Per-head q/k RMSNorm (use_qk_norm=true): VALUE transforms (w - 1),
    // never keep-quant.
    layer.attn.q_norm = MapleOwnNormMinus1(gguf, Blk(il, "attn_q_norm.weight"));
    layer.attn.k_norm = MapleOwnNormMinus1(gguf, Blk(il, "attn_k_norm.weight"));

    // MoE block: router F32 stored as... GGUF ffn_gate_inp is F32; route it
    // like any weight (expands to bf16 unless keep rules say otherwise).
    layer.moe.router_gate = MapleOwnWeight(gguf, Blk(il, "ffn_gate_inp.weight"),
                                           pol, GgufTensorRole::kMatmulWeight);
    MapleLoadExperts(gguf, il, "ffn_gate_exps.weight", config.num_experts, pol,
                     layer.moe, "gate");
    MapleLoadExperts(gguf, il, "ffn_up_exps.weight", config.num_experts, pol,
                     layer.moe, "up");
    MapleLoadExperts(gguf, il, "ffn_down_exps.weight", config.num_experts, pol,
                     layer.moe, "down");
    // NO shared expert: maple ships none; the shared fields stay EMPTY.

    w.layers.push_back(std::move(layer));
  }
  return w;
}

}  // namespace vllm
