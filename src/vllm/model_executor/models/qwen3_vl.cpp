// Qwen3-VL (`Qwen3VLForConditionalGeneration`) — M2c e2e image->text forward.
// See qwen3_vl.h for the port map. The forked VL decode reuses the landed
// dense_attn machinery (Dev/DBuf/ResidentWeight/KvSlice/AttnBlock preamble) and
// forks only the three vision-conditioned points (inputs_embeds merge, 3-section
// MRoPE, DeepStack inject at layers 0/1/2). ADDITIVE TU — no shared-forward edit.
#include "vllm/model_executor/models/qwen3_vl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"  // LinearMethod seam
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"    // Dev/DBuf/ResidentWeight/KvSlice
#include "vllm/model_executor/models/dense_weight_loaders.h"  // dense_loaders::*
#include "vllm/model_executor/models/interfaces.h"          // #607 L3 SkipTowerForModalities
#include "vllm/model_executor/models/model_registry.h"      // MM-ENGINE-FORWARD: ModelForwardInput/MultiModalForwardInput/ModelRegistry
#include "vllm/model_executor/models/qwen3_5.h"             // GdnStateCache (ModelForwardInput field)
#include "vllm/model_executor/models/qwen3_vl_text.h"       // Qwen3VL{GetRopeIndex,MergeMultimodal,ComputeDeepstack}
#include "vllm/v1/attention/backends/gdn_attn.h"            // v1::GDNAttentionMetadata (ModelForwardInput field)
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/unaligned.h"  // LoadUnaligned — mmap'd payloads have no alignment guarantee

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using namespace dense_attn;  // Dev, DBuf, ResidentWeight, ResidentWeightF32, KvSlice, MakeTensor, Reshape

// One forward STEP of the forked VL decode: given the already-merged host bf16
// embeddings [T*H], the 3-D MRoPE positions [3*T], the (possibly empty) DeepStack
// [L*T*H], the step attention metadata, the persistent paged KV, and the bf16
// cos|sin cache, return the LAST row's logits [1, vocab] as a ForwardLogits
// carrier — ON DEVICE by default (device_tensor; the runner/greedy loop samples
// straight off device via vt::GreedyArgmax, no full-vocab D2H), or host on the
// opt-out. RUNNER-ROUTE: the ONLY production step is the ENGINE registered forward
// (MakeRegistryStep -> ModelRegistry::Forward -> ForwardQwen3VL), so every driver
// (image/video/registry) enters through ModelRegistry::Forward and gets the same
// on-device logits carrier the text models return (qwen3_dense.cpp WrapDeviceLogits).
using VLStepFn = std::function<ForwardLogits(
    const std::vector<uint16_t>& inputs_embeds_bf16,
    const std::vector<int32_t>& positions3, int64_t num_tokens,
    const std::vector<uint16_t>& deepstack_bf16, const CommonAttentionMetadata& meta,
    std::vector<PagedKvCache>& attn_kv, const Tensor& cos_sin_cache)>;

// bf16 raw bits <-> f32 host conversions.
std::vector<float> Bf16BitsToF32(const uint16_t* p, int64_t n) {
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
  return out;
}
std::vector<uint16_t> F32ToBf16Bits(const float* p, int64_t n) {
  std::vector<uint16_t> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::F32ToBF16(p[i]);
  return out;
}
// Round f32 values through bf16 (matching vLLM's bf16 tower/embed store).
void RoundToBf16(std::vector<float>& v) {
  for (float& x : v) x = vt::BF16ToF32(vt::F32ToBF16(x));
}

// ---- vision weight loader: model.visual.* bf16 -> host f32 (matches M2a dump) ----
// `t.data` points into the safetensors mmap, whose payload offset carries NO
// alignment guarantee (issue #772), so the bytes are read through
// vt::LoadUnaligned rather than handed to Bf16BitsToF32 as a `const uint16_t*`.
// Bf16BitsToF32 keeps that signature for its OTHER callers, which pass
// std::vector<uint16_t>::data() and are suitably aligned by construction.
std::vector<float> LoadVisionF32(const TensorResolver& get,
                                 const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3-vl vision: expected BF16 for " + name);
  const auto n = static_cast<size_t>(t.nbytes / sizeof(uint16_t));
  const auto* src = static_cast<const unsigned char*>(static_cast<const void*>(t.data));
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(src + i * 2));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return out;
}

multimodal::VisionMergerWeights LoadMerger(const TensorResolver& get,
                                           const std::string& prefix, bool postshuffle) {
  multimodal::VisionMergerWeights m;
  m.use_postshuffle_norm = postshuffle;
  m.norm_w = LoadVisionF32(get, prefix + ".norm.weight");
  m.norm_b = LoadVisionF32(get, prefix + ".norm.bias");
  m.fc1_w = LoadVisionF32(get, prefix + ".linear_fc1.weight");
  m.fc1_b = LoadVisionF32(get, prefix + ".linear_fc1.bias");
  m.fc2_w = LoadVisionF32(get, prefix + ".linear_fc2.weight");
  m.fc2_b = LoadVisionF32(get, prefix + ".linear_fc2.bias");
  return m;
}

// ---- text weight loader: model.language_model.* bf16 (dense, tied, no fp4) ----
Qwen3DenseWeights LoadTextBackbone(const TensorResolver& get,
                                   const HfConfig& config) {
  using dense_loaders::LoadBf16Direct;
  using dense_loaders::LoadMergedBf16RawNK;
  const std::string P = "model.language_model.";
  Qwen3DenseWeights w;
  w.tie_word_embeddings = true;   // Qwen3-VL-4B text_config.tie_word_embeddings
  w.attention_bias = false;
  w.embed_tokens = LoadBf16Direct(get, P + "embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, P + "norm.weight");
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string base = P + "layers." + std::to_string(l) + ".";
    const std::string sa = base + "self_attn.";
    const std::string mlp = base + "mlp.";
    Qwen3DenseLayerWeights lw;
    lw.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
    lw.post_attention_layernorm =
        LoadBf16Direct(get, base + "post_attention_layernorm.weight");
    lw.attn.qkv_proj = LoadMergedBf16RawNK(
        get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
    lw.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
    lw.attn.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
    lw.attn.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");
    lw.mlp.gate_up_proj = LoadMergedBf16RawNK(
        get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
    lw.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// MRoPE RopeArgs for the Qwen3-VL text backbone (full 128-dim, 3-section
// interleaved, theta 5e6). Independent of HfConfig rope parsing so the fork does
// not depend on a partial_rotary_factor default.
vt::RopeArgs MropeArgs(const HfConfig& config) {
  vt::RopeArgs a;
  a.base = static_cast<float>(config.rope_theta);
  a.rotary_dim = static_cast<int>(config.head_dim);  // 128, full rotary
  a.is_neox_style = true;
  a.mrope_section = {24, 20, 20};
  a.mrope_interleaved = true;
  return a;
}

// The forked VL self-attention block — the dense bf16 default path (3-shard qkv,
// bf16 per-head q/k RMSNorm) with the RoPE step replaced by 3-section MRoPE off a
// GLOBAL absolute-position cos|sin cache and positions[3,T].
DBuf VLAttnBlock(Dev d, const Qwen3DenseAttnWeights& w, const HfConfig& cfg,
                 const Tensor& dhn, const Tensor& positions3, const Tensor& cos_sin_cache,
                 const Tensor& slot_mapping, const Tensor& block_table,
                 const Tensor& seq_lens, const Tensor& query_start_loc,
                 const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                 const vt::RopeArgs& rope, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const DType adt = DType::kBF16;

  // Merged qkv owner, 3-shard GEMMs (the dense default, byte-identical path).
  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kdim});
  DBuf v(d, adt, {T, kdim});
  {
    Tensor wqkv = ResidentWeight(d, w.qkv_proj);
    Tensor wq = wqkv.Slice(0, 0, qdim);
    Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
    Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
    vt::MatmulBT(d.q, q.t(), dhn, wq);
    vt::MatmulBT(d.q, k.t(), dhn, wk);
    vt::MatmulBT(d.q, v.t(), dhn, wv);
  }

  // Per-head q/k RMSNorm (bf16) then 3-section MRoPE off the global cache.
  Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
  Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  {
    Tensor wqn = ResidentWeight(d, w.q_norm, {Dh});
    Tensor wkn = ResidentWeight(d, w.k_norm, {Dh});
    vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
  }
  Tensor k3v = k3;
  vt::RopeFromCache(d.q, q3, &k3v, positions3, cos_sin_cache, rope);

  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});

  // Write K/V into the paged cache (bf16 == cache dtype), then causal paged attn.
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, k3, v3, k_cache, v_cache, slot_mapping);

  DBuf attn(d, adt, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  vt::PagedAttentionArgs pa{scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, block_table, seq_lens,
                     query_start_loc, pa);

  // o_proj.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// One decoder layer: input norm -> attn -> post norm -> MLP (dense layout).
void VLRunLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
                DBuf& hidden, DBuf& res, const Tensor& positions3,
                const Tensor& cos_sin_cache, const Tensor& slot_mapping,
                const Tensor& block_table, const Tensor& seq_lens,
                const Tensor& query_start_loc, const CommonAttentionMetadata& meta,
                const PagedKvCache& kv, const vt::RopeArgs& rope, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

  DBuf attn = VLAttnBlock(d, layer.attn, cfg, dhn.t(), positions3, cos_sin_cache,
                          slot_mapping, block_table, seq_lens, query_start_loc, meta, kv,
                          rope, T);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

  // Dense SwiGLU MLP via the LinearMethod seam (bf16 == UnquantizedLinearMethod),
  // byte-identical to the landed dense MlpBlock.
  auto gate_up = layers::MakeMlpGateUpMethod(layer.mlp.gate_up_proj,
                                             layer.mlp.gate_proj_fp4,
                                             layer.mlp.up_proj_fp4, I);
  DBuf act = gate_up->Apply(d, dh2.t());
  auto down = layers::MakeLinearMethod(layer.mlp.down_proj, layer.mlp.down_proj_fp4);
  hidden = down->Apply(d, act.t(), DType::kBF16);
}

// Move a [rows, vocab] f32 DEVICE DBuf into a pool-returning ForwardLogits carrier
// so the runner / greedy loop samples straight off device (the sampler's argmax /
// temperature / top-k/p kernels run on-device — no per-step full-vocab D2H). The
// pool block's lifetime moves into a shared_ptr whose deleter returns it to the
// DevicePool (no per-step cudaMalloc/cudaFree). Byte-for-byte the qwen3.cpp /
// qwen3_moe.cpp WrapDeviceLogits idiom (RUNNER-ROUTE: mirror the text device path).
ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

// Full forked forward for ONE step. `inputs_embeds_bf16` [T*H] host bf16 bits are
// the already-merged embeddings. `deepstack_bf16` (may be empty) is the [L*T*H]
// host bf16 DeepStack tensor injected after layers 0/1/2. Returns the LAST row's
// logits as an ON-DEVICE [1, vocab] f32 DBuf (NO host Download) — the device-
// resident forward core shared by the host + on-device logits wrappers.
DBuf VLForwardLastLogitsDBuf(
    Dev d, const std::vector<uint16_t>& inputs_embeds_bf16,
    const std::vector<int32_t>& positions3_host, int64_t T,
    const std::vector<uint16_t>& deepstack_bf16, int64_t L,
    const Tensor& cos_sin_cache, const CommonAttentionMetadata& meta,
    const std::vector<PagedKvCache>& attn_kv, const Qwen3DenseWeights& weights,
    const HfConfig& config, const vt::RopeArgs& rope) {
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);

  DBuf hidden(d, DType::kBF16, {T, H}, inputs_embeds_bf16.data());
  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  // Per-step device inputs.
  DBuf positions3(d, DType::kI32, {3, T}, positions3_host.data());
  DBuf slot_mapping(d, DType::kI64, {T}, meta.slot_mapping.data());
  DBuf block_table(d, DType::kI32, {meta.num_reqs, meta.block_table_num_cols},
                   meta.block_table_tensor.data());
  DBuf seq_lens(d, DType::kI32, {meta.num_reqs}, meta.seq_lens.data());
  DBuf query_start_loc(d, DType::kI32, {meta.num_reqs + 1}, meta.query_start_loc.data());

  // DeepStack device tensor [L,T,H] bf16 (empty on decode steps).
  const bool has_ds = !deepstack_bf16.empty();
  DBuf ds(d, DType::kBF16, has_ds ? std::vector<int64_t>{L, T, H}
                                  : std::vector<int64_t>{1, 1},
          has_ds ? deepstack_bf16.data() : nullptr);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    VLRunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
               positions3.t(), cos_sin_cache, slot_mapping.t(), block_table.t(),
               seq_lens.t(), query_start_loc.t(), meta, attn_kv[static_cast<size_t>(l)],
               rope, T);
    // DeepStack: add the l-th multiscale merger output after layers 0..L-1
    // (qwen3_vl.py:1589-1594). hidden is the post-layer delta stream.
    if (has_ds && l < L) {
      Tensor ds_l = MakeTensor(static_cast<char*>(ds.ptr()) +
                                   static_cast<size_t>(l * T * H) * vt::SizeOf(DType::kBF16),
                               DType::kBF16, d.q.device, {T, H});
      vt::Add(d.q, hidden.t(), hidden.t(), ds_l);
    }
  }

  // Final RMSNorm over the fused stream.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  // Gather the LAST row, then tied lm_head (== embed_tokens^T via MatmulBT).
  DBuf last(d, DType::kBF16, {1, H});
  d.b.Copy(d.q, last.ptr(),
           static_cast<char*>(dnorm.ptr()) + static_cast<size_t>((T - 1) * H) * vt::SizeOf(DType::kBF16),
           static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));
  Tensor lm = ResidentWeight(d, weights.embed_tokens, {vocab, H});
  DBuf logits(d, DType::kF32, {1, vocab});
  vt::MatmulBT(d.q, logits.t(), last.t(), lm);
  return logits;
}

// Host wrapper over the device core (VT_LOGITS_GATHER=0-style opt-out + the
// exposed host public API): run the forward, then Download the [vocab] f32 row.
std::vector<float> VLForwardLastLogits(
    Dev d, const std::vector<uint16_t>& inputs_embeds_bf16,
    const std::vector<int32_t>& positions3_host, int64_t T,
    const std::vector<uint16_t>& deepstack_bf16, int64_t L,
    const Tensor& cos_sin_cache, const CommonAttentionMetadata& meta,
    const std::vector<PagedKvCache>& attn_kv, const Qwen3DenseWeights& weights,
    const HfConfig& config, const vt::RopeArgs& rope) {
  DBuf logits = VLForwardLastLogitsDBuf(d, inputs_embeds_bf16, positions3_host, T,
                                        deepstack_bf16, L, cos_sin_cache, meta,
                                        attn_kv, weights, config, rope);
  std::vector<float> out(static_cast<size_t>(config.vocab_size));
  logits.Download(d, out.data());
  return out;
}

// Build a single-sequence CommonAttentionMetadata for a step.
CommonAttentionMetadata StepMeta(int64_t T, int64_t seq_len, int64_t first_slot) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(seq_len)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(seq_len);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t)
    m.slot_mapping.push_back(static_cast<int64_t>(first_slot + t));
  m.causal = true;
  return m;
}

int64_t ArgMax(const std::vector<float>& logits) {
  int64_t am = 0;
  for (int64_t v = 1; v < static_cast<int64_t>(logits.size()); ++v)
    if (logits[static_cast<size_t>(v)] > logits[static_cast<size_t>(am)]) am = v;
  return am;
}

// Greedy token from a step's ForwardLogits. On the DEVICE path (default) reduce
// the [1, vocab] f32 logits with vt::GreedyArgmax (LOWEST-index tie-break ==
// torch.argmax, the exact winner the old host scan produced) and Download only the
// winning int64 — instead of the full-vocab (~600 KiB) f32 D2H + a host scan.
// This is the on-GPU greedy sampler the text runner uses (runner.cpp:1185-1186 ->
// sampler.cpp GreedyArgmax). Host path (opt-out) falls back to the host scan.
int32_t VLArgMaxFromForward(Dev d, const ForwardLogits& fl) {
  if (fl.on_device()) {
    DBuf ids(d, DType::kI64, {1});
    vt::GreedyArgmax(d.q, ids.t(), fl.device_tensor);
    int64_t id = 0;
    ids.Download(d, &id);
    return static_cast<int32_t>(id);
  }
  return static_cast<int32_t>(ArgMax(fl.host));
}

}  // namespace

Qwen3VLWeights LoadQwen3VLWeights(const std::vector<SafetensorsFile>& shards,
                                  const HfConfig& config,
                                  const MultiModalConfig* mm_config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3-vl: tensor not found: " + name);
    return it->second->Get(name);
  };

  Qwen3VLWeights w;
  w.text = LoadTextBackbone(get, config);
  // Vision tower (model.visual.*), bf16 -> f32; w.vision_cfg holds the Qwen3-VL-4B
  // defaults. Shared with the 27B path via LoadQwen3VLVisionWeights.
  //
  // #607 L3, the TOWER SKIP. `w.vision_cfg` above is the constructed-but-
  // uninitialised half: the geometry is resolved either way, only the storage is
  // conditional, which is what `no_init_weights` over `torch.device("meta")`
  // does upstream (interfaces.py:288-293, utils.py:762). The modality set is
  // `{"image", "video"}` because that is exactly how upstream marks this tower
  // (qwen3_vl.py:1747, and qwen3_5.py:422,634 for the 27B/35B wrappers that
  // compose the SAME Qwen3_VisionTransformer), so `image: 0` alone keeps it.
  if (SkipTowerForModalities(mm_config, {"image", "video"})) {
    w.vision_skipped = true;
  } else {
    w.vision = LoadQwen3VLVisionWeights(shards, w.vision_cfg);
    w.vision_loaded = true;
  }
  return w;
}

multimodal::Qwen3VLVisionWeights LoadQwen3VLVisionWeights(
    const std::vector<SafetensorsFile>& shards,
    const multimodal::Qwen3VLVisionConfig& vc) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3-vl vision: tensor not found: " + name);
    return it->second->Get(name);
  };

  multimodal::Qwen3VLVisionWeights vw;
  const std::string V = "model.visual.";
  vw.patch_proj_w = LoadVisionF32(get, V + "patch_embed.proj.weight");
  vw.patch_proj_b = LoadVisionF32(get, V + "patch_embed.proj.bias");
  vw.pos_embed_w = LoadVisionF32(get, V + "pos_embed.weight");
  vw.blocks.resize(static_cast<size_t>(vc.depth));
  for (int64_t l = 0; l < vc.depth; ++l) {
    const std::string p = V + "blocks." + std::to_string(l);
    multimodal::VisionBlockWeights& b = vw.blocks[static_cast<size_t>(l)];
    b.norm1_w = LoadVisionF32(get, p + ".norm1.weight");
    b.norm1_b = LoadVisionF32(get, p + ".norm1.bias");
    b.norm2_w = LoadVisionF32(get, p + ".norm2.weight");
    b.norm2_b = LoadVisionF32(get, p + ".norm2.bias");
    b.qkv_w = LoadVisionF32(get, p + ".attn.qkv.weight");
    b.qkv_b = LoadVisionF32(get, p + ".attn.qkv.bias");
    b.proj_w = LoadVisionF32(get, p + ".attn.proj.weight");
    b.proj_b = LoadVisionF32(get, p + ".attn.proj.bias");
    b.fc1_w = LoadVisionF32(get, p + ".mlp.linear_fc1.weight");
    b.fc1_b = LoadVisionF32(get, p + ".mlp.linear_fc1.bias");
    b.fc2_w = LoadVisionF32(get, p + ".mlp.linear_fc2.weight");
    b.fc2_b = LoadVisionF32(get, p + ".mlp.linear_fc2.bias");
  }
  vw.merger = LoadMerger(get, V + "merger", /*postshuffle=*/false);
  // Qwen3.6-27B has EMPTY deepstack_visual_indexes ⇒ this loop runs zero times
  // (no deepstack_merger_list.* tensors exist in the checkpoint).
  for (size_t i = 0; i < vc.deepstack_visual_indexes.size(); ++i)
    vw.deepstack_mergers.push_back(
        LoadMerger(get, V + "deepstack_merger_list." + std::to_string(i),
                   /*postshuffle=*/true));
  return vw;
}

namespace {

// Common forked-forward core (shared by the image + video gate drivers): given the
// precomputed visual mask + MRoPE prefill positions [3,T0] + decode delta, run
// embed(prompt_ids) + masked-scatter merge + DeepStack inject prefill, then paged
// greedy decode. `weights_text` is the dense text backbone; `L` = deepstack levels.
std::vector<int32_t> VLGenerateCore(
    Dev d, const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::vector<float>& mm_deepstack, int64_t L, const std::vector<bool>& mask,
    const std::vector<int32_t>& pos_prefill, int64_t delta, int32_t eos_token_id,
    const Qwen3DenseWeights& weights_text, const HfConfig& config,
    int max_new_tokens, const VLStepFn& step) {
  Backend& backend = d.b;
  const int64_t H = config.hidden_size;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());
  const int64_t N = static_cast<int64_t>(mm_main.size()) / (H > 0 ? H : 1);

  // Global absolute-position cos|sin cache [Pmax, rotary_dim] bf16 — the ONE
  // build path shared with the registered forward (Qwen3VLMakeCosSinCache).
  const Qwen3VLCosSinCache cos_sin = Qwen3VLMakeCosSinCache(d.q, config);

  // KV caches: one big block per layer sized for T0 + max_new_tokens.
  const int64_t block_size = T0 + max_new_tokens + 8;
  const size_t kv_bytes =
      static_cast<size_t>(1 * 2 * block_size * Hkv * Dh) * vt::SizeOf(DType::kBF16);
  std::vector<std::shared_ptr<void>> kv_storage;
  std::vector<PagedKvCache> attn_kv;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    void* p = backend.Alloc(kv_bytes);
    backend.Memset(d.q, p, 0, kv_bytes);
    kv_storage.emplace_back(p, [&backend](void* q) { backend.Free(q); });
    PagedKvCache kv;
    kv.data = p;
    kv.dtype = DType::kBF16;
    kv.num_blocks = 1;
    kv.block_size = block_size;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
    attn_kv.push_back(kv);
  }

  // ---- PREFILL: embed + merge + deepstack ----
  // Embed prompt ids, download, merge mm_main (bf16-rounded) into image rows.
  std::vector<uint16_t> emb_bits(static_cast<size_t>(T0 * H));
  {
    DBuf ids(d, DType::kI32, {T0}, prompt_ids.data());
    DBuf emb(d, DType::kBF16, {T0, H});
    Tensor tab = ResidentWeight(d, weights_text.embed_tokens, {config.vocab_size, H});
    vt::Embedding(d.q, emb.t(), tab, ids.t());
    emb.Download(d, emb_bits.data());
  }
  std::vector<float> embeds = Bf16BitsToF32(emb_bits.data(), T0 * H);
  std::vector<float> main_bf = mm_main;
  RoundToBf16(main_bf);
  multimodal::Qwen3VLMergeMultimodal(embeds, T0, H, main_bf, mask);
  std::vector<uint16_t> merged_bits = F32ToBf16Bits(embeds.data(), T0 * H);

  // DeepStack [L,T0,H] host bf16.
  std::vector<float> ds_f32 =
      multimodal::Qwen3VLComputeDeepstack(mm_deepstack, N, L, H, mask, T0);
  RoundToBf16(ds_f32);
  std::vector<uint16_t> ds_bits = F32ToBf16Bits(ds_f32.data(), L * T0 * H);

  const CommonAttentionMetadata pm = StepMeta(T0, T0, 0);
  ForwardLogits logits =
      step(merged_bits, pos_prefill, T0, ds_bits, pm, attn_kv, cos_sin.tensor);

  std::vector<int32_t> generated;
  int32_t next = VLArgMaxFromForward(d, logits);
  generated.push_back(next);

  // ---- DECODE: single token per step, MRoPE decode position, no deepstack ----
  // The greedy token is picked on-device from the step's device logits carrier
  // (VLArgMaxFromForward -> vt::GreedyArgmax), so the only per-step D2H is the
  // winning int64 — the full-vocab logit round-trip is GONE. The one remaining
  // host round-trip is the per-token embed Download below (the mm-forward contract
  // still carries HOST merged embeds via MultiModalForwardInput; a device-embeds
  // seam is the scoped RUNNER-ROUTE residual).
  const std::vector<uint16_t> no_ds;
  for (int dstep = 1; dstep < max_new_tokens; ++dstep) {
    if (next == eos_token_id) break;
    const int64_t abs_idx = T0 + (dstep - 1);  // sequence index of the token being fed
    const int32_t p = static_cast<int32_t>(abs_idx + delta);
    const std::vector<int32_t> pos1 = {p, p, p};  // [3,1] all axes equal (text decode)

    std::vector<uint16_t> tok_emb(static_cast<size_t>(H));
    {
      const std::vector<int32_t> one = {next};
      DBuf ids(d, DType::kI32, {1}, one.data());
      DBuf emb(d, DType::kBF16, {1, H});
      Tensor tab = ResidentWeight(d, weights_text.embed_tokens, {config.vocab_size, H});
      vt::Embedding(d.q, emb.t(), tab, ids.t());
      emb.Download(d, tok_emb.data());
    }
    const int64_t seq_len = abs_idx + 1;
    const CommonAttentionMetadata dm = StepMeta(1, seq_len, abs_idx);
    logits = step(tok_emb, pos1, 1, no_ds, dm, attn_kv, cos_sin.tensor);
    next = VLArgMaxFromForward(d, logits);
    generated.push_back(next);
  }
  return generated;
}

// The PRODUCTION step: pack the already-merged embeddings / 3-D MRoPE positions /
// DeepStack into ModelForwardInput.mm and drive ModelRegistry::Forward, which
// dispatches to the registered ForwardQwen3VL → the SHARED
// Qwen3VLForwardStepLastLogitsDevice. `gather_logits=true` requests ON-DEVICE
// logits (the sampler-on-device carrier), consumed by VLArgMaxFromForward. This is
// the ONE step every driver (image / video / registry) runs — so the Qwen3-VL
// decode enters through ModelRegistry::Forward exactly like the text models, no
// off-framework standalone forward (RUNNER-ROUTE).
VLStepFn MakeRegistryStep(LoadedModel& model, const HfConfig& config,
                          vt::Queue& queue, int64_t deepstack_levels) {
  return [&model, &config, &queue, deepstack_levels](
             const std::vector<uint16_t>& embeds, const std::vector<int32_t>& pos3,
             int64_t T, const std::vector<uint16_t>& ds,
             const CommonAttentionMetadata& meta, std::vector<PagedKvCache>& attn_kv,
             const Tensor& /*cos_sin (the model owns an identical cache)*/)
             -> ForwardLogits {
    const std::vector<int32_t> no_tokens;
    const std::vector<int32_t> no_pos;
    std::vector<GdnStateCache> no_gdn_state;
    v1::GDNAttentionMetadata gdn_meta{};
    const std::vector<int32_t> gather_li = {static_cast<int32_t>(T - 1)};
    const MultiModalForwardInput mm{&embeds, &pos3, &ds, deepstack_levels};
    ModelForwardInput in{
        .token_ids = no_tokens,
        .positions = no_pos,
        .attn_meta = meta,
        .gdn_meta = gdn_meta,
        .attn_kv = attn_kv,
        .gdn_state = no_gdn_state,
        .config = config,
        .queue = queue,
        .logits_indices = gather_li,
        .num_reqs = meta.num_reqs,
        .pure_decode = false,
        .gather_logits = true,
        .mm = mm,
    };
    return ModelRegistry::Forward(model, in);
  };
}

}  // namespace

// ── MM-ENGINE-FORWARD shared building blocks (reused by the registered forward) ──

Qwen3VLCosSinCache Qwen3VLMakeCosSinCache(vt::Queue& queue, const HfConfig& config) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const vt::RopeArgs rope = MropeArgs(config);
  const int64_t Pmax = 8192;
  const int rot = rope.rotary_dim;
  std::vector<int32_t> arange(static_cast<size_t>(Pmax));
  for (int64_t i = 0; i < Pmax; ++i) arange[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  DBuf ar(d, DType::kI32, {Pmax}, arange.data());
  DBuf cache_f32(d, DType::kF32, {Pmax, rot});
  vt::RopeCosSinCache(d.q, cache_f32.t(), ar.t(), rope);
  // Persistent bf16 storage (owned by the returned cache; freed on last ref).
  const size_t bytes = static_cast<size_t>(Pmax * rot) * vt::SizeOf(DType::kBF16);
  void* p = backend.Alloc(bytes);
  std::shared_ptr<void> storage(p, [&backend](void* q) { backend.Free(q); });
  Tensor tensor = MakeTensor(p, DType::kBF16, d.q.device, {Pmax, rot});
  vt::CastBf16(d.q, tensor, cache_f32.t());
  return Qwen3VLCosSinCache{std::move(storage), tensor};
}

std::vector<float> Qwen3VLForwardStepLastLogits(
    vt::Queue& queue, const Qwen3DenseWeights& weights_text, const HfConfig& config,
    const std::vector<uint16_t>& inputs_embeds_bf16,
    const std::vector<int32_t>& positions3, int64_t num_tokens,
    const std::vector<uint16_t>& deepstack_bf16, int64_t deepstack_levels,
    const vt::Tensor& cos_sin_cache_bf16, const v1::CommonAttentionMetadata& meta,
    const std::vector<PagedKvCache>& attn_kv) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const vt::RopeArgs rope = MropeArgs(config);
  return VLForwardLastLogits(d, inputs_embeds_bf16, positions3, num_tokens,
                             deepstack_bf16, deepstack_levels, cos_sin_cache_bf16,
                             meta, attn_kv, weights_text, config, rope);
}

ForwardLogits Qwen3VLForwardStepLastLogitsDevice(
    vt::Queue& queue, const Qwen3DenseWeights& weights_text, const HfConfig& config,
    const std::vector<uint16_t>& inputs_embeds_bf16,
    const std::vector<int32_t>& positions3, int64_t num_tokens,
    const std::vector<uint16_t>& deepstack_bf16, int64_t deepstack_levels,
    const vt::Tensor& cos_sin_cache_bf16, const v1::CommonAttentionMetadata& meta,
    const std::vector<PagedKvCache>& attn_kv) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const vt::RopeArgs rope = MropeArgs(config);
  DBuf logits = VLForwardLastLogitsDBuf(
      d, inputs_embeds_bf16, positions3, num_tokens, deepstack_bf16,
      deepstack_levels, cos_sin_cache_bf16, meta, attn_kv, weights_text, config,
      rope);
  return WrapDeviceLogits(std::move(logits), /*rows=*/1, config.vocab_size);
}

std::vector<int32_t> Qwen3VLGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::vector<float>& mm_deepstack, int64_t num_deepstack_levels,
    const std::array<int64_t, 3>& grid_thw, int32_t image_token_id,
    int32_t eos_token_id, const Qwen3VLWeights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const int64_t H = config.hidden_size;
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());

  // Image span offset + visual mask.
  int64_t offset = -1;
  int64_t n_img = 0;
  std::vector<bool> mask(static_cast<size_t>(T0), false);
  for (int64_t t = 0; t < T0; ++t) {
    if (prompt_ids[static_cast<size_t>(t)] == image_token_id) {
      if (offset < 0) offset = t;
      mask[static_cast<size_t>(t)] = true;
      ++n_img;
    }
  }
  VT_CHECK(offset >= 0, "qwen3-vl: no image token in prompt");
  const int64_t N = static_cast<int64_t>(mm_main.size()) / (H > 0 ? H : 1);
  VT_CHECK(N == n_img, "qwen3-vl: mm_main rows != image-token count");

  // MRoPE positions [3,T0] + decode delta (spatial_merge_size == 2 for Qwen3-VL).
  std::vector<multimodal::MmImageSpan> spans = {{offset, grid_thw}};
  int64_t delta = 0;
  std::vector<int32_t> pos_prefill =
      multimodal::Qwen3VLGetRopeIndex(prompt_ids, spans, /*spatial_merge_size=*/2, &delta);

  // RUNNER-ROUTE: every decode step enters through ModelRegistry::Forward on a
  // borrowed (non-owning) registered Qwen3-VL LoadedModel — no standalone forward.
  std::unique_ptr<LoadedModel> model = BorrowQwen3VLLoadedModel(weights);
  const VLStepFn step =
      MakeRegistryStep(*model, config, queue, num_deepstack_levels);
  return VLGenerateCore(d, prompt_ids, mm_main, mm_deepstack, num_deepstack_levels,
                        mask, pos_prefill, delta, eos_token_id, weights.text, config,
                        max_new_tokens, step);
}

std::vector<int32_t> Qwen3VLGenerateGreedyVideo(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::vector<float>& mm_deepstack, int64_t num_deepstack_levels,
    const std::array<int64_t, 3>& grid_thw, int32_t video_token_id,
    int32_t vision_start_token_id, int32_t vision_end_token_id,
    int32_t eos_token_id, const Qwen3VLWeights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const int64_t H = config.hidden_size;
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());

  // Video merge mask (all video_token_id positions across all frames).
  int64_t n_vid = 0;
  std::vector<bool> mask(static_cast<size_t>(T0), false);
  for (int64_t t = 0; t < T0; ++t) {
    if (prompt_ids[static_cast<size_t>(t)] == video_token_id) {
      mask[static_cast<size_t>(t)] = true;
      ++n_vid;
    }
  }
  VT_CHECK(n_vid > 0, "qwen3-vl: no video token in prompt");
  const int64_t N = static_cast<int64_t>(mm_main.size()) / (H > 0 ? H : 1);
  VT_CHECK(N == n_vid, "qwen3-vl: mm_main rows != video-token count");

  // MRoPE positions [3,T0] + decode delta via the per-frame video scan.
  int64_t delta = 0;
  std::vector<int32_t> pos_prefill = multimodal::Qwen3VLGetRopeIndexVideo(
      prompt_ids, grid_thw, /*spatial_merge_size=*/2, vision_start_token_id,
      video_token_id, vision_end_token_id, &delta);

  // RUNNER-ROUTE: drive every decode step through ModelRegistry::Forward (see
  // Qwen3VLGenerateGreedy) — the image and video drivers share the same
  // registered mm-forward path, differing only in mask / MRoPE construction.
  std::unique_ptr<LoadedModel> model = BorrowQwen3VLLoadedModel(weights);
  const VLStepFn step =
      MakeRegistryStep(*model, config, queue, num_deepstack_levels);
  return VLGenerateCore(d, prompt_ids, mm_main, mm_deepstack, num_deepstack_levels,
                        mask, pos_prefill, delta, eos_token_id, weights.text, config,
                        max_new_tokens, step);
}

std::vector<int32_t> Qwen3VLGenerateGreedyViaRegistry(
    LoadedModel& model, const std::vector<int32_t>& prompt_ids,
    const std::vector<float>& mm_main, const std::vector<float>& mm_deepstack,
    int64_t num_deepstack_levels, const std::array<int64_t, 3>& grid_thw,
    int32_t image_token_id, int32_t eos_token_id, const Qwen3VLWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const int64_t H = config.hidden_size;
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());

  // Image span offset + visual mask (identical to Qwen3VLGenerateGreedy).
  int64_t offset = -1;
  int64_t n_img = 0;
  std::vector<bool> mask(static_cast<size_t>(T0), false);
  for (int64_t t = 0; t < T0; ++t) {
    if (prompt_ids[static_cast<size_t>(t)] == image_token_id) {
      if (offset < 0) offset = t;
      mask[static_cast<size_t>(t)] = true;
      ++n_img;
    }
  }
  VT_CHECK(offset >= 0, "qwen3-vl: no image token in prompt");
  const int64_t N = static_cast<int64_t>(mm_main.size()) / (H > 0 ? H : 1);
  VT_CHECK(N == n_img, "qwen3-vl: mm_main rows != image-token count");

  std::vector<multimodal::MmImageSpan> spans = {{offset, grid_thw}};
  int64_t delta = 0;
  std::vector<int32_t> pos_prefill =
      multimodal::Qwen3VLGetRopeIndex(prompt_ids, spans, /*spatial_merge_size=*/2, &delta);

  // The REGISTERED step (shared by all three drivers): drive ModelRegistry::Forward
  // with ModelForwardInput.mm carrying the merged embeddings / 3-D MRoPE positions /
  // DeepStack, which dispatches to the registered ForwardQwen3VL → the SHARED
  // Qwen3VLForwardStepLastLogitsDevice. Returns ON-DEVICE logits (the sampler-on-
  // device carrier), consumed by VLGenerateCore's on-GPU VLArgMaxFromForward.
  const VLStepFn step =
      MakeRegistryStep(model, config, queue, num_deepstack_levels);
  return VLGenerateCore(d, prompt_ids, mm_main, mm_deepstack, num_deepstack_levels,
                        mask, pos_prefill, delta, eos_token_id, weights.text, config,
                        max_new_tokens, step);
}

}  // namespace vllm
