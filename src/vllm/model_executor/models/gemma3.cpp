// Gemma-3 text (`Gemma3ForCausalLM`) dense forward — sweep W2, the first
// Gemma-family token-exact gate vehicle. Composed from the public vt:: ops and
// the shared dense-attention device glue (dense_attn_block.h: Dev/DBuf/
// ResidentWeight/KvSlice/StepInputs), with a Gemma-specific attention block and
// sandwich-norm decoder layer.
//
// Grounding: vllm/model_executor/models/gemma3.py @ e24d1b24. The Gemma
// vocabulary vs the Qwen3-dense path (qwen3.cpp):
//   - GemmaRMSNorm (1+w, fp32) at EVERY norm — RmsNormArgs{eps, gemma=true};
//   - the Gemma-2/3 SANDWICH layout (GLM-4 pattern, glm4.cpp): input +
//     pre_feedforward are fused add+RMSNorm; post_attention + post_feedforward
//     are standalone norms on the sublayer output before it re-enters residual;
//   - per-head Gemma q/k RMSNorm (gemma=true) before RoPE;
//   - GeGLU MLP (vt::GeluAndMul, gelu_pytorch_tanh) instead of SwiGLU;
//   - embedding scaled by sqrt(hidden) cast to bf16 (vt::MulScalar);
//   - attention scale = query_pre_attn_scalar**-0.5 (not head_dim**-0.5);
//   - DUAL per-layer RoPE theta: rope_theta on full-attn layers,
//     rope_local_base_freq on sliding layers, routed by sliding_window_pattern;
//   - per-layer sliding window (masked at the FA kernel; inert for contexts
//     < sliding_window, e.g. the short gate battery);
//   - NO attention output gate, NO attention logit soft-cap, NO qkv bias.
//
// Numeric contract: bf16 per-op, matching vLLM's stores (dense_attn_block.h).
#include "vllm/model_executor/models/gemma3.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/glue
#include "vllm/model_executor/models/device_pool.h"       // Pool
#include "vllm/model_executor/models/qwen3_5_common.h"     // HostLogits
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using namespace dense_attn;  // Dev/DBuf/ResidentWeight/KvSlice/StepInputs/Reshape

// Raw config.json scalar readers (Gemma-specific fields not yet typed on
// HfConfig; flat doc for the standalone Gemma3ForCausalLM checkpoint).
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return fallback;
  return it->get<double>();
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number_integer()) return fallback;
  return it->get<int64_t>();
}

// VT_GEMMA3_SLIDING (default ON): thread the per-layer sliding window into paged
// attention on sliding layers (faithful to vLLM). Inert for contexts <
// sliding_window (the gate battery is short), so the SACRED gate is unaffected;
// the toggle exists only for a same-binary A/B if the windowed kernel path ever
// perturbs bf16 rounding. Read once (process-stable).
bool SlidingWindowEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GEMMA3_SLIDING");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Per-layer Gemma-3 routing derived from the config.
struct Gemma3Layout {
  int64_t sliding_window_pattern;  // 1-in-N layers are full attention
  double rope_theta_global;        // full-attn layers
  double rope_theta_local;         // sliding layers (rope_local_base_freq)
  float attn_scale;                // query_pre_attn_scalar**-0.5
  int64_t sliding_window;          // window length (tokens)
  bool IsSliding(int64_t l) const {
    return ((l + 1) % sliding_window_pattern) != 0;
  }
};

Gemma3Layout MakeLayout(const HfConfig& cfg) {
  Gemma3Layout g;
  g.sliding_window_pattern = RawInt(cfg.raw, "sliding_window_pattern", 6);
  if (g.sliding_window_pattern <= 0) g.sliding_window_pattern = 1;
  g.rope_theta_global = cfg.rope_theta;
  g.rope_theta_local = RawDouble(cfg.raw, "rope_local_base_freq", 10000.0);
  const double qpas =
      RawDouble(cfg.raw, "query_pre_attn_scalar", static_cast<double>(cfg.head_dim));
  g.attn_scale = static_cast<float>(1.0 / std::sqrt(qpas));
  g.sliding_window = cfg.sliding_window.value_or(RawInt(cfg.raw, "sliding_window", 0));
  return g;
}

// One Gemma-3 self-attention block (gemma3.py::Gemma3Attention.forward). `dhn` is
// the input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16.
// `rope_base` is this layer's RoPE theta (dual: global vs local). `attn_scale` is
// query_pre_attn_scalar**-0.5. `sliding_window` (>0) masks to the last W keys.
DBuf Gemma3AttnBlock(Dev d, const Gemma3AttnWeights& w, const HfConfig& cfg,
                     const Tensor& dhn, const StepInputs& si,
                     const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                     int64_t T, double rope_base, float attn_scale,
                     std::optional<int64_t> sliding_window) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const DType adt = DType::kBF16;  // vLLM runs Gemma-3 bf16 (per-op stores)
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "gemma3: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "gemma3: KV cache head dims mismatch config");

  // Merged QKVParallelLinear (no bias): one raw-NK owner [qdim+2kdim, H], sliced
  // into q/k/v shards and projected (3-shard; the tiny GQA k/v GEMMs mirror
  // vLLM's single qkv GEMM + split numerically).
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

  // Per-head Gemma q/k RMSNorm (GemmaRMSNorm(head_dim), 1+w) BEFORE RoPE, then
  // full-dim NeoX RoPE with this layer's theta.
  Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
  Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  Tensor wqn = ResidentWeight(d, w.q_norm, {Dh});
  Tensor wkn = ResidentWeight(d, w.k_norm, {Dh});
  vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, true});  // gemma=true (1+w)
  vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, true});
  vt::RopeArgs ra;
  ra.base = static_cast<float>(rope_base);
  ra.rotary_dim = static_cast<int>(Dh);  // full rotary_dim = head_dim
  vt::RopeNeox(d.q, q3, k3, si.positions.t(), ra);

  // Write rope'd K + V into the paged cache. On the bf16 default (== cache dtype)
  // no cast; an f32 cache (CPU-synthetic A/B) down/up-casts K/V to match.
  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
  Tensor kw = k3;
  Tensor vw = v3;
  DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
  DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
  if (kv.dtype != adt) {
    if (kv.dtype == DType::kBF16) {
      vt::CastBf16(d.q, kcast.t(), k3);
      vt::CastBf16(d.q, vcast.t(), v3);
    } else {
      vt::CastF32(d.q, kcast.t(), k3);
      vt::CastF32(d.q, vcast.t(), v3);
    }
    kw = kcast.t();
    vw = vcast.t();
  }
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());

  // Paged GQA attention: scale = query_pre_attn_scalar**-0.5; sliding layers
  // mask to the last `sliding_window` keys (FA window convention (W-1, 0)).
  DBuf attn(d, adt, {T, Hq, Dh});
  vt::PagedAttentionArgs pa{attn_scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  if (sliding_window.has_value() && *sliding_window > 0 && SlidingWindowEnabled())
    pa.window_size = vt::AttentionWindow{static_cast<int32_t>(*sliding_window - 1), 0};
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  // o_proj (RowParallelLinear, no bias): [T, Hq*Dh] -> [T,H] bf16.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// Gemma-3 GeGLU MLP (gemma3.py::Gemma3MLP): merged gate_up -> GeluAndMul(tanh) ->
// down. `dh2` is the pre-FF-normed hidden [T,H] bf16.
DBuf Gemma3MlpBlock(Dev d, const Gemma3MlpWeights& w, const HfConfig& cfg,
                    const Tensor& dh2, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  Tensor wgu = ResidentWeight(d, w.gate_up_proj);
  DBuf gate_up(d, DType::kBF16, {T, 2 * I});
  vt::MatmulBT(d.q, gate_up.t(), dh2, wgu);
  DBuf act(d, DType::kBF16, {T, I});
  vt::GeluAndMul(d.q, act.t(), gate_up.t());
  Tensor wd = ResidentWeight(d, w.down_proj);
  DBuf down(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, down.t(), act.t(), wd);
  return down;
}

// One Gemma-3 decoder layer (gemma3.py::Gemma3DecoderLayer.forward), the sandwich
// pattern (GLM-4, glm4.cpp:158-188) with GemmaRMSNorm (1+w) throughout:
//   res += hidden;   dhn = gemmaNorm(res)      # input_layernorm  (fused)
//   attn = Attn(dhn)
//   attn = gemmaNorm(attn)                      # post_attention   (standalone)
//   res += attn;     dh2 = gemmaNorm(res)       # pre_feedforward  (fused)
//   mlp  = Mlp(dh2)
//   hidden = gemmaNorm(mlp)                      # post_feedforward (standalone)
void RunLayer(Dev d, const Gemma3LayerWeights& layer, const HfConfig& cfg,
              const Gemma3Layout& g, int64_t l, DBuf& hidden, DBuf& res,
              const StepInputs& si, const CommonAttentionMetadata& meta,
              const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const vt::RmsNormArgs gemma{eps, true};

  // input_layernorm (fused add+GemmaRMSNorm): res += hidden; dhn = norm(res)
  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, gemma, &res.t());

  // attention (dual-theta rope + per-layer sliding window)
  const bool sliding = g.IsSliding(l);
  const double rope_base = sliding ? g.rope_theta_local : g.rope_theta_global;
  std::optional<int64_t> window;
  if (sliding) window = g.sliding_window;
  DBuf attn = Gemma3AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T,
                              rope_base, g.attn_scale, window);

  // post_attention_layernorm (STANDALONE GemmaRMSNorm, sandwich): attn = norm(attn)
  Tensor w_pa = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf attn_n(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa, gemma);

  // pre_feedforward_layernorm (fused add+GemmaRMSNorm): res += attn_n; dh2=norm(res)
  Tensor w_pf = ResidentWeight(d, layer.pre_feedforward_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn_n.t(), w_pf, gemma, &res.t());

  // GeGLU MLP
  DBuf mlp = Gemma3MlpBlock(d, layer.mlp, cfg, dh2.t(), T);

  // post_feedforward_layernorm (STANDALONE GemmaRMSNorm, sandwich): hidden=norm(mlp)
  Tensor w_pff = ResidentWeight(d, layer.post_feedforward_layernorm, {H});
  hidden = DBuf(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, hidden.t(), mlp.t(), w_pff, gemma);
}

// GatherRows: gather the idx-indexed rows of `src` [.,H] into contiguous `dst`.
void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Gemma3Weights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "gemma3: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "gemma3: one PagedKvCache per layer required");

  const Gemma3Layout g = MakeLayout(config);

  // Embed then scale by sqrt(hidden) cast to bf16 (gemma3.py:328-341). The
  // normalizer is bf16(sqrt(H)); the multiply is f32 then rounded to bf16,
  // matching torch's bf16-scalar multiply.
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }
  const float nsqrt = std::sqrt(static_cast<float>(H));
  const double normalizer = static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(nsqrt)));
  vt::MulScalar(d.q, hidden.t(), hidden.t(), normalizer);

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, g, l, hidden, res, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final GemmaRMSNorm over the fused stream (res += hidden; gemma norm).
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, true}, &res.t());

  // lm_head. Tied (gemma-3-1b): logits = hidden @ embed_tokens^T (MatmulBT over
  // the [vocab,H] embed table). Note the embed table used here is the UNSCALED
  // one — the sqrt(H) normalizer applies only to the input embedding lookup.
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);

  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kBF16, do_gather ? std::vector<int64_t>{
                                                static_cast<int64_t>(logits_indices.size()), H}
                                          : std::vector<int64_t>{1, 1});
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);
  return logits;
}

ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  const size_t alloc = dlogits.alloc_bytes();
  void* p = dlogits.Release();
  fl.device_storage =
      std::shared_ptr<void>(p, [alloc](void* q) { Pool().Put(alloc, q); });
  (void)d;
  return fl;
}

}  // namespace

std::vector<float> Gemma3Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma3Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Gemma3Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma3Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
