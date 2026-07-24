// Gemma-1 text (`GemmaForCausalLM`) dense forward — sweep W5. The original Gemma:
// the SIMPLEST Gemma-family bring-up, a plain Llama-style pre-norm decoder (two
// fused add+RMSNorm per layer) with the Gemma vocabulary (GemmaRMSNorm (1+w),
// GeGLU, sqrt(hidden) embed-scale, tied lm_head), reusing the W1 GeGLU/embed-scale
// primitives with NO soft-cap, NO QK-norm, NO sliding window.
//
// Grounding: vllm/model_executor/models/gemma.py @ e24d1b24. The Gemma-1
// vocabulary vs the Gemma-2 path (gemma2.cpp): the GeGLU MLP, sqrt(hidden)
// embed-scale and tied lm_head are IDENTICAL; the delta is
//   - only TWO norms per layer, BOTH fused add+RMSNorm (no sandwich standalones);
//   - `head_dim**-0.5` attention scaling (not query_pre_attn_scalar);
//   - NO attention/final logit soft-cap, NO sliding window, single RoPE theta.
//
// Numeric contract: bf16 per-op, matching vLLM's stores (dense_attn_block.h).
#include "vllm/model_executor/models/gemma.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

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

// One Gemma-1 self-attention block (gemma.py::GemmaAttention.forward). `dhn` is
// the input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16.
DBuf GemmaAttnBlock(Dev d, const GemmaAttnWeights& w, const HfConfig& cfg,
                    const Tensor& dhn, const StepInputs& si,
                    const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                    int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const DType adt = DType::kBF16;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "gemma: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "gemma: KV cache head dims mismatch config");

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

  // NeoX RoPE (single theta, full rotary_dim = head_dim). NO q/k norm.
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  vt::RopeArgs ra;
  ra.base = static_cast<float>(cfg.rope_theta);
  ra.rotary_dim = static_cast<int>(Dh);
  vt::RopeNeox(d.q, q3, k3, si.positions.t(), ra);

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

  // Paged GQA attention: scale = head_dim**-0.5, no soft-cap, no sliding window.
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(Dh)));
  DBuf attn(d, adt, {T, Hq, Dh});
  vt::PagedAttentionArgs pa{scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

DBuf GemmaMlpBlock(Dev d, const GemmaMlpWeights& w, const HfConfig& cfg,
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

// One Gemma-1 decoder layer (gemma.py::GemmaDecoderLayer.forward), two fused
// add+RMSNorm (GemmaRMSNorm 1+w):
//   res += hidden;   dhn = gemmaNorm(res)      # input_layernorm      (fused)
//   attn = Attn(dhn)
//   res += attn;     dh2 = gemmaNorm(res)       # post_attention_ln   (fused)
//   hidden = Mlp(dh2)                            # returned unnormed; res carried
void RunLayer(Dev d, const GemmaLayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, DBuf& res, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const vt::RmsNormArgs gemma{eps, true};

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, gemma, &res.t());

  DBuf attn = GemmaAttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T);

  Tensor w_pa = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), w_pa, gemma, &res.t());

  hidden = GemmaMlpBlock(d, layer.mlp, cfg, dh2.t(), T);
}

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
                 const GemmaWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "gemma: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "gemma: one PagedKvCache per layer required");

  // Embed then scale by sqrt(hidden) cast to bf16 (gemma.py:288-295).
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
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T);

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, true}, &res.t());

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

std::vector<float> GemmaModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const GemmaWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits GemmaModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const GemmaWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
