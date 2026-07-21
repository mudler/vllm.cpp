// Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`) forward — the first full-attention
// MoE bring-up W3 (the capstone). It COMPOSES two already-DONE pieces: the shared
// dense self-attention block (dense_attn_block.h `AttnBlock` — merged qkv, per-head
// q/k RMSNorm, NeoX RoPE θ=1e7, d128 FA2, o_proj) and the exposed sparse-MoE block
// (qwen3_5_moe_block.h `RunMoeBlock` — router softmax+top-k+renorm, per-expert bf16
// SwiGLU MLP, weighted combine), with NO GDN and NO shared expert. Structurally it
// is the qwen3.cpp dense ForwardBody with the per-layer SwiGLU MLP replaced by the
// MoE block, plus an UNTIED lm_head.
//
// Grounding: vllm/model_executor/models/qwen3_moe.py @ e24d1b24
//   Qwen3MoeAttention (:254-354, == qwen3.py Qwen3Attention), Qwen3MoeSparseMoeBlock
//   (:130-251), Qwen3MoeDecoderLayer (:357-429), Qwen3MoeForCausalLM (:541-657).
// See .agents/specs/sweep-qwen3-coder-30b.md §4 (forward wiring), §7 W3.
//
// Numeric contract (mirrors qwen3.cpp / the qwen3_5 full-attention path): the
// residual stream is bf16 (vLLM's fused_add_rms_norm residual); the attention
// preamble follows the dense AttnBlock discipline; the MoE block runs the bf16
// reference path (per-expert host-gather — correctness for W3/W4; the bf16 fast
// grouped-MoE GEMM is W5). Router numerics: vt::MoeRouterTopK(renormalize=true) =
// softmax+top-k+renorm, matching Qwen3Moe scoring_func="softmax",
// norm_topk_prob=true. Returns [n_out, vocab] f32 logits.
#include "vllm/model_executor/models/qwen3_moe.h"

#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"    // shared AttnBlock + device glue
#include "vllm/model_executor/models/device_pool.h"         // DevicePool/Pool/ActivePool (shared)
#include "vllm/model_executor/models/qwen3_5_moe_block.h"   // RunMoeBlock (SEAM GAP #2)
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// Reuse the shared dense self-attention block + all its device glue (Dev/DBuf/pool
// policy, ResidentWeight, KvSlice, StepInputs/BuildStepInputs, AttnBlock) exactly
// as the Qwen3-dense forward does — the attention preamble is byte-for-byte the
// dense path (the spike's "combine two done pieces" thesis).
using namespace dense_attn;

// One Qwen3-Coder decoder layer (qwen3_moe.py::Qwen3MoeDecoderLayer): input norm
// (std add+RMSNorm) -> attention -> post norm (std add+RMSNorm) -> MoE block. The
// residual accumulator `res` (bf16 [T,H]) is threaded through the two fused
// add+RMSNorm producers; `hidden`/`hidden_hold` carry the current block-output
// delta — a device tensor whose storage is either the previous layer's owning
// MoeBlockOutput (`hidden_hold`) or the embedding buffer (held by the caller). The
// MoE block output becomes the new delta, so the final RMSNorm fuses it into res.
void RunMoeLayer(Dev d, const Qwen3MoeLayerWeights& layer, const HfConfig& cfg,
                 Tensor& hidden, std::shared_ptr<void>& hidden_hold, DBuf& res,
                 const StepInputs& si, const CommonAttentionMetadata& meta,
                 const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dhn.t(), hidden, w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dhn.t(), hidden, w_in, vt::RmsNormArgs{eps, false}, &res.t());
  }

  DBuf attn = AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
  }

  // Sparse-MoE block (router + top-k experts, NO shared expert — guarded on
  // shared_expert_intermediate_size==0 inside MoeBlock). RunMoeBlock returns an
  // owning device [T,H] bf16 buffer; it becomes the new residual-stream delta.
  MoeBlockOutput moe = RunMoeBlock(d.q, layer.moe, cfg, dh2.t(), T);
  hidden = moe.tensor;
  hidden_hold = std::move(moe.storage);
}

// GatherRows: gather the idx-indexed rows of `src` [.,H] into contiguous `dst`.
// (Copied from qwen3.cpp — the logits_indices gather-before-lm_head helper.)
void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// embed -> N MoE layers -> final RMSNorm -> untied lm_head. Returns [n_out, vocab]
// f32 as a device DBuf (no host Download; the caller Downloads or wraps it).
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Qwen3MoeWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3 moe: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3 moe: one PagedKvCache per layer required");

  // Embed: hidden[T,H] bf16 = embed_tokens[token_ids]. The embedding buffer holds
  // the layer-0 input delta; it stays alive (embed_buf) until layer 0 consumes it.
  DBuf embed_buf(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    vt::Embedding(d.q, embed_buf.t(), dtab, dids.t());
  }
  Tensor hidden = embed_buf.t();
  std::shared_ptr<void> hidden_hold;  // owns the current MoE-output delta storage

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunMoeLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden,
                hidden_hold, res, si, attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm.t(), hidden, w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm.t(), hidden, w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  }

  // lm_head. UNTIED (Qwen3-Coder): the loaded Matmul-B [H,vocab] lm_head via
  // vt::Matmul. The tied branch (aliases embed_tokens via MatmulBT) is kept for
  // completeness but Qwen3-Coder never takes it (tie_word_embeddings=false).
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
  // The pool block's lifetime moves into a shared_ptr whose deleter returns it to
  // the DevicePool — no per-step cudaMalloc/cudaFree (mirrors qwen3.cpp).
  const size_t alloc = dlogits.alloc_bytes();
  void* p = dlogits.Release();
  fl.device_storage =
      std::shared_ptr<void>(p, [alloc](void* q) { Pool().Put(alloc, q); });
  (void)d;
  return fl;
}

}  // namespace

std::vector<float> Qwen3MoeModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3MoeWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Qwen3MoeModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3MoeWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
