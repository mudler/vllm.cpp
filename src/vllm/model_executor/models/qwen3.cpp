// Qwen3 DENSE (`Qwen3ForCausalLM`) forward — the first ADDITIVE-MODEL bring-up
// W3 (the capstone). A pure standard-dense transformer forward COMPOSED from the
// public vt:: ops + the fusion catalog (kFusedAddRmsNormStd / kAttnQkNormRope,
// include/vt/recipes.h), with NO GDN, NO MoE and NO attention output gate. It is
// the Qwen3.6-dense full-attention path (qwen3_5.cpp DenseForwardLayers /
// FullAttnBlockPaged) stripped to the pure-dense subset:
//   - ONE full-attention KV group per layer (no MambaSpec/GDN, no hybrid split);
//   - STANDARD (non-gemma) RMSNorm at input/post/final norms;
//   - per-head q_norm/k_norm (RMSNorm(head_dim), non-gemma) applied BEFORE RoPE;
//   - NO attention gate (Qwen3 has none);
//   - a TIED lm_head (aliases embed_tokens).
//
// Grounding: vllm/model_executor/models/qwen3.py @ e24d1b24
//   Qwen3Attention (:65-168), Qwen3MLP=Qwen2MLP (:58), Qwen3DecoderLayer
//   (:171-242), Qwen3Model=Qwen2Model (:260), tied lm_head (:294-295).
// See .agents/specs/first-additive-model-qwen3-dense.md §2/§4/§6.
//
// Numeric contract (mirrors the qwen3_5 full-attention FALLBACK path — the
// token-exact paged==dense anchor): the residual stream is the model dtype (bf16,
// matching vLLM's fused_add_rms_norm residual); the qkv GEMM emits f32 q/k/v so
// the per-head q/k RMSNorm + RoPE run in f32; the paged KV cache is written bf16
// (down-cast K/V) while the query stays f32 into vt::PagedAttention; o_proj and
// the whole MLP flow bf16. Returns [n_out, vocab] f32 logits.
//
// Self-contained device glue (Dev/DBuf/ResidentWeight): the DBuf here draws its
// scratch from the SHARED DevicePool (include/vllm/model_executor/models/
// device_pool.h — extracted verbatim from qwen3_5.cpp), so the dense forward
// reuses freed blocks instead of a per-op cudaMalloc/cudaFree. This is a pure
// allocation-source change (identical computation ⇒ byte-identical output; all
// gate models unchanged). NOTE: a clean same-binary A/B (Qwen3-4B c1+c8)
// measured the pool PERF-NEUTRAL on this model — the async scheduler already
// overlaps the host-side alloc syncs with GPU compute; it is kept as byte-safe
// hygiene + code sharing, not a measured TTFT lever. The real dense-TTFT lever
// is the RoPE cos|sin cache below.
#include "vllm/model_executor/models/qwen3.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // shared AttnBlock + device glue
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // NVFP4 W4A16 dispatch
#include "vllm/model_executor/models/device_pool.h"     // DevicePool/Pool/ActivePool (shared)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
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

// The dense self-attention block + all its device glue (Dev/DBuf/pool policy,
// ResidentWeight[F32]/WeightF32, KvSlice, StepInputs/BuildStepInputs, the
// env-flag readers and AttnBlock itself) were EXTRACTED VERBATIM to the shared
// header include/vllm/model_executor/models/dense_attn_block.h so the first
// full-attention MoE (Qwen3-Coder `Qwen3MoeForCausalLM`, qwen3_moe.cpp W3)
// reuses the exact same attention preamble. This is a PURE RELOCATION: the
// definitions are byte-for-byte the same and the dense-only MLP / decoder-layer
// / forward-body machinery below composes them via `using namespace dense_attn`,
// so the Qwen3-dense (0.6B/4B) forward is byte-identical (same vt:: op order).
using namespace dense_attn;

// Dense SwiGLU MLP (qwen3.py::Qwen3MLP=Qwen2MLP): merged gate_up_proj ->
// SiluAndMul -> down_proj, all bf16. `dh2` is the post-norm hidden [T,H] bf16.
DBuf MlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const HfConfig& cfg,
              const Tensor& dh2, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  if (w.IsNvfp4()) {
    // NVFP4 W4A16 MLP (compressed-tensors `nvfp4-pack-quantized`). The activation
    // is bf16 throughout, which is exactly the a16 contract; the ONLY change vs
    // the BF16 arm above is which GEMM the same [T,H]/[T,I] activations flow
    // through — the SwiGLU, shapes and residual handling are untouched.
    DBuf act = [&] {
#ifdef VT_MARLIN_NVFP4
      if (d.q.device.type == vt::DeviceType::kCUDA &&
          dense_nvfp4::MarlinW4A16Enabled() &&
          dense_nvfp4::GateUpFusedEligible(w.gate_proj_fp4, w.up_proj_fp4)) {
        // vLLM's shape: ONE Marlin GEMM over the merged gate_up operand
        // (size_n = 2I) + SiluAndMul on the halves.
        return dense_nvfp4::GateUpFusedMarlinD(d, dh2, w.gate_proj_fp4,
                                               w.up_proj_fp4);
      }
#endif
      // SPLIT A/B fallback (and the CPU reference path): two separate GEMMs fed
      // to the two-input MoeSiluMul — bit-identical to the fused arm's
      // SiluAndMul over the merged halves (test_ops_moe_grouped probe).
      DBuf gate = dense_nvfp4::MatmulNvfp4W4A16D(d, dh2, w.gate_proj_fp4,
                                                 DType::kBF16);
      DBuf up = dense_nvfp4::MatmulNvfp4W4A16D(d, dh2, w.up_proj_fp4,
                                               DType::kBF16);
      DBuf a(d, DType::kBF16, {T, I});
      vt::MoeSiluMul(d.q, a.t(), gate.t(), up.t());
      return a;
    }();
    return dense_nvfp4::MatmulNvfp4W4A16D(d, act.t(), w.down_proj_fp4,
                                          DType::kBF16);
  }
  Tensor wgu = ResidentWeight(d, w.gate_up_proj);  // [2I, H] raw-NK
  DBuf gate_up(d, DType::kBF16, {T, 2 * I});
  vt::MatmulBT(d.q, gate_up.t(), dh2, wgu);
  DBuf act(d, DType::kBF16, {T, I});
  vt::SiluAndMul(d.q, act.t(), gate_up.t());  // silu(gate)*up
  Tensor wdn = ResidentWeight(d, w.down_proj);  // [H, I] raw-NK
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wdn);
  return out;
}

// One dense decoder layer (qwen3.py::Qwen3DecoderLayer): input norm (std
// add+RMSNorm) -> attention -> post norm (std add+RMSNorm) -> MLP. `hidden` (bf16
// [T,H]) is the delta; `res` (bf16 [T,H]) the residual accumulator.
void RunLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, DBuf& res, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());
  }

  DBuf attn = AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
  }

  hidden = MlpBlock(d, layer.mlp, cfg, dh2.t(), T);
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

// embed -> N layers -> final RMSNorm -> lm_head. Returns [n_out, vocab] f32 as a
// device DBuf (no host Download; the caller Downloads or wraps it).
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Qwen3DenseWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3 dense: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3 dense: one PagedKvCache per layer required");

  // Embed: hidden[T,H] bf16 = embed_tokens[token_ids].
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  }

  // lm_head. Tied (Qwen3-0.6B): logits = hidden @ embed_tokens^T via MatmulBT
  // over the [vocab,H] embed table (== [N=vocab,K=H]). Untied: the loaded
  // Matmul-B [H,vocab] lm_head via vt::Matmul.
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
  // the DevicePool — no per-step cudaMalloc/cudaFree, and the buffer safely
  // outlives sampling (mirrors qwen3_5.cpp WrapDeviceLogits).
  const size_t alloc = dlogits.alloc_bytes();
  void* p = dlogits.Release();
  fl.device_storage =
      std::shared_ptr<void>(p, [alloc](void* q) { Pool().Put(alloc, q); });
  (void)d;
  return fl;
}

}  // namespace

std::vector<float> Qwen3DenseModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Qwen3DenseModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
