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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"  // LinearMethod seam
#include "vllm/model_executor/models/decode_graph_sizes.h"  // DecodeGraphSizes/PadToCaptureSize
#include "vllm/model_executor/models/dense_attn_block.h"  // shared AttnBlock + device glue
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // NVFP4 W4A16 dispatch
#include "vllm/model_executor/models/device_pool.h"     // DevicePool/Pool/ActivePool (shared)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::DeviceTokenIds seam
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
// SiluAndMul -> down_proj. `dh2` is the post-norm hidden [T,H] bf16.
//
// Routed through the LinearMethod seam (S4): the gate_up+SiluAndMul and the
// down projection each go through a method chosen ONCE by the checkpoint's
// scheme (bf16 UnquantizedLinearMethod vs NVFP4 W4A16), so this forward no
// longer carries the `IsNvfp4()` probe, the `device == kCUDA` gate or the
// `#ifdef VT_MARLIN_NVFP4` fused-path dispatch — they live in the shared
// quantization scheme headers. Byte-identical: the methods run the exact same
// vt:: ops in the same order the inline path did.
DBuf MlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const HfConfig& cfg,
              const Tensor& dh2, int64_t /*T*/,
              const TensorParallel* tp = nullptr) {
  const int64_t I = cfg.intermediate_size;
  // gate_up is a MergedColumnParallelLinear (sharded on I, no comm); down is a
  // RowParallelLinear whose per-rank partial [T,H] products are all-reduced below
  // (linear.py:1766). tp_size==1 ⇒ whole tensors + the all-reduce is a no-op, so
  // this is byte-identical to the single-GPU MLP.
  auto gate_up = layers::MakeMlpGateUpMethod(w.gate_up_proj, w.gate_proj_fp4,
                                             w.up_proj_fp4, I);
  DBuf act = gate_up->Apply(d, dh2);
  auto down = layers::MakeLinearMethod(w.down_proj, w.down_proj_fp4);
  DBuf out = down->Apply(d, act.t(), DType::kBF16);
  Tensor ot = out.t();
  TpAllReduceSum(tp, d.q, ot);
  return out;
}

// One dense decoder layer (qwen3.py::Qwen3DecoderLayer): input norm (std
// add+RMSNorm) -> attention -> post norm (std add+RMSNorm) -> MLP. `hidden` (bf16
// [T,H]) is the delta; `res` (bf16 [T,H]) the residual accumulator.
void RunLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, DBuf& res, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T,
              const TensorParallel* tp = nullptr) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());
  }

  DBuf attn = AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T, tp);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
  }

  hidden = MlpBlock(d, layer.mlp, cfg, dh2.t(), T, tp);
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

// ROW-SERVE-ASYNC-DENSE-MIRROR (ENG-ASYNC-SCHED W4 / the #31 P0, ported to the
// classic dense family): overwrite the REAL prefix of a freshly uploaded input-id
// buffer with the device-resident ids the async runner's combine produced. The
// exact analogue of qwen3_5.cpp's ApplyDeviceTokenIdsOverride — that TU wired it
// for the gate models (MoE + 27B dense); this is the identical consumer for the
// SHARED pure-dense driver (Qwen3ForCausalLM and every registry that routes
// through Qwen3DenseModel / EmbedInto: InternLM2, Mistral, Llama).
//
// WHY: on the async serving loop (AsyncLLM depth-2) the sampled token is NOT
// written to token_ids_cpu synchronously; the runner's device combine splices each
// decode row's real token into the device input-ids on the MAIN QUEUE while the
// host `token_ids` vector stays stale. The default host upload below then RACES
// that device write (unsynchronized device-write/host-read), nondeterministically
// embedding the stale/zero placeholder -> token-0 degeneration. Copying the
// device ids over the DBuf prefix here is main-queue-ordered AFTER the combine, so
// the embed never does the racing host read — exactly upstream (states.py:64
// device-resident prev_sampled_token_ids + gpu_model_runner.py GPU gather).
//
// The override is published by the registry forward's detail::DeviceTokenIdsScope
// and CONSUMED here on first use; null on every path except the CUDA async runner,
// so with no override this is byte-identical to the pre-fix host upload.
static void ApplyDeviceTokenIdsOverride(Dev d, DBuf& dids, int64_t T) {
  const detail::DeviceTokenIds ov = detail::DeviceTokenIdsOverride();
  if (ov.ids == nullptr) return;
  detail::DeviceTokenIdsOverride() = detail::DeviceTokenIds{};
  // A device buffer LONGER than the embed's input would run past the end. That can
  // only mean the runner and the model disagree about this step's shape, so fail
  // loudly rather than corrupt the embedding.
  VT_CHECK(ov.count <= T,
           "qwen3 dense embed: device input ids longer than the embed input");
  d.b.Copy(d.q, dids.ptr(), ov.ids,
           static_cast<size_t>(ov.count) * sizeof(int32_t));
}

// Embed: hidden[T,H] bf16 = embed_tokens[token_ids] (device-resident table). KEPT
// OUTSIDE THE CUDA-GRAPH (mirrors qwen3_moe.cpp / qwen3_5.cpp EmbedInto): the CUDA
// Embedding op allocates a device bounds-check flag (cudaMalloc/cudaFree) and syncs
// the stream, both illegal inside a capture region — and it consumes the HOST
// token_ids. The graph driver runs this per step into its PERSISTENT hidden buffer,
// then captures/replays ForwardLayers over that fixed hidden address.
void EmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
               const Qwen3DenseWeights& weights, const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  Tensor dtab = ResidentWeight(d, weights.embed_tokens,
                               {config.vocab_size, config.hidden_size});
  // ROW-SERVE-ASYNC-DENSE-MIRROR: when the async runner has already placed this
  // step's input ids on the device (and spliced each decode row's sampled token
  // into them there), embed straight from that buffer. `token_ids` is stale for
  // decode rows in that case BY DESIGN — materializing it on the host is the
  // synchronize the async path removes — so its real prefix is overwritten here.
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  ApplyDeviceTokenIdsOverride(d, dids, T);
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());
}

// The CAPTURABLE region: everything AFTER the embedding — the residual stream
// (res=0), the N dense decoder layers, the final RMSNorm and the (tied/untied)
// lm_head — returning [n_out, vocab] f32 as a device DBuf (no host Download). Split
// out of the eager forward body so the exact op sequence is what the decode graph
// captures/replays; every per-step-varying input is read from a HOST vector
// argument (positions / the attention-metadata vectors, via BuildStepInputs) whose
// host->device copies are capturable on GB10, and which the graph driver keeps
// persistent + mutates in place so a replay picks up the new step's inputs.
//
// `hidden_in` is the embedded input (a view over the graph's persistent hidden
// buffer on the replay path). It is COPIED into a working buffer so the per-layer
// `hidden` DBuf reassignment (RunLayer's `hidden = MlpBlock(...)`) never disturbs
// the persistent embedding — the copy is a pure device->device data move, so the
// layer sequence and its output are BYTE-IDENTICAL to the pre-split forward.
// `return_hidden` (ARCH-ONE-SURFACE ROW 6, default false = byte-identical
// text path): when true, STOP after the final RMSNorm (+ the logits_indices
// gather) and return the [n_out, H] hidden rows upcast to f32 — the pooling
// forward of an embedding conversion, whose model has NO lm_head at all
// (adapters.py:135-151: as_embedding_model replaces the output layer with a
// missing-layer stage; the pooler consumes the post-final-norm hidden). Every
// existing caller leaves the default, so the lm_head tail is untouched.
DBuf ForwardLayers(Dev d, const Tensor& hidden_in,
                   const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& attn_meta,
                   const std::vector<PagedKvCache>& attn_kv,
                   const Qwen3DenseWeights& weights, const HfConfig& config,
                   const std::vector<int32_t>& logits_indices,
                   bool return_hidden = false) {
  const int64_t T = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3 dense: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3 dense: one PagedKvCache per layer required");

  // Working copy of the embedded hidden (device->device; captured). RunLayer
  // reassigns `hidden` per layer, so it must NOT alias the persistent buffer.
  DBuf hidden(d, DType::kBF16, {T, H});
  d.b.Copy(d.q, hidden.ptr(), hidden_in.data,
           static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));

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

  // ARCH-ONE-SURFACE ROW 6 pooling tail: the post-final-norm hidden rows,
  // upcast bf16 -> f32 (vt::CastF32), with NO lm_head — an embedding-converted
  // checkpoint has no output layer to multiply by. Never taken by any text
  // caller (return_hidden defaults false).
  if (return_hidden) {
    DBuf dhid(d, DType::kF32, {n_out, H});
    vt::CastF32(d.q, dhid.t(), src);
    return dhid;
  }

  // lm_head. Tied (Qwen3-0.6B): logits = hidden @ embed_tokens^T via MatmulBT
  // over the [vocab,H] embed table (== [N=vocab,K=H]). Untied: the loaded
  // Matmul-B [H,vocab] lm_head via vt::Matmul.
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);
  return logits;
}

// Full eager forward body: embed (host token_ids) then the capturable layer region.
// Used by Qwen3DenseModel::Forward/ForwardDevice and by the graph driver's eager
// fallback / cold-size pre-warm step (one contiguous stream, no capture). Byte-
// identical op sequence to the graph (eager output == replay output).
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Qwen3DenseWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices,
                 bool return_hidden = false) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  DBuf hidden(d, DType::kBF16, {T, config.hidden_size});
  EmbedInto(d, hidden, token_ids, weights, config);
  return ForwardLayers(d, hidden.t(), positions, attn_meta, attn_kv, weights, config,
                       logits_indices, return_hidden);
}

ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  // The pool block's lifetime moves into a shared_ptr whose deleter returns it to
  // the DevicePool — no per-step cudaMalloc/cudaFree, and the buffer safely
  // outlives sampling (mirrors qwen3_5.cpp WrapDeviceLogits).
  fl.device_storage = dlogits.ReleaseShared();
  (void)d;
  return fl;
}

// NON-OWNING [rows, vocab] f32 view over a buffer the graph slot keeps alive
// (mirrors qwen3_moe.cpp / qwen3_5.cpp ViewDeviceLogits). Stream ordering
// guarantees the sampler's later reads see the replay's writes; the next same-size
// replay overwrites the buffer, so in-place sampler mutation is safe.
ForwardLogits ViewDeviceLogits(void* base, vt::Device device, int64_t rows,
                               int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = MakeTensor(base, DType::kF32, device, {rows, vocab});
  fl.device_storage = std::shared_ptr<void>(base, [](void*) {});
  return fl;
}

// Overwrite dst's CONTENTS from src WITHOUT changing dst.data() when the sizes
// already match (preserves the fixed address a captured host->device copy reads
// from); reallocate only when the shape actually changed (qwen3_moe.cpp CopyInPlace).
template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}

// Build the S-padded PURE-DECODE inputs from the real B-request step (B<=S). The
// ATTENTION-ONLY analogue of qwen3_5.cpp's BuildPaddedDecode (pure dense has no GDN
// metadata) — byte-for-byte the qwen3_moe.cpp BuildPaddedDecodeAttn.
//
// The decode forward is ROW-INDEPENDENT (paged attention is per-request causal; the
// norm / SwiGLU MLP / lm_head are per-token with no cross-row reduction), so
// appending S-B INERT rows cannot perturb the real rows' logits. The padding rows
// are made inert exactly as vLLM's cudagraph padding:
//   * token id / position 0 (the embed row is discarded);
//   * slot_mapping = -1 -> ReshapeAndCache skips the KV write, so no real KV block
//     is touched;
//   * seq_lens = 1 + block_table row 0 -> paged attention does a valid in-bounds
//     read of block 0 whose output row is discarded (never returned).
// The real prefix [0,B) is copied verbatim, so at S==B this is a bit-identical
// rebuild of the eager inputs.
void BuildPaddedDecodeAttn(int64_t S, const std::vector<int32_t>& tok,
                           const std::vector<int32_t>& pos,
                           const CommonAttentionMetadata& am,
                           std::vector<int32_t>& tok_out,
                           std::vector<int32_t>& pos_out,
                           CommonAttentionMetadata& am_out) {
  const int64_t cols = am.block_table_num_cols;

  tok_out.assign(static_cast<size_t>(S), 0);
  pos_out.assign(static_cast<size_t>(S), 0);
  std::copy(tok.begin(), tok.end(), tok_out.begin());
  std::copy(pos.begin(), pos.end(), pos_out.begin());

  am_out = am;  // carries causal + block_table_num_cols + max_seq_len
  am_out.num_reqs = static_cast<int>(S);
  am_out.num_actual_tokens = static_cast<int>(S);
  am_out.max_query_len = 1;  // pure decode
  am_out.slot_mapping.assign(static_cast<size_t>(S), -1);
  std::copy(am.slot_mapping.begin(), am.slot_mapping.end(),
            am_out.slot_mapping.begin());
  am_out.seq_lens.assign(static_cast<size_t>(S), 1);
  std::copy(am.seq_lens.begin(), am.seq_lens.end(), am_out.seq_lens.begin());
  am_out.block_table_tensor.assign(static_cast<size_t>(S * cols), 0);
  std::copy(am.block_table_tensor.begin(), am.block_table_tensor.end(),
            am_out.block_table_tensor.begin());
  am_out.query_start_loc.resize(static_cast<size_t>(S + 1));
  for (int64_t i = 0; i <= S; ++i)
    am_out.query_start_loc[static_cast<size_t>(i)] = static_cast<int32_t>(i);
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

ForwardLogits Qwen3DenseModel::ForwardHidden(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  // ARCH-ONE-SURFACE ROW 6: the POOLING forward — the same embed + layer stack
  // as Forward/ForwardDevice, stopping after the final RMSNorm (+ gather) with
  // NO lm_head, mirroring an as_embedding_model conversion whose output layer
  // is a missing-layer stage (adapters.py:135-151). The [n_out, H] f32 rows are
  // downloaded to the host carrier: the landed pooler ops are host-side, and an
  // embedding batch is one prefill (no per-step decode loop to keep resident).
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dhidden = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices, /*return_hidden=*/true);
  const int64_t n_out = dhidden.t().shape[0];
  const int64_t H = config.hidden_size;
  ForwardLogits fl;
  fl.rows = n_out;
  fl.vocab = H;  // the carrier's row width IS the hidden size on this path
  fl.host.resize(static_cast<size_t>(n_out) * static_cast<size_t>(H));
  dhidden.Download(d, fl.host.data());
  return fl;
}

// ─── Qwen3DenseDecodeGraph (shared pure-dense decode CUDA-graph driver) ───────
// The pure-dense sibling of Qwen3MoeDecodeGraph (qwen3_moe.cpp) — SAME cold ->
// warm -> capture -> replay state machine, SAME padded-batch capture set
// (decode_graph_sizes.h) and SAME persistent fixed-address host inputs + persistent
// embed/logits buffers, driving the dense forward (ForwardLayers over EmbedInto)
// with a dense SwiGLU MLP instead of the MoE block. NO GDN (attention-only).
//
// Ported from: vllm/v1/worker/gpu_model_runner.py::GPUModelRunner @ e24d1b24
//   (`_dummy_run` warm-up then capture, then graph dispatch per decode step) +
//   vllm/compilation/cuda_graph.py (`CUDAGraphWrapper.__call__`: pad the batch to a
//   captured size, replay, else run eager).
//
// GRAPH-SAFETY AUDIT of the bf16 dense decode path (capture requires stable pointers
// and no host sync / stream-ordered alloc inside the region) — identical to the
// already-shipped Qwen3MoeDecodeGraph (its d128 full-attention capture path IS this
// one, minus the MoE-only scratch):
//   * Embedding (device flag cudaMalloc + stream sync) stays OUTSIDE (EmbedInto).
//   * All device scratch comes from the shared DevicePool, whose blocks are recycled
//     (never returned to the driver) — the cold pre-warm step at this exact size
//     populates every size class the capture then reuses, so capture itself performs
//     no cudaMalloc.
//   * The graph-safe persistent RoPE row-index table (BuildStepInputs, W7) is baked
//     once per T and never moved.
//   * ResidentWeight uploads every weight once, on first touch (pre-warm).
//   * The FA-2 varlen-decode launcher's per-shape scratch throws if it misses during
//     capture (cuda_flash_attn_fa2.cu) — the pre-warm step at the same padded size
//     populates it. Its host `max_seq_len` only sizes the split-KV grid; the
//     per-request causal geometry is read from the DEVICE seq_lens, and each split's
//     KV range is derived in-kernel from `seqused_k`, so a captured graph stays
//     CORRECT as the sequences grow (identical contract to the shipped decode graphs).
//   * cuBLASLt's workspace is a one-time per-context cudaMalloc.
struct Qwen3DenseDecodeGraph::Impl {
  Impl(const Qwen3DenseWeights& w, const HfConfig& c, vt::Queue q, int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    const char* env = std::getenv("VLLM_CPP_CUDAGRAPH");
    const bool env_on = (env == nullptr) || std::string(env) != "0";
    Backend& b = vt::GetBackend(queue.device.type);
    enabled = env_on &&
              platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
              b.SupportsGraphCapture();
  }
  ~Impl() {
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr,
                   "[Qwen3DenseDecodeGraph] dense decode graph: %lld total replays "
                   "across %zu captured size(s)\n",
                   static_cast<long long>(replays), slots.size());
    Backend& b = vt::GetBackend(queue.device.type);
    for (auto& kv : slots)
      if (kv.second.graph != nullptr) b.DestroyGraph(kv.second.graph);
  }

  // One captured padded batch size. Owns its OWN persistent host inputs (the
  // captured graph's host->device copies bake these addresses, so each size needs
  // its own fixed-address buffers), its persistent embed target + logits output, and
  // its instantiated graph.
  struct SizeSlot {
    std::vector<int32_t> token_ids;  // [S]
    std::vector<int32_t> positions;  // [S]
    CommonAttentionMetadata attn_meta;
    std::unique_ptr<DBuf> hidden;  // [S,H] bf16 persistent embed target
    std::unique_ptr<DBuf> logits;  // [S,vocab] f32 held graph output
    void* graph = nullptr;         // instantiated cudaGraphExec (opaque)
    int fa_cols = -1;              // captured block-table column count
    bool captured = false;
    bool warm = false;
    int64_t replays = 0;

    // In-place refresh of the persistent host inputs (fixed addresses once the
    // slot's vectors reach size S) so a replay re-reads this step's tokens.
    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const CommonAttentionMetadata& am) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
    }
  };

  const Qwen3DenseWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;

  std::map<int64_t, SizeSlot> slots;  // padded size S -> slot
  int64_t replays = 0;                // total replays (diagnostics)
  bool any_captured = false;          // diagnostics: at least one live graph
};

Qwen3DenseDecodeGraph::Qwen3DenseDecodeGraph(const Qwen3DenseWeights& weights,
                                             const HfConfig& config, vt::Queue queue,
                                             int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

Qwen3DenseDecodeGraph::~Qwen3DenseDecodeGraph() = default;

bool Qwen3DenseDecodeGraph::captured() const { return impl_->any_captured; }
int64_t Qwen3DenseDecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits Qwen3DenseDecodeGraph::Step(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv) {
  const int64_t B = static_cast<int64_t>(token_ids.size());
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  const int64_t vocab = impl_->config.vocab_size;
  const int64_t H = impl_->config.hidden_size;

  // Pure decode passes identity logits_indices (gather is a no-op), so the
  // capturable region returns the full [S,vocab].
  const std::vector<int32_t> kNoGather;
  const int64_t S = PadToCaptureSize(B, impl_->max_num_reqs);
  if (!impl_->enabled || S < 0) {
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, attn_kv,
                          impl_->weights, impl_->config, kNoGather);
    return WrapDeviceLogits(d, std::move(lg), B, vocab);
  }

  // Pad this step's real B-request inputs up to S (inert padding rows), then refresh
  // THIS size's persistent host buffers in place.
  Impl::SizeSlot& s = impl_->slots[S];
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  CommonAttentionMetadata pam;
  BuildPaddedDecodeAttn(S, token_ids, positions, attn_meta, ptok, ppos, pam);

  // A block-table column-count change reallocates the persistent block_table (the
  // captured H2D copy's source address moves) -> invalidate this slot's graph and
  // re-warm/re-capture.
  const bool cols_changed = (s.fa_cols != -1 && s.fa_cols != cols);
  s.Refresh(ptok, ppos, pam);
  s.fa_cols = cols;
  if (cols_changed && s.graph != nullptr) {
    b.DestroyGraph(s.graph);
    s.graph = nullptr;
    s.captured = false;
    s.warm = false;
  }

  // Fast path: this size's graph is captured. Embed OUTSIDE the graph into the
  // persistent hidden buffer, then relaunch the captured layer region.
  if (s.captured) {
    EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    b.ReplayGraph(impl_->queue, s.graph);
    ++s.replays;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Warm: the pool + weight residency + per-shape kernel scratch were warmed for
  // this size by the previous (eager) step. CAPTURE the layer region once,
  // instantiate the graph, then launch it.
  if (s.warm) {
    EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    b.BeginCapture(impl_->queue);
    DBuf lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv,
                            impl_->weights, impl_->config, kNoGather);
    s.graph = b.EndCaptureGraph(impl_->queue);
    s.logits = std::make_unique<DBuf>(std::move(lg));
    s.captured = true;
    impl_->any_captured = true;
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr,
                   "[Qwen3DenseDecodeGraph] captured dense decode graph for padded "
                   "size S=%lld (real B=%lld)\n",
                   static_cast<long long>(S), static_cast<long long>(B));
    b.ReplayGraph(impl_->queue, s.graph);
    s.replays = 1;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool size classes, the
  // resident weights, and the FA-2 per-shape scratch for this size) and defer
  // capture to the next same-size step. This is a real decode step — nothing wasted.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
  DBuf lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv,
                          impl_->weights, impl_->config, kNoGather);
  s.warm = true;
  s.captured = false;
  // lg is [S,vocab]; hand ownership out but expose only the first B (real) rows.
  ForwardLogits fl = WrapDeviceLogits(d, std::move(lg), B, vocab);
  fl.device_tensor =
      MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
  return fl;
}

// Per-family gate (see qwen3.h). DEFAULT ON (row QUANT-CT-MXFP4-MARLIN-STRUCT step 1,
// parity-enabler): the shared dense decode CUDA-graph is byte-coherent + token-exact
// vs the eager forward on both dense checkpoints — test_qwen3_paged_engine 184/184
// (Qwen3-0.6B near-tie + Qwen3-4B) and test_qwen3_dense_async_serving 82/82, graph
// ON == OFF — and on the Qwen3-8B-MXFP4 #44 smoke (deterministic 3/3 token-exact +
// coherent), all captured on GB10. An explicit VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH=0
// opts back out to the eager path (byte-identical to the pre-graph forward); the
// framework kill switch VLLM_CPP_CUDAGRAPH=0 additionally forces eager inside the
// driver (Impl::enabled), so the graph never captures under either opt-out.
bool DenseDecodeGraphEnabled() {
  const char* value = std::getenv("VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH");
  return !(value != nullptr && value[0] == '0');
}

std::optional<ForwardLogits> DenseDecodeGraphForward(
    std::unique_ptr<Qwen3DenseDecodeGraph>& graph,
    const Qwen3DenseWeights& weights, const ModelForwardInput& input) {
  // Only a graph-eligible PURE decode on a CUDA static-graph platform routes here;
  // otherwise the caller runs its existing eager Forward/ForwardDevice path. The
  // driver itself caps the padded batch at max_num_reqs and falls back internally
  // for larger batches / when the framework kill switch is set.
  if (!DenseDecodeGraphEnabled() || !input.pure_decode ||
      !platforms::GetPlatform(input.queue.device.type).support_static_graph_mode()) {
    return std::nullopt;
  }
  // #323 — CORRECTNESS FIRST. `Step()` below replays against the HOST
  // `input.token_ids` and never reads `input.device_token_ids`. On the depth-2
  // async path the combine has patched the DEVICE ids and `token_ids` is
  // deliberately stale for decode rows (runner.cpp), so the replay generates
  // from stale ids and every concurrent request past slot 0 degenerates — the
  // #31 signature, reproduced on Mistral-7B-v0.3 and InternLM2-chat-1.8B and
  // latent for EVERY classic-dense model, since the graph is default-ON.
  //
  // Measured, same binary, 4-concurrent battery vs a batch-1 sync anchor:
  //   depth-1, graph ON   PASS 78/78      (no async pipelining)
  //   depth-2, graph OFF  PASS 82/82      (eager path honours the scope)
  //   depth-2, graph ON   FAIL, slots 1-3 degenerate
  // Both conditions are required, which is why the registry-level
  // DeviceTokenIdsScope (60e71a0e) did not close it: this path returns BEFORE
  // the eager forward ever runs.
  //
  // Declining the graph while the mirror is live falls back to that
  // proven-correct eager path. This is a MITIGATION, not the end state — the
  // real fix is for Step() to read the ids at REPLAY time (a stable device
  // buffer), which restores graphed decode for async serving. Until then a
  // correct stream outranks the graph's throughput.
  if (input.device_token_ids != nullptr) {
    return std::nullopt;
  }
  // gdn_state_slots carries max_num_reqs for EVERY arch (the runner sets it from
  // max_num_reqs_ regardless of whether the model has GDN layers), so a pure
  // full-attention model reads its capture-size cap from it unchanged.
  if (!graph) {
    graph = std::make_unique<Qwen3DenseDecodeGraph>(weights, input.config,
                                                    input.queue, input.gdn_state_slots);
  }
  return graph->Step(input.token_ids, input.positions, input.attn_meta,
                     input.attn_kv);
}

}  // namespace vllm
