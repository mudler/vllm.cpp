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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/decode_graph_sizes.h"  // DecodeGraphSizes/PadToCaptureSize
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

// Embed: hidden[T,H] bf16 = embed_tokens[token_ids] (device-resident table).
// KEPT OUTSIDE THE CUDA-GRAPH (mirrors qwen3_5.cpp `EmbedInto`): the CUDA
// Embedding op allocates a device bounds-check flag (cudaMalloc/cudaFree) and
// syncs the stream (cuda_ops.cu:525,535), all illegal inside a capture region —
// and it consumes the HOST token_ids. The graph driver runs this per step into
// its PERSISTENT hidden buffer, then captures/replays ForwardLayers over that
// fixed hidden address.
void EmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
               const Qwen3MoeWeights& weights, const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  Tensor dtab = ResidentWeight(d, weights.embed_tokens,
                               {config.vocab_size, config.hidden_size});
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());
}

// The CAPTURABLE region: everything AFTER the embedding — the residual stream
// (res=0), the N full-attention MoE decoder layers, the final RMSNorm and the
// untied lm_head — returning [n_out, vocab] f32 as a device DBuf (no host
// Download). Split out of ForwardBody (W7) so the exact op sequence is what the
// graph captures/replays; every per-step-varying input is read from a HOST vector
// argument (positions / the attention-metadata vectors) whose host->device copies
// are capturable on GB10, and which the graph driver keeps persistent + mutates
// in place so a replay picks up the new step's inputs.
//
// `hidden_in` is the embedded input (a view over the graph's persistent hidden
// buffer on the replay path). The layer loop only ever READS it (the first fused
// add+RMSNorm accumulates into `res` and reassigns `hidden` to the MoE output), so
// unlike the 35B — whose RunLayerPaged writes the residual stream in place — no
// defensive copy of the persistent buffer is needed.
DBuf ForwardLayers(Dev d, const Tensor& hidden_in,
                   const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& attn_meta,
                   const std::vector<PagedKvCache>& attn_kv,
                   const Qwen3MoeWeights& weights, const HfConfig& config,
                   const std::vector<int32_t>& logits_indices) {
  const int64_t T = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3 moe: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3 moe: one PagedKvCache per layer required");

  Tensor hidden = hidden_in;
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

// Full eager forward body: embed (host token_ids) then the capturable layer
// region. Used by Qwen3MoeModel::Forward/ForwardDevice and by the graph driver's
// eager fallback / cold-size pre-warm step (one contiguous stream, no capture).
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Qwen3MoeWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  DBuf hidden(d, DType::kBF16, {T, config.hidden_size});
  EmbedInto(d, hidden, token_ids, weights, config);
  return ForwardLayers(d, hidden.t(), positions, attn_meta, attn_kv, weights,
                       config, logits_indices);
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

// NON-OWNING [rows, vocab] f32 view over a buffer the graph slot keeps alive
// (mirrors qwen3_5.cpp:5177 ViewDeviceLogits). Stream ordering guarantees the
// sampler's later reads see the replay's writes; the next same-size replay
// overwrites the buffer, so in-place sampler mutation is safe.
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
// from); reallocate only when the shape actually changed. (qwen3_5.cpp:5800.)
template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}

// Build the S-padded PURE-DECODE inputs from the real B-request step (B<=S). The
// ATTENTION-ONLY analog of qwen3_5.cpp's BuildPaddedDecode — Qwen3-Coder has no
// GDN metadata, so only CommonAttentionMetadata is padded.
//
// The decode forward is ROW-INDEPENDENT (paged attention is per-request causal;
// the router/grouped-MoE GEMM/norm/lm_head are per-token with no cross-row
// reduction — the grouped GEMM's counting sort groups (token,expert) PAIRS and
// each output row reduces only over its own K), so appending S-B INERT rows
// cannot perturb the real rows' logits. The padding rows are made inert exactly
// as vLLM's cudagraph padding:
//   * token id / position 0 (the embed row is discarded);
//   * slot_mapping = -1 -> ReshapeAndCache skips the KV write (cuda_cache.cu:50),
//     so no real KV block is touched;
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

// ─── Qwen3MoeDecodeGraph (BF16 full-attention-MoE decode CUDA-graph driver) ───
// The Qwen3-Coder sibling of Qwen3_5DecodeGraph (qwen3_5.cpp:5902) and
// Qwen3_5DenseDecodeGraph (qwen3_5.cpp:6104): SAME cold -> warm -> capture ->
// replay state machine, SAME padded-batch capture set (decode_graph_sizes.h,
// mirroring vLLM `_set_cudagraph_sizes` reduced to the full-decode-cudagraph
// regime + `cuda_graph.py`'s pad-to-nearest dispatch), SAME persistent
// fixed-address host inputs and persistent embed/logits buffers — driving the
// BF16 full-attention MoE forward (ForwardLayers over EmbedInto) with NO GDN.
//
// Ported from: vllm/v1/worker/gpu_model_runner.py::GPUModelRunner @ e24d1b24
//   (`_dummy_run` warm-up then capture, then graph dispatch per decode step) +
//   vllm/compilation/cuda_graph.py (`CUDAGraphWrapper.__call__`: pad the batch to
//   a captured size, replay, else run eager).
//
// WHY THIS IS THE C1 LEVER (spec §9 W6 residual 1): at concurrency 1 the summed
// decode GPU kernel time is ~31 ms against a 36.22 ms TPOT (~86% GPU-busy), i.e.
// ~5 ms/step of pure host/launch tax — essentially the entire 4.5 ms/step deficit
// vs GRAPHED vLLM. The kernels are already at 70-77% of GB10 memory peak, so the
// win is removing per-step launch overhead, not making kernels faster.
//
// GRAPH-SAFETY AUDIT of the bf16 decode path (capture requires stable pointers
// and no host sync / stream-ordered alloc inside the region):
//   * Embedding (device flag cudaMalloc + stream sync) stays OUTSIDE (EmbedInto).
//   * All device scratch comes from the shared DevicePool, whose blocks are
//     recycled (never returned to the driver) — the cold pre-warm step at this
//     exact size populates every size class the capture then reuses, so capture
//     itself performs no cudaMalloc.
//   * The grouped-MoE index scratch (`EnsureMoeScratch`) and split-K partials
//     (`EnsureMoePartials`) are graph-safe BY DESIGN (retire-don't-free, so a
//     later growth never dangles a baked pointer — cuda_matmul_nvfp4.cu:767,986).
//   * The per-layer expert device-pointer arrays + pair->token row map
//     (`MoeBf16Resident`) are uploaded ONCE at first touch and the token map is
//     memoized per T, both during the pre-warm step (qwen3_5.cpp:4436-4505).
//   * ResidentWeight uploads every weight once, on first touch (pre-warm).
//   * The FA-2 varlen-decode launcher's per-shape scratch throws if it misses
//     during capture (cuda_flash_attn_fa2.cu:706) — the pre-warm step at the same
//     padded size populates it. Its host `max_seq_len` only sizes the split-KV
//     grid; the per-request causal geometry is read from the DEVICE seq_lens, and
//     each split's KV range is derived in-kernel from `seqused_k`, so a captured
//     graph stays CORRECT as the sequences grow (identical contract to the
//     already-gated 35B/27B decode graphs).
//   * cuBLASLt's workspace is a one-time per-context cudaMalloc (cuda_matmul.cu:101).
struct Qwen3MoeDecodeGraph::Impl {
  Impl(const Qwen3MoeWeights& w, const HfConfig& c, vt::Queue q, int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    // The framework-wide graph switch, plus a Qwen3-Coder-local rollback for a
    // same-binary A/B of exactly this lever.
    const char* env = std::getenv("VLLM_CPP_CUDAGRAPH");
    const bool env_on = (env == nullptr) || std::string(env) != "0";
    const char* local = std::getenv("VT_QWEN3MOE_CUDAGRAPH");
    const bool local_on = (local == nullptr) || local[0] != '0';
    Backend& b = vt::GetBackend(queue.device.type);
    enabled = env_on && local_on &&
              platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
              b.SupportsGraphCapture();
  }
  ~Impl() {
    Backend& b = vt::GetBackend(queue.device.type);
    for (auto& kv : slots)
      if (kv.second.graph != nullptr) b.DestroyGraph(kv.second.graph);
  }

  // One captured padded batch size. Owns its OWN persistent host inputs (the
  // captured graph's host->device copies bake these addresses, so each size needs
  // its own fixed-address buffers), its persistent embed target + logits output,
  // and its instantiated graph.
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

  const Qwen3MoeWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;

  std::map<int64_t, SizeSlot> slots;  // padded size S -> slot
  int64_t replays = 0;                // total replays (diagnostics)
  bool any_captured = false;          // diagnostics: at least one live graph
};

Qwen3MoeDecodeGraph::Qwen3MoeDecodeGraph(const Qwen3MoeWeights& weights,
                                         const HfConfig& config, vt::Queue queue,
                                         int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

Qwen3MoeDecodeGraph::~Qwen3MoeDecodeGraph() = default;

bool Qwen3MoeDecodeGraph::captured() const { return impl_->any_captured; }
int64_t Qwen3MoeDecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits Qwen3MoeDecodeGraph::Step(
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

  // Pad this step's real B-request inputs up to S (inert padding rows), then
  // refresh THIS size's persistent host buffers in place.
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
                   "[Qwen3MoeDecodeGraph] captured bf16 MoE decode graph for "
                   "padded size S=%lld (real B=%lld)\n",
                   static_cast<long long>(S), static_cast<long long>(B));
    b.ReplayGraph(impl_->queue, s.graph);
    s.replays = 1;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool size classes, the
  // resident weights / MoE expert pointer arrays + token map, and the FA-2
  // per-shape scratch for this size) and defer capture to the next same-size
  // step. This is a real decode step — nothing is wasted.
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

}  // namespace vllm
