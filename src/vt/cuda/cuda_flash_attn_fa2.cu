// FlashAttention-2 paged prefill + pure-decode launcher (torch-free, sm_121a),
// ported from vllm-project/flash-attention @ 2c839c33742309ec41e620bf837495ec9926c56e:
//   csrc/flash_attn/flash_api.cpp:30-147,262-327,587-714,754-779.
// This is the exact FA2 source pinned by vLLM v0.25.0 target 702f481. The TU is
// a thin torch-free replacement for mha_varlen_fwd: it fills Flash_fwd_params
// from vt::Tensor views and calls the vendored CUTLASS split-KV instantiations.
//
// Interface matched to vLLM's flash_attn backend prefill call
// (vllm/v1/attention/backends/flash_attn.py -> flash_attn_varlen_func ->
// vllm_flash_attn/flash_attn_interface.py varlen path): head_dim 256, GQA
// (h_h_k_ratio), bf16 in / bf16 out, per-sequence causal, PAGED KV (block_table)
// + varlen (query_start_loc as cu_seqlens_q, seq_lens as seqused_k).
//
// Ragged prefill keeps num_splits=1: its packed varlen output cannot use the
// batched combine addressing. Pure decode is a separate upstream path. When
// max_seqlen_q==1 and Hq>Hkv, upstream reshapes [B,Hkv,G,D] to logical
// [B,G,Hkv,D], clears cu_seqlens_q, and applies num_splits_heuristic. We present
// that same logical view directly with the ABI's independent batch/row/head
// strides, avoiding upstream's materialized transpose and output copy. Split
// partials remain F32 and the vendored combine writes the original [B,Hq,D]
// layout through those strides.
//
// The launcher is SYNC-FREE on the production path: max_seqlen_q comes from
// PagedAttentionArgs::query_start_loc_host and max_seqlen_k from
// PagedAttentionArgs::max_seq_len (both host-known per step, set by
// FullAttnBlockPaged from CommonAttentionMetadata — the same device-resident-
// metadata pattern the WMMA prefill launchers use). The previous revision did a
// per-layer D2H copy + cudaStreamSynchronize here, re-introducing the exact
// pipeline drain the query_start_loc_host mechanism was built to remove (ops.h
// PagedAttentionArgs note: ~10-12 syncs/step, prefill only 43.7% GPU-busy);
// op-unit-test callers without host metadata still get the D2H+sync fallback.
//
// It is also CAST-FREE: bf16 query/out only. The previous revision accepted an
// f32 query/out and bridged with f32<->bf16 cast kernels + per-call
// cudaMallocAsync scratch — measured to ERASE the attention win end-to-end
// (.agents/parity-ledger.md 2026-07-06: "the f32<->bf16 CAST kernels (~35ms/run)
// ... ERASE the attention win"). The production 27B path now hands bf16
// natively (fused preamble emits bf16 q/k, attention output is bf16 into the
// sigmoid gate); anything else falls through to the WMMA ladder.
//
// VT_FA2_PREFILL, VT_FA2_DECODE (27B ratio-6) and VT_FA2_DECODE_35B (35B
// ratio-8) are independent runtime toggles. Decode covers the two hd-256 BF16
// GQA topologies of the two gate models — Qwen3.6-27B Hq/Hkv=24/4 and
// Qwen3.6-35B Hq/Hkv=16/2 — through the same vendored split-KV kernel; the
// group-swap presentation is generic in the GQA ratio. Its scratch is keyed by
// device+stream+capture-stable shape and never moved until queue teardown: the
// cold eager graph step allocates it, capture/replay only reuses the same
// pointers. Other ratios / windows / non-256 shapes remain on the fallback.
#ifdef VLLM_CPP_FLASH_ATTN

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cutlass/numeric_types.h>  // cutlass::bfloat16_t

// Torch-free FA-2 params struct + the splitkv dispatch *declaration* (the
// definition lives in the vendored flash_attn/src/*.cu instantiations, linked in).
#include "namespace_config.h"
#include "flash.h"

#include "vt/cuda/cuda_flash_attn_fa2_internal.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cuda flash-attn-2: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

inline int RoundMultiple(int x, int m) { return (x + m - 1) / m * m; }

// Exact port of flash_api.cpp::num_splits_heuristic @ FA2 2c839c33. Upstream
// passes 2*numSM because this kernel uses 128 threads per CTA.
int NumSplitsHeuristic(int batch_nheads_mblocks, int num_sms, int num_n_blocks,
                       int max_splits) {
  if (batch_nheads_mblocks >= 0.8F * static_cast<float>(num_sms)) return 1;
  max_splits = std::min({max_splits, num_sms, num_n_blocks});
  if (max_splits < 1) return 1;

  const auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  const auto eligible = [&](int splits) {
    return splits == 1 ||
           ceildiv(num_n_blocks, splits) != ceildiv(num_n_blocks, splits - 1);
  };
  float max_efficiency = 0.0F;
  std::vector<float> efficiency;
  efficiency.reserve(static_cast<size_t>(max_splits));
  for (int splits = 1; splits <= max_splits; ++splits) {
    if (!eligible(splits)) {
      efficiency.push_back(0.0F);
      continue;
    }
    const float waves = static_cast<float>(batch_nheads_mblocks * splits) /
                        static_cast<float>(num_sms);
    const float value = waves / std::ceil(waves);
    max_efficiency = std::max(max_efficiency, value);
    efficiency.push_back(value);
  }
  for (int splits = 1; splits <= max_splits; ++splits) {
    if (eligible(splits) &&
        efficiency[static_cast<size_t>(splits - 1)] >= 0.85F * max_efficiency) {
      return splits;
    }
  }
  return 1;
}

struct ScratchBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;
};

struct DecodeShapeKey {
  int batch = 0;
  int query_heads = 0;
  int kv_heads = 0;
  int groups = 0;
  int head_dim = 0;
  int max_blocks = 0;
  int block_size = 0;
  int num_splits = 0;

  bool operator==(const DecodeShapeKey& other) const {
    return batch == other.batch && query_heads == other.query_heads &&
           kv_heads == other.kv_heads && groups == other.groups &&
           head_dim == other.head_dim && max_blocks == other.max_blocks &&
           block_size == other.block_size && num_splits == other.num_splits;
  }
};

void HashCombine(size_t& seed, int value) {
  const size_t h = std::hash<int>{}(value);
  seed ^= h + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}

struct DecodeShapeKeyHash {
  size_t operator()(const DecodeShapeKey& key) const {
    size_t h = 0;
    HashCombine(h, key.batch);
    HashCombine(h, key.query_heads);
    HashCombine(h, key.kv_heads);
    HashCombine(h, key.groups);
    HashCombine(h, key.head_dim);
    HashCombine(h, key.max_blocks);
    HashCombine(h, key.block_size);
    HashCombine(h, key.num_splits);
    return h;
  }
};

struct DecodeScratch {
  ScratchBuffer softmax_lse;
  ScratchBuffer softmax_lse_accum;
  ScratchBuffer out_accum;
};

struct Fa2StreamScratch {
  // Serializes host submissions that share scratch on one stream. CUDA stream
  // ordering makes reuse safe after submission returns.
  std::mutex submit_mu;
  ScratchBuffer prefill_lse;
  std::unordered_map<DecodeShapeKey, DecodeScratch, DecodeShapeKeyHash> decode;
  int num_sms = 0;
};

struct Fa2StreamKey {
  int device = 0;
  cudaStream_t stream = nullptr;

  bool operator==(const Fa2StreamKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct Fa2StreamKeyHash {
  size_t operator()(const Fa2StreamKey& key) const {
    size_t h = std::hash<int>{}(key.device);
    const size_t sh = std::hash<void*>{}(static_cast<void*>(key.stream));
    h ^= sh + 0x9e3779b9U + (h << 6U) + (h >> 2U);
    return h;
  }
};

using Fa2ScratchMap =
    std::unordered_map<Fa2StreamKey, std::shared_ptr<Fa2StreamScratch>, Fa2StreamKeyHash>;

Fa2ScratchMap& Fa2ScratchPool() {
  static Fa2ScratchMap pool;
  return pool;
}

std::mutex& Fa2ScratchPoolMutex() {
  static std::mutex mu;
  return mu;
}

std::shared_ptr<Fa2StreamScratch> Fa2ScratchFor(int device, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(Fa2ScratchPoolMutex());
  auto& entry = Fa2ScratchPool()[Fa2StreamKey{device, stream}];
  if (!entry) {
    entry = std::make_shared<Fa2StreamScratch>();
    Check(cudaDeviceGetAttribute(&entry->num_sms, cudaDevAttrMultiProcessorCount, device),
          "query SM count");
  }
  return entry;
}

void* EnsureGrowOnly(ScratchBuffer& buffer, size_t bytes, cudaStream_t stream,
                     const char* what) {
  if (bytes <= buffer.bytes) return buffer.ptr;
  void* replacement = nullptr;
  Check(cudaMallocAsync(&replacement, bytes, stream), what);
  void* old = buffer.ptr;
  buffer.ptr = replacement;
  buffer.bytes = bytes;
  if (old != nullptr) Check(cudaFreeAsync(old, stream), "grow scratch free");
  return buffer.ptr;
}

void AllocateFixed(ScratchBuffer& buffer, size_t bytes, cudaStream_t stream,
                   const char* what) {
  if (bytes == 0) return;
  Check(cudaMallocAsync(&buffer.ptr, bytes, stream), what);
  buffer.bytes = bytes;
}

struct Fa2DebugCounters {
  std::atomic<bool> enabled{false};
  std::atomic<uint64_t> decode_launches{0};
  std::atomic<uint64_t> split_launches{0};
  std::atomic<uint64_t> no_split_launches{0};
  std::atomic<uint64_t> scratch_allocations{0};
  std::atomic<uint64_t> scratch_reuses{0};
};

Fa2DebugCounters& DebugCounters() {
  static Fa2DebugCounters counters;
  return counters;
}

void RecordDecodeLaunch(bool split, bool allocated) {
  Fa2DebugCounters& counters = DebugCounters();
  if (!counters.enabled.load(std::memory_order_relaxed)) return;
  counters.decode_launches.fetch_add(1, std::memory_order_relaxed);
  (split ? counters.split_launches : counters.no_split_launches)
      .fetch_add(1, std::memory_order_relaxed);
  (allocated ? counters.scratch_allocations : counters.scratch_reuses)
      .fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

// Launch FA-2 paged varlen prefill for a bf16 query + bf16 KV cache + bf16 out,
// head_dim 256. Layouts (elements, matching our LaunchPaged dispatch):
//   query [total_q, hq, d]                          (varlen-packed, bf16)
//   k/v   [num_blocks, block_size, num_kv_heads, d] (paged, NHD block layout, bf16)
//   out   [total_q, hq, d]                          (varlen-packed, bf16)
//   block_table [num_reqs, max_blocks] i32, seq_lens [num_reqs] i32 (K lengths
//   incl. context), query_start_loc [num_reqs+1] i32 (cumulative q offsets).
void LaunchPrefillFA2Bf16(cudaStream_t s, Tensor& out, const Tensor& query,
                          const Tensor& k_cache, const Tensor& v_cache,
                          const Tensor& block_table, const Tensor& seq_lens,
                          const Tensor& query_start_loc, const PagedAttentionArgs& args,
                          int64_t hq, int64_t d, int64_t num_reqs, int64_t num_kv_heads,
                          int64_t block_size) {
  const int64_t total_q = query.shape[0];
  if (total_q == 0 || num_reqs == 0 || hq == 0 || d == 0) return;
  if (query.dtype != DType::kBF16 || out.dtype != DType::kBF16) {
    throw std::runtime_error(
        "cuda flash-attn-2 prefill: bf16 query/out required (dispatch gate must enforce)");
  }

  // FA-2 needs max_seqlen_q/max_seqlen_k on the host — for GRID SIZING and the
  // rounded dims only; the per-request causal geometry reads the DEVICE
  // cu_seqlens_q/seqused_k via BlockInfo, so host UPPER BOUNDS are safe.
  // Production callers hand both through PagedAttentionArgs
  // (query_start_loc_host + max_seq_len, built once per step); callers without
  // them (op unit tests) fall back to a small D2H + sync, same as the WMMA
  // launchers' fallback.
  int max_seqlen_q = 0;
  int max_seqlen_k = args.max_seq_len;
  if (args.query_start_loc_host != nullptr) {
    for (int64_t i = 0; i < num_reqs; ++i) {
      max_seqlen_q = std::max(
          max_seqlen_q, args.query_start_loc_host[i + 1] - args.query_start_loc_host[i]);
    }
  }
  if (max_seqlen_q <= 0 || max_seqlen_k <= 0) {
    std::vector<int32_t> qsl(static_cast<size_t>(num_reqs) + 1);
    std::vector<int32_t> sk(static_cast<size_t>(num_reqs));
    Check(cudaMemcpyAsync(qsl.data(), query_start_loc.Ptr<int32_t>(),
                          sizeof(int32_t) * (num_reqs + 1), cudaMemcpyDeviceToHost, s),
          "query_start_loc D2H");
    Check(cudaMemcpyAsync(sk.data(), seq_lens.Ptr<int32_t>(), sizeof(int32_t) * num_reqs,
                          cudaMemcpyDeviceToHost, s),
          "seq_lens D2H");
    Check(cudaStreamSynchronize(s), "seqlen sync");
    max_seqlen_q = 0;
    max_seqlen_k = 0;
    for (int64_t i = 0; i < num_reqs; ++i) {
      max_seqlen_q = std::max(max_seqlen_q, qsl[i + 1] - qsl[i]);
      max_seqlen_k = std::max(max_seqlen_k, sk[i]);
    }
  }
  if (max_seqlen_q == 0 || max_seqlen_k == 0) return;

  // softmax_lse [nheads, total_q] f32 (unpadded/varlen LSE). Written by the
  // kernel; not consumed. The per-stream grow-only buffer mirrors the caching
  // allocator upstream receives from PyTorch and is released with the queue.
  const auto scratch = Fa2ScratchFor(query.device.index, s);
  std::lock_guard<std::mutex> submit_lock(scratch->submit_mu);
  float* softmax_lse = static_cast<float*>(EnsureGrowOnly(
      scratch->prefill_lse,
      static_cast<size_t>(hq) * static_cast<size_t>(total_q) * sizeof(float), s,
      "prefill softmax_lse alloc"));

  FLASH_NAMESPACE::Flash_fwd_params p{};  // zero-init: nulls knew/rotary/alibi/accum/leftpad
  p.is_bf16 = true;

  p.q_ptr = query.data;
  p.k_ptr = k_cache.data;
  p.v_ptr = v_cache.data;
  p.o_ptr = out.data;

  // Strides in ELEMENTS. Varlen q/o: no batch stride (cu_seqlens_q drives offset).
  p.q_row_stride = query.stride[0];
  p.q_head_stride = query.stride[1];
  p.o_row_stride = out.stride[0];
  p.o_head_stride = out.stride[1];
  // Paged k/v [num_blocks, block_size, num_kv_heads, d]:
  //   batch_stride = per-block, row_stride = per page-position, head_stride = per head.
  p.k_batch_stride = k_cache.stride[0];
  p.k_row_stride = k_cache.stride[1];
  p.k_head_stride = k_cache.stride[2];
  p.v_batch_stride = v_cache.stride[0];
  p.v_row_stride = v_cache.stride[1];
  p.v_head_stride = v_cache.stride[2];

  p.cu_seqlens_q = query_start_loc.Ptr<int32_t>();
  p.cu_seqlens_k = nullptr;               // paged: block_table + seqused_k drive K
  p.seqused_k = seq_lens.Ptr<int32_t>();  // per-request K length (BlockInfo::actual_seqlen_k)
  p.softmax_lse_ptr = softmax_lse;

  p.b = static_cast<int>(num_reqs);
  p.h = static_cast<int>(hq);
  p.h_k = static_cast<int>(num_kv_heads);
  p.h_h_k_ratio = static_cast<int>(hq / num_kv_heads);
  p.seqlen_q = max_seqlen_q;
  p.seqlen_k = max_seqlen_k;
  p.seqlen_q_rounded = RoundMultiple(max_seqlen_q, 128);
  p.seqlen_k_rounded = RoundMultiple(max_seqlen_k, 128);
  p.d = static_cast<int>(d);
  p.d_rounded = RoundMultiple(static_cast<int>(d), 32);
  p.total_q = static_cast<int>(total_q);

  p.scale_softmax = args.scale;
  p.scale_softmax_log2 = args.scale * static_cast<float>(M_LOG2E);
  p.softcap = 0.0f;

  // Dropout disabled (inference): keep-prob 1.0.
  p.p_dropout = 1.0f;
  p.p_dropout_in_uint8_t = uint8_t(255);
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = args.scale;
  p.philox_args = at::PhiloxCudaState(0, 0);

  // The pinned FA-2 API normalizes a finite window to the LOCAL specialization,
  // even for decoder attention: local (W-1,0) already contains the causal right
  // bound, while the compile-time Is_causal specialization deliberately ignores
  // window_size_left (flash_fwd_launch_template.h LOCAL_SWITCH). Because this
  // torch-free adapter bypasses flash_api.cpp, reproduce that normalization
  // here before selecting the explicit template instantiation.
  const bool is_local = args.window_size.has_value();
  p.is_causal = args.causal && !is_local;
  p.window_size_left = is_local ? args.window_size->left : -1;
  p.window_size_right =
      is_local ? args.window_size->right : (args.causal ? 0 : -1);
  p.is_seqlens_k_cumulative = true;  // ignored while cu_seqlens_k == nullptr
  p.is_rotary_interleaved = false;
  p.rotary_dim = 0;

  p.block_table = block_table.Ptr<int32_t>();
  p.block_table_batch_stride = block_table.stride[0];
  p.page_block_size = static_cast<int>(block_size);

  p.unpadded_lse = true;  // LSE is [nheads, total_q]
  p.seqlenq_ngroups_swapped = false;

  // ALWAYS 1 on the varlen prefill path (see header): Split=false, the main
  // kernel writes O directly through cu_seqlens_q. o_batch_stride is unused on
  // this path (varlen q_offset ignores it) but set consistently anyway.
  p.num_splits = 1;
  p.o_batch_stride = static_cast<int64_t>(max_seqlen_q) * p.o_row_stride;

  // head_dim 256 is gated by the caller. Causal is per-layer (qwen3.5 mixes full-
  // attn causal / GDN); a finite decoder or encoder window dispatches the
  // non-causal template so its runtime LOCAL_SWITCH selects the exact local
  // mask. Both instantiations are compiled for sm_121a.
  // With num_splits==1 run_mha_fwd_splitkv_dispatch runs the Split=false kernel
  // only — no combine pass, exactly vLLM's varlen prefill.
  if (args.causal && !is_local) {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(p, s);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(p, s);
  }
  Check(cudaGetLastError(), "splitkv dispatch launch");
}

// Launch the pinned FA2 pure-decode optimization: bf16 paged KV, D256, one
// query per request, global causal decoder attention, for either Hq/Hkv=24/4
// (G=6, 27B) or Hq/Hkv=16/2 (G=8, 35B). flash_api.cpp first normalizes
// max_seqlen_q==1 to non-causal, then swaps query groups into the sequence
// dimension and applies split-KV. Independent strides let us expose that
// logical swap without copies:
//   physical [B,Hkv,G,D] address = b*Hq*D + kv*G*D + g*D + d
//   logical  [B,G,Hkv,D] strides = {Hq*D, D, G*D, 1}.
void LaunchDecodeFA2Bf16(cudaStream_t stream, Tensor& out, const Tensor& query,
                         const Tensor& k_cache, const Tensor& v_cache,
                         const Tensor& block_table, const Tensor& seq_lens,
                         const PagedAttentionArgs& args, int64_t hq, int64_t d,
                         int64_t num_reqs, int64_t num_kv_heads,
                         int64_t block_size) {
  if (query.shape[0] == 0) return;
  const int64_t groups = hq / num_kv_heads;
  // Two supported topologies, both hd-256 BF16 paged global-causal pure decode:
  //   * 27B ratio-6 Hq/Hkv=24/4 (W3-G);
  //   * 35B ratio-8 Hq/Hkv=16/2 (CLAIM-35B-FA2-DECODE-1).
  // The split-KV group-swap below is generic in groups/heads, so the body is
  // shared; only this admission and the caller gates encode the ratio.
  const bool ratio6 = hq == 24 && num_kv_heads == 4 && groups == 6;
  const bool ratio8 = hq == 16 && num_kv_heads == 2 && groups == 8;
  if (query.dtype != DType::kBF16 || k_cache.dtype != DType::kBF16 ||
      v_cache.dtype != DType::kBF16 || out.dtype != DType::kBF16 ||
      query.shape[0] != num_reqs || !(ratio6 || ratio8) || d != 256 ||
      block_size % 16 != 0 || !args.causal || args.window_size.has_value()) {
    throw std::runtime_error(
        "cuda flash-attn-2 decode: dispatch called outside ratio-6/ratio-8 BF16/D256 eligibility");
  }

  const int batch = static_cast<int>(num_reqs);
  const int heads = static_cast<int>(num_kv_heads);
  const int query_groups = static_cast<int>(groups);
  const int head_dim = static_cast<int>(d);
  const int max_blocks = static_cast<int>(block_table.shape[1]);
  const int page_size = static_cast<int>(block_size);
  const int max_seqlen_k = max_blocks * page_size;
  if (max_blocks <= 0 || max_seqlen_k <= 0) return;

  const auto stream_scratch = Fa2ScratchFor(query.device.index, stream);
  std::lock_guard<std::mutex> submit_lock(stream_scratch->submit_mu);

  // flash_api.cpp::set_params_splitkv: block-N is 64 for D256, block-M is 64,
  // and the heuristic receives 2*numSM for its 128-thread CTA occupancy model.
  constexpr int kBlockN = 64;
  constexpr int kBlockM = 64;
  const int num_n_blocks = (max_seqlen_k + kBlockN - 1) / kBlockN;
  const int num_m_blocks = (query_groups + kBlockM - 1) / kBlockM;
  const int num_splits = NumSplitsHeuristic(
      batch * heads * num_m_blocks, stream_scratch->num_sms * 2, num_n_blocks, 128);

  const DecodeShapeKey key{batch,       static_cast<int>(hq), heads,
                           query_groups, head_dim,              max_blocks,
                           page_size,   num_splits};
  auto [it, inserted] = stream_scratch->decode.try_emplace(key);
  DecodeScratch& scratch = it->second;
  if (inserted) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    Check(cudaStreamIsCapturing(stream, &capture_status), "decode capture-status query");
    if (capture_status != cudaStreamCaptureStatusNone) {
      stream_scratch->decode.erase(it);
      throw std::runtime_error(
          "cuda flash-attn-2 decode: scratch miss during CUDA graph capture");
    }

    const size_t rows = static_cast<size_t>(batch) * static_cast<size_t>(heads) *
                        static_cast<size_t>(query_groups);
    try {
      AllocateFixed(scratch.softmax_lse, rows * sizeof(float), stream,
                    "decode softmax_lse alloc");
      if (num_splits > 1) {
        const size_t split_rows = static_cast<size_t>(num_splits) * rows;
        const int head_dim_rounded = RoundMultiple(head_dim, 64);
        AllocateFixed(scratch.softmax_lse_accum, split_rows * sizeof(float), stream,
                      "decode partial LSE alloc");
        AllocateFixed(scratch.out_accum,
                      split_rows * static_cast<size_t>(head_dim_rounded) * sizeof(float),
                      stream, "decode partial output alloc");
      }
    } catch (...) {
      // Preserve stream ordering while making a failed partial construction
      // reusable only through a clean retry.
      if (scratch.softmax_lse.ptr != nullptr)
        cudaFreeAsync(scratch.softmax_lse.ptr, stream);
      if (scratch.softmax_lse_accum.ptr != nullptr)
        cudaFreeAsync(scratch.softmax_lse_accum.ptr, stream);
      if (scratch.out_accum.ptr != nullptr)
        cudaFreeAsync(scratch.out_accum.ptr, stream);
      stream_scratch->decode.erase(it);
      throw;
    }
  }

  FLASH_NAMESPACE::Flash_fwd_params p{};
  p.is_bf16 = true;
  p.q_ptr = query.data;
  p.k_ptr = k_cache.data;
  p.v_ptr = v_cache.data;
  p.o_ptr = out.data;

  p.q_batch_stride = query.stride[0];
  p.q_row_stride = query.stride[1];
  p.q_head_stride = query_groups * query.stride[1];
  p.o_batch_stride = out.stride[0];
  p.o_row_stride = out.stride[1];
  p.o_head_stride = query_groups * out.stride[1];
  p.k_batch_stride = k_cache.stride[0];
  p.k_row_stride = k_cache.stride[1];
  p.k_head_stride = k_cache.stride[2];
  p.v_batch_stride = v_cache.stride[0];
  p.v_row_stride = v_cache.stride[1];
  p.v_head_stride = v_cache.stride[2];

  p.cu_seqlens_q = nullptr;
  p.cu_seqlens_k = nullptr;
  p.seqused_k = seq_lens.Ptr<int32_t>();
  p.softmax_lse_ptr = scratch.softmax_lse.ptr;
  p.softmax_lseaccum_ptr = scratch.softmax_lse_accum.ptr;
  p.oaccum_ptr = scratch.out_accum.ptr;

  p.b = batch;
  p.h = heads;
  p.h_k = heads;
  p.h_h_k_ratio = 1;
  p.seqlen_q = query_groups;
  p.seqlen_k = max_seqlen_k;
  p.seqlen_q_rounded = RoundMultiple(query_groups, 128);
  p.seqlen_k_rounded = RoundMultiple(max_seqlen_k, 128);
  p.d = head_dim;
  p.d_rounded = RoundMultiple(head_dim, 64);
  p.total_q = batch * query_groups;

  p.scale_softmax = args.scale;
  p.scale_softmax_log2 = args.scale * static_cast<float>(M_LOG2E);
  p.softcap = 0.0F;
  p.p_dropout = 1.0F;
  p.p_dropout_in_uint8_t = uint8_t(255);
  p.rp_dropout = 1.0F;
  p.scale_softmax_rp_dropout = args.scale;
  p.philox_args = at::PhiloxCudaState(0, 0);

  // Upstream changes max_seqlen_q==1 causal decode to non-causal before the
  // group swap. Each logical group row is the same time position, so this is
  // semantically identical and selects the exact non-causal instantiation.
  p.is_causal = false;
  p.window_size_left = -1;
  p.window_size_right = -1;
  p.is_seqlens_k_cumulative = true;
  p.is_rotary_interleaved = false;
  p.rotary_dim = 0;

  p.block_table = block_table.Ptr<int32_t>();
  p.block_table_batch_stride = block_table.stride[0];
  p.page_block_size = page_size;
  p.unpadded_lse = true;
  p.seqlenq_ngroups_swapped = true;
  p.num_splits = num_splits;

  RecordDecodeLaunch(num_splits > 1, inserted);
  FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(p,
                                                                                stream);
  Check(cudaGetLastError(), "decode splitkv dispatch launch");
}

void ReleaseFa2Scratch(int device, void* stream_handle) {
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  std::shared_ptr<Fa2StreamScratch> scratch;
  {
    std::lock_guard<std::mutex> pool_lock(Fa2ScratchPoolMutex());
    auto& pool = Fa2ScratchPool();
    const auto it = pool.find(Fa2StreamKey{device, stream});
    if (it == pool.end()) return;
    scratch = std::move(it->second);
    pool.erase(it);
  }

  std::lock_guard<std::mutex> submit_lock(scratch->submit_mu);
  const auto release = [&](ScratchBuffer& buffer) {
    if (buffer.ptr == nullptr) return;
    Check(cudaFreeAsync(buffer.ptr, stream), "queue-destroy scratch free");
    buffer.ptr = nullptr;
    buffer.bytes = 0;
  };
  release(scratch->prefill_lse);
  for (auto& entry : scratch->decode) {
    release(entry.second.softmax_lse);
    release(entry.second.softmax_lse_accum);
    release(entry.second.out_accum);
  }
  scratch->decode.clear();
}

namespace testing {

int Fa2DecodeNumSplitsForTesting(int batch_nheads_mblocks, int num_sms,
                                 int num_n_blocks, int max_splits) {
  return NumSplitsHeuristic(batch_nheads_mblocks, num_sms, num_n_blocks,
                            max_splits);
}

void ResetFa2DecodeDebugCounters() {
  Fa2DebugCounters& counters = DebugCounters();
  counters.decode_launches.store(0, std::memory_order_relaxed);
  counters.split_launches.store(0, std::memory_order_relaxed);
  counters.no_split_launches.store(0, std::memory_order_relaxed);
  counters.scratch_allocations.store(0, std::memory_order_relaxed);
  counters.scratch_reuses.store(0, std::memory_order_relaxed);
  counters.enabled.store(true, std::memory_order_release);
}

void DisableFa2DecodeDebugCounters() {
  DebugCounters().enabled.store(false, std::memory_order_release);
}

uint64_t Fa2DecodeLaunchesForTesting() {
  return DebugCounters().decode_launches.load(std::memory_order_relaxed);
}

uint64_t Fa2DecodeSplitLaunchesForTesting() {
  return DebugCounters().split_launches.load(std::memory_order_relaxed);
}

uint64_t Fa2DecodeNoSplitLaunchesForTesting() {
  return DebugCounters().no_split_launches.load(std::memory_order_relaxed);
}

uint64_t Fa2DecodeScratchAllocationsForTesting() {
  return DebugCounters().scratch_allocations.load(std::memory_order_relaxed);
}

uint64_t Fa2DecodeScratchReusesForTesting() {
  return DebugCounters().scratch_reuses.load(std::memory_order_relaxed);
}

size_t Fa2DecodeScratchShapeCountForTesting(int device, void* stream_handle) {
  std::shared_ptr<Fa2StreamScratch> scratch;
  {
    std::lock_guard<std::mutex> pool_lock(Fa2ScratchPoolMutex());
    const auto it = Fa2ScratchPool().find(
        Fa2StreamKey{device, static_cast<cudaStream_t>(stream_handle)});
    if (it == Fa2ScratchPool().end()) return 0;
    scratch = it->second;
  }
  std::lock_guard<std::mutex> submit_lock(scratch->submit_mu);
  return scratch->decode.size();
}

}  // namespace testing

}  // namespace vt::cuda

#endif  // VLLM_CPP_FLASH_ATTN
