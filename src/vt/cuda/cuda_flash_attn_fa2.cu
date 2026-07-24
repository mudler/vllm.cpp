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
  // VARLEN d128 decode (Qwen3-dense) scratch — separate map so it can never
  // alias the d256 group-swap decode arms above (byte-identical d256 required).
  std::unordered_map<DecodeShapeKey, DecodeScratch, DecodeShapeKeyHash> varlen_decode;
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
// head_dim 256 (gate models) or 128 (Qwen3-dense — vLLM runs its prefill on the
// SAME flash_attn_varlen_func FA2 family). Layouts (elements, matching our
// LaunchPaged dispatch):
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
  // head_dim 256 (gate models Qwen3.6-27B/35B) and head_dim 128 (Qwen3-dense —
  // vLLM runs its prefill on the SAME flash_attn_varlen_func FA2 family) share
  // this launcher; both split-KV instantiations are vendored + compiled. The
  // d128 arm NEVER engages the d256 gate models (they are d256 by construction),
  // so the d256 arm is byte-identical.
  if (d != 128 && d != 256) {
    throw std::runtime_error(
        "cuda flash-attn-2 prefill: head_dim 128 or 256 only (dispatch gate must enforce)");
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

  // head_dim {128,256} is gated by the caller. Causal is per-layer (qwen3.5 mixes
  // full-attn causal / GDN); a finite decoder or encoder window dispatches the
  // non-causal template so its runtime LOCAL_SWITCH selects the exact local
  // mask. All instantiations are compiled for sm_121a.
  // With num_splits==1 run_mha_fwd_splitkv_dispatch runs the Split=false kernel
  // only — no combine pass, exactly vLLM's varlen prefill.
  const bool run_causal = args.causal && !is_local;
  if (d == 128) {
    if (run_causal) {
      FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(p, s);
    } else {
      FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, false>(p, s);
    }
  } else {
    if (run_causal) {
      FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(p, s);
    } else {
      FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(p, s);
    }
  }
  Check(cudaGetLastError(), "splitkv dispatch launch");
}

// ─── MLA PREFILL (MLA campaign W5) ──────────────────────────────────────────
// Launch FA-2 over CONTIGUOUS varlen q/k/v at head_dim 192 — the MLA prefill
// call `FlashAttnPrefillBackend` makes on GB10
// (vllm/v1/attention/backends/mla/prefill/flash_attn.py:182-188 @ e24d1b24,
// reached from `:205 run_prefill_new_tokens` causal / `:229
// run_prefill_context_chunk` non-causal). OBSERVED, not inferred: the oracle
// logs `Using FLASH_ATTN MLA prefill backend` on sm_121, and the MLA prefill
// selector gives major != 10 that single entry (`prefill/selector.py:66-76`).
//
// THIS IS A SEPARATE ENTRY POINT ON PURPOSE. `LaunchPrefillFA2Bf16` above is the
// PAGED launcher (block_table + seqused_k, 4-D k/v caches, head_dim {128,256});
// nothing in it is touched by this addition, so the 27B / 35B / Qwen3-dense
// prefill paths are byte-identical by construction rather than by measurement.
// What the two share is the vendored kernel template, which already supports the
// contiguous varlen mode (`block_info.h` k_offset via `sum_s_k` when
// `block_table == nullptr`, `flash_fwd_kernel.h:584-590`) and every head dim
// upstream FA-2 supports; W5 added only the two head_dim-192 explicit
// instantiations.
//
//   query/out    [total_q, h, d]  (out at the SAME d — the caller pads V and
//                                  slices the output back, mirroring
//                                  `_flash_attn_varlen_diff_headdims`)
//   k/v          [total_k, h, d]  contiguous varlen, cu_seqlens_k-addressed
//   lse_out      [h, total_q] f32 or nullptr
//
// LSE LAYOUT — a deliberate deviation from upstream's call, forced by an upstream
// FA-2 quirk. With `unpadded_lse = true` the MAIN path writes LSE at the
// unpadded offset (`flash_fwd_kernel.h:1038-1041`) but the EMPTY-K EARLY EXIT
// (`:1039` in the `n_block_min >= n_block_max` branch, `:1030-1043`) ignores
// `unpadded_lse` and always writes at the PADDED
// `((split*b + bidb)*h + bidh)*seqlen_q` offset. A request with ZERO keys in a
// chunk is REACHABLE in the chunked-context loop — it is exactly the case
// `merge_attn_states.cu:100-106` documents — so that mixed layout would both
// clobber valid rows and write past an `[h, total_q]` buffer. We therefore run
// the kernel with `unpadded_lse = false` (BOTH paths then use the padded layout,
// consistently and in-bounds) into a `[b, h, max_seqlen_q]` scratch, and convert
// to the caller's unpadded `[h, total_q]` afterwards. The conversion also
// normalizes the `+INFINITY` FA-2 writes for an empty-K row (`:573`) to `-inf`,
// which is the value our CPU reference produces and the value
// `merge_attn_states.cu:97-98` normalizes it to anyway.
__global__ void Fa2UnpadLseKernel(float* __restrict__ dst, const float* __restrict__ src,
                                  const int32_t* __restrict__ cu_seqlens_q, int num_reqs,
                                  int num_heads, int max_seqlen_q, int total_q,
                                  int64_t dst_row_stride) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total = num_reqs * num_heads * max_seqlen_q;
  if (idx >= total) return;
  const int iq = idx % max_seqlen_q;
  const int h = (idx / max_seqlen_q) % num_heads;
  const int b = idx / (max_seqlen_q * num_heads);
  const int q_begin = cu_seqlens_q[b];
  const int len_q = cu_seqlens_q[b + 1] - q_begin;
  if (iq >= len_q) return;
  const int t = q_begin + iq;
  if (t >= total_q) return;
  const float v = src[(static_cast<int64_t>(b) * num_heads + h) * max_seqlen_q + iq];
  dst[static_cast<int64_t>(h) * dst_row_stride + t] =
      isinf(v) ? -INFINITY : v;
}

void LaunchMlaPrefillFA2Bf16(cudaStream_t s, Tensor& out, float* lse_out,
                             int64_t lse_row_stride, const Tensor& query, const Tensor& key,
                             const Tensor& value, const Tensor& cu_seqlens_q,
                             const Tensor& cu_seqlens_k,
                             const MlaPrefillAttentionArgs& args, int64_t num_reqs,
                             int64_t hq, int64_t d) {
  const int64_t total_q = query.shape[0];
  const int64_t total_k = key.shape[0];
  if (total_q == 0 || num_reqs == 0 || hq == 0 || d == 0) return;
  if (query.dtype != DType::kBF16 || out.dtype != DType::kBF16 ||
      key.dtype != DType::kBF16 || value.dtype != DType::kBF16) {
    throw std::runtime_error(
        "cuda flash-attn-2 MLA prefill: bf16 query/key/value/out required "
        "(dispatch gate must enforce)");
  }
  // The head_dims the MLA prefill path needs, one per q_lora family:
  //   * DeepSeek-V2/V3: qk_nope 128 + qk_rope 64 = 192, v_head_dim 128
  //     (V zero-padded 128 -> 192 by the caller);
  //   * GLM-4.7-Flash (`Glm4MoeLite`): qk_nope 192 + qk_rope 64 = 256, v_head_dim
  //     256 (the caller's V pad is then a no-op, since v_head_dim == d already).
  // V arrives already zero-padded to the SAME width by the caller, mirroring
  // `requires_v_padding` (flash_attn.py:88-99 — TRUE on GB10 because FA3-on-SM90
  // and FA4 are the only exemptions). Both split-KV instantiations are already
  // compiled (flash_fwd_split_hdim{192,256}_bf16{,_causal}_sm80.cu), so this is a
  // dispatch addition only — the 192 path is byte-identical.
  if (d != 192 && d != 256) {
    throw std::runtime_error(
        "cuda flash-attn-2 MLA prefill: head_dim 192 or 256 only (dispatch gate must enforce)");
  }

  // max_seqlen_q / max_seqlen_k: host UPPER BOUNDS are safe for grid sizing and
  // the rounded dims (the per-request geometry reads the DEVICE cu_seqlens via
  // BlockInfo). max_seqlen_q additionally sizes the padded LSE scratch, where an
  // upper bound only over-allocates.
  int max_seqlen_q = args.max_seqlen_q;
  int max_seqlen_k = args.max_seqlen_k;
  if (max_seqlen_q <= 0 || max_seqlen_k <= 0) {
    std::vector<int32_t> qsl(static_cast<size_t>(num_reqs) + 1);
    std::vector<int32_t> ksl(static_cast<size_t>(num_reqs) + 1);
    Check(cudaMemcpyAsync(qsl.data(), cu_seqlens_q.Ptr<int32_t>(),
                          sizeof(int32_t) * (num_reqs + 1), cudaMemcpyDeviceToHost, s),
          "cu_seqlens_q D2H");
    Check(cudaMemcpyAsync(ksl.data(), cu_seqlens_k.Ptr<int32_t>(),
                          sizeof(int32_t) * (num_reqs + 1), cudaMemcpyDeviceToHost, s),
          "cu_seqlens_k D2H");
    Check(cudaStreamSynchronize(s), "mla prefill seqlen sync");
    max_seqlen_q = 0;
    max_seqlen_k = 0;
    for (int64_t i = 0; i < num_reqs; ++i) {
      max_seqlen_q = std::max(max_seqlen_q, qsl[i + 1] - qsl[i]);
      max_seqlen_k = std::max(max_seqlen_k, ksl[i + 1] - ksl[i]);
    }
  }
  if (max_seqlen_q == 0) return;
  // Every request may legitimately have ZERO keys in a context chunk; FA-2's
  // early-exit writes zeros to O and +INFINITY to LSE, which is exactly what the
  // merge expects. Guard only against a degenerate rounded dim.
  if (max_seqlen_k == 0) max_seqlen_k = 1;

  const auto scratch = Fa2ScratchFor(query.device.index, s);
  std::lock_guard<std::mutex> submit_lock(scratch->submit_mu);
  const size_t lse_elems =
      static_cast<size_t>(num_reqs) * static_cast<size_t>(hq) * static_cast<size_t>(max_seqlen_q);
  float* softmax_lse = static_cast<float*>(
      EnsureGrowOnly(scratch->prefill_lse, lse_elems * sizeof(float), s,
                     "mla prefill softmax_lse alloc"));

  FLASH_NAMESPACE::Flash_fwd_params p{};  // zero-init: nulls knew/rotary/alibi/accum/leftpad
  p.is_bf16 = true;

  p.q_ptr = query.data;
  p.k_ptr = key.data;
  p.v_ptr = value.data;
  p.o_ptr = out.data;

  // Contiguous varlen on ALL of q/k/v/o: cu_seqlens drive the row offsets, so
  // the batch strides are unused. Strides are in ELEMENTS.
  p.q_row_stride = query.stride[0];
  p.q_head_stride = query.stride[1];
  p.o_row_stride = out.stride[0];
  p.o_head_stride = out.stride[1];
  p.k_row_stride = key.stride[0];
  p.k_head_stride = key.stride[1];
  p.v_row_stride = value.stride[0];
  p.v_head_stride = value.stride[1];
  p.k_batch_stride = 0;
  p.v_batch_stride = 0;

  p.cu_seqlens_q = cu_seqlens_q.Ptr<int32_t>();
  p.cu_seqlens_k = cu_seqlens_k.Ptr<int32_t>();
  p.seqused_k = nullptr;  // contiguous varlen: cu_seqlens_k IS the K geometry
  p.softmax_lse_ptr = softmax_lse;

  p.b = static_cast<int>(num_reqs);
  p.h = static_cast<int>(hq);
  p.h_k = static_cast<int>(hq);  // MLA prefill is multi-head on BOTH sides
  p.h_h_k_ratio = 1;
  p.seqlen_q = max_seqlen_q;
  p.seqlen_k = max_seqlen_k;
  p.seqlen_q_rounded = RoundMultiple(max_seqlen_q, 128);
  p.seqlen_k_rounded = RoundMultiple(max_seqlen_k, 128);
  p.d = static_cast<int>(d);
  p.d_rounded = RoundMultiple(static_cast<int>(d), 32);
  p.total_q = static_cast<int>(total_q);
  (void)total_k;

  p.scale_softmax = args.scale;
  p.scale_softmax_log2 = args.scale * static_cast<float>(M_LOG2E);
  p.softcap = 0.0f;

  p.p_dropout = 1.0f;
  p.p_dropout_in_uint8_t = uint8_t(255);
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = args.scale;
  p.philox_args = at::PhiloxCudaState(0, 0);

  // `causal=True` for new tokens (flash_attn.py:223), `causal=False` for a
  // context chunk (`:246`). No local window on any MLA path — TritonMLAImpl
  // rejects `sliding_window` outright (triton_mla.py:165-171).
  p.is_causal = args.causal;
  p.window_size_left = -1;
  p.window_size_right = args.causal ? 0 : -1;
  p.is_seqlens_k_cumulative = true;
  p.is_rotary_interleaved = false;
  p.rotary_dim = 0;

  p.block_table = nullptr;  // contiguous, NOT paged
  p.block_table_batch_stride = 0;
  p.page_block_size = 1;

  p.unpadded_lse = false;  // see the LSE LAYOUT note above
  p.seqlenq_ngroups_swapped = false;

  // num_splits == 1 => the Split=false kernel writes O directly through
  // cu_seqlens_q with no combine pass — vLLM's varlen prefill exactly, and our
  // fixed-order determinism convention for free.
  p.num_splits = 1;
  p.o_batch_stride = static_cast<int64_t>(max_seqlen_q) * p.o_row_stride;

  if (d == 256) {
    if (args.causal) {
      FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(p, s);
    } else {
      FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(p, s);
    }
  } else if (args.causal) {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, true>(p, s);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, false>(p, s);
  }
  Check(cudaGetLastError(), "mla prefill splitkv dispatch launch");

  if (lse_out != nullptr) {
    const int total = static_cast<int>(lse_elems);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    Fa2UnpadLseKernel<<<blocks, threads, 0, s>>>(
        lse_out, softmax_lse, cu_seqlens_q.Ptr<int32_t>(), static_cast<int>(num_reqs),
        static_cast<int>(hq), max_seqlen_q, static_cast<int>(total_q), lse_row_stride);
    Check(cudaGetLastError(), "mla prefill lse unpad launch");
  }
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

// Launch the pinned FA2 VARLEN decode — the EXACT reduction vLLM's
// flash_attn_varlen_func runs for a paged bf16 KV cache at head_dim 128
// (Qwen3-dense). Unlike LaunchDecodeFA2Bf16 (the d256 group-swap
// seqlenq_ngroups_swapped route), this presents the decode batch as PLAIN
// varlen: one query row per request (cu_seqlens_q = query_start_loc), full
// query heads (h = hq, h_h_k_ratio = groups), per-request K length via
// seqused_k, causal = true, NO group swap. Because a block_table is present,
// upstream forces the split-KV kernel even at num_splits==1, so we dispatch
// run_mha_fwd_splitkv_dispatch<..., 128, causal> exactly like the varlen
// prefill launcher (Split=false when num_splits==1). num_splits is the exact
// port of upstream num_splits_heuristic (=1 for the short gate contexts), and
// for num_splits>1 the split combine addresses O via batch_idx*o_batch_stride;
// with seqlen_q==1 per request that equals o_row_stride, so we set
// o_batch_stride = o_row_stride and the combine writes the packed [total_q,Hq,D]
// output correctly (this is why the packed-prefill combine restriction —
// seqlen_q>1, irregular row spacing — does NOT apply to decode).
// Ported from vllm-project/flash-attention @ 2c839c33 (mha_varlen_fwd paged
// path + set_params_splitkv) and vLLM v0.25.0 flash_attn.py flash_attn_varlen_func.
void LaunchDecodeVarlenFA2Bf16(cudaStream_t s, Tensor& out, const Tensor& query,
                               const Tensor& k_cache, const Tensor& v_cache,
                               const Tensor& block_table, const Tensor& seq_lens,
                               const Tensor& query_start_loc, const PagedAttentionArgs& args,
                               int64_t hq, int64_t d, int64_t num_reqs, int64_t num_kv_heads,
                               int64_t block_size) {
  const int64_t total_q = query.shape[0];
  if (total_q == 0 || num_reqs == 0 || hq == 0 || d == 0) return;
  if (query.dtype != DType::kBF16 || out.dtype != DType::kBF16 ||
      k_cache.dtype != DType::kBF16 || v_cache.dtype != DType::kBF16) {
    throw std::runtime_error(
        "cuda flash-attn-2 varlen decode: bf16 q/kv/out required (dispatch gate must enforce)");
  }
  if (d != 128) {
    throw std::runtime_error(
        "cuda flash-attn-2 varlen decode: head_dim 128 only (dispatch gate must enforce)");
  }

  // Host UPPER BOUNDS for grid sizing + rounded dims only; per-request geometry
  // reads device cu_seqlens_q/seqused_k via BlockInfo (same as the prefill
  // launcher). Production callers hand query_start_loc_host + max_seq_len;
  // op-test callers without them fall back to a small D2H + sync.
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
          "varlen decode query_start_loc D2H");
    Check(cudaMemcpyAsync(sk.data(), seq_lens.Ptr<int32_t>(), sizeof(int32_t) * num_reqs,
                          cudaMemcpyDeviceToHost, s),
          "varlen decode seq_lens D2H");
    Check(cudaStreamSynchronize(s), "varlen decode seqlen sync");
    max_seqlen_q = 0;
    max_seqlen_k = 0;
    for (int64_t i = 0; i < num_reqs; ++i) {
      max_seqlen_q = std::max(max_seqlen_q, qsl[i + 1] - qsl[i]);
      max_seqlen_k = std::max(max_seqlen_k, sk[i]);
    }
  }
  if (max_seqlen_q == 0 || max_seqlen_k == 0) return;

  const int batch = static_cast<int>(num_reqs);
  const int heads = static_cast<int>(hq);
  const int kv_heads = static_cast<int>(num_kv_heads);
  const int head_dim = static_cast<int>(d);
  const int max_blocks = static_cast<int>(block_table.shape[1]);

  const auto stream_scratch = Fa2ScratchFor(query.device.index, s);
  std::lock_guard<std::mutex> submit_lock(stream_scratch->submit_mu);

  // Exact port of set_params_splitkv for head_dim 128: block-N is 128, block-M
  // is 64, and the heuristic receives 2*numSM for the 128-thread CTA occupancy
  // model. For the gate contexts (<=2 key blocks, batch 1) this yields 1.
  constexpr int kBlockN = 128;  // Headdim <= 128
  constexpr int kBlockM = 64;
  const int num_n_blocks = (max_seqlen_k + kBlockN - 1) / kBlockN;
  const int num_m_blocks = (max_seqlen_q + kBlockM - 1) / kBlockM;
  const int num_splits = NumSplitsHeuristic(
      batch * heads * num_m_blocks, stream_scratch->num_sms * 2, num_n_blocks, 128);

  // softmax_lse is [nheads, total_q] f32 (unpadded/varlen LSE, written+consumed
  // by the combine). oaccum/lseaccum only when the KV dimension is split.
  const DecodeShapeKey key{batch,     heads,     kv_heads,  0,
                           head_dim,  max_blocks, static_cast<int>(block_size), num_splits};
  auto [it, inserted] = stream_scratch->varlen_decode.try_emplace(key);
  DecodeScratch& scratch = it->second;
  if (inserted) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    Check(cudaStreamIsCapturing(s, &capture_status), "varlen decode capture-status query");
    if (capture_status != cudaStreamCaptureStatusNone) {
      stream_scratch->varlen_decode.erase(it);
      throw std::runtime_error(
          "cuda flash-attn-2 varlen decode: scratch miss during CUDA graph capture");
    }
    const size_t rows = static_cast<size_t>(heads) * static_cast<size_t>(total_q);
    try {
      AllocateFixed(scratch.softmax_lse, rows * sizeof(float), s,
                    "varlen decode softmax_lse alloc");
      if (num_splits > 1) {
        const size_t split_rows = static_cast<size_t>(num_splits) * rows;
        const int head_dim_rounded = RoundMultiple(head_dim, 64);
        AllocateFixed(scratch.softmax_lse_accum, split_rows * sizeof(float), s,
                      "varlen decode partial LSE alloc");
        AllocateFixed(scratch.out_accum,
                      split_rows * static_cast<size_t>(head_dim_rounded) * sizeof(float), s,
                      "varlen decode partial output alloc");
      }
    } catch (...) {
      if (scratch.softmax_lse.ptr != nullptr) cudaFreeAsync(scratch.softmax_lse.ptr, s);
      if (scratch.softmax_lse_accum.ptr != nullptr) cudaFreeAsync(scratch.softmax_lse_accum.ptr, s);
      if (scratch.out_accum.ptr != nullptr) cudaFreeAsync(scratch.out_accum.ptr, s);
      stream_scratch->varlen_decode.erase(it);
      throw;
    }
  }

  FLASH_NAMESPACE::Flash_fwd_params p{};  // zero-init nulls knew/rotary/alibi/leftpad
  p.is_bf16 = true;
  p.q_ptr = query.data;
  p.k_ptr = k_cache.data;
  p.v_ptr = v_cache.data;
  p.o_ptr = out.data;

  // Varlen q/o: cu_seqlens_q drives the row offset; batch strides are ignored on
  // the non-split path but MUST equal o_row_stride so the split combine (which
  // writes O via batch_idx*o_batch_stride, seqlen_q==1) lands each request's row.
  p.q_batch_stride = query.stride[0];
  p.q_row_stride = query.stride[0];
  p.q_head_stride = query.stride[1];
  p.o_batch_stride = out.stride[0];
  p.o_row_stride = out.stride[0];
  p.o_head_stride = out.stride[1];
  // Paged k/v [num_blocks, block_size, num_kv_heads, d].
  p.k_batch_stride = k_cache.stride[0];
  p.k_row_stride = k_cache.stride[1];
  p.k_head_stride = k_cache.stride[2];
  p.v_batch_stride = v_cache.stride[0];
  p.v_row_stride = v_cache.stride[1];
  p.v_head_stride = v_cache.stride[2];

  p.cu_seqlens_q = query_start_loc.Ptr<int32_t>();  // [0,1,2,...,num_reqs]
  p.cu_seqlens_k = nullptr;                          // paged: block_table + seqused_k
  p.seqused_k = seq_lens.Ptr<int32_t>();
  p.softmax_lse_ptr = scratch.softmax_lse.ptr;
  p.softmax_lseaccum_ptr = scratch.softmax_lse_accum.ptr;
  p.oaccum_ptr = scratch.out_accum.ptr;

  p.b = batch;
  p.h = heads;
  p.h_k = kv_heads;
  p.h_h_k_ratio = static_cast<int>(hq / num_kv_heads);
  p.seqlen_q = max_seqlen_q;  // 1 per decode request
  p.seqlen_k = max_seqlen_k;
  p.seqlen_q_rounded = RoundMultiple(max_seqlen_q, 128);
  p.seqlen_k_rounded = RoundMultiple(max_seqlen_k, 128);
  p.d = head_dim;
  p.d_rounded = RoundMultiple(head_dim, 64);
  p.total_q = static_cast<int>(total_q);

  p.scale_softmax = args.scale;
  p.scale_softmax_log2 = args.scale * static_cast<float>(M_LOG2E);
  p.softcap = 0.0F;
  p.p_dropout = 1.0F;
  p.p_dropout_in_uint8_t = uint8_t(255);
  p.rp_dropout = 1.0F;
  p.scale_softmax_rp_dropout = args.scale;
  p.philox_args = at::PhiloxCudaState(0, 0);

  // Plain causal decode (NO group-swap normalization): vLLM passes causal=True to
  // flash_attn_varlen_func; with seqlen_q==1 the single query still sees the full
  // context, and the causal template selects the exact n_block geometry vLLM runs.
  p.is_causal = args.causal;
  p.window_size_left = -1;
  p.window_size_right = args.causal ? 0 : -1;
  p.is_seqlens_k_cumulative = true;  // ignored while cu_seqlens_k == nullptr
  p.is_rotary_interleaved = false;
  p.rotary_dim = 0;

  p.block_table = block_table.Ptr<int32_t>();
  p.block_table_batch_stride = block_table.stride[0];
  p.page_block_size = static_cast<int>(block_size);
  p.unpadded_lse = true;             // LSE is [nheads, total_q]
  p.seqlenq_ngroups_swapped = false;  // the whole point: no group swap
  p.num_splits = num_splits;

  RecordDecodeLaunch(num_splits > 1, inserted);
  if (args.causal) {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(p, s);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, false>(p, s);
  }
  Check(cudaGetLastError(), "varlen decode splitkv dispatch launch");
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
  for (auto& entry : scratch->varlen_decode) {
    release(entry.second.softmax_lse);
    release(entry.second.softmax_lse_accum);
    release(entry.second.out_accum);
  }
  scratch->varlen_decode.clear();
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
