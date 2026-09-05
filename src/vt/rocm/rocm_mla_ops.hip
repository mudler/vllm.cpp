// ROCm MLA data-movement ops — the W1 slice of the MLA/DSA campaign
// (`BACKEND-ROCM`, issue #2715, spec `.agents/specs/rocm-mla-dsa-ops.md`).
//
// Three ops, and every one of them is a PURE COPY:
//
//   kConcatAndCacheMla  — write one token's latent + decoupled-pe into its KV
//                         cache slot, as one 576-wide concatenated entry.
//   kConcatMlaNopeRope  — concatenate the NoPE and RoPE halves of a q or k row.
//   kGatherMlaCache     — gather a request's cache entries back into a dense
//                         [total_tokens, head_dim] buffer for prefill.
//
// WHY THEY MATTER HERE, AND WHAT THEY DO NOT FIX. An op with no native ROCm
// kernel meets one of two fates, and WHICH ONE on this board is a live question
// this branch measures rather than assumes.
//
// Where the portable CPU reference tier is eligible it installs a host kernel
// and the op RUNS, slowly: not a correctness failure, but a MEASUREMENT one,
// because `docs/ROCM.md:60-61` disqualifies any performance result from a run
// whose `GetReferenceTierHits()` is non-zero. That is the state #2715 describes
// and the state `9f3e6e223`'s GLM-5.3 run observed, with five ops on the tier.
//
// Where it is not eligible, `GetOp` REFUSES by name rather than hand a host
// kernel a pointer the host may not dereference. That is a discrete AMD card —
// and, since `6b97a6800` (#2511) narrowed managed allocation to
// `PageableMemoryAccess == 1` and made the unified claim follow the allocator,
// possibly `gfx1151` too: it reports that attribute 0, and `docs/ROCM.md:83-85`
// says outright that this part gets "plain `hipMalloc` and no reference tier".
// `9f3e6e223` is NOT a descendant of that commit, so its observation predates
// the narrowing. `tests/vt/test_backend_cross_device.cpp` asks the board through
// `vt::ReferenceTierEligible` and asserts the consequence either way; this
// header does not pick an answer.
//
// Registering these three does not make a GLM-5.3 speed number admissible. The
// campaign's W2 (`kDsaIndexerLogits`, `kDsaTopkSelect`) and W3
// (`kMlaDecodeAttention`) are still unregistered after this file lands, so the
// hit count on a real run stays non-zero and the axis stays VOID. The spec's
// §Split says why the campaign is staged rather than landed at once.
//
// UPSTREAM (pinned oracle `~/_git/vllm` @ `5559679229`).
//   ConcatAndCacheMla  <- vllm/csrc/libtorch_stable/cache_kernels.cu:401-442
//                         (`concat_and_cache_mla_kernel`), launched :899-900
//   ConcatMlaNopeRope  <- vllm/csrc/libtorch_stable/cache_kernels.cu:1572-1584
//                         plus concat_mla_q.cuh:13,21-24,50-53
//   GatherMlaCache     <- vllm/csrc/libtorch_stable/cache_kernels.cu:992-1064,
//                         launched :1142-1145 (`thread_block_size = 64`)
//
// PORTED FROM the CUDA siblings, element for element:
//   src/vt/cuda/cuda_cache.cu:243-297        (ConcatAndCacheMla)
//   src/vt/cuda/cuda_mla_attn.cu:752-802     (ConcatMlaNopeRope)
//   src/vt/cuda/cuda_mla_prefill.cu:302-361  (GatherMlaCache)
//
// WORD TYPING, and why it is not a shortcut. Every kernel here is typed on the
// raw STORAGE word (`uint16_t` / `uint32_t`) rather than on a float type. That
// is the CUDA donor's own choice for this op class and it states the reason at
// `cuda_cache.cu:233-236`: the copy is bit-exact, so no dtype conversion ever
// appears and the storage type is the only thing the kernel needs to know. It
// collapses f32/bf16/f16 into two cases, and it makes bit-exactness against the
// CPU oracle STRUCTURAL rather than argued — which is why the gate for these
// three asserts equality rather than an NMSE bound.
//
// WAVEFRONT WIDTH is not a concern in this file: none of the three kernels uses
// a warp-level primitive or a cross-lane reduction. They are grid-strided
// element copies with `__syncthreads()` nowhere, so they are wavefront-width
// agnostic by construction, unlike the W2 indexer pair and the W3 decode kernel.
//
// Gate: the three arms added to tests/vt/test_backend_cross_device.cpp. Each
// asserts BIT-EXACT equality against the CPU oracle, that
// `vt::OpRegistered(op, device)` is true, and that `vt::GetReferenceTierHits()`
// did not move across the call. The middle assertion is the load-bearing one:
// `OpRegistered` is a native-only probe (`src/vt/op_provider.cpp:788-806`), so
// it is the only one of the three that can tell a native kernel from the
// reference tier — the tier computes the SAME answer on the host, so the
// equality assertion alone is green before any of this exists.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"

namespace vt::rocm {
namespace {

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm mla_ops: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

// ─── kConcatAndCacheMla ─────────────────────────────────────────────────────
// One block per token, threads stride the copied run — upstream's `copy` lambda
// (`cache_kernels.cu:426-438`) and the CUDA donor's kernel, unchanged.
//
// Destination arithmetic is upstream `:431-433`:
//   dst = block_idx*block_stride + block_offset*entry_stride + i + offset
// with offset 0 for the latent and `kv_lora_rank` for the rope part, so the two
// sources are CONCATENATED into one entry.
template <typename Word>
__global__ void ConcatAndCacheMlaKernelHip(const Word* __restrict__ kv_c,
                                           const Word* __restrict__ k_pe,
                                           Word* __restrict__ kv_cache,
                                           const int64_t* __restrict__ slot_mapping,
                                           int64_t block_size, int64_t kv_lora_rank,
                                           int64_t pe_dim, int64_t block_stride,
                                           int64_t entry_stride, int64_t kv_c_stride,
                                           int64_t k_pe_stride) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  // Padded token -> skip. Upstream `:419-422`; dropping this writes block 0.
  if (slot < 0) return;
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  const int64_t entry = block * block_stride + offset * entry_stride;
  const int64_t csrc = token * kv_c_stride;
  const int64_t psrc = token * k_pe_stride;
  for (int64_t i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    kv_cache[entry + i] = kv_c[csrc + i];
  }
  for (int64_t i = threadIdx.x; i < pe_dim; i += blockDim.x) {
    kv_cache[entry + kv_lora_rank + i] = k_pe[psrc + i];
  }
}

// ─── kConcatMlaNopeRope ─────────────────────────────────────────────────────
// Grid-strided over every written element. `rope_broadcast` is the single-head
// rope row shared across all q heads (`concat_mla_q.cuh:50-53`); the CUDA donor
// derives it the same way, from `rope.shape[1] == 1 && heads > 1`.
//
// RECORDED DEVIATION, inherited from the donor and stated there: upstream
// vectorizes at 128/256 bits and instantiates only NOPE_DIM=512 / rope 64
// (`concat_mla_q.cuh:13,21-24`). This is SCALAR and width-generic. A concat is a
// pure copy, so the bytes written are identical either way.
template <typename Word>
__global__ void ConcatMlaNopeRopeKernelHip(Word* __restrict__ out,
                                           const Word* __restrict__ nope,
                                           const Word* __restrict__ rope, int64_t heads,
                                           int64_t dn, int64_t dr, int64_t out_s0,
                                           int64_t out_s1, int64_t nope_s0, int64_t nope_s1,
                                           int64_t rope_s0, int64_t rope_s1,
                                           bool rope_broadcast, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t width = dn + dr;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t d = idx % width;
    const int64_t h = (idx / width) % heads;
    const int64_t t = idx / (width * heads);
    Word* dst = out + t * out_s0 + h * out_s1 + d;
    if (d < dn) {
      *dst = nope[t * nope_s0 + h * nope_s1 + d];
    } else {
      const int64_t rh = rope_broadcast ? 0 : h;
      *dst = rope[t * rope_s0 + rh * rope_s1 + (d - dn)];
    }
  }
}

// ─── kGatherMlaCache ────────────────────────────────────────────────────────
// One block per token (upstream `dim3 grid(num_tokens)`, `:1145`), 64 threads
// striding the entry (`constexpr int32_t thread_block_size = 64;`, `:1142`).
// The index arithmetic is `:1013-1031` verbatim, including the
// `token_id >= batch_end` early return at `:1019` — which is a RETURN, not a
// `continue`, so a grid smaller than `num_tokens` stops the whole strided walk
// for that block exactly as upstream does.
template <typename Word>
__global__ void GatherMlaCacheKernelHip(Word* __restrict__ dst,
                                        const Word* __restrict__ src_cache,
                                        const int32_t* __restrict__ block_table,
                                        const int32_t* __restrict__ cu_seq_lens,
                                        const int32_t* __restrict__ token_to_seq,
                                        const int32_t* __restrict__ seq_starts, int num_tokens,
                                        int block_size, int head_dim,
                                        int64_t block_table_stride, int64_t cache_block_stride,
                                        int64_t cache_entry_stride, int64_t dst_entry_stride) {
  for (int token_id = blockIdx.x; token_id < num_tokens; token_id += gridDim.x) {
    const int batch_id = token_to_seq[token_id];
    const int batch_start = cu_seq_lens[batch_id];
    const int batch_end = cu_seq_lens[batch_id + 1];
    if (token_id >= batch_end) return;  // upstream `:1019`
    int batch_offset = token_id - batch_start;
    if (seq_starts != nullptr) batch_offset += seq_starts[batch_id];
    const int block_table_id = batch_offset / block_size;
    const int slot_id = batch_offset % block_size;
    const int block_id =
        block_table[static_cast<int64_t>(batch_id) * block_table_stride + block_table_id];
    const int64_t cache_offset =
        static_cast<int64_t>(block_id) * cache_block_stride + slot_id * cache_entry_stride;
    Word* d = dst + static_cast<int64_t>(token_id) * dst_entry_stride;
    const Word* sp = src_cache + cache_offset;
    for (int idx = threadIdx.x; idx < head_dim; idx += blockDim.x) d[idx] = sp[idx];
  }
}

}  // namespace

// The registered ConcatAndCacheMlaFn (include/vt/ops.h:2517). Every argument
// check already ran in `ops.cpp`; a backend writes zero validation.
void ConcatAndCacheMlaKernelRocm(Queue& q, const Tensor& kv_c, const Tensor& k_pe,
                                 Tensor& kv_cache, const Tensor& slot_mapping) {
  EnsureQueueDevice(q);
  const int64_t num_slots = slot_mapping.shape[0];
  if (num_slots == 0) return;
  const int64_t block_size = kv_cache.shape[1];
  const int64_t kv_lora_rank = kv_c.shape[1];
  const int64_t pe_dim = k_pe.shape[1];
  const unsigned grid = static_cast<unsigned>(num_slots);
  // Upstream `:899-900`: block(min(kv_lora_rank, 512)).
  const unsigned block = static_cast<unsigned>(kv_lora_rank < 512 ? kv_lora_rank : 512);
  hipStream_t s = AsStream(q);
  const int64_t* slots = slot_mapping.Ptr<int64_t>();

#define VT_ROCM_CONCAT_CACHE(W)                                                              \
  ConcatAndCacheMlaKernelHip<W><<<grid, block, 0, s>>>(                                       \
      kv_c.Ptr<W>(), k_pe.Ptr<W>(), kv_cache.Ptr<W>(), slots, block_size, kv_lora_rank,        \
      pe_dim, kv_cache.stride[0], kv_cache.stride[1], kv_c.stride[0], k_pe.stride[0])
  switch (SizeOf(kv_c.dtype)) {
    case 4: VT_ROCM_CONCAT_CACHE(uint32_t); break;
    case 2: VT_ROCM_CONCAT_CACHE(uint16_t); break;
    default: VT_CHECK(false, "rocm concat_and_cache_mla: unsupported dtype element size");
  }
#undef VT_ROCM_CONCAT_CACHE
  Check(hipGetLastError(), "concat_and_cache_mla launch");
}

// The registered ConcatMlaNopeRopeFn (include/vt/ops.h:2153).
void ConcatMlaNopeRopeKernelRocm(Queue& q, Tensor& out, const Tensor& nope,
                                 const Tensor& rope) {
  EnsureQueueDevice(q);
  const int64_t tokens = out.shape[0], heads = out.shape[1];
  const int64_t dn = nope.shape[2], dr = rope.shape[2];
  const int64_t n = tokens * heads * (dn + dr);
  if (n == 0) return;
  constexpr int kBlock = 256;
  const int64_t blocks = std::min<int64_t>((n + kBlock - 1) / kBlock, 65535);
  const bool broadcast = rope.shape[1] == 1 && heads > 1;
  hipStream_t s = AsStream(q);

#define VT_ROCM_CONCAT_NOPE_ROPE(W)                                                          \
  ConcatMlaNopeRopeKernelHip<W><<<static_cast<unsigned>(blocks), kBlock, 0, s>>>(             \
      out.Ptr<W>(), nope.Ptr<W>(), rope.Ptr<W>(), heads, dn, dr, out.stride[0],                \
      out.stride[1], nope.stride[0], nope.stride[1], rope.stride[0], rope.stride[1],           \
      broadcast, n)
  switch (SizeOf(out.dtype)) {
    case 4: VT_ROCM_CONCAT_NOPE_ROPE(uint32_t); break;
    case 2: VT_ROCM_CONCAT_NOPE_ROPE(uint16_t); break;
    default: VT_CHECK(false, "rocm concat_mla_nope_rope: unsupported dtype");
  }
#undef VT_ROCM_CONCAT_NOPE_ROPE
  Check(hipGetLastError(), "concat_mla_nope_rope launch");
}

// The registered GatherMlaCacheFn (include/vt/ops.h:2545).
void GatherMlaCacheKernelRocm(Queue& q, Tensor& dst, const Tensor& src_cache,
                              const Tensor& block_table, const Tensor& cu_seq_lens,
                              const Tensor& token_to_seq, const Tensor* seq_starts,
                              int64_t num_tokens) {
  EnsureQueueDevice(q);
  if (num_tokens == 0) return;
  hipStream_t s = AsStream(q);
  const int block_size = static_cast<int>(src_cache.shape[1]);
  const int head_dim = static_cast<int>(src_cache.shape[2]);
  const int threads = 64;  // upstream `:1142`
  const unsigned grid = static_cast<unsigned>(num_tokens);
  const int32_t* starts = seq_starts != nullptr ? seq_starts->Ptr<int32_t>() : nullptr;

#define VT_ROCM_GATHER_MLA(W)                                                                \
  GatherMlaCacheKernelHip<W><<<grid, threads, 0, s>>>(                                        \
      dst.Ptr<W>(), src_cache.Ptr<W>(), block_table.Ptr<int32_t>(),                            \
      cu_seq_lens.Ptr<int32_t>(), token_to_seq.Ptr<int32_t>(), starts,                         \
      static_cast<int>(num_tokens), block_size, head_dim, block_table.stride[0],               \
      src_cache.stride[0], src_cache.stride[1], dst.stride[0])
  switch (SizeOf(dst.dtype)) {
    case 4: VT_ROCM_GATHER_MLA(uint32_t); break;
    case 2: VT_ROCM_GATHER_MLA(uint16_t); break;
    default: VT_CHECK(false, "rocm gather_mla_cache: unsupported dtype");
  }
#undef VT_ROCM_GATHER_MLA
  Check(hipGetLastError(), "gather_mla_cache launch");
}

}  // namespace vt::rocm
