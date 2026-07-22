// MLA decode attention (CUDA) — MLA campaign W4.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ───────────────────────
// This is a 1:1 STRUCTURE port of the two-stage split-KV flash-decode that the
// vLLM 0.25.0 oracle was OBSERVED executing on GB10/sm_121 at W0
// (`_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2` — see
// .agents/specs/mla-deepseek-campaign.md `### Work breakdown` W0(f)):
//
//   OURS                          <-  UPSTREAM (@ pin e24d1b24)
//   MlaDecodeStage1 (below)       <-  vllm/v1/attention/ops/triton_decode_attention.py
//                                     :278-458 `_fwd_grouped_kernel_stage1`
//                                     (the IS_MLA=True branch)
//   MlaDecodeStage2 (below)       <-  ...:575-639 `_fwd_kernel_stage2`
//   LaunchMlaDecode (below)       <-  ...:470-573 `_decode_grouped_att_m_fwd`
//                                     + :642-682 `_decode_softmax_reducev_fwd`
//                                     + :719-754 `decode_attention_fwd_grouped`
//   ComputeNumKvSplits (below)    <-  vllm/v1/attention/backends/mla/triton_mla.py
//                                     :40-47 `_compute_num_kv_splits`
//   the caller contract           <-  ...triton_mla.py:189-260 `forward_mqa`
//
// Upstream's own lineage, recorded at triton_decode_attention.py:1-11:
// vLLM <- SGLang `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`
// <- lightllm `.../deepseek2/triton_kernel/gqa_flash_decoding_stage{1,2}.py`.
//
// ─── THE MLA-SPECIFIC TRICK, ported verbatim ────────────────────────────────
// triton_decode_attention.py:424-431: under `IS_MLA` the kernel does NOT load a
// V tile at all —
//     # MLA uses a single c_kv. loading the same c_kv to interpret it as v is
//     # not necessary. transpose the existing c_kv (aka k) for the dot product.
//     v = tl.trans(k)
// i.e. V is the LEADING `v_head_dim` slice of the SAME latent row already loaded
// as K. Our shared-memory K tile is therefore also the V tile: the QK dot runs
// over all `head_size` (576) columns and the output accumulation over the first
// `v_head_dim` (512) of the very same `k_s` rows. Upstream expresses the 512/64
// nope/rope split as two tiles (BLOCK_DMODEL=512 + BLOCK_DPE=64, `:494-496`)
// purely because Triton needs power-of-two tile shapes; the arithmetic is one
// 576-wide dot either way, so we load one 576-wide row and skip the split.
//
// ─── WHERE OUR FA-2 SPLIT+COMBINE MACHINERY FIT, AND WHERE IT DID NOT ───────
// FIT (the algorithm): the split-KV SCHEDULE and the combine ALGEBRA are the
// same shape as our existing FA-2 decode — partition [0, seq_len) into
// `num_kv_splits` contiguous chunks, emit one normalized partial + its LSE per
// chunk, then merge in a FIXED ASCENDING order with an online-softmax rescale
// (no atomicAdd). That is our house determinism convention and it transfers
// unchanged, which is exactly what the W0 observation predicted.
// DID NOT FIT (the code): the vendored FA-2 launcher in cuda_flash_attn_fa2.cu
// cannot be reused, and forcing it would have been dishonest —
//   1. it consumes SEPARATE `k_cache` / `v_cache` 4-D tensors
//      (num_blocks, block_size, num_kv_heads, head_size); MLA's cache is 3-D
//      with no K/V and no head axis, and K and V are the same bytes;
//   2. its CUTLASS instantiations are compiled for head_dim {128, 256} with
//      QK dim == V dim; MLA needs QK 576 / V 512, an ASYMMETRIC pair that no
//      vendored instantiation exists for (and 576 is not a supported FA-2
//      head_dim at all);
//   3. its combine kernel addresses a batched [B, Hq, D] output through the FA-2
//      params struct, which has no notion of a V width different from the QK
//      width.
// So the reuse here is STRUCTURAL (the schedule + the merge algebra + the
// determinism rule), not code-level. Generalizing the vendored FA-2 launcher is
// W5's problem for PREFILL (qk 192 / v 128), where it is tractable.
//
// ─── DETERMINISM ────────────────────────────────────────────────────────────
// Stage 2 loops `for (split = 0 .. num_splits)` in ascending order (upstream
// `:607`), and stage 1 accumulates over keys in ascending order within a split.
// There is no atomicAdd and no run-to-run-variable reduction tree anywhere, so
// the op is bit-reproducible run-to-run for a fixed `num_kv_splits`. Changing
// `num_kv_splits` changes the f32 summation order and may change the last bits —
// that is upstream's behavior too (which is precisely why upstream forces
// `num_kv_splits = 1` under VLLM_BATCH_INVARIANT, triton_mla.py:212-213).
//
// ─── DELIBERATE DEVIATIONS, recorded ────────────────────────────────────────
//   * fp8 KV cache (`if k.dtype.is_fp8(): k = (k.to(f32) * ks).to(q.dtype)`,
//     `:390-391` / `:409-410`) is NOT ported — out of campaign scope; the op
//     wrapper refuses a mismatched cache dtype loudly (same discipline as W3's
//     `fp8_ds_mla` refusal).
//   * `logit_cap` (`:404-405`) is NOT ported: `TritonMLAImpl.__init__` REJECTS
//     `logits_soft_cap` (triton_mla.py:165-171), so it is 0 on every reachable
//     MLA path.
//   * `seq_len == 0` writes ZEROS (upstream's stage 2 would divide by an
//     `e_sum` of 0). Unreachable upstream — a scheduled decode request always
//     has >= 1 computed token — and matched exactly by our CPU reference.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/cuda/cuda_device_caps.h"
#include "vt/cuda/graph_safe_scratch.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: mla_decode_attention: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

__device__ __forceinline__ float Ld(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float Ld(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ float Ld(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ __forceinline__ void St(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}
__device__ __forceinline__ void St(__half* p, int64_t i, float v) { p[i] = __float2half(v); }

// BLOCK_H — upstream's head tile (triton_decode_attention.py:513 `BLOCK_H = 16`).
// One WARP per head in the tile, so the whole tile shares each K row it loads
// from shared memory: that head-group amortization is the entire reason the
// grouped kernel exists (MLA is MQA — all Hq heads read the SAME single-head
// latent).
constexpr int kBlockH = 16;
constexpr int kThreads = kBlockH * 32;
// BLOCK_N — upstream's key tile (`:509 BLOCK = 32`). We use 8 by default so the
// per-block shared memory (kBlockH + kNTile) * head_size * 4 B stays under the
// 48 KiB every architecture guarantees for head_size == 576 plus headroom;
// kNTileSmall is the fallback for a head_size large enough that even that does
// not fit. Purely a tiling choice: the numbers do not depend on it beyond f32
// summation order within a tile.
constexpr int kNTile = 8;
constexpr int kNTileSmall = 1;

// ─── STAGE 1 ────────────────────────────────────────────────────────────────
// Port of `_fwd_grouped_kernel_stage1` (triton_decode_attention.py:278-458),
// IS_MLA branch. Grid (batch, head_tiles, num_splits) mirrors `:517-521`.
// Writes, per (b, head, split), the NORMALIZED partial `acc / e_sum` into
// mid[..., 0:v_head_dim] and `e_max + log(e_sum)` into mid[..., v_head_dim]
// (upstream `:436-457`). A split whose range is empty writes NOTHING (upstream
// `:361 if split_kv_end > split_kv_start`) and stage 2 skips the same range.
//
// DVREGS: v_head_dim / 32 rounded up, as a COMPILE-TIME bound so the per-lane
// output accumulator lives in registers (a runtime-bounded array would spill to
// local memory). Lane `l` owns output dims l, l+32, l+64, ... — the same lane
// striding used for the QK partial sums, which keeps every shared-memory access
// bank-conflict-free.
template <typename T, int DVREGS, int NTILE>
__global__ __launch_bounds__(kThreads) void MlaDecodeStage1(
    float* __restrict__ mid, const T* __restrict__ query, const T* __restrict__ kv_cache,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ seq_lens,
    int64_t q_s0, int64_t q_s1, int64_t c_s0, int64_t c_s1, int64_t bt_s0, int head_size,
    int v_head_dim, int block_size, int num_heads, int valid_h, int num_splits,
    int64_t mid_s0, int64_t mid_s1, int64_t mid_s2, float scale) {
  const int b = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.z);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  const int seq_len = seq_lens[b];
  // `:352-354`: kv_len_per_split = cdiv(seq_len, NUM_KV_SPLITS);
  //             start = kv_len_per_split * split_kv_id;
  //             end   = min(start + kv_len_per_split, seq_len).
  const int per_split = (seq_len + num_splits - 1) / num_splits;
  const int split_start = per_split * split;
  const int split_end = min(split_start + per_split, seq_len);
  if (split_end <= split_start) return;  // `:361` — block-uniform, so safe here

  // `:319-324`: cur_head = cur_head_id * VALID_BLOCK_H + arange(0, BLOCK_H),
  //             mask_h = (cur_head < (cur_head_id+1)*VALID_BLOCK_H)
  //                      & (cur_head < q_head_num).
  const int my_head = static_cast<int>(blockIdx.y) * valid_h + warp;
  const bool h_valid = warp < valid_h && my_head < num_heads;

  extern __shared__ float smem[];
  float* q_s = smem;                                    // [kBlockH][head_size]
  float* k_s = smem + static_cast<int64_t>(kBlockH) * head_size;  // [NTILE][head_size]

  // `:332-338` — load this warp's whole query row once.
  if (h_valid) {
    const T* qp = query + b * q_s0 + static_cast<int64_t>(my_head) * q_s1;
    for (int d = lane; d < head_size; d += 32) {
      q_s[static_cast<int64_t>(warp) * head_size + d] = Ld(qp, d);
    }
  }

  float m = -CUDART_INF_F;
  float l = 0.0f;
  float acc[DVREGS];
#pragma unroll
  for (int i = 0; i < DVREGS; ++i) acc[i] = 0.0f;

  for (int n0 = split_start; n0 < split_end; n0 += NTILE) {
    const int n_cnt = min(NTILE, split_end - n0);
    __syncthreads();
    // `:372-390` — gather the key tile through the block table. Whole-block
    // cooperative + contiguous in `d`, so the loads coalesce; the tile is the K
    // AND the V tile (`:424-431`).
    for (int n = 0; n < n_cnt; ++n) {
      const int j = n0 + n;
      const int blk = block_table[b * bt_s0 + j / block_size];
      const T* src = kv_cache + static_cast<int64_t>(blk) * c_s0 +
                     static_cast<int64_t>(j % block_size) * c_s1;
      for (int d = static_cast<int>(threadIdx.x); d < head_size; d += kThreads) {
        k_s[static_cast<int64_t>(n) * head_size + d] = Ld(src, d);
      }
    }
    __syncthreads();
    if (!h_valid) continue;  // warp-uniform; both __syncthreads still execute

    // `:392-402` — qk = dot(q, k) over the FULL head_size (upstream splits it
    // into the BLOCK_DMODEL + BLOCK_DPE tiles and ADDS the two dots, `:393-402`;
    // one contiguous 576-wide dot is the same sum).
    float qk[NTILE];
#pragma unroll
    for (int n = 0; n < NTILE; ++n) {
      if (n >= n_cnt) {
        qk[n] = -CUDART_INF_F;
        continue;
      }
      float part = 0.0f;
      const float* kp = k_s + static_cast<int64_t>(n) * head_size;
      const float* qp = q_s + static_cast<int64_t>(warp) * head_size;
      for (int d = lane; d < head_size; d += 32) part += qp[d] * kp[d];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        part += __shfl_xor_sync(0xffffffffu, part, off);
      }
      qk[n] = part * scale;  // `:403` qk *= sm_scale
    }

    // `:433-441` — online softmax rescale, then acc += p @ v with v == k.
    float m_new = m;
#pragma unroll
    for (int n = 0; n < NTILE; ++n) {
      if (n < n_cnt) m_new = fmaxf(m_new, qk[n]);
    }
    const float rescale = expf(m - m_new);
#pragma unroll
    for (int i = 0; i < DVREGS; ++i) acc[i] *= rescale;
    float lsum = 0.0f;
#pragma unroll
    for (int n = 0; n < NTILE; ++n) {
      if (n >= n_cnt) continue;
      const float p = expf(qk[n] - m_new);
      lsum += p;
      const float* kp = k_s + static_cast<int64_t>(n) * head_size;
#pragma unroll
      for (int i = 0; i < DVREGS; ++i) {
        const int d = lane + 32 * i;
        if (d < v_head_dim) acc[i] += p * kp[d];
      }
    }
    l = l * rescale + lsum;
    m = m_new;
  }

  if (!h_valid) return;
  // `:443-457` — store the NORMALIZED partial and its LSE.
  float* dst = mid + b * mid_s0 + static_cast<int64_t>(my_head) * mid_s1 +
               static_cast<int64_t>(split) * mid_s2;
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
  for (int i = 0; i < DVREGS; ++i) {
    const int d = lane + 32 * i;
    if (d < v_head_dim) dst[d] = acc[i] * inv;
  }
  if (lane == 0) dst[v_head_dim] = m + logf(l);
}

// ─── STAGE 2 ────────────────────────────────────────────────────────────────
// Port of `_fwd_kernel_stage2` (triton_decode_attention.py:575-639). One block
// per (batch, head), mirroring `:669 grid = (batch, head_num)`. The split loop
// runs ASCENDING (`:607`) and recomputes the SAME [start, end) partition stage 1
// used, so an empty split is skipped identically and its never-written scratch
// row is never read (`:610`).
//
// Every thread tracks the scalar (e_max, e_sum) redundantly — upstream keeps
// them as Triton scalars broadcast across the BLOCK_DV lanes; the arithmetic and
// the order are identical, so the result is the same and there is no cross-lane
// reduction to make non-deterministic.
template <typename T>
__global__ void MlaDecodeStage2(T* __restrict__ out, float* __restrict__ lse,
                                const float* __restrict__ mid,
                                const int32_t* __restrict__ seq_lens, int64_t o_s0,
                                int64_t o_s1, int64_t lse_s0, int64_t mid_s0, int64_t mid_s1,
                                int64_t mid_s2, int v_head_dim, int num_splits) {
  const int b = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int seq_len = seq_lens[b];
  const int per_split = (seq_len + num_splits - 1) / num_splits;
  const float* base = mid + b * mid_s0 + static_cast<int64_t>(h) * mid_s1;

  for (int d0 = 0; d0 < v_head_dim; d0 += static_cast<int>(blockDim.x)) {
    const int d = d0 + static_cast<int>(threadIdx.x);
    const bool active = d < v_head_dim;
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      const int split_start = per_split * s;
      const int split_end = min(split_start + per_split, seq_len);
      if (split_end <= split_start) continue;  // `:610`
      const float* p = base + static_cast<int64_t>(s) * mid_s2;
      const float tlogic = p[v_head_dim];              // `:613`
      const float tv = active ? p[d] : 0.0f;           // `:611-612`
      const float m_new = fmaxf(tlogic, m);            // `:614`
      const float old_scale = expf(m - m_new);       // `:616`
      const float exp_logic = expf(tlogic - m_new);  // `:618`
      acc = acc * old_scale + exp_logic * tv;          // `:617,619`
      l = l * old_scale + exp_logic;                   // `:621`
      m = m_new;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;  // `:625` (0-length guard, ours)
    if (active) St(out, b * o_s0 + static_cast<int64_t>(h) * o_s1 + d, acc * inv);
    if (lse != nullptr && d0 == 0 && threadIdx.x == 0) {
      // `:636` lse_val = e_max + tl.log(e_sum)
      lse[b * lse_s0 + h] = l > 0.0f ? m + logf(l) : -CUDART_INF_F;
    }
  }
}

// ─── the split-KV workspace ─────────────────────────────────────────────────
// Upstream reserves `(B, q_num_heads, max_splits, kv_lora_rank + 1)` f32 from the
// shared workspace manager at build time so the decode hot path never allocates
// (triton_mla.py:57-78 `_reserve_attn_logits_workspace`, consumed at `:221-234`).
// Our equivalent is the house grow-only PER-STREAM scratch: on growth the old
// block is RETIRED, never freed, because a captured decode CUDA graph may have
// baked the pointer (see graph_safe_scratch.h).
struct StreamScratch {
  void* buf = nullptr;
  size_t bytes = 0;
};

StreamScratch& ScratchFor(cudaStream_t s) {
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, StreamScratch> m;
  std::lock_guard<std::mutex> lk(mu);
  return m[s];
}

float* EnsureMidScratch(size_t need, cudaStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.bytes) {
    RetireGraphScratch(sc.buf);
    Check(cudaMallocAsync(&sc.buf, need, s), "cudaMallocAsync attn-logits workspace");
    sc.bytes = need;
  }
  return static_cast<float*>(sc.buf);
}

// Port of `_compute_num_kv_splits` (triton_mla.py:40-47):
//     ideal   = next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
//     maximum = sm_count * _SPLIT_OCCUPANCY_MULTIPLIER
//     return min(ideal, maximum)
// with _MIN_WORK_PER_SPLIT = 512 (`:36`) and _SPLIT_OCCUPANCY_MULTIPLIER = 2
// (`:37`). Any value >= 1 is CORRECT (the partition always covers [0, seq_len));
// this is purely the occupancy heuristic.
int ComputeNumKvSplits(int max_seq_len, int sm_count) {
  int ideal = 1;
  const int work = max_seq_len / 512;
  const int target = work > 1 ? work : 1;
  while (ideal < target) ideal <<= 1;
  const int max_splits = sm_count > 0 ? sm_count * 2 : 1;
  return ideal < max_splits ? ideal : max_splits;
}

template <typename T, int DVREGS>
void LaunchStage1(cudaStream_t s, dim3 grid, size_t smem, int n_tile, float* mid,
                  const T* query, const T* kv_cache, const int32_t* block_table,
                  const int32_t* seq_lens, int64_t q_s0, int64_t q_s1, int64_t c_s0,
                  int64_t c_s1, int64_t bt_s0, int head_size, int v_head_dim, int block_size,
                  int num_heads, int valid_h, int num_splits, int64_t mid_s0, int64_t mid_s1,
                  int64_t mid_s2, float scale) {
  if (n_tile == kNTile) {
    const void* fn = reinterpret_cast<const void*>(&MlaDecodeStage1<T, DVREGS, kNTile>);
    if (smem > 48u * 1024u) {
      Check(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)),
            "cudaFuncSetAttribute stage1 smem");
    }
    MlaDecodeStage1<T, DVREGS, kNTile><<<grid, kThreads, smem, s>>>(
        mid, query, kv_cache, block_table, seq_lens, q_s0, q_s1, c_s0, c_s1, bt_s0, head_size,
        v_head_dim, block_size, num_heads, valid_h, num_splits, mid_s0, mid_s1, mid_s2, scale);
  } else {
    const void* fn = reinterpret_cast<const void*>(&MlaDecodeStage1<T, DVREGS, kNTileSmall>);
    if (smem > 48u * 1024u) {
      Check(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)),
            "cudaFuncSetAttribute stage1 smem");
    }
    MlaDecodeStage1<T, DVREGS, kNTileSmall><<<grid, kThreads, smem, s>>>(
        mid, query, kv_cache, block_table, seq_lens, q_s0, q_s1, c_s0, c_s1, bt_s0, head_size,
        v_head_dim, block_size, num_heads, valid_h, num_splits, mid_s0, mid_s1, mid_s2, scale);
  }
}

// Port of `decode_attention_fwd_grouped` (triton_decode_attention.py:719-754):
// stage 1 then stage 2, no synchronization between them beyond the stream order.
template <typename T>
void LaunchMlaDecode(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                     const Tensor& kv_cache, const Tensor& block_table, const Tensor& seq_lens,
                     const MlaDecodeAttentionArgs& args) {
  const cudaStream_t s = AsStream(q);
  const int batch = static_cast<int>(query.shape[0]);
  const int num_heads = static_cast<int>(query.shape[1]);
  const int head_size = static_cast<int>(query.shape[2]);
  const int v_head_dim = static_cast<int>(out.shape[2]);
  const int block_size = static_cast<int>(kv_cache.shape[1]);

  const DeviceCaps& caps = GetDeviceCaps();
  int num_splits = args.num_kv_splits;
  if (num_splits <= 0) {
    num_splits = ComputeNumKvSplits(args.max_seq_len, caps.multiprocessor_count);
  }

  // The [B, Hq, S, Dv+1] f32 workspace — upstream's `logits_shape`
  // (triton_mla.py:228 `(B, q_num_heads, num_kv_splits, kv_lora_rank + 1)`).
  const int64_t mid_s2 = static_cast<int64_t>(v_head_dim) + 1;
  const int64_t mid_s1 = mid_s2 * num_splits;
  const int64_t mid_s0 = mid_s1 * num_heads;
  const size_t need = static_cast<size_t>(mid_s0) * static_cast<size_t>(batch) * sizeof(float);
  float* mid = EnsureMidScratch(need, s);

  // `:517-521` grid = (batch, cdiv(head_num, min(BLOCK_H, kv_group_num)), splits)
  // with kv_group_num == num_heads (MLA has exactly 1 KV head).
  const int valid_h = num_heads < kBlockH ? num_heads : kBlockH;
  const int head_tiles = (num_heads + valid_h - 1) / valid_h;
  const dim3 grid1(static_cast<unsigned>(batch), static_cast<unsigned>(head_tiles),
                   static_cast<unsigned>(num_splits));

  int n_tile = kNTile;
  size_t smem = static_cast<size_t>(kBlockH + n_tile) * static_cast<size_t>(head_size) *
                sizeof(float);
  if (smem > 48u * 1024u && !DynamicSmemFits(static_cast<long long>(smem))) {
    n_tile = kNTileSmall;
    smem = static_cast<size_t>(kBlockH + n_tile) * static_cast<size_t>(head_size) *
           sizeof(float);
  }
  VT_CHECK(smem <= 48u * 1024u || DynamicSmemFits(static_cast<long long>(smem)),
           "cuda mla_decode_attention: head_size too large for this device's shared memory");

  const auto* qp = query.Ptr<T>();
  const auto* cp = kv_cache.Ptr<T>();
  const int32_t* btp = block_table.Ptr<int32_t>();
  const int32_t* slp = seq_lens.Ptr<int32_t>();

#define VT_MLA_STAGE1(DVREGS)                                                                 \
  LaunchStage1<T, DVREGS>(s, grid1, smem, n_tile, mid, qp, cp, btp, slp, query.stride[0],     \
                          query.stride[1], kv_cache.stride[0], kv_cache.stride[1],            \
                          block_table.stride[0], head_size, v_head_dim, block_size,           \
                          num_heads, valid_h, num_splits, mid_s0, mid_s1, mid_s2, args.scale)
  if (v_head_dim <= 64) {
    VT_MLA_STAGE1(2);
  } else if (v_head_dim <= 128) {
    VT_MLA_STAGE1(4);
  } else if (v_head_dim <= 256) {
    VT_MLA_STAGE1(8);
  } else if (v_head_dim <= 512) {
    VT_MLA_STAGE1(16);
  } else if (v_head_dim <= 1024) {
    VT_MLA_STAGE1(32);
  } else {
    VT_CHECK(false, "cuda mla_decode_attention: v_head_dim > 1024 is not instantiated");
  }
#undef VT_MLA_STAGE1
  Check(cudaGetLastError(), "stage1 launch");

  // `:669` grid = (batch, head_num). One thread per output dim (upstream's
  // BLOCK_DV lanes), capped at 1024 with an outer chunk loop for wider V.
  int threads2 = ((v_head_dim + 31) / 32) * 32;
  if (threads2 > 1024) threads2 = 1024;
  if (threads2 < 32) threads2 = 32;
  const dim3 grid2(static_cast<unsigned>(batch), static_cast<unsigned>(num_heads));
  MlaDecodeStage2<T><<<grid2, static_cast<unsigned>(threads2), 0, s>>>(
      out.Ptr<T>(), lse != nullptr ? lse->Ptr<float>() : nullptr, mid, slp, out.stride[0],
      out.stride[1], lse != nullptr ? lse->stride[0] : 0, mid_s0, mid_s1, mid_s2, v_head_dim,
      num_splits);
  Check(cudaGetLastError(), "stage2 launch");
}

void MlaDecodeAttentionKernelCuda(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                                  const Tensor& kv_cache, const Tensor& block_table,
                                  const Tensor& seq_lens, const MlaDecodeAttentionArgs& args) {
  if (query.shape[0] == 0) return;
  switch (query.dtype) {
    case DType::kF32:
      LaunchMlaDecode<float>(q, out, lse, query, kv_cache, block_table, seq_lens, args);
      break;
    case DType::kBF16:
      LaunchMlaDecode<__nv_bfloat16>(q, out, lse, query, kv_cache, block_table, seq_lens, args);
      break;
    case DType::kF16:
      LaunchMlaDecode<__half>(q, out, lse, query, kv_cache, block_table, seq_lens, args);
      break;
    default: VT_CHECK(false, "cuda mla_decode_attention: unsupported dtype");
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(
        OpId::kMlaDecodeAttention, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<MlaDecodeAttentionFn>(&MlaDecodeAttentionKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
