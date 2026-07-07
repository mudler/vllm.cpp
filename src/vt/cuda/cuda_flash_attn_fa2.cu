// FlashAttention-2 prefill launcher (torch-free, sm_121a) — vendored from
// vllm-project/flash-attention @ 2c839c33 (the exact FA-2 source vLLM 0.24.0
// builds as _vllm_fa2_C and runs on GB10/sm_121; its flash_fwd_splitkv_kernel is
// what vLLM's own nsys trace shows for prefill). This TU is a thin, torch-free
// replacement for FA-2's flash_api.cpp: it fills Flash_fwd_params directly from
// our vt::Tensor views and calls the CUTLASS-templated splitkv dispatch. The
// kernel itself is compiled from the vendored flash_attn/src/*.cu instantiations.
//
// Requirements matched to our WMMA prefill path: head_dim 256, GQA (h_h_k_ratio),
// bf16, per-sequence causal, PAGED KV (block_table) + varlen (query_start_loc as
// cu_seqlens_q, seq_lens as seqused_k). Paged KV forces FA-2's split kernel with
// num_splits=1 (Split=false → the main kernel writes O directly; no combine pass),
// exactly as FA-2's mha_varlen_fwd does (run_mha_fwd(params, stream, paged_KV)).
//
// Gated at compile time by VLLM_CPP_FLASH_ATTN and at runtime by VT_ATTN_FA2
// (default OFF); the proven WMMA kernel remains the default prefill path.
#ifdef VLLM_CPP_FLASH_ATTN

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cutlass/numeric_types.h>  // cutlass::bfloat16_t

// Torch-free FA-2 params struct + the splitkv dispatch *declaration* (the
// definition lives in the vendored flash_attn/src/*.cu instantiations, linked in).
#include "namespace_config.h"
#include "flash.h"

#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cuda flash-attn-2 prefill: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

inline int RoundMultiple(int x, int m) { return (x + m - 1) / m * m; }

// vLLM/FA-2's split-KV heuristic, ported verbatim from flash-attn's
// flash_fwd_launch_template.h / flash_api.cpp num_splits_heuristic. Given the
// grid work (batch * nheads * num_m_blocks), the SM count, and the number of KV
// n-blocks, it picks num_splits>1 to fill idle SMs when the base grid is small.
// This is exactly the path vLLM's nsys trace shows (flash_fwd_splitkv_kernel,
// Split=TRUE) — for a large-grid prefill it returns 1 (main kernel writes O
// directly, no combine), for a small grid it returns >1 (KV split + combine).
inline int NumSplitsHeuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks,
                              int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split.
  if (batch_nheads_mblocks >= 0.8f * num_SMs) return 1;
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible: if the last split has fewer n-blocks than the
  // others, load balancing is poor — skip those (mirrors upstream).
  auto is_split_eligible = [&ceildiv, num_n_blocks](int num_splits) {
    return num_splits == 1 ||
           ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; ++num_splits) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = static_cast<float>(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / std::ceil(n_waves);
      if (eff > max_efficiency) max_efficiency = eff;
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; ++num_splits) {
    if (!is_split_eligible(num_splits)) continue;
    if (efficiency[num_splits - 1] >= 0.85f * max_efficiency) return num_splits;
  }
  return 1;
}

// Persistent (grow-only) device scratch for the split-KV accumulators. Mirrors
// vLLM's set_params_splitkv softmax_lse_accum / out_accum tensors:
//   oaccum   [num_splits, b, h, seqlen_q, d_rounded] f32
//   lseaccum [num_splits, b, h, seqlen_q]            f32
// Allocated once and reused across layers/forwards (not a per-call malloc).
struct SplitKvAccumPool {
  std::mutex mu;
  float* oaccum = nullptr;
  float* lseaccum = nullptr;
  size_t oaccum_floats = 0;
  size_t lseaccum_floats = 0;

  // Ensure both buffers hold at least the requested float counts (grow-only).
  void Ensure(size_t need_o, size_t need_lse) {
    std::lock_guard<std::mutex> lock(mu);
    if (need_o > oaccum_floats) {
      if (oaccum) cudaFree(oaccum);
      if (cudaMalloc(&oaccum, need_o * sizeof(float)) != cudaSuccess) {
        oaccum = nullptr;
        oaccum_floats = 0;
        throw std::runtime_error("cuda flash-attn-2 prefill: split-KV oaccum pool alloc failed");
      }
      oaccum_floats = need_o;
    }
    if (need_lse > lseaccum_floats) {
      if (lseaccum) cudaFree(lseaccum);
      if (cudaMalloc(&lseaccum, need_lse * sizeof(float)) != cudaSuccess) {
        lseaccum = nullptr;
        lseaccum_floats = 0;
        throw std::runtime_error("cuda flash-attn-2 prefill: split-KV lseaccum pool alloc failed");
      }
      lseaccum_floats = need_lse;
    }
  }
};

SplitKvAccumPool& AccumPool() {
  static SplitKvAccumPool pool;
  return pool;
}

// FA-2's kernel is bf16 in / bf16 out. The production full-attn prefill hands us
// an f32 query (residual-stream precision) and an f32 output tensor with a bf16
// KV cache (see qwen3_5.cpp FullAttnBlockPaged). We cast f32→bf16 on the way in
// and bf16→f32 on the way out so FA-2 engages for the f32-output path too; a
// native-bf16 query/out is used in place (no copy). Casts are flat over the
// contiguous [total_q, hq, d] buffers.
__global__ void CastF32ToBf16Kernel(const float* __restrict__ in,
                                    __nv_bfloat16* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}

__global__ void CastBf16ToF32Kernel(const __nv_bfloat16* __restrict__ in,
                                    float* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}

}  // namespace

// Launch FA-2 paged prefill for a bf16 query + bf16 KV cache, head_dim 256.
// Layouts (elements, matching our LaunchPaged dispatch):
//   query [total_q, hq, d]                          (varlen-packed)
//   k/v   [num_blocks, block_size, num_kv_heads, d] (paged, NHD block layout)
//   out   [total_q, hq, d]
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

  // FA-2 needs max_seqlen_q/max_seqlen_k on the host (grid + rounded dims). Small
  // D2H (num_reqs ints) + one stream sync — the same host-build shape our WMMA
  // prefill launcher already uses. query_start_loc is cumulative; seq_lens holds
  // the per-request K length (context + query) — our seqused_k.
  std::vector<int32_t> qsl(static_cast<size_t>(num_reqs) + 1);
  std::vector<int32_t> sk(static_cast<size_t>(num_reqs));
  Check(cudaMemcpyAsync(qsl.data(), query_start_loc.Ptr<int32_t>(),
                        sizeof(int32_t) * (num_reqs + 1), cudaMemcpyDeviceToHost, s),
        "query_start_loc D2H");
  Check(cudaMemcpyAsync(sk.data(), seq_lens.Ptr<int32_t>(), sizeof(int32_t) * num_reqs,
                        cudaMemcpyDeviceToHost, s),
        "seq_lens D2H");
  Check(cudaStreamSynchronize(s), "seqlen sync");
  int max_seqlen_q = 0;
  int max_seqlen_k = 0;
  for (int64_t i = 0; i < num_reqs; ++i) {
    max_seqlen_q = std::max(max_seqlen_q, qsl[i + 1] - qsl[i]);
    max_seqlen_k = std::max(max_seqlen_k, sk[i]);
  }
  if (max_seqlen_q == 0 || max_seqlen_k == 0) return;

  // softmax_lse [nheads, total_q] f32 (unpadded/varlen LSE). Written by the main
  // splitkv kernel; we don't consume it but the kernel requires a valid buffer.
  float* softmax_lse = nullptr;
  Check(cudaMallocAsync(&softmax_lse, sizeof(float) * hq * total_q, s), "softmax_lse alloc");

  // FA-2's kernel is bf16. Cast the query to bf16 when the caller hands us an f32
  // query (production full-attn prefill), and target a bf16 output buffer when the
  // caller's output is f32; both are cast back after. Native-bf16 in/out are used
  // in place. Buffers are contiguous [total_q, hq, d].
  const int64_t n_qo = total_q * hq * d;
  const int cast_threads = 256;
  const unsigned cast_blocks =
      static_cast<unsigned>((n_qo + cast_threads - 1) / cast_threads);

  __nv_bfloat16* q_bf = nullptr;  // owned bf16 query (only when query is f32)
  void* q_kernel_ptr = query.data;
  int64_t q_row_stride = query.stride[0];
  int64_t q_head_stride = query.stride[1];
  if (query.dtype == DType::kF32) {
    Check(cudaMallocAsync(&q_bf, sizeof(__nv_bfloat16) * n_qo, s), "fa2 q bf16 alloc");
    CastF32ToBf16Kernel<<<cast_blocks, cast_threads, 0, s>>>(query.Ptr<float>(), q_bf, n_qo);
    Check(cudaGetLastError(), "fa2 q f32->bf16 cast");
    q_kernel_ptr = q_bf;
    q_row_stride = hq * d;  // contiguous temp
    q_head_stride = d;
  }

  __nv_bfloat16* o_bf = nullptr;  // owned bf16 output (only when out is f32)
  void* o_kernel_ptr = out.data;
  int64_t o_row_stride = out.stride[0];
  int64_t o_head_stride = out.stride[1];
  const bool out_is_f32 = out.dtype == DType::kF32;
  if (out_is_f32) {
    Check(cudaMallocAsync(&o_bf, sizeof(__nv_bfloat16) * n_qo, s), "fa2 o bf16 alloc");
    o_kernel_ptr = o_bf;
    o_row_stride = hq * d;  // contiguous temp
    o_head_stride = d;
  }

  FLASH_NAMESPACE::Flash_fwd_params p{};  // zero-init: nulls knew/rotary/alibi/accum/leftpad
  p.is_bf16 = true;

  p.q_ptr = q_kernel_ptr;
  p.k_ptr = k_cache.data;
  p.v_ptr = v_cache.data;
  p.o_ptr = o_kernel_ptr;

  // Strides in ELEMENTS. Varlen q/o: no batch stride (cu_seqlens_q drives offset).
  p.q_row_stride = q_row_stride;
  p.q_head_stride = q_head_stride;
  p.o_row_stride = o_row_stride;
  p.o_head_stride = o_head_stride;
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

  p.is_causal = args.causal;
  p.window_size_left = -1;
  p.window_size_right = args.causal ? 0 : -1;
  p.is_seqlens_k_cumulative = true;  // ignored while cu_seqlens_k == nullptr
  p.is_rotary_interleaved = false;
  p.rotary_dim = 0;

  p.block_table = block_table.Ptr<int32_t>();
  p.block_table_batch_stride = block_table.stride[0];
  p.page_block_size = static_cast<int>(block_size);

  p.unpadded_lse = true;    // LSE is [nheads, total_q]
  p.seqlenq_ngroups_swapped = false;

  // -- SPLIT-KV heuristic (mirrors vLLM's set_params_splitkv) -----------------
  // The splitkv dispatch uses kBlockM=64 and, for hdim256, kBlockN=64. num_splits
  // splits the KV (n) dimension across extra CTAs so idle SMs get work when the
  // base grid (b * h * m_blocks) is small; the combine kernel then reduces the
  // partials into O. For a large prefill grid the heuristic returns 1 (Split=false,
  // main kernel writes O directly — identical to the pre-split behaviour).
  constexpr int kSplitKvBlockM = 64;                        // run_mha_fwd_splitkv_dispatch
  constexpr int kSplitKvBlockN = 64;                        // hdim256 -> kBlockN=64
  const int num_m_blocks = (max_seqlen_q + kSplitKvBlockM - 1) / kSplitKvBlockM;
  const int num_n_blocks = (max_seqlen_k + kSplitKvBlockN - 1) / kSplitKvBlockN;
  int num_SMs = 48;
  {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, dev);
  }
  const int batch_nheads_mblocks =
      static_cast<int>(num_reqs) * static_cast<int>(hq) * num_m_blocks;
  int num_splits = NumSplitsHeuristic(batch_nheads_mblocks, num_SMs, num_n_blocks, 128);

  // Optional override for A/B probing: VT_ATTN_FA2_SPLITS forces num_splits (the
  // heuristic returns 1 for large prefill grids, so this is the only way to
  // exercise/measure the Split=TRUE + combine path on those shapes). Clamped to
  // [1,128]. The uniform-seqlen benchmark keeps the combine's batched O-write
  // valid; ragged varlen would need the packed O-write the combine kernel lacks.
  if (const char* ov = std::getenv("VT_ATTN_FA2_SPLITS")) {
    int forced = std::atoi(ov);
    if (forced >= 1) num_splits = std::min(forced, 128);
  }
  num_splits = std::max(1, std::min(num_splits, num_n_blocks));  // can't split more than n-blocks
  p.num_splits = num_splits;

  // The combine kernel writes the final O with batched/padded addressing:
  //   o_ptr + batch_idx*o_batch_stride + head_idx*o_head_stride + row*o_row_stride
  // For our varlen-packed O [total_q,hq,d] this is exact only for UNIFORM seqlens
  // with o_batch_stride = seqlen_q * o_row_stride (row runs [0,seqlen_q) per seq).
  // It is ignored by the num_splits==1 direct path (that uses cu_seqlens_q), so
  // setting it here is harmless when no split occurs.
  p.o_batch_stride = static_cast<int64_t>(max_seqlen_q) * o_row_stride;

  float* oaccum = nullptr;
  float* lseaccum = nullptr;
  if (num_splits > 1) {
    // oaccum [num_splits,b,h,seqlen_q,d_rounded], lseaccum [num_splits,b,h,seqlen_q]
    // (+ one kBlockM row of margin per (b*h) so the last m-block's full-tile store
    // stays in-bounds). Persistent pool, reused across the 40 layers.
    const size_t bh = static_cast<size_t>(num_reqs) * static_cast<size_t>(hq);
    const size_t per_split_lse = bh * static_cast<size_t>(p.seqlen_q);
    const size_t per_split_o = per_split_lse * static_cast<size_t>(p.d_rounded);
    const size_t margin_o = bh * static_cast<size_t>(kSplitKvBlockM) *
                            static_cast<size_t>(p.d_rounded);
    const size_t margin_lse = bh * static_cast<size_t>(kSplitKvBlockM);
    AccumPool().Ensure(static_cast<size_t>(num_splits) * per_split_o + margin_o,
                       static_cast<size_t>(num_splits) * per_split_lse + margin_lse);
    oaccum = AccumPool().oaccum;
    lseaccum = AccumPool().lseaccum;
    p.oaccum_ptr = oaccum;
    p.softmax_lseaccum_ptr = lseaccum;
  }

  // One-shot diagnostic: report the gate shape and the num_splits actually used.
  static bool logged = false;
  if (!logged) {
    logged = true;
    std::fprintf(stderr,
                 "[FA2 split-KV] b=%lld hq=%lld d=%lld max_sq=%d max_sk=%d "
                 "m_blocks=%d n_blocks=%d SMs=%d batch*h*mblk=%d -> num_splits=%d%s\n",
                 static_cast<long long>(num_reqs), static_cast<long long>(hq),
                 static_cast<long long>(d), max_seqlen_q, max_seqlen_k, num_m_blocks,
                 num_n_blocks, num_SMs, batch_nheads_mblocks, num_splits,
                 std::getenv("VT_ATTN_FA2_SPLITS") ? " (forced)" : " (heuristic)");
  }

  // head_dim 256 is gated by the caller. Causal is per-layer (qwen3.5 mixes full-
  // attn causal / GDN); dispatch both instantiations (compiled for sm_121a).
  // run_mha_fwd_splitkv_dispatch runs the Split kernel AND, when num_splits>1, the
  // flash_fwd_splitkv_combine reduction kernel — exactly vLLM's path.
  if (args.causal) {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(p, s);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(p, s);
  }
  Check(cudaGetLastError(), "splitkv dispatch launch");

  // Cast the bf16 result back into the caller's f32 output (contiguous), and free
  // the scratch buffers (async on the same stream — safe, ordered after the cast).
  if (out_is_f32) {
    CastBf16ToF32Kernel<<<cast_blocks, cast_threads, 0, s>>>(o_bf, out.Ptr<float>(), n_qo);
    Check(cudaGetLastError(), "fa2 o bf16->f32 cast");
    Check(cudaFreeAsync(o_bf, s), "fa2 o bf16 free");
  }
  if (q_bf != nullptr) Check(cudaFreeAsync(q_bf, s), "fa2 q bf16 free");
  Check(cudaFreeAsync(softmax_lse, s), "softmax_lse free");
}

}  // namespace vt::cuda

#endif  // VLLM_CPP_FLASH_ATTN
