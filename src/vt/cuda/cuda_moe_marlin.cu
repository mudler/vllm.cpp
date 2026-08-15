// vllm.cpp — Marlin NVFP4 W4A16 grouped-MoE GEMM drop-in (vt::Tensor launcher).
//
// This is the torch-free host launcher for the vendored Marlin MoE kernel
// (src/vt/cuda/marlin/, a 1:1 lift of vLLM's moe/marlin_moe_wna16 @ e24d1b24 —
// marlin-dropin-feasibility.md). It mirrors the NVFP4 branch of vLLM's
// moe_wna16_marlin_gemm (ops.cu:543): b_type=kFE2M1f + s_type=kFE4M3fn,
// group_blocks=1 (group size 16), bf16 activation/output. All the act-order /
// zero-point / bias / 8-bit-activation branches of the torch launcher are
// irrelevant to NVFP4 W4A16 and are dropped; the compute call into
// MARLIN_NAMESPACE_NAME::marlin_mm is the verbatim vendored dispatcher.
//
// Weights MUST be pre-repacked into Marlin's interleaved layout (b_q_weight)
// with processed fp8 block scales (b_scales) + per-expert global scale
// (global_scale) — see the load-time repack (mirror of
// marlin_utils_fp4.prepare_nvfp4_moe_layer_for_marlin). The align inputs
// (sorted_token_ids / expert_ids / num_tokens_past_padded / topk_weights) are
// vLLM's moe_align_block_size outputs.
//
// Isolated TU (heavy templated kernel). Gated by VT_MARLIN_NVFP4.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "core/scalar_type.hpp"
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_mm.h"

#include "vt/cuda/graph_safe_scratch.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: moe_marlin: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Persistent C_tmp workspace pool (VT_MARLIN_WS_POOL, default ON).
//
// vLLM allocates c_tmp — the fp32 global-reduce scratch — per moe_wna16_marlin_gemm
// call (torch::stable::new_empty, ops.cu:708-715), but that goes through PyTorch's
// CACHING device allocator, so it is a cheap pool hit, NOT a raw cudaMalloc. Our
// port had replaced it with a raw cudaMallocAsync + cudaFreeAsync PER GEMM (3 per
// MoE layer x num_layers x step). Stream-ordered async alloc/free serialize on the
// forward's host thread and were the #1 steady-state prefill idle (nsys: ~3893
// cudaMallocAsync ~1.1s host-time, ~242ms idle at the memset->Marlin boundary).
//
// This mirrors PyTorch's caching allocator with a persistent, grown-on-demand
// buffer PER STREAM, reused every call. c_tmp is scratch the kernel fully writes
// before it reads (vLLM uses new_empty — uninitialized; there is NO zero-on-entry
// invariant, unlike the `workspace` locks), so reuse is race-free under the
// forward's single-stream ordering: the prior GEMM's kernels complete before the
// next GEMM on the SAME stream is issued. The buffer grows monotonically and leaks
// at process exit (like the cublasLt workspace / the resident Marlin workspace).
// VT_MARLIN_WS_POOL=0 restores the per-call cudaMallocAsync/cudaFreeAsync (A/B).
bool MarlinWsPoolEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MARLIN_WS_POOL");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Ensure the per-stream c_tmp scratch holds >= `bytes`; regrow (async) if short.
float* EnsureCtmp(cudaStream_t s, size_t bytes) {
  struct Scratch {
    void* p = nullptr;
    size_t cap = 0;
  };
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, Scratch> pool;
  std::lock_guard<std::mutex> lk(mu);
  Scratch& sc = pool[s];
  if (bytes > sc.cap) {
    // RETIRE the old block instead of freeing it: this per-stream Marlin MoE c_tmp
    // reduce scratch pointer is baked into the captured pure-decode CUDA graph (it is
    // passed to marlin_mm, the faulting kernel a cuda-gdb catch pinned as
    // marlin_moe_wna16::Marlin<...>). A later, larger forward — a bigger co-scheduled
    // prefill or a larger decode batch, only reachable at concurrency > 1 — grows
    // c_tmp (its size is size_n * sorted_token_ids, which scales with the token
    // count); freeing the old block here dangles the captured graph's pointer, so the
    // next graph replay reads freed memory → Warp Illegal Address (surfaced at the
    // next cudaEventSynchronize). This is exactly the 35B c2+ online-serving crash;
    // single-stream (c1) never grows c_tmp after its one prefill, so it never hits it.
    // See graph_safe_scratch.h. (Growth is O(log); retired memory is negligible.)
    RetireGraphScratch(sc.p);
    Check(cudaMallocAsync(&sc.p, bytes, s), "cudaMallocAsync c_tmp (pool)");
    sc.cap = bytes;
  }
  return static_cast<float*>(sc.p);
}

// vt::MoeGroupedGemmNvfp4Marlin registered kernel.
void MoeGroupedGemmNvfp4MarlinKernelCuda(Queue& q, Tensor& c, const Tensor& a,
                                         const Tensor& b_q_weight, const Tensor& b_scales,
                                         const Tensor& global_scale, Tensor& workspace,
                                         const Tensor& sorted_token_ids, const Tensor& expert_ids,
                                         const Tensor& num_tokens_past_padded,
                                         const Tensor& topk_weights, const MoeMarlinArgs& args) {
  cudaStream_t s = AsStream(q);
  const int dev = q.device.index;

  // NVFP4 W4A16, bf16 activation/output (marlin_moe_wna16 generate_kernels.py:94),
  // OR MXFP4 W4A16 when args.mxfp4 (E8M0 scales => s_type kFE8M0fnu, group_size 32
  // => group_blocks 2, NO global scale). Mirrors vLLM's is_nvfp4 branch.
  const vllm::ScalarType a_type = vllm::kBFloat16;
  const vllm::ScalarType b_type = vllm::kFE2M1f;
  const vllm::ScalarType c_type = vllm::kBFloat16;
  const vllm::ScalarType s_type = args.mxfp4 ? vllm::kFE8M0fnu : vllm::kFE4M3fn;

  const int num_experts = static_cast<int>(b_q_weight.shape[0]);
  const int size_m = args.size_m;
  const int size_n = args.size_n;
  const int size_k = args.size_k;
  const int moe_block_size = args.moe_block_size;
  const int top_k = args.top_k;
  const int group_size = args.group_size;    // 16 (nvfp4) or 32 (mxfp4)
  const int num_groups = size_k / group_size;
  // MXFP4 has NO global scale — the kernel only reads global_scale_ptr under
  // (b_type==kFE2M1f && s_type==kFE4M3fn), so pass nullptr on the mxfp4 path.
  void* global_scale_ptr = args.mxfp4 ? nullptr : global_scale.data;

  int sms = -1;
  Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev),
        "cudaDeviceGetAttribute(sms)");

  // C_tmp for the fp32 global reduce (use_fp32_reduce && !use_atomic_add).
  //
  // We used to allocate vLLM's UPPER bound (size_n * sorted_token_ids) and note
  // that it is "always >= the capped size the kernel indexes". Safe, but not
  // free: upstream takes the MIN of that bound and a CTA-derived cap
  // (ops.cu:709-713), because the grid is at most `sms * 4` CTAs and each writes
  // at most `moe_block_size * max_thread_n` floats, so the cap IS the true
  // requirement. On the 35B/GB10 decode shape the difference is 15.3 MB vs
  // 3.15 MB (gate_up) and 30.5 MB vs 3.15 MB (down) -- 4.9x and 9.7x.
  //
  // That is not just wasted memory: the split-K reduce WRITES partial sums here
  // and READS them back, so an oversized buffer turns a cache-resident working
  // set into DRAM traffic on every launch. This kernel is DRAM-bound (measured
  // L2 hit rate 9.5%), which is why the same weight bytes were being read at
  // lower effective bandwidth than upstream (186.6 vs 210.7 GB/s, #442).
  const bool use_atomic_add = false;
  const bool use_fp32_reduce = true;
  const int64_t sorted_len = sorted_token_ids.shape[0];
  float* c_tmp = nullptr;
  int64_t c_tmp_elems = 0;
  bool c_tmp_pooled = false;
  if (use_fp32_reduce && !use_atomic_add) {
    // marlin.cuh:28 `max_thread_n` (256). Spelled locally because that constant
    // lives in the QUANTIZATION marlin namespace, not the MoE one this TU uses.
    constexpr int64_t kMarlinMaxThreadN = 256;
    const int64_t cta_cap =
        static_cast<int64_t>(sms) * 4 * moe_block_size * kMarlinMaxThreadN;
    // VT_MARLIN_CTMP_UNCAPPED=1 restores the pre-fix upper-bound sizing, so the
    // cap can be A/B'd WITHIN one session. GB10 cannot lock memory clocks
    // ("Setting locked Memory clocks is not supported"), so a DRAM-bound kernel's
    // absolute throughput drifts between sessions and a before/after taken across
    // two runs cannot attribute anything.
    static const bool ctmp_uncapped = [] {
      const char* v = std::getenv("VT_MARLIN_CTMP_UNCAPPED");
      return v != nullptr && v[0] == '1';
    }();
    const int64_t upper = static_cast<int64_t>(size_n) * sorted_len;
    c_tmp_elems = ctmp_uncapped ? upper : std::min(upper, cta_cap);
    if (moe_block_size == 8) c_tmp_elems *= 2;
    const size_t c_tmp_bytes = static_cast<size_t>(c_tmp_elems) * sizeof(float);
    if (MarlinWsPoolEnabled()) {
      c_tmp = EnsureCtmp(s, c_tmp_bytes);  // persistent per-stream, reused
      c_tmp_pooled = true;
    } else {
      Check(cudaMallocAsync(&c_tmp, c_tmp_bytes, s), "cudaMallocAsync c_tmp");
    }
  }

  MARLIN_NAMESPACE_NAME::marlin_mm(
      a.data, b_q_weight.data, c.data, c_tmp, /*b_bias=*/nullptr, /*a_s=*/nullptr,
      b_scales.data, global_scale_ptr, /*zp=*/nullptr, /*g_idx=*/nullptr, /*perm=*/nullptr,
      /*a_tmp=*/nullptr, sorted_token_ids.data, expert_ids.data, num_tokens_past_padded.data,
      topk_weights.data, moe_block_size, num_experts, top_k, args.mul_topk_weights, size_m,
      size_n, size_k, workspace.data, a_type, b_type, c_type, s_type, /*has_bias=*/false,
      /*has_act_order=*/false, /*is_k_full=*/true, /*has_zp=*/false, num_groups, group_size, dev,
      s, /*thread_k=*/-1, /*thread_n=*/-1, sms, /*blocks_per_sm=*/0, use_atomic_add,
      use_fp32_reduce, /*is_zp_float=*/false);

  if (c_tmp && !c_tmp_pooled) Check(cudaFreeAsync(c_tmp, s), "cudaFreeAsync c_tmp");
  Check(cudaGetLastError(), "moe_marlin marlin_mm launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMoeGroupedGemmNvfp4Marlin, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MoeGroupedGemmNvfp4MarlinFn>(&MoeGroupedGemmNvfp4MarlinKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda
