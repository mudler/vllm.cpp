// vllm.cpp original (vt runtime, inventory deviation §9). NO upstream vLLM
// mirror — this is the PORTABLE tile-pipeline component ("what Triton does,
// internally"): the async-copy software-pipelining primitives that give
// Triton/CUTLASS-grade gmem→smem staging, hand-transcribed 1:1 from CUTLASS C++
// (NOT AOT-compiled from any DSL — see .agents/discipline.md "Kernel-DSL source
// is a PORTING REFERENCE"). Hot kernels (GDN chunk, fused norm+quant, attn) are
// written against this so the pipelining expertise lives in ONE place and every
// backend supplies its own atoms (CUDA: cp.async/TMA; Metal: simdgroup; CPU: ref).
//
// Ported FROM (cutlass @ /home/mudler/cutlass_probe, v3.x):
//   - cp.async atom          : include/cute/arch/copy_sm80.hpp:40-193
//   - cast_smem_ptr_to_uint  : include/cute/arch/util.hpp (__cvta_generic_to_shared)
//   - PipelineState ring     : include/cutlass/pipeline/sm90_pipeline.hpp:171-250
// Design record: .agents/specs/tile-pipeline-component-2026-07-08.md
#ifndef VT_CUDA_TILE_CP_ASYNC_CUH
#define VT_CUDA_TILE_CP_ASYNC_CUH

#include <cstdint>

namespace vt::cuda::tile {

// sm80+ generic→shared address cast for the cp.async destination operand.
// Ported 1:1 from cutlass cute/arch/util.hpp cast_smem_ptr_to_uint().
__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(void const* ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
#endif
}

// cp.async atom — asynchronous, non-blocking gmem→smem copy. `pred=false`
// zero-fills the destination (ZFILL variant): src_size becomes 0 so the copy
// engine writes zeros for the out-of-bounds tail, which is exactly how CUTLASS/
// Triton mask partial tiles without a branch. `Bytes` is the per-thread transfer
// width (16 = one 128-bit vector, the coalesced sweet spot).
//
// CACHEGLOBAL (.cg): bypass L1, cache at L2 — for operands read once (the streamed
// K/V/W tiles). Requires 16-byte transfers. Ported from copy_sm80.hpp:99-152
// SM80_CP_ASYNC_CACHEGLOBAL_ZFILL.
template <int Bytes>
__device__ __forceinline__ void cp_async_cg(void* smem_dst, void const* gmem_src, bool pred) {
  static_assert(Bytes == 16, "cp.async.cg requires 16-byte transfers");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t smem = cast_smem_ptr_to_uint(smem_dst);
  int src_size = pred ? Bytes : 0;
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem),
               "l"(gmem_src), "n"(Bytes), "r"(src_size));
#endif
}

// CACHEALWAYS (.ca): cache at all levels — for operands re-read soon. Supports
// 4/8/16-byte transfers. Ported from copy_sm80.hpp:40-97 SM80_CP_ASYNC_CACHEALWAYS_ZFILL.
template <int Bytes>
__device__ __forceinline__ void cp_async_ca(void* smem_dst, void const* gmem_src, bool pred) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp.async.ca: 4/8/16 bytes only");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t smem = cast_smem_ptr_to_uint(smem_dst);
  int src_size = pred ? Bytes : 0;
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem),
               "l"(gmem_src), "n"(Bytes), "r"(src_size));
#endif
}

// Establishes ordering: batches all cp.async issued since the last fence into one
// "commit group". Non-blocking. Ported from copy_sm80.hpp:158-168 cp_async_fence().
__device__ __forceinline__ void cp_async_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Blocks the thread until all but the N most-recent commit groups have landed in
// smem — i.e. keeps N groups in flight (the depth of the software pipeline).
// N==0 waits for all. Ported from copy_sm80.hpp:173-193 cp_async_wait<N>().
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
#endif
}

// Ring-buffer state for an S-stage pipeline: which smem buffer to touch (index),
// the phase bit (flips each time the ring wraps — used by mbarrier waits in the
// Rung-2 TMA pipeline), and a monotone iteration count. Mirrors cutlass
// PipelineState<Stages> (sm90_pipeline.hpp:171-250) so the Rung-2 TMA/mbarrier
// pipeline is a drop-in extension, not a rewrite.
template <int Stages_>
struct PipelineState {
  static constexpr int Stages = Stages_;
  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  __device__ PipelineState() {}
  __device__ PipelineState(int index, uint32_t phase, uint32_t count)
      : index_(index), phase_(phase), count_(count) {}

  __device__ int index() const { return index_; }
  __device__ uint32_t phase() const { return phase_; }
  __device__ uint32_t count() const { return count_; }

  __device__ void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }
};

}  // namespace vt::cuda::tile

#endif  // VT_CUDA_TILE_CP_ASYNC_CUH
