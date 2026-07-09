// vllm.cpp original (vt runtime, inventory deviation §9). NO upstream vLLM
// mirror — this is the Rung-2 tier of the PORTABLE tile-pipeline component
// ("what Triton does, internally"): the TMA (Tensor Memory Accelerator) bulk
// gmem->smem copy coordinated by an mbarrier producer/consumer state machine —
// the Blackwell-native async pipeline (cp.async.bulk.tensor + mbarrier) that
// Triton/CUTLASS emit on Hopper/Blackwell, superseding the sm80 cp.async ring
// (Rung-1, cp_async.cuh) which is what Ampere emits. Hand-transcribed 1:1 from
// CUTLASS C++ (NOT AOT-compiled from any DSL — see .agents/discipline.md
// "Kernel-DSL source is a PORTING REFERENCE"). Behind the portable vt:: seam:
// CUDA supplies the TMA+mbarrier atoms; other backends supply theirs; the CPU
// ref path is unaffected (this header is CUDA-only).
//
// Ported FROM (cutlass @ /home/mudler/cutlass_probe, v3.x):
//   - mbarrier.init                : cutlass/arch/barrier.h:397 (ClusterBarrier::init)
//   - mbarrier.arrive.expect_tx    : cutlass/arch/barrier.h:593 (ClusterTransactionBarrier::arrive_and_expect_tx)
//   - mbarrier.arrive (local)      : cutlass/arch/barrier.h:514 (ClusterBarrier::arrive)
//   - mbarrier.try_wait.parity spin: cutlass/arch/barrier.h:416-427 (ClusterBarrier::wait)
//   - fence.mbarrier_init          : cutlass/arch/barrier.h:717 (fence_barrier_init)
//   - cp.async.bulk.tensor.3d G2S  : cute/arch/copy_sm90_tma.hpp:159-191 (SM90_TMA_LOAD_3D,
//                                    CUTE_ARCH_TMA_SM120_ENABLED .shared::cta variant)
//   - cp.async.bulk.commit/wait    : cute/arch/copy_sm90_tma.hpp:1228,1251 (SM90_TMA_STORE / wait)
// Verified on GB10 sm_121a by a standalone probe (token-exact 3D tile load).
// Design record: .agents/tile-pipeline-component-2026-07-08.md (Rung-2).
#ifndef VT_CUDA_TILE_TMA_PIPELINE_CUH
#define VT_CUDA_TILE_TMA_PIPELINE_CUH

#include <cstdint>

namespace vt::cuda::tile {

// generic->shared address cast for the smem operands (barrier + TMA dst).
// Ported 1:1 from cute/arch/util.hpp cast_smem_ptr_to_uint().
__device__ __forceinline__ uint32_t tma_smem_u32(void const* ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
#endif
}

// mbarrier.init — arm a shared-memory barrier with an expected arrival count.
// Ported from barrier.h:390-402 ClusterBarrier::init.
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t arrive_count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t a = tma_smem_u32(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;\n" ::"r"(arrive_count), "r"(a) : "memory");
#endif
}

// fence.mbarrier_init — order the barrier inits before any arrive/copy that
// targets them (issued once after init, before the producer starts).
// Ported from barrier.h:711-719 fence_barrier_init.
__device__ __forceinline__ void mbarrier_init_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}

// fence.proxy.async — make prior generic-proxy smem writes (e.g. the barrier
// init) visible to the async (TMA) proxy before the copy is issued.
__device__ __forceinline__ void fence_proxy_async_shared() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

// mbarrier.arrive.expect_tx — combined arrive + set the expected transaction
// byte count for one pipeline stage. The paired TMA copy's completion delivers
// those bytes; the barrier flips phase when arrivals AND tx-bytes are satisfied.
// Ported from barrier.h:586-602 ClusterTransactionBarrier::arrive_and_expect_tx.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t a = tma_smem_u32(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n" ::"r"(bytes), "r"(a)
               : "memory");
#endif
}

// mbarrier.arrive — plain local arrive (no tx bytes). Used when a warp signals
// completion of a consumer phase. Ported from barrier.h:508-519 ClusterBarrier::arrive.
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t a = tma_smem_u32(bar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory");
#endif
}

// mbarrier.try_wait.parity spin — block until the barrier has completed the
// phase with the given parity. Ported 1:1 from barrier.h:409-432 ClusterBarrier::wait
// (the timeout-retry spin loop; ticks matches CUTLASS's 0x989680).
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t a = tma_smem_u32(bar);
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT_TMA:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra DONE_TMA;\n\t"
      "bra LAB_WAIT_TMA;\n\t"
      "DONE_TMA:\n\t"
      "}\n" ::"r"(a),
      "r"(phase), "r"(ticks)
      : "memory");
#endif
}

// cp.async.bulk.tensor.3d gmem->smem — the TMA bulk-tensor load. `desc` is a
// host-built CUtensorMap (passed by __grid_constant__ value; &desc is its gmem
// address); the copy's completion is signalled on `bar` (transaction bytes).
// crd0/crd1/crd2 are the tile's tensor coordinates (innermost-first). Ported 1:1
// from cute/arch/copy_sm90_tma.hpp:159-191 SM90_TMA_LOAD_3D; arch>=1200 uses the
// .shared::cta form (CUTE_ARCH_TMA_SM120_ENABLED), else the sm90 .shared::cluster.
__device__ __forceinline__ void tma_load_3d(void const* desc, uint64_t* bar, void* smem_dst,
                                             int32_t crd0, int32_t crd1, int32_t crd2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64_t d = reinterpret_cast<uint64_t>(desc);
  uint32_t m = tma_smem_u32(bar);
  uint32_t s = tma_smem_u32(smem_dst);
  uint64_t hint = 0;
#if (__CUDA_ARCH__ >= 1200)
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;\n" ::"r"(s),
      "l"(d), "r"(m), "r"(crd0), "r"(crd1), "r"(crd2), "l"(hint)
      : "memory");
#else
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;\n" ::"r"(s),
      "l"(d), "r"(m), "r"(crd0), "r"(crd1), "r"(crd2), "l"(hint)
      : "memory");
#endif
#endif
}

}  // namespace vt::cuda::tile

#endif  // VT_CUDA_TILE_TMA_PIPELINE_CUH
