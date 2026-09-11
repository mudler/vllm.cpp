// ROCm `vt::OpId::kMoeGateUpSwiGLUGrouped` — the fused MoE gate+up+clamped-SwiGLU
// keep-quant grouped GEMM (BACKEND-ROCM, #2942,
// `.agents/specs/rocm-moe-gateup-swiglu-grouped.md`).
//
// WHY THIS OP. `glm5_next_forward.cpp:307-310` asks the op table for exactly two
// ops on any non-CPU queue, before it builds a single operand:
// `kMoeGateUpSwiGLUGrouped` and `kMatmulBTQuantGrouped`. ROCm has had the second
// since `rocm_ops.hip:261`. It did not have the first, so `:315-327` refused
// every ROCm queue by name and GLM-5.3-Flash could only run `--device cpu`.
//
// DONOR: `src/vt/cpu/cpu_quant_gemm.cpp:282-299`, `MoeGateUpSwiGLUGroupedKernel`
// — the CPU GOLDEN, and NOT `src/vt/cuda/cuda_quant_dot.cu`'s 2923 lines. That
// is #2942's own load-bearing instruction, and the reason is visible in the two
// donors side by side. The CUDA kernel
// (`cuda_quant_dot.cu:1104-1142`) fuses both dot products into one warp and
// reduces them with
//
//     for (int off = 16; off > 0; off >>= 1) pg += __shfl_down_sync(0xffffffffu, pg, off);
//
// which is a 32-lane assumption. AMD's wavefront is 64 lanes
// (`rocm_rmsnorm.hip:39-44`), so that tree does not survive translation, and
// gfx1151 is RDNA and has no MFMA to replace it with. The CPU golden has no such
// construct to translate, because it is a COMPOSITE: two grouped keep-quant
// GEMMs through the SAME `MatmulBTQuantGroupedKernel` into f32 temporaries, then
// one elementwise pass
//
//     gate = min(g, limit);  up = clamp(u, -limit, limit);
//     out  = gate * sigmoid(gate) * up
//
// with NO extra scale, because the grouped GEMM has already folded the weight
// `FinalFactor` into `g` and `u`. `include/vt/merged_gemm.h:117` states the same
// polarity from the other side: the fused op is DEFINED as bit-identical to that
// composite.
//
// So this file is that composite. It calls the already-landed, already-gated
// `MatmulBTQuantGroupedKernelRocm` twice and adds ONE new kernel — the epilogue.
// Writing a second copy of the Q8_0 / Q4_K / Q5_K / Q6_K dot cores here to fuse
// the two dots would be the parallel path the protocol forbids, and it would be
// a SPEED change, not a correctness one. It is recorded as owed in the spec.
//
// The epilogue is ELEMENTWISE. It contains no cross-lane reduction at all, so
// there is no wavefront width to get wrong and no warp-level primitive is used
// anywhere in this file.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/grow_only_stream_scratch.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"

namespace vt::rocm {

// Defined in rocm_grouped_gemm.hip. Declared here rather than in a header
// because that is how this backend's TUs reach each other (rocm_ops.hip:97 does
// the same for the same symbol); there is no `src/vt/rocm/*.h`.
void MatmulBTQuantGroupedKernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                    const Tensor& weight, const Tensor& expert_ids);

namespace {

constexpr int kBlock = 256;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm moe_gate_up_swiglu: ") + what + ": " +
                             hipGetErrorString(err));
}
inline hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }
inline unsigned GridFor(int64_t n) {
  if (n <= 0) return 1;
  const int64_t g = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(g > 65535 ? 65535 : g);
}

// Grow-only, per-stream f32 scratch for the two GEMM temporaries. Same
// discipline as `rocm_grouped_gemm.hip:606-635` and for the same reason:
// hipMalloc/hipFree/hipStreamSynchronize per call are ILLEGAL under hipGraph
// stream capture, and a captured graph may have baked the pointer, so the pool
// allocates stream-ordered and never frees during the process.
//
// A SEPARATE pool from that file's `EnsureQuantScratch`, deliberately. That
// block holds the QUANTIZED ACTIVATION and is rewritten by each of the two
// grouped-GEMM calls below; sharing it would overwrite the gate result while the
// up GEMM ran.
vt::GrowOnlyStreamScratch<hipStream_t>& GateUpScratchPool() {
  static vt::GrowOnlyStreamScratch<hipStream_t> pool;
  return pool;
}

void* EnsureGateUpScratch(size_t need, hipStream_t s) {
  return GateUpScratchPool().Ensure(s, need, [s](size_t bytes) {
    void* block = nullptr;
    Check(hipMallocAsync(&block, bytes, s), "gate/up scratch grow");
    return block;
  });
}

// The clamped-SwiGLU epilogue — `cpu_quant_gemm.cpp:294-298` transcribed, and
// bit-for-bit the same expression as `cuda_quant_dot.cu:1138-1141`
// (alpha=1, beta=0). `limit = +inf` collapses both clamps to no-ops and leaves
// plain silu(g)*u, which is the standard SwiGLU MLP.
//
// f32 throughout, which is not a widening: `out` is f32 BY CONTRACT
// (`ops.cpp:269-270` refuses anything else) because this value is the routed
// experts' `adown` operand, and the grouped down-projection that consumes it
// reads f32.
__global__ void MoeGateUpSwiGLUEpilogueK(float* __restrict__ out,
                                         const float* __restrict__ g,
                                         const float* __restrict__ u, int64_t n,
                                         float limit) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    const float gate = fminf(g[i], limit);
    const float up = fminf(fmaxf(u[i], -limit), limit);
    out[i] = gate * (1.0f / (1.0f + expf(-gate))) * up;
  }
}

}  // namespace

void MoeGateUpSwiGLUGroupedKernelRocm(Queue& q, Tensor& out, const Tensor& act,
                                      const Tensor& gate_w, const Tensor& up_w,
                                      const Tensor& expert_ids, float limit) {
  EnsureQueueDevice(q);
  // The seam (`ops.cpp:256-285`) has already validated rank, shapes, the
  // same-block-quant dtype across both towers, `K % BlockElems == 0`, packed
  // activation rows, contiguity and device match. These two are re-asserted here
  // anyway because `Tensor::Ptr<T>()` is an UNCHECKED cast — the hazard the #509
  // review sweep found in `rocm_moe_chain.hip:141-146` — and because the
  // grouped-GEMM call below would otherwise write bf16 into an f32 read.
  VT_CHECK(out.dtype == DType::kF32,
           "rocm moe_gate_up_swiglu: f32 output required");
  VT_CHECK(gate_w.dtype == up_w.dtype,
           "rocm moe_gate_up_swiglu: gate_w/up_w must be the SAME block-quantized dtype");

  const int64_t P = out.shape[0], N = out.shape[1];
  const int64_t n = P * N;
  if (n == 0) return;

  hipStream_t s = AsStream(q);
  // One block, two halves: the up half starts at `n` floats in. One allocation
  // rather than two, so the pool grows once.
  float* scratch = static_cast<float*>(
      EnsureGateUpScratch(static_cast<size_t>(n) * 2 * sizeof(float), s));
  float* g = scratch;
  float* u = scratch + n;

  Tensor gt = Tensor::Contiguous(g, DType::kF32, out.device, {P, N});
  Tensor ut = Tensor::Contiguous(u, DType::kF32, out.device, {P, N});
  // The SAME kernel the CPU golden composes, so the arm this file adds is the
  // epilogue and nothing else. Both calls quantize `act` into the grouped GEMM's
  // own scratch; they are stream-ordered, so the second sees a correctly
  // rewritten buffer. The duplicated quantization is redundant work, priced
  // under the spec's `## Owed` as the fused single-kernel wave.
  MatmulBTQuantGroupedKernelRocm(q, gt, act, gate_w, expert_ids);
  MatmulBTQuantGroupedKernelRocm(q, ut, act, up_w, expert_ids);

  MoeGateUpSwiGLUEpilogueK<<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), g, u, n, limit);
  Check(hipGetLastError(), "gate_up_swiglu epilogue launch");
}

}  // namespace vt::rocm
