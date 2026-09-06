// ROCm RmsNorm — the ONE op the W0 skeleton registers (BACKEND-ROCM, W0).
//
// **UNBUILT.** Never compiled: no AMD GPU on the authoring machine. See the
// header of rocm_backend.hip.
//
// PORTED FROM: src/vt/cuda/cuda_ops.cu:96-126 `RmsNormRowKernel` (one block per
// row, shared-memory f32 tree reduction), whose own upstream counterpart is
// vLLM csrc/layernorm_kernels.cu `rms_norm_kernel` / `fused_add_rms_norm_kernel`.
// This is what a hipify pass produces for that kernel: __nv_bfloat16 becomes
// __hip_bfloat16, the intrinsics change spelling, the algorithm does not change
// at all. Diff the two files side by side — that is the point of writing it out.
//
// WHY THIS OP AND NOT A SIMPLER ONE. tests/vt/test_backend_cross_device.cpp runs
// against every registered non-CPU backend and SKIPS ops a device has not
// registered, comparing the rest against the CPU backend at NMSE <= 5e-4 from
// the same binary on the same inputs. So registering RmsNorm here means a
// contributor gets a real numeric gate on their own silicon without writing a
// test. RmsNorm specifically because it is a CROSS-LANE REDUCTION, which is
// where a hipified kernel can actually be wrong — AMD's wavefront is 64 lanes
// where NVIDIA's warp is 32, so a kernel that quietly assumed 32 breaks here.
// This one does not assume: the reduction is a __syncthreads() tree over shared
// memory with no warp-level primitives, so it is wavefront-width agnostic. An
// elementwise op like Add would have passed the same gate while proving nothing.
//
// NOT PORTED, deliberately: the VT_RMSNORM_DECODE_FAST paths
// (cuda_ops.cu:421-437). They exist to be BIT-IDENTICAL to this kernel while
// being faster on NVIDIA, they are tuned against sm_121a, and W0 is not a speed
// milestone. RmsNormQuantFp8 and the gated variants are likewise out of scope.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <string>
#include <type_traits>

#include "vt/cpu/cpu_quant_blocks.h"
#include "vt/rocm/rocm_act_quant.h"
#include "vt/rocm/rocm_norm_quant_bridge.h"

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"

namespace vt::rocm {

using vt::cpu::kQK_K;
namespace {

// Block width, from cuda_ops.cu:24. Kept at 256 rather than raised to a multiple
// of AMD's 64-lane wavefront: 256 is already 4 whole wavefronts, and matching the
// CUDA kernel's block width keeps the REDUCTION ORDER identical, which is what
// keeps the two backends' outputs comparable at the NMSE bar.
constexpr int kBlock = 256;

__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline float Load(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, as the CUDA path and the host do
}

// Round-trip a f32 through the residual store dtype so the variance squares the
// SAME rounded value that is written back — mirrors vLLM's fused_add_rms_norm,
// whose bf16 residual add rounds to model dtype before the f32 variance.
// Identity for a f32 residual. cuda_ops.cu:85-93.
template <typename Tres>
__device__ inline float ResRound(float v);
template <>
__device__ inline float ResRound<float>(float v) {
  return v;
}
template <>
__device__ inline float ResRound<__hip_bfloat16>(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

// Lever C: `q8_out` (nullptr on every pre-existing path) turns the kernel into
// a PRODUCER of Q8_K activation scratch: after the output row is stored, one
// thread per 256-wide superblock requantizes the STORED bf16/f32 row through
// the SAME shared QuantQ8KSBlock body the standalone QuantizeQ8KK uses, so
// the scratch is byte-identical to the standalone path by construction
// (asserted in tests/vt/test_rocm_quant_dot.cpp). Requires nsb <= kBlock: one
// thread per superblock within this single-row block.
template <typename Tin, typename Tw, typename Tout, typename Tres>
__global__ void RmsNormRowKernel(Tout* out, const Tin* x, const Tw* w, Tres* residual, int64_t h,
                                 float eps, bool gemma, vt::cpu::BlockQ8_K* q8_out, int nsb,
                                 ActDT q8_adt) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * h;
  Tout* orow = out + row * h;
  Tres* rrow = residual == nullptr ? nullptr : residual + row * h;

  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float v = Load(xrow, j);
    if (rrow != nullptr) {
      v = ResRound<Tres>(v + Load(rrow, j));  // new residual stream: f32 add, round to Tres
      Store(rrow, j, v);                      // updated in place
    }
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v = rrow != nullptr ? Load(rrow, j) : Load(xrow, j);
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(orow, j, v * inv * wj);
  }
  if (q8_out != nullptr) {
    // Barrier first: the epilogue reads the WHOLE stored row (written by all
    // 256 threads) back from global memory, exactly as the standalone
    // QuantizeQ8KK would -- same bytes, same serial per-superblock walk, so
    // the amax first-occurrence tie-break and every scale bit match.
    __syncthreads();
    const int sb = static_cast<int>(threadIdx.x);
    if (sb < nsb)
      QuantQ8KSBlock(q8_out[static_cast<int64_t>(row) * nsb + sb], orow, q8_adt,
                     static_cast<int64_t>(sb) * kQK_K);
  }
}

// --- T8 (GFX1100-TG200): cooperative single-row remap -----------------------
// Decode launches ONE 256-thread block per row; the ported body chains three
// strided scalar passes, a nine-step __syncthreads shared-memory tree, and a
// fused-q8 epilogue whose per-superblock QuantQ8KSBlock walks 256 elements
// serially on ONE thread (nsb<=10 of 256 threads active). rocpd prices the
// fused instantiation at 65 launches/tok x 18.1us = 1.178 ms/tok against a
// microsecond-class floor. This arm keeps the grid shape and the byte
// contracts but rebuilds the internals, behind VT_RMSNORM_ROW_COOP=1
// (default OFF leaves every path byte-unchanged):
//
//   1. Reduction: two wavefront shfl_down trees + one cross-wavefront
//      combine through shared memory -- TWO __syncthreads instead of nine.
//      The float association CHANGES (recorded adjudication required; the
//      opt-in flag rides the campaign config like GDN_SCAN_COOP).
//   2. Vector passes: 16-byte loads/stores where the row base allows,
//      scalar fallback otherwise (uniform per launch).
//   3. Cooperative q8 epilogue: the whole block quantizes ONE superblock
//      at a time, thread i owning element i. BYTE CONTRACT PRESERVED BY
//      CONSTRUCTION: the (mx, amax) pair comes from a LEFT-BIASED max over
//      ascending element positions (shfl trees keep the earlier element on
//      magnitude ties, which is exactly the scalar scan's first-occurrence
//      rule), iscale/DNearestInt/clamp arithmetic is verbatim, and bsums
//      stay exact integer sums. Asserted against the standalone quantizer
//      by the focused test under BOTH flag states.
template <typename Tin, typename Tw, typename Tout, typename Tres>
__global__ void RmsNormRowCoopKernel(Tout* out, const Tin* x, const Tw* w, Tres* residual,
                                     int64_t h, float eps, bool gemma,
                                     vt::cpu::BlockQ8_K* q8_out, int nsb, ActDT q8_adt,
                                     bool lds_quant) {
  const int64_t row = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);
  const Tin* xrow = x + row * h;
  Tout* orow = out + row * h;
  Tres* rrow = residual == nullptr ? nullptr : residual + row * h;
  const int waveSz = warpSize;  // runtime width (32 on RDNA default); never hardcode
  const int kWaves = kBlock / waveSz;
  __shared__ float partial[kBlock];
  // Statically sized for the narrowest wavefront (32); kWaves is runtime.
  __shared__ float wave_mx[kBlock / 32];
  __shared__ float wave_amax[kBlock / 32];
  extern __shared__ char dyn_smem[];
  Tout* lds_out = reinterpret_cast<Tout*>(dyn_smem);
  // The vector body decodes the gamma as 8 bf16 halves per uint4, so a
  // non-bf16 gamma (kF32/kF16 are admitted by DispatchRmsNormWeight,
  // independent of Tin) MUST take the scalar path, whose Load() overloads
  // decode per dtype. sizeof(Tw)==2 alone would still misread an f16 gamma.
  const bool vec =
                   std::is_same_v<Tw, __hip_bfloat16> && sizeof(Tin) == 2 &&
                   (rrow == nullptr || sizeof(Tres) == 2) &&
                   (reinterpret_cast<uintptr_t>(xrow) & 15u) == 0 &&
                   (reinterpret_cast<uintptr_t>(orow) & 15u) == 0 &&
                   (rrow == nullptr || (reinterpret_cast<uintptr_t>(rrow) & 15u) == 0);

  // Pass 1: sum of squares (+ residual fold-in). Per-thread element set is
  // the plain kernel's; the vector path regroups WITHIN a thread's step.
  float acc = 0.0f;
  if (vec && sizeof(Tin) == 2) {
    const uint4* xv = reinterpret_cast<const uint4*>(xrow);
    uint4* rv = reinterpret_cast<uint4*>(rrow);
    const int step8 = kBlock * 8;  // elements per block-step (bf16)
    for (int j8 = tid * 8; j8 < static_cast<int>(h); j8 += step8) {
      const int elems = j8 + 8 <= static_cast<int>(h) ? 8 : static_cast<int>(h) - j8;
      if (elems == 8) {
        const uint4 rawx = xv[j8 / 8];
        uint16_t hx[8] = {static_cast<uint16_t>(rawx.x & 0xFFFF), static_cast<uint16_t>(rawx.x >> 16),
                          static_cast<uint16_t>(rawx.y & 0xFFFF), static_cast<uint16_t>(rawx.y >> 16),
                          static_cast<uint16_t>(rawx.z & 0xFFFF), static_cast<uint16_t>(rawx.z >> 16),
                          static_cast<uint16_t>(rawx.w & 0xFFFF), static_cast<uint16_t>(rawx.w >> 16)};
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          float fv = DBF16ToF32(hx[u]);
          if (rrow != nullptr) {
            fv = ResRound<Tres>(fv + Load(rrow, j8 + u));
          }
          v[u] = fv;
        }
        if (rrow != nullptr) {
          uint4 rout;
          uint16_t hout[8];
          for (int u = 0; u < 8; ++u) hout[u] = DF32ToBF16(v[u]);
          rout.x = static_cast<unsigned>(hout[0]) | (static_cast<unsigned>(hout[1]) << 16);
          rout.y = static_cast<unsigned>(hout[2]) | (static_cast<unsigned>(hout[3]) << 16);
          rout.z = static_cast<unsigned>(hout[4]) | (static_cast<unsigned>(hout[5]) << 16);
          rout.w = static_cast<unsigned>(hout[6]) | (static_cast<unsigned>(hout[7]) << 16);
          rv[j8 / 8] = rout;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) acc += v[u] * v[u];
      } else {
        for (int j = j8; j < j8 + elems; ++j) {
          float v = Load(xrow, j);
          if (rrow != nullptr) {
            v = ResRound<Tres>(v + Load(rrow, j));
            Store(rrow, j, v);
          }
          acc += v * v;
        }
      }
    }
  } else {
    for (int64_t j = tid; j < h; j += kBlock) {
      float v = Load(xrow, j);
      if (rrow != nullptr) {
        v = ResRound<Tres>(v + Load(rrow, j));
        Store(rrow, j, v);
      }
      acc += v * v;
    }
  }

  // Two-level reduction: wavefront shfl trees, one cross-wavefront combine.
  float wacc = acc;
#pragma unroll
  for (int off = waveSz / 2; off > 0; off >>= 1)
    wacc += __shfl_down_sync(0xffffffffULL, wacc, off);
  if (tid % waveSz == 0) partial[tid / waveSz] = wacc;
  __syncthreads();
  if (tid == 0) {
    float t = partial[0];
#pragma unroll
    for (int wv = 1; wv < kWaves; ++wv) t += partial[wv];
    partial[0] = t;
  }
  __syncthreads();
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);

  // Pass 3: scaled output.
  if (vec && sizeof(Tin) == 2 && sizeof(Tout) == 2) {
    const uint4* xv = reinterpret_cast<const uint4*>(
        rrow != nullptr ? static_cast<const void*>(rrow) : static_cast<const void*>(xrow));
    uint4* ov = reinterpret_cast<uint4*>(orow);
    const uint4* wv = reinterpret_cast<const uint4*>(w);
    const bool waligned = (reinterpret_cast<uintptr_t>(w) & 15u) == 0;
    for (int j8 = tid * 8; j8 < static_cast<int>(h); j8 += kBlock * 8) {
      const int elems = j8 + 8 <= static_cast<int>(h) ? 8 : static_cast<int>(h) - j8;
      if (elems == 8 && waligned) {
        const uint4 rawx = xv[j8 / 8];
        const uint4 raww = wv[j8 / 8];
        uint16_t hx[8] = {static_cast<uint16_t>(rawx.x & 0xFFFF), static_cast<uint16_t>(rawx.x >> 16),
                          static_cast<uint16_t>(rawx.y & 0xFFFF), static_cast<uint16_t>(rawx.y >> 16),
                          static_cast<uint16_t>(rawx.z & 0xFFFF), static_cast<uint16_t>(rawx.z >> 16),
                          static_cast<uint16_t>(rawx.w & 0xFFFF), static_cast<uint16_t>(rawx.w >> 16)};
        uint16_t hw[8] = {static_cast<uint16_t>(raww.x & 0xFFFF), static_cast<uint16_t>(raww.x >> 16),
                          static_cast<uint16_t>(raww.y & 0xFFFF), static_cast<uint16_t>(raww.y >> 16),
                          static_cast<uint16_t>(raww.z & 0xFFFF), static_cast<uint16_t>(raww.z >> 16),
                          static_cast<uint16_t>(raww.w & 0xFFFF), static_cast<uint16_t>(raww.w >> 16)};
        uint16_t ho[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          float wj = DBF16ToF32(hw[u]);
          if (gemma) wj += 1.0f;
          ho[u] = DF32ToBF16(DBF16ToF32(hx[u]) * inv * wj);
        }
        uint4 o;
        o.x = static_cast<unsigned>(ho[0]) | (static_cast<unsigned>(ho[1]) << 16);
        o.y = static_cast<unsigned>(ho[2]) | (static_cast<unsigned>(ho[3]) << 16);
        o.z = static_cast<unsigned>(ho[4]) | (static_cast<unsigned>(ho[5]) << 16);
        o.w = static_cast<unsigned>(ho[6]) | (static_cast<unsigned>(ho[7]) << 16);
        ov[j8 / 8] = o;
      } else {
        for (int j = j8; j < j8 + elems; ++j) {
          const float v = rrow != nullptr ? Load(rrow, j) : Load(xrow, j);
          float wj = Load(w, j);
          if (gemma) wj += 1.0f;
          Store(orow, j, v * inv * wj);
        }
      }
    }
  } else {
    for (int64_t j = tid; j < h; j += kBlock) {
      const float v = rrow != nullptr ? Load(rrow, j) : Load(xrow, j);
      float wj = Load(w, j);
      if (gemma) wj += 1.0f;
      Store(orow, j, v * inv * wj);
    }
  }

  if (q8_out != nullptr) {
    // Barrier: the epilogue reads the WHOLE stored row back, exactly as the
    // standalone QuantizeQ8KK would.
    __syncthreads();
    __shared__ int8_t sq[kQK_K];
    __shared__ float lead_iscale;
    const void* quant_src = lds_quant ? static_cast<const void*>(lds_out)
                                      : static_cast<const void*>(orow);
    for (int sb = 0; sb < nsb; ++sb) {
      vt::cpu::BlockQ8_K& y = q8_out[static_cast<int64_t>(row) * nsb + sb];
      const float x = DLoadAct(quant_src, q8_adt, static_cast<int64_t>(sb) * kQK_K + tid);
      // Seed 0 like the standalone QuantizeQ8KK (strict > keeps the first
      // occurrence and rejects NaN): seeding from x let a NaN element poison
      // d and diverge from the scalar contract's zero block.
      float mx = 0.0f, amax = 0.0f;
      if (x == x) { mx = x; amax = fabsf(x); }
      for (int off = waveSz / 2; off > 0; off >>= 1) {
        const float ox = __shfl_down_sync(0xffffffffULL, mx, off);
        const float oa = __shfl_down_sync(0xffffffffULL, amax, off);
        if (oa > amax) { amax = oa; mx = ox; }
      }
      if (tid % waveSz == 0) {
        wave_mx[tid / waveSz] = mx;
        wave_amax[tid / waveSz] = amax;
      }
      __syncthreads();
      if (tid == 0) {
        float bmx = wave_mx[0], bamax = wave_amax[0];
#pragma unroll
        for (int wv = 1; wv < kWaves; ++wv) {
          if (wave_amax[wv] > bamax) { bamax = wave_amax[wv]; bmx = wave_mx[wv]; }
        }
        lead_iscale = bamax == 0.0f ? 0.0f : -127.0f / bmx;
      }
      __syncthreads();
      if (lead_iscale == 0.0f) {
        if (tid < kQK_K) y.qs[tid] = 0;
        if (tid < kQK_K / 16) y.bsums[tid] = 0;
        if (tid == 0) y.d = 0.0f;
      } else {
        const int qv = DNearestInt(lead_iscale * x);
        const int8_t q = static_cast<int8_t>(qv < 127 ? qv : 127);
        y.qs[tid] = q;
        sq[tid] = q;
        __syncthreads();
        if (tid < kQK_K / 16) {
          int sum = 0;
#pragma unroll
          for (int ii = 0; ii < 16; ++ii) sum += sq[tid * 16 + ii];
          y.bsums[tid] = static_cast<int16_t>(sum);
        }
        if (tid == 0) y.d = 1.0f / lead_iscale;
      }
      __syncthreads();  // sq reuse guard across superblocks
    }
  }
}

template <typename Tin, typename Tw, typename Tout>
void LaunchRmsNormRes(hipStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const RmsNormArgs& args, Tensor* residual, unsigned rows, int64_t h,
                      vt::cpu::BlockQ8_K* q8_out) {
  const int nsb = q8_out != nullptr ? static_cast<int>(h / kQK_K) : 0;
  // T8 opt-in arm (read PER CALL like the sibling flags): the cooperative
  // remap requires one 256-wide superblock per thread and h a multiple of
  // kQK_K; anything else falls back to the ported body.
  const char* coop_e = std::getenv("VT_RMSNORM_ROW_COOP");
  const bool coop = coop_e != nullptr && coop_e[0] == '1' && coop_e[1] == '\0' &&
                    h % kQK_K == 0 && h / kQK_K <= kBlock;
  if (coop) {
    // T24: LDS-buffered quant epilogue (VT_RMSNORM_LDS_QUANT, default ON).
    // Stores the normalized row to LDS during Pass 3 so the Q8_K quant
    // epilogue reads from LDS instead of re-reading global memory.
    const char* lds_e = std::getenv("VT_RMSNORM_LDS_QUANT");
    const bool lds_quant =
        lds_e == nullptr || lds_e[0] == '1';  // default ON when q8_out active
    const size_t smem =
        (q8_out != nullptr && lds_quant) ? static_cast<size_t>(h) * sizeof(Tout) : 0;
    if (residual != nullptr && residual->dtype == DType::kBF16) {
      RmsNormRowCoopKernel<Tin, Tw, Tout, __hip_bfloat16><<<rows, kBlock, smem, s>>>(
          out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tw>(), residual->Ptr<__hip_bfloat16>(), h,
          args.eps, args.gemma, q8_out, nsb, ActDtOf(out.dtype), lds_quant);
    } else {
      float* res = residual == nullptr ? nullptr : residual->Ptr<float>();
      RmsNormRowCoopKernel<Tin, Tw, Tout, float><<<rows, kBlock, smem, s>>>(
          out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tw>(), res, h, args.eps, args.gemma, q8_out,
          nsb, ActDtOf(out.dtype), lds_quant);
    }
    return;
  }
  if (residual != nullptr && residual->dtype == DType::kBF16) {
    RmsNormRowKernel<Tin, Tw, Tout, __hip_bfloat16><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tw>(), residual->Ptr<__hip_bfloat16>(), h, args.eps,
        args.gemma, q8_out, nsb, ActDtOf(out.dtype));
  } else {
    float* res = residual == nullptr ? nullptr : residual->Ptr<float>();
    RmsNormRowKernel<Tin, Tw, Tout, float><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tw>(), res, h, args.eps, args.gemma, q8_out,
        nsb, ActDtOf(out.dtype));
  }
}

// Lever C opt-in gate: VT_NORM_QUANT_FUSED=1 (read PER CALL like the sibling
// arms' flags so in-process tests and graph capture pick it up at dispatch
// time), dense rows, whole Q8_K superblocks, and at most one thread's worth of
// superblocks per row. Default OFF leaves every path byte-unchanged.
vt::cpu::BlockQ8_K* NormQuantEpilogueFor(int64_t rows, int64_t h, hipStream_t s,
                                         size_t* scratch_bytes) {
  const char* e = std::getenv("VT_NORM_QUANT_FUSED");
  if (e == nullptr || e[0] != '1' || e[1] != '\0') return nullptr;
  if (rows <= 0 || h % kQK_K != 0 || h / kQK_K > kBlock) return nullptr;
  *scratch_bytes = static_cast<size_t>(rows) * static_cast<size_t>(h / kQK_K) *
                   sizeof(vt::cpu::BlockQ8_K);
  return static_cast<vt::cpu::BlockQ8_K*>(NormQuantProducerScratch(*scratch_bytes, s));
}

template <typename Tin, typename Tw>
void LaunchRmsNorm(hipStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  const unsigned rows = static_cast<unsigned>(t);
  size_t q8_bytes = 0;
  vt::cpu::BlockQ8_K* q8_out =
      NormQuantEpilogueFor(t, h, s, &q8_bytes);  // nullptr unless opt-in flag
  switch (out.dtype) {
    case DType::kF32:
      LaunchRmsNormRes<Tin, Tw, float>(s, out, x, w, args, residual, rows, h, q8_out);
      break;
    case DType::kBF16:
      LaunchRmsNormRes<Tin, Tw, __hip_bfloat16>(s, out, x, w, args, residual, rows, h, q8_out);
      break;
    default:
      VT_CHECK(false, "rocm rmsnorm: unsupported out dtype (f32/bf16 only)");
  }
  if (q8_out != nullptr) {
    NormQuantRecordProducer(out.data, t, h, /*row_stride=*/h, out.dtype, q8_out, s);
  }
}

// The gamma's dtype is dispatched SEPARATELY from the activation's, mirroring
// `DispatchRmsNormWeight` in cuda_ops.cu. The refusal that used to live in
// `RmsNormKernelRocm` as `VT_CHECK(w.dtype == x.dtype)` is NARROWED here rather
// than deleted: accepting a mismatch without retyping the kernel above would have
// read a bf16 gamma through a `const float*` and run off the end of it. An
// unsupported gamma dtype is still refused, and now says which one it got.
template <typename Tin>
void DispatchRmsNormWeight(hipStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                           const RmsNormArgs& args, Tensor* residual) {
  switch (w.dtype) {
    case DType::kF32:
      LaunchRmsNorm<Tin, float>(s, out, x, w, args, residual);
      break;
    case DType::kF16:
      LaunchRmsNorm<Tin, __half>(s, out, x, w, args, residual);
      break;
    case DType::kBF16:
      LaunchRmsNorm<Tin, __hip_bfloat16>(s, out, x, w, args, residual);
      break;
    default:
      VT_CHECK(false, std::string("rocm rmsnorm: unsupported weight dtype "
                                  "(f32/f16/bf16 only), got ") +
                          Name(w.dtype));
  }
}

// out = rmsnorm(x, w) + addend  — Gemma-4 residual join (NOT residual+=x then norm).
template <typename T>
__global__ void RmsNormPlusAddKernel(T* out, const T* x, const T* w, const T* add, int64_t h,
                                     float eps, bool gemma) {
  const int64_t row = blockIdx.x;
  const T* xrow = x + row * h;
  const T* arow = add + row * h;
  T* orow = out + row * h;
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v = Load(xrow, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(orow, j, Load(xrow, j) * inv * wj + Load(arow, j));
  }
}

}  // namespace

// The registered RmsNormFn (include/vt/ops.h:810). Signature is the shared vt
// one, unchanged — that is the whole contract a backend has to meet.
void RmsNormKernelRocm(Queue& q,
  // bind
 Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual) {
  EnsureQueueDevice(q);
  VT_CHECK(w.dtype == x.dtype, "rocm rmsnorm: weight dtype must match x");
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  switch (x.dtype) {
    case DType::kF32:
      DispatchRmsNormWeight<float>(s, out, x, w, args, residual);
      break;
    case DType::kBF16:
      DispatchRmsNormWeight<__hip_bfloat16>(s, out, x, w, args, residual);
      break;
    default:
      VT_CHECK(false, "rocm rmsnorm: unsupported input dtype (f32/bf16 only)");
  }
}

// out = rmsnorm(x, w) + addend
void RmsNormPlusAddRocm(Queue& q,
  // bind
 Tensor& out, const Tensor& x, const Tensor& w,
                        const Tensor& addend, const RmsNormArgs& args) {
  EnsureQueueDevice(q);
  VT_CHECK(w.dtype == x.dtype && out.dtype == x.dtype && addend.dtype == x.dtype,
           "rocm rmsnorm+add: dtype match");
  VT_CHECK(x.rank >= 2 && out.shape[0] == x.shape[0] && out.shape[1] == x.shape[1] &&
               addend.shape[0] == x.shape[0] && addend.shape[1] == x.shape[1],
           "rocm rmsnorm+add: shape");
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (x.dtype == DType::kBF16) {
    RmsNormPlusAddKernel<__hip_bfloat16><<<static_cast<unsigned>(t), kBlock, 0, s>>>(
        out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), w.Ptr<__hip_bfloat16>(),
        addend.Ptr<__hip_bfloat16>(), h, args.eps, args.gemma);
  } else if (x.dtype == DType::kF32) {
    RmsNormPlusAddKernel<float><<<static_cast<unsigned>(t), kBlock, 0, s>>>(
        out.Ptr<float>(), x.Ptr<float>(), w.Ptr<float>(), addend.Ptr<float>(), h, args.eps,
        args.gemma);
  } else {
    VT_CHECK(false, "rocm rmsnorm+add dtype");
  }
}

// out = rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual
// Gemma-4 MoE post-FF: dual expert streams + join residual.
template <typename T>
__global__ void DualRmsNormPlusResKernel(T* out, const T* x1, const T* w1, const T* x2,
                                         const T* w2, const T* w3, const T* residual, int64_t h,
                                         float eps1, float eps2, float eps3, bool gemma) {
  const int64_t row = blockIdx.x;
  const T* x1r = x1 + row * h;
  const T* x2r = x2 + row * h;
  const T* rr = residual + row * h;
  T* orow = out + row * h;
  __shared__ float p1[kBlock];
  __shared__ float p2[kBlock];
  __shared__ float p3[kBlock];

  float a1 = 0.f, a2 = 0.f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v1 = Load(x1r, j);
    const float v2 = Load(x2r, j);
    a1 += v1 * v1;
    a2 += v2 * v2;
  }
  p1[threadIdx.x] = a1;
  p2[threadIdx.x] = a2;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) {
      p1[threadIdx.x] += p1[threadIdx.x + s];
      p2[threadIdx.x] += p2[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float inv1 = 1.f / sqrtf(p1[0] / static_cast<float>(h) + eps1);
  const float inv2 = 1.f / sqrtf(p2[0] / static_cast<float>(h) + eps2);

  // Second pass: build mid = n1+n2 and accumulate variance of mid.
  // Use shared? too large for H=2816. Recompute n1,n2 per thread.
  float a3 = 0.f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float w1j = Load(w1, j);
    float w2j = Load(w2, j);
    if (gemma) {
      w1j += 1.f;
      w2j += 1.f;
    }
    const float mid = Load(x1r, j) * inv1 * w1j + Load(x2r, j) * inv2 * w2j;
    a3 += mid * mid;
  }
  p3[threadIdx.x] = a3;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) p3[threadIdx.x] += p3[threadIdx.x + s];
    __syncthreads();
  }
  const float inv3 = 1.f / sqrtf(p3[0] / static_cast<float>(h) + eps3);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float w1j = Load(w1, j);
    float w2j = Load(w2, j);
    float w3j = Load(w3, j);
    if (gemma) {
      w1j += 1.f;
      w2j += 1.f;
      w3j += 1.f;
    }
    const float mid = Load(x1r, j) * inv1 * w1j + Load(x2r, j) * inv2 * w2j;
    Store(orow, j, mid * inv3 * w3j + Load(rr, j));
  }
}

void DualRmsNormPlusResRocm(Queue& q,
  // bind
 Tensor& out, const Tensor& x1, const Tensor& w1,
                            const Tensor& x2, const Tensor& w2, const Tensor& w3,
                            const Tensor& residual, const RmsNormArgs& args) {
  EnsureQueueDevice(q);
  VT_CHECK(out.dtype == x1.dtype && x1.dtype == x2.dtype && x1.dtype == residual.dtype,
           "dual rms dtype");
  const int64_t t = x1.shape[0], h = x1.shape[1];
  if (t == 0 || h == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  // plain RMS (no gemma +1) for Gemma-4 layer norms
  const bool gemma = args.gemma;
  if (x1.dtype == DType::kBF16) {
    DualRmsNormPlusResKernel<__hip_bfloat16><<<static_cast<unsigned>(t), kBlock, 0, s>>>(
        out.Ptr<__hip_bfloat16>(), x1.Ptr<__hip_bfloat16>(), w1.Ptr<__hip_bfloat16>(),
        x2.Ptr<__hip_bfloat16>(), w2.Ptr<__hip_bfloat16>(), w3.Ptr<__hip_bfloat16>(),
        residual.Ptr<__hip_bfloat16>(), h, args.eps, args.eps, args.eps, gemma);
  } else if (x1.dtype == DType::kF32) {
    DualRmsNormPlusResKernel<float><<<static_cast<unsigned>(t), kBlock, 0, s>>>(
        out.Ptr<float>(), x1.Ptr<float>(), w1.Ptr<float>(), x2.Ptr<float>(), w2.Ptr<float>(),
        w3.Ptr<float>(), residual.Ptr<float>(), h, args.eps, args.eps, args.eps, gemma);
  } else {
    VT_CHECK(false, "dual rms dtype");
  }
}

}  // namespace vt::rocm
