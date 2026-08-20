// CUDA provider for `vt::Conv3d` — the LTX-2.5 video VAE decode's device arm
// (LTX25-DEVICE-RESIDENCY W5, #1007; .agents/specs/ltx25-device-residency.md).
//
// WHY THIS EXISTS. The LTX-2.5 video VAE decode is ~7.25 TFLOP of dense 3x3x3
// convolution and it ran entirely on the host, because `vt` had no 3-D
// convolution on ANY device: `vt::OpId::kLtx2` is the DiT glue table (seven
// small ops, no convolution) and `kConv2d`/`kConv1d` cannot express a temporal
// axis. Every reference runs this decode on an accelerator and fixes the
// placement at BUILD time rather than per call — Lightricks/LTX-2 @ fd4ded7f2
// `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:267-288`
// (CUDA by default at `:273`), SGLang @ f63458b5b
// `.../stages/model_specific_stages/ltx_2/decoding_av.py:71`, vLLM-Omni @
// a4ea67a21 `vllm_omni/diffusion/models/interface.py:92` ("VAE(s) (always on
// GPU)").
//
// Upstream semantics: `torch.nn.functional.conv3d` as `CausalConv3d`
// instantiates it (convolution.py:292-302, called at :312). The CPU provider
// (src/vt/cpu/cpu_conv3d.cpp) is the numeric reference this must agree with,
// because it is the host loop every committed LTX-2.5 video VAE golden was taken
// through.
//
// THE ONE DESIGN DECISION WORTH READING, and it is the same one
// src/vt/cuda/cuda_conv1d_general.cu makes: this is GATHER form, one thread per
// OUTPUT element, and each thread walks its inputs in the SAME ORDER the CPU
// provider walks them into that same output cell — the f32 accumulator seeded
// with the bias, then one f32 PARTIAL PER INPUT CHANNEL swept (kt, kh, kw) and
// added in. The host loop is already a gather, so the transcription is direct.
//
// That leaves exactly ONE way the two arms could disagree: FMA contraction,
// which fuses `acc + v*w` into a single-rounding operation and drops the
// intermediate rounding of the product. The host side is pinned against it
// project-wide by `-ffp-contract=off` (CMakeLists.txt); nvcc has no such pin and
// its `-fmad` default is on, so this file pins itself, locally and visibly, with
// `__fmul_rn` / `__fadd_rn`. With that, every operation on both arms is an
// IEEE-754 single-precision multiply or add with round-to-nearest-even,
// performed in the same order on the same values, and the arms are
// BIT-IDENTICAL rather than "within a tolerance".
//
// NOT VERIFIED ON HARDWARE. Nothing in this tree has executed this kernel: the
// row that landed it had no GPU lease and no CI job here has a GPU runner. It is
// registered and it is reached by the same dispatch the CPU arm takes, and the
// hardware run is owed — see `## Owed` in the spec. Read the bit-identity
// paragraph above as a design argument, not as a measurement.
//
// F32 ONLY, refused by name. The op's contract admits f16 and bf16 storage and
// the CPU arm serves them; this arm does not, and says so rather than reading
// half-width bytes as floats. The narrow-dtype device arm is owed with the
// hardware run.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (src/vt/cuda/cuda_conv1d_general.cu, src/vt/cuda/cuda_layernorm.cu): no
// existing kernel TU and no shared op array is edited.
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Every extent resolved on the host, so the device code does no division it does
// not need. Mirrors cuda_conv1d_general.cu's `Plan`.
struct Plan {
  int64_t in_channels;
  int64_t in_t;
  int64_t in_h;
  int64_t in_w;
  int64_t out_channels;
  int64_t out_t;
  int64_t out_h;
  int64_t out_w;
  int64_t kt;
  int64_t kh;
  int64_t kw;
  int64_t in_per_group;
  int64_t out_per_group;
  int64_t stride_t;
  int64_t stride_h;
  int64_t stride_w;
  int64_t pad_t;
  int64_t pad_h;
  int64_t pad_w;
  int64_t dilation_t;
  int64_t dilation_h;
  int64_t dilation_w;
};

// out[oc,ot,oh,ow] = bias[oc] + Sum_ic ( Sum_{kt,kh,kw} x[...] * w[...] ), the
// inner sum in its OWN f32 partial — the exact order of
// src/vt/cpu/cpu_conv3d.cpp Conv3dKernel.
__global__ void Conv3dKernelCudaImpl(float* __restrict__ out, const float* __restrict__ x,
                                     const float* __restrict__ w, const float* __restrict__ bias,
                                     Plan p) {
  const int64_t total = p.out_channels * p.out_t * p.out_h * p.out_w;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ow = idx % p.out_w;
    const int64_t row = idx / p.out_w;
    const int64_t oh = row % p.out_h;
    const int64_t ot = (row / p.out_h) % p.out_t;
    const int64_t oc = row / (p.out_h * p.out_t);
    const int64_t gc0 = (oc / p.out_per_group) * p.in_per_group;

    float acc = bias != nullptr ? bias[oc] : 0.0f;
    for (int64_t ic = 0; ic < p.in_per_group; ++ic) {
      const int64_t xc = (gc0 + ic) * p.in_t;
      const int64_t wc = (oc * p.in_per_group + ic) * p.kt;
      float tap = 0.0f;
      for (int64_t kt = 0; kt < p.kt; ++kt) {
        const int64_t it = ot * p.stride_t - p.pad_t + kt * p.dilation_t;
        if (it < 0 || it >= p.in_t) continue;
        const int64_t xt = (xc + it) * p.in_h;
        const int64_t wt = (wc + kt) * p.kh;
        for (int64_t kh = 0; kh < p.kh; ++kh) {
          const int64_t ih = oh * p.stride_h - p.pad_h + kh * p.dilation_h;
          if (ih < 0 || ih >= p.in_h) continue;
          const int64_t xrow = (xt + ih) * p.in_w;
          const int64_t wrow = (wt + kh) * p.kw;
          for (int64_t kw = 0; kw < p.kw; ++kw) {
            const int64_t iw = ow * p.stride_w - p.pad_w + kw * p.dilation_w;
            if (iw < 0 || iw >= p.in_w) continue;
            // __fmul_rn/__fadd_rn, never `tap += a * b`: see the file header.
            // nvcc would contract the latter into an fma and break byte
            // agreement with the -ffp-contract=off host provider.
            tap = __fadd_rn(tap, __fmul_rn(x[xrow + iw], w[wrow + kw]));
          }
        }
      }
      acc = __fadd_rn(acc, tap);
    }
    out[idx] = acc;
  }
}

unsigned GridFor(int64_t total) {
  const int64_t blocks = (total + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 65535 ? (blocks < 1 ? 1 : blocks) : 65535);
}

void Conv3dKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias,
                      const Conv3dArgs& args) {
  VT_CHECK(out.dtype == DType::kF32 && x.dtype == DType::kF32 && w.dtype == DType::kF32 &&
               (bias == nullptr || bias->dtype == DType::kF32),
           "cuda conv3d: this arm serves f32 only; f16/bf16 storage is owed (#1007)");
  Plan p{};
  p.in_channels = x.shape[0];
  p.in_t = x.shape[1];
  p.in_h = x.shape[2];
  p.in_w = x.shape[3];
  p.out_channels = out.shape[0];
  p.out_t = out.shape[1];
  p.out_h = out.shape[2];
  p.out_w = out.shape[3];
  p.kt = w.shape[1];
  p.kh = w.shape[2];
  p.kw = w.shape[3];
  p.in_per_group = p.in_channels / args.groups;
  p.out_per_group = p.out_channels / args.groups;
  p.stride_t = args.stride_t;
  p.stride_h = args.stride_h;
  p.stride_w = args.stride_w;
  p.pad_t = args.pad_t;
  p.pad_h = args.pad_h;
  p.pad_w = args.pad_w;
  p.dilation_t = args.dilation_t;
  p.dilation_h = args.dilation_h;
  p.dilation_w = args.dilation_w;
  const int64_t total = p.out_channels * p.out_t * p.out_h * p.out_w;
  if (total == 0) return;
  Conv3dKernelCudaImpl<<<GridFor(total), kBlock, 0, AsStream(q)>>>(
      out.Ptr<float>(), x.Ptr<float>(), w.Ptr<float>(),
      bias != nullptr ? bias->Ptr<float>() : nullptr, p);
  Check(cudaGetLastError(), "conv3d launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kConv3d, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Conv3dFn>(&Conv3dKernelCuda)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cuda
