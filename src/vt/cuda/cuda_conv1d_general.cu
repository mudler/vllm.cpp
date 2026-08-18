// CUDA providers for `vt::Conv1d` and `vt::ConvTranspose1d` — the BigVGAN / DAC
// vocoder convolutions (#672, .agents/specs/minimax-music3.md §11.4).
//
// WHY THIS EXISTS. The transposed convolution chain is 88.5 % of MiniMax-Music3's
// acoustic-half profile, and before this row `vt` had no transposed 1-D
// convolution of ANY kind, on any device — so the single most expensive stage in
// the model had no device op to route to and the whole vocoder ran as scalar
// host loops. `vt::Conv2d` and `vt::DepthwiseConv1d` are likewise CPU-only
// (cpu_conv2d.cpp, cpu_conv1d_depthwise.cpp), so widening one of them would not
// have helped either.
//
// Upstream semantics: `torch.nn.functional.conv1d` / `conv_transpose1d` as
// instantiated at minimax_music3_vocoder.py:42,44,55,89,98 and LTX-2.5
// audio_vae/vocoder.py:104-184. The CPU provider
// (src/vt/cpu/cpu_conv1d_general.cpp) is the numeric reference these must agree
// with, because it is the host loop the committed goldens were taken through.
//
// THE ONE DESIGN DECISION WORTH READING: both kernels are GATHER form, one
// thread per OUTPUT element, and each thread walks its inputs in the SAME ORDER
// the CPU provider walks them into that same output cell.
//
// For Conv1d that is trivial — the host loop is already a gather, so (ic
// ascending, k ascending) with the bias seeded first transcribes directly.
//
// For ConvTranspose1d it is the whole trick. The host loop is a SCATTER: for
// each input channel `ic` ascending, for each input position `t` ascending, it
// adds `x[ic,t] * w[ic,oc,k]` into destination cell `t*stride + k*dilation`. Fix
// a destination cell `p` and ask which additions land in it, in what order: `ic`
// ascending, then `t` ascending, and for each `t` at most ONE tap `k`, namely
// the one with `t*stride + k*dilation == p`. So a thread that owns `p` and sweeps
// `ic` then `t` in increasing order performs the identical sequence of f64
// additions into the identical f64 accumulator. Not "within a tolerance" —
// the same additions in the same order.
//
// Two details that are load-bearing rather than cosmetic:
//   * the `value == 0.0` SKIP is reproduced exactly. It is not an optimisation:
//     dropping it changes the sign of a zero output cell, because
//     (-0.0) + (+0.0) == +0.0 while (-0.0) left alone stays -0.0.
//   * the accumulator is f64, and the bias is added LAST for the transposed op
//     and FIRST for the forward one, matching each host loop respectively.
//
// That leaves exactly ONE way the two arms could still disagree: FMA
// contraction, which fuses `acc + v*w` into a single-rounding operation and so
// drops the intermediate rounding of the product. This project already pins the
// HOST side against it — `CMakeLists.txt:40-56` compiles every C++ TU with
// `-ffp-contract=off` precisely so two textually identical reductions cannot
// compile one contracted and one not. nvcc has no such pin (its `-fmad` default
// is on and its flags are separate), so the device side pins itself, locally and
// visibly, with `__dmul_rn` / `__dadd_rn`.
//
// With that, every arithmetic operation on both arms is an IEEE-754 double
// multiply or add with round-to-nearest-even, performed in the same order on the
// same values. The arms are BIT-IDENTICAL, and the gate asserts `memcmp`
// equality rather than a tolerance — tests/vt/test_ops_conv1d_general.cpp,
// `CUDA ... is byte-identical to the CPU provider`. A tolerance here would have
// been the wrong instrument anyway: every defect this pairing exists to catch —
// a transposed weight axis, a dropped zero-skip, a reassociated sweep — lands
// well inside any epsilon anyone would write.
//
// f32 in memory, f64 in the accumulator: see include/vt/ops.h at vt::Conv1d for
// why that widening is deliberate and what it costs (nothing in bytes moved).
//
// SELF-REGISTERING translation unit in the established additive pattern
// (src/vt/cuda/cuda_glue.cu, src/vt/cuda/cuda_layernorm.cu): no existing kernel
// TU and no shared op array is edited.
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

// Every extent the kernels need, resolved on the host so the device code does
// no division by a runtime `groups` beyond the one it needs for its own row.
struct Plan {
  int64_t batch;
  int64_t in_channels;
  int64_t in_len;
  int64_t out_channels;
  int64_t out_len;
  int64_t kernel;
  int64_t in_per_group;
  int64_t out_per_group;
  int64_t stride;
  int64_t padding;
  int64_t dilation;
};

// out[n, oc, t] = bias[oc] + Sum_{ic,k} x[n, g*in_per_group + ic,
//                                        t*stride - padding + k*dilation]
//                            * w[oc, ic, k]
// walked (ic ascending, k ascending) with the bias seeded FIRST — the exact
// order of src/vt/cpu/cpu_conv1d_general.cpp Conv1dKernel.
__global__ void Conv1dKernelCudaImpl(float* __restrict__ out, const float* __restrict__ x,
                                     const float* __restrict__ w, const float* __restrict__ bias,
                                     Plan p) {
  const int64_t total = p.batch * p.out_channels * p.out_len;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = idx % p.out_len;
    const int64_t row = idx / p.out_len;
    const int64_t oc = row % p.out_channels;
    const int64_t n = row / p.out_channels;
    const int64_t g = oc / p.out_per_group;
    const float* xn = x + n * p.in_channels * p.in_len;

    double acc = bias != nullptr ? static_cast<double>(bias[oc]) : 0.0;
    for (int64_t ic = 0; ic < p.in_per_group; ++ic) {
      const int64_t src_c = g * p.in_per_group + ic;
      const float* wc = w + (oc * p.in_per_group + ic) * p.kernel;
      for (int64_t k = 0; k < p.kernel; ++k) {
        const int64_t pos = t * p.stride - p.padding + k * p.dilation;
        if (pos < 0 || pos >= p.in_len) continue;
        // __dmul_rn/__dadd_rn, never `a += b * c`: see the file header. nvcc
        // would contract the latter into an fma and break byte agreement with
        // the -ffp-contract=off host provider.
        acc = __dadd_rn(acc, __dmul_rn(static_cast<double>(xn[src_c * p.in_len + pos]),
                                       static_cast<double>(wc[k])));
      }
    }
    out[row * p.out_len + t] = static_cast<float>(acc);
  }
}

// The gather transcription of the host SCATTER — see the file header for the
// argument that the addition sequence is identical. `full` is the un-cropped
// scatter extent; cells at or past it are the `output_padding` tail, which torch
// leaves at zero (plus bias).
__global__ void ConvTranspose1dKernelCudaImpl(float* __restrict__ out, const float* __restrict__ x,
                                              const float* __restrict__ w,
                                              const float* __restrict__ bias, Plan p,
                                              int64_t full) {
  const int64_t total = p.batch * p.out_channels * p.out_len;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t_out = idx % p.out_len;
    const int64_t row = idx / p.out_len;
    const int64_t dst_c = row % p.out_channels;
    const int64_t n = row / p.out_channels;
    const int64_t g = dst_c / p.out_per_group;
    const int64_t oc = dst_c - g * p.out_per_group;
    const float* xn = x + n * p.in_channels * p.in_len;

    const int64_t pos = t_out + p.padding;
    double acc = 0.0;
    if (pos < full) {
      // t*stride + k*dilation == pos, with 0 <= k < kernel and 0 <= t < in_len.
      const int64_t span = p.dilation * (p.kernel - 1);
      int64_t t_lo = pos - span;
      // ceil-divide the lower bound by stride without touching negatives.
      t_lo = t_lo <= 0 ? 0 : (t_lo + p.stride - 1) / p.stride;
      int64_t t_hi = pos / p.stride;
      if (t_hi > p.in_len - 1) t_hi = p.in_len - 1;
      for (int64_t ic = g * p.in_per_group; ic < (g + 1) * p.in_per_group; ++ic) {
        const float* wc = w + (ic * p.out_per_group + oc) * p.kernel;
        const float* xc = xn + ic * p.in_len;
        for (int64_t t = t_lo; t <= t_hi; ++t) {
          const int64_t off = pos - t * p.stride;
          if (off % p.dilation != 0) continue;
          const double value = xc[t];
          // The host loop skips a zero input BEFORE touching the destination;
          // reproducing that is what keeps the sign of a zero cell.
          if (value == 0.0) continue;
          // Non-contracted, as in Conv1dKernelCudaImpl above.
          acc = __dadd_rn(acc, __dmul_rn(value, static_cast<double>(wc[off / p.dilation])));
        }
      }
    }
    // Bias added LAST, matching the host scatter's crop-then-bias tail.
    if (bias != nullptr) acc = __dadd_rn(acc, static_cast<double>(bias[dst_c]));
    out[row * p.out_len + t_out] = static_cast<float>(acc);
  }
}

unsigned GridFor(int64_t total) {
  const int64_t blocks = (total + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 65535 ? (blocks < 1 ? 1 : blocks) : 65535);
}

Plan MakePlan(const Tensor& out, const Tensor& x, const Tensor& w, int64_t groups, int64_t stride,
              int64_t padding, int64_t dilation, int64_t in_per_group, int64_t out_per_group) {
  Plan p{};
  p.batch = x.shape[0];
  p.in_channels = x.shape[1];
  p.in_len = x.shape[2];
  p.out_channels = out.shape[1];
  p.out_len = out.shape[2];
  p.kernel = w.shape[2];
  p.in_per_group = in_per_group;
  p.out_per_group = out_per_group;
  p.stride = stride;
  p.padding = padding;
  p.dilation = dilation;
  (void)groups;
  return p;
}

void Conv1dKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias,
                      const Conv1dArgs& args) {
  const Plan p = MakePlan(out, x, w, args.groups, args.stride, args.padding, args.dilation,
                          x.shape[1] / args.groups, out.shape[1] / args.groups);
  const int64_t total = p.batch * p.out_channels * p.out_len;
  if (total == 0) return;
  Conv1dKernelCudaImpl<<<GridFor(total), kBlock, 0, AsStream(q)>>>(
      out.Ptr<float>(), x.Ptr<float>(), w.Ptr<float>(),
      bias != nullptr ? bias->Ptr<float>() : nullptr, p);
  Check(cudaGetLastError(), "conv1d launch");
}

void ConvTranspose1dKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                               const Tensor* bias, const ConvTranspose1dArgs& args) {
  const Plan p = MakePlan(out, x, w, args.groups, args.stride, args.padding, args.dilation,
                          x.shape[1] / args.groups, w.shape[1]);
  const int64_t full = (p.in_len - 1) * p.stride + p.dilation * (p.kernel - 1) + 1;
  const int64_t total = p.batch * p.out_channels * p.out_len;
  if (total == 0) return;
  ConvTranspose1dKernelCudaImpl<<<GridFor(total), kBlock, 0, AsStream(q)>>>(
      out.Ptr<float>(), x.Ptr<float>(), w.Ptr<float>(),
      bias != nullptr ? bias->Ptr<float>() : nullptr, p, full);
  Check(cudaGetLastError(), "conv_transpose1d launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kConv1d, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<Conv1dFn>(&Conv1dKernelCuda)));
    RegisterOp(OpId::kConvTranspose1d, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ConvTranspose1dFn>(&ConvTranspose1dKernelCuda)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cuda
