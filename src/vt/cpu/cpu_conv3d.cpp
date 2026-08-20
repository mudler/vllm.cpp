// CPU Conv3d — the portable-tier kernel for `vt::Conv3d`
// (LTX25-DEVICE-RESIDENCY W5, #1007; .agents/specs/ltx25-device-residency.md).
//
// Ported FROM (semantics, 1:1): torch `nn.Conv3d` as `CausalConv3d` instantiates
// it — Lightricks/LTX-2 @ fd4ded7f2,
//   packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:292-302 (ctor)
//   packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:312       (call)
// which is the single call site the whole LTX-2.5 video VAE's arithmetic passes
// through. The temporal and non-`zeros` spatial padding upstream applies around
// that call (`torch.concatenate` at :305-311, `padding_mode` at :299) is
// MATERIALIZED BY THE CALLER, exactly as torch itself materializes a non-`zeros`
// `padding_mode` before calling a zero-padded convolution — so this kernel only
// ever sees explicit zero padding.
//
// It is the numeric reference the CUDA arm (src/vt/cuda/cuda_conv3d.cu) must
// agree with BYTE FOR BYTE, because it is the loop every committed LTX-2.5 video
// VAE golden was taken through: this file is
// `src/vllm/model_executor/models/ltx2_video_vae.cpp`'s `CausalConv3d` inner
// convolution, moved behind the op seam without one arithmetic change.
//
// Like src/vt/cpu/cpu_conv2d.cpp this is a SELF-REGISTERING translation unit
// (the src/vt/cpu/cpu_ops.cpp Registrar idiom), so adding the op edited no
// existing kernel file.
//
// ─── THE ACCUMULATION ORDER, WHICH IS THE CONTRACT ───────────────────────────
//
// One f32 accumulator per output element, SEEDED WITH THE BIAS, then ONE f32
// PARTIAL PER INPUT CHANNEL walked strictly in (kt, kh, kw) order and added in.
// This is NOT `cpu_conv2d.cpp`'s single flat accumulator, and the difference is
// measurable rather than stylistic: `ltx2_video_vae.cpp` records that a naive
// serial f32 sum over all `ci * kernel^3` taps pushes the non-causal tiled golden
// to 5.00679e-06 against a 5e-06 tolerance, because torch's f32 convolution is a
// blocked GEMM and sums one partial per input channel. The bias is seeded FIRST,
// not added last — `cpu_conv2d.cpp` adds it last, and the two orders are
// different numbers.
//
// ─── DETERMINISM ─────────────────────────────────────────────────────────────
//
// The parallel axis is the OUTPUT LINE (oc, ot, oh), whose `wout` elements are a
// contiguous span of `out` that no other worker writes, and the whole
// `cin_g * KT*KH*KW` reduction for one element stays inside one iteration of one
// row. So a worker executes precisely the serial arm's instruction sequence, in
// the serial arm's order, for every element it owns: the result is byte-identical
// at any thread count and under any row-to-thread assignment, which matters
// because ParallelForRows steals work through an atomic cursor. Splitting the
// REDUCTION axis into per-thread partials would also be a legal convolution and
// is deliberately not taken — it would make the summation order a function of
// the thread count. tests/vt/test_ops_conv3d.cpp gates both halves.
//
// Zero padding is realised by SKIPPING out-of-range taps rather than adding an
// explicit 0.0 product, matching cpu_conv2d.cpp and the in-test reference; that
// keeps the two exact even for a -0.0 accumulator.
#include "cpu_threadpool.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

float LoadF32At(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "cpu conv3d: unsupported input dtype"); return 0.0f;
  }
}

void StoreF32At(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kF16: t.Ptr<uint16_t>()[i] = F32ToF16(v); break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "cpu conv3d: unsupported output dtype");
  }
}

// out[oc,ot,oh,ow] = bias[oc] + Σ_ic ( Σ_{kt,kh,kw} x[gc0+ic, ot*st-pt+kt*dt,
//                                                    oh*sh-ph+kh*dh,
//                                                    ow*sw-pw+kw*dw]
//                                      * w[(oc*cin_g + ic), kt, kh, kw] )
// with taps outside the input skipped (zero padding), `gc0` the first input
// channel of output channel oc's group, and the INNER sum kept in its own f32
// partial — see the header comment for why that nesting is the contract.
void Conv3dKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias,
                  const Conv3dArgs& args) {
  const int64_t cin = x.shape[0], tin = x.shape[1], hin = x.shape[2], win = x.shape[3];
  const int64_t cout = out.shape[0], tout = out.shape[1], hout = out.shape[2],
                wout = out.shape[3];
  const int64_t kt_n = w.shape[1], kh_n = w.shape[2], kw_n = w.shape[3];
  const int64_t cin_g = cin / args.groups;
  const int64_t cout_g = cout / args.groups;
  const int64_t st = args.stride_t, sh = args.stride_h, sw = args.stride_w;
  const int64_t pt = args.pad_t, ph = args.pad_h, pw = args.pad_w;
  const int64_t dt = args.dilation_t, dh = args.dilation_h, dw = args.dilation_w;
  const int64_t rows = cout * tout * hout;
  ParallelForRows(CurrentThreadpool(), rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t oh = r % hout;
      const int64_t ot = (r / hout) % tout;
      const int64_t oc = r / (hout * tout);
      const int64_t gc0 = (oc / cout_g) * cin_g;  // first input channel of oc's group
      const float b = bias != nullptr ? LoadF32At(*bias, oc) : 0.0f;
      for (int64_t ow = 0; ow < wout; ++ow) {
        float acc = b;
        for (int64_t ic = 0; ic < cin_g; ++ic) {
          const int64_t xc = (gc0 + ic) * tin;
          const int64_t wc = (oc * cin_g + ic) * kt_n;
          float tap = 0.0f;
          for (int64_t kt = 0; kt < kt_n; ++kt) {
            const int64_t it = ot * st - pt + kt * dt;
            if (it < 0 || it >= tin) continue;
            const int64_t xt = (xc + it) * hin;
            const int64_t wt = (wc + kt) * kh_n;
            for (int64_t kh = 0; kh < kh_n; ++kh) {
              const int64_t ih = oh * sh - ph + kh * dh;
              if (ih < 0 || ih >= hin) continue;
              const int64_t xrow = (xt + ih) * win;
              const int64_t wrow = (wt + kh) * kw_n;
              for (int64_t kw = 0; kw < kw_n; ++kw) {
                const int64_t iw = ow * sw - pw + kw * dw;
                if (iw < 0 || iw >= win) continue;
                tap += LoadF32At(x, xrow + iw) * LoadF32At(w, wrow + kw);
              }
            }
          }
          acc += tap;
        }
        StoreF32At(out, ((oc * tout + ot) * hout + oh) * wout + ow, acc);
      }
    }
  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kConv3d, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Conv3dFn>(&Conv3dKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
