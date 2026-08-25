// LTX-2.5 CONV VIDEO VAE — the DEVICE-RESIDENT KERNEL SEAM (vt::OpId::kLtx2Vae).
//
// Row: LTX25-VAE-DEVICE-RESIDENCY. Spec:
// .agents/specs/ltx25-vae-device-residency.md. Issue #1451.
//
// This header is DELIBERATELY THIN, and mirrors ltx2_kernels.h for the reason
// that header states at its own :6-10: `src/vt/cuda/cuda_ltx2_vae.cu` includes
// it, and a backend kernel translation unit must not be asked to compile the
// model's own headers — `ltx2_video_vae.h` pulls in the model config types and
// with them `nlohmann/json.hpp`, which nvcc has no business parsing. Nothing
// here includes anything but `vt/`.
//
// ─── WHAT THIS CLOSES ────────────────────────────────────────────────────────
//
// W5 (#1007) put the conv video VAE's CONVOLUTION on the device and left every
// buffer BETWEEN the convolutions on the host, so a non-CPU queue paid an upload
// and a download per `nn.Conv3d` call and re-uploaded each weight. That is a
// divergence in memory behaviour and not only a cost: upstream builds the
// decoder onto a device once (Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca,
// packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:273 defaults
// it onto CUDA), the latent follows the weights
// (model/video_vae/conv_video_decoder.py:283-284), and the conv decoder's
// forward contains NO host round-trip at all — `grep -rn '\.cpu()' ` over
// `model/video_vae/` returns only the checkpoint loader and the DIFFUSION
// decoder's timestep schedule.
//
// ─── WHY THE TABLE IS THIS LONG, WHERE ltx2_kernels.h's IS SEVEN ─────────────
//
// The DiT is a transformer and the shared `vt::` surface already expresses
// almost all of it. A convolutional VAE decoder is mostly SHAPE MOVEMENT, and
// the shared surface has none: there is no `vt::` permute, transpose,
// depth-to-space, pixel-shuffle or slice op anywhere in the tree, and no
// GroupNorm and no pixel-norm either. What the decode DOES reuse is named here
// so the reuse is checkable rather than asserted:
//
//   nn.Conv3d              -> vt::Conv3d          (W5, #1007)
//   ungated SiLU           -> vt::OpId::kLtx2's `silu` (cpu_ltx2.cpp:188)
//   residual add           -> vt::Add             (ops.h:2440)
//
// Everything else is below, and every entry names the host helper it replaces
// and the upstream line that helper came from.
//
// ─── DTYPE: f32, AND THAT IS NOT A WIDENING ─────────────────────────────────
//
// Every kernel here serves f32 STORAGE only and refuses anything else BY NAME,
// the same way `src/vt/cuda/cuda_conv3d.cu:154-156` refuses f16/bf16. This is
// not the "f32 is a rare annotated exception" case AGENTS.md warns about: the
// whole LTX-2.5 conv video VAE decode is f32 in this tree, deliberately and
// after a measurement — #1008 found the decode computing in f64 where torch
// computes in f32 and narrowed it, and `ltx2_video_vae.cpp:50` records the
// resulting polarity. A bf16 storage arm is owed with the CUDA measurement
// (#1452) rather than guessed at here.
//
// ─── SEAM ────────────────────────────────────────────────────────────────────
//
// One OpProvider entry whose payload is a static kernels-struct of typed
// launchers, mirroring the kLtx2 and kMiniMaxH3 precedents. Registered for BOTH
// kCPU (src/vt/cpu/cpu_ltx2_vae.cpp) and kCUDA (src/vt/cuda/cuda_ltx2_vae.cu) —
// so the RESIDENT STRUCTURE is covered by CPU CI and a GPU is needed to gate the
// KERNELS, not the port. `.agents/specs/ltx25-vae-device-residency.md`
// `## What a CPU-only run can and cannot establish` states exactly which of
// those two this row measured.
//
// ─── LAYOUT ──────────────────────────────────────────────────────────────────
//
// Every volume is [C, T, H, W] at batch 1, contiguous, which is what both arms
// of the decode already consume. `spatial` below always means T*H*W. #1011, the
// memory-format rider, is NOT taken here: this row makes the volume resident and
// does not re-choose its format.
#pragma once

#include <cstdint>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt {
struct Queue;
}  // namespace vt

namespace vllm {
namespace ltx2_vae {

struct Ltx2VaeDeviceKernels {
  // `PixelNorm` (ltx2_video_vae.cpp:381), IN PLACE, over [C, spatial]:
  //   ms[i] = mean over c of x[c, i]^2
  //   x[c, i] /= sqrt(ms[i] + eps)
  // Upstream: model/common/normalization.py:37-40 — `torch.mean(x**2, dim=1,
  // keepdim=True)`, `torch.sqrt(mean_sq + eps)`, `x / rms`. The DEFAULT eps is
  // 1e-8 (`:22`), reached bare from video_vae/resnet.py and
  // conv_video_decoder.py — NOT the 1e-6 the audio VAE gets through
  // `build_normalization_layer`.
  //
  // THE RECIPROCAL IS FORMED ONCE PER PIXEL AND MULTIPLIED, which is what the
  // host loop this replaces did. **No gate holds that form**, and the honest
  // statement is worth more here than the confident one: mutating this to a
  // per-channel divide leaves `test_ltx2_vae` at 45/45 and 3152/3152 green,
  // because the two differ only in the last ulp and the golden tolerance absorbs
  // it. The form is kept because it is the one the replaced loop had, so this
  // move changes nothing -- not because anything would catch a change.
  //
  // `eps` is f32 HERE and f64 in the config that carries it
  // (`Ltx2ConvVideoDecoderConfig::pixel_norm_eps`): it is a pinned threshold up
  // to the point it enters the arithmetic, and it is narrowed at exactly that
  // point, which is where `ltx2_video_vae.cpp:390-394` already narrows it.
  void (*pixel_norm)(vt::Queue&, void* x, int64_t channels, int64_t spatial, float eps,
                     vt::DType dtype);

  // The GroupNorm arm of `ApplyNorm` (ltx2_video_vae.cpp:443), IN PLACE, over
  // [C, spatial] with `groups` groups of C/groups channels:
  //   over each group: mean and biased variance across (C/groups * spatial)
  //   x[c, i] = (x[c, i] - mean) / sqrt(var + eps) * weight[c] + bias[c]
  // This is `MiniMaxH3GroupNorm3d`'s contract, and the two must not drift: the
  // decode calls that host function at four sites today (`:443, :588, :929`, and
  // the encoder's `:1291`) and the goldens were taken through it.
  //
  // `num_groups == 1` is a LayerNorm over (C, T, H, W) and is reached that way
  // by `norm3` (`:586-590`). It is served here rather than re-pointed at
  // `vt::LayerNorm`, because `vt::LayerNorm` normalizes over the LAST axis of a
  // row-major [rows, width] and this volume is channel-major: the two reduce
  // over different elements and only agree when spatial == 1.
  //
  // `eps` IS f64 HERE AND f32 IN `pixel_norm`, and the asymmetry is faithful
  // rather than sloppy. The host loop this replaces
  // (`minimax_h3_vae_cnn.cpp:117-147`) takes `double eps` and adds it to a double
  // `var`; `PixelNorm`'s host loop narrowed its own eps at the point it entered
  // the arithmetic (`ltx2_video_vae.cpp`'s `static_cast<float>(eps)`). Each
  // kernel keeps the width its caller already had, because a golden cannot see a
  // width that changed by 1e-19 in a sum of order one -- and that is exactly the
  // class of drift AGENTS.md warns a token gate cannot detect.
  void (*group_norm)(vt::Queue&, void* x, int64_t channels, int64_t spatial, int64_t groups,
                     const void* weight, const void* bias, double eps, vt::DType dtype);

  // `ApplyAdaLn` (ltx2_video_vae.cpp:529), IN PLACE, over [C, spatial]:
  //   shift = table[shift_row * C + c] + embed[shift_row * C + c]
  //   scale = table[scale_row * C + c] + embed[scale_row * C + c]
  //   x[c, i] = x[c, i] * (1 + scale) + shift
  // Upstream: model/video_vae/resnet.py:135-148 — `ada_values = scale_shift_table
  // + timestep`, unbound into (shift1, scale1, shift2, scale2), then
  // `hidden_states * (1 + scale1) + shift1`.
  //
  // THE TWO ADDITIONS HAPPEN BEFORE THE AFFINE and in that order. Upstream forms
  // `table + timestep` first and only then applies the affine; folding them the
  // other way round rounds differently. That is the same argument
  // ltx2_kernels.h's `output_modulate` makes for the DiT's output head.
  //
  // `rows` is the table's leading extent — FOUR for a resnet block
  // (resnet.py:102, `torch.zeros(4, in_channels)`) and TWO for the decode's
  // final `last_scale_shift_table` (ltx2_video_vae.cpp:937). It is passed rather
  // than inferred so a caller cannot silently read the wrong pair.
  void (*ada_ln)(vt::Queue&, void* x, const void* table, const void* embed, int64_t channels,
                 int64_t spatial, int64_t rows, int64_t shift_row, int64_t scale_row,
                 vt::DType dtype);

  // `FeedSpatialNoise` (ltx2_video_vae.cpp:507), IN PLACE, over [C, T, H, W]:
  //   x[c, t, h, w] += plane[h * W + w] * scale[c]
  // Upstream: model/video_vae/resnet.py:112-117 — ONE [H, W] draw
  // (`torch.randn(spatial_shape)`), broadcast over batch, channels AND TIME by
  // the `[None]` / `[None, :, None, ...]` indexing, scaled per channel.
  //
  // THE PLANE IS AN INPUT, NOT DRAWN HERE. `Ltx2NoiseStream::Draw` is the
  // reproducibility seam and it stays on the host: a device-side generator would
  // be a different stream, and the renders this project has captured are keyed
  // to that one.
  void (*spatial_noise)(vt::Queue&, void* x, const void* plane, const void* scale,
                        int64_t channels, int64_t t, int64_t h, int64_t w, vt::DType dtype);

  // `DepthToSpaceUpsample`'s channel unpack (ltx2_video_vae.cpp:605-631):
  //   out[c, t*st + p1, h*sh + p2, w*sw + p3] = x[((c*st + p1)*sh + p2)*sw + p3, t, h, w]
  // Upstream: model/video_vae/sampling.py:112-118 —
  // `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)`, p1 TEMPORAL and p2/p3
  // spatial. `in_channels` is `out_channels * st * sh * sw`; the shape is derived
  // from `out_channels` and the strides rather than passed twice.
  //
  // p1 IS THE OUTER FACTOR of the packed channel index. Reading the unpack in
  // any other order transposes every patch and still produces a plausible clip,
  // which is why the index expression is written out above rather than left to
  // the reader.
  void (*depth_to_space)(vt::Queue&, void* out, const void* x, int64_t out_channels, int64_t t,
                         int64_t h, int64_t w, int64_t st, int64_t sh, int64_t sw,
                         vt::DType dtype);

  // `drop_first_frame` (ltx2_video_vae.cpp:632-654): out[c, i, h, w] = x[c, i + drop, h, w],
  // with `t` the INPUT frame count and the output carrying `t - drop`.
  // Upstream: model/video_vae/sampling.py:119-120 — `x[:, :, 1:, :, :]` when
  // `stride[0] == 2`, applied to the main branch and, at :109-110, to the
  // residual. NOT :121-122, which is `if self.residual: x = x + x_in` -- a
  // different operation, and the miscitation a fresh review caught here.
  void (*frame_slice)(vt::Queue&, void* out, const void* x, int64_t channels, int64_t t,
                      int64_t h, int64_t w, int64_t drop, vt::DType dtype);

  // The residual's channel repeat (ltx2_video_vae.cpp:660-670):
  //   out[r * C + c, i] = x[c, i] for r in [0, repeat)
  // Upstream: model/video_vae/sampling.py:108 — `x_in.repeat(1, num_repeat, 1, 1, 1)`
  // with `num_repeat = prod(stride) // out_channels_reduction_factor`.
  // torch's `repeat` TILES the whole tensor, so the block index is the OUTER
  // axis; `repeat_interleave` would put it inner and is a different tensor.
  void (*channel_repeat)(vt::Queue&, void* out, const void* x, int64_t channels, int64_t spatial,
                         int64_t repeat, vt::DType dtype);

  // `Linear3d` (ltx2_video_vae.cpp:339-368) over a channel-major volume:
  //   out[oc, i] = bias[oc] + sum over ic of x[ic, i] * weight[oc * Cin + ic]
  // Upstream: model/video_vae/convolution.py:84-85 — `make_linear_nd` for
  // `dims == 3` is a 1x1x1 `nn.Conv3d`, which is why the accumulator is f32 and
  // seeded with the bias, exactly as `vt::Conv3d`'s contract requires
  // (`vt::Conv3d`'s ACCUMULATION ORDER contract in ops.h).
  //
  // NOT `vt::Matmul` PLUS `vt::Add`. `vt::Add` broadcasts a rank-1 operand over
  // the LAST axis; this bias is per CHANNEL, which is the FIRST axis of a
  // channel-major [Cout, N]. Routing it through the shared pair would need a
  // materialized [Cout, N] bias — more bytes moved than the whole operation —
  // or a transpose op the tree does not have.
  void (*linear_cn)(vt::Queue&, void* out, const void* x, const void* weight, const void* bias,
                    int64_t out_channels, int64_t in_channels, int64_t n, vt::DType dtype);

  // The decode's `unpatchify` tail (ltx2_video_vae.cpp:945-970):
  //   out[c, f, h*q + qi, w*r + ri] = x[(c * r + ri) * q + qi, f, h, w]
  // Upstream: model/video_vae/ops.py:35-60 —
  // `b (c p r q) f h w -> b c (f p) (h q) (w r)` with p = patch_size_t = 1.
  //
  // H TAKES q AND W TAKES r. Swapping them transposes every patch, which the
  // host helper's own comment at `:945-947` already flags, and which a
  // shape-valid gate cannot see.
  void (*unpatchify)(vt::Queue&, void* out, const void* x, int64_t channels, int64_t t,
                     int64_t h, int64_t w, int64_t q, int64_t r, vt::DType dtype);

  // The pad `CausalConv3d` materializes before it convolves
  // (ltx2_video_vae.cpp:262-300), out is [C, T + pad_t, H + 2*pad_hw, W + 2*pad_hw]:
  //   temporal: frame index ti - pad_t, CLAMPED AT 0, so frame 0 is REPLICATED
  //             pad_t times (upstream concatenates k_t - 1 copies of frame 0,
  //             model/video_vae/convolution.py:305-311)
  //   spatial:  `mode` applied independently on H and W — kZeros writes 0,
  //             kReflect uses torch's edge-EXCLUDING reflection, kReplicate clamps
  //
  // WHY THIS IS A KERNEL AT ALL, when `vt::Conv3d` takes a `pad_*`: that argument
  // is ZERO padding only (ops.h:3089), and this decode's spatial mode is a config
  // field with three values. Materializing the pad is what the host path already
  // does; making it a kernel is what stops the volume leaving the device to be
  // padded. `pad_t_leading` is `k_t - 1` on the causal arm and `(k_t - 1) / 2` on
  // the non-causal arm, and `pad_t_trailing` is 0 and `(k_t - 1) / 2` — passed
  // separately because the two arms differ ONLY there and a single "temporal pad"
  // argument would have to encode the causality flag.
  void (*pad)(vt::Queue&, void* out, const void* x, int64_t channels, int64_t t, int64_t h,
              int64_t w, int64_t pad_t_leading, int64_t pad_t_trailing, int64_t pad_hw,
              int mode, vt::DType dtype);
};

// `mode` values for `pad`, matching `Ltx2PaddingMode` in ltx2_video_vae.h. They
// are re-declared as plain ints here rather than including that header, for the
// nvcc reason at the top of this file. `ltx2_video_vae.cpp` static_asserts the
// two enumerations agree, so a reordering there cannot silently change the pad.
inline constexpr int kLtx2VaePadZeros = 0;
inline constexpr int kLtx2VaePadReflect = 1;
inline constexpr int kLtx2VaePadReplicate = 2;

// Resolver. Throws when nothing is registered for (kLtx2Vae, device) — which
// cannot happen for kCPU in any build, but keeps the failure explicit rather
// than a null dereference on an unexpected backend.
const Ltx2VaeDeviceKernels* Ltx2VaeDevice(vt::DeviceType device);

// There is deliberately NO `Ltx2VaeDeviceKernelsAvailable` here, though the
// sibling `kLtx2` table has one (`ltx2_device.h`). The decode's single entry
// point resolves and refuses by name in one step -- `VaeKernels()` in
// ltx2_video_vae.cpp -- so a separate predicate would have no caller, and a
// landed function with no caller is the shape `AGENTS.md` `## Nothing lands
// dead` names. A fresh review found the first draft carrying exactly that,
// described in its own comment as guarding something it did not guard.

}  // namespace ltx2_vae
}  // namespace vllm
