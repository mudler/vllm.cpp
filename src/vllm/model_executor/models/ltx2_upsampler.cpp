// LTX-2.5 LATENT SPATIAL UPSAMPLER — see
// include/vllm/model_executor/models/ltx2_upsampler.h for the upstream mapping
// and the refusals.
//
// WHY THE CONVOLUTION IS LOCAL AND NOT THE VAE'S. `LatentUpsampler` builds plain
// `torch.nn.Conv3d`/`Conv2d` with `padding=1`, i.e. ZERO padding on every axis
// including time. `ltx2_video_vae.cpp`'s `CausalConv3d` prepends replicated
// copies of frame 0 instead (convolution.py:306-307). Reusing that kernel here
// would shift the whole clip while still producing a correctly shaped, finite,
// plausible latent — so the two stay separate, deliberately.
//
// TWO STORAGE WIDTHS SINCE A24 WAVE 5 (row LTX25-A24-UPSAMPLER-BF16, #2857).
// Upstream resolves ONE model dtype and it is bfloat16 (`distilled.py:109`),
// handed to the upsampler it constructs at `distilled.py:138-141`. `Volume`
// therefore carries a `vt::DType` and REAL bytes of that width, not an f32
// buffer holding narrowed values -- the wave 3 rule, because a token gate
// cannot see a dtype that is too wide.
//
// The compute width is unchanged and is NOT the storage width: every kernel
// still reduces in `double` and rounds ONCE into the volume's dtype. Six
// rounding rules decide where that single rounding falls, and each was MEASURED
// against the executed module rather than read off it. They are stated at their
// sites and tabulated in .agents/specs/ltx25-a24-upsampler-bf16.md section 3.
//
// ARITHMETIC WIDTH, stated once and referenced per site below. Upstream is f32
// everywhere in this file; every `double` here is an ESCAPE and each one is
// annotated at its site. They come in two kinds and only the first is justified
// by the suite's convention:
//
//   REDUCTIONS (conv accumulators, GroupNorm mean/var, the blur accumulator).
//     Upstream reduces in f32 but in a blocked/vectorized order no straight loop
//     reproduces, so accumulating exactly and rounding ONCE is the closest
//     single-rounding approximation to any order. This is the same escape L3
//     took and documented at ltx2_text_encoder.cpp:259-269.
//
//   POINTWISE (`Silu` here, `GeluTanh` in ltx2_duration_head.cpp). These are NOT
//     reductions, so the convention above does not cover them: upstream rounds to
//     f32 at each step of the expression and computing wider is numerically FINER
//     than the mirror, not equal to it — the polarity AGENTS.md warns about, where
//     a too-wide dtype still passes a value gate. Left as-is here rather than
//     narrowed in a review-repair branch, because narrowing moves the upsampler
//     and duration-head goldens and so owes its own red-first change. Recorded so
//     it is visible debt rather than an unremarked default.
//
//     ON THE bf16 ARM THIS ESCAPE IS INVISIBLE, and that is measured rather than
//     hoped: at a bf16 store the f32-vs-f64 accumulation difference sits below
//     one ulp over the shipped fan-in, and first parts at `mid_channels = 512`
//     (fan-in 13824) in ONE element of 768. The polarity warning above therefore
//     applies to the f32 arm and not to this one.
#include "vllm/model_executor/models/ltx2_upsampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/dtype.h"

// The temporal arm's refusal is shared with every other L5 out-of-scope feature,
// so it is raised through the one seam that names them all.
#include "vllm/model_executor/models/ltx2_pipeline.h"

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& message) { throw std::runtime_error(message); }

void Require(bool condition, const std::string& message) {
  if (!condition) Refuse(message);
}

// The two widths this file serves, refused by name in ONE place so a third
// cannot arrive by silence. FP8 and NVFP4 are A22. Mirrors `RequireVaeDType`
// (ltx2_video_vae.cpp), because a second refusal with its own wording is a
// second thing to keep in agreement.
void RequireUpsamplerDType(vt::DType dtype) {
  if (dtype != vt::DType::kF32 && dtype != vt::DType::kBF16) {
    Refuse(std::string("ltx2 upsampler: this stage serves f32 (the CPU parity arm every "
                       "committed golden is measured against) and bf16 (upstream's own model "
                       "dtype, distilled.py:109) storage. The FP8 and NVFP4 arms are A22 and "
                       "are not implemented; it was handed ") +
           vt::Name(dtype));
  }
}

// ─── THE STORAGE OBSERVABLE ──────────────────────────────────────────────────
//
// Declared at ltx2_upsampler.h's `Ltx2UpsamplerStorage`, where the reason lives:
// a bf16 arm that sizes its buffers by `sizeof(float)` passes every value gate
// this tree has, so the bytes have to be counted where they are reserved and
// where they are read. It accumulates and the reader CLEARS, so a caller
// brackets its own call and no byte is counted twice.
//
// `thread_local` and not a mutex: nothing in this file threads work, and a
// shared counter would let two concurrent renders report each other's bytes.
thread_local Ltx2UpsamplerStorage g_storage;

void RecordVolumeBytes(int64_t elems, size_t bytes) {
  ++g_storage.volumes;
  g_storage.elems += elems;
  g_storage.bytes += static_cast<int64_t>(bytes);
}

void RecordParamBytes(size_t elems, vt::DType read_width) {
  ++g_storage.param_views;
  g_storage.param_elems += static_cast<int64_t>(elems);
  g_storage.param_bytes += static_cast<int64_t>(elems * vt::SizeOf(read_width));
}

// A [channels, frames, height, width] volume at batch 1 — the shape every stage
// below operates on. Batch is carried by the caller loop.
//
// STORAGE IS THE DELIVERABLE. `bytes` holds `elems()` elements AT `dtype`, so a
// bf16 volume really is half an f32 one rather than an f32 buffer of narrowed
// values. Reads widen and writes round, through the same `LoadElem`/`StoreElem`
// shape `ltx2_video_vae.cpp:200-211` uses; `Set` is the ONE place a value
// becomes an element of this volume, which is what makes "round exactly once per
// store" a property of the type instead of a rule each kernel has to remember.
struct Volume {
  int64_t channels = 0, frames = 0, height = 0, width = 0;
  vt::DType dtype = vt::DType::kF32;
  std::vector<uint8_t> bytes;

  int64_t elems() const { return channels * frames * height * width; }
  size_t index(int64_t c, int64_t f, int64_t y, int64_t x) const {
    return static_cast<size_t>(((c * frames + f) * height + y) * width + x);
  }
  void Alloc() {
    RequireUpsamplerDType(dtype);
    bytes.assign(static_cast<size_t>(elems()) * vt::SizeOf(dtype), 0);
    // What was RESERVED, read back off the vector rather than recomputed from
    // `dtype`. Recomputing it here would be the tautology this instrument exists
    // to avoid: it would report the intended width on a buffer that took the
    // other one.
    RecordVolumeBytes(elems(), bytes.size());
  }
  // Shape and WIDTH from another volume. A derived volume that picked its own
  // dtype would put a bf16 input through an f32 output and reinterpret the bytes
  // rather than refuse, exactly as wave 3 found at its own seam.
  void LikeWidth(const Volume& other) { dtype = other.dtype; }

  float Load(size_t i) const {
    return dtype == vt::DType::kBF16
               ? vt::BF16ToF32(reinterpret_cast<const uint16_t*>(bytes.data())[i])
               : reinterpret_cast<const float*>(bytes.data())[i];
  }
  void Store(size_t i, float v) {
    if (dtype == vt::DType::kBF16) {
      reinterpret_cast<uint16_t*>(bytes.data())[i] = vt::F32ToBF16(v);
    } else {
      reinterpret_cast<float*>(bytes.data())[i] = v;
    }
  }
  float at(int64_t c, int64_t f, int64_t y, int64_t x) const {
    return Load(index(c, f, y, x));
  }
  void Set(int64_t c, int64_t f, int64_t y, int64_t x, float v) {
    Store(index(c, f, y, x), v);
  }
};

// A parameter read AT THE BAG'S OWN WIDTH. `Ltx2VaeWeights` populates exactly one
// of its two maps (ltx2_audio_vae.h:71-85) and `Get` refuses on a bf16 bag, so a
// bf16 checkpoint could not reach this file at all before this row.
//
// It is a VIEW and not a widened copy on purpose. Widening every parameter to
// f32 on the way in would make the bf16 arm hold the same bytes the f32 arm
// does, which is the widening A24 exists to remove -- at the shipped
// `mid_channels = 512` one convolution weight alone is 7.1 M parameters.
class WeightView {
 public:
  WeightView(const Ltx2VaeWeights& weights, const std::string& name) {
    if (weights.dtype == vt::DType::kBF16) {
      bf16_ = &weights.GetBf16(name);
    } else {
      f32_ = &weights.Get(name);
    }
    // The bytes this view will be READ THROUGH, taken off `read_width()`, which
    // dispatches on the same member `operator[]` does. That is what makes the
    // number a measurement of the claim above rather than a restatement of the
    // bag's `dtype`: a view that materialised a widened f32 copy would read
    // through the copy, and would report f32's four bytes per element.
    RecordParamBytes(size(), read_width());
  }
  size_t size() const { return bf16_ != nullptr ? bf16_->size() : f32_->size(); }
  // The storage `operator[]` actually reads. Kept beside it, and derived from the
  // same member, so the two cannot disagree.
  vt::DType read_width() const {
    return bf16_ != nullptr ? vt::DType::kBF16 : vt::DType::kF32;
  }
  float operator[](size_t i) const {
    return bf16_ != nullptr ? vt::BF16ToF32((*bf16_)[i]) : (*f32_)[i];
  }

 private:
  const std::vector<float>* f32_ = nullptr;
  const std::vector<uint16_t>* bf16_ = nullptr;
};

// AND THE VIEW IS A VIEW BY CONSTRUCTION, not only by what it reports.
// `RecordParamBytes` dispatches on `read_width()`, which reads the same member
// `operator[]` reads -- so the counter catches a widened copy that reports
// honestly, and a mutation that edited BOTH lines together would defeat it. That
// is not a limit of this process, it is a limit of one instrument, and the shape
// it misses is exactly the shape a size closes: an owned copy has storage, and
// storage is bytes. Two pointers is what a view costs.
static_assert(sizeof(WeightView) == 2 * sizeof(void*),
              "WeightView must be a VIEW: two pointers and no owned storage");

WeightView W(const Ltx2VaeWeights& weights, const std::string& name) {
  return WeightView(weights, name);
}

// `torch.nn.Conv3d(in, out, kernel_size=3, padding=1)` — zero padding on ALL
// three axes, unlike the VAE's causal replication.
Volume Conv3dPad1(const Volume& in, int64_t out_channels, const WeightView& weight,
                  const WeightView& bias) {
  constexpr int64_t k = 3;
  constexpr int64_t pad = 1;
  Volume out;
  out.channels = out_channels;
  out.frames = in.frames;
  out.height = in.height;
  out.width = in.width;
  out.LikeWidth(in);
  out.Alloc();
  Require(weight.size() ==
              static_cast<size_t>(out_channels * in.channels * k * k * k),
          "ltx2 upsampler: conv3d weight has the wrong element count");
  Require(bias.size() == static_cast<size_t>(out_channels),
          "ltx2 upsampler: conv3d bias has the wrong element count");

  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t f = 0; f < out.frames; ++f) {
      for (int64_t y = 0; y < out.height; ++y) {
        for (int64_t x = 0; x < out.width; ++x) {
          // f64 REDUCTION escape -- see the width note in the file header.
          double acc = static_cast<double>(bias[static_cast<size_t>(oc)]);
          for (int64_t ic = 0; ic < in.channels; ++ic) {
            for (int64_t kf = 0; kf < k; ++kf) {
              const int64_t sf = f + kf - pad;
              if (sf < 0 || sf >= in.frames) continue;
              for (int64_t ky = 0; ky < k; ++ky) {
                const int64_t sy = y + ky - pad;
                if (sy < 0 || sy >= in.height) continue;
                for (int64_t kx = 0; kx < k; ++kx) {
                  const int64_t sx = x + kx - pad;
                  if (sx < 0 || sx >= in.width) continue;
                  const size_t widx = static_cast<size_t>(
                      (((oc * in.channels + ic) * k + kf) * k + ky) * k + kx);
                  acc += static_cast<double>(weight[widx]) *
                         static_cast<double>(in.at(ic, sf, sy, sx));
                }
              }
            }
          }
          // R1: the single rounding, into the volume's own width. MEASURED
          // against upstream: at bf16 this reproduces `torch.nn.Conv3d`'s own
          // output, and the f64-vs-f32 accumulator question that dominates the
          // f32 arm separates NOTHING here (section 3 R1 of the row's spec).
          out.Set(oc, f, y, x, static_cast<float>(acc));
        }
      }
    }
  }
  return out;
}

// `torch.nn.Conv2d(in, out, kernel_size=3, padding=1)` applied PER FRAME — what
// upstream reaches by `rearrange(x, "b c f h w -> (b f) c h w")` (model.py:117,
// spatial_rational_resampler.py:42).
Volume Conv2dPad1PerFrame(const Volume& in, int64_t out_channels, const WeightView& weight,
                          const WeightView& bias) {
  constexpr int64_t k = 3;
  constexpr int64_t pad = 1;
  Volume out;
  out.channels = out_channels;
  out.frames = in.frames;
  out.height = in.height;
  out.width = in.width;
  out.LikeWidth(in);
  out.Alloc();
  Require(weight.size() == static_cast<size_t>(out_channels * in.channels * k * k),
          "ltx2 upsampler: conv2d weight has the wrong element count");
  // The bias half of the contract, which `Conv3dPad1` above has always had and
  // this helper did not. Indexing `bias[oc]` unchecked reads past the end of a
  // std::vector when a checkpoint carries a correct kernel and a short bias --
  // UB and silent garbage, where the identical 3-D defect gets a named refusal.
  // Nothing upstream of here validates it: `Ltx2LoadVaeWeights` loads by NAME and
  // checks no shape. The dims=2 arm routes four parameter groups through this
  // helper, so it is the path that made the gap reachable.
  Require(bias.size() == static_cast<size_t>(out_channels),
          "ltx2 upsampler: conv2d bias has the wrong element count");

  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t f = 0; f < out.frames; ++f) {
      for (int64_t y = 0; y < out.height; ++y) {
        for (int64_t x = 0; x < out.width; ++x) {
          // f64 REDUCTION escape -- see the width note in the file header.
          double acc = static_cast<double>(bias[static_cast<size_t>(oc)]);
          for (int64_t ic = 0; ic < in.channels; ++ic) {
            for (int64_t ky = 0; ky < k; ++ky) {
              const int64_t sy = y + ky - pad;
              if (sy < 0 || sy >= in.height) continue;
              for (int64_t kx = 0; kx < k; ++kx) {
                const int64_t sx = x + kx - pad;
                if (sx < 0 || sx >= in.width) continue;
                const size_t widx =
                    static_cast<size_t>(((oc * in.channels + ic) * k + ky) * k + kx);
                acc += static_cast<double>(weight[widx]) *
                       static_cast<double>(in.at(ic, f, sy, sx));
              }
            }
          }
          out.Set(oc, f, y, x, static_cast<float>(acc));  // R1, as above
        }
      }
    }
  }
  return out;
}

// `torch.nn.GroupNorm(32, channels)`: statistics over (channels_per_group, F, H, W)
// per group, per sample. The group COUNT is a literal upstream, not a config key.
//
// R2 AND R3, both MEASURED against the executed module and both per-SITE:
//
//   R2  The statistics, the normalization and the AFFINE all happen at compute
//       width and the result is rounded ONCE. Rounding the normalized value and
//       then applying the affine is a different function -- it separates in 394
//       of 1536 elements. Note that at GroupNorm's DEFAULT init (`weight = 1`,
//       `bias = 0`) the two are the same expression, so a probe that leaves the
//       affine at identity measures nothing; the goldens randomize it.
//   R3  `eps` STAYS f32. It is a plain Python attribute of `torch.nn.GroupNorm`
//       and not a registered buffer, so `.to(bfloat16)` does not touch it -- the
//       distinction this campaign has now paid for four times. Adding the bf16
//       value instead separates 0 elements at variance ~1 and 227 at ~1e-6,
//       which is why the golden carries a SMALL-VARIANCE case whose only job is
//       to make this width observable.
void GroupNorm(Volume& x, const WeightView& weight, const WeightView& bias) {
  const int64_t groups = kLtx2UpsamplerNormGroups;
  Require(x.channels % groups == 0,
          "ltx2 upsampler: GroupNorm(32) requires channels divisible by 32, got " +
              std::to_string(x.channels));
  const int64_t per_group = x.channels / groups;
  const int64_t spatial = x.frames * x.height * x.width;
  const int64_t elems = per_group * spatial;

  for (int64_t g = 0; g < groups; ++g) {
    // f64 REDUCTION escape (mean and var) -- see the width note in the header.
    double mean = 0.0;
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        mean += static_cast<double>(x.Load(static_cast<size_t>(c * spatial + i)));
      }
    }
    mean /= static_cast<double>(elems);
    double var = 0.0;
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double d = static_cast<double>(x.Load(static_cast<size_t>(c * spatial + i))) - mean;
        var += d * d;
      }
    }
    // torch's GroupNorm uses the BIASED variance (divide by N), unlike `std`.
    var /= static_cast<double>(elems);
    const double inv = 1.0 / std::sqrt(var + kLtx2UpsamplerNormEps);
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      const double gain = static_cast<double>(weight[static_cast<size_t>(c)]);
      const double shift = static_cast<double>(bias[static_cast<size_t>(c)]);
      for (int64_t i = 0; i < spatial; ++i) {
        const size_t idx = static_cast<size_t>(c * spatial + i);
        const double value = static_cast<double>(x.Load(idx));
        x.Store(idx, static_cast<float>((value - mean) * inv * gain + shift));
      }
    }
  }
}

// R4, MEASURED: upstream computes `x * sigmoid(x)` at compute width and rounds
// ONCE. Rounding the sigmoid to bf16 before the multiply separates in 952 of
// 4096 elements, so it is a different function and not a smaller error.
void Silu(Volume& x) {
  const size_t n = static_cast<size_t>(x.elems());
  for (size_t i = 0; i < n; ++i) {
    // POINTWISE f64, WIDER than upstream's f32 -- see the width note in the
    // file header. Not covered by the reduction convention.
    const double v = static_cast<double>(x.Load(i));
    x.Store(i, static_cast<float>(v / (1.0 + std::exp(-v))));
  }
}

// ResBlock.forward (res_block.py:29-37). The residual is added BEFORE the
// activation — `activation(x + residual)`, not `activation(x) + residual`.
// `two_d` selects the same way `res_block.py:21` does. It is a parameter rather
// than two functions because upstream expresses the difference as one
// constructor choice over an otherwise identical block, and a second copy here
// would be a second thing to keep in agreement for no gain.
Volume ResBlockForward(const Ltx2VaeWeights& weights, const std::string& prefix, bool two_d,
                       const Volume& in) {
  const auto conv = [&](const Volume& v, int64_t out_ch, const WeightView& w,
                        const WeightView& b) {
    return two_d ? Conv2dPad1PerFrame(v, out_ch, w, b) : Conv3dPad1(v, out_ch, w, b);
  };
  Volume x = conv(in, in.channels, W(weights, prefix + "conv1.weight"),
                  W(weights, prefix + "conv1.bias"));
  GroupNorm(x, W(weights, prefix + "norm1.weight"), W(weights, prefix + "norm1.bias"));
  Silu(x);
  Volume y = conv(x, in.channels, W(weights, prefix + "conv2.weight"),
                  W(weights, prefix + "conv2.bias"));
  GroupNorm(y, W(weights, prefix + "norm2.weight"), W(weights, prefix + "norm2.bias"));
  // R5, MEASURED: `x + residual` is a TENSOR OPERATION upstream, so it lands in
  // the model dtype BEFORE the activation reads it. Keeping the sum at compute
  // width and activating that is a different function -- it separates in 793 of
  // 4096 elements. `Store` is what makes the rounding happen, which is why the
  // sum is written back rather than carried in a local.
  const size_t n = static_cast<size_t>(y.elems());
  for (size_t i = 0; i < n; ++i) y.Store(i, y.Load(i) + in.Load(i));
  Silu(y);
  return y;
}

// PixelShuffleND(2) — `b (c p1 p2) h w -> b c (h p1) (w p2)` (pixel_shuffle.py:41-47).
// p1 takes HEIGHT and p2 takes WIDTH; swapping them transposes every block.
Volume PixelShuffle2d(const Volume& in, int64_t up_h, int64_t up_w) {
  Require(in.channels % (up_h * up_w) == 0,
          "ltx2 upsampler: pixel shuffle requires channels divisible by the upscale product");
  Volume out;
  out.channels = in.channels / (up_h * up_w);
  out.frames = in.frames;
  out.height = in.height * up_h;
  out.width = in.width * up_w;
  // MOVEMENT, not arithmetic: `rearrange` never rounds, so this copies elements
  // at the volume's own width and adds no rounding site.
  out.LikeWidth(in);
  out.Alloc();
  for (int64_t c = 0; c < out.channels; ++c) {
    for (int64_t p1 = 0; p1 < up_h; ++p1) {
      for (int64_t p2 = 0; p2 < up_w; ++p2) {
        const int64_t src_c = (c * up_h + p1) * up_w + p2;
        for (int64_t f = 0; f < in.frames; ++f) {
          for (int64_t y = 0; y < in.height; ++y) {
            for (int64_t x = 0; x < in.width; ++x) {
              out.Set(c, f, y * up_h + p1, x * up_w + p2, in.at(src_c, f, y, x));
            }
          }
        }
      }
    }
  }
  return out;
}

// PixelShuffleND(1) — `b (c p1) f h w -> b c (f p1) h w` (pixel_shuffle.py:47-52).
// Both groupings put `p1` FASTEST: the source channel is `c * p1 + j` because the
// pattern is `(c p1)`, and the destination frame is `f * p1 + j` because it is
// `(f p1)`. Reversing either factor order yields a correctly shaped, finite,
// plausible latent, which is why the golden — not a shape check — is what holds
// this down.
Volume PixelShuffle1d(const Volume& in, int64_t up_f) {
  Require(in.channels % up_f == 0,
          "ltx2 upsampler: temporal pixel shuffle requires channels divisible by " +
              std::to_string(up_f));
  Volume out;
  out.channels = in.channels / up_f;
  out.frames = in.frames * up_f;
  out.height = in.height;
  out.width = in.width;
  out.LikeWidth(in);  // movement, as above
  out.Alloc();
  for (int64_t c = 0; c < out.channels; ++c) {
    for (int64_t j = 0; j < up_f; ++j) {
      const int64_t src_c = c * up_f + j;
      for (int64_t f = 0; f < in.frames; ++f) {
        for (int64_t y = 0; y < in.height; ++y) {
          for (int64_t x = 0; x < in.width; ++x) {
            out.Set(c, f * up_f + j, y, x, in.at(src_c, f, y, x));
          }
        }
      }
    }
  }
  return out;
}

// `x = x[:, :, 1:, :, :]` (model.py:111-113): "remove the first frame after
// upsampling. This is done because the first frame encodes one pixel frame."
// Frames go 2F -> 2F - 1, which is exactly the count the only upstream consumer
// keeps for itself (`num_frames = 2 * (num_frames - 1) + 1`, dfr_pipeline.py:408).
Volume DropFirstFrame(const Volume& in) {
  Require(in.frames >= 2,
          "ltx2 upsampler: the temporal arm drops the first frame after upsampling "
          "(model/upsampler/model.py:109-113), so it needs at least 2 frames out of the "
          "shuffle");
  Volume out;
  out.channels = in.channels;
  out.frames = in.frames - 1;
  out.height = in.height;
  out.width = in.width;
  out.LikeWidth(in);  // a slice, so movement again
  out.Alloc();
  for (int64_t c = 0; c < out.channels; ++c) {
    for (int64_t f = 0; f < out.frames; ++f) {
      for (int64_t y = 0; y < out.height; ++y) {
        for (int64_t x = 0; x < out.width; ++x) {
          out.Set(c, f, y, x, in.at(c, f + 1, y, x));
        }
      }
    }
  }
  return out;
}

// BlurDownsample._apply_2d (blur_downsample.py:49-53): a DEPTHWISE conv2d with
// the fixed binomial kernel, stride `den`, padding `kernel_size // 2`, per frame.
Volume BlurDownsample(const Volume& in, int64_t stride, int64_t kernel_size) {
  if (stride == 1) return in;  // :36-37, the short circuit
  // R6, MEASURED, and it is the rule that reads as a no-op and is not.
  // `BlurDownsample` REGISTERS its binomial kernel as a buffer
  // (blur_downsample.py:33), and a registered buffer is exactly what
  // `.to(bfloat16)` narrows -- unlike a Python float, which it does not touch.
  // So the kernel this arm convolves with is the NARROWED one, and it is
  // narrowed here for that reason and not for symmetry.
  //
  // At the pinned `kernel_size = 5` the narrowing changes NO entry, because
  // every value is `{1,4,6,16,24,36}/256`, a dyadic rational bf16 holds exactly.
  // That is reported rather than used to skip the cast: at kernel_size 9 one
  // entry moves and at 11 fifty-seven do, so the site is live and only this
  // width is quiet.
  std::vector<float> kernel = Ltx2BlurKernel(kernel_size);
  if (in.dtype == vt::DType::kBF16) {
    for (float& k_value : kernel) k_value = vt::BF16ToF32(vt::F32ToBF16(k_value));
  }
  const int64_t pad = kernel_size / 2;
  Volume out;
  out.channels = in.channels;
  out.frames = in.frames;
  out.height = (in.height + 2 * pad - kernel_size) / stride + 1;
  out.width = (in.width + 2 * pad - kernel_size) / stride + 1;
  out.LikeWidth(in);
  out.Alloc();

  for (int64_t c = 0; c < in.channels; ++c) {
    for (int64_t f = 0; f < in.frames; ++f) {
      for (int64_t y = 0; y < out.height; ++y) {
        for (int64_t x = 0; x < out.width; ++x) {
          // f64 REDUCTION escape -- see the width note in the file header.
          double acc = 0.0;
          for (int64_t ky = 0; ky < kernel_size; ++ky) {
            const int64_t sy = y * stride + ky - pad;
            if (sy < 0 || sy >= in.height) continue;
            for (int64_t kx = 0; kx < kernel_size; ++kx) {
              const int64_t sx = x * stride + kx - pad;
              if (sx < 0 || sx >= in.width) continue;
              acc += static_cast<double>(kernel[static_cast<size_t>(ky * kernel_size + kx)]) *
                     static_cast<double>(in.at(c, f, sy, sx));
            }
          }
          out.Set(c, f, y, x, static_cast<float>(acc));  // one rounding, as R1
        }
      }
    }
  }
  return out;
}

}  // namespace

Ltx2UpsamplerStorage Ltx2TakeUpsamplerStorage() {
  const Ltx2UpsamplerStorage taken = g_storage;
  g_storage = Ltx2UpsamplerStorage();
  return taken;
}

Ltx2RationalScale Ltx2RationalForScale(double scale) {
  // spatial_rational_resampler.py:11-14, exactly this map and no nearest match.
  if (scale == 0.75) return {3, 4};
  if (scale == 1.5) return {3, 2};
  if (scale == 2.0) return {2, 1};
  if (scale == 4.0) return {4, 1};
  Refuse("ltx2 upsampler: Unsupported scale " + std::to_string(scale) +
         ". Choose from [0.75, 1.5, 2.0, 4.0] (spatial_rational_resampler.py:11-14).");
}

std::vector<float> Ltx2BlurKernel(int64_t kernel_size) {
  Require(kernel_size >= 3 && kernel_size % 2 == 1,
          "ltx2 upsampler: BlurDownsample kernel_size must be odd and >= 3, got " +
              std::to_string(kernel_size));
  // blur_downsample.py:29-33: Pascal's row `kernel_size - 1`, outer-producted and
  // normalized to sum 1. Built in integer arithmetic then normalized ONCE, which
  // is what upstream's `k2d / k2d.sum()` on an integer tensor does.
  std::vector<double> row(static_cast<size_t>(kernel_size));
  double value = 1.0;
  for (int64_t i = 0; i < kernel_size; ++i) {
    row[static_cast<size_t>(i)] = value;
    value = value * static_cast<double>(kernel_size - 1 - i) / static_cast<double>(i + 1);
  }
  double total = 0.0;
  for (int64_t y = 0; y < kernel_size; ++y) {
    for (int64_t x = 0; x < kernel_size; ++x) {
      total += row[static_cast<size_t>(y)] * row[static_cast<size_t>(x)];
    }
  }
  std::vector<float> kernel(static_cast<size_t>(kernel_size * kernel_size));
  for (int64_t y = 0; y < kernel_size; ++y) {
    for (int64_t x = 0; x < kernel_size; ++x) {
      kernel[static_cast<size_t>(y * kernel_size + x)] = static_cast<float>(
          row[static_cast<size_t>(y)] * row[static_cast<size_t>(x)] / total);
    }
  }
  return kernel;
}

std::vector<Ltx2UpsamplerTensorSpec> EnumerateLtx2UpsamplerTensors(
    const Ltx2UpsamplerConfig& config) {
  // `named_parameters()` order, the same for both ranks: initial_conv,
  // initial_norm, res_blocks, upsampler, post_upsample_res_blocks, final_conv.
  const std::string p = config.prefix;
  const int64_t in_c = config.in_channels;
  const int64_t mid = config.mid_channels;
  std::vector<Ltx2UpsamplerTensorSpec> specs;

  // model.py:47 — `conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d`.
  // That ONE line reaches four parameter groups: `initial_conv` (:49), both
  // ResBlock stacks (:53 and :76-78, whose own conv is chosen the same way at
  // res_block.py:21) and `final_conv` (:80). The `upsampler` branch (:55-72)
  // never reads `dims`, so it is deliberately NOT built through this helper —
  // its rank is decided by the two FLAGS instead, below.
  //
  // The `else` is upstream's and is mirrored as written: any `dims` that is not
  // 2 builds Conv3d there, so 3 is the shipped value rather than the only legal
  // one, and this port does not invent a refusal upstream does not raise.
  const auto conv_w = [&](int64_t out_ch, int64_t in_ch) {
    return config.dims == 2 ? std::vector<int64_t>{out_ch, in_ch, 3, 3}
                            : std::vector<int64_t>{out_ch, in_ch, 3, 3, 3};
  };

  specs.push_back({p + "initial_conv.weight", conv_w(mid, in_c)});
  specs.push_back({p + "initial_conv.bias", {mid}});
  specs.push_back({p + "initial_norm.weight", {mid}});
  specs.push_back({p + "initial_norm.bias", {mid}});

  auto res_block = [&](const std::string& prefix) {
    specs.push_back({prefix + "conv1.weight", conv_w(mid, mid)});
    specs.push_back({prefix + "conv1.bias", {mid}});
    specs.push_back({prefix + "norm1.weight", {mid}});
    specs.push_back({prefix + "norm1.bias", {mid}});
    specs.push_back({prefix + "conv2.weight", conv_w(mid, mid)});
    specs.push_back({prefix + "conv2.bias", {mid}});
    specs.push_back({prefix + "norm2.weight", {mid}});
    specs.push_back({prefix + "norm2.bias", {mid}});
  };
  for (int64_t i = 0; i < config.num_blocks_per_stage; ++i) {
    res_block(p + "res_blocks." + std::to_string(i) + ".");
  }

  if (config.temporal_upsample && !config.spatial_upsample) {
    // `torch.nn.Sequential(Conv3d(mid, 2*mid, 3, padding=1), PixelShuffleND(1))`
    // names its conv `upsampler.0` (model.py:68-71). Note the RANK: this is a
    // Conv3d, so the weight is 5-D, where the non-rational SPATIAL arm's
    // identically-named tensor is a 4-D Conv2d kernel (model.py:64-66).
    specs.push_back({p + "upsampler.0.weight",
                     {kLtx2UpsamplerTemporalFactor * mid, mid, 3, 3, 3}});
    specs.push_back({p + "upsampler.0.bias", {kLtx2UpsamplerTemporalFactor * mid}});
  } else if (config.spatial_upsample && !config.temporal_upsample) {
    if (config.rational_resampler) {
      // SpatialRationalResampler names its conv `upsampler.conv`
      // (spatial_rational_resampler.py:36); the blur kernel is a BUFFER, not a
      // parameter, so it never appears here.
      const Ltx2RationalScale rational = Ltx2RationalForScale(config.spatial_scale);
      specs.push_back({p + "upsampler.conv.weight",
                       {rational.num * rational.num * mid, mid, 3, 3}});
      specs.push_back({p + "upsampler.conv.bias", {rational.num * rational.num * mid}});
    } else {
      // `torch.nn.Sequential(Conv2d, PixelShuffleND)` names its conv `upsampler.0`
      // (model.py:64-67).
      specs.push_back({p + "upsampler.0.weight", {4 * mid, mid, 3, 3}});
      specs.push_back({p + "upsampler.0.bias", {4 * mid}});
    }
  }

  for (int64_t i = 0; i < config.num_blocks_per_stage; ++i) {
    res_block(p + "post_upsample_res_blocks." + std::to_string(i) + ".");
  }
  specs.push_back({p + "final_conv.weight", conv_w(in_c, mid)});
  specs.push_back({p + "final_conv.bias", {in_c}});
  return specs;
}

Ltx2LatentVolume Ltx2LatentUpsample(const Ltx2UpsamplerConfig& config,
                                    const Ltx2VaeWeights& weights,
                                    const Ltx2LatentVolume& latent) {
  // model.py:73-74 — upstream's own ValueError, raised at CONSTRUCTION there and
  // here at the first forward, which is the earliest this port sees the config.
  Require(config.spatial_upsample || config.temporal_upsample,
          "ltx2 upsampler: either spatial_upsample or temporal_upsample must be True "
          "(model/upsampler/model.py:73-74)");
  // The SPATIOTEMPORAL arm is a different operator, not "the temporal arm with
  // spatial on": model.py:55-59 builds `Conv3d(mid, 8*mid)` + `PixelShuffleND(3)`.
  // Refused BEFORE any weight is touched, so it reports an unported feature
  // rather than a wrong element count on `upsampler.0.weight`.
  if (config.temporal_upsample && config.spatial_upsample) {
    Ltx2RefuseUnportedPipelineFeature(Ltx2UnportedPipelineFeature::kSpatiotemporalUpsampler);
  }
  // model.py:47's `else` is mirrored as written: `dims == 2` is Conv2d and
  // EVERYTHING else is Conv3d, so there is no refusal here — upstream raises
  // none, and inventing one would refuse a checkpoint upstream runs.
  const bool two_d = config.dims == 2;
  // The one combination upstream cannot run. `dims=2` takes the forward at :85,
  // which hands `self.upsampler` a 4-D tensor, while `temporal_upsample` built
  // that module as a Conv3d (:57 for both flags, :70 for temporal alone).
  //
  // WHAT ACTUALLY FIRES UPSTREAM IS NOT THE RANK, and this comment said it was.
  // `Conv3d` accepts a 4-D input as an UNBATCHED 5-D one, so the folded
  // `(b*f, c, h, w)` is read as `(C, D, H, W)` and the CHANNEL count is what
  // disagrees. Measured at the pin with `_UPS_MID = 32` and 3 frames:
  //   RuntimeError: Given groups=1, weight of size [64, 32, 3, 3, 3], expected
  //   input[1, 3, 32, 4, 6] to have 32 channels, but got 3 channels instead
  // The conclusion survives the correction and the reasoning does not: at
  // `frames == mid_channels` that Conv3d would PASS, and `PixelShuffleND(1)`'s
  // einops pattern (pixel_shuffle.py:47-52) is what fails instead. So upstream
  // refuses the configuration by two different mechanisms depending on a shape
  // coincidence, which is exactly why this refuses it by NAME once and up front.
  // `scripts/gen-ltx2-pipeline-goldens.py` runs both contradictions through the
  // real module and asserts the message, so a pin that changed either mechanism
  // fails the generator rather than leaving this paragraph wrong again.
  //
  // ORDER IS PART OF THE MESSAGE. This `Require` comes BEFORE the rational one
  // because upstream's `__init__` is an if/elif chain: `elif temporal_upsample`
  // (model.py:68) builds the Conv3d and never consults `rational_resampler`, so
  // a config with BOTH flags at dims=2 owns no SpatialRationalResampler and must
  // not be told about one. Refuse/accept polarity is the same either way -- the
  // eighth cell is refused whichever fires -- which is exactly why only reading
  // the message catches a swap back.
  Require(!(two_d && config.temporal_upsample),
          "ltx2 upsampler: dims=2 with temporal_upsample=true is not a configuration upstream "
          "can run. model/upsampler/model.py:68-71 builds the temporal upsampler as a Conv3d, "
          "and the dims=2 forward (:85-100) folds the frame axis into the batch and so feeds "
          "it a 4-D tensor. The frame axis is gone by then, which is why no 2-D arm can "
          "upsample it.");

  // The SIBLING contradiction, and it fails differently from the temporal one:
  // every operator inside the rational branch is already per-frame, so a folded
  // one-frame volume satisfies all of them and this config computes a complete,
  // finite, plausible latent. Upstream cannot: `SpatialRationalResampler.forward`
  // opens with `b, _, f, _, _ = x.shape`
  // (model/upsampler/spatial_rational_resampler.py:41) and the dims=2 forward
  // reaches it at model.py:94 having already folded the frame axis away at :86,
  // so torch raises `not enough values to unpack (expected 5, got 4)`. No shape
  // check here can see this one, which is why it has to be a refusal.
  Require(!(two_d && config.rational_resampler),
          "ltx2 upsampler: dims=2 with rational_resampler=true is not a configuration upstream "
          "can run. SpatialRationalResampler.forward unpacks five dimensions at "
          "model/upsampler/spatial_rational_resampler.py:41, and the dims=2 forward "
          "(model/upsampler/model.py:85-100) folds the frame axis into the batch at :86 before "
          "reaching it at :94, so it is handed a 4-D tensor and raises.");
  Require(latent.channels == config.in_channels,
          "ltx2 upsampler: latent has " + std::to_string(latent.channels) +
              " channels, config declares " + std::to_string(config.in_channels));

  // THE ARM IS THE BAG'S, exactly as waves 2-4 route it: `Ltx2VaeWeights`
  // populates one of its two maps and says which (ltx2_audio_vae.h:71-85). It is
  // not a parameter of this function, because a caller that could pick a width
  // the checkpoint is not stored at would reinterpret the parameter bytes rather
  // than refuse.
  RequireUpsamplerDType(weights.dtype);
  const vt::DType dtype = weights.dtype;

  const std::string p = config.prefix;
  Ltx2LatentVolume result;
  result.batch = latent.batch;
  result.channels = config.in_channels;

  // model.py:47 again, at the four call sites rather than at the shapes.
  const auto conv = [&](const Volume& v, int64_t out_ch, const WeightView& w,
                        const WeightView& b_) {
    return two_d ? Conv2dPad1PerFrame(v, out_ch, w, b_) : Conv3dPad1(v, out_ch, w, b_);
  };

  // ONE stack for both ranks, because upstream's two forward branches
  // (model.py:87-99 and :102-124) run the identical module sequence and differ
  // only in the fold around them and in the rank of the four convolutions
  // inside. Writing it twice would be the parallel path AGENTS.md forbids.
  const auto run_stack = [&](Volume x) {
    // model.py:87-89 / :102-104.
    x = conv(x, config.mid_channels, W(weights, p + "initial_conv.weight"),
             W(weights, p + "initial_conv.bias"));
    GroupNorm(x, W(weights, p + "initial_norm.weight"), W(weights, p + "initial_norm.bias"));
    Silu(x);

    for (int64_t i = 0; i < config.num_blocks_per_stage; ++i) {
      x = ResBlockForward(weights, p + "res_blocks." + std::to_string(i) + ".", two_d, x);
    }

    if (config.temporal_upsample) {
      // model.py:109-113, and the branch order is upstream's: `if
      // self.temporal_upsample` (:109) is tested BEFORE the resampler check
      // (:114). A full 3-D conv (not per-frame): the temporal arm's
      // `upsampler.0` is a Conv3d (model.py:70), unlike the spatial arm's Conv2d.
      // Unreachable when `two_d`, which is refused above.
      x = Conv3dPad1(x, kLtx2UpsamplerTemporalFactor * config.mid_channels,
                     W(weights, p + "upsampler.0.weight"), W(weights, p + "upsampler.0.bias"));
      x = PixelShuffle1d(x, kLtx2UpsamplerTemporalFactor);
      x = DropFirstFrame(x);
    } else if (config.rational_resampler) {
      // SpatialRationalResampler.forward (:40-47): per-frame conv, pixel shuffle
      // by `num`, then an anti-aliased stride-`den` blur.
      const Ltx2RationalScale rational = Ltx2RationalForScale(config.spatial_scale);
      x = Conv2dPad1PerFrame(x, rational.num * rational.num * config.mid_channels,
                             W(weights, p + "upsampler.conv.weight"),
                             W(weights, p + "upsampler.conv.bias"));
      x = PixelShuffle2d(x, rational.num, rational.num);
      x = BlurDownsample(x, rational.den, kLtx2BlurKernelSize);
    } else {
      // model.py:117-119 — per-frame Conv2d then PixelShuffleND(2). Already
      // per-frame for BOTH ranks, because the `upsampler` branch never reads
      // `dims`: at dims=2 the fold has already made the volume one frame, and
      // the operator is the same either way.
      x = Conv2dPad1PerFrame(x, 4 * config.mid_channels, W(weights, p + "upsampler.0.weight"),
                             W(weights, p + "upsampler.0.bias"));
      x = PixelShuffle2d(x, 2, 2);
    }

    for (int64_t i = 0; i < config.num_blocks_per_stage; ++i) {
      x = ResBlockForward(weights, p + "post_upsample_res_blocks." + std::to_string(i) + ".",
                          two_d, x);
    }
    x = conv(x, config.in_channels, W(weights, p + "final_conv.weight"),
             W(weights, p + "final_conv.bias"));
    return x;
  };

  for (int64_t b = 0; b < latent.batch; ++b) {
    Volume x;
    x.channels = latent.channels;
    x.frames = latent.frames;
    x.height = latent.height;
    x.width = latent.width;
    // The interface value stays f32 and the STORAGE does not, which is the same
    // split `Ltx2ConvVideoEncode` makes: a latent is what a caller holds, and
    // `Ltx2LatentVolume` is that. Rounding here is mirroring rather than
    // truncation -- upstream's latent at this point came out of a bf16 DiT and
    // is already at this width.
    x.dtype = dtype;
    x.Alloc();
    const int64_t stride = latent.channels * latent.frames * latent.height * latent.width;
    for (int64_t i = 0; i < stride; ++i) {
      x.Store(static_cast<size_t>(i), latent.data[static_cast<size_t>(b * stride + i)]);
    }

    if (two_d) {
      // model.py:86 — `rearrange(latent, "b c f h w -> (b f) c h w")`, undone at
      // :100. Running one frame at a time IS that fold: `(b f)` makes every
      // frame its own sample, and GroupNorm here reduces over
      // `frames * height * width`, so a ONE-frame volume hands it exactly the
      // per-frame statistic upstream computes. Passing the whole clip to the
      // 2-D convolutions instead would agree on every shape and be wrong in
      // every element, which is why the dims=2 golden — not a shape check — is
      // what holds this down.
      const int64_t in_plane = x.height * x.width;
      Volume folded;
      for (int64_t f = 0; f < x.frames; ++f) {
        Volume plane;
        plane.channels = x.channels;
        plane.frames = 1;
        plane.height = x.height;
        plane.width = x.width;
        plane.LikeWidth(x);
        plane.Alloc();
        for (int64_t c = 0; c < x.channels; ++c) {
          for (int64_t i = 0; i < in_plane; ++i) {
            plane.Store(static_cast<size_t>(c * in_plane + i),
                        x.Load(static_cast<size_t>((c * x.frames + f) * in_plane + i)));
          }
        }
        plane = run_stack(plane);
        if (f == 0) {
          folded.channels = plane.channels;
          folded.frames = x.frames;
          folded.height = plane.height;
          folded.width = plane.width;
          folded.LikeWidth(plane);
          folded.Alloc();
        }
        const int64_t out_plane = plane.height * plane.width;
        for (int64_t c = 0; c < plane.channels; ++c) {
          for (int64_t i = 0; i < out_plane; ++i) {
            folded.Store(static_cast<size_t>((c * folded.frames + f) * out_plane + i),
                         plane.Load(static_cast<size_t>(c * out_plane + i)));
          }
        }
      }
      x = folded;
    } else {
      x = run_stack(x);
    }

    result.frames = x.frames;
    result.height = x.height;
    result.width = x.width;
    // Widening on the way OUT is lossless: every element already carries only
    // the bits `dtype` holds. It is the container changing, not the value.
    const size_t out_n = static_cast<size_t>(x.elems());
    for (size_t i = 0; i < out_n; ++i) result.data.push_back(x.Load(i));
    result.dtype = dtype;
  }
  return result;
}

Ltx2LatentVolume Ltx2UpsampleVideoLatent(const Ltx2UpsamplerConfig& config,
                                         const Ltx2VaeWeights& weights,
                                         const Ltx2LatentVolume& latent,
                                         const std::vector<float>& std_of_means,
                                         const std::vector<float>& mean_of_means) {
  // model.py:140-142, with PerChannelStatistics (video_vae/ops.py:76-84):
  //   un_normalize(x) = x * std + mean
  //   normalize(x)    = (x - mean) / std
  Require(std_of_means.size() == static_cast<size_t>(latent.channels) &&
              mean_of_means.size() == static_cast<size_t>(latent.channels),
          "ltx2 upsample_video: per-channel statistics must carry one value per latent channel");
  RequireUpsamplerDType(weights.dtype);
  const vt::DType dtype = weights.dtype;

  // R7, MEASURED, and it has TWO halves that a single hypothesis would miss.
  //
  //   (a) BOTH statistics NARROW. `un_normalize` writes
  //       `self.get_buffer("std-of-means").view(...).to(x)` (ops.py:77-79), and
  //       `x` is the bf16 latent, so each is rounded to the model dtype before
  //       it multiplies anything. Against the un-narrowed values this separates
  //       38 of 144 elements on `un_normalize` and 31 on `normalize`.
  //   (b) There are TWO roundings, not one. `x * std` is a tensor operation that
  //       lands in bf16 and `+ mean` is a second one; fusing them separates 40
  //       and 16 elements respectively. This is the site where a C++ `a * b + c`
  //       written as one expression -- which a compiler may contract to an FMA --
  //       is a DIFFERENT function from upstream's, so the intermediate is
  //       rounded explicitly rather than left to the expression.
  const auto narrow = [&](float v) {
    return dtype == vt::DType::kBF16 ? vt::BF16ToF32(vt::F32ToBF16(v)) : v;
  };
  const auto round_store = [&](float v) { return narrow(v); };

  Ltx2LatentVolume denormalized = latent;
  const int64_t spatial = latent.frames * latent.height * latent.width;
  for (int64_t b = 0; b < latent.batch; ++b) {
    for (int64_t c = 0; c < latent.channels; ++c) {
      const float std_value = narrow(std_of_means[static_cast<size_t>(c)]);    // R7(a)
      const float mean_value = narrow(mean_of_means[static_cast<size_t>(c)]);  // R7(a)
      for (int64_t i = 0; i < spatial; ++i) {
        float& value =
            denormalized.data[static_cast<size_t>((b * latent.channels + c) * spatial + i)];
        const float scaled = round_store(narrow(value) * std_value);  // R7(b), first rounding
        value = round_store(scaled + mean_value);                     // R7(b), second
      }
    }
  }

  Ltx2LatentVolume upsampled = Ltx2LatentUpsample(config, weights, denormalized);
  const int64_t out_spatial = upsampled.frames * upsampled.height * upsampled.width;
  for (int64_t b = 0; b < upsampled.batch; ++b) {
    for (int64_t c = 0; c < upsampled.channels; ++c) {
      const float std_value = narrow(std_of_means[static_cast<size_t>(c)]);
      const float mean_value = narrow(mean_of_means[static_cast<size_t>(c)]);
      for (int64_t i = 0; i < out_spatial; ++i) {
        float& value = upsampled.data[static_cast<size_t>(
            (b * upsampled.channels + c) * out_spatial + i)];
        const float centered = round_store(value - mean_value);  // R7(b), first rounding
        value = round_store(centered / std_value);               // R7(b), second
      }
    }
  }
  return upsampled;
}

}  // namespace vllm
