// Byte-exactness + determinism gate for `vt::Conv3d` (LTX25-DEVICE-RESIDENCY
// W5, #1007, .agents/specs/ltx25-device-residency.md `## W5`; kernel
// src/vt/cpu/cpu_conv3d.cpp).
//
// The op mirrors torch `nn.Conv3d` as `CausalConv3d` instantiates it —
// Lightricks/LTX-2 @ fd4ded7f2,
// packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:292-302, called
// once at :312 — which is the whole of the LTX-2.5 video VAE's arithmetic.
//
// GATE, matching tests/vt/test_ops_conv2d.cpp discipline: `memcmp` against an
// INDEPENDENT in-test scalar reference written from the convolution definition,
// NOT against the kernel. That is achievable exactly because the accumulation
// order is published contract — one f32 accumulator per output element SEEDED
// WITH THE BIAS, then one f32 PARTIAL PER INPUT CHANNEL swept (kt, kh, kw) — and
// the reference reproduces that nesting.
//
// THE ORDER IS THE POINT, AND IT IS NOT kConv2d's. `RefConv3dFlat` below is the
// same convolution with kConv2d's single flat accumulator, and the case
// `conv3d: the per-input-channel partial is NOT the flat accumulator` asserts
// the two DISAGREE on a shape where the difference is representable.
//
// THAT CASE IS THE ONLY THING IN THIS TREE THAT HOLDS THE ORDER, AND THAT IS
// MEASURED RATHER THAN ASSUMED. Mutating the kernel to the flat accumulator
// (bias last) reds 2 of this file's 4 cases and 140 of its 2035 assertions —
// and leaves `test_ltx2_vae` at 44/44 GREEN. The record in
// src/vllm/model_executor/models/ltx2_video_vae.cpp that a naive sum pushed the
// non-causal tiled golden to 5.00679e-06 against a 5e-06 tolerance is about a
// mutation that ALSO narrowed the accumulator width; the order ALONE does not
// move that golden at its fixture scale. So a reader must not take the video
// goldens as the instrument for this contract. They are the instrument for the
// ROUTING — deleting the dispatch reds 12 of their 44 cases — and this file is
// the instrument for the arithmetic.
//
// Coverage: every x/weight/out dtype combination over f32/f16/bf16; dense,
// grouped, depthwise, pointwise (kernel 1), dilated, strided and padded forms;
// ragged extents that make the window fall off every edge; the LTX shapes
// themselves (k3, pad 0, because the model materialises its own pad); and thread
// counts 1/2/4/8.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting (via -I src)
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Conv3dArgs;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

// Deterministic LCG so every case is reproducible without a seed corpus.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float Uniform() { return static_cast<float>(Next() % 20001) / 10000.0f - 1.0f; }
};

// Storage for one operand plus the EXACT f32 value of every element after
// rounding into the storage dtype (what the kernel will read back).
struct Buf {
  DType dtype;
  std::vector<uint8_t> bytes;
  std::vector<float> ref;

  Buf(DType dt, int64_t n, Rng& rng) : dtype(dt) {
    ref.resize(static_cast<size_t>(n));
    bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
    for (int64_t i = 0; i < n; ++i) {
      const float v = rng.Uniform();
      switch (dt) {
        case DType::kF32:
          reinterpret_cast<float*>(bytes.data())[i] = v;
          ref[i] = v;
          break;
        case DType::kF16: {
          const uint16_t h = vt::F32ToF16(v);
          reinterpret_cast<uint16_t*>(bytes.data())[i] = h;
          ref[i] = vt::F16ToF32(h);
          break;
        }
        case DType::kBF16: {
          const uint16_t h = vt::F32ToBF16(v);
          reinterpret_cast<uint16_t*>(bytes.data())[i] = h;
          ref[i] = vt::BF16ToF32(h);
          break;
        }
        default:
          break;
      }
    }
  }
  void* Data() { return bytes.data(); }
};

void StoreAt(std::vector<uint8_t>* out, int64_t i, DType dt, float v) {
  switch (dt) {
    case DType::kF32: reinterpret_cast<float*>(out->data())[i] = v; break;
    case DType::kF16: reinterpret_cast<uint16_t*>(out->data())[i] = vt::F32ToF16(v); break;
    case DType::kBF16: reinterpret_cast<uint16_t*>(out->data())[i] = vt::F32ToBF16(v); break;
    default: break;
  }
}

struct Case {
  const char* name;
  int64_t cin, tin, hin, win;
  int64_t cout, kt, kh, kw;
  int64_t st, sh, sw, pt, ph, pw, dt_, dh, dw, groups;
  bool bias;
};

int64_t OutLen(int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride) {
  return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

// INDEPENDENT scalar reference. Written from the convolution definition:
// out[oc,ot,oh,ow] = bias[oc] + Σ_ic ( Σ_{kt,kh,kw} x * w ), taps outside the
// input skipped (zero padding), the bias SEEDED into the accumulator and the
// inner sum kept in its own f32 partial.
void RefConv3d(const Case& c, const std::vector<float>& x, const std::vector<float>& w,
               const std::vector<float>* bias, DType odt, std::vector<uint8_t>* out) {
  const int64_t tout = OutLen(c.tin, c.pt, c.dt_, c.kt, c.st);
  const int64_t hout = OutLen(c.hin, c.ph, c.dh, c.kh, c.sh);
  const int64_t wout = OutLen(c.win, c.pw, c.dw, c.kw, c.sw);
  const int64_t cin_g = c.cin / c.groups;
  const int64_t cout_g = c.cout / c.groups;
  out->assign(static_cast<size_t>(c.cout * tout * hout * wout) * vt::SizeOf(odt), 0);
  for (int64_t oc = 0; oc < c.cout; ++oc) {
    const int64_t gc0 = (oc / cout_g) * cin_g;
    for (int64_t ot = 0; ot < tout; ++ot) {
      for (int64_t oh = 0; oh < hout; ++oh) {
        for (int64_t ow = 0; ow < wout; ++ow) {
          float acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0f;
          for (int64_t ic = 0; ic < cin_g; ++ic) {
            float tap = 0.0f;
            for (int64_t kt = 0; kt < c.kt; ++kt) {
              const int64_t it = ot * c.st - c.pt + kt * c.dt_;
              if (it < 0 || it >= c.tin) continue;
              for (int64_t kh = 0; kh < c.kh; ++kh) {
                const int64_t ih = oh * c.sh - c.ph + kh * c.dh;
                if (ih < 0 || ih >= c.hin) continue;
                for (int64_t kw = 0; kw < c.kw; ++kw) {
                  const int64_t iw = ow * c.sw - c.pw + kw * c.dw;
                  if (iw < 0 || iw >= c.win) continue;
                  const size_t xi = static_cast<size_t>(
                      (((gc0 + ic) * c.tin + it) * c.hin + ih) * c.win + iw);
                  const size_t wi = static_cast<size_t>(
                      (((oc * cin_g + ic) * c.kt + kt) * c.kh + kh) * c.kw + kw);
                  tap += x[xi] * w[wi];
                }
              }
            }
            acc += tap;
          }
          StoreAt(out, ((oc * tout + ot) * hout + oh) * wout + ow, odt, acc);
        }
      }
    }
  }
}

// The SAME convolution with kConv2d's accumulator shape: one flat f32 sum over
// (ic, kt, kh, kw) with the bias added LAST. Present only so the difference
// between the two orders is executable rather than asserted in prose.
void RefConv3dFlat(const Case& c, const std::vector<float>& x, const std::vector<float>& w,
                   const std::vector<float>* bias, DType odt, std::vector<uint8_t>* out) {
  const int64_t tout = OutLen(c.tin, c.pt, c.dt_, c.kt, c.st);
  const int64_t hout = OutLen(c.hin, c.ph, c.dh, c.kh, c.sh);
  const int64_t wout = OutLen(c.win, c.pw, c.dw, c.kw, c.sw);
  const int64_t cin_g = c.cin / c.groups;
  const int64_t cout_g = c.cout / c.groups;
  out->assign(static_cast<size_t>(c.cout * tout * hout * wout) * vt::SizeOf(odt), 0);
  for (int64_t oc = 0; oc < c.cout; ++oc) {
    const int64_t gc0 = (oc / cout_g) * cin_g;
    for (int64_t ot = 0; ot < tout; ++ot) {
      for (int64_t oh = 0; oh < hout; ++oh) {
        for (int64_t ow = 0; ow < wout; ++ow) {
          float acc = 0.0f;
          for (int64_t ic = 0; ic < cin_g; ++ic) {
            for (int64_t kt = 0; kt < c.kt; ++kt) {
              const int64_t it = ot * c.st - c.pt + kt * c.dt_;
              if (it < 0 || it >= c.tin) continue;
              for (int64_t kh = 0; kh < c.kh; ++kh) {
                const int64_t ih = oh * c.sh - c.ph + kh * c.dh;
                if (ih < 0 || ih >= c.hin) continue;
                for (int64_t kw = 0; kw < c.kw; ++kw) {
                  const int64_t iw = ow * c.sw - c.pw + kw * c.dw;
                  if (iw < 0 || iw >= c.win) continue;
                  const size_t xi = static_cast<size_t>(
                      (((gc0 + ic) * c.tin + it) * c.hin + ih) * c.win + iw);
                  const size_t wi = static_cast<size_t>(
                      (((oc * cin_g + ic) * c.kt + kt) * c.kh + kh) * c.kw + kw);
                  acc += x[xi] * w[wi];
                }
              }
            }
          }
          if (bias != nullptr) acc += (*bias)[static_cast<size_t>(oc)];
          StoreAt(out, ((oc * tout + ot) * hout + oh) * wout + ow, odt, acc);
        }
      }
    }
  }
}

// Shapes: the LTX-2.5 decoder's own form first (cubic k3, pad 0 — the model
// materialises its temporal and spatial pad itself, convolution.py:305-311),
// then the generalizations torch admits.
const std::vector<Case>& Cases() {
  static const std::vector<Case> c = {
      // The LTX conv as `CausalConv3d` issues it: k3 cubic, unit stride, NO
      // padding, bias present (convolution.py:292-302 with padding materialised).
      {"ltx k3 pad0", 4, 5, 6, 7, 3, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, true},
      {"ltx k3 pad0 nobias", 3, 4, 5, 5, 2, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, false},
      // `make_linear_nd` for dims == 3 is a 1x1x1 Conv3d (convolution.py:84-85).
      {"pointwise 1x1x1", 5, 3, 4, 4, 7, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, true},
      // The video ENCODER's strided form (a stride is applied after the pad).
      {"strided t2", 3, 7, 6, 6, 4, 3, 3, 3, 2, 1, 1, 0, 0, 0, 1, 1, 1, 1, true},
      {"strided hw2", 2, 4, 9, 9, 3, 3, 3, 3, 1, 2, 2, 0, 0, 0, 1, 1, 1, 1, false},
      // generalizations torch admits and the op contract publishes
      {"zero padded", 3, 5, 5, 5, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, true},
      {"grouped g2", 4, 4, 5, 5, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, true},
      {"depthwise", 4, 5, 5, 6, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, true},
      {"dilated d2", 2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, true},
      {"anisotropic kernel", 2, 6, 5, 7, 3, 1, 3, 5, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, false},
      // degenerate / ragged extents: single frame, single row, single column, and
      // an extent that is not a multiple of the stride so the last output cell
      // reads a truncated window.
      {"single frame", 2, 1, 6, 6, 3, 1, 3, 3, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, true},
      {"single row", 2, 5, 1, 8, 3, 3, 1, 3, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, true},
      {"single col", 2, 5, 8, 1, 3, 3, 3, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, false},
      {"ragged stride", 1, 9, 11, 13, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, true},
      // every knob on at once
      {"kitchen sink", 6, 7, 9, 10, 9, 2, 3, 2, 2, 2, 3, 1, 1, 1, 2, 1, 1, 3, true},
  };
  return c;
}

void RunCase(const Case& c, DType xdt, DType wdt, DType odt, uint64_t seed) {
  Queue q = CpuQueue();
  const int64_t tout = OutLen(c.tin, c.pt, c.dt_, c.kt, c.st);
  const int64_t hout = OutLen(c.hin, c.ph, c.dh, c.kh, c.sh);
  const int64_t wout = OutLen(c.win, c.pw, c.dw, c.kw, c.sw);
  REQUIRE(tout > 0);
  REQUIRE(hout > 0);
  REQUIRE(wout > 0);
  Rng rng(seed);
  Buf x(xdt, c.cin * c.tin * c.hin * c.win, rng);
  Buf w(wdt, c.cout * (c.cin / c.groups) * c.kt * c.kh * c.kw, rng);
  Buf b(wdt, c.cout, rng);

  std::vector<uint8_t> got(static_cast<size_t>(c.cout * tout * hout * wout) * vt::SizeOf(odt),
                           0xAB);
  Tensor tx = Tensor::Contiguous(x.Data(), xdt, Cpu(), {c.cin, c.tin, c.hin, c.win});
  Tensor tw = Tensor::Contiguous(w.Data(), wdt, Cpu(),
                                 {c.cout * (c.cin / c.groups), c.kt, c.kh, c.kw});
  Tensor tb = Tensor::Contiguous(b.Data(), wdt, Cpu(), {c.cout});
  Tensor to = Tensor::Contiguous(got.data(), odt, Cpu(), {c.cout, tout, hout, wout});
  Conv3dArgs args;
  args.stride_t = c.st;
  args.stride_h = c.sh;
  args.stride_w = c.sw;
  args.pad_t = c.pt;
  args.pad_h = c.ph;
  args.pad_w = c.pw;
  args.dilation_t = c.dt_;
  args.dilation_h = c.dh;
  args.dilation_w = c.dw;
  args.groups = c.groups;
  vt::Conv3d(q, to, tx, tw, c.bias ? &tb : nullptr, args);

  std::vector<uint8_t> want;
  RefConv3d(c, x.ref, w.ref, c.bias ? &b.ref : nullptr, odt, &want);
  REQUIRE(want.size() == got.size());
  CHECK_MESSAGE(std::memcmp(got.data(), want.data(), want.size()) == 0, c.name);
}

}  // namespace

TEST_CASE("conv3d: byte-identical to the scalar reference over every dtype x shape") {
  const DType kDt[3] = {DType::kF32, DType::kF16, DType::kBF16};
  uint64_t seed = 1;
  for (DType xdt : kDt) {
    for (DType wdt : kDt) {
      for (DType odt : kDt) {
        for (const Case& c : Cases()) RunCase(c, xdt, wdt, odt, seed++);
      }
    }
  }
}

TEST_CASE("conv3d: the per-input-channel partial is NOT the flat accumulator") {
  // The op's published order sums one f32 PARTIAL PER INPUT CHANNEL; kConv2d
  // keeps a single flat accumulator and adds the bias last. The two are
  // different numbers. This case is what makes that difference executable: it
  // asserts the kernel matches the BLOCKED reference and does NOT match the flat
  // one, on a shape whose channel count is large enough for the reassociation to
  // be representable in f32. See the file header for why the LTX-2.5 video
  // goldens are NOT a substitute — measured, they do not move under this
  // mutation.
  Queue q = CpuQueue();
  const Case c = {"order", 64, 3, 4, 5, 4, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, true};
  const int64_t tout = OutLen(c.tin, c.pt, c.dt_, c.kt, c.st);
  const int64_t hout = OutLen(c.hin, c.ph, c.dh, c.kh, c.sh);
  const int64_t wout = OutLen(c.win, c.pw, c.dw, c.kw, c.sw);
  Rng rng(4242);
  Buf x(DType::kF32, c.cin * c.tin * c.hin * c.win, rng);
  Buf w(DType::kF32, c.cout * c.cin * c.kt * c.kh * c.kw, rng);
  Buf b(DType::kF32, c.cout, rng);
  Conv3dArgs args;  // every knob at its default: k3, unit stride, no padding
  std::vector<uint8_t> got(static_cast<size_t>(c.cout * tout * hout * wout) * sizeof(float), 0xAB);
  Tensor tx = Tensor::Contiguous(x.Data(), DType::kF32, Cpu(), {c.cin, c.tin, c.hin, c.win});
  Tensor tw =
      Tensor::Contiguous(w.Data(), DType::kF32, Cpu(), {c.cout * c.cin, c.kt, c.kh, c.kw});
  Tensor tb = Tensor::Contiguous(b.Data(), DType::kF32, Cpu(), {c.cout});
  Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {c.cout, tout, hout, wout});
  vt::Conv3d(q, to, tx, tw, &tb, args);

  std::vector<uint8_t> blocked;
  RefConv3d(c, x.ref, w.ref, &b.ref, DType::kF32, &blocked);
  std::vector<uint8_t> flat;
  RefConv3dFlat(c, x.ref, w.ref, &b.ref, DType::kF32, &flat);
  CHECK(std::memcmp(got.data(), blocked.data(), blocked.size()) == 0);
  CHECK(std::memcmp(blocked.data(), flat.data(), flat.size()) != 0);
}

TEST_CASE("conv3d: byte-identical across thread counts") {
  // The determinism contract (src/vt/cpu/cpu_conv3d.cpp): parallelism partitions
  // output LINES only, so the result must not depend on the worker count. The
  // shape is chosen so the (cout, tout, hout) row count does not divide evenly
  // by 8 — a partition-dependent kernel shows up here.
  Queue q = CpuQueue();
  const Case c = {"threads", 6, 7, 9, 10, 9, 2, 3, 2, 2, 2, 3, 1, 1, 1, 2, 1, 1, 3, true};
  const int64_t tout = OutLen(c.tin, c.pt, c.dt_, c.kt, c.st);
  const int64_t hout = OutLen(c.hin, c.ph, c.dh, c.kh, c.sh);
  const int64_t wout = OutLen(c.win, c.pw, c.dw, c.kw, c.sw);
  REQUIRE((c.cout * tout * hout) % 8 != 0);
  Rng rng(9001);
  Buf x(DType::kBF16, c.cin * c.tin * c.hin * c.win, rng);
  Buf w(DType::kBF16, c.cout * (c.cin / c.groups) * c.kt * c.kh * c.kw, rng);
  Buf b(DType::kBF16, c.cout, rng);
  Conv3dArgs args;
  args.stride_t = c.st;
  args.stride_h = c.sh;
  args.stride_w = c.sw;
  args.pad_t = c.pt;
  args.pad_h = c.ph;
  args.pad_w = c.pw;
  args.dilation_t = c.dt_;
  args.dilation_h = c.dh;
  args.dilation_w = c.dw;
  args.groups = c.groups;
  std::vector<uint8_t> base;
  for (int nth : {1, 2, 4, 8}) {
    vt::cpu::Threadpool tp(nth);
    vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
    std::vector<uint8_t> got(static_cast<size_t>(c.cout * tout * hout * wout) * sizeof(float),
                             0xAB);
    Tensor tx = Tensor::Contiguous(x.Data(), DType::kBF16, Cpu(), {c.cin, c.tin, c.hin, c.win});
    Tensor tw = Tensor::Contiguous(w.Data(), DType::kBF16, Cpu(),
                                   {c.cout * (c.cin / c.groups), c.kt, c.kh, c.kw});
    Tensor tb = Tensor::Contiguous(b.Data(), DType::kBF16, Cpu(), {c.cout});
    Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {c.cout, tout, hout, wout});
    vt::Conv3d(q, to, tx, tw, &tb, args);
    vt::cpu::Threadpool::SwapForTesting(prev);
    if (base.empty()) {
      base = got;
    } else {
      CHECK(std::memcmp(base.data(), got.data(), base.size()) == 0);
    }
  }
}

TEST_CASE("conv3d: the shape contract refuses by name") {
  Queue q = CpuQueue();
  std::vector<float> xs(2 * 3 * 4 * 4, 0.5f);
  std::vector<float> ws(2 * 2 * 3 * 3 * 3, 0.25f);
  std::vector<float> os(2 * 1 * 2 * 2, 0.0f);
  Tensor tx = Tensor::Contiguous(xs.data(), DType::kF32, Cpu(), {2, 3, 4, 4});
  Tensor to = Tensor::Contiguous(os.data(), DType::kF32, Cpu(), {2, 1, 2, 2});
  Conv3dArgs args;

  SUBCASE("a weight whose merged leading axis is wrong") {
    // [Cout*Cin/groups, KT, KH, KW] is 4 here, not 3: a rank-5 caller that
    // forgot to merge would land exactly on this.
    Tensor tw = Tensor::Contiguous(ws.data(), DType::kF32, Cpu(), {3, 3, 3, 3});
    CHECK_THROWS(vt::Conv3d(q, to, tx, tw, nullptr, args));
  }
  SUBCASE("groups that do not divide the channels") {
    Tensor tw = Tensor::Contiguous(ws.data(), DType::kF32, Cpu(), {4, 3, 3, 3});
    Conv3dArgs g = args;
    g.groups = 3;  // 2 % 3 != 0
    CHECK_THROWS(vt::Conv3d(q, to, tx, tw, nullptr, g));
  }
  SUBCASE("an out extent that does not match the stride arithmetic") {
    Tensor tw = Tensor::Contiguous(ws.data(), DType::kF32, Cpu(), {4, 3, 3, 3});
    std::vector<float> bad(2 * 1 * 3 * 3, 0.0f);
    Tensor tbad = Tensor::Contiguous(bad.data(), DType::kF32, Cpu(), {2, 1, 3, 3});
    CHECK_THROWS(vt::Conv3d(q, tbad, tx, tw, nullptr, args));
  }
  SUBCASE("a kernel larger than the padded input") {
    std::vector<float> big(2 * 2 * 5 * 5 * 5, 0.25f);
    Tensor tw = Tensor::Contiguous(big.data(), DType::kF32, Cpu(), {4, 5, 5, 5});
    CHECK_THROWS(vt::Conv3d(q, to, tx, tw, nullptr, args));
  }
  SUBCASE("a kernel larger than the padded input, WITH a stride over 1") {
    // #1007 fresh review F7. The extent formula's numerator is NEGATIVE here —
    // `tin + 2*pad - dilation*(kt-1) - 1` = `2 + 0 - 2 - 1` = -1 — and C++
    // integer division TRUNCATES TOWARD ZERO while torch's shape contract
    // FLOORS. Truncation gives `-1 / 2 + 1` = 1 and accepts a `Tout` of 1,
    // convolving over taps the stride skipped; torch gives `floor(-1/2) + 1` = 0
    // and raises "Output size is too small".
    //
    // The header at vt::Conv3d claims to mirror torch's shape contract, and
    // `LTX25-DEVICE-RESIDENCY` §W5.4 offers this op to other models as a shared
    // seam, so the divergence had to close rather than be documented.
    // UNREACHABLE FROM LTX — `CausalConv3d` materialises a pad of at least the
    // kernel on every axis — which is why no model gate here can hold it and
    // why this case exists.
    //
    // Stride 1 keeps the sibling SUBCASE above red for its own reason: at
    // stride 1 truncation and floor AGREE (`-1 / 1` is -1 either way), so that
    // case cannot detect this defect and this one cannot replace it.
    // Tin must be 2, not the enclosing fixture's 3: at Tin = 3 the span is
    // `3 + 0 - 2 - 1` = 0, which is NON-NEGATIVE, and floor and truncation agree
    // on it. The case is only discriminating where the span is negative.
    std::vector<float> xs2(2 * 2 * 4 * 4, 0.5f);
    Tensor tx2 = Tensor::Contiguous(xs2.data(), DType::kF32, Cpu(), {2, 2, 4, 4});
    Conv3dArgs strided = args;
    strided.stride_t = 2;
    Tensor tw = Tensor::Contiguous(ws.data(), DType::kF32, Cpu(), {4, 3, 3, 3});
    CHECK_THROWS(vt::Conv3d(q, to, tx2, tw, nullptr, strided));
  }
}
