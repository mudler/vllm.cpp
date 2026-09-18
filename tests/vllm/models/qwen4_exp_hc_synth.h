// The `vt::Qwen4ExpGatedResidual` grouped-norm SYNTHETIC FIXTURE, shared by the
// CUDA and the ROCm device gates.
// Row MODEL-MM-QWEN4-EXP, spec `.agents/specs/qwen4-exp-rocm-hc-norm.md`.
//
// ─── WHY THIS IS A HEADER AND NOT A SECOND COPY ──────────────────────────────
// `tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp` defined all of this
// when it was the only device gate for the kernel. The ROCm gate needs the
// IDENTICAL fixture: the same generator, the same per-group scale pattern and
// above all the same DERIVED bound, because two arms of one kernel held to two
// separately-maintained copies of `W7NormRel` would drift and the first repair
// to either copy would create the contradiction. The CUDA file now includes this
// header and names what it uses; it did not gain a copy.
//
// ─── WHAT THE FIXTURE IS FOR ─────────────────────────────────────────────────
// The committed goldens are six elements wide and the ROCm cross-device case is
// seven. Neither can express a REDUCTION SHAPE: at that width one thread does
// the whole group either way, so a kernel that dropped its cross-lane fold, its
// grid stride or its last strided trip passes them all. These shapes reach the
// model's own `hidden = 2560`, past the 4096-block grid cap, and across widths
// that are a multiple of no wavefront size.
//
// THE PROJECTIONS ARE ZEROED ON PURPOSE. `mix_down` and `mix_up` are all-zero,
// which makes the mixer a GEMM-free function of the normed stream. That is what
// lets the bound be a pure reduction-width bound derived from `H`: with a live
// projection the dominant term would be `vt::MatmulBT`'s K-reassociation, which
// is a different phenomenon and is already gated by the oracle cases at 1e-5.
#pragma once

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace q4hc {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Qwen4ExpGatedResidualArgs;
using vt::Queue;
using vt::Tensor;

inline constexpr double kAbsFloor = 1e-30;

// ─── THE BOUND, DERIVED ──────────────────────────────────────────────────────
// The CPU arm walks the group ascending in `double`; a device arm takes
// `ceil(H / 256)` serial adds inside one thread, then a cross-lane tree, then a
// second tree across the per-wave partials, in f32. Call the tree depth 16,
// which is more than three times the 10 it actually is at a 256-thread block on
// either wavefront size. The standard `n u` bound with `u = 2^-24` gives a
// relative error on the sum of squares, `r = 1/sqrt(.)` halves it, and a factor
// of 4 is carried on top as stated margin:
//
//   rel(H) = 4 * 0.5 * (ceil(H/256) + 16) * u
//
// At H = 2560 that is 3.1e-6. The defects it has to separate — a broadcast `r`,
// a dropped grid stride, a missing cross-wave stage — are all O(1) RELATIVE, so
// the discrimination band is six orders of magnitude wide and is MEASURED on
// every run by the printed ratio. NEVER WIDEN IT TO MAKE A CASE PASS.
inline constexpr double kW7UnitRoundoff = 1.0 / 16777216.0;  // u = 2^-24
inline double W7NormRel(int64_t H) {
  const double depth = static_cast<double>((H + 255) / 256) + 16.0;
  return 4.0 * 0.5 * depth * kW7UnitRoundoff;
}

// max|a-b| with `std::max`'s NaN blindness removed, plus the count of elements
// that are not BITWISE equal. Both are reported, because "within one ulp" and
// "identical" are different findings.
struct Agreement {
  double worst = 0.0;
  size_t not_bitwise_equal = 0;
};

inline Agreement Compare(const std::vector<float>& got, const std::vector<float>& want) {
  REQUIRE(got.size() == want.size());
  Agreement a;
  // MaxAbsDiff returns +infinity on ANY non-finite operand and raises its own
  // doctest failure, so an all-NaN device output cannot read as a perfect match
  // here. That defect is issue #449 and it must not be re-introduced by
  // hand-rolling a reduction.
  a.worst = vllm_test::MaxAbsDiff(got, want);
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) ++a.not_bitwise_equal;
  }
  return a;
}

// max|v| over ONE array, with the SAME NaN polarity `Compare` has: a non-finite
// element gives +infinity, never 0.0. A scale is not a diff, so it is easy to
// think the #449 rule does not reach it -- it does, and in the worse direction.
// A `std::max(worst, std::fabs(v))` scale reduces an all-NaN array to 0.0, and
// a bound DERIVED from that scale then SHRINKS, so the hand-rolled spelling
// turns a poisoned run into a stricter-looking green rather than a red. Routed
// through the same hardened scan so there is one spelling of the reduction in
// this row, not two.
inline double MaxAbsFinite(const std::vector<float>& v) {
  const std::vector<float> zeros(v.size(), 0.0f);
  return vllm_test::MaxAbsDiff(v, zeros);
}

// BOTH REPORTERS PRINT ON SUCCESS, not only on failure. doctest's INFO/CAPTURE
// are emitted only when an assertion fails, so a passing run of a numeric gate
// says nothing about HOW closely it passed. These numbers are the wave's
// evidence and have to survive a green run.
inline void CheckBitwise(const std::vector<float>& got, const std::vector<float>& want,
                         const char* what) {
  const Agreement a = Compare(got, want);
  std::printf("[MEASURED] %-48s BYTE gate: %zu/%zu differ, max|diff| = %.9g\n", what,
              a.not_bitwise_equal, want.size(), a.worst);
  INFO(what << ": " << a.not_bitwise_equal << " of " << want.size()
            << " elements differ; max|diff| = " << a.worst);
  CHECK(a.not_bitwise_equal == 0);
}

inline void CheckWithin(const std::vector<float>& got, const std::vector<float>& want,
                        double rel, const char* what) {
  const Agreement a = Compare(got, want);
  double scale = 0.0;
  for (float v : want) scale = std::max(scale, static_cast<double>(std::fabs(v)));
  const double bound = kAbsFloor + rel * scale;
  std::printf("[MEASURED] %-48s max|diff| = %.9g  bound = %.9g  not-bitwise-equal = %zu/%zu\n",
              what, a.worst, bound, a.not_bitwise_equal, want.size());
  INFO(what << ": max|diff| = " << a.worst << " vs bound " << bound << "; "
            << a.not_bitwise_equal << " of " << want.size()
            << " elements not bitwise equal (0 means byte-identical)");
  CHECK(a.worst <= bound);
}

// ─── THE FIXTURE ─────────────────────────────────────────────────────────────

struct SynthHc {
  std::string name;
  int64_t hidden = 0, hc = 0, lowrank = 0, T = 0;
  float eps = 1e-6f;
  std::vector<float> hyper, w_hf, down, up;
};

// A fixed 64-bit LCG, the one `test_qwen4_exp_hc_device.cpp`'s model-width case
// uses, so the case is reproducible without depending on any standard-library
// distribution's implementation.
inline SynthHc MakeSynthHc(const char* name, int64_t H, int64_t hc, int64_t R, int64_t T,
                           uint64_t seed) {
  SynthHc y;
  y.name = name;
  y.hidden = H;
  y.hc = hc;
  y.lowrank = R;
  y.T = T;
  const int64_t flat = hc * H;
  uint64_t state = seed;
  const auto next = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<int32_t>(state >> 33)) / 2147483648.0f;
  };
  // PER-GROUP SCALES, period 7. Seven is coprime with `hc` (3 and 4 here) and
  // with the 4096-block grid cap, so the pattern does not align with either —
  // a kernel that computed group `(t, 0)` and broadcast it, that walked the
  // group stride as `j` instead of `j * H`, or that lost a token to a dropped
  // grid stride, lands on a DIFFERENT scale and cannot hide inside any
  // reduction-width bound. The spread is 8x at most, which keeps every
  // intermediate far from an f32 edge.
  y.hyper.resize(static_cast<size_t>(T * flat));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < hc; ++j) {
      const int k = static_cast<int>((t * hc + j) % 7) - 3;
      const float scale = std::ldexp(1.0f, k);
      for (int64_t h = 0; h < H; ++h) {
        y.hyper[static_cast<size_t>((t * hc + j) * H + h)] = scale * next();
      }
    }
  }
  // The RAW HuggingFace gamma (#2218): centred on ZERO, because the op adds the
  // 1 itself. Drawn per element so no two groups share a gamma row.
  y.w_hf.resize(static_cast<size_t>(flat));
  for (float& v : y.w_hf) v = 0.1f * next();
  // ZERO. See the isolation note in the file header — this is the whole reason
  // the bound is a reduction-width bound.
  y.down.assign(static_cast<size_t>(R * flat), 0.0f);
  y.up.assign(static_cast<size_t>(flat * R), 0.0f);
  return y;
}

// ─── THE MAGNITUDE-SEPARATED FIXTURE ─────────────────────────────────────────
// A TRANSCRIPTION of `test_qwen4_exp_hc_device.cpp`'s "the grouped norm needs a
// WIDER-THAN-f32 accumulator" case (`:504`), which exists to make an accumulator
// width VISIBLE: every element is 1.0f except one dominant element per group,
// and the dominant values are four orders of magnitude apart across the four
// streams. That file measured a `double` accumulator at 1.173e-06 and a SERIAL
// `float ss` at 6.702e-04 against its 1e-5 bar — a 571x separation.
//
// IT RUNS THE CPU ARM ONLY THERE, and that is correct: the CPU arm is the host
// REFERENCE and the reference is the thing that must be `double`. The device
// arms are fp32 by contract (`qwen4_exp_hc.h:99-104`: upstream runs the norm in
// fp32 and vLLM likewise), so this fixture cannot red them into being `double`
// and is not here to. It is here because a WIDTH change deserves a case that can
// see a width, and the serial 571x is NOT the number a block tree pays: the
// tree is ~10 deep at H = 2560 rather than 2560 deep. How much of the 571x the
// tree gives back is a measurement, printed by the case that calls this.
inline SynthHc MakeMagnitudeSeparatedHc(const char* name, int64_t H, int64_t hc, int64_t R,
                                        int64_t T, const std::vector<float>& dominant) {
  REQUIRE(static_cast<int64_t>(dominant.size()) == hc);
  SynthHc y;
  y.name = name;
  y.hidden = H;
  y.hc = hc;
  y.lowrank = R;
  y.T = T;
  const int64_t flat = hc * H;
  y.hyper.assign(static_cast<size_t>(T * flat), 1.0f);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < hc; ++j) {
      // The dominant element sits at a DIFFERENT offset in each group, so a
      // kernel whose strided walk skipped one fixed lane cannot miss them all
      // the same way, and the offset is inside the first strided trip for every
      // width this is used at.
      const int64_t at = (j * 37) % H;
      y.hyper[static_cast<size_t>((t * hc + j) * H + at)] = dominant[static_cast<size_t>(j)];
    }
  }
  // A ZERO gamma, so the op's own `1 +` fold makes the multiplier exactly 1.0
  // and the only thing this case measures is the reduction.
  y.w_hf.assign(static_cast<size_t>(flat), 0.0f);
  y.down.assign(static_cast<size_t>(R * flat), 0.0f);
  y.up.assign(static_cast<size_t>(flat * R), 0.0f);
  return y;
}

struct MixerResult {
  std::vector<float> mixed;
  std::vector<float> injection;
  std::vector<float> hyper_after;  // the stream must come back UNCHANGED
};

namespace detail {

inline Tensor MakeTensor(void* data, DType dt, Device dev,
                         const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

// The device type is a CONSTRUCTOR ARGUMENT rather than a hard-coded `kCUDA`,
// which is the one behavioural difference between this and the copy that lived
// in the CUDA file. It is what lets one fixture drive two arms.
class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DeviceType dev, DType dt,
               const std::vector<int64_t>& shape, const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, Device{dev, 0}, shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

}  // namespace detail

// Runs the mixer on `dev`. `dev == kCPU` takes the host arm directly; any other
// value goes through that backend's registered op. The caller decides which,
// so one fixture drives the CPU reference and every device arm.
inline MixerResult RunSynthMixer(DeviceType dev, const SynthHc& c) {
  const int64_t flat = c.hc * c.hidden;
  MixerResult r;
  r.mixed.assign(static_cast<size_t>(c.T * c.hidden), 0.0f);
  r.injection.clear();
  r.hyper_after = c.hyper;
  std::vector<float> w = c.w_hf, down = c.down, up = c.up;

  Qwen4ExpGatedResidualArgs args;
  args.hc_count = c.hc;
  args.hidden_size = c.hidden;
  args.lowrank = c.lowrank;
  args.eps = c.eps;

  if (dev == DeviceType::kCPU) {
    const Device cpu{DeviceType::kCPU, 0};
    Queue q = Queue{cpu, nullptr};
    Tensor t_h = detail::MakeTensor(r.hyper_after.data(), DType::kF32, cpu, {c.T, flat});
    Tensor t_w = detail::MakeTensor(w.data(), DType::kF32, cpu, {flat});
    Tensor t_d = detail::MakeTensor(down.data(), DType::kF32, cpu, {c.lowrank, flat});
    Tensor t_u = detail::MakeTensor(up.data(), DType::kF32, cpu, {flat, c.lowrank});
    Tensor t_m = detail::MakeTensor(r.mixed.data(), DType::kF32, cpu, {c.T, c.hidden});
    vt::Qwen4ExpGatedResidual(q, t_m, nullptr, t_h, t_w, t_d, t_u, nullptr, args);
    return r;
  }
  Backend& b = vt::GetBackend(dev);
  detail::QueueGuard qg(b);
  detail::DeviceTensor d_h(b, qg.q, dev, DType::kF32, {c.T, flat}, c.hyper.data());
  detail::DeviceTensor d_w(b, qg.q, dev, DType::kF32, {flat}, w.data());
  detail::DeviceTensor d_d(b, qg.q, dev, DType::kF32, {c.lowrank, flat}, down.data());
  detail::DeviceTensor d_u(b, qg.q, dev, DType::kF32, {flat, c.lowrank}, up.data());
  detail::DeviceTensor d_m(b, qg.q, dev, DType::kF32, {c.T, c.hidden});
  vt::Qwen4ExpGatedResidual(qg.q, d_m.tensor(), nullptr, d_h.tensor(), d_w.tensor(),
                            d_d.tensor(), d_u.tensor(), nullptr, args);
  b.Synchronize(qg.q);
  d_m.Download(qg.q, r.mixed.data());
  d_h.Download(qg.q, r.hyper_after.data());
  return r;
}

}  // namespace q4hc
