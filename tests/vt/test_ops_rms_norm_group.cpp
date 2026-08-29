// `vt::RmsNormGroup` — the UNGATED per-group RMS norm, row MODEL-MM-QWEN4-EXP
// W5d-1, issue #2249 item 1, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT IS UNDER TEST. One op: `out[i] = x[i] * rsqrt(mean_{i in group}(x^2) +
// eps) * (1 + w[i])`, the `group_size is not None` arm of
// `Qwen4ExpTextRMSNorm` (transformers **v5.16.0**
// `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py:158-181`, sha256
// 77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459). That is
// this row's accepted lane pin (`.agents/oracles/transformers.md`); vLLM
// registers no `qwen4_exp` at `6a5e8f5979`, so there is no primary oracle to
// mirror instead. vLLM DOES define the op FORM this mirrors — `RmsNormArgs`,
// the f32 interior and the round-on-store — and `vt::RmsNorm` right beside it
// is where those come from.
//
// WHAT IT IS COMPARED AGAINST, AND WHY THAT IS NOT THIS FILE. The correctness
// assertions run against `qwen4_exp_hc_goldens.inc`, dumped by
// `scripts/gen-qwen4-exp-hc-goldens.py`, which lifts `Qwen4ExpTextRMSNorm`
// VERBATIM by line range out of the pinned oracle and EXECUTES it under torch:
// `normed = mod.hc_norm(hyper)` where `hc_norm` is
// `Qwen4ExpTextRMSNorm(hc*hidden, group_size=hidden, eps)`. `k*_normed` is
// therefore the oracle's own output of exactly the function this op implements,
// over the oracle's own RAW gamma `k*_norm_w_hf`. Nothing in the correctness
// path is transcribed here; a transcription cannot gate the function it
// transcribes.
//
// WHY THE FIXTURE DISCRIMINATES, MEASURED RATHER THAN ASSERTED. Six waves on
// this row have shipped a fixture that could not see the defect it was written
// for, so each of the three ways this op can be wrong is separated from the
// oracle IN THIS FILE, by a `> kTol` assertion on a locally computed wrong
// answer, before the `< kTol` assertion on the op:
//
//   * REDUCING OVER THE ROW instead of the group. Separation 4.0e-1 to 1.2e+0
//     against kTol = 1e-5, on the goldens; and a hand-built case whose two
//     groups differ by four orders of magnitude in scale carries it further, so
//     the discrimination does not rest on random draws happening to differ.
//   * DROPPING THE `+ 1` on the gamma. Separation ~2.0 on every golden case.
//     This is the #2218 defect — a gamma centred on 0 applied without the fold
//     scales the stream by ~0, which is a plausible tensor and not a crash —
//     and the goldens can see it only because their gamma is NOT near zero,
//     which this file asserts rather than hopes.
//   * DROPPING EPS. This one is scale-dependent and a previous wave got it
//     wrong: at case A's `hyper_scale = 1.7` the mean square is O(1) and an eps
//     of 1e-6 moves the answer by 4.1e-6, BELOW kTol, so an eps probe run there
//     is a mute switch. Case D exists for this — the generator says so in its
//     own comment — and at `hyper_scale = 0.01` the separation is 2.6e-2.
//
// The reference used for those separations is a local double-precision
// transcription. It is NEVER the thing the op is asserted against; it exists to
// prove the golden can tell right from wrong, which is a question about the
// GOLDEN and not about the kernel.
//
// SCOPE, HONESTLY. CPU only. No CUDA arm of this op exists, and one written on
// this host could not be gated on it. Nothing production-side calls this op yet
// — the `qwen4_exp` forward still refuses by name — so this file makes no token
// claim and no speed claim; the spec's `## Owed` records the unreached state and
// the row that owns the wiring.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm_test::MaxAbsDiff;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormGroupArgs;
using vt::Tensor;

namespace {

#include "vllm/models/qwen4_exp_hc_goldens.inc"  // NOLINT — golden literals

// The goldens are fp32 out of torch and this op's interior is fp32, so at these
// widths (24 and 15 elements per row, 5 or 6 per group) the two agree to a few
// ulps. The same value `test_qwen4_exp_hc.cpp` and `test_qwen4_exp_hc_device.cpp`
// already justify for THESE shapes.
constexpr double kTol = 1e-5;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor MakeT(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= t.shape[i];
  }
  return t;
}

// The op, driven end to end over f32 in and f32 out.
std::vector<float> RunOp(const float* x, const float* w, int64_t rows, int64_t width,
                         int64_t group_size, float eps, bool gemma) {
  std::vector<float> xs(x, x + rows * width);
  std::vector<float> ws(w, w + width);
  std::vector<float> out(static_cast<size_t>(rows * width), 0.0f);
  Tensor tx = MakeT(xs.data(), DType::kF32, {rows, width});
  Tensor tw = MakeT(ws.data(), DType::kF32, {width});
  Tensor to = MakeT(out.data(), DType::kF32, {rows, width});
  RmsNormGroupArgs args;
  args.eps = eps;
  args.gemma = gemma;
  args.group_size = group_size;
  Queue q = CpuQ();
  vt::RmsNormGroup(q, to, tx, tw, args);
  return out;
}

// The local double-precision transcription of :167-178, with one flag per
// plausible single-character defect. USED ONLY to measure how far each defect
// sits from the oracle. The op is never compared against it.
struct Variant {
  bool full_row = false;  // reduce over the whole row, ignoring group_size
  bool no_fold = false;   // multiply by `w` instead of `(1 + w)`
  bool no_eps = false;    // drop eps from inside the rsqrt
};

std::vector<double> Reference(const float* x, const float* w, int64_t rows, int64_t width,
                              int64_t group_size, double eps, Variant v) {
  const int64_t extent = v.full_row ? width : group_size;
  std::vector<double> out(static_cast<size_t>(rows * width), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t g = 0; g < width / extent; ++g) {
      const int64_t base = g * extent;
      double sumsq = 0.0;
      for (int64_t j = 0; j < extent; ++j) {
        const double t = x[r * width + base + j];
        sumsq += t * t;
      }
      const double inv =
          1.0 / std::sqrt(sumsq / static_cast<double>(extent) + (v.no_eps ? 0.0 : eps));
      for (int64_t j = 0; j < extent; ++j) {
        const int64_t idx = base + j;
        const double wj = v.no_fold ? w[idx] : 1.0 + w[idx];
        out[static_cast<size_t>(r * width + idx)] = x[r * width + idx] * inv * wj;
      }
    }
  }
  return out;
}

struct Case {
  // `std::string`, not `const char*`: doctest stringifies a `const char*` INFO
  // argument through its bool overload, so every case would log the same thing.
  std::string name;
  int64_t hidden, hc, T;
  float eps;
  const float* norm_w_hf;
  const float* hyper;
  const float* normed;
};

// hidden is the group_size, hc*hidden the row width — `hc_norm =
// Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, ...)`
// (modeling_qwen4_exp.py:947), which is the same shape the three PLE norms take
// (:1138-1140).
const Case kCases[] = {
    {"A", 6, 4, 3, 1e-6f, kA_norm_w_hf, kA_hyper, kA_normed},
    {"B", 5, 3, 2, 1e-5f, kB_norm_w_hf, kB_hyper, kB_normed},
    {"C", 6, 4, 2, 1e-6f, kC_norm_w_hf, kC_hyper, kC_normed},
    // D is the eps case: `hyper_scale = 0.01`, so the mean square is ~1e-4 and
    // eps is 1% of it rather than 5e-7 of it.
    {"D", 6, 4, 2, 1e-6f, kD_norm_w_hf, kD_hyper, kD_normed},
};

}  // namespace

TEST_CASE("vt::RmsNormGroup reproduces the pinned oracle's Qwen4ExpTextRMSNorm") {
  for (const Case& c : kCases) {
    INFO("case ", c.name);
    const int64_t width = c.hidden * c.hc;
    const std::vector<float> got =
        RunOp(c.hyper, c.norm_w_hf, c.T, width, c.hidden, c.eps, /*gemma=*/true);
    CHECK(MaxAbsDiff(got, c.normed, static_cast<size_t>(c.T * width)) < kTol);
  }
}

TEST_CASE("vt::RmsNormGroup: the goldens SEPARATE the group reduction from the row") {
  // The claim this case makes is about the FIXTURE, not the kernel: a full-row
  // reduction must land far outside kTol on every golden, or the case above
  // could not tell a grouped norm from an ungrouped one.
  for (const Case& c : kCases) {
    INFO("case ", c.name);
    const int64_t width = c.hidden * c.hc;
    const size_t n = static_cast<size_t>(c.T * width);
    const std::vector<double> row =
        Reference(c.hyper, c.norm_w_hf, c.T, width, c.hidden, c.eps, Variant{true, false, false});
    const double sep = MaxAbsDiff(row, c.normed, n);
    INFO("full-row separation ", sep);
    CHECK(sep > 1e2 * kTol);
  }
}

TEST_CASE("vt::RmsNormGroup: two groups four orders of magnitude apart") {
  // The goldens are `torch.randn * 1.7`, so their per-group magnitudes differ by
  // chance rather than by construction. This case builds the discrimination in:
  // group 0 has an RMS of ~1e-2 and group 1 of ~1e+2, so a whole-row reduction
  // is dominated entirely by group 1 and group 0's output is wrong by ~4 orders
  // of magnitude. The expectation is computed in DOUBLE from the upstream lines,
  // not read back from the op.
  constexpr int64_t kT = 3, kGroup = 4, kGroups = 2, kWidth = kGroup * kGroups;
  constexpr float kEps = 1e-6f;
  std::vector<float> x(static_cast<size_t>(kT * kWidth));
  std::vector<float> w(kWidth);
  for (int64_t r = 0; r < kT; ++r) {
    for (int64_t j = 0; j < kWidth; ++j) {
      // A deterministic, non-symmetric fill; the SCALE is the point.
      const float unit = static_cast<float>(1 + ((r * kWidth + j) % 7)) / 4.0f;
      const float scale = (j < kGroup) ? 1e-2f : 1e2f;
      x[static_cast<size_t>(r * kWidth + j)] = unit * scale;
    }
  }
  // A gamma that is neither all-zero nor all-one after the fold: at raw 0 both
  // polarities agree, and at raw -1 the output is zero.
  for (int64_t j = 0; j < kWidth; ++j)
    w[static_cast<size_t>(j)] = 0.25f * static_cast<float>(j) - 0.75f;

  const std::vector<double> want =
      Reference(x.data(), w.data(), kT, kWidth, kGroup, kEps, Variant{});
  const std::vector<double> row =
      Reference(x.data(), w.data(), kT, kWidth, kGroup, kEps, Variant{true, false, false});
  const double sep = MaxAbsDiff(row, want.data(), want.size());
  INFO("full-row separation ", sep);
  REQUIRE(sep > 1e2 * kTol);  // the case discriminates before it asserts

  const std::vector<float> got =
      RunOp(x.data(), w.data(), kT, kWidth, kGroup, kEps, /*gemma=*/true);
  CHECK(MaxAbsDiff(got, want.data(), want.size()) < kTol);
}

TEST_CASE("vt::RmsNormGroup: the `+ 1` fold, and a gamma that can see it") {
  for (const Case& c : kCases) {
    INFO("case ", c.name);
    const int64_t width = c.hidden * c.hc;
    const size_t n = static_cast<size_t>(c.T * width);
    // A gamma centred on zero makes `w` and `1 + w` agree, and #2218 is exactly
    // the defect that hides there. Assert the fixture is not standing on it.
    double smallest = 1e30;
    for (int64_t j = 0; j < width; ++j)
      smallest = std::min(smallest, std::abs(static_cast<double>(c.norm_w_hf[j])));
    INFO("min |w_hf| ", smallest);

    const std::vector<double> unfolded =
        Reference(c.hyper, c.norm_w_hf, c.T, width, c.hidden, c.eps, Variant{false, true, false});
    const double sep = MaxAbsDiff(unfolded, c.normed, n);
    INFO("unfolded separation ", sep);
    CHECK(sep > 1e2 * kTol);

    // And the op must FOLLOW the flag rather than baking the fold in: with
    // gemma = false it computes the unfolded value, which is the same assertion
    // read the other way.
    const std::vector<float> got =
        RunOp(c.hyper, c.norm_w_hf, c.T, width, c.hidden, c.eps, /*gemma=*/false);
    CHECK(MaxAbsDiff(got, unfolded.data(), n) < kTol);
  }
}

TEST_CASE("vt::RmsNormGroup: eps is inside the rsqrt, probed where it is visible") {
  // Case A is carried alongside D on purpose: it is the SCALE at which an eps
  // probe reports nothing, and stating that here stops the next reader from
  // moving the probe onto it.
  for (const Case& c : kCases) {
    INFO("case ", c.name);
    const int64_t width = c.hidden * c.hc;
    const size_t n = static_cast<size_t>(c.T * width);
    const std::vector<double> no_eps =
        Reference(c.hyper, c.norm_w_hf, c.T, width, c.hidden, c.eps, Variant{false, false, true});
    const double sep = MaxAbsDiff(no_eps, c.normed, n);
    INFO("no-eps separation ", sep);
    if (c.name == "D") {
      // hyper_scale = 0.01: eps is ~1% of the mean square.
      CHECK(sep > 1e2 * kTol);
    } else if (c.name == "A") {
      // hyper_scale = 1.7: eps is ~5e-7 of the mean square, and the difference
      // is BELOW the tolerance. This is a recorded property of the fixture, not
      // a weakness of the op.
      CHECK(sep < kTol);
    }
  }
  // The op itself, at D, must be inside the tolerance the separation above
  // makes meaningful.
  const Case& d = kCases[3];
  const int64_t width = d.hidden * d.hc;
  const std::vector<float> got =
      RunOp(d.hyper, d.norm_w_hf, d.T, width, d.hidden, d.eps, /*gemma=*/true);
  CHECK(MaxAbsDiff(got, d.normed, static_cast<size_t>(d.T * width)) < kTol);
}

TEST_CASE("vt::RmsNormGroup: ONE rounding, on the store") {
  // `output = self._norm(x.float()); output = output * (1.0 + self.weight.float());
  // return output.type_as(x)` (:174-178), with upstream's own comment at
  // :175-176 saying what it is NOT: "Llama does x.to(float16) * w whilst
  // Qwen4ExpText is (x * w).to(float16)". A kernel that narrows the normalized
  // value before the weight multiply is a different model, and a token gate
  // cannot see the difference.
  constexpr int64_t kT = 2, kGroup = 4, kWidth = 8;
  constexpr float kEps = 1e-6f;
  std::vector<float> x(static_cast<size_t>(kT * kWidth));
  std::vector<float> w(kWidth);
  for (int64_t r = 0; r < kT; ++r)
    for (int64_t j = 0; j < kWidth; ++j)
      x[static_cast<size_t>(r * kWidth + j)] =
          0.37f + 0.11f * static_cast<float>(j) + 0.53f * static_cast<float>(r);
  for (int64_t j = 0; j < kWidth; ++j)
    w[static_cast<size_t>(j)] = -0.37f + 0.29f * static_cast<float>(j);

  const std::vector<double> exact =
      Reference(x.data(), w.data(), kT, kWidth, kGroup, kEps, Variant{});

  // The two orders, both realized here in bf16, so the case can say it
  // discriminates before it asserts which one the op takes.
  std::vector<float> round_late(exact.size()), round_early(exact.size());
  for (int64_t r = 0; r < kT; ++r) {
    for (int64_t g = 0; g < kWidth / kGroup; ++g) {
      double sumsq = 0.0;
      for (int64_t j = 0; j < kGroup; ++j) {
        const double t = x[static_cast<size_t>(r * kWidth + g * kGroup + j)];
        sumsq += t * t;
      }
      const double inv = 1.0 / std::sqrt(sumsq / static_cast<double>(kGroup) + kEps);
      for (int64_t j = 0; j < kGroup; ++j) {
        const size_t idx = static_cast<size_t>(r * kWidth + g * kGroup + j);
        const double wj = 1.0 + w[static_cast<size_t>(g * kGroup + j)];
        const double normed = x[idx] * inv;
        round_late[idx] = vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(normed * wj)));
        const float narrowed = vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(normed)));
        round_early[idx] =
            vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(narrowed * wj)));
      }
    }
  }
  size_t differing = 0;
  for (size_t i = 0; i < round_late.size(); ++i)
    if (round_late[i] != round_early[i]) ++differing;
  INFO("elements where the two rounding orders differ: ", differing);
  REQUIRE(differing > 0);  // the case discriminates before it asserts

  std::vector<float> xs = x, ws = w;
  std::vector<uint16_t> out_bf16(exact.size(), 0);
  Tensor tx = MakeT(xs.data(), DType::kF32, {kT, kWidth});
  Tensor tw = MakeT(ws.data(), DType::kF32, {kWidth});
  Tensor to = MakeT(out_bf16.data(), DType::kBF16, {kT, kWidth});
  RmsNormGroupArgs args;
  args.eps = kEps;
  args.gemma = true;
  args.group_size = kGroup;
  Queue q = CpuQ();
  vt::RmsNormGroup(q, to, tx, tw, args);

  std::vector<float> got(exact.size());
  for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(out_bf16[i]);
  // EXACT, not a tolerance: bf16 is the store width and the claim is which
  // value was stored.
  for (size_t i = 0; i < got.size(); ++i) {
    INFO("element ", i);
    CHECK(got[i] == round_late[i]);
  }
}

TEST_CASE("vt::RmsNormGroup: refusals name the caller's mistake") {
  constexpr int64_t kT = 2, kWidth = 6;
  std::vector<float> x(static_cast<size_t>(kT * kWidth), 1.0f);
  std::vector<float> w(kWidth, 0.0f);
  std::vector<float> out(static_cast<size_t>(kT * kWidth), 0.0f);
  Tensor tx = MakeT(x.data(), DType::kF32, {kT, kWidth});
  Tensor tw = MakeT(w.data(), DType::kF32, {kWidth});
  Tensor to = MakeT(out.data(), DType::kF32, {kT, kWidth});
  Queue q = CpuQ();

  // The message is checked, not only the throw: a refusal that does not name the
  // caller's mistake sends the reader to the kernel instead of to their call.
  auto refusal = [](auto&& fn) -> std::string {
    try {
      fn();
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
    return std::string();
  };

  SUBCASE("group_size 0 is refused, NOT read as the whole row") {
    RmsNormGroupArgs args;  // the default, deliberately unusable
    const std::string msg = refusal([&] { vt::RmsNormGroup(q, to, tx, tw, args); });
    INFO("message: ", msg);
    REQUIRE(!msg.empty());
    CHECK(msg.find("group_size must be >= 1") != std::string::npos);
    CHECK(msg.find("vt::RmsNorm") != std::string::npos);
  }
  SUBCASE("a group_size that does not divide the width is refused") {
    RmsNormGroupArgs args;
    args.group_size = 4;  // 6 % 4 != 0, upstream's own ValueError (:164-165)
    const std::string msg = refusal([&] { vt::RmsNormGroup(q, to, tx, tw, args); });
    INFO("message: ", msg);
    REQUIRE(!msg.empty());
    CHECK(msg.find("must divide the last dim") != std::string::npos);
  }
  SUBCASE("a weight of the wrong width is refused") {
    std::vector<float> bad(kWidth - 1, 0.0f);
    Tensor tb = MakeT(bad.data(), DType::kF32, {kWidth - 1});
    RmsNormGroupArgs args;
    args.group_size = 3;
    const std::string msg = refusal([&] { vt::RmsNormGroup(q, to, tx, tb, args); });
    INFO("message: ", msg);
    REQUIRE(!msg.empty());
    CHECK(msg.find("weight size mismatch") != std::string::npos);
  }
  SUBCASE("the legal shape is accepted, so the refusals above are not vacuous") {
    RmsNormGroupArgs args;
    args.group_size = 3;
    CHECK_NOTHROW(vt::RmsNormGroup(q, to, tx, tw, args));
  }
}
