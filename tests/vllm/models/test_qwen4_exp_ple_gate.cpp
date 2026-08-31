// Qwen4-Exp (Qwen3.8-Flash-Next) W5e-1 GATE — `vt::Qwen4ExpPleGate`, the PLE
// signed-square-root gate and the broadcast sigmoid scale it feeds, as a `vt::`
// op over `vt::Tensor`.
// Issue #2336, campaign issue #1978, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT IS UNDER TEST, AND WHAT IT IS COMPARED AGAINST. The op is compared
// DIRECTLY against the lane-pinned oracle, never against this repository's host
// reference: section J of `qwen4_exp_ple_goldens.inc` is dumped by
// `scripts/gen-qwen4-exp-ple-goldens.py`, which `exec`s
// `modeling_qwen4_exp.py:1181-1182` and the `:1184` flatten VERBATIM by line
// range out of transformers **v5.16.0** (sha256 77fec77d…c459) and runs them
// under torch. The W2 host reference `PleForward` computes the same arithmetic
// inline and is gated on section G of the same file, so the two arms answer to
// ONE oracle rather than to each other.
//
// THE ANCHOR. #2336 cites this block as `:1179-1183`. At the pinned file `:1179`
// is the `query_normed` unflatten and `:1183` is the `norm_conv` call, so the
// gate is `:1180-1182` and the flatten it feeds is `:1184`. Every citation in
// this file and in the generator uses the corrected numbers.
//
// THE CLAMP IS THE VARIABLE, AND THE ORACLE SUPPLIES BOTH SIDES OF IT. A fixture
// on which `clamp_min(1e-6)` never bound would pass with the clamp deleted —
// the blind spot #2272 recorded for an eps invisible at two of four goldens. The
// generator therefore builds three of the twelve `(t, j)` pairs to fall under
// the floor and leaves nine dense above it, records WHICH in
// `kGateClampBinds`, and measures what the clamp is worth on the output
// (`kGateClampSeparation`). `the fixture actually probes the clamp` below
// asserts that population rather than trusting the comment beside it, so a
// future regeneration that stopped probing could not pass in silence.
//
// THE ORIGIN IS A REAL CASE, NOT AN EDGE. `torch.sign(0) == 0`, so a zero score
// maps to a zero gate and `0.5 * value`, NOT to the 1e-3 floor the clamp puts
// under every other tiny score. A fully masked row scores exactly zero, so the
// discontinuity is reachable in production and is pinned in BOTH directions:
// the origin must not move to the floor, and its tiny neighbours must not move
// to the origin.
//
// A NaN SCORE IS OUT OF CONTRACT AND STILL HAS AN UPSTREAM ANSWER, WHICH IS
// NaN. `torch.sign(NaN) == 0` and `NaN * 0.0 == NaN`, so the gate and the
// output are both NaN at the pin. Every comparison in `SignedSqrt` is false for
// NaN, so an unguarded kernel falls through the sign branches and returns 0.0,
// and the output becomes exactly `0.5 * value` — a poison operand rendered as a
// plausible number, #2272's polarity, and one `max_abs_diff.h`'s finiteness
// guard cannot see because the non-finite operand is gone by then. `a NaN score
// PROPAGATES` pins it, and pins the origin row beside it so a guard that leaked
// finite scores would red as well.
//
// WHAT IS NOT THIS OP, AND IS TESTED ANYWAY. The dot at `:1180` is
// `vt::BatchedMatmul` over strided views of the two `[T, hc*H]` buffers, not new
// code. `composed with vt::BatchedMatmul` RUNS that composition end to end
// against the same golden, so #2336's claim that the dot needs no new op is
// measured here rather than asserted in a comment.
//
// SCOPE, HONESTLY. This file is the CPU arm's gate. A CUDA arm of this op DOES
// now exist (`src/vt/cuda/cuda_qwen4_exp_ple.cu`, W6-CUDA) and is gated in
// `test_qwen4_exp_cuda.cpp`, which carries the NaN case this op's declaration
// requires of a device arm; nothing below runs on a device. The text that stood
// here said no CUDA arm exists, and it read: CPU only — no CUDA arm of this op exists, and one written on
// this CPU-only host could not be gated on it. Nothing calls this op from a
// production entry point yet: `ForwardQwen4ExpForConditionalGeneration` still
// refuses by name, the PLE block that will call it is W5e-2, the wiring is owned
// by row `MODEL-MM-QWEN4-EXP` and tracked by #2031, and the spec's `## Owed`
// records it. No token claim and no speed claim.
#include <doctest/doctest.h>

#include <cmath>
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
using vt::Qwen4ExpPleGateArgs;
using vt::Queue;
using vt::Tensor;

namespace {

#include "qwen4_exp_ple_goldens.inc"  // NOLINT — golden literals

// The goldens are fp32 out of torch; the op's interior is double with one f32
// store. The two agree to well under an ulp of the sigmoid, so this bound is
// loose by orders of magnitude for everything except a real defect — and the
// defect it has to separate first, a dropped clamp, sits at
// kGateClampSeparation = 1.56e-3, which `the fixture actually probes the clamp`
// re-measures against this number.
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

// A VIEW with caller-chosen strides. vt::BatchedMatmul constrains only the
// innermost stride, which is what lets the [T, hc*H] key and query buffers be
// read as [T*hc, 1, H] and [T*hc, H, 1] with no copy.
Tensor MakeView(void* data, DType dt, const std::vector<int64_t>& shape,
                const std::vector<int64_t>& stride) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride[static_cast<size_t>(i)];
  }
  return t;
}

double Sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// One run of the op over caller-supplied scores and values.
std::vector<float> RunGate(const std::vector<float>& score, const std::vector<float>& value,
                           int64_t tokens, int64_t hc, int64_t hidden, float divisor,
                           float clamp_min = 1e-6f) {
  Queue q = CpuQ();
  std::vector<float> s = score;
  std::vector<float> v = value;
  std::vector<float> out(static_cast<size_t>(tokens * hc * hidden), 0.0f);
  Tensor t_s = MakeT(s.data(), DType::kF32, {tokens, hc});
  Tensor t_v = MakeT(v.data(), DType::kF32, {tokens, hidden});
  Tensor t_o = MakeT(out.data(), DType::kF32, {tokens, hc * hidden});
  Qwen4ExpPleGateArgs args;
  args.gate_divisor = divisor;
  args.clamp_min = clamp_min;
  vt::Qwen4ExpPleGate(q, t_o, t_s, t_v, args);
  return out;
}

constexpr int64_t kT = 6;    // kGateT
constexpr int64_t kHc = 2;   // kGateHc
constexpr int64_t kH = 8;    // kGateH

}  // namespace

TEST_CASE("vt::Qwen4ExpPleGate reproduces the pinned oracle") {
  REQUIRE(kGateT == kT);
  REQUIRE(kGateHc == kHc);
  REQUIRE(kGateH == kH);
  // The scores are upstream's OWN :1180 output, so `gate_divisor` is the
  // identity here and the op is held to :1181-1182 alone. The divide is
  // exercised by `composed with vt::BatchedMatmul` below, which starts one line
  // earlier and passes the real sqrt(hidden_size).
  const std::vector<float> score(kGateScaledDot, kGateScaledDot + kT * kHc);
  const std::vector<float> value(kGateValueIn, kGateValueIn + kT * kH);
  const std::vector<float> got = RunGate(score, value, kT, kHc, kH, 1.0f);
  CHECK(MaxAbsDiff(got, kGateExpectedOut, static_cast<size_t>(kT * kHc * kH)) < kTol);
}

TEST_CASE("the fixture actually probes the clamp, in both directions") {
  // A tolerance cannot tell "the clamp is inert on this input" from "the clamp
  // is not implemented". This case makes the fixture state, in its own output,
  // which of the twelve pairs it is probing with.
  const double floor_value = std::sqrt(1e-6);
  int binding = 0, inert = 0;
  for (int64_t i = 0; i < kT * kHc; ++i) {
    const double pre = kGateScaledDot[i];
    const double post = kGatePostSqrt[i];
    const bool binds = kGateClampBinds[i] != 0;
    // The recorded population must be what the recorded scores say it is.
    CHECK(binds == (std::abs(pre) < 1e-6));
    if (i == 0) {
      // The ORIGIN. `sign(0) == 0` cancels the floor, so this pair is under the
      // clamp AND still maps to zero. It is the one place where "binds" does not
      // mean "reads 1e-3".
      CHECK(pre == 0.0);
      CHECK(post == 0.0);
      ++binding;
      continue;
    }
    if (binds) {
      CHECK(std::abs(std::abs(post) - floor_value) < 1e-9);
      ++binding;
    } else {
      CHECK(std::abs(post) > floor_value);
      ++inert;
    }
  }
  // Three under the floor (one of them the origin) and nine above it. Both
  // halves are load-bearing: without the first the clamp is untested, without
  // the second an unconditional floor would pass.
  CHECK(binding == 3);
  CHECK(inert == kT * kHc - 3);
  // And the clamp has to be WORTH something on the output, not merely present.
  // The generator measured this by running the same three upstream lines with
  // the floor taken to zero.
  INFO("clamp separation " << kGateClampSeparation << " vs tolerance " << kTol);
  CHECK(static_cast<double>(kGateClampSeparation) > 100.0 * kTol);
}

TEST_CASE("composed with vt::BatchedMatmul: the DOT needs no new op") {
  // #2336 claims the `:1180` dot composes from vt::BatchedMatmul over VIEWS of
  // the [T, hc*H] key and query buffers, with no copy and no tiling. This runs
  // it: key as [T*hc, 1, H], query as [T*hc, H, 1], out as [T*hc, 1, 1], which
  // is the [T, hc] score buffer the gate wants. Only the innermost stride is
  // constrained, so the batch and row strides below are the buffers' own.
  Queue q = CpuQ();
  const int64_t g = kT * kHc;
  std::vector<float> key(kGateKeyNormed, kGateKeyNormed + g * kH);
  std::vector<float> query(kGateQueryNormed, kGateQueryNormed + g * kH);
  std::vector<float> score(static_cast<size_t>(g), 0.0f);

  Tensor t_k = MakeView(key.data(), DType::kF32, {g, 1, kH}, {kH, kH, 1});
  Tensor t_q = MakeView(query.data(), DType::kF32, {g, kH, 1}, {kH, 1, 1});
  Tensor t_s = MakeView(score.data(), DType::kF32, {g, 1, 1}, {1, 1, 1});
  vt::BatchedMatmul(q, t_s, t_k, t_q);

  // The composed dot must be upstream's :1180 numerator. Compared against the
  // ORACLE's own scaled score times the divisor it was divided by, so this is
  // not a read-back of the line under test.
  std::vector<float> want_dot(static_cast<size_t>(g));
  for (int64_t i = 0; i < g; ++i) {
    want_dot[static_cast<size_t>(i)] =
        static_cast<float>(static_cast<double>(kGateScaledDot[i]) *
                           static_cast<double>(kGateDivisor));
  }
  CHECK(MaxAbsDiff(score, want_dot) < 1e-5);

  const std::vector<float> value(kGateValueIn, kGateValueIn + kT * kH);
  const std::vector<float> got = RunGate(score, value, kT, kHc, kH, kGateDivisor);
  CHECK(MaxAbsDiff(got, kGateExpectedOut, static_cast<size_t>(kT * kHc * kH)) < kTol);
}

TEST_CASE("the ORIGIN maps to zero, and its neighbours do NOT map to the origin") {
  // Two assertions in opposite directions, because each alone passes a wrong
  // port: `sign(0) == +1` moves the origin onto the 1e-3 floor, and a
  // `if (|g| < eps) return 0` shortcut moves every clamped score onto the
  // origin. The scores here are section E's own oracle probes.
  const std::vector<float> value{1.0f, -2.0f, 0.0f, 0.5f};
  const int64_t hidden = static_cast<int64_t>(value.size());
  // score row: [0, +1e-12, -1e-12] — the origin and its two tiniest neighbours.
  const std::vector<float> score{kGateInput[0], kGateInput[1], kGateInput[2]};
  REQUIRE(score[0] == 0.0f);
  const std::vector<float> got = RunGate(score, value, 1, 3, hidden, 1.0f);

  // j = 0: gate 0, so exactly 0.5 * value.
  for (int64_t d = 0; d < hidden; ++d) {
    CHECK(static_cast<double>(got[static_cast<size_t>(d)]) ==
          doctest::Approx(0.5 * static_cast<double>(value[static_cast<size_t>(d)])));
  }
  // j = 1 and j = 2: the ORACLE's post-sqrt values, which are +/-1e-3 and not 0.
  for (int64_t j = 1; j < 3; ++j) {
    const double w = Sigmoid(static_cast<double>(kGateExpected[j]));
    for (int64_t d = 0; d < hidden; ++d) {
      CHECK(static_cast<double>(got[static_cast<size_t>(j * hidden + d)]) ==
            doctest::Approx(w * static_cast<double>(value[static_cast<size_t>(d)])));
    }
  }
  // And the two must be DISTINGUISHABLE on a non-zero value, or neither
  // assertion above gates anything.
  CHECK(std::abs(static_cast<double>(got[1]) -
                 static_cast<double>(got[static_cast<size_t>(hidden + 1)])) > kTol);
  CHECK(std::abs(static_cast<double>(got[1]) -
                 static_cast<double>(got[static_cast<size_t>(2 * hidden + 1)])) > kTol);
}

TEST_CASE("a NaN score PROPAGATES; it is not swallowed into 0.5 * value") {
  // OUT-OF-CONTRACT INPUT, AND THE ANSWER IS STILL UPSTREAM'S. At the pin,
  // `torch.sign(NaN) == 0` but `NaN * 0.0 == NaN`, so
  // `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` is NaN and
  // `sigmoid(gate) * value` is NaN. That was measured by running the pinned
  // expression itself under torch on `[nan, inf, -inf, -0.0, 0.0]`, which
  // returns `[nan, inf, -inf, 0.0, 0.0]` — so only the NaN arm needed a guard
  // and the two infinities and the two zeros did not.
  //
  // WHAT THIS SEPARATES. `SignedSqrt`'s clamp comparison, its two sign branches
  // and its fall-through are all FALSE for NaN, so an unguarded kernel returns
  // 0.0 and the gate becomes exactly `0.5 * value` — a poison operand rendered
  // as a plausible number, which is #2272's polarity. `MaxAbsDiff` cannot catch
  // it, because by then there is no non-finite operand left to catch.
  const std::vector<float> value{1.0f, -2.0f, 0.0f, 0.5f};
  const int64_t hidden = static_cast<int64_t>(value.size());
  const std::vector<float> score{std::nanf(""), 0.0f};
  const std::vector<float> got = RunGate(score, value, 1, 2, hidden, 1.0f);

  // j = 0 is the NaN row. Every d must be NaN, INCLUDING d = 2 where `value` is
  // 0: `NaN * 0` is NaN, and a swallowed NaN would read 0 there too, so that
  // column is the one an `isfinite`-shaped check would miss.
  for (int64_t d = 0; d < hidden; ++d) {
    CHECK(std::isnan(static_cast<double>(got[static_cast<size_t>(d)])));
  }
  // j = 1 is the ORIGIN, and it must STILL be `0.5 * value`. This is the other
  // direction: a guard written one line too early, or widened to `!isfinite`,
  // would leak a finite score out of the op unchanged and this row would move.
  for (int64_t d = 0; d < hidden; ++d) {
    CHECK(static_cast<double>(got[static_cast<size_t>(hidden + d)]) ==
          doctest::Approx(0.5 * static_cast<double>(value[static_cast<size_t>(d)])));
  }
}

TEST_CASE("section E's eleven scalar probes, through the op") {
  // kGateExpected is upstream's :1181 output for kGateInput, measured. The
  // expectation below is sigmoid() of THAT, times a value this test chose — an
  // independent composition of the oracle's number, never a read-back.
  const std::vector<float> value{1.0f, -0.75f, 0.0f, 3.0f};
  const int64_t hidden = static_cast<int64_t>(value.size());
  const int64_t hc = kGateCount;
  const std::vector<float> score(kGateInput, kGateInput + hc);
  const std::vector<float> got = RunGate(score, value, 1, hc, hidden, 1.0f);

  std::vector<double> want(static_cast<size_t>(hc * hidden));
  for (int64_t j = 0; j < hc; ++j) {
    const double w = Sigmoid(static_cast<double>(kGateExpected[j]));
    for (int64_t d = 0; d < hidden; ++d) {
      want[static_cast<size_t>(j * hidden + d)] =
          w * static_cast<double>(value[static_cast<size_t>(d)]);
    }
  }
  CHECK(MaxAbsDiff(got, want.data(), want.size()) < kTol);

  // The clamped probes must SEPARATE from the origin, so the eleven are not
  // eleven copies of 0.5 * value. Indices 1..4 are +/-1e-12 and +/-1e-6, all
  // under the floor; index 0 is the origin.
  for (int64_t j = 1; j <= 4; ++j) {
    CHECK(std::abs(want[static_cast<size_t>(j * hidden)] - want[0]) > kTol);
  }
}

TEST_CASE("vt::Qwen4ExpPleGate: bf16 storage rounds ONCE, on the store") {
  // The interior is double whatever the store width is, so a bf16 out must be
  // the bf16 rounding of the f32 answer and never a bf16-rounded interior. The
  // value operand is bf16 too, which IS a value change and is the caller's.
  Queue q = CpuQ();
  std::vector<float> score(kGateScaledDot, kGateScaledDot + kT * kHc);
  std::vector<float> value32(kGateValueIn, kGateValueIn + kT * kH);
  std::vector<float> out32(static_cast<size_t>(kT * kHc * kH), 0.0f);
  Qwen4ExpPleGateArgs args;

  Tensor t_s = MakeT(score.data(), DType::kF32, {kT, kHc});
  Tensor t_v32 = MakeT(value32.data(), DType::kF32, {kT, kH});
  Tensor t_o32 = MakeT(out32.data(), DType::kF32, {kT, kHc * kH});
  vt::Qwen4ExpPleGate(q, t_o32, t_s, t_v32, args);

  // bf16 OUT over the same f32 value: the answer must be out32 rounded once.
  std::vector<uint16_t> outbf(static_cast<size_t>(kT * kHc * kH), 0);
  Tensor t_obf = MakeT(outbf.data(), DType::kBF16, {kT, kHc * kH});
  vt::Qwen4ExpPleGate(q, t_obf, t_s, t_v32, args);
  for (size_t i = 0; i < out32.size(); ++i) {
    CHECK(vt::BF16ToF32(outbf[i]) == vt::BF16ToF32(vt::F32ToBF16(out32[i])));
  }

  // bf16 VALUE: the product is still formed in double from the widened operand,
  // so the answer is the f32 one recomputed on the rounded value — not the f32
  // answer rounded.
  std::vector<uint16_t> valuebf(static_cast<size_t>(kT * kH), 0);
  for (size_t i = 0; i < value32.size(); ++i) valuebf[i] = vt::F32ToBF16(value32[i]);
  std::vector<float> value_round(value32.size());
  for (size_t i = 0; i < value32.size(); ++i) value_round[i] = vt::BF16ToF32(valuebf[i]);
  std::vector<float> out_from_bf(static_cast<size_t>(kT * kHc * kH), 0.0f);
  Tensor t_vbf = MakeT(valuebf.data(), DType::kBF16, {kT, kH});
  Tensor t_obf2 = MakeT(out_from_bf.data(), DType::kF32, {kT, kHc * kH});
  vt::Qwen4ExpPleGate(q, t_obf2, t_s, t_vbf, args);
  const std::vector<float> want =
      RunGate(score, value_round, kT, kHc, kH, 1.0f);
  CHECK(MaxAbsDiff(out_from_bf, want) == 0.0);
}

TEST_CASE("vt::Qwen4ExpPleGate at a MODEL-shaped width, against a second reference") {
  // The goldens are hc = 2, H = 8, which cannot separate a transposed
  // (H, hc) flatten from the (hc, H) one upstream produces: at those sizes a
  // wrong axis order still lands inside the buffer. Non-power-of-two extents
  // here, and a second reference computed independently in double.
  const int64_t tokens = 5, hc = 3, hidden = 257;
  std::vector<float> score(static_cast<size_t>(tokens * hc));
  std::vector<float> value(static_cast<size_t>(tokens * hidden));
  for (int64_t i = 0; i < tokens * hc; ++i) {
    // A spread that straddles the floor: an exact zero, two clamped, the rest
    // dense and of both signs.
    const double table[5] = {0.0, 1e-9, -3e-8, 0.75, -1.5};
    score[static_cast<size_t>(i)] = static_cast<float>(table[i % 5] * (1.0 + 0.1 * i));
  }
  for (int64_t i = 0; i < tokens * hidden; ++i) {
    value[static_cast<size_t>(i)] =
        static_cast<float>(std::sin(0.37 * static_cast<double>(i)) * 1.75);
  }
  const float divisor = static_cast<float>(std::sqrt(static_cast<double>(hidden)));
  const std::vector<float> got = RunGate(score, value, tokens, hc, hidden, divisor);

  std::vector<double> want(static_cast<size_t>(tokens * hc * hidden));
  int clamped = 0, dense = 0;
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t j = 0; j < hc; ++j) {
      const double g =
          static_cast<double>(score[static_cast<size_t>(t * hc + j)]) /
          static_cast<double>(divisor);
      const double mag = std::abs(g);
      if (mag < 1e-6) ++clamped; else ++dense;
      const double root = std::sqrt(mag < 1e-6 ? 1e-6 : mag);
      const double gate = g > 0.0 ? root : (g < 0.0 ? -root : 0.0);
      const double w = Sigmoid(gate);
      for (int64_t d = 0; d < hidden; ++d) {
        want[static_cast<size_t>((t * hc + j) * hidden + d)] =
            w * static_cast<double>(value[static_cast<size_t>(t * hidden + d)]);
      }
    }
  }
  INFO("clamped pairs " << clamped << ", dense pairs " << dense);
  CHECK(clamped >= 3);
  CHECK(dense >= 3);
  CHECK(MaxAbsDiff(got, want.data(), want.size()) < 1e-6);
}

TEST_CASE("vt::Qwen4ExpPleGate refuses by name") {
  Queue q = CpuQ();
  std::vector<float> score(static_cast<size_t>(kT * kHc), 0.25f);
  std::vector<float> value(static_cast<size_t>(kT * kH), 0.5f);
  std::vector<float> out(static_cast<size_t>(kT * kHc * kH), 0.0f);
  Tensor t_s = MakeT(score.data(), DType::kF32, {kT, kHc});
  Tensor t_v = MakeT(value.data(), DType::kF32, {kT, kH});
  Tensor t_o = MakeT(out.data(), DType::kF32, {kT, kHc * kH});
  const Qwen4ExpPleGateArgs ok;

  SUBCASE("out must be [T, hc*H], and the message names both factors") {
    Tensor bad = MakeT(out.data(), DType::kF32, {kT, kHc * kH - 1});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpPleGate(q, bad, t_s, t_v, ok),
                         doctest::Contains("out must be [T, hc*H] = [T,2*8] = [T,16]"),
                         std::runtime_error);
  }
  SUBCASE("T must agree across all three") {
    std::vector<float> shortv(static_cast<size_t>((kT - 1) * kH), 0.5f);
    Tensor bad = MakeT(shortv.data(), DType::kF32, {kT - 1, kH});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpPleGate(q, t_o, t_s, bad, ok),
                         doctest::Contains("must agree on T"), std::runtime_error);
  }
  SUBCASE("score must be f32") {
    std::vector<uint16_t> sbf(static_cast<size_t>(kT * kHc), 0);
    Tensor bad = MakeT(sbf.data(), DType::kBF16, {kT, kHc});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpPleGate(q, t_o, bad, t_v, ok),
                         doctest::Contains("score must be f32"), std::runtime_error);
  }
  SUBCASE("gate_divisor must be positive") {
    Qwen4ExpPleGateArgs bad = ok;
    bad.gate_divisor = 0.0f;
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpPleGate(q, t_o, t_s, t_v, bad),
                         doctest::Contains("gate_divisor must be > 0"), std::runtime_error);
  }
  SUBCASE("clamp_min 0 is refused, because it is not 'no floor'") {
    Qwen4ExpPleGateArgs bad = ok;
    bad.clamp_min = 0.0f;
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpPleGate(q, t_o, t_s, t_v, bad),
                         doctest::Contains("clamp_min must be > 0"), std::runtime_error);
  }
  SUBCASE("rank") {
    Tensor bad = MakeT(out.data(), DType::kF32, {kT, kHc, kH});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpPleGate(q, bad, t_s, t_v, ok),
                         doctest::Contains("out [T,hc*H], score [T,hc], value [T,H]"),
                         std::runtime_error);
  }
}
