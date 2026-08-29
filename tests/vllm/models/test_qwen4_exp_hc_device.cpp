// Qwen4-Exp (Qwen3.8-Flash-Next) W5b-2 DEVICE-ARM GATE — `vt::Qwen4ExpGatedResidual`
// and `vt::Qwen4ExpGatedResidualWriteBack`, the 4-branch gated-residual
// hyper-connection stream as `vt::` ops over `vt::Tensor`.
// Issue #2031, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT IS UNDER TEST, AND WHAT IT IS COMPARED AGAINST. The op is compared
// DIRECTLY against the lane-pinned oracle, not against this repository's host
// reference: `qwen4_exp_hc_goldens.inc` is dumped by
// `scripts/gen-qwen4-exp-hc-goldens.py`, which lifts `Qwen4ExpTextRMSNorm`
// (:158-181) and `Qwen4ExpTextGatedResidual` (:941-969) VERBATIM by line range
// out of transformers **v5.16.0**
// `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`, sha256
// 77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459, and
// EXECUTES them under torch. The write-back goldens replay the two verbatim
// lines of `Qwen4ExpTextDecoderLayer.forward`. The same file already gates the
// W3 host reference (`test_qwen4_exp_hc.cpp`); reusing it means the two arms are
// held to ONE oracle rather than to each other.
//
// WHY THE GOLDENS DISCRIMINATE, stated rather than assumed. `test_qwen4_exp_hc.cpp`
// carries the measurement: an independent double-precision `Variant` reference
// with one flag per plausible single-character defect, and the separation of the
// narrowest flip (the `/ hc_count` moved outside the SiLU) from the oracle is
// 6.63e-3 against a 1e-5 tolerance — a 663x band. That sweep is not repeated
// here, because it measures the GOLDENS and the goldens have not changed; what
// is measured here is whether the OP lands inside the band.
//
// THE BATCH AXIS IS THE NEW THING. The host reference is per token by
// construction (`.h`: "the host signatures are per-sequence precisely so it
// drops in"). Every golden case below carries T > 1 and is driven through the op
// in ONE call, so a kernel that silently computed token 0 and broadcast it, or
// that walked the low-rank intermediate with the wrong row stride, fails here
// and could not fail there.
//
// SCOPE, HONESTLY. CPU only — no CUDA arm of this op exists, and one written on
// this host could not be gated on it. No token claim and no speed claim: nothing
// calls these ops from a production entry point yet (the forward is owed, see
// the spec's `## Owed`), and no `qwen4_exp` arm decodes.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vllm/model_executor/models/qwen4_exp_hc.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm_test::MaxAbsDiff;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Qwen4ExpGatedResidualArgs;
using vt::Queue;
using vt::Tensor;

namespace {

#include "qwen4_exp_hc_goldens.inc"  // NOLINT — golden literals

// The goldens are fp32 out of torch and the op's interior is fp32 with a double
// per-group sum of squares, so at these widths (flat = 24 and 15) the two are
// bit-identical or within one ulp. `kTol` is the value `test_qwen4_exp_hc.cpp`
// already justifies for THESE SHAPES and nothing else; the real-width case at
// the bottom of this file is the measurement that says why it cannot be reused
// at hidden_size 2560.
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

// The gamma the OP takes, and since #2218 that is the goldens' own value with
// nothing done to it. The goldens store the HuggingFace parameter (zero-init,
// so upstream spells the norm `output * (1.0 + weight)`) and
// `vt::Qwen4ExpGatedResidual` now adds the 1 itself, the same way
// `vt::RmsNorm(gemma=true)` and `vt::Qwen4ExpQsaCompress` do for this
// architecture's other gammas — so a loaded `Qwen4ExpWeights` can be handed
// straight to it. `vllm::qwen4_exp::HcNormWeightFromHf` remains the
// `w_hf -> 1 + w_hf` bridge for the HOST reference, whose `GroupedRmsNorm`
// keeps vLLM's `out * w` form and which `test_qwen4_exp_hc.cpp` drives; this
// suite compares the op against the transformers goldens directly, so it no
// longer needs the transform at all.
//
// A PASS-THROUGH THAT IS NOT DECORATION: it names, at every call site, WHICH
// parameterization the op is being handed, which is the entire content of
// #2218. An edit that reintroduces a fold here has to say so out loud.
std::vector<float> OpGamma(const float* w_hf, int64_t n) {
  return std::vector<float>(w_hf, w_hf + n);
}

struct Case {
  // `std::string`, NOT `const char*`: doctest stringifies a `const char*` INFO
  // argument through its bool overload, so `INFO("case ", c.name, ...)` logged
  // `case 1` for every one of A/B/C/D and no failure said which case reddened.
  std::string name;
  int64_t hidden, hc, lowrank, T;
  float eps;
  const float *norm_w_hf, *down, *up, *inject, *hyper, *mixed, *block_out, *inj_w, *written;
};

const Case kCaseA{"A", 6, 4, 5, 3, 1e-6f, kA_norm_w_hf, kA_down,      kA_up,
                  kA_inject,          kA_hyper,   kA_mixed,     kA_block_out, kA_inj_w,
                  kA_written};
const Case kCaseB{"B", 5, 3, 7, 2, 1e-5f, kB_norm_w_hf, kB_down,      kB_up,
                  kB_inject,          kB_hyper,   kB_mixed,     kB_block_out, kB_inj_w,
                  kB_written};
// C is the `use_combine=False` arm — the model-level mixer. Upstream returns
// `mixed_input` alone there and registers NO `block_inject_weight`, which is why
// the last four pointers are null.
const Case kCaseC{"C",     6,        4,       5,       2, 1e-6f, kC_norm_w_hf,
                  kC_down, kC_up,    nullptr, kC_hyper, kC_mixed, nullptr,
                  nullptr, nullptr};
// D is the SMALL-MAGNITUDE case, and it is here because A, B and C structurally
// cannot see where `eps` goes. eps is INSIDE the rsqrt, added to the MEAN SQUARE
// (`torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`, :170), and the
// plausible slip adds it to the norm instead. At A/B/C's `hyper_scale = 1.7` the
// mean square is O(1), so with eps = 1e-6 the two spellings differ by ~5e-7 —
// under `kTol`, and the wrong one passes. MEASURED, not supposed: mutation M7 of
// the W5b-2 battery SURVIVED all three cases and REDS this one. At
// `hyper_scale = 0.01` the mean square is ~1e-4, eps is 1% of it, and the two
// answers separate by ~0.5%.
const Case kCaseD{"D", 6, 4, 5, 2, 1e-6f, kD_norm_w_hf, kD_down,      kD_up,
                  kD_inject,          kD_hyper,   kD_mixed,     kD_block_out, kD_inj_w,
                  kD_written};

// Drive one golden case through the op, batched: all T tokens in ONE call.
void RunReadCase(const Case& c) {
  Queue q = CpuQ();
  const int64_t flat = c.hc * c.hidden;
  std::vector<float> w = OpGamma(c.norm_w_hf, flat);
  std::vector<float> hyper(c.hyper, c.hyper + c.T * flat);
  std::vector<float> down(c.down, c.down + c.lowrank * flat);
  std::vector<float> up(c.up, c.up + flat * c.lowrank);
  std::vector<float> mixed(static_cast<size_t>(c.T * c.hidden), 0.0f);
  std::vector<float> inj(static_cast<size_t>(c.T * c.hc), 0.0f);
  std::vector<float> inject_w;
  if (c.inject != nullptr) inject_w.assign(c.inject, c.inject + c.hc * flat);

  Tensor t_hyper = MakeT(hyper.data(), DType::kF32, {c.T, flat});
  Tensor t_w = MakeT(w.data(), DType::kF32, {flat});
  Tensor t_down = MakeT(down.data(), DType::kF32, {c.lowrank, flat});
  Tensor t_up = MakeT(up.data(), DType::kF32, {flat, c.lowrank});
  Tensor t_mixed = MakeT(mixed.data(), DType::kF32, {c.T, c.hidden});
  Tensor t_inj = MakeT(inj.data(), DType::kF32, {c.T, c.hc});
  Tensor t_inject_w =
      inject_w.empty() ? Tensor{} : MakeT(inject_w.data(), DType::kF32, {c.hc, flat});

  Qwen4ExpGatedResidualArgs args;
  args.hc_count = c.hc;
  args.hidden_size = c.hidden;
  args.lowrank = c.lowrank;
  args.eps = c.eps;

  const bool combine = c.inject != nullptr;
  vt::Qwen4ExpGatedResidual(q, t_mixed, combine ? &t_inj : nullptr, t_hyper, t_w, t_down,
                            t_up, combine ? &t_inject_w : nullptr, args);

  INFO("case ", c.name, " mixed");
  CHECK(MaxAbsDiff(mixed, std::vector<float>(c.mixed, c.mixed + c.T * c.hidden)) < kTol);
  if (combine) {
    INFO("case ", c.name, " injection_weights");
    CHECK(MaxAbsDiff(inj, std::vector<float>(c.inj_w, c.inj_w + c.T * c.hc)) < kTol);
  }

  // THE STREAM IS READ-ONLY. Upstream returns `hyper_input` RAW and it is the
  // raw stream the write-back adds to, so an op that normalized in place would
  // double-normalize at the second site of every layer — plausible output, wrong
  // model. Asserted byte-for-byte, not within a tolerance.
  INFO("case ", c.name, " hyper unmodified");
  CHECK(std::memcmp(hyper.data(), c.hyper,
                    sizeof(float) * static_cast<size_t>(c.T * flat)) == 0);
}

void RunWriteBackCase(const Case& c) {
  REQUIRE(c.written != nullptr);
  Queue q = CpuQ();
  const int64_t flat = c.hc * c.hidden;
  std::vector<float> hyper(c.hyper, c.hyper + c.T * flat);
  std::vector<float> block_out(c.block_out, c.block_out + c.T * c.hidden);
  std::vector<float> inj(c.inj_w, c.inj_w + c.T * c.hc);

  Tensor t_hyper = MakeT(hyper.data(), DType::kF32, {c.T, flat});
  Tensor t_block = MakeT(block_out.data(), DType::kF32, {c.T, c.hidden});
  Tensor t_inj = MakeT(inj.data(), DType::kF32, {c.T, c.hc});

  Qwen4ExpGatedResidualArgs args;
  args.hc_count = c.hc;
  args.hidden_size = c.hidden;
  args.lowrank = c.lowrank;
  args.eps = c.eps;

  vt::Qwen4ExpGatedResidualWriteBack(q, t_hyper, t_block, t_inj, args);
  INFO("case ", c.name, " write-back");
  CHECK(MaxAbsDiff(hyper, std::vector<float>(c.written, c.written + c.T * flat)) < kTol);
}

}  // namespace

TEST_CASE("vt::Qwen4ExpGatedResidual reproduces the pinned oracle, batched over T") {
  RunReadCase(kCaseA);
  RunReadCase(kCaseB);
  RunReadCase(kCaseD);
}

TEST_CASE("vt::Qwen4ExpGatedResidual: use_combine=false is the mixer arm") {
  // `block_inject_weight is None` (modeling_qwen4_exp.py:966-967) returns
  // `mixed_input` alone. That early return IS `Qwen4ExpTextModel`'s terminal
  // mixer, and `Qwen4ExpTextModel` has NO final RMSNorm after it, so this arm's
  // output is the last thing before `lm_head`.
  RunReadCase(kCaseC);
}

TEST_CASE("vt::Qwen4ExpGatedResidualWriteBack reproduces the pinned oracle") {
  RunWriteBackCase(kCaseA);
  RunWriteBackCase(kCaseB);
  RunWriteBackCase(kCaseD);
}

TEST_CASE("vt::Qwen4ExpGatedResidual: the per-token rows are independent") {
  // A BATCHED kernel that leaked one token's low-rank intermediate into the next
  // still matches a golden whose tokens happen to be similar. Drive case A's
  // three tokens ONE AT A TIME and require the batched answer for each row to be
  // bit-identical to the single-token answer. Bit-identical, not within kTol:
  // the arithmetic per row is the same sequence of operations either way, so any
  // difference at all is cross-row contamination.
  Queue q = CpuQ();
  const Case& c = kCaseA;
  const int64_t flat = c.hc * c.hidden;
  std::vector<float> w = OpGamma(c.norm_w_hf, flat);
  std::vector<float> down(c.down, c.down + c.lowrank * flat);
  std::vector<float> up(c.up, c.up + flat * c.lowrank);
  std::vector<float> inject_w(c.inject, c.inject + c.hc * flat);

  Qwen4ExpGatedResidualArgs args;
  args.hc_count = c.hc;
  args.hidden_size = c.hidden;
  args.lowrank = c.lowrank;
  args.eps = c.eps;

  Tensor t_w = MakeT(w.data(), DType::kF32, {flat});
  Tensor t_down = MakeT(down.data(), DType::kF32, {c.lowrank, flat});
  Tensor t_up = MakeT(up.data(), DType::kF32, {flat, c.lowrank});
  Tensor t_inject_w = MakeT(inject_w.data(), DType::kF32, {c.hc, flat});

  std::vector<float> hyper_all(c.hyper, c.hyper + c.T * flat);
  std::vector<float> mixed_all(static_cast<size_t>(c.T * c.hidden), 0.0f);
  std::vector<float> inj_all(static_cast<size_t>(c.T * c.hc), 0.0f);
  Tensor t_hyper_all = MakeT(hyper_all.data(), DType::kF32, {c.T, flat});
  Tensor t_mixed_all = MakeT(mixed_all.data(), DType::kF32, {c.T, c.hidden});
  Tensor t_inj_all = MakeT(inj_all.data(), DType::kF32, {c.T, c.hc});
  vt::Qwen4ExpGatedResidual(q, t_mixed_all, &t_inj_all, t_hyper_all, t_w, t_down, t_up,
                            &t_inject_w, args);

  for (int64_t t = 0; t < c.T; ++t) {
    std::vector<float> hyper_one(c.hyper + t * flat, c.hyper + (t + 1) * flat);
    std::vector<float> mixed_one(static_cast<size_t>(c.hidden), 0.0f);
    std::vector<float> inj_one(static_cast<size_t>(c.hc), 0.0f);
    Tensor t_hyper_one = MakeT(hyper_one.data(), DType::kF32, {1, flat});
    Tensor t_mixed_one = MakeT(mixed_one.data(), DType::kF32, {1, c.hidden});
    Tensor t_inj_one = MakeT(inj_one.data(), DType::kF32, {1, c.hc});
    vt::Qwen4ExpGatedResidual(q, t_mixed_one, &t_inj_one, t_hyper_one, t_w, t_down, t_up,
                              &t_inject_w, args);
    INFO("token ", t);
    CHECK(std::memcmp(mixed_all.data() + t * c.hidden, mixed_one.data(),
                      sizeof(float) * static_cast<size_t>(c.hidden)) == 0);
    CHECK(std::memcmp(inj_all.data() + t * c.hc, inj_one.data(),
                      sizeof(float) * static_cast<size_t>(c.hc)) == 0);
  }
}

TEST_CASE("vt::Qwen4ExpGatedResidual: bf16 storage rounds ONCE, on the store") {
  // THE CLAIM IS AN IDENTITY, NOT A TOLERANCE. vt's house dtype contract is
  // "widen on load, compute in f32, round once on the store" (`vt::RmsNorm`,
  // `vt::MoeRelu2`). Feed the op inputs that are ALREADY bf16-exact, run it once
  // with f32 tensors and once with bf16 tensors, and the bf16 outputs must be
  // EXACTLY `F32ToBF16` of the f32 outputs — every interior value is identical
  // by construction, so the only admissible difference is the store.
  //
  // This is strictly stronger than a bf16-eps tolerance against the goldens: a
  // kernel that narrowed the ACCUMULATOR, or that rounded the normed stream
  // before the down projection, changes the interior and breaks the identity,
  // while a loose tolerance would absorb both.
  Queue q = CpuQ();
  const Case& c = kCaseA;
  const int64_t flat = c.hc * c.hidden;
  const auto bf16_exact = [](std::vector<float> v) {
    for (float& x : v) x = vt::BF16ToF32(vt::F32ToBF16(x));
    return v;
  };
  const auto to_bf16 = [](const std::vector<float>& src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = vt::F32ToBF16(src[i]);
    return out;
  };
  const std::vector<float> w = bf16_exact(OpGamma(c.norm_w_hf, flat));
  const std::vector<float> hyper = bf16_exact({c.hyper, c.hyper + c.T * flat});
  const std::vector<float> down = bf16_exact({c.down, c.down + c.lowrank * flat});
  const std::vector<float> up = bf16_exact({c.up, c.up + flat * c.lowrank});
  const std::vector<float> inject = bf16_exact({c.inject, c.inject + c.hc * flat});

  Qwen4ExpGatedResidualArgs args;
  args.hc_count = c.hc;
  args.hidden_size = c.hidden;
  args.lowrank = c.lowrank;
  args.eps = c.eps;

  // f32 arm.
  std::vector<float> h32 = hyper, w32 = w, d32 = down, u32 = up, i32 = inject;
  std::vector<float> mixed32(static_cast<size_t>(c.T * c.hidden), 0.0f);
  std::vector<float> inj32(static_cast<size_t>(c.T * c.hc), 0.0f);
  Tensor t_h32 = MakeT(h32.data(), DType::kF32, {c.T, flat});
  Tensor t_w32 = MakeT(w32.data(), DType::kF32, {flat});
  Tensor t_d32 = MakeT(d32.data(), DType::kF32, {c.lowrank, flat});
  Tensor t_u32 = MakeT(u32.data(), DType::kF32, {flat, c.lowrank});
  Tensor t_i32 = MakeT(i32.data(), DType::kF32, {c.hc, flat});
  Tensor t_m32 = MakeT(mixed32.data(), DType::kF32, {c.T, c.hidden});
  Tensor t_j32 = MakeT(inj32.data(), DType::kF32, {c.T, c.hc});
  vt::Qwen4ExpGatedResidual(q, t_m32, &t_j32, t_h32, t_w32, t_d32, t_u32, &t_i32, args);

  // bf16 arm, same values.
  std::vector<uint16_t> hbf = to_bf16(hyper), wbf = to_bf16(w), dbf = to_bf16(down),
                        ubf = to_bf16(up), ibf = to_bf16(inject);
  std::vector<uint16_t> mixedbf(static_cast<size_t>(c.T * c.hidden), 0);
  std::vector<uint16_t> injbf(static_cast<size_t>(c.T * c.hc), 0);
  Tensor t_hbf = MakeT(hbf.data(), DType::kBF16, {c.T, flat});
  Tensor t_wbf = MakeT(wbf.data(), DType::kBF16, {flat});
  Tensor t_dbf = MakeT(dbf.data(), DType::kBF16, {c.lowrank, flat});
  Tensor t_ubf = MakeT(ubf.data(), DType::kBF16, {flat, c.lowrank});
  Tensor t_ibf = MakeT(ibf.data(), DType::kBF16, {c.hc, flat});
  Tensor t_mbf = MakeT(mixedbf.data(), DType::kBF16, {c.T, c.hidden});
  Tensor t_jbf = MakeT(injbf.data(), DType::kBF16, {c.T, c.hc});
  vt::Qwen4ExpGatedResidual(q, t_mbf, &t_jbf, t_hbf, t_wbf, t_dbf, t_ubf, &t_ibf, args);

  for (int64_t i = 0; i < c.T * c.hidden; ++i) {
    INFO("mixed element ", i);
    CHECK(mixedbf[static_cast<size_t>(i)] ==
          vt::F32ToBF16(mixed32[static_cast<size_t>(i)]));
  }
  for (int64_t i = 0; i < c.T * c.hc; ++i) {
    INFO("injection element ", i);
    CHECK(injbf[static_cast<size_t>(i)] == vt::F32ToBF16(inj32[static_cast<size_t>(i)]));
  }
  // The bf16 store must actually LOSE something, or the identity above holds for
  // the uninteresting reason that every value happened to be representable.
  bool any_rounded = false;
  for (int64_t i = 0; i < c.T * c.hidden; ++i) {
    if (vt::BF16ToF32(mixedbf[static_cast<size_t>(i)]) != mixed32[static_cast<size_t>(i)]) {
      any_rounded = true;
    }
  }
  CHECK(any_rounded);
}

TEST_CASE("vt::Qwen4ExpGatedResidual agrees with the host reference at MODEL WIDTH") {
  // The one thing the golden shapes cannot say. `test_qwen4_exp_hc.cpp` measures
  // that its own `kTol = 1e-5` does not survive a rescale to hidden_size 2560 —
  // the oracle itself sits at the same ORDER as 1e-5 against an exact
  // double evaluation of its own algorithm, because torch runs this in fp32 too
  // — so the bound here is RELATIVE, and it is the one the spec derives:
  // `4e-5`, 6.6x the sqrt(K)*u random-walk bound for K = 10240.
  //
  // WHAT THIS COMPARES. The vt op against `vllm::qwen4_exp::GatedResidualForward`,
  // which is a SEPARATE implementation gated against the same oracle at the
  // golden widths. It is an agreement check between two independently written
  // arms at a width neither has been run at, not a second look at one of them.
  Queue q = CpuQ();
  constexpr int64_t H = 2560, HC = 4, R = 320, T = 2;
  constexpr int64_t kFlat = HC * H;
  constexpr float kEps = 1e-6f;
  // Deterministic inputs; a fixed 64-bit LCG so the case is reproducible without
  // depending on any standard-library distribution's implementation.
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  const auto next = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<int32_t>(state >> 33)) / 2147483648.0f;
  };
  std::vector<float> hyper(static_cast<size_t>(T * kFlat));
  for (float& v : hyper) v = next();
  // THE OP TAKES `w_hf`, THE HOST REFERENCE TAKES `1 + w_hf` (#2218). The same
  // numbers reach the same arithmetic either way — the draw below is centred on
  // zero and the fold puts it back on one — but the two arms are handed
  // DIFFERENT parameterizations of it, which is what keeps this an agreement
  // check between two implementations rather than between two spellings.
  std::vector<float> w_hf(static_cast<size_t>(kFlat));
  for (float& v : w_hf) v = 0.1f * next();
  std::vector<float> down(static_cast<size_t>(R * kFlat));
  for (float& v : down) v = 0.02f * next();
  std::vector<float> up(static_cast<size_t>(kFlat * R));
  for (float& v : up) v = 0.02f * next();
  std::vector<float> inject(static_cast<size_t>(HC * kFlat));
  for (float& v : inject) v = 0.02f * next();

  std::vector<float> mixed(static_cast<size_t>(T * H), 0.0f);
  std::vector<float> inj(static_cast<size_t>(T * HC), 0.0f);
  Tensor t_hyper = MakeT(hyper.data(), DType::kF32, {T, kFlat});
  Tensor t_w = MakeT(w_hf.data(), DType::kF32, {kFlat});
  Tensor t_down = MakeT(down.data(), DType::kF32, {R, kFlat});
  Tensor t_up = MakeT(up.data(), DType::kF32, {kFlat, R});
  Tensor t_inject = MakeT(inject.data(), DType::kF32, {HC, kFlat});
  Tensor t_mixed = MakeT(mixed.data(), DType::kF32, {T, H});
  Tensor t_inj = MakeT(inj.data(), DType::kF32, {T, HC});

  Qwen4ExpGatedResidualArgs args;
  args.hc_count = HC;
  args.hidden_size = H;
  args.lowrank = R;
  args.eps = kEps;
  vt::Qwen4ExpGatedResidual(q, t_mixed, &t_inj, t_hyper, t_w, t_down, t_up, &t_inject,
                            args);

  vllm::qwen4_exp::GatedResidualWeights hw;
  hw.hc_norm_weight = vllm::qwen4_exp::HcNormWeightFromHf(w_hf);
  hw.mix_down = down;
  hw.mix_up = up;
  hw.block_inject = inject;
  // The spec's `kRealWidthMixedRel`: relative, because an ABSOLUTE bound at this
  // width tests the accumulator and not the port.
  constexpr double kRealWidthRel = 4e-5;
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> one(hyper.begin() + t * kFlat, hyper.begin() + (t + 1) * kFlat);
    const auto ref = vllm::qwen4_exp::GatedResidualForward(one, hw, HC, H, kEps);
    double worst = 0.0, scale = 0.0;
    for (int64_t h = 0; h < H; ++h) {
      const double a = mixed[static_cast<size_t>(t * H + h)];
      const double b = ref.mixed_input[static_cast<size_t>(h)];
      worst = std::max(worst, std::abs(a - b));
      scale = std::max(scale, std::abs(b));
    }
    INFO("token ", t, " mixed max|diff| ", worst, " scale ", scale);
    CHECK(worst <= kRealWidthRel * std::max(scale, 1e-3));
    double winj = 0.0;
    for (int64_t j = 0; j < HC; ++j) {
      winj = std::max(winj, std::abs(static_cast<double>(inj[static_cast<size_t>(t * HC + j)]) -
                                     static_cast<double>(ref.injection_weights[static_cast<size_t>(j)])));
    }
    INFO("token ", t, " injection max|diff| ", winj);
    CHECK(winj <= kRealWidthRel * 2.0);  // injection_weights live in (0, 2)
  }
}

TEST_CASE("vt::Qwen4ExpGatedResidualWriteBack agrees with the host reference at MODEL WIDTH") {
  Queue q = CpuQ();
  constexpr int64_t H = 2560, HC = 4, T = 2;
  constexpr int64_t kFlat = HC * H;
  uint64_t state = 0xD1B54A32D192ED03ULL;
  const auto next = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<int32_t>(state >> 33)) / 2147483648.0f;
  };
  std::vector<float> hyper(static_cast<size_t>(T * kFlat));
  for (float& v : hyper) v = next();
  std::vector<float> block_out(static_cast<size_t>(T * H));
  for (float& v : block_out) v = next();
  std::vector<float> inj(static_cast<size_t>(T * HC));
  for (float& v : inj) v = 1.0f + 0.5f * next();

  std::vector<float> device = hyper;
  Tensor t_hyper = MakeT(device.data(), DType::kF32, {T, kFlat});
  Tensor t_block = MakeT(block_out.data(), DType::kF32, {T, H});
  Tensor t_inj = MakeT(inj.data(), DType::kF32, {T, HC});
  Qwen4ExpGatedResidualArgs args;
  args.hc_count = HC;
  args.hidden_size = H;
  vt::Qwen4ExpGatedResidualWriteBack(q, t_hyper, t_block, t_inj, args);

  // The write-back is one fused multiply-add per element — no reduction — so the
  // two arms are bit-identical here and the gate says so rather than allowing a
  // tolerance a reordering could hide in.
  for (int64_t t = 0; t < T; ++t) {
    const auto ref = vllm::qwen4_exp::GatedResidualWriteBack(
        std::vector<float>(hyper.begin() + t * kFlat, hyper.begin() + (t + 1) * kFlat),
        std::vector<float>(block_out.begin() + t * H, block_out.begin() + (t + 1) * H),
        std::vector<float>(inj.begin() + t * HC, inj.begin() + (t + 1) * HC), HC, H);
    INFO("token ", t);
    CHECK(std::memcmp(device.data() + t * kFlat, ref.data(),
                      sizeof(float) * static_cast<size_t>(kFlat)) == 0);
  }
}

TEST_CASE("vt::Qwen4ExpGatedResidual: the grouped norm needs a WIDER-THAN-f32 accumulator") {
  // WHY THIS CASE EXISTS AND WHY IT IS SHAPED LIKE THIS. The per-group sum of
  // squares runs over 2560 terms in the real model, and `double ss` -> `float ss`
  // is a one-word edit that changes nothing measurable on ordinary data — a sum
  // of similar-sized positive squares loses about sqrt(N)*u either way. It was
  // mutation M9 of the W5b-2 battery and it SURVIVED every other case in this
  // file. The loss appears only when a partial sum grows past the point where the
  // next term falls under its own ulp: 4096^2 is exactly 2^24, the float integer
  // ceiling, so with one element at 4096 and the rest at 1.0 every following
  // `+1.0f` rounds away and `float ss` misses the tail ENTIRELY.
  //
  // THE PROJECTIONS ARE ZEROED ON PURPOSE. `mixed` is what the op returns, and at
  // flat = 10240 the two f32 projections contribute their own ~1e-6 relative
  // noise, which would sit on top of the signal this case is trying to read. With
  // `mix_down` and `mix_up` both zero the intermediate is exactly 0, `silu(0)` is
  // exactly 0 and every gate is `sigmoid(0)` = 0.5 exactly, so `mixed[h]` is a
  // four-term f32 mean of the normed stream and the ONLY error source left is the
  // reduction under test. That is a shape chosen to isolate a variable, not a
  // shape chosen to pass: the projections are gated by every other case here.
  Queue q = CpuQ();
  constexpr int64_t H = 2560, HC = 4, R = 1, T = 1;
  constexpr int64_t kFlat = HC * H;
  constexpr float kEps = 1e-6f;
  const float dominant[HC] = {4096.0f, 2048.0f, 8192.0f, 1024.0f};
  uint64_t state = 12345;
  const auto next = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<int32_t>(state >> 33)) / 2147483648.0f;
  };
  // `w_hf` is the RAW gamma the op takes (#2218); the double reference below
  // spells the `1 +` itself, IN f32, so both arms describe the same multiplier
  // bit for bit and the only thing this case widens is the reduction it is about.
  std::vector<float> hyper(static_cast<size_t>(kFlat)), w_hf(static_cast<size_t>(kFlat));
  for (int64_t j = 0; j < HC; ++j) {
    for (int64_t d = 0; d < H; ++d) {
      hyper[static_cast<size_t>(j * H + d)] = (d == 0) ? dominant[j] : 1.0f;
      w_hf[static_cast<size_t>(j * H + d)] = 0.5f * next();
    }
  }
  std::vector<float> down(static_cast<size_t>(R * kFlat), 0.0f);
  std::vector<float> up(static_cast<size_t>(kFlat * R), 0.0f);
  std::vector<float> mixed(static_cast<size_t>(T * H), 0.0f);

  Tensor t_hyper = MakeT(hyper.data(), DType::kF32, {T, kFlat});
  Tensor t_w = MakeT(w_hf.data(), DType::kF32, {kFlat});
  Tensor t_down = MakeT(down.data(), DType::kF32, {R, kFlat});
  Tensor t_up = MakeT(up.data(), DType::kF32, {kFlat, R});
  Tensor t_mixed = MakeT(mixed.data(), DType::kF32, {T, H});
  Qwen4ExpGatedResidualArgs args;
  args.hc_count = HC;
  args.hidden_size = H;
  args.lowrank = R;
  args.eps = kEps;
  // The mixer arm: no injection, so nothing but the norm and the collapse runs.
  vt::Qwen4ExpGatedResidual(q, t_mixed, nullptr, t_hyper, t_w, t_down, t_up, nullptr,
                            args);

  // The reference, in full double EXCEPT the gamma fold. `sigmoid(0)` is 0.5
  // exactly in any precision, so the reduction is the only thing left for the
  // widening to isolate -- provided the fold does not quietly widen with it.
  // Upstream folds in f32: `Qwen4ExpTextRMSNorm.forward` is
  // `output * (1.0 + self.weight.float())` (`modeling_qwen4_exp.py:177`), where
  // the Python `1.0` is a weak scalar and the promotion stays fp32, and the
  // kernel mirrors that with `1.0f + LoadF32At(hc_norm_w, ...)`. Folding in
  // double here instead would leave the two arms up to a float ulp apart on the
  // multiplier, and this case's band would absorb the difference silently -- a
  // tolerance covering a dtype gap, which is the shape AGENTS.md "Inherit vLLM
  // defaults" warns a token gate cannot see. So the `+ 1` is spelled `1.0f` and
  // widened AFTERWARDS, matching the kernel exactly.
  std::vector<double> want(static_cast<size_t>(H), 0.0);
  for (int64_t j = 0; j < HC; ++j) {
    double ss = 0.0;
    for (int64_t d = 0; d < H; ++d) {
      const double v = hyper[static_cast<size_t>(j * H + d)];
      ss += v * v;
    }
    const double r = 1.0 / std::sqrt(ss / static_cast<double>(H) + static_cast<double>(kEps));
    for (int64_t d = 0; d < H; ++d) {
      const float w_folded = 1.0f + w_hf[static_cast<size_t>(j * H + d)];
      want[static_cast<size_t>(d)] +=
          0.5 * hyper[static_cast<size_t>(j * H + d)] * r * static_cast<double>(w_folded);
    }
  }
  for (double& v : want) v /= static_cast<double>(HC);

  double worst = 0.0, peak = 0.0;
  for (int64_t d = 0; d < H; ++d) {
    worst = std::max(worst, std::abs(static_cast<double>(mixed[static_cast<size_t>(d)]) -
                                     want[static_cast<size_t>(d)]));
    peak = std::max(peak, std::abs(want[static_cast<size_t>(d)]));
  }
  // MEASURED BAND. Both ends are re-run mutations on exactly the data above, not
  // estimates, and they are recorded in the W5b-2 mutation table in
  // `.agents/specs/qwen4-exp-flash-next.md`:
  //
  //   max|reference|                     3.2895e+01
  //   ours, double accumulator           1.173e-06   (3.6e-08 relative, ~0.6 ulp)
  //   the same kernel with `float ss`    6.702e-04   (571x worse)
  //
  // The bound sits 8.5x above the first and 67x below the second, which is a band
  // rather than a fitted number. A tolerance chosen anywhere inside it says the
  // same thing; one chosen outside it says nothing.
  constexpr double kAccumBound = 1e-5;
  INFO("max|mixed - double reference| ", worst, " peak ", peak, " bound ", kAccumBound);
  CHECK(worst < kAccumBound);
}

TEST_CASE("vt::Qwen4ExpGatedResidual refuses by name") {
  Queue q = CpuQ();
  constexpr int64_t H = 6, HC = 4, R = 5, T = 1;
  constexpr int64_t kFlat = HC * H;
  std::vector<float> hyper(static_cast<size_t>(T * kFlat), 0.5f);
  std::vector<float> w(static_cast<size_t>(kFlat), 1.0f);
  std::vector<float> down(static_cast<size_t>(R * kFlat), 0.1f);
  std::vector<float> up(static_cast<size_t>(kFlat * R), 0.1f);
  std::vector<float> inject(static_cast<size_t>(HC * kFlat), 0.1f);
  std::vector<float> mixed(static_cast<size_t>(T * H), 0.0f);
  std::vector<float> inj(static_cast<size_t>(T * HC), 0.0f);

  Tensor t_hyper = MakeT(hyper.data(), DType::kF32, {T, kFlat});
  Tensor t_w = MakeT(w.data(), DType::kF32, {kFlat});
  Tensor t_down = MakeT(down.data(), DType::kF32, {R, kFlat});
  Tensor t_up = MakeT(up.data(), DType::kF32, {kFlat, R});
  Tensor t_inject = MakeT(inject.data(), DType::kF32, {HC, kFlat});
  Tensor t_mixed = MakeT(mixed.data(), DType::kF32, {T, H});
  Tensor t_inj = MakeT(inj.data(), DType::kF32, {T, HC});

  Qwen4ExpGatedResidualArgs ok;
  ok.hc_count = HC;
  ok.hidden_size = H;
  ok.lowrank = R;
  ok.eps = 1e-6f;

  SUBCASE("hc_count <= 1, the refusal upstream's __post_init__ makes") {
    Qwen4ExpGatedResidualArgs bad = ok;
    bad.hc_count = 1;
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpGatedResidual(q, t_mixed, &t_inj, t_hyper, t_w,
                                                   t_down, t_up, &t_inject, bad),
                         doctest::Contains("hc_count must be > 1"), std::exception);
  }
  SUBCASE("an injection output with no block_inject_weight") {
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpGatedResidual(q, t_mixed, &t_inj, t_hyper, t_w,
                                                   t_down, t_up, nullptr, ok),
                         doctest::Contains("null TOGETHER"), std::exception);
  }
  SUBCASE("a block_inject_weight with nowhere to put the injection") {
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpGatedResidual(q, t_mixed, nullptr, t_hyper, t_w,
                                                   t_down, t_up, &t_inject, ok),
                         doctest::Contains("null TOGETHER"), std::exception);
  }
  SUBCASE("a hyper stream that is not hc_count * hidden_size wide") {
    Tensor narrow = MakeT(hyper.data(), DType::kF32, {T, kFlat - HC});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpGatedResidual(q, t_mixed, &t_inj, narrow, t_w,
                                                   t_down, t_up, &t_inject, ok),
                         doctest::Contains("hyper must be"), std::exception);
  }
  SUBCASE("a mix_up in mix_down's orientation") {
    // [R, flat] where [flat, R] belongs: the single most likely porting slip,
    // because both tensors are square-free and the product still type-checks in
    // a language without shapes.
    Tensor swapped = MakeT(up.data(), DType::kF32, {R, kFlat});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpGatedResidual(q, t_mixed, &t_inj, t_hyper, t_w,
                                                   t_down, swapped, &t_inject, ok),
                         doctest::Contains("input_mix_weight_up"), std::exception);
  }
  SUBCASE("a write-back whose injection does not carry one weight per branch") {
    Tensor wrong = MakeT(inj.data(), DType::kF32, {T, HC - 1});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpGatedResidualWriteBack(q, t_hyper, t_mixed, wrong, ok),
        doctest::Contains("injection must be"), std::exception);
  }
}
