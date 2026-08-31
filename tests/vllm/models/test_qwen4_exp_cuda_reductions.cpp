// Qwen4-Exp (Qwen3.8-Flash-Next) W6-CUDA-B DEVICE-ARM GATE — the CUDA arms of
// the three ops that own a REDUCTION: `vt::Qwen4ExpGatedResidual` (the mixer),
// `vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention`.
// Row MODEL-MM-QWEN4-EXP, campaign issue #1978, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// `vt::RmsNormGroup`'s CUDA arm is the fourth and is gated in
// `tests/vt/test_ops_rms_norm_group_cuda.cpp`, beside its CPU suite, because it
// is a shared `vt::` seam and not a `qwen4_exp` op.
//
// WHAT IS UNDER TEST, AND WHAT IT IS COMPARED AGAINST. Two things, and the
// distinction is the point of the file — the arrangement
// `test_qwen4_exp_cuda.cpp` already states for the W6-CUDA arms:
//
//   1. THE ORACLE. The mixer against `qwen4_exp_hc_goldens.inc` and the two QSA
//      ops against `fixtures/qwen4_exp_qsa_goldens.inc`, both dumped by lifting
//      transformers v5.16.0 VERBATIM by line range and EXECUTING it under torch.
//      The CPU arms are held to the same files, so the two arms answer to ONE
//      oracle instead of to each other.
//   2. THE CPU ARM, on identical inputs. A DEVICE ARM GATED ONLY AGAINST ITSELF
//      WOULD BE WORTHLESS, and the goldens' widths (16-element index heads,
//      6-element hidden) cannot see a defect that needs a warp to express.
//
// ─── THE THREE BOUNDS, AND WHY THEY ARE NOT THE SAME NUMBER ──────────────────
//
// `vt::Qwen4ExpQsaCompress` IS BYTE-IDENTICAL to its CPU arm and is gated by a
// byte comparison, in BOTH arms of `round_intermediates_to_bf16`. It has no
// transcendental, every reduction runs in the host's ascending f32 order, and
// every multiply/add is spelled `__fmul_rn`/`__fadd_rn` against the host's
// pinned `-ffp-contract=off`. **AND THAT MAKES ITS ORACLE GATE TRANSITIVE,
// EXACTLY:** the CPU arm is held to the goldens by
// `test_qwen4_exp_qsa_device.cpp`, the CUDA arm is held to the CPU arm by
// equality, and equality composes. That argument is available only BECAUSE the
// relation is equality; it would be invalid for a tolerance, and it is stated
// rather than assumed.
//
// `vt::Qwen4ExpQsaGatherAttention` has ONE divergence source and it is named:
// `exp`. The CPU arm calls `std::exp` on a float, the device arm calls `expf`,
// and CUDA documents up to 2 ulp for `expf` where glibc's is correctly rounded.
// Every other operation is IEEE-exact or `_rn`, and all four of the CPU arm's
// reduction ORDERS are preserved (the kernel's header enumerates them), so the
// arms are close but NOT byte-identical, and the gate does not pretend
// otherwise: it holds them to a bound DERIVED from each fixture's own inputs
// (see the derivation beside `kExpRelDiff`) and prints the scale-free
// actual/bound ratio, so a future toolkit that moves `exp` is visible as a
// number rather than as a red gate nobody can read.
//
// **AN EARLIER REVISION USED A FITTED `kUlpTol = 1.20e-7` HERE AND IT FAILED ON
// CORRECT KERNELS**, by 112% to 290% of its own budget as the selection grew.
// The derivation replacing it records why; do not reintroduce a constant.
//
// `vt::Qwen4ExpGatedResidual` has TWO, and the second is structural. Its three
// projections go through the shared seam `vt::MatmulBT` — a device GEMM, which
// re-associates the K reduction — where the CPU arm uses its private
// `LinearNoBias`, an ascending f32 walk. That is a deliberate trade the kernel's
// header argues (a hand-written device GEMV beside `vt::MatmulBT` is the
// parallel path AGENTS.md forbids, and it would give the released checkpoint's
// 194 Q8_0 mix weights a second private route), and the COST is this bound.
// `kMixerTol` is therefore the ORACLE's 1e-5 and not one ulp: both arms are held
// to the golden at 1e-5, so their mutual difference cannot usefully be asserted
// tighter than the band they each live in. The number is MEASURED and printed on
// every run.
//
// **NEVER WIDEN A BOUND TO MAKE A CASE PASS.** The defects each bound has to
// separate are orders of magnitude away and this file measures the separations
// rather than asserting them: the mixer's eps probe, the compressor's rope
// position, and the gather's gather-vs-mask observable are all carried below.
//
// ─── SCOPE, HONESTLY ─────────────────────────────────────────────────────────
// ALL SIX `qwen4_exp` OPS PLUS `vt::RmsNormGroup` NOW HAVE CUDA ARMS. What is
// still missing is the block-decoding n-gram gather `vt::Embedding` needs
// (`EmbeddingKernelCuda` refuses a block-quantized table by name), which is
// another wave's file territory. `ModelRegistry::Forward` is all-or-nothing, so
// until that lands NOTHING IN PRODUCTION REACHES THESE KERNELS and their
// reachability from a production entry point is VACUOUS rather than proven. The
// spec's `## Owed` names the remaining blocker and the row that owns the wiring.
// No token claim and no speed claim.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm_test::MaxAbsDiff;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Qwen4ExpGatedResidualArgs;
using vt::Qwen4ExpQsaAttnArgs;
using vt::Qwen4ExpQsaCompressArgs;
using vt::Queue;
using vt::Tensor;

namespace {

#include "fixtures/qwen4_exp_qsa_goldens.inc"  // NOLINT — golden literals
#include "qwen4_exp_hc_goldens.inc"            // NOLINT — golden literals

namespace g = qwen4_exp_qsa_goldens;

constexpr double kAbsFloor = 1e-30;

// ─── THE GATHER'S ARM-VS-ARM BOUND IS DERIVED, NOT FITTED ────────────────────
//
// **THE VALUE THAT STOOD HERE WAS A TOLERANCE FITTED TO ONE FIXTURE AND IT WAS
// WRONG.** It read `constexpr double kUlpTol = 1.20e-7` -- "one f32 ulp
// relative" -- applied as `kAbsFloor + kUlpTol * max|want|`. A fresh review
// measured this kernel against a faithful copy of the CPU body on `thor:gpu0`
// under the project's `-ffp-contract=off` pin and it failed at every shape it
// tried, by a margin that GREW with the selection:
//
//   suite geometry, different data   2.384e-07 vs 2.127e-07   112% of budget
//   |sel| = 202,  head_dim = 32      8.941e-08 vs 4.803e-08   186%
//   |sel| = 2050, head_dim = 256     4.470e-08 vs 1.542e-08   290%   <- released config
//   pairs = 9000                     3.576e-07 vs 2.958e-07   121%
//
// It fails with a CORRECT KERNEL, which is the worst kind of bound: the file's
// own "NEVER WIDEN A BOUND TO MAKE A CASE PASS" would then send the next reader
// hunting a defect that is not there. Two separate errors produced it.
//
// FIRST, ONE ULP WAS THE WRONG NUMBER. The sole divergence source is `expf`,
// and CUDA documents `expf` at **2 ulp**, not 1. A 1-ulp bound on a 2-ulp
// function derives the wrong way round. Perturbing `exp` by +/-1 ulp on the
// committed fixture -- HALF what CUDA permits itself -- already took the old
// bound to 140% of itself.
//
// SECOND, AND WORSE, IT SCALED THE WRONG WAY. `out` is a WEIGHTED AVERAGE of
// the selected value rows, so as the selection grows `max|out|` shrinks toward
// the mean of `v` while the error does not: the error is governed by how far
// the value rows sit FROM that mean. A bound proportional to `max|out|` therefore
// tightens exactly where the true error is unchanged, which is why the overshoot
// runs 112% -> 186% -> 290% as `|sel|` goes 11 -> 202 -> 2050.
//
// ─── THE DERIVATION ──────────────────────────────────────────────────────────
//
//   out[d] = SUM_p w_p v_p[d] / SUM_p w_p ,   w_p = exp(s_p - m)
//
// The two arms compute the same `s_p` and the same `m` bit for bit -- the dot is
// sequential ascending f32 on one thread and the max is exact -- so they differ
// ONLY in `w_p`. Write w_p^cuda = w_p (1 + e_p). Then, to first order in e,
//
//   out'[d] - out[d] = SUM_p (w_p/W) e_p (v_p[d] - out[d])
//   |out'[d] - out[d]| <= max|e| * SUM_p (w_p/W) |v_p[d] - out[d]|
//                      <= max|e| * max_p |v_p[d] - out[d]|
//
// because the weights are a convex combination. THAT is the term the old bound
// got backwards: `max_p |v_p - out|` is a spread, and it does not shrink as the
// selection grows. The second-order term is O(e^2) ~ 6e-14 and is dropped.
//
// The two arms also round their `|sel|` accumulations differently, because they
// are adding operands that already differ. The standard bound for `n` f32
// additions gives each arm at most `n u SUM|terms|` away from the exact sum of
// its own operands, with `u = 2^-24`; dividing the accumulator by `W` turns
// `SUM_p w_p |v_p[d]|` into at most `max_p |v_p[d]|`.
//
// **BOTH accumulations round, not one.** `out = fl(acc) / fl(den)`, and the
// DENOMINATOR is a second `|sel|`-long f32 sum whose error passes through the
// quotient as a further `<= gamma_n |out|` per arm. So the rigorous factor is
// FOUR n u, not two: one `n u` for `acc` and one for `den`, in each of two arms.
// An earlier revision wrote 2, which held empirically with about 25% margin --
// and that is exactly the residue of fitting this derivation exists to remove.
// The analysis gives 4; the fixtures would have accepted 2; 4 is what is used.
// It loosens rather than tightens, and it moves the discrimination window from
// ">= 10 ulp of exp error" to ">= 20", which is still far inside anything a
// legal `expf` can do.
//
// WHY THE ulp -> RELATIVE CONVERSION IS SAFE HERE, which is not obvious: `k`
// ulp is `k 2^-23` relative only when the value is normal and its mantissa is
// near 1, and a subnormal `w_p` would break it. Nothing is subnormal, because
// **`W >= 1` exactly**. `m` is bit-identically one of the `__fmul_rn(dot,
// scale)` values -- the max is exact and both arms compute the same dots -- so
// the argmax row contributes `expf(0.0f)`, which is `1.0f` in both arms and in
// any conforming library. Every other term is non-negative, so the denominator
// is at least 1 and the quotient cannot amplify.
//
// Both terms are computed FROM THE FIXTURE'S OWN INPUTS in `CheckGatherDerived`
// below, per output element. Nothing here is fitted, and the bound follows the
// data instead of the other way round.
//
// `expf`: CUDA documents 2 ulp (CUDA C Programming Guide, single-precision
// intrinsics table). glibc's `expf` is correctly rounded, so <= 0.5 ulp. A ulp
// is at most `2^-23` relative, so the worst relative disagreement is
// `2*2^-23 + 0.5*2^-23`.
constexpr double kUlpRel = 1.0 / 8388608.0;          // 2^-23, one f32 ulp, relative
constexpr double kExpRelDiff = 2.5 * kUlpRel;        // CUDA 2 ulp + glibc 0.5 ulp
constexpr double kUnitRoundoff = 1.0 / 16777216.0;   // u = 2^-24
// The band both arms of the mixer already live in against the torch golden.
constexpr double kMixerTol = 1e-5;
// The bound the CPU compressor is held to against the golden, unchanged here.
constexpr double kCompressL2 = 1e-6;
// The bound the CPU gather is held to against the golden, unchanged here: the
// oracle reduces in torch's order over the PADDED row and we reduce over the
// gathered subset, so this is a summation-order band and not a slack.
constexpr double kGatherL2 = 2e-3;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

// LOUD, because a silent skip on a CPU box is how a device arm goes un-gated for
// a release.
bool SkipNoCuda(const char* what) {
  if (HasCuda()) return false;
  std::printf("[SKIP] no CUDA backend: %s NOT exercised\n", what);
  return true;
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void* raw() { return p_; }
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

std::vector<float> Slice(const float* p, int64_t n) { return std::vector<float>(p, p + n); }

double RelL2(const std::vector<float>& a, const float* b, int64_t n) {
  double num = 0.0, den = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num) / std::max(std::sqrt(den), 1e-30);
}

// max|a-b| with `std::max`'s NaN blindness removed, plus the count of elements
// that are not BITWISE equal. Both are reported, because "within one ulp" and
// "identical" are different findings and this file claims the second wherever it
// can get it.
struct Agreement {
  double worst = 0.0;
  size_t not_bitwise_equal = 0;
};

Agreement Compare(const std::vector<float>& got, const std::vector<float>& want) {
  REQUIRE(got.size() == want.size());
  Agreement a;
  // MaxAbsDiff returns +infinity on ANY non-finite operand and raises its own
  // doctest failure, so an all-NaN device output cannot read as a perfect match
  // here. That defect is issue #449 and this file must not re-introduce it by
  // hand-rolling a reduction.
  a.worst = MaxAbsDiff(got, want);
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) ++a.not_bitwise_equal;
  }
  return a;
}

// BOTH REPORTERS PRINT ON SUCCESS, not only on failure. doctest's INFO/CAPTURE
// are emitted only when an assertion fails, so a passing run of a numeric gate
// says nothing about HOW closely it passed. These numbers are the wave's
// evidence and have to survive a green run.
void CheckBitwise(const std::vector<float>& got, const std::vector<float>& want,
                  const char* what) {
  const Agreement a = Compare(got, want);
  std::printf("[MEASURED] %-48s BYTE gate: %zu/%zu differ, max|diff| = %.9g\n", what,
              a.not_bitwise_equal, want.size(), a.worst);
  INFO(what << ": " << a.not_bitwise_equal << " of " << want.size()
            << " elements differ; max|diff| = " << a.worst);
  CHECK(a.not_bitwise_equal == 0);
}

void CheckWithin(const std::vector<float>& got, const std::vector<float>& want, double rel,
                 const char* what) {
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

// ═════════════════════════════════════════════════════════════════════════════
// vt::Qwen4ExpGatedResidual — the mixer
// ═════════════════════════════════════════════════════════════════════════════

struct HcCase {
  std::string name;
  int64_t hidden, hc, lowrank, T;
  float eps;
  const float *norm_w_hf, *down, *up, *inject, *hyper, *mixed, *inj_w;
};

const HcCase kHcA{"A", 6, 4, 5, 3, 1e-6f, kA_norm_w_hf, kA_down, kA_up,
                  kA_inject, kA_hyper, kA_mixed, kA_inj_w};
const HcCase kHcB{"B", 5, 3, 7, 2, 1e-5f, kB_norm_w_hf, kB_down, kB_up,
                  kB_inject, kB_hyper, kB_mixed, kB_inj_w};
// C is the `use_combine=False` arm — the model-level terminal mixer. Upstream
// returns `mixed_input` alone there and registers NO `block_inject_weight`.
const HcCase kHcC{"C", 6, 4, 5, 2, 1e-6f, kC_norm_w_hf, kC_down, kC_up,
                  nullptr, kC_hyper, kC_mixed, nullptr};
// D is the SMALL-MAGNITUDE case and it is here because A, B and C structurally
// cannot see WHERE eps goes: at their `hyper_scale = 1.7` the mean square is
// O(1) and the two spellings differ by ~5e-7, UNDER the bound, so an eps probe
// run there is a mute switch. At `hyper_scale = 0.01` they separate by ~0.5%.
const HcCase kHcD{"D", 6, 4, 5, 2, 1e-6f, kD_norm_w_hf, kD_down, kD_up,
                  kD_inject, kD_hyper, kD_mixed, kD_inj_w};

struct MixerResult {
  std::vector<float> mixed;
  std::vector<float> injection;
  std::vector<float> hyper_after;  // the stream must come back UNCHANGED
};

MixerResult RunMixer(DeviceType dev, const HcCase& c) {
  const int64_t flat = c.hc * c.hidden;
  const bool combine = c.inject != nullptr;
  std::vector<float> hyper = Slice(c.hyper, c.T * flat);
  std::vector<float> w = Slice(c.norm_w_hf, flat);
  std::vector<float> down = Slice(c.down, c.lowrank * flat);
  std::vector<float> up = Slice(c.up, flat * c.lowrank);
  std::vector<float> inject_w;
  if (combine) inject_w = Slice(c.inject, c.hc * flat);
  MixerResult r;
  r.mixed.assign(static_cast<size_t>(c.T * c.hidden), 0.0f);
  r.injection.assign(static_cast<size_t>(c.T * c.hc), 0.0f);
  r.hyper_after = hyper;

  Qwen4ExpGatedResidualArgs args;
  args.hc_count = c.hc;
  args.hidden_size = c.hidden;
  args.lowrank = c.lowrank;
  args.eps = c.eps;

  if (dev == DeviceType::kCPU) {
    Queue q = CpuQ();
    Tensor t_h = MakeTensor(r.hyper_after.data(), DType::kF32, Cpu(), {c.T, flat});
    Tensor t_w = MakeTensor(w.data(), DType::kF32, Cpu(), {flat});
    Tensor t_d = MakeTensor(down.data(), DType::kF32, Cpu(), {c.lowrank, flat});
    Tensor t_u = MakeTensor(up.data(), DType::kF32, Cpu(), {flat, c.lowrank});
    Tensor t_m = MakeTensor(r.mixed.data(), DType::kF32, Cpu(), {c.T, c.hidden});
    Tensor t_j = MakeTensor(r.injection.data(), DType::kF32, Cpu(), {c.T, c.hc});
    Tensor t_i = combine ? MakeTensor(inject_w.data(), DType::kF32, Cpu(), {c.hc, flat})
                         : Tensor{};
    vt::Qwen4ExpGatedResidual(q, t_m, combine ? &t_j : nullptr, t_h, t_w, t_d, t_u,
                              combine ? &t_i : nullptr, args);
    return r;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_h(b, qg.q, DType::kF32, {c.T, flat}, hyper.data());
  DeviceTensor d_w(b, qg.q, DType::kF32, {flat}, w.data());
  DeviceTensor d_d(b, qg.q, DType::kF32, {c.lowrank, flat}, down.data());
  DeviceTensor d_u(b, qg.q, DType::kF32, {flat, c.lowrank}, up.data());
  DeviceTensor d_m(b, qg.q, DType::kF32, {c.T, c.hidden});
  DeviceTensor d_j(b, qg.q, DType::kF32, {c.T, c.hc});
  DeviceTensor d_i(b, qg.q, DType::kF32, {combine ? c.hc : 1, flat},
                   combine ? inject_w.data() : nullptr);
  vt::Qwen4ExpGatedResidual(qg.q, d_m.tensor(), combine ? &d_j.tensor() : nullptr,
                            d_h.tensor(), d_w.tensor(), d_d.tensor(), d_u.tensor(),
                            combine ? &d_i.tensor() : nullptr, args);
  b.Synchronize(qg.q);
  d_m.Download(qg.q, r.mixed.data());
  d_j.Download(qg.q, r.injection.data());
  d_h.Download(qg.q, r.hyper_after.data());
  return r;
}

// ═════════════════════════════════════════════════════════════════════════════
// The QSA fixture: the indexer runs ON THE CPU so the op under test is isolated
// ═════════════════════════════════════════════════════════════════════════════

struct QsaCase {
  std::string name;
  int64_t seq;
  const float *k_raw, *cos, *sin, *k_norm_w, *q_post, *block_keys;
  const float *attn_q, *attn_k, *attn_v, *attn_out;
};

const QsaCase kSubBudget{"sub_budget",         g::kSubBudgetSeq,     g::kSubBudgetKRaw,
                         g::kSubBudgetCos,     g::kSubBudgetSin,     g::kSubBudgetKNormW,
                         g::kSubBudgetQPost,   g::kSubBudgetBlockKeys, g::kSubBudgetAttnQ,
                         g::kSubBudgetAttnK,   g::kSubBudgetAttnV,   g::kSubBudgetAttnOut};
const QsaCase kOverBudget{"over_budget",        g::kOverBudgetSeq,    g::kOverBudgetKRaw,
                          g::kOverBudgetCos,    g::kOverBudgetSin,    g::kOverBudgetKNormW,
                          g::kOverBudgetQPost,  g::kOverBudgetBlockKeys, g::kOverBudgetAttnQ,
                          g::kOverBudgetAttnK,  g::kOverBudgetAttnV,  g::kOverBudgetAttnOut};

constexpr int64_t kTopk = g::kTokenBudget / g::kCompressRatio;

// The compressor, on either device, over the golden's own raw keys.
std::vector<float> RunCompress(DeviceType dev, const QsaCase& c, bool round_bf16) {
  const int64_t D = g::kIndexHeadDim, CR = g::kCompressRatio, rot = g::kRotaryDim;
  const int64_t complete = (c.seq / CR) * CR;
  const int64_t nb = complete / CR;
  std::vector<float> raw = Slice(c.k_raw, complete * D);
  std::vector<float> knw = Slice(c.k_norm_w, D);
  std::vector<float> cos = Slice(c.cos, c.seq * rot);
  std::vector<float> sin = Slice(c.sin, c.seq * rot);
  std::vector<float> out(static_cast<size_t>(nb * D), 0.0f);
  Qwen4ExpQsaCompressArgs args;
  args.compress_ratio = CR;
  args.rotary_dim = rot;
  args.eps = g::kRmsNormEps;
  args.round_intermediates_to_bf16 = round_bf16;
  if (dev == DeviceType::kCPU) {
    Queue q = CpuQ();
    Tensor t_raw = MakeTensor(raw.data(), DType::kF32, Cpu(), {complete, D});
    Tensor t_knw = MakeTensor(knw.data(), DType::kF32, Cpu(), {D});
    Tensor t_cos = MakeTensor(cos.data(), DType::kF32, Cpu(), {c.seq, rot});
    Tensor t_sin = MakeTensor(sin.data(), DType::kF32, Cpu(), {c.seq, rot});
    Tensor t_out = MakeTensor(out.data(), DType::kF32, Cpu(), {nb, D});
    vt::Qwen4ExpQsaCompress(q, t_out, t_raw, t_knw, t_cos, t_sin, args);
    return out;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_raw(b, qg.q, DType::kF32, {complete, D}, raw.data());
  DeviceTensor d_knw(b, qg.q, DType::kF32, {D}, knw.data());
  DeviceTensor d_cos(b, qg.q, DType::kF32, {c.seq, rot}, cos.data());
  DeviceTensor d_sin(b, qg.q, DType::kF32, {c.seq, rot}, sin.data());
  DeviceTensor d_out(b, qg.q, DType::kF32, {nb, D});
  vt::Qwen4ExpQsaCompress(qg.q, d_out.tensor(), d_raw.tensor(), d_knw.tensor(),
                          d_cos.tensor(), d_sin.tensor(), args);
  b.Synchronize(qg.q);
  d_out.Download(qg.q, out.data());
  return out;
}

// The SELECTION, computed ON THE CPU by the two shared DSA ops, so that the
// gather arm under test is isolated: a difference below is this op's, not the
// indexer's. `weights == 1` and `n_head_scale == 1` collapse
// `DsaIndexerLogits`'s fold to QSA's single `index_head_dim ** -0.5`.
struct Selection {
  std::vector<int32_t> block_ids;  // [T, topk], ASCENDING, -1 padded
  std::vector<int32_t> kv_lens;    // [T]
  // The geometry travels WITH the selection rather than being read from the
  // golden fixture's file-scope constants, so a synthetic shape can differ from
  // it. Every helper below reads these two.
  int64_t topk = kTopk;
  int64_t CR = g::kCompressRatio;
};

Selection RunIndexerCpu(const QsaCase& c) {
  const int64_t D = g::kIndexHeadDim, CR = g::kCompressRatio, rot = g::kRotaryDim;
  const int64_t H = g::kIndexNHeads, T = c.seq;
  const int64_t complete = (T / CR) * CR;
  const int64_t nb = complete / CR;
  Queue q = CpuQ();
  Selection sel;
  sel.kv_lens.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) sel.kv_lens[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);

  std::vector<float> block_keys = RunCompress(DeviceType::kCPU, c, /*round_bf16=*/true);
  std::vector<int32_t> ws(static_cast<size_t>(T), 0), we(static_cast<size_t>(T), 0);
  for (int64_t t = 0; t < T; ++t) {
    we[static_cast<size_t>(t)] = static_cast<int32_t>((t + 1) / CR);
  }
  std::vector<float> ones(static_cast<size_t>(T * H), 1.0f);
  std::vector<float> logits(static_cast<size_t>(T * nb), 0.0f);
  std::vector<float> qi = Slice(c.q_post, T * H * D);
  (void)rot;

  Tensor t_bk = MakeTensor(block_keys.data(), DType::kF32, Cpu(), {nb, D});
  Tensor t_q = MakeTensor(qi.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor t_w = MakeTensor(ones.data(), DType::kF32, Cpu(), {T, H});
  Tensor t_ws = MakeTensor(ws.data(), DType::kI32, Cpu(), {T});
  Tensor t_we = MakeTensor(we.data(), DType::kI32, Cpu(), {T});
  Tensor t_lg = MakeTensor(logits.data(), DType::kF32, Cpu(), {T, nb});
  vt::DsaIndexerLogitsArgs largs;
  largs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  largs.n_head_scale = 1.0f;
  vt::DsaIndexerLogits(q, t_lg, t_q, t_bk, t_w, t_ws, t_we, largs);

  sel.block_ids.assign(static_cast<size_t>(T * kTopk), -1);
  std::vector<int32_t> counts(static_cast<size_t>(T), 0);
  Tensor t_ids = MakeTensor(sel.block_ids.data(), DType::kI32, Cpu(), {T, kTopk});
  Tensor t_cnt = MakeTensor(counts.data(), DType::kI32, Cpu(), {T});
  vt::DsaTopkSelect(q, t_ids, t_cnt, t_lg, t_ws, t_we);
  return sel;
}

struct GatherResult {
  std::vector<float> out;
  int64_t keys_visited = -1;
};

// The gather, on either device, over a CONTIGUOUS [max_kv, Hkv, Dh] cache.
GatherResult RunGather(DeviceType dev, const std::vector<float>& qa,
                       const std::vector<float>& ka, const std::vector<float>& va,
                       const Selection& sel, int64_t T, int64_t HQ, int64_t HKV, int64_t DH,
                       int64_t max_kv, bool want_counter) {
  GatherResult r;
  r.out.assign(static_cast<size_t>(T * HQ * DH), 0.0f);
  Qwen4ExpQsaAttnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(DH));
  args.compress_ratio = sel.CR;
  if (want_counter) args.keys_visited = &r.keys_visited;
  std::vector<float> qc = qa, kc = ka, vc = va;
  std::vector<int32_t> ids = sel.block_ids, lens = sel.kv_lens;
  const int64_t topk = sel.topk;
  if (dev == DeviceType::kCPU) {
    Queue q = CpuQ();
    Tensor t_q = MakeTensor(qc.data(), DType::kF32, Cpu(), {T, HQ, DH});
    Tensor t_k = MakeTensor(kc.data(), DType::kF32, Cpu(), {max_kv, HKV, DH});
    Tensor t_v = MakeTensor(vc.data(), DType::kF32, Cpu(), {max_kv, HKV, DH});
    Tensor t_i = MakeTensor(ids.data(), DType::kI32, Cpu(), {T, topk});
    Tensor t_l = MakeTensor(lens.data(), DType::kI32, Cpu(), {T});
    Tensor t_o = MakeTensor(r.out.data(), DType::kF32, Cpu(), {T, HQ, DH});
    vt::Qwen4ExpQsaGatherAttention(q, t_o, t_q, t_k, t_v, t_i, t_l, args);
    return r;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_q(b, qg.q, DType::kF32, {T, HQ, DH}, qc.data());
  DeviceTensor d_k(b, qg.q, DType::kF32, {max_kv, HKV, DH}, kc.data());
  DeviceTensor d_v(b, qg.q, DType::kF32, {max_kv, HKV, DH}, vc.data());
  DeviceTensor d_i(b, qg.q, DType::kI32, {T, topk}, ids.data());
  DeviceTensor d_l(b, qg.q, DType::kI32, {T}, lens.data());
  DeviceTensor d_o(b, qg.q, DType::kF32, {T, HQ, DH});
  vt::Qwen4ExpQsaGatherAttention(qg.q, d_o.tensor(), d_q.tensor(), d_k.tensor(),
                                 d_v.tensor(), d_i.tensor(), d_l.tensor(), args);
  b.Synchronize(qg.q);
  d_o.Download(qg.q, r.out.data());
  return r;
}

// The HOST expansion of a device selection, used ONLY to derive the EXPECTED
// read count and the expected READ SET. It never touches the kernel's counter.
std::vector<int64_t> ExpandHost(const Selection& sel, int64_t t, int64_t kv_len) {
  const int64_t CR = sel.CR;
  const int64_t complete = kv_len / CR;
  std::vector<int64_t> out;
  for (int64_t j = 0; j < sel.topk; ++j) {
    const int64_t b = sel.block_ids[static_cast<size_t>(t * sel.topk + j)];
    if (b < 0) break;
    for (int64_t i = 0; i < CR; ++i) out.push_back(b * CR + i);
  }
  for (int64_t p = complete * CR; p < kv_len; ++p) out.push_back(p);
  return out;
}

// THE DERIVED ARM-VS-ARM GATE for the gather. The bound is computed per output
// element FROM THE FIXTURE'S OWN INPUTS, exactly as the derivation above states:
//
//   bound[t,h,d] = kExpRelDiff * max_p |v_p[d] - out[d]|      (the exp term)
//                + 4 * |sel| * u * max_p |v_p[d]|             (the accumulation term)
//
// THE ACCUMULATION TERM EARNS ITS PLACE AGAINST A WORSE `expf`, NOT AGAINST
// THIS ONE. Measured on `thor:gpu0`, the exp term ALONE suffices at every shape
// (ratios 0.049 to 0.87) because that device's `expf` is far inside its 2 ulp
// budget. Against a legal-but-worse `expf` the exp term alone is exceeded by up
// to 22.8x, and the accumulation term is what absorbs it. Dropping it because
// "it never binds here" would fit the bound to one device.
//
// The first term does NOT shrink as the selection grows, which is the property
// the old fitted tolerance got backwards. The second grows with `|sel|`, which
// is the honest cost of comparing two long f32 accumulations that started from
// slightly different addends.
//
// The WORST RATIO is printed on every run, pass or fail. That number, not the
// bound, is what a reader should watch: it is scale-free, so a change in the
// kernel or in the toolchain's `expf` moves it visibly while the bound follows
// the fixture.
void CheckGatherDerived(const std::vector<float>& got, const std::vector<float>& want,
                        const std::vector<float>& va, const Selection& sel, int64_t T,
                        int64_t HQ, int64_t HKV, int64_t DH, const char* what) {
  REQUIRE(got.size() == want.size());
  const int64_t groups = HQ / HKV;
  double worst_ratio = 0.0, worst_abs = 0.0, at_bound = 0.0;
  size_t notbit = 0, n_over = 0;
  int64_t worst_sel = 0;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t kv_len = sel.kv_lens[static_cast<size_t>(t)];
    const std::vector<int64_t> rows = ExpandHost(sel, t, kv_len);
    const double n = static_cast<double>(rows.size());
    for (int64_t h = 0; h < HQ; ++h) {
      const int64_t kvh = h / groups;
      for (int64_t d = 0; d < DH; ++d) {
        const size_t o = static_cast<size_t>((t * HQ + h) * DH + d);
        const double ref = want[o];
        double spread = 0.0, mag = 0.0;
        for (int64_t pr : rows) {
          const double vv = va[static_cast<size_t>((pr * HKV + kvh) * DH + d)];
          spread = std::max(spread, std::fabs(vv - ref));
          mag = std::max(mag, std::fabs(vv));
        }
        const double bound =
            kAbsFloor + kExpRelDiff * spread + 4.0 * n * kUnitRoundoff * mag;
        const double diff = std::fabs(static_cast<double>(got[o]) - ref);
        if (std::memcmp(&got[o], &want[o], sizeof(float)) != 0) ++notbit;
        const double ratio = diff / bound;
        if (ratio > worst_ratio) {
          worst_ratio = ratio;
          worst_abs = diff;
          at_bound = bound;
          worst_sel = static_cast<int64_t>(rows.size());
        }
        if (diff > bound) ++n_over;
      }
    }
  }
  std::printf("[MEASURED] %-44s worst ratio = %.4f (|diff| %.9g vs derived bound %.9g at "
              "|sel|=%lld); over = %zu/%zu; not-bitwise-equal = %zu/%zu\n",
              what, worst_ratio, worst_abs, at_bound, static_cast<long long>(worst_sel),
              n_over, want.size(), notbit, want.size());
  INFO(what << ": worst actual/derived-bound ratio " << worst_ratio << " (" << n_over
            << " elements over bound); the bound is computed from the fixture, not fitted");
  CHECK(n_over == 0);
}

// ─── A SYNTHETIC FIXTURE, because every scale loop in the golden ones runs ONCE
//
// The committed QSA goldens are seq 11 and 23. That makes `|sel|` at most 11
// against a `kSelTile` of 32, and `T*HQ` at most 92 against a 4096-block grid,
// so the gather's TILE loop and its GRID-STRIDE loop each take exactly one
// iteration in every case above. A fresh review proved that is not a
// hypothetical gap: deleting the cross-tile denominator carry
// (`cuda_qwen4_exp_qsa.cu`, the `float denom = s_denom;` / `s_denom = denom;`
// pair) leaves the output BYTE-FOR-BYTE identical at one tile, and collapsing
// the grid stride to `blockIdx.x` leaves it byte-identical at one grid trip.
// Both mutations were invisible to the entire committed suite.
//
// These fixtures exist to cross those two boundaries. They carry no oracle and
// they do not need one: the claim is CUDA-vs-CPU agreement under the derived
// bound, and the CPU arm is gated against the transformers goldens elsewhere.
struct Synth {
  int64_t T, HQ, HKV, DH, max_kv;
  std::vector<float> q, k, v;
  Selection sel;
};

Synth MakeSynth(int64_t T, int64_t HQ, int64_t HKV, int64_t DH, int64_t kv_len, int64_t CR,
                int64_t topk, uint32_t seed) {
  Synth y;
  y.T = T; y.HQ = HQ; y.HKV = HKV; y.DH = DH; y.max_kv = kv_len;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  auto fill = [&](std::vector<float>& out, size_t n) {
    out.resize(n);
    for (auto& x : out) x = dist(rng);
  };
  fill(y.q, static_cast<size_t>(T * HQ * DH));
  fill(y.k, static_cast<size_t>(kv_len * HKV * DH));
  fill(y.v, static_cast<size_t>(kv_len * HKV * DH));
  y.sel.topk = topk;
  y.sel.CR = CR;
  y.sel.kv_lens.assign(static_cast<size_t>(T), static_cast<int32_t>(kv_len));
  y.sel.block_ids.assign(static_cast<size_t>(T * topk), -1);
  const int64_t complete = kv_len / CR;
  const int64_t nsel = std::min<int64_t>(complete, topk);
  for (int64_t t = 0; t < T; ++t) {
    // ASCENDING and inside the complete-block count, which is what the op
    // requires; taking the FIRST `nsel` blocks keeps the ragged tail live too.
    for (int64_t j = 0; j < nsel; ++j) {
      y.sel.block_ids[static_cast<size_t>(t * topk + j)] = static_cast<int32_t>(j);
    }
  }
  return y;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
// vt::Qwen4ExpGatedResidual
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("vt::Qwen4ExpGatedResidual CUDA reproduces the pinned oracle") {
  if (SkipNoCuda("vt::Qwen4ExpGatedResidual CUDA arm vs the transformers oracle")) return;
  for (const HcCase* c : {&kHcA, &kHcB, &kHcC, &kHcD}) {
    INFO("case ", c->name);
    const MixerResult gpu = RunMixer(DeviceType::kCUDA, *c);
    const std::vector<float> want_mixed = Slice(c->mixed, c->T * c->hidden);
    const double d = MaxAbsDiff(gpu.mixed, want_mixed);
    std::printf("[MEASURED] hc mixer case %-4s vs ORACLE mixed max|diff| = %.9g bound %g\n",
                c->name.c_str(), d, kMixerTol);
    CHECK(d < kMixerTol);
    if (c->inj_w != nullptr) {
      const std::vector<float> want_inj = Slice(c->inj_w, c->T * c->hc);
      const double dj = MaxAbsDiff(gpu.injection, want_inj);
      std::printf("[MEASURED] hc mixer case %-4s vs ORACLE inject max|diff| = %.9g\n",
                  c->name.c_str(), dj);
      CHECK(dj < kMixerTol);
    }
    // THE STREAM IS READ-ONLY. Upstream returns `hyper_input` RAW and it is the
    // raw stream the write-back adds to, so an arm that normalized in place
    // would double-normalize every layer and still look plausible.
    CheckBitwise(gpu.hyper_after, Slice(c->hyper, c->T * c->hc * c->hidden),
                 ("hc mixer case " + c->name + " stream unchanged").c_str());
  }
}

TEST_CASE("vt::Qwen4ExpGatedResidual CUDA agrees with the CPU arm inside the oracle band") {
  if (SkipNoCuda("vt::Qwen4ExpGatedResidual CUDA vs CPU")) return;
  // NOT a byte gate, and the header says exactly why: the three projections go
  // through `vt::MatmulBT`, a device GEMM that re-associates the K reduction,
  // where the CPU arm walks it ascending in f32. The number is MEASURED and
  // printed; a change in it is a finding whether or not it crosses the bound.
  for (const HcCase* c : {&kHcA, &kHcB, &kHcC, &kHcD}) {
    INFO("case ", c->name);
    const MixerResult gpu = RunMixer(DeviceType::kCUDA, *c);
    const MixerResult cpu = RunMixer(DeviceType::kCPU, *c);
    CheckWithin(gpu.mixed, cpu.mixed, kMixerTol,
                ("hc mixer case " + c->name + " mixed CUDA-vs-CPU").c_str());
    if (c->inj_w != nullptr) {
      CheckWithin(gpu.injection, cpu.injection, kMixerTol,
                  ("hc mixer case " + c->name + " inject CUDA-vs-CPU").c_str());
    }
  }
}

TEST_CASE("vt::Qwen4ExpGatedResidual CUDA: eps is INSIDE the rsqrt, probed where visible") {
  if (SkipNoCuda("vt::Qwen4ExpGatedResidual CUDA eps probe")) return;
  // Case D is the only one that can see this: at A/B/C's `hyper_scale = 1.7` the
  // mean square is O(1) and moving eps changes the answer by ~5e-7, UNDER the
  // bound. Running the probe there is a mute switch, and mutation M7 of the
  // W5b-2 battery SURVIVED all three and reds this one.
  HcCase moved = kHcD;
  moved.eps = 1.0f;  // an eps this large cannot hide anywhere
  const MixerResult ok = RunMixer(DeviceType::kCUDA, kHcD);
  const MixerResult bad = RunMixer(DeviceType::kCUDA, moved);
  double sep = 0.0;
  for (size_t i = 0; i < ok.mixed.size(); ++i) {
    sep = std::max(sep, std::fabs(static_cast<double>(ok.mixed[i]) - bad.mixed[i]));
  }
  std::printf("[MEASURED] hc mixer CUDA eps separation (case D) = %.9g\n", sep);
  CHECK(sep > kMixerTol);  // `args.eps` is READ on the device
  CHECK(MaxAbsDiff(ok.mixed, Slice(kHcD.mixed, kHcD.T * kHcD.hidden)) < kMixerTol);
}

// ═════════════════════════════════════════════════════════════════════════════
// vt::Qwen4ExpQsaCompress
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("vt::Qwen4ExpGatedResidual CUDA: a Q8_0 mix weight ENTERS the block branch") {
  if (SkipNoCuda("vt::Qwen4ExpGatedResidual CUDA block-dtype projection")) return;
  // **THE BRANCH THIS WAVE ARGUES FOR WAS NEVER ENTERED ON A DEVICE.**
  // `cuda_qwen4_exp.cu` routes the three projections through `vt::MatmulBT`
  // rather than a hand-written GEMV, and its stated reason is the released
  // checkpoint's 194 Q8_0 hyper-connection mix weights (W5p). Every golden case
  // in this file uses f32 weights, so `MatmulBT`'s `IsBlockQuant(b.dtype)`
  // dispatch to `kMatmulBTQuant` never fired on CUDA and the argument was
  // untested. The CPU sibling gates both directions (W5p M2/M3); this is the
  // device half.
  //
  // The weights are EXACT in Q8_0 -- codes in [-127,127] against f16-exact
  // power-of-two scales, so `dequant(quant(w)) == w` bit for bit -- which is
  // what makes the two arms comparable at all. The scales cycle over four
  // values so a kernel reading block 0's scale for a whole row is separated.
  constexpr int64_t hc = 4, H = 8, R = 5, T = 3;
  constexpr int64_t flat = hc * H;
  struct Lcg {
    uint32_t s;
    uint32_t Next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    int Code() { return static_cast<int>(Next() % 255u) - 127; }
  };
  auto exact_q8 = [](int64_t n, int64_t k, uint32_t seed,
                     std::vector<float>* f32, std::vector<uint8_t>* blocks) {
    REQUIRE(k % 32 == 0);
    static const float kScales[4] = {1.0f / 256.0f, 1.0f / 512.0f, 1.0f / 1024.0f,
                                     1.0f / 2048.0f};
    const int64_t nb = n * (k / 32);
    f32->resize(static_cast<size_t>(n * k));
    blocks->assign(static_cast<size_t>(nb) * 34, 0);
    Lcg rng{seed};
    for (int64_t b = 0; b < nb; ++b) {
      const float d = kScales[b % 4];
      const uint16_t half = vt::F32ToF16(d);
      REQUIRE(vt::F16ToF32(half) == d);  // the scale survives the f16 store EXACTLY
      std::memcpy(blocks->data() + static_cast<size_t>(b) * 34, &half, 2);
      for (int64_t i = 0; i < 32; ++i) {
        const int code = rng.Code();
        (*blocks)[static_cast<size_t>(b) * 34 + 2 + static_cast<size_t>(i)] =
            static_cast<uint8_t>(static_cast<int8_t>(code));
        (*f32)[static_cast<size_t>(b * 32 + i)] = d * static_cast<float>(code);
      }
    }
  };
  std::vector<float> down_f, up_f, inj_f;
  std::vector<uint8_t> down_q, inj_q;
  exact_q8(R, flat, 11u, &down_f, &down_q);
  // `mix_up` is [flat, R] and R = 5 is not a multiple of 32, so it CANNOT be
  // Q8_0 -- a Q8_0 row is a whole number of 32-element blocks. Only the two
  // operands whose inner dim is block-aligned are quantized, which is exactly
  // the split the loader produces on the released file.
  up_f.assign(static_cast<size_t>(flat * R), 0.0f);
  {
    Lcg rng{5u};
    for (auto& x : up_f) x = static_cast<float>(rng.Code()) / 512.0f;
  }
  exact_q8(hc, flat, 13u, &inj_f, &inj_q);

  std::vector<float> hyper(static_cast<size_t>(T * flat));
  std::vector<float> gamma(static_cast<size_t>(flat));
  {
    Lcg rng{7u};
    for (auto& x : hyper) x = static_cast<float>(rng.Code()) / 64.0f;
    for (auto& x : gamma) x = static_cast<float>(rng.Code()) / 256.0f;
  }

  Qwen4ExpGatedResidualArgs args;
  args.hc_count = hc;
  args.hidden_size = H;
  args.lowrank = R;
  args.eps = 1e-6f;

  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_h(b, qg.q, DType::kF32, {T, flat}, hyper.data());
  DeviceTensor d_w(b, qg.q, DType::kF32, {flat}, gamma.data());
  DeviceTensor d_up(b, qg.q, DType::kF32, {flat, R}, up_f.data());
  DeviceTensor d_m(b, qg.q, DType::kF32, {T, H});
  DeviceTensor d_j(b, qg.q, DType::kF32, {T, hc});

  // ── the f32 reference arm, on the SAME device ──────────────────────────────
  DeviceTensor d_df(b, qg.q, DType::kF32, {R, flat}, down_f.data());
  DeviceTensor d_if(b, qg.q, DType::kF32, {hc, flat}, inj_f.data());
  vt::Qwen4ExpGatedResidual(qg.q, d_m.tensor(), &d_j.tensor(), d_h.tensor(), d_w.tensor(),
                            d_df.tensor(), d_up.tensor(), &d_if.tensor(), args);
  b.Synchronize(qg.q);
  std::vector<float> mixed_f32(static_cast<size_t>(T * H), 0.0f);
  std::vector<float> inj_out_f32(static_cast<size_t>(T * hc), 0.0f);
  d_m.Download(qg.q, mixed_f32.data());
  d_j.Download(qg.q, inj_out_f32.data());

  // ── the Q8_0 arm. `DeviceTensor` cannot size a block dtype -- `vt::SizeOf`
  // ── refuses one by name -- so the payload is allocated by BYTE COUNT and the
  // ── tensor built over it with the shape in ELEMENTS, which is the contract
  // ── `vt::MatmulBTQuant` states.
  void* p_dq = b.Alloc(down_q.size());
  void* p_iq = b.Alloc(inj_q.size());
  b.Copy(qg.q, p_dq, down_q.data(), down_q.size());
  b.Copy(qg.q, p_iq, inj_q.data(), inj_q.size());
  Tensor t_dq = MakeTensor(p_dq, DType::kQ8_0, Gpu(), {R, flat});
  Tensor t_iq = MakeTensor(p_iq, DType::kQ8_0, Gpu(), {hc, flat});

  std::vector<float> mixed_q8(static_cast<size_t>(T * H), 0.0f);
  std::string refusal;
  try {
    vt::Qwen4ExpGatedResidual(qg.q, d_m.tensor(), &d_j.tensor(), d_h.tensor(), d_w.tensor(),
                              t_dq, d_up.tensor(), &t_iq, args);
    b.Synchronize(qg.q);
    d_m.Download(qg.q, mixed_q8.data());
  } catch (const std::exception& e) {
    refusal = e.what();
  }
  b.Free(p_dq);
  b.Free(p_iq);

  // ── the SAME quantized route on the CPU, which is what this case compares ──
  // **THE FIRST VERSION OF THIS CASE COMPARED THE WRONG PAIR AND FAILED ON A
  // CORRECT KERNEL, WHICH IS THE F1 MISTAKE IN A SECOND PLACE.** It asserted
  // `Q8_0 vs f32 on CUDA` against `kMixerTol`, reasoning that the weights are
  // exact in Q8_0 so the two arms compute the same function. That reasoning is
  // wrong: `kMatmulBTQuant` also QUANTIZES THE ACTIVATION to the weight's
  // `vec_dot_type`, and `normed` is not exact in Q8_0. The two routes therefore
  // compute genuinely different numbers, and on the device the gap measured
  // 7.549e-05 against a 1e-05 bound -- a real difference held to a bound derived
  // for something else.
  //
  // The comparison that actually tests THIS WAVE'S KERNEL is the same route on
  // both devices: CUDA Q8_0 against CPU Q8_0. Both quantize the same activation
  // the same way, so a disagreement is the device arm's and nothing else. The
  // f32 gap is still computed and PRINTED, because it measures the activation
  // quantization this architecture accepts on the released checkpoint, but it is
  // not asserted against the oracle band.
  std::vector<float> cpu_q8(static_cast<size_t>(T * H), 0.0f);
  std::string cpu_refusal;
  {
    Queue q = CpuQ();
    std::vector<float> h2 = hyper, g2 = gamma, u2 = up_f;
    std::vector<uint8_t> dq2 = down_q, iq2 = inj_q;
    std::vector<float> j2(static_cast<size_t>(T * hc), 0.0f);
    Tensor t_h = MakeTensor(h2.data(), DType::kF32, Cpu(), {T, flat});
    Tensor t_w = MakeTensor(g2.data(), DType::kF32, Cpu(), {flat});
    Tensor t_u = MakeTensor(u2.data(), DType::kF32, Cpu(), {flat, R});
    Tensor t_m = MakeTensor(cpu_q8.data(), DType::kF32, Cpu(), {T, H});
    Tensor t_j = MakeTensor(j2.data(), DType::kF32, Cpu(), {T, hc});
    Tensor t_d = MakeTensor(dq2.data(), DType::kQ8_0, Cpu(), {R, flat});
    Tensor t_i = MakeTensor(iq2.data(), DType::kQ8_0, Cpu(), {hc, flat});
    try {
      vt::Qwen4ExpGatedResidual(q, t_m, &t_j, t_h, t_w, t_d, t_u, &t_i, args);
    } catch (const std::exception& e) {
      cpu_refusal = e.what();
    }
  }

  // WHICHEVER ARM THE DEVICE TAKES IS GATED, and which one it took is PRINTED,
  // because that is a property of this build's keep-quant support and not of
  // this wave.
  if (!refusal.empty()) {
    std::printf("[MEASURED] hc mixer Q8_0 on CUDA: REFUSED BY NAME -- %.140s\n",
                refusal.c_str());
    INFO("refusal: " << refusal);
    // A refusal must NAME something. An empty or generic message would leave a
    // caller with no idea which operand or which support is missing.
    CHECK(refusal.size() > 20);
    // **IT MUST NAME THE KEEP-QUANT GAP SPECIFICALLY.** An earlier revision
    // accepted any message containing `q8_0`, `quant` or `matmul`, which a
    // SHAPE or DTYPE error thrown BEFORE the block branch also satisfies -- so
    // the case could have passed having never entered the branch it exists to
    // test. `matmul_bt_quant` is the entry point the block dtype dispatches to,
    // and only a refusal from inside it proves the branch was taken.
    const bool from_the_quant_gemm = refusal.find("matmul_bt_quant") != std::string::npos;
    INFO("the refusal must come from the quantized GEMM, not from a shape check");
    CHECK(from_the_quant_gemm);
  } else {
    // The device ENTERED the block branch and returned, which is the fact this
    // case exists to establish.
    std::printf("[MEASURED] hc mixer Q8_0 on CUDA: block branch ENTERED, no refusal\n");
    REQUIRE(cpu_refusal.empty());  // else the two arms are not comparable at all
    const double same_route = MaxAbsDiff(mixed_q8, cpu_q8);
    const double vs_f32 = MaxAbsDiff(mixed_q8, mixed_f32);
    std::printf("[MEASURED] hc mixer Q8_0 CUDA-vs-CPU (same route) max|diff| = %.9g "
                "bound %g\n", same_route, kMixerTol);
    std::printf("[MEASURED] hc mixer Q8_0-vs-f32 on CUDA max|diff| = %.9g (ACTIVATION "
                "quantization, not the device arm; reported, not asserted)\n", vs_f32);
    CHECK(same_route < kMixerTol);
    // And the quantized route must actually DIFFER from the f32 one, or the
    // block branch was not really taken and this case proves nothing.
    CHECK(vs_f32 > 0.0);
  }
}

TEST_CASE("vt::Qwen4ExpQsaCompress CUDA matches the oracle AND the CPU arm BITWISE") {
  if (SkipNoCuda("vt::Qwen4ExpQsaCompress CUDA arm")) return;
  const int64_t D = g::kIndexHeadDim, CR = g::kCompressRatio;
  for (const QsaCase* c : {&kSubBudget, &kOverBudget}) {
    INFO("case ", c->name);
    const int64_t nb = (c->seq / CR);
    for (bool round : {true, false}) {
      CAPTURE(round);
      const std::vector<float> gpu = RunCompress(DeviceType::kCUDA, *c, round);
      const std::vector<float> cpu = RunCompress(DeviceType::kCPU, *c, round);
      // BOTH arms of `round_intermediates_to_bf16`. The flag is load-bearing —
      // the mean pool is the one place a bf16 round trip changes which four raw
      // keys a state can represent — so an arm gated in one setting only is
      // gated in the setting the goldens happen to use.
      CheckBitwise(gpu, cpu,
                   ("qsa_compress " + c->name + (round ? " bf16-round" : " f32")).c_str());
      if (round) {
        // The oracle bound the CPU arm already answers to, unchanged. Byte
        // identity above makes this transitive.
        const double l2 = RelL2(gpu, c->block_keys, nb * D);
        std::printf("[MEASURED] qsa_compress %-12s vs ORACLE relL2 = %.9g  bound = %g\n",
                    c->name.c_str(), l2, kCompressL2);
        CHECK(l2 < kCompressL2);
      }
    }
  }
}

TEST_CASE("vt::Qwen4ExpQsaCompress CUDA: the GRID STRIDE takes more than one trip") {
  if (SkipNoCuda("vt::Qwen4ExpQsaCompress CUDA grid stride")) return;
  // The compressor caps its grid at 4096 blocks and walks `b += gridDim.x`. Both
  // golden fixtures have `nb` of 2 and 5, so that loop ran ONCE and a collapsed
  // stride would have been invisible. 4100 blocks makes the second trip
  // mandatory, and the last 4 blocks are only written by it.
  const int64_t D = g::kIndexHeadDim, CR = g::kCompressRatio, rot = g::kRotaryDim;
  constexpr int64_t kNb = 4100;
  const int64_t num_keys = kNb * CR;
  CHECK(kNb > 4096);  // the fixture must cross the cap
  std::mt19937 rng(31337u);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  auto fill = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
  };
  const std::vector<float> raw = fill(static_cast<size_t>(num_keys * D));
  const std::vector<float> knw = fill(static_cast<size_t>(D));
  const std::vector<float> cos = fill(static_cast<size_t>(num_keys * rot));
  const std::vector<float> sin = fill(static_cast<size_t>(num_keys * rot));

  Qwen4ExpQsaCompressArgs args;
  args.compress_ratio = CR;
  args.rotary_dim = rot;
  args.eps = g::kRmsNormEps;
  args.round_intermediates_to_bf16 = true;

  std::vector<float> cpu(static_cast<size_t>(kNb * D), 0.0f);
  {
    Queue q = CpuQ();
    std::vector<float> r = raw, w = knw, c = cos, si = sin;
    Tensor t_raw = MakeTensor(r.data(), DType::kF32, Cpu(), {num_keys, D});
    Tensor t_knw = MakeTensor(w.data(), DType::kF32, Cpu(), {D});
    Tensor t_cos = MakeTensor(c.data(), DType::kF32, Cpu(), {num_keys, rot});
    Tensor t_sin = MakeTensor(si.data(), DType::kF32, Cpu(), {num_keys, rot});
    Tensor t_out = MakeTensor(cpu.data(), DType::kF32, Cpu(), {kNb, D});
    vt::Qwen4ExpQsaCompress(q, t_out, t_raw, t_knw, t_cos, t_sin, args);
  }
  std::vector<float> gpu(static_cast<size_t>(kNb * D), 0.0f);
  {
    Backend& b = vt::GetBackend(DeviceType::kCUDA);
    QueueGuard qg(b);
    DeviceTensor d_raw(b, qg.q, DType::kF32, {num_keys, D}, raw.data());
    DeviceTensor d_knw(b, qg.q, DType::kF32, {D}, knw.data());
    DeviceTensor d_cos(b, qg.q, DType::kF32, {num_keys, rot}, cos.data());
    DeviceTensor d_sin(b, qg.q, DType::kF32, {num_keys, rot}, sin.data());
    DeviceTensor d_out(b, qg.q, DType::kF32, {kNb, D});
    vt::Qwen4ExpQsaCompress(qg.q, d_out.tensor(), d_raw.tensor(), d_knw.tensor(),
                            d_cos.tensor(), d_sin.tensor(), args);
    b.Synchronize(qg.q);
    d_out.Download(qg.q, gpu.data());
  }
  std::printf("[MEASURED] qsa_compress grid stride: %lld blocks over a 4096-block cap\n",
              static_cast<long long>(kNb));
  // The compressor has no transcendental, so the byte contract holds at scale
  // exactly as it does at the golden widths.
  CheckBitwise(gpu, cpu, "qsa_compress 4100 blocks / grid stride");
  // A collapsed stride leaves the tail AS ALLOCATED, which `DeviceTensor` does
  // not zero, so it is undefined rather than zero. Agreement alone could be two
  // zeros agreeing, so the tail is also asserted to be non-zero.
  double tail = 0.0;
  for (size_t i = static_cast<size_t>(4096 * D); i < gpu.size(); ++i) {
    tail = std::max(tail, std::fabs(static_cast<double>(gpu[i])));
  }
  std::printf("[MEASURED] qsa_compress grid-stride tail max = %.9g (0 means never written)\n",
              tail);
  CHECK(tail > 0.0);
}

TEST_CASE("vt::Qwen4ExpQsaCompress CUDA: the rope position is the BLOCK'S FIRST token") {
  if (SkipNoCuda("vt::Qwen4ExpQsaCompress CUDA rope-position probe")) return;
  // Taking the block's LAST position instead is a silent one-block phase error
  // that no shape check can see, and it is the defect this case separates ON THE
  // DEVICE. The probe shifts the cos/sin tables by one block and measures the
  // distance; a kernel that ignored `pos` entirely would give zero separation.
  const QsaCase& c = kSubBudget;
  const int64_t D = g::kIndexHeadDim, CR = g::kCompressRatio, rot = g::kRotaryDim;
  const int64_t complete = (c.seq / CR) * CR, nb = complete / CR;
  const std::vector<float> ref = RunCompress(DeviceType::kCUDA, c, true);

  // Rebuild with tables rotated by one block: row p reads what row p+CR read.
  std::vector<float> cos = Slice(c.cos, c.seq * rot), sin = Slice(c.sin, c.seq * rot);
  std::vector<float> cos2(cos.size()), sin2(sin.size());
  for (int64_t p = 0; p < c.seq; ++p) {
    const int64_t src = std::min<int64_t>(p + CR, c.seq - 1);
    std::copy(cos.begin() + src * rot, cos.begin() + (src + 1) * rot, cos2.begin() + p * rot);
    std::copy(sin.begin() + src * rot, sin.begin() + (src + 1) * rot, sin2.begin() + p * rot);
  }
  std::vector<float> raw = Slice(c.k_raw, complete * D);
  std::vector<float> knw = Slice(c.k_norm_w, D);
  std::vector<float> out(static_cast<size_t>(nb * D), 0.0f);
  Qwen4ExpQsaCompressArgs args;
  args.compress_ratio = CR;
  args.rotary_dim = rot;
  args.eps = g::kRmsNormEps;
  args.round_intermediates_to_bf16 = true;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_raw(b, qg.q, DType::kF32, {complete, D}, raw.data());
  DeviceTensor d_knw(b, qg.q, DType::kF32, {D}, knw.data());
  DeviceTensor d_cos(b, qg.q, DType::kF32, {c.seq, rot}, cos2.data());
  DeviceTensor d_sin(b, qg.q, DType::kF32, {c.seq, rot}, sin2.data());
  DeviceTensor d_out(b, qg.q, DType::kF32, {nb, D});
  vt::Qwen4ExpQsaCompress(qg.q, d_out.tensor(), d_raw.tensor(), d_knw.tensor(),
                          d_cos.tensor(), d_sin.tensor(), args);
  b.Synchronize(qg.q);
  d_out.Download(qg.q, out.data());
  double sep = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    sep = std::max(sep, std::fabs(static_cast<double>(out[i]) - ref[i]));
  }
  std::printf("[MEASURED] qsa_compress CUDA rope-position separation = %.9g\n", sep);
  CHECK(sep > 1e-3);  // the position IS read on the device
}

// ═════════════════════════════════════════════════════════════════════════════
// vt::Qwen4ExpQsaGatherAttention
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("vt::Qwen4ExpQsaGatherAttention CUDA reproduces the oracle and the CPU arm") {
  if (SkipNoCuda("vt::Qwen4ExpQsaGatherAttention CUDA arm")) return;
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads, DH = g::kHeadDim;
  for (const QsaCase* c : {&kSubBudget, &kOverBudget}) {
    INFO("case ", c->name);
    const Selection sel = RunIndexerCpu(*c);
    const std::vector<float> qa = Slice(c->attn_q, c->seq * HQ * DH);
    const std::vector<float> ka = Slice(c->attn_k, c->seq * HKV * DH);
    const std::vector<float> va = Slice(c->attn_v, c->seq * HKV * DH);
    const GatherResult gpu =
        RunGather(DeviceType::kCUDA, qa, ka, va, sel, c->seq, HQ, HKV, DH, c->seq, false);
    const GatherResult cpu =
        RunGather(DeviceType::kCPU, qa, ka, va, sel, c->seq, HQ, HKV, DH, c->seq, false);
    const double l2 = RelL2(gpu.out, c->attn_out, c->seq * HQ * DH);
    std::printf("[MEASURED] qsa_gather %-12s vs ORACLE relL2 = %.9g  bound = %g\n",
                c->name.c_str(), l2, kGatherL2);
    CHECK(l2 < kGatherL2);
    // The arm-vs-arm bound is DERIVED from this fixture's own inputs, not a
    // fitted constant -- see the derivation beside `kExpRelDiff`. The case
    // prints the scale-free ratio, which is the number to watch.
    CheckGatherDerived(gpu.out, cpu.out, va, sel, c->seq, HQ, HKV, DH,
                       ("qsa_gather " + c->name + " CUDA-vs-CPU").c_str());
  }
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention CUDA: |sel| CROSSES the tile boundary") {
  if (SkipNoCuda("vt::Qwen4ExpQsaGatherAttention CUDA multi-tile")) return;
  // THE CASE THE COMMITTED SUITE DID NOT HAVE. Pass 2 carries the running
  // denominator across `kSelTile`-sized tiles through shared memory; at one tile
  // that carry is dead code, and a fresh review showed deleting it leaves the
  // output BYTE-FOR-BYTE identical on every golden fixture. Each shape below
  // spans many tiles, so the carry is load-bearing in all of them.
  struct Shape { const char* name; int64_t T, HQ, HKV, DH, kv, CR, topk; };
  const Shape kShapes[] = {
      // 300 complete blocks -> |sel| = 1200 -> ceil(1200/32) = 38 tiles.
      {"1200 rows / 38 tiles", 2, 2, 1, 32, 1200, 4, 300},
      // The RELEASED geometry's selection size, at head_dim 64:
      // ceil(2050/32) = 65 tiles.
      {"2050 rows / 65 tiles", 1, 2, 1, 64, 2050, 4, 512},
      // 64 selected rows is EXACTLY two tiles; the 2-row ragged tail makes a
      // third, partial one. ceil(66/32) = 3.
      {"2 full tiles + a 2-row tail", 3, 4, 2, 32, 66, 4, 16},
  };
  for (const Shape& sh : kShapes) {
    CAPTURE(std::string(sh.name));
    const Synth y = MakeSynth(sh.T, sh.HQ, sh.HKV, sh.DH, sh.kv, sh.CR, sh.topk, 909u);
    const int64_t tiles = (static_cast<int64_t>(ExpandHost(y.sel, 0, sh.kv).size()) + 31) / 32;
    // The fixture must actually cross a tile, or the case is the old one again.
    CHECK(tiles > 1);
    const GatherResult gpu = RunGather(DeviceType::kCUDA, y.q, y.k, y.v, y.sel, sh.T, sh.HQ,
                                       sh.HKV, sh.DH, y.max_kv, false);
    const GatherResult cpu = RunGather(DeviceType::kCPU, y.q, y.k, y.v, y.sel, sh.T, sh.HQ,
                                       sh.HKV, sh.DH, y.max_kv, false);
    char nm[96];
    std::snprintf(nm, sizeof nm, "qsa_gather %s", sh.name);
    std::printf("[MEASURED] %-44s tiles = %lld\n", nm, static_cast<long long>(tiles));
    CheckGatherDerived(gpu.out, cpu.out, y.v, y.sel, sh.T, sh.HQ, sh.HKV, sh.DH, nm);
  }
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention CUDA: the GRID STRIDE takes more than one trip") {
  if (SkipNoCuda("vt::Qwen4ExpQsaGatherAttention CUDA grid stride")) return;
  // The launcher caps the grid at 4096 blocks and the kernel walks
  // `pair += gridDim.x`. Every golden fixture has `T*HQ` <= 92, so that loop ran
  // ONCE and collapsing it to `blockIdx.x` was byte-identical. 5200 pairs makes
  // the second trip mandatory: without it, every pair at index >= 4096 keeps the
  // zero its output buffer was allocated with.
  constexpr int64_t T = 1300, HQ = 4, HKV = 2, DH = 32, KV = 8, CR = 4, TOPK = 2;
  const int64_t pairs = T * HQ;
  CHECK(pairs > 4096);  // the fixture must cross the cap
  const Synth y = MakeSynth(T, HQ, HKV, DH, KV, CR, TOPK, 4242u);
  const GatherResult gpu =
      RunGather(DeviceType::kCUDA, y.q, y.k, y.v, y.sel, T, HQ, HKV, DH, y.max_kv, false);
  const GatherResult cpu =
      RunGather(DeviceType::kCPU, y.q, y.k, y.v, y.sel, T, HQ, HKV, DH, y.max_kv, false);
  std::printf("[MEASURED] qsa_gather grid stride: %lld pairs over a 4096-block cap\n",
              static_cast<long long>(pairs));
  CheckGatherDerived(gpu.out, cpu.out, y.v, y.sel, T, HQ, HKV, DH,
                     "qsa_gather 5200 pairs / grid stride");
  // A collapsed stride leaves the tail AS ALLOCATED. `DeviceTensor` does not
  // zero its buffer, so "as allocated" is undefined rather than zero -- the
  // point stands either way and the wording is corrected: the pair of
  // assertions is what carries it. Agreement alone could be two zeros agreeing,
  // so the tail is also asserted to be non-zero.
  double tail_mag = 0.0;
  for (size_t i = static_cast<size_t>(4096 * DH); i < gpu.out.size(); ++i) {
    tail_mag = std::max(tail_mag, std::fabs(static_cast<double>(gpu.out[i])));
  }
  std::printf("[MEASURED] qsa_gather grid-stride tail max|out| = %.9g (0 means never written)\n",
              tail_mag);
  CHECK(tail_mag > 0.0);
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention CUDA: the ROW SET decides, not how it is described") {
  if (SkipNoCuda("vt::Qwen4ExpQsaGatherAttention CUDA row-set invariance")) return;
  // **THE CASE THAT STOOD HERE WAS VACUOUS AND IS REPLACED.** It read "a
  // sub-budget gather is BIT-IDENTICAL to dense" and built a `dense` selection
  // to compare against the indexer's. On `kSubBudget` -- seq 11, CR 4, topk 2 --
  // every query has `complete = kv_len/4 <= 2 = topk`, so `DsaTopkSelect` takes
  // its all-select branch and emits ascending: the `dense` buffer it was
  // compared against was THE SAME BUFFER. It measured determinism, which this
  // file's own comment already said is insufficient. Its vacuity guard was an
  // `INFO`, not a `CHECK`, so nothing caught that.
  //
  // Running it on `kOverBudget` does not fix it either, and that is worth
  // stating: above the budget the selection is a STRICT SUBSET of the complete
  // blocks, so it is not equal to a dense walk and must not be -- that
  // inequality IS the sparsity this op exists for.
  //
  // The non-vacuous form of the same claim is INVARIANCE: two selections that
  // name the SAME rows through DIFFERENT descriptions must agree to the bit.
  // Here the two differ in `topk` width, so the kernel's `-1` terminator is met
  // at a different `j` and the expansion arithmetic runs over a different buffer
  // stride, while the row set is identical. That is what llama.cpp #27742's
  // "max logit delta 0.0" claim actually needs: with every candidate selected,
  // the gather reduces over exactly the dense sequence, in exactly the dense
  // order. The fixture also spans 7 tiles, so the carry is live.
  constexpr int64_t T = 2, HQ = 4, HKV = 2, DH = 32, KV = 200, CR = 4;
  const Synth narrow = MakeSynth(T, HQ, HKV, DH, KV, CR, /*topk=*/50, 77u);
  Synth wide = MakeSynth(T, HQ, HKV, DH, KV, CR, /*topk=*/64, 77u);

  // The two descriptions must genuinely DIFFER ...
  CHECK(narrow.sel.topk != wide.sel.topk);
  CHECK(narrow.sel.block_ids.size() != wide.sel.block_ids.size());
  // ... and must name exactly the same rows, or the comparison below is not an
  // invariance claim at all. This is the guard the old case left as an INFO.
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<int64_t> a = ExpandHost(narrow.sel, t, KV);
    const std::vector<int64_t> b = ExpandHost(wide.sel, t, KV);
    REQUIRE(a == b);
    REQUIRE(a.size() == static_cast<size_t>(KV));  // every row, i.e. the dense prefix
  }
  const int64_t tiles = (KV + 31) / 32;
  CHECK(tiles > 1);

  const GatherResult a = RunGather(DeviceType::kCUDA, narrow.q, narrow.k, narrow.v,
                                   narrow.sel, T, HQ, HKV, DH, narrow.max_kv, false);
  const GatherResult b = RunGather(DeviceType::kCUDA, wide.q, wide.k, wide.v, wide.sel, T,
                                   HQ, HKV, DH, wide.max_kv, false);
  std::printf("[MEASURED] qsa_gather row-set invariance: topk %lld vs %lld over %lld tiles\n",
              static_cast<long long>(narrow.sel.topk), static_cast<long long>(wide.sel.topk),
              static_cast<long long>(tiles));
  CheckBitwise(a.out, b.out, "qsa_gather CUDA row set vs description");
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention CUDA: the GATHER reads only the selected rows") {
  if (SkipNoCuda("vt::Qwen4ExpQsaGatherAttention CUDA read set")) return;
  // TWO INSTRUMENTS, because neither alone is a set equality and the wave's
  // whole point is that a MASK over a dense cache is correct and passes every
  // value comparison:
  //
  //   COUNT  `keys_visited`, incremented AT THE READ inside the kernel and
  //          block-reduced, against a count derived independently from the HOST
  //          expansion of the selection. Two quantities, two derivations.
  //   MEMBERSHIP  the cache's UNSELECTED rows are NaN. A gather never addresses
  //          them; a mask multiplies them by a zero weight into `0.0f * NaN`,
  //          which is NaN. `MaxAbsDiff` returns +infinity on any non-finite
  //          operand (#449), so a mask cannot pass this.
  //
  // Together they are SET EQUALITY: membership gives read-set is a subset of
  // selected, count gives |read set| == |selected|.
  const QsaCase& c = kOverBudget;
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads, DH = g::kHeadDim;
  constexpr int64_t kReadsPerRowPerHead = 2;  // two softmax passes
  const Selection sel = RunIndexerCpu(c);
  const std::vector<float> qa = Slice(c.attn_q, c.seq * HQ * DH);
  std::vector<float> ka = Slice(c.attn_k, c.seq * HKV * DH);
  std::vector<float> va = Slice(c.attn_v, c.seq * HKV * DH);

  int64_t want = 0, dense = 0, strictly_sparse = 0;
  std::vector<char> ever_selected(static_cast<size_t>(c.seq), 0);
  for (int64_t t = 0; t < c.seq; ++t) {
    const std::vector<int64_t> e = ExpandHost(sel, t, t + 1);
    for (int64_t p : e) ever_selected[static_cast<size_t>(p)] = 1;
    const int64_t sel_reads = static_cast<int64_t>(e.size()) * HQ * kReadsPerRowPerHead;
    const int64_t dense_reads = (t + 1) * HQ * kReadsPerRowPerHead;
    want += sel_reads;
    dense += dense_reads;
    if (sel_reads < dense_reads) ++strictly_sparse;
  }
  const GatherResult clean =
      RunGather(DeviceType::kCUDA, qa, ka, va, sel, c.seq, HQ, HKV, DH, c.seq, true);
  std::printf("[MEASURED] qsa_gather CUDA keys_visited = %lld  selected-derived = %lld  "
              "dense = %lld  margin = %lld\n",
              static_cast<long long>(clean.keys_visited), static_cast<long long>(want),
              static_cast<long long>(dense), static_cast<long long>(dense - want));
  INFO("keys_visited ", clean.keys_visited, " want ", want, " dense ", dense);
  CHECK(clean.keys_visited == want);
  CHECK(clean.keys_visited < dense);
  // Above the budget the selection MUST discard blocks; a fixture that never
  // crossed it would leave the assertion above trivially true.
  CHECK(strictly_sparse > 0);

  // MEMBERSHIP, AND IT IS PER-QUERY RATHER THAN PER-SEQUENCE. The first draft of
  // this case poisoned the rows NO query anywhere in the batch selects, ran, and
  // asserted the answer was unchanged. On this fixture that set is EMPTY — every
  // cache row is selected by SOME query — so the probe measured nothing while
  // reporting `0/2944 differ`, which is a byte gate over a cache it never
  // poisoned. **THE GUARD IS WHAT CAUGHT IT** (`CHECK(poisoned > 0)` reddened on
  // the device), and the repair is to ask the question the op actually answers:
  // for ONE query token, does the kernel read outside THAT token's selection?
  //
  // So the probe runs a single-token call and poisons every row below that
  // token's `kv_len` that its own selection does not name. A gather never
  // addresses them; a mask multiplies them by a zero weight into `0.0f * NaN`,
  // which is NaN. `MaxAbsDiff` returns +infinity on any non-finite operand
  // (#449), so a mask cannot pass this.
  const int64_t probe_t = c.seq - 1;  // the largest kv_len, so the most to skip
  const int64_t probe_kv = probe_t + 1;
  const std::vector<int64_t> probe_sel = ExpandHost(sel, probe_t, probe_kv);
  std::vector<char> in_sel(static_cast<size_t>(c.seq), 0);
  for (int64_t p : probe_sel) in_sel[static_cast<size_t>(p)] = 1;

  Selection one;
  one.block_ids.assign(sel.block_ids.begin() + probe_t * kTopk,
                       sel.block_ids.begin() + (probe_t + 1) * kTopk);
  one.kv_lens.assign(1, static_cast<int32_t>(probe_kv));
  const std::vector<float> q_one(qa.begin() + probe_t * HQ * DH,
                                 qa.begin() + (probe_t + 1) * HQ * DH);
  const GatherResult one_clean =
      RunGather(DeviceType::kCUDA, q_one, ka, va, one, 1, HQ, HKV, DH, c.seq, false);

  int64_t poisoned = 0;
  for (int64_t p = 0; p < probe_kv; ++p) {
    if (in_sel[static_cast<size_t>(p)] != 0) continue;
    ++poisoned;
    for (int64_t e = 0; e < HKV * DH; ++e) {
      ka[static_cast<size_t>(p * HKV * DH + e)] = std::nanf("");
      va[static_cast<size_t>(p * HKV * DH + e)] = std::nanf("");
    }
  }
  std::printf("[MEASURED] qsa_gather CUDA query %lld: %lld of its %lld visible rows "
              "poisoned (%lld selected)\n",
              static_cast<long long>(probe_t), static_cast<long long>(poisoned),
              static_cast<long long>(probe_kv),
              static_cast<long long>(probe_sel.size()));
  // A fixture with nothing to poison proves nothing. This is the assertion that
  // caught the per-sequence draft.
  CHECK(poisoned > 0);
  const GatherResult one_probed =
      RunGather(DeviceType::kCUDA, q_one, ka, va, one, 1, HQ, HKV, DH, c.seq, false);
  CheckBitwise(one_probed.out, one_clean.out, "qsa_gather CUDA NaN-unselected walk");
  // And the clean single-token answer must agree with the batched one, so the
  // probe is measuring the same computation the case above measured.
  const std::vector<float> batched_row(clean.out.begin() + probe_t * HQ * DH,
                                       clean.out.begin() + (probe_t + 1) * HQ * DH);
  CheckBitwise(one_clean.out, batched_row, "qsa_gather CUDA single-token == batched row");
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention CUDA: the PAGED address mode agrees with contiguous") {
  if (SkipNoCuda("vt::Qwen4ExpQsaGatherAttention CUDA paged arm")) return;
  // The engine allocates this model's QSA K/V as a PAGED FullAttentionSpec
  // group, so the contiguous arm alone could serve nothing a runner hands a
  // forward. The paged view is STRIDED by construction — K and V interleave at
  // dim 1 of the flash cache — and the two arms are ONE body whose only fork is
  // the resolution of one row address, so the answers must be BITWISE equal.
  const QsaCase& c = kOverBudget;
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads, DH = g::kHeadDim;
  constexpr int64_t kPage = 4;
  const Selection sel = RunIndexerCpu(c);
  const std::vector<float> qa = Slice(c.attn_q, c.seq * HQ * DH);
  const std::vector<float> ka = Slice(c.attn_k, c.seq * HKV * DH);
  const std::vector<float> va = Slice(c.attn_v, c.seq * HKV * DH);
  const int64_t pages = (c.seq + kPage - 1) / kPage;
  const int64_t row = HKV * DH;

  // The flash cache: [pages, 2, kPage, HKV, DH], K at dim1 == 0, V at dim1 == 1.
  // The page TABLE names them in REVERSE, which is what says the kernel reads
  // the table rather than assuming identity.
  std::vector<float> cache(static_cast<size_t>(pages * 2 * kPage * row), 0.0f);
  std::vector<int32_t> table(static_cast<size_t>(pages), 0);
  for (int64_t lp = 0; lp < pages; ++lp) table[static_cast<size_t>(lp)] =
      static_cast<int32_t>(pages - 1 - lp);
  for (int64_t p = 0; p < c.seq; ++p) {
    const int64_t phys = table[static_cast<size_t>(p / kPage)];
    const int64_t off = p % kPage;
    for (int64_t i = 0; i < row; ++i) {
      cache[static_cast<size_t>(((phys * 2 + 0) * kPage + off) * row + i)] =
          ka[static_cast<size_t>(p * row + i)];
      cache[static_cast<size_t>(((phys * 2 + 1) * kPage + off) * row + i)] =
          va[static_cast<size_t>(p * row + i)];
    }
  }

  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_cache(b, qg.q, DType::kF32, {pages, 2, kPage, HKV, DH}, cache.data());
  DeviceTensor d_tab(b, qg.q, DType::kI32, {1, pages}, table.data());
  DeviceTensor d_q(b, qg.q, DType::kF32, {c.seq, HQ, DH}, qa.data());
  std::vector<int32_t> ids = sel.block_ids, lens = sel.kv_lens;
  DeviceTensor d_i(b, qg.q, DType::kI32, {c.seq, kTopk}, ids.data());
  DeviceTensor d_l(b, qg.q, DType::kI32, {c.seq}, lens.data());
  DeviceTensor d_o(b, qg.q, DType::kF32, {c.seq, HQ, DH});

  // The two unbind views, built the way `dense_attn::KvSlice` builds them: the
  // page stride is `2 * kPage * HKV * DH` and V starts one `kPage * HKV * DH`
  // block in. EVERY FIELD IS SET EXPLICITLY; a marker or a stride dropped at
  // this boundary produces plausible numbers and no crash.
  auto kv_view = [&](int64_t which) {
    Tensor t;
    t.data = static_cast<float*>(d_cache.raw()) + which * kPage * row;
    t.dtype = DType::kF32;
    t.device = Gpu();
    t.rank = 4;
    t.shape[0] = pages;   t.stride[0] = 2 * kPage * row;
    t.shape[1] = kPage;   t.stride[1] = row;
    t.shape[2] = HKV;     t.stride[2] = DH;
    t.shape[3] = DH;      t.stride[3] = 1;
    return t;
  };
  Tensor t_k = kv_view(0), t_v = kv_view(1);

  Qwen4ExpQsaAttnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(DH));
  args.compress_ratio = g::kCompressRatio;
  args.kv_block_table = &d_tab.tensor();
  args.kv_block_size = kPage;
  vt::Qwen4ExpQsaGatherAttention(qg.q, d_o.tensor(), d_q.tensor(), t_k, t_v, d_i.tensor(),
                                 d_l.tensor(), args);
  b.Synchronize(qg.q);
  std::vector<float> paged(static_cast<size_t>(c.seq * HQ * DH), 0.0f);
  d_o.Download(qg.q, paged.data());

  const GatherResult contig =
      RunGather(DeviceType::kCUDA, qa, ka, va, sel, c.seq, HQ, HKV, DH, c.seq, false);
  CheckBitwise(paged, contig.out, "qsa_gather CUDA paged vs contiguous");
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention CUDA: a malformed selection POISONS the row") {
  if (SkipNoCuda("vt::Qwen4ExpQsaGatherAttention CUDA malformed-selection refusal")) return;
  // The CPU arm `VT_CHECK`s that block ids are ASCENDING and inside the complete
  // block count. A device kernel cannot throw, and `block_ids` is device-resident
  // so the host dispatcher cannot read it either. This arm therefore reads
  // NOTHING out of range and writes NaN across the row — chosen over a clamp or
  // a skip, both of which return a plausible tensor. This case asserts the
  // poison actually appears, so a future edit that turns it into a silent skip
  // reds here.
  const QsaCase& c = kSubBudget;
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads, DH = g::kHeadDim;
  Selection bad = RunIndexerCpu(c);
  // Make query token seq-1 name a block far past its own complete-block count.
  const int64_t t = c.seq - 1;
  bad.block_ids[static_cast<size_t>(t * kTopk + 0)] = static_cast<int32_t>(c.seq);
  for (int64_t j = 1; j < kTopk; ++j) bad.block_ids[static_cast<size_t>(t * kTopk + j)] = -1;
  const GatherResult r =
      RunGather(DeviceType::kCUDA, Slice(c.attn_q, c.seq * HQ * DH),
                Slice(c.attn_k, c.seq * HKV * DH), Slice(c.attn_v, c.seq * HKV * DH), bad,
                c.seq, HQ, HKV, DH, c.seq, false);
  size_t nan_in_row = 0;
  for (int64_t i = 0; i < HQ * DH; ++i) {
    if (std::isnan(r.out[static_cast<size_t>(t * HQ * DH + i)])) ++nan_in_row;
  }
  std::printf("[MEASURED] qsa_gather CUDA malformed row: %zu/%lld outputs are NaN\n",
              nan_in_row, static_cast<long long>(HQ * DH));
  CHECK(nan_in_row == static_cast<size_t>(HQ * DH));
  // And the OTHER rows are untouched: the poison is per (token, head) and not a
  // whole-tensor abort.
  size_t nan_elsewhere = 0;
  for (int64_t i = 0; i < t * HQ * DH; ++i) {
    if (std::isnan(r.out[static_cast<size_t>(i)])) ++nan_elsewhere;
  }
  CHECK(nan_elsewhere == 0);
}
