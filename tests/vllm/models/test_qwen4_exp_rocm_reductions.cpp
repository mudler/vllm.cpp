// Qwen4-Exp (Qwen3.8-Flash-Next) ROCm DEVICE-ARM GATE for the grouped
// hyper-connection norm inside `vt::Qwen4ExpGatedResidual`.
// Row MODEL-MM-QWEN4-EXP, issue ISSUE-LOCAL-01M2CJXQMV9R9JGRSKZMW4W21F, spec
// `.agents/specs/qwen4-exp-rocm-hc-norm.md`.
//
// ─── WHY THIS FILE EXISTS ────────────────────────────────────────────────────
// Until it did, ONE committed test reached the ROCm arm of this op:
// `tests/vt/test_backend_cross_device.cpp:5580`, at `hidden_size = 7` against
// `kNmseTol = 5e-4`. Seven elements cannot express a reduction SHAPE — one
// thread does the whole group whatever the launch looks like — so a kernel that
// dropped its cross-lane fold, its group grid stride or its last strided trip
// passed it. That case is not replaced and is not widened; this file is the
// shape gate beside it.
//
// ─── THE FIXTURE IS SHARED WITH THE CUDA ARM, DELIBERATELY ───────────────────
// `qwen4_exp_hc_synth.h` holds the generator, the derived bound `W7NormRel` and
// the runner, and `test_qwen4_exp_cuda_reductions.cpp` includes the same header.
// Two arms of one kernel held to two separately-maintained copies of one
// derivation drift, and the first repair to either copy creates the
// contradiction. Read the derivation in that header; it is not restated here.
//
// ─── WHAT IS COMPARED AGAINST WHAT, AND WHAT THAT IS WORTH ───────────────────
// The ROCm arm against the CPU arm on identical inputs, at widths the goldens
// cannot express. The CPU arm is itself held to the transformers goldens by
// `test_qwen4_exp_hc_device.cpp`, so the chain reaches an oracle — but only as
// a TOLERANCE, and the tolerance is not transitive the way an equality would be.
// The bound here is a reduction-width bound and nothing else, which is why the
// fixture zeroes both mix projections: with a live projection the dominant term
// would be `vt::MatmulBT`'s K-reassociation, a different phenomenon that the
// oracle cases at 1e-5 already carry.
//
// ─── REACHABILITY, HONESTLY ──────────────────────────────────────────────────
// These cases construct the op by hand on the `vt::` seam, exactly as the CUDA
// gate does, and a hand-constructed call is NOT a reachability proof. The kernel
// IS reached in production on this board: `examples/vllm-cli --device auto` runs
// the released `Qwen3.8-Flash-Next` UD-IQ1_S through `ModelRegistry::Forward` on
// `strix:gpu0` and emits 32 tokens (docs/USAGE.md, measured 2026-09-13), which
// is what makes this kernel 35.10% of that run's decode kernel time. This file
// gates the NUMBERS; that run is the reach.
//
// ─── EVERY CASE NAME CARRIES THE LITERAL `ROCM W7` ───────────────────────────
// That is load-bearing rather than decorative: `-tc='*ROCM W7*'` is how a
// reviewer runs this battery alone while mutating the kernel. The CUDA twin's
// first mutation round used a filter that matched NO case and doctest printed
// `0 passed | 0 failed`, `Status: SUCCESS!`, and exited 0 on all five mutations.
// Run every filtered arm through `scripts/run-doctest-selected.sh <binary>
// -tc='<filter>'`, which asks doctest how many cases the filter selects and
// REFUSES with exit 3 at zero, and read the ASSERTION count as well as the case
// count — a selected case that early-returns through the skip guard below
// reports as passed having asserted nothing.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "qwen4_exp_hc_synth.h"
#include "vt/backend.h"
#include "vt/ops.h"

using vt::DeviceType;

namespace {

using q4hc::CheckBitwise;
using q4hc::CheckWithin;
using q4hc::Compare;
using q4hc::kAbsFloor;
using q4hc::MakeMagnitudeSeparatedHc;
using q4hc::MakeSynthHc;
using q4hc::MaxAbsFinite;
using q4hc::MixerResult;
using q4hc::RunSynthMixer;
using q4hc::SynthHc;
using q4hc::W7NormRel;

bool HasRocm() {
  try {
    vt::GetBackend(DeviceType::kROCM);
  } catch (const std::runtime_error&) {
    return false;
  }
  return vt::OpRegistered(vt::OpId::kQwen4ExpGatedResidual, DeviceType::kROCM);
}

// LOUD, because a silent skip on a box without a ROCm device is how a device arm
// goes un-gated for a release. It also names the op, not just the backend: a
// built-but-unregistered arm and an absent backend are different failures and a
// single "no ROCm" line would merge them.
bool SkipNoRocm(const char* what) {
  if (HasRocm()) return false;
  std::printf("[SKIP] no ROCm backend with kQwen4ExpGatedResidual: %s NOT exercised\n", what);
  return true;
}

// The portable CPU reference tier can serve an op a device did not register, and
// a run that fell through to it would compare the CPU arm against the CPU arm
// and report a perfect match. Every case asserts the counter did not move, so
// that mode reads as a failure instead of as the best result in the file.
struct TierWatch {
  unsigned long long before = vt::GetReferenceTierHits();
  void Check(const char* what) const {
    const unsigned long long after = vt::GetReferenceTierHits();
    INFO(what << ": reference-tier hits went " << before << " -> " << after
              << "; a non-zero delta means the CPU tier served this, not ROCm");
    CHECK(after == before);
  }
};

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
// The four SHAPE cases. Each names the structural property it alone can see.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("vt::Qwen4ExpGatedResidual ROCM W7: the grouped norm at MODEL WIDTH, many elements per thread") {
  if (SkipNoRocm("ROCM W7 grouped norm at model width")) return;
  // hidden_size 2560, hc_count 4, lowrank 320 — the released
  // Qwen3.8-Flash-Next's own numbers (`qwen4_exp.h`). At a 256-thread block that
  // is TEN elements per thread, so the per-thread strided walk is live and every
  // wave contributes: eight of them on a wave32 target, four on wave64. At the
  // goldens' hidden 6 and the cross-device case's hidden 7, neither is.
  //
  // THIS IS ALSO THE SHAPE THE 35.10% WAS MEASURED AT. The kernel this case
  // gates is the one `rocprofv3` ranked at the top of the decode budget, and it
  // is at THIS width that four threads used to do all of it.
  TierWatch tier;
  const SynthHc c = MakeSynthHc("model_width", 2560, 4, 320, 3, 0x9E3779B97F4A7C15ULL);
  const MixerResult gpu = RunSynthMixer(DeviceType::kROCM, c);
  const MixerResult cpu = RunSynthMixer(DeviceType::kCPU, c);
  CheckWithin(gpu.mixed, cpu.mixed, W7NormRel(c.hidden), "ROCM W7 model width mixed vs CPU");
  // THE STREAM IS READ-ONLY, asserted at model width too: a normalize-in-place
  // slip would double-normalize at the second site of every layer.
  CheckBitwise(gpu.hyper_after, c.hyper, "ROCM W7 model width stream unchanged");
  tier.Check("ROCM W7 model width");
}

TEST_CASE("vt::Qwen4ExpGatedResidual ROCM W7: MORE GROUPS THAN BLOCKS, so the group grid stride runs") {
  if (SkipNoRocm("ROCM W7 grouped norm past the grid cap")) return;
  // `GridForGroups` caps the grid at 4096 blocks, so 4800 groups forces the
  // kernel's group loop to take a SECOND trip — the only case in this file that
  // does. That second trip is what re-reads the two shared slots after a block
  // has already used them, so it is also the case to drive any future race
  // instrument at. The grid stride is DEAD at `T * hc <= 4096`, which is every
  // other case here.
  TierWatch tier;
  const SynthHc c = MakeSynthHc("grid_cap", 512, 4, 32, 1200, 0xD1B54A32D192ED03ULL);
  REQUIRE(c.T * c.hc > 4096);  // the property this case exists for, asserted
  const MixerResult gpu = RunSynthMixer(DeviceType::kROCM, c);
  const MixerResult cpu = RunSynthMixer(DeviceType::kCPU, c);
  CheckWithin(gpu.mixed, cpu.mixed, W7NormRel(c.hidden), "ROCM W7 grid cap mixed vs CPU");
  CheckBitwise(gpu.hyper_after, c.hyper, "ROCM W7 grid cap stream unchanged");
  tier.Check("ROCM W7 grid cap");
}

TEST_CASE("vt::Qwen4ExpGatedResidual ROCM W7: groups SHORTER than a block, and not a multiple of a wavefront") {
  if (SkipNoRocm("ROCM W7 grouped norm at ragged widths")) return;
  // 100 is under the 256-thread block, so most waves contribute an exact zero
  // and the cross-wave stage must not read a slot no wave wrote. On wave32 that
  // is four live waves of eight; on wave64, two of four — the guard has to hold
  // on both and 100 is under a block on either. 777 is a multiple of neither 32
  // nor 64 nor 256, so the last trip of the strided walk is ragged and the tail
  // threads sit out whatever the wavefront size is. 3 streams keeps `hc` off a
  // power of two.
  TierWatch tier;
  for (int64_t H : {static_cast<int64_t>(100), static_cast<int64_t>(777)}) {
    INFO("hidden ", H);
    const SynthHc c = MakeSynthHc("ragged", H, 3, 16, 4,
                                  0xBF58476D1CE4E5B9ULL + static_cast<uint64_t>(H));
    const MixerResult gpu = RunSynthMixer(DeviceType::kROCM, c);
    const MixerResult cpu = RunSynthMixer(DeviceType::kCPU, c);
    CheckWithin(gpu.mixed, cpu.mixed, W7NormRel(c.hidden),
                ("ROCM W7 ragged hidden=" + std::to_string(H) + " mixed vs CPU").c_str());
    CheckBitwise(gpu.hyper_after, c.hyper,
                 ("ROCM W7 ragged hidden=" + std::to_string(H) + " stream unchanged").c_str());
  }
  tier.Check("ROCM W7 ragged");
}

TEST_CASE("vt::Qwen4ExpGatedResidual ROCM W7: the per-group scales SEPARATE, so a broadcast cannot hide") {
  if (SkipNoRocm("ROCM W7 grouped norm separation probe")) return;
  // A gate that passes says nothing unless the defects it is aimed at would have
  // failed it by a visible margin. This probe MEASURES one such margin instead
  // of asserting it from prose: it replays the model-width case with every group
  // of token 0 forced to group 0's data, which is precisely what a kernel that
  // computed one group and broadcast its reciprocal would produce, and reports
  // how far that lands from the correct answer.
  //
  // WHAT THE NUMBER BOUNDS, AND WHAT IT DOES NOT. It is the signal of ONE
  // defect — a broadcast `r`. It is NOT the floor of the mutation battery and
  // was never shown to be; the other four mutations have their own signals and
  // the spec's §6 table records them separately.
  TierWatch tier;
  const SynthHc c = MakeSynthHc("model_width", 2560, 4, 320, 3, 0x9E3779B97F4A7C15ULL);
  SynthHc broadcast = c;
  const int64_t flat = c.hc * c.hidden;
  for (int64_t j = 1; j < c.hc; ++j) {
    for (int64_t h = 0; h < c.hidden; ++h) {
      broadcast.hyper[static_cast<size_t>(j * c.hidden + h)] =
          c.hyper[static_cast<size_t>(h)];
    }
  }
  (void)flat;
  const MixerResult good = RunSynthMixer(DeviceType::kROCM, c);
  const MixerResult bad = RunSynthMixer(DeviceType::kROCM, broadcast);
  // NOT a hand-rolled `std::max(worst, std::fabs(a - b))`. That spelling is
  // issue #449's form B, it is NaN-BLIND, and `max_abs_diff.h:36-41` records it
  // recurring in THIS model family (#1988). This case is fail-CLOSED against a
  // NaN today -- a blind `sep` of 0.0 misses a `> bound * 1000` bar -- but the
  // spelling is the next recurrence waiting to happen, so it goes too. Both
  // reductions route through the shared hardened scan.
  const double sep = Compare(good.mixed, bad.mixed).worst;
  const double scale = MaxAbsFinite(good.mixed);
  const double bound = kAbsFloor + W7NormRel(c.hidden) * scale;
  std::printf("[MEASURED] ROCM W7 separation probe: broadcast signal = %.9g, case bound = %.9g,"
              " ratio = %.1fx\n", sep, bound, sep / bound);
  INFO("broadcast-group signal " << sep << " against the case bound " << bound);
  CHECK(sep > bound * 1000.0);
  tier.Check("ROCM W7 separation probe");
}

// ═════════════════════════════════════════════════════════════════════════════
// The WIDTH case. This one is not about the launch shape at all.
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("vt::Qwen4ExpGatedResidual ROCM W7: MAGNITUDE-SEPARATED data, where the ACCUMULATOR WIDTH shows") {
  if (SkipNoRocm("ROCM W7 grouped norm magnitude separation")) return;
  // The four shape cases above cannot see an accumulator width: their data is
  // well scaled, so f32 and double answer the same number to well inside
  // `W7NormRel`. This case is the transcription of
  // `test_qwen4_exp_hc_device.cpp:504`, which exists precisely to make a width
  // visible — every element 1.0f but one dominant element per group, four
  // orders of magnitude apart across the streams. That file measured a `double`
  // accumulator at 1.173e-06 and a SERIAL `float ss` at 6.702e-04 against its
  // 1e-5 bar, a 571x separation, AND IT RUNS THE CPU ARM ONLY.
  //
  // WHY IT RUNS CPU-ONLY THERE AND WHY THAT IS CORRECT. The CPU arm is the host
  // REFERENCE and the reference is the thing that must be `double`. The device
  // arms are fp32 BY CONTRACT — `qwen4_exp_hc.h:99-104` says upstream runs the
  // norm in fp32 (`self._norm(x.float())`) and vLLM likewise, and that "the
  // device arm is the thing that must be fp32-accumulate". So this fixture
  // cannot red a device arm into being `double` and is not here to.
  //
  // IT IS HERE BECAUSE A WIDTH CHANGE DESERVES A CASE THAT CAN SEE A WIDTH, and
  // because the 571x is a SERIAL f32 walk and this kernel is not one: its tree
  // is about ten deep at H = 2560 rather than 2560 deep, so it should give most
  // of that back. HOW MUCH IS THE MEASUREMENT, PRINTED BELOW. The bar is the
  // CPU case's own 1e-5, unmoved: if a block tree in f32 cannot hold the bound
  // that a serial f32 walk misses by 67x ON THAT FIXTURE -- this one is a
  // different fixture at a ~5.2x smaller scale, so the 67x is not restated as
  // if it were measured here -- that is a finding about this kernel and not a
  // reason to widen anything.
  TierWatch tier;
  constexpr int64_t kH = 2560, kHc = 4, kR = 1, kT = 1;
  constexpr double kAccumBound = 1e-5;  // `test_qwen4_exp_hc_device.cpp:606`
  const SynthHc c = MakeMagnitudeSeparatedHc("magnitude", kH, kHc, kR, kT,
                                             {4096.0f, 2048.0f, 8192.0f, 1024.0f});
  const MixerResult gpu = RunSynthMixer(DeviceType::kROCM, c);
  const MixerResult cpu = RunSynthMixer(DeviceType::kCPU, c);
  // THE ASSERTION BELOW MUST BE ABLE TO SEE A NaN. It could not until now: the
  // reduction here was `worst = std::max(worst, std::fabs(gpu - cpu))`, which
  // is issue #449's form B. `std::max(a, b)` is `a < b ? b : a` and `a < NaN`
  // is false, so an ALL-NaN device output reduced to `worst = 0.0` and passed
  // `worst < 1e-5`. An Inf output was caught; a NaN was not. That is the third
  // recurrence of one defect -- #449, then #1988 in `test_qwen4_exp_hc.cpp` in
  // this same model family, then here -- in a file whose own shared header
  // forbids the spelling by name eleven lines above the `Compare()` it exports.
  // `Compare` returns +infinity AND raises its own doctest failure naming the
  // offending index, so a poisoned device arm now reds this case.
  const q4hc::Agreement agree = Compare(gpu.mixed, cpu.mixed);
  const double worst = agree.worst;
  const double scale = MaxAbsFinite(cpu.mixed);
  // THE `6.702e-04` IS FROM A DIFFERENT FIXTURE AND IS LABELLED AS SUCH. It was
  // measured on `test_qwen4_exp_hc_device.cpp:504`, whose peak |reference| is
  // 32.895; this case's scale is ~6.335, so the two numbers are not directly
  // comparable and dividing them is not a result. Scale-normalised, that walk
  // predicts 6.702e-04 / 32.895 * `scale` here, which is what the last column
  // prints -- a normalised PREDICTION beside our measurement, never a measured
  // serial-f32 arm on this fixture. No such arm has been run here.
  const double serial_f32_normalised = 6.702e-04 / 32.895 * scale;
  std::printf("[MEASURED] ROCM W7 magnitude-separated: max|diff| = %.9g  scale = %.9g"
              "  bound = %.9g  (serial-f32 walk from hc_device.cpp:504, SCALE-NORMALISED to"
              " this fixture: %.9g)\n",
              worst, scale, kAccumBound, serial_f32_normalised);
  INFO("ROCm f32 tree vs CPU double reference on magnitude-separated data: max|diff| = "
       << worst << " against the CPU case's own bound " << kAccumBound);
  CHECK(worst < kAccumBound);
  CheckBitwise(gpu.hyper_after, c.hyper, "ROCM W7 magnitude stream unchanged");
  tier.Check("ROCM W7 magnitude separated");
}
