// NemotronH A2-Q1 (#810, parent #517) — the DEVICE Mamba2 arm on a SYNTHETIC
// FP8 W8A8 fixture.
//
// ─── WHY THIS FILE EXISTS AT ALL ────────────────────────────────────────────
//
// The A2-Q1 spec's §5.1 gate is per-block numeric equivalence on the REAL 21 GB
// checkpoint. This file is the CHEAP arm in front of it, and it is not a
// substitute:
//
//   * `test_nemotron_h_forward.cpp`'s `BuildTiny` fixture is all `kDense`, so
//     nothing in the existing device suite can reach the FP8 W8A8 arm at all.
//     Without this file the real checkpoint would be the ONLY instrument, and
//     every RED would cost a GB10 window plus a 21 GB load.
//   * The same reasoning that put `test_nemotron_h_moe_device.cpp` in front of
//     A2-Q2a's real-checkpoint gate, for the same row.
//
// ─── WHAT THE COMPARISON ACTUALLY IS, WHICH IS NOT "TWO WAYS TO DO bf16" ────
//
// The HOST reference is W8A16: `Linear(..., const NemotronHOwned&)`
// (nemotron_h.cpp:291) reaches `DenseFor` (:245), which calls
// `NemotronHOwned::DenseBf16()` — a full dequant of the fp8 weight into bf16 —
// and then runs a bf16 GEMM on an UNQUANTIZED activation. `DenseBf16` says so
// itself (nemotron_h.cpp:419-422: "Weight-only: `input_scale` is carried, not
// applied").
//
// The DEVICE arm is W8A8, which is what vLLM does: static per-tensor activation
// quant against `input_scale` (`quant_utils.py:124`) then an fp8 GEMM with the
// folded `alpha = input_scale * weight_scale`.
//
// So the two arms are NOT bit-comparable by construction, and the difference is
// the e4m3 activation quantization — three mantissa bits, averaged down over the
// K-sum. That is exactly why the band below is MEASURED IN THE RUN and the
// acceptance is derived from a defect this fixture can separate, rather than
// chosen. A band chosen up front is how a 3e-2 bf16 tolerance came to sit ABOVE
// a 2.11e-2 defect on this very row and accept a wrong answer.
//
// ─── THE GEOMETRY IS NOT ARBITRARY ──────────────────────────────────────────
//
// Every GEMM extent is a multiple of 16, because a cuBLASLt fp8 plan needs
// 16-byte-aligned leading dimensions and a shape it declines to run would make a
// refusal here a statement about alignment rather than about values:
//   in_proj   [N=336, K=128]   336 = I(128) + conv_dim(192) + num_heads(16)
//   out_proj  [N=128, K=128]
// and the recurrence divides: num_heads*head_dim = 16*8 = 128 = intermediate,
// intermediate + 2*n_groups*state = 128 + 2*2*16 = 192 = conv_dim, and
// n_groups(2) divides intermediate(128) for the group RMS norm.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // load_stats
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"  // vt::F32ToF8E4M3 — the fixture's e4m3 encoder

namespace {

using vllm::NemotronHBlock;
using vllm::NemotronHMambaState;
using vllm::NemotronHMambaWeights;
using vllm::NemotronHOwned;
using vllm::NemotronHParams;
using vllm::NemotronHWeightForm;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// NOT 1.0, either of them. A `weight_scale` of 1 is invisible to a mutation that
// ignores it, and an `input_scale` of 1 is invisible to the mutation that drops
// it — which is Q1-M1, the first row of the spec's mutation table.
constexpr float kWeightScale = 0.125F;
constexpr float kInputScale = 0.5F;

// The same deterministic generator the forward and MoE-device suites use, so a
// value here is reproducible and independent of any RNG seeding order.
std::vector<float> SynthVec(size_t n, uint32_t salt, float k) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const double a = static_cast<double>(i);
    const double b = static_cast<double>(salt);
    v[i] = static_cast<float>(k * std::sin(0.7 * a + 1.3 * b + 0.21 * a * b));
  }
  return v;
}

// An FP8 W8A8 static projection [rows, cols]. The stored e4m3 bytes are the
// QUANTIZED weight; the logical value is `byte * weight_scale`, which is what
// `DenseBf16` reconstructs on the host arm and what the folded alpha applies on
// the device arm.
NemotronHOwned MakeFp8(int64_t rows, int64_t cols, uint32_t salt, DType logical) {
  NemotronHOwned w;
  w.form = NemotronHWeightForm::kFp8W8A8Static;
  w.dtype = logical;
  w.shape = {rows, cols};
  const std::vector<float> v = SynthVec(static_cast<size_t>(rows * cols), salt, 3.0F);
  w.bytes.resize(v.size());
  for (size_t i = 0; i < v.size(); ++i) w.bytes[i] = vt::F32ToF8E4M3(v[i]);
  w.global_scale = kWeightScale;
  w.input_scale = kInputScale;
  w.has_input_scale = true;
  return w;
}

NemotronHOwned OwnF32(const std::vector<float>& v, DType dt, std::vector<int64_t> shape) {
  return NemotronHOwned::FromF32(v, dt, std::move(shape));
}

// A NemotronH mamba geometry whose every GEMM extent is a multiple of 16.
NemotronHParams MambaParams() {
  NemotronHParams p;
  p.hidden_size = 128;
  p.mamba_num_heads = 16;
  p.mamba_head_dim = 8;
  p.n_groups = 2;
  p.ssm_state_size = 16;
  p.conv_kernel = 4;
  p.chunk_size = 8;
  p.use_conv_bias = true;
  p.mamba_hidden_act = "silu";
  // The RELEASED checkpoint's value, and it is resolved INDEPENDENTLY of the
  // model dtype (mamba_utils.py:96-107). Left unset the fixture fell back to
  // bf16, which is not the configuration the paged forward selects the device
  // arm for (`ssm_dtype == f32`), so the cheap arm was gating a path production
  // does not take.
  p.mamba_ssm_cache_dtype = "float32";
  p.layer_norm_epsilon = 1e-5;
  p.vocab_size = 32;
  p.layers_block_type = {NemotronHBlock::kMamba};
  return p;
}

NemotronHMambaWeights MakeFp8Mamba(const NemotronHParams& p, DType dt) {
  NemotronHMambaWeights w;
  const int64_t H = p.hidden_size;
  const int64_t I = p.mamba_intermediate_size();
  const int64_t Cd = p.conv_dim();
  const int64_t K = p.conv_kernel;
  const int64_t Hh = p.mamba_num_heads;
  w.in_proj = MakeFp8(p.in_proj_out_features(), H, 11, dt);
  w.out_proj = MakeFp8(H, I, 12, dt);
  w.conv1d_weight = OwnF32(SynthVec(static_cast<size_t>(Cd * K), 13, 0.4F), dt, {Cd, K});
  w.conv1d_bias = OwnF32(SynthVec(static_cast<size_t>(Cd), 14, 0.2F), dt, {Cd});
  // f32 BY CONTRACT on both arms — `vt::Mamba2ChunkScan` validates A/D/dt_bias
  // as f32, mirroring upstream's `-torch.exp(self.A_log.float())`.
  w.A_log = OwnF32(SynthVec(static_cast<size_t>(Hh), 15, 0.5F), DType::kF32, {Hh});
  w.D = OwnF32(SynthVec(static_cast<size_t>(Hh), 16, 0.6F), DType::kF32, {Hh});
  w.dt_bias = OwnF32(SynthVec(static_cast<size_t>(Hh), 17, 0.3F), DType::kF32, {Hh});
  w.norm_weight = OwnF32(SynthVec(static_cast<size_t>(I), 18, 0.7F), dt, {I});
  return w;
}

bool TryCudaQueue(Queue* q) {
  try {
    *q = vt::GetBackend(DeviceType::kCUDA).CreateQueue();
    return true;
  } catch (...) {
    return false;
  }
}

// A GPU-less box must SKIP LOUDLY. A device case that silently reports a pass
// over zero device work is indistinguishable from a real one.
void NoteDeviceSkip(const std::string& case_name) {
  MESSAGE("SKIPPED '" << case_name
                      << "': no CUDA device on this box. A2-Q1's device Mamba2 arm "
                         "gates on dgx.casa (GB10 sm_121a) and 192.168.68.23 (Thor "
                         "sm_110); a result from a GPU-less box is not an A2-Q1 result.");
  CHECK(true);  // the skip path ran and said so
}

double MaxAbs(const std::vector<float>& v) {
  double m = 0.0;
  for (float x : v) m = std::max(m, std::abs(static_cast<double>(x)));
  return m;
}

// The largest relative deviation between two answers, against a scale-aware
// denominator. ★ IT REPORTS HOW MANY ELEMENTS IT EXAMINED, AND EVERY CALLER
// ASSERTS THAT. A maximum over ZERO elements is 0.0 — and 0.0 is also exactly
// what a bit-exact comparison prints, so the deviation ALONE cannot distinguish
// "the two arms agree exactly over 512 elements" from "the loop ran over
// nothing".
double MaxRel(const std::vector<float>& got, const std::vector<float>& want,
              int64_t* examined) {
  const double scale = std::max(MaxAbs(want), 1e-30);
  double worst = 0.0;
  int64_t n = 0;
  for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
    const double d = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    worst = std::max(worst, d / scale);
    ++n;
  }
  if (examined != nullptr) *examined = n;
  return worst;
}

// One carried SSM state as f32, whatever it is stored as. The state dtype is
// `mamba_ssm_cache_dtype`'s and NOT the activation dtype's, so a reader that
// assumed one would be right on this fixture and wrong on the next.
std::vector<float> SsmToF32(const NemotronHOwned& t) {
  const int64_t n = t.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  if (t.dtype == DType::kF32) {
    std::memcpy(out.data(), t.bytes.data(), out.size() * sizeof(float));
  } else if (t.dtype == DType::kBF16) {
    const auto* src = reinterpret_cast<const uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  } else {
    // Refuse rather than reinterpret: a wrong-dtype read produces plausible
    // numbers and the comparison below would then be measuring the reader.
    REQUIRE_MESSAGE(false, "carried SSM state is neither f32 nor bf16");
  }
  return out;
}

}  // namespace

// ─── the gate ───────────────────────────────────────────────────────────────
//
// NO COMMA IN ANY CASE NAME, EVER. doctest's `-tc` filter splits on commas, so a
// comma here makes a targeted run select ZERO cases, print `SUCCESS!` and exit
// 0 — a whole mutation pass once read GREEN that way on this repository,
// including the row that deleted the guard.
TEST_CASE("NemotronH A2-Q1: the device Mamba2 block matches the host reference on FP8 W8A8") {
  Queue dq{Device{DeviceType::kCPU, 0}, nullptr};
  if (!TryCudaQueue(&dq)) {
    NoteDeviceSkip("device Mamba2 block vs host reference");
    return;
  }
  const NemotronHParams p = MambaParams();
  Queue hq{Device{DeviceType::kCPU, 0}, nullptr};
  const DType dt = DType::kBF16;  // the released checkpoint's model dtype
  const int64_t H = p.hidden_size;
  const NemotronHMambaWeights w = MakeFp8Mamba(p, dt);

  // T == 1 IS THE DECODE SHAPE and it is the width this model spends its whole
  // decode in; 8 is one full chunk and 12 straddles two, which is where the
  // varlen chunk metadata stops being degenerate. A width loop rather than three
  // copies, and the count is asserted afterwards so a loop that ran over nothing
  // cannot report a clean pass.
  int64_t widths_covered = 0;
  for (const int64_t T : {int64_t{1}, int64_t{8}, int64_t{12}}) {
    INFO("token count T=" << T);
    const std::vector<float> x = SynthVec(static_cast<size_t>(T * H), 77, 0.5F);

    const std::vector<float> host = vllm::NemotronHMamba2Mixer(w, p, x, T, dt, hq);
    const std::vector<float> dev =
        vllm::NemotronHMamba2MixerDeviceHostIO(w, p, x, T, dt, dq);

    REQUIRE(host.size() == static_cast<size_t>(T * H));
    REQUIRE(dev.size() == host.size());

    int64_t examined = 0;
    const double agreed = MaxRel(dev, host, &examined);
    MESSAGE("device-vs-host worst relative deviation: " << agreed << " over " << examined
                                                        << " elements");
    // THE COUNT IS THE REPORT. Asserted against the GEOMETRY rather than against
    // `dev.size()`, so a comparison that silently shortened would still red.
    REQUIRE(examined == T * H);
    REQUIRE(examined > 0);

    // ── THE SEPARATION IS A REAL DEFECT OF THIS BLOCK ────────────────────────
    // Q1-M2 is "alpha folded as weight_scale alone", which scales every output
    // of both fp8 GEMMs by 1/input_scale. Applied to the host answer through the
    // SAME arithmetic and the SAME element count, that is what this fixture can
    // resolve, and the band is derived from it rather than chosen.
    std::vector<float> perturbed = host;
    for (float& v : perturbed) v = static_cast<float>(static_cast<double>(v) / kInputScale);
    int64_t sep_examined = 0;
    const double separation = MaxRel(perturbed, host, &sep_examined);
    MESSAGE("separation of a dropped input_scale: " << separation << " over " << sep_examined
                                                    << " elements");
    REQUIRE(sep_examined == examined);
    REQUIRE(separation > 0.0);

    // Half the measured separation: strictly above any agreement this fixture
    // can show, strictly below the defect it must reject, and it keeps that
    // property at `agreed == 0` (unlike a geometric mean, which degenerates to 0
    // and then fails on the best possible outcome).
    const double band = separation / 2.0;
    MESSAGE("accepting at band " << band);
    CHECK(agreed < band);

    // ── THE GUARD IS A PROPERTY, NOT A TWIN ─────────────────────────────────
    // The perturbed answer, run through the SAME comparison that accepted the
    // real one, must come out REJECTED. Widen `band` past the separation and
    // this line reds, which is the historical hole reproduced by construction.
    int64_t guard_examined = 0;
    const double guard = MaxRel(perturbed, host, &guard_examined);
    REQUIRE(guard_examined == examined);
    INFO("does the band " << band << " REJECT a dropped input_scale?");
    CHECK(guard >= band);
    ++widths_covered;
  }
  REQUIRE(widths_covered == 3);
}

// The CARRY. With one leg and fresh state the recurrence is unobservable — the
// conv window and the SSM state are both zero, so a dropped carry computes the
// identical answer. Two legs is the smallest arrangement in which the device
// arm's in-place state advance is visible at all.
//
// ★ THIS CASE GATES THE STATE, NOT THE SECOND LEG'S OUTPUT, AND THAT IS A
// MEASURED DECISION RATHER THAN A CONVENIENCE.
//
// The first version of this case banded the second leg's OUTPUT against the
// separation of a dropped carry. It failed on Thor (sm_110): the second leg
// agreed to 0.705 while a dropped carry separated by only 0.205, so the band it
// derived (0.102) sat BELOW the deviation a FRESH leg already shows on this
// fixture (0.164 at T=1, measured in the case above and re-measured here).
//
// That is the instrument being wrong, not the arm — and the difference is
// checkable rather than assertable, which is why the noise floor is measured
// HERE, in this run, and printed beside the separation. The two arms are W8A8
// against W8A16 (see the file header), so a fresh leg already disagrees by the
// e4m3 activation quantization; a second leg compounds that with the same
// disagreement propagated through the carried state. A defect whose separation
// is SMALLER than the noise the comparison must accept is not resolvable, and
// widening the band until it passes is exactly what the A2-Q1 spec §8.1 says to
// stop for.
//
// So the assertion moves to the thing the carry actually is: the STATE. A
// dropped carry hands the next leg ZEROS, so the separation between the advanced
// state and a zeroed one is 1.0 by construction — roughly six times the noise
// floor, which is a band this fixture can genuinely resolve. The second leg's
// output is still measured and printed, because a number that cannot carry an
// assertion can still carry a diagnosis.
//
// WHAT THIS CANNOT SEE, stated rather than left implied: a carry that is
// advanced but wrong by less than the band. The real-checkpoint per-block gate
// (spec §5.1), comparing every one of the 23 mamba layers against
// `trace.mixer[l]`, is the instrument for that, and it is still owed.
TEST_CASE("NemotronH A2-Q1: the device Mamba2 arm carries conv and SSM state across two legs") {
  Queue dq{Device{DeviceType::kCPU, 0}, nullptr};
  if (!TryCudaQueue(&dq)) {
    NoteDeviceSkip("device Mamba2 carry across two legs");
    return;
  }
  const NemotronHParams p = MambaParams();
  Queue hq{Device{DeviceType::kCPU, 0}, nullptr};
  const DType dt = DType::kBF16;
  const int64_t H = p.hidden_size;
  const NemotronHMambaWeights w = MakeFp8Mamba(p, dt);

  const int64_t T0 = 8;  // the prefill leg
  const int64_t T1 = 1;  // the decode leg that can only be right if the carry is
  const std::vector<float> x0 = SynthVec(static_cast<size_t>(T0 * H), 31, 0.5F);
  const std::vector<float> x1 = SynthVec(static_cast<size_t>(T1 * H), 32, 0.5F);

  NemotronHMambaState hs;
  NemotronHMambaState ds;
  (void)vllm::NemotronHMamba2Mixer(w, p, x0, T0, dt, hq, &hs);
  (void)vllm::NemotronHMamba2MixerDeviceHostIO(w, p, x0, T0, dt, dq, &ds);
  REQUIRE(hs.has_initial);
  REQUIRE(ds.has_initial);
  REQUIRE(ds.conv.size() == hs.conv.size());
  REQUIRE(ds.conv.size() == static_cast<size_t>(p.conv_dim() * (p.conv_kernel - 1)));
  REQUIRE(ds.ssm.dtype == hs.ssm.dtype);
  REQUIRE(ds.ssm.Numel() == hs.ssm.Numel());
  REQUIRE(ds.ssm.Numel() == p.mamba_num_heads * p.mamba_head_dim * p.ssm_state_size);
  // The state dtype is `mamba_ssm_cache_dtype`'s, resolved independently of the
  // model dtype. The released checkpoint says float32 and so does the fixture;
  // asserted because a state silently halved to bf16 is invisible to every
  // comparison below.
  REQUIRE(ds.ssm.dtype == DType::kF32);

  // ── THE NOISE FLOOR, MEASURED IN THIS RUN AT THIS WIDTH ──────────────────
  // A FRESH device leg against a FRESH host leg over the same tokens: no carry
  // is involved, so whatever they disagree by is the fp8 activation
  // quantization alone. Every band below is read against this.
  const std::vector<float> fresh1_host = vllm::NemotronHMamba2Mixer(w, p, x1, T1, dt, hq);
  const std::vector<float> fresh1_dev =
      vllm::NemotronHMamba2MixerDeviceHostIO(w, p, x1, T1, dt, dq);
  int64_t floor_examined = 0;
  const double noise_floor = MaxRel(fresh1_dev, fresh1_host, &floor_examined);
  MESSAGE("W8A8-vs-W8A16 noise floor at T=" << T1 << ": " << noise_floor << " over "
                                            << floor_examined << " elements");
  REQUIRE(floor_examined == T1 * H);
  REQUIRE(floor_examined > 0);

  // ── THE CONV WINDOW ─────────────────────────────────────────────────────
  // The raw pre-activation input the block just consumed. A device arm that
  // advanced the SSM state but left the conv window unwritten would still
  // return a plausible first-leg answer, so it is compared on its own.
  int64_t conv_examined = 0;
  const double conv_agreed = MaxRel(ds.conv, hs.conv, &conv_examined);
  MESSAGE("carried conv window device-vs-host: " << conv_agreed << " over " << conv_examined
                                                 << " elements");
  REQUIRE(conv_examined == static_cast<int64_t>(hs.conv.size()));
  REQUIRE(conv_examined > 0);
  // The separation of a DROPPED carry, through the SAME arithmetic and the same
  // population: zeros, which is what the next leg would be handed.
  const std::vector<float> conv_zero(hs.conv.size(), 0.0F);
  int64_t conv_sep_examined = 0;
  const double conv_sep = MaxRel(conv_zero, hs.conv, &conv_sep_examined);
  MESSAGE("separation of a DROPPED conv window: " << conv_sep << " over "
                                                  << conv_sep_examined << " elements");
  REQUIRE(conv_sep_examined == conv_examined);
  REQUIRE(conv_sep > 0.0);
  const double conv_band = conv_sep / 2.0;
  MESSAGE("accepting the conv window at band " << conv_band);
  CHECK(conv_agreed < conv_band);
  INFO("does the band " << conv_band << " REJECT a dropped conv window?");
  CHECK(conv_sep >= conv_band);

  // ── THE SSM STATE ───────────────────────────────────────────────────────
  const std::vector<float> hs_ssm = SsmToF32(hs.ssm);
  const std::vector<float> ds_ssm = SsmToF32(ds.ssm);
  int64_t ssm_examined = 0;
  const double ssm_agreed = MaxRel(ds_ssm, hs_ssm, &ssm_examined);
  MESSAGE("carried SSM state device-vs-host: " << ssm_agreed << " over " << ssm_examined
                                               << " elements");
  REQUIRE(ssm_examined ==
          p.mamba_num_heads * p.mamba_head_dim * p.ssm_state_size);
  REQUIRE(ssm_examined > 0);
  const std::vector<float> ssm_zero(hs_ssm.size(), 0.0F);
  int64_t ssm_sep_examined = 0;
  const double ssm_sep = MaxRel(ssm_zero, hs_ssm, &ssm_sep_examined);
  MESSAGE("separation of a DROPPED SSM state: " << ssm_sep << " over " << ssm_sep_examined
                                                << " elements");
  REQUIRE(ssm_sep_examined == ssm_examined);
  REQUIRE(ssm_sep > 0.0);
  const double ssm_band = ssm_sep / 2.0;
  MESSAGE("accepting the SSM state at band " << ssm_band);
  CHECK(ssm_agreed < ssm_band);
  INFO("does the band " << ssm_band << " REJECT a dropped SSM state?");
  CHECK(ssm_sep >= ssm_band);

  // ── THE SECOND LEG: MEASURED AND REPORTED, WITH ITS OWN VERDICT ─────────
  // Both arms consume THEIR OWN carried state, so this compounds the noise floor
  // above with the state disagreement above. It carries an assertion only when
  // the fixture can resolve one — that is, when a dropped carry separates by
  // more than twice the measured noise floor. The condition is evaluated and
  // PRINTED either way, so "no assertion" is a stated measurement rather than a
  // silent hole.
  const std::vector<float> host1 = vllm::NemotronHMamba2Mixer(w, p, x1, T1, dt, hq, &hs);
  const std::vector<float> dev1 =
      vllm::NemotronHMamba2MixerDeviceHostIO(w, p, x1, T1, dt, dq, &ds);
  REQUIRE(host1.size() == static_cast<size_t>(T1 * H));
  REQUIRE(dev1.size() == host1.size());
  int64_t examined = 0;
  const double agreed = MaxRel(dev1, host1, &examined);
  REQUIRE(examined == T1 * H);
  int64_t sep_examined = 0;
  const double separation = MaxRel(fresh1_host, host1, &sep_examined);
  REQUIRE(sep_examined == examined);
  MESSAGE("second leg: agreed " << agreed << " ; dropped-carry separation " << separation
                                << " ; noise floor " << noise_floor << " over " << examined
                                << " elements");
  const bool resolvable = separation > 2.0 * noise_floor;
  // ★ std::string, NOT a char* ternary. doctest stringifies a `const char*` as a
  // BOOL, so `<< (cond ? "yes" : "no ...")` prints `1` whichever branch is live
  // -- the Thor run printed exactly that, and this line's entire job is to make
  // "no assertion was made here" a STATED result rather than a silent hole. A
  // diagnostic that cannot say what it means is the same class of defect as the
  // band it is reporting on. Reproduced and fixed against doctest 2.5.2:
  // `MESSAGE("x " << (false ? "yes" : "prose"))` prints `x 1`, and the
  // std::string form prints `x prose`.
  const std::string resolvable_verdict =
      resolvable ? std::string("yes")
                 : std::string("no -- the separation is inside the noise, so the "
                               "assertion is on the STATE above");
  MESSAGE("is a dropped carry RESOLVABLE from the second leg's output here? "
          << resolvable_verdict);
  if (resolvable) {
    const double band = separation / 2.0;
    MESSAGE("accepting the second leg at band " << band);
    CHECK(agreed < band);
    INFO("does the band " << band << " REJECT a dropped carry?");
    CHECK(separation >= band);
  }
}

// A separate case so a `-tc` run can select it alone. Same no-comma rule.
TEST_CASE("NemotronH A2-Q1: the device Mamba2 arm refuses a dense projection rather than reading garbage") {
  Queue dq{Device{DeviceType::kCPU, 0}, nullptr};
  if (!TryCudaQueue(&dq)) {
    NoteDeviceSkip("device Mamba2 refuses a dense projection");
    return;
  }
  const NemotronHParams p = MambaParams();
  const DType dt = DType::kBF16;
  const int64_t T = 4;
  const int64_t H = p.hidden_size;
  const std::vector<float> x = SynthVec(static_cast<size_t>(T * H), 78, 0.5F);

  // A layer whose `in_proj` is DENSE, which is what `BuildTiny` produces and what
  // an unquantized NemotronH would load. The whole-forward path takes the host
  // bounce for exactly this case; the block entry point refuses, because a direct
  // caller asked for the device.
  NemotronHMambaWeights dense = MakeFp8Mamba(p, dt);
  dense.in_proj = OwnF32(SynthVec(static_cast<size_t>(p.in_proj_out_features() * H), 5, 0.3F),
                         dt, {p.in_proj_out_features(), H});
  CHECK_THROWS(vllm::NemotronHMamba2MixerDeviceHostIO(dense, p, x, T, dt, dq));

  // And an `input_scale` the checkpoint never shipped: `Fp8Weight` has no
  // `has_input_scale` field, so the conversion must REFUSE rather than let a
  // defaulted divisor reach the activation quant (spec §4.1).
  NemotronHMambaWeights noscale = MakeFp8Mamba(p, dt);
  noscale.in_proj.has_input_scale = false;
  CHECK_THROWS(vllm::NemotronHMamba2MixerDeviceHostIO(noscale, p, x, T, dt, dq));
}

// THE RESIDENCY IS THE POINT OF THE WHOLE UNIT, and it is not visible in any
// numeric comparison: an arm that re-uploaded the 890 MB fp8 tower on every step
// would return exactly the same numbers as one that uploads it once. So it is
// asserted directly, against the geometry, through the ONE counter the tree
// already keeps for this (`load_stats::AddDeviceUpload`, which
// `dense_attn::ResidentWeight` and `ResidentNvfp4` both call).
//
// This is also the reachability instrument this row needs. `dense_fp8::
// ResidentFp8` does NOT account its own upload — that is issue #974 and A2-Q1
// does not change the shared header — so A2-Q1 accounts what IT uploads at the
// site that causes it. A step that took the host bounce instead uploads nothing,
// and the delta is 0.
TEST_CASE("NemotronH A2-Q1: the FP8 mamba tower uploads ONCE and the second call uploads nothing") {
  Queue dq{Device{DeviceType::kCPU, 0}, nullptr};
  if (!TryCudaQueue(&dq)) {
    NoteDeviceSkip("device Mamba2 residency and upload accounting");
    return;
  }
  const NemotronHParams p = MambaParams();
  const DType dt = DType::kBF16;
  const int64_t T = 4;
  const int64_t H = p.hidden_size;
  const int64_t I = p.mamba_intermediate_size();
  const int64_t Cd = p.conv_dim();
  const int64_t Kw = p.conv_kernel;
  const int64_t Hh = p.mamba_num_heads;
  const NemotronHMambaWeights w = MakeFp8Mamba(p, dt);
  const std::vector<float> x = SynthVec(static_cast<size_t>(T * H), 79, 0.5F);

  // What the arm OWES the device, derived from the geometry rather than read
  // back from the run: the two e4m3 towers, the conv weight and bias, the three
  // f32 [num_heads] recurrence parameters, and the gated-norm weight.
  const uint64_t esz = static_cast<uint64_t>(vt::SizeOf(dt));
  const uint64_t expect =
      static_cast<uint64_t>(p.in_proj_out_features() * H) +   // in_proj  e4m3
      static_cast<uint64_t>(H * I) +                          // out_proj e4m3
      static_cast<uint64_t>(Cd * Kw) * esz +                  // conv1d.weight
      static_cast<uint64_t>(Cd) * esz +                       // conv1d.bias
      static_cast<uint64_t>(Hh) * 4U * 3U +                   // A, D, dt_bias (f32)
      static_cast<uint64_t>(I) * esz;                         // norm.weight

  const uint64_t before = vllm::load_stats::Snapshot().device_upload_bytes;
  (void)vllm::NemotronHMamba2MixerDeviceHostIO(w, p, x, T, dt, dq);
  const uint64_t after_first = vllm::load_stats::Snapshot().device_upload_bytes;
  (void)vllm::NemotronHMamba2MixerDeviceHostIO(w, p, x, T, dt, dq);
  const uint64_t after_second = vllm::load_stats::Snapshot().device_upload_bytes;

  const uint64_t first = after_first - before;
  const uint64_t second = after_second - after_first;
  MESSAGE("device upload: first call " << first << " B, second call " << second
                                       << " B, expected " << expect << " B");
  CHECK(first == expect);
  CHECK(second == 0U);
}

// ─── the ONE substitution, gated on a box with no GPU at all ────────────────
//
// Every case above needs a device, so on a GPU-less box this whole file is a
// skip and gates nothing. That is a real hole: the device arm replaces the host
// arm's three `SliceCols` column copies (nemotron_h.cpp:333) with ONE
// `vt::QkvSplit`, and the `zxbcdt` widths are the only place that substitution
// can go wrong. `mamba_mixer2.py:692-696` reads xBC and dt off the TAIL and
// `:583` reads the gate off the HEAD, so an offset shifted by one column is
// finite, correctly shaped and plausible -- mutation Q1-M5.
//
// `vt::QkvSplit` is registered on CPU (ops.h:3512, "CPU + CUDA"), so the
// geometry is checkable everywhere even though the arithmetic is not. This case
// therefore runs in CI and on a laptop, and it is the only thing in this file
// that does.
//
// ★ BE PRECISE ABOUT WHAT IT PROVES. It pins the OP CONTRACT the device arm
// depends on — that three outputs of widths (I, conv_dim, num_heads) take the
// head, the middle and the tail of a `[T, in_proj_out_features]` row, in that
// order. It does NOT pin the production CALL SITE, because it issues its own
// `QkvSplit` rather than reaching `NemotronHMamba2MixerDevice`: reordering the
// arguments in the device arm would not red this case. The device numeric case
// above is what covers the call site, and it needs a GPU. Saying otherwise would
// make this look like a cheap substitute for a gate it cannot replace.
//
// The reference is INDEPENDENT and in-test: it re-derives the column ranges from
// the params rather than calling the file-local `SliceCols` it is checking
// against, so the two cannot agree by construction.
TEST_CASE("NemotronH A2-Q1: the device zxbcdt split lands on the same columns the host arm slices") {
  const NemotronHParams p = MambaParams();
  const int64_t I = p.mamba_intermediate_size();
  const int64_t Cd = p.conv_dim();
  const int64_t Hh = p.mamba_num_heads;
  const int64_t proj = p.in_proj_out_features();
  REQUIRE(I + Cd + Hh == proj);

  const int64_t T = 3;
  // Each element encodes its own (row, column), so a split that reads one column
  // early or late lands on a DIFFERENT value rather than a coincidentally equal
  // one -- the same reason the MoE fixture refuses nibble-symmetric bytes.
  std::vector<float> fused(static_cast<size_t>(T * proj));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < proj; ++c) {
      fused[static_cast<size_t>(t * proj + c)] =
          static_cast<float>(t) * 1000.0F + static_cast<float>(c);
    }
  }
  std::vector<float> z(static_cast<size_t>(T * I), -1.0F);
  std::vector<float> xbc(static_cast<size_t>(T * Cd), -1.0F);
  std::vector<float> dt(static_cast<size_t>(T * Hh), -1.0F);

  const Device cpu{DeviceType::kCPU, 0};
  Queue q{cpu, nullptr};
  vt::Tensor tf = vt::Tensor::Contiguous(fused.data(), DType::kF32, cpu, {T, proj});
  vt::Tensor tz = vt::Tensor::Contiguous(z.data(), DType::kF32, cpu, {T, I});
  vt::Tensor tx = vt::Tensor::Contiguous(xbc.data(), DType::kF32, cpu, {T, Cd});
  vt::Tensor td = vt::Tensor::Contiguous(dt.data(), DType::kF32, cpu, {T, Hh});
  vt::QkvSplit(q, tz, tx, td, tf);

  // The independent reference: z off the HEAD, then xBC, then dt off the TAIL.
  int64_t examined = 0;
  int64_t wrong = 0;
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < I; ++c, ++examined)
      if (z[static_cast<size_t>(t * I + c)] != fused[static_cast<size_t>(t * proj + c)]) ++wrong;
    for (int64_t c = 0; c < Cd; ++c, ++examined)
      if (xbc[static_cast<size_t>(t * Cd + c)] != fused[static_cast<size_t>(t * proj + I + c)])
        ++wrong;
    for (int64_t c = 0; c < Hh; ++c, ++examined)
      if (dt[static_cast<size_t>(t * Hh + c)] !=
          fused[static_cast<size_t>(t * proj + I + Cd + c)])
        ++wrong;
  }
  MESSAGE("zxbcdt split: " << wrong << " wrong of " << examined << " elements examined");
  // ASSERTED AGAINST THE GEOMETRY, not against a container's size: a loop that
  // silently shortened would otherwise report `0 wrong` and read as a pass.
  REQUIRE(examined == T * proj);
  CHECK(wrong == 0);

  // ── THE GUARD IS A PROPERTY ─────────────────────────────────────────────
  // The SAME comparison, against a reference whose `dt` offset is shifted by one
  // column, must come out REJECTED. If it does not, the check above cannot see
  // Q1-M5 either.
  int64_t shifted_wrong = 0;
  int64_t shifted_examined = 0;
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < Hh; ++c, ++shifted_examined) {
      if (dt[static_cast<size_t>(t * Hh + c)] !=
          fused[static_cast<size_t>(t * proj + I + Cd - 1 + c)])
        ++shifted_wrong;
    }
  }
  REQUIRE(shifted_examined == T * Hh);
  INFO("does the comparison REJECT a dt offset shifted by one column?");
  CHECK(shifted_wrong > 0);
}
