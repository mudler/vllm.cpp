// NemotronH A2-Q2a (#810, parent #517) — the DEVICE MoE arm on a SYNTHETIC
// NVFP4 fixture.
//
// ─── WHY THIS FILE EXISTS AT ALL ────────────────────────────────────────────
//
// The A2-Q2 spec's §5.1 gate is per-block numeric equivalence on the REAL
// 21 GB checkpoint, and that is still owed. This file is the CHEAP arm in front
// of it, and it is not a substitute:
//
//   * `test_nemotron_h_forward.cpp`'s `BuildTiny` fixture is ALL `kDense`
//     (:892 -> `PackMoe` :873 -> `Own(...)`), so nothing in the existing device
//     suite can reach the NVFP4 arena at all. Without this file the real
//     checkpoint would be the ONLY instrument, and every RED would cost a GB10
//     window plus a 21 GB load.
//   * Three earlier GB10 attempts on this row were VOID for reasons a cheap
//     local arm would have caught first. Spending a scarce window to discover a
//     flipped nibble order is the mistake this file prevents.
//
// ─── THE GEOMETRY IS NOT ARBITRARY ──────────────────────────────────────────
//
// Marlin refuses a shape it has no thread config for: `min_thread_n` and
// `min_thread_k` are both 64 (marlin.cuh:26-27) and `is_valid_config` requires
// `prob_k % thread_k == 0 && prob_n % thread_n == 0` (marlin_mm_moe.cu:245),
// over configs {128,128,256}, {64,128,128}, {128,64,128}. `TinyParams`
// (hidden_size 24, moe_intermediate_size 10) satisfies NONE of them, which is
// why this file cannot reuse it.
//
// The shape below is the smallest that resolves on BOTH routed GEMMs:
//   up   K=H=128, N=I=64    -> {128, 64, 128}   (128%128==0, 64%64==0)
//   down K=I=64,  N=H=128   -> {64, 128, 128}   (64%64==0, 128%128==0)
//   shared up/down K=N=128  -> {128, 128, 256}
// Each has at least one valid config, so a refusal here is about VALUES, never
// about a shape the kernel declined to launch.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::NemotronHBlock;
using vllm::NemotronHExpertWeights;
using vllm::NemotronHMoeWeights;
using vllm::NemotronHOwned;
using vllm::NemotronHParams;
using vllm::NemotronHWeightForm;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// The fixture's NVFP4 constants, taken verbatim from the form suite
// (test_nemotron_h_quantized_forms.cpp:155-158) so the two files describe the
// same bytes. `kWeightScale2` is deliberately NOT 1.0: a weight_scale_2 of 1 is
// invisible to a mutation that ignores it.
const uint8_t kGroupScaleA = 0x38;  // fp8-e4m3 1.0
const uint8_t kGroupScaleB = 0x40;  // fp8-e4m3 2.0
const float kWeightScale2 = 0.25F;

// [rows, cols/2] packed nibbles, LOW nibble first (the torchao/ModelOpt
// convention the loader reads, nemotron_h_weights.cpp:583-600). Never
// nibble-symmetric within a byte, so swapping the halves cannot leave a pair
// unchanged — that is what arms the nibble-order mutation.
std::vector<uint8_t> PackedNibbles(int64_t rows, int64_t cols, uint32_t salt) {
  std::vector<uint8_t> p(static_cast<size_t>(rows * cols / 2));
  for (size_t b = 0; b < p.size(); ++b) {
    const uint8_t lo = static_cast<uint8_t>((b * 5U + salt) % 16U);
    uint8_t hi = static_cast<uint8_t>((b * 3U + salt + 7U) % 16U);
    if ((hi & 0x07U) == (lo & 0x07U)) hi = static_cast<uint8_t>((hi + 1U) % 16U);
    p[b] = static_cast<uint8_t>(lo | (hi << 4));
  }
  return p;
}

// One fp8 group scale per 16 inputs, alternating so no two adjacent groups of a
// row share a value — a transposed or mis-strided scale grid therefore lands on
// a DIFFERENT value rather than a coincidentally equal one.
std::vector<uint8_t> GroupScales(int64_t rows, int64_t cols) {
  std::vector<uint8_t> s(static_cast<size_t>(rows * cols / 16));
  for (size_t i = 0; i < s.size(); ++i) s[i] = (i % 2 == 0) ? kGroupScaleA : kGroupScaleB;
  return s;
}

NemotronHOwned MakeNvfp4(int64_t rows, int64_t cols, uint32_t salt, DType logical) {
  NemotronHOwned w;
  w.form = NemotronHWeightForm::kNvfp4W4A16G16;
  w.dtype = logical;
  w.shape = {rows, cols};
  w.bytes = PackedNibbles(rows, cols, salt);
  w.scale = GroupScales(rows, cols);
  w.global_scale = kWeightScale2;
  return w;
}

NemotronHOwned OwnF32(const std::vector<float>& v, DType dt,
                      std::vector<int64_t> shape) {
  return NemotronHOwned::FromF32(v, dt, std::move(shape));
}

// The same deterministic generator the forward suite uses (:244), so a value
// here is reproducible and independent of any RNG seeding order.
std::vector<float> SynthVec(size_t n, uint32_t salt, float k) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const double a = static_cast<double>(i);
    const double b = static_cast<double>(salt);
    v[i] = static_cast<float>(k * std::sin(0.7 * a + 1.3 * b + 0.21 * a * b));
  }
  return v;
}

// A Marlin-legal NemotronH MoE geometry. Everything not listed is irrelevant to
// a single MoE block and is left at whatever keeps `NemotronHParams` coherent.
NemotronHParams MoeParams() {
  NemotronHParams p;
  p.hidden_size = 128;
  p.moe_intermediate_size = 64;
  p.n_routed_experts = 8;
  p.num_experts_per_tok = 2;
  p.n_group = 1;
  p.topk_group = 1;
  p.moe_shared_expert_intermediate_size = 128;
  p.n_shared_experts = 1;
  p.routed_scaling_factor = 2.5;  // the released value; NOT 1, so it is observable
  p.norm_topk_prob = true;
  p.vocab_size = 32;
  p.intermediate_size = 64;
  p.mlp_hidden_act = "relu2";
  p.layers_block_type = {NemotronHBlock::kMoe};
  return p;
}

NemotronHMoeWeights MakeNvfp4Moe(const NemotronHParams& p, DType dt) {
  NemotronHMoeWeights w;
  const int64_t E = p.n_routed_experts;
  const int64_t H = p.hidden_size;
  const int64_t I = p.moe_intermediate_size;
  const int64_t Is = p.moe_shared_expert_intermediate_size * p.n_shared_experts;
  // f32 on BOTH arms — `force_fp32_compute=True` (nemotron_h.py:150-156).
  w.gate = OwnF32(SynthVec(static_cast<size_t>(E * H), 1, 0.35F), DType::kF32, {E, H});
  w.e_score_correction_bias =
      OwnF32(SynthVec(static_cast<size_t>(E), 2, 0.4F), DType::kF32, {E});
  for (int64_t e = 0; e < E; ++e) {
    NemotronHExpertWeights ew;
    ew.up_proj = MakeNvfp4(I, H, static_cast<uint32_t>(10 + e), dt);
    ew.down_proj = MakeNvfp4(H, I, static_cast<uint32_t>(40 + e), dt);
    w.experts.push_back(std::move(ew));
  }
  w.shared.up_proj = MakeNvfp4(Is, H, 90, dt);
  w.shared.down_proj = MakeNvfp4(H, Is, 91, dt);
  w.has_shared = true;
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
                      << "': no CUDA device on this box. A2-Q2a's device MoE arm "
                         "gates on dgx.casa (GB10 sm_121a); a result from a "
                         "GPU-less box is not an A2-Q2a result.");
  CHECK(true);  // the skip path ran and said so
}

double MaxAbs(const std::vector<float>& v) {
  double m = 0.0;
  for (float x : v) m = std::max(m, std::abs(static_cast<double>(x)));
  return m;
}

// The largest relative deviation between two answers, measured element-wise
// against a scale-aware denominator. Returned rather than asserted, so the case
// can REPORT what the two arms actually agree to and then decide.
// ★ IT REPORTS HOW MANY ELEMENTS IT EXAMINED, AND EVERY CALLER ASSERTS THAT.
// A maximum over ZERO elements is 0.0 — and 0.0 is also exactly what a
// bit-exact comparison prints. So the deviation ALONE cannot distinguish "the
// two arms agree exactly over 512 elements" from "the loop ran over nothing".
// The first GB10 run of this file printed `worst relative deviation: 0`, and
// that ambiguity was real rather than theoretical: nothing on that line
// separated the best possible result from a mute instrument. `examined` is what
// makes a 0 mean something.
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

}  // namespace

// ─── the gate ───────────────────────────────────────────────────────────────
//
// NO COMMA IN THIS NAME, EVER. doctest's `-tc` filter splits on commas, so a
// comma here makes a targeted run select ZERO cases, print `SUCCESS!` and exit
// 0 — a whole mutation pass once read GREEN that way, including the row that
// deleted the guard. The case-count assertion below is the second half of that
// defence.
TEST_CASE("NemotronH A2-Q2a: the device MoE block matches the host reference on NVFP4 experts") {
  Queue dq{Device{DeviceType::kCPU, 0}, nullptr};
  if (!TryCudaQueue(&dq)) {
    NoteDeviceSkip("device MoE block vs host reference");
    return;
  }
  const NemotronHParams p = MoeParams();
  Queue hq{Device{DeviceType::kCPU, 0}, nullptr};
  const DType dt = DType::kBF16;  // Marlin's a/c contract (ops.cpp:879)
  const int64_t H = p.hidden_size;

  const NemotronHMoeWeights w = MakeNvfp4Moe(p, dt);

  // ★ T == 1 IS THE DECODE SHAPE, AND UNTIL #1157 THIS CASE NEVER RAN IT.
  // The widths here were 4 and 2, both of them PREFILL shapes. Every token
  // after the first comes out of a step with exactly one token, so the arm this
  // model spends its whole decode in was the one width the gate did not cover —
  // and `MarlinMoeAlignBlockSizeSelect` / `MarlinMoeAlignSizes` take different
  // branches at a token count below the expert count, which is what T=1 with
  // 128 experts is. A width loop rather than a third copy, so the three cannot
  // drift apart, and the count is asserted afterwards: a loop that ran over
  // nothing would otherwise report a clean pass.
  int64_t widths_covered = 0;
  for (const int64_t T : {int64_t{1}, int64_t{2}, int64_t{4}}) {
  INFO("token count T=" << T);
  const std::vector<float> x = SynthVec(static_cast<size_t>(T * H), 77, 0.5F);

  // The HOST arm dequantizes each touched expert to bf16 and runs the per-pair
  // MatmulBT loop (nemotron_h.cpp:780-802). The DEVICE arm dequantizes inside
  // the Marlin kernel and accumulates in f32. They are NOT bit-identical by
  // construction, which is exactly why the band below is measured rather than
  // chosen.
  const std::vector<float> host = vllm::NemotronHMoeMixer(w, p, x, T, dt, hq);
  const std::vector<float> dev =
      vllm::NemotronHMoeBlockDeviceHostIO(w, p, x, T, dt, dq);

  REQUIRE(host.size() == static_cast<size_t>(T * H));
  REQUIRE(dev.size() == host.size());

  // ── THE BAND IS MEASURED IN THIS RUN, NOT STORED ──────────────────────────
  // A band chosen up front is how a 3e-2 bf16 tolerance came to sit ABOVE a
  // 2.11e-2 defect on this very row and accept a wrong answer. So: report what
  // the two arms agree to, and derive the acceptance from the fixture's own
  // separation rather than from a constant anyone can widen.
  int64_t examined = 0;
  const double agreed = MaxRel(dev, host, &examined);
  MESSAGE("device-vs-host worst relative deviation: " << agreed << " over " << examined
                                                     << " elements");
  // THE COUNT IS THE REPORT. Without this the `agreed` value below cannot be
  // read at all — see MaxRel. Asserted against the geometry rather than against
  // `dev.size()`, so a comparison that silently shortened would still red.
  REQUIRE(examined == T * H);
  REQUIRE(examined > 0);

  // The SEPARATION this fixture can resolve: perturb the host answer by one
  // routed-scale step (the coarsest defect this block can carry) and measure how
  // far that moves it, through the SAME arithmetic and the same element count.
  std::vector<float> perturbed = host;
  const double step = 1.0 / static_cast<double>(p.routed_scaling_factor);
  for (float& v : perturbed) v = static_cast<float>(static_cast<double>(v) * step);
  int64_t sep_examined = 0;
  const double separation = MaxRel(perturbed, host, &sep_examined);
  MESSAGE("separation of a routed-scale defect: " << separation << " over " << sep_examined
                                                  << " elements");
  // The separation must be measured over the SAME population as the agreement,
  // or the two numbers are not comparable and the band between them is fiction.
  REQUIRE(sep_examined == examined);

  // ── THE BAND MUST ADMIT EXACT AGREEMENT ───────────────────────────────────
  // The first GB10 run took the geometric mean, `sqrt(agreed * separation)`.
  // That is right in spirit — both operands measured in this run — but it
  // DEGENERATES precisely when the arms agree exactly: `sqrt(0 * 0.6) == 0`, and
  // no value satisfies a strict `agreed < 0`. It failed on the best possible
  // outcome, which is a defect in the instrument and not in the product.
  //
  // Half the measured separation has the property the geometric mean was reaching
  // for and keeps it at `agreed == 0`: strictly above any agreement this fixture
  // can show, strictly below the defect it must reject. NOT a relaxation of `<`
  // to `<=`, which would accept a band of 0 and pass even if separation were 0
  // too — the failure this comment exists to keep closed.
  REQUIRE(separation > 0.0);
  const double band = separation / 2.0;
  MESSAGE("accepting at band " << band);

  CHECK(agreed < band);

  // ── THE GUARD IS A PROPERTY, NOT A TWIN ───────────────────────────────────
  // The perturbed answer, run through the SAME comparison that accepted the real
  // one, must come out REJECTED. Widen `band` past the separation and this line
  // reds — which is the historical hole, reproduced by construction.
  int64_t guard_examined = 0;
  const double guard = MaxRel(perturbed, host, &guard_examined);
  REQUIRE(guard_examined == examined);
  INFO("does the band " << band << " REJECT a routed-scale defect?");
  CHECK(guard >= band);
  ++widths_covered;
  }
  // Three widths, or the loop did not run the one this case was extended for.
  REQUIRE(widths_covered == 3);
}

// A separate case so a `-tc` run can select it alone. Same no-comma rule.
TEST_CASE("NemotronH A2-Q2a: the device MoE arm refuses a dense expert rather than reading garbage") {
  Queue dq{Device{DeviceType::kCPU, 0}, nullptr};
  if (!TryCudaQueue(&dq)) {
    NoteDeviceSkip("device MoE refuses a dense expert");
    return;
  }
  const NemotronHParams p = MoeParams();
  const DType dt = DType::kBF16;
  const int64_t T = 2;
  const int64_t H = p.hidden_size;

  // A layer whose experts are DENSE, which is what `BuildTiny` produces and what
  // an unquantized NemotronH would load. The arena is built from the NVFP4 form
  // alone, so this must refuse BY NAME rather than repack whatever bytes it
  // finds. The whole-forward path takes the host bounce for exactly this case;
  // the block entry point refuses, because a direct caller asked for the device.
  NemotronHMoeWeights w = MakeNvfp4Moe(p, dt);
  const int64_t I = p.moe_intermediate_size;
  w.experts[0].up_proj = OwnF32(SynthVec(static_cast<size_t>(I * H), 5, 0.3F), dt, {I, H});

  const std::vector<float> x = SynthVec(static_cast<size_t>(T * H), 78, 0.5F);
  CHECK_THROWS(vllm::NemotronHMoeBlockDeviceHostIO(w, p, x, T, dt, dq));
}
