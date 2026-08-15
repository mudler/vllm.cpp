// Nemotron-H — the MIXED_PRECISION memory forms, OFFLINE (#517, spec
// `.agents/specs/nemotron-h-model.md` §6d).
//
// ─── WHY THIS FILE EXISTS ───────────────────────────────────────────────────
//
// The weight loader's whole design claim is that every weight is held in the
// format the checkpoint SHIPS it in — NVFP4 W4A16 group-16 for the routed and
// shared experts and `lm_head`, FP8 W8A8 static for the 46 mamba projections,
// plain bf16/f32 for the rest — and that the HOST reference forward widens a
// quantized operand only TRANSIENTLY, at the GEMM call site. Two seams carry
// that claim: `NemotronHOwned::View`, which must REFUSE a non-dense weight
// rather than reinterpret packed nibbles as the model dtype, and
// `NemotronHOwned::DenseBf16`, the declared dequant.
//
// Every one of those seams was reachable ONLY through the 20.1 GiB checkpoint
// gate. On a runner with no `CHECKPOINT_ROOT` that gate returns early, so
// deleting the `View` guard outright left the whole suite green — and the guard
// is not decorative: without it `View` hands out a 128-element bf16 tensor over
// a 64-byte NVFP4 buffer, 192 bytes OUT OF BOUNDS. This file makes the seams
// gateable with no checkpoint at all, so they run on EVERY CI arm.
//
// ─── HOW THE REFERENCE IS DERIVED ───────────────────────────────────────────
//
// NOT from the code under test, and not through a helper both arms share
// ([[gate-comparing-shared-helper-proves-consistency-not-correctness]]). The
// expected values are computed here from upstream's own formula —
// `nvfp4_emulation_utils.dequantize_to_dtype` (swizzle=False) and ModelOpt's
// `W4A16_NVFP4` recipe, as transcribed in
// `model_loader/nvfp4_dequant.h` @ pin e24d1b24:
//
//   scale[o, g] = f32(weight_scale[o, g]) * weight_scale_2      // f32, MULTIPLIED
//   out[o, i]   = bf16( e2m1_lut[nibble(o, i)] * scale[o, i/16] )
//
// with element `2j` in the LOW nibble of byte `j` (torchao `pack_uint4`;
// `.agents/specs/nvfp4-nibble-order.md`), and for FP8 W8A8 static:
//
//   out[i] = bf16( f8_e4m3(weight[i]) * weight_scale )          // input_scale UNUSED
//
// The fp8-e4m3 decode and the bf16 round are written out here from their format
// definitions rather than called out of `vt`/`vllm`, so a defect in either
// cannot cancel itself out. Both are ALSO anchored by literal spot values
// (0x38 -> 1.0, 0x40 -> 2.0, nibble 0x7 -> 6.0) that are checked by hand below.
//
// TOLERANCE: there is none, deliberately. Every fixture value is a small dyadic
// rational, so every expected result is EXACT in bf16 and the comparison is on
// the bf16 BIT PATTERN. That sidesteps `doctest::Approx`'s ~1.19e-5 absolute
// floor ([[doctest-approx-scale-term-floor]]) entirely: a band that cannot be
// wrong is better than a band nobody re-derived.
//
// THE TWO DEFECTS THIS IS BUILT TO CATCH are §6d's M4 and M5 — the NVFP4 nibble
// order flipped to `kHighFirst`, and `weight_scale_2` read (so the accounting
// stays right) and then ignored. Both leave EVERY structural count of the
// checkpoint gate correct, so a counts-only gate calls them clean. The fixtures
// below are chosen so each one moves a value: the two nibbles of a byte never
// carry the same magnitude, the two groups of a row never carry the same scale,
// and `weight_scale_2` is never 1.
//
// CPU-only. No checkpoint, no golden, no speed claim.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // kNvfp4GroupSize
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace {

using vllm::NemotronHMlpWeights;
using vllm::NemotronHOwned;
using vllm::NemotronHParams;
using vllm::NemotronHWeightForm;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

// ─── the independent reference ──────────────────────────────────────────────

// IEEE fp8-e4m3fn -> f32, from the format (1 sign, 4 exp bias 7, 3 mantissa; no
// inf; 0x7F/0xFF NaN; subnormals at exponent 0). Written here, NOT called out of
// `vllm::F8E4M3ToF32`, because that is the decoder the dequant under test uses.
double RefF8E4M3(uint8_t byte) {
  const int sign = (byte & 0x80U) != 0 ? -1 : 1;
  const int exp = static_cast<int>((byte >> 3) & 0x0FU);
  const int man = static_cast<int>(byte & 0x07U);
  // Subnormal: (m/8) * 2^(1-bias) = (m/8) * 2^-6. Normal: (1 + m/8) * 2^(e-7).
  if (exp == 0) return sign * (man / 8.0) * std::ldexp(1.0, -6);
  return sign * (1.0 + man / 8.0) * std::ldexp(1.0, exp - 7);
}

// The E2M1 magnitude table, transcribed from nvfp4_emulation_utils.py:20-22.
// Index is the 3 low magnitude bits; bit 3 is the sign.
double RefE2M1(uint8_t nibble) {
  static const double kMag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  const double m = kMag[nibble & 0x07U];
  return (nibble & 0x08U) != 0 ? -m : m;
}

// f32 -> bf16 bit pattern, round-to-nearest-even, from the bf16 definition
// (truncate the low 16 bits with an RNE carry). Independent of `vt::F32ToBF16`.
uint16_t RefBf16Bits(double value) {
  const float f = static_cast<float>(value);
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1U;
  u += 0x7FFFU + lsb;
  return static_cast<uint16_t>(u >> 16);
}

// The whole ModelOpt W4A16_NVFP4 dequant, independently. `packed` is
// [rows, cols/2] with element 2j in the LOW nibble; `gscale` is [rows, cols/16]
// fp8-e4m3 bytes; `ws2` is the per-tensor `weight_scale_2`, MULTIPLIED.
std::vector<uint16_t> RefDequantNvfp4(const std::vector<uint8_t>& packed,
                                      const std::vector<uint8_t>& gscale, float ws2,
                                      int64_t rows, int64_t cols) {
  std::vector<uint16_t> out(static_cast<size_t>(rows * cols));
  for (int64_t o = 0; o < rows; ++o) {
    for (int64_t i = 0; i < cols; ++i) {
      const size_t byte = static_cast<size_t>(o * (cols / 2) + i / 2);
      const uint8_t nib = (i % 2 == 0) ? (packed[byte] & 0x0FU)
                                       : static_cast<uint8_t>(packed[byte] >> 4);
      const size_t g = static_cast<size_t>(o * (cols / vllm::kNvfp4GroupSize) +
                                           i / vllm::kNvfp4GroupSize);
      const double s = RefF8E4M3(gscale[g]) * static_cast<double>(ws2);
      out[static_cast<size_t>(o * cols + i)] = RefBf16Bits(RefE2M1(nib) * s);
    }
  }
  return out;
}

std::vector<uint16_t> RefDequantFp8(const std::vector<uint8_t>& bytes, float scale) {
  std::vector<uint16_t> out(bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) {
    out[i] = RefBf16Bits(RefF8E4M3(bytes[i]) * static_cast<double>(scale));
  }
  return out;
}

// ─── fixtures ───────────────────────────────────────────────────────────────
//
// Both nibbles of a byte carry DIFFERENT magnitudes (so a flipped nibble order
// moves every element), and the two groups of a row carry DIFFERENT fp8 scales
// (so a transposed or per-tensor group scale moves the second group).
const uint8_t kGroupScaleA = 0x38;  // 1.0
const uint8_t kGroupScaleB = 0x40;  // 2.0
const float kWeightScale2 = 0.25F;  // never 1: M5 ignores it, and that must move

// [rows, cols/2] packed nibbles, deterministic and never nibble-symmetric.
std::vector<uint8_t> PackedNibbles(int64_t rows, int64_t cols, uint32_t salt) {
  std::vector<uint8_t> p(static_cast<size_t>(rows * cols / 2));
  for (size_t b = 0; b < p.size(); ++b) {
    const uint8_t lo = static_cast<uint8_t>((b * 5U + salt) % 16U);
    uint8_t hi = static_cast<uint8_t>((b * 3U + salt + 7U) % 16U);
    // Never equal in MAGNITUDE, so swapping the two halves of the byte cannot
    // leave the pair unchanged.
    if ((hi & 0x07U) == (lo & 0x07U)) hi = static_cast<uint8_t>((hi + 1U) % 16U);
    p[b] = static_cast<uint8_t>(lo | (hi << 4));
  }
  return p;
}

// One fp8 group scale per 16 inputs, alternating so no two adjacent groups of a
// row share a value.
std::vector<uint8_t> GroupScales(int64_t rows, int64_t cols) {
  std::vector<uint8_t> s(static_cast<size_t>(rows * cols / vllm::kNvfp4GroupSize));
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

NemotronHOwned MakeFp8(int64_t rows, int64_t cols, uint32_t salt, DType logical) {
  NemotronHOwned w;
  w.form = NemotronHWeightForm::kFp8W8A8Static;
  w.dtype = logical;
  w.shape = {rows, cols};
  w.bytes.resize(static_cast<size_t>(rows * cols));
  for (size_t i = 0; i < w.bytes.size(); ++i) {
    // Stay inside the finite, non-NaN e4m3 range: exponent field 1..14.
    const uint8_t exp = static_cast<uint8_t>(1U + (i * 3U + salt) % 14U);
    const uint8_t man = static_cast<uint8_t>((i * 5U + salt) % 8U);
    const uint8_t sign = static_cast<uint8_t>(((i + salt) % 3U == 0) ? 0x80U : 0U);
    w.bytes[i] = static_cast<uint8_t>(sign | (exp << 3) | man);
  }
  w.global_scale = 0.5F;
  w.input_scale = 8.0F;  // CARRIED, never applied on the host path
  w.has_input_scale = true;
  return w;
}

// A dense bf16 weight whose CONTENT is the INDEPENDENT reference dequant of `q`.
// This is what makes the reach case sensitive to M4/M5: if `DenseBf16` flips a
// nibble order or drops `weight_scale_2`, this arm does not move with it.
NemotronHOwned DenseFromReference(const NemotronHOwned& q) {
  const int64_t rows = q.shape[0];
  const int64_t cols = q.shape[1];
  const std::vector<uint16_t> ref =
      q.form == NemotronHWeightForm::kNvfp4W4A16G16
          ? RefDequantNvfp4(q.bytes, q.scale, q.global_scale, rows, cols)
          : RefDequantFp8(q.bytes, q.global_scale);
  NemotronHOwned d;
  d.form = NemotronHWeightForm::kDense;
  d.dtype = DType::kBF16;
  d.shape = q.shape;
  d.bytes.resize(ref.size() * sizeof(uint16_t));
  std::memcpy(d.bytes.data(), ref.data(), d.bytes.size());
  return d;
}

std::vector<uint16_t> Bits(const std::vector<uint8_t>& bytes) {
  std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(uint16_t));
  return out;
}

// How many elements of two bf16 series differ, and the first index that does.
int CountDiff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
              size_t* first) {
  int n = 0;
  *first = a.size();
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    if (a[i] != b[i]) {
      if (n == 0) *first = i;
      ++n;
    }
  }
  return n;
}

}  // namespace

// ─── (0) the reference itself ───────────────────────────────────────────────
//
// The reference above is only worth something if it is right, so it is anchored
// to values that can be read off the format by hand before it judges anything.
TEST_CASE("NemotronH quantized forms: the independent reference is anchored by hand") {
  // fp8-e4m3fn: 0x38 = 0 0111 000 -> 2^(7-7) * 1.0 = 1.0.
  CHECK(RefF8E4M3(0x38) == doctest::Approx(1.0));
  CHECK(RefF8E4M3(0x40) == doctest::Approx(2.0));   // 0 1000 000 -> 2^1
  CHECK(RefF8E4M3(0x3C) == doctest::Approx(1.5));   // 0 0111 100 -> 1.5
  CHECK(RefF8E4M3(0x30) == doctest::Approx(0.5));   // 0 0110 000 -> 2^-1
  CHECK(RefF8E4M3(0xB8) == doctest::Approx(-1.0));  // sign bit set
  CHECK(RefF8E4M3(0x00) == doctest::Approx(0.0));
  // Subnormals (exponent field 0), which no fixture below reaches — anchored so
  // the reference is right rather than merely unexercised. 0x01 = (1/8)*2^-6.
  CHECK(RefF8E4M3(0x01) == doctest::Approx(0.001953125));
  CHECK(RefF8E4M3(0x07) == doctest::Approx(0.013671875));  // the largest subnormal
  CHECK(RefF8E4M3(0x08) == doctest::Approx(0.015625));     // the smallest normal
  // ...and the subnormal ladder is monotone into the normals.
  CHECK(RefF8E4M3(0x07) < RefF8E4M3(0x08));

  // E2M1: {0, .5, 1, 1.5, 2, 3, 4, 6}, sign in bit 3.
  CHECK(RefE2M1(0x0) == doctest::Approx(0.0));
  CHECK(RefE2M1(0x1) == doctest::Approx(0.5));
  CHECK(RefE2M1(0x7) == doctest::Approx(6.0));
  CHECK(RefE2M1(0xF) == doctest::Approx(-6.0));

  // bf16 bit patterns of exactly representable values.
  CHECK(RefBf16Bits(1.0) == 0x3F80);
  CHECK(RefBf16Bits(-1.0) == 0xBF80);
  CHECK(RefBf16Bits(0.0) == 0x0000);
  CHECK(RefBf16Bits(1.5) == 0x3FC0);
  CHECK(RefBf16Bits(6.0) == 0x40C0);
  // And it agrees with the runtime's own converter on those values, which is a
  // CONSISTENCY check on top of the hand anchors, not a substitute for them.
  CHECK(RefBf16Bits(1.5) == vt::F32ToBF16(1.5F));
  CHECK(RefBf16Bits(-6.0) == vt::F32ToBF16(-6.0F));
}

// ─── (1) View REFUSES a non-dense weight, BY NAME ───────────────────────────
//
// A view over packed nibbles typed as the model dtype is finite, correctly
// shaped, plausible garbage — no kernel, no shape check and no token gate can
// see it. Worse, the buffer is HALF the size the view claims: `View` on a
// [8, 16] NVFP4 weight would describe 128 bf16 elements (256 bytes) over a
// 64-byte payload, 192 bytes out of bounds.
TEST_CASE("NemotronHOwned::View refuses a weight held in its shipped quantized form") {
  const NemotronHOwned nvfp4 = MakeNvfp4(8, 16, 1, DType::kBF16);
  REQUIRE(nvfp4.bytes.size() == 64);      // [8, 16/2]
  REQUIRE(nvfp4.scale.size() == 8);       // [8, 16/16]
  REQUIRE_FALSE(nvfp4.IsDense());
  // What the refusal is protecting: the LOGICAL extent against the payload.
  REQUIRE(static_cast<size_t>(nvfp4.Numel()) * sizeof(uint16_t) ==
          nvfp4.bytes.size() * 4);

  bool threw = false;
  std::string msg;
  try {
    (void)nvfp4.View(Cpu());
  } catch (const std::runtime_error& e) {
    threw = true;
    msg = e.what();
  }
  CHECK(threw);
  // BY NAME: the message has to say which form it is holding and where to go,
  // or the refusal is just a crash with extra steps.
  CHECK(msg.find("NemotronHOwned::View") != std::string::npos);
  CHECK(msg.find("NVFP4") != std::string::npos);
  CHECK(msg.find("DenseBf16") != std::string::npos);

  const NemotronHOwned fp8 = MakeFp8(4, 16, 2, DType::kBF16);
  REQUIRE_FALSE(fp8.IsDense());
  bool threw_fp8 = false;
  try {
    (void)fp8.View(Cpu());
  } catch (const std::runtime_error&) {
    threw_fp8 = true;
  }
  CHECK(threw_fp8);

  // The dense form is still handed out, unchanged — the guard refuses the
  // quantized forms, not every weight.
  const NemotronHOwned dense = DenseFromReference(nvfp4);
  REQUIRE(dense.IsDense());
  const vt::Tensor t = dense.View(Cpu());
  CHECK(t.rank == 2);
  CHECK(t.shape[0] == 8);
  CHECK(t.shape[1] == 16);
  CHECK(t.dtype == DType::kBF16);
  CHECK(t.data == static_cast<const void*>(dense.bytes.data()));
}

// ─── (2) the NVFP4 dequant, against the independent reference ───────────────
TEST_CASE("NemotronHOwned::DenseBf16 reproduces the ModelOpt NVFP4 W4A16 g16 dequant") {
  // 3 rows x 32 cols = TWO groups per row, so the group scale is per-group and
  // not per-row or per-tensor, and 48 bytes of nibbles.
  const NemotronHOwned w = MakeNvfp4(3, 32, 11, DType::kBF16);
  REQUIRE(w.bytes.size() == 48);
  REQUIRE(w.scale.size() == 6);

  const std::vector<uint16_t> got = Bits(w.DenseBf16());
  const std::vector<uint16_t> want =
      RefDequantNvfp4(w.bytes, w.scale, w.global_scale, 3, 32);
  REQUIRE(got.size() == want.size());
  REQUIRE(got.size() == 96);
  size_t first = 0;
  const int diff = CountDiff(got, want, &first);
  // Plain decimal, NOT `std::hex`: doctest 2.5.2's MESSAGE stream renders the
  // manipulator as `{?}` and then leaves the stream in hex, which is how this
  // line first printed `got 0x{?}16000`.
  CHECK_MESSAGE(diff == 0,
                "first differing element "
                    << first << " of " << got.size() << ": got bf16 bits "
                    << static_cast<int>(first < got.size() ? got[first] : 0)
                    << ", want " << static_cast<int>(first < want.size() ? want[first] : 0)
                    << " (" << diff << " of " << got.size() << " differ)");

  // ── the three properties a counts-only gate cannot see ──────────────────

  // NIBBLE ORDER (§6d M4). Element 2j is the LOW nibble. Assert it on a byte
  // whose two halves are known, rather than inferring it from the bulk compare.
  NemotronHOwned probe;
  probe.form = NemotronHWeightForm::kNvfp4W4A16G16;
  probe.dtype = DType::kBF16;
  probe.shape = {1, 16};
  probe.bytes.assign(8, 0x00);
  probe.bytes[0] = 0x71;  // low nibble 0x1 -> 0.5, high nibble 0x7 -> 6.0
  probe.scale.assign(1, kGroupScaleA);  // 1.0
  probe.global_scale = 1.0F;
  std::vector<uint16_t> p = Bits(probe.DenseBf16());
  REQUIRE(p.size() == 16);
  CHECK(p[0] == RefBf16Bits(0.5));  // element 0 came from the LOW nibble
  CHECK(p[1] == RefBf16Bits(6.0));  // element 1 from the HIGH nibble
  CHECK(p[0] != p[1]);              // the pair is not symmetric, so a flip moves it

  // THE GROUP SCALE is per-16, and bound to the right group. Same nibbles in
  // both groups of a row, different fp8 scales -> the second group is 2x.
  NemotronHOwned two;
  two.form = NemotronHWeightForm::kNvfp4W4A16G16;
  two.dtype = DType::kBF16;
  two.shape = {1, 32};
  two.bytes.assign(16, 0x22);  // every element = 1.0 before scaling
  two.scale = {kGroupScaleA, kGroupScaleB};  // 1.0 then 2.0
  two.global_scale = 1.0F;
  const std::vector<uint16_t> t2 = Bits(two.DenseBf16());
  REQUIRE(t2.size() == 32);
  for (size_t i = 0; i < 16; ++i) CHECK(t2[i] == RefBf16Bits(1.0));
  for (size_t i = 16; i < 32; ++i) CHECK(t2[i] == RefBf16Bits(2.0));

  // `weight_scale_2` IS MULTIPLIED, not reciprocated and not ignored (§6d M5).
  NemotronHOwned scaled = two;
  scaled.global_scale = kWeightScale2;  // 0.25
  const std::vector<uint16_t> t3 = Bits(scaled.DenseBf16());
  REQUIRE(t3.size() == 32);
  for (size_t i = 0; i < 16; ++i) CHECK(t3[i] == RefBf16Bits(0.25));
  for (size_t i = 16; i < 32; ++i) CHECK(t3[i] == RefBf16Bits(0.5));
  // Reciprocated would be 4.0/8.0, ignored would be 1.0/2.0 — both distinct.
  CHECK(t3[0] != RefBf16Bits(4.0));
  CHECK(t3[0] != RefBf16Bits(1.0));

  // The weight KEEPS its packed form: dequant is transient, per call.
  CHECK(w.bytes.size() == 48);
  CHECK(w.form == NemotronHWeightForm::kNvfp4W4A16G16);
  CHECK(w.HostBytes() == 48 + 6);
}

// ─── (3) the FP8 W8A8 static arm, and the scale it does NOT apply ───────────
TEST_CASE("NemotronHOwned::DenseBf16 reproduces the FP8 W8A8 static dequant") {
  const NemotronHOwned w = MakeFp8(4, 16, 3, DType::kBF16);
  REQUIRE(w.bytes.size() == 64);

  const std::vector<uint16_t> got = Bits(w.DenseBf16());
  const std::vector<uint16_t> want = RefDequantFp8(w.bytes, w.global_scale);
  REQUIRE(got.size() == want.size());
  size_t first = 0;
  CHECK_MESSAGE(CountDiff(got, want, &first) == 0,
                "first differing element " << first);

  // `input_scale` is CARRIED, NOT APPLIED: nothing on the host path quantizes
  // the activation, so applying it here would scale the product by a factor
  // upstream applies to the OTHER operand. `has_input_scale` distinguishes
  // "the checkpoint shipped 1.0" from "no scale shipped".
  CHECK(w.has_input_scale);
  CHECK(w.input_scale == 8.0F);
  NemotronHOwned other = w;
  other.input_scale = 1.0F / 8.0F;
  const std::vector<uint16_t> got2 = Bits(other.DenseBf16());
  size_t f2 = 0;
  CHECK_MESSAGE(CountDiff(got, got2, &f2) == 0,
                "input_scale moved the dequant at element " << f2);

  // `weight_scale` IS applied. Doubling it doubles every non-zero element.
  NemotronHOwned doubled = w;
  doubled.global_scale = w.global_scale * 2.0F;
  const std::vector<uint16_t> got3 = Bits(doubled.DenseBf16());
  size_t f3 = 0;
  CHECK(CountDiff(got, got3, &f3) > 0);
  const std::vector<uint16_t> want3 = RefDequantFp8(w.bytes, w.global_scale * 2.0F);
  size_t f4 = 0;
  CHECK_MESSAGE(CountDiff(got3, want3, &f4) == 0, "first differing element " << f4);
}

// ─── (4) a quantized weight actually REACHES a GEMM ─────────────────────────
//
// The two cases above gate `DenseBf16` in isolation. This one gates the WIRING:
// `DenseFor`/`DenseCopy` are file-private to nemotron_h.cpp, so the only way to
// prove a quantized weight is widened at the GEMM call site (rather than viewed,
// refused, or silently skipped) is to run a mixer on one.
//
// The dense arm's content comes from the INDEPENDENT reference, never from
// `DenseBf16`, so a flipped nibble order or a dropped `weight_scale_2` breaks
// this case too instead of cancelling out. Both arms then feed byte-identical
// bf16 operands to the same `vt::MatmulBT`, so the outputs are bit-identical and
// the comparison needs no tolerance.
TEST_CASE("NemotronH forward: a quantized MLP is widened at the GEMM call site") {
  NemotronHParams p;
  p.hidden_size = 16;
  p.intermediate_size = 32;
  p.mlp_hidden_act = "relu2";
  p.mlp_bias = false;

  const NemotronHOwned up_q = MakeNvfp4(p.intermediate_size, p.hidden_size, 5,
                                        DType::kBF16);
  const NemotronHOwned down_q = MakeNvfp4(p.hidden_size, p.intermediate_size, 9,
                                          DType::kBF16);

  NemotronHMlpWeights quant;
  quant.up_proj = up_q;
  quant.down_proj = down_q;
  NemotronHMlpWeights dense;
  dense.up_proj = DenseFromReference(up_q);
  dense.down_proj = DenseFromReference(down_q);
  REQUIRE_FALSE(quant.up_proj.IsDense());
  REQUIRE(dense.up_proj.IsDense());

  const int64_t T = 3;
  std::vector<float> hidden(static_cast<size_t>(T * p.hidden_size));
  for (size_t i = 0; i < hidden.size(); ++i) {
    hidden[i] = static_cast<float>(static_cast<double>((i * 7) % 13) - 6.0) * 0.125F;
  }

  Queue q = CpuQ();
  const std::vector<float> from_quant =
      vllm::NemotronHMlpMixer(quant, p, hidden, T, DType::kBF16, q);
  const std::vector<float> from_dense =
      vllm::NemotronHMlpMixer(dense, p, hidden, T, DType::kBF16, q);
  REQUIRE(from_quant.size() == static_cast<size_t>(T * p.hidden_size));
  REQUIRE(from_dense.size() == from_quant.size());
  CHECK(std::memcmp(from_quant.data(), from_dense.data(),
                    from_quant.size() * sizeof(float)) == 0);

  // NOT VACUOUS: the block computes something, and a DIFFERENT quantized weight
  // gives a different answer. Otherwise an all-zeros forward would pass.
  bool nonzero = false;
  for (float v : from_quant) nonzero = nonzero || v != 0.0F;
  CHECK(nonzero);
  NemotronHMlpWeights other = quant;
  other.up_proj.global_scale = quant.up_proj.global_scale * 2.0F;
  const std::vector<float> moved =
      vllm::NemotronHMlpMixer(other, p, hidden, T, DType::kBF16, q);
  CHECK(std::memcmp(from_quant.data(), moved.data(),
                    from_quant.size() * sizeof(float)) != 0);

  // And the weight is STILL packed afterwards: the widening was transient, not a
  // dequantize-at-load in disguise.
  CHECK(quant.up_proj.form == NemotronHWeightForm::kNvfp4W4A16G16);
  CHECK(quant.up_proj.bytes.size() ==
        static_cast<size_t>(p.intermediate_size * p.hidden_size / 2));
}
