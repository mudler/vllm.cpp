// LTX-2.5 phase L6 — the quantized loaders.
//
// Row MODEL-DIFFUSION-LTX25, .agents/specs/ltx-2-5.md, issue #435.
//
// THREE KINDS OF GATE, and they are not interchangeable:
//
//  * REAL MANIFEST. ltx2_fp8_dit_manifest.inc / ltx2_nvfp4_te_manifest.inc are
//    the SHIPPED checkpoints' own safetensors headers — 6124 and 1688 entries,
//    names/dtypes/shapes, not one weight byte. Every claim this port makes about
//    what those files contain is asserted against them, so a claim cannot drift
//    from the artifact. This mirrors how MiniMax-H3 gated its GGUF and NVFP4
//    arms (test_minimax_h3.cpp:2348).
//  * REAL BYTES. ltx2_quant_goldens.inc carries a few hundred bytes read at
//    their own offsets out of those same files, with the expected values decoded
//    by TORCH. The fp8 half of both dequant paths is therefore held against an
//    implementation that is not ours.
//  * SYNTHETIC FILES. Whole-model materialization, missing-tensor refusals and
//    device staging need a file small enough to build in a test, so those are
//    written here from a deterministic byte stream whose header MIRRORS the real
//    layout (prefix, dtypes, packed widths, swizzled scale shapes).
//
// The NVFP4 DiT arm is gated on a SYNTHETIC file only. Lightricks' first-party
// NVFP4 DiT sits behind an un-accepted HF gate (HTTP 403) and was NOT
// downloaded; that is recorded as owed in .agents/porting-inventory.md rather
// than papered over with a fabricated manifest.
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <doctest/doctest.h>

// Issue #449: every max|diff| reduction here goes through the ONE hardened
// helper, where a non-finite operand on either side is a FAILURE. The two
// obvious spellings (`if (d > worst)` / `std::max`) are both NaN-BLIND, and this
// suite had seven copies of the first: injecting all-NaN for exactly the
// real-bytes probe geometry left `19 passed | 1664 passed | SUCCESS!`, byte for
// byte, with the function under test returning nothing but NaN.
#include "support/max_abs_diff.h"

#include "ltx2_fp8_dit_manifest.inc"
#include "ltx2_nvfp4_dit_manifest.inc"
#include "ltx2_nvfp4_te_manifest.inc"
#include "ltx2_quant_goldens.inc"

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vt/backend.h"

using vllm::Ltx2DitLoadOptions;
using vllm::Ltx2DitParams;
using vllm::Ltx2DitQuant;
using vllm::Ltx2TensorSpec;
using vllm::SafetensorsFile;

namespace {

// ---------------------------------------------------------------------------
// The deterministic byte stream, mirrored bit-for-bit by
// scripts/gen-ltx2-quant-goldens.py (fnv1a64 + splitmix64).
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& s) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::vector<uint8_t> RandBytes(const std::string& name, size_t count) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<uint8_t> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = static_cast<uint8_t>((SplitMix64(seed + i) >> 24) & 0xFFU);
  }
  return out;
}

float Bf16ToF32(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t F32ToBf16(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t rounded = bits + 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(rounded >> 16);
}

// True when `v` survives a bf16 store UNCHANGED, i.e. its low 16 mantissa bits
// are already zero. Used to state, as a gated fact rather than a comment, which
// fixtures do and do not reach the bf16 rounding regime at all.
bool IsBf16Exact(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return (bits & 0xFFFFU) == 0U;
}

// ---------------------------------------------------------------------------
// A synthetic .safetensors writer. Same shape as
// test_minimax_h3.cpp:509 WriteSafetensorsFromEntries.
// ---------------------------------------------------------------------------

struct StEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

void WriteSafetensors(const std::vector<StEntry>& entries, const std::string& path) {
  std::string header = "{";
  size_t offset = 0;
  for (size_t i = 0; i < entries.size(); ++i) {
    const StEntry& e = entries[i];
    if (i != 0) header += ",";
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t d = 0; d < e.shape.size(); ++d) {
      if (d != 0) header += ",";
      header += std::to_string(e.shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  FILE* fh = std::fopen(path.c_str(), "wb");
  REQUIRE(fh != nullptr);
  const uint64_t len = header.size();
  uint8_t le[8];
  for (int i = 0; i < 8; ++i) le[i] = static_cast<uint8_t>((len >> (8 * i)) & 0xFFU);
  std::fwrite(le, 1, 8, fh);
  std::fwrite(header.data(), 1, header.size(), fh);
  for (const StEntry& e : entries) std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
  std::fclose(fh);
}

std::string PackF32(const std::vector<float>& v) {
  std::string out(v.size() * sizeof(float), '\0');
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

std::string PackBf16(const std::vector<float>& v) {
  std::string out(v.size() * sizeof(uint16_t), '\0');
  for (size_t i = 0; i < v.size(); ++i) {
    const uint16_t b = F32ToBf16(v[i]);
    std::memcpy(out.data() + i * sizeof(uint16_t), &b, sizeof(b));
  }
  return out;
}

std::string PackBytes(const std::vector<uint8_t>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size());
}

// The forward swizzle, so the synthetic files carry a genuinely swizzled scale
// and the loader's inverse has something real to invert. Transcribed from the
// SAME source the header cites (vLLM qutlass_utils.py:177-179): source (r, c)
// with r = 128*rt + 32*a + s, c = 4*ct + q lands at
// ((((rt * ctiles) + ct) * 32 + s) * 4 + a) * 4 + q.
std::vector<uint8_t> SwizzleBlockScale(const std::vector<uint8_t>& linear, int64_t rows,
                                       int64_t cols) {
  REQUIRE(rows % 128 == 0);
  REQUIRE(cols % 4 == 0);
  const int64_t ctiles = cols / 4;
  std::vector<uint8_t> out(static_cast<size_t>(rows * cols), 0);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t rt = r / 128, rem = r % 128, a = rem / 32, s = rem % 32;
      const int64_t ct = c / 4, q = c % 4;
      out[static_cast<size_t>(((((rt * ctiles) + ct) * 32 + s) * 4 + a) * 4 + q)] =
          linear[static_cast<size_t>(r * cols + c)];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// A REDUCED-dimension LTX-2.5 DiT, written as a real checkpoint would be:
// `model.diffusion_model.` prefixed, FP8 or NVFP4 weights with their scale
// sidecars, BF16 biases and norms, F32 tables.
// ---------------------------------------------------------------------------

Ltx2DitParams TinyParams() {
  Ltx2DitParams p;
  p.num_layers = 2;
  p.num_attention_heads = 2;
  p.attention_head_dim = 64;   // dim = 128
  p.audio_num_attention_heads = 2;
  p.audio_attention_head_dim = 32;  // adim = 64
  p.in_channels = 16;
  p.out_channels = 16;
  p.audio_in_channels = 16;
  p.audio_out_channels = 16;
  p.cross_attention_dim = 128;
  p.audio_cross_attention_dim = 64;
  p.apply_gated_attention = true;
  p.cross_attention_adaln = true;
  p.use_prompt_adaln_single = false;
  p.ff_bias = false;
  p.audio_ff_bias = true;
  return p;
}

bool IsTable(const std::string& name) {
  return name.find("scale_shift_table") != std::string::npos;
}

// Deterministic f32 "true" value for element `i` of tensor `name`, so both the
// file and the expectation come from one rule.
float TrueValue(const std::string& name, size_t i) {
  const uint64_t u = SplitMix64(Fnv1a64(name) + i);
  return static_cast<float>(static_cast<double>(u >> 11) * 0x1p-53 * 2.0 - 1.0) * 0.05F;
}

// The FP8 arm's per-tensor scale. It is deliberately NOT a power of two, and
// that is the whole point: fp8-e4m3 carries 4 significant bits, so against a
// power-of-two scale (this fixture used 2^-8) every product is EXACTLY
// bf16-representable and the bf16 store never rounds. Under such a scale
// replacing round-nearest-even with truncation in DequantFp8ToBf16 changes
// nothing anywhere in this suite — measured, and it is why the real-bytes case
// above cannot gate the rounding mode either. This value carries a full f32
// significand, so 252 of the 256 fp8 encodings times it land off a bf16 grid
// point and the rounding mode becomes load-bearing. The whole-model FP8 test
// below counts how many elements actually reach that regime, so the guarantee
// cannot quietly evaporate if this constant is ever changed again.
constexpr float kSyntheticFp8Scale = 0.011234567F;

struct SyntheticDit {
  std::vector<StEntry> entries;
  // name (contract, unprefixed) -> the exact bf16 the loader must produce.
  std::map<std::string, std::vector<uint16_t>> expected;
  // How many FP8 elements the fixture holds, and how many of them need a real
  // bf16 round. The second is what makes the RNE claim load-bearing.
  int64_t fp8_elements = 0;
  int64_t fp8_rounding_elements = 0;
  // Must stay 0: a NaN weight is not something a real checkpoint stores, and it
  // makes a gate compare nan against nan.
  int64_t fp8_nonfinite_elements = 0;
};

SyntheticDit BuildSyntheticDit(const Ltx2DitParams& p, Ltx2DitQuant quant,
                               const std::vector<std::string>& extra_modules) {
  SyntheticDit out;
  const std::string pre = vllm::kLtx2DitCheckpointPrefix;
  for (const Ltx2TensorSpec& spec : vllm::EnumerateLtx2DitTensors(p)) {
    int64_t numel = 1;
    for (int64_t d : spec.shape) numel *= d;
    const bool table = IsTable(spec.name);
    const bool quantized = spec.shape.size() == 2 && !table;

    if (table) {
      std::vector<float> v(static_cast<size_t>(numel));
      for (size_t i = 0; i < v.size(); ++i) v[i] = TrueValue(spec.name, i);
      out.entries.push_back({pre + spec.name, "F32", spec.shape, PackF32(v)});
      continue;  // f32 tables are checked separately, against TrueValue directly
    }
    if (!quantized) {
      // Biases and q/k norms: BF16, stored as-is.
      std::vector<float> v(static_cast<size_t>(numel));
      std::vector<uint16_t> want(v.size());
      for (size_t i = 0; i < v.size(); ++i) {
        v[i] = TrueValue(spec.name, i);
        want[i] = F32ToBf16(v[i]);
      }
      out.entries.push_back({pre + spec.name, "BF16", spec.shape, PackBf16(v)});
      out.expected[spec.name] = want;
      continue;
    }

    const int64_t rows = spec.shape[0], cols = spec.shape[1];
    if (quant == Ltx2DitQuant::kFp8) {
      std::vector<uint8_t> raw = RandBytes(spec.name, static_cast<size_t>(numel));
      // Never emit the fp8-e4m3 NaN encodings, exactly as the NVFP4 arm below
      // already does. A NaN WEIGHT is not something a real checkpoint stores,
      // and it makes the gate compare nan against nan — which the old NaN-blind
      // reductions read as agreement (issue #449). Found by hardening them.
      for (uint8_t& b : raw) {
        if (b == 0x7FU || b == 0xFFU) b = 0x38U;
      }
      const float scale = kSyntheticFp8Scale;
      std::vector<uint16_t> want(raw.size());
      for (size_t i = 0; i < raw.size(); ++i) {
        // `want` rounds with this file's OWN round-nearest-even, which is a copy
        // of the rule and not a call into the code under test, so a rounding
        // change in DequantFp8ToBf16 shows up as a bit mismatch.
        const float product = vllm::F8E4M3ToF32(raw[i]) * scale;
        want[i] = F32ToBf16(product);
        ++out.fp8_elements;
        if (!std::isfinite(product)) ++out.fp8_nonfinite_elements;
        if (!IsBf16Exact(product)) ++out.fp8_rounding_elements;
      }
      out.entries.push_back({pre + spec.name, "F8_E4M3", spec.shape, PackBytes(raw)});
      out.entries.push_back({pre + spec.name + "_scale", "F32", {},
                             std::string(reinterpret_cast<const char*>(&scale), 4)});
      out.expected[spec.name] = want;
    } else {
      REQUIRE(cols % 16 == 0);
      const std::vector<uint8_t> packed =
          RandBytes(spec.name, static_cast<size_t>(rows * cols / 2));
      // The scale grid is [rows, cols/16]; the swizzle needs rows % 128 == 0 and
      // (cols/16) % 4 == 0, which a real checkpoint always satisfies. The tiny
      // config does not, so the synthetic NVFP4 arm pads the scale grid the way
      // torchao's own producer does.
      const int64_t groups = cols / 16;
      const int64_t prows = ((rows + 127) / 128) * 128;
      const int64_t pcols = ((groups + 3) / 4) * 4;
      std::vector<uint8_t> lin(static_cast<size_t>(prows * pcols), 0);
      const std::vector<uint8_t> live =
          RandBytes(spec.name + ".scale", static_cast<size_t>(rows * groups));
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g = 0; g < groups; ++g) {
          uint8_t b = live[static_cast<size_t>(r * groups + g)];
          if (b == 0x7F || b == 0xFF) b = 0x38;  // never emit the NaN encoding
          lin[static_cast<size_t>(r * pcols + g)] = b;
        }
      }
      const std::vector<uint8_t> sw = SwizzleBlockScale(lin, prows, pcols);
      const float scale2 = 0.0078125F;
      std::vector<uint8_t> lin_live(static_cast<size_t>(rows * groups));
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g = 0; g < groups; ++g) {
          lin_live[static_cast<size_t>(r * groups + g)] =
              lin[static_cast<size_t>(r * pcols + g)];
        }
      }
      // The synthetic NVFP4 DiT mirrors the SHIPPED one, which is Lightricks'
      // nvfp4-prequant: no torchao marker, the cuBLAS-PADDED scale framing, and
      // HIGH-nibble-first packing. It used to declare the to_blocked framing with
      // no marker — a combination NEITHER producer emits, which the discriminator
      // now refuses by name. That refusal is correct and this fixture was the
      // thing that was wrong: a synthetic file has to be a file some real producer
      // could have written, or it gates a shape nothing ships.
      std::vector<uint16_t> want(static_cast<size_t>(rows * cols));
      vllm::DequantNvfp4ToBf16(packed.data(), lin_live.data(), scale2, rows, cols,
                               want.data(), vllm::Nvfp4NibbleOrder::kHighFirst);
      out.entries.push_back({pre + spec.name, "U8", {rows, cols / 2}, PackBytes(packed)});
      out.entries.push_back({pre + spec.name + "_scale", "F8_E4M3", {prows, pcols},
                             PackBytes(sw)});
      out.entries.push_back({pre + spec.name + "_scale_2", "F32", {},
                             std::string(reinterpret_cast<const char*>(&scale2), 4)});
      out.expected[spec.name] = want;
    }
  }
  // Whatever unported family the caller wants present.
  for (const std::string& m : extra_modules) {
    const std::vector<float> v(4, 0.5F);
    out.entries.push_back({pre + m, "F32", {2, 2}, PackF32(v)});
  }
  return out;
}

// A path no CONCURRENT run can collide on. This box carries ~150 worktrees, and
// a fixed /tmp/ltx2_loader_fp8.safetensors means two suites running at once
// write and unlink each other's file — a failure that reads as a loader bug.
std::string TmpPath(const char* stem) {
  static const std::string unique = std::to_string(static_cast<long long>(::getpid())) + "_" +
                                    std::to_string(static_cast<unsigned long long>(
                                        std::chrono::steady_clock::now().time_since_epoch().count()));
  return std::string("/tmp/ltx2_loader_") + stem + "_" + unique + ".safetensors";
}

// ---------------------------------------------------------------------------
// Real-manifest helpers
// ---------------------------------------------------------------------------

std::vector<int64_t> ShapeOf(const int64_t* s, int32_t rank) {
  return std::vector<int64_t>(s, s + rank);
}

const vllm_test::Ltx25Fp8DitTensor* FindDit(const std::string& name) {
  for (const auto& t : vllm_test::kLtx25Fp8DitTensors) {
    if (name == t.name) return &t;
  }
  return nullptr;
}

const vllm_test::Ltx25Nvfp4TeTensor* FindTe(const std::string& name) {
  for (const auto& t : vllm_test::kLtx25Nvfp4TeTensors) {
    if (name == t.name) return &t;
  }
  return nullptr;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The RELATIVE companion to vllm_test::MaxAbsDiff, with the SAME polarity: a
// non-finite operand on either side is a failure, never a zero. It does not
// restate the finiteness rule — it asks MaxAbsDiffScan for the verdict — and it
// accumulates with `!(rel <= worst)` so a NaN ratio cannot be dropped either.
// Issue #449 is about `if (d > worst)`; the ratio form of the same loop is the
// same blindness one division later.
double MaxRelDiff(const std::vector<float>& got, const float* want, size_t count) {
  REQUIRE(got.size() == count);
  const vllm_test::MaxAbsDiffScanResult r = vllm_test::MaxAbsDiffScan(got.data(), want, count);
  if (!r.ok()) {
    FAIL_CHECK("max rel: NON-FINITE operand at index "
               << r.bad_index << " (got = " << r.bad_got << ", want = " << r.bad_want
               << "). A NaN here used to reduce to 0.0 and PASS — see issue #449.");
    return std::numeric_limits<double>::infinity();
  }
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    // A relative error is undefined where the reference is zero; the ABSOLUTE
    // bound covers those elements.
    if (want[i] == 0.0F) continue;
    const double rel =
        std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i])) /
        std::abs(static_cast<double>(want[i]));
    if (!(rel <= worst)) worst = rel;
  }
  return worst;
}

}  // namespace

// ===========================================================================
// 1. The one delta: the scale swizzle
// ===========================================================================

TEST_CASE("ltx2 loader: the unswizzle inverts vLLM's own block-scale permutation") {
  for (int64_t c = 0; c < vllm_test::kLtx2BlockedCaseCount; ++c) {
    const auto& cs = vllm_test::kLtx2BlockedCases[c];
    const uint8_t* swizzled = nullptr;
    const uint8_t* linear = nullptr;
    int64_t count = 0;
    switch (c) {
      case 0:
        swizzled = vllm_test::kLtx2BlockedSwizzled0;
        linear = vllm_test::kLtx2BlockedLinear0;
        count = vllm_test::kLtx2BlockedLinear0Count;
        break;
      case 1:
        swizzled = vllm_test::kLtx2BlockedSwizzled1;
        linear = vllm_test::kLtx2BlockedLinear1;
        count = vllm_test::kLtx2BlockedLinear1Count;
        break;
      case 2:
        swizzled = vllm_test::kLtx2BlockedSwizzled2;
        linear = vllm_test::kLtx2BlockedLinear2;
        count = vllm_test::kLtx2BlockedLinear2Count;
        break;
      case 3:
        swizzled = vllm_test::kLtx2BlockedSwizzled3;
        linear = vllm_test::kLtx2BlockedLinear3;
        count = vllm_test::kLtx2BlockedLinear3Count;
        break;
      default:
        swizzled = vllm_test::kLtx2BlockedSwizzled4;
        linear = vllm_test::kLtx2BlockedLinear4;
        count = vllm_test::kLtx2BlockedLinear4Count;
        break;
    }
    REQUIRE(count == cs.rows * cs.cols);
    std::vector<uint8_t> got(static_cast<size_t>(count), 0);
    vllm::Ltx2UnswizzleNvfp4BlockScale(swizzled, static_cast<size_t>(count), cs.rows,
                                       cs.cols, got.data());
    int64_t mismatches = 0;
    int64_t first_bad = -1;
    for (int64_t i = 0; i < count; ++i) {
      if (got[static_cast<size_t>(i)] != linear[i]) {
        ++mismatches;
        if (first_bad < 0) first_bad = i;
      }
    }
    INFO("case " << c << " rows=" << cs.rows << " cols=" << cs.cols
                 << " mismatches=" << mismatches << " first_bad=" << first_bad);
    CHECK(mismatches == 0);
  }
}

TEST_CASE("ltx2 loader: the unswizzle refuses a buffer that is not the padded size") {
  std::vector<uint8_t> src(128 * 4, 0);
  std::vector<uint8_t> dst(128 * 4, 0);
  // rows=100 pads to 128, so 100*4 bytes is NOT what the layout stores.
  CHECK_THROWS(vllm::Ltx2UnswizzleNvfp4BlockScale(src.data(), 100 * 4, 100, 4, dst.data()));
}

// ===========================================================================
// 2. The torchao marker — read, not assumed
// ===========================================================================

TEST_CASE("ltx2 loader: the shipped torchao marker says swizzled fp4, and is parsed") {
  const std::string json = vllm_test::kLtx2RealTeMarkerJson;
  vllm::StTensor t;
  t.dtype = "U8";
  t.shape = {static_cast<int64_t>(json.size())};
  t.data = reinterpret_cast<const uint8_t*>(json.data());
  t.nbytes = json.size();
  const vllm::Ltx2TorchaoNvfp4Marker m =
      vllm::ParseLtx2TorchaoNvfp4Marker("video_aggregate_embed", t);
  CHECK(m.format == "torchao_nvfp4");
  CHECK(m.block_size == 16);
  CHECK(m.is_swizzled_scales);
  CHECK(m.config == "NVFP4DynamicActivationNVFP4WeightConfig");

  // The refusals. Each is a layout this port does NOT implement, and each would
  // otherwise dequantize to finite, wrongly-scaled weights.
  const char* rejects[] = {
      R"({"format": "compressed_tensors", "block_size": 16, "is_swizzled_scales": true})",
      R"({"format": "torchao_nvfp4", "block_size": 32, "is_swizzled_scales": true})",
      R"({"format": "torchao_nvfp4", "block_size": 16, "is_swizzled_scales": false})",
      "not json at all",
  };
  for (const char* bad : rejects) {
    vllm::StTensor b;
    b.dtype = "U8";
    b.shape = {static_cast<int64_t>(std::strlen(bad))};
    b.data = reinterpret_cast<const uint8_t*>(bad);
    b.nbytes = std::strlen(bad);
    const std::string payload_msg = std::string("payload ") + bad;
    INFO(payload_msg);
    CHECK_THROWS(vllm::ParseLtx2TorchaoNvfp4Marker("m", b));
  }
}

// ===========================================================================
// 3. The REAL checkpoints' own bytes
// ===========================================================================

TEST_CASE("ltx2 loader: the shipped text encoder's swizzled scale tile unswizzles") {
  REQUIRE(vllm_test::kLtx2RealTeScaleTileSwizzledCount == 512);
  REQUIRE(vllm_test::kLtx2RealTeScaleTileLinearCount == 512);
  std::vector<uint8_t> got(512, 0);
  vllm::Ltx2UnswizzleNvfp4BlockScale(vllm_test::kLtx2RealTeScaleTileSwizzled, 512, 128, 4,
                                     got.data());
  int64_t mismatches = 0;
  for (int i = 0; i < 512; ++i) {
    if (got[static_cast<size_t>(i)] != vllm_test::kLtx2RealTeScaleTileLinear[i]) ++mismatches;
  }
  INFO("mismatches=" << mismatches);
  CHECK(mismatches == 0);

  // And the bytes decode to the values TORCH read out of them. F8E4M3ToF32
  // returns NaN for the two NaN encodings (0x7F / 0xFF); the shipped tile
  // carries neither, which the count below states rather than assumes, so the
  // hardened reduction cannot be red for a legitimate reason here.
  int64_t nan_encodings = 0;
  std::vector<float> ours(512, 0.0F);
  for (int i = 0; i < 512; ++i) {
    const uint8_t b = got[static_cast<size_t>(i)];
    if (b == 0x7FU || b == 0xFFU) ++nan_encodings;
    ours[static_cast<size_t>(i)] = vllm::F8E4M3ToF32(b);
  }
  CHECK(nan_encodings == 0);
  const double max_abs =
      vllm_test::MaxAbsDiff(ours, vllm_test::kLtx2RealTeScaleTileLinearF32, 512);
  INFO("max abs diff vs torch fp8-e4m3 = " << max_abs);
  CHECK(max_abs == 0.0);
}

TEST_CASE("ltx2 loader: the shipped FP8 DiT's own bytes dequantize to torch's values") {
  const int64_t n = vllm_test::kLtx2RealDitFp8HeadCount;
  REQUIRE(n == vllm_test::kLtx2RealDitFp8HeadF32Count);
  std::vector<uint16_t> bf16(static_cast<size_t>(n), 0);
  vllm::DequantFp8ToBf16(vllm_test::kLtx2RealDitFp8Head, vllm_test::kLtx2RealDitFp8Scale, n,
                         bf16.data());
  std::vector<float> got(static_cast<size_t>(n), 0.0F);
  for (int64_t i = 0; i < n; ++i) {
    got[static_cast<size_t>(i)] = Bf16ToF32(bf16[static_cast<size_t>(i)]);
  }
  const double max_abs =
      vllm_test::MaxAbsDiff(got, vllm_test::kLtx2RealDitFp8HeadF32, static_cast<size_t>(n));
  const double max_rel =
      MaxRelDiff(got, vllm_test::kLtx2RealDitFp8HeadF32, static_cast<size_t>(n));
  INFO("max abs = " << max_abs << " max rel = " << max_rel);
  // bf16 carries 8 significand bits, so RNE's worst relative error is half an
  // ulp = 2^-8. (An earlier revision of this line wrote 2^-9 by counting the 7
  // EXPLICIT mantissa bits and forgetting the implicit one — a miscounted
  // constant, not a tolerance anyone widened to pass.)
  CHECK(max_rel <= 0.00390625);
  // And tighter, because it can be: our path computes f32(byte) * scale, which
  // is torch's association exactly, so the bf16 bits must match bit for bit.
  // That pins the SCALE and the ASSOCIATION.
  //
  // WHAT IT DOES NOT PIN, measured rather than assumed. An earlier revision of
  // this comment claimed "a rounding-mode or association change breaks this",
  // and the rounding half was false: replacing RNE with truncation in
  // DequantFp8ToBf16 left the whole suite green. The reason is structural, not a
  // property of which 32 bytes were sampled, so a different slice of this tensor
  // would not help either: an fp8-e4m3 value carries at most 4 significant bits,
  // this tensor's per-tensor scale (0x3A880000, mantissa 1.0001b) carries 4, and
  // 4 x 4 lands inside bf16's 8, so the product is ALWAYS exactly representable
  // and the bf16 store never rounds at all. The two counts below say so, so the
  // claim cannot drift from the fixture. The rounding mode is gated where the
  // regime is actually reachable: the synthetic FP8 arm below, whose scale is
  // deliberately not a power of two.
  int64_t bit_mismatches = 0, bf16_exact_goldens = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (bf16[static_cast<size_t>(i)] != F32ToBf16(vllm_test::kLtx2RealDitFp8HeadF32[i])) {
      ++bit_mismatches;
    }
    if (IsBf16Exact(vllm_test::kLtx2RealDitFp8HeadF32[i])) ++bf16_exact_goldens;
  }
  INFO("bf16 bit mismatches = " << bit_mismatches);
  CHECK(bit_mismatches == 0);
  INFO("bf16-exact goldens = " << bf16_exact_goldens << " of " << n);
  CHECK(bf16_exact_goldens == n);
  // ... and the structural statement, over EVERY fp8 byte rather than the 32
  // sampled: none of the 256 encodings times this scale needs a bf16 round.
  int64_t rounding_bytes = 0;
  for (int b = 0; b < 256; ++b) {
    const float v = vllm::F8E4M3ToF32(static_cast<uint8_t>(b));
    if (std::isnan(v)) continue;  // the two NaN encodings carry no value
    if (!IsBf16Exact(v * vllm_test::kLtx2RealDitFp8Scale)) ++rounding_bytes;
  }
  INFO("fp8 encodings that would need a bf16 round at this scale = " << rounding_bytes);
  CHECK(rounding_bytes == 0);
}

TEST_CASE("ltx2 loader: a torchao module built from the shipped bytes dequantizes") {
  // out=128, in=64 is exactly the geometry of the (0,0) scale tile: the scale
  // grid is [128, 4] and 4 groups of 16 is 64 inputs. Row 0's packed nibbles are
  // the shipped weight's first 32 bytes, so row 0 IS the golden.
  const int64_t out_features = 128, in_features = 64;
  std::vector<uint8_t> packed(static_cast<size_t>(out_features * in_features / 2), 0);
  std::memcpy(packed.data(), vllm_test::kLtx2RealTePackedHead,
              static_cast<size_t>(vllm_test::kLtx2RealTePackedHeadCount));

  vllm::StTensor w;
  w.dtype = "U8";
  w.shape = {out_features, in_features / 2};
  w.data = packed.data();
  w.nbytes = packed.size();
  vllm::StTensor s;
  s.dtype = "F8_E4M3";
  s.shape = {out_features / 4, (in_features / 16) * 4};
  s.data = vllm_test::kLtx2RealTeScaleTileSwizzled;
  s.nbytes = 512;
  const float scale2 = vllm_test::kLtx2RealTeScale2;
  vllm::StTensor g;
  g.dtype = "F32";
  g.shape = {};
  g.data = reinterpret_cast<const uint8_t*>(&scale2);
  g.nbytes = sizeof(float);

  std::vector<uint16_t> bf16(static_cast<size_t>(out_features * in_features), 0);
  vllm::Ltx2DequantNvfp4ToBf16("video_aggregate_embed", w, s, g, out_features, in_features,
                              vllm::Ltx2Nvfp4Producer::kTorchao, bf16.data());
  const size_t head = static_cast<size_t>(vllm_test::kLtx2RealTeWeightHeadF32Count);
  std::vector<float> got(head, 0.0F);
  for (size_t i = 0; i < head; ++i) got[i] = Bf16ToF32(bf16[i]);
  const double max_abs =
      vllm_test::MaxAbsDiff(got, vllm_test::kLtx2RealTeWeightHeadF32, head);
  const double max_rel = MaxRelDiff(got, vllm_test::kLtx2RealTeWeightHeadF32, head);
  INFO("max abs = " << max_abs << " max rel = " << max_rel);
  // Half an ulp of bf16 (2^-8). Not bit-exact here, unlike the FP8 case: the
  // generator multiplies nibble * group_scale * global while
  // DequantNvfp4ToBf16 folds group_scale * global FIRST (nvfp4_dequant.h:17),
  // so the two f32 associations differ by the last ulp before the bf16 round.
  CHECK(max_rel <= 0.00390625);

  // Row 0 alone is NOT enough: for a [128, 4] scale grid the (0, 0..3) cells sit
  // at swizzled offsets 0..3, so row 0 reads the same whether the unswizzle runs
  // or not. Probe EVERY row of the real tile by dequantizing an all-1.0 fp4
  // pattern, which makes each output literally its own group scale.
  std::vector<uint8_t> ones(packed.size(), 0x22);  // both nibbles -> e2m1 1.0
  vllm::StTensor w1 = w;
  w1.data = ones.data();
  std::vector<uint16_t> probe(bf16.size(), 0);
  vllm::Ltx2DequantNvfp4ToBf16("video_aggregate_embed", w1, s, g, out_features, in_features,
                              vllm::Ltx2Nvfp4Producer::kTorchao, probe.data());
  const size_t probe_n = static_cast<size_t>(out_features) * static_cast<size_t>(in_features);
  std::vector<float> probe_got(probe_n, 0.0F);
  std::vector<float> probe_want(probe_n, 0.0F);
  for (int64_t r = 0; r < out_features; ++r) {
    for (int64_t i = 0; i < in_features; ++i) {
      const size_t k = static_cast<size_t>(r * in_features + i);
      probe_want[k] = static_cast<float>(
          static_cast<double>(vllm_test::kLtx2RealTeScaleTileLinearF32[r * 4 + i / 16]) *
          static_cast<double>(scale2));
      probe_got[k] = Bf16ToF32(probe[k]);
    }
  }
  const double probe_max_rel = MaxRelDiff(probe_got, probe_want.data(), probe_n);
  INFO("per-row group-scale probe max rel = " << probe_max_rel);
  CHECK(probe_max_rel <= 0.00390625);

  // And the SHAPE trap: reading the swizzled scale as if it were linear
  // [out, in/16] type-checks on element count. It must not be accepted.
  vllm::StTensor bad = s;
  bad.shape = {out_features, in_features / 16};
  CHECK_THROWS(vllm::Ltx2DequantNvfp4ToBf16("video_aggregate_embed", w, bad, g, out_features,
                                            in_features, vllm::Ltx2Nvfp4Producer::kTorchao,
                                            bf16.data()));

  // The PACKED-WIDTH refusal has to say something a reader can act on. Naming
  // the stored shape on both sides of "is X but the module is X" is a
  // contradiction, not a diagnosis, so the message states the LOGICAL width the
  // stored shape implies.
  std::string width_what;
  try {
    vllm::Ltx2DequantNvfp4ToBf16("video_aggregate_embed", w, s, g, out_features,
                                  in_features / 2, vllm::Ltx2Nvfp4Producer::kTorchao,
                                  bf16.data());
  } catch (const std::exception& e) {
    width_what = e.what();
  }
  const std::string width_msg = "what: " + width_what;
  INFO(width_msg);
  // stored [128, 32] -> logical [128, 64], asked for [128, 32].
  CHECK(width_what.find("stored [128, 32]") != std::string::npos);
  CHECK(width_what.find("LOGICAL [128, 64]") != std::string::npos);
  CHECK(width_what.find("the module is [128, 32]") != std::string::npos);
}

TEST_CASE("ltx2 loader: the NVFP4 refusal names a tensor the checkpoint ACTUALLY has") {
  // "Refuse BY NAME" is this phase's headline promise, and a name that appears
  // nowhere in the file does not keep it. Ltx2DequantTorchaoNvfp4ToBf16 takes a
  // MODULE and appends `.weight` / `.weight_scale` / `.weight_scale_2`; the DiT
  // call site used to hand it the full tensor name, so the refusal read
  // 'patchify_proj.weight.weight_scale' and a user grepping the checkpoint for
  // it found nothing.
  const Ltx2DitParams p = TinyParams();
  SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kNvfp4, {});
  const std::string pre = vllm::kLtx2DitCheckpointPrefix;
  const std::string scale_name = pre + "patchify_proj.weight_scale";

  // Corrupt ONLY the stored SHAPE of one scale sidecar: the same 512 bytes
  // declared in the torchao `to_blocked` framing [32, 16] rather than the
  // cuBLAS-padded [128, 4] this (marker-less) file actually uses. Same element
  // count, so the safetensors reader accepts it and the refusal fires in the
  // producer resolver, which is the call site under test. It is also a REAL
  // disagreement rather than a nonsense shape: [32, 16] is exactly what a torchao
  // file of this geometry would store, so nothing but the missing marker says it
  // is wrong.
  std::set<std::string> present;
  bool patched = false;
  for (StEntry& e : syn.entries) {
    present.insert(e.name);
    if (e.name == scale_name) {
      REQUIRE(e.shape == std::vector<int64_t>{128, 4});
      e.shape = {32, 16};
      patched = true;
    }
  }
  REQUIRE(patched);
  const std::string path = TmpPath("nvfp4_name");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  std::string what;
  try {
    vllm::Ltx2LoadDitFromSafetensors(file);
  } catch (const std::exception& e) {
    what = e.what();
  }
  const std::string what_msg = "what: " + what;
  INFO(what_msg);
  CHECK_FALSE(what.empty());
  // The doubled component, stated directly: it must not come back.
  CHECK(what.find(".weight.weight") == std::string::npos);
  // Every name the message quotes must be a tensor this checkpoint carries,
  // which is the property "refuse by name" actually claims.
  size_t open = what.find('\'');
  int64_t quoted = 0;
  while (open != std::string::npos) {
    const size_t close = what.find('\'', open + 1);
    if (close == std::string::npos) break;
    const std::string quoted_name = what.substr(open + 1, close - open - 1);
    ++quoted;
    const std::string quoted_msg = "quoted name: " + quoted_name;
    INFO(quoted_msg);
    CHECK(present.count(pre + quoted_name) == 1);
    open = what.find('\'', close + 1);
  }
  CHECK(quoted > 0);
  std::remove(path.c_str());
}

// ===========================================================================
// 4. The REAL DiT manifest
// ===========================================================================

TEST_CASE("ltx2 loader: the shipped FP8 DiT manifest is fully accounted for") {
  REQUIRE(vllm_test::kLtx25Fp8DitTensorCount == 6124);
  const std::string pre = vllm::kLtx2DitCheckpointPrefix;

  int64_t scales = 0, f8 = 0, bf16 = 0, f32 = 0, unprefixed = 0;
  std::set<std::string> families;
  for (const auto& t : vllm_test::kLtx25Fp8DitTensors) {
    const std::string name = t.name;
    if (name.compare(0, pre.size(), pre) != 0) {
      ++unprefixed;
      continue;
    }
    const std::string bare = name.substr(pre.size());
    if (EndsWith(bare, "_scale")) {
      ++scales;
      continue;
    }
    if (std::string(t.dtype) == "F8_E4M3") ++f8;
    if (std::string(t.dtype) == "BF16") ++bf16;
    if (std::string(t.dtype) == "F32") ++f32;
    families.insert(bare.substr(0, bare.find('.')));
  }
  // EVERY tensor carries the ComfyUI prefix; the loader strips exactly one.
  CHECK(unprefixed == 0);
  CHECK(scales == 1775);
  CHECK(f8 == 1775);   // one scale per quantized weight, and no more
  CHECK(bf16 == 2284);
  CHECK(f32 == 290);   // the tables; the 1775 scalar scales were counted above

  // The five families the file carries beyond the ORIGINAL L2 contract, named so
  // their status cannot be discovered later. Two are ported (the prompt AdaLN
  // pair), two are loaded elsewhere (the connectors), one is genuinely unported
  // (keyframes).
  CHECK(families.count("prompt_adaln_single") == 1);
  CHECK(families.count("audio_prompt_adaln_single") == 1);
  CHECK(families.count("keyframes_abs_pos_embedding") == 1);
  CHECK(families.count("video_embeddings_connector") == 1);
  CHECK(families.count("audio_embeddings_connector") == 1);

  // Spot-check the two shapes the ported forward is most sensitive to, straight
  // out of the shipped header. `audio_to_video_attn` is the asymmetric pair the
  // spec's test-trap list names: query/out width from the VIDEO stream, key/value
  // from the AUDIO one. A square assumption transposes these and still runs.
  const auto* a2v_q = FindDit(pre + "transformer_blocks.0.audio_to_video_attn.to_q.weight");
  const auto* a2v_o =
      FindDit(pre + "transformer_blocks.0.audio_to_video_attn.to_out.0.weight");
  REQUIRE(a2v_q != nullptr);
  REQUIRE(a2v_o != nullptr);
  CHECK(a2v_q->shape[0] == 2048);
  CHECK(a2v_q->shape[1] == 4096);
  CHECK(a2v_o->shape[0] == 4096);
  CHECK(a2v_o->shape[1] == 2048);
  // The FP8 arm's scale really is a per-tensor SCALAR, not a per-channel vector.
  const auto* scale = FindDit(pre + "transformer_blocks.0.attn1.to_q.weight_scale");
  REQUIRE(scale != nullptr);
  CHECK(std::string(scale->dtype) == "F32");
  CHECK(scale->rank == 0);
  // The video `ff` has NO bias and the audio one does — upstream's
  // ff_bias=false / audio_ff_bias=true asymmetry, in the shipped weights.
  CHECK(FindDit(pre + "transformer_blocks.0.ff.net.0.proj.bias") == nullptr);
  CHECK(FindDit(pre + "transformer_blocks.0.audio_ff.net.0.proj.bias") != nullptr);
}

TEST_CASE("ltx2 loader: the L2 contract's every name is present in the shipped DiT") {
  const std::string pre = vllm::kLtx2DitCheckpointPrefix;
  std::vector<Ltx2TensorSpec> manifest;
  std::set<std::string> present;
  for (const auto& t : vllm_test::kLtx25Fp8DitTensors) {
    const std::string bare = std::string(t.name).substr(pre.size());
    if (EndsWith(bare, "_scale")) continue;
    present.insert(bare);
    manifest.push_back({bare, ShapeOf(t.shape, t.rank)});
  }

  Ltx2DitParams p = vllm::ParseLtx2DitParamsFromManifest(manifest);
  CHECK(p.num_layers == 48);
  CHECK(p.num_attention_heads == 32);
  CHECK(p.audio_num_attention_heads == 32);
  CHECK(p.inner_dim() == 4096);
  CHECK(p.audio_inner_dim() == 2048);
  CHECK(p.attention_head_dim == 128);
  CHECK(p.audio_attention_head_dim == 64);
  CHECK(p.in_channels == 128);
  CHECK(p.out_channels == 128);
  CHECK(p.cross_attention_dim == 4096);
  CHECK(p.audio_cross_attention_dim == 2048);
  CHECK(p.apply_gated_attention);
  CHECK(p.cross_attention_adaln);
  CHECK_FALSE(p.ff_bias);
  CHECK(p.audio_ff_bias);
  // MEASURED: the SHIPPED checkpoint carries prompt_adaln_single, which upstream
  // builds only when use_prompt_adaln_single is TRUE (model.py:222-226), so the
  // prompt-K/V cache's premise does not hold for this checkpoint
  // (.agents/specs/ltx-2-5.md §1.2) and the module is PORTED
  // (.agents/specs/ltx25-prompt-adaln.md, issue #644).
  CHECK(p.use_prompt_adaln_single);
  // MEASURED the same way: the SHIPPED vonkaiser FP8 checkpoint carries
  // `keyframes_abs_pos_embedding`, so `supports_keyframes_abs_pos_embedding`
  // (model.py:166-173) holds for it and the marker is live on every forward.
  // This is the file's OWN evidence that ltx2.h's old "LTX-2.5's checkpoint does
  // not carry the parameter" was false (row LTX25-KEYFRAMES-ABS-POS, issue #658).
  CHECK(p.use_keyframes_abs_pos_embedding);
  {
    // The tensor the flag was resolved from, named and shaped, with a positive
    // control in the same loop so "found it" is not an artefact of the search.
    int64_t found = 0;
    int64_t control = 0;
    for (const Ltx2TensorSpec& spec : manifest) {
      if (spec.name == "keyframes_abs_pos_embedding") {
        ++found;
        CHECK(spec.shape == std::vector<int64_t>{1, 4096});
      }
      if (spec.name == "patchify_proj.weight") ++control;
    }
    CHECK(found == 1);
    CHECK(control == 1);
  }

  // Enumerate the contract AS THE FILE DESCRIBES IT — no flag is forced here any
  // more — and require every one of its names in the file. This line used to read
  // `contract.use_prompt_adaln_single = false`, which is the shape of the defect:
  // the gate agreed with the port about a module they were both dropping.
  const std::vector<Ltx2TensorSpec> want = vllm::EnumerateLtx2DitTensors(p);
  //   4078 = the original L2 contract
  //     12 = prompt_adaln_single + audio_prompt_adaln_single (issue #644)
  //      1 = keyframes_abs_pos_embedding (issue #658)
  CHECK(want.size() == 4078 + 12 + 1);
  int64_t missing = 0;
  std::string first_missing;
  for (const Ltx2TensorSpec& spec : want) {
    if (present.count(spec.name) == 0) {
      ++missing;
      if (first_missing.empty()) first_missing = spec.name;
    }
  }
  const std::string missing_msg = "missing=" + std::to_string(missing) + " first=" + first_missing;
  INFO(missing_msg);
  CHECK(missing == 0);

  // ... and account for every name the file has that the contract does not, so
  // "the rest is fine" is a counted claim rather than a hope.
  //   258 = 2 connectors x (8 blocks x 16 + 1 learnable_registers)
  // NOTHING ELSE. `keyframes_abs_pos_embedding` used to be the `+ 1` on this
  // line; it moved INTO `want` when row LTX25-KEYFRAMES-ABS-POS ported it, which
  // is the whole delta this row lands — exactly as the 12 prompt-AdaLN tensors
  // did before it. The unported group is now EMPTY for this checkpoint, and the
  // only names outside the DiT contract are the two connectors, which
  // `Ltx2LoadConnectorWeights` owns.
  std::set<std::string> want_set;
  for (const Ltx2TensorSpec& spec : want) want_set.insert(spec.name);
  int64_t extra = 0;
  for (const std::string& name : present) {
    if (want_set.count(name) == 0) ++extra;
  }
  CHECK(extra == 258);
}

// ===========================================================================
// 5. The REAL text-encoder manifest
// ===========================================================================

TEST_CASE("ltx2 loader: the shipped torchao text encoder's widths are the LOGICAL ones") {
  REQUIRE(vllm_test::kLtx25Nvfp4TeTensorCount == 1688);

  // model.norm.weight is BF16 and therefore UNPACKED — the width authority.
  const auto* norm = FindTe("model.norm.weight");
  REQUIRE(norm != nullptr);
  CHECK(std::string(norm->dtype) == "BF16");
  CHECK(norm->rank == 1);
  CHECK(norm->shape[0] == 3840);

  // The two caption projections, the trap this campaign already fell into once.
  struct Proj {
    const char* module;
    int64_t out_features;
  };
  const Proj projections[] = {
      {"text_embedding_projection.video_aggregate_embed", 4096},
      {"text_embedding_projection.audio_aggregate_embed", 2048},
  };
  for (const Proj& pr : projections) {
    const auto* w = FindTe(std::string(pr.module) + ".weight");
    const auto* s = FindTe(std::string(pr.module) + ".weight_scale");
    const auto* g = FindTe(std::string(pr.module) + ".weight_scale_2");
    const auto* b = FindTe(std::string(pr.module) + ".bias");
    const auto* m = FindTe(std::string(pr.module) + ".torchao_nvfp4");
    REQUIRE(w != nullptr);
    REQUIRE(s != nullptr);
    REQUIRE(g != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(m != nullptr);
    CHECK(std::string(w->dtype) == "U8");
    CHECK(w->shape[0] == pr.out_features);
    // TWO values per byte: the stored width is HALF the logical one, and the
    // logical one is the Gemma hidden size times (num_hidden_layers + 1).
    const int64_t in_features = w->shape[1] * 2;
    CHECK(in_features == 3840 * 49);
    CHECK(std::string(s->dtype) == "F8_E4M3");
    CHECK(s->shape[0] == pr.out_features / 4);
    CHECK(s->shape[1] == (in_features / 16) * 4);
    CHECK(std::string(g->dtype) == "F32");
    CHECK(g->rank == 0);
    // The bias is BF16 — a DIFFERENT unpack path from the weight, which is
    // exactly the module ltx2_text_encoder.h:264-269 warns a loader drops.
    CHECK(std::string(b->dtype) == "BF16");
    CHECK(b->shape[0] == pr.out_features);
  }

  // Every quantized module carries all four tensors, tower and vision included.
  int64_t markers = 0, complete = 0;
  std::set<std::string> towers;
  for (const auto& t : vllm_test::kLtx25Nvfp4TeTensors) {
    const std::string name = t.name;
    if (!EndsWith(name, ".torchao_nvfp4")) continue;
    ++markers;
    const std::string module = name.substr(0, name.size() - std::strlen(".torchao_nvfp4"));
    towers.insert(module.substr(0, module.find('.')));
    if (FindTe(module + ".weight") != nullptr &&
        FindTe(module + ".weight_scale") != nullptr &&
        FindTe(module + ".weight_scale_2") != nullptr) {
      ++complete;
    }
  }
  CHECK(markers == 334);
  CHECK(complete == markers);
  // The file ships the FULL multimodal Gemma-4. Text conditioning is the scope,
  // but a loader that chokes on these cannot read this checkpoint at all.
  CHECK(towers.count("vision_model") == 1);
  CHECK(towers.count("multi_modal_projector") == 1);
  CHECK(towers.count("audio_projector") == 1);

  // The tokenizer ships AS A TENSOR, with its sidecars.
  const auto* tok = FindTe("tokenizer_json");
  REQUIRE(tok != nullptr);
  CHECK(std::string(tok->dtype) == "U8");
  CHECK(tok->shape[0] == 32169626);
  for (const char* asset : {"hf_asset__tokenizer_config.json",
                            "hf_asset__processor_config.json",
                            "hf_asset__generation_config.json",
                            "hf_asset__chat_template.jinja"}) {
    const std::string asset_msg = std::string("asset ") + asset;
    INFO(asset_msg);
    CHECK(FindTe(asset) != nullptr);
  }
}

// ===========================================================================
// 5b. The FIRST-PARTY NVFP4 DiT, held against the FP8 DiT as an ORACLE
//
// .agents/specs/nvfp4-nibble-order.md section 5.2. The two shipped DiTs quantize
// the SAME base weights, so the FP8 file is an independent oracle for the NVFP4
// read — which matters because NOTHING else here can tell a correct dequant from
// a plausible wrong one. A wrong nibble order or a wrong scale indexing produces
// finite, correctly shaped, correctly scaled garbage.
//
// The CONTROL arm is what makes this a gate rather than a witness: the same NVFP4
// read is also correlated against a DIFFERENT module's FP8 weights, and that must
// COLLAPSE. Without it, a fixture that passes on any two finite arrays would look
// exactly like a fixture that works (spec ltx-2-5.md section 7.0(c)).
// ===========================================================================

namespace {

// Pearson correlation. Non-finite on either side is a FAILURE, never a 0 — the
// same polarity issue #449 hardened MaxAbsDiff for.
double Correlation(const std::vector<float>& a, const float* b, size_t n) {
  REQUIRE(a.size() == n);
  REQUIRE(n > 1);
  const vllm_test::MaxAbsDiffScanResult scan = vllm_test::MaxAbsDiffScan(a.data(), b, n);
  if (!scan.ok()) {
    FAIL_CHECK("correlation: NON-FINITE operand at index "
               << scan.bad_index << " (got = " << scan.bad_got
               << ", want = " << scan.bad_want << ")");
    return std::numeric_limits<double>::quiet_NaN();
  }
  double ma = 0.0, mb = 0.0;
  for (size_t i = 0; i < n; ++i) {
    ma += static_cast<double>(a[i]);
    mb += static_cast<double>(b[i]);
  }
  ma /= static_cast<double>(n);
  mb /= static_cast<double>(n);
  double num = 0.0, da = 0.0, db = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double x = static_cast<double>(a[i]) - ma;
    const double y = static_cast<double>(b[i]) - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  REQUIRE(da > 0.0);
  REQUIRE(db > 0.0);
  return num / std::sqrt(da * db);
}

double RelRms(const std::vector<float>& a, const float* b, size_t n) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  REQUIRE(den > 0.0);
  return std::sqrt(num / den);
}

// Unswizzle the real [128, 16] scale tile and decode the committed real rows.
// `order` is the thing under test: both arms are exercised by the gate below, so
// the "wrong" one is not hypothetical.
std::vector<float> DequantRealNvfp4RowsWith(vllm::Nvfp4NibbleOrder order) {
  const int64_t rows = vllm_test::kLtx2RealDitNvfp4ScaleTileRows;
  const int64_t cols = vllm_test::kLtx2RealDitNvfp4ScaleTileCols;
  const int64_t elems = vllm_test::kLtx2RealDitNvfp4RowElems;
  const int64_t nrows = vllm_test::kLtx2RealDitNvfp4RowCount;
  REQUIRE(vllm_test::kLtx2RealDitNvfp4ScaleTileCount == rows * cols);
  REQUIRE(vllm_test::kLtx2RealDitNvfp4PackedCount == nrows * elems / 2);
  REQUIRE(cols == elems / 16);

  // The REAL unswizzle, on the REAL bytes.
  std::vector<uint8_t> linear(static_cast<size_t>(rows * cols), 0);
  vllm::Ltx2UnswizzleNvfp4BlockScale(vllm_test::kLtx2RealDitNvfp4ScaleTile,
                                     static_cast<size_t>(rows * cols), rows, cols,
                                     linear.data());

  std::vector<float> got;
  got.reserve(static_cast<size_t>(nrows * elems));
  for (int64_t i = 0; i < nrows; ++i) {
    const int64_t row = vllm_test::kLtx2RealDitNvfp4Rows[i];
    std::vector<uint16_t> bf16(static_cast<size_t>(elems), 0);
    vllm::DequantNvfp4ToBf16(
        vllm_test::kLtx2RealDitNvfp4Packed + static_cast<size_t>(i * elems / 2),
        linear.data() + static_cast<size_t>(row * cols),
        vllm_test::kLtx2RealDitNvfp4Scale2, 1, elems, bf16.data(), order);
    for (int64_t j = 0; j < elems; ++j) {
      got.push_back(Bf16ToF32(bf16[static_cast<size_t>(j)]));
    }
  }
  return got;
}

// RESOLVE the producer from the artifact's own marker state and declared shape,
// then decode with the order that producer implies.
//
// The resolver is INSIDE this helper deliberately. If it ever returned kTorchao
// for this file the nibble order would flip, the correlation would collapse, and
// the gate below would fail — which is what ties "the discriminator is right" to
// "the numbers are right" instead of testing them as two unrelated facts.
std::vector<float> DequantRealNvfp4Rows() {
  const std::vector<int64_t> declared(
      vllm_test::kLtx2RealDitNvfp4DeclaredScaleShape,
      vllm_test::kLtx2RealDitNvfp4DeclaredScaleShape + 2);
  // The real file carries NO marker for this module — that absence IS the
  // evidence, so it is passed as one.
  const vllm::Ltx2Nvfp4Producer producer = vllm::Ltx2ResolveNvfp4Producer(
      vllm_test::kLtx2RealDitNvfp4Module, nullptr, declared,
      vllm_test::kLtx2RealDitNvfp4OutFeatures, vllm_test::kLtx2RealDitNvfp4InFeatures);
  CHECK(producer == vllm::Ltx2Nvfp4Producer::kNvfp4Prequant);
  const vllm::Nvfp4NibbleOrder order = vllm::Ltx2Nvfp4NibbleOrderFor(producer);
  CHECK(order == vllm::Nvfp4NibbleOrder::kHighFirst);
  return DequantRealNvfp4RowsWith(order);
}

}  // namespace

TEST_CASE("ltx2 loader: the first-party NVFP4 DiT agrees with the FP8 DiT") {
  const size_t n = static_cast<size_t>(vllm_test::kLtx2RealDitNvfp4RowCount *
                                       vllm_test::kLtx2RealDitNvfp4RowElems);
  REQUIRE(vllm_test::kLtx2RealDitFp8OracleF32Count == static_cast<int64_t>(n));
  REQUIRE(vllm_test::kLtx2RealDitFp8ControlF32Count == static_cast<int64_t>(n));

  const std::vector<float> got = DequantRealNvfp4Rows();

  const double corr = Correlation(got, vllm_test::kLtx2RealDitFp8OracleF32, n);
  const double rel = RelRms(got, vllm_test::kLtx2RealDitFp8OracleF32, n);
  const double ctrl = Correlation(got, vllm_test::kLtx2RealDitFp8ControlF32, n);

  // VALUES, not booleans (spec section 7.0).
  INFO("corr(oracle) = " << corr << "  rel_rms = " << rel << "  corr(control) = " << ctrl);
  CHECK(corr >= 0.99);
  // ── rel_rms IS A BAND, NOT A CEILING ───────────────────────────────────────
  //
  // rel_rms here is not an error budget to stay under; it is a PREDICTED
  // QUANTITY — the disagreement two different quantizations of the same base
  // weights must show. Measured 0.100672 (spec section 7), which is NVFP4's own
  // group-scaled E2M1 error combined with the oracle's fp8-e4m3 error.
  //
  // A one-sided `<= 0.15` was wrong in both directions, and both were measured
  // on these bytes rather than argued:
  //
  //   TOO LOOSE ABOVE. A uniform group-scale error of x1.10 lands at rel 0.1489
  //   and PASSED. Pearson correlation cannot help: it is scale-INVARIANT, so a
  //   systematic scale defect leaves corr at 0.994968 to every printed digit.
  //   rel is the ONLY statistic in this gate that can see it, which is exactly
  //   why it may not be spent as slack. Both facts are asserted below.
  //
  //   UNBOUNDED BELOW. rel had no floor at all, so an arm that reproduced the
  //   FP8 oracle EXACTLY — the shape a fixture takes when the "independent"
  //   oracle has quietly become the thing under test, this project's 7.0(c)
  //   failure — scored rel 0.0, corr 1.0 and passed with room to spare.
  //
  // The band is the measured value +/- about 15%. The width is for compiler and
  // libm drift, NOT for run-to-run noise: every input here is a committed byte
  // array and the reduction is a plain sequential double loop, so the value is
  // deterministic. It admits at worst a ~4% uniform scale error, against the
  // ~10% the ceiling alone admitted.
  CHECK(rel >= 0.085);
  CHECK(rel <= 0.115);
  // ...and the same ceiling on the correlation, for the same reason: an oracle
  // that agrees TOO well is not a better result, it is a broken oracle.
  CHECK(corr <= 0.998);
  // The control MUST collapse, or this fixture cannot separate right from wrong.
  CHECK(ctrl < 0.2);

  // ── THE TWO MUTATIONS THE BAND EXISTS FOR, RUN HERE ─────────────────────────
  //
  // Same discipline as the wrong-nibble arm below: the defect is executed by the
  // gate rather than described in a comment a future reader has to re-derive.

  // (1) A uniform +10% group-scale error. Invisible to corr BY CONSTRUCTION;
  //     caught only by the band's upper edge.
  std::vector<float> overscaled(got);
  for (float& v : overscaled) v = static_cast<float>(static_cast<double>(v) * 1.10);
  const double over_corr = Correlation(overscaled, vllm_test::kLtx2RealDitFp8OracleF32, n);
  const double over_rel = RelRms(overscaled, vllm_test::kLtx2RealDitFp8OracleF32, n);
  INFO("x1.10 group scale: corr = " << over_corr << "  rel_rms = " << over_rel);
  // Pearson is scale-invariant: the defect moves corr by less than a printed digit.
  CHECK(std::fabs(over_corr - corr) < 1e-6);
  // The old `rel <= 0.15` admitted this. The band does not.
  CHECK(over_rel > 0.115);

  // (2) The too-good arm: hand the gate the oracle itself. Rejected by the rel
  //     FLOOR and by the corr ceiling, neither of which existed before.
  const std::vector<float> perfect(vllm_test::kLtx2RealDitFp8OracleF32,
                                   vllm_test::kLtx2RealDitFp8OracleF32 + n);
  const double perfect_corr = Correlation(perfect, vllm_test::kLtx2RealDitFp8OracleF32, n);
  const double perfect_rel = RelRms(perfect, vllm_test::kLtx2RealDitFp8OracleF32, n);
  INFO("oracle vs itself: corr = " << perfect_corr << "  rel_rms = " << perfect_rel);
  CHECK(perfect_rel < 0.085);
  CHECK(perfect_corr > 0.998);

  // ── AND THE WRONG ORDER MUST FAIL, MEASURED RATHER THAN ASSUMED ─────────────
  //
  // A gate that only ever runs the right answer proves the right answer is
  // self-consistent. Running the SAME bytes low-first is the defect this whole
  // change exists to prevent, so its collapse is asserted here — the mutation
  // built into the gate rather than left for a reviewer to perform.
  const std::vector<float> wrong =
      DequantRealNvfp4RowsWith(vllm::Nvfp4NibbleOrder::kLowFirst);
  const double wrong_corr = Correlation(wrong, vllm_test::kLtx2RealDitFp8OracleF32, n);
  const double wrong_rel = RelRms(wrong, vllm_test::kLtx2RealDitFp8OracleF32, n);
  INFO("low-first (WRONG for this file): corr = " << wrong_corr
                                                  << "  rel_rms = " << wrong_rel);
  CHECK(wrong_corr < 0.2);
  CHECK(wrong_rel > 0.5);
  // Both readings decode the same MULTISET per group, so absmax cannot separate
  // them. Stated as a gated fact: it is why "the output looked reasonable" is not
  // evidence here, and why the correlation is what this gate reduces to.
  double got_absmax = 0.0, wrong_absmax = 0.0;
  for (size_t i = 0; i < n; ++i) {
    got_absmax = std::max(got_absmax, std::fabs(static_cast<double>(got[i])));
    wrong_absmax = std::max(wrong_absmax, std::fabs(static_cast<double>(wrong[i])));
  }
  INFO("absmax right = " << got_absmax << " wrong = " << wrong_absmax);
  CHECK(got_absmax == wrong_absmax);
}

TEST_CASE("ltx2 loader: the producer discriminator, every row of the table") {
  // .agents/specs/nvfp4-nibble-order.md section 3.2. The marker decides, the shape
  // corroborates, and every other combination REFUSES.
  const int64_t out_features = 4096, in_features = 4096;
  const std::vector<int64_t> blocked =
      vllm::Ltx2Nvfp4ToBlockedScaleShape(out_features, in_features);
  const std::vector<int64_t> padded =
      vllm::Ltx2Nvfp4PaddedScaleShape(out_features, in_features);
  CHECK(blocked == std::vector<int64_t>{1024, 1024});
  CHECK(padded == std::vector<int64_t>{4096, 256});

  vllm::Ltx2TorchaoNvfp4Marker marker;
  marker.format = "torchao_nvfp4";
  marker.block_size = 16;
  marker.is_swizzled_scales = true;

  // Marker present + to_blocked framing -> torchao, LOW-first.
  const vllm::Ltx2Nvfp4Producer torchao = vllm::Ltx2ResolveNvfp4Producer(
      "m", &marker, blocked, out_features, in_features);
  CHECK(torchao == vllm::Ltx2Nvfp4Producer::kTorchao);
  CHECK(vllm::Ltx2Nvfp4NibbleOrderFor(torchao) == vllm::Nvfp4NibbleOrder::kLowFirst);

  // Marker absent + cuBLAS-padded framing -> nvfp4-prequant, HIGH-first.
  const vllm::Ltx2Nvfp4Producer prequant =
      vllm::Ltx2ResolveNvfp4Producer("m", nullptr, padded, out_features, in_features);
  CHECK(prequant == vllm::Ltx2Nvfp4Producer::kNvfp4Prequant);
  CHECK(vllm::Ltx2Nvfp4NibbleOrderFor(prequant) == vllm::Nvfp4NibbleOrder::kHighFirst);

  // THE TWO DISAGREEMENTS. Each is a shape that is perfectly valid for the OTHER
  // producer, so neither is malformed — which is exactly why picking one would be
  // silent and wrong.
  CHECK_THROWS(
      vllm::Ltx2ResolveNvfp4Producer("m", &marker, padded, out_features, in_features));
  CHECK_THROWS(
      vllm::Ltx2ResolveNvfp4Producer("m", nullptr, blocked, out_features, in_features));

  // A shape that is neither framing, with and without a marker.
  const std::vector<int64_t> nonsense = {512, 2048};
  CHECK_THROWS(
      vllm::Ltx2ResolveNvfp4Producer("m", &marker, nonsense, out_features, in_features));
  CHECK_THROWS(
      vllm::Ltx2ResolveNvfp4Producer("m", nullptr, nonsense, out_features, in_features));

  // Both refusals must NAME the module, which is what "refuse by name" claims —
  // and every SINGLE-QUOTED span in them must be a tensor name, because that is
  // the convention the whole loader refuses in and the convention a user greps
  // the checkpoint with. An apostrophe in ordinary prose silently breaks it: it
  // opens a quote that closes on the next one, so the message starts "naming" a
  // fragment of its own explanation. This suite already asserts that for the
  // whole-file refusal path; asserting it HERE too is what stopped a stray
  // "the other's assumption" from shipping in the marker-present branch.
  for (bool with_marker : {true, false}) {
    std::string what;
    try {
      vllm::Ltx2ResolveNvfp4Producer("transformer_blocks.0.attn1.to_q",
                                     with_marker ? &marker : nullptr,
                                     with_marker ? padded : blocked, out_features,
                                     in_features);
    } catch (const std::exception& e) {
      what = e.what();
    }
    const std::string msg = "what: " + what;
    INFO(msg);
    CHECK(what.find("transformer_blocks.0.attn1.to_q") != std::string::npos);
    // An even count means every quote pairs; an odd one means prose ate a
    // delimiter.
    const int64_t quotes = std::count(what.begin(), what.end(), '\'');
    INFO("single-quote count = " << quotes);
    CHECK(quotes % 2 == 0);
    CHECK(quotes >= 2);
    // ...and each pair really is the tensor this refusal is about.
    size_t open = what.find('\'');
    while (open != std::string::npos) {
      const size_t close = what.find('\'', open + 1);
      if (close == std::string::npos) break;
      const std::string quoted = what.substr(open + 1, close - open - 1);
      const std::string qmsg = "quoted span: " + quoted;
      INFO(qmsg);
      CHECK(quoted == "transformer_blocks.0.attn1.to_q.weight_scale");
      open = what.find('\'', close + 1);
    }
  }
}

TEST_CASE("ltx2 loader: EVERY quantized module of the shipped NVFP4 DiT resolves alike") {
  // The correlation gate proves ONE module decodes correctly. This proves the
  // producer resolves the SAME way for all 1176 of them, straight off the real
  // header — because "I sampled a few and they agreed" is how the linear-layout
  // diagnosis got written in the first place.
  REQUIRE(vllm_test::kLtx25Nvfp4DitTensorCount == 7876);
  const std::string pre = vllm::kLtx2DitCheckpointPrefix;

  // A name -> shape index over the manifest, so the scale lookup is not O(n^2).
  std::map<std::string, const vllm_test::Ltx25Nvfp4DitTensor*> by_name;
  for (const auto& t : vllm_test::kLtx25Nvfp4DitTensors) by_name[t.name] = &t;

  int64_t packed = 0, markers = 0, scale2 = 0, input_scale = 0, resolved_prequant = 0;
  int64_t padded_framing = 0, would_be_blocked = 0, ambiguous_with_linear = 0;
  for (const auto& t : vllm_test::kLtx25Nvfp4DitTensors) {
    const std::string name = t.name;
    if (EndsWith(name, ".torchao_nvfp4")) ++markers;
    if (EndsWith(name, ".weight_scale_2")) ++scale2;
    if (EndsWith(name, ".input_scale")) ++input_scale;
    if (std::string(t.dtype) != "U8") continue;
    ++packed;
    REQUIRE(t.rank == 2);
    const int64_t out_features = t.shape[0];
    const int64_t in_features = t.shape[1] * 2;  // TWO values per byte

    const auto it = by_name.find(name + "_scale");
    REQUIRE(it != by_name.end());
    const std::vector<int64_t> scale_shape = ShapeOf(it->second->shape, it->second->rank);

    // No marker ANYWHERE in this file, so every module takes the marker-less arm.
    const vllm::Ltx2Nvfp4Producer p = vllm::Ltx2ResolveNvfp4Producer(
        name, nullptr, scale_shape, out_features, in_features);
    if (p == vllm::Ltx2Nvfp4Producer::kNvfp4Prequant) ++resolved_prequant;
    if (scale_shape == vllm::Ltx2Nvfp4PaddedScaleShape(out_features, in_features)) {
      ++padded_framing;
    }
    if (scale_shape == vllm::Ltx2Nvfp4ToBlockedScaleShape(out_features, in_features)) {
      ++would_be_blocked;
    }
    // THE AMBIGUITY, counted rather than asserted in prose: for these geometries
    // the padded framing IS the linear [N, K/16] shape, so a shape test alone
    // could not have told swizzled from linear for a single one of them.
    if (scale_shape == std::vector<int64_t>{out_features, in_features / 16}) {
      ++ambiguous_with_linear;
    }
  }
  INFO("packed=" << packed << " markers=" << markers << " scale2=" << scale2
                 << " input_scale=" << input_scale << " prequant=" << resolved_prequant
                 << " padded=" << padded_framing << " blocked=" << would_be_blocked
                 << " ambiguous_with_linear=" << ambiguous_with_linear);
  CHECK(packed == 1176);
  // torchao ALWAYS writes its marker. Zero of them is what excludes torchao, and
  // it is the whole basis of the marker-less arm.
  CHECK(markers == 0);
  CHECK(scale2 == 1176);
  // W4A4-shaped export: the activation scales are present and UNUSED on this
  // weight-only path, exactly as vLLM's W4A16 method discards them
  // (modelopt.py:1264-1268).
  CHECK(input_scale == 1176);
  CHECK(resolved_prequant == 1176);
  CHECK(padded_framing == 1176);
  CHECK(would_be_blocked == 0);
  CHECK(ambiguous_with_linear == 1176);
}

TEST_CASE("ltx2 loader: the two scale framings are the same buffer, differently dressed") {
  // Why the shape cannot discriminate alone, as a gated fact rather than a claim
  // in a comment: for every geometry the shipped DiT uses, the cuBLAS-padded
  // framing has the SAME element count as the to_blocked one AND is numerically
  // identical to the LINEAR [N, K/16] shape.
  struct Geom { int64_t out, in; };
  const Geom geoms[] = {{4096, 4096}, {2048, 2048}, {16384, 4096}, {4096, 16384},
                        {8192, 2048}, {2048, 4096}, {4096, 256}};
  for (const Geom& g : geoms) {
    const std::vector<int64_t> blocked = vllm::Ltx2Nvfp4ToBlockedScaleShape(g.out, g.in);
    const std::vector<int64_t> padded = vllm::Ltx2Nvfp4PaddedScaleShape(g.out, g.in);
    const std::string msg = "geom [" + std::to_string(g.out) + ", " + std::to_string(g.in) + "]";
    INFO(msg);
    CHECK(blocked[0] * blocked[1] == padded[0] * padded[1]);
    // The trap: padded == linear [N, K/16] exactly, because N % 128 == 0 and
    // (K/16) % 4 == 0 for all of these.
    CHECK(padded == std::vector<int64_t>{g.out, g.in / 16});
    // ...and the two FRAMINGS are never equal to each other, so a marker-bearing
    // file can always be told from a marker-less one by shape once the marker has
    // already narrowed it.
    CHECK(blocked != padded);
  }
}

// ===========================================================================
// 6. Whole-model materialization, on synthetic files
// ===========================================================================

TEST_CASE("ltx2 loader: the FP8 DiT materializes onto the L2 contract, exactly") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("fp8");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  // The fixture must REACH the bf16 rounding regime, or the bit-exact comparison
  // below silently stops gating the rounding mode — which is exactly what the
  // old power-of-two scale did. Stated as a gate, not as a comment.
  INFO("fp8 elements = " << syn.fp8_elements
                         << " needing a bf16 round = " << syn.fp8_rounding_elements
                         << " non-finite = " << syn.fp8_nonfinite_elements);
  CHECK(syn.fp8_elements > 0);
  CHECK(syn.fp8_rounding_elements * 2 > syn.fp8_elements);
  CHECK(syn.fp8_nonfinite_elements == 0);

  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);
  CHECK(ck.quant == Ltx2DitQuant::kFp8);
  CHECK(ck.unported.empty());
  CHECK(ck.params.num_layers == p.num_layers);
  CHECK(ck.params.inner_dim() == p.inner_dim());
  CHECK(ck.params.audio_inner_dim() == p.audio_inner_dim());
  CHECK_FALSE(ck.params.ff_bias);
  CHECK(ck.params.audio_ff_bias);

  // Every quantized/bf16 tensor lands bit-exactly on the bf16 the checkpoint
  // encodes. bf16 is the DEFAULT: a wider default would still pass a value gate.
  int64_t checked = 0, bad = 0;
  std::string first_bad;
  for (const auto& kv : syn.expected) {
    auto it = ck.views.find(kv.first);
    REQUIRE(it != ck.views.end());
    const vt::Tensor& t = it->second;
    REQUIRE(t.dtype == vt::DType::kBF16);
    const uint16_t* got = t.Ptr<uint16_t>();
    for (size_t i = 0; i < kv.second.size(); ++i) {
      ++checked;
      if (got[i] != kv.second[i]) {
        ++bad;
        if (first_bad.empty()) first_bad = kv.first;
      }
    }
  }
  const std::string count_msg = "checked=" + std::to_string(checked) + " bad=" +
                                std::to_string(bad) + " first=" + first_bad;
  INFO(count_msg);
  CHECK(checked > 0);
  CHECK(bad == 0);

  // The tables stay F32 because the CHECKPOINT stores them F32.
  auto tbl = ck.views.find("scale_shift_table");
  REQUIRE(tbl != ck.views.end());
  CHECK(tbl->second.dtype == vt::DType::kF32);
  const size_t table_n = static_cast<size_t>(2 * p.inner_dim());
  std::vector<float> table_got(tbl->second.Ptr<float>(), tbl->second.Ptr<float>() + table_n);
  std::vector<float> table_want(table_n, 0.0F);
  for (size_t i = 0; i < table_n; ++i) table_want[i] = TrueValue("scale_shift_table", i);
  const double table_max_abs = vllm_test::MaxAbsDiff(table_got, table_want);
  INFO("table max abs = " << table_max_abs);
  CHECK(table_max_abs == 0.0);

  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: the NVFP4 DiT arm materializes onto the same contract") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kNvfp4, {});
  const std::string path = TmpPath("nvfp4");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);
  CHECK(ck.quant == Ltx2DitQuant::kNvfp4);
  int64_t checked = 0, bad = 0;
  std::string first_bad;
  for (const auto& kv : syn.expected) {
    auto it = ck.views.find(kv.first);
    REQUIRE(it != ck.views.end());
    const uint16_t* got = it->second.Ptr<uint16_t>();
    for (size_t i = 0; i < kv.second.size(); ++i) {
      ++checked;
      if (got[i] != kv.second[i]) {
        ++bad;
        if (first_bad.empty()) first_bad = kv.first;
      }
    }
  }
  const std::string count_msg = "checked=" + std::to_string(checked) + " bad=" +
                                std::to_string(bad) + " first=" + first_bad;
  INFO(count_msg);
  CHECK(checked > 0);
  CHECK(bad == 0);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: a missing tensor throws BY NAME and never reads as zeros") {
  const Ltx2DitParams p = TinyParams();
  SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string drop =
      std::string(vllm::kLtx2DitCheckpointPrefix) + "transformer_blocks.1.audio_attn2.to_v.bias";
  std::vector<StEntry> kept;
  bool found = false;
  for (const StEntry& e : syn.entries) {
    if (e.name == drop) {
      found = true;
      continue;
    }
    kept.push_back(e);
  }
  REQUIRE(found);
  const std::string path = TmpPath("missing");
  WriteSafetensors(kept, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  bool named = false;
  try {
    vllm::Ltx2LoadDitFromSafetensors(file);
  } catch (const std::exception& e) {
    const std::string what = e.what();
    const std::string what_msg = "what: " + what;
  INFO(what_msg);
    named = what.find("transformer_blocks.1.audio_attn2.to_v.bias") != std::string::npos;
  }
  CHECK(named);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: the unported families are refused by name, not absorbed") {
  const Ltx2DitParams p = TinyParams();
  // `caption_projection` is the 19B form the DiT would carry under
  // `caption_proj_before_connector=false` (text_projection.py:31-38), which phase
  // L3 owes. It replaced `keyframes_abs_pos_embedding` here when row
  // LTX25-KEYFRAMES-ABS-POS ported that one (issue #658) — the mechanism is what
  // this case gates, and it needs SOME family that is genuinely unported.
  const SyntheticDit syn = BuildSyntheticDit(
      p, Ltx2DitQuant::kFp8,
      {"caption_projection.linear_1.weight", "video_embeddings_connector.learnable_registers",
       "audio_embeddings_connector.learnable_registers"});
  const std::string path = TmpPath("unported");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  std::string what;
  try {
    vllm::Ltx2LoadDitFromSafetensors(file);
  } catch (const std::exception& e) {
    what = e.what();
  }
  const std::string what_msg = "what: " + what;
  INFO(what_msg);
  // ONE family in the LIST now, not five. `prompt_adaln_single` /
  // `audio_prompt_adaln_single` left it when they were PORTED (issue #644), and
  // `keyframes_abs_pos_embedding` left it for the same reason (issue #658); the
  // cases below prove a checkpoint carrying either needs no opt-in at all.
  //
  // The list is asserted as a whole rather than by substring, because the message
  // deliberately goes on to NAME the families that are not in it — a
  // `find(...) == npos` over the whole message would only test the prose.
  const std::string head = "does NOT carry: ";
  const size_t at = what.find(head);
  REQUIRE(at != std::string::npos);
  const std::string list = what.substr(at + head.size(), what.find('.', at) - at - head.size());
  CHECK(list == "caption_projection");
  // ASSERTED AS AN ABSENCE, with the positive control right above it: the retired
  // family must not reappear in the refusal list.
  CHECK(list.find("keyframes_abs_pos_embedding") == std::string::npos);

  // THE TWO CONNECTOR FAMILIES ARE NOT UNPORTED AS OF PHASE L9c. They are
  // outside the DiT's contract by design — upstream loads them into the text
  // encoder's EmbeddingsProcessor (encoder_configurator.py:331-346) and so does
  // this port, through `Ltx2LoadConnectorWeights`, which the video engine calls.
  // Listing them here would say something untrue about the tree AND would demand
  // `allow_unported_modules` from a caller whose checkpoint this port reads
  // completely. Asserted as an ABSENCE, so restoring the old behaviour REDs.
  CHECK(what.find("video_embeddings_connector") == std::string::npos);
  CHECK(what.find("audio_embeddings_connector") == std::string::npos);

  // The opt-in still REPORTS it; it does not make it vanish.
  Ltx2DitLoadOptions options;
  options.allow_unported_modules = true;
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file, options);
  CHECK(ck.unported.size() == 1);
  for (const std::string& family : ck.unported) {
    CHECK(family != "video_embeddings_connector");
    CHECK(family != "audio_embeddings_connector");
  }
  std::remove(path.c_str());

  // AND THE CONNECTOR-ONLY CHECKPOINT LOADS WITH NO OPT-IN AT ALL, which is the
  // half of the change a message assertion cannot see: a file whose only
  // out-of-contract modules are the two connectors is now fully read by this
  // port, so requiring the flag would be requiring an admission of a gap that
  // has been closed.
  const SyntheticDit conn_only = BuildSyntheticDit(
      p, Ltx2DitQuant::kFp8,
      {"video_embeddings_connector.learnable_registers",
       "audio_embeddings_connector.learnable_registers"});
  const std::string conn_path = TmpPath("connector_only");
  WriteSafetensors(conn_only.entries, conn_path);
  const SafetensorsFile conn_file = SafetensorsFile::Open(conn_path);
  const vllm::Ltx2DitCheckpoint conn_ck = vllm::Ltx2LoadDitFromSafetensors(conn_file);
  CHECK(conn_ck.unported.empty());
  std::remove(conn_path.c_str());
}

// REPLACES the half of the case above that used `keyframes_abs_pos_embedding` as
// its unported example. The module is ported now (row LTX25-KEYFRAMES-ABS-POS,
// issue #658), so what is gated here is the OPPOSITE claim: a checkpoint carrying
// it loads with NO opt-in, resolves the flag, and BINDS the tensor rather than
// stepping over it. That is what neither shipped LTX-2.5 DiT could do before.
TEST_CASE("ltx2 loader: a keyframes-carrying checkpoint loads with NO opt-in") {
  Ltx2DitParams p = TinyParams();
  p.use_keyframes_abs_pos_embedding = true;
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("keyframes");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  // No `allow_unported_modules`. Before this row this threw by name.
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);
  CHECK(ck.unported.empty());
  // Resolved from the file's own shapes, which is the only evidence a manifest
  // carries — `supports_keyframes_abs_pos_embedding` (model.py:166-173).
  CHECK(ck.params.use_keyframes_abs_pos_embedding);

  // BOUND, not merely enumerated. A contract entry nothing binds is the exact
  // failure mode the prompt-AdaLN row found (issue #644 row 0).
  REQUIRE(ck.weights.keyframes_abs_pos_embedding.data != nullptr);
  REQUIRE(ck.weights.keyframes_abs_pos_embedding.rank == 2);
  CHECK(ck.weights.keyframes_abs_pos_embedding.shape[0] == 1);
  CHECK(ck.weights.keyframes_abs_pos_embedding.shape[1] == p.inner_dim());

  // THE FP8 CONVENTION, stated rather than assumed. The synthetic file stores
  // this tensor exactly as the shipped vonkaiser DiT does — `F8_E4M3` plus a
  // scalar `F32` `<name>_scale` — and `MaterializeDitTensor`'s F8_E4M3 arm is the
  // ONE convention that reads it: `ReadScalarF32` then `DequantFp8ToBf16`, giving
  // a BF16 view. No second convention was invented for a rank-2 [1, D] tensor.
  CHECK(ck.weights.keyframes_abs_pos_embedding.dtype == vt::DType::kBF16);
  {
    const auto want = syn.expected.find("keyframes_abs_pos_embedding");
    REQUIRE(want != syn.expected.end());
    const uint16_t* got = ck.weights.keyframes_abs_pos_embedding.Ptr<uint16_t>();
    REQUIRE(want->second.size() == static_cast<size_t>(p.inner_dim()));
    for (size_t i = 0; i < want->second.size(); ++i) {
      CAPTURE(i);
      CHECK(got[i] == want->second[i]);
    }
  }

  // ... and the f32 widening the L2 forward requires reaches it too, so the
  // dtype assertion in `PrepareStream` is satisfiable rather than a dead end.
  Ltx2DitLoadOptions widen;
  widen.widen_to_f32 = true;
  const vllm::Ltx2DitCheckpoint wide = vllm::Ltx2LoadDitFromSafetensors(file, widen);
  CHECK(wide.weights.keyframes_abs_pos_embedding.dtype == vt::DType::kF32);
  std::remove(path.c_str());
}

// THE REGRESSION GATE FOR ISSUE #644 ROW 0.
//
// A checkpoint that carries `prompt_adaln_single` — which the shipped LTX-2.5 DiT
// does — used to be refused without `allow_unported_modules=1`, and setting that
// extra reached three assignments that CLEARED `use_prompt_adaln_single`. So the
// only way to load a real DiT was also the way to silently drop the timestep half
// of every cross-attention K/V modulation.
//
// Both halves are asserted here, and neither is true by construction: the load
// with no opt-in exercises the contract, and the resolved flag is read back off
// the checkpoint the loader actually bound.
TEST_CASE("ltx2 loader: a DiT carrying prompt_adaln_single loads with NO opt-in") {
  Ltx2DitParams p = TinyParams();
  p.use_prompt_adaln_single = true;
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("prompt_adaln");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  // No options at all: the families are in the contract, so nothing is unported.
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);
  CHECK(ck.unported.empty());
  CHECK(ck.checkpoint_params.use_prompt_adaln_single);
  // The flag the FORWARD reads. This is the assertion the defect broke: it was
  // false while `checkpoint_params` said true.
  CHECK(ck.params.use_prompt_adaln_single);
  // And the weights are bound, not merely enumerated — a contract that listed the
  // tensors while `BindLtx2DitWeights` left the views null would render the same
  // wrong picture with a different failure mode.
  CHECK(ck.weights.prompt_adaln_single.linear.weight.data != nullptr);
  CHECK(ck.weights.audio_prompt_adaln_single.linear.weight.data != nullptr);
  CHECK(ck.weights.prompt_adaln_single.linear.weight.shape[0] == 2 * p.inner_dim());
  CHECK(ck.weights.audio_prompt_adaln_single.linear.weight.shape[0] == 2 * p.audio_inner_dim());
  // The 12 tensors, counted rather than assumed.
  Ltx2DitParams off = p;
  off.use_prompt_adaln_single = false;
  CHECK(vllm::EnumerateLtx2DitTensors(p).size() -
            vllm::EnumerateLtx2DitTensors(off).size() ==
        12);
  std::remove(path.c_str());

  // AND THE OPT-IN CANNOT UNDO IT. `allow_unported_modules` is what a real render
  // still passes for `keyframes_abs_pos_embedding`; it must leave a ported feature
  // alone. Same bytes, opt-in set, same resolved flag.
  const std::string path2 = TmpPath("prompt_adaln_optin");
  WriteSafetensors(syn.entries, path2);
  const SafetensorsFile file2 = SafetensorsFile::Open(path2);
  Ltx2DitLoadOptions options;
  options.allow_unported_modules = true;
  const vllm::Ltx2DitCheckpoint ck2 = vllm::Ltx2LoadDitFromSafetensors(file2, options);
  CHECK(ck2.params.use_prompt_adaln_single);
  CHECK(ck2.weights.prompt_adaln_single.linear.weight.data != nullptr);
  std::remove(path2.c_str());
}

// The `transformer` object `ParseLtx2DitParams` needs in order to say YES: every
// geometry key the shapes carry, plus every `check_config_value` the configurator
// asserts verbatim (model_configurator.py:26-44). Shared by the two adoption
// cases below so neither drifts from the other's idea of a valid config.
nlohmann::json MinimalTransformerConfig(const Ltx2DitParams& shapes) {
  nlohmann::json t;
  t["num_attention_heads"] = shapes.num_attention_heads;
  t["attention_head_dim"] = shapes.attention_head_dim;
  t["in_channels"] = shapes.in_channels;
  t["out_channels"] = shapes.out_channels;
  t["num_layers"] = shapes.num_layers;
  t["cross_attention_dim"] = shapes.cross_attention_dim;
  t["audio_num_attention_heads"] = shapes.audio_num_attention_heads;
  t["audio_attention_head_dim"] = shapes.audio_attention_head_dim;
  t["audio_in_channels"] = shapes.audio_in_channels;
  t["audio_out_channels"] = shapes.audio_out_channels;
  t["audio_cross_attention_dim"] = shapes.audio_cross_attention_dim;
  t["apply_gated_attention"] = true;
  t["cross_attention_adaln"] = true;
  t["ff_bias"] = false;
  t["audio_ff_bias"] = true;
  t["rope_type"] = "split";
  // The checks ParseLtx2DitParams asserts verbatim (model_configurator.py:26-44).
  t["dropout"] = 0.0;
  t["attention_bias"] = true;
  t["num_vector_embeds"] = nullptr;
  t["activation_fn"] = "gelu-approximate";
  t["num_embeds_ada_norm"] = 1000;
  t["use_linear_projection"] = false;
  t["only_cross_attention"] = false;
  t["cross_attention_norm"] = true;
  t["double_self_attention"] = false;
  t["upcast_attention"] = false;
  t["standardization_norm"] = "rms_norm";
  t["norm_elementwise_affine"] = false;
  t["qk_norm"] = "rms_norm";
  t["positional_embedding_type"] = "rope";
  t["use_audio_video_cross_attention"] = true;
  t["share_ff"] = false;
  t["av_cross_ada_norm"] = true;
  t["use_middle_indices_grid"] = true;
  t["caption_proj_before_connector"] = true;
  return t;
}

// The DECLARED config path. `Ltx2AdoptDeclaredDitParams` no longer forces both
// sides of its comparison to a cleared flag, so a config that disagrees with the
// file's shapes about `use_prompt_adaln_single` now produces two DIFFERENT
// contracts and is refused — which is an INPUT-driven gate, not a mutation-only
// one. Both directions are checked, because the equality is the point.
TEST_CASE("ltx2 loader: a config that disagrees about use_prompt_adaln_single is REFUSED") {
  Ltx2DitParams shapes = TinyParams();
  shapes.use_prompt_adaln_single = true;
  const nlohmann::json t = MinimalTransformerConfig(shapes);

  nlohmann::json agreeing;
  agreeing["transformer"] = t;
  agreeing["transformer"]["use_prompt_adaln_single"] = true;
  // Agreement is adopted, and carries the flag through.
  const Ltx2DitParams adopted =
      vllm::Ltx2AdoptDeclaredDitParams(agreeing, shapes, "the test config");
  CHECK(adopted.use_prompt_adaln_single);

  nlohmann::json disagreeing;
  disagreeing["transformer"] = t;
  disagreeing["transformer"]["use_prompt_adaln_single"] = false;
  CHECK_THROWS(vllm::Ltx2AdoptDeclaredDitParams(disagreeing, shapes, "the test config"));

  // And the other direction: shapes WITHOUT the module against a config that
  // declares it.
  Ltx2DitParams shapes_off = shapes;
  shapes_off.use_prompt_adaln_single = false;
  CHECK_THROWS(vllm::Ltx2AdoptDeclaredDitParams(agreeing, shapes_off, "the test config"));
}

// REPLACES the test that asserted `use_keyframes_abs_pos_embedding` was cleared
// only under `allow_unported_modules`. That refusal is retired (row
// LTX25-KEYFRAMES-ABS-POS, issue #658); what took its place is upstream's
// `supports_keyframes_abs_pos_embedding` (model.py:166-173), resolved against
// what the FILE carries — and this asserts all three of its outcomes, plus the
// one thing that must NOT happen.
TEST_CASE("ltx2 loader: a declared keyframes flag is RESOLVED against the file, not refused") {
  Ltx2DitParams shapes = TinyParams();
  shapes.use_prompt_adaln_single = false;

  nlohmann::json t = MinimalTransformerConfig(shapes);
  t["use_prompt_adaln_single"] = false;

  SUBCASE("declared TRUE, tensor ABSENT: adopted as FALSE, and NOT refused") {
    // The shipped first-party NVFP4 DiT exactly: `__metadata__` declares the
    // flag, `declared - state_dict` is precisely this one key, and upstream's
    // meta-device load leaves the parameter unmaterialized so the add is never
    // reached. Before this row, `ltx2.cpp` threw here unless the caller passed
    // `allow_unported_modules` — stricter than upstream.
    nlohmann::json cfg;
    cfg["transformer"] = t;
    cfg["transformer"]["use_keyframes_abs_pos_embedding"] = true;
    REQUIRE_FALSE(shapes.use_keyframes_abs_pos_embedding);
    const Ltx2DitParams adopted =
        vllm::Ltx2AdoptDeclaredDitParams(cfg, shapes, "the test config");
    CHECK_FALSE(adopted.use_keyframes_abs_pos_embedding);
    // NOT a synthesised zero: the contract must not name the tensor at all, or a
    // loader would go looking for a weight the file does not have.
    const std::vector<vllm::Ltx2TensorSpec> contract = vllm::EnumerateLtx2DitTensors(adopted);
    int64_t named = 0;
    for (const vllm::Ltx2TensorSpec& spec : contract) {
      if (spec.name == "keyframes_abs_pos_embedding") ++named;
    }
    CHECK(named == 0);
    // Positive control for that count, so a renamed contract entry cannot make
    // the assertion above pass by finding nothing at all.
    int64_t control = 0;
    for (const vllm::Ltx2TensorSpec& spec : contract) {
      if (spec.name == "patchify_proj.weight") ++control;
    }
    CHECK(control == 1);
  }

  SUBCASE("declared TRUE, tensor PRESENT: adopted TRUE and the contract names it") {
    // The vonkaiser FP8 DiT plus the first-party config it needs (that file
    // carries no `__metadata__` at all, so its config always arrives separately).
    Ltx2DitParams with = shapes;
    with.use_keyframes_abs_pos_embedding = true;
    nlohmann::json cfg;
    cfg["transformer"] = t;
    cfg["transformer"]["use_keyframes_abs_pos_embedding"] = true;
    const Ltx2DitParams adopted = vllm::Ltx2AdoptDeclaredDitParams(cfg, with, "the test config");
    CHECK(adopted.use_keyframes_abs_pos_embedding);
    const std::vector<vllm::Ltx2TensorSpec> contract = vllm::EnumerateLtx2DitTensors(adopted);
    REQUIRE_FALSE(contract.empty());
    // Registration order, which no shape encodes: model.py:217 is BEFORE :230.
    CHECK(contract[0].name == "keyframes_abs_pos_embedding");
    CHECK(contract[0].shape == std::vector<int64_t>{1, adopted.inner_dim()});
  }

  SUBCASE("NOT declared while the file CARRIES it is a real disagreement, refused") {
    // Upstream would build no parameter and `strict=False` would drop the key.
    // We refuse instead, because the tensor is already BOUND by the time the
    // config is adopted and the two would describe different contracts — the
    // same equality every other key is held to. Named rather than silent.
    Ltx2DitParams with = shapes;
    with.use_keyframes_abs_pos_embedding = true;
    nlohmann::json cfg;
    cfg["transformer"] = t;
    cfg["transformer"]["use_keyframes_abs_pos_embedding"] = false;
    CHECK_THROWS(vllm::Ltx2AdoptDeclaredDitParams(cfg, with, "the test config"));
  }
}

TEST_CASE("ltx2 loader: the f32 widening is OPT-IN and bit-exact over bf16") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("widen");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file, options);
  double max_abs = 0.0;
  for (const auto& kv : syn.expected) {
    auto it = ck.views.find(kv.first);
    REQUIRE(it != ck.views.end());
    REQUIRE(it->second.dtype == vt::DType::kF32);
    const std::vector<float> got(it->second.Ptr<float>(),
                                 it->second.Ptr<float>() + kv.second.size());
    std::vector<float> want(kv.second.size(), 0.0F);
    for (size_t i = 0; i < kv.second.size(); ++i) want[i] = Bf16ToF32(kv.second[i]);
    const double d = vllm_test::MaxAbsDiff(got, want);
    if (!(d <= max_abs)) max_abs = d;
  }
  INFO("widen max abs = " << max_abs);
  CHECK(max_abs == 0.0);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: staging at load produces the same weights as the host load") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("stage");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  vt::Queue queue;
  const vllm::Ltx2DitCheckpoint staged = vllm::Ltx2StreamDitToDevice(queue, file);
  int64_t bad = 0, checked = 0;
  for (const auto& kv : syn.expected) {
    auto it = staged.views.find(kv.first);
    REQUIRE(it != staged.views.end());
    const uint16_t* got = it->second.Ptr<uint16_t>();
    for (size_t i = 0; i < kv.second.size(); ++i) {
      ++checked;
      if (got[i] != kv.second[i]) ++bad;
    }
  }
  INFO("staged checked=" << checked << " bad=" << bad);
  CHECK(checked > 0);
  CHECK(bad == 0);

  // Staging exists to avoid moving twice the bytes; widening while staging
  // would defeat it, so it is refused rather than silently honoured.
  Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;
  CHECK_THROWS(vllm::Ltx2StreamDitToDevice(queue, file, options));
  std::remove(path.c_str());
}

// ===========================================================================
// 7. The text-encoder loader, on a synthetic torchao file
// ===========================================================================

namespace {

// A torchao-NVFP4 text encoder with the shipped file's SHAPE RULES at reduced
// dimensions: the prefixless projections, a BF16 model.norm, the embedded
// tokenizer pack, and a vision module that must not choke the loader.
std::vector<StEntry> BuildSyntheticTe(int64_t hidden, int64_t layers, int64_t video_out,
                                      int64_t audio_out,
                                      std::map<std::string, std::vector<uint16_t>>* expected) {
  std::vector<StEntry> entries;
  const std::vector<float> norm(static_cast<size_t>(hidden), 1.0F);
  entries.push_back({"model.norm.weight", "BF16", {hidden}, PackBf16(norm)});
  for (int64_t l = 0; l < layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l);
    entries.push_back({b + ".input_layernorm.weight", "BF16", {hidden}, PackBf16(norm)});
  }
  const int64_t in_features = hidden * (layers + 1);
  const char* marker =
      R"({"format": "torchao_nvfp4", "block_size": 16, "scope": "full", )"
      R"("config": "NVFP4DynamicActivationNVFP4WeightConfig", "is_swizzled_scales": true, )"
      R"("use_triton_kernel": true, "use_dynamic_activation": true, )"
      R"("use_dynamic_per_tensor_scale": true})";

  struct P {
    const char* module;
    int64_t out;
  };
  const P projections[] = {
      {"text_embedding_projection.video_aggregate_embed", video_out},
      {"text_embedding_projection.audio_aggregate_embed", audio_out},
      {"vision_model.patch_dense", 128},  // present, out of scope, must not choke
  };
  for (const P& pr : projections) {
    const std::string m = pr.module;
    const int64_t groups = in_features / 16;
    const std::vector<uint8_t> packed =
        RandBytes(m + ".w", static_cast<size_t>(pr.out * in_features / 2));
    std::vector<uint8_t> lin(static_cast<size_t>(pr.out * groups));
    const std::vector<uint8_t> raw = RandBytes(m + ".s", lin.size());
    for (size_t i = 0; i < lin.size(); ++i) {
      lin[i] = (raw[i] == 0x7F || raw[i] == 0xFF) ? 0x38 : raw[i];
    }
    const std::vector<uint8_t> sw = SwizzleBlockScale(lin, pr.out, groups);
    const float scale2 = 0.0078125F;
    entries.push_back({m + ".weight", "U8", {pr.out, in_features / 2}, PackBytes(packed)});
    entries.push_back(
        {m + ".weight_scale", "F8_E4M3", {pr.out / 4, groups * 4}, PackBytes(sw)});
    entries.push_back({m + ".weight_scale_2", "F32", {},
                       std::string(reinterpret_cast<const char*>(&scale2), 4)});
    entries.push_back({m + ".torchao_nvfp4", "U8",
                       {static_cast<int64_t>(std::strlen(marker))}, std::string(marker)});
    if (expected != nullptr && m.rfind("text_embedding_projection", 0) == 0) {
      std::vector<uint16_t> want(static_cast<size_t>(pr.out * in_features));
      vllm::DequantNvfp4ToBf16(packed.data(), lin.data(), scale2, pr.out, in_features,
                               want.data());
      (*expected)[m] = want;
    }
    if (m.rfind("text_embedding_projection", 0) == 0) {
      std::vector<float> bias(static_cast<size_t>(pr.out));
      for (size_t i = 0; i < bias.size(); ++i) bias[i] = TrueValue(m + ".bias", i);
      entries.push_back({m + ".bias", "BF16", {pr.out}, PackBf16(bias)});
    }
  }
  entries.push_back({"tokenizer_json", "U8", {5}, std::string("{\"a\":")});
  entries.push_back({"hf_asset__tokenizer_config.json", "U8", {2}, std::string("{}")});
  entries.push_back({"hf_asset__processor_config.json", "U8", {2}, std::string("{}")});
  return entries;
}

}  // namespace

TEST_CASE("ltx2 loader: the torchao text encoder materializes onto the L3 contract") {
  const int64_t hidden = 128, layers = 3, video_out = 256, audio_out = 128;
  std::map<std::string, std::vector<uint16_t>> expected;
  const std::vector<StEntry> entries =
      BuildSyntheticTe(hidden, layers, video_out, audio_out, &expected);
  const std::string path = TmpPath("te");
  WriteSafetensors(entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  const vllm::Ltx2TextEncoderCheckpoint ck =
      vllm::Ltx2LoadTextEncoderFromSafetensors(file);
  CHECK(ck.gemma_hidden_size == hidden);
  CHECK(ck.gemma_num_hidden_layers == layers);
  CHECK(ck.video.out_features == video_out);
  CHECK(ck.video.in_features == hidden * (layers + 1));
  CHECK(ck.audio.out_features == audio_out);
  // The bias comes off a DIFFERENT dtype path and is the one a U8-only loader
  // drops; ltx2_text_encoder.h:264-269 names that failure exactly.
  CHECK(ck.video.bias_bf16.size() == static_cast<size_t>(video_out));
  CHECK(ck.audio.bias_bf16.size() == static_cast<size_t>(audio_out));
  // The multimodal tower is present and did NOT choke the loader.
  CHECK(ck.quantized_modules.size() == 3);
  // The tokenizer came out of the tensor, not a sibling file.
  CHECK(ck.assets.tokenizer_json.size() == 5);
  CHECK_FALSE(ck.assets.has_config);

  int64_t bad = 0, checked = 0;
  for (const auto& kv : expected) {
    const std::vector<uint16_t>& want = kv.second;
    const std::vector<uint16_t>& got =
        kv.first.find("video") != std::string::npos ? ck.video.weight_bf16
                                                    : ck.audio.weight_bf16;
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < want.size(); ++i) {
      ++checked;
      if (got[i] != want[i]) ++bad;
    }
  }
  INFO("te checked=" << checked << " bad=" << bad);
  CHECK(checked > 0);
  CHECK(bad == 0);

  // The f32 widening is opt-in and lands on L3's own contract.
  const vllm::Ltx2TextEncoderWeights w = vllm::Ltx2WidenTextProjectionsToF32(ck);
  CHECK(w.video.out_features == video_out);
  CHECK(w.video.in_features == hidden * (layers + 1));
  CHECK(w.video.bias.size() == static_cast<size_t>(video_out));
  REQUIRE(w.video.weight.size() == ck.video.weight_bf16.size());
  std::vector<float> te_want(w.video.weight.size(), 0.0F);
  for (size_t i = 0; i < te_want.size(); ++i) te_want[i] = Bf16ToF32(ck.video.weight_bf16[i]);
  const double max_abs = vllm_test::MaxAbsDiff(w.video.weight, te_want);
  INFO("te widen max abs = " << max_abs);
  CHECK(max_abs == 0.0);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: an incomplete torchao module throws BY NAME") {
  std::vector<StEntry> entries = BuildSyntheticTe(128, 3, 256, 128, nullptr);
  std::vector<StEntry> kept;
  for (const StEntry& e : entries) {
    if (e.name == "vision_model.patch_dense.weight_scale") continue;
    kept.push_back(e);
  }
  const std::string path = TmpPath("te_incomplete");
  WriteSafetensors(kept, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);
  std::string what;
  try {
    vllm::Ltx2LoadTextEncoderFromSafetensors(file);
  } catch (const std::exception& e) {
    what = e.what();
  }
  const std::string what_msg = "what: " + what;
  INFO(what_msg);
  CHECK(what.find("vision_model.patch_dense") != std::string::npos);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: require_config mirrors upstream's refusal of a metadata-less pack") {
  const std::vector<StEntry> entries = BuildSyntheticTe(128, 3, 256, 128, nullptr);
  const std::string path = TmpPath("te_cfg");
  WriteSafetensors(entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);
  vllm::Ltx2TextEncoderLoadOptions options;
  options.require_config = true;
  CHECK_THROWS(vllm::Ltx2LoadTextEncoderFromSafetensors(file, options));
  std::remove(path.c_str());
}

// ===========================================================================
// 8. The contract's OWN refusal, which ltx2.h:228-232 promises and did not give
// ===========================================================================

TEST_CASE("ltx2 loader: BindLtx2DitWeights names the tensor it is missing") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("bindname");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);

  // ltx2.h:228-232 states that a name the contract requires and the map lacks
  // "throws BY NAME rather than reading as zeros". The loader above enforces
  // that before it binds, but the SEAM itself must too: a caller assembling its
  // own map — which is exactly what the L2 parity suite does — gets no such
  // pre-pass, and "a required tensor is missing" does not tell it which.
  std::map<std::string, vt::Tensor> views = ck.views;
  const std::string dropped = "transformer_blocks.1.audio_to_video_attn.to_out.0.weight";
  REQUIRE(views.erase(dropped) == 1);
  std::string what;
  try {
    vllm::BindLtx2DitWeights(ck.params, views);
  } catch (const std::exception& e) {
    what = e.what();
  }
  const std::string bind_msg = "what: " + what;
  INFO(bind_msg);
  CHECK(what.find(dropped) != std::string::npos);
  std::remove(path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// IC-LoRA fusion on the QUANTIZED arms (row LTX25-IC-LORA, issue #923)
// ─────────────────────────────────────────────────────────────────────────────
//
// WHY THESE TWO CASES EXIST AND ARE NOT REDUNDANT WITH test_ltx2_lora. That
// suite gates the fusion arithmetic against a bf16 buffer it hands over itself.
// It cannot show that an FP8 or NVFP4 CHECKPOINT reaches the same code, and the
// design claim this row makes is precisely that one hook serves every arm
// because `MaterializeDitTensor`'s `F8_E4M3` and `U8` branches both
// `return vt::DType::kBF16` before anything else sees a byte.
//
// If that claim were wrong — if either quantized branch returned its packed
// dtype — the fuser's final `Fail` would fire and these cases would throw. So
// they are the gate on the "one hook serves four arms" design, not a second
// copy of the arithmetic gate.
namespace {

// Write a one-target IC-LoRA whose delta is a known constant, against the same
// contract `BuildSyntheticDit` writes.
std::string WriteLoraFor(const Ltx2DitParams& p, const std::string& target, float scale,
                         const std::string& path) {
  std::vector<int64_t> shape;
  for (const Ltx2TensorSpec& spec : vllm::EnumerateLtx2DitTensors(p)) {
    if (spec.name == target) shape = spec.shape;
  }
  REQUIRE_MESSAGE(shape.size() == 2, "LoRA target '", target, "' is not rank 2");
  const int64_t rank = 2;
  const std::string module = target.substr(0, target.size() - std::string(".weight").size());

  std::vector<StEntry> entries;
  {
    StEntry a;
    a.name = "diffusion_model." + module + ".lora_A.weight";
    a.dtype = "BF16";
    a.shape = {rank, shape[1]};
    for (int64_t i = 0; i < rank * shape[1]; ++i) {
      const uint16_t v = vt::F32ToBF16(1.0F);
      a.bytes.append(reinterpret_cast<const char*>(&v), sizeof(v));
    }
    entries.push_back(std::move(a));

    StEntry b;
    b.name = "diffusion_model." + module + ".lora_B.weight";
    b.dtype = "BF16";
    b.shape = {shape[0], rank};
    for (int64_t i = 0; i < shape[0] * rank; ++i) {
      const uint16_t v = vt::F32ToBF16(scale);
      b.bytes.append(reinterpret_cast<const char*>(&v), sizeof(v));
    }
    entries.push_back(std::move(b));
  }
  WriteSafetensors(entries, path);
  return path;
}

// The delta every case above produces: rank terms of `1.0 * scale`, so a
// uniform `rank * scale` on every element of the target.
constexpr float kLoraScale = 0.5F;
constexpr float kExpectedDelta = 2 * kLoraScale;

void CheckArmFuses(Ltx2DitQuant quant, const char* tag) {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, quant, {});
  const std::string dit_path = TmpPath((std::string("lora_") + tag).c_str());
  WriteSafetensors(syn.entries, dit_path);

  // A target that is quantized on BOTH arms: a rank-2 non-table weight.
  const std::string target = "transformer_blocks.0.attn1.to_q.weight";
  const std::string lora_path =
      WriteLoraFor(p, target, kLoraScale, TmpPath((std::string("lora_a_") + tag).c_str()));

  const SafetensorsFile file = SafetensorsFile::Open(dit_path);

  // The control: the same checkpoint with no adapter, so the delta is measured
  // against what this loader actually produces rather than against an
  // independently computed dequantization.
  const vllm::Ltx2DitCheckpoint plain = vllm::Ltx2LoadDitFromSafetensors(file);
  REQUIRE(plain.lora_fused_tensors == 0);

  vllm::Ltx2DitLoadOptions options;
  vllm::Ltx2LoraSpec spec;
  spec.path = lora_path;
  spec.strength = 1.0;
  options.loras.push_back(spec);
  const vllm::Ltx2DitCheckpoint fused = vllm::Ltx2LoadDitFromSafetensors(file, options);

  INFO("arm = ", std::string(tag));
  // Exactly one contract tensor was touched.
  CHECK(fused.lora_fused_tensors == 1);
  // No adapter metadata, so both factors are upstream's default of 1.
  CHECK(fused.lora_reference.downscale == 1);
  CHECK(fused.lora_reference.temporal == 1);

  const vt::Tensor& before = plain.views.at(target);
  const vt::Tensor& after = fused.views.at(target);
  REQUIRE(before.dtype == vt::DType::kBF16);
  REQUIRE(after.dtype == vt::DType::kBF16);
  REQUIRE(before.Numel() == after.Numel());

  // THE VALUE CLAIM: every element moved by exactly the delta. Checked
  // element-wise rather than as a norm, because a norm cannot see a delta
  // applied to the wrong half of the tensor.
  const uint16_t* b0 = before.Ptr<uint16_t>();
  const uint16_t* a0 = after.Ptr<uint16_t>();
  int64_t moved = 0;
  for (int64_t i = 0; i < before.Numel(); ++i) {
    // Rounded to bf16, because the STORE is bf16: comparing against an f32 sum
    // fails on the last mantissa bit for every element whose sum is not exactly
    // representable, which is most of them.
    const float want = vt::BF16ToF32(vt::F32ToBF16(vt::BF16ToF32(b0[i]) + kExpectedDelta));
    CHECK(vt::BF16ToF32(a0[i]) == doctest::Approx(want));
    if (a0[i] != b0[i]) ++moved;
  }
  // And it is not vacuous: the delta is non-zero, so the bytes must actually
  // differ. A fuser that returned early would pass the Approx loop above only
  // if `kExpectedDelta` were 0.
  CHECK(moved == before.Numel());

  // A tensor NO adapter targets is byte-identical, which is what shows the hook
  // is keyed on the name rather than applied to everything materialized.
  const std::string untouched = "transformer_blocks.0.attn1.to_k.weight";
  const vt::Tensor& u0 = plain.views.at(untouched);
  const vt::Tensor& u1 = fused.views.at(untouched);
  REQUIRE(u0.Numel() == u1.Numel());
  int64_t untouched_moved = 0;
  for (int64_t i = 0; i < u0.Numel(); ++i) {
    if (u0.Ptr<uint16_t>()[i] != u1.Ptr<uint16_t>()[i]) ++untouched_moved;
  }
  CHECK(untouched_moved == 0);

  std::remove(dit_path.c_str());
  std::remove(lora_path.c_str());
}

}  // namespace

TEST_CASE("ltx2 loader: an IC-LoRA fuses into the FP8 arm") {
  // The arm most users run. `MaterializeDitTensor`'s F8_E4M3 branch calls
  // DequantFp8ToBf16 and returns kBF16, so the delta lands on a dequantized
  // weight and no FP8 quantizer is needed — which is the whole reason this port
  // can serve every arm with one hook (ltx2_lora.h, the WHERE IT IS APPLIED
  // note).
  CheckArmFuses(Ltx2DitQuant::kFp8, "fp8");
}

TEST_CASE("ltx2 loader: an IC-LoRA fuses into the NVFP4 arm") {
  // Same claim, the other quantized branch: U8-packed NVFP4 with its two scale
  // sidecars, dequantized by Ltx2DequantNvfp4ToBf16 which also returns kBF16.
  //
  // Upstream RE-QUANTIZES here (quantization/nvfp4/fuse.py:40-47) because it
  // keeps the packed weight resident for its NVFP4 kernels. This tree keeps
  // bf16 (ltx2_loader.h's DTYPE note), so there is nothing to re-quantize into
  // and no NVFP4 quantizer in the tree to do it with. The divergence is
  // deliberate and recorded in the row's spec §3.1: our fused weight SKIPS
  // upstream's lossy round trip, at no extra bytes.
  CheckArmFuses(Ltx2DitQuant::kNvfp4, "nvfp4");
}

TEST_CASE("ltx2 loader: an adapter that fuses into NOTHING refuses rather than loading green") {
  // Reachable because the contract carries rank-1 and rank-3 tensors a LoRA
  // pair can legitimately name but never fuse into. A load that reported
  // success here would render byte-identically to no adapter at all.
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string dit_path = TmpPath("lora_none");
  WriteSafetensors(syn.entries, dit_path);

  // A non-rank-2 contract tensor: the scale-shift table is [rows, inner] rank 2,
  // so pick a genuinely rank-1 one.
  std::string rank1;
  for (const Ltx2TensorSpec& spec : vllm::EnumerateLtx2DitTensors(p)) {
    const bool is_weight = spec.name.size() > 7 &&
                           spec.name.compare(spec.name.size() - 7, 7, ".weight") == 0;
    if (spec.shape.size() == 1 && is_weight) {
      rank1 = spec.name;
      break;
    }
  }
  REQUIRE_MESSAGE(!rank1.empty(), "the contract has no rank-1 tensor to build this case on");

  const std::string module = rank1.substr(0, rank1.size() - std::string(".weight").size());
  std::vector<StEntry> entries;
  for (const char* side : {".lora_A.weight", ".lora_B.weight"}) {
    StEntry e;
    e.name = "diffusion_model." + module + side;
    e.dtype = "BF16";
    e.shape = {1, 1};
    const uint16_t v = vt::F32ToBF16(1.0F);
    e.bytes.append(reinterpret_cast<const char*>(&v), sizeof(v));
    entries.push_back(std::move(e));
  }
  const std::string lora_path = TmpPath("lora_none_a");
  WriteSafetensors(entries, lora_path);

  const SafetensorsFile file = SafetensorsFile::Open(dit_path);
  vllm::Ltx2DitLoadOptions options;
  vllm::Ltx2LoraSpec spec;
  spec.path = lora_path;
  options.loras.push_back(spec);

  std::string what;
  try {
    (void)vllm::Ltx2LoadDitFromSafetensors(file, options);
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("fused into ZERO tensors") != std::string::npos);
  CHECK(what.find(lora_path) != std::string::npos);

  std::remove(dit_path.c_str());
  std::remove(lora_path.c_str());
}

// ─── LTX25-PHASE-LORA (#1118) ────────────────────────────────────────────────

namespace {

// Every bound view of `a` against `b`, byte for byte. Returns the number of
// views that DIFFER, so a caller can assert both directions and neither
// assertion is vacuous.
//
// Byte-for-byte and not a tolerance: the claim `Ltx2RebindDitLoras` makes is
// that a rebound checkpoint is INDISTINGUISHABLE from one the loader produced,
// and a tolerance would pass an implementation that reconstructs the base by
// subtracting the delta — which is the wrong implementation this case exists to
// catch.
int64_t ViewsDiffering(const vllm::Ltx2DitCheckpoint& a, const vllm::Ltx2DitCheckpoint& b) {
  REQUIRE(a.views.size() == b.views.size());
  int64_t differing = 0;
  for (const auto& kv : a.views) {
    const auto it = b.views.find(kv.first);
    REQUIRE(it != b.views.end());
    const vt::Tensor& x = kv.second;
    const vt::Tensor& y = it->second;
    REQUIRE(x.dtype == y.dtype);
    REQUIRE(x.Numel() == y.Numel());
    const size_t bytes = static_cast<size_t>(x.Numel()) * vt::SizeOf(x.dtype);
    if (std::memcmp(x.data, y.data, bytes) != 0) ++differing;
  }
  return differing;
}

}  // namespace

TEST_CASE("ltx2 loader: a phase rebind reproduces the load BYTE-FOR-BYTE in both directions") {
  // THE EXACTNESS CLAIM of row LTX25-PHASE-LORA (#1118), made executable.
  //
  // Upstream gives each `DiffusionStage` its own adapter set by building a
  // SECOND stage from the same checkpoint (ic_lora.py:104 and :115). This port
  // holds one DiT and moves it between the two states, so the thing that has to
  // be proved is that "moved back" and "loaded that way" are the same bytes —
  // otherwise the seam buys per-phase adapters at the cost of a silent numeric
  // drift, which is the trade this row rejected when it rejected unfused runtime
  // LoRA.
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string dit_path = TmpPath("rebind_dit");
  WriteSafetensors(syn.entries, dit_path);

  const std::string target = "transformer_blocks.0.attn1.to_q.weight";
  const std::string lora_path = WriteLoraFor(p, target, kLoraScale, TmpPath("rebind_lora"));

  const SafetensorsFile file = SafetensorsFile::Open(dit_path);

  vllm::Ltx2DitLoadOptions options;
  vllm::Ltx2LoraSpec spec;
  spec.path = lora_path;
  spec.strength = 1.0;
  options.loras.push_back(spec);

  // The two REFERENCE images, each produced by the loader itself.
  const vllm::Ltx2DitCheckpoint plain = vllm::Ltx2LoadDitFromSafetensors(file);
  const vllm::Ltx2DitCheckpoint fused = vllm::Ltx2LoadDitFromSafetensors(file, options);
  REQUIRE(plain.lora_fused_tensors == 0);
  REQUIRE(fused.lora_fused_tensors == 1);
  // The instrument is armed: the two references genuinely differ, in exactly the
  // one tensor the adapter targets. Without this line every equality below could
  // pass on a checkpoint where the adapter did nothing.
  REQUIRE(ViewsDiffering(plain, fused) == 1);

  // The one under test, loaded FUSED exactly as `Ltx2VideoEngine::Load` loads it.
  vllm::Ltx2DitCheckpoint live = vllm::Ltx2LoadDitFromSafetensors(file, options);
  REQUIRE(live.lora_fused_tensors == 1);
  // The pointer the bound weights read through. `Ltx2DitWeights` is a pure view
  // struct, so a rebind that reallocated would leave `live.weights` dangling and
  // every forward would read freed memory. Captured before, checked after.
  const void* const target_before = live.views.at(target).data;

  SUBCASE("rebound OFF, it is the checkpoint the loader builds with no adapter") {
    vllm::Ltx2RebindDitLoras(/*queue=*/nullptr, file, options, /*fuse=*/false, live);
    CHECK(live.lora_fused_tensors == 0);
    CHECK(live.views.at(target).data == target_before);
    // The whole claim, in one number: nothing distinguishes it from `plain`.
    CHECK(ViewsDiffering(live, plain) == 0);
    // And it really moved — this is what fails if the rebind quietly did nothing.
    CHECK(ViewsDiffering(live, fused) == 1);
  }

  SUBCASE("rebound OFF then ON, it is the checkpoint the loader builds WITH the adapter") {
    vllm::Ltx2RebindDitLoras(/*queue=*/nullptr, file, options, /*fuse=*/false, live);
    vllm::Ltx2RebindDitLoras(/*queue=*/nullptr, file, options, /*fuse=*/true, live);
    CHECK(live.lora_fused_tensors == 1);
    CHECK(live.views.at(target).data == target_before);
    // THE ROUND TRIP. An implementation that reconstructed the base by
    // SUBTRACTING the delta would land here at
    // `round_bf16(round_bf16(W + d) - d) + d`, which is not `round_bf16(W + d)`
    // for every element, and this line is what tells the two apart.
    CHECK(ViewsDiffering(live, fused) == 0);
    CHECK(ViewsDiffering(live, plain) == 1);
  }

  SUBCASE("a rebind to the state it is already in changes nothing") {
    // The no-op a one-stage recipe and every recipe predating the phase field
    // relies on: they never ask for a different set, and they must not pay a
    // re-materialization to be told so.
    vllm::Ltx2RebindDitLoras(/*queue=*/nullptr, file, options, /*fuse=*/true, live);
    CHECK(live.lora_fused_tensors == 1);
    CHECK(ViewsDiffering(live, fused) == 0);
  }

  SUBCASE("a queue for a HOST checkpoint refuses by name") {
    // The two address spaces. Guessing which one a view points at is how a
    // rebind would corrupt weights silently instead of refusing.
    vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    std::string what;
    try {
      vllm::Ltx2RebindDitLoras(&queue, file, options, /*fuse=*/false, live);
    } catch (const std::exception& e) {
      what = e.what();
    }
    INFO("what: ", what);
    CHECK(what.find("a queue for a HOST-resident checkpoint") != std::string::npos);
    // It refused BEFORE touching anything.
    CHECK(live.lora_fused_tensors == 1);
    CHECK(ViewsDiffering(live, fused) == 0);
  }

  std::remove(dit_path.c_str());
  std::remove(lora_path.c_str());
}

TEST_CASE("ltx2 loader: a rebind of a WIDENED checkpoint fuses in bf16 and stores f32") {
  // The host arm the CPU parity forward actually runs: `widen_to_f32` is set
  // whenever the engine is off-device (`ltx2_video.cpp`), so the bound view is
  // an f32 copy while `MaterializeDitTensor` still returns bf16. If the rebind
  // widened BEFORE fusing it would accumulate the delta in f32 and quietly undo
  // the accumulator dtype `ltx2_lora.h` pins — which no token gate and no
  // golden could see, because the numbers would still be finite and close.
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string dit_path = TmpPath("rebind_wide_dit");
  WriteSafetensors(syn.entries, dit_path);

  const std::string target = "transformer_blocks.0.attn1.to_q.weight";
  const std::string lora_path = WriteLoraFor(p, target, kLoraScale, TmpPath("rebind_wide_lora"));

  const SafetensorsFile file = SafetensorsFile::Open(dit_path);

  vllm::Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;
  vllm::Ltx2LoraSpec spec;
  spec.path = lora_path;
  spec.strength = 1.0;
  options.loras.push_back(spec);

  vllm::Ltx2DitLoadOptions plain_options;
  plain_options.widen_to_f32 = true;

  const vllm::Ltx2DitCheckpoint plain = vllm::Ltx2LoadDitFromSafetensors(file, plain_options);
  const vllm::Ltx2DitCheckpoint fused = vllm::Ltx2LoadDitFromSafetensors(file, options);
  REQUIRE(plain.views.at(target).dtype == vt::DType::kF32);
  REQUIRE(ViewsDiffering(plain, fused) == 1);

  vllm::Ltx2DitCheckpoint live = vllm::Ltx2LoadDitFromSafetensors(file, options);
  const void* const target_before = live.views.at(target).data;

  vllm::Ltx2RebindDitLoras(/*queue=*/nullptr, file, options, /*fuse=*/false, live);
  CHECK(live.views.at(target).data == target_before);
  CHECK(live.views.at(target).dtype == vt::DType::kF32);
  CHECK(ViewsDiffering(live, plain) == 0);

  vllm::Ltx2RebindDitLoras(/*queue=*/nullptr, file, options, /*fuse=*/true, live);
  CHECK(ViewsDiffering(live, fused) == 0);
  CHECK(ViewsDiffering(live, plain) == 1);

  std::remove(dit_path.c_str());
  std::remove(lora_path.c_str());
}
