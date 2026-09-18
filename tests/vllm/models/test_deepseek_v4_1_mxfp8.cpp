// DeepSeek-V4.1-Flash W3b UNIT GATE — the MXFP8 32x32 UE8M0 linear family.
//
// PORTED from `tests/quantization/test_fp8.py` @ vLLM `e77daef89e`:
//   :67-199  `test_deepseek_v41_mxfp8_scale_loading` — a BIT-EXACT
//            (`rtol=0, atol=0`) oracle for the scale expansion, parametrized
//            over `tp_rank in [0, 1]` and both checkpoint scale dtypes, over the
//            six checkpoint shapes it names, with the shared-expert 96 -> 128
//            zero pad, and with the shard taken AFTER the expansion.
//   :202-238 `test_deepseek_v41_vl_mapper_routes_linear_scales` — the three
//            `(weight_block_size, expert_dtype) -> scale name` cases.
//
// HARNESS ADAPTATIONS — three, and they are adaptations of HOW the same
// assertion is spelled, not of WHAT is asserted:
//   * Upstream parametrizes `scale_dtype in [torch.uint8, torch.float8_e8m0fnu]`
//     and applies it with `.view(scale_dtype)`, a REINTERPRET of the same bytes
//     with no conversion. On the host both arms are the identical `uint8_t`
//     buffer, so the parametrization collapses to one case rather than being
//     dropped. `Mxfp8LinearScaleParamName` keeps the dtype-independent half of
//     what that parametrization was checking: the NAME the bytes land under.
//   * Upstream builds its tensors with `torch.randint`. The assertions here are
//     equivalences between two computations of the SAME buffer, so the RNG
//     STREAM is not part of the oracle; a fixed-seed `mt19937` draws from the
//     upstream ranges (`randint(-4, 5)` for the e4m3 bytes, `randint(124, 131)`
//     for the scale bytes) and the shapes are upstream's verbatim.
//   * Upstream drives the expansion through vLLM's `ColumnParallelLinear` /
//     `MergedColumnParallelLinear` / `RowParallelLinear` weight loaders, which
//     this project does not have (the loader arm is W8). The test shards
//     explicitly along the same axis each of those classes shards along, which
//     is what `dequant.chunk(2, dim=axis)[tp_rank]` on the reference side does
//     upstream.
//
// NOT PORTED — three upstream assertions this gate does NOT carry, each named
// with what it guarantees and who owes it. Every one of them asserts a property
// of the LOADER, which this tree does not have yet; none of them is a property
// of the functions W3b lands, so there is nothing here they could be attached
// to. All three are owed by the wiring waves on this row and are
// recorded in `ISSUE-LOCAL-01M2C2QSFZWCXNQBJFBWYFNV2P`:
//   * `test_fp8.py:186-188` — the OTHER half of the expansion guarantee. The
//     MoE expert scale (`layers.0.ffn.experts.0.w1.weight_scale`) and the engram
//     scale (`layers.0.engram.weight_scale_inv`) are asserted EQUAL to the raw
//     3x4 checkpoint bytes, which is to say the row expansion must NOT be
//     applied to them. `ExpandMxfp8CheckpointScale` is an explicit call, so
//     nothing here can apply it by accident; what can apply it by accident is a
//     loader that wires it onto the wrong parameter. **Owed by W8.**
//   * `test_fp8.py:191` — `isinstance(linear.quant_method, ModelOptLinearMethod)`:
//     the routing assertion. `quant_config.py:186-201` is listed in the header's
//     port table as the ROUTING that selects this family, and W3b ports no
//     routing predicate at all, only the five leaves it routes to. **Owed by W8**
//     together with the `get_quant_method` port itself.
//   * `test_fp8.py:226-237` — the mapper-rename half of the name split. This
//     gate holds `_linear_scale_param_name` (`:223-224`); it does not hold that
//     `_make_deepseek_v4_vl_weights_mapper` then renames `...attn.wq_a.scale` to
//     `...attn.wq_a.<that name>` while leaving `...attn.wkv.weight` alone.
//     **Owed by W8** (the mapper is loader-side).
//
// DERIVED, because upstream has no test that can be transcribed: no upstream
// test exercises the emulation kernel by name, and every mxfp8 linear test is
// CUDA/ROCm-gated (`@pytest.mark.skipif(not current_platform.is_cuda())` on
// every case in this file). So the quantizer, the emulation linear arm and the
// 0xFF divergence are gated from FIRST PRINCIPLES in DOUBLE precision, exactly
// as `test_deepseek_v4_moe.cpp` / `test_deepseek_v4_mhc.cpp` did for V4.
#include "vllm/model_executor/models/deepseek_v4_1_mxfp8.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/mxfp4_dequant.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/dtype.h"

using namespace vllm::deepseek_v4_1;

namespace {

// One checkpoint tensor, upstream's tuple minus the vLLM module plumbing:
// (name, N, K, shard axis). `axis` is the dimension the owning parallel linear
// shards along — 0 for the Column/MergedColumn cases, 1 for the Row case —
// exactly as upstream's `dequant.chunk(2, dim=axis)[tp_rank]`.
struct CkptTensor {
  const char* source;
  int64_t n;
  int64_t k;
  int axis;
  bool shared_expert;  // upstream zero-pads these to 128x128 before chunking
};

// test_fp8.py:126-150, verbatim shapes and axes.
const CkptTensor kCheckpoints[] = {
    {"attn.wq_b", 128, 128, 0, false},
    {"attn.wq_a", 128, 128, 0, false},
    {"attn.wkv", 64, 128, 0, false},
    {"ffn.shared_experts.w1", 96, 128, 0, true},
    {"ffn.shared_experts.w3", 96, 128, 0, true},
    {"ffn.shared_experts.down_proj", 128, 96, 1, true},
};

std::vector<uint8_t> DrawWeightBytes(std::mt19937& rng, int64_t n, int64_t k) {
  // `torch.randint(-4, 5, (n, k)).to(torch.float8_e4m3fn)`: small exact integers,
  // every one of which e4m3 represents exactly.
  std::uniform_int_distribution<int> d(-4, 4);
  std::vector<uint8_t> out(static_cast<size_t>(n * k));
  for (auto& b : out) b = vllm::F32ToF8E4M3(static_cast<float>(d(rng)));
  return out;
}

std::vector<uint8_t> DrawScaleBytes(std::mt19937& rng, int64_t rows, int64_t cols) {
  // `torch.randint(124, 131, ..., dtype=torch.uint8)`: exponents around 1.0.
  std::uniform_int_distribution<int> d(124, 130);
  std::vector<uint8_t> out(static_cast<size_t>(rows * cols));
  for (auto& b : out) b = static_cast<uint8_t>(d(rng));
  return out;
}

// Upstream's REFERENCE side (test_fp8.py:157-161): the CHECKPOINT layout,
// `weight.reshape(n/32, 32, k/32, 32) * exp2(scale - 127)[:, None, :, None]`.
// Computed by the function this tree ALREADY has rather than re-written here.
std::vector<float> CheckpointLayoutDequant(const std::vector<uint8_t>& w,
                                           const std::vector<uint8_t>& s, int64_t n,
                                           int64_t k) {
  std::vector<float> out(static_cast<size_t>(n * k));
  vllm::DequantFp8BlockToF32(w.data(), s.data(), n, k, kMxfp8BlockSize, kMxfp8BlockSize,
                             out.data());
  return out;
}

// `padded = torch.zeros(128, 128); padded[:n, :k] = dequant` (test_fp8.py:162-165).
std::vector<float> ZeroPad(const std::vector<float>& in, int64_t n, int64_t k,
                           int64_t pn, int64_t pk) {
  std::vector<float> out(static_cast<size_t>(pn * pk), 0.0F);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < k; ++j) out[i * pk + j] = in[i * k + j];
  return out;
}

// `t.chunk(2, dim=axis)[tp_rank]` for an even extent.
std::vector<float> Chunk2(const std::vector<float>& in, int64_t n, int64_t k, int axis,
                          int rank) {
  if (axis == 0) {
    const int64_t rows = n / 2;
    std::vector<float> out(static_cast<size_t>(rows * k));
    for (int64_t i = 0; i < rows; ++i)
      for (int64_t j = 0; j < k; ++j) out[i * k + j] = in[(rank * rows + i) * k + j];
    return out;
  }
  const int64_t cols = k / 2;
  std::vector<float> out(static_cast<size_t>(n * cols));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < cols; ++j) out[i * cols + j] = in[i * k + rank * cols + j];
  return out;
}

// Byte-level equivalents of ZeroPad / Chunk2 for the [rows, cols] scale plane,
// so the runtime side can be built the way the loader builds it: expand, pad,
// shard, THEN dequant. The scale plane's column extent is K/32, so a shard along
// K (axis 1) shards the scale along K/32.
std::vector<uint8_t> ZeroPadBytes(const std::vector<uint8_t>& in, int64_t n, int64_t k,
                                  int64_t pn, int64_t pk) {
  std::vector<uint8_t> out(static_cast<size_t>(pn * pk), 0U);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < k; ++j) out[i * pk + j] = in[i * k + j];
  return out;
}

std::vector<uint8_t> Chunk2Bytes(const std::vector<uint8_t>& in, int64_t n, int64_t k,
                                 int axis, int rank) {
  if (axis == 0) {
    const int64_t rows = n / 2;
    std::vector<uint8_t> out(static_cast<size_t>(rows * k));
    for (int64_t i = 0; i < rows; ++i)
      for (int64_t j = 0; j < k; ++j) out[i * k + j] = in[(rank * rows + i) * k + j];
    return out;
  }
  const int64_t cols = k / 2;
  std::vector<uint8_t> out(static_cast<size_t>(n * cols));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < cols; ++j) out[i * cols + j] = in[i * k + rank * cols + j];
  return out;
}

// An independent DOUBLE-PRECISION dequant, written from the upstream FORMULA
// rather than by calling the f32 implementation.
std::vector<double> RuntimeDequantD(const std::vector<uint8_t>& w,
                                    const std::vector<uint8_t>& s, int64_t n,
                                    int64_t k) {
  const int64_t sc = k / kMxfp8BlockSize;
  std::vector<double> out(static_cast<size_t>(n * k));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < k; ++j)
      out[i * k + j] = static_cast<double>(vllm::F8E4M3ToF32(w[i * k + j])) *
                       std::exp2(static_cast<double>(s[i * sc + j / kMxfp8BlockSize]) -
                                 127.0);
  return out;
}

double RelL2(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return std::sqrt(num) / std::max(std::sqrt(den), 1e-30);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// (1) THE PORTED BIT-EXACT ORACLE — test_fp8.py:67-199
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE(
    "V4.1 MXFP8: expanding the 32x32 checkpoint scale reproduces the checkpoint-layout "
    "dequant exactly, for every tp_rank and every V4.1 linear shape") {
  // test_fp8.py:186-199. `torch.testing.assert_close(actual, reference,
  // rtol=0, atol=0)` — BIT-exact, not a tolerance.
  for (int tp_rank : {0, 1}) {
    CAPTURE(tp_rank);
    // The SAME seed for both ranks, deliberately: upstream draws its tensors
    // once and then takes `chunk(2, dim=axis)[tp_rank]` of that one tensor, so
    // the two ranks are two shards of ONE checkpoint. Seeding per rank would
    // give each rank a different tensor and would stop the pair from saying
    // anything about sharding at all.
    std::mt19937 rng(0xD54100U);
    for (const auto& t : kCheckpoints) {
      CAPTURE(t.source);
      const auto w = DrawWeightBytes(rng, t.n, t.k);
      const auto s = DrawScaleBytes(rng, t.n / kMxfp8BlockSize, t.k / kMxfp8BlockSize);

      // REFERENCE side: the checkpoint layout, padded, then sharded
      // (test_fp8.py:157-166).
      auto reference = CheckpointLayoutDequant(w, s, t.n, t.k);
      int64_t rn = t.n, rk = t.k;
      if (t.shared_expert) {
        reference = ZeroPad(reference, t.n, t.k, 128, 128);
        rn = 128;
        rk = 128;
      }
      reference = Chunk2(reference, rn, rk, t.axis, tp_rank);

      // ACTUAL side, the way the loader builds it: EXPAND the scale rows first
      // (modelopt.py:2186-2201), then pad, then shard, then dequant with the
      // RUNTIME per-32-column rule (test_fp8.py:193-196).
      auto es = ExpandMxfp8CheckpointScale(s.data(), t.n / kMxfp8BlockSize,
                                           t.k / kMxfp8BlockSize, kMxfp8BlockSize,
                                           kMxfp8BlockSize);
      auto pw = w;
      int64_t an = t.n, ak = t.k;
      if (t.shared_expert) {
        // The zero pad reaches the weight AND the scale plane; a zero scale byte
        // is 2^-127, and a zero e4m3 weight makes the product 0 either way.
        std::vector<float> wf(static_cast<size_t>(t.n * t.k));
        for (int64_t i = 0; i < t.n * t.k; ++i) wf[i] = vllm::F8E4M3ToF32(w[i]);
        const auto pwf = ZeroPad(wf, t.n, t.k, 128, 128);
        pw.assign(128 * 128, 0U);
        for (size_t i = 0; i < pwf.size(); ++i) pw[i] = vllm::F32ToF8E4M3(pwf[i]);
        es = ZeroPadBytes(es, t.n, t.k / kMxfp8BlockSize, 128, 128 / kMxfp8BlockSize);
        an = 128;
        ak = 128;
      }
      const auto sw = Chunk2Bytes(pw, an, ak, t.axis, tp_rank);
      const auto ss = Chunk2Bytes(es, an, ak / kMxfp8BlockSize, t.axis, tp_rank);
      const int64_t sn = t.axis == 0 ? an / 2 : an;
      const int64_t sk = t.axis == 0 ? ak : ak / 2;

      std::vector<float> actual(static_cast<size_t>(sn * sk));
      DequantMxfp8ToF32(sw.data(), ss.data(), sn, sk, actual.data());

      REQUIRE(actual.size() == reference.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != reference[i]) {
          CAPTURE(i);
          CAPTURE(actual[i]);
          CAPTURE(reference[i]);
          FAIL("bit-exact mismatch (upstream asserts rtol=0, atol=0)");
        }
      }
    }
  }
}

TEST_CASE("V4.1 MXFP8: repeat_interleave, not tile — row n reads checkpoint row n/32") {
  // modelopt.py:2196-2197. `repeat_interleave` repeats each row CONTIGUOUSLY;
  // `repeat`/`tile` would concatenate whole copies of the plane. On a 3-row
  // checkpoint scale the two disagree everywhere except row 0.
  const std::vector<uint8_t> s = {10U, 20U, 30U};
  const auto e = ExpandMxfp8CheckpointScale(s.data(), 3, 1, kMxfp8BlockSize,
                                            kMxfp8BlockSize);
  REQUIRE(e.size() == 96U);
  for (int64_t n = 0; n < 96; ++n) {
    CAPTURE(n);
    CHECK(e[static_cast<size_t>(n)] == s[static_cast<size_t>(n / kMxfp8BlockSize)]);
  }
}

TEST_CASE("V4.1 MXFP8: the expansion must precede the TP shard, and the padded "
          "shared-expert row count is what proves it") {
  // A V4.1 shared-expert projection is 96 rows: its checkpoint scale has 96/32 = 3
  // rows, which does not divide by a TP world of 2 at all. Expanding first gives
  // 96 rows (padded to 128, 64 per rank); sharding the CHECKPOINT scale first
  // cannot even be spelled, and the nearest thing that can — sharding the 3 rows
  // 2/1 — puts a different scale on rank 1's rows. This is why
  // `get_scale_weight_loader` wraps the loader instead of running after it.
  std::mt19937 rng(0xD5410FU);
  const int64_t n = 96, k = 32;
  const auto s = DrawScaleBytes(rng, n / kMxfp8BlockSize, k / kMxfp8BlockSize);

  const auto expanded = ExpandMxfp8CheckpointScale(s.data(), 3, 1, kMxfp8BlockSize,
                                                   kMxfp8BlockSize);
  const auto padded = ZeroPadBytes(expanded, n, 1, 128, 1);
  const auto rank1_correct = Chunk2Bytes(padded, 128, 1, 0, 1);

  // Shard-then-expand: rank 1 takes checkpoint row 2 (of 3) and expands it.
  const std::vector<uint8_t> ckpt_rank1 = {s[2]};
  const auto rank1_wrong =
      ExpandMxfp8CheckpointScale(ckpt_rank1.data(), 1, 1, kMxfp8BlockSize,
                                 kMxfp8BlockSize);

  REQUIRE(rank1_correct.size() == 64U);
  // The correct rank-1 plane is 32 rows of checkpoint row 2 followed by 32 ZERO
  // pad rows. The shard-then-expand plane is 32 rows and has no pad at all, so
  // the two do not even agree on the row COUNT, let alone the values.
  CHECK(rank1_wrong.size() == 32U);
  for (size_t i = 0; i < 32U; ++i) CHECK(rank1_correct[i] == s[2]);
  for (size_t i = 32U; i < 64U; ++i) CHECK(rank1_correct[i] == 0U);
}

TEST_CASE("V4.1 MXFP8: a checkpoint scale block that is not 32 columns is refused") {
  // modelopt.py:2189-2192 raises NotImplementedError when
  // `block_cols != MXFP8_BLOCK_SIZE` or `block_rows < 1`.
  const std::vector<uint8_t> s = {1U, 2U, 3U, 4U};
  CHECK_THROWS(ExpandMxfp8CheckpointScale(s.data(), 2, 2, 32, 128));
  CHECK_THROWS(ExpandMxfp8CheckpointScale(s.data(), 2, 2, 32, 16));
  CHECK_THROWS(ExpandMxfp8CheckpointScale(s.data(), 2, 2, 0, 32));
  CHECK_THROWS(ExpandMxfp8CheckpointScale(s.data(), 2, 2, -1, 32));
  // block_rows == 1 is legal and is the identity (modelopt.py:2201 returns the
  // unwrapped loader in that case).
  const auto e = ExpandMxfp8CheckpointScale(s.data(), 2, 2, 1, 32);
  CHECK(e == s);
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) THE RUNTIME DEQUANT — against a first-principles double reference
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("V4.1 MXFP8: the runtime dequant indexes the scale by ROW n, not n/32") {
  // The whole point of the expansion. Two adjacent output rows whose expanded
  // scale bytes differ must dequant differently; an implementation that kept the
  // checkpoint's `n / 32` indexing would give them the SAME scale.
  const int64_t n = 2, k = 32;
  std::vector<uint8_t> w(static_cast<size_t>(n * k), vllm::F32ToF8E4M3(1.0F));
  const std::vector<uint8_t> s = {127U, 130U};  // 2^0 and 2^3
  std::vector<float> out(static_cast<size_t>(n * k));
  DequantMxfp8ToF32(w.data(), s.data(), n, k, out.data());
  for (int64_t j = 0; j < k; ++j) {
    CHECK(out[j] == doctest::Approx(1.0F));
    CHECK(out[k + j] == doctest::Approx(8.0F));
  }
}

TEST_CASE("V4.1 MXFP8: runtime dequant matches a double-precision recompute, and the "
          "bf16 emitter is the same math narrowed once") {
  std::mt19937 rng(0xD54102U);
  const int64_t n = 37, k = 96;  // ragged rows, 3 scale columns
  const auto w = DrawWeightBytes(rng, n, k);
  const auto s = DrawScaleBytes(rng, n, k / kMxfp8BlockSize);  // already expanded

  std::vector<float> got(static_cast<size_t>(n * k));
  DequantMxfp8ToF32(w.data(), s.data(), n, k, got.data());
  const auto want = RuntimeDequantD(w, s, n, k);
  // Every operand is an exact power of two times a small integer, so f32 carries
  // the product exactly: this is an equality, not a tolerance.
  for (size_t i = 0; i < got.size(); ++i)
    CHECK(static_cast<double>(got[i]) == want[i]);

  std::vector<uint16_t> gotb(static_cast<size_t>(n * k));
  DequantMxfp8ToBf16(w.data(), s.data(), n, k, gotb.data());
  for (size_t i = 0; i < gotb.size(); ++i)
    CHECK(vt::BF16ToF32(gotb[i]) == vt::BF16ToF32(vt::F32ToBF16(got[i])));
}

TEST_CASE("V4.1 MXFP8: scale byte 0 decodes ARITHMETICALLY to 2^-127, not to the "
          "bitcast decode's zero") {
  // THE DECODE CHOICE, held here and nowhere else. Two E8M0 decodes exist in
  // this tree's reach and they agree on every byte except 0 and 0xFF:
  //   arithmetic  `exp2(byte - 127)`  — `vllm::E8M0ToF32` (mxfp4_dequant.cpp:15-22),
  //                                     which this family uses, and which
  //                                     mirrors `torch.exp2(scales - 127)`
  //                                     (mxfp8_utils.py:237);
  //   bitcast     `(byte << 23)` read as f32 — upstream's Engram kernel
  //                                     (`deepseek_v4_1/common/engram.py:613-614`),
  //                                     which W3a owns.
  // Byte 0 separates them: arithmetic gives 2^-127, bitcast gives +0.0. The
  // suite's drawn scale bytes are `randint(124, 131)` (upstream's range,
  // `DrawScaleBytes` below), so without this case the only byte that separated
  // the two decodes anywhere in the file was 0xFF — and 0xFF is itself a
  // recorded, revisitable divergence. Reconciling 0xFF to upstream's `+inf`
  // would then leave NOTHING holding the decode on this arm.
  CHECK(vllm::E8M0ToF32(0U) == std::ldexp(1.0F, -127));
  CHECK(vllm::E8M0ToF32(0U) != 0.0F);
  // And through the family's own entry point, on a NONZERO weight, because a
  // zero weight makes both decodes agree at 0.
  const std::vector<uint8_t> w(kMxfp8BlockSize, vllm::F32ToF8E4M3(1.0F));
  const std::vector<uint8_t> s = {0U};
  std::vector<float> out(kMxfp8BlockSize);
  DequantMxfp8ToF32(w.data(), s.data(), 1, kMxfp8BlockSize, out.data());
  for (float v : out) {
    CHECK(v == std::ldexp(1.0F, -127));
    CHECK(v != 0.0F);
  }
}

TEST_CASE("V4.1 MXFP8: scale byte 0xFF is the OCP NaN encoding, a NAMED divergence "
          "from upstream's exp2(255-127) == +inf") {
  // The quantizer clamps sb to 254 so it cannot PRODUCE 0xFF; only a hand-written
  // checkpoint could carry one. We keep `E8M0ToF32`'s OCP reading rather than
  // fork a second E8M0 decode. Pinned so the divergence is visible, not latent.
  CHECK(std::isnan(vllm::E8M0ToF32(0xFFU)));
  const std::vector<uint8_t> w(kMxfp8BlockSize, vllm::F32ToF8E4M3(1.0F));
  const std::vector<uint8_t> s = {0xFFU};
  std::vector<float> out(kMxfp8BlockSize);
  DequantMxfp8ToF32(w.data(), s.data(), 1, kMxfp8BlockSize, out.data());
  for (float v : out) CHECK(std::isnan(v));
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) THE DYNAMIC QUANTIZER — derived, in double precision
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("V4.1 MXFP8 dynamic quant: the exponent byte is ceil(log2(amax/448))+127, "
          "clamped to [0, 254]") {
  // mxfp8_utils.py:127-128. Hand-derived literals: with amax == 448 exactly,
  // log2(1) == 0, so sb == 127 (scale 2^0). With amax == 896, sb == 128.
  const int64_t k = kMxfp8BlockSize;
  auto sb_of = [&](float amax) {
    std::vector<float> x(static_cast<size_t>(k), 0.0F);
    x[0] = amax;
    std::vector<uint8_t> q(static_cast<size_t>(k));
    std::vector<uint8_t> s(1);
    QuantizeMxfp8E4m3(x.data(), 1, k, q.data(), s.data());
    return static_cast<int>(s[0]);
  };
  CHECK(sb_of(448.0F) == 127);
  CHECK(sb_of(896.0F) == 128);
  CHECK(sb_of(224.0F) == 126);
  // CEIL, not floor, and these are the only literals here that can tell them
  // apart. Every literal above is 448 * 2^k, where `log2(amax / 448)` is an
  // INTEGER and `ceil == floor`; the case is silent about the rounding it
  // names. 300 / 448 == 0.6696, log2 == -0.5787: ceil gives 0 and sb == 127,
  // floor would give -1 and sb == 126. 500 / 448 == 1.1161, log2 == 0.1583:
  // ceil gives 1 and sb == 128, floor would give 0 and sb == 127. The
  // polarity matters — ceil is what keeps the scaled amax at or below 448
  // rather than one binade above it, which the round-trip case below checks.
  CHECK(sb_of(300.0F) == 127);
  CHECK(sb_of(500.0F) == 128);
  // Clamp LOW: a tiny amax would drive sb far negative; 0 is the floor.
  CHECK(sb_of(std::numeric_limits<float>::min()) == 0);
  CHECK(sb_of(0.0F) == 0);
  // Clamp HIGH: 254 is the ceiling, and it is NOT reachable from any finite f32.
  // The largest finite float gives ceil(log2(FLT_MAX / 448)) + 127 == 247, so
  // every finite activation lands at or below 247. Recorded because a reader
  // would otherwise assume the upper clamp bounds ordinary data; it does not.
  CHECK(sb_of(std::numeric_limits<float>::max()) == 247);
  // What DOES reach it is a non-finite amax, and there the clamp is
  // load-bearing rather than decorative: `ceil(log2(inf)) + 127` is `inf`, and
  // narrowing `inf` to an int without the clamp is undefined behaviour.
  CHECK(sb_of(std::numeric_limits<float>::infinity()) == 254);
  CHECK(sb_of(-std::numeric_limits<float>::infinity()) == 254);
}

TEST_CASE("V4.1 MXFP8 dynamic quant: an all-zero block stays ZERO and FINITE") {
  // mxfp8_utils.py:129-135 states the hazard in upstream's own words: sb == 0
  // makes the DIVISOR 2**-127, subnormal in fp32, which flushes to zero on CDNA
  // and turns the block's zeros into 0/0 == NaN. The reciprocal 2^(127-0) == 2^127
  // is a normal float, which is why the port multiplies.
  //
  // **WHAT THIS CASE DOES NOT GATE, measured rather than assumed.** Replacing
  // the multiply with a true divide by `exp2(sb - 127)` leaves this suite GREEN
  // with the binary proved changed. See the `## Mutations` table in
  // `ISSUE-LOCAL-01M2C2QSFZWCXNQBJFBWYFNV2P`, which carries every digest beside
  // the build recipe that produced it; digests are not repeated here, because a
  // digest without its recipe is not reproducible and a digest stored in two
  // files drifts. The two forms are exact powers of two and x86 has no
  // flush-to-zero here, so they are bit-identical on every host WITHOUT
  // flush-to-zero, which is every host this gate runs on; upstream's own comment
  // says the weaker "nothing else changes". The multiply is therefore a
  // SOURCE-level decision whose difference only a CDNA device arm can measure,
  // and W5 owes that measurement. This case gates what it can: sb == 0 and a
  // finite zero result, which the LOW CLAMP produces and which the low-clamp
  // mutation does kill.
  const int64_t k = kMxfp8BlockSize;
  const std::vector<float> x(static_cast<size_t>(k), 0.0F);
  std::vector<uint8_t> q(static_cast<size_t>(k));
  std::vector<uint8_t> s(1);
  QuantizeMxfp8E4m3(x.data(), 1, k, q.data(), s.data());
  CHECK(static_cast<int>(s[0]) == 0);
  std::vector<float> back(static_cast<size_t>(k));
  DequantMxfp8ToF32(q.data(), s.data(), 1, k, back.data());
  for (float v : back) {
    CHECK(std::isfinite(v));
    CHECK(v == 0.0F);
  }
}

TEST_CASE("V4.1 MXFP8 dynamic quant: the scale puts the block amax at the TOP of the "
          "e4m3 range, so the block round-trips inside one e4m3 step") {
  // mxfp8_utils.py:123-125: "the scale has to put the block amax at the top of
  // the e4m3 range rather than at 1.0, or small elements of the block end up in
  // the subnormals". Derived: after scaling, |amax| lands in (224, 448], and e4m3
  // has 3 mantissa bits, so the worst-case relative error of the round is
  // 2^-4 = 0.0625.
  std::mt19937 rng(0xD54103U);
  std::uniform_real_distribution<float> d(-3.0F, 3.0F);
  const int64_t m = 5, k = 64;
  std::vector<float> x(static_cast<size_t>(m * k));
  for (auto& v : x) v = d(rng);

  std::vector<uint8_t> q(static_cast<size_t>(m * k));
  std::vector<uint8_t> s(static_cast<size_t>(m * k / kMxfp8BlockSize));
  QuantizeMxfp8E4m3(x.data(), m, k, q.data(), s.data());

  std::vector<float> back(static_cast<size_t>(m * k));
  DequantMxfp8ToF32(q.data(), s.data(), m, k, back.data());

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t b = 0; b < k / kMxfp8BlockSize; ++b) {
      double amax = 0.0;
      for (int64_t j = 0; j < kMxfp8BlockSize; ++j)
        amax = std::max(amax, std::fabs(static_cast<double>(
                                  x[i * k + b * kMxfp8BlockSize + j])));
      // The scale byte, recomputed from the formula in double.
      const int sb = std::max(
          0, std::min(254, static_cast<int>(std::ceil(std::log2(
                               amax / static_cast<double>(kMxfp8Fp8Max)))) +
                               127));
      CHECK(static_cast<int>(s[i * (k / kMxfp8BlockSize) + b]) == sb);
      // Post-scale amax in (224, 448]: the top e4m3 binade, not 1.0.
      const double scaled = amax * std::exp2(127.0 - sb);
      CHECK(scaled <= 448.0);
      CHECK(scaled > 224.0);
      for (int64_t j = 0; j < kMxfp8BlockSize; ++j) {
        const size_t idx = static_cast<size_t>(i * k + b * kMxfp8BlockSize + j);
        CHECK(std::fabs(static_cast<double>(back[idx]) -
                        static_cast<double>(x[idx])) <= 0.0625 * amax);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) THE EMULATION LINEAR ARM — derived, in double precision
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("V4.1 MXFP8 emulation linear: matches a double-precision recompute that "
          "narrows the weight through BF16 first") {
  // emulation.py:45-48 replaces the 1-byte MXFP8 weight with the BF16 dequant at
  // load, then :59-63 runs a plain `F.linear`. The bf16 narrowing is part of the
  // arm's numerics: it is the dtype the emulated weight actually has.
  std::mt19937 rng(0xD54104U);
  std::uniform_real_distribution<float> dx(-1.0F, 1.0F);
  const int64_t m = 3, k = 64, n = 5;
  const auto w = DrawWeightBytes(rng, n, k);
  const auto s = DrawScaleBytes(rng, n, k / kMxfp8BlockSize);
  std::vector<float> x(static_cast<size_t>(m * k));
  for (auto& v : x) v = dx(rng);
  std::vector<float> bias(static_cast<size_t>(n));
  for (auto& v : bias) v = dx(rng);

  std::vector<float> got(static_cast<size_t>(m * n));
  Mxfp8LinearEmulation(x.data(), m, k, w.data(), s.data(), n, bias.data(), got.data());

  const auto wd = RuntimeDequantD(w, s, n, k);
  std::vector<double> want(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t p = 0; p < k; ++p)
        acc += static_cast<double>(x[i * k + p]) *
               // BF16 round of the dequant, recomputed independently.
               static_cast<double>(vt::BF16ToF32(
                   vt::F32ToBF16(static_cast<float>(wd[p + j * k]))));
      want[i * n + j] = acc + static_cast<double>(bias[j]);
    }
  CHECK(RelL2(got, want) < 1e-6);

  // No bias is the `bias=None` arm (emulation.py:54).
  std::vector<float> nob(static_cast<size_t>(m * n));
  Mxfp8LinearEmulation(x.data(), m, k, w.data(), s.data(), n, nullptr, nob.data());
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j)
      CHECK(nob[i * n + j] == doctest::Approx(got[i * n + j] - bias[j]));
}

TEST_CASE("V4.1 MXFP8: the bf16 narrowing is a NO-OP on all but the bottom of the "
          "range, and the one value that separates it is 2^-136") {
  // A dtype that is too WIDE is invisible to a token gate (AGENTS.md), so the
  // narrowing needs an assertion that can actually SEE it. Most candidate values
  // cannot: an e4m3 datum is an integer in [1, 15] times a power of two (3
  // explicit mantissa bits, or 3 subnormal bits), the scale is a power of two,
  // so EVERY dequant product carries at most 4 significant bits where bf16 holds
  // 8. Over the whole product space `bf16(v) == v`, and the earlier version of
  // this case — `1.875 * 2^-6` — was therefore vacuous: 1.875 is `1.111b`, which
  // bf16 holds exactly.
  //
  // What DOES separate f32 from bf16 is the EXPONENT floor, not the mantissa.
  // f32's smallest subnormal is 2^-149; bf16's is 2^-133, because its 8-bit
  // significand eats 7 of the 8 exponent steps below 2^-126. So the smallest
  // product this format can build — the smallest e4m3 subnormal `0x01` == 2^-9
  // times scale byte 0 == 2^-127 — is 2^-136: exact in f32, and ZERO in bf16.
  const float kTiniest = std::ldexp(1.0F, -136);
  REQUIRE(vllm::F8E4M3ToF32(0x01U) == std::ldexp(1.0F, -9));
  REQUIRE(vllm::E8M0ToF32(0U) == std::ldexp(1.0F, -127));
  REQUIRE(kTiniest != 0.0F);                            // f32 holds it
  REQUIRE(vt::BF16ToF32(vt::F32ToBF16(kTiniest)) == 0.0F);  // bf16 does not

  const int64_t k = kMxfp8BlockSize;
  std::vector<uint8_t> w(static_cast<size_t>(k), 0U);
  w[0] = 0x01U;                        // e4m3 smallest subnormal, 2^-9
  const std::vector<uint8_t> s = {0U};  // 2^-127

  // (a) The f32 emitter keeps it and the bf16 emitter loses it. This is the
  // MEMORY-FORMAT assertion: delete the `F32ToBF16` in `DequantMxfp8ToBf16` and
  // this reds.
  std::vector<float> df(static_cast<size_t>(k));
  DequantMxfp8ToF32(w.data(), s.data(), 1, k, df.data());
  CHECK(df[0] == kTiniest);
  std::vector<uint16_t> db(static_cast<size_t>(k));
  DequantMxfp8ToBf16(w.data(), s.data(), 1, k, db.data());
  CHECK(vt::BF16ToF32(db[0]) == 0.0F);

  // (b) The emulation arm inherits it, because it dequantizes the weight to
  // BF16 once (emulation.py:45-46) rather than holding an f32 weight. Delete
  // that narrowing and the product is 2^-136 instead of 0.
  std::vector<float> x(static_cast<size_t>(k), 0.0F);
  x[0] = 1.0F;
  std::vector<float> out(1);
  Mxfp8LinearEmulation(x.data(), 1, k, w.data(), s.data(), 1, nullptr, out.data());
  CHECK(out[0] == 0.0F);

  // (c) The narrowing ROUNDS to nearest even, it does not truncate, and the
  // bf16 subnormal range is again the only place that can tell the two apart.
  // e4m3 `0x03` is the subnormal 3 * 2^-9; scale byte 2 is 2^-125; the product
  // is 3 * 2^-134 == 1.5 bf16 subnormal steps, a dead tie. RNE takes the even
  // neighbour 2^-132; a truncating high-half store takes 2^-133. Without this,
  // a "narrowing" that merely drops the low 16 bits passes (a) and (b), because
  // 2^-136 truncates to zero as well.
  std::vector<uint8_t> w3(static_cast<size_t>(k), 0U);
  w3[0] = 0x03U;
  const std::vector<uint8_t> s2 = {2U};
  REQUIRE(vllm::F8E4M3ToF32(0x03U) == 3.0F * std::ldexp(1.0F, -9));
  std::vector<uint16_t> tie(static_cast<size_t>(k));
  DequantMxfp8ToBf16(w3.data(), s2.data(), 1, k, tie.data());
  CHECK(vt::BF16ToF32(tie[0]) == std::ldexp(1.0F, -132));
  CHECK(vt::BF16ToF32(tie[0]) != std::ldexp(1.0F, -133));
  Mxfp8LinearEmulation(x.data(), 1, k, w3.data(), s2.data(), 1, nullptr, out.data());
  CHECK(out[0] == std::ldexp(1.0F, -132));

  // (d) And the ordinary case still round-trips, so (a) through (c) are reading
  // a floor and not a broken emitter: 1.875 at 2^-6 survives both paths intact.
  w[0] = vllm::F32ToF8E4M3(1.875F);  // e4m3 exact: 1.111b
  const std::vector<uint8_t> s6 = {127U - 6U};
  Mxfp8LinearEmulation(x.data(), 1, k, w.data(), s6.data(), 1, nullptr, out.data());
  CHECK(out[0] == 1.875F * std::ldexp(1.0F, -6));
}

TEST_CASE("V4.1 MXFP8 emulation linear: the BF16 arm is the model-path arm, and it "
          "matches the f32 reference narrowed at the store") {
  // emulation.py:59-63 runs `F.linear(x, weight.to(x.dtype), bias).to(x.dtype)`
  // with `x` the model dtype, which V4.1 resolves to bf16
  // (test_fp8.py:83 sets `dtype=torch.bfloat16`). So the upstream arm is bf16
  // in and bf16 out; `Mxfp8LinearEmulationBf16` is that arm, and the f32 entry
  // point beside it is a REFERENCE, not a model-path buffer.
  std::mt19937 rng(0xD54105U);
  std::uniform_real_distribution<float> dx(-1.0F, 1.0F);
  const int64_t m = 3, k = 64, n = 5;
  const auto w = DrawWeightBytes(rng, n, k);
  const auto s = DrawScaleBytes(rng, n, k / kMxfp8BlockSize);

  // Draw the activation and the bias AS BF16, so the two arms get identical
  // inputs and the only difference between them is the output store.
  std::vector<uint16_t> xb(static_cast<size_t>(m * k));
  std::vector<float> xf(static_cast<size_t>(m * k));
  for (size_t i = 0; i < xb.size(); ++i) {
    xb[i] = vt::F32ToBF16(dx(rng));
    xf[i] = vt::BF16ToF32(xb[i]);
  }
  std::vector<uint16_t> bb(static_cast<size_t>(n));
  std::vector<float> bf(static_cast<size_t>(n));
  for (size_t i = 0; i < bb.size(); ++i) {
    bb[i] = vt::F32ToBF16(dx(rng));
    bf[i] = vt::BF16ToF32(bb[i]);
  }

  std::vector<float> ref(static_cast<size_t>(m * n));
  Mxfp8LinearEmulation(xf.data(), m, k, w.data(), s.data(), n, bf.data(), ref.data());
  std::vector<uint16_t> got(static_cast<size_t>(m * n));
  Mxfp8LinearEmulationBf16(xb.data(), m, k, w.data(), s.data(), n, bb.data(),
                           got.data());
  // f32 accumulation in both, narrowed ONCE at the store: this is an equality,
  // and it reds if the bf16 arm ever stores something wider or rounds twice.
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == vt::F32ToBF16(ref[i]));

  // The `bias=None` arm (emulation.py:54), and the polarity that says the store
  // really is bf16: a result bf16 cannot hold comes back rounded.
  std::vector<uint16_t> nob(static_cast<size_t>(m * n));
  Mxfp8LinearEmulationBf16(xb.data(), m, k, w.data(), s.data(), n, nullptr,
                           nob.data());
  std::vector<float> noref(static_cast<size_t>(m * n));
  Mxfp8LinearEmulation(xf.data(), m, k, w.data(), s.data(), n, nullptr, noref.data());
  bool any_rounded = false;
  for (size_t i = 0; i < nob.size(); ++i) {
    CHECK(nob[i] == vt::F32ToBF16(noref[i]));
    if (vt::BF16ToF32(nob[i]) != noref[i]) any_rounded = true;
  }
  // Not decoration: without at least one value the bf16 store actually changes,
  // the equality above would hold for an f32 passthrough too.
  CHECK(any_rounded);
}

TEST_CASE("V4.1 MXFP8 emulation linear: a non-positive K is REFUSED before the "
          "weight buffer is sized") {
  // The arm sizes `std::vector<uint16_t>(N * K)` before anything validates K.
  // For K == 0 the refusal already arrived one level down, from the dequant's
  // own `N > 0 && K > 0`; for K < 0 it did NOT, because `static_cast<size_t>(N *
  // K)` on a negative product asks for ~2^64 elements and the allocator throws
  // `std::length_error` — a `std::logic_error`, not the `std::runtime_error`
  // every VT_CHECK in this family raises. Pinning the EXCEPTION TYPE is what
  // makes this case able to tell a refusal from an accident.
  const std::vector<uint8_t> w(32U, 0U), s(1U, 127U);
  std::vector<float> x(32U, 0.0F), out(1U);
  CHECK_THROWS_AS(
      Mxfp8LinearEmulation(x.data(), 1, 0, w.data(), s.data(), 1, nullptr, out.data()),
      std::runtime_error);
  CHECK_THROWS_AS(
      Mxfp8LinearEmulation(x.data(), 1, -32, w.data(), s.data(), 1, nullptr, out.data()),
      std::runtime_error);
  CHECK_THROWS_AS(Mxfp8LinearEmulationBf16(nullptr, 1, -32, w.data(), s.data(), 1,
                                           nullptr, nullptr),
                  std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) THE CHECKPOINT NAME SPLIT — test_fp8.py:202-238
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("V4.1 MXFP8: the linear scale parameter name is weight_scale only for "
          "[32,32] blocks WITH fp4 experts") {
  // test_fp8.py:204-210, the three parametrizations verbatim.
  CHECK(Mxfp8LinearScaleParamName({32, 32}, "fp4") == "weight_scale");
  CHECK(Mxfp8LinearScaleParamName({128, 128}, "fp4") == "weight_scale_inv");
  CHECK(Mxfp8LinearScaleParamName({32, 32}, "fp8") == "weight_scale_inv");
  // Both halves of the conjunction are load-bearing (nvidia/model.py:878-882):
  // neither an absent block size nor a near-miss block size is native MXFP8.
  CHECK(Mxfp8LinearScaleParamName({}, "fp4") == "weight_scale_inv");
  CHECK(Mxfp8LinearScaleParamName({32, 128}, "fp4") == "weight_scale_inv");
  CHECK(Mxfp8LinearScaleParamName({128, 128}, "fp8") == "weight_scale_inv");
}
