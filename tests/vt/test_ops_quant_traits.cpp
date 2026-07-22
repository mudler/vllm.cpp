// QUANT-GGUF-CIQ-GEMM work row G1 — block dtypes, the CPU quant traits table,
// and the `kMatmulBTQuant` skeleton's generic-composite fallback.
//
// The load-bearing case in this file is the TRAIT CROSS-CHECK: vt's block
// geometry table (src/vt/dtype.cpp) and the GGUF reader's `GgmlTraits`
// (src/vllm/model_executor/model_loader/gguf_reader.cpp) are two INDEPENDENT
// ports of the same llama.cpp facts (@ 237ad9b96, ggml/include/ggml.h:390-432
// + ggml/src/ggml-common.h block structs). A silent disagreement between them
// would mis-stride every packed weight buffer downstream, so they are compared
// element-for-element here, together with the third independent statement of
// the same facts — the block struct arithmetic spelled out from ggml-common.h.
//
// Traits-table cases mirror llama.cpp `ggml/src/ggml-cpu/ggml-cpu.c:211-406`
// (`type_traits_cpu[]`): the `vec_dot_type` column is the dispatch fact G1
// owns (Q8_0 for the legacy 32-element types, Q8_K for the K-quants).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

namespace {

struct BlockCase {
  vt::DType dtype;
  uint32_t ggml_type;
  int64_t block_elems;
  int64_t block_bytes;
  vt::DType vec_dot_type;
  const char* name;
};

// Expected values written out from the ggml-common.h struct definitions, NOT
// copied from either table under test:
//   q4_0  :213-218  f16 d + 32/2 qs                        = 2 + 16  = 18
//   q8_0  :242-245  f16 d + 32 qs                          = 2 + 32  = 34
//   q3_K  :305-310  256/8 hmask + 256/4 qs + 12 sc + f16 d = 32+64+12+2  = 110
//   q4_K  :317-327  2*f16 dm + 12 sc + 256/2 qs            = 4+12+128    = 144
//   q5_K  :334-345  2*f16 dm + 12 sc + 256/8 qh + 256/2 qs = 4+12+32+128 = 176
//   q6_K  :352-357  256/2 ql + 256/4 qh + 256/16 sc + f16 d= 128+64+16+2 = 210
//   q8_K  :361-365  f32 d + 256 qs + 256/16 i16 bsums      = 4+256+32    = 292
const BlockCase kBlockCases[] = {
    {vt::DType::kQ4_0, 2, 32, 2 + 16, vt::DType::kQ8_0, "q4_0"},
    {vt::DType::kQ8_0, 8, 32, 2 + 32, vt::DType::kQ8_0, "q8_0"},
    {vt::DType::kQ3_K, 11, 256, 32 + 64 + 12 + 2, vt::DType::kQ8_K, "q3_K"},
    {vt::DType::kQ4_K, 12, 256, 4 + 12 + 128, vt::DType::kQ8_K, "q4_K"},
    {vt::DType::kQ5_K, 13, 256, 4 + 12 + 32 + 128, vt::DType::kQ8_K, "q5_K"},
    {vt::DType::kQ6_K, 14, 256, 128 + 64 + 16 + 2, vt::DType::kQ8_K, "q6_K"},
    {vt::DType::kQ8_K, 15, 256, 4 + 256 + 32, vt::DType::kQ8_K, "q8_K"},
};

// Deterministic pseudo-random block bytes. Any bit pattern is a legal block for
// every one of these encodings (all fields are unconstrained integers/halfs),
// except that f16 scale fields must avoid inf/nan — bit patterns with all
// exponent bits set. Those are masked out below so the composite comparison is
// finite.
std::vector<uint8_t> RandomBlocks(const BlockCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  // Overwrite every f16/f32 scale field with a small, exactly-representable
  // value so no block carries an inf/nan delta.
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    switch (c.dtype) {
      case vt::DType::kQ4_0:
      case vt::DType::kQ8_0:
        put_f16(0, 0.0125F);
        break;
      case vt::DType::kQ3_K:
        put_f16(108, 0.0125F);
        break;
      case vt::DType::kQ4_K:
      case vt::DType::kQ5_K:
        put_f16(0, 0.0125F);
        put_f16(2, 0.0075F);
        break;
      case vt::DType::kQ6_K:
        put_f16(208, 0.0125F);
        break;
      case vt::DType::kQ8_K: {
        const float d = 0.0125F;
        std::memcpy(blk, &d, sizeof(d));
        break;
      }
      default:
        break;
    }
  }
  return bytes;
}

}  // namespace

// --- The cross-check ---------------------------------------------------------

TEST_CASE("vt block geometry agrees with the GGUF reader's GgmlTraits") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    // vt's own table vs the values written out from ggml-common.h.
    CHECK(vt::IsBlockQuant(c.dtype));
    CHECK(vt::BlockElems(c.dtype) == c.block_elems);
    CHECK(vt::BlockBytes(c.dtype) == c.block_bytes);
    CHECK(vt::GgmlTypeId(c.dtype) == c.ggml_type);
    CHECK(std::string(vt::Name(c.dtype)) == c.name);

    // Q8_K (id 15) is activation-only: it never appears in a GGUF file, so the
    // reader's table deliberately does not carry it. Every FILE type must.
    if (c.dtype == vt::DType::kQ8_K) {
      CHECK_THROWS(vllm::GgmlTraits(c.ggml_type));
      continue;
    }
    const vllm::GgmlTypeTraits& g = vllm::GgmlTraits(c.ggml_type);
    CHECK(g.block_elems == vt::BlockElems(c.dtype));
    CHECK(g.block_bytes == vt::BlockBytes(c.dtype));

    // ...and the id round-trips back to the same vt dtype.
    vt::DType back = vt::DType::kF32;
    REQUIRE(vt::BlockDTypeFromGgmlTypeId(c.ggml_type, &back));
    CHECK(back == c.dtype);
  }
}

TEST_CASE("elementwise dtypes are not block-quantized and reject block queries") {
  for (vt::DType d : {vt::DType::kF32, vt::DType::kF16, vt::DType::kBF16,
                      vt::DType::kI8, vt::DType::kI32, vt::DType::kI64}) {
    CHECK_FALSE(vt::IsBlockQuant(d));
    CHECK_THROWS(vt::BlockElems(d));
    CHECK_THROWS(vt::BlockBytes(d));
  }
  // The GGUF reader knows F32/F16/BF16 but they are not BLOCK dtypes, so the
  // id->block-dtype map must reject them rather than aliasing onto one.
  for (uint32_t id : {0U, 1U, 30U}) {
    vt::DType out = vt::DType::kF32;
    CHECK_FALSE(vt::BlockDTypeFromGgmlTypeId(id, &out));
  }
}

TEST_CASE("block dtypes are storage-only: SizeOf throws") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    CHECK_THROWS(vt::SizeOf(c.dtype));
  }
}

TEST_CASE("RowSizeBytes mirrors ggml_row_size") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    const int64_t be = c.block_elems;
    CHECK(vt::RowSizeBytes(c.dtype, be) == static_cast<size_t>(c.block_bytes));
    CHECK(vt::RowSizeBytes(c.dtype, 8 * be) ==
          static_cast<size_t>(8 * c.block_bytes));
    // ggml asserts rows are whole blocks.
    CHECK_THROWS(vt::RowSizeBytes(c.dtype, be + 1));
  }
  // Elementwise dtypes stay k * SizeOf so callers can use one helper.
  CHECK(vt::RowSizeBytes(vt::DType::kF32, 7) == 28);
  CHECK(vt::RowSizeBytes(vt::DType::kBF16, 7) == 14);
}

// --- The traits table --------------------------------------------------------

TEST_CASE("quant traits mirror type_traits_cpu (ggml-cpu.c:211-406)") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    const vt::cpu::QuantTypeTraits& t = vt::cpu::QuantTraits(c.dtype);
    CHECK(t.vec_dot_type == c.vec_dot_type);
    // The generic tier pins nrows == 1; nrows == 2 arrives only with the i8mm
    // mmla kernels (G6) and their odd-shape boundary guards.
    CHECK(t.nrows == 1);
    // Every executable block type must decode.
    CHECK(t.to_float != nullptr);
    // G1 is the skeleton: no quantized dot exists yet, so every type must route
    // to the generic composite. This CHECK is what G2/G3 flip.
    CHECK(t.vec_dot == nullptr);
    CHECK(t.from_float == nullptr);
    CHECK_FALSE(vt::cpu::HasQuantDotKernel(c.dtype));
  }
  CHECK_THROWS(vt::cpu::QuantTraits(vt::DType::kF32));
}

TEST_CASE("BlockToFloat matches the GGUF loader's dequant byte-for-byte") {
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    if (c.dtype == vt::DType::kQ8_K) continue;  // not a file type
    constexpr int64_t kBlocks = 5;
    const std::vector<uint8_t> bytes = RandomBlocks(c, kBlocks, 1234U);
    const int64_t numel = kBlocks * c.block_elems;

    std::vector<float> direct(static_cast<size_t>(numel));
    vt::cpu::BlockToFloat(c.dtype)(bytes.data(), direct.data(), numel);

    const std::vector<float> loader =
        vllm::DequantGgufRowToF32(c.ggml_type, bytes.data(), numel);
    REQUIRE(loader.size() == direct.size());
    for (size_t i = 0; i < direct.size(); ++i) {
      // Byte-identical, not approximate: they are the same implementation.
      CHECK(loader[i] == direct[i]);
    }
  }
}

// --- The op skeleton ---------------------------------------------------------

namespace {

// Reference: dequantize the whole weight to f32, then the same f32 dot the
// composite performs. This is the `DequantGgufRowToF32 + MatmulBTKernel`
// oracle the spec's ported MUL_MAT cases use.
std::vector<float> ReferenceMatmul(const std::vector<float>& a,
                                   const std::vector<float>& w, int64_t m,
                                   int64_t k, int64_t n) {
  std::vector<float> out(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0F;
      for (int64_t p = 0; p < k; ++p) {
        acc += a[static_cast<size_t>(i * k + p)] *
               w[static_cast<size_t>(j * k + p)];
      }
      out[static_cast<size_t>(i * n + j)] = acc;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("MatmulBTQuant generic-composite fallback == dequant-then-matmul") {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  for (const BlockCase& c : kBlockCases) {
    CAPTURE(c.name);
    if (c.dtype == vt::DType::kQ8_K) continue;  // activation-only

    // N weight rows of K elements each; M activation rows. K spans several
    // blocks so the per-row block walk is exercised.
    const int64_t k = 4 * c.block_elems;
    const int64_t n = 3;
    for (int64_t m : {int64_t{1}, int64_t{4}}) {
      CAPTURE(m);
      const std::vector<uint8_t> wq =
          RandomBlocks(c, n * (k / c.block_elems), 99U);

      std::vector<float> a(static_cast<size_t>(m * k));
      std::mt19937 rng(7U);
      std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
      for (float& v : a) v = dist(rng);

      // The reference decodes the SAME bytes through the loader path.
      const std::vector<float> w =
          vllm::DequantGgufRowToF32(c.ggml_type, wq.data(), n * k);
      const std::vector<float> expected = ReferenceMatmul(a, w, m, k, n);

      vt::Tensor at = vt::Tensor::Contiguous(a.data(), vt::DType::kF32,
                                             q.device, {m, k});
      vt::Tensor bt = vt::Tensor::Contiguous(const_cast<uint8_t*>(wq.data()),
                                             vt::DType::kF32, q.device, {n, k});
      bt.dtype = c.dtype;  // block dtype: strides are not meaningful
      std::vector<float> got(static_cast<size_t>(m * n));
      vt::Tensor ot = vt::Tensor::Contiguous(got.data(), vt::DType::kF32,
                                             q.device, {m, n});

      vt::MatmulBTQuant(q, ot, at, bt);
      for (size_t i = 0; i < got.size(); ++i) {
        CHECK(got[i] == doctest::Approx(expected[i]).epsilon(1e-6));
      }
    }
  }
}

TEST_CASE("MatmulBTQuant rejects the shapes/dtypes ggml would assert on") {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t k = 256, n = 2, m = 1;
  const std::vector<uint8_t> wq =
      RandomBlocks(kBlockCases[3] /*q4_K*/, n * (k / 256), 5U);
  std::vector<float> a(static_cast<size_t>(m * k), 0.5F);
  std::vector<float> got(static_cast<size_t>(m * n));

  vt::Tensor at =
      vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, k});
  vt::Tensor ot =
      vt::Tensor::Contiguous(got.data(), vt::DType::kF32, q.device, {m, n});
  vt::Tensor bt = vt::Tensor::Contiguous(const_cast<uint8_t*>(wq.data()),
                                         vt::DType::kF32, q.device, {n, k});

  // An elementwise weight belongs on MatmulBT, not here.
  CHECK_THROWS(vt::MatmulBTQuant(q, ot, at, bt));

  bt.dtype = vt::DType::kQ4_K;
  // K that is not a whole number of blocks (ggml_row_size's assert).
  vt::Tensor a_bad = vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device,
                                            {m, k - 1});
  vt::Tensor b_bad = bt;
  b_bad.shape[1] = k - 1;
  CHECK_THROWS(vt::MatmulBTQuant(q, ot, a_bad, b_bad));

  // Inner-dim mismatch.
  vt::Tensor a_k2 =
      vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, 128});
  CHECK_THROWS(vt::MatmulBTQuant(q, ot, a_k2, bt));

  // The well-formed call still works (guards are not over-tight).
  CHECK_NOTHROW(vt::MatmulBTQuant(q, ot, at, bt));
}
