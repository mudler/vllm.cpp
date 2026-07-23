// GGUF keep-quantized loader — work rows L2 (quant residency) and L3 (routing
// policy + VT_CPU_REF oracle switch) of
// .agents/specs/gguf-keep-quant-loader.md.
//
// Upstream idea being gated (llama.cpp @ 237ad9b96): a weight KEEPS the
// ggml_type it has in the file (src/llama-model-loader.cpp:1047 create_tensor,
// :1385 load_data_for) and rows are whole blocks (ggml_row_size, ggml/src/
// ggml.c). Ported test shape follows llama.cpp tests/gguf-model-data.cpp's
// harness idea — round-trip resident blocks through the decoder and demand the
// SAME bytes the direct-from-file decode produces.
//
// THE GATE (spec gate 1) IS BYTE-IDENTITY, PROVEN PER ENCODING. Keeping a
// weight in its blocks is only safe if dequantizing the resident blocks is
// indistinguishable from today's load-time expansion — not "close", identical.
// Each of the six executable encodings gets its own TEST_CASE so a failure
// names the encoding that broke.
//
// Block bytes are generated PSEUDO-RANDOMLY rather than from an encoder: the
// residency property must hold for every bit pattern a file can contain, and
// only Q8_0/Q8_K have a `from_float` in this project anyway. The one
// constraint applied is that every f16 scale field decodes finite — all six
// block structs place their f16 fields at EVEN offsets and all six block sizes
// are even, so clearing bit 6 of every odd-indexed byte (the f16 exponent MSB)
// bounds every scale away from Inf/NaN without otherwise restricting the data.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;
using vllm::GgufLoadPolicy;
using vllm::GgufResidency;
using vllm::GgufTensorRole;
using vllm::KeepQuantDType;
using vllm::OwnGgufQuantBlocks;
using vllm::RouteGgufTensor;

namespace {

// ggml type ids (ggml/include/ggml.h:390-432).
constexpr uint32_t kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ8_0 = 8, kQ3_K = 11,
                   kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kQ8_K = 15,
                   kIQ2_S = 22, kIQ4_XS = 23, kBF16 = 30;

// Every executable weight encoding, with a K that is a whole number of blocks.
struct Encoding {
  uint32_t ggml_type;
  const char* name;
  int64_t k;
};
const Encoding kEncodings[] = {
    {kQ4_0, "Q4_0", 64},  {kQ8_0, "Q8_0", 64},  {kQ3_K, "Q3_K", 256},
    {kQ4_K, "Q4_K", 256}, {kQ5_K, "Q5_K", 256}, {kQ6_K, "Q6_K", 256},
};

// xorshift32 — deterministic across platforms and compilers, so a failure is
// reproducible from the seed alone.
uint32_t NextRand(uint32_t* s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

// `nbytes` of block payload; see the header comment for the odd-byte mask.
std::string RandomBlockBytes(size_t nbytes, uint32_t seed) {
  uint32_t s = seed | 1U;
  std::string out(nbytes, '\0');
  for (size_t i = 0; i < nbytes; ++i) {
    uint8_t b = static_cast<uint8_t>(NextRand(&s) & 0xFF);
    if ((i % 2) == 1) b &= 0xBF;  // clear the f16 exponent MSB -> finite
    out[i] = static_cast<char>(b);
  }
  return out;
}

// A GGUF holding exactly one block-quantized tensor with torch shape
// [n, k] (ggml dims are the reverse).
std::string OneTensorGguf(uint32_t ggml_type, int64_t n, int64_t k,
                          const std::string& blocks) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "test"));
  b.AddTensor("w", {static_cast<uint64_t>(k), static_cast<uint64_t>(n)},
              ggml_type, blocks);
  return b.Build();
}

size_t BlockBytesFor(uint32_t ggml_type, int64_t numel) {
  vt::DType dt = vt::DType::kF32;
  REQUIRE(KeepQuantDType(ggml_type, &dt));
  return vt::RowSizeBytes(dt, numel);
}

// A policy with keep-quant forced ON (what G4 made the default wherever the
// running device can execute kMatmulBTQuant). `expand_nk` is left OFF here so
// the pre-existing L2/L3 losslessness cases keep comparing against the
// historical Matmul-B expansion; the orientation switch has its own cases.
GgufLoadPolicy KeepQuantOn() {
  GgufLoadPolicy p;
  p.keep_quant = true;
  return p;
}

}  // namespace

// ===========================================================================
// L2 gate 1 — LOSSLESSNESS, one case per encoding.
// ===========================================================================

namespace {

void CheckLossless(const Encoding& enc, int64_t n) {
  const int64_t numel = n * enc.k;
  const size_t nbytes = BlockBytesFor(enc.ggml_type, numel);
  const std::string blocks = RandomBlockBytes(nbytes, enc.ggml_type * 7919U + 1);
  const TempFile f(OneTensorGguf(enc.ggml_type, n, enc.k, blocks));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufTensorInfo& t = g.Get("w");
  REQUIRE(t.nbytes == nbytes);

  // Residency: the raw ggml blocks, copied verbatim, [N,K] with nk = true.
  const vllm::OwnedTensor res = OwnGgufQuantBlocks(t, n, enc.k);
  vt::DType dt = vt::DType::kF32;
  REQUIRE(KeepQuantDType(enc.ggml_type, &dt));
  CHECK(res.dtype == dt);
  CHECK(res.rank == 2);
  CHECK(res.shape[0] == n);
  CHECK(res.shape[1] == enc.k);
  CHECK(res.nk == true);  // GGUF [out,in] IS the MatmulBT orientation
  REQUIRE(res.bytes.size() == nbytes);
  // Byte-for-byte the file's blocks — residency copies, it does not transform.
  CHECK(std::memcmp(res.bytes.data(), t.data, nbytes) == 0);

  // THE GATE: dequantizing the resident blocks is byte-identical to today's
  // load-time expansion straight from the mapped file, in BOTH target dtypes.
  const std::vector<float> from_file =
      vllm::DequantGgufRowToF32(enc.ggml_type, t.data, numel);
  const std::vector<float> from_resident =
      vllm::DequantGgufRowToF32(enc.ggml_type, res.bytes.data(), numel);
  REQUIRE(from_file.size() == static_cast<size_t>(numel));
  REQUIRE(from_resident.size() == from_file.size());
  CHECK(std::memcmp(from_resident.data(), from_file.data(),
                    from_file.size() * sizeof(float)) == 0);

  const std::vector<uint16_t> bf_file =
      vllm::DequantGgufRowToBf16(enc.ggml_type, t.data, numel);
  const std::vector<uint16_t> bf_resident =
      vllm::DequantGgufRowToBf16(enc.ggml_type, res.bytes.data(), numel);
  CHECK(std::memcmp(bf_resident.data(), bf_file.data(),
                    bf_file.size() * sizeof(uint16_t)) == 0);

  // Sanity that the random bytes decoded to real numbers (the odd-byte mask):
  // a losslessness gate over all-NaN data would prove much less.
  for (float v : from_file) REQUIRE(std::isfinite(v));
}

}  // namespace

TEST_CASE("keep-quant residency is lossless: Q4_0") {
  CheckLossless(kEncodings[0], /*n=*/5);
}
TEST_CASE("keep-quant residency is lossless: Q8_0") {
  CheckLossless(kEncodings[1], /*n=*/5);
}
TEST_CASE("keep-quant residency is lossless: Q3_K") {
  CheckLossless(kEncodings[2], /*n=*/3);
}
TEST_CASE("keep-quant residency is lossless: Q4_K") {
  CheckLossless(kEncodings[3], /*n=*/3);
}
TEST_CASE("keep-quant residency is lossless: Q5_K") {
  CheckLossless(kEncodings[4], /*n=*/3);
}
TEST_CASE("keep-quant residency is lossless: Q6_K") {
  CheckLossless(kEncodings[5], /*n=*/3);
}

TEST_CASE("keep-quant expert split is lossless per expert") {
  // A stacked [E, out, in] expert tensor: each expert is a whole number of
  // ROWS, hence whole blocks, so the split must be a pure byte range.
  const int64_t e_count = 4, out_dim = 3, in_dim = 256;
  const int64_t numel = e_count * out_dim * in_dim;
  const size_t nbytes = BlockBytesFor(kQ4_K, numel);
  const std::string blocks = RandomBlockBytes(nbytes, 4242);

  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "test"));
  b.AddTensor("exps", {static_cast<uint64_t>(in_dim),
                       static_cast<uint64_t>(out_dim),
                       static_cast<uint64_t>(e_count)},
              kQ4_K, blocks);
  const TempFile f(b.Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufTensorInfo& t = g.Get("exps");
  REQUIRE(t.shape.size() == 3);
  REQUIRE(t.shape[0] == e_count);

  const std::vector<float> whole =
      vllm::DequantGgufRowToF32(kQ4_K, t.data, numel);
  const int64_t per = out_dim * in_dim;
  for (int64_t e = 0; e < e_count; ++e) {
    CAPTURE(e);
    const vllm::OwnedTensor res =
        OwnGgufQuantBlocks(t, out_dim, in_dim, /*row_offset=*/e * out_dim);
    const std::vector<float> slice =
        vllm::DequantGgufRowToF32(kQ4_K, res.bytes.data(), per);
    CHECK(std::memcmp(slice.data(), whole.data() + e * per,
                      static_cast<size_t>(per) * sizeof(float)) == 0);
  }
}

TEST_CASE("keep-quant residency refuses ragged K and out-of-span slices") {
  const int64_t n = 2, k = 64;
  const size_t nbytes = BlockBytesFor(kQ8_0, n * k);
  const TempFile f(OneTensorGguf(kQ8_0, n, k, RandomBlockBytes(nbytes, 9)));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufTensorInfo& t = g.Get("w");

  // K must be a whole number of blocks (ggml_row_size's precondition).
  CHECK_THROWS(OwnGgufQuantBlocks(t, n, /*k=*/33));
  // Rows beyond the validated tensor span must never be handed out.
  CHECK_THROWS(OwnGgufQuantBlocks(t, /*n=*/3, k));
  CHECK_THROWS(OwnGgufQuantBlocks(t, n, k, /*row_offset=*/1));
  // A ragged-K weight is not merely refused at residency: the POLICY routes it
  // to expansion, so the loader never reaches the throw.
  CHECK(RouteGgufTensor(true, false, false, GgufTensorRole::kMatmulWeight, kQ8_0,
                        {n, 33}) == GgufResidency::kExpandBf16);
}

// ===========================================================================
// L3 — the routing policy table.
// ===========================================================================

TEST_CASE("KeepQuantDType covers exactly the six executable encodings") {
  vt::DType dt = vt::DType::kF32;
  for (const Encoding& e : kEncodings) {
    CAPTURE(e.name);
    CHECK(KeepQuantDType(e.ggml_type, &dt));
    CHECK(vt::cpu::HasQuantDotKernel(dt));
  }
  // Unquantized file types, the activation-only encoding, and every unported
  // encoding are NOT keep-quant capable.
  for (uint32_t id : {kF32, kF16, kBF16, kQ8_K, kIQ2_S, kIQ4_XS}) {
    CAPTURE(id);
    CHECK_FALSE(KeepQuantDType(id, &dt));
  }
}

TEST_CASE("routing table is TOTAL: every role x every encoding is explicit") {
  // The expectation is written out LONGHAND here rather than derived from the
  // implementation, so this is a real cross-check and not a tautology.
  const GgufTensorRole all_roles[] = {
      GgufTensorRole::kMatmulWeight,      GgufTensorRole::kStackedExpertWeight,
      GgufTensorRole::kTransformedWeight, GgufTensorRole::kEmbeddingTable,
      GgufTensorRole::kConvWeight,        GgufTensorRole::kVector,
  };
  const uint32_t all_types[] = {kF32,  kF16,  kBF16,   kQ4_0,   kQ8_0,  kQ3_K,
                                kQ4_K, kQ5_K, kQ6_K,   kQ8_K,   kIQ2_S, kIQ4_XS};

  int kept = 0;
  int expanded = 0;
  for (GgufTensorRole role : all_roles) {
    for (uint32_t type : all_types) {
      // Shapes: correct rank for the role, plus wrong-rank and ragged-K probes.
      const std::vector<std::vector<int64_t>> shapes = {
          {8, 256}, {2, 8, 256}, {256}, {8, 255}, {2, 8, 255}, {},
      };
      for (const std::vector<int64_t>& shape : shapes) {
        CAPTURE(vllm::Name(role));
        CAPTURE(type);
        CAPTURE(shape.size());

        // --- the independent expectation ---
        const bool block_capable =
            type == kQ4_0 || type == kQ8_0 || type == kQ3_K || type == kQ4_K ||
            type == kQ5_K || type == kQ6_K;
        const int64_t blk = (type == kQ4_0 || type == kQ8_0) ? 32 : 256;
        bool expect_keep = false;
        if (block_capable) {
          if (role == GgufTensorRole::kMatmulWeight && shape.size() == 2) {
            expect_keep = shape[1] % blk == 0;
          } else if (role == GgufTensorRole::kStackedExpertWeight &&
                     shape.size() == 3) {
            expect_keep = shape[2] % blk == 0;
          }
        }
        const GgufResidency expected = expect_keep
                                           ? GgufResidency::kKeepQuant
                                           : GgufResidency::kExpandBf16;

        CHECK(RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/false,
                              /*cpu_ref=*/false, role, type, shape) == expected);
        (expect_keep ? kept : expanded)++;

        // The master switch OFF expands everything, always.
        CHECK(RouteGgufTensor(/*keep_quant=*/false, /*keep_f16=*/false,
                              /*cpu_ref=*/false, role, type,
                              shape) == GgufResidency::kExpandBf16);
        // The VT_CPU_REF oracle wins over keep-quant, always.
        CHECK(RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/false,
                              /*cpu_ref=*/true, role, type,
                              shape) == GgufResidency::kExpandBf16);
      }
    }
  }
  // Both outcomes are actually exercised (a table that never keeps anything
  // would pass every assertion above vacuously).
  CHECK(kept == 12);          // 6 encodings x 2 keep-capable roles, right rank
  CHECK(expanded == 12 * 36 - 12);
}

TEST_CASE("tensors that are value- or layout-rewritten NEVER keep quant") {
  // These are the routes that would silently CORRUPT a model: the (w-1) norm
  // rewrite, ssm_a = log(-x), the V-head reorders, the embedding gather and the
  // conv filter. Every encoding, both keep-quant-capable ranks.
  for (const Encoding& e : kEncodings) {
    CAPTURE(e.name);
    for (GgufTensorRole role :
         {GgufTensorRole::kTransformedWeight, GgufTensorRole::kEmbeddingTable,
          GgufTensorRole::kConvWeight, GgufTensorRole::kVector}) {
      CAPTURE(vllm::Name(role));
      CHECK(RouteGgufTensor(true, false, false, role, e.ggml_type, {8, e.k}) ==
            GgufResidency::kExpandBf16);
      CHECK(RouteGgufTensor(true, false, false, role, e.ggml_type,
                            {2, 8, e.k}) == GgufResidency::kExpandBf16);
    }
  }
}

TEST_CASE("GgufLoadPolicy::FromEnv reads VT_CPU_REF and VT_GGUF_KEEP_QUANT") {
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_KEEP_QUANT");
  {
    // PRODUCTION DEFAULT SINCE CIQ G4: keep-quant follows the running device's
    // ability to EXECUTE the quantized GEMM. The expectation is derived from
    // the op registry, not hardcoded to a build flavour, so this case states
    // the same rule on a CPU-only build (available -> ON) and on a CUDA build
    // (kMatmulBTQuant unregistered for kCUDA -> OFF, and the loader keeps
    // expanding to bf16 exactly as before).
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv();
    CHECK(p.keep_quant == vllm::GgufQuantComputeAvailable());
    CHECK(p.expand_nk == vllm::GgufQuantComputeAvailable());
    // L6: keep-f16 is OPT-IN (default OFF) — the binding measurement refuted the
    // RSS-closing premise and it regresses prefill, so it never defaults on.
    CHECK_FALSE(p.keep_f16);
    CHECK_FALSE(p.cpu_ref);
  }
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  CHECK(GgufLoadPolicy::FromEnv().keep_quant);
  CHECK(GgufLoadPolicy::FromEnv().expand_nk);
  CHECK_FALSE(GgufLoadPolicy::FromEnv().keep_f16);  // opt-in, still off
  // The OPT-OUT the spec promised must survive the default flip.
  for (const char* off : {"0", "false", "off", ""}) {
    ::setenv("VT_GGUF_KEEP_QUANT", off, 1);
    CAPTURE(off);
    CHECK_FALSE(GgufLoadPolicy::FromEnv().keep_quant);
    CHECK_FALSE(GgufLoadPolicy::FromEnv().expand_nk);
    CHECK_FALSE(GgufLoadPolicy::FromEnv().keep_f16);
  }
  // VT_GGUF_KEEP_F16=1 opts IN, but ONLY where expand_nk holds (CPU, not oracle);
  // it is inert with keep-quant off (nothing to keep) or under VT_CPU_REF.
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  ::setenv("VT_GGUF_KEEP_F16", "1", 1);
  CHECK(GgufLoadPolicy::FromEnv().keep_f16 ==
        vllm::GgufQuantComputeAvailable());
  for (const char* on : {"1", "true", "on"}) {
    ::setenv("VT_GGUF_KEEP_F16", on, 1);
    CAPTURE(on);
    CHECK(GgufLoadPolicy::FromEnv().keep_f16 ==
          vllm::GgufQuantComputeAvailable());
  }
  ::unsetenv("VT_GGUF_KEEP_F16");
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  ::setenv("VT_CPU_REF", "1", 1);
  {
    // The oracle switch: keep-quant requested, oracle wins — and it takes the
    // orientation and keep-f16 with it, so VT_CPU_REF=1 is the FULL historical
    // load.
    ::setenv("VT_GGUF_KEEP_F16", "1", 1);
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv();
    CHECK(p.keep_quant);
    CHECK(p.cpu_ref);
    CHECK_FALSE(p.expand_nk);
    CHECK_FALSE(p.keep_f16);
    ::unsetenv("VT_GGUF_KEEP_F16");
    CHECK(p.Route(vllm::GgufTensorInfo{"w", {8, 256}, kQ4_K, nullptr, 0},
                  GgufTensorRole::kMatmulWeight) ==
          GgufResidency::kExpandBf16);
  }
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_KEEP_QUANT");
}

// The default is only correct if it means "a block weight has a consumer". On
// this CPU test binary the quantized GEMM IS registered, so the availability
// probe must say so — otherwise the flip above would be vacuous.
TEST_CASE("GgufQuantComputeAvailable tracks the kMatmulBTQuant registration") {
  CHECK(vllm::GgufQuantComputeAvailable() ==
        vt::OpRegistered(vt::OpId::kMatmulBTQuant,
                         vllm::platforms::CurrentPlatform().device_type()));
  if (vllm::platforms::CurrentPlatform().is_cpu()) {
    // CPU-only build: the CPU kernel IS registered, so keep-quant is live and
    // the default flip is not vacuous.
    CHECK(vllm::GgufQuantComputeAvailable());
  }
}

// ===========================================================================
// L3 — the loader routes EVERY tensor, and keep-quant stays lossless there.
// ===========================================================================

namespace {

struct DenseDims {
  int64_t H = 64, vocab = 32, n_head = 2, n_head_kv = 1, head_dim = 32,
          I = 64, n_layer = 2;
};

std::string F32Bytes(int64_t n, float base) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float v = base + 0.25F * static_cast<float>(i % 7);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    for (int k = 0; k < 4; ++k) {
      s.push_back(static_cast<char>((bits >> (8 * k)) & 0xFF));
    }
  }
  return s;
}

// Real Q8_0 blocks produced by the ported `quantize_row_q8_0_ref`, so the
// loader test runs over data a converter could actually have written.
std::string Q8_0Bytes(int64_t rows, int64_t k, uint32_t seed) {
  std::vector<float> src(static_cast<size_t>(rows * k));
  uint32_t s = seed | 1U;
  for (float& v : src) {
    v = static_cast<float>(static_cast<int32_t>(NextRand(&s) % 2001) - 1000) /
        128.0F;
  }
  const size_t nbytes = vt::RowSizeBytes(vt::DType::kQ8_0, rows * k);
  std::string out(nbytes, '\0');
  vt::cpu::FromFloatFn q = vt::cpu::BlockFromFloat(vt::DType::kQ8_0);
  REQUIRE(q != nullptr);
  // Row by row: ggml quantizes whole rows, and a row is whole blocks.
  const size_t row_bytes = vt::RowSizeBytes(vt::DType::kQ8_0, k);
  for (int64_t r = 0; r < rows; ++r) {
    q(src.data() + r * k, out.data() + static_cast<size_t>(r) * row_bytes, k);
  }
  return out;
}

void AddQ8_0(GgufModelBuilder& b, const std::string& name, int64_t out_dim,
             int64_t in_dim, uint32_t seed) {
  b.AddTensor(name,
              {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
              kQ8_0, Q8_0Bytes(out_dim, in_dim, seed));
}

void AddF32T(GgufModelBuilder& b, const std::string& name,
             const std::vector<int64_t>& torch_shape, float base) {
  std::vector<uint64_t> dims;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it) {
    dims.push_back(static_cast<uint64_t>(*it));
  }
  int64_t n = 1;
  for (int64_t d : torch_shape) n *= d;
  b.AddTensor(name, dims, kF32, F32Bytes(n, base));
}

// A tiny DENSE (`qwen35`) GGUF whose GEMM weights are Q8_0 and whose norms /
// embedding stay F32 — the realistic mixed-type file the routing policy exists
// for. All layers are full attention (interval 1), so no GDN tensors appear.
// `tied` omits `output.weight`, which is how a tied-embedding GGUF says "the
// head IS token_embd" (llama.cpp TENSOR_DUPLICATED) — the L5 sharing case.
std::string BuildDenseQ8Gguf(const DenseDims& d, bool tied = false) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35"));
  b.AddKv(U32Kv("qwen35.embedding_length", static_cast<uint32_t>(d.H)));
  b.AddKv(U32Kv("qwen35.block_count", static_cast<uint32_t>(d.n_layer)));
  b.AddKv(U32Kv("qwen35.attention.head_count", static_cast<uint32_t>(d.n_head)));
  b.AddKv(U32Kv("qwen35.attention.head_count_kv",
                static_cast<uint32_t>(d.n_head_kv)));
  b.AddKv(U32Kv("qwen35.attention.key_length",
                static_cast<uint32_t>(d.head_dim)));
  b.AddKv(U32Kv("qwen35.feed_forward_length", static_cast<uint32_t>(d.I)));
  b.AddKv(F32Kv("qwen35.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen35.rope.freq_base", 1000000.0F));
  b.AddKv(U32Kv("qwen35.full_attention_interval", 1));
  b.AddKv(U32Kv("qwen35.context_length", 4096));

  AddF32T(b, "token_embd.weight", {d.vocab, d.H}, 0.5F);
  AddF32T(b, "output_norm.weight", {d.H}, 1.5F);
  if (!tied) AddQ8_0(b, "output.weight", d.vocab, d.H, 11);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    AddF32T(b, p + "attn_norm.weight", {d.H}, 1.25F);
    AddF32T(b, p + "post_attention_norm.weight", {d.H}, 1.75F);
    AddQ8_0(b, p + "attn_q.weight", d.n_head * d.head_dim, d.H, 21 + il);
    AddQ8_0(b, p + "attn_k.weight", d.n_head_kv * d.head_dim, d.H, 31 + il);
    AddQ8_0(b, p + "attn_v.weight", d.n_head_kv * d.head_dim, d.H, 41 + il);
    AddQ8_0(b, p + "attn_output.weight", d.H, d.n_head * d.head_dim, 51 + il);
    AddF32T(b, p + "attn_q_norm.weight", {d.head_dim}, 1.5F);
    AddF32T(b, p + "attn_k_norm.weight", {d.head_dim}, 1.5F);
    AddQ8_0(b, p + "ffn_gate.weight", d.I, d.H, 61 + il);
    AddQ8_0(b, p + "ffn_up.weight", d.I, d.H, 71 + il);
    AddQ8_0(b, p + "ffn_down.weight", d.H, d.I, 81 + il);
  }
  return b.Build();
}

// The bf16 [K,N] tensor today's loader produces from a [N,K] GGUF weight:
// dequant, then transpose. Used to prove the keep-quant weight carries the
// SAME information as the expanded one.
std::vector<uint16_t> ExpandAndTranspose(uint32_t ggml_type,
                                         const uint8_t* blocks, int64_t n,
                                         int64_t k) {
  const std::vector<uint16_t> dq =
      vllm::DequantGgufRowToBf16(ggml_type, blocks, n * k);
  std::vector<uint16_t> out(dq.size());
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t c = 0; c < k; ++c) out[c * n + r] = dq[r * k + c];
  }
  return out;
}

}  // namespace

TEST_CASE("loader routes EVERY tensor in the file (total coverage)") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  std::set<std::string> routed;
  std::set<std::string> kept;
  GgufLoadPolicy pol = KeepQuantOn();
  pol.audit = [&](const std::string& name, GgufTensorRole,
                  GgufResidency res) {
    routed.insert(name);
    if (res == GgufResidency::kKeepQuant) kept.insert(name);
  };
  const vllm::Qwen3_5DenseWeights w =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &pol);
  REQUIRE(w.layers.size() == static_cast<size_t>(d.n_layer));

  // TOTALITY: the audited set is exactly the file's tensor list. Nothing the
  // loader consumed skipped the policy, and the policy saw nothing spurious.
  std::set<std::string> in_file;
  for (const vllm::GgufTensorInfo& t : g.Tensors()) in_file.insert(t.name);
  CHECK(routed == in_file);

  // And the decisions are the intended ones: every Q8_0 GEMM weight kept, the
  // F32 norms / embedding expanded.
  std::set<std::string> expect_kept;
  expect_kept.insert("output.weight");
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    for (const char* s : {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                          "attn_output.weight", "ffn_gate.weight",
                          "ffn_up.weight", "ffn_down.weight"}) {
      expect_kept.insert(p + s);
    }
  }
  CHECK(kept == expect_kept);
  CHECK(kept.count("token_embd.weight") == 0);
  CHECK(kept.count("output_norm.weight") == 0);
  CHECK(kept.count("blk.0.attn_q_norm.weight") == 0);
}

TEST_CASE("loader keep-quant weights are lossless vs the bf16 expansion") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  const GgufLoadPolicy off;  // production default: expand everything
  GgufLoadPolicy on = KeepQuantOn();
  const vllm::Qwen3_5DenseWeights base =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &off);
  const vllm::Qwen3_5DenseWeights kept =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &on);

  // Pairs of (kept weight, expanded weight, source tensor).
  struct Pair {
    const vllm::OwnedTensor* q;
    const vllm::OwnedTensor* b;
    std::string name;
  };
  std::vector<Pair> pairs = {{&kept.lm_head, &base.lm_head, "output.weight"}};
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto& kl = kept.layers[static_cast<size_t>(il)];
    const auto& bl = base.layers[static_cast<size_t>(il)];
    pairs.push_back({&kl.attn.q_proj, &bl.attn.q_proj, p + "attn_q.weight"});
    pairs.push_back({&kl.attn.k_proj, &bl.attn.k_proj, p + "attn_k.weight"});
    pairs.push_back({&kl.attn.v_proj, &bl.attn.v_proj, p + "attn_v.weight"});
    pairs.push_back(
        {&kl.attn.o_proj, &bl.attn.o_proj, p + "attn_output.weight"});
    pairs.push_back(
        {&kl.mlp.gate_proj, &bl.mlp.gate_proj, p + "ffn_gate.weight"});
    pairs.push_back({&kl.mlp.up_proj, &bl.mlp.up_proj, p + "ffn_up.weight"});
    pairs.push_back(
        {&kl.mlp.down_proj, &bl.mlp.down_proj, p + "ffn_down.weight"});
  }

  for (const Pair& pr : pairs) {
    CAPTURE(pr.name);
    const vllm::GgufTensorInfo& t = g.Get(pr.name);
    const int64_t n = t.shape[0];
    const int64_t k = t.shape[1];

    // Kept: block dtype, file orientation [N,K], nk = true, raw file bytes.
    CHECK(pr.q->dtype == vt::DType::kQ8_0);
    CHECK(pr.q->nk == true);
    CHECK(pr.q->shape[0] == n);
    CHECK(pr.q->shape[1] == k);
    REQUIRE(pr.q->bytes.size() == t.nbytes);
    CHECK(std::memcmp(pr.q->bytes.data(), t.data, t.nbytes) == 0);

    // Expanded: bf16, Matmul-B orientation [K,N] — today's exact behavior.
    CHECK(pr.b->dtype == vt::DType::kBF16);
    CHECK(pr.b->nk == false);
    CHECK(pr.b->shape[0] == k);
    CHECK(pr.b->shape[1] == n);

    // THE GATE: dequantizing the resident blocks (and applying the same
    // transpose) reproduces the expanded tensor BYTE for BYTE.
    const std::vector<uint16_t> rehydrated =
        ExpandAndTranspose(kQ8_0, pr.q->bytes.data(), n, k);
    REQUIRE(rehydrated.size() * sizeof(uint16_t) == pr.b->bytes.size());
    CHECK(std::memcmp(rehydrated.data(), pr.b->bytes.data(),
                      pr.b->bytes.size()) == 0);
  }

  // Tensors that must NOT keep quant are bit-identical between the two loads,
  // so enabling keep-quant cannot perturb them.
  CHECK(kept.embed_tokens.bytes == base.embed_tokens.bytes);
  CHECK(kept.final_norm.bytes == base.final_norm.bytes);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const auto& kl = kept.layers[static_cast<size_t>(il)];
    const auto& bl = base.layers[static_cast<size_t>(il)];
    CHECK(kl.input_layernorm.bytes == bl.input_layernorm.bytes);
    CHECK(kl.post_attention_layernorm.bytes ==
          bl.post_attention_layernorm.bytes);
    CHECK(kl.attn.q_norm.bytes == bl.attn.q_norm.bytes);
    CHECK(kl.attn.k_norm.bytes == bl.attn.k_norm.bytes);
  }
}

TEST_CASE("VT_CPU_REF forces the full dequant oracle path in the loader") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  const GgufLoadPolicy off;
  GgufLoadPolicy oracle = KeepQuantOn();
  oracle.cpu_ref = true;  // VT_CPU_REF=1
  std::set<std::string> kept;
  oracle.audit = [&](const std::string& n, GgufTensorRole, GgufResidency r) {
    if (r == GgufResidency::kKeepQuant) kept.insert(n);
  };

  const vllm::Qwen3_5DenseWeights base =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &off);
  const vllm::Qwen3_5DenseWeights ref =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &oracle);

  // Nothing stayed quantized...
  CHECK(kept.empty());
  // ...and every weight is BIT-IDENTICAL to the historical load. This is the
  // whole point of the oracle switch: the reference numerics stay reachable.
  CHECK(ref.embed_tokens.bytes == base.embed_tokens.bytes);
  CHECK(ref.final_norm.bytes == base.final_norm.bytes);
  CHECK(ref.lm_head.bytes == base.lm_head.bytes);
  CHECK(ref.lm_head.dtype == base.lm_head.dtype);
  REQUIRE(ref.layers.size() == base.layers.size());
  for (size_t il = 0; il < ref.layers.size(); ++il) {
    CAPTURE(il);
    const auto& r = ref.layers[il];
    const auto& b = base.layers[il];
    CHECK(r.input_layernorm.bytes == b.input_layernorm.bytes);
    CHECK(r.post_attention_layernorm.bytes == b.post_attention_layernorm.bytes);
    CHECK(r.attn.q_proj.bytes == b.attn.q_proj.bytes);
    CHECK(r.attn.k_proj.bytes == b.attn.k_proj.bytes);
    CHECK(r.attn.v_proj.bytes == b.attn.v_proj.bytes);
    CHECK(r.attn.o_proj.bytes == b.attn.o_proj.bytes);
    CHECK(r.attn.q_norm.bytes == b.attn.q_norm.bytes);
    CHECK(r.attn.k_norm.bytes == b.attn.k_norm.bytes);
    CHECK(r.mlp.gate_proj.bytes == b.mlp.gate_proj.bytes);
    CHECK(r.mlp.up_proj.bytes == b.mlp.up_proj.bytes);
    CHECK(r.mlp.down_proj.bytes == b.mlp.down_proj.bytes);
  }
}

// ===========================================================================
// L2/L3 — the MoE loader's STACKED-EXPERT keep-quant split.
// ===========================================================================

namespace {

struct MoeDims {
  int64_t H = 64, vocab = 32, n_head = 2, n_head_kv = 1, head_dim = 32,
          E = 3, used = 2, I = 64, Is = 64, n_layer = 1;
};

// A tiny `qwen35moe` GGUF: Q8_0 GEMM weights (including the STACKED expert
// tensors) and F32 norms/embedding. full_attention_interval = 1 makes every
// layer full-attention, so no GDN tensors are needed.
std::string BuildMoeQ8Gguf(const MoeDims& d) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35moe"));
  b.AddKv(U32Kv("qwen35moe.embedding_length", static_cast<uint32_t>(d.H)));
  b.AddKv(U32Kv("qwen35moe.block_count", static_cast<uint32_t>(d.n_layer)));
  b.AddKv(U32Kv("qwen35moe.attention.head_count",
                static_cast<uint32_t>(d.n_head)));
  b.AddKv(U32Kv("qwen35moe.attention.head_count_kv",
                static_cast<uint32_t>(d.n_head_kv)));
  b.AddKv(U32Kv("qwen35moe.attention.key_length",
                static_cast<uint32_t>(d.head_dim)));
  b.AddKv(U32Kv("qwen35moe.expert_count", static_cast<uint32_t>(d.E)));
  b.AddKv(U32Kv("qwen35moe.expert_used_count", static_cast<uint32_t>(d.used)));
  b.AddKv(U32Kv("qwen35moe.expert_feed_forward_length",
                static_cast<uint32_t>(d.I)));
  b.AddKv(U32Kv("qwen35moe.expert_shared_feed_forward_length",
                static_cast<uint32_t>(d.Is)));
  b.AddKv(F32Kv("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen35moe.rope.freq_base", 1000000.0F));
  b.AddKv(U32Kv("qwen35moe.full_attention_interval", 1));
  b.AddKv(U32Kv("qwen35moe.context_length", 4096));

  AddF32T(b, "token_embd.weight", {d.vocab, d.H}, 0.5F);
  AddF32T(b, "output_norm.weight", {d.H}, 1.5F);
  AddQ8_0(b, "output.weight", d.vocab, d.H, 111);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    AddF32T(b, p + "attn_norm.weight", {d.H}, 1.25F);
    AddF32T(b, p + "post_attention_norm.weight", {d.H}, 1.75F);
    AddQ8_0(b, p + "attn_q.weight", d.n_head * d.head_dim, d.H, 121 + il);
    AddQ8_0(b, p + "attn_k.weight", d.n_head_kv * d.head_dim, d.H, 131 + il);
    AddQ8_0(b, p + "attn_v.weight", d.n_head_kv * d.head_dim, d.H, 141 + il);
    AddQ8_0(b, p + "attn_output.weight", d.H, d.n_head * d.head_dim, 151 + il);
    AddF32T(b, p + "attn_q_norm.weight", {d.head_dim}, 1.5F);
    AddF32T(b, p + "attn_k_norm.weight", {d.head_dim}, 1.5F);
    AddQ8_0(b, p + "ffn_gate_inp.weight", d.E, d.H, 161 + il);
    AddF32T(b, p + "ffn_gate_inp_shexp.weight", {d.H}, 0.75F);
    // Stacked experts: torch [E, out, in].
    b.AddTensor(p + "ffn_gate_exps.weight",
                {static_cast<uint64_t>(d.H), static_cast<uint64_t>(d.I),
                 static_cast<uint64_t>(d.E)},
                kQ8_0, Q8_0Bytes(d.E * d.I, d.H, 171 + il));
    b.AddTensor(p + "ffn_up_exps.weight",
                {static_cast<uint64_t>(d.H), static_cast<uint64_t>(d.I),
                 static_cast<uint64_t>(d.E)},
                kQ8_0, Q8_0Bytes(d.E * d.I, d.H, 181 + il));
    b.AddTensor(p + "ffn_down_exps.weight",
                {static_cast<uint64_t>(d.I), static_cast<uint64_t>(d.H),
                 static_cast<uint64_t>(d.E)},
                kQ8_0, Q8_0Bytes(d.E * d.H, d.I, 191 + il));
    AddQ8_0(b, p + "ffn_gate_shexp.weight", d.Is, d.H, 201 + il);
    AddQ8_0(b, p + "ffn_up_shexp.weight", d.Is, d.H, 211 + il);
    AddQ8_0(b, p + "ffn_down_shexp.weight", d.H, d.Is, 221 + il);
  }
  return b.Build();
}

}  // namespace

TEST_CASE("loader keep-quant expert split is lossless per expert") {
  const MoeDims d;
  const TempFile f(BuildMoeQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  const GgufLoadPolicy off;
  GgufLoadPolicy on = KeepQuantOn();
  std::set<std::string> routed;
  std::set<std::string> kept_names;
  on.audit = [&](const std::string& n, GgufTensorRole, GgufResidency r) {
    routed.insert(n);
    if (r == GgufResidency::kKeepQuant) kept_names.insert(n);
  };
  const vllm::Qwen3_5MoeWeights base = vllm::LoadQwen3_5MoeFromGguf(g, c, &off);
  const vllm::Qwen3_5MoeWeights kept = vllm::LoadQwen3_5MoeFromGguf(g, c, &on);

  // Totality on the MoE tensor list too.
  std::set<std::string> in_file;
  for (const vllm::GgufTensorInfo& t : g.Tensors()) in_file.insert(t.name);
  CHECK(routed == in_file);

  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto& kl = kept.layers[static_cast<size_t>(il)].moe;
    const auto& bl = base.layers[static_cast<size_t>(il)].moe;
    struct Stack {
      const std::vector<vllm::OwnedTensor>* q;
      const std::vector<vllm::OwnedTensor>* b;
      std::string name;
    };
    const Stack stacks[] = {
        {&kl.expert_gate, &bl.expert_gate, p + "ffn_gate_exps.weight"},
        {&kl.expert_up, &bl.expert_up, p + "ffn_up_exps.weight"},
        {&kl.expert_down, &bl.expert_down, p + "ffn_down_exps.weight"},
    };
    for (const Stack& s : stacks) {
      CAPTURE(s.name);
      CHECK(kept_names.count(s.name) == 1);
      const vllm::GgufTensorInfo& t = g.Get(s.name);
      const int64_t out_dim = t.shape[1];
      const int64_t in_dim = t.shape[2];
      REQUIRE(s.q->size() == static_cast<size_t>(d.E));
      REQUIRE(s.b->size() == static_cast<size_t>(d.E));
      const size_t per_bytes =
          vt::RowSizeBytes(vt::DType::kQ8_0, out_dim * in_dim);
      for (int64_t e = 0; e < d.E; ++e) {
        CAPTURE(e);
        const vllm::OwnedTensor& q = (*s.q)[static_cast<size_t>(e)];
        const vllm::OwnedTensor& bexp = (*s.b)[static_cast<size_t>(e)];
        CHECK(q.dtype == vt::DType::kQ8_0);
        CHECK(q.nk == true);
        CHECK(q.shape[0] == out_dim);
        CHECK(q.shape[1] == in_dim);
        REQUIRE(q.bytes.size() == per_bytes);
        // The slice is EXACTLY this expert's byte range of the stacked tensor —
        // a wrong offset (e.g. always expert 0) is caught here.
        CHECK(std::memcmp(q.bytes.data(),
                          t.data + static_cast<size_t>(e) * per_bytes,
                          per_bytes) == 0);
        // ...and dequantizing it reproduces the expanded expert byte for byte.
        const std::vector<uint16_t> rehydrated =
            ExpandAndTranspose(kQ8_0, q.bytes.data(), out_dim, in_dim);
        REQUIRE(rehydrated.size() * sizeof(uint16_t) == bexp.bytes.size());
        CHECK(std::memcmp(rehydrated.data(), bexp.bytes.data(),
                          bexp.bytes.size()) == 0);
      }
    }
    // The 1-D shared gate is a vector: never kept, and bit-identical.
    CHECK(kept_names.count(p + "ffn_gate_inp_shexp.weight") == 0);
    CHECK(kl.shared_gate.bytes == bl.shared_gate.bytes);
  }
}

// CIQ G4. The production default is no longer "expand everything": where the
// device can run the quantized GEMM, an env-driven load must equal a load
// under an explicitly-ON policy, block dtypes and orientation included.
TEST_CASE("production default is keep-quant wherever the quant GEMM exists") {
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_KEEP_QUANT");
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // The explicit policy must mirror FromEnv's FULL CPU default, otherwise the
  // byte comparison below is not apples-to-apples. On an i8mm box (dgx aarch64)
  // FromEnv turns quant_repack ON, which PERMUTES the q8_0 bytes, so an `expect`
  // that leaves it off would differ from the env load on the repacked weights
  // (this bit was invisible on x86 CI, where repack is off — L6 fix).
  GgufLoadPolicy expect;
  expect.keep_quant = vllm::GgufQuantComputeAvailable();
  expect.expand_nk = expect.keep_quant;
  // keep_f16 is opt-in (default OFF), so FromEnv leaves it off here too.
  expect.mmap_residency = expect.keep_quant;
  expect.share_tied_head = expect.keep_quant;
  expect.gdn_expand_nk = expect.keep_quant;
  expect.quant_repack = expect.keep_quant && vt::cpu::QuantRepackActive();

  const vllm::Qwen3_5DenseWeights from_env =
      vllm::LoadQwen3_5DenseFromGguf(g, c, /*policy=*/nullptr);
  const vllm::Qwen3_5DenseWeights from_policy =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &expect);

  const vt::DType want =
      expect.keep_quant ? vt::DType::kQ8_0 : vt::DType::kBF16;
  CHECK(from_env.lm_head.bytes == from_policy.lm_head.bytes);
  CHECK(from_env.lm_head.dtype == want);
  CHECK(from_env.lm_head.nk == expect.keep_quant);
  REQUIRE(from_env.layers.size() == from_policy.layers.size());
  for (size_t il = 0; il < from_env.layers.size(); ++il) {
    const auto& a = from_env.layers[il];
    const auto& b = from_policy.layers[il];
    CHECK(a.attn.q_proj.bytes == b.attn.q_proj.bytes);
    CHECK(a.mlp.down_proj.bytes == b.mlp.down_proj.bytes);
    CHECK(a.attn.q_proj.dtype == want);
    CHECK(a.attn.q_proj.nk == expect.keep_quant);
  }
}

// ===========================================================================
// G4 — orientation: an EXPANDED matmul weight keeps the file's [N, K] order.
// ===========================================================================

// The measured 1.3-3.0x lever. Its correctness argument is that the two CPU
// GEMM kernels differ ONLY in the weight offset, so this must be BIT-EXACT
// against the transposed load, element by element — not merely "close".
TEST_CASE("expand_nk is the untransposed view of the SAME expanded weight") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // keep_quant OFF so every matmul weight EXPANDS; only the orientation moves.
  const GgufLoadPolicy transposed;
  GgufLoadPolicy raw;
  raw.expand_nk = true;

  const vllm::Qwen3_5DenseWeights t =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &transposed);
  const vllm::Qwen3_5DenseWeights r = vllm::LoadQwen3_5DenseFromGguf(g, c, &raw);

  auto same_weight = [](const vllm::OwnedTensor& kn,
                        const vllm::OwnedTensor& nk) {
    REQUIRE(kn.rank == 2);
    REQUIRE(nk.rank == 2);
    CHECK_FALSE(kn.nk);
    CHECK(nk.nk);
    CHECK(kn.dtype == vt::DType::kBF16);
    CHECK(nk.dtype == vt::DType::kBF16);
    // [K, N] vs [N, K]: the same matrix, transposed.
    REQUIRE(kn.shape[0] == nk.shape[1]);
    REQUIRE(kn.shape[1] == nk.shape[0]);
    const auto* a = reinterpret_cast<const uint16_t*>(kn.bytes.data());
    const auto* b = reinterpret_cast<const uint16_t*>(nk.bytes.data());
    const int64_t K = kn.shape[0], N = kn.shape[1];
    REQUIRE(kn.bytes.size() == nk.bytes.size());
    for (int64_t i = 0; i < K; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        // A single mismatched element fails loudly with its index.
        if (a[i * N + j] != b[j * K + i]) {
          CAPTURE(i);
          CAPTURE(j);
          REQUIRE(a[i * N + j] == b[j * K + i]);
        }
      }
    }
  };

  same_weight(t.lm_head, r.lm_head);
  REQUIRE(t.layers.size() == r.layers.size());
  for (size_t il = 0; il < t.layers.size(); ++il) {
    CAPTURE(il);
    same_weight(t.layers[il].attn.q_proj, r.layers[il].attn.q_proj);
    same_weight(t.layers[il].attn.o_proj, r.layers[il].attn.o_proj);
    same_weight(t.layers[il].mlp.gate_proj, r.layers[il].mlp.gate_proj);
    same_weight(t.layers[il].mlp.down_proj, r.layers[il].mlp.down_proj);
    // Tensors that are NOT matmul weights are untouched by the orientation
    // switch — it must not leak into the norm/embedding/conv paths.
    CHECK(t.layers[il].input_layernorm.bytes ==
          r.layers[il].input_layernorm.bytes);
  }
  CHECK(t.embed_tokens.bytes == r.embed_tokens.bytes);
  CHECK_FALSE(r.embed_tokens.nk);
  CHECK(t.final_norm.bytes == r.final_norm.bytes);
}

// ===========================================================================
// G4 — the ROUTING itself: vt::MatmulBT consumes a block weight.
// ===========================================================================

// The whole enablement in one assertion: a block-typed [N, K] weight handed to
// the SAME entry point every model matmul helper already calls must produce
// the quantized GEMM's answer, not throw. Before G4 this threw
// "matmul_bt: float inputs and f32/bf16 output required".
TEST_CASE("vt::MatmulBT routes a block-quantized weight to MatmulBTQuant") {
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = backend.CreateQueue();

  const int64_t M = 3, N = 5, K = 64;
  std::vector<float> act(static_cast<size_t>(M) * K);
  for (size_t i = 0; i < act.size(); ++i) {
    act[i] = 0.25F * static_cast<float>((i % 17)) - 1.5F;
  }
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = 0.125F * static_cast<float>((i % 23)) - 1.0F;
  }
  // Quantize the weight rows to Q8_0 exactly as a GGUF file stores them.
  std::vector<uint8_t> blocks(vt::RowSizeBytes(vt::DType::kQ8_0, K) *
                              static_cast<size_t>(N));
  for (int64_t j = 0; j < N; ++j) {
    vt::cpu::QuantTraits(vt::DType::kQ8_0)
        .from_float(w.data() + j * K,
                    blocks.data() + static_cast<size_t>(j) *
                                        vt::RowSizeBytes(vt::DType::kQ8_0, K),
                    K);
  }

  vt::Tensor a = vt::Tensor::Contiguous(act.data(), vt::DType::kF32,
                                        q.device, {M, K});
  vt::Tensor b = vt::Tensor::Contiguous(blocks.data(), vt::DType::kQ8_0,
                                        q.device, {N, K});
  std::vector<float> got(static_cast<size_t>(M) * N, 0.0F);
  std::vector<float> want(static_cast<size_t>(M) * N, 0.0F);
  vt::Tensor og = vt::Tensor::Contiguous(got.data(), vt::DType::kF32,
                                         q.device, {M, N});
  vt::Tensor ow = vt::Tensor::Contiguous(want.data(), vt::DType::kF32,
                                         q.device, {M, N});

  vt::MatmulBT(q, og, a, b);        // the routed call site
  vt::MatmulBTQuant(q, ow, a, b);   // the op it must have reached
  backend.Synchronize(q);

  // Bit-identical: MatmulBT must DELEGATE, not approximate.
  CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
  // ...and the answer is a real GEMM, not zeros.
  bool nonzero = false;
  for (float v : got) nonzero = nonzero || v != 0.0F;
  CHECK(nonzero);
}

// ===========================================================================
// L5 — mmap residency and tied-head sharing.
//
// Both replace a COPY with a VIEW, so both raise the same two questions: are
// the bytes identical to the copy arm, and can the view outlive what it views?
// The second is the real hazard the spec names, so it is gated by actually
// destroying the producer rather than by reasoning about it.
// ===========================================================================

namespace {

GgufLoadPolicy KeepQuantMmap() {
  GgufLoadPolicy p = KeepQuantOn();
  p.mmap_residency = true;
  return p;
}

// The production CPU shape: keep-quant + untransposed expansion + both L5
// residency refinements, exactly as GgufLoadPolicy::FromEnv builds it there.
GgufLoadPolicy ProductionCpu() {
  GgufLoadPolicy p = KeepQuantOn();
  p.expand_nk = true;
  p.mmap_residency = true;
  p.share_tied_head = true;
  // keep_f16 stays OFF: it is opt-in (default OFF), see FromEnv / L6.
  return p;
}

bool FileHasTensor(const vllm::GgufFile& g, const std::string& name) {
  for (const vllm::GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("mmap residency is byte-identical to the copy arm, and borrows") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  GgufLoadPolicy copy = KeepQuantOn();    // L2 arm: blocks copied out
  GgufLoadPolicy mmap = KeepQuantMmap();  // L5 arm: blocks viewed in place
  const vllm::Qwen3_5DenseWeights wc =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &copy);
  const vllm::Qwen3_5DenseWeights wm =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);

  // The kept weights: same dtype/shape/orientation, same BYTES, different
  // residency — and the mmap arm's bytes are literally the file's own.
  const vllm::GgufTensorInfo& t = g.Get("output.weight");
  CHECK(wc.lm_head.bytes.borrowed() == false);
  CHECK(wm.lm_head.bytes.borrowed() == true);
  CHECK(wm.lm_head.dtype == wc.lm_head.dtype);
  CHECK(wm.lm_head.nk == wc.lm_head.nk);
  CHECK(wm.lm_head.shape[0] == wc.lm_head.shape[0]);
  CHECK(wm.lm_head.shape[1] == wc.lm_head.shape[1]);
  CHECK(wm.lm_head.bytes == wc.lm_head.bytes);
  CHECK(wm.lm_head.bytes.data() == t.data);  // IN PLACE, not a copy
  CHECK(wm.lm_head.View().data == const_cast<uint8_t*>(t.data));

  for (int64_t il = 0; il < d.n_layer; ++il) {
    CAPTURE(il);
    const auto& lc = wc.layers[static_cast<size_t>(il)];
    const auto& lm = wm.layers[static_cast<size_t>(il)];
    CHECK(lm.attn.q_proj.bytes.borrowed());
    CHECK(lm.attn.q_proj.bytes == lc.attn.q_proj.bytes);
    CHECK(lm.attn.o_proj.bytes == lc.attn.o_proj.bytes);
    CHECK(lm.mlp.down_proj.bytes == lc.mlp.down_proj.bytes);
    // An EXPANDED tensor owns its bytes in BOTH arms — mmap residency applies
    // to kept blocks only and must not leak into the expansion path.
    CHECK(lm.input_layernorm.bytes.borrowed() == false);
    CHECK(lm.input_layernorm.bytes == lc.input_layernorm.bytes);
  }
}

TEST_CASE("a borrowed weight OUTLIVES the GgufFile and the file itself") {
  const DenseDims d;
  std::vector<uint8_t> expected;
  std::unique_ptr<vllm::Qwen3_5DenseWeights> w;
  int64_t numel = 0;

  {
    // Both the reader and the on-disk file go away inside this scope. The
    // weights must stay readable: the mapping is refcounted (GgufMapping), and
    // a POSIX mapping survives unlink of its path.
    const TempFile f(BuildDenseQ8Gguf(d));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
    GgufLoadPolicy mmap = KeepQuantMmap();
    w = std::make_unique<vllm::Qwen3_5DenseWeights>(
        vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap));
    const vllm::GgufTensorInfo& t = g.Get("output.weight");
    expected.assign(t.data, t.data + t.nbytes);
    numel = w->lm_head.shape[0] * w->lm_head.shape[1];
  }

  REQUIRE(w->lm_head.bytes.borrowed());
  REQUIRE(w->lm_head.bytes.size() == expected.size());
  // Reading through the view that would be dangling if it were not refcounted.
  CHECK(std::memcmp(w->lm_head.bytes.data(), expected.data(),
                    expected.size()) == 0);
  // ...and decoding it still gives the right answer, i.e. these are the file's
  // pages and not recycled memory that happens to compare equal.
  const std::vector<uint16_t> re =
      vllm::DequantGgufRowToBf16(kQ8_0, w->lm_head.bytes.data(), numel);
  const std::vector<uint16_t> ref =
      vllm::DequantGgufRowToBf16(kQ8_0, expected.data(), numel);
  CHECK(re == ref);

  w.reset();  // last reference: the mapping is unmapped exactly once, here
}

TEST_CASE("tied lm_head SHARES the embedding expansion instead of copying it") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  REQUIRE_FALSE(FileHasTensor(g, "output.weight"));

  GgufLoadPolicy shared = ProductionCpu();
  GgufLoadPolicy unshared = ProductionCpu();
  unshared.share_tied_head = false;
  const vllm::Qwen3_5DenseWeights ws =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
  const vllm::Qwen3_5DenseWeights wu =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &unshared);

  // THE GATE: sharing changes WHERE the bytes live, never WHAT they are.
  CHECK(ws.embed_tokens.bytes == wu.embed_tokens.bytes);
  CHECK(ws.lm_head.bytes == wu.lm_head.bytes);
  CHECK(ws.lm_head.dtype == wu.lm_head.dtype);
  CHECK(ws.lm_head.nk == wu.lm_head.nk);
  CHECK(ws.lm_head.rank == wu.lm_head.rank);
  CHECK(ws.lm_head.shape[0] == wu.lm_head.shape[0]);
  CHECK(ws.lm_head.shape[1] == wu.lm_head.shape[1]);

  // ONE buffer in the shared arm, TWO in the unshared one. This is the whole
  // point of the row: a failure here means the second vocab matrix is still
  // being paid for.
  CHECK(ws.embed_tokens.bytes.data() == ws.lm_head.bytes.data());
  CHECK(wu.embed_tokens.bytes.data() != wu.lm_head.bytes.data());
  CHECK(ws.embed_tokens.bytes.borrowed());
  CHECK(ws.lm_head.bytes.borrowed());

  // The head is in the file's own [N = vocab, K = H] order, which is the ONLY
  // orientation in which the two byte images can coincide...
  CHECK(ws.lm_head.nk == true);
  CHECK(ws.lm_head.shape[0] == d.vocab);
  CHECK(ws.lm_head.shape[1] == d.H);
  // ...and the gather table keeps its own metadata over the same bytes.
  CHECK(ws.embed_tokens.nk == false);
  CHECK(ws.embed_tokens.shape[0] == d.vocab);
  CHECK(ws.embed_tokens.shape[1] == d.H);
}

TEST_CASE("a shared tied head is freed ONCE: either half may die first") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  GgufLoadPolicy shared = ProductionCpu();

  // Drop the EMBEDDING first, then read the head.
  {
    vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
    const vllm::OwnedBytes& head = w.lm_head.bytes;
    const std::vector<uint8_t> want(head.begin(), head.end());
    w.embed_tokens.bytes.Reset();
    REQUIRE(head.size() == want.size());
    CHECK(std::memcmp(head.data(), want.data(), want.size()) == 0);
  }
  // Drop the HEAD first, then read the embedding.
  {
    vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
    const vllm::OwnedBytes& emb = w.embed_tokens.bytes;
    const std::vector<uint8_t> want(emb.begin(), emb.end());
    w.lm_head.bytes.Reset();
    REQUIRE(emb.size() == want.size());
    CHECK(std::memcmp(emb.data(), want.data(), want.size()) == 0);
  }
  // ReleaseHost() on a borrowed weight drops only the reference: it must not
  // madvise pages it does not own, and the other half must stay readable.
  {
    vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
    const vllm::OwnedBytes& head = w.lm_head.bytes;
    const std::vector<uint8_t> want(head.begin(), head.end());
    w.embed_tokens.ReleaseHost();
    CHECK(w.embed_tokens.bytes.empty());
    REQUIRE(head.size() == want.size());
    CHECK(std::memcmp(head.data(), want.data(), want.size()) == 0);
  }
}

TEST_CASE("the oracle path shares NOTHING and borrows NOTHING") {
  const DenseDims d;
  const TempFile f(BuildDenseQ8Gguf(d, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // VT_CPU_REF forces every L5 refinement off (see FromEnv), so the load is the
  // historical two-buffer, transposed-head one, allocation for allocation.
  GgufLoadPolicy oracle = ProductionCpu();
  oracle.cpu_ref = true;
  oracle.expand_nk = false;
  oracle.mmap_residency = false;
  oracle.share_tied_head = false;
  const vllm::Qwen3_5DenseWeights wo =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &oracle);

  CHECK(wo.embed_tokens.bytes.borrowed() == false);
  CHECK(wo.lm_head.bytes.borrowed() == false);
  CHECK(wo.embed_tokens.bytes.data() != wo.lm_head.bytes.data());
  // Transposed to Matmul-B [K = H, N = vocab] — genuinely different bytes from
  // the gather table, which is precisely why this arm cannot share.
  CHECK(wo.lm_head.nk == false);
  CHECK(wo.lm_head.shape[0] == d.H);
  CHECK(wo.lm_head.shape[1] == d.vocab);
  CHECK(wo.lm_head.bytes != wo.embed_tokens.bytes);
}

TEST_CASE("FromEnv derives both L5 switches, and VT_CPU_REF overrides them") {
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_SHARE_TIED_HEAD");
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  {
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv();
    CHECK(p.keep_quant);
    CHECK(p.expand_nk);
    CHECK(p.mmap_residency);   // defaults ON with keep-quant
    CHECK(p.share_tied_head);  // defaults ON with expand_nk
  }
  // Each is independently opt-out-able, so the copy and duplicate arms stay
  // A/B-able against the production default.
  for (const char* off : {"0", "false", "off", ""}) {
    CAPTURE(off);
    ::setenv("VT_GGUF_MMAP", off, 1);
    ::setenv("VT_GGUF_SHARE_TIED_HEAD", off, 1);
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv();
    CHECK(p.keep_quant);
    CHECK_FALSE(p.mmap_residency);
    CHECK_FALSE(p.share_tied_head);
  }
  // Turning keep-quant off takes both with it: there is nothing to borrow when
  // every weight expands, and the head is transposed again.
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_SHARE_TIED_HEAD");
  ::setenv("VT_GGUF_KEEP_QUANT", "0", 1);
  {
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv();
    CHECK_FALSE(p.mmap_residency);
    CHECK_FALSE(p.share_tied_head);
  }
  // The oracle switch wins over both, even when they are asked for explicitly.
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  ::setenv("VT_GGUF_MMAP", "1", 1);
  ::setenv("VT_GGUF_SHARE_TIED_HEAD", "1", 1);
  ::setenv("VT_CPU_REF", "1", 1);
  {
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv();
    CHECK(p.cpu_ref);
    CHECK_FALSE(p.expand_nk);
    CHECK_FALSE(p.mmap_residency);
    CHECK_FALSE(p.share_tied_head);
  }
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_SHARE_TIED_HEAD");
  ::unsetenv("VT_GGUF_KEEP_QUANT");
}

TEST_CASE("OwnedBytes refuses to MUTATE a borrowed buffer") {
  auto holder = std::make_shared<const std::vector<uint8_t>>(
      std::vector<uint8_t>{1, 2, 3, 4});
  vllm::OwnedBytes b =
      vllm::OwnedBytes::Borrow(holder->data(), holder->size(), holder);
  REQUIRE(b.borrowed());
  REQUIRE(b.size() == 4);
  const vllm::OwnedBytes& rb = b;
  CHECK(rb.data()[0] == 1);  // a const read is fine

  // Every write path is a loud failure instead of a silent write through
  // read-only memory. That is what makes the borrowed residency safe to
  // introduce underneath ~300 existing `.bytes` call sites.
  CHECK_THROWS_AS(b.resize(8), std::runtime_error);
  CHECK_THROWS_AS(b.assign(size_t{2}, uint8_t{0}), std::runtime_error);
  CHECK_THROWS_AS(b.Share(), std::runtime_error);
  // Element access, const or not, is NOT rejected: constness does not
  // distinguish a read from a write, and this tree reads weights through
  // non-const handles everywhere. Only the STRUCTURAL mutations above — the
  // ones that would move the buffer out from under the owner — are refused.
  CHECK(b.data() == rb.data());

  // A borrow with no keep-alive is unrepresentable.
  CHECK_THROWS_AS(vllm::OwnedBytes::Borrow(holder->data(), 4, nullptr),
                  std::runtime_error);

  b.Reset();
  CHECK_FALSE(b.borrowed());
  CHECK(b.empty());
  b.resize(2);  // and it is reusable as an owned buffer afterwards
  CHECK(b.size() == 2);
}

TEST_CASE("OwnedBytes::Share hands the SAME bytes to a second viewer") {
  vllm::OwnedBytes a(std::vector<uint8_t>{9, 8, 7, 6, 5});
  const std::vector<uint8_t> want(a.begin(), a.end());
  REQUIRE_FALSE(a.borrowed());

  std::shared_ptr<const void> owner = a.Share();
  REQUIRE(a.borrowed());
  const vllm::OwnedBytes& ra = a;
  vllm::OwnedBytes b = vllm::OwnedBytes::Borrow(ra.data(), ra.size(), owner);
  const vllm::OwnedBytes& rb = b;

  CHECK(ra.data() == rb.data());  // one buffer, two viewers
  CHECK(a == b);
  REQUIRE(ra.size() == want.size());
  CHECK(std::memcmp(ra.data(), want.data(), want.size()) == 0);  // no copy

  owner.reset();
  a.Reset();  // the first viewer goes; the second still reads
  REQUIRE(rb.size() == want.size());
  CHECK(std::memcmp(rb.data(), want.data(), want.size()) == 0);
}

// ===========================================================================
// L6 — keep-f16 residency. An F16 file weight stays F16 (no bf16 re-expansion),
// resident in the file's own [N, K] order, consumed by the elementwise f16 GEMM.
// The residency is byte-lossless; the compute dtype changes bf16 -> f16, which
// the engine gate (test_qwen36_gguf_engine, same-file llama.cpp oracle) covers.
// These units gate the LOADER: routing, byte-identity, borrow/lifetime, sharing.
// ===========================================================================

namespace {

// F16 bit patterns for a torch [n, k] tensor, ggml dims reversed (ne0 = k).
std::string F16Bytes(int64_t n, float base) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 2);
  for (int64_t i = 0; i < n; ++i) {
    const float v = base + 0.25F * static_cast<float>(i % 7) -
                    0.125F * static_cast<float>(i % 3);
    const uint16_t h = vt::F32ToF16(v);
    s.push_back(static_cast<char>(h & 0xFF));
    s.push_back(static_cast<char>((h >> 8) & 0xFF));
  }
  return s;
}

void AddF16T(GgufModelBuilder& b, const std::string& name, int64_t out_dim,
             int64_t in_dim, float base) {
  b.AddTensor(name,
              {static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim)},
              kF16, F16Bytes(out_dim * in_dim, base));
}

// A tiny DENSE GGUF whose GEMM weights AND token_embd are F16 (norms stay F32) —
// the "56 f16 tensors" shape of the real 2B bench file, the case keep-f16 exists
// for. `tied` omits output.weight so the head IS the f16 token_embd.
std::string BuildDenseF16Gguf(const DenseDims& d, bool tied = false) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35"));
  b.AddKv(U32Kv("qwen35.embedding_length", static_cast<uint32_t>(d.H)));
  b.AddKv(U32Kv("qwen35.block_count", static_cast<uint32_t>(d.n_layer)));
  b.AddKv(U32Kv("qwen35.attention.head_count", static_cast<uint32_t>(d.n_head)));
  b.AddKv(U32Kv("qwen35.attention.head_count_kv",
                static_cast<uint32_t>(d.n_head_kv)));
  b.AddKv(U32Kv("qwen35.attention.key_length",
                static_cast<uint32_t>(d.head_dim)));
  b.AddKv(U32Kv("qwen35.feed_forward_length", static_cast<uint32_t>(d.I)));
  b.AddKv(F32Kv("qwen35.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen35.rope.freq_base", 1000000.0F));
  b.AddKv(U32Kv("qwen35.full_attention_interval", 1));
  b.AddKv(U32Kv("qwen35.context_length", 4096));

  AddF16T(b, "token_embd.weight", d.vocab, d.H, 0.5F);
  AddF32T(b, "output_norm.weight", {d.H}, 1.5F);
  if (!tied) AddF16T(b, "output.weight", d.vocab, d.H, 0.6F);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    AddF32T(b, p + "attn_norm.weight", {d.H}, 1.25F);
    AddF32T(b, p + "post_attention_norm.weight", {d.H}, 1.75F);
    AddF16T(b, p + "attn_q.weight", d.n_head * d.head_dim, d.H, 0.11F);
    AddF16T(b, p + "attn_k.weight", d.n_head_kv * d.head_dim, d.H, 0.13F);
    AddF16T(b, p + "attn_v.weight", d.n_head_kv * d.head_dim, d.H, 0.17F);
    AddF16T(b, p + "attn_output.weight", d.H, d.n_head * d.head_dim, 0.19F);
    AddF32T(b, p + "attn_q_norm.weight", {d.head_dim}, 1.5F);
    AddF32T(b, p + "attn_k_norm.weight", {d.head_dim}, 1.5F);
    AddF16T(b, p + "ffn_gate.weight", d.I, d.H, 0.21F);
    AddF16T(b, p + "ffn_up.weight", d.I, d.H, 0.23F);
    AddF16T(b, p + "ffn_down.weight", d.H, d.I, 0.27F);
  }
  return b.Build();
}

// keep-f16 forced ON, copy residency (mmap off) — the A/B analogue of KeepQuantOn.
GgufLoadPolicy KeepF16On() {
  GgufLoadPolicy p;
  p.keep_quant = true;   // GgufQuantComputeAvailable() would set this on CPU
  p.expand_nk = true;
  p.keep_f16 = true;
  return p;
}

}  // namespace

TEST_CASE("keep-f16 routing: F16 matmul/expert/embed keep, others expand") {
  const uint32_t f16 = kF16;
  // The keep-eligible roles + right rank keep f16 ONLY when keep_f16 is on.
  CHECK(RouteGgufTensor(false, true, false, GgufTensorRole::kMatmulWeight, f16,
                        {8, 64}) == GgufResidency::kKeepF16);
  CHECK(RouteGgufTensor(false, true, false, GgufTensorRole::kStackedExpertWeight,
                        f16, {2, 8, 64}) == GgufResidency::kKeepF16);
  // The embedding table keeps f16 too (a gather widens f16 like bf16); this is
  // what lets a tied token_embd/lm_head be ONE resident f16 vocab matrix.
  CHECK(RouteGgufTensor(false, true, false, GgufTensorRole::kEmbeddingTable, f16,
                        {32, 64}) == GgufResidency::kKeepF16);
  // Value/layout-rewritten, conv and vector roles NEVER keep, even f16.
  for (GgufTensorRole role :
       {GgufTensorRole::kTransformedWeight, GgufTensorRole::kConvWeight,
        GgufTensorRole::kVector}) {
    CAPTURE(vllm::Name(role));
    CHECK(RouteGgufTensor(false, true, false, role, f16, {8, 64}) ==
          GgufResidency::kExpandBf16);
  }
  // Wrong rank never keeps.
  CHECK(RouteGgufTensor(false, true, false, GgufTensorRole::kMatmulWeight, f16,
                        {64}) == GgufResidency::kExpandBf16);
  // keep_f16 OFF -> expand; VT_CPU_REF -> expand; a non-f16 type -> no keep-f16.
  CHECK(RouteGgufTensor(false, false, false, GgufTensorRole::kMatmulWeight, f16,
                        {8, 64}) == GgufResidency::kExpandBf16);
  CHECK(RouteGgufTensor(true, true, true, GgufTensorRole::kMatmulWeight, f16,
                        {8, 64}) == GgufResidency::kExpandBf16);
  CHECK(RouteGgufTensor(false, true, false, GgufTensorRole::kMatmulWeight, kQ8_0,
                        {8, 64}) == GgufResidency::kExpandBf16);
  // keep-quant WINS over keep-f16 when both are on and the encoding is a block
  // type (they are mutually exclusive by encoding, so this only asserts order).
  CHECK(RouteGgufTensor(true, true, false, GgufTensorRole::kMatmulWeight, kQ8_0,
                        {8, 64}) == GgufResidency::kKeepQuant);
}

TEST_CASE("OwnGgufF16 keeps the file's f16 bytes verbatim, [N,K] nk=true") {
  const int64_t n = 5, k = 64;
  const TempFile f([&] {
    GgufModelBuilder b;
    b.AddKv(StrKv("general.architecture", "test"));
    b.AddTensor("w", {static_cast<uint64_t>(k), static_cast<uint64_t>(n)}, kF16,
                F16Bytes(n * k, 0.3F));
    return b.Build();
  }());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufTensorInfo& t = g.Get("w");
  REQUIRE(t.nbytes == static_cast<size_t>(n) * k * 2);

  const vllm::OwnedTensor res = vllm::OwnGgufF16(t, n, k);
  CHECK(res.dtype == vt::DType::kF16);
  CHECK(res.rank == 2);
  CHECK(res.shape[0] == n);
  CHECK(res.shape[1] == k);
  CHECK(res.nk == true);
  REQUIRE(res.bytes.size() == t.nbytes);
  CHECK(std::memcmp(res.bytes.data(), t.data, t.nbytes) == 0);  // verbatim

  // nk=false builds a [vocab,H] gather table over the SAME bytes.
  const vllm::OwnedTensor emb = vllm::OwnGgufF16(t, n, k, 0, nullptr, false);
  CHECK(emb.nk == false);
  CHECK(std::memcmp(emb.bytes.data(), t.data, t.nbytes) == 0);

  // Guards: not f16, ragged span, bad slice.
  CHECK_THROWS(vllm::OwnGgufF16(t, n, k, /*row_offset=*/1));  // slice past span
  const std::string q = OneTensorGguf(kQ8_0, n, k, RandomBlockBytes(
      BlockBytesFor(kQ8_0, n * k), 7));
  const TempFile qf(q);
  const vllm::GgufFile qg = vllm::GgufFile::Open(qf.path());
  CHECK_THROWS(vllm::OwnGgufF16(qg.Get("w"), n, k));  // not f16
}

TEST_CASE("loader keeps F16 weights f16, byte-identical, and expands norms") {
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  GgufLoadPolicy on = KeepF16On();
  const vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &on);

  // The F16 GEMM weights STAY f16, in [N,K] nk=true, byte-identical to the file.
  const vllm::GgufTensorInfo& oh = g.Get("output.weight");
  CHECK(w.lm_head.dtype == vt::DType::kF16);
  CHECK(w.lm_head.nk == true);
  CHECK(std::memcmp(w.lm_head.bytes.data(), oh.data, oh.nbytes) == 0);
  for (int64_t il = 0; il < d.n_layer; ++il) {
    CAPTURE(il);
    const auto& la = w.layers[static_cast<size_t>(il)];
    for (const vllm::OwnedTensor* wt :
         {&la.attn.q_proj, &la.attn.o_proj, &la.mlp.gate_proj, &la.mlp.down_proj}) {
      CHECK(wt->dtype == vt::DType::kF16);
      CHECK(wt->nk == true);
    }
    const std::string p = "blk." + std::to_string(il) + ".";
    const vllm::GgufTensorInfo& gq = g.Get(p + "attn_q.weight");
    CHECK(std::memcmp(la.attn.q_proj.bytes.data(), gq.data, gq.nbytes) == 0);
    // Norms are F32 in the file and (w-1)-rewritten -> still bf16, never kept.
    CHECK(la.input_layernorm.dtype == vt::DType::kBF16);
  }
  // The embedding table (F16) is kept f16 as a [vocab,H] gather, nk=false.
  CHECK(w.embed_tokens.dtype == vt::DType::kF16);
  CHECK(w.embed_tokens.nk == false);
}

TEST_CASE("VT_CPU_REF / keep_f16=off reproduce the historical bf16 expansion") {
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // Oracle: everything expands to bf16, exactly as before L6.
  GgufLoadPolicy oracle;  // default-constructed = all-expand
  const vllm::Qwen3_5DenseWeights wo =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &oracle);
  CHECK(wo.lm_head.dtype == vt::DType::kBF16);
  CHECK(wo.embed_tokens.dtype == vt::DType::kBF16);
  CHECK(wo.layers[0].attn.q_proj.dtype == vt::DType::kBF16);

  // The kept-f16 weight carries the SAME information as the bf16 expansion, but
  // MORE precisely: dequantizing f16->bf16 from the resident f16 equals the
  // loader's bf16 expansion byte-for-byte (both round the file's f16 to bf16).
  GgufLoadPolicy on = KeepF16On();
  const vllm::Qwen3_5DenseWeights wk =
      vllm::LoadQwen3_5DenseFromGguf(g, c, &on);
  const vllm::GgufTensorInfo& gq = g.Get("blk.0.attn_q.weight");
  const int64_t numel = gq.shape[0] * gq.shape[1];
  const std::vector<uint16_t> bf_from_f16 =
      vllm::DequantGgufRowToBf16(kF16, wk.layers[0].attn.q_proj.bytes.data(),
                                 numel);
  // wo.q_proj is the bf16 expansion in [N,K] (expand_nk off here -> [K,N]); to
  // compare information not layout, expand the SAME file tensor to bf16 [N,K].
  const std::vector<uint16_t> bf_from_file =
      vllm::DequantGgufRowToBf16(kF16, gq.data, numel);
  CHECK(bf_from_f16 == bf_from_file);
}

TEST_CASE("keep-f16 mmap residency BORROWS in place; copy arm OWNS") {
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  GgufLoadPolicy copy = KeepF16On();       // mmap off
  GgufLoadPolicy mmap = KeepF16On();
  mmap.mmap_residency = true;
  const vllm::Qwen3_5DenseWeights wc = vllm::LoadQwen3_5DenseFromGguf(g, c, &copy);
  const vllm::Qwen3_5DenseWeights wm = vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap);

  const vllm::GgufTensorInfo& oh = g.Get("output.weight");
  CHECK(wc.lm_head.bytes.borrowed() == false);
  CHECK(wm.lm_head.bytes.borrowed() == true);
  CHECK(wm.lm_head.bytes == wc.lm_head.bytes);      // same bytes
  CHECK(wm.lm_head.bytes.data() == oh.data);        // IN PLACE, the file's own
  CHECK(wm.lm_head.dtype == vt::DType::kF16);
}

TEST_CASE("a borrowed F16 weight OUTLIVES the GgufFile and the file") {
  const DenseDims d;
  std::vector<uint8_t> expected;
  std::unique_ptr<vllm::Qwen3_5DenseWeights> w;

  {
    const TempFile f(BuildDenseF16Gguf(d));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
    GgufLoadPolicy mmap = KeepF16On();
    mmap.mmap_residency = true;
    w = std::make_unique<vllm::Qwen3_5DenseWeights>(
        vllm::LoadQwen3_5DenseFromGguf(g, c, &mmap));
    const vllm::GgufTensorInfo& t = g.Get("output.weight");
    expected.assign(t.data, t.data + t.nbytes);
  }

  REQUIRE(w->lm_head.bytes.borrowed());
  REQUIRE(w->lm_head.bytes.size() == expected.size());
  CHECK(std::memcmp(w->lm_head.bytes.data(), expected.data(),
                    expected.size()) == 0);
  w.reset();  // last reference to the refcounted mapping
}

TEST_CASE("tied F16 head SHARES one f16 vocab matrix (copy AND mmap)") {
  const DenseDims d;
  const TempFile f(BuildDenseF16Gguf(d, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  REQUIRE_FALSE(FileHasTensor(g, "output.weight"));

  for (bool use_mmap : {false, true}) {
    CAPTURE(use_mmap);
    GgufLoadPolicy shared = KeepF16On();
    shared.share_tied_head = true;
    shared.mmap_residency = use_mmap;
    const vllm::Qwen3_5DenseWeights ws =
        vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);

    // ONE f16 buffer serves both: the gather table [vocab,H] nk=false and the
    // GEMM head [N=vocab,K=H] nk=true, over the same bytes.
    CHECK(ws.embed_tokens.dtype == vt::DType::kF16);
    CHECK(ws.lm_head.dtype == vt::DType::kF16);
    CHECK(ws.embed_tokens.bytes.data() == ws.lm_head.bytes.data());
    CHECK(ws.embed_tokens.bytes == ws.lm_head.bytes);
    CHECK(ws.embed_tokens.nk == false);
    CHECK(ws.lm_head.nk == true);
    CHECK(ws.lm_head.shape[0] == d.vocab);
    CHECK(ws.lm_head.shape[1] == d.H);
    CHECK(ws.embed_tokens.bytes.borrowed());
    CHECK(ws.lm_head.bytes.borrowed());

    // The share is freed exactly once, either half first.
    {
      vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, c, &shared);
      const vllm::OwnedBytes& head = w.lm_head.bytes;
      const std::vector<uint8_t> want(head.begin(), head.end());
      w.embed_tokens.bytes.Reset();
      REQUIRE(head.size() == want.size());
      CHECK(std::memcmp(head.data(), want.data(), want.size()) == 0);
    }
  }
}
