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
  CHECK(RouteGgufTensor(true, false, GgufTensorRole::kMatmulWeight, kQ8_0,
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

        CHECK(RouteGgufTensor(/*keep_quant=*/true, /*cpu_ref=*/false, role,
                              type, shape) == expected);
        (expect_keep ? kept : expanded)++;

        // The master switch OFF expands everything, always.
        CHECK(RouteGgufTensor(/*keep_quant=*/false, /*cpu_ref=*/false, role,
                              type, shape) == GgufResidency::kExpandBf16);
        // The VT_CPU_REF oracle wins over keep-quant, always.
        CHECK(RouteGgufTensor(/*keep_quant=*/true, /*cpu_ref=*/true, role, type,
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
      CHECK(RouteGgufTensor(true, false, role, e.ggml_type, {8, e.k}) ==
            GgufResidency::kExpandBf16);
      CHECK(RouteGgufTensor(true, false, role, e.ggml_type, {2, 8, e.k}) ==
            GgufResidency::kExpandBf16);
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
    CHECK_FALSE(p.cpu_ref);
  }
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  CHECK(GgufLoadPolicy::FromEnv().keep_quant);
  CHECK(GgufLoadPolicy::FromEnv().expand_nk);
  // The OPT-OUT the spec promised must survive the default flip.
  for (const char* off : {"0", "false", "off", ""}) {
    ::setenv("VT_GGUF_KEEP_QUANT", off, 1);
    CAPTURE(off);
    CHECK_FALSE(GgufLoadPolicy::FromEnv().keep_quant);
    CHECK_FALSE(GgufLoadPolicy::FromEnv().expand_nk);
  }
  ::setenv("VT_GGUF_KEEP_QUANT", "1", 1);
  ::setenv("VT_CPU_REF", "1", 1);
  {
    // The oracle switch: keep-quant requested, oracle wins — and it takes the
    // orientation with it, so VT_CPU_REF=1 is the FULL historical load.
    const GgufLoadPolicy p = GgufLoadPolicy::FromEnv();
    CHECK(p.keep_quant);
    CHECK(p.cpu_ref);
    CHECK_FALSE(p.expand_nk);
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
std::string BuildDenseQ8Gguf(const DenseDims& d) {
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
  AddQ8_0(b, "output.weight", d.vocab, d.H, 11);
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

  GgufLoadPolicy expect;
  expect.keep_quant = vllm::GgufQuantComputeAvailable();
  expect.expand_nk = expect.keep_quant;

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
