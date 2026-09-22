// Test for the qwen3 GGUF loader (`Qwen3HfConfigFromGguf`, `IsQwen3Gguf`,
// `LoadQwen3FromGguf`). Builds a tiny synthetic Qwen3 GGUF in-memory with F32
// tensors and verifies config parsing and weight layout: merged QKV concat
// (rows q|k|v), merged gate/up concat (rows gate|up), per-head q/k norm
// presence, tied/untied lm_head, and the all-bf16 expand path on CPU.
//
// Mirrors test_laguna_gguf_load.cpp's pattern: synthetic GGUF via
// gguf_builder.h, direct loader calls, shape/dtype/nk assertions. No model
// file is needed, so the test runs on every CPU CI node.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_gguf_weights.h"
#include "vt/dtype.h"

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

namespace {

// Tiny geometry. inter != H so gate/down transposed shapes are unambiguous.
struct Dims {
  int64_t H = 64, vocab = 16, head_dim = 32;
  int64_t n_heads = 2, n_kv_heads = 1;  // GQA
  int64_t n_layer = 2, inter = 96;
  int64_t context_len = 256;
  float rope_theta = 1000000.0F;
  float rms_eps = 1e-6F;
};

int64_t Prod(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

std::vector<uint64_t> GgmlDims(const std::vector<int64_t>& torch_shape) {
  std::vector<uint64_t> d;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it)
    d.push_back(static_cast<uint64_t>(*it));
  return d;
}

float WFill(int64_t i) { return 0.05F * static_cast<float>((i % 13) - 6); }

std::string F32Data(int64_t n, int64_t offset = 0) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float v = WFill(offset + i);
    uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    for (int k = 0; k < 4; ++k) s.push_back(static_cast<char>((bits >> (8 * k)) & 0xff));
  }
  return s;
}

std::string Blk(int64_t l, const std::string& s) {
  return "blk." + std::to_string(l) + "." + s;
}

void AddF32(GgufModelBuilder& b, const std::string& name,
             const std::vector<int64_t>& shape, int64_t offset = 0) {
  b.AddTensor(name, GgmlDims(shape), /*F32=*/0, F32Data(Prod(shape), offset));
}

// Per-tensor fill offsets. Tensors that share a shape (k/v, gate/up) get
// distinct seeds so the merged concat order is provable: each component's
// row-0 element is WFill(offset), and a k<->v or gate<->up swap moves the
// wrong seed into a checked row and fails.
const int64_t kQOff = 0, kKOff = 1, kVOff = 2;
const int64_t kGateOff = 3, kUpOff = 4;

std::string BuildGguf(const Dims& d, bool tied, bool with_qk_norm) {
  GgufModelBuilder b;
  const std::string p = "qwen3.";
  b.AddKv(StrKv("general.architecture", "qwen3"));
  b.AddKv(U32Kv(p + "embedding_length", d.H));
  b.AddKv(U32Kv(p + "block_count", d.n_layer));
  b.AddKv(U32Kv(p + "attention.head_count", d.n_heads));
  b.AddKv(U32Kv(p + "attention.head_count_kv", d.n_kv_heads));
  b.AddKv(U32Kv(p + "attention.key_length", d.head_dim));
  b.AddKv(U32Kv(p + "feed_forward_length", d.inter));
  b.AddKv(U32Kv(p + "vocab_size", d.vocab));
  b.AddKv(U32Kv(p + "context_length", d.context_len));
  b.AddKv(U32Kv(p + "rope.dimension_count", d.head_dim));
  b.AddKv(F32Kv(p + "rope.freq_base", d.rope_theta));
  b.AddKv(F32Kv(p + "attention.layer_norm_rms_epsilon", d.rms_eps));

  const int64_t q_rows = d.n_heads * d.head_dim;
  const int64_t kv_rows = d.n_kv_heads * d.head_dim;

  AddF32(b, "token_embd.weight", {d.vocab, d.H});
  AddF32(b, "output_norm.weight", {d.H});
  if (!tied) AddF32(b, "output.weight", {d.vocab, d.H});

  for (int64_t l = 0; l < d.n_layer; ++l) {
    AddF32(b, Blk(l, "attn_norm.weight"), {d.H});
    AddF32(b, Blk(l, "post_attention_norm.weight"), {d.H});
    AddF32(b, Blk(l, "attn_q.weight"), {q_rows, d.H}, kQOff);
    AddF32(b, Blk(l, "attn_k.weight"), {kv_rows, d.H}, kKOff);
    AddF32(b, Blk(l, "attn_v.weight"), {kv_rows, d.H}, kVOff);
    AddF32(b, Blk(l, "attn_output.weight"), {d.H, q_rows});
    if (with_qk_norm) {
      AddF32(b, Blk(l, "attn_q_norm.weight"), {d.head_dim});
      AddF32(b, Blk(l, "attn_k_norm.weight"), {d.head_dim});
    }
    AddF32(b, Blk(l, "ffn_gate.weight"), {d.inter, d.H}, kGateOff);
    AddF32(b, Blk(l, "ffn_up.weight"), {d.inter, d.H}, kUpOff);
    AddF32(b, Blk(l, "ffn_down.weight"), {d.H, d.inter});
  }
  return b.Build();
}

// Default-constructed policy: all-expand to bf16 (CPU default).
const vllm::GgufLoadPolicy kExpandAll;

}  // namespace

// ── config ─────────────────────────────────────────────────────────────────

TEST_CASE("Qwen3HfConfigFromGguf: parses all metadata (untied)") {
  Dims d;
  TempFile f(BuildGguf(d, /*tied=*/false, /*with_qk_norm=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::Qwen3HfConfigFromGguf(g);

  CHECK(c.model_type == "qwen3");
  REQUIRE(c.architectures.size() == 1);
  CHECK(c.architectures[0] == "Qwen3ForCausalLM");
  CHECK(c.hidden_size == d.H);
  CHECK(c.num_hidden_layers == d.n_layer);
  CHECK(c.num_attention_heads == d.n_heads);
  CHECK(c.num_key_value_heads == d.n_kv_heads);
  CHECK(c.head_dim == d.head_dim);
  CHECK(c.intermediate_size == d.inter);
  CHECK(c.vocab_size == d.vocab);
  CHECK(c.rope_theta == doctest::Approx(d.rope_theta));
  CHECK(c.rotary_dim == d.head_dim);
  CHECK(c.rms_norm_eps == doctest::Approx(d.rms_eps));
  CHECK(c.max_position_embeddings == d.context_len);
  CHECK(c.torch_dtype == "bfloat16");
  CHECK(c.raw.value("tie_word_embeddings", true) == false);
  CHECK(c.raw.value("attention_bias", true) == false);
  CHECK(vllm::IsQwen3Gguf(g));
}

TEST_CASE("Qwen3HfConfigFromGguf: tied (no output.weight)") {
  Dims d;
  TempFile f(BuildGguf(d, /*tied=*/true, /*with_qk_norm=*/false));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::Qwen3HfConfigFromGguf(g);
  CHECK(c.raw.value("tie_word_embeddings", false) == true);
}

TEST_CASE("Qwen3HfConfigFromGguf: rejects wrong architecture") {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "llama"));
  b.AddKv(U32Kv("qwen3.embedding_length", 64));
  TempFile f(b.Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  CHECK_THROWS(vllm::Qwen3HfConfigFromGguf(g));
  CHECK_FALSE(vllm::IsQwen3Gguf(g));
}

// ── weights ───────────────────────────────────────────────────────────────

TEST_CASE("LoadQwen3FromGguf: shapes, dtypes, nk, QKV/gate-up concat") {
  Dims d;
  TempFile f(BuildGguf(d, /*tied=*/false, /*with_qk_norm=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::Qwen3HfConfigFromGguf(g);
  const vllm::Qwen3DenseWeights w =
      vllm::LoadQwen3FromGguf(g, c, &kExpandAll);

  const int64_t q_rows = d.n_heads * d.head_dim;
  const int64_t kv_rows = d.n_kv_heads * d.head_dim;

  // Embedding: bf16 [vocab, H], not nk (gather table, not a GEMM weight).
  CHECK(w.embed_tokens.dtype == vt::DType::kBF16);
  REQUIRE(w.embed_tokens.rank == 2);
  CHECK(w.embed_tokens.shape[0] == d.vocab);
  CHECK(w.embed_tokens.shape[1] == d.H);
  CHECK_FALSE(w.embed_tokens.nk);

  // Final norm: bf16 [H].
  CHECK(w.final_norm.dtype == vt::DType::kBF16);
  REQUIRE(w.final_norm.rank == 1);
  CHECK(w.final_norm.shape[0] == d.H);

  // lm_head: untied → non-empty, bf16, Matmul-B [H, vocab] (transposed).
  CHECK_FALSE(w.lm_head.Empty());
  CHECK(w.lm_head.dtype == vt::DType::kBF16);
  REQUIRE(w.lm_head.rank == 2);
  CHECK(w.lm_head.shape[0] == d.H);
  CHECK(w.lm_head.shape[1] == d.vocab);
  CHECK_FALSE(w.lm_head.nk);

  REQUIRE(w.layers.size() == static_cast<size_t>(d.n_layer));
  for (int64_t l = 0; l < d.n_layer; ++l) {
    const auto& ly = w.layers[static_cast<size_t>(l)];
    INFO("layer", l);

    // Norms: bf16 [H], nk=false (OwnBf16, no w+1 shift).
    CHECK(ly.input_layernorm.dtype == vt::DType::kBF16);
    REQUIRE(ly.input_layernorm.rank == 1);
    CHECK(ly.input_layernorm.shape[0] == d.H);
    CHECK_FALSE(ly.input_layernorm.nk);
    CHECK(ly.post_attention_layernorm.dtype == vt::DType::kBF16);
    CHECK(ly.post_attention_layernorm.shape[0] == d.H);

    // QKV merged: bf16 [q + k + v rows, H], nk=true.
    CHECK(ly.attn.qkv_proj.dtype == vt::DType::kBF16);
    REQUIRE(ly.attn.qkv_proj.rank == 2);
    CHECK(ly.attn.qkv_proj.shape[0] == q_rows + 2 * kv_rows);
    CHECK(ly.attn.qkv_proj.shape[1] == d.H);
    CHECK(ly.attn.qkv_proj.nk);

    // o_proj: bf16 Matmul-B [q_rows, H] (transposed from [H, q_rows]).
    CHECK(ly.attn.o_proj.dtype == vt::DType::kBF16);
    REQUIRE(ly.attn.o_proj.rank == 2);
    CHECK(ly.attn.o_proj.shape[0] == q_rows);
    CHECK(ly.attn.o_proj.shape[1] == d.H);
    CHECK_FALSE(ly.attn.o_proj.nk);

    // q_norm / k_norm: present → bf16 [head_dim].
    CHECK_FALSE(ly.attn.q_norm.Empty());
    CHECK(ly.attn.q_norm.dtype == vt::DType::kBF16);
    CHECK(ly.attn.q_norm.shape[0] == d.head_dim);
    CHECK_FALSE(ly.attn.k_norm.Empty());
    CHECK(ly.attn.k_norm.dtype == vt::DType::kBF16);
    CHECK(ly.attn.k_norm.shape[0] == d.head_dim);

    // gate_up merged: bf16 [2*inter, H], nk=true.
    CHECK(ly.mlp.gate_up_proj.dtype == vt::DType::kBF16);
    REQUIRE(ly.mlp.gate_up_proj.rank == 2);
    CHECK(ly.mlp.gate_up_proj.shape[0] == 2 * d.inter);
    CHECK(ly.mlp.gate_up_proj.shape[1] == d.H);
    CHECK(ly.mlp.gate_up_proj.nk);

    // down_proj: bf16 Matmul-B [inter, H] (transposed from [H, inter]).
    CHECK(ly.mlp.down_proj.dtype == vt::DType::kBF16);
    REQUIRE(ly.mlp.down_proj.rank == 2);
    CHECK(ly.mlp.down_proj.shape[0] == d.inter);
    CHECK(ly.mlp.down_proj.shape[1] == d.H);
    CHECK_FALSE(ly.mlp.down_proj.nk);
  }

  // Norm value spot-check: input_layernorm[0] == F32ToBF16(WFill(0)).
  {
    const auto& norm = w.layers[0].input_layernorm;
    const uint16_t* bf = reinterpret_cast<const uint16_t*>(norm.bytes.data());
    // BF16 has ~3 decimal digits of precision, so allow 1% relative error.
    CHECK(vt::BF16ToF32(bf[0]) == doctest::Approx(WFill(0)).epsilon(0.01));
  }

  // QKV concat: each component has a distinct fill offset, so the row-0
  // element of each block carries its own seed. Verifying the actual seeds at
  // each boundary proves q|k|v order: a k<->v swap moves the wrong seed into the
  // checked row and fails.
  {
    const auto& qkv = w.layers[0].attn.qkv_proj;
    const uint16_t* bf = reinterpret_cast<const uint16_t*>(qkv.bytes.data());
    const int64_t cols = d.H;
    // Row 0 is q (kQOff), row q_rows is k (kKOff), row q_rows+kv_rows is v.
    CHECK(vt::BF16ToF32(bf[0]) == doctest::Approx(WFill(kQOff)).epsilon(0.01));
    CHECK(vt::BF16ToF32(bf[q_rows * cols]) ==
          doctest::Approx(WFill(kKOff)).epsilon(0.01));
    CHECK(vt::BF16ToF32(bf[(q_rows + kv_rows) * cols]) ==
          doctest::Approx(WFill(kVOff)).epsilon(0.01));
    // Adjacent rows within a component differ (fill advances by cols).
    CHECK(bf[cols] != bf[0]);
    // The three component seeds are mutually distinct.
    CHECK(bf[0] != bf[q_rows * cols]);
    CHECK(bf[0] != bf[(q_rows + kv_rows) * cols]);
    CHECK(bf[q_rows * cols] != bf[(q_rows + kv_rows) * cols]);
  }

  // gate/up concat: gate and up have distinct fill offsets, so the row-0
  // element of each block carries its own seed. Verifying the actual seeds
  // proves gate|up order: a swap moves the wrong seed and fails.
  {
    const auto& gu = w.layers[0].mlp.gate_up_proj;
    const uint16_t* bf = reinterpret_cast<const uint16_t*>(gu.bytes.data());
    const int64_t cols = d.H;
    // Row 0 is gate (kGateOff), row d.inter is up (kUpOff).
    CHECK(vt::BF16ToF32(bf[0]) ==
          doctest::Approx(WFill(kGateOff)).epsilon(0.01));
    CHECK(vt::BF16ToF32(bf[d.inter * cols]) ==
          doctest::Approx(WFill(kUpOff)).epsilon(0.01));
    // Adjacent rows within a component differ.
    CHECK(bf[cols] != bf[0]);
    // The two component seeds are distinct.
    CHECK(bf[0] != bf[d.inter * cols]);
  }
}

TEST_CASE("LoadQwen3FromGguf: tied + no qk_norm") {
  Dims d;
  TempFile f(BuildGguf(d, /*tied=*/true, /*with_qk_norm=*/false));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::Qwen3HfConfigFromGguf(g);
  const vllm::Qwen3DenseWeights w =
      vllm::LoadQwen3FromGguf(g, c, &kExpandAll);

  // Tied: tie_word_embeddings is set; the forward path aliases embed_tokens.
  CHECK(w.tie_word_embeddings);
  // No qk_norm tensors: q_norm and k_norm are empty on every layer.
  for (const auto& ly : w.layers) {
    CHECK(ly.attn.q_norm.Empty());
    CHECK(ly.attn.k_norm.Empty());
  }
}
