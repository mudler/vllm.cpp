// Muse Glimmer GGUF k-quant gate (`.agents/porting-a-model.md` §2, "Weight
// formats — all of them"). Muse Glimmer previously REFUSED GGUF outright, which
// was never a decision; this is the gate for the arm that closes it.
//
// What each case proves, and what none of them prove:
//
//   (1) CONFIG-FROM-GGUF. A `muse-glimmer` file's metadata descends into the
//       SAME `MuseGlimmerParams` a safetensors `config.json` produces, including
//       the iRoPE mask read off `attention.sliding_window_pattern` and the
//       untied-head decision read off the presence of `output.weight`.
//   (2) THE NAME MAP, both directions. Canonical -> GGUF for every enumerated
//       tensor, and the GGUF-only `attn_{q,k}_norm` pair accounted explicitly
//       rather than tolerated as strangers.
//   (3) THE THREE CONVERT-TIME TRANSFORMS, each asserted against the value the
//       bug would produce: the sandwich norms un-shifted by ONE (and the final
//       norm NOT), the query pre-scale RECOVERED from the folded `attn_q_norm`
//       (with a non-constant q-norm and a non-ones k-norm both REFUSED), and
//       the iRoPE mask.
//   (4) STRUCTURAL ACCOUNTING against the REAL released k-quant, both
//       directions, zero unaccounted — off a COMMITTED header-only manifest, so
//       CI never needs the 16.76 GB asset. A second case re-reads the live file
//       when VLLM_MUSE_GGUF points at it.
//   (5) The mmproj perception encoder REFUSES BY NAME, and the refusal names the
//       missing piece (the `patch_temporal` axis of `patch_embd`), proven
//       against the real mmproj manifest rather than asserted in prose.
//   (6) The `dflash` drafter's manifest is fully covered by the EXISTING
//       `qwen3_dflash_gguf` name map — a structural reachability claim only.
//
// NOT ESTABLISHED HERE: no forward, no e2e, no token-exactness, and NO SPEED
// AXIS AT ALL. The pinned oracle cannot load `muse_glimmer` in any weight format
// (.agents/specs/muse-glimmer.md §0), so there is no denominator to compare
// against and none is claimed.
#include "vllm/model_executor/models/muse_glimmer_gguf_weights.h"

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/muse_glimmer.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../gguf_builder.h"
#include "muse_glimmer_dflash_gguf_manifest.inc"
#include "muse_glimmer_gguf_manifest.inc"
#include "muse_glimmer_mmproj_gguf_manifest.inc"

namespace {

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

// ── a tiny but STRUCTURALLY COMPLETE synthetic muse-glimmer GGUF ─────────────
// Reduced geometry (4 layers, hidden 8, 2 q-heads / 1 kv-head, head_dim 4,
// vocab 6, intermediate 16) so the whole file is a few KB and the test needs no
// asset. Every tensor the real file ships is present, with the real file's
// dtypes for the small tensors (F32 norms) and F32 for the matmul operands, so
// the loader's name map, shapes, un-shift and pre-scale recovery all run.
struct SyntheticMuse {
  static constexpr int64_t kLayers = 4;
  static constexpr int64_t kHidden = 8;
  static constexpr int64_t kHeads = 2;
  static constexpr int64_t kKvHeads = 1;
  static constexpr int64_t kHeadDim = 4;
  static constexpr int64_t kVocab = 6;
  static constexpr int64_t kInter = 16;
  static constexpr float kQueryPreScale = 3.87f;

  // ggml type ids.
  static constexpr uint32_t kF32 = 0;

  // Deterministic filler so a mis-sliced tensor is visible.
  static float Fill(int64_t seed, int64_t i) {
    return 0.125f * static_cast<float>((seed * 31 + i * 7) % 17) - 1.0f;
  }

  static std::string F32Data(int64_t n, int64_t seed) {
    std::string s(static_cast<size_t>(n) * 4, '\0');
    for (int64_t i = 0; i < n; ++i) {
      const float v = Fill(seed, i);
      std::memcpy(s.data() + static_cast<size_t>(i) * 4, &v, 4);
    }
    return s;
  }

  static std::string ConstData(int64_t n, float v) {
    std::string s(static_cast<size_t>(n) * 4, '\0');
    for (int64_t i = 0; i < n; ++i)
      std::memcpy(s.data() + static_cast<size_t>(i) * 4, &v, 4);
    return s;
  }

  // `dims` are ggml ne order (inner dim FIRST), i.e. reversed torch.
  static void AddF32(GgufModelBuilder& b, const std::string& name,
                     const std::vector<uint64_t>& ne, int64_t seed) {
    int64_t n = 1;
    for (uint64_t d : ne) n *= static_cast<int64_t>(d);
    b.AddTensor(name, ne, kF32, F32Data(n, seed));
  }

  // `q_norm_scale` / `k_norm_scale` are the MUTATION handles: the real converter
  // folds the query pre-scale into attn_q_norm and leaves attn_k_norm at ones.
  static std::string Build(float q_norm_scale = kQueryPreScale,
                           float k_norm_scale = 1.0f,
                           bool q_norm_constant = true, bool untied = true) {
    GgufModelBuilder b;
    const std::string a = "muse-glimmer.";
    b.AddKv(StrKv("general.architecture", "muse-glimmer"));
    b.AddKv(U32Kv(a + "block_count", static_cast<uint32_t>(kLayers)));
    b.AddKv(U32Kv(a + "context_length", 4096));
    b.AddKv(U32Kv(a + "embedding_length", static_cast<uint32_t>(kHidden)));
    b.AddKv(U32Kv(a + "feed_forward_length", static_cast<uint32_t>(kInter)));
    b.AddKv(U32Kv(a + "attention.head_count", static_cast<uint32_t>(kHeads)));
    b.AddKv(U32Kv(a + "attention.head_count_kv", static_cast<uint32_t>(kKvHeads)));
    b.AddKv(U32Kv(a + "attention.key_length", static_cast<uint32_t>(kHeadDim)));
    b.AddKv(U32Kv(a + "attention.value_length", static_cast<uint32_t>(kHeadDim)));
    b.AddKv(F32Kv(a + "rope.freq_base", 500000.0f));
    b.AddKv(F32Kv(a + "attention.layer_norm_rms_epsilon", 1e-5f));
    b.AddKv(F32Kv(a + "final_logit_softcapping", 20.0f));
    b.AddKv(F32Kv(a + "logit_scale", 0.19611613f));
    b.AddKv(U32Kv(a + "attention.sliding_window", 2048));
    // The released 30B pattern, truncated: NoPE (false) every 4th layer.
    b.AddKv(gguf_test::BoolArrayKv(a + "attention.sliding_window_pattern",
                                   {true, true, true, false}));

    b.AddTensor("token_embd.weight", {kHidden, kVocab}, kF32,
                F32Data(kHidden * kVocab, 1));
    for (int64_t l = 0; l < kLayers; ++l) {
      const std::string p = "blk." + std::to_string(l) + ".";
      // Sandwich norms, stored PRE-OFFSET by the converter (w_hf + 1).
      AddF32(b, p + "attn_norm.weight", {kHidden}, 10 + l);
      AddF32(b, p + "post_attention_norm.weight", {kHidden}, 20 + l);
      AddF32(b, p + "ffn_norm.weight", {kHidden}, 30 + l);
      AddF32(b, p + "post_ffw_norm.weight", {kHidden}, 40 + l);
      // The FOLDED weightless QK-norms.
      if (q_norm_constant) {
        b.AddTensor(p + "attn_q_norm.weight", {kHeadDim}, kF32,
                    ConstData(kHeadDim, q_norm_scale));
      } else {
        AddF32(b, p + "attn_q_norm.weight", {kHeadDim}, 99);
      }
      b.AddTensor(p + "attn_k_norm.weight", {kHeadDim}, kF32,
                  ConstData(kHeadDim, k_norm_scale));
      AddF32(b, p + "attn_q.weight", {kHidden, kHeads * kHeadDim}, 50 + l);
      AddF32(b, p + "attn_k.weight", {kHidden, kKvHeads * kHeadDim}, 60 + l);
      AddF32(b, p + "attn_v.weight", {kHidden, kKvHeads * kHeadDim}, 70 + l);
      AddF32(b, p + "attn_output.weight", {kHeads * kHeadDim, kHidden}, 80 + l);
      AddF32(b, p + "attn_gate.weight", {kHidden, kHeads * kHeadDim}, 90 + l);
      AddF32(b, p + "ffn_gate.weight", {kHidden, kInter}, 100 + l);
      AddF32(b, p + "ffn_up.weight", {kHidden, kInter}, 110 + l);
      AddF32(b, p + "ffn_down.weight", {kInter, kHidden}, 120 + l);
    }
    AddF32(b, "output_norm.weight", {kHidden}, 200);
    if (untied) AddF32(b, "output.weight", {kHidden, kVocab}, 201);
    return b.Build();
  }
};

float Bf16ToF32Bits(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

std::vector<float> OwnedBf16ToF32(const vllm::OwnedTensor& t) {
  const auto* src = reinterpret_cast<const uint16_t*>(t.bytes.data());
  const int64_t n = t.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    out[static_cast<size_t>(i)] = Bf16ToF32Bits(src[static_cast<size_t>(i)]);
  return out;
}

// The converter's Q/K RoPE row permutation (transform 4), reproduced INDEPENDENTLY
// of the loader from llama.cpp's own `LlamaModel.permute`
// (`conversion/llama.py:163-169`) so a failure names the transform rather than
// restating the implementation:
//     gguf = w.reshape(heads, 2, head_dim/2, K).swapaxes(1, 2).reshape(w.shape)
// Returns, for a DESTINATION (HF-order) row, the SOURCE row in the GGUF.
int64_t QkSrcRow(int64_t dst_row, int64_t heads, int64_t rows) {
  const int64_t head_dim = rows / heads;
  const int64_t half = head_dim / 2;
  const int64_t h = dst_row / head_dim;
  const int64_t i = dst_row % head_dim;
  return h * head_dim + (i % half) * 2 + i / half;
}

// ── a SECOND synthetic file, geometry chosen so the qkv trio can stay QUANTIZED ─
// The released 17 GB k-quant's qkv trio is heterogeneous (Q6_K `attn_v` against
// Q4_K `attn_q`/`attn_k`), so it always takes `LoadMerged`'s DEQUANTIZED branch.
// A homogeneous trio takes the KEPT-QUANT block-concat branch instead, and the
// Q/K un-permute has to happen there too — on whole rows, which is byte-safe
// because a quantized row is a whole number of blocks. This file exists to reach
// that branch: hidden 64 = two Q8_0 blocks per row, one layer, everything else F32.
struct SyntheticMuseQ8 {
  static constexpr int64_t kLayers = 1;
  static constexpr int64_t kHidden = 64;
  static constexpr int64_t kHeads = 2;
  static constexpr int64_t kKvHeads = 1;
  static constexpr int64_t kHeadDim = 32;
  static constexpr int64_t kVocab = 4;
  static constexpr int64_t kInter = 64;
  static constexpr uint32_t kF32 = 0;
  static constexpr uint32_t kQ8_0 = 8;

  // block_q8_0 = { fp16 d; int8 qs[32] }. `d` is fixed at 1.0 so a dequantized
  // element is EXACTLY the stored int8 and a permuted row is unmistakable.
  static int8_t Q(int64_t row, int64_t col) {
    return static_cast<int8_t>((row * 13 + col * 5) % 101 - 50);
  }
  static std::string Q8Data(int64_t rows, int64_t cols) {
    std::string s;
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t b = 0; b < cols / 32; ++b) {
        s += std::string("\x00\x3c", 2);  // fp16 1.0, little-endian
        for (int64_t i = 0; i < 32; ++i)
          s.push_back(static_cast<char>(Q(r, b * 32 + i)));
      }
    return s;
  }

  static std::string Build() {
    GgufModelBuilder b;
    const std::string a = "muse-glimmer.";
    b.AddKv(StrKv("general.architecture", "muse-glimmer"));
    b.AddKv(U32Kv(a + "block_count", static_cast<uint32_t>(kLayers)));
    b.AddKv(U32Kv(a + "context_length", 4096));
    b.AddKv(U32Kv(a + "embedding_length", static_cast<uint32_t>(kHidden)));
    b.AddKv(U32Kv(a + "feed_forward_length", static_cast<uint32_t>(kInter)));
    b.AddKv(U32Kv(a + "attention.head_count", static_cast<uint32_t>(kHeads)));
    b.AddKv(U32Kv(a + "attention.head_count_kv", static_cast<uint32_t>(kKvHeads)));
    b.AddKv(U32Kv(a + "attention.key_length", static_cast<uint32_t>(kHeadDim)));
    b.AddKv(U32Kv(a + "attention.value_length", static_cast<uint32_t>(kHeadDim)));
    b.AddKv(F32Kv(a + "rope.freq_base", 500000.0f));
    b.AddKv(F32Kv(a + "attention.layer_norm_rms_epsilon", 1e-5f));
    b.AddKv(U32Kv(a + "attention.sliding_window", 2048));
    b.AddKv(gguf_test::BoolArrayKv(a + "attention.sliding_window_pattern", {true}));

    SyntheticMuse::AddF32(b, "token_embd.weight", {kHidden, kVocab}, 1);
    const std::string p = "blk.0.";
    SyntheticMuse::AddF32(b, p + "attn_norm.weight", {kHidden}, 10);
    SyntheticMuse::AddF32(b, p + "post_attention_norm.weight", {kHidden}, 20);
    SyntheticMuse::AddF32(b, p + "ffn_norm.weight", {kHidden}, 30);
    SyntheticMuse::AddF32(b, p + "post_ffw_norm.weight", {kHidden}, 40);
    b.AddTensor(p + "attn_q_norm.weight", {kHeadDim}, kF32,
                SyntheticMuse::ConstData(kHeadDim, SyntheticMuse::kQueryPreScale));
    b.AddTensor(p + "attn_k_norm.weight", {kHeadDim}, kF32,
                SyntheticMuse::ConstData(kHeadDim, 1.0f));
    // The homogeneous Q8_0 trio — the whole point of this file.
    b.AddTensor(p + "attn_q.weight", {kHidden, kHeads * kHeadDim}, kQ8_0,
                Q8Data(kHeads * kHeadDim, kHidden));
    b.AddTensor(p + "attn_k.weight", {kHidden, kKvHeads * kHeadDim}, kQ8_0,
                Q8Data(kKvHeads * kHeadDim, kHidden));
    b.AddTensor(p + "attn_v.weight", {kHidden, kKvHeads * kHeadDim}, kQ8_0,
                Q8Data(kKvHeads * kHeadDim, kHidden));
    SyntheticMuse::AddF32(b, p + "attn_output.weight", {kHeads * kHeadDim, kHidden}, 80);
    SyntheticMuse::AddF32(b, p + "attn_gate.weight", {kHidden, kHeads * kHeadDim}, 90);
    SyntheticMuse::AddF32(b, p + "ffn_gate.weight", {kHidden, kInter}, 100);
    SyntheticMuse::AddF32(b, p + "ffn_up.weight", {kHidden, kInter}, 110);
    SyntheticMuse::AddF32(b, p + "ffn_down.weight", {kInter, kHidden}, 120);
    SyntheticMuse::AddF32(b, "output_norm.weight", {kHidden}, 200);
    SyntheticMuse::AddF32(b, "output.weight", {kHidden, kVocab}, 201);
    return b.Build();
  }
};

// The released 30B geometry, so the manifest cases enumerate what the real file
// must contain without needing the file.
vllm::MuseGlimmerParams Released30BParams() {
  vllm::MuseGlimmerParams p;
  vllm::MuseGlimmerTextParams& t = p.text;
  t.vocab_size = 202048;
  t.hidden_size = 6656;
  t.intermediate_size = 19968;
  t.num_hidden_layers = 52;
  t.num_attention_heads = 32;
  t.num_key_value_heads = 2;
  t.head_dim = 128;
  t.max_position_embeddings = 131072;
  t.sliding_window = 2048;
  t.rope_theta = 500000.0;
  t.tie_word_embeddings = false;
  t.use_qk_norm = true;
  t.use_attn_output_gate = true;
  t.scale_query_by = 3.87;
  t.no_rope_layers = vllm::DefaultMuseGlimmerNoRopeLayers(52);
  return p;
}

}  // namespace

TEST_CASE("muse glimmer gguf: config descends from the file's own metadata") {
  const TempFile f(SyntheticMuse::Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  CHECK(vllm::IsMuseGlimmerGguf(g));

  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  REQUIRE(!cfg.architectures.empty());
  // The registry must RESOLVE what the GGUF config announces, or a k-quant file
  // is unloadable no matter how correct the weights are.
  CHECK_NOTHROW(vllm::ModelRegistry::Resolve(cfg));

  const vllm::MuseGlimmerParams p = vllm::ParseMuseGlimmerParams(cfg);
  CHECK(p.text.num_hidden_layers == SyntheticMuse::kLayers);
  CHECK(p.text.hidden_size == SyntheticMuse::kHidden);
  CHECK(p.text.intermediate_size == SyntheticMuse::kInter);
  CHECK(p.text.num_attention_heads == SyntheticMuse::kHeads);
  CHECK(p.text.num_key_value_heads == SyntheticMuse::kKvHeads);
  CHECK(p.text.head_dim == SyntheticMuse::kHeadDim);
  CHECK(p.text.vocab_size == SyntheticMuse::kVocab);
  CHECK(p.text.sliding_window == 2048);
  CHECK(p.text.rope_theta == doctest::Approx(500000.0));
  CHECK(p.text.final_logit_softcapping == doctest::Approx(20.0));
  CHECK(p.text.output_multiplier == doctest::Approx(0.19611613));
  CHECK(p.text.tie_word_embeddings == false);
  // Transform 3: `sliding_window_pattern` IS the iRoPE mask. true => RoPE +
  // sliding (1), false => NoPE + full attention (0).
  REQUIRE(p.text.no_rope_layers.size() == 4);
  CHECK(p.text.no_rope_layers == std::vector<int64_t>{1, 1, 1, 0});
  // Transform 2: the pre-scale is recovered from the folded attn_q_norm, NOT
  // from a metadata key (the file carries none).
  CHECK(p.text.scale_query_by == doctest::Approx(SyntheticMuse::kQueryPreScale));
  // The text GGUF carries no perception encoder; that ships as a separate file.
  CHECK(p.vision.present == false);
}

TEST_CASE("muse glimmer gguf: a tied checkpoint is read off the missing output.weight") {
  const TempFile f(SyntheticMuse::Build(SyntheticMuse::kQueryPreScale, 1.0f, true,
                                        /*untied=*/false));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  CHECK(vllm::ParseMuseGlimmerParams(cfg).text.tie_word_embeddings == true);
}

TEST_CASE("muse glimmer gguf: the canonical -> gguf name map") {
  std::string out;
  REQUIRE(vllm::MuseGlimmerGgufTensorName("model.embed_tokens.weight", &out));
  CHECK(out == "token_embd.weight");
  REQUIRE(vllm::MuseGlimmerGgufTensorName("model.norm.weight", &out));
  CHECK(out == "output_norm.weight");
  REQUIRE(vllm::MuseGlimmerGgufTensorName("lm_head.weight", &out));
  CHECK(out == "output.weight");

  // The four sandwich norms. llama.cpp gives the PRE-feedforward norm the name
  // `ffn_norm` and the POST-feedforward one `post_ffw_norm`; reading those two
  // the other way round swaps a pre-norm with a post-norm (different eps, and a
  // different point in the residual stream) while still producing fluent text.
  const std::pair<const char*, const char*> kNorms[] = {
      {"input_layernorm.weight", "attn_norm.weight"},
      {"post_attention_layernorm.weight", "post_attention_norm.weight"},
      {"pre_feedforward_layernorm.weight", "ffn_norm.weight"},
      {"post_feedforward_layernorm.weight", "post_ffw_norm.weight"},
  };
  for (const auto& [canon, gg] : kNorms) {
    REQUIRE(vllm::MuseGlimmerGgufTensorName(std::string("model.layers.7.") + canon,
                                            &out));
    CHECK(out == std::string("blk.7.") + gg);
  }

  // The attention OUTPUT GATE and the MLP gate share a suffix on the HF side;
  // they must land on DIFFERENT GGUF names.
  REQUIRE(vllm::MuseGlimmerGgufTensorName(
      "model.layers.7.self_attn.output_gate_proj.weight", &out));
  CHECK(out == "blk.7.attn_gate.weight");
  REQUIRE(vllm::MuseGlimmerGgufTensorName("model.layers.7.mlp.gate_proj.weight",
                                          &out));
  CHECK(out == "blk.7.ffn_gate.weight");

  const std::pair<const char*, const char*> kRest[] = {
      {"self_attn.q_proj.weight", "attn_q.weight"},
      {"self_attn.k_proj.weight", "attn_k.weight"},
      {"self_attn.v_proj.weight", "attn_v.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      {"mlp.up_proj.weight", "ffn_up.weight"},
      {"mlp.down_proj.weight", "ffn_down.weight"},
  };
  for (const auto& [canon, gg] : kRest) {
    REQUIRE(vllm::MuseGlimmerGgufTensorName(std::string("model.layers.7.") + canon,
                                            &out));
    CHECK(out == std::string("blk.7.") + gg);
  }

  // The perception encoder has NO counterpart in a text-tower GGUF.
  CHECK(!vllm::MuseGlimmerGgufTensorName("vision_encoder.ln_pre.weight", &out));
  CHECK(!vllm::MuseGlimmerGgufTensorName("vision_projection.weight", &out));
}

TEST_CASE("muse glimmer gguf: the sandwich norms are un-shifted, the final norm is not") {
  const TempFile f(SyntheticMuse::Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  const vllm::MuseGlimmerWeights w = vllm::LoadMuseGlimmerFromGguf(g, cfg);

  REQUIRE(w.text_loaded);
  REQUIRE(w.layers.size() == static_cast<size_t>(SyntheticMuse::kLayers));

  // Transform 1. The file stores `w_hf + 1`; the OwnedTensor must hold `w_hf`,
  // because the forward adds the +1 itself (RmsNormArgs{gemma=true}). The
  // assertion is against the value the BUG produces: skipping the un-shift
  // leaves every element exactly 1.0 too large.
  struct NormCase {
    const vllm::OwnedTensor vllm::MuseGlimmerLayerWeights::*field;
    int64_t seed;
  };
  const NormCase kCases[] = {
      {&vllm::MuseGlimmerLayerWeights::input_layernorm, 10},
      {&vllm::MuseGlimmerLayerWeights::post_attention_layernorm, 20},
      {&vllm::MuseGlimmerLayerWeights::pre_feedforward_layernorm, 30},
      {&vllm::MuseGlimmerLayerWeights::post_feedforward_layernorm, 40},
  };
  for (int64_t l = 0; l < SyntheticMuse::kLayers; ++l) {
    for (const NormCase& c : kCases) {
      const std::vector<float> got = OwnedBf16ToF32(w.layers[static_cast<size_t>(l)].*
                                                    (c.field));
      REQUIRE(got.size() == static_cast<size_t>(SyntheticMuse::kHidden));
      for (int64_t i = 0; i < SyntheticMuse::kHidden; ++i) {
        const float stored = SyntheticMuse::Fill(c.seed + l, i);
        CHECK(got[static_cast<size_t>(i)] == doctest::Approx(stored - 1.0f).epsilon(0.01));
      }
    }
  }

  // The FINAL norm takes no offset in the model, so it is stored RAW and must
  // NOT be un-shifted. Same tensor shape, opposite rule — this is the case that
  // catches an over-eager blanket un-shift.
  const std::vector<float> fin = OwnedBf16ToF32(w.final_norm);
  REQUIRE(fin.size() == static_cast<size_t>(SyntheticMuse::kHidden));
  for (int64_t i = 0; i < SyntheticMuse::kHidden; ++i)
    CHECK(fin[static_cast<size_t>(i)] ==
          doctest::Approx(SyntheticMuse::Fill(200, i)).epsilon(0.01));
}

TEST_CASE("muse glimmer gguf: a genuinely weighted qk-norm is refused, not averaged") {
  // A NON-CONSTANT attn_q_norm is a real per-channel QK-norm weight, which this
  // architecture does not have. Silently taking its mean (or its first element)
  // would be a plausible-looking wrong model.
  {
    const TempFile f(SyntheticMuse::Build(SyntheticMuse::kQueryPreScale, 1.0f,
                                          /*q_norm_constant=*/false));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    CHECK_THROWS_AS(vllm::MuseGlimmerHfConfigFromGguf(g), std::runtime_error);
  }
  // An attn_k_norm that is not ones is the same defect on the key side.
  {
    const TempFile f(SyntheticMuse::Build(SyntheticMuse::kQueryPreScale,
                                          /*k_norm_scale=*/2.5f));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    CHECK_THROWS_AS(vllm::MuseGlimmerHfConfigFromGguf(g), std::runtime_error);
  }
}

TEST_CASE("muse glimmer gguf: shapes and the qkv/gate_up merges") {
  const TempFile f(SyntheticMuse::Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  const vllm::MuseGlimmerWeights w = vllm::LoadMuseGlimmerFromGguf(g, cfg);

  const int64_t H = SyntheticMuse::kHidden;
  const int64_t qdim = SyntheticMuse::kHeads * SyntheticMuse::kHeadDim;
  const int64_t kdim = SyntheticMuse::kKvHeads * SyntheticMuse::kHeadDim;
  const vllm::MuseGlimmerLayerWeights& l0 = w.layers[0];

  // Merged QKV, rows q|k|v — the order is load-bearing (a k shard landing where
  // q is expected permutes attention silently).
  CHECK(l0.attn.qkv_proj.rank == 2);
  CHECK(l0.attn.qkv_proj.shape[0] == qdim + 2 * kdim);
  CHECK(l0.attn.qkv_proj.shape[1] == H);
  CHECK(l0.attn.qkv_proj.nk == true);
  CHECK(l0.attn.o_proj.shape[0] == H);
  CHECK(l0.attn.o_proj.shape[1] == qdim);
  CHECK(l0.attn.output_gate_proj.shape[0] == qdim);
  CHECK(l0.attn.output_gate_proj.shape[1] == H);
  // Merged gate|up.
  CHECK(l0.mlp.gate_up_proj.shape[0] == 2 * SyntheticMuse::kInter);
  CHECK(l0.mlp.gate_up_proj.shape[1] == H);
  CHECK(l0.mlp.down_proj.shape[0] == H);
  CHECK(l0.mlp.down_proj.shape[1] == SyntheticMuse::kInter);

  // The embedding is a [vocab, H] gather table (nk = false); the UNTIED head is
  // the Matmul-B [H, vocab] orientation the forward consumes.
  CHECK(w.embed_tokens.shape[0] == SyntheticMuse::kVocab);
  CHECK(w.embed_tokens.shape[1] == H);
  CHECK(w.embed_tokens.nk == false);
  CHECK(w.lm_head.shape[0] == H);
  CHECK(w.lm_head.shape[1] == SyntheticMuse::kVocab);

  // The QKV row order, asserted against the source bytes: row 0 of the q block,
  // of the k block and of the v block must each come from their own shard.
  const std::vector<float> qkv = OwnedBf16ToF32(l0.attn.qkv_proj);
  for (int64_t i = 0; i < H; ++i) {
    CHECK(qkv[static_cast<size_t>(i)] ==
          doctest::Approx(SyntheticMuse::Fill(50, i)).epsilon(0.01));
    CHECK(qkv[static_cast<size_t>(qdim * H + i)] ==
          doctest::Approx(SyntheticMuse::Fill(60, i)).epsilon(0.01));
    CHECK(qkv[static_cast<size_t>((qdim + kdim) * H + i)] ==
          doctest::Approx(SyntheticMuse::Fill(70, i)).epsilon(0.01));
  }
}

// ── transform 4: the converter's Q/K RoPE row permutation ────────────────────
// llama.cpp stores `attn_q` / `attn_k` in ggml's INTERLEAVED-RoPE row order
// (`LlamaModel.permute`, conversion/llama.py:163-169, which the `muse-glimmer`
// converter of ggml-org/llama.cpp#26841 inherits). Our forward runs
// `vt::RopeNeox`, which is HF's half-offset pairing, so the loader must invert it.
//
// This is the defect behind issue #359: the permutation is INVISIBLE to every
// structural check (right names, right shapes, right counts) and invisible to
// attention itself on the 13 NoPE layers (a permutation applied to BOTH q and k
// leaves q·k unchanged), so the 17 GB k-quant loaded, ran, and emitted degenerate
// text — " is is is is ..." — instead of failing.
TEST_CASE("muse glimmer gguf: the converter's Q/K RoPE row permutation is inverted") {
  const TempFile f(SyntheticMuse::Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  const vllm::MuseGlimmerWeights w = vllm::LoadMuseGlimmerFromGguf(g, cfg);

  const int64_t H = SyntheticMuse::kHidden;
  const int64_t qdim = SyntheticMuse::kHeads * SyntheticMuse::kHeadDim;
  const int64_t kdim = SyntheticMuse::kKvHeads * SyntheticMuse::kHeadDim;
  const std::vector<float> qkv = OwnedBf16ToF32(w.layers[0].attn.qkv_proj);

  // The synthetic filler is a multiple of 1/8, so every value is EXACT in bf16
  // and these are equalities, not tolerances.
  // q: permuted with `n_head` (muse `attention.head_count`).
  for (int64_t r = 0; r < qdim; ++r) {
    const int64_t src = QkSrcRow(r, SyntheticMuse::kHeads, qdim);
    for (int64_t c = 0; c < H; ++c)
      CHECK(qkv[static_cast<size_t>(r * H + c)] == SyntheticMuse::Fill(50, src * H + c));
  }
  // k: permuted with `n_head_kv`, NOT `n_head` — llama.cpp substitutes the KV
  // head count for the key side, so a k shard un-permuted with 2 q-heads instead
  // of 1 kv-head lands on different rows.
  for (int64_t r = 0; r < kdim; ++r) {
    const int64_t src = QkSrcRow(r, SyntheticMuse::kKvHeads, kdim);
    for (int64_t c = 0; c < H; ++c)
      CHECK(qkv[static_cast<size_t>((qdim + r) * H + c)] ==
            SyntheticMuse::Fill(60, src * H + c));
  }
  // v is NOT permuted upstream and must stay VERBATIM. Un-permuting it too would
  // be just as wrong as not un-permuting q/k, and just as quiet.
  for (int64_t r = 0; r < kdim; ++r)
    for (int64_t c = 0; c < H; ++c)
      CHECK(qkv[static_cast<size_t>((qdim + kdim + r) * H + c)] ==
            SyntheticMuse::Fill(70, r * H + c));

  // The map must not be the identity, or the three loops above would pass on a
  // loader that ignores the transform entirely.
  CHECK(QkSrcRow(1, SyntheticMuse::kHeads, qdim) != 1);
  CHECK(qkv[static_cast<size_t>(1 * H)] != SyntheticMuse::Fill(50, 1 * H));

  // The MLP shards share `LoadMerged` and carry no permutation at all.
  const std::vector<float> gu = OwnedBf16ToF32(w.layers[0].mlp.gate_up_proj);
  for (int64_t c = 0; c < H; ++c) {
    CHECK(gu[static_cast<size_t>(1 * H + c)] == SyntheticMuse::Fill(100, 1 * H + c));
    CHECK(gu[static_cast<size_t>((SyntheticMuse::kInter + 1) * H + c)] ==
          SyntheticMuse::Fill(110, 1 * H + c));
  }
}

TEST_CASE("muse glimmer gguf: the Q/K un-permute also runs on the KEPT-QUANT concat") {
  // A homogeneous quantized qkv trio is merged as a BYTE CONCAT, and the row
  // reorder has to happen there too. Rows are whole blocks, so this is a pure
  // byte-level row permutation and nothing is requantized.
  const TempFile f(SyntheticMuseQ8::Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  vllm::GgufLoadPolicy pol;
  pol.keep_quant = true;
  // Pin the two byte-rewriting levers OFF so this case reads the concat itself.
  pol.quant_repack = false;
  pol.mmap_residency = false;
  const vllm::MuseGlimmerWeights w = vllm::LoadMuseGlimmerFromGguf(g, cfg, &pol);

  const int64_t H = SyntheticMuseQ8::kHidden;
  const int64_t qdim = SyntheticMuseQ8::kHeads * SyntheticMuseQ8::kHeadDim;
  const int64_t kdim = SyntheticMuseQ8::kKvHeads * SyntheticMuseQ8::kHeadDim;
  const vllm::OwnedTensor& qkv = w.layers[0].attn.qkv_proj;
  REQUIRE(qkv.shape[0] == qdim + 2 * kdim);
  REQUIRE(qkv.shape[1] == H);
  // block_q8_0 is { fp16 d; int8 qs[32] } = 34 bytes per 32 columns. Restated
  // here rather than read from the loader so the test is not circular.
  const size_t row_bytes = 34 * static_cast<size_t>(H / 32);
  REQUIRE(qkv.bytes.size() == static_cast<size_t>(qdim + 2 * kdim) * row_bytes);

  const auto* q_src = g.Get("blk.0.attn_q.weight").data;
  const auto* k_src = g.Get("blk.0.attn_k.weight").data;
  const auto* v_src = g.Get("blk.0.attn_v.weight").data;
  const uint8_t* dst = qkv.bytes.data();
  for (int64_t r = 0; r < qdim; ++r) {
    const size_t src = static_cast<size_t>(QkSrcRow(r, SyntheticMuseQ8::kHeads, qdim));
    CHECK(std::memcmp(dst + static_cast<size_t>(r) * row_bytes,
                      q_src + src * row_bytes, row_bytes) == 0);
  }
  for (int64_t r = 0; r < kdim; ++r) {
    const size_t src = static_cast<size_t>(QkSrcRow(r, SyntheticMuseQ8::kKvHeads, kdim));
    CHECK(std::memcmp(dst + static_cast<size_t>(qdim + r) * row_bytes,
                      k_src + src * row_bytes, row_bytes) == 0);
  }
  for (int64_t r = 0; r < kdim; ++r)
    CHECK(std::memcmp(dst + static_cast<size_t>(qdim + kdim + r) * row_bytes,
                      v_src + static_cast<size_t>(r) * row_bytes, row_bytes) == 0);
  // Not the identity: row 1 of the merged q block is NOT row 1 of the file.
  CHECK(std::memcmp(dst + row_bytes, q_src + row_bytes, row_bytes) != 0);
}

TEST_CASE("muse glimmer gguf: synthetic accounting is exact in both directions") {
  const TempFile f(SyntheticMuse::Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  const vllm::MuseGlimmerParams p = vllm::ParseMuseGlimmerParams(cfg);

  const std::vector<std::string> expected = vllm::EnumerateMuseGlimmerGgufTensors(p);
  std::set<std::string> present;
  for (const vllm::GgufTensorInfo& t : g.Tensors()) present.insert(t.name);

  std::vector<std::string> missing;
  for (const std::string& n : expected)
    if (present.count(n) == 0) missing.push_back(n);
  std::vector<std::string> unaccounted;
  const std::set<std::string> want(expected.begin(), expected.end());
  for (const std::string& n : present)
    if (want.count(n) == 0) unaccounted.push_back(n);

  CHECK(missing.empty());
  CHECK(unaccounted.empty());
  CHECK(static_cast<int64_t>(expected.size()) ==
        static_cast<int64_t>(g.Tensors().size()));

  const vllm::MuseGlimmerWeights w = vllm::LoadMuseGlimmerFromGguf(g, cfg);
  CHECK(w.enumerated_tensors == static_cast<int64_t>(expected.size()));
  CHECK(w.accounted_tensors == w.enumerated_tensors);
}

TEST_CASE("muse glimmer gguf: the REAL 731-tensor k-quant manifest accounts exactly") {
  // Committed header-only projection of
  // meta-models/Muse-Glimmer-30B-GGUF @ 2fb01e4e6f muse-glimmer-30B-kquant-17gb.gguf.
  CHECK(std::string(vllm_test::kMuseGgufArchitecture) == "muse-glimmer");
  CHECK(vllm_test::kMuseGgufVersion == 3);
  REQUIRE(vllm_test::kMuseGgufTensorCount == 731);

  std::map<std::string, const vllm_test::MuseGgufTensor*> present;
  for (const vllm_test::MuseGgufTensor& t : vllm_test::kMuseGgufTensors)
    present.emplace(t.name, &t);
  REQUIRE(present.size() == static_cast<size_t>(vllm_test::kMuseGgufTensorCount));

  const vllm::MuseGlimmerParams p = Released30BParams();
  const std::vector<std::string> expected = vllm::EnumerateMuseGlimmerGgufTensors(p);

  std::vector<std::string> missing;
  for (const std::string& n : expected)
    if (present.count(n) == 0) missing.push_back(n);
  const std::set<std::string> want(expected.begin(), expected.end());
  std::vector<std::string> unaccounted;
  for (const auto& [name, t] : present)
    if (want.count(name) == 0) unaccounted.push_back(name);

  INFO("missing=", missing.empty() ? std::string("-") : missing.front(),
       " unaccounted=", unaccounted.empty() ? std::string("-") : unaccounted.front());
  CHECK(missing.empty());
  CHECK(unaccounted.empty());
  // 3 + 52 * 14: the trunk's embed/final-norm/head plus fourteen per layer, two
  // of which (attn_{q,k}_norm) exist only in the GGUF.
  CHECK(static_cast<int64_t>(expected.size()) == 731);

  // SHAPES, in the file's own ne order (reversed vs torch). A name map that is
  // right about names and wrong about orientation is the failure this catches.
  struct ShapeCase {
    const char* name;
    int64_t ne0;
    int64_t ne1;  // 0 => 1-D
  };
  const ShapeCase kShapes[] = {
      {"token_embd.weight", 6656, 202048},
      {"output.weight", 6656, 202048},
      {"output_norm.weight", 6656, 0},
      {"blk.0.attn_norm.weight", 6656, 0},
      {"blk.0.post_attention_norm.weight", 6656, 0},
      {"blk.0.ffn_norm.weight", 6656, 0},
      {"blk.0.post_ffw_norm.weight", 6656, 0},
      {"blk.0.attn_q.weight", 6656, 4096},
      {"blk.0.attn_k.weight", 6656, 256},
      {"blk.0.attn_v.weight", 6656, 256},
      {"blk.0.attn_output.weight", 4096, 6656},
      {"blk.0.attn_gate.weight", 6656, 4096},
      {"blk.0.attn_q_norm.weight", 128, 0},
      {"blk.0.attn_k_norm.weight", 128, 0},
      {"blk.0.ffn_gate.weight", 6656, 19968},
      {"blk.0.ffn_up.weight", 6656, 19968},
      {"blk.0.ffn_down.weight", 19968, 6656},
  };
  for (const ShapeCase& s : kShapes) {
    const auto it = present.find(s.name);
    REQUIRE_MESSAGE(it != present.end(), s.name);
    CHECK(it->second->dims[0] == s.ne0);
    if (s.ne1 != 0) {
      CHECK(it->second->n_dims == 2);
      CHECK(it->second->dims[1] == s.ne1);
    } else {
      CHECK(it->second->n_dims == 1);
    }
  }

  // The weightless-norm vectors are F32 in the file (they encode a scalar), and
  // the sandwich norms too — a k-quant of a 6656-vector would be lossy for no
  // gain, and the un-shift needs real values.
  for (const char* n : {"blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
                        "blk.0.attn_norm.weight", "output_norm.weight"})
    CHECK(present.at(n)->ggml_type == 0u);
}

TEST_CASE("muse glimmer gguf: the mmproj perception encoder refuses by name") {
  CHECK(std::string(vllm_test::kMuseMmprojGgufArchitecture) == "clip");
  REQUIRE(vllm_test::kMuseMmprojGgufTensorCount == 809);

  std::map<std::string, const vllm_test::MuseMmprojGgufTensor*> present;
  for (const vllm_test::MuseMmprojGgufTensor& t : vllm_test::kMuseMmprojGgufTensors)
    present.emplace(t.name, &t);

  // THE EVIDENCE for the refusal, read off the real file rather than asserted in
  // prose: `v.patch_embd.weight` is ne [14, 14, 3, 1536] = torch [1536, 3*14*14]
  // = [1536, 588], while our conv1_linear needs
  // patch_temporal * 3 * patch^2 = 2*3*196 = 1176 input features (and the
  // safetensors ships exactly [1536, 1176]). The patch_temporal axis is absent.
  const auto it = present.find("v.patch_embd.weight");
  REQUIRE(it != present.end());
  CHECK(it->second->n_dims == 4);
  CHECK(it->second->dims[0] == 14);
  CHECK(it->second->dims[1] == 14);
  CHECK(it->second->dims[2] == 3);  // NOT patch_temporal * 3 == 6
  CHECK(it->second->dims[3] == 1536);
  const int64_t gguf_in_features =
      it->second->dims[0] * it->second->dims[1] * it->second->dims[2];
  CHECK(gguf_in_features == 588);
  CHECK(gguf_in_features * 2 == 1176);  // exactly the missing temporal half

  // Every OTHER tower tensor is present, so this is one missing axis rather than
  // a naming problem — which is what makes the refusal precise instead of a
  // blanket "unsupported".
  for (const char* n : {"v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight",
                        "v.post_ln.bias", "v.position_embd.weight", "mm.0.weight",
                        "mm.1.weight", "mm.2.weight", "v.blk.49.attn_q.weight",
                        "v.blk.49.attn_out.bias", "v.blk.49.ffn_down.weight"})
    CHECK_MESSAGE(present.count(n) == 1, n);
  CHECK(present.at("v.position_embd.weight")->dims[0] == 1536);
  CHECK(present.at("v.position_embd.weight")->dims[1] == 1024);  // 32 x 32 grid
  CHECK(present.at("mm.2.weight")->dims[0] == 4096);             // adapter_dim
  CHECK(present.at("mm.2.weight")->dims[1] == 6656);             // text hidden

  // And the refusal itself names the missing piece.
  bool threw = false;
  try {
    vllm::MuseGlimmerRefuseMmproj();
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string msg = e.what();
    CHECK(msg.find("patch_temporal") != std::string::npos);
    CHECK(msg.find("mmproj") != std::string::npos);
  }
  CHECK(threw);
}

TEST_CASE("muse glimmer gguf: the dflash drafter manifest is covered by the existing seam") {
  // REACHABILITY ONLY. This says the drafter file's every tensor has a home in
  // the ALREADY-LANDED `qwen3_dflash_gguf` name map — no new seam is needed. It
  // says NOTHING about whether a Muse Glimmer draft proposes useful tokens; that
  // needs an acceptance-rate run this row does not have hardware for.
  CHECK(std::string(vllm_test::kMuseDflashGgufArchitecture) == "dflash");
  REQUIRE(vllm_test::kMuseDflashGgufTensorCount == 58);

  std::set<std::string> present;
  for (const vllm_test::MuseDflashGgufTensor& t : vllm_test::kMuseDflashGgufTensors)
    present.insert(t.name);

  // The names `qwen3_dflash_gguf.cpp`'s MapName emits, for the 5 blocks this
  // file declares (`dflash.block_count`).
  std::vector<std::string> expected = {"fc.weight", "enc.output_norm.weight",
                                       "output_norm.weight"};
  for (int i = 0; i < 5; ++i) {
    const std::string b = "blk." + std::to_string(i) + ".";
    for (const char* s : {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                          "attn_output.weight", "attn_q_norm.weight",
                          "attn_k_norm.weight", "attn_norm.weight",
                          "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                          "ffn_down.weight"})
      expected.push_back(b + s);
  }
  std::vector<std::string> missing;
  for (const std::string& n : expected)
    if (present.count(n) == 0) missing.push_back(n);
  const std::set<std::string> want(expected.begin(), expected.end());
  std::vector<std::string> unaccounted;
  for (const std::string& n : present)
    if (want.count(n) == 0) unaccounted.push_back(n);
  INFO("missing=", missing.empty() ? std::string("-") : missing.front(),
       " unaccounted=", unaccounted.empty() ? std::string("-") : unaccounted.front());
  CHECK(missing.empty());
  CHECK(unaccounted.empty());
  CHECK(expected.size() == 58u);

  // The draft carries NEITHER token_embd NOR output: it runs the TARGET's
  // embedding table and head, which the text GGUF does ship. That is what makes
  // `LoadGgufSharedEmbedAndHeadBf16` the right source for them.
  CHECK(present.count("token_embd.weight") == 0);
  CHECK(present.count("output.weight") == 0);
}

TEST_CASE("muse glimmer gguf: the LIVE k-quant, when VLLM_MUSE_GGUF names it") {
  const char* path = std::getenv("VLLM_MUSE_GGUF");
  if (path == nullptr || *path == '\0') {
    MESSAGE("skipped: set VLLM_MUSE_GGUF to muse-glimmer-30B-kquant-17gb.gguf");
    return;
  }
  const vllm::GgufFile g = vllm::GgufFile::Open(path);
  REQUIRE(vllm::IsMuseGlimmerGguf(g));

  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  const vllm::MuseGlimmerParams p = vllm::ParseMuseGlimmerParams(cfg);
  CHECK(p.text.num_hidden_layers == 52);
  CHECK(p.text.hidden_size == 6656);
  CHECK(p.text.num_attention_heads == 32);
  CHECK(p.text.num_key_value_heads == 2);
  CHECK(p.text.head_dim == 128);
  CHECK(p.text.vocab_size == 202048);
  CHECK(p.text.intermediate_size == 19968);
  CHECK(p.text.tie_word_embeddings == false);
  // Recovered from the folded attn_q_norm, and it must equal the safetensors
  // config's `qk_scale_factor` — two independent encodings of one number.
  CHECK(p.text.scale_query_by == doctest::Approx(3.87).epsilon(1e-4));
  // The iRoPE mask the released config encodes as layer_types/layer_rope_theta:
  // NoPE at 3, 7, ... 51.
  REQUIRE(p.text.no_rope_layers.size() == 52);
  for (int64_t i = 0; i < 52; ++i)
    CHECK(p.text.no_rope_layers[static_cast<size_t>(i)] == ((i % 4) == 3 ? 0 : 1));

  // Accounting against the LIVE file, both directions.
  const std::vector<std::string> expected = vllm::EnumerateMuseGlimmerGgufTensors(p);
  std::set<std::string> present;
  for (const vllm::GgufTensorInfo& t : g.Tensors()) present.insert(t.name);
  std::vector<std::string> missing;
  for (const std::string& n : expected)
    if (present.count(n) == 0) missing.push_back(n);
  const std::set<std::string> want(expected.begin(), expected.end());
  std::vector<std::string> unaccounted;
  for (const std::string& n : present)
    if (want.count(n) == 0) unaccounted.push_back(n);
  INFO("live missing=", missing.empty() ? std::string("-") : missing.front(),
       " unaccounted=", unaccounted.empty() ? std::string("-") : unaccounted.front());
  CHECK(missing.empty());
  CHECK(unaccounted.empty());
  CHECK(expected.size() == present.size());
  MESSAGE("live accounting: ", expected.size(), " enumerated / ", present.size(),
          " present");
}

TEST_CASE("muse glimmer gguf: the LIVE k-quant MATERIALIZES, when VLLM_MUSE_GGUF_LOAD names it") {
  // Separate from the accounting case above because this one actually reads the
  // weights: ~22 GB resident and a few minutes of dequant. Names lining up is not
  // the same claim as the tower being built, which is why this exists — but it is
  // still a STRUCTURAL claim. No forward is run and none is implied.
  const char* path = std::getenv("VLLM_MUSE_GGUF_LOAD");
  if (path == nullptr || *path == '\0') {
    MESSAGE("skipped: set VLLM_MUSE_GGUF_LOAD to load the whole 16.76 GB k-quant");
    return;
  }
  const vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig cfg = vllm::MuseGlimmerHfConfigFromGguf(g);
  const vllm::MuseGlimmerWeights w = vllm::LoadMuseGlimmerFromGguf(g, cfg);

  REQUIRE(w.text_loaded);
  CHECK(w.enumerated_tensors == 731);
  CHECK(w.accounted_tensors == 731);
  REQUIRE(w.layers.size() == 52u);
  CHECK(w.vision.loaded == false);  // the perception encoder is a separate file

  CHECK(w.embed_tokens.shape[0] == 202048);
  CHECK(w.embed_tokens.shape[1] == 6656);
  CHECK(w.embed_tokens.nk == false);
  CHECK(w.lm_head.shape[0] == 6656);  // Matmul-B [H, vocab]
  CHECK(w.lm_head.shape[1] == 202048);
  CHECK(w.final_norm.Numel() == 6656);

  for (size_t l = 0; l < w.layers.size(); ++l) {
    const vllm::MuseGlimmerLayerWeights& lw = w.layers[l];
    CHECK(lw.attn.qkv_proj.shape[0] == 4096 + 2 * 256);
    CHECK(lw.attn.qkv_proj.shape[1] == 6656);
    CHECK(lw.attn.qkv_proj.nk == true);
    CHECK(lw.attn.o_proj.shape[0] == 6656);
    CHECK(lw.attn.o_proj.shape[1] == 4096);
    CHECK(lw.attn.output_gate_proj.shape[0] == 4096);
    CHECK(lw.mlp.gate_up_proj.shape[0] == 2 * 19968);
    CHECK(lw.mlp.gate_up_proj.shape[1] == 6656);
    CHECK(lw.mlp.down_proj.shape[0] == 6656);
    CHECK(lw.mlp.down_proj.shape[1] == 19968);
    CHECK(lw.input_layernorm.Numel() == 6656);
    CHECK(lw.post_feedforward_layernorm.Numel() == 6656);
  }

  // The un-shift, on the REAL weights. The released checkpoint's layer-0
  // `input_layernorm` has min exactly -1.0 (so the GGUF's stored min is 0.0);
  // an un-shift that did not fire would leave every element >= 0.
  const std::vector<float> n0 = OwnedBf16ToF32(w.layers[0].input_layernorm);
  float mn = n0[0];
  for (float v : n0) mn = std::min(mn, v);
  CHECK(mn == doctest::Approx(-1.0f).epsilon(0.01));
  MESSAGE("live load: 52 layers, layer-0 input_layernorm min = ", mn,
          " (un-shifted)");
}
