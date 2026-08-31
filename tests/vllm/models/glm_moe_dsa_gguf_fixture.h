// The synthetic `glm-dsa` GGUF fixture, shared by W7's load suite and W9's
// forward suite.
//
// WHY IT IS A HEADER. W7 built this model inside its own test TU, and W9 needs
// the SAME model: a second copy would be two descriptions of one artifact, and
// the first thing that would drift is the indexer schedule — the one field whose
// disagreement produces a plausible, finite, wrong answer rather than a crash.
// The builder is unchanged; it moved.
//
// WHAT IT IS FOR. The real gate is a 201.83 GiB load under an `rc` lease, which
// CI cannot run. What CI can run is the same loader and the same forward over a
// model with the same STRUCTURE at 1/1000th the size: the same tensor names, the
// same heterogeneous indexer schedule with `full` and `shared` layers
// interleaved, the same leading dense block, the same stacked keep-quant towers,
// and the same multi-token-prediction tail to drop.
#ifndef VLLM_TESTS_VLLM_MODELS_GLM_MOE_DSA_GGUF_FIXTURE_H_
#define VLLM_TESTS_VLLM_MODELS_GLM_MOE_DSA_GGUF_FIXTURE_H_

#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../gguf_builder.h"

namespace glm_dsa_fixture {

// The tiny model's geometry. Every number is chosen so the shapes are legal for
// the real code path: `K` is a whole number of Q8_0 blocks on every tower, the
// MLA latent is stated consistently three times, and the indexer schedule has
// both kinds of layer with a `full` one first.
constexpr int64_t kHidden = 64;
constexpr int64_t kVocab = 32;
constexpr int64_t kBackbone = 4;   // block_count 5 minus one MTP block
constexpr int64_t kHeads = 4;
constexpr int64_t kQLora = 48;
constexpr int64_t kKvLora = 32;
constexpr int64_t kQkRope = 8;
constexpr int64_t kQkHead = 24;    // key_length_mla
constexpr int64_t kQkNope = kQkHead - kQkRope;  // 16
constexpr int64_t kVHead = 24;
constexpr int64_t kInter = 128;    // the leading dense block's MLP
constexpr int64_t kMoeInter = 32;
constexpr int64_t kExperts = 8;
constexpr int64_t kIdxHeads = 4;
constexpr int64_t kIdxHead = 16;
constexpr int64_t kLeadingDense = 1;
// `indexer.types` = full, shared, full, shared. Layer 0 is full, which
// `GlmMoeDsaMlaSchedule` requires: a `shared` layer 0 would attend through a
// buffer nothing had written.
constexpr int64_t kFullLayers = 2;
constexpr int64_t kSharedLayers = kBackbone - kFullLayers;

// ─── Q8_0 blocks, built ──────────────────────────────────────────────────────
// `block_q8_0` is `{ ggml_fp16_t d; int8_t qs[32]; }` = 34 bytes for 32
// elements, which `gguf_reader.cpp` case 8 sizes as `{32, 34}`.
constexpr uint16_t kFp16One = 0x3C00;  // 1.0 in IEEE half

inline std::string Q8_0Blocks(int64_t numel, uint32_t seed) {
  REQUIRE(numel % 32 == 0);
  std::string out;
  out.reserve(static_cast<size_t>(numel) / 32 * 34);
  uint32_t s = seed | 1u;
  for (int64_t b = 0; b < numel / 32; ++b) {
    char d[2];
    std::memcpy(d, &kFp16One, 2);
    out.append(d, 2);
    for (int i = 0; i < 32; ++i) {
      s = s * 1664525u + 1013904223u;
      // A small signed value, deterministic and finite. The scale is exactly
      // 1.0, so the dequantized weight is this integer.
      out.push_back(static_cast<char>(static_cast<int8_t>((s >> 24) % 7) - 3));
    }
  }
  return out;
}

// ─── F32 payload, and why it is NOT ZEROS ────────────────────────────────────
//
// W7 filled every non-tower tensor with zeros, which was right for what W7
// asserted: shapes, dtypes, tensor counts and refusals do not care about values.
// It is WRONG for a forward gate, and wrong in the way that hurts. A zero
// embedding table makes every hidden state zero, every norm output zero and
// every logit zero — so the model emits token id 0, every "are these two runs
// identical" check passes trivially, and every "did this change move the
// output" check fails for a reason that has nothing to do with the code under
// test. W9's first run of this suite produced exactly that: `min=0 max=0
// mean=0 sd=0` across all 32 logits, a token id of 0, and a streamed-vs-
// resident comparison that was true of two forwards that computed nothing.
//
// THE VALUES ARE BF16-EXACT BY CONSTRUCTION, and that is not incidental. The
// loader NARROWS `indexer.proj.weight` from the file's F32 to the model dtype
// and REFUSES a value that does not round-trip (spec §3.7 W9), because upstream
// builds that projection at the model dtype and a genuinely-f32 one describes a
// different operator. Every value here is `k / 64` for integer `k` in
// `[-16, 16]`, which has at most 5 mantissa bits and is therefore exact in
// bf16's 8. A fixture built from arbitrary floats would trip that refusal and
// look like a loader defect.
//
// Small and mean-centred, so an 80-token forward through four layers of
// unnormalised random weights stays in range: the RMSNorms bound the magnitude
// per layer, and the residual stream does not run away.
inline std::string F32Pattern(int64_t numel, uint32_t seed) {
  std::string out;
  out.resize(static_cast<size_t>(numel) * 4);
  uint32_t s = seed | 1u;
  for (int64_t i = 0; i < numel; ++i) {
    s = s * 1664525u + 1013904223u;
    const int32_t k = static_cast<int32_t>((s >> 24) % 33) - 16;
    const float v = static_cast<float>(k) / 64.0f;
    std::memcpy(&out[static_cast<size_t>(i) * 4], &v, 4);
  }
  return out;
}

// A per-tensor seed from the tensor NAME (FNV-1a), so two tensors of the same
// shape do not carry identical values. Identical operands are their own
// degeneracy: a `gate_proj` equal to its `up_proj` makes `silu(g)*u` a function
// of one matrix, and a `k_b` equal to its `v_b` hides a swapped absorption.
inline uint32_t SeedFor(const std::string& name) {
  uint32_t h = 2166136261u;
  for (char c : name) {
    h ^= static_cast<uint8_t>(c);
    h *= 16777619u;
  }
  return h;
}

inline int64_t Prod(const std::vector<uint64_t>& dims) {
  int64_t n = 1;
  for (uint64_t d : dims) n *= static_cast<int64_t>(d);
  return n;
}

// `GgufModelBuilder::AddTensor` takes dims in the FILE's ggml `ne` order, which
// is the reverse of the `[N, K]` torch order `GgufTensorInfo::shape` reports.
// So a `[N, K]` matmul weight is added as `{K, N}`.
inline void AddF32(gguf_test::GgufModelBuilder& b, const std::string& name,
            const std::vector<uint64_t>& ne) {
  b.AddTensor(name, ne, /*ggml_type=*/0, F32Pattern(Prod(ne), SeedFor(name)));
}

inline void AddQ8(gguf_test::GgufModelBuilder& b, const std::string& name,
           const std::vector<uint64_t>& ne, uint32_t seed) {
  b.AddTensor(name, ne, /*ggml_type=*/8, Q8_0Blocks(Prod(ne), seed));
}

inline std::string Blk(int64_t l, const std::string& s) {
  return "blk." + std::to_string(l) + "." + s;
}

// A COMPLETE `glm-dsa` file: header, tokenizer, and every tensor this port
// claims plus the multi-token-prediction tail it drops.
//
// `extra` and `omit` exist for the negative cases, which are the half of this
// suite that proves the accounting is a gate rather than a decoration.
inline std::string BuildCompleteGlmDsa(const std::string& extra_tensor = "",
                                bool expand_one_tower = false) {
  gguf_test::GgufModelBuilder b;
  const std::string p = "glm-dsa.";
  b.AddKv(gguf_test::StrKv("general.architecture", "glm-dsa"));
  b.AddKv(gguf_test::U32Kv(p + "block_count", kBackbone + 1));
  b.AddKv(gguf_test::U32Kv(p + "nextn_predict_layers", 1));
  b.AddKv(gguf_test::U32Kv(p + "embedding_length", kHidden));
  // Small, unlike the real file's 1,048,576: the rope cos/sin cache is
  // `[max_position_embeddings, qk_rope_head_dim]` and the real one is 134 MB.
  b.AddKv(gguf_test::U32Kv(p + "context_length", 256));
  b.AddKv(gguf_test::U32Kv(p + "attention.head_count", kHeads));
  b.AddKv(gguf_test::U32Kv(p + "attention.head_count_kv", 1));
  b.AddKv(gguf_test::U32Kv(p + "feed_forward_length", kInter));
  b.AddKv(gguf_test::F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-5f));
  b.AddKv(gguf_test::F32Kv(p + "rope.freq_base", 8e6f));
  b.AddKv(gguf_test::U32Kv(p + "attention.kv_lora_rank", kKvLora));
  b.AddKv(gguf_test::U32Kv(p + "rope.dimension_count", kQkRope));
  b.AddKv(gguf_test::U32Kv(p + "attention.key_length", kKvLora + kQkRope));
  b.AddKv(gguf_test::U32Kv(p + "attention.key_length_mla", kQkHead));
  b.AddKv(gguf_test::U32Kv(p + "attention.value_length_mla", kVHead));
  b.AddKv(gguf_test::U32Kv(p + "attention.q_lora_rank", kQLora));
  b.AddKv(gguf_test::U32Kv(p + "expert_count", kExperts));
  b.AddKv(gguf_test::U32Kv(p + "expert_used_count", 2));
  b.AddKv(gguf_test::U32Kv(p + "expert_feed_forward_length", kMoeInter));
  b.AddKv(gguf_test::U32Kv(p + "expert_shared_count", 1));
  b.AddKv(gguf_test::U32Kv(p + "leading_dense_block_count", kLeadingDense));
  b.AddKv(gguf_test::F32Kv(p + "expert_weights_scale", 2.5f));
  b.AddKv(gguf_test::BoolKv(p + "expert_weights_norm", true));
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.head_count", kIdxHeads));
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.key_length", kIdxHead));
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.top_k", 64));
  // full, shared, full, shared.
  b.AddKv(gguf_test::BoolArrayKv(p + "attention.indexer.types",
                                 std::vector<bool>{true, false, true, false}));
  {
    b.AddKv(gguf_test::StrKv("tokenizer.ggml.model", "gpt2"));
    b.AddKv(gguf_test::StrKv("tokenizer.ggml.pre", "glm4"));
    std::vector<std::string> toks;
    std::vector<int32_t> types;
    for (int i = 0; i < kVocab; ++i) {
      toks.push_back("t" + std::to_string(i));
      types.push_back(1);
    }
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.tokens", toks));
    b.AddKv(gguf_test::I32ArrayKv("tokenizer.ggml.token_type", types));
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.merges",
                                  std::vector<std::string>{}));
  }

  // ── model level ──
  AddF32(b, "token_embd.weight", {kHidden, kVocab});
  AddF32(b, "output_norm.weight", {kHidden});
  AddF32(b, "output.weight", {kHidden, kVocab});

  // ── every block, backbone and MTP tail ──
  for (int64_t l = 0; l <= kBackbone; ++l) {
    AddF32(b, Blk(l, "attn_norm.weight"), {kHidden});
    AddF32(b, Blk(l, "ffn_norm.weight"), {kHidden});
    AddF32(b, Blk(l, "attn_q_a.weight"), {kHidden, kQLora});
    AddF32(b, Blk(l, "attn_q_a_norm.weight"), {kQLora});
    AddF32(b, Blk(l, "attn_q_b.weight"), {kQLora, kHeads * kQkHead});
    AddF32(b, Blk(l, "attn_kv_a_mqa.weight"), {kHidden, kKvLora + kQkRope});
    AddF32(b, Blk(l, "attn_kv_a_norm.weight"), {kKvLora});
    AddF32(b, Blk(l, "attn_k_b.weight"), {kQkNope, kKvLora, kHeads});
    AddF32(b, Blk(l, "attn_v_b.weight"), {kKvLora, kVHead, kHeads});
    AddF32(b, Blk(l, "attn_output.weight"), {kHeads * kVHead, kHidden});
    // The conversion broadcasts these onto every block; the loader reads the
    // schedule and drops the surplus.
    AddF32(b, Blk(l, "indexer.attn_q_b.weight"), {kQLora, kIdxHeads * kIdxHead});
    AddF32(b, Blk(l, "indexer.attn_k.weight"), {kHidden, kIdxHead});
    AddF32(b, Blk(l, "indexer.k_norm.weight"), {kIdxHead});
    AddF32(b, Blk(l, "indexer.k_norm.bias"), {kIdxHead});
    AddF32(b, Blk(l, "indexer.proj.weight"), {kHidden, kIdxHeads});

    if (l < kLeadingDense) {
      AddF32(b, Blk(l, "ffn_gate.weight"), {kHidden, kInter});
      AddF32(b, Blk(l, "ffn_up.weight"), {kHidden, kInter});
      AddF32(b, Blk(l, "ffn_down.weight"), {kInter, kHidden});
    } else {
      AddF32(b, Blk(l, "ffn_gate_inp.weight"), {kHidden, kExperts});
      AddF32(b, Blk(l, "exp_probs_b.bias"), {kExperts});
      const bool expand_this = expand_one_tower && l == kLeadingDense;
      if (expand_this) {
        // F32 has no keep-quant residency, so this tower expands and the
        // loader must refuse it by name.
        AddF32(b, Blk(l, "ffn_gate_exps.weight"),
               {kHidden, kMoeInter, kExperts});
      } else {
        AddQ8(b, Blk(l, "ffn_gate_exps.weight"),
              {kHidden, kMoeInter, kExperts}, 11u + static_cast<uint32_t>(l));
      }
      AddQ8(b, Blk(l, "ffn_up_exps.weight"), {kHidden, kMoeInter, kExperts},
            41u + static_cast<uint32_t>(l));
      AddQ8(b, Blk(l, "ffn_down_exps.weight"), {kMoeInter, kHidden, kExperts},
            71u + static_cast<uint32_t>(l));
      AddF32(b, Blk(l, "ffn_gate_shexp.weight"), {kHidden, kMoeInter});
      AddF32(b, Blk(l, "ffn_up_shexp.weight"), {kHidden, kMoeInter});
      AddF32(b, Blk(l, "ffn_down_shexp.weight"), {kMoeInter, kHidden});
    }
  }
  // The MTP tail's own four tensors, on top of the full block above — the real
  // file's `blk.78` carries both.
  AddF32(b, Blk(kBackbone, "nextn.eh_proj.weight"), {kHidden, 2 * kHidden});
  AddF32(b, Blk(kBackbone, "nextn.enorm.weight"), {kHidden});
  AddF32(b, Blk(kBackbone, "nextn.hnorm.weight"), {kHidden});
  AddF32(b, Blk(kBackbone, "nextn.shared_head_norm.weight"), {kHidden});

  if (!extra_tensor.empty()) AddF32(b, extra_tensor, {kHidden});
  return b.Build();
}

}  // namespace glm_dsa_fixture

#endif  // VLLM_TESTS_VLLM_MODELS_GLM_MOE_DSA_GGUF_FIXTURE_H_
