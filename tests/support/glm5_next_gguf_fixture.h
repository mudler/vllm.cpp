// The ONE synthetic `glm5next` GGUF file the W5c loader suite reads, and the
// production entry point it reaches the loader through.
//
// Issue [#2242](https://github.com/mudler/vllm.cpp/issues/2242), spec
// `.agents/specs/glm5-next-flash.md` §W5c.
//
// ─── WHY A MINIATURE AND NOT THE ARTIFACT ────────────────────────────────────
// The published `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL` is 101.2535 GiB in four
// shards. Its NAMES, SHAPES and ENCODINGS are gated against this loader with no
// asset at all, out of the committed header manifest
// (`tests/vllm/models/glm5_next_gguf_manifest.inc`). What a manifest cannot do
// is run a load, so this file is a byte-exact miniature of the same topology
// with payloads small enough to build in a test.
//
// ─── WHY THE SCHEDULE IS `[0, 0, 1, 0, 1]` AND NOT `idx % 4 == 3` ────────────
// This is the #2177 trap made expressible. The published checkpoint's own
// schedule happens to be `idx % 4 == 3` over its 45 model layers, so a reader
// that SYNTHESIZES that stride and a reader that READS the file agree on the
// only artifact that exists — and disagree silently on any other. A fixture at
// the published stride therefore cannot tell the two apart. This one puts the
// single DSA layer at index 2, where the stride would put a KDA layer, so a
// synthesizing loader builds the wrong kind of block on two of four layers and
// fails on the first tensor name it looks up.
//
// ─── WHY BLOCK 4 EXISTS ──────────────────────────────────────────────────────
// `block_count` is 5 and `nextn_predict_layers` is 1, so the backbone is 4
// layers deep and `blk.4` is the multi-token-prediction block the reference
// discards (`_keys_to_ignore_on_load_unexpected`). It is WRITTEN into the
// fixture, complete with its `nextn.*` tensors, because an exclusion cannot be
// tested against a file that has nothing to exclude: `layers.size() == 4` is
// equally true of a stack built from blocks 0..3 and one built from 1..4.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/glm5_next_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vt/dtype.h"

namespace glm5_next_fixture {

// ── the miniature's geometry ─────────────────────────────────────────────────
//
// Every width below is a real field of the published checkpoint scaled down,
// and the two that are NOT scaled are the two whose values are load-bearing:
// `kHcMult` is 4 because `mix = (2 + hc_mult) * hc_mult` is nonlinear in it, and
// `kKpool` is 4 because it is the published value against a class default of 16.
constexpr int64_t kH = 32;         // hidden_size
constexpr int64_t kVocab = 32;
constexpr int64_t kBlocks = 5;     // block_count, INCLUDING the MTP block
constexpr int64_t kMtpBlocks = 1;  // nextn_predict_layers
constexpr int64_t kLayers = kBlocks - kMtpBlocks;  // 4 backbone layers

constexpr int64_t kDenseInter = 64;  // feed_forward_length (layer 0 only)

// KDA
constexpr int64_t kKdaHeads = 2;
constexpr int64_t kKdaHeadDim = 16;
constexpr int64_t kKdaQkv = kKdaHeads * kKdaHeadDim;  // 32
constexpr int64_t kConvKernel = 4;

// MLA. `qk_rope_head_dim` is ZERO — this architecture is fully NoPE — so
// `key_length == kv_lora_rank` and `key_length_mla == qk_nope_head_dim`.
constexpr int64_t kHeads = 2;      // attention.head_count
constexpr int64_t kQLora = 32;
constexpr int64_t kKvLora = 32;
constexpr int64_t kQkNope = 16;
constexpr int64_t kVHead = 16;

// The DSA indexer, including the k-pool stage.
constexpr int64_t kIdxHeads = 2;
constexpr int64_t kIdxHeadDim = 16;
constexpr int64_t kIdxTopk = 8;
constexpr int64_t kKpool = 4;  // `index_topk % index_kpool == 0` is validated

// mHC
constexpr int64_t kHcMult = 4;
constexpr int64_t kStream = kHcMult * kH;             // 128
constexpr int64_t kHcMix = (2 + kHcMult) * kHcMult;   // 24

// MoE
constexpr int64_t kExperts = 4;
constexpr int64_t kExpertsPerTok = 2;
constexpr int64_t kMoeI = 32;
constexpr int64_t kSharedExperts = 1;

// The per-block attention schedule, read out of `attention.head_count_kv`:
// 0 is a KDA `linear_attention` block, non-zero a DSA/MLA one. Block 4 is the
// MTP block and is MLA-shaped, exactly as the published artifact's entry 45 is.
inline constexpr int32_t kHeadCountKv[kBlocks] = {0, 0, 1, 0, 1};

inline bool IsKda(int64_t block) { return kHeadCountKv[block] == 0; }
// Layer 0 is dense and the rest are sparse, which mirrors the published
// checkpoint's `["dense"] * 3 + ["sparse"] * 42` at a depth where three dense
// layers would leave only one sparse one.
inline bool IsDense(int64_t block) { return block == 0; }

inline std::string Blk(int64_t l, const char* suffix) {
  return "blk." + std::to_string(l) + "." + suffix;
}

// ── deterministic payloads ───────────────────────────────────────────────────

inline std::string F32Bytes(const std::vector<float>& v) {
  std::string s(v.size() * 4, '\0');
  std::memcpy(s.data(), v.data(), v.size() * 4);
  return s;
}

// A distinguishable value per element: no two positions of any tensor share a
// value, so a permutation or a cross-wiring cannot hide behind a repeated
// number. `base` is per tensor.
inline std::vector<float> Ramp(int64_t n, float base) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = base + static_cast<float>(i);
  return v;
}

inline std::string RampF32(int64_t n, float base) {
  return F32Bytes(Ramp(n, base));
}

// A NORM gamma, and it gets its own generator for a measured reason. Every
// gamma in this loader is stored bf16, whose step is 16 by the time a plain
// ramp reaches 3001 — so a value defect smaller than 16 would round away and
// the comparison would be testing bf16 rather than the loader. Every value here
// is `1 + k/128` with `k` in [0, 127], which bf16 represents EXACTLY. `tag`
// gives each tensor its own sequence, so a cross-wired pair (`q_a_norm` read
// into `kv_a_norm`) is visible.
inline float NormValue(int64_t i, int64_t tag) {
  return 1.0F + static_cast<float>((i + 13 * tag) % 128) / 128.0F;
}

inline std::string NormF32(int64_t n, int64_t tag) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = NormValue(i, tag);
  return F32Bytes(v);
}

// `ssm_a` holds `-exp(A_log)` (llama.cpp #27752 `conversion/glm5next.py`), so
// every entry is STRICTLY NEGATIVE and the loader recovers `A_log = log(-x)`.
// The values are chosen so the recovered `A_log` is exactly representable in
// f32 after a round trip through `exp`/`log` to within the tolerance the suite
// asserts, and so that no two heads share one.
inline float SsmAValue(int64_t head, int64_t layer) {
  return -std::exp(static_cast<float>(layer) - 0.25F * static_cast<float>(head));
}

inline std::string SsmABytes(int64_t heads, int64_t layer) {
  std::vector<float> v(static_cast<size_t>(heads));
  for (int64_t i = 0; i < heads; ++i)
    v[static_cast<size_t>(i)] = SsmAValue(i, layer);
  return F32Bytes(v);
}

// Q8_0 payload for `rows x cols` — `cols / 32` blocks per row, encoded the way
// `DequantGgufRowToF32` reads it back: an f16 scale then 32 int8 codes. Used
// for the three stacked expert banks, so the keep-quant residency arm is
// EXERCISED rather than described: an all-F32 fixture routes every tensor to
// `kExpandBf16` and the 3-D reshape the keep-quant arm does would never run.
inline std::string Q8_0Bytes(int64_t rows, int64_t cols, int64_t tag) {
  const int64_t blocks = rows * (cols / 32);
  std::string s(static_cast<size_t>(blocks) * 34, '\0');
  auto* p = reinterpret_cast<uint8_t*>(s.data());
  for (int64_t b = 0; b < blocks; ++b) {
    const uint16_t half = vt::F32ToF16(0.5F);
    std::memcpy(p + b * 34, &half, 2);
    for (int64_t i = 0; i < 32; ++i) {
      p[b * 34 + 2 + i] =
          static_cast<uint8_t>(static_cast<int8_t>((b + i + tag) % 100 - 50));
    }
  }
  return s;
}

// The f32 value `Q8_0Bytes` decodes to at flat element `i` of a `rows x cols`
// tensor, so a value assertion has something to compare against.
inline float Q8_0ValueAt(int64_t i, int64_t cols, int64_t tag) {
  const int64_t b = i / 32;
  const int64_t j = i % 32;
  (void)cols;
  return 0.5F * static_cast<float>(static_cast<int8_t>((b + j + tag) % 100 - 50));
}

// ── the synthetic file ───────────────────────────────────────────────────────

// `drop` names a tensor to OMIT and `bad_shape` one to write at a wrong shape,
// so the refusal cases enter through the SAME builder the happy path does. A
// second builder would be free to disagree with this one, and then the refusal
// cases would be testing the second builder.
struct FixtureOpts {
  std::string drop;
  std::string bad_shape;
  // Write `ssm_a` on layer 0 as a POSITIVE value — what a converter that
  // skipped llama.cpp #27752's `-torch.exp` would emit. `log(-x)` is then
  // undefined and the loader must refuse by name rather than store a NaN.
  bool positive_ssm_a = false;
  // Omit `output.weight`, which is how llama.cpp's writer states a TIED head.
  bool tie_lm_head = false;
};

namespace detail {

inline void Add(gguf_test::GgufModelBuilder& b, const FixtureOpts& o,
                const std::string& name, const std::vector<uint64_t>& ne_dims,
                uint32_t ggml_type, const std::string& data) {
  if (name == o.drop) return;
  if (name == o.bad_shape) {
    // One extra element on the innermost axis: enough to be a different
    // tensor, small enough that the payload stays valid for the new shape.
    std::vector<uint64_t> bad = ne_dims;
    bad[0] += 1;
    uint64_t n = 1;
    for (uint64_t d : bad) n *= d;
    b.AddTensor(name, bad, 0, std::string(static_cast<size_t>(n) * 4, '\0'));
    return;
  }
  b.AddTensor(name, ne_dims, ggml_type, data);
}

// GGUF `ne` order is the REVERSE of the torch shape this project reads back, so
// every call site below states the torch shape and this reverses it once.
inline std::vector<uint64_t> Ne(std::vector<int64_t> torch_shape) {
  std::vector<uint64_t> ne;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it)
    ne.push_back(static_cast<uint64_t>(*it));
  return ne;
}

inline void AddF32(gguf_test::GgufModelBuilder& b, const FixtureOpts& o,
                   const std::string& name, std::vector<int64_t> shape,
                   float base) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  Add(b, o, name, Ne(shape), 0, RampF32(n, base));
}

inline void AddNorm(gguf_test::GgufModelBuilder& b, const FixtureOpts& o,
                    const std::string& name, int64_t n, int64_t tag) {
  Add(b, o, name, Ne({n}), 0, NormF32(n, tag));
}

}  // namespace detail

// A per-tensor base value, so no two tensors in the file share a ramp and a
// tensor read into the wrong field is visible in the first element.
inline float Base(int64_t layer, int64_t slot) {
  return 1.0F + 1000.0F * static_cast<float>(layer) +
         10.0F * static_cast<float>(slot);
}

// A per-norm tag, disjoint from every other norm in the same layer.
inline int64_t NormTag(int64_t layer, int64_t slot) {
  return 1 + 16 * layer + slot;
}

inline std::string BuildFixture(const FixtureOpts& o = FixtureOpts{}) {
  using gguf_test::BoolKv;
  using gguf_test::F32ArrayKv;
  using gguf_test::F32Kv;
  using gguf_test::I32ArrayKv;
  using gguf_test::StrArrayKv;
  using gguf_test::StrKv;
  using gguf_test::U32Kv;
  using detail::AddF32;
  using detail::AddNorm;
  using detail::Ne;

  gguf_test::GgufModelBuilder b;
  const std::string k = "glm5next.";
  b.AddKv(StrKv("general.architecture", "glm5next"));
  b.AddKv(U32Kv(k + "vocab_size", static_cast<uint32_t>(kVocab)));
  b.AddKv(U32Kv(k + "context_length", 4096));
  b.AddKv(U32Kv(k + "embedding_length", static_cast<uint32_t>(kH)));
  b.AddKv(U32Kv(k + "block_count", static_cast<uint32_t>(kBlocks)));
  b.AddKv(U32Kv(k + "nextn_predict_layers", static_cast<uint32_t>(kMtpBlocks)));
  b.AddKv(U32Kv(k + "feed_forward_length", static_cast<uint32_t>(kDenseInter)));
  b.AddKv(U32Kv(k + "expert_feed_forward_length", static_cast<uint32_t>(kMoeI)));
  b.AddKv(U32Kv(k + "expert_shared_feed_forward_length",
                static_cast<uint32_t>(kMoeI)));
  b.AddKv(U32Kv(k + "expert_count", static_cast<uint32_t>(kExperts)));
  b.AddKv(U32Kv(k + "expert_used_count", static_cast<uint32_t>(kExpertsPerTok)));
  b.AddKv(U32Kv(k + "expert_shared_count", static_cast<uint32_t>(kSharedExperts)));
  b.AddKv(U32Kv(k + "expert_group_count", 1));
  b.AddKv(U32Kv(k + "expert_group_used_count", 1));
  b.AddKv(F32Kv(k + "expert_weights_scale", 2.5F));
  b.AddKv(BoolKv(k + "expert_weights_norm", true));
  b.AddKv(U32Kv(k + "attention.head_count", static_cast<uint32_t>(kHeads)));
  b.AddKv(I32ArrayKv(k + "attention.head_count_kv",
                     std::vector<int32_t>(kHeadCountKv,
                                          kHeadCountKv + kBlocks)));
  b.AddKv(F32Kv(k + "attention.layer_norm_rms_epsilon", 1e-5F));
  b.AddKv(U32Kv(k + "attention.q_lora_rank", static_cast<uint32_t>(kQLora)));
  b.AddKv(U32Kv(k + "attention.kv_lora_rank", static_cast<uint32_t>(kKvLora)));
  // llama.cpp's MLA convention: `key_length = kv_lora_rank + qk_rope_head_dim`,
  // `value_length = kv_lora_rank`, `key_length_mla = qk_nope + qk_rope`,
  // `value_length_mla = v_head_dim` (b10451:conversion/deepseek.py:345-348).
  b.AddKv(U32Kv(k + "attention.key_length", static_cast<uint32_t>(kKvLora)));
  b.AddKv(U32Kv(k + "attention.value_length", static_cast<uint32_t>(kKvLora)));
  b.AddKv(U32Kv(k + "attention.key_length_mla", static_cast<uint32_t>(kQkNope)));
  b.AddKv(U32Kv(k + "attention.value_length_mla", static_cast<uint32_t>(kVHead)));
  b.AddKv(U32Kv(k + "rope.dimension_count", 0));
  b.AddKv(U32Kv(k + "attention.indexer.head_count",
                static_cast<uint32_t>(kIdxHeads)));
  b.AddKv(U32Kv(k + "attention.indexer.key_length",
                static_cast<uint32_t>(kIdxHeadDim)));
  b.AddKv(U32Kv(k + "attention.indexer.top_k", static_cast<uint32_t>(kIdxTopk)));
  b.AddKv(U32Kv(k + "attention.indexer.kpool", static_cast<uint32_t>(kKpool)));
  b.AddKv(BoolKv(k + "attention.indexer.kpool_always_select_tail", true));
  b.AddKv(U32Kv(k + "kda.head_dim", static_cast<uint32_t>(kKdaHeadDim)));
  b.AddKv(U32Kv(k + "ssm.group_count", static_cast<uint32_t>(kKdaHeads)));
  b.AddKv(U32Kv(k + "ssm.conv_kernel", static_cast<uint32_t>(kConvKernel)));
  b.AddKv(F32Kv(k + "kda.gate_lower_bound", -5.0F));
  b.AddKv(U32Kv(k + "hyper_connection.count", static_cast<uint32_t>(kHcMult)));
  b.AddKv(U32Kv(k + "hyper_connection.sinkhorn_iterations", 20));
  b.AddKv(F32Kv(k + "hyper_connection.epsilon", 1e-6F));
  {
    std::vector<std::string> mlp;
    for (int64_t i = 0; i < kBlocks; ++i)
      mlp.push_back(IsDense(i) ? "dense" : "sparse");
    b.AddKv(StrArrayKv(k + "mlp_layer_types", mlp));
    // The published artifact states the SwiGLU clamp as a per-block f32 array,
    // not as a scalar, and states it twice under two keys that must agree.
    b.AddKv(F32ArrayKv(k + "swiglu_clamp_exp",
                       std::vector<float>(kBlocks, 10.0F)));
    b.AddKv(F32ArrayKv(k + "swiglu_clamp_shexp",
                       std::vector<float>(kBlocks, 10.0F)));
  }

  // ── model level ────────────────────────────────────────────────────────────
  AddF32(b, o, "token_embd.weight", {kVocab, kH}, 1.0F);
  AddNorm(b, o, "output_norm.weight", kH, /*tag=*/101);
  if (!o.tie_lm_head) AddF32(b, o, "output.weight", {kVocab, kH}, 5000.0F);

  // ── every block, INCLUDING the MTP one ─────────────────────────────────────
  for (int64_t L = 0; L < kBlocks; ++L) {
    AddNorm(b, o, Blk(L, "attn_norm.weight"), kH, NormTag(L, 0));
    AddNorm(b, o, Blk(L, "ffn_norm.weight"), kH, NormTag(L, 1));
    // The mHC pair. `blk.4` — the MTP block — carries NONE of these in the
    // published artifact, and this fixture mirrors that: the MTP block's
    // absence of hyper-connection parameters is itself a structural fact a
    // loader that built it would trip over.
    if (L < kLayers) {
      for (const char* side : {"attn", "ffn"}) {
        const std::string pfx = std::string("hc_") + side + "_";
        const int64_t slot = (side[0] == 'a') ? 2 : 5;
        AddF32(b, o, Blk(L, (pfx + "fn.weight").c_str()), {kHcMix, kStream},
               Base(L, slot));
        AddF32(b, o, Blk(L, (pfx + "base.weight").c_str()), {kHcMix},
               Base(L, slot + 1));
        AddF32(b, o, Blk(L, (pfx + "scale.weight").c_str()), {3},
               Base(L, slot + 2));
      }
    }

    if (IsKda(L)) {
      AddF32(b, o, Blk(L, "attn_q.weight"), {kKdaQkv, kH}, Base(L, 8));
      AddF32(b, o, Blk(L, "attn_k.weight"), {kKdaQkv, kH}, Base(L, 9));
      AddF32(b, o, Blk(L, "attn_v.weight"), {kKdaQkv, kH}, Base(L, 10));
      AddF32(b, o, Blk(L, "attn_output.weight"), {kH, kKdaQkv}, Base(L, 11));
      // `nn.Conv1d` stores `[C, 1, K]`, and the middle axis is REAL in the
      // file: a loader that expected `[C, K]` reads a shape mismatch.
      AddF32(b, o, Blk(L, "ssm_conv1d_q.weight"), {kKdaQkv, 1, kConvKernel},
             Base(L, 12));
      AddF32(b, o, Blk(L, "ssm_conv1d_k.weight"), {kKdaQkv, 1, kConvKernel},
             Base(L, 13));
      AddF32(b, o, Blk(L, "ssm_conv1d_v.weight"), {kKdaQkv, 1, kConvKernel},
             Base(L, 14));
      AddF32(b, o, Blk(L, "ssm_f_a.weight"), {kKdaHeadDim, kH}, Base(L, 15));
      AddF32(b, o, Blk(L, "ssm_f_b.weight"), {kKdaQkv, kKdaHeadDim}, Base(L, 16));
      AddF32(b, o, Blk(L, "ssm_g_a.weight"), {kKdaHeadDim, kH}, Base(L, 17));
      AddF32(b, o, Blk(L, "ssm_g_b.weight"), {kKdaQkv, kKdaHeadDim}, Base(L, 18));
      // ONE ROW PER HEAD, not per channel.
      AddF32(b, o, Blk(L, "ssm_beta.weight"), {kKdaHeads, kH}, Base(L, 19));
      if (o.positive_ssm_a && L == 0) {
        AddF32(b, o, Blk(L, "ssm_a"), {kKdaHeads}, 1.0F);
      } else {
        detail::Add(b, o, Blk(L, "ssm_a"), Ne({kKdaHeads}), 0,
                    SsmABytes(kKdaHeads, L));
      }
      AddF32(b, o, Blk(L, "ssm_dt.bias"), {kKdaQkv}, Base(L, 20));
      AddNorm(b, o, Blk(L, "ssm_norm.weight"), kKdaHeadDim, NormTag(L, 2));
    } else {
      AddF32(b, o, Blk(L, "attn_q_a.weight"), {kQLora, kH}, Base(L, 8));
      AddNorm(b, o, Blk(L, "attn_q_a_norm.weight"), kQLora, NormTag(L, 3));
      AddF32(b, o, Blk(L, "attn_q_b.weight"), {kHeads * kQkNope, kQLora},
             Base(L, 9));
      AddF32(b, o, Blk(L, "attn_kv_a_mqa.weight"), {kKvLora, kH}, Base(L, 10));
      AddNorm(b, o, Blk(L, "attn_kv_a_norm.weight"), kKvLora, NormTag(L, 4));
      // The two absorbed halves. `attn_k_b` is TRANSPOSED, so its trailing
      // extent is `qk_nope_head_dim` while `attn_v_b`'s is `kv_lora_rank`.
      AddF32(b, o, Blk(L, "attn_k_b.weight"), {kHeads, kKvLora, kQkNope},
             Base(L, 11));
      AddF32(b, o, Blk(L, "attn_v_b.weight"), {kHeads, kVHead, kKvLora},
             Base(L, 12));
      AddF32(b, o, Blk(L, "attn_output.weight"), {kH, kHeads * kVHead},
             Base(L, 13));
      AddF32(b, o, Blk(L, "indexer.attn_q_b.weight"),
             {kIdxHeads * kIdxHeadDim, kQLora}, Base(L, 14));
      AddF32(b, o, Blk(L, "indexer.attn_k.weight"), {kIdxHeadDim, kH},
             Base(L, 15));
      AddNorm(b, o, Blk(L, "indexer.k_norm.weight"), kIdxHeadDim, NormTag(L, 5));
      AddNorm(b, o, Blk(L, "indexer.k_norm.bias"), kIdxHeadDim, NormTag(L, 6));
      AddF32(b, o, Blk(L, "indexer.proj.weight"), {kIdxHeads, kH}, Base(L, 16));
      AddF32(b, o, Blk(L, "indexer_compressor_ape.weight"),
             {kKpool, kIdxHeadDim}, Base(L, 17));
      AddF32(b, o, Blk(L, "indexer_compressor_gate.weight"), {kIdxHeadDim, kH},
             Base(L, 18));
    }

    if (IsDense(L)) {
      AddF32(b, o, Blk(L, "ffn_gate.weight"), {kDenseInter, kH}, Base(L, 21));
      AddF32(b, o, Blk(L, "ffn_up.weight"), {kDenseInter, kH}, Base(L, 22));
      AddF32(b, o, Blk(L, "ffn_down.weight"), {kH, kDenseInter}, Base(L, 23));
    } else {
      AddF32(b, o, Blk(L, "ffn_gate_inp.weight"), {kExperts, kH}, Base(L, 24));
      AddF32(b, o, Blk(L, "exp_probs_b.bias"), {kExperts}, Base(L, 25));
      // The three stacked banks, at Q8_0 so the keep-quant arm runs.
      detail::Add(b, o, Blk(L, "ffn_gate_exps.weight"),
                  Ne({kExperts, kMoeI, kH}), 8,
                  Q8_0Bytes(kExperts * kMoeI, kH, 10 * L + 1));
      detail::Add(b, o, Blk(L, "ffn_up_exps.weight"), Ne({kExperts, kMoeI, kH}),
                  8, Q8_0Bytes(kExperts * kMoeI, kH, 10 * L + 2));
      detail::Add(b, o, Blk(L, "ffn_down_exps.weight"),
                  Ne({kExperts, kH, kMoeI}), 8,
                  Q8_0Bytes(kExperts * kH, kMoeI, 10 * L + 3));
      AddF32(b, o, Blk(L, "ffn_gate_shexp.weight"), {kMoeI, kH}, Base(L, 26));
      AddF32(b, o, Blk(L, "ffn_up_shexp.weight"), {kMoeI, kH}, Base(L, 27));
      AddF32(b, o, Blk(L, "ffn_down_shexp.weight"), {kH, kMoeI}, Base(L, 28));
    }

    // The MTP block's own four tensors, which no backbone layer has. They are
    // written so the fixture's `blk.4` is a real MTP block and not a decoder
    // layer wearing the number.
    if (L >= kLayers) {
      AddF32(b, o, Blk(L, "nextn.eh_proj.weight"), {2 * kH, kH}, Base(L, 29));
      AddNorm(b, o, Blk(L, "nextn.enorm.weight"), kH, NormTag(L, 7));
      AddNorm(b, o, Blk(L, "nextn.hnorm.weight"), kH, NormTag(L, 8));
      AddNorm(b, o, Blk(L, "nextn.shared_head_norm.weight"), kH, NormTag(L, 9));
    }
  }
  return b.Build();
}

// THE PRODUCTION ENTRY POINT, reached the way a user reaches it: the GGUF
// architecture dispatch builds the config, the registry resolves the
// architecture from it, and the registration's own `load_weights` hook runs.
// Nothing in the suite constructs a `Glm5NextWeights` by hand.
//
// Calling `LoadGlm5NextFromGguf` directly would skip `ModelRegistry::Resolve`
// and the registry's factory, and that skip is exactly what would hide a
// registration this wave never wired.
inline std::unique_ptr<vllm::LoadedModel> LoadThroughRegistry(
    const vllm::GgufFile& g) {
  const vllm::HfConfig config = vllm::Glm5NextHfConfigFromGguf(g);
  const vllm::ModelSource source = vllm::ModelSource::FromGguf(g);
  return vllm::ModelRegistry::Load(config, source);
}

}  // namespace glm5_next_fixture
