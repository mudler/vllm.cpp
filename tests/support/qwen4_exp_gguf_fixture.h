// The synthetic tiny `qwen4exp` GGUF, shared by every suite that needs a REAL
// `Qwen4ExpLoadedModel` rather than a hand-built `Qwen4ExpWeights`.
//
// EXTRACTED VERBATIM from `tests/vllm/models/test_qwen4_exp_gguf_weights.cpp`
// (W5a, #2031) when the W5b layer loop needed the same file. The extraction is
// not tidiness: the loader suite chose every dimension below to make a specific
// defect expressible — `kNumKHeads` 2 against `kNumVHeads` 6 so the V-head
// permutation is not its own inverse, `kPleRow` 96 so `ple_embed_dim` is
// distinct from both `hidden_size` and `hidden_size * ngram_heads`, gammas on a
// bf16-exact `1 + k/128` grid so the convert-time `+1` fold cannot round away —
// and a SECOND builder would be free to disagree with all of it. The pair that
// must agree here is exactly the pair a copy would let diverge: the suite that
// gates what the loader PRODUCED and the suite that gates what the forward
// CONSUMES.
//
// Everything is `inline` in one namespace, so both suites link one definition.
// No case lives here — a fixture that asserts is a suite.
//
// THE GAMMA POLARITY THIS FIXTURE CARRIES IS THE FILE'S, NOT THE LOADER'S.
// `NormValue` returns `1 + k/128`, centred on 1.0, because that is what
// ggml-org/llama.cpp#27742 writes: every tensor whose name ends in
// `norm.weight` is stored with the `+1` fold applied, `linear_attn.norm.weight`
// excepted. `LoadNormBf16(..., unshift=true)` inverts it, so what a loaded
// `Qwen4ExpWeights` holds is the RAW HuggingFace gamma centred on 0. A test
// that wants the value the FILE carried must call `NormValue` and a test that
// wants the value the MODEL holds must subtract one. Composing the two is what
// #2218 was about.
#pragma once

#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_gguf_weights.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"
#include "vt/quant.h"

namespace qwen4_exp_fixture {

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::I32ArrayKv;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

// ── the tiny fixture geometry ────────────────────────────────────────────────
//
// Every dimension is the smallest one that keeps a STRUCTURE the released
// config has and that a smaller value would erase:
//
//   * `kNumKHeads` is 2, not 1. The V-head reorder maps grouped head `k*R + r`
//     to tiled head `r*K + k`; at K = 1 that is the identity, so a fixture with
//     one key head cannot tell a correct un-reorder from no un-reorder at all.
//   * `kLayers` is 4 with `full_attention_interval` 4, which is the released
//     3-linear-then-1-sparse pattern at its shortest: layers 0..2 are Gated
//     DeltaNet and layer 3 is QSA, so both arms of the per-layer branch run.
//   * `kHeadsPerNgram` is 1, and `kPleRow` (below) is a multiple of 32, so
//     `head_dim_per_ngram` is 96 — three whole Q8_0 blocks. The n-gram table is
//     the ONE gather this model keeps quantized (W6a, #1989), and a table whose
//     row is not a whole number of blocks could not exercise that at all.
constexpr int64_t kH = 64;      // hidden_size
constexpr int64_t kLayers = 4;  // 0,1,2 linear_attention; 3 qwen_sparse_attention
constexpr int64_t kVocab = 16;
constexpr int64_t kHcCount = 2;
constexpr int64_t kHcLowrank = 8;
constexpr int64_t kStream = kHcCount * kH;  // the residual stream width, 128
constexpr int64_t kExperts = 2;
constexpr int64_t kExpertsPerTok = 1;
constexpr int64_t kMoeI = 8;
constexpr int64_t kSharedI = 8;
constexpr int64_t kQHeads = 2;
constexpr int64_t kKvHeads = 1;
constexpr int64_t kHeadDim = 8;
constexpr int64_t kRotaryDim = 4;
constexpr int64_t kIdxHeads = 2;
constexpr int64_t kIdxKvHeads = 1;
constexpr int64_t kIdxHeadDim = 8;
constexpr int64_t kIdxBudget = 8;
constexpr int64_t kCompressRatio = 4;
constexpr int64_t kNumKHeads = 2;   // linear_num_key_heads
// SIX, not four, and the reason is a mutation this gate failed before it was
// six. The V-head reorder maps grouped head `k*R + r` to tiled head `r*K + k`.
// At K == R that permutation is its OWN INVERSE, so a loader that applied the
// map in the wrong direction produced byte-identical output and the whole
// reorder suite stayed green (mutation M5). K = 2 with R = 3 is the smallest
// pair where the map and its inverse differ, and it is also the released
// model's own ratio: 16 key heads to 48 value heads is R = 3.
constexpr int64_t kNumVHeads = 6;   // linear_num_value_heads
constexpr int64_t kLinHeadDim = 8;  // linear_{key,value}_head_dim
constexpr int64_t kConvKernel = 4;
constexpr int64_t kNgramSize = 3;
constexpr int64_t kHeadsPerNgram = 1;
constexpr int64_t kPleLayer = 1;  // 0-based, and a linear_attention layer
constexpr int64_t kEosTokenId = 3;

constexpr int64_t kKeyDim = kNumKHeads * kLinHeadDim;    // 16
constexpr int64_t kValueDim = kNumVHeads * kLinHeadDim;  // 48
constexpr int64_t kConvDim = 2 * kKeyDim + kValueDim;    // 80
constexpr int64_t kNgramHeads = (kNgramSize - 1) * kHeadsPerNgram;  // 2
// 96, and it is DELIBERATELY NEITHER `kH / kNgramHeads` NOR `kH`. This is the
// fixture shape that gates `ple_embed_dim`, and neither value it replaced could.
//
// The GGUF states the PER-HEAD row width and HF states the TOTAL; the builder
// reconstructs the total as `ple_row * ngram_heads`, and `ParseQwen4ExpParams`
// falls back to `hidden_size` when the total is absent. On the RELEASED config
// those two happen to coincide (160 * 16 == 2560 == hidden_size), which is the
// coincidence #2064 was filed about. A fixture that DEFINES `kPleRow` as
// `kH / kNgramHeads` reproduces that coincidence by construction, so deleting
// the builder's `text["ple_embed_dim"]` line left the whole suite green
// (mutation MUT-C).
//
// 64 broke MUT-C but left a SECOND coincidence standing, because `kH` is also
// 64: a builder that wrote `hidden_size * ngram_heads` instead of
// `ple_row * ngram_heads` still produced 128, the correct total, and that
// mutation survived the whole suite (MUT-D). At 96 the correct total is 192,
// the `hidden_size` product is 128 and the bare `hidden_size` fallback is 64,
// so all three are distinct and each wrong one refuses the file by shape —
// which is what makes the builder's line observable at all.
//
// 96 rather than any other triply-distinct value because
// `head_dim_per_ngram() == kPleEmbedDim / kNgramHeads` must stay a whole number
// of Q8_0 blocks: the n-gram table is the one gather this model keeps
// quantized, and a ragged row cannot be kept at all. 96 is three blocks, and it
// is the smallest multiple of 32 that is neither `kH` nor `kH / kNgramHeads`.
constexpr int64_t kPleRow = 96;
// The TOTAL width, HF's own `ple_embed_dim`. 192 != kH, which is the point.
constexpr int64_t kPleEmbedDim = kPleRow * kNgramHeads;  // 192
static_assert(kPleEmbedDim != kH,
              "the fixture must not reproduce the released checkpoint's "
              "ple_embed_dim == hidden_size coincidence (#2064)");
static_assert(kPleEmbedDim != kH * kNgramHeads,
              "the fixture must not let `hidden_size * ngram_heads` stand in "
              "for `ple_row * ngram_heads` (#2064)");
static_assert(kPleRow % 32 == 0, "an n-gram row must be whole Q8_0 blocks");
// The two head vocabularies the fixture STATES, the way a real `qwen4exp` file
// does (`qwen4exp.ple.head_vocab_sizes`). Their sum is 52 and
// `make_ngram_vocab_size_divisible_by` defaults to 128, so the padded table is
// 128 rows. 23 and 29 are the successive primes after 19, which is what the HF
// derivation would produce from `ngram_vocab_size_base = 20` — stated here so
// the two routes into `NgramTableRows` are the same arithmetic on a small
// config, and the released-config case gates them at 320001536.
constexpr int64_t kNgramHead0Vocab = 23;
constexpr int64_t kNgramHead1Vocab = 29;
constexpr int64_t kNgramRows = 128;

// One `tag` per NORM tensor, so a cross-wired pair reads a different sequence.
// The per-layer ones are offset by layer as well, so a loader that read layer 0's
// gamma into layer 3 would be visible too.
constexpr int64_t kMixerNormTag = 1;
constexpr int64_t kQNormTag = 2;
constexpr int64_t kKNormTag = 3;
constexpr int64_t kIdxQNormTag = 4;
constexpr int64_t kIdxKNormTag = 5;
constexpr int64_t kPleNormKeyTag = 6;
constexpr int64_t kPleNormQueryTag = 7;
constexpr int64_t kPleNormConvTag = 8;
inline int64_t HcNormTag(int64_t layer, const char* side) {
  return 10 + 2 * layer + (side[0] == 'a' ? 0 : 1);
}
inline int64_t SsmNormTag(int64_t layer) { return 30 + layer; }

inline std::string Blk(int64_t l, const char* suffix) {
  return "blk." + std::to_string(l) + "." + suffix;
}

inline bool IsLinear(int64_t l) { return ((l + 1) % 4) != 0; }

// ── deterministic payloads ───────────────────────────────────────────────────

inline std::string F32Bytes(const std::vector<float>& v) {
  std::string s(v.size() * 4, '\0');
  std::memcpy(s.data(), v.data(), v.size() * 4);
  return s;
}

// A distinguishable value per element: no two positions of any tensor share a
// value, so a permutation defect cannot hide behind a repeated number.
inline std::vector<float> Ramp(int64_t n, float base) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = base + static_cast<float>(i);
  return v;
}

inline std::string RampF32(int64_t n, float base) { return F32Bytes(Ramp(n, base)); }

// NORM gammas get their own generator, and the reason is a measurement rather
// than tidiness. The `+1` fold this loader inverts is a subtraction of ONE, and
// bf16's step is 16 by the time a plain ramp reaches 3001 — so on a gamma
// written as `3001 + i` the fold and its absence round to the SAME bf16 value
// and the check passes either way. Every value here is `1 + k/128` with
// `k` in [0, 127], which bf16 represents exactly, and so is `k/128` after the
// fold is removed. `tag` gives each tensor its own sequence so a cross-wired
// pair (norm_key read into norm_query) is visible.
inline float NormValue(int64_t i, int64_t tag) {
  return 1.0F + static_cast<float>((i + 13 * tag) % 128) / 128.0F;
}

inline std::string NormF32(int64_t n, int64_t tag) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = NormValue(i, tag);
  return F32Bytes(v);
}

// Q8_0 payload for `rows x 32` — one block per row, encoded the way
// `DequantGgufRowToF32` reads it back: an f16 scale then 32 int8 codes.
inline std::string Q8_0Bytes(int64_t rows, int64_t cols) {
  REQUIRE(cols % 32 == 0);
  const int64_t blocks = rows * (cols / 32);
  std::string s(static_cast<size_t>(blocks) * 34, '\0');
  auto* p = reinterpret_cast<uint8_t*>(s.data());
  for (int64_t b = 0; b < blocks; ++b) {
    const uint16_t half = vt::F32ToF16(0.5F);
    std::memcpy(p + b * 34, &half, 2);
    for (int64_t i = 0; i < 32; ++i)
      p[b * 34 + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>((b + i) % 100 - 50));
  }
  return s;
}

// ── the synthetic file ───────────────────────────────────────────────────────

// `drop` names a tensor to OMIT and `bad_shape` one to write at a wrong shape,
// so the refusal cases enter through the same builder the happy path does. A
// second builder would be free to disagree with this one, and then the refusal
// cases would be testing the second builder.
struct FixtureOpts {
  std::string drop;
  std::string bad_shape;
  // W5c (#2031): make `attention.compress_ratios` DISAGREE between two sparse
  // layers. The file states the ratio per LAYER while HF states one value, so
  // the config builder takes the first non-zero and requires the rest to
  // match; a mixed schedule that silently first-wins would size the QSA
  // indexer side cache for one ratio while another layer compressed at a
  // different one.
  //
  // It DOUBLES `block_count`, and that is what makes the defect expressible at
  // all. The miniature is four layers at `full_attention_interval` 4, so it has
  // exactly ONE sparse layer and one non-zero ratio, which cannot disagree with
  // itself; and a stray non-zero on a LINEAR layer is caught one check earlier
  // by "compress_ratios disagrees with the full_attention_interval schedule".
  // Eight layers give two sparse ones, 3 and 7, so the array can be
  // schedule-consistent AND non-uniform. Only `Qwen4ExpHfConfigFromGguf` is
  // driven with this option — it reads metadata and never walks the per-layer
  // tensors, which stay at four layers.
  bool mixed_compress_ratios = false;
};

inline void Add(GgufModelBuilder& b, const FixtureOpts& o, const std::string& name,
         std::vector<uint64_t> ne, uint32_t ggml_type, const std::string& data) {
  if (name == o.drop) return;
  if (name == o.bad_shape) {
    // One extra row: a shape a reader that only checks rank would accept.
    ne.back() += 1;
    const int64_t elems_per_row =
        static_cast<int64_t>(ne.front());
    return b.AddTensor(name, ne, ggml_type,
                       data + std::string(static_cast<size_t>(elems_per_row) * 4, '\0'));
  }
  b.AddTensor(name, ne, ggml_type, data);
}

inline std::string BuildFixture(const FixtureOpts& o = {}) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen4exp"));
  b.AddKv(U32Kv("qwen4exp.embedding_length", kH));
  const int64_t layers_kv = o.mixed_compress_ratios ? kLayers * 2 : kLayers;
  b.AddKv(U32Kv("qwen4exp.block_count", layers_kv));
  b.AddKv(U32Kv("qwen4exp.attention.head_count", kQHeads));
  b.AddKv(U32Kv("qwen4exp.attention.head_count_kv", kKvHeads));
  b.AddKv(U32Kv("qwen4exp.attention.key_length", kHeadDim));
  b.AddKv(U32Kv("qwen4exp.attention.value_length", kHeadDim));
  b.AddKv(U32Kv("qwen4exp.context_length", 256));
  b.AddKv(F32Kv("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen4exp.rope.freq_base", 10000.0F));
  b.AddKv(U32Kv("qwen4exp.rope.dimension_count", kRotaryDim));
  b.AddKv(U32Kv("qwen4exp.expert_count", kExperts));
  b.AddKv(U32Kv("qwen4exp.expert_used_count", kExpertsPerTok));
  b.AddKv(U32Kv("qwen4exp.expert_feed_forward_length", kMoeI));
  b.AddKv(U32Kv("qwen4exp.expert_shared_feed_forward_length", kSharedI));
  b.AddKv(U32Kv("qwen4exp.ssm.group_count", kNumKHeads));
  b.AddKv(U32Kv("qwen4exp.ssm.time_step_rank", kNumVHeads));
  b.AddKv(U32Kv("qwen4exp.ssm.state_size", kLinHeadDim));
  b.AddKv(U32Kv("qwen4exp.ssm.conv_kernel", kConvKernel));
  b.AddKv(U32Kv("qwen4exp.ssm.inner_size", kValueDim));
  b.AddKv(U32Kv("qwen4exp.full_attention_interval", 4));
  b.AddKv(U32Kv("qwen4exp.hyper_connection.count", kHcCount));
  b.AddKv(U32Kv("qwen4exp.hyper_connection.low_rank", kHcLowrank));
  b.AddKv(U32Kv("qwen4exp.attention.indexer.head_count", kIdxHeads));
  b.AddKv(U32Kv("qwen4exp.attention.indexer.key_length", kIdxHeadDim));
  b.AddKv(U32Kv("qwen4exp.attention.indexer.top_k", kIdxBudget));
  b.AddKv(U32Kv("qwen4exp.embedding_length_per_layer_input", kPleRow));
  b.AddKv(U32Kv("qwen4exp.ple.ngram_size", kNgramSize));
  b.AddKv(U32Kv("qwen4exp.ple.heads_per_ngram", kHeadsPerNgram));
  b.AddKv(U32Kv("qwen4exp.ple.conv_kernel", kConvKernel));
  b.AddKv(U32Kv("qwen4exp.ple.eos_token_id", kEosTokenId));
  b.AddKv(I32ArrayKv("qwen4exp.ple.head_vocab_sizes",
                     {static_cast<int32_t>(kNgramHead0Vocab),
                      static_cast<int32_t>(kNgramHead1Vocab)}));
  b.AddKv(I32ArrayKv("qwen4exp.ple.head_offsets",
                     {0, static_cast<int32_t>(kNgramHead0Vocab)}));
  b.AddKv(I32ArrayKv("qwen4exp.ple.layers", {static_cast<int32_t>(kPleLayer)}));
  std::vector<int32_t> ratios;
  for (int64_t i = 0; i < layers_kv; ++i)
    ratios.push_back(IsLinear(i) ? 0 : static_cast<int32_t>(kCompressRatio));
  if (o.mixed_compress_ratios) {
    // The LAST sparse layer compresses at a different ratio from the first, so
    // the array still agrees with the schedule and no longer agrees with
    // itself.
    ratios.back() = static_cast<int32_t>(kCompressRatio) * 2;
  }
  b.AddKv(I32ArrayKv("qwen4exp.attention.compress_ratios", ratios));

  // Tensor dims are in GGUF `ne` order (inner/fastest dim first), which is the
  // REVERSE of the torch [out, in] order the reader hands back.
  Add(b, o, "token_embd.weight", {kH, kVocab}, 0, RampF32(kH * kVocab, 1.0F));
  Add(b, o, "output.weight", {kH, kVocab}, 0, RampF32(kH * kVocab, 2.0F));
  Add(b, o, "per_layer_token_embd.weight", {kPleRow, kNgramRows}, 8,
      Q8_0Bytes(kNgramRows, kPleRow));
  Add(b, o, "output_hc_norm.weight", {kStream}, 0,
      NormF32(kStream, kMixerNormTag));
  Add(b, o, "output_hc_down.weight", {kStream, kHcLowrank}, 0,
      RampF32(kStream * kHcLowrank, 3.0F));
  Add(b, o, "output_hc_up.weight", {kHcLowrank, kStream}, 0,
      RampF32(kStream * kHcLowrank, 4.0F));

  for (int64_t l = 0; l < kLayers; ++l) {
    const float base = static_cast<float>(l * 1000 + 1);
    for (const char* side : {"attn", "ffn"}) {
      const std::string p = std::string("hc_") + side + "_";
      Add(b, o, Blk(l, (p + "norm.weight").c_str()), {kStream}, 0,
          NormF32(kStream, HcNormTag(l, side)));
      Add(b, o, Blk(l, (p + "down.weight").c_str()), {kStream, kHcLowrank}, 0,
          RampF32(kStream * kHcLowrank, base));
      Add(b, o, Blk(l, (p + "up.weight").c_str()), {kHcLowrank, kStream}, 0,
          RampF32(kStream * kHcLowrank, base));
      Add(b, o, Blk(l, (p + "inject.weight").c_str()), {kStream, kHcCount}, 0,
          RampF32(kStream * kHcCount, base));
    }
    Add(b, o, Blk(l, "ffn_gate_inp.weight"), {kH, kExperts}, 0,
        RampF32(kH * kExperts, base));
    Add(b, o, Blk(l, "ffn_gate_inp_shexp.weight"), {kH}, 0, RampF32(kH, base));
    Add(b, o, Blk(l, "ffn_gate_exps.weight"), {kH, kMoeI, kExperts}, 0,
        RampF32(kH * kMoeI * kExperts, base));
    Add(b, o, Blk(l, "ffn_up_exps.weight"), {kH, kMoeI, kExperts}, 0,
        RampF32(kH * kMoeI * kExperts, base));
    Add(b, o, Blk(l, "ffn_down_exps.weight"), {kMoeI, kH, kExperts}, 0,
        RampF32(kH * kMoeI * kExperts, base));
    Add(b, o, Blk(l, "ffn_gate_shexp.weight"), {kH, kSharedI}, 0,
        RampF32(kH * kSharedI, base));
    Add(b, o, Blk(l, "ffn_up_shexp.weight"), {kH, kSharedI}, 0,
        RampF32(kH * kSharedI, base));
    Add(b, o, Blk(l, "ffn_down_shexp.weight"), {kSharedI, kH}, 0,
        RampF32(kH * kSharedI, base));

    if (IsLinear(l)) {
      Add(b, o, Blk(l, "attn_qkv.weight"), {kH, kConvDim}, 0,
          RampF32(kH * kConvDim, base));
      Add(b, o, Blk(l, "attn_gate.weight"), {kH, kValueDim}, 0,
          RampF32(kH * kValueDim, base));
      Add(b, o, Blk(l, "ssm_alpha.weight"), {kH, kNumVHeads}, 0,
          RampF32(kH * kNumVHeads, base));
      Add(b, o, Blk(l, "ssm_beta.weight"), {kH, kNumVHeads}, 0,
          RampF32(kH * kNumVHeads, base));
      Add(b, o, Blk(l, "ssm_conv1d.weight"), {kConvKernel, kConvDim}, 0,
          RampF32(kConvDim * kConvKernel, base));
      Add(b, o, Blk(l, "ssm_norm.weight"), {kLinHeadDim}, 0,
          NormF32(kLinHeadDim, SsmNormTag(l)));
      Add(b, o, Blk(l, "ssm_out.weight"), {kValueDim, kH}, 0,
          RampF32(kH * kValueDim, base));
      // `ssm_a` is stored as -exp(A_log); the loader recovers log(-x). Negative
      // by construction, and distinct per head.
      std::vector<float> a(static_cast<size_t>(kNumVHeads));
      for (int64_t i = 0; i < kNumVHeads; ++i)
        a[static_cast<size_t>(i)] = -static_cast<float>(i + 1);
      Add(b, o, Blk(l, "ssm_a"), {kNumVHeads}, 0, F32Bytes(a));
      Add(b, o, Blk(l, "ssm_dt.bias"), {kNumVHeads}, 0,
          RampF32(kNumVHeads, base));
    } else {
      Add(b, o, Blk(l, "attn_q.weight"), {kH, kQHeads * kHeadDim * 2}, 0,
          RampF32(kH * kQHeads * kHeadDim * 2, base));
      Add(b, o, Blk(l, "attn_k.weight"), {kH, kKvHeads * kHeadDim}, 0,
          RampF32(kH * kKvHeads * kHeadDim, base));
      Add(b, o, Blk(l, "attn_v.weight"), {kH, kKvHeads * kHeadDim}, 0,
          RampF32(kH * kKvHeads * kHeadDim, base));
      Add(b, o, Blk(l, "attn_output.weight"), {kQHeads * kHeadDim, kH}, 0,
          RampF32(kH * kQHeads * kHeadDim, base));
      Add(b, o, Blk(l, "attn_q_norm.weight"), {kHeadDim}, 0,
          NormF32(kHeadDim, kQNormTag));
      Add(b, o, Blk(l, "attn_k_norm.weight"), {kHeadDim}, 0,
          NormF32(kHeadDim, kKNormTag));
      Add(b, o, Blk(l, "indexer.q_proj.weight"), {kH, kIdxHeads * kIdxHeadDim},
          0, RampF32(kH * kIdxHeads * kIdxHeadDim, base));
      Add(b, o, Blk(l, "indexer.k_proj.weight"),
          {kH, kIdxKvHeads * kIdxHeadDim}, 0,
          RampF32(kH * kIdxKvHeads * kIdxHeadDim, base));
      Add(b, o, Blk(l, "indexer.q_norm.weight"), {kIdxHeadDim}, 0,
          NormF32(kIdxHeadDim, kIdxQNormTag));
      Add(b, o, Blk(l, "indexer.k_norm.weight"), {kIdxHeadDim}, 0,
          NormF32(kIdxHeadDim, kIdxKNormTag));
    }

    if (l == kPleLayer) {
      // [stream, ple_embed_dim] and [hidden_size, ple_embed_dim] in TORCH
      // order, so the GGUF `ne` is reversed. Both are ple_embed_dim wide and
      // NOT hidden_size wide, which is what MUT-C now runs into.
      Add(b, o, Blk(l, "ple_key.weight"), {kPleEmbedDim, kStream}, 0,
          RampF32(kPleEmbedDim * kStream, base));
      Add(b, o, Blk(l, "ple_value.weight"), {kPleEmbedDim, kH}, 0,
          RampF32(kPleEmbedDim * kH, base));
      Add(b, o, Blk(l, "ple_norm_key.weight"), {kStream}, 0,
          NormF32(kStream, kPleNormKeyTag));
      Add(b, o, Blk(l, "ple_norm_query.weight"), {kStream}, 0,
          NormF32(kStream, kPleNormQueryTag));
      Add(b, o, Blk(l, "ple_norm_conv.weight"), {kStream}, 0,
          NormF32(kStream, kPleNormConvTag));
      Add(b, o, Blk(l, "ple_conv1d.weight"), {kConvKernel, kStream}, 0,
          RampF32(kStream * kConvKernel, base));
    }
  }
  return b.Build();
}

// `ModelRegistry::Load`, not `reg.factory->load_weights`, and the difference is
// load-bearing rather than stylistic. `Load` resolves the architecture, refuses
// an unsupported FP8-block quantization, runs `parse_config` and THEN the weight
// loader — which is the sequence `LoadedEngine::FromModelDir` runs at
// `entrypoints/model_loader.cpp` (`ModelSource::FromGguf(gguf)` ->
// `ModelRegistry::Load(config, gguf_source)`). Calling the hook directly skips
// `parse_config`, and that skip is exactly what hid #2064: the config builder
// and the config VALIDATOR had never been composed, so a file that built a
// config fine was refused the moment anything parsed it.
inline std::unique_ptr<vllm::LoadedModel> LoadThroughRegistry(
    const vllm::GgufFile& g) {
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  const vllm::ModelSource source = vllm::ModelSource::FromGguf(g);
  return vllm::ModelRegistry::Load(config, source);
}

}  // namespace qwen4_exp_fixture
