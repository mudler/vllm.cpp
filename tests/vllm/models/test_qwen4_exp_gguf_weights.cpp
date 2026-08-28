// MODEL-MM-QWEN4-EXP W5a — the `qwen4exp` GGUF WEIGHT LOADER, entered through
// the production `load_weights` hook.
//
// Issue #2031, campaign issue #1978, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT THIS GATES, AND WHAT IT DOES NOT. Nothing here runs a forward and nothing
// here is a token or a speed claim: `Qwen4ExpForConditionalGeneration`'s forward
// and KV-cache spec still refuse by name after this wave, and the spec's
// `## Owed` says so. What this file gates is the load: which file tensor becomes
// which model weight, at which shape, under which residency, and with which
// convert-time transform inverted.
//
// TWO INSTRUMENTS, DELIBERATELY DIFFERENT.
//
//   * The COMMITTED MANIFEST of the real 67.56 GiB artifact
//     (`qwen4_exp_gguf_manifest.inc`, `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S`
//     @ 8bdc666649440e9bdc97e16f3f75782c98478ff5) gates the NAME MAP and the
//     SHAPE RULES at the released config, with no asset in CI. It is the only
//     instrument that can catch a name or a shape our synthetic fixture would
//     agree with by construction, because we write the fixture and we did not
//     write the checkpoint.
//   * A SYNTHETIC tiny GGUF, built byte-by-byte here, gates the VALUES — the
//     `+1` norm fold and its one exception, the V-head reorder, `log(-ssm_a)`,
//     and keep-quant residency — by carrying bytes we chose and reading back
//     what the loader made of them.
//
// Neither is sufficient alone. The manifest has no bytes; the fixture has no
// authority over names.
//
// ORACLE. vLLM implements `qwen4_exp` at no revision, so the algorithm oracle is
// transformers **5.16.0** (this row's accepted lane pin) and the CONTAINER
// oracle for the convert-time transforms is llama.cpp pull request
// [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) at head
// `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, read at source rather than
// relayed. Every transform this file asserts cites the line that produces it.
#include "vllm/model_executor/models/qwen4_exp_weights.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_gguf_weights.h"
#include "vllm/model_executor/models/qwen4_exp_ple.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"
#include "vt/quant.h"

#include "qwen4_exp_gguf_manifest.inc"

namespace {

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

std::string Blk(int64_t l, const char* suffix) {
  return "blk." + std::to_string(l) + "." + suffix;
}

bool IsLinear(int64_t l) { return ((l + 1) % 4) != 0; }

// ── deterministic payloads ───────────────────────────────────────────────────

std::string F32Bytes(const std::vector<float>& v) {
  std::string s(v.size() * 4, '\0');
  std::memcpy(s.data(), v.data(), v.size() * 4);
  return s;
}

// A distinguishable value per element: no two positions of any tensor share a
// value, so a permutation defect cannot hide behind a repeated number.
std::vector<float> Ramp(int64_t n, float base) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = base + static_cast<float>(i);
  return v;
}

std::string RampF32(int64_t n, float base) { return F32Bytes(Ramp(n, base)); }

// NORM gammas get their own generator, and the reason is a measurement rather
// than tidiness. The `+1` fold this loader inverts is a subtraction of ONE, and
// bf16's step is 16 by the time a plain ramp reaches 3001 — so on a gamma
// written as `3001 + i` the fold and its absence round to the SAME bf16 value
// and the check passes either way. Every value here is `1 + k/128` with
// `k` in [0, 127], which bf16 represents exactly, and so is `k/128` after the
// fold is removed. `tag` gives each tensor its own sequence so a cross-wired
// pair (norm_key read into norm_query) is visible.
float NormValue(int64_t i, int64_t tag) {
  return 1.0F + static_cast<float>((i + 13 * tag) % 128) / 128.0F;
}

std::string NormF32(int64_t n, int64_t tag) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = NormValue(i, tag);
  return F32Bytes(v);
}

// Q8_0 payload for `rows x 32` — one block per row, encoded the way
// `DequantGgufRowToF32` reads it back: an f16 scale then 32 int8 codes.
std::string Q8_0Bytes(int64_t rows, int64_t cols) {
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

void Add(GgufModelBuilder& b, const FixtureOpts& o, const std::string& name,
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

std::string BuildFixture(const FixtureOpts& o = {}) {
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

// The production entry point, reached the way a user reaches it: the GGUF
// architecture dispatch builds the config, the registry resolves the
// architecture, and the registration's own `load_weights` hook runs. Nothing
// here constructs a `Qwen4ExpWeights` by hand.
// The released `Qwen/Qwen3.8-Flash-Next` config, resolved through the same
// parser production uses. The fixture is that checkpoint's config.json verbatim
// (W1 committed it), so every number below comes from the model's author.
vllm::Qwen4ExpParams ReleasedParams() {
  const std::string path = std::string(QWEN4_EXP_CKPT_FIXTURE_DIR) + "/config.json";
  return vllm::ParseQwen4ExpParams(vllm::LoadHfConfig(path));
}

// One manifest row by name, in TORCH order (the manifest stores GGUF `ne`,
// which is reversed).
std::vector<int64_t> ManifestShape(const std::string& name) {
  for (const auto& t : vllm_test::kQwen4ExpGgufTensors) {
    if (name != t.name) continue;
    std::vector<int64_t> ne(t.dims, t.dims + t.n_dims);
    return std::vector<int64_t>(ne.rbegin(), ne.rend());
  }
  FAIL("manifest has no tensor named " << name);
  return {};
}

// The fixture's ramp runs past 256, where bf16 stops representing consecutive
// integers exactly (the step is 2 by 256 and 16 by 2048). An expectation has to
// be rounded the same way the loader's store rounds it, or the comparison is
// testing bf16 and not the loader. Every reorder subcase below ALSO asserts that
// the rounded value it wants differs from the one a no-reorder loader would
// produce, so a comparison bf16 has collapsed cannot pass for a measurement.
float Rounded(float f) { return vt::BF16ToF32(vt::F32ToBF16(f)); }

float Bf16At(const vllm::OwnedTensor& t, int64_t i) {
  REQUIRE(t.dtype == vt::DType::kBF16);
  const auto* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
  return vt::BF16ToF32(p[i]);
}

float F32At(const vllm::OwnedTensor& t, int64_t i) {
  REQUIRE(t.dtype == vt::DType::kF32);
  const auto* p = reinterpret_cast<const float*>(t.bytes.data());
  return p[i];
}

std::vector<int64_t> ShapeOf(const vllm::OwnedTensor& t) {
  return std::vector<int64_t>(t.shape, t.shape + t.rank);
}

// The whole file's tensor names, as the reader sees them.
std::set<std::string> FileNames(const vllm::GgufFile& g) {
  std::set<std::string> out;
  for (const auto& t : g.Tensors()) out.insert(t.name);
  return out;
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
std::unique_ptr<vllm::LoadedModel> LoadThroughRegistry(
    const vllm::GgufFile& g) {
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  const vllm::ModelSource source = vllm::ModelSource::FromGguf(g);
  return vllm::ModelRegistry::Load(config, source);
}

}  // namespace

// --- the load runs at all ---------------------------------------------------

TEST_CASE("qwen4_exp GGUF: the production load_weights hook LOADS the file") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  // AND THE HANDLE IS OPENED AND READ. `REQUIRE_NOTHROW` plus `!= nullptr` is
  // where this case stopped, and both of those hold for a `load_weights` that
  // returns a default-constructed `Qwen4ExpWeights{}` — so deleting the
  // `LoadQwen4ExpFromGguf` call site (mutation M1) left this case green and
  // only the refusal subcases below reddened. A case named for a LOAD has to
  // observe what the load produced.
  //
  // `ModelAs`, never a `static_cast`: the checked form establishes the dynamic
  // type first (#775, #730).
  const vllm::Qwen4ExpLoadedModel& typed =
      vllm::ModelAs<vllm::Qwen4ExpLoadedModel>(
          *model, "Qwen4ExpForConditionalGeneration");
  const vllm::Qwen4ExpWeights& w = typed.weights();

  // STRUCTURE, then BYTES. The counts a stub reports as zero.
  CHECK(w.enumerated_tensors == static_cast<int64_t>(FileNames(g).size()));
  CHECK(w.accounted_tensors == w.enumerated_tensors);
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));

  // The bytes. `token_embd.weight` is a plain ramp from 1.0 in the file's own
  // order, so element `i` is `1 + i` rounded through bf16 — a value no empty
  // tensor and no zero-filled one carries.
  REQUIRE(ShapeOf(w.embed_tokens) == std::vector<int64_t>{kVocab, kH});
  for (int64_t i : {int64_t{0}, int64_t{1}, int64_t{37}}) {
    CAPTURE(i);
    CHECK(Bf16At(w.embed_tokens, i) ==
          doctest::Approx(Rounded(1.0F + static_cast<float>(i))));
  }
  // And one weight from the far end of the per-layer loop, so a hook that
  // loaded only the prologue is visible too: layer 3 is the sparse one and its
  // `attn_q.weight` is written from `base = 3 * 1000 + 1`.
  const auto& qsa = w.layers[3].qsa;
  REQUIRE(ShapeOf(qsa.q_proj) ==
          std::vector<int64_t>{kQHeads * kHeadDim * 2, kH});
  CHECK(Bf16At(qsa.q_proj, 0) == doctest::Approx(Rounded(3001.0F)));
}

// --- (1) the name map, against the REAL 1224-tensor table -------------------

TEST_CASE("qwen4_exp GGUF: the name map accounts the shipped file BOTH WAYS") {
  // The strongest instrument this wave has, and the only one with authority
  // over names: we wrote the fixture, we did not write the checkpoint. A name
  // the loader invents is invisible to the fixture — the fixture would simply
  // carry it — and fatal here.
  const std::vector<std::string> enumerated =
      vllm::EnumerateQwen4ExpGgufTensors(ReleasedParams());
  const std::set<std::string> ours(enumerated.begin(), enumerated.end());
  CHECK(ours.size() == enumerated.size());  // no name enumerated twice

  std::set<std::string> shipped;
  for (const auto& t : vllm_test::kQwen4ExpGgufTensors) shipped.insert(t.name);
  REQUIRE(shipped.size() ==
          static_cast<size_t>(vllm_test::kQwen4ExpGgufTensorCount));

  // Direction 1: nothing the loader asks for is missing from the file.
  for (const std::string& n : ours) {
    CAPTURE(n);
    CHECK(shipped.count(n) == 1);
  }
  // Direction 2: nothing in the file is left unaccounted. An unaccounted tensor
  // is a weight the model would run WITHOUT, which is the failure a one-way
  // check cannot see.
  for (const std::string& n : shipped) {
    CAPTURE(n);
    CHECK(ours.count(n) == 1);
  }
  CHECK(static_cast<int64_t>(ours.size()) ==
        vllm_test::kQwen4ExpGgufTensorCount);
}

// --- (2) the SHAPE rules, against the same table ----------------------------

TEST_CASE("qwen4_exp GGUF: every shape rule agrees with the shipped file") {
  // Each assertion joins a number DERIVED from the released config.json to a
  // number READ from the released checkpoint. Neither side can be adjusted to
  // suit the other, which is what makes this different from a table that
  // restates itself.
  const vllm::Qwen4ExpParams p = ReleasedParams();
  const int64_t h = p.hidden_size;
  const int64_t stream = p.stream_width();
  REQUIRE(h == 2560);
  REQUIRE(stream == 10240);

  SUBCASE("the hyper-connection tower") {
    CHECK(ManifestShape("blk.0.hc_attn_norm.weight") ==
          std::vector<int64_t>{stream});
    CHECK(ManifestShape("blk.0.hc_attn_down.weight") ==
          std::vector<int64_t>{p.hc_lowrank, stream});
    CHECK(ManifestShape("blk.0.hc_attn_up.weight") ==
          std::vector<int64_t>{stream, p.hc_lowrank});
    CHECK(ManifestShape("blk.0.hc_attn_inject.weight") ==
          std::vector<int64_t>{p.hc_count, stream});
    // The MIXER carries no `inject`, which is `use_combine=false`, and the
    // absence is asserted rather than assumed: an enumerated-but-absent tensor
    // would have failed the both-ways case above, but a present-and-ignored one
    // would not.
    CHECK(ManifestShape("output_hc_norm.weight") ==
          std::vector<int64_t>{stream});
    bool mixer_inject = false;
    for (const auto& t : vllm_test::kQwen4ExpGgufTensors)
      if (std::string(t.name) == "output_hc_inject.weight") mixer_inject = true;
    CHECK_FALSE(mixer_inject);
    // And there is NO final RMSNorm. The mixer's own `hc_norm` is the last
    // normalization before `lm_head`; a port that copies our DeepSeek-V4 tail
    // inserts one that does not exist.
    bool final_norm = false;
    for (const auto& t : vllm_test::kQwen4ExpGgufTensors)
      if (std::string(t.name) == "output_norm.weight") final_norm = true;
    CHECK_FALSE(final_norm);
  }

  SUBCASE("Gated DeltaNet") {
    // 16 key heads and 48 value heads at 128 dims: key_dim 2048, value_dim
    // 6144, conv_dim 2*2048 + 6144 = 10240 — which is the same number as the
    // residual stream width and is NOT the same quantity. The manifest is what
    // separates them.
    CHECK(p.linear_key_dim() == 2048);
    CHECK(p.linear_value_dim() == 6144);
    CHECK(ManifestShape("blk.0.attn_qkv.weight") ==
          std::vector<int64_t>{p.linear_conv_dim(), h});
    CHECK(ManifestShape("blk.0.attn_gate.weight") ==
          std::vector<int64_t>{p.linear_value_dim(), h});
    CHECK(ManifestShape("blk.0.ssm_alpha.weight") ==
          std::vector<int64_t>{p.linear_num_value_heads, h});
    CHECK(ManifestShape("blk.0.ssm_beta.weight") ==
          std::vector<int64_t>{p.linear_num_value_heads, h});
    CHECK(ManifestShape("blk.0.ssm_a") ==
          std::vector<int64_t>{p.linear_num_value_heads});
    CHECK(ManifestShape("blk.0.ssm_dt.bias") ==
          std::vector<int64_t>{p.linear_num_value_heads});
    CHECK(ManifestShape("blk.0.ssm_conv1d.weight") ==
          std::vector<int64_t>{p.linear_conv_dim(), p.linear_conv_kernel_dim});
    // `RMSNormGated` over ONE value head, not over the value stream.
    CHECK(ManifestShape("blk.0.ssm_norm.weight") ==
          std::vector<int64_t>{p.linear_value_head_dim});
    CHECK(ManifestShape("blk.0.ssm_out.weight") ==
          std::vector<int64_t>{h, p.linear_value_dim()});
    // The V-head reorder is ACTIVE on this checkpoint, which is what makes the
    // whole un-reorder path load-bearing rather than dead.
    CHECK(p.linear_num_key_heads != p.linear_num_value_heads);
    CHECK(p.linear_num_value_heads % p.linear_num_key_heads == 0);
  }

  SUBCASE("Qwen Sparse Attention") {
    // `* 2`: query and output gate share one projection. Read straight, 12288
    // looks like 48 heads of 256 and is 24 heads of 256 twice over.
    CHECK(ManifestShape("blk.3.attn_q.weight") ==
          std::vector<int64_t>{p.num_attention_heads * p.head_dim * 2, h});
    CHECK(ManifestShape("blk.3.attn_k.weight") ==
          std::vector<int64_t>{p.num_key_value_heads * p.head_dim, h});
    CHECK(ManifestShape("blk.3.attn_v.weight") ==
          std::vector<int64_t>{p.num_key_value_heads * p.head_dim, h});
    CHECK(ManifestShape("blk.3.attn_output.weight") ==
          std::vector<int64_t>{h, p.num_attention_heads * p.head_dim});
    CHECK(ManifestShape("blk.3.attn_q_norm.weight") ==
          std::vector<int64_t>{p.head_dim});
    CHECK(ManifestShape("blk.3.attn_k_norm.weight") ==
          std::vector<int64_t>{p.head_dim});
    // The indexer's single `index_qk_proj` arrives SPLIT, at exactly the point
    // `indexer_n_heads * indexer_head_dim`.
    CHECK(ManifestShape("blk.3.indexer.q_proj.weight") ==
          std::vector<int64_t>{p.qsa.n_heads * p.qsa.head_dim, h});
    CHECK(ManifestShape("blk.3.indexer.k_proj.weight") ==
          std::vector<int64_t>{p.qsa.kv_heads * p.qsa.head_dim, h});
    CHECK(ManifestShape("blk.3.indexer.q_norm.weight") ==
          std::vector<int64_t>{p.qsa.head_dim});
    CHECK(p.qsa.kv_heads == 1);
  }

  SUBCASE("the MoE block") {
    CHECK(ManifestShape("blk.0.ffn_gate_inp.weight") ==
          std::vector<int64_t>{p.num_experts, h});
    // ONE-DIMENSIONAL. `Linear(H, 1)` squeezed; a loader expecting [1, H]
    // refuses a correct file.
    CHECK(ManifestShape("blk.0.ffn_gate_inp_shexp.weight") ==
          std::vector<int64_t>{h});
    CHECK(ManifestShape("blk.0.ffn_gate_exps.weight") ==
          std::vector<int64_t>{p.num_experts, p.moe_intermediate_size, h});
    CHECK(ManifestShape("blk.0.ffn_up_exps.weight") ==
          std::vector<int64_t>{p.num_experts, p.moe_intermediate_size, h});
    CHECK(ManifestShape("blk.0.ffn_down_exps.weight") ==
          std::vector<int64_t>{p.num_experts, h, p.moe_intermediate_size});
    CHECK(ManifestShape("blk.0.ffn_down_shexp.weight") ==
          std::vector<int64_t>{h, p.shared_expert_intermediate_size});
  }

  SUBCASE("the PLE layer, and the n-gram table's derived row count") {
    // ONE PLE layer, 0-based 1, and the checkpoint agrees: no other layer
    // carries a `ple_*` tensor.
    REQUIRE(p.ple.layer_ids_zero_based == std::vector<int64_t>{1});
    int64_t ple_tensors = 0;
    for (const auto& t : vllm_test::kQwen4ExpGgufTensors) {
      const std::string n = t.name;
      if (n.find(".ple_") != std::string::npos) {
        ++ple_tensors;
        CHECK(n.rfind("blk.1.", 0) == 0);
      }
    }
    CHECK(ple_tensors == 6);
    CHECK(ManifestShape("blk.1.ple_key.weight") ==
          std::vector<int64_t>{stream, p.ple.embed_dim});
    CHECK(ManifestShape("blk.1.ple_value.weight") ==
          std::vector<int64_t>{h, p.ple.embed_dim});
    CHECK(ManifestShape("blk.1.ple_norm_conv.weight") ==
          std::vector<int64_t>{stream});
    CHECK(ManifestShape("blk.1.ple_conv1d.weight") ==
          std::vector<int64_t>{stream, p.ple.conv_kernel_size});

    // THE ONE THAT MAKES W2 LOAD-BEARING. The released config states no head
    // sizes, so the row count comes from the prime chain: 16 successive primes
    // after 19999999, summed and padded to a multiple of 128. If that chain is
    // off by a single prime the number moves and this reds — against a
    // dimension read from the checkpoint, not written by us.
    CHECK(p.ple.head_vocab_sizes.empty());
    vllm::qwen4_exp::PleGeometry geom;
    geom.hidden_size = p.hidden_size;
    geom.hc_count = p.hc_count;
    geom.ple_embed_dim = p.ple.embed_dim;
    geom.ngram_size = p.ple.ngram_size;
    geom.heads_per_ngram = p.ple.heads_per_ngram;
    geom.ngram_vocab_size_base = p.ple.ngram_vocab_size_base;
    geom.make_ngram_vocab_size_divisible_by =
        p.ple.make_ngram_vocab_size_divisible_by;
    geom.vocab_size = p.vocab_size;
    geom.eos_token_id = p.eos_token_id;
    geom.seed = p.ple.seed;
    const vllm::qwen4_exp::NGramTableLayout layout =
        vllm::qwen4_exp::BuildNGramTableLayout(geom, /*ple_layer_index=*/0);
    CHECK(layout.padded_vocab_size == 320001536);
    CHECK(p.ple.head_dim_per_ngram() == 160);
    CHECK(ManifestShape("per_layer_token_embd.weight") ==
          std::vector<int64_t>{layout.padded_vocab_size,
                               p.ple.head_dim_per_ngram()});
  }

  SUBCASE("the token table and the head") {
    CHECK(ManifestShape("token_embd.weight") ==
          std::vector<int64_t>{p.vocab_size, h});
    CHECK(ManifestShape("output.weight") ==
          std::vector<int64_t>{p.vocab_size, h});
  }
}

// --- (3) the VALUES, on bytes we wrote --------------------------------------
//
// These call `LoadQwen4ExpFromGguf` directly, which is the house shape
// (`test_muse_glimmer_gguf.cpp`, `test_deepseek_v4_gguf_load.cpp`): the
// concrete `LoadedModel` lives in the registry's anonymous namespace and there
// is nothing to downcast to. The REACHABILITY of this function is proven
// separately, by the hook case above and by the deletion mutation the review
// runs on its call site — a green suite with that call site removed would mean
// this file measures a function and not a load.

TEST_CASE("qwen4_exp GGUF: the `+1` norm fold is inverted everywhere but ssm_norm") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::Qwen4ExpHfConfigFromGguf(g);
  const vllm::Qwen4ExpWeights w =
      vllm::LoadQwen4ExpFromGguf(g, cfg, vt::DeviceType::kCPU);

  // The fixture wrote `1 + k/128` into every gamma. A FOLDED read gives
  // `k/128`, an UNFOLDED one gives `1 + k/128`, and both are exact in bf16 — so
  // each assertion below has a wrong answer available and can land on it.
  auto Folded = [](int64_t i, int64_t tag) { return NormValue(i, tag) - 1.0F; };

  SUBCASE("folded: the hyper-connection, PLE, q/k and indexer gammas") {
    for (int64_t l = 0; l < kLayers; ++l) {
      CAPTURE(l);
      const auto& lw = w.layers[static_cast<size_t>(l)];
      for (int64_t i : {int64_t{0}, int64_t{5}, kStream - 1}) {
        CHECK(Bf16At(lw.attn_hc.hc_norm, i) ==
              doctest::Approx(Folded(i, HcNormTag(l, "attn"))));
        CHECK(Bf16At(lw.mlp_hc.hc_norm, i) ==
              doctest::Approx(Folded(i, HcNormTag(l, "ffn"))));
      }
    }
    CHECK(Bf16At(w.mixer.hc_norm, 2) == doctest::Approx(Folded(2, kMixerNormTag)));

    // The three PLE gammas the INHERITED rule would have missed: they are
    // spelled `norm_key` / `norm_query` / `norm_conv`, so
    // `endswith("norm.weight")` never reaches them and #27742's own
    // early-returning branch is what folds them. Each carries its own sequence,
    // so a loader that read one into another is visible here as well.
    const auto& ple = w.layers[static_cast<size_t>(kPleLayer)].ple;
    for (int64_t i : {int64_t{0}, int64_t{4}, kStream - 1}) {
      CHECK(Bf16At(ple.norm_key, i) == doctest::Approx(Folded(i, kPleNormKeyTag)));
      CHECK(Bf16At(ple.norm_query, i) ==
            doctest::Approx(Folded(i, kPleNormQueryTag)));
      CHECK(Bf16At(ple.norm_conv, i) ==
            doctest::Approx(Folded(i, kPleNormConvTag)));
    }

    // Layer 3 is the sparse one: the per-head q/k gammas and both indexer ones.
    const auto& qsa = w.layers[3].qsa;
    for (int64_t i = 0; i < kHeadDim; ++i) {
      CAPTURE(i);
      CHECK(Bf16At(qsa.q_norm, i) == doctest::Approx(Folded(i, kQNormTag)));
      CHECK(Bf16At(qsa.k_norm, i) == doctest::Approx(Folded(i, kKNormTag)));
    }
    for (int64_t i = 0; i < kIdxHeadDim; ++i) {
      CAPTURE(i);
      CHECK(Bf16At(qsa.idx_q_norm, i) ==
            doctest::Approx(Folded(i, kIdxQNormTag)));
      CHECK(Bf16At(qsa.idx_k_norm, i) ==
            doctest::Approx(Folded(i, kIdxKNormTag)));
    }
  }

  SUBCASE("UNFOLDED: ssm_norm, the one exception") {
    // `linear_attn.norm.weight` is excluded BY NAME from the inherited rule and
    // is not in #27742's explicit list either, so it reaches us raw. Subtracting
    // one here would be a ~2x error on the Gated DeltaNet output gate that no
    // shape check could see, and the published artifacts corroborate the
    // polarity: `ssm_norm` sits in [0.875, 1.023] while every sibling gamma is
    // centred on 1.0.
    for (int64_t l = 0; l < kLayers; ++l) {
      if (!IsLinear(l)) continue;
      CAPTURE(l);
      const auto& gdn = w.layers[static_cast<size_t>(l)].gdn;
      for (int64_t i = 0; i < kLinHeadDim; ++i) {
        CHECK(Bf16At(gdn.norm_weight, i) ==
              doctest::Approx(NormValue(i, SsmNormTag(l))));
      }
    }
  }
}

TEST_CASE("qwen4_exp GGUF: ssm_a is recovered as log(-x), and a bad sign refuses") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::Qwen4ExpHfConfigFromGguf(g);
  const vllm::Qwen4ExpWeights w =
      vllm::LoadQwen4ExpFromGguf(g, cfg, vt::DeviceType::kCPU);

  // The fixture wrote `-(head + 1)` in GGUF TILED order; the loader recovers
  // `log(head + 1)` and then un-reorders. Grouped head g = k*R + r reads tiled
  // head t = r*K + k, with K = 2 and R = 3: g -> t is 0->0, 1->2, 2->4, 3->1,
  // 4->3, 5->5. That map is NOT its own inverse, which is what makes a
  // wrong-direction un-reorder visible.
  const int64_t tiled_for_grouped[6] = {0, 2, 4, 1, 3, 5};
  REQUIRE(w.layers[0].gdn.a_log.dtype == vt::DType::kF32);
  for (int64_t gidx = 0; gidx < kNumVHeads; ++gidx) {
    CAPTURE(gidx);
    const float want =
        std::log(static_cast<float>(tiled_for_grouped[gidx] + 1));
    CHECK(F32At(w.layers[0].gdn.a_log, gidx) == doctest::Approx(want));
  }
  // Not the identity: a loader that skipped the un-reorder would read
  // log(gidx + 1) and disagree on heads 1 and 2.
  CHECK(F32At(w.layers[0].gdn.a_log, 1) !=
        doctest::Approx(std::log(2.0F)).epsilon(1e-6));
  // And not the INVERSE permutation either: `t = k*R + r`, `g = r*K + k` sends
  // grouped head 1 to tiled head 3, where the correct map sends it to 2.
  CHECK(F32At(w.layers[0].gdn.a_log, 1) !=
        doctest::Approx(std::log(4.0F)).epsilon(1e-6));
}

TEST_CASE("qwen4_exp GGUF: the V-head reorder is inverted on every GDN tensor") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::Qwen4ExpHfConfigFromGguf(g);
  const vllm::Qwen4ExpWeights w =
      vllm::LoadQwen4ExpFromGguf(g, cfg, vt::DeviceType::kCPU);
  const vllm::Qwen4ExpGdnWeights& gdn = w.layers[0].gdn;
  const float base = 1.0F;
  // Grouped head `g = k*R + r` reads tiled head `t = r*K + k`, with K = 2 key
  // heads and R = 3 value heads each: 0->0, 1->2, 2->4, 3->1, 4->3, 5->5. Four
  // of the six MOVE, and the map differs from its own inverse (which would send
  // 1 to 3), so a loader that ran the permutation backwards lands somewhere
  // wrong. At K = 1 the map is the identity and at K == R it is an involution;
  // this fixture is neither.
  const int64_t tiled_for_grouped[6] = {0, 2, 4, 1, 3, 5};

  // One element of a [rows, cols] buffer whose file value was `base + index`.
  // `src_index` is where the value came from and `identity_index` is where a
  // loader that skipped the un-reorder would have read it. Asserting the two
  // ROUNDED values differ is what stops a bf16-collapsed pair from reading as a
  // pass; the CHECK then has somewhere wrong to land.
  auto CheckMoved = [&](const vllm::OwnedTensor& t, int64_t dst_index,
                        int64_t src_index, int64_t identity_index) {
    CAPTURE(dst_index);
    CAPTURE(src_index);
    CAPTURE(identity_index);
    const float want = Rounded(base + static_cast<float>(src_index));
    if (src_index != identity_index) {
      REQUIRE(want != Rounded(base + static_cast<float>(identity_index)));
    }
    CHECK(Bf16At(t, dst_index) == doctest::Approx(want));
  };

  SUBCASE("in_proj_z: every row moves with its head") {
    REQUIRE(ShapeOf(gdn.in_proj_z) == std::vector<int64_t>{kValueDim, kH});
    for (int64_t gidx = 0; gidx < kNumVHeads; ++gidx) {
      for (int64_t r = 0; r < kLinHeadDim; r += 3) {
        CAPTURE(gidx);
        const int64_t dst = (gidx * kLinHeadDim + r) * kH + 7;
        const int64_t src = (tiled_for_grouped[gidx] * kLinHeadDim + r) * kH + 7;
        CheckMoved(gdn.in_proj_z, dst, src, dst);
      }
    }
  }

  SUBCASE("in_proj_qkv: ONLY the trailing V rows move") {
    REQUIRE(ShapeOf(gdn.in_proj_qkv) == std::vector<int64_t>{kConvDim, kH});
    // The leading 2*key_dim q and k rows are untouched. A loader that reordered
    // the whole tensor would scramble the query stream and the model would
    // still run, which is why this half is asserted and not assumed.
    for (int64_t row : {int64_t{0}, int64_t{5}, 2 * kKeyDim - 1}) {
      CAPTURE(row);
      const int64_t idx = row * kH + 1;
      CHECK(Bf16At(gdn.in_proj_qkv, idx) ==
            doctest::Approx(Rounded(base + static_cast<float>(idx))));
    }
    for (int64_t gidx = 0; gidx < kNumVHeads; ++gidx) {
      CAPTURE(gidx);
      const int64_t dst = (2 * kKeyDim + gidx * kLinHeadDim) * kH + 2;
      const int64_t src =
          (2 * kKeyDim + tiled_for_grouped[gidx] * kLinHeadDim) * kH + 2;
      CheckMoved(gdn.in_proj_qkv, dst, src, dst);
    }
  }

  SUBCASE("in_proj_a / in_proj_b: one row per head") {
    // `ssm_beta` is `in_proj_b` and `ssm_alpha` is `in_proj_a` — the names
    // cross, and this subcase deliberately does NOT claim to see a swap: the
    // fixture writes the same ramp into both, so their contents are identical
    // and no assertion here could tell them apart. What it does see is the
    // reorder, which is the thing with a wrong answer available. The crossing
    // itself is gated by the name map against the shipped file, not here.
    REQUIRE(ShapeOf(gdn.in_proj_a) == std::vector<int64_t>{kNumVHeads, kH});
    REQUIRE(ShapeOf(gdn.in_proj_b) == std::vector<int64_t>{kNumVHeads, kH});
    for (int64_t gidx = 0; gidx < kNumVHeads; ++gidx) {
      CAPTURE(gidx);
      const int64_t dst = gidx * kH + 3;
      const int64_t src = tiled_for_grouped[gidx] * kH + 3;
      CheckMoved(gdn.in_proj_a, dst, src, dst);
      CheckMoved(gdn.in_proj_b, dst, src, dst);
    }
  }

  SUBCASE("conv1d: only the V CHANNELS move, and the q|k channels do not") {
    REQUIRE(ShapeOf(gdn.conv1d) == std::vector<int64_t>{kConvDim, kConvKernel});
    for (int64_t ch : {int64_t{0}, 2 * kKeyDim - 1}) {
      CAPTURE(ch);
      const int64_t idx = ch * kConvKernel;
      CHECK(Bf16At(gdn.conv1d, idx) ==
            doctest::Approx(Rounded(base + static_cast<float>(idx))));
    }
    for (int64_t gidx = 0; gidx < kNumVHeads; ++gidx) {
      CAPTURE(gidx);
      const int64_t dst = (2 * kKeyDim + gidx * kLinHeadDim) * kConvKernel + 1;
      const int64_t src =
          (2 * kKeyDim + tiled_for_grouped[gidx] * kLinHeadDim) * kConvKernel + 1;
      CheckMoved(gdn.conv1d, dst, src, dst);
    }
  }

  SUBCASE("out_proj: the reorder is on COLUMNS, not rows") {
    // The one that can never ride a kept-quant block stream: a COLUMN
    // permutation cuts across a superblock where a row permutation never does,
    // which is why this tensor is `kTransformedWeight` whenever the reorder is
    // active.
    REQUIRE(ShapeOf(gdn.out_proj) == std::vector<int64_t>{kH, kValueDim});
    for (int64_t row : {int64_t{0}, int64_t{9}}) {
      for (int64_t gidx = 0; gidx < kNumVHeads; ++gidx) {
        CAPTURE(row);
        CAPTURE(gidx);
        const int64_t dst = row * kValueDim + gidx * kLinHeadDim + 2;
        const int64_t src =
            row * kValueDim + tiled_for_grouped[gidx] * kLinHeadDim + 2;
        CheckMoved(gdn.out_proj, dst, src, dst);
      }
    }
  }
}

// --- (4) residency and accounting -------------------------------------------

TEST_CASE("qwen4_exp GGUF: the n-gram gather table KEEPS its blocks on CPU") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::Qwen4ExpHfConfigFromGguf(g);

  SUBCASE("under the production policy") {
    // W6a (#1989) made `kEmbeddingTable` keep-quant eligible and taught
    // `vt::Embedding` to decode a row per gathered token; without it a 51.2 G-
    // parameter Q4_K table expands from 28.8 GB to 102.4 GB at load and the arm
    // dies before the first forward. `FromEnv()` is what a production load gets,
    // and on a CPU build it turns keep-quant on.
    vllm::GgufLoadPolicy pol = vllm::GgufLoadPolicy::FromEnv();
    pol.keep_quant = true;
    const vllm::Qwen4ExpWeights w =
        vllm::LoadQwen4ExpFromGguf(g, cfg, vt::DeviceType::kCPU, &pol);
    CHECK(w.ngram_table.dtype == vt::DType::kQ8_0);
    CHECK(ShapeOf(w.ngram_table) == std::vector<int64_t>{kNgramRows, kPleRow});
    // A GATHER, never a MatmulBT operand: `nk` is what tells a consumer which
    // it is, and a table marked `nk` would be handed to a GEMM.
    CHECK_FALSE(w.ngram_table.nk);
    // The plain token table is a gather too and expands, which is what every
    // other token table in this tree does.
    CHECK(w.embed_tokens.dtype == vt::DType::kBF16);
    CHECK_FALSE(w.embed_tokens.nk);
  }

  SUBCASE("with keep-quant OFF the same table expands, and the load still works") {
    vllm::GgufLoadPolicy pol;  // the default-constructed all-expand policy
    const vllm::Qwen4ExpWeights w =
        vllm::LoadQwen4ExpFromGguf(g, cfg, vt::DeviceType::kCPU, &pol);
    CHECK(w.ngram_table.dtype == vt::DType::kBF16);
    CHECK(ShapeOf(w.ngram_table) == std::vector<int64_t>{kNgramRows, kPleRow});
  }
}

TEST_CASE("qwen4_exp GGUF: the load accounts every tensor in the file") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::Qwen4ExpHfConfigFromGguf(g);
  const vllm::Qwen4ExpWeights w =
      vllm::LoadQwen4ExpFromGguf(g, cfg, vt::DeviceType::kCPU);

  const std::set<std::string> in_file = FileNames(g);
  const std::vector<std::string> enumerated =
      vllm::EnumerateQwen4ExpGgufTensors(w.params);
  const std::set<std::string> ours(enumerated.begin(), enumerated.end());
  CHECK(ours == in_file);
  CHECK(w.enumerated_tensors == static_cast<int64_t>(in_file.size()));
  CHECK(w.accounted_tensors == w.enumerated_tensors);

  // The structure the config implied is the structure that was built.
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));
  for (int64_t l = 0; l < kLayers; ++l) {
    CAPTURE(l);
    CHECK(w.layers[static_cast<size_t>(l)].is_linear_attention == IsLinear(l));
    CHECK(w.layers[static_cast<size_t>(l)].has_ple == (l == kPleLayer));
    CHECK(w.layers[static_cast<size_t>(l)].attn_hc.has_inject);
    CHECK(w.layers[static_cast<size_t>(l)].mlp_hc.has_inject);
  }
  // `use_combine = false` on the mixer alone.
  CHECK_FALSE(w.mixer.has_inject);
  CHECK(w.mixer.inject.bytes.empty());
  // The file carries `output.weight`, so the head is untied.
  CHECK_FALSE(w.tied_word_embeddings);
  CHECK(ShapeOf(w.lm_head) == std::vector<int64_t>{kVocab, kH});
  // The PLE resolution the GGUF path owes: `qwen4exp.ple.layers` is ZERO-based
  // (`ple_layers = [i - 1 for i in hp["ple_layer_ids"]]`, #27742), so a file
  // saying [1] must produce a PLE on 0-based layer 1 — and not on none at all,
  // which is what an unmapped key produced.
  CHECK(w.params.ple.layer_ids_zero_based == std::vector<int64_t>{kPleLayer});
  CHECK(w.params.number_of_conv_states() == 3);
  CHECK(w.params.eos_token_id == kEosTokenId);

  // The OTHER half of #2064, and the half that had no instrument at all until
  // this fixture stopped defining `kPleRow` as `kH / kNgramHeads`. The GGUF
  // states the per-head row width; HF states the total; the builder must
  // reconstruct the total, because `ParseQwen4ExpParams` otherwise falls back
  // to `hidden_size`. On the released checkpoint the two agree by coincidence,
  // so the fallback is right there and wrong everywhere else.
  //
  // Three assertions, not one. The first states the value; the other two state
  // that the value is NEITHER wrong answer available to land on — the bare
  // `hidden_size` fallback, and a builder that multiplied `hidden_size` by the
  // head count instead of the per-head row width. At the old `kPleRow` of 64
  // the second wrong answer was numerically equal to the right one, so it was
  // unobservable (mutation MUT-D); at 96 the three values are 192, 64 and 128.
  CHECK(w.params.ple.embed_dim == kPleEmbedDim);
  CHECK(w.params.ple.embed_dim != w.params.hidden_size);
  CHECK(w.params.ple.embed_dim !=
        w.params.hidden_size * w.params.ple.ngram_heads());
  CHECK(w.params.ple.head_dim_per_ngram() == kPleRow);
  // And the shapes that width decides, read off the loaded weights rather than
  // off the config that produced them.
  const auto& ple = w.layers[static_cast<size_t>(kPleLayer)].ple;
  CHECK(ShapeOf(ple.key_proj) == std::vector<int64_t>{kStream, kPleEmbedDim});
  CHECK(ShapeOf(ple.value_proj) == std::vector<int64_t>{kH, kPleEmbedDim});
}

// --- (5) the refusals, through the PRODUCTION hook ---------------------------

TEST_CASE("qwen4_exp GGUF: a device with no block gather refuses BEFORE the load") {
  // #2083. The n-gram table is the ONE gather this model keeps quantized, and
  // `DeviceQuantGatherSupported` is true for `kCPU` alone
  // (`gguf_keep_quant.cpp`) because only the CPU `Embedding` kernel decodes
  // blocks. On any other device `RouteGgufTensor` therefore sends
  // `per_layer_token_embd.weight` to `kExpandBf16`, and on the shipped
  // artifact that tensor is [320001536, 160]: 320001536 * 160 * 2 =
  // 102,400,491,520 bytes = 95.368 GiB of ANONYMOUS host memory, against
  // 26.822 GiB on disk.
  //
  // The #1123 device-fit guard cannot see it. That guard sums the file's
  // ON-DISK bytes — 67.554 GiB for this artifact, comfortably under a GB10's
  // ~119.6 GiB — so it admits the load and the expansion happens anyway. What
  // the user got was `model_loader.cpp`'s own worst case, quoted there:
  // "Loading for 26 minutes and dying mid-stream is the worst of the available
  // behaviours."
  //
  // So the arm is REFUSED BY NAME on the device that cannot serve it, before
  // any tensor I/O, naming the missing part the way AGENTS.md requires of an
  // unimplemented arm. The device is a parameter rather than a read of
  // `CurrentPlatform()` precisely so this case can enter the production
  // refusal on a CPU-only host instead of asserting the predicate beside it.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::Qwen4ExpHfConfigFromGguf(g);

  SUBCASE("cuda refuses, and the message names the tensor and the missing arm") {
    std::string msg;
    try {
      (void)vllm::LoadQwen4ExpFromGguf(g, cfg, vt::DeviceType::kCUDA);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CAPTURE(msg);
    // CHECK, not REQUIRE: a fatal assertion here aborts the whole test case and
    // the two subcases below never report at all.
    CHECK_FALSE(msg.empty());
    CHECK(msg.find("per_layer_token_embd.weight") != std::string::npos);
    CHECK(msg.find("cuda") != std::string::npos);
    // The refusal has to say what is MISSING, not only that something is.
    CHECK(msg.find("gather") != std::string::npos);
  }

  SUBCASE("every non-CPU device refuses, because none of them decodes blocks") {
    for (vt::DeviceType d :
         {vt::DeviceType::kCUDA, vt::DeviceType::kMETAL,
          vt::DeviceType::kVULKAN, vt::DeviceType::kROCM}) {
      CAPTURE(vt::DeviceTypeName(d));
      CHECK_THROWS_AS(
          (void)vllm::LoadQwen4ExpFromGguf(g, cfg, d),
          std::exception);
    }
  }

  SUBCASE("CPU loads, and so does an explicit kCPU") {
    CHECK_NOTHROW((void)vllm::LoadQwen4ExpFromGguf(g, cfg,
                                                   vt::DeviceType::kCPU));
    // And the refusal is keyed on the DEVICE, not on the residency the policy
    // happens to resolve: with keep-quant off the CPU table expands too, and
    // that is a small, correct, supported load rather than a 95 GiB one.
    vllm::GgufLoadPolicy pol;  // default-constructed: all-expand
    CHECK_NOTHROW((void)vllm::LoadQwen4ExpFromGguf(g, cfg,
                                                   vt::DeviceType::kCPU, &pol));
  }
}


TEST_CASE("qwen4_exp GGUF: a malformed file refuses BY NAME at load_weights") {
  auto Message = [](const FixtureOpts& o) {
    TempFile f(BuildFixture(o));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    try {
      (void)LoadThroughRegistry(g);
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
    return std::string();
  };

  SUBCASE("a missing tensor") {
    FixtureOpts o;
    o.drop = "blk.2.ssm_a";
    const std::string msg = Message(o);
    CHECK(msg.find("blk.2.ssm_a") != std::string::npos);
    CHECK_FALSE(msg.empty());
  }

  SUBCASE("a mis-shaped tensor names the tensor AND both shapes") {
    FixtureOpts o;
    o.bad_shape = "blk.0.hc_attn_down.weight";
    const std::string msg = Message(o);
    CHECK(msg.find("blk.0.hc_attn_down.weight") != std::string::npos);
    CHECK(msg.find("shape mismatch") != std::string::npos);
    // Both the shape found and the shape wanted, or the reader cannot tell
    // which end is wrong.
    CHECK(msg.find("got [") != std::string::npos);
    CHECK(msg.find("expected [") != std::string::npos);
  }

  SUBCASE("a compress_ratios schedule that is not uniform") {
    // W5c (#2031). The refusal ALREADY existed — the reader takes the first
    // non-zero ratio and requires the rest to agree — and NOTHING gated it:
    // deleting its `VT_CHECK` left this whole suite green. It matters to the
    // KV-cache spec, which publishes ONE `MLAAttentionSpec` for every QSA
    // layer at ONE `compress_ratio`, mirroring upstream's own
    // `MLAAttentionSpec.merge` assert that a group carries a single ratio
    // (`vllm/v1/kv_cache_interface.py:424-435` at the pin `5559679229`). A
    // first-wins read would size the side cache for one ratio while another
    // layer compressed at a different one, which is a short cache and wrong
    // tokens rather than a crash.
    FixtureOpts o;
    o.mixed_compress_ratios = true;
    TempFile f(BuildFixture(o));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    std::string msg;
    try {
      (void)vllm::Qwen4ExpHfConfigFromGguf(g);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("compress_ratios") != std::string::npos);
    CHECK(msg.find("not uniform") != std::string::npos);
    // The unmodified fixture does NOT throw, so the refusal is scoped to the
    // defect rather than to the key.
    TempFile good(BuildFixture());
    const vllm::GgufFile g2 = vllm::GgufFile::Open(good.path());
    CHECK_NOTHROW((void)vllm::Qwen4ExpHfConfigFromGguf(g2));
  }

  SUBCASE("a GGUF source with no file") {
    // Reached by a caller that set the KIND without the FILE. It must refuse,
    // not segfault, and it must not degrade into the safetensors message.
    const nlohmann::json doc = nlohmann::json::object();
    TempFile f(BuildFixture());
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    const vllm::HfConfig cfg = vllm::Qwen4ExpHfConfigFromGguf(g);
    const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(cfg);
    vllm::ModelSource src{};
    src.kind = vllm::ModelSource::Kind::kGguf;
    std::string msg;
    try {
      (void)reg.factory->load_weights(reg, cfg, src);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(msg.find("carries no file") != std::string::npos);
    CHECK(msg.find("safetensors") == std::string::npos);
    (void)doc;
  }
}

TEST_CASE("qwen4_exp GGUF: a non-negative ssm_a refuses rather than making a NaN") {
  // The converter writes `-exp(A_log)`, so every value is strictly negative.
  // `log(-x)` of a non-negative value is NaN, and a NaN A_log poisons the whole
  // recurrence with nothing upstream of the first garbage token to explain it.
  GgufModelBuilder b;
  const std::string good = BuildFixture();
  TempFile f(good);
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig cfg = vllm::Qwen4ExpHfConfigFromGguf(g);
  (void)b;

  // Rebuild with a POSITIVE ssm_a on layer 0 by rewriting those four floats in
  // place: the tensor is f32 and the builder wrote them contiguously, so the
  // pattern is unique enough to find and the rest of the file is untouched.
  std::string bad = good;
  const std::vector<float> neg = {-1.0F, -2.0F, -3.0F, -4.0F};
  std::string needle(neg.size() * 4, '\0');
  std::memcpy(needle.data(), neg.data(), needle.size());
  const size_t at = bad.find(needle);
  REQUIRE(at != std::string::npos);
  const std::vector<float> pos = {1.0F, 2.0F, 3.0F, 4.0F};
  std::memcpy(bad.data() + at, pos.data(), needle.size());

  TempFile bf(bad);
  const vllm::GgufFile bg = vllm::GgufFile::Open(bf.path());
  std::string msg;
  try {
    (void)vllm::LoadQwen4ExpFromGguf(bg, cfg, vt::DeviceType::kCPU);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  CHECK(msg.find("ssm_a") != std::string::npos);
  CHECK(msg.find("must be negative") != std::string::npos);
}
