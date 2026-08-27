// Qwen4-Exp W2 gate — issue #1987, campaign issue #1978, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHY THIS FILE IS THE WHOLE GATE FOR THESE TWO COMPONENTS. They are the only
// parts of `Qwen4ExpForConditionalGeneration` with no vLLM op, so there is no
// mirrored implementation to diff against, and the spec's `## Gates` admits no
// token gate for this row at all until an arm runs — nothing published fits any
// fleet device. G0, component goldens, is the only gate reachable today. Below
// it there is nothing: an n-gram id computed wrong does not crash and does not
// change a shape, it reads a different row of a 320-million-row table and the
// model emits plausible text.
//
// ORACLE: huggingface/transformers **v5.16.0**, this row's accepted lane pin
// (spec `## Oracles`). Every expected value in
// `qwen4_exp_ple_goldens.inc` was produced by EXECUTING upstream's own bytes:
// `scripts/gen-qwen4-exp-ple-goldens.py` fetches
// `models/qwen4_exp/modeling_qwen4_exp.py` and `cache_utils.py` from
// raw.githubusercontent.com at the `v5.16.0` tag and `exec`s the named line
// ranges verbatim, never transcribing them, so a golden here is an oracle
// observation and not a prediction. The n-gram ids are the one value upstream
// does not put on a return path (`forward` returns embeddings, :1114); they are
// RECOVERED from its own output by filling row i of the embedding with the
// scalar i, never rebuilt by re-running the loop at :1097-1112, because the
// generator and `BuildNGramIds` would then share one reading of those lines.
// The real-config block is confirmed a fourth way beyond the three in #1987:
// `vocab_size = 248320`, read from the released `config.json`, is the UNIQUE
// preimage below 2e6 of the published `layer_multipliers`.
//
// What it proves, on CPU, with no GPU and without the ~360 GB checkpoint:
//   (1) the splitmix64 chain is UNSIGNED — divergence site #1. Four of the
//       eight probes return a value above 2^63, so an `int64_t` port whose
//       `>>` went arithmetic fails on the raw chain, before any derived value;
//   (2) `% half_bound` is UNSIGNED — divergence site #2 — pinned through the
//       three published multipliers at the REAL config;
//   (3) the head vocab sizes are the successive primes after 19999999, their
//       offsets are the exclusive prefix sum, and the padded table has exactly
//       90 unaddressable rows;
//   (4) `_shift_right_ignore_eos` gets the EOS-SEGMENT semantics right,
//       including the case that separates it from a plain shift: an EOS token
//       belongs to the segment it TERMINATES, not the one it opens;
//   (5) the n-gram ids from prefill(10)+decode+decode equal the single-shot
//       12-token prefill — the conv-state-2 history, EOS-padded and never
//       zero-padded;
//   (6) the signed-sqrt gate CLAMPS BEFORE THE SQRT: the magnitude floor is
//       1e-3, tiny scores are amplified rather than squashed, and exactly zero
//       maps to zero;
//   (7) the dilated depthwise conv reads lags {9, 6, 3, 0}, proved by a
//       one-hot tap per channel over an impulse, so a unit-stride or reversed
//       tap order moves the response;
//   (8) the whole PLE forward matches upstream, and its incremental arm equals
//       its single-shot arm through the 9-column state;
//   (9) `conv_mask` masks BOTH the skip term and the conv input (:1185-1187),
//       and the mask carries through the 9-column state across a chunk break;
//  (10) `eos_token_id` is range-checked like any other id. It is the one id in
//       the mix that does not come from `input_ids`, it is on the FIRST TOKEN
//       OF EVERY SEQUENCE, and at the struct's own `-1` default ours and
//       upstream produce DIFFERENT rows of the table with no exception on
//       either side.
//
// `qwen4_exp_ple.h` is a MODEL-PRIVATE header under `src/`, like
// `dots3_note.h`: W2 ships no public ABI, because nothing is reachable from a
// production entry point until W5 assembles the model. See the header's scope
// block and the spec's `## Owed`.

#include "vllm/model_executor/models/qwen4_exp_ple.h"

#include <doctest/doctest.h>

#include "support/max_abs_diff.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

#include "vllm/models/qwen4_exp_ple_goldens.inc"

using vllm::qwen4_exp::BuildLayerMultipliers;
using vllm::qwen4_exp::BuildNGramIds;
using vllm::qwen4_exp::BuildNGramTableLayout;
using vllm::qwen4_exp::FindNthPrimeAfter;
using vllm::qwen4_exp::IsPrime;
using vllm::qwen4_exp::NGramTableLayout;
using vllm::qwen4_exp::PleForward;
using vllm::qwen4_exp::PleGeometry;
using vllm::qwen4_exp::PleSequenceState;
using vllm::qwen4_exp::PleShortConv;
using vllm::qwen4_exp::PleWeights;
using vllm::qwen4_exp::SignedSqrtGate;
using vllm::qwen4_exp::SplitMix64;

// The tiny config the runnable goldens were generated at. Kept in one place so
// a test cannot silently disagree with the generator about a shape.
PleGeometry TinyGeometry() {
  PleGeometry geom;
  geom.hidden_size = kTinyHiddenSize;
  geom.hc_count = kTinyHcCount;
  geom.ple_embed_dim = kTinyPleEmbedDim;
  geom.ple_conv_kernel_size = kTinyConvKernel;
  geom.ngram_size = kTinyNgramSize;
  geom.heads_per_ngram = kTinyHeadsPerNgram;
  geom.ngram_vocab_size_base = kTinyNgramVocabBase;
  geom.make_ngram_vocab_size_divisible_by = kTinyVocabDivisor;
  geom.vocab_size = kTinyVocabSize;
  geom.eos_token_id = kTinyEosTokenId;
  geom.seed = kTinySeed;
  geom.rms_norm_eps = 1e-6;
  return geom;
}

PleWeights TinyWeights() {
  PleWeights w;
  w.ngram_embedding = kPleNgramEmbeddingWeight;
  w.key_proj = kPleKeyProjWeight;
  w.value_proj = kPleValueProjWeight;
  w.norm_key = kPleNormKeyWeight;
  w.norm_query = kPleNormQueryWeight;
  w.norm_conv = kPleNormConvWeight;
  w.conv1d = kPleConv1dWeight;
  return w;
}

}  // namespace

TEST_CASE("qwen4_exp splitmix64 is unsigned end to end") {
  // Divergence site #1. Assert the property that makes it load-bearing rather
  // than only the values: a signed `>>` cannot be told apart on inputs whose
  // top bit is clear, so the probe set has to contain some whose top bit is set.
  int top_bit_set = 0;
  for (const auto& probe : kSplitMix64) {
    CHECK(SplitMix64(probe.in) == probe.out);
    if ((probe.out >> 63) != 0U) ++top_bit_set;
  }
  CHECK(top_bit_set >= 3);
}

TEST_CASE("qwen4_exp layer multipliers match the released checkpoint") {
  // Divergence site #2, at the REAL config. These three values were published
  // in #1987 from three independent readings (the chain, a range read of
  // `model-00005-of-00131.safetensors`, and the GGUF key), and `seed` is 1234
  // because `config.seed` is ABSENT from the published config.json.
  const std::vector<int64_t> got =
      BuildLayerMultipliers(kRealVocabSize, 3, /*ple_layer_index=*/0, kRealSeed);
  REQUIRE(got.size() == 3U);
  for (int i = 0; i < 3; ++i) CHECK(got[static_cast<size_t>(i)] == kRealLayerMultipliers[i]);

  // Every multiplier is ODD by construction (`2 * x + 1`) and POSITIVE. A
  // signed modulo makes the residue negative, which makes the multiplier even
  // AND negative; both halves are checked so neither alone can carry the case.
  for (int64_t m : got) {
    CHECK(m > 0);
    CHECK(m % 2 == 1);
  }
  // The bound the int64 forward rests on: multiplier_max * vocab_size < 2^63.
  for (int64_t m : got) CHECK(m < INT64_MAX / kRealVocabSize);
}

TEST_CASE("qwen4_exp n-gram head vocab sizes are the successive primes") {
  // `_is_prime` (modeling_qwen4_exp.py:998-1006) branch by branch. Both call
  // sites start above 19999998, so nothing else in this file ever reaches the
  // even arm, the `< 2` arm or 2 itself; without the block below,
  // `if (value % 2 == 0) return value == 2;` can be replaced by `return false`
  // and the whole gate stays green (mutation C3 in the fresh review).
  CHECK(IsPrime(2));
  CHECK_FALSE(IsPrime(4));
  CHECK_FALSE(IsPrime(1));
  CHECK_FALSE(IsPrime(0));
  CHECK_FALSE(IsPrime(-7));
  CHECK(IsPrime(3));
  CHECK_FALSE(IsPrime(9));
  CHECK_FALSE(IsPrime(20000000));   // even, at the scale the model uses
  CHECK_FALSE(IsPrime(20000001));   // odd composite: the trial-division arm

  int64_t running = 0;
  for (int i = 0; i < 16; ++i) {
    CHECK(FindNthPrimeAfter(20000000 - 1, i + 1) == kRealHeadVocabSizes[i]);
    CHECK(kRealHeadOffsets[i] == running);
    running += kRealHeadVocabSizes[i];
  }
  CHECK(running == kRealTotalVocabSize);

  PleGeometry real;
  real.hidden_size = 2560;
  real.hc_count = 4;
  real.ple_embed_dim = 2560;
  real.ngram_size = 3;
  real.heads_per_ngram = 8;
  real.vocab_size = kRealVocabSize;
  real.eos_token_id = 248044;
  real.seed = kRealSeed;
  const NGramTableLayout layout = BuildNGramTableLayout(real, /*ple_layer_index=*/0);
  REQUIRE(layout.head_vocab_sizes.size() == 16U);
  for (int i = 0; i < 16; ++i) {
    CHECK(layout.head_vocab_sizes[static_cast<size_t>(i)] == kRealHeadVocabSizes[i]);
    CHECK(layout.head_offsets[static_cast<size_t>(i)] == kRealHeadOffsets[i]);
  }
  CHECK(layout.total_vocab_size == kRealTotalVocabSize);
  CHECK(layout.padded_vocab_size == kRealPaddedVocabSize);
  // 90 rows the hash can never address, which is what the padding costs.
  CHECK(layout.padded_vocab_size - layout.total_vocab_size == 90);
  CHECK(real.head_dim_per_ngram() == 160);
  CHECK(real.short_conv_state_len() == 9);   // (4-1)*3, NOT kernel-1
  CHECK(real.stream_width() == 10240);
}

TEST_CASE("qwen4_exp _shift_right_ignore_eos honours EOS segments") {
  std::vector<int64_t> got(static_cast<size_t>(kShiftSeqLen));
  for (int64_t shift = 0; shift < 3; ++shift) {
    vllm::qwen4_exp::ShiftRightIgnoreEos(kShiftInput, kShiftSeqLen, shift,
                                         kTinyEosTokenId, got.data());
    for (int64_t i = 0; i < kShiftSeqLen; ++i) {
      CHECK(got[static_cast<size_t>(i)] == kShiftExpected[shift][i]);
    }
  }
  // The one case that separates this from a plain shift-with-EOS-fill: index 4
  // IS an EOS and still reads token 3, because the "previous EOS" scan is
  // strictly-before, so an EOS belongs to the segment it terminates. Index 5
  // opens a new segment and therefore reads EOS.
  CHECK(kShiftInput[4] == kTinyEosTokenId);
  CHECK(kShiftExpected[1][4] == kShiftInput[3]);
  CHECK(kShiftExpected[1][5] == kTinyEosTokenId);
}

TEST_CASE("qwen4_exp n-gram ids: incremental decode equals single-shot prefill") {
  const PleGeometry geom = TinyGeometry();
  const NGramTableLayout layout = BuildNGramTableLayout(geom, /*ple_layer_index=*/0);
  REQUIRE(layout.head_vocab_sizes.size() == static_cast<size_t>(geom.ngram_heads()));
  for (int64_t h = 0; h < geom.ngram_heads(); ++h) {
    CHECK(layout.head_vocab_sizes[static_cast<size_t>(h)] == kTinyHeadVocabSizes[h]);
    CHECK(layout.head_offsets[static_cast<size_t>(h)] == kTinyHeadOffsets[h]);
  }
  for (int64_t i = 0; i < geom.ngram_size; ++i) {
    CHECK(layout.layer_multipliers[static_cast<size_t>(i)] == kTinyLayerMultipliers[i]);
  }
  CHECK(layout.total_vocab_size == kTinyTotalVocabSize);
  CHECK(layout.padded_vocab_size == kTinyPaddedVocabSize);

  const int64_t heads = geom.ngram_heads();
  SUBCASE("single shot") {
    PleSequenceState state;
    state.Reset(geom);
    std::vector<int64_t> ids(static_cast<size_t>(kNgramTotalLen * heads));
    BuildNGramIds(geom, layout, kNgramTokens, kNgramTotalLen, &state, ids.data());
    for (int64_t t = 0; t < kNgramTotalLen; ++t) {
      for (int64_t h = 0; h < heads; ++h) {
        CHECK(ids[static_cast<size_t>(t * heads + h)] == kNgramExpectedIds[t][h]);
      }
    }
    // The history the next step will read is the LAST context_len tokens.
    REQUIRE(state.tokens.size() == 2U);
    CHECK(state.tokens[0] == kNgramTokens[kNgramTotalLen - 2]);
    CHECK(state.tokens[1] == kNgramTokens[kNgramTotalLen - 1]);
  }

  SUBCASE("prefill then two decode steps") {
    PleSequenceState state;
    state.Reset(geom);
    // Seeded with EOS, never with zero: `update_conv_state` pads with 0, which
    // is a VALID token id, so upstream works around it explicitly.
    CHECK(state.tokens[0] == kTinyEosTokenId);
    CHECK(state.tokens[1] == kTinyEosTokenId);

    std::vector<int64_t> ids(static_cast<size_t>(kNgramTotalLen * heads));
    BuildNGramIds(geom, layout, kNgramTokens, kNgramPrefillLen, &state, ids.data());
    for (int64_t t = kNgramPrefillLen; t < kNgramTotalLen; ++t) {
      BuildNGramIds(geom, layout, kNgramTokens + t, 1, &state,
                    ids.data() + t * heads);
    }
    for (int64_t t = 0; t < kNgramTotalLen; ++t) {
      for (int64_t h = 0; h < heads; ++h) {
        CHECK(ids[static_cast<size_t>(t * heads + h)] == kNgramExpectedIds[t][h]);
      }
    }
  }

  SUBCASE("an out-of-range token id is refused by name, not silently overflowed") {
    PleSequenceState state;
    state.Reset(geom);
    const int64_t bad[1] = {kTinyVocabSize};
    std::vector<int64_t> ids(static_cast<size_t>(heads));
    CHECK_THROWS_AS(BuildNGramIds(geom, layout, bad, 1, &state, ids.data()),
                    std::invalid_argument);
  }

  SUBCASE("an out-of-range eos_token_id is refused too") {
    // `eos_token_id` is the one id in the mix that does NOT come from
    // `input_ids`, so the loop above cannot see it: `Reset` seeds the history
    // with it and `_shift_right_ignore_eos` emits it at every segment start,
    // which puts it on the FIRST TOKEN OF EVERY SEQUENCE. Measured against
    // transformers v5.16.0 on the 12 tokens below, reading the ids out of
    // upstream's own `forward` with an invertible embedding:
    //     eos = -1       upstream row 0 [2, 35, 67, 96]   ours [8, 30, 67, 96]
    //     eos = 1000000  upstream row 0 [9, 38, 65, 108]  ours [9, 38, 81, 83]
    // No exception, no shape change, a different row of the table. Upstream is
    // safe without a check because `config.eos_token_id` cannot be out of
    // range; our geometry is a plain struct whose eos DEFAULTS TO -1, and
    // upstream's config even admits a LIST (modeling_qwen4_exp.py:1032 takes
    // element [0]), so a loader has a real way to mis-set it.
    for (const int64_t bad_eos : {int64_t{-1}, kTinyVocabSize, int64_t{1000000}}) {
      PleGeometry bad = geom;
      bad.eos_token_id = bad_eos;
      CHECK_THROWS_AS(BuildNGramTableLayout(bad, /*ple_layer_index=*/0),
                      std::invalid_argument);
      PleSequenceState state;
      state.Reset(bad);
      std::vector<int64_t> ids(static_cast<size_t>(heads));
      CHECK_THROWS_AS(BuildNGramIds(bad, layout, kNgramTokens, 1, &state, ids.data()),
                      std::invalid_argument);
    }
    // The valid boundary values are NOT refused, so the guard cannot be a
    // blanket refusal wearing a range check.
    for (const int64_t ok_eos : {int64_t{0}, kTinyVocabSize - 1}) {
      PleGeometry ok = geom;
      ok.eos_token_id = ok_eos;
      CHECK_NOTHROW(BuildNGramTableLayout(ok, /*ple_layer_index=*/0));
      PleSequenceState state;
      state.Reset(ok);
      std::vector<int64_t> ids(static_cast<size_t>(heads));
      CHECK_NOTHROW(BuildNGramIds(ok, layout, kNgramTokens, 1, &state, ids.data()));
    }
  }
}

TEST_CASE("qwen4_exp signed-sqrt gate clamps before the sqrt") {
  std::vector<float> got;
  got.reserve(static_cast<size_t>(kGateCount));
  for (int64_t i = 0; i < kGateCount; ++i) got.push_back(SignedSqrtGate(kGateInput[i]));
  // Bounds, and why they are what they are. Observed max|diff| against the
  // oracle is 2.38419e-07 = 2^-22 for BOTH the conv and the full forward, and
  // below 1e-07 for the gate, whose sqrt is effectively exact. That single
  // number means two different things on the two sites, so state them apart:
  //   * conv, bound 1e-6:  argmax element silu(3.0) = 2.8577, one ULP there is
  //     2.384e-07, so the observation is ONE ULP and the bound is ~4x above it;
  //   * forward, bound 1e-5: argmax element 1.08830333, one ULP there is
  //     1.192e-07, so the observation is TWO ULP and the bound is ~42x above it.
  // Both bounds sit above the observation rather than on it, to survive a
  // different libm, and ~4 orders BELOW the O(0.1) error any real defect here
  // produces: every mutation in the table went red. R5 in the fresh review —
  // dropping `rms_norm_eps` entirely, a 1e-6 perturbation — still goes red at
  // 1e-5, so the looser of the two bounds discriminates.
  const double worst =
      vllm_test::MaxAbsDiff(got, kGateExpected, static_cast<size_t>(kGateCount));
  MESSAGE("signed-sqrt gate max|diff| vs transformers v5.16.0 = " << worst);
  CHECK(worst < 1e-7);
  // The three properties the values encode, asserted so the intent survives a
  // regenerated golden: the origin maps to zero, the magnitude floor is 1e-3
  // rather than 1e-6, and a tiny score is AMPLIFIED.
  CHECK(SignedSqrtGate(0.0F) == 0.0F);
  CHECK(std::abs(SignedSqrtGate(1e-12F)) > 9.9e-4F);
  CHECK(std::abs(SignedSqrtGate(1e-12F)) > std::abs(1e-12F) * 1e6F);
  CHECK(SignedSqrtGate(-1e-12F) < 0.0F);
}

TEST_CASE("qwen4_exp PLE conv reads lags 9, 6, 3 and 0") {
  PleGeometry geom = TinyGeometry();
  const int64_t width = geom.stream_width();
  const int64_t kernel = geom.ple_conv_kernel_size;
  REQUIRE(geom.short_conv_state_len() == kTinyShortConvStateLen);

  // One-hot tap per channel: channel c carries its whole weight at kernel index
  // c % 4. A unit-stride conv, a reversed tap order or a `kernel-1` state all
  // move where the impulse lands.
  std::vector<float> conv_weight(static_cast<size_t>(width * kernel), 0.0F);
  for (int64_t c = 0; c < width; ++c) {
    conv_weight[static_cast<size_t>(c * kernel + (c % kernel))] = 1.0F;
  }
  std::vector<float> normed(static_cast<size_t>(kTapSeqLen * width), 0.0F);
  for (int64_t c = 0; c < width; ++c) normed[static_cast<size_t>(c)] = kTapImpulse;

  PleSequenceState state;
  state.Reset(geom);
  std::vector<float> got(static_cast<size_t>(kTapSeqLen * width));
  PleShortConv(geom, conv_weight.data(), normed.data(), kTapSeqLen, &state,
               got.data());
  const double worst = vllm_test::MaxAbsDiff(got, &kTapExpected[0][0],
                                             static_cast<size_t>(kTapSeqLen * width));
  MESSAGE("dilated conv max|diff| vs transformers v5.16.0 = " << worst);
  CHECK(worst < 1e-6);
  // Read the lag structure straight off the response so a regenerated golden
  // cannot quietly change it: channel c responds at t = (3 - c % 4) * 3.
  for (int64_t c = 0; c < width; ++c) {
    const int64_t lag = (kernel - 1 - (c % kernel)) * geom.ngram_size;
    CHECK(got[static_cast<size_t>(lag * width + c)] > 1.0F);
  }
}

TEST_CASE("qwen4_exp PLE forward matches transformers v5.16.0") {
  const PleGeometry geom = TinyGeometry();
  const NGramTableLayout layout = BuildNGramTableLayout(geom, /*ple_layer_index=*/0);
  const PleWeights weights = TinyWeights();
  const int64_t width = geom.stream_width();

  SUBCASE("single shot") {
    PleSequenceState state;
    state.Reset(geom);
    std::vector<float> got(static_cast<size_t>(kNgramTotalLen * width));
    PleForward(geom, layout, weights, kPleHiddenStates, kNgramTokens, kNgramTotalLen,
               /*conv_mask=*/nullptr, &state, got.data());
    const double worst = vllm_test::MaxAbsDiff(got, kPleExpectedOutput,
                                              static_cast<size_t>(kNgramTotalLen * width));
    MESSAGE("PLE forward max|diff| vs transformers v5.16.0 = " << worst);
    CHECK(worst < 1e-5);
  }

  SUBCASE("prefill then two decode steps, through the 9-column state") {
    PleSequenceState state;
    state.Reset(geom);
    std::vector<float> got(static_cast<size_t>(kNgramTotalLen * width));
    PleForward(geom, layout, weights, kPleHiddenStates, kNgramTokens, kNgramPrefillLen,
               nullptr, &state, got.data());
    for (int64_t t = kNgramPrefillLen; t < kNgramTotalLen; ++t) {
      PleForward(geom, layout, weights, kPleHiddenStates + t * width, kNgramTokens + t,
                 1, nullptr, &state, got.data() + t * width);
    }
    const double worst = vllm_test::MaxAbsDiff(got, kPleExpectedOutput,
                                              static_cast<size_t>(kNgramTotalLen * width));
    MESSAGE("PLE forward max|diff| vs transformers v5.16.0 = " << worst);
    CHECK(worst < 1e-5);
  }

  SUBCASE("the skip term is the UN-NORMED copy") {
    // Zero the conv weights: the conv contributes silu(0) = 0 and the output
    // collapses to the skip term alone. If the skip took the NORMED copy the
    // result would be RMS-normalised, so its per-group RMS would be pinned near
    // 1; it is not, and that is the fork the spec warns about.
    std::vector<float> zero_conv(static_cast<size_t>(width * geom.ple_conv_kernel_size),
                                 0.0F);
    PleWeights w = weights;
    w.conv1d = zero_conv.data();
    PleSequenceState state;
    state.Reset(geom);
    std::vector<float> got(static_cast<size_t>(kNgramTotalLen * width));
    PleForward(geom, layout, w, kPleHiddenStates, kNgramTokens, kNgramTotalLen,
               nullptr, &state, got.data());

    bool any_group_far_from_unit_rms = false;
    for (int64_t t = 0; t < kNgramTotalLen; ++t) {
      for (int64_t s = 0; s < geom.hc_count; ++s) {
        double sumsq = 0.0;
        for (int64_t d = 0; d < geom.hidden_size; ++d) {
          const double v = got[static_cast<size_t>(t * width + s * geom.hidden_size + d)];
          sumsq += v * v;
        }
        const double rms = std::sqrt(sumsq / static_cast<double>(geom.hidden_size));
        if (std::abs(rms - 1.0) > 0.2) any_group_far_from_unit_rms = true;
      }
    }
    CHECK(any_group_far_from_unit_rms);
  }
}

TEST_CASE("qwen4_exp PLE conv_mask masks BOTH the skip term and the conv input") {
  // modeling_qwen4_exp.py:1185-1187 masks `gated_value` AND
  // `gated_value_normed`; :204-213 is the multiply it does it with. Masking one
  // of the two is a real port defect and it is invisible to every other case in
  // this file, because they all pass `conv_mask = nullptr`.
  //
  // The mask is a PAIRED obligation with the caller (see the header): a masked
  // position must already carry EOS in `input_ids`, because the hash reads
  // token ids and not activations. `kPleMaskTokens` honours that, so this pins
  // the contract rather than a state nobody would produce. Zeros sit at 3, 4
  // and 11; 3 and 4 are INTERIOR, so the dilation carries them into t = 6, 9
  // and 12 as well, and the 9-column state carries them across a chunk break.
  const PleGeometry geom = TinyGeometry();
  const NGramTableLayout layout = BuildNGramTableLayout(geom, /*ple_layer_index=*/0);
  const PleWeights weights = TinyWeights();
  const int64_t width = geom.stream_width();

  SUBCASE("single shot") {
    PleSequenceState state;
    state.Reset(geom);
    std::vector<float> got(static_cast<size_t>(kNgramTotalLen * width));
    PleForward(geom, layout, weights, kPleHiddenStates, kPleMaskTokens, kNgramTotalLen,
               kPleConvMask, &state, got.data());
    const double worst = vllm_test::MaxAbsDiff(got, kPleMaskedExpectedOutput,
                                              static_cast<size_t>(kNgramTotalLen * width));
    MESSAGE("masked PLE forward max|diff| vs transformers v5.16.0 = " << worst);
    CHECK(worst < 1e-5);
  }

  SUBCASE("prefill then two decode steps: the mask reaches the 9-column state") {
    PleSequenceState state;
    state.Reset(geom);
    std::vector<float> got(static_cast<size_t>(kNgramTotalLen * width));
    PleForward(geom, layout, weights, kPleHiddenStates, kPleMaskTokens, kNgramPrefillLen,
               kPleConvMask, &state, got.data());
    for (int64_t t = kNgramPrefillLen; t < kNgramTotalLen; ++t) {
      PleForward(geom, layout, weights, kPleHiddenStates + t * width, kPleMaskTokens + t,
                 1, kPleConvMask + t, &state, got.data() + t * width);
    }
    const double worst = vllm_test::MaxAbsDiff(got, kPleMaskedExpectedOutput,
                                              static_cast<size_t>(kNgramTotalLen * width));
    MESSAGE("masked PLE forward max|diff| vs transformers v5.16.0 = " << worst);
    CHECK(worst < 1e-5);
  }

  SUBCASE("the mask is load-bearing: a nullptr mask gives a different answer") {
    // Without this, a golden that happened to equal the unmasked one would gate
    // nothing at all. The masked and unmasked answers must be far apart.
    PleSequenceState masked_state;
    masked_state.Reset(geom);
    std::vector<float> masked(static_cast<size_t>(kNgramTotalLen * width));
    PleForward(geom, layout, weights, kPleHiddenStates, kPleMaskTokens, kNgramTotalLen,
               kPleConvMask, &masked_state, masked.data());

    PleSequenceState plain_state;
    plain_state.Reset(geom);
    std::vector<float> plain(static_cast<size_t>(kNgramTotalLen * width));
    PleForward(geom, layout, weights, kPleHiddenStates, kPleMaskTokens, kNgramTotalLen,
               /*conv_mask=*/nullptr, &plain_state, plain.data());
    const double apart = vllm_test::MaxAbsDiff(masked, plain.data(),
                                               static_cast<size_t>(kNgramTotalLen * width));
    MESSAGE("masked vs unmasked max|diff| = " << apart);
    CHECK(apart > 1e-2);

    // A masked row keeps NO skip term, so what survives there is the conv
    // output alone. Zero the conv weights and the masked rows must be exactly
    // zero, which separates "the skip term was masked" from "something was".
    std::vector<float> zero_conv(
        static_cast<size_t>(width * geom.ple_conv_kernel_size), 0.0F);
    PleWeights w = weights;
    w.conv1d = zero_conv.data();
    PleSequenceState skip_state;
    skip_state.Reset(geom);
    std::vector<float> skip_only(static_cast<size_t>(kNgramTotalLen * width));
    PleForward(geom, layout, w, kPleHiddenStates, kPleMaskTokens, kNgramTotalLen,
               kPleConvMask, &skip_state, skip_only.data());
    int64_t masked_rows = 0;
    for (int64_t t = 0; t < kNgramTotalLen; ++t) {
      if (kPleConvMask[t] != 0) continue;
      ++masked_rows;
      for (int64_t c = 0; c < width; ++c) {
        CHECK(skip_only[static_cast<size_t>(t * width + c)] == 0.0F);
      }
    }
    CHECK(masked_rows == 3);
  }
}
