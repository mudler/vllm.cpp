// Qwen4-Exp W4 — Qwen Sparse Attention UNIT GATE (row MODEL-MM-QWEN4-EXP,
// issue #1991, .agents/specs/qwen4-exp-flash-next.md).
//
// The goldens in `fixtures/qwen4_exp_qsa_goldens.inc` are the output of the
// UNMODIFIED `Qwen4ExpTextQSAIndexer.forward` at transformers 5.16.0, this row's
// accepted ALGORITHM lane pin (vLLM registers no `qwen4_exp` at 6a5e8f5979, so
// there is nothing on the primary oracle to run). See
// `fixtures/gen_qwen4_exp_qsa_goldens.py` for exactly what was captured and what
// little was transcribed.
//
// TWO ORACLES HERE NEED NO WEIGHTS AND NO GPU, and both are used:
//
//  (1) Below `indexer_budget + compress_ratio - 1` cached tokens every candidate
//      is selected, so QSA must be BIT-IDENTICAL to dense attention. llama.cpp
//      #27742 measures a max logit delta of 0.0 over all 2051 such rows. That
//      gates the whole selection, ordering and masking path.
//  (2) Above it, the selected token index sets must equal the transformers
//      reference, INCLUDING the ragged tail, which is always attended.
//
// AND ONE GATE THAT IS NOT ABOUT CORRECTNESS. `keys_visited` asserts that the
// gather reads only the selected rows. It COUNTS reads at the key-row read; the
// first revision assigned it `sel.size()`, which made the assertion a statement
// about `indices` rather than about the loop, and a gather body doing the full
// dense work passed it (W4 fresh review, mutation M22c). A mask-only QSA is CORRECT and would pass
// every value comparison in this file while forfeiting the long-context speed
// lever the row is aiming at (llama.cpp #27739: a sparse mask over a dense cache
// costs the same as dense attention under CUDA flash attention, because
// `flash_attn_mask_to_KV_max` only scans back to the first tile that is not all
// -inf). Delete the `keys_visited` cases and this suite stops being able to tell
// the two implementations apart.
//
// SCOPE, honestly. Host/CPU reference math only. `Qwen4ExpTextModel` does not
// exist yet (W2/W3/W6a are three sibling waves and the registry wiring is W5),
// so nothing here enters through a production entry point; the spec lists the
// slice under `## Owed` and issue #1978 owns the wiring. No checkpoint, no GPU,
// no speed claim — the speed axis opens at G4, after W6a.
#include "vllm/model_executor/models/qwen4_exp_qsa.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixtures/qwen4_exp_qsa_goldens.inc"

using namespace vllm::qwen4_exp;
namespace g = qwen4_exp_qsa_goldens;

namespace {

// The tiny config the goldens were captured under. Kept beside the goldens'
// own constants so a regenerated fixture with different shapes cannot silently
// keep passing against a stale literal.
QsaConfig GoldenConfig() {
  QsaConfig cfg;
  cfg.index_n_heads = g::kIndexNHeads;
  cfg.index_kv_heads = g::kIndexKvHeads;
  cfg.index_head_dim = g::kIndexHeadDim;
  cfg.token_budget = g::kTokenBudget;
  cfg.compress_ratio = g::kCompressRatio;
  cfg.rotary_dim = g::kRotaryDim;
  cfg.rms_norm_eps = g::kRmsNormEps;
  return cfg;
}

std::vector<float> Slice(const float* p, int64_t n) {
  return std::vector<float>(p, p + n);
}

double RelL2(const std::vector<float>& a, const float* b, int64_t n) {
  double num = 0.0, den = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num) / std::max(std::sqrt(den), 1e-30);
}

// One golden case, resolved from the flat arrays the fixture emits.
struct Case {
  const char* name;
  int64_t seq;
  const float* q_raw;
  const float* k_raw;
  const float* cos;
  const float* sin;
  const float* q_norm_w;
  const float* k_norm_w;
  const float* q_post;
  const float* block_keys;
  const int32_t* selected;
  const float* attn_q;
  const float* attn_k;
  const float* attn_v;
  const float* attn_out;
};

const Case kSubBudget{"sub_budget",       g::kSubBudgetSeq,      g::kSubBudgetQRaw,
                      g::kSubBudgetKRaw,  g::kSubBudgetCos,      g::kSubBudgetSin,
                      g::kSubBudgetQNormW, g::kSubBudgetKNormW,  g::kSubBudgetQPost,
                      g::kSubBudgetBlockKeys, g::kSubBudgetSelected,
                      g::kSubBudgetAttnQ, g::kSubBudgetAttnK,    g::kSubBudgetAttnV,
                      g::kSubBudgetAttnOut};

const Case kOverBudget{"over_budget",       g::kOverBudgetSeq,     g::kOverBudgetQRaw,
                       g::kOverBudgetKRaw,  g::kOverBudgetCos,     g::kOverBudgetSin,
                       g::kOverBudgetQNormW, g::kOverBudgetKNormW, g::kOverBudgetQPost,
                       g::kOverBudgetBlockKeys, g::kOverBudgetSelected,
                       g::kOverBudgetAttnQ, g::kOverBudgetAttnK,   g::kOverBudgetAttnV,
                       g::kOverBudgetAttnOut};

// q after `q_layernorm` and the partial rope, [seq, H, D].
std::vector<float> BuildQ(const Case& c, const QsaConfig& cfg) {
  const int64_t H = cfg.index_n_heads, D = cfg.index_head_dim;
  const std::vector<float> raw = Slice(c.q_raw, c.seq * H * D);
  // The norm is per index head, so [seq * H] rows of width D.
  const std::vector<float> normed =
      QsaRmsNorm(raw, c.seq * H, D, Slice(c.q_norm_w, D), cfg.rms_norm_eps,
                 /*round_to_bf16=*/true);
  // The rope's cos/sin are per POSITION and shared across the heads, so expand
  // the [seq, rotary_dim] table to one row per (position, head).
  std::vector<float> cos(c.seq * H * cfg.rotary_dim);
  std::vector<float> sin(cos.size());
  for (int64_t t = 0; t < c.seq; ++t) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t d = 0; d < cfg.rotary_dim; ++d) {
        cos[(t * H + h) * cfg.rotary_dim + d] = c.cos[t * cfg.rotary_dim + d];
        sin[(t * H + h) * cfg.rotary_dim + d] = c.sin[t * cfg.rotary_dim + d];
      }
    }
  }
  return QsaApplyRotaryLeadingHalf(normed, c.seq * H, D, cos, sin, cfg.rotary_dim,
                                   /*round_to_bf16=*/true);
}

// The side cache: one pooled key per complete block over the whole case.
std::vector<float> BuildBlockKeys(const Case& c, const QsaConfig& cfg) {
  const int64_t D = cfg.index_head_dim;
  const int64_t complete = (c.seq / cfg.compress_ratio) * cfg.compress_ratio;
  return QsaCompressNormRope(Slice(c.k_raw, c.seq * D), complete,
                             Slice(c.k_norm_w, D),
                             Slice(c.cos, c.seq * cfg.rotary_dim),
                             Slice(c.sin, c.seq * cfg.rotary_dim), cfg,
                             /*round_to_bf16=*/true);
}

// The full per-query selection, ascending and -1 padded to index_width().
std::vector<int32_t> SelectFor(const QsaConfig& cfg,
                               const std::vector<float>& q,
                               const std::vector<float>& block_keys,
                               int64_t query_idx) {
  const int64_t H = cfg.index_n_heads, D = cfg.index_head_dim;
  const int64_t kv_len = query_idx + 1;  // causal, no interior masking
  const int64_t nb = kv_len / cfg.compress_ratio;
  std::vector<float> q_row(q.begin() + query_idx * H * D,
                           q.begin() + (query_idx + 1) * H * D);
  std::vector<float> keys(block_keys.begin(), block_keys.begin() + nb * D);
  const std::vector<float> scores = QsaBlockScores(q_row, keys, nb, cfg);
  const std::vector<int64_t> blocks =
      QsaTopkBlocks(scores, nb, std::min(cfg.block_topk(), nb));
  return QsaSelectedTokenIndices(blocks, nb, kv_len, cfg);
}

// Which refusal fired. Two config refusals can cover for each other, and
// `CHECK_THROWS` cannot tell them apart: it only reports that SOMETHING threw.
// VT_CHECK appends ` at <file>:<line>`, so match on the message text only — an
// exact-`what()` assertion would go stale on any edit above the line it names.
template <typename F>
bool RefusesWith(F&& fn, const char* needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

int64_t SelectedCount(const std::vector<int32_t>& idx) {
  int64_t n = 0;
  for (int32_t v : idx) {
    if (v < 0) break;
    ++n;
  }
  return n;
}

}  // namespace

// ── The config invariants ────────────────────────────────────────────────────

TEST_CASE("qsa-config: the published shape resolves, and every upstream refusal fires") {
  QsaConfig cfg;  // the published Qwen3.8-Flash-Next values
  CHECK(cfg.index_n_heads == 4);
  CHECK(cfg.index_kv_heads == 1);
  CHECK(cfg.index_head_dim == 128);
  CHECK(cfg.token_budget == 2048);
  CHECK(cfg.compress_ratio == 4);
  CHECK(cfg.block_topk() == 512);
  // budget + compress_ratio - 1: the incomplete trailing block is attended ON
  // TOP of the budget, which is why the buffer is not just `budget` wide.
  CHECK(cfg.index_width() == 2051);
  QsaValidateConfig(cfg);

  // configuration_qwen4_exp.py:221 — "Qwen4-Exp QSA requires indexer_kv_heads=1."
  QsaConfig two_kv = cfg;
  two_kv.index_kv_heads = 2;
  CHECK_THROWS(QsaValidateConfig(two_kv));

  // :223 — "indexer_budget must be divisible by indexer_compress_ratio."
  QsaConfig ragged = cfg;
  ragged.token_budget = 2049;
  CHECK_THROWS(QsaValidateConfig(ragged));

  // :227-231 — the attention rope must FIT the index head. The probe is EVEN,
  // so the evenness refusal below cannot cover for this one. `index_head_dim + 1`
  // was the original probe and it is odd: EITHER refusal alone still threw on it,
  // so deleting this one — the refusal upstream actually raises — left the suite
  // green (W4 fresh review, mutation M3).
  QsaConfig wide_rope = cfg;
  wide_rope.rotary_dim = cfg.index_head_dim + 2;
  CHECK(RefusesWith([&] { QsaValidateConfig(wide_rope); },
                    "RoPE dimensions must fit the QSA index head"));

  // OURS, with no upstream counterpart: `rotate_half` needs an even span. The
  // probe FITS the index head, so only this refusal can fire on it (M3b).
  QsaConfig odd_rope = cfg;
  odd_rope.rotary_dim = cfg.index_head_dim - 1;
  CHECK(RefusesWith([&] { QsaValidateConfig(odd_rope); },
                    "rotary_dim must be even"));

  QsaConfig zero = cfg;
  zero.index_n_heads = 0;
  CHECK_THROWS(QsaValidateConfig(zero));
}

// ── The side cache ───────────────────────────────────────────────────────────

TEST_CASE("qsa-cache: one key vector per compress_ratio tokens, 64 B/token/layer") {
  const QsaConfig cfg;
  const QsaSideCacheSpec spec = QsaMakeSideCacheSpec(cfg, /*elem_bytes=*/2);
  // MLAAttentionSpec(num_kv_heads=1, head_size=128, tokens_per_state=4).
  CHECK(spec.num_kv_heads == 1);
  CHECK(spec.head_size == 128);
  CHECK(spec.tokens_per_state == 4);

  // KEY-ONLY: one vector per state, not 2x for K+V. 2 * 1 * 128 / 4 = 64.
  CHECK(spec.BytesPerTokenPerLayer() == 64);
  // A quarter of what a per-token index cache would cost, which is the whole
  // argument for compressing at all.
  CHECK(spec.BytesPerTokenPerLayer() * spec.tokens_per_state == 2 * 128);
  CHECK(spec.BytesForTokens(4096) == 4096 * 64);

  // Only a COMPLETE block produces a state: the compressor early-exits unless
  // `(position + 1) % compress_ratio == 0`. The ragged tail costs no state.
  CHECK(spec.StatesForTokens(0) == 0);
  CHECK(spec.StatesForTokens(3) == 0);
  CHECK(spec.StatesForTokens(4) == 1);
  CHECK(spec.StatesForTokens(7) == 1);
  CHECK(spec.StatesForTokens(8) == 2);

  // get_compressed_slot_mapping's per-token body: PAD_ID -1 off a boundary.
  CHECK(QsaCompressedSlot(0, 4) == -1);
  CHECK(QsaCompressedSlot(2, 4) == -1);
  CHECK(QsaCompressedSlot(3, 4) == 0);
  CHECK(QsaCompressedSlot(7, 4) == 1);
  CHECK(QsaCompressedSlot(11, 4) == 2);
}

// ── The indexer's own math, against the oracle ───────────────────────────────

TEST_CASE("qsa-indexer: the block score is relu(q.k) summed over heads, / sqrt(D)") {
  // A HAND-DERIVED case, because the selection goldens cannot see this. Top-k is
  // invariant under any positive rescale of every score, so inheriting
  // DeepSeek-V4's extra `head_scale = n_heads ** -0.5` fold
  // (models/deepseek_v4/attention.py:930) changes every score and moves no
  // selection at all. A mutation proved that: it left the whole suite green
  // until this case existed. The constant still matters — it is the logits'
  // dynamic range, which is what an fp8 indexer kernel is quantised against.
  QsaConfig cfg;
  cfg.index_n_heads = 2;
  cfg.index_head_dim = 4;
  cfg.token_budget = 4;
  cfg.compress_ratio = 2;
  cfg.rotary_dim = 2;

  // q[h=0] = e0, q[h=1] = e1.
  const std::vector<float> q = {1, 0, 0, 0, 0, 1, 0, 0};
  const std::vector<float> keys = {
      2,  3,  0, 0,  // dots  2 and  3 -> relu sum 5 -> 5 / sqrt(4) = 2.5
      -1, -1, 0, 0,  // dots -1 and -1 -> BOTH clamped -> 0
      4,  -2, 0, 0,  // dots  4 and -2 -> only the first survives -> 4/2 = 2.0
  };
  const std::vector<float> scores = QsaBlockScores(q, keys, 3, cfg);
  REQUIRE(scores.size() == 3);
  CHECK(scores[0] == doctest::Approx(2.5));
  CHECK(scores[1] == doctest::Approx(0.0));
  CHECK(scores[2] == doctest::Approx(2.0));
  // Without the ReLU block 1 would score -1 and block 2 would score 1.0, so the
  // two are ordered differently as well as scaled differently.
  CHECK(scores[0] > scores[2]);
  CHECK(scores[2] > scores[1]);
}

TEST_CASE("qsa-indexer: q_layernorm + partial rope match transformers 5.16.0") {
  const QsaConfig cfg = GoldenConfig();
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));  // doctest stringifies a char* as a bool
    const std::vector<float> q = BuildQ(*c, cfg);
    const int64_t n = c->seq * cfg.index_n_heads * cfg.index_head_dim;
    REQUIRE(static_cast<int64_t>(q.size()) == n);
    // Both sides are bf16-valued, so this is exact rather than a tolerance.
    for (int64_t i = 0; i < n; ++i) CHECK(q[i] == c->q_post[i]);
  }
}

TEST_CASE("qsa-indexer: mean-pooled, normed, block-start-roped keys match the oracle") {
  const QsaConfig cfg = GoldenConfig();
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));  // doctest stringifies a char* as a bool
    const std::vector<float> keys = BuildBlockKeys(*c, cfg);
    const int64_t nb = c->seq / cfg.compress_ratio;
    REQUIRE(static_cast<int64_t>(keys.size()) == nb * cfg.index_head_dim);
    // Tight but not exact: the oracle's mean reduces in torch's order and ours
    // in ascending token order over `compress_ratio` = 4 terms. Everything after
    // the pool — the .to(dtype) round-trip, k_layernorm, the block-start rope —
    // is reproduced operation for operation.
    CHECK(RelL2(keys, c->block_keys, nb * cfg.index_head_dim) < 1e-6);
  }
}

TEST_CASE("qsa-indexer: the pool is a MEAN, not a sum — the /compress_ratio is pinned") {
  // A HAND-DERIVED case, because no golden can see this constant. `k_layernorm`
  // runs on the POOLED key, and RMSNorm is scale-invariant whenever its epsilon
  // is negligible against the mean square. At the published eps = 1e-6, dropping
  // the `/ compress_ratio` and storing a SUM therefore changes nothing any
  // downstream value can observe — it left the whole suite green (W4 fresh
  // review, mutation M12) — and `/4` is exact in binary floating point, so not
  // even the bf16 round-trip catches it. The window is gated (M14) and the
  // round-trip is gated (M13); the WEIGHTING was not.
  //
  // So probe where the norm is not scale-invariant. With eps dominating the mean
  // square the norm is linear in its input, the pooled scale reaches the output,
  // and mean versus sum is a factor of compress_ratio. Everything else is
  // neutralised: cos = 1 and sin = 0 make the rope the identity, the norm weight
  // is zero so `(1.0 + w)` is 1, and round_to_bf16 is off — which is what makes
  // every expected value below EXACT rather than toleranced.
  QsaConfig cfg;
  cfg.index_n_heads = 1;
  cfg.index_kv_heads = 1;
  cfg.index_head_dim = 4;
  cfg.compress_ratio = 2;
  cfg.token_budget = 2;
  cfg.rotary_dim = 2;
  cfg.rms_norm_eps = 10.0f;

  // Two blocks of two raw keys each. Block 0 MEAN-pools to [4, 6, 8, 10]: mean
  // square 54, + eps = 64, so the norm divides by exactly 8. Block 1 pools to
  // [1, 3, 5, 5]: mean square 15, + eps = 25, so it divides by exactly 5.
  const std::vector<float> raw = {
      2.0f, 4.0f, 6.0f,  8.0f,   // block 0, key 0
      6.0f, 8.0f, 10.0f, 12.0f,  // block 0, key 1
      0.0f, 2.0f, 4.0f,  4.0f,   // block 1, key 0
      2.0f, 4.0f, 6.0f,  6.0f};  // block 1, key 1
  const std::vector<float> w(4, 0.0f);
  const std::vector<float> cos(4 * cfg.rotary_dim, 1.0f);
  const std::vector<float> sin(4 * cfg.rotary_dim, 0.0f);

  const std::vector<float> got =
      QsaCompressNormRope(raw, 4, w, cos, sin, cfg, /*round_to_bf16=*/false);
  const std::vector<float> want = {0.5f, 0.75f, 1.0f, 1.25f,
                                   0.2f, 0.6f,  1.0f, 1.0f};
  REQUIRE(got.size() == want.size());
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == want[i]);
  // An unnormalised SUM would pool block 0 to [8, 12, 16, 20], mean square 216,
  // + eps = 226, and emit 8/sqrt(226) = 0.5322... where 0.5 is asserted.
}

TEST_CASE("qsa-indexer: more selected blocks than the budget allows is refused") {
  // `selected_blocks.size() <= cfg.block_topk()`. Deleting it left the suite
  // green (W4 fresh review, mutation M34), because the index-width check a few
  // lines later throws as well: compress_ratio * (block_topk + 1) is
  // token_budget + compress_ratio, exactly ONE token past an index_width() of
  // token_budget + compress_ratio - 1. No token count can separate the two, so
  // the only available discriminator is WHICH refusal fires, and that is what
  // this case asserts.
  QsaConfig cfg;
  cfg.index_head_dim = 4;
  cfg.rotary_dim = 2;
  cfg.token_budget = 8;
  cfg.compress_ratio = 2;  // block_topk() = 4, index_width() = 9
  const int64_t nb = cfg.block_topk() + 1;
  const int64_t kv_len = nb * cfg.compress_ratio;
  std::vector<int64_t> blocks;
  for (int64_t b = 0; b < nb; ++b) blocks.push_back(b);
  CHECK(RefusesWith([&] { QsaSelectedTokenIndices(blocks, nb, kv_len, cfg); },
                    "more blocks selected than the budget allows"));

  // One block fewer is inside the budget and returns normally, so the refusal
  // above is not this function simply refusing everything.
  blocks.pop_back();
  const std::vector<int32_t> ok =
      QsaSelectedTokenIndices(blocks, nb, kv_len, cfg);
  CHECK(SelectedCount(ok) == cfg.block_topk() * cfg.compress_ratio);
}

TEST_CASE("qsa-indexer: selected token sets equal the oracle, ragged tail included") {
  const QsaConfig cfg = GoldenConfig();
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));  // doctest stringifies a char* as a bool
    const std::vector<float> q = BuildQ(*c, cfg);
    const std::vector<float> keys = BuildBlockKeys(*c, cfg);
    for (int64_t qi = 0; qi < c->seq; ++qi) {
      CAPTURE(qi);
      const std::vector<int32_t> got = SelectFor(cfg, q, keys, qi);
      REQUIRE(static_cast<int64_t>(got.size()) == g::kIndexWidth);
      for (int64_t j = 0; j < g::kIndexWidth; ++j) {
        CHECK(got[j] == c->selected[qi * g::kIndexWidth + j]);
      }
    }
  }
}

TEST_CASE("qsa-indexer: the ragged tail is ALWAYS attended") {
  // Not a restatement of the case above: it names the property, and it fires on
  // the queries where the tail is the only thing keeping the count off a
  // multiple of compress_ratio.
  const QsaConfig cfg = GoldenConfig();
  const Case& c = kOverBudget;
  const std::vector<float> q = BuildQ(c, cfg);
  const std::vector<float> keys = BuildBlockKeys(c, cfg);
  int64_t ragged_queries = 0;
  for (int64_t qi = 0; qi < c.seq; ++qi) {
    const int64_t kv_len = qi + 1;
    const int64_t tail = kv_len % cfg.compress_ratio;
    if (tail == 0) continue;
    ++ragged_queries;
    const std::vector<int32_t> got = SelectFor(cfg, q, keys, qi);
    // The last `tail` entries of the ascending buffer are the tail tokens, and
    // they are there whatever the scores said.
    const int64_t n = SelectedCount(got);
    for (int64_t j = 0; j < tail; ++j) {
      CHECK(got[n - tail + j] == static_cast<int32_t>(kv_len - tail + j));
    }
  }
  CHECK(ragged_queries > 0);  // a gate that never fired is not a gate
}

// ── Free oracle (1): sub-budget bit-identity with dense attention ────────────

TEST_CASE("qsa-oracle: below budget + compress_ratio - 1 EVERY candidate is selected") {
  const QsaConfig cfg = GoldenConfig();
  const Case& c = kSubBudget;
  REQUIRE(c.seq == cfg.index_width());  // the largest all-select context
  const std::vector<float> q = BuildQ(c, cfg);
  const std::vector<float> keys = BuildBlockKeys(c, cfg);
  for (int64_t qi = 0; qi < c.seq; ++qi) {
    CAPTURE(qi);
    const std::vector<int32_t> got = SelectFor(cfg, q, keys, qi);
    REQUIRE(SelectedCount(got) == qi + 1);
    for (int64_t j = 0; j <= qi; ++j) CHECK(got[j] == static_cast<int32_t>(j));
    for (int64_t j = qi + 1; j < g::kIndexWidth; ++j) CHECK(got[j] == -1);
  }
}

TEST_CASE("qsa-oracle: a sub-budget GATHER is BIT-IDENTICAL to dense attention") {
  // llama.cpp #27742 claims a max logit delta of 0.0 over all 2051 rows. This is
  // the C++ statement of that claim: with every candidate selected the gather
  // reduces over exactly the dense sequence, in exactly the dense order.
  const QsaConfig cfg = GoldenConfig();
  const Case& c = kSubBudget;
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads,
                DH = g::kHeadDim;
  const std::vector<float> q_all = Slice(c.attn_q, c.seq * HQ * DH);
  const std::vector<float> k_all = Slice(c.attn_k, c.seq * HKV * DH);
  const std::vector<float> v_all = Slice(c.attn_v, c.seq * HKV * DH);
  const std::vector<float> q = BuildQ(c, cfg);
  const std::vector<float> keys = BuildBlockKeys(c, cfg);

  for (int64_t qi = 0; qi < c.seq; ++qi) {
    CAPTURE(qi);
    const int64_t kv_len = qi + 1;
    const std::vector<float> q_row(q_all.begin() + qi * HQ * DH,
                                   q_all.begin() + (qi + 1) * HQ * DH);
    // Dense: the causal prefix, nothing removed, through the INDEPENDENT walk.
    // `want` must not come from the function under test. It did in the first
    // revision of this case, and then the `==` said only that the gather is
    // deterministic under two equal inputs: scaling its output by 2 left the
    // case green (W4 fresh review, mutation M40).
    std::vector<int32_t> dense(cfg.index_width(), -1);
    for (int64_t j = 0; j < kv_len; ++j) dense[j] = static_cast<int32_t>(j);
    int64_t dense_visited = 0, sparse_visited = 0;
    const std::vector<float> want = QsaMaskedAttention(
        q_row, k_all, v_all, dense, kv_len, HQ, HKV, DH, &dense_visited);
    const std::vector<int32_t> sel = SelectFor(cfg, q, keys, qi);
    const std::vector<float> got = QsaGatherAttention(
        q_row, k_all, v_all, sel, kv_len, HQ, HKV, DH, &sparse_visited);
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == want[i]);
    // Below budget the gather is not merely correct, it is doing exactly the
    // dense amount of work: every candidate is selected, so there is nothing to
    // skip. The saving starts above the budget, in the case below.
    CHECK(sparse_visited == dense_visited);
  }
}

// ── Free oracle (2): the consumer against the oracle's own masked attention ──

TEST_CASE("qsa-consumer: the gather reproduces the oracle's masked attention") {
  const QsaConfig cfg = GoldenConfig();
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads,
                DH = g::kHeadDim;
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));  // doctest stringifies a char* as a bool
    const std::vector<float> q_all = Slice(c->attn_q, c->seq * HQ * DH);
    const std::vector<float> k_all = Slice(c->attn_k, c->seq * HKV * DH);
    const std::vector<float> v_all = Slice(c->attn_v, c->seq * HKV * DH);
    const std::vector<float> q = BuildQ(*c, cfg);
    const std::vector<float> keys = BuildBlockKeys(*c, cfg);
    std::vector<float> out;
    out.reserve(c->seq * HQ * DH);
    for (int64_t qi = 0; qi < c->seq; ++qi) {
      const std::vector<float> q_row(q_all.begin() + qi * HQ * DH,
                                     q_all.begin() + (qi + 1) * HQ * DH);
      const std::vector<int32_t> sel = SelectFor(cfg, q, keys, qi);
      const std::vector<float> row = QsaGatherAttention(
          q_row, k_all, v_all, sel, qi + 1, HQ, HKV, DH, nullptr);
      out.insert(out.end(), row.begin(), row.end());
    }
    // The oracle reduces in torch's order over the padded row; we reduce over
    // the gathered subset. Same values, different summation order.
    CHECK(RelL2(out, c->attn_out, c->seq * HQ * DH) < 2e-3);
  }
}

TEST_CASE("qsa-consumer: gather and mask agree value for value") {
  // The mask reference is the red-first path. It must agree exactly, because it
  // adds only exact zeros to the same accumulation in the same ascending order.
  const QsaConfig cfg = GoldenConfig();
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads,
                DH = g::kHeadDim;
  const Case& c = kOverBudget;
  const std::vector<float> q_all = Slice(c.attn_q, c.seq * HQ * DH);
  const std::vector<float> k_all = Slice(c.attn_k, c.seq * HKV * DH);
  const std::vector<float> v_all = Slice(c.attn_v, c.seq * HKV * DH);
  const std::vector<float> q = BuildQ(c, cfg);
  const std::vector<float> keys = BuildBlockKeys(c, cfg);
  for (int64_t qi = 0; qi < c.seq; ++qi) {
    CAPTURE(qi);
    const std::vector<float> q_row(q_all.begin() + qi * HQ * DH,
                                   q_all.begin() + (qi + 1) * HQ * DH);
    const std::vector<int32_t> sel = SelectFor(cfg, q, keys, qi);
    const std::vector<float> gathered =
        QsaGatherAttention(q_row, k_all, v_all, sel, qi + 1, HQ, HKV, DH, nullptr);
    const std::vector<float> masked =
        QsaMaskedAttention(q_row, k_all, v_all, sel, qi + 1, HQ, HKV, DH, nullptr);
    REQUIRE(gathered.size() == masked.size());
    for (size_t i = 0; i < gathered.size(); ++i) CHECK(gathered[i] == masked[i]);
  }
}

// ── The gate that a mask-only implementation CANNOT pass ─────────────────────

TEST_CASE("qsa-consumer: the GATHER touches only the selected rows") {
  // This is the wave's whole point, and it is not a correctness property. A mask
  // over a dense cache is CORRECT and would pass every case above. llama.cpp
  // #27739 names the mechanism by which it is also not faster.
  const QsaConfig cfg = GoldenConfig();
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads,
                DH = g::kHeadDim;
  const Case& c = kOverBudget;
  const std::vector<float> q_all = Slice(c.attn_q, c.seq * HQ * DH);
  const std::vector<float> k_all = Slice(c.attn_k, c.seq * HKV * DH);
  const std::vector<float> v_all = Slice(c.attn_v, c.seq * HKV * DH);
  const std::vector<float> q = BuildQ(c, cfg);
  const std::vector<float> keys = BuildBlockKeys(c, cfg);

  // The two softmax passes: the max, then the weights and the reduction. Each
  // reads every row it visits, so reading a row twice over — as a redundant pass
  // would — is visible here (W4 fresh review, mutation M31). A single-pass
  // online-softmax rewrite would legitimately halve this, and this is where that
  // shows up and gets re-derived on purpose.
  constexpr int64_t kReadsPerRowPerHead = 2;

  int64_t strictly_sparse_queries = 0;
  for (int64_t qi = 0; qi < c.seq; ++qi) {
    CAPTURE(qi);
    const int64_t kv_len = qi + 1;
    const std::vector<float> q_row(q_all.begin() + qi * HQ * DH,
                                   q_all.begin() + (qi + 1) * HQ * DH);
    const std::vector<int32_t> sel = SelectFor(cfg, q, keys, qi);
    // READS, not selected positions. Each consumer reads one key row per query
    // head in each of its two softmax passes, so an honest gather comes out at
    // `selected * HQ * 2` and any walk over the whole cache at `kv_len * HQ * 2`.
    // `SelectedCount(sel)` alone was the expectation once, matching a counter
    // that was assigned `sel.size()`; both sides were then the same quantity
    // computed the same way, and a gather body that dot-products every cached
    // row and masks the rest — QsaMaskedAttention's cost — passed this case (W4
    // fresh review, mutation M22c).
    const int64_t want = SelectedCount(sel) * HQ * kReadsPerRowPerHead;
    const int64_t dense = kv_len * HQ * kReadsPerRowPerHead;
    int64_t gather_visited = -1, mask_visited = -1;
    QsaGatherAttention(q_row, k_all, v_all, sel, kv_len, HQ, HKV, DH,
                       &gather_visited);
    QsaMaskedAttention(q_row, k_all, v_all, sel, kv_len, HQ, HKV, DH,
                       &mask_visited);
    CHECK(gather_visited == want);
    CHECK(mask_visited == dense);  // the mask cannot do better, by construction
    if (want < dense) {
      ++strictly_sparse_queries;
      CHECK(gather_visited < mask_visited);
    }
  }
  // Above `indexer_budget` the selection MUST discard blocks. A fixture that
  // never crossed the budget would leave every assertion above trivially true —
  // the exact failure the spec's `## Gates` G2 warns about.
  CHECK(strictly_sparse_queries > 0);
}
