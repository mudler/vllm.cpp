// GLiNER2 NER decoder tests — MODEL-GLINER25 Phase 4a.
//
// Exercises the marginal-to-entity decoder with hand-crafted BoundaryMarginals
// that verify:
//   - basic span scoring and threshold filtering
//   - sigmoid numerics (positive and negative logits)
//   - greedy overlap suppression
//   - token-to-character offset mapping and surface text extraction
//   - multi-query decoding and global sort order
//
// The marginals are small (Q=2, L=6) so every assertion is by hand.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/gliner2.h"
#include "vllm/model_executor/models/gliner2_ner.h"

namespace {

using vllm::gliner2::BoundaryMarginals;
using vllm::gliner2::DecodeNer;
using vllm::gliner2::NerEntity;
using vllm::gliner2::NerParams;
using vllm::gliner2::Sigmoid;

// Build marginals for Q queries over L tokens.
// Layouts: start_logits [Q, L+1], end_logits [Q, L+1],
//          inside_logits [Q, L], inside_prefix [Q, L+1].
BoundaryMarginals MakeMarginals(int64_t Q, int64_t L) {
  const int64_t B = L + 1;
  BoundaryMarginals m;
  m.start_logits.assign(static_cast<size_t>(Q * B), 0.0F);
  m.end_logits.assign(static_cast<size_t>(Q * B), 0.0F);
  m.inside_logits.assign(static_cast<size_t>(Q * L), 0.0F);
  m.inside_prefix.assign(static_cast<size_t>(Q * B), 0.0F);
  return m;
}

// Helper to set a marginal value at [q, i] in a [Q, B] layout.
void SetVal(std::vector<float>& v, int64_t B, int64_t q, int64_t i,
            float val) {
  v[static_cast<size_t>(q * B + i)] = val;
}

}  // namespace

TEST_CASE("gliner2_ner: sigmoid is numerically stable") {
  CHECK(Sigmoid(0.0F) == doctest::Approx(0.5F).epsilon(0.001F));
  // Large positive: should be ~1.0, not inf
  CHECK(Sigmoid(100.0F) == doctest::Approx(1.0F).epsilon(0.001F));
  // Large negative: should be ~0.0, not nan
  CHECK(Sigmoid(-100.0F) == doctest::Approx(0.0F).epsilon(0.001F));
  // Symmetry: sigmoid(x) + sigmoid(-x) = 1
  CHECK((Sigmoid(2.5F) + Sigmoid(-2.5F)) == doctest::Approx(1.0F).epsilon(0.001F));
}

TEST_CASE("gliner2_ner: basic single-entity extraction") {
  // Q=1 ("person"), L=4 tokens.
  // Strong start at token 1, strong end at token 2, high inside at token 1.
  // The entity should be [1, 3) with text "John".
  const int64_t Q = 1, L = 4, B = L + 1;
  auto m = MakeMarginals(Q, L);

  // start_logits: token 1 is a start boundary (high logit).
  SetVal(m.start_logits, B, 0, 0, -5.0F);
  SetVal(m.start_logits, B, 0, 1, 5.0F);   // start at token 1
  SetVal(m.start_logits, B, 0, 2, -5.0F);
  SetVal(m.start_logits, B, 0, 3, -5.0F);
  SetVal(m.start_logits, B, 0, 4, -5.0F);

  // end_logits: token 2 is an end boundary (end=3, so end-1=2).
  SetVal(m.end_logits, B, 0, 0, -5.0F);
  SetVal(m.end_logits, B, 0, 1, -5.0F);
  SetVal(m.end_logits, B, 0, 2, 5.0F);   // end at token 2 (end=3)
  SetVal(m.end_logits, B, 0, 3, -5.0F);
  SetVal(m.end_logits, B, 0, 4, -5.0F);

  // inside_logits: tokens 1 and 2 are inside.
  m.inside_logits[0] = -1.0F;  // token 0
  m.inside_logits[1] = 3.0F;   // token 1
  m.inside_logits[2] = 3.0F;   // token 2
  m.inside_logits[3] = -1.0F;  // token 3

  // inside_prefix: cumsum of inside_logits, mean-centered.
  // raw cumsum: [0, -1, 2, 5, 4], mean = 10/5 = 2, centered: [-2, -3, 0, 3, 2]
  // But our implementation centers after cumsum, so:
  // raw: [0, -1, 2, 5, 4], mean=2, centered: [-2, -3, 0, 3, 2]
  m.inside_prefix[0] = -2.0F;
  m.inside_prefix[1] = -3.0F;
  m.inside_prefix[2] = 0.0F;
  m.inside_prefix[3] = 3.0F;
  m.inside_prefix[4] = 2.0F;

  // Token-to-char mappings for "Hi John !!" (L=4 tokens)
  // token 0: "Hi" (0,2), token 1: "John" (3,7), token 2: "!" (8,9), token 3: "!" (9,10)
  std::vector<int64_t> start_maps = {0, 3, 8, 9};
  std::vector<int64_t> end_maps = {2, 7, 9, 10};
  std::string text = "Hi John !!";

  NerParams params;
  params.threshold = 0.5F;
  params.max_width = 12;

  auto entities = DecodeNer(m, {"person"}, L, start_maps, end_maps, text, params);

  REQUIRE(entities.size() == 1);
  const auto& e = entities[0];
  CHECK(e.label == "person");
  CHECK(e.token_start == 1);
  CHECK(e.token_end == 3);
  CHECK(e.char_start == 3);
  CHECK(e.char_end == 9);
  CHECK(e.text == "John !");
  CHECK(e.confidence > 0.5F);
}

TEST_CASE("gliner2_ner: threshold filtering rejects weak spans") {
  // All logits negative: no entity should survive threshold 0.5.
  const int64_t Q = 1, L = 3, B = L + 1;
  auto m = MakeMarginals(Q, L);

  for (int64_t i = 0; i < B; ++i) {
    SetVal(m.start_logits, B, 0, i, -5.0F);
    SetVal(m.end_logits, B, 0, i, -5.0F);
  }
  for (int64_t i = 0; i < L; ++i) {
    m.inside_logits[static_cast<size_t>(i)] = -5.0F;
  }

  std::vector<int64_t> start_maps = {0, 1, 2};
  std::vector<int64_t> end_maps = {1, 2, 3};
  std::string text = "abc";

  NerParams params;
  params.threshold = 0.5F;

  auto entities = DecodeNer(m, {"label"}, L, start_maps, end_maps, text, params);
  CHECK(entities.empty());
}

TEST_CASE("gliner2_ner: overlap suppression keeps highest-confidence") {
  // Q=1, L=6.
  // Two overlapping spans: [1,4) with high confidence, [2,5) with lower.
  // The decoder should keep [1,4) and suppress [2,5).
  const int64_t Q = 1, L = 6, B = L + 1;
  auto m = MakeMarginals(Q, L);

  // Strong starts at tokens 1 and 2.
  for (int64_t i = 0; i < B; ++i) {
    SetVal(m.start_logits, B, 0, i, -5.0F);
    SetVal(m.end_logits, B, 0, i, -5.0F);
  }
  SetVal(m.start_logits, B, 0, 1, 6.0F);   // start at 1
  SetVal(m.start_logits, B, 0, 2, 4.0F);   // start at 2 (weaker)

  // Strong ends at tokens 3 and 4.
  SetVal(m.end_logits, B, 0, 3, 6.0F);   // end at 3 (end=4) — strong
  SetVal(m.end_logits, B, 0, 4, 4.0F);   // end at 4 (end=5) — weaker

  // inside_prefix: set so that [1,4) has higher score than [2,5).
  // inside_prefix[4] - inside_prefix[1] should be large (for [1,4))
  // inside_prefix[5] - inside_prefix[2] should be smaller (for [2,5))
  // Use simple values: prefix = [0, 0, 0, 0, 3, 3]
  // [1,4): 3-0=3, [2,5): 3-0=3 — same. Let's differentiate:
  // prefix = [0, 0, 0, 0, 5, 3]
  // [1,4): 5-0=5, [2,5): 3-0=3
  m.inside_prefix[0] = 0.0F;
  m.inside_prefix[1] = 0.0F;
  m.inside_prefix[2] = 0.0F;
  m.inside_prefix[3] = 0.0F;
  m.inside_prefix[4] = 5.0F;
  m.inside_prefix[5] = 3.0F;

  std::vector<int64_t> start_maps = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> end_maps = {1, 2, 3, 4, 5, 6};
  std::string text = "abcdef";

  NerParams params;
  params.threshold = 0.5F;
  params.max_width = 12;

  auto entities = DecodeNer(m, {"label"}, L, start_maps, end_maps, text, params);

  // [1,4) score = 6+6+5 = 17, [2,5) score = 4+4+3 = 11
  // [1,4) has higher confidence, should suppress [2,5) (they overlap)
  REQUIRE(entities.size() == 1);
  CHECK(entities[0].token_start == 1);
  CHECK(entities[0].token_end == 4);
}

TEST_CASE("gliner2_ner: non-overlapping spans both kept") {
  // Q=1, L=6.
  // Two non-overlapping spans: [0,2) and [3,5).
  const int64_t Q = 1, L = 6, B = L + 1;
  auto m = MakeMarginals(Q, L);

  for (int64_t i = 0; i < B; ++i) {
    SetVal(m.start_logits, B, 0, i, -5.0F);
    SetVal(m.end_logits, B, 0, i, -5.0F);
  }
  // Span [0,2): start at 0, end at 1 (end=2)
  SetVal(m.start_logits, B, 0, 0, 5.0F);
  SetVal(m.end_logits, B, 0, 1, 5.0F);
  // Span [3,5): start at 3, end at 4 (end=5)
  SetVal(m.start_logits, B, 0, 3, 5.0F);
  SetVal(m.end_logits, B, 0, 4, 5.0F);

  // inside_prefix: [0,2) and [3,5) have high inside scores
  for (auto& v : m.inside_prefix) v = 0.0F;
  m.inside_prefix[2] = 5.0F;  // [0,2): prefix[2]-prefix[0] = 5
  m.inside_prefix[5] = 5.0F;  // [3,5): prefix[5]-prefix[3] = 5

  std::vector<int64_t> start_maps = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> end_maps = {1, 2, 3, 4, 5, 6};
  std::string text = "abcdef";

  NerParams params;
  params.threshold = 0.5F;
  params.max_width = 12;

  auto entities = DecodeNer(m, {"label"}, L, start_maps, end_maps, text, params);
  REQUIRE(entities.size() == 2);
  // Sorted by (-prob, start, end). Both have same score (5+5+5=15),
  // so sorted by start: [0,2) first, [3,5) second.
  CHECK(entities[0].token_start == 0);
  CHECK(entities[0].token_end == 2);
  CHECK(entities[1].token_start == 3);
  CHECK(entities[1].token_end == 5);
}

TEST_CASE("gliner2_ner: multi-query with global sort") {
  // Q=2 ("person", "org"), L=4.
  // Query 0 ("person"): one entity [0,2) with score 15
  // Query 1 ("org"): one entity [2,4) with score 18
  // Global sort by -confidence: "org" entity first (higher score).
  const int64_t Q = 2, L = 4, B = L + 1;
  auto m = MakeMarginals(Q, L);

  for (int64_t q = 0; q < Q; ++q) {
    for (int64_t i = 0; i < B; ++i) {
      SetVal(m.start_logits, B, q, i, -5.0F);
      SetVal(m.end_logits, B, q, i, -5.0F);
    }
  }

  // Query 0: entity [0,2)
  SetVal(m.start_logits, B, 0, 0, 5.0F);
  SetVal(m.end_logits, B, 0, 1, 5.0F);
  m.inside_prefix[0 * B + 2] = 5.0F;

  // Query 1: entity [2,4)
  SetVal(m.start_logits, B, 1, 2, 6.0F);
  SetVal(m.end_logits, B, 1, 3, 6.0F);
  m.inside_prefix[1 * B + 4] = 6.0F;

  std::vector<int64_t> start_maps = {0, 1, 2, 3};
  std::vector<int64_t> end_maps = {1, 2, 3, 4};
  std::string text = "abcd";

  NerParams params;
  params.threshold = 0.5F;
  params.max_width = 12;

  auto entities = DecodeNer(m, {"person", "org"}, L, start_maps, end_maps, text, params);
  REQUIRE(entities.size() == 2);
  // Query 1 (org) has higher score (6+6+6=18 vs 5+5+5=15), so it's first.
  CHECK(entities[0].label == "org");
  CHECK(entities[0].token_start == 2);
  CHECK(entities[0].token_end == 4);
  CHECK(entities[1].label == "person");
  CHECK(entities[1].token_start == 0);
  CHECK(entities[1].token_end == 2);
}

TEST_CASE("gliner2_ner: max_width limits span enumeration") {
  // Q=1, L=6.
  // Strong start at 0, strong end at 5 (end=6), but max_width=2 means
  // only spans of width <= 2 are considered, so [0,6) is never checked.
  const int64_t Q = 1, L = 6, B = L + 1;
  auto m = MakeMarginals(Q, L);

  for (int64_t i = 0; i < B; ++i) {
    SetVal(m.start_logits, B, 0, i, 5.0F);
    SetVal(m.end_logits, B, 0, i, 5.0F);
  }
  for (auto& v : m.inside_prefix) v = 5.0F;

  std::vector<int64_t> start_maps = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> end_maps = {1, 2, 3, 4, 5, 6};
  std::string text = "abcdef";

  NerParams params;
  params.threshold = 0.0F;  // accept everything
  params.max_width = 2;     // only spans up to 2 tokens wide

  auto entities = DecodeNer(m, {"label"}, L, start_maps, end_maps, text, params);

  // Every span [i, j) with j-i <= 2 should be present.
  // After overlap suppression (greedy, all same confidence), the first
  // in sort order (highest prob, then smallest start, then smallest end)
  // wins and suppresses overlapping ones. With identical scores, the
  // smallest-start, smallest-end spans are kept first.
  // Span [0,1) is kept, [0,2) overlaps, [1,2) overlaps with [0,1),
  // [1,3) overlaps, etc. Actually [0,1) and [1,2) don't overlap (half-open).
  // Let's just verify max_width is respected: no span wider than 2.
  for (const auto& e : entities) {
    CHECK(e.token_end - e.token_start <= 2);
  }
  // At least one entity must survive.
  CHECK_FALSE(entities.empty());
}
