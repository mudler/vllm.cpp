// GLiNER2 NER decoder implementation (MODEL-GLINER25).
//
// See gliner2_ner.h for the design. The algorithm:
//
// 1. For each query q and each half-open span [start, end):
//    - Compute span_score = start_logit[q,start] + end_logit[q,end-1]
//      + (inside_prefix[q,end] - inside_prefix[q,start])
//    - span_prob = sigmoid(span_score)
//    - Keep if span_prob >= threshold
// 2. Per-query greedy overlap suppression (sort by -prob, start, end;
//    reject overlapping).
// 3. Global sort by (-confidence, token_start, token_end, label).
// 4. Map token offsets to character offsets and extract surface text.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gliner2_ner.h"

namespace vllm {
namespace gliner2 {

float Sigmoid(float x) {
  if (x >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-x));
  }
  float z = std::exp(x);
  return z / (1.0F + z);
}

namespace {

// Internal candidate before formatting.
struct Candidate {
  int64_t query_id;
  int64_t start;      // token start (inclusive)
  int64_t end;        // token end (exclusive)
  float logit;        // raw span score
  float prob;         // sigmoid(logit)
};

// Greedy overlap suppression: sort by descending prob, then ascending
// start/end; reject any candidate that overlaps an already-kept one.
// Two half-open spans [s1, e1) and [s2, e2) overlap iff
// s1 < e2 && s2 < e1.
std::vector<Candidate> SuppressOverlaps(std::vector<Candidate>& cands) {
  std::sort(cands.begin(), cands.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.prob != b.prob) return a.prob > b.prob;
              if (a.start != b.start) return a.start < b.start;
              return a.end < b.end;
            });
  std::vector<Candidate> kept;
  kept.reserve(cands.size());
  for (const auto& c : cands) {
    bool overlaps = false;
    for (const auto& k : kept) {
      if (c.start < k.end && k.start < c.end) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) {
      kept.push_back(c);
    }
  }
  return kept;
}

}  // namespace

std::vector<NerEntity> DecodeNer(
    const BoundaryMarginals& marginals,
    const std::vector<std::string>& labels,
    int64_t seq_len,
    const std::vector<int64_t>& start_mappings,
    const std::vector<int64_t>& end_mappings,
    const std::string& text,
    const NerParams& params) {
  const int64_t B = seq_len + 1;  // boundary positions: 0..L
  const int64_t num_queries = static_cast<int64_t>(labels.size());

  // Per-query candidate lists.
  std::vector<std::vector<Candidate>> per_query(num_queries);

  for (int64_t q = 0; q < num_queries; ++q) {
    const float* start_logits = &marginals.start_logits[static_cast<size_t>(q * B)];
    const float* end_logits = &marginals.end_logits[static_cast<size_t>(q * B)];
    const float* inside_prefix = &marginals.inside_prefix[static_cast<size_t>(q * B)];

    for (int64_t start = 0; start < seq_len; ++start) {
      // Quick reject: start probability must clear threshold.
      if (Sigmoid(start_logits[start]) < params.threshold) continue;

      const int64_t max_end = std::min(start + params.max_width, seq_len);
      for (int64_t end = start + 1; end <= max_end; ++end) {
        // end-1 is the last token in the span; it must clear threshold.
        if (Sigmoid(end_logits[end - 1]) < params.threshold) continue;

        float span_score = start_logits[start] + end_logits[end - 1] +
                            (inside_prefix[end] - inside_prefix[start]);
        float prob = Sigmoid(span_score);
        if (prob < params.threshold) continue;

        per_query[q].push_back(
            {q, start, end, span_score, prob});
      }
    }

    per_query[q] = SuppressOverlaps(per_query[q]);
  }

  // Flatten, sort globally by (-prob, start, end, label).
  std::vector<Candidate> all;
  for (auto& pq : per_query) {
    for (auto& c : pq) {
      all.push_back(c);
    }
  }
  std::sort(all.begin(), all.end(), [&](const Candidate& a, const Candidate& b) {
    if (a.prob != b.prob) return a.prob > b.prob;
    if (a.start != b.start) return a.start < b.start;
    if (a.end != b.end) return a.end < b.end;
    return labels[static_cast<size_t>(a.query_id)] <
           labels[static_cast<size_t>(b.query_id)];
  });

  // Format into NerEntity with character offsets.
  std::vector<NerEntity> entities;
  entities.reserve(all.size());
  for (const auto& c : all) {
    NerEntity e;
    e.label = labels[static_cast<size_t>(c.query_id)];
    e.token_start = c.start;
    e.token_end = c.end;
    e.confidence = c.prob;
    // Half-open [start, end) -> char offsets:
    //   char_start = start_mappings[start]
    //   char_end   = end_mappings[end - 1]
    if (c.start >= 0 && c.start < seq_len &&
        c.end - 1 >= 0 && c.end - 1 < seq_len) {
      e.char_start = start_mappings[static_cast<size_t>(c.start)];
      e.char_end = end_mappings[static_cast<size_t>(c.end - 1)];
      if (e.char_start >= 0 && e.char_end <= static_cast<int64_t>(text.size()) &&
          e.char_start <= e.char_end) {
        e.text = text.substr(static_cast<size_t>(e.char_start),
                             static_cast<size_t>(e.char_end - e.char_start));
      }
    }
    entities.push_back(std::move(e));
  }

  return entities;
}

}  // namespace gliner2
}  // namespace vllm
