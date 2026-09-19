// GLiNER2 NER decoder — boundary marginal to entity extraction (MODEL-GLINER25).
//
// Ported from the GLiNER2 library (github.com/fastino-ai/GLiNER2):
//   inference/candidate_decoder.py   decode_candidate_set, format_candidate
//   inference/overlap.py             resolve_overlaps (greedy default)
//   models/boundary/heads.py         BoundaryMarginals
//
// Phase 4a: the marginal-based NER pipeline. Consumes the BoundaryMarginals
// produced by BoundaryQueryHeadForward and decodes them into thresholded,
// overlap-resolved, ranked entities with token-to-character offset mapping.
//
// The full pair scorer (SparseBoundaryPairScorer) is an owed correctness
// refinement. This decoder uses the marginal scores directly: for each
// half-open span [start, end) the score is
//
//   start_logit[q, start] + end_logit[q, end-1]
//     + (inside_prefix[q, end] - inside_prefix[q, start])
//
// which, because inside_prefix is a mean-centered cumsum, equals the sum of
// inside_logits over the span tokens. Sigmoid and threshold follow the
// upstream decode_candidate_set contract.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gliner2.h"

namespace vllm {
namespace gliner2 {

// A single extracted entity.
struct NerEntity {
  std::string label;         // entity type / query name
  std::string text;          // surface text from the original input
  int64_t char_start = 0;    // character start offset (inclusive)
  int64_t char_end = 0;      // character end offset (exclusive)
  int64_t token_start = 0;   // token start index (inclusive)
  int64_t token_end = 0;     // token end index (exclusive)
  float confidence = 0.0F;   // sigmoid probability of the span score
};

// NER decoding parameters.
struct NerParams {
  float threshold = 0.5F;             // minimum sigmoid probability to keep
  int64_t max_width = 12;              // maximum span width in tokens
  // Greedy overlap suppression: sort by descending confidence, then ascending
  // start and end; reject any candidate that overlaps an already-kept one.
  // Matches the finalize_spans default in candidate_decoder.py.
};

// Numerically stable sigmoid (matches upstream _sigmoid).
float Sigmoid(float x);

// Decode boundary marginals into named entities.
//
// marginals:       from BoundaryQueryHeadForward [Q, L+1] / [Q, L]
// labels:          entity type names, one per query (size = num_queries)
// seq_len:         number of text tokens (L)
// start_mappings:   char start offset for each token [L]
// end_mappings:     char end offset for each token [L]
// text:            original input text (for slicing surface text)
// params:          decoding parameters
//
// Returns entities sorted by (-confidence, token_start, token_end, label).
std::vector<NerEntity> DecodeNer(
    const BoundaryMarginals& marginals,
    const std::vector<std::string>& labels,
    int64_t seq_len,
    const std::vector<int64_t>& start_mappings,
    const std::vector<int64_t>& end_mappings,
    const std::string& text,
    const NerParams& params);

}  // namespace gliner2

// ── Phase 4: NER inference entry point ───────────────────────────────────
// Full NER pipeline: tokenize text + entity labels, run the DeBERTa v2 encoder,
// run the boundary head (encoder + query head), decode marginals into entities.
// Exposed so the C ABI (vllm_gliner_ner) and the server endpoint can reach the
// NER capability without going through the text-generation/embedding forward.
class LoadedModel;
namespace tok { class Tokenizer; }

struct Gliner2NerResult {
  std::vector<gliner2::NerEntity> entities;
};

// Run NER inference on `text` for the given entity `labels`.
// `model` must be a BoundaryExtractor-loaded model (checked via ModelAs).
// `tokenizer` tokenizes text and labels.
// `params` controls threshold, max_width, and overlap suppression.
Gliner2NerResult Gliner2NerInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    std::string_view text,
    const std::vector<std::string>& labels,
    const gliner2::NerParams& params);

}  // namespace vllm
