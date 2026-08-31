#include "vllm/model_executor/models/deepseek_v4_dspark.h"

#include <cmath>
#include <limits>
#include <string>

#include "vt/dtype.h"

namespace vllm::dspark {

std::vector<float> StreamMeanTap(const std::vector<float>& x, int64_t num_tokens,
                                 int64_t hc_mult, int64_t hidden) {
  VT_CHECK(hc_mult > 0 && hidden > 0 && num_tokens >= 0,
           "dspark tap: degenerate shape");
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * hc_mult * hidden,
           "dspark tap: expected [num_tokens, hc_mult, hidden] = " +
               std::to_string(num_tokens * hc_mult * hidden) + " values, got " +
               std::to_string(x.size()));
  std::vector<float> out(static_cast<size_t>(num_tokens) * hidden, 0.0f);
  const float inv = 1.0f / static_cast<float>(hc_mult);
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t d = 0; d < hidden; ++d) {
      // Accumulate in double: hc_mult is small, but the streams start as
      // BROADCAST COPIES of the embedding, so the summands are near-identical and
      // a float accumulator's rounding is systematic rather than cancelling.
      double acc = 0.0;
      for (int64_t s = 0; s < hc_mult; ++s)
        acc += static_cast<double>(x[static_cast<size_t>((t * hc_mult + s) * hidden + d)]);
      out[static_cast<size_t>(t * hidden + d)] = static_cast<float>(acc) * inv;
    }
  }
  return out;
}

std::vector<float> ProjectTaps(const std::vector<std::vector<float>>& taps,
                               const DeepseekV4MtpTensorView& main_proj,
                               const std::vector<float>& main_norm_weight, float eps,
                               int64_t num_tokens, int64_t hidden) {
  const int64_t n_taps = static_cast<int64_t>(taps.size());
  VT_CHECK(n_taps > 0, "dspark entry: no taps; main_proj reads n_taps * hidden");
  for (int64_t i = 0; i < n_taps; ++i) {
    VT_CHECK(static_cast<int64_t>(taps[static_cast<size_t>(i)].size()) ==
                 num_tokens * hidden,
             "dspark entry: tap " + std::to_string(i) +
                 " is not [num_tokens, hidden]; a tap is the STREAM MEAN, not the "
                 "stream stack (exllamav3 transformer.py:198-203)");
  }
  // The stored width IS the logical one here: `main_proj` is fp8-block, whose
  // shape is not nibble-packed. `DequantizeDeepseekV4MtpTensor` resolves that from
  // the view's own format, which is why this does not test the dtype itself.
  VT_CHECK(main_proj.out_dim == hidden && main_proj.in_dim == n_taps * hidden,
           "dspark entry: main_proj is [" + std::to_string(main_proj.out_dim) + ", " +
               std::to_string(main_proj.in_dim) + "], expected [" +
               std::to_string(hidden) + ", " + std::to_string(n_taps * hidden) +
               "]; `dspark_target_layer_ids` and this projection must agree on "
               "n_taps");
  VT_CHECK(static_cast<int64_t>(main_norm_weight.size()) == hidden,
           "dspark entry: main_norm is not [hidden]");

  const std::vector<float> w = DequantizeDeepseekV4MtpTensor(main_proj);
  const int64_t in = n_taps * hidden;

  std::vector<float> out(static_cast<size_t>(num_tokens) * hidden, 0.0f);
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t o = 0; o < hidden; ++o) {
      double acc = 0.0;
      // The concatenation is on the LAST axis, so tap `k` occupies input columns
      // [k*hidden, (k+1)*hidden) -- it is never interleaved per element.
      for (int64_t k = 0; k < n_taps; ++k) {
        const std::vector<float>& tk = taps[static_cast<size_t>(k)];
        const float* wrow = &w[static_cast<size_t>(o * in + k * hidden)];
        const float* trow = &tk[static_cast<size_t>(t * hidden)];
        for (int64_t d = 0; d < hidden; ++d)
          acc += static_cast<double>(wrow[d]) * trow[d];
      }
      out[static_cast<size_t>(t * hidden + o)] = static_cast<float>(acc);
    }
  }

  // `main_norm` is applied to the PROJECTION's output, per row.
  std::vector<float> normed(out.size(), 0.0f);
  for (int64_t t = 0; t < num_tokens; ++t) {
    double ss = 0.0;
    for (int64_t d = 0; d < hidden; ++d) {
      const double v = out[static_cast<size_t>(t * hidden + d)];
      ss += v * v;
    }
    const float r =
        1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(hidden)) + eps);
    for (int64_t d = 0; d < hidden; ++d)
      normed[static_cast<size_t>(t * hidden + d)] =
          out[static_cast<size_t>(t * hidden + d)] * r *
          main_norm_weight[static_cast<size_t>(d)];
  }
  return normed;
}

std::vector<int32_t> MarkovDraftLoop(const std::vector<float>& logits, int32_t seed_id,
                                     const std::vector<float>& markov_w1,
                                     const std::vector<float>& markov_w2, int64_t block,
                                     int64_t vocab, int64_t rank) {
  VT_CHECK(block > 0 && vocab > 0 && rank > 0, "dspark draft: degenerate shape");
  VT_CHECK(static_cast<int64_t>(logits.size()) == block * vocab,
           "dspark draft: logits must be [block, vocab]");
  VT_CHECK(static_cast<int64_t>(markov_w1.size()) == vocab * rank,
           "dspark draft: markov_w1 is an Embedding [vocab, rank]");
  VT_CHECK(static_cast<int64_t>(markov_w2.size()) == vocab * rank,
           "dspark draft: markov_w2 is a Linear [vocab, rank]");
  VT_CHECK(seed_id >= 0 && seed_id < vocab, "dspark draft: seed id out of range");

  std::vector<int32_t> out(static_cast<size_t>(block) + 1, 0);
  out[0] = seed_id;
  for (int64_t i = 0; i < block; ++i) {
    // The bias is conditioned on the PREVIOUSLY SAMPLED id, not on position i's
    // own argmax -- that is what makes the chain a bigram and the loop serial.
    const float* emb = &markov_w1[static_cast<size_t>(out[static_cast<size_t>(i)]) *
                                  static_cast<size_t>(rank)];
    const float* row = &logits[static_cast<size_t>(i * vocab)];
    int32_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int64_t v = 0; v < vocab; ++v) {
      double bias = 0.0;
      const float* w2row = &markov_w2[static_cast<size_t>(v * rank)];
      for (int64_t r = 0; r < rank; ++r) bias += static_cast<double>(w2row[r]) * emb[r];
      const float cand = row[v] + static_cast<float>(bias);
      // Strictly greater, so the LOWEST id wins a tie -- `torch.argmax`'s own
      // convention, and a draft that disagrees on ties diverges from the oracle
      // while still being a valid token.
      if (cand > best_v) {
        best_v = cand;
        best = static_cast<int32_t>(v);
      }
    }
    out[static_cast<size_t>(i) + 1] = best;
  }
  return out;
}

}  // namespace vllm::dspark
