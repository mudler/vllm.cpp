// The supplied emotion vector. See emovec.h for the upstream anchors.
#include "vllm/model_executor/models/emovec.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace emovec {

std::vector<float> Select(const std::vector<float>& style, int64_t style_dim,
                          const std::vector<EmotionBank>& banks,
                          const std::vector<float>& weights, int64_t out_dim,
                          std::vector<int64_t>* chosen_rows) {
  VT_CHECK(style_dim > 0 && out_dim > 0, "emovec: style_dim and out_dim must be positive");
  VT_CHECK(style.size() == static_cast<size_t>(style_dim),
           "emovec: style must be [style_dim]");
  VT_CHECK(!banks.empty(), "emovec: at least one emotion bank is required");
  VT_CHECK(weights.size() == banks.size(),
           "emovec: one weight per emotion bank");

  double style_norm = 0.0;
  for (const float v : style) {
    style_norm += static_cast<double>(v) * static_cast<double>(v);
  }
  style_norm = std::sqrt(style_norm);

  std::vector<float> out(static_cast<size_t>(out_dim), 0.0F);
  if (chosen_rows != nullptr) {
    chosen_rows->clear();
  }

  for (size_t e = 0; e < banks.size(); ++e) {
    const EmotionBank& b = banks[e];
    VT_CHECK(b.rows > 0, "emovec: an emotion bank is empty");
    VT_CHECK(b.speakers.size() == static_cast<size_t>(b.rows * style_dim),
             "emovec: a speaker matrix is not [rows, style_dim]");
    VT_CHECK(b.emotions.size() == static_cast<size_t>(b.rows * out_dim),
             "emovec: an emotion matrix is not [rows, out_dim]");

    // argmax over COSINE similarity, searched PER EMOTION against this bank.
    int64_t best = 0;
    double best_sim = 0.0;
    for (int64_t r = 0; r < b.rows; ++r) {
      double dot = 0.0;
      double row_norm = 0.0;
      for (int64_t d = 0; d < style_dim; ++d) {
        const double v = static_cast<double>(b.speakers[static_cast<size_t>(r * style_dim + d)]);
        dot += v * static_cast<double>(style[static_cast<size_t>(d)]);
        row_norm += v * v;
      }
      row_norm = std::sqrt(row_norm);
      const double denom = style_norm * row_norm;
      // torch's cosine_similarity clamps the denominator rather than dividing
      // by zero; a zero row therefore scores 0, not NaN.
      const double sim = denom > 0.0 ? dot / denom : 0.0;
      // Strictly greater, so a tie keeps the LOWEST index.
      if (r == 0 || sim > best_sim) {
        best_sim = sim;
        best = r;
      }
    }
    if (chosen_rows != nullptr) {
      chosen_rows->push_back(best);
    }

    const double w = static_cast<double>(weights[e]);
    for (int64_t d = 0; d < out_dim; ++d) {
      out[static_cast<size_t>(d)] = static_cast<float>(
          static_cast<double>(out[static_cast<size_t>(d)]) +
          w * static_cast<double>(b.emotions[static_cast<size_t>(best * out_dim + d)]));
    }
  }
  return out;
}

}  // namespace emovec
}  // namespace models
}  // namespace vllm
