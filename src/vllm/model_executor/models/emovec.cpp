// The supplied emotion vector. See emovec.h for the upstream anchors.
#include "vllm/model_executor/models/emovec.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include <cstring>
#include <stdexcept>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
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

namespace {

std::vector<float> ReadF32(const SafetensorsFile& file, const std::string& name,
                           std::vector<int64_t>* shape) {
  const StTensor* t = nullptr;
  try {
    t = &file.Get(name);
  } catch (const std::exception&) {
    throw std::runtime_error("IndexTTS-2.5 emovec: missing tensor '" + name + "'");
  }
  if (t->dtype != "F32") {
    throw std::runtime_error("IndexTTS-2.5 emovec: tensor '" + name + "' is " + t->dtype +
                             ", expected F32");
  }
  *shape = t->shape;
  std::vector<float> out(t->nbytes / sizeof(float));
  std::memcpy(out.data(), t->data, t->nbytes);
  return out;
}

}  // namespace

std::vector<EmotionBank> LoadBanks(const std::string& aux_path,
                                   const std::vector<int64_t>& emo_num,
                                   int64_t* style_dim, int64_t* out_dim) {
  VT_CHECK(!emo_num.empty(), "emovec: emo_num must name at least one emotion");
  VT_CHECK(style_dim != nullptr && out_dim != nullptr,
           "emovec: style_dim and out_dim out-parameters are required");
  const SafetensorsFile file = SafetensorsFile::Open(aux_path);

  std::vector<int64_t> spk_shape;
  std::vector<int64_t> emo_shape;
  const std::vector<float> spk = ReadF32(file, "feat1", &spk_shape);
  const std::vector<float> emo = ReadF32(file, "feat2", &emo_shape);
  VT_CHECK(spk_shape.size() == 2 && emo_shape.size() == 2,
           "emovec: feat1 and feat2 must both be 2-D");
  VT_CHECK(spk_shape[0] == emo_shape[0],
           "emovec: feat1 and feat2 must have the same number of rows");

  const int64_t rows = spk_shape[0];
  *style_dim = spk_shape[1];
  *out_dim = emo_shape[1];

  int64_t total = 0;
  for (const int64_t n : emo_num) {
    VT_CHECK(n > 0, "emovec: every emotion must own at least one row");
    total += n;
  }
  VT_CHECK(total == rows,
           "emovec: emo_num sums to " + std::to_string(total) + " but feat1 has " +
               std::to_string(rows) +
               " rows; a mismatch reassigns rows to the wrong emotions silently");

  std::vector<EmotionBank> banks;
  int64_t offset = 0;
  for (const int64_t n : emo_num) {
    EmotionBank b;
    b.rows = n;
    b.speakers.assign(spk.begin() + static_cast<ptrdiff_t>(offset * *style_dim),
                      spk.begin() + static_cast<ptrdiff_t>((offset + n) * *style_dim));
    b.emotions.assign(emo.begin() + static_cast<ptrdiff_t>(offset * *out_dim),
                      emo.begin() + static_cast<ptrdiff_t>((offset + n) * *out_dim));
    banks.push_back(std::move(b));
    offset += n;
  }
  return banks;
}

}  // namespace emovec
}  // namespace models
}  // namespace vllm
