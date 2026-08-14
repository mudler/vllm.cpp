// The SUPPLIED emotion vector (#634).
//
// Upstream `indextts/infer_v2_5.py:668-679` with `find_most_similar_cosine` at
// `:901-907`, index-tts @4f8792ff120cd3ea470dd511e997a17c86cddd10.
//
//   for each emotion e:
//       i_e = argmax_i cosine(style, spk_matrix[e][i])
//       row_e = emo_matrix[e][i_e]
//   out = sum_e weight[e] * row_e
//
// This is the CHEAP emotion path: when the caller states the emotion, upstream
// runs neither the `emo_conditioning_encoder` Conformer nor the
// `emo_perceiver_encoder` Perceiver. `spk_matrix` and `emo_matrix` are `feat1.pt`
// and `feat2.pt`, both shipped in the checkpoint.
//
// Three details, each of which yields a real vector when wrong:
//   - the argmax is over COSINE similarity, not distance. A nearest-neighbour by
//     Euclidean distance picks a different row whenever the norms differ.
//   - the search runs PER EMOTION against that emotion's own speaker matrix, so
//     the emotions can select different rows. One shared index is the obvious
//     simplification and is wrong.
//   - ties keep the LOWEST index, which is argmax's convention.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace emovec {

// One emotion's pair of matrices. `speakers` is [rows, style_dim] and
// `emotions` is [rows, out_dim]; the same row index addresses both.
struct EmotionBank {
  std::vector<float> speakers;
  std::vector<float> emotions;
  int64_t rows = 0;
};

// `style` is the CAMPPlus vector, [style_dim]. `weights` has one entry per bank.
// Returns [out_dim].
std::vector<float> Select(const std::vector<float>& style, int64_t style_dim,
                          const std::vector<EmotionBank>& banks,
                          const std::vector<float>& weights, int64_t out_dim,
                          std::vector<int64_t>* chosen_rows = nullptr);

}  // namespace emovec
}  // namespace models
}  // namespace vllm
