#include "vllm/model_executor/models/indextts2_conditioning.h"

#include <stdexcept>
#include <string>

namespace vllm::indextts2 {

void MeanCentreColumns(std::vector<float>& mel, int64_t bins, int64_t frames) {
  if (bins <= 0 || frames <= 0) {
    throw std::runtime_error(
        "indextts2: mean-centring needs a positive frame and bin count; got " +
        std::to_string(frames) + " frames of " + std::to_string(bins) + " bins");
  }
  if (static_cast<size_t>(bins * frames) != mel.size()) {
    throw std::runtime_error(
        "indextts2: the log-mel buffer holds " + std::to_string(mel.size()) +
        " entries, which is not " + std::to_string(frames) + " frames of " +
        std::to_string(bins) + " bins");
  }
  for (int64_t c = 0; c < bins; ++c) {
    double m = 0.0;
    for (int64_t f = 0; f < frames; ++f) {
      m += static_cast<double>(mel[static_cast<size_t>(f * bins + c)]);
    }
    m /= static_cast<double>(frames);
    for (int64_t f = 0; f < frames; ++f) {
      mel[static_cast<size_t>(f * bins + c)] -= static_cast<float>(m);
    }
  }
}

std::vector<float> ProjectSpeaker(const std::vector<float>& style,
                                  const std::vector<float>& w,
                                  const std::vector<float>& b, int64_t out_dim) {
  const int64_t style_dim = static_cast<int64_t>(style.size());
  if (out_dim <= 0 || style_dim <= 0) {
    throw std::runtime_error(
        "indextts2: the speaker projection needs a non-empty style and a "
        "positive output width; got " + std::to_string(style_dim) + " -> " +
        std::to_string(out_dim));
  }
  if (static_cast<size_t>(out_dim * style_dim) != w.size()) {
    throw std::runtime_error(
        "indextts2: spk_emb_proj holds " + std::to_string(w.size()) +
        " weights, which is not " + std::to_string(out_dim) + " x " +
        std::to_string(style_dim));
  }
  if (!b.empty() && static_cast<int64_t>(b.size()) != out_dim) {
    throw std::runtime_error(
        "indextts2: spk_emb_proj's bias is " + std::to_string(b.size()) +
        " wide but the projection emits " + std::to_string(out_dim));
  }
  std::vector<float> out(static_cast<size_t>(out_dim), 0.0F);
  for (int64_t o = 0; o < out_dim; ++o) {
    double acc = b.empty() ? 0.0 : static_cast<double>(b[static_cast<size_t>(o)]);
    for (int64_t i = 0; i < style_dim; ++i) {
      acc += static_cast<double>(style[static_cast<size_t>(i)]) *
             static_cast<double>(w[static_cast<size_t>(o * style_dim + i)]);
    }
    out[static_cast<size_t>(o)] = static_cast<float>(acc);
  }
  return out;
}

}  // namespace vllm::indextts2
