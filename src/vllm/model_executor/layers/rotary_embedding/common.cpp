// Ported from:
//   vllm/model_executor/layers/rotary_embedding/common.py:32-76
// @ e24d1b24fe96.
#include "vllm/model_executor/layers/rotary_embedding/common.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vllm {

double yarn_find_correction_dim(int64_t num_rotations, int64_t dim,
                                double base,
                                int64_t max_position_embeddings) {
  if (num_rotations <= 0 || dim <= 0 || base <= 0.0 || base == 1.0 ||
      max_position_embeddings <= 0) {
    throw std::invalid_argument("invalid YaRN correction-dimension arguments");
  }
  return static_cast<double>(dim) *
         std::log(static_cast<double>(max_position_embeddings) /
                  (static_cast<double>(num_rotations) * 2.0 *
                   std::acos(-1.0))) /
         (2.0 * std::log(base));
}

std::pair<double, double> yarn_find_correction_range(
    int64_t low_rot, int64_t high_rot, int64_t dim, double base,
    int64_t max_position_embeddings, bool truncate) {
  double low = yarn_find_correction_dim(
      low_rot, dim, base, max_position_embeddings);
  double high = yarn_find_correction_dim(
      high_rot, dim, base, max_position_embeddings);
  if (truncate) {
    low = std::floor(low);
    high = std::ceil(high);
  }
  return {std::max(low, 0.0),
          std::min(high, static_cast<double>(dim - 1))};
}

std::vector<float> yarn_linear_ramp_mask(double low, double high,
                                         int64_t dim) {
  if (dim < 0) throw std::invalid_argument("negative YaRN ramp dimension");
  if (low == high) high += 0.001;
  std::vector<float> ramp(static_cast<size_t>(dim));
  const float low_f = static_cast<float>(low);
  const float width_f = static_cast<float>(high - low);
  for (int64_t i = 0; i < dim; ++i) {
    const float linear = (static_cast<float>(i) - low_f) / width_f;
    ramp[static_cast<size_t>(i)] = std::clamp(linear, 0.0F, 1.0F);
  }
  return ramp;
}

double yarn_get_mscale(double scale) {
  if (scale <= 1.0) return 1.0;
  return 0.1 * std::log(scale) + 1.0;
}

}  // namespace vllm
