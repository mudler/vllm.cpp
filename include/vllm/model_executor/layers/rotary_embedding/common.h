// Ported from:
//   vllm/model_executor/layers/rotary_embedding/common.py:32-76
// @ e24d1b24fe96.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace vllm {

double yarn_find_correction_dim(
    int64_t num_rotations, int64_t dim, double base = 10000.0,
    int64_t max_position_embeddings = 2048);

std::pair<double, double> yarn_find_correction_range(
    int64_t low_rot, int64_t high_rot, int64_t dim, double base = 10000.0,
    int64_t max_position_embeddings = 2048, bool truncate = true);

std::vector<float> yarn_linear_ramp_mask(double low, double high,
                                         int64_t dim);

double yarn_get_mscale(double scale = 1.0);

}  // namespace vllm
