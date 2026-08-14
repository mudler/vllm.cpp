// FactorizedVectorQuantize — EnhancedCodec's quantizer (W3/W4, #634).
//
// `infer_v2_5.py:293` calls `semantic_codec.quantize(...)`, and its DISCRETE
// output is the semantic code the talker consumes. Amphion's implementation with
// `use_l2_normlize=True`.
//
// Layout is [dim, frames] (channel-major), matching the torch [B, D, T] tensor.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace fvq {

struct Weights {
  // torch weight_norm stores (g, v); the effective weight is g * v / ||v||,
  // with the norm taken over every dimension except the first. The checkpoint
  // carries the LEGACY spelling weight_g / weight_v.
  std::vector<float> in_g, in_v, in_bias;    // [codebook_dim, input_dim, 1]
  std::vector<float> out_g, out_v, out_bias; // [input_dim, codebook_dim, 1]
  std::vector<float> codebook;               // [codebook_size, codebook_dim]
};

// The fold w = g * v / ||v|| is NOT declared here. It has one home,
// `vocoder1d::MaterializeWeightNorm`, and this file forked it: the copy that
// lived here was byte-equivalent but skipped that one's check that `g` carries
// one magnitude per dim-0 slice. Call the shared one.

struct QuantizeResult {
  std::vector<int64_t> indices;  // [frames]
  std::vector<float> z_q;        // [input_dim, frames], AFTER out_project
};

// FactorizedVectorQuantize::forward.
//
//   z_e      = in_project(z)                         (1x1 conv, weight-normed)
//   distance = |e|^2 - 2 e.c^T + |c|^2  over L2-NORMALIZED e and c
//   indices  = argmin(distance)
//   z_q      = out_project(codebook[indices])
//
// THE NORMALIZATION IS SEARCH-ONLY. Distances are computed on normalized
// vectors, but the entry returned by `decode_code` is the RAW codebook row.
// Returning the normalized row instead still produces plausible embeddings of
// the right shape, and the indices would be identical -- so only the VALUES
// catch it.
QuantizeResult Quantize(const std::vector<float>& z, int64_t frames, int64_t input_dim,
                        int64_t codebook_dim, int64_t codebook_size, const Weights& weights);

}  // namespace fvq
}  // namespace models
}  // namespace vllm
