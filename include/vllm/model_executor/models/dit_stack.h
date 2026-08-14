// The S2Mel DiT transformer STACK: what sits between front end and tail (#634).
//
// Upstream `indextts/s2mel/modules/gpt_fast/model.py:161-191`
// (Transformer.forward), index-tts @4f8792ff120cd3ea470dd511e997a17c86cddd10.
// N blocks, the U-Net skip routing across them, and a final AdaptiveLayerNorm.
//
// This is composition only: the block is `dit::Block`, the routing is
// `dit_skip::Plan`, and the per-layer skip merge is `dit_skip::ApplySkip`.
// Nothing here reimplements any of them.
//
// The rotary table is an INPUT. Upstream precomputes `freqs_cis` once for the
// whole model and indexes it by position, so passing it in keeps this a gate on
// composition rather than on a second copy of that computation.
//
// Upstream builds a `skip_in_linear` on EVERY layer when uvit_skip_connection is
// set, even the layers that never receive one, so the checkpoint carries
// unused ones. They are loaded and left alone rather than treated as an error.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/dit.h"

namespace vllm {
namespace models {
namespace dit_stack {

struct LayerWeights {
  dit::BlockWeights block;
  // Present on every layer upstream; consulted only on receiving layers.
  std::vector<float> skip_in_w;  // [dim, 2 * dim]
  std::vector<float> skip_in_b;  // [dim]
};

struct Weights {
  std::vector<LayerWeights> layers;
  // transformer.norm, an AdaptiveLayerNorm like the per-block ones.
  std::vector<float> norm_proj_w, norm_proj_b, norm_w;
};

struct Config {
  int64_t dim = 0;
  int64_t heads = 0;
  int64_t head_dim = 0;
  int64_t intermediate = 0;
  int64_t frames = 0;
  double eps = 1e-5;
};

// x is [frames, dim]; cond is [dim] (one conditioning vector, as upstream passes
// t1 unsqueezed); freqs is the rotary table for these positions.
// Returns [frames, dim].
std::vector<float> Forward(const Config& cfg, const Weights& w,
                           const std::vector<float>& x, const std::vector<float>& cond,
                           const std::vector<float>& freqs);

}  // namespace dit_stack
}  // namespace models
}  // namespace vllm
