// The S2Mel DiT TAIL — everything after the transformer stack (#634).
//
// Upstream `indextts/s2mel/modules/diffusion_transformer.py:243-253`, index-tts
// @4f8792ff120cd3ea470dd511e997a17c86cddd10, under the shipped config
// (`long_skip_connection: true`, `final_layer_type: wavenet`):
//
//   x_res = skip_linear(cat([x_res, x], -1))          // the LONG skip
//   h     = conv1(x_res)                              // Linear D -> wavenet H
//   t2    = t_embedder2(t)                            // a SECOND embedder
//   h     = wavenet(h^T, mask, g=t2)^T + res_projection(x_res)
//   h     = final_layer(h, t1)
//   out   = conv2(h^T)                                // Conv1d H -> in_channels
//
// This composes `wavenet::Forward`, `cfm::TimestepFeatures` and
// `adaln::FinalLayer` rather than reimplementing any of them.
//
// One coupling is NOT obvious from the config and is asserted here: the DiT's
// hidden width must equal the wavenet width, because `final_layer` is built at
// the wavenet width but conditioned on `t1`, which the DiT embeds at its own.
// Both are 512 upstream, so the constraint never shows there.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/adaln.h"
#include "vllm/model_executor/models/wavenet.h"

namespace vllm {
namespace models {
namespace dit_tail {

struct Linear {
  std::vector<float> weight;  // [out, in], torch row-major
  std::vector<float> bias;    // [out]
};

struct Weights {
  Linear skip_linear;      // [hidden, hidden + in_channels]
  Linear conv1;            // [wn_hidden, hidden]
  Linear res_projection;   // [wn_hidden, hidden]
  Linear conv2;            // Conv1d kernel 1: [in_channels, wn_hidden]
  Linear t_embedder2_mlp0;  // [wn_hidden, freq_size]
  Linear t_embedder2_mlp2;  // [wn_hidden, wn_hidden]
  wavenet::Weights wn;
  adaln::FinalLayerWeights final_layer;
};

struct Config {
  int64_t hidden = 0;
  int64_t wn_hidden = 0;  // must equal `hidden`; see the header comment
  int64_t in_channels = 0;
  int64_t frames = 0;
  int64_t freq_size = 256;  // TimestepEmbedder's frequency_embedding_size
  wavenet::Config wn;
};

// x_res is [frames, hidden] (the transformer output, frame-major).
// x is [frames, in_channels] (the noisy input at this step, frame-major).
// t1 is [hidden] (the DiT's own timestep embedding), t is the raw timestep.
// mask is [frames] or empty.
// Returns [in_channels, frames], CHANNEL-major, as upstream's conv2 does.
std::vector<float> Forward(const Config& cfg, const Weights& w,
                           const std::vector<float>& x_res, const std::vector<float>& x,
                           float t, const std::vector<float>& t1,
                           const std::vector<float>& mask);

}  // namespace dit_tail
}  // namespace models
}  // namespace vllm
