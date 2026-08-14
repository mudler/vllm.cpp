// EnhancedCodec.quantize — reference features become SEMANTIC CODES (#634).
//
// Upstream `indextts/codec/models.py:179-199`, index-tts
// @4f8792ff120cd3ea470dd511e997a17c86cddd10:
//
//   if downsample_scale > 1:
//       x = gelu(down(x^T))^T        // stride-2 Conv1d, kernel 3, padding 1
//   x = encoder(x^T)^T               // VocosBackbone then Linear(vocos_dim, hidden)
//   indices, quantized = quantizer(x)
//
// This is the second half of the reference-audio path: w2v-bert features in, the
// discrete codes the talker consumes out. The backbone is `vocos::Backbone` and
// the quantizer is `fvq::Quantize`, both already ported; what this adds is the
// stride-2 downsample, the GELU between them, and the projection.
//
// THE INDICES ARE DISCRETE. A code that is off by one is a different utterance,
// so they are gated EXACTLY, never within a tolerance.
//
// `downsample_scale` halves the frame count, so the codes are at HALF the rate
// of the features. A port that skipped the downsample would emit twice as many
// codes and the talker would still consume them.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/fvq.h"
#include "vllm/model_executor/models/vocos.h"

namespace vllm {
namespace models {
namespace codec_encoder {

struct Weights {
  // down: Conv1d(hidden, hidden, kernel 3, stride 2, padding 1).
  std::vector<float> down_w, down_b;
  vocos::BackboneWeights backbone;
  // The Linear that follows the backbone: [hidden, vocos_dim].
  std::vector<float> proj_w, proj_b;
  fvq::Weights quantizer;
};

struct Config {
  int64_t hidden = 0;
  int64_t vocos_dim = 0;
  int64_t vocos_intermediate = 0;
  int64_t codebook_size = 0;
  int64_t codebook_dim = 0;
  bool downsample = true;  // downsample_scale > 1
  double eps = 1e-6;
};

struct Result {
  std::vector<int64_t> indices;  // [out_frames]
  std::vector<float> quantized;  // [hidden, out_frames], channel-major
  // The latent the quantizer saw, [hidden, out_frames]. Exposed because it is
  // what THIS code computes -- the quantizer is `fvq`, gated separately -- and
  // because at reduced dims with untrained weights every frame collapses onto
  // one codebook entry, so the indices alone cannot tell a correct encoder from
  // a broken one.
  std::vector<float> latent;
  // After down -> gelu only, [hidden, out_frames]. Exposed for localisation.
  std::vector<float> after_down;
  int64_t out_frames = 0;
};

// x is [frames, hidden] FRAME-major, as upstream passes w2v-bert features.
Result Encode(const Config& cfg, const Weights& w, const std::vector<float>& x,
              int64_t frames);

}  // namespace codec_encoder
}  // namespace models
}  // namespace vllm
