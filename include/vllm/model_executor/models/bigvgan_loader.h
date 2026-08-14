// Bind the converted BigVGAN checkpoint to the ported generator (#634).
//
// `nvidia/bigvgan_v2_22khz_80band_256x`, which IndexTTS-2.5 fetches separately
// (`model_download.py:33`) rather than shipping. Converted to safetensors
// offline like the rest, so the engine reads no pickle.
//
// The checkpoint's own config confirms every choice `bigvgan.h` records:
// upsample_rates [4,4,2,2,2,2] whose product is 256 -- exactly `kHopLength` --
// resblock "1", snakebeta with logscale, and use_tanh_at_final AND
// use_bias_at_final both FALSE.
//
// Weights ship WEIGHT-NORMED (`weight_g` / `weight_v`); upstream calls
// `remove_weight_norm()` before inference, so the fold happens here at load
// through `vocoder1d::MaterializeWeightNorm` and never per forward.
#pragma once

#include <string>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/bigvgan.h"

namespace vllm {
namespace models {
namespace bigvgan {

struct Loaded {
  Config config;
  Weights weights;
};

// Throws std::runtime_error naming the missing or misshapen tensor.
Loaded Load(const SafetensorsFile& file);
Loaded Load(const std::string& path);

}  // namespace bigvgan
}  // namespace models
}  // namespace vllm
