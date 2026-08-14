// Bind the converted S2Mel checkpoint to the ported DiT tail (#634).
//
// Reads `s2mel.safetensors`, produced offline by
// `scripts/convert-indextts2-checkpoint.py` from upstream's `s2mel.pth`. The
// engine never reads pickle; see that script's header for why.
//
// Names are the manifest's names: the converter joins nested state dicts with
// '.', so `net.cfm.estimator.conv1.weight` here is the same string
// `tests/vllm/models/indextts2_pth_manifest.json` records.
//
// DIMENSIONS ARE RESOLVED FROM THE TENSORS, not from a config. The shipped
// checkpoint says hidden 512, in_channels 80, wavenet 512 wide over 8 layers of
// kernel 5, and `t_embedder2.mlp.0.weight` is [512, 256], so the sinusoidal
// feature width is 256. A config that disagreed with the weights would be the
// config that is wrong.
//
// The shipped S2Mel tower is F32 throughout, which is recorded rather than
// assumed: this is one of the rare places where f32 is what upstream ACTUALLY
// stores, not a widening we introduced.
#pragma once

#include <string>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dit_tail.h"

namespace vllm {
namespace models {
namespace indextts2 {

struct S2MelTail {
  dit_tail::Config config;
  dit_tail::Weights weights;
};

// Throws std::runtime_error naming the missing or misshapen tensor. A checkpoint
// that is merely INCOMPLETE must not load as if it were whole.
S2MelTail LoadS2MelTail(const SafetensorsFile& file);

// Convenience: open the file and load. `path` is the converted safetensors.
S2MelTail LoadS2MelTail(const std::string& path);

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
