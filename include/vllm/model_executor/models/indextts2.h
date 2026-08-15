// IndexTTS-2.5 — the speech family registration (#634).
//
// Upstream: vllm-project/vllm-omni registers TWO architectures for this model,
// `IndexTTS2TalkerForConditionalGeneration` (stage 0) and `IndexTTS2S2MelDecoder`
// (stage 1), at `vllm_omni/model_executor/models/registry.py`.
//
// WHAT EXISTS TODAY. The GPT-2 backbone the talker is built on (gpt2.h, W2) and
// the BigVGAN 1-D core the vocoder needs (vocoder1d.h, W1). What does NOT exist
// is the reference-audio conditioning path (w2v-bert-2.0, the MaskGCT semantic
// codec, CAMPPlus), the EnhancedCodec, and the S2Mel CFM/DiT decoder — W3-W5 in
// .agents/specs/indextts-2-5.md.
//
// So this registration DETECTS the checkpoint and REFUSES the load, naming the
// missing stages. That is deliberate: an unimplemented arm that is silently
// absent is a failure this project has already recorded, while one that refuses
// by name is visible debt a reader can act on.
#pragma once

#include "vllm/multimodal/speech_engine.h"

namespace vllm {
namespace models {

// Register the IndexTTS-2.5 family into `registry`. Explicit rather than
// self-registering at static-init time so a test can drive a local registry.
void RegisterIndexTts2SpeechFamily(multimodal::SpeechRegistry& registry);

}  // namespace models
}  // namespace vllm
