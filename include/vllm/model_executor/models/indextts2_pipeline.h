// IndexTTS-2.5 pipeline composition (#634).
//
// Every stage's numerics are ported and gated individually. This wires them into
// ONE path in upstream's order (infer_v2_5.py:569-660):
//
//   1. w2v-bert-2.0        reference clip -> semantic features
//   2. EnhancedCodec       features       -> semantic codes + quantized feat
//   3. CAMPPlus            feat           -> 192-d style vector
//   4. length regulator    codes          -> prompt condition at mel rate
//   5. GPT-2 talker        text + style   -> mel codes
//   6. S2Mel + BigVGAN     mel codes      -> 22.05 kHz waveform
//
// WHAT THIS IS AND IS NOT. This is a STRUCTURAL composition: it proves the
// stages connect, that each one's output shape is the next one's input shape,
// and that a change at the front propagates to the back. It runs at reduced
// dimensions on synthetic weights and is NOT a quality result, NOT a parity
// result, and NOT a render. The same distinction the MiniMax-H3 lane drew when
// it composed t2va before real weights existed.
//
// A real render additionally needs the checkpoint loader and the vLLM-Omni
// oracle (#633), neither of which exists yet.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {
namespace models {
namespace indextts2 {

// Reduced-dimension shape contract. The real model's values come from
// `config.yaml`; these are the knobs the composition needs to agree on.
struct PipelineDims {
  int64_t ref_frames = 0;        // reference-clip frames into w2v-bert
  int64_t semantic_dim = 0;      // w2v-bert hidden
  int64_t codec_dim = 0;         // EnhancedCodec input width
  int64_t codebook_dim = 0;
  int64_t codebook_size = 0;
  int64_t style_feat_dim = 0;    // CAMPPlus feat_dim
  int64_t style_dim = 0;         // CAMPPlus embedding (192 in the real model)
  int64_t mel_frames = 0;        // target mel length
  int64_t mel_channels = 0;
  int64_t talker_dim = 0;
  int64_t talker_vocab = 0;
};

struct PipelineResult {
  std::vector<int64_t> semantic_codes;  // stage 2
  std::vector<float> style;             // stage 3, [style_dim]
  std::vector<float> quantized;         // stage 2 output, [codec_dim, ref_frames]
  std::vector<float> prompt_condition;  // stage 4, [codec_dim, mel_frames]
  std::vector<float> mel;               // stage 5+6 input, [mel_channels, mel_frames]
  std::vector<float> waveform;          // stage 6 output
  int64_t sample_rate = 22050;
};

// Run the composed path over a reference clip and a text token sequence.
//
// Throws std::runtime_error naming the stage when a shape does not line up --
// which is the failure this seam exists to make loud, since a silent reshape
// between stages is how a pipeline produces audio from the wrong tensor.
PipelineResult RunReduced(const PipelineDims& dims, const std::vector<float>& reference_clip,
                          const std::vector<int64_t>& text_tokens, uint64_t seed);

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
