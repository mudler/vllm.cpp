// IndexTTS-2.5 pipeline composition. See indextts2_pipeline.h for what this
// does and does not claim.
#include "vllm/model_executor/models/indextts2_pipeline.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/campplus.h"
#include "vllm/model_executor/models/fvq.h"
#include "vllm/model_executor/models/lenreg.h"
#include "vllm/model_executor/models/talker.h"
#include "vllm/model_executor/models/vocoder1d.h"
#include "vllm/model_executor/models/w2vbert.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace indextts2 {
namespace {

// Deterministic synthetic weights: this composition runs on generated tensors,
// never on a checkpoint. Named so a stage's inputs are reproducible.
std::vector<float> Synth(const std::string& name, int64_t count, double scale, uint64_t seed) {
  uint64_t s = 0xCBF29CE484222325ULL ^ seed;
  for (const char c : name) {
    s ^= static_cast<unsigned char>(c);
    s *= 0x100000001B3ULL;
  }
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    uint64_t x = s + static_cast<uint64_t>(i);
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    out[static_cast<size_t>(i)] =
        static_cast<float>(((static_cast<double>(z >> 11) * 0x1.0p-53) * 2.0 - 1.0) * scale);
  }
  return out;
}

void Expect(bool ok, const std::string& stage, const std::string& what) {
  if (!ok) throw std::runtime_error("indextts2 pipeline: " + stage + ": " + what);
}

}  // namespace

PipelineResult RunReduced(const PipelineDims& d, const std::vector<float>& reference_clip,
                          const std::vector<int64_t>& text_tokens, uint64_t seed) {
  Expect(d.ref_frames > 0 && d.semantic_dim > 0 && d.mel_frames > 0, "dims",
         "reference frames, semantic dim and mel frames must be positive");
  Expect(reference_clip.size() == static_cast<size_t>(d.ref_frames * d.semantic_dim),
         "stage 1 (w2v-bert)", "reference clip does not match [ref_frames, semantic_dim]");

  PipelineResult r;

  // ── stage 1: semantic features. The reference clip stands in for the
  // Conformer's output; the encoder itself is gated in test_w2vbert.
  std::vector<float> features = reference_clip;

  // ── stage 2: EnhancedCodec quantization -> semantic codes.
  // The codec works [dim, frames]; the features arrive [frames, dim].
  Expect(d.codec_dim == d.semantic_dim, "stage 2 (EnhancedCodec)",
         "codec input width must equal the semantic dim");
  std::vector<float> codec_in(features.size());
  for (int64_t t = 0; t < d.ref_frames; ++t) {
    for (int64_t c = 0; c < d.codec_dim; ++c) {
      codec_in[static_cast<size_t>(c * d.ref_frames + t)] =
          features[static_cast<size_t>(t * d.semantic_dim + c)];
    }
  }
  fvq::Weights qw;
  qw.in_g = Synth("q.in_g", d.codebook_dim, 0.5, seed);
  qw.in_v = Synth("q.in_v", d.codebook_dim * d.codec_dim, 0.3, seed);
  qw.in_bias = Synth("q.in_b", d.codebook_dim, 0.2, seed);
  qw.out_g = Synth("q.out_g", d.codec_dim, 0.5, seed);
  qw.out_v = Synth("q.out_v", d.codec_dim * d.codebook_dim, 0.3, seed);
  qw.out_bias = Synth("q.out_b", d.codec_dim, 0.2, seed);
  qw.codebook = Synth("q.cb", d.codebook_size * d.codebook_dim, 0.6, seed);
  const fvq::QuantizeResult q =
      fvq::Quantize(codec_in, d.ref_frames, d.codec_dim, d.codebook_dim, d.codebook_size, qw);
  Expect(static_cast<int64_t>(q.indices.size()) == d.ref_frames, "stage 2 (EnhancedCodec)",
         "one semantic code per reference frame");
  r.semantic_codes = q.indices;
  r.quantized = q.z_q;

  // ── stage 3: CAMPPlus style vector from the quantized feature.
  // CAMPPlus consumes (T, F); the quantized output is [dim, frames].
  Expect(d.style_feat_dim > 0 && d.style_dim > 0, "stage 3 (CAMPPlus)",
         "style dims must be positive");
  // CHANNEL-MAJOR [feat, frames] -- StatsPool's layout. Building this
  // frame-major silently transposes the view, so the pooled "channels" mix
  // features and frames; the statistics still look plausible and are wrong.
  // Caught by the permutation case below, which requires a frame permutation to
  // leave the style untouched.
  std::vector<float> style_in(static_cast<size_t>(d.style_feat_dim * d.ref_frames));
  for (int64_t f = 0; f < d.style_feat_dim; ++f) {
    // Fold the codec width onto the style feature width deterministically; the
    // real model feeds CAMPPlus its own 80-bin fbank.
    const int64_t src = f % d.codec_dim;
    for (int64_t t = 0; t < d.ref_frames; ++t) {
      style_in[static_cast<size_t>(f * d.ref_frames + t)] =
          q.z_q[static_cast<size_t>(src * d.ref_frames + t)];
    }
  }
  // StatsPool alone stands in for the full encoder here; CAMPPlus::Forward is
  // gated whole in test_campplus, and re-running its 52 layers would make this
  // seam a duplicate of that gate rather than a composition check.
  const std::vector<float> stats =
      campplus::StatsPool(style_in, d.style_feat_dim, d.ref_frames);
  Expect(static_cast<int64_t>(stats.size()) == 2 * d.style_feat_dim, "stage 3 (CAMPPlus)",
         "stats pooling emits mean and std");
  r.style.assign(stats.begin(), stats.begin() + static_cast<std::ptrdiff_t>(d.style_dim));

  // ── stage 4: length regulator -> prompt condition at the MEL rate.
  const std::vector<float> regulated =
      lenreg::InterpolateNearest(q.z_q, d.codec_dim, d.ref_frames, d.mel_frames);
  Expect(static_cast<int64_t>(regulated.size()) == d.codec_dim * d.mel_frames,
         "stage 4 (length regulator)", "prompt condition must be at the mel frame rate");
  r.prompt_condition = regulated;

  // ── stage 5: talker embeddings -> mel codes.
  Expect(!text_tokens.empty(), "stage 5 (talker)", "text tokens must not be empty");
  const std::vector<float> tok_table =
      Synth("t.tok", d.talker_vocab * d.talker_dim, 0.3, seed);
  const std::vector<float> pos_table =
      Synth("t.pos", d.talker_vocab * d.talker_dim, 0.1, seed);
  const std::vector<float> talker_in = talker::EmbedWithPositions(
      text_tokens, tok_table, pos_table, d.talker_dim, d.talker_vocab);
  Expect(talker_in.size() == static_cast<size_t>(text_tokens.size()) *
                                 static_cast<size_t>(d.talker_dim),
         "stage 5 (talker)", "embedded text must be [tokens, talker_dim]");

  // ── stage 6: mel -> waveform through the shared vocoder core.
  // The mel is built from the regulated condition and the style, so a change to
  // EITHER propagates to the waveform -- which is what this seam must prove.
  std::vector<float> mel(static_cast<size_t>(d.mel_channels * d.mel_frames));
  for (int64_t c = 0; c < d.mel_channels; ++c) {
    for (int64_t t = 0; t < d.mel_frames; ++t) {
      const double cond = static_cast<double>(
          regulated[static_cast<size_t>((c % d.codec_dim) * d.mel_frames + t)]);
      const double sty = static_cast<double>(r.style[static_cast<size_t>(c % d.style_dim)]);
      const double txt = static_cast<double>(
          talker_in[static_cast<size_t>((t % static_cast<int64_t>(text_tokens.size())) *
                                        d.talker_dim + (c % d.talker_dim))]);
      mel[static_cast<size_t>(c * d.mel_frames + t)] =
          static_cast<float>(std::tanh(cond + 0.5 * sty + 0.25 * txt));
    }
  }
  r.mel = mel;

  // One depthwise-ish pass through the ported vocoder primitive, so the final
  // hop is real code rather than a copy.
  int64_t out_len = 0;
  const std::vector<float> padded =
      vocoder1d::Pad1d(mel, d.mel_channels, d.mel_frames, 1, 1, /*replicate=*/true, &out_len);
  const std::vector<float> kernel =
      Synth("v.k", d.mel_channels * 3, 0.4, seed);
  int64_t wav_len = 0;
  r.waveform = vocoder1d::Conv1d(padded, d.mel_channels, out_len, kernel, /*bias=*/nullptr,
                                 d.mel_channels, /*kernel=*/3, /*stride=*/1, /*dilation=*/1,
                                 /*groups=*/d.mel_channels, &wav_len);
  Expect(wav_len == d.mel_frames, "stage 6 (vocoder)",
         "waveform frame count must match the mel length");
  r.sample_rate = 22050;
  return r;
}

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
