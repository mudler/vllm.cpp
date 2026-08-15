// LTX-2.5 AUDIO VAE ENCODER — the LOAD path, row LTX25-A2V-AUDIO-INPUT (#922).
//
// Spec: .agents/specs/ltx25-a2v-audio-input.md §1, §3.
//
// ─── WHY THIS IS ITS OWN TU ──────────────────────────────────────────────────
// The DECODER's twin of this code lives in `ltx2_loader.cpp:1295-1300` and
// `:1410-1442`, a 1500-line file that concurrent rows of the #644 campaign all
// need to touch. Putting the encoder's two functions there would have made that
// file the lock AGENTS.md §Records names. This mirrors the choice
// `ltx2_video_vae_encoder_load.cpp:5-12` already made for the video half, and
// like that file it duplicates only the small config accessors that
// `ltx2_loader.cpp` keeps in its own anonymous namespace and does not export.
//
// ─── WHY THIS DID NOT EXIST ──────────────────────────────────────────────────
// The reference-audio refusal at `ltx2_video.cpp:1402-1409` names this exact
// gap: "there is no AUDIO_VAE_ENCODER key filter — so nothing can turn a WAV
// into audio latents here". `Ltx2AudioEncoderForward` and the mel front-end have
// been ported and gated since `cefacd2d0`; what they never had was weights.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca,
// packages/ltx-core/src/ltx_core/model/audio_vae/
//   OURS                            <-  UPSTREAM
//   Ltx2AudioVaeEncoderKeyRules     <-  model_configurator.py:194-200
//                                       (AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER)
//   Ltx2ParseAudioEncoderConfig     <-  model_configurator.py:144-182
//                                       (AudioEncoderConfigurator.from_metadata)
//
// The key filter differs from the decoder's (`:187-192`) in exactly one token —
// `encoder` for `decoder` — and shares the `per_channel_statistics.` rule,
// because both directions need the SAME normalization statistics: the encoder
// applies them and the decoder undoes them. Dropping that second rule would
// produce a latent that is silently off by a per-channel affine, which is a
// finite, correctly shaped tensor and therefore invisible to a shape check.
#include "vllm/model_executor/models/ltx2_audio_input.h"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& why) {
  throw std::runtime_error("ltx2 audio vae encoder: " + why);
}

bool StartsWith(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

// `config.get(key, fallback)`, refusing a present-but-wrong-typed value rather
// than falling back. Same polarity as `ltx2_loader.cpp:883-893` and
// `ltx2_video_vae_encoder_load.cpp:43-56`, for the same reason: a checkpoint
// that says `"z_channels": "8"` means something, and treating it as absent
// builds a different latent.
template <typename T>
T ConfigGet(const nlohmann::json& config, const std::string& key, T fallback,
            const std::string& where) {
  const auto it = config.find(key);
  if (it == config.end() || it->is_null()) return fallback;
  try {
    return it->get<T>();
  } catch (const std::exception&) {
    Fail("'" + where + "." + key + "' is present but not the expected type (" + it->dump() + ")");
  }
}

const nlohmann::json& ConfigObject(const nlohmann::json& parent, const std::string& key,
                                   const std::string& where) {
  static const nlohmann::json kEmpty = nlohmann::json::object();
  const auto it = parent.find(key);
  if (it == parent.end() || it->is_null()) return kEmpty;
  if (!it->is_object()) Fail("'" + where + "." + key + "' is not an object");
  return *it;
}

// causality_axis.py:4-10, and common/normalization.py:7-11. Duplicated from
// `ltx2_loader.cpp:935-948` rather than exported, per the TU note above.
Ltx2NormType ParseAudioNormType(const std::string& name) {
  if (name == "pixel") return Ltx2NormType::kPixel;
  if (name == "group") return Ltx2NormType::kGroup;
  Fail("audio_vae ddconfig.norm_type is '" + name + "'; only 'pixel' and 'group' are ported");
}

Ltx2CausalityAxis ParseCausalityAxis(const std::string& name) {
  if (name == "none") return Ltx2CausalityAxis::kNone;
  if (name == "width") return Ltx2CausalityAxis::kWidth;
  if (name == "height") return Ltx2CausalityAxis::kHeight;
  if (name == "width_compatibility") return Ltx2CausalityAxis::kWidthCompatibility;
  Fail("audio_vae ddconfig.causality_axis is '" + name + "'; it is not one of the four");
}

}  // namespace

bool Ltx2CheckpointHasAudioEncoder(const std::vector<std::string>& names) {
  // Asked of the `audio_vae.encoder.` prefix ALONE, deliberately. The key rules
  // also match `audio_vae.per_channel_statistics.`, which a DECODER-only
  // checkpoint carries too, so a presence test written as "did the encoder
  // filter match anything" answers yes for every LTX-2.5 checkpoint ever
  // shipped and the encoder then runs on statistics and no convolutions.
  for (const std::string& name : names) {
    if (StartsWith(name, "audio_vae.encoder.")) return true;
  }
  return false;
}

std::vector<Ltx2VaeKeyRule> Ltx2AudioVaeEncoderKeyRules() {
  // AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER (model_configurator.py:194-200).
  return {
      {"audio_vae.encoder.", ""},
      {"audio_vae.per_channel_statistics.", "per_channel_statistics."},
  };
}

Ltx2AudioEncoderLoad Ltx2ParseAudioEncoderConfig(const nlohmann::json& config) {
  // `AudioEncoderConfigurator.from_metadata` (audio_vae/model_configurator.py:144-182).
  const nlohmann::json& audio_vae = ConfigObject(config, "audio_vae", "config");
  if (audio_vae.empty()) Fail("the audio VAE config carries no 'audio_vae' object");
  const nlohmann::json& model = ConfigObject(audio_vae, "model", "audio_vae");
  const nlohmann::json& params = ConfigObject(model, "params", "audio_vae.model");
  const nlohmann::json& dd = ConfigObject(params, "ddconfig", "audio_vae.model.params");
  const nlohmann::json& pre = ConfigObject(audio_vae, "preprocessing", "audio_vae");
  const nlohmann::json& stft = ConfigObject(pre, "stft", "audio_vae.preprocessing");
  const nlohmann::json& mel = ConfigObject(pre, "mel", "audio_vae.preprocessing");
  const nlohmann::json& variables = ConfigObject(audio_vae, "variables", "audio_vae");

  Ltx2AudioEncoderLoad out;

  // ── the encoder proper (:161-182) ──────────────────────────────────────────
  out.encoder.ch = ConfigGet<int64_t>(dd, "ch", 128, "ddconfig");
  out.encoder.in_channels = ConfigGet<int64_t>(dd, "in_channels", 2, "ddconfig");
  out.encoder.ch_mult = ConfigGet<std::vector<int64_t>>(dd, "ch_mult", {1, 2, 4}, "ddconfig");
  out.encoder.num_res_blocks = ConfigGet<int64_t>(dd, "num_res_blocks", 2, "ddconfig");
  out.encoder.attn_resolutions =
      ConfigGet<std::vector<int64_t>>(dd, "attn_resolutions", {8, 16, 32}, "ddconfig");
  out.encoder.resolution = ConfigGet<int64_t>(dd, "resolution", 256, "ddconfig");
  out.encoder.z_channels = ConfigGet<int64_t>(dd, "z_channels", 8, "ddconfig");
  out.encoder.double_z = ConfigGet<bool>(dd, "double_z", true, "ddconfig");
  out.encoder.resamp_with_conv = ConfigGet<bool>(dd, "resamp_with_conv", true, "ddconfig");
  out.encoder.mid_block_add_attention =
      ConfigGet<bool>(dd, "mid_block_add_attention", true, "ddconfig");
  out.encoder.norm_type =
      ParseAudioNormType(ConfigGet<std::string>(dd, "norm_type", "pixel", "ddconfig"));
  out.encoder.causality_axis =
      ParseCausalityAxis(ConfigGet<std::string>(dd, "causality_axis", "height", "ddconfig"));

  // ── the mel front-end (:156-160) ───────────────────────────────────────────
  // These come from THREE different objects and one of them is not under
  // `preprocessing.mel`: `sampling_rate` sits on `model.params`, and the FFT
  // size is `stft.filter_length`, not `stft.n_fft`. Reading them from the
  // obvious-looking neighbours yields a front-end that runs and mis-bins every
  // frame.
  out.processor.target_sample_rate = ConfigGet<int64_t>(params, "sampling_rate", 16000, "params");
  out.processor.mel_hop_length = ConfigGet<int64_t>(stft, "hop_length", 160, "stft");
  out.processor.n_fft = ConfigGet<int64_t>(stft, "filter_length", 1024, "stft");

  // `ddconfig.mel_bins or mel.n_mel_channels or variables.mel_bins` (:160) — a
  // PYTHON `or` chain, so a 0 falls through exactly as an absent key does. Same
  // shape as the decoder's at `ltx2_loader.cpp:1435-1440`.
  int64_t mel_bins = ConfigGet<int64_t>(dd, "mel_bins", 0, "ddconfig");
  if (mel_bins == 0) mel_bins = ConfigGet<int64_t>(mel, "n_mel_channels", 0, "mel");
  if (mel_bins == 0) mel_bins = ConfigGet<int64_t>(variables, "mel_bins", 0, "variables");
  // Upstream's chain can end in `None`, which `AudioEncoder` then treats as "the
  // spectrogram decides". Here the mel front-end must BUILD the spectrogram, so
  // there is no later source for the value and a missing one is refused rather
  // than defaulted: a guessed bin count silently rescales the frequency axis.
  if (mel_bins <= 0) {
    Fail("the audio VAE config carries no mel bin count. `AudioEncoderConfigurator` reads "
         "`ddconfig.mel_bins`, then `preprocessing.mel.n_mel_channels`, then "
         "`variables.mel_bins` (model_configurator.py:160) and all three are absent or zero "
         "here. The mel front-end BUILDS the spectrogram, so nothing downstream can supply the "
         "value later; guessing it would resample the frequency axis and still produce a "
         "finite latent");
  }
  out.processor.mel_bins = mel_bins;
  return out;
}

}  // namespace vllm
