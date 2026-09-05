// dots3-note IMAGE processor (W6a, #2512). Ported from
// `vllm/models/dots3_note/common/processor.py` read in `~/_git/vllm` at
// `9035151d6`. See the header for the full provenance and for the three ways
// this differs from the Qwen3-VL processor beside it.
#include "vllm/multimodal/dots3_note_processor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/multimodal/audio_resample.h"
#include "vllm/multimodal/hasher.h"
#include "vllm/multimodal/mel_filter_bank.h"
#include "vllm/v1/engine/validation_error.h"
#include "vllm/multimodal/pil_resize.h"
#include "vt/dtype.h"

namespace vllm::multimodal {

namespace {

nlohmann::json LoadJson(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open json: " + path);
  nlohmann::json j;
  f >> j;
  return j;
}

// `round(v / factor) * factor` with Python's round-half-to-EVEN
// (`processor.py:86-88` @ `9035151d6`). `std::round` is half-away-from-zero and
// would disagree at exactly .5, which is one grid row.
int64_t RoundByFactor(int64_t v, int64_t factor) {
  return static_cast<int64_t>(std::nearbyint(static_cast<double>(v) /
                                             static_cast<double>(factor))) *
         factor;
}
int64_t CeilByFactor(double v, int64_t factor) {
  return static_cast<int64_t>(
             std::ceil(v / static_cast<double>(factor))) * factor;
}
int64_t FloorByFactor(double v, int64_t factor) {
  return static_cast<int64_t>(
             std::floor(v / static_cast<double>(factor))) * factor;
}

// Resolve one of the three image token ids from `added_tokens.json` (upstream's
// own source, `multimodal.py:82-90` @ `9035151d6`) or, failing that, from the
// `config.json` key a converted checkpoint may carry. Returns -1 when neither
// answers; the caller refuses BY NAME rather than defaulting.
int32_t ResolveTokenId(const nlohmann::json& added, const nlohmann::json& cfg,
                       const char* marker, const char* config_key) {
  if (added.is_object()) {
    const auto it = added.find(marker);
    if (it != added.end() && it->is_number_integer())
      return it->get<int32_t>();
  }
  if (cfg.is_object()) {
    const auto it = cfg.find(config_key);
    if (it != cfg.end() && it->is_number_integer())
      return it->get<int32_t>();
  }
  return -1;
}

}  // namespace

Dots3NoteProcessorConfig LoadDots3NoteProcessorConfig(
    const std::string& preprocessor_config_json_path,
    const std::string& config_json_path, const std::string& model_id) {
  Dots3NoteProcessorConfig cfg;
  cfg.model_id = model_id;

  const nlohmann::json pp = LoadJson(preprocessor_config_json_path);
  cfg.patch_size = pp.value("patch_size", cfg.patch_size);
  cfg.temporal_patch_size =
      pp.value("temporal_patch_size", cfg.temporal_patch_size);
  cfg.merge_size = pp.value("merge_size", cfg.merge_size);
  cfg.pre_pixel_shuffle = pp.value("pre_pixel_shuffle", cfg.pre_pixel_shuffle);
  cfg.min_pixels = pp.value("min_pixels", cfg.min_pixels);
  cfg.max_pixels = pp.value("max_pixels", cfg.max_pixels);
  // The `size` shorthand the HF image-processor family also writes.
  if (pp.contains("size") && pp["size"].is_object()) {
    const auto& sz = pp["size"];
    cfg.min_pixels = sz.value("shortest_edge", cfg.min_pixels);
    cfg.max_pixels = sz.value("longest_edge", cfg.max_pixels);
  }
  // PER CHANNEL. Reading only `[0]` — which is what the Qwen3-VL loader beside
  // this one can afford, because its mean and std are 0.5 on all three — would
  // silently normalize green and blue with red's statistics.
  const auto read3 = [](const nlohmann::json& j, const char* key,
                        std::array<double, 3>* out) {
    if (!j.contains(key)) return;
    const auto& a = j[key];
    if (!a.is_array()) {
      throw std::runtime_error(std::string("dots3-note processor: '") + key +
                               "' must be a 3-entry list, got " + a.dump());
    }
    if (a.size() == 1) {
      (*out) = {a[0].get<double>(), a[0].get<double>(), a[0].get<double>()};
      return;
    }
    if (a.size() != 3) {
      throw std::runtime_error(
          std::string("dots3-note processor: '") + key + "' has " +
          std::to_string(a.size()) +
          " entries; the RGB pipeline needs 1 or 3 (processor.py:76-77 @ "
          "9035151d6)");
    }
    for (int i = 0; i < 3; ++i) (*out)[static_cast<size_t>(i)] = a[i].get<double>();
  };
  read3(pp, "image_mean", &cfg.image_mean);
  read3(pp, "image_std", &cfg.image_std);
  cfg.rescale_factor = pp.value("rescale_factor", cfg.rescale_factor);

  const nlohmann::json cj = LoadJson(config_json_path);
  // `vision_config` is the AUTHORITY on the patch/merge geometry, exactly as it
  // is for the tower: a `preprocessor_config.json` that disagrees with the
  // model it belongs to would move the grid.
  if (cj.contains("vision_config") && cj["vision_config"].is_object()) {
    const auto& vc = cj["vision_config"];
    cfg.merge_size = vc.value("spatial_merge_size", cfg.merge_size);
    cfg.patch_size = vc.value("patch_size", cfg.patch_size);
    cfg.temporal_patch_size =
        vc.value("temporal_patch_size", cfg.temporal_patch_size);
    cfg.pre_pixel_shuffle = vc.value("pre_pixel_shuffle", cfg.pre_pixel_shuffle);
  }

  // The three ids. Upstream reads `added_tokens.json` and RAISES when
  // `<|imgpad|>` is absent (`multimodal.py:86-90` @ `9035151d6`); this mirrors
  // that, and extends it to the two markers around it, because injecting a
  // start/end marker the tokenizer does not know breaks the prompt just as
  // quietly.
  nlohmann::json added = nlohmann::json::object();
  {
    const std::string dir =
        config_json_path.substr(0, config_json_path.find_last_of("/\\") + 1);
    std::ifstream f(dir + "added_tokens.json");
    if (f) f >> added;
  }
  cfg.image_token_id =
      ResolveTokenId(added, cj, "<|imgpad|>", "image_token_id");
  cfg.image_start_token_id =
      ResolveTokenId(added, cj, "<|img|>", "image_start_token_id");
  cfg.image_end_token_id =
      ResolveTokenId(added, cj, "<|endofimg|>", "image_end_token_id");
  const auto require = [](int32_t id, const char* marker, const char* key) {
    if (id >= 0) return;
    throw std::runtime_error(
        std::string("dots3-note processor: the image token '") + marker +
        "' has no id. Upstream reads it from `added_tokens.json` and raises "
        "when it is missing (multimodal.py:86-90 @ 9035151d6); this port also "
        "accepts `config.json`'s `" + key +
        "`. Refusing rather than guessing an id: a marker the tokenizer does "
        "not know is injected as ordinary text and the image is dropped.");
  };
  require(cfg.image_token_id, "<|imgpad|>", "image_token_id");
  require(cfg.image_start_token_id, "<|img|>", "image_start_token_id");
  require(cfg.image_end_token_id, "<|endofimg|>", "image_end_token_id");
  return cfg;
}

std::array<int64_t, 2> Dots3NoteResizedSize(int64_t height, int64_t width,
                                            int64_t factor, int64_t min_pixels,
                                            int64_t max_pixels) {
  // `processor.py:131-146` @ `9035151d6`, in upstream's own order.
  if (std::min(height, width) < factor / 4) {
    throw std::runtime_error(
        "dots3-note processor: image height and width must be at least " +
        std::to_string(factor / 4) + ", got " + std::to_string(height) + "x" +
        std::to_string(width));
  }
  if (std::min(height, width) <= 0 ||
      static_cast<double>(std::max(height, width)) /
              static_cast<double>(std::min(height, width)) >
          200.0) {
    throw std::runtime_error(
        "dots3-note processor: image aspect ratio must be smaller than 200");
  }
  int64_t rh = std::max(factor, RoundByFactor(height, factor));
  int64_t rw = std::max(factor, RoundByFactor(width, factor));
  const double hw = static_cast<double>(height) * static_cast<double>(width);
  if (rh * rw > max_pixels) {
    const double beta = std::sqrt(hw / static_cast<double>(max_pixels));
    rh = std::max(factor, FloorByFactor(static_cast<double>(height) / beta, factor));
    rw = std::max(factor, FloorByFactor(static_cast<double>(width) / beta, factor));
  } else if (rh * rw < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / hw);
    rh = CeilByFactor(static_cast<double>(height) * beta, factor);
    rw = CeilByFactor(static_cast<double>(width) * beta, factor);
    if (rh * rw > max_pixels) {
      const double b2 = std::sqrt(static_cast<double>(rh) *
                                  static_cast<double>(rw) /
                                  static_cast<double>(max_pixels));
      rh = std::max(factor, FloorByFactor(static_cast<double>(rh) / b2, factor));
      rw = std::max(factor, FloorByFactor(static_cast<double>(rw) / b2, factor));
    }
  }
  return {rh, rw};
}

std::string Dots3NoteImageProcessor::HashImage(const uint8_t* rgb,
                                               int64_t height,
                                               int64_t width) const {
  return MultiModalHasher::HashImageRGB(cfg_.model_id, rgb, height, width);
}

ImageKwargs Dots3NoteImageProcessor::ProcessImage(const uint8_t* rgb,
                                                  int64_t height,
                                                  int64_t width) const {
  const int64_t patch = cfg_.patch_size;
  const int64_t merge = cfg_.merge_size;
  const int64_t tp = cfg_.temporal_patch_size;
  const int64_t f = factor();

  const std::array<int64_t, 2> rs =
      Dots3NoteResizedSize(height, width, f, cfg_.min_pixels, cfg_.max_pixels);
  const int64_t rh = rs[0], rw = rs[1];

  // THE RESIZE (`image.resize((resized_w, resized_h),
  // Image.Resampling.BICUBIC)`, `processor.py:174` @ `9035151d6`). Upstream
  // ALWAYS resizes, and `factor = patch_size * merge_size` is 28 on the
  // released checkpoint, so an image that clears the identity below is the rare
  // case rather than the ordinary one: this call is the served path, not a
  // fallback. W6c, issue #2537.
  //
  // `PilResizeBicubicRgb` is PIL's resampler and NOT a four-tap cubic — see
  // `pil_resize.h` for why the difference is structural on a downscale. Getting
  // it wrong here would move every patch, and §6.4 records that this row has no
  // token-exact denominator that could catch it.
  //
  // NOT resizing when the size already matches is upstream's own short circuit
  // and keeps the conformant path byte-identical to what W6a served.
  std::vector<uint8_t> resized;
  if (rh != height || rw != width) {
    resized = PilResizeBicubicRgb(rgb, height, width, rh, rw);
    rgb = resized.data();
  }

  const int64_t grid_h = rh / patch;
  const int64_t grid_w = rw / patch;
  const int64_t grid_t = 1;  // ONE image; video grids are W7's
  if (cfg_.pre_pixel_shuffle && (grid_h % merge != 0 || grid_w % merge != 0)) {
    throw std::runtime_error(
        "Dots3NoteImageProcessor: the " + std::to_string(grid_h) + "x" +
        std::to_string(grid_w) +
        " patch grid does not group into whole " + std::to_string(merge) +
        "x" + std::to_string(merge) +
        " blocks, which `pre_pixel_shuffle` requires (processor.py:185-197 @ "
        "9035151d6)");
  }
  const int64_t num_patches = grid_t * grid_h * grid_w;
  const int64_t feat = 3 * tp * patch * patch;

  // Fused rescale + normalize, PER CHANNEL:
  //   (raw * rescale - mean) / std  ==  (raw - mean/rescale) / (std/rescale)
  // `processor.py:166-167`.
  double shift[3], scale[3];
  for (int c = 0; c < 3; ++c) {
    shift[c] = cfg_.image_mean[static_cast<size_t>(c)] / cfg_.rescale_factor;
    scale[c] = cfg_.image_std[static_cast<size_t>(c)] / cfg_.rescale_factor;
  }

  ImageKwargs out;
  out.num_patches = num_patches;
  out.patch_feature_dim = feat;
  out.image_grid_thw = {grid_t, grid_h, grid_w};
  out.pixel_values_f32.resize(static_cast<size_t>(num_patches * feat));
  out.pixel_values_bf16.resize(static_cast<size_t>(num_patches * feat));

  // The RESIZED width, not the requested one: `rgb` points at the resampled
  // buffer whenever the two differ, and reading it at the source stride would
  // shear the image while leaving every shape and every count valid.
  const int64_t rowstride = rw * 3;  // HWC uint8 stride of the patchified image
  // The column index is the same under both row orders:
  //   k = ((c * tp + t) * patch + ph) * patch + pw
  // because both transposes end `..., C, tp, ph, pw` (`processor.py:196`,
  // `:207`). Only the ROW index differs.
  const auto emit = [&](int64_t r, int64_t src_h0, int64_t src_w0) {
    for (int64_t c = 0; c < 3; ++c) {
      for (int64_t t = 0; t < tp; ++t) {
        for (int64_t ph = 0; ph < patch; ++ph) {
          const int64_t H = src_h0 + ph;
          for (int64_t pw = 0; pw < patch; ++pw) {
            const int64_t W = src_w0 + pw;
            const uint8_t raw = rgb[H * rowstride + W * 3 + c];
            const float v = static_cast<float>(
                (static_cast<double>(raw) - shift[c]) / scale[c]);
            const int64_t k = ((c * tp + t) * patch + ph) * patch + pw;
            const size_t idx = static_cast<size_t>(r * feat + k);
            out.pixel_values_f32[idx] = v;
            out.pixel_values_bf16[idx] = vt::F32ToBF16(v);
          }
        }
      }
    }
  };

  if (cfg_.pre_pixel_shuffle) {
    // `reshape(t, tp, C, Gh, m, p, Gw, m, p).transpose(0,3,6,4,7,2,1,5,8)`
    // (`processor.py:185-197`): row index
    //   r = ((gh * Gw + gw) * m + mh) * m + mw
    const int64_t Gh = grid_h / merge, Gw = grid_w / merge;
    for (int64_t gh = 0; gh < Gh; ++gh) {
      for (int64_t gw = 0; gw < Gw; ++gw) {
        for (int64_t mh = 0; mh < merge; ++mh) {
          for (int64_t mw = 0; mw < merge; ++mw) {
            const int64_t r = ((gh * Gw + gw) * merge + mh) * merge + mw;
            emit(r, (gh * merge + mh) * patch, (gw * merge + mw) * patch);
          }
        }
      }
    }
  } else {
    // `reshape(t, tp, C, gh, p, gw, p).transpose(0,3,5,2,1,4,6)`
    // (`processor.py:199-208`): plain row-major, r = gh * grid_w + gw.
    for (int64_t gh = 0; gh < grid_h; ++gh) {
      for (int64_t gw = 0; gw < grid_w; ++gw) {
        emit(gh * grid_w + gw, gh * patch, gw * patch);
      }
    }
  }
  return out;
}


// ─── THE AUDIO PROCESSOR (W7a, #2703) ───────────────────────────────────────
// See the header for the provenance and for why the front end is CONFIGURED
// rather than re-written.

Dots3NoteAudioProcessorConfig LoadDots3NoteAudioProcessorConfig(
    const std::string& config_json_path, const std::string& model_id) {
  Dots3NoteAudioProcessorConfig cfg;
  cfg.model_id = model_id;
  const nlohmann::json cj = LoadJson(config_json_path);
  if (!cj.contains("audio_config") || !cj["audio_config"].is_object()) {
    return cfg;  // present == false
  }
  const auto& ac = cj["audio_config"];
  cfg.present = true;
  cfg.sampling_rate = ac.value("sampling_rate", cfg.sampling_rate);
  cfg.chunk_seconds = ac.value("chunk_seconds", cfg.chunk_seconds);
  cfg.merge_factor = ac.value("merge_factor", cfg.merge_factor);
  // `conv_temporal_stride` is a PROPERTY of `use_conv2d_stem`
  // (`audio.py:67-69`), never its own key. Deriving it here rather than reading
  // one keeps the stride and the stem from ever disagreeing; the conv1d arm is
  // refused by name below, so the 2 branch is unreachable in production and is
  // written anyway because it is upstream's.
  cfg.conv_temporal_stride = ac.value("use_conv2d_stem", true) ? 8 : 2;
  if (ac.contains("whisper_config") && ac["whisper_config"].is_object()) {
    cfg.n_mels = ac["whisper_config"].value("num_mel_bins", cfg.n_mels);
  }
  cfg.audio_comp_start = ac.value("audio_comp_start", cfg.audio_comp_start);
  cfg.audio_comp_span = ac.value("audio_comp_span", cfg.audio_comp_span);
  cfg.audio_comp_end = ac.value("audio_comp_end", cfg.audio_comp_end);
  return cfg;
}

void Dots3NoteResolveAudioTokenIds(
    Dots3NoteAudioProcessorConfig* cfg,
    const std::function<int32_t(const std::string&)>& lookup) {
  const auto resolve = [&](const std::string& marker, const char* which,
                           int32_t* out) {
    const int32_t id = lookup(marker);
    if (id < 0) {
      throw std::runtime_error(
          "dots3-note audio processor: this checkpoint's tokenizer carries no "
          "added token '" + marker + "', which `audio_config." + which +
          "` names. Upstream resolves the three audio markers out of the "
          "tokenizer's vocabulary (common/processor.py:757-760 @ 9035151d6) "
          "and reads `added_tokens.json` for the image one "
          "(nvidia/multimodal.py:86-89). REFUSING rather than defaulting to an "
          "id: the three are NOT consecutive in the released checkpoint "
          "(start 151718, end 151719, pad 151720), so a guess would inject a "
          "marker the tokenizer maps to something else and the audio would be "
          "dropped from a request that answered 200.");
    }
    *out = id;
  };
  resolve(cfg->audio_comp_start, "audio_comp_start", &cfg->audio_start_token_id);
  resolve(cfg->audio_comp_span, "audio_comp_span", &cfg->audio_token_id);
  resolve(cfg->audio_comp_end, "audio_comp_end", &cfg->audio_end_token_id);
}

std::string Dots3NoteAudioProcessorRefusal(
    const Dots3NoteAudioProcessorConfig& cfg) {
  if (!cfg.present) {
    return "this checkpoint's config.json carries no `audio_config`, so "
           "upstream builds no audio tower for it either "
           "(nvidia/multimodal.py:119-126 @ 9035151d6) and there is nothing to "
           "serve an audio part with";
  }
  // ORDER MATTERS: the first unrepresentable feature wins, and the order is
  // upstream's own constructor order so a reader can diff the two.
  if (cfg.merge_factor != 1) {
    return "`audio_config.merge_factor` is " +
           std::to_string(cfg.merge_factor) +
           "; only 1 is ported. Upstream reshapes [B, T, D] to "
           "[B, T/merge, D*merge] before the adapter "
           "(nvidia/audio.py:269-276 @ 9035151d6), which also multiplies the "
           "adapter's input width, and the released checkpoint's "
           "`whisper_adapter_in_dim` 1280 is the unmerged one. No published "
           "checkpoint sets it; refused rather than reshaped";
  }
  if (cfg.sampling_rate <= 0) {
    return "`audio_config.sampling_rate` is " +
           std::to_string(cfg.sampling_rate) + ", which is not a rate";
  }
  if (cfg.chunk_seconds <= 0) {
    return "`audio_config.chunk_seconds` is " +
           std::to_string(cfg.chunk_seconds) + ", which is not a duration";
  }
  // `assert mel.shape[1] == self.chunk_mel_frames` (`audio.py:215`) is
  // upstream's own, and it only holds when the hop divides the chunk exactly.
  if (cfg.chunk_samples() % cfg.hop_length != 0) {
    return "`audio_config` gives " + std::to_string(cfg.chunk_samples()) +
           " chunk samples, which is not a whole number of " +
           std::to_string(cfg.hop_length) +
           "-sample hops; upstream asserts the mel frame count equals "
           "`chunk_seconds * 100` (nvidia/audio.py:215 @ 9035151d6) and that "
           "assert would not hold";
  }
  return "";
}

Dots3NoteAudioProcessor::Dots3NoteAudioProcessor(
    Dots3NoteAudioProcessorConfig cfg)
    : cfg_(std::move(cfg)) {
  const std::string why = Dots3NoteAudioProcessorRefusal(cfg_);
  if (!why.empty()) {
    throw std::runtime_error("dots3-note audio processor: " + why);
  }
  // The FRONT END, configured rather than re-written. Every field below is a
  // dots3 value flowing into the Whisper log-mel this tree already gates
  // (tests/vllm/multimodal/test_voxtral_e2e.cpp:157-178 drives it at exactly
  // this n_fft / hop / n_mels / rate).
  AudioProcessorConfig fe;
  fe.n_fft = cfg_.n_fft;
  fe.hop_length = cfg_.hop_length;
  fe.n_mels = cfg_.n_mels;
  fe.sampling_rate = cfg_.sampling_rate;
  fe.chunk_length_s = cfg_.chunk_seconds;
  // `max_source_positions` is Whisper's fixed token count and is the ONE field
  // of that struct dots3 must not use: dots3's token count is
  // `ceil(samples / 1280)` and depends on the waveform. It is left at its
  // default and `NumAudioTokens` below is what answers instead. This is also
  // why `RouteAudioWav` (chat_mm.cpp:131-160) cannot serve this model — it
  // reads that field — and why W7a wrote `RouteDots3NoteAudioWav` beside it
  // rather than editing another row's gated function.
  fe.model_id = cfg_.model_id;
  fe.audio_placeholder_id = cfg_.audio_token_id;
  front_end_ = std::make_shared<const WhisperAudioProcessor>(
      fe, MelFilterBankSlaney(cfg_.num_freq_bins(), cfg_.n_mels,
                              /*min_frequency=*/0.0,
                              /*max_frequency=*/
                              static_cast<double>(cfg_.sampling_rate) / 2.0,
                              cfg_.sampling_rate));
}

const std::vector<float>& Dots3NoteAudioProcessor::mel_filters() const {
  return front_end_->mel_filters();
}

int64_t Dots3NoteAudioProcessor::NumAudioTokens(int64_t num_samples) const {
  const int64_t stride = cfg_.token_stride();
  if (num_samples <= 0) return 0;
  // `math.ceil(length / stride)` (`processor.py:771`), written as the integer
  // form `(n - 1) // stride + 1` that `audio.py:210-212` uses, so no double
  // rounds at a boundary.
  return (num_samples - 1) / stride + 1;
}

std::vector<Dots3NoteAudioProcessor::AudioChunk>
Dots3NoteAudioProcessor::SegmentWaveform(int64_t num_samples) const {
  // `encode_waveform`'s slicing loop, `nvidia/audio.py:196-203` @ `9035151d6`:
  //
  //   while time_step * SAMPLE_RATE < audio_waveform.shape[0]:
  //       segments.append(audio_waveform[time_step * SAMPLE_RATE :
  //                                      (time_step + chunk_seconds) * SAMPLE_RATE])
  //       time_step += chunk_seconds
  //
  // Upstream steps in SECONDS and multiplies by the module constant
  // SAMPLE_RATE (`audio.py:15`), not by `self.sampling_rate`; the two are the
  // same 16000 wherever this port runs. Before W7c-2 (#2828) that held because a
  // rate that was not `audio_config.sampling_rate` was REFUSED before this was
  // reached; it now holds because such a rate is RESAMPLED to it, which is the
  // stronger of the two reasons and reaches this function the same way. The
  // stride is written in SAMPLES here so the loop cannot overflow a second
  // count on a long recording.
  std::vector<AudioChunk> out;
  if (num_samples <= 0) return out;
  const int64_t chunk = cfg_.chunk_samples();
  for (int64_t start = 0; start < num_samples; start += chunk) {
    AudioChunk c;
    c.start = start;
    // Python's slice CLAMPS its stop, so the last segment is short rather than
    // padded here; `pad_or_trim` is what pads it, and only for the mel.
    c.length = std::min(chunk, num_samples - start);
    // `token_len = (segment_length - 1) // stride + 1` (`:210-212`), which is
    // `NumAudioTokens` — the same function, not a second copy. The loop above
    // never emits a zero-length segment, which is the condition under which the
    // two forms are the same function; see the header.
    c.num_tokens = NumAudioTokens(c.length);
    out.push_back(c);
  }
  return out;
}

AudioKwargs Dots3NoteAudioProcessor::ProcessWaveform(
    const float* samples, int64_t num_samples, int sample_rate,
    std::vector<float>* resampled_out) const {
  // W7c-2 (#2828): a rate that is not `audio_config.sampling_rate` is
  // RESAMPLED, not refused. Upstream resamples in its data parser
  // (`MultiModalDataParser(target_sr=..., target_channels=1)`,
  // `vllm/models/dots3_note/common/processor.py:523-525` @ `9035151d6` ->
  // `vllm/multimodal/parse.py:695`), and this is the function that is handed a
  // rate, so this is where the conversion goes.
  //
  // `ResampleAudioScipy` is upstream's `"scipy"` arm and NOT its `"pyav"`
  // default, and that is a recorded DIVERGENCE rather than an oversight:
  // libswresample is not bit-identical to itself across CPU dispatch on one
  // binary and one input, so no bit-exact gate against the default can exist.
  // vLLM ships the scipy arm in production for another model
  // (`vllm/model_executor/models/phi4mm.py:580`). See the seam header and
  // `.agents/specs/dots3-note.md` §4.17.
  //
  // THIS IS THE PRODUCTION CALL SITE the reachability mutation deletes. The
  // guard below is upstream's own `if orig_sr_int == target_sr_int: return
  // audio` (`audio.py:241-242`) hoisted to the caller, and `ResampleAudioScipy`
  // repeats it internally rather than trusting it, so a 16 kHz waveform cannot
  // move by a bit whichever way it arrives.
  //
  // THE THREE VARIABLES DESCRIBING THE WAVEFORM MOVE TOGETHER, and `sample_rate`
  // is one of them. Rebinding the pointer and the length while leaving the rate
  // at the request's value left the resampled buffer described as 22050 Hz, and
  // `WhisperAudioProcessor::ProcessWaveform` — which this drives once per chunk
  // and which carries its OWN rate refusal for the Whisper/Voxtral row
  // (`audio_processor.cpp:214-221`) — threw a bare `runtime_error`. That is
  // HTTP 500, not 400: the served suite read `500 == 400` before this line was
  // written. Before W7c-2 the assignment was unreachable, because the refusal
  // above it guaranteed the two rates were already equal.
  //
  // THE BUFFER IS HANDED BACK when the caller asked for it (PR #2842 F2). The
  // route needs the SAME resampled waveform for the encoder-cache key, and
  // before this it got it by resampling a second time — 1220.7 MB twice on the
  // request measured in spec §4.17.10. `resampled_out` is used as the local,
  // so there is one buffer and no copy, and it stays empty at the target rate
  // because then the caller's own pointer already is the consumed waveform.
  std::vector<float> owned;
  std::vector<float>& resampled = resampled_out != nullptr ? *resampled_out : owned;
  resampled.clear();
  if (sample_rate != cfg_.sampling_rate) {
    resampled = ResampleAudioScipy(samples, num_samples, sample_rate,
                                   cfg_.sampling_rate);
    samples = resampled.data();
    num_samples = static_cast<int64_t>(resampled.size());
    sample_rate = cfg_.sampling_rate;
  }

  if (num_samples <= 0) {
    throw v1::InputValidationError(
        "dots3-note audio processor: the request carries an empty waveform");
  }

  // `segments` (`audio.py:194-203`).
  const std::vector<AudioChunk> chunks = SegmentWaveform(num_samples);
  const int64_t total_tokens = NumAudioTokens(num_samples);

  // THE ONE THING SEGMENTATION CANNOT MAKE SAFE, refused BY NAME rather than
  // spliced. The tower emits `sum_i ceil(seg_i / stride)` rows — upstream's own
  // `compute_audio_token_length` (`audio.py:129-147`) — while the prompt side
  // expands ONE `ceil(total / stride)` span (`common/processor.py:771`). Every
  // segment but the last is exactly `chunk_samples` long, so those two are
  // equal for every waveform exactly when `chunk_samples % token_stride == 0`.
  // The released config satisfies it (960000 = 750 * 1280); the check is on the
  // NUMBERS rather than on the config so it cannot drift from the loop above.
  //
  // A SINGLE-chunk waveform is served either way, because a one-segment sum IS
  // `ceil(n / stride)`. That is why this is refused per request and not at
  // install: refusing the capability would turn away clips upstream serves.
  if (chunks.size() > 1) {
    int64_t summed = 0;
    for (const AudioChunk& c : chunks) summed += c.num_tokens;
    if (summed != total_tokens) {
      throw v1::InputValidationError(
          "dots3-note audio processor: this request's " +
          std::to_string(num_samples) + " samples need " +
          std::to_string(chunks.size()) + " chunks, and this checkpoint's "
          "`audio_config` gives " + std::to_string(cfg_.chunk_samples()) +
          " chunk samples, which is not a whole number of " +
          std::to_string(cfg_.token_stride()) +
          "-sample token strides. The tower would produce " +
          std::to_string(summed) + " rows (the per-segment sum, "
          "nvidia/audio.py:129-147 @ 9035151d6) against a placeholder span of " +
          std::to_string(total_tokens) + " (the prompt side's one "
          "ceil(total / stride), common/processor.py:771), and a masked "
          "scatter that does not balance splices audio features onto text "
          "rows. Upstream computes both numbers and never compares them. A "
          "waveform inside ONE chunk is served on this config; a longer one is "
          "refused. See .agents/specs/dots3-note.md §4.15.3 and issue #2797.");
    }
  }

  AudioKwargs out;
  out.n_mels = cfg_.n_mels;
  out.n_frames = cfg_.chunk_mel_frames();
  out.num_samples = num_samples;
  out.num_tokens = total_tokens;
  out.num_chunks = static_cast<int64_t>(chunks.size());
  out.input_features.reserve(static_cast<size_t>(out.num_chunks) *
                             static_cast<size_t>(out.n_mels) *
                             static_cast<size_t>(out.n_frames));

  for (const AudioChunk& c : chunks) {
    // `pad_or_trim(audio_segment.flatten(), length=self.chunk_samples)` then
    // `log_mel_spectrogram(pad_audio)` (`audio.py:213-214`), PER SEGMENT.
    // `WhisperAudioProcessor::ProcessWaveform` performs the pad itself
    // (audio_processor.cpp:228-232, `padding="max_length"` with truncation), so
    // this call IS that pair. The `-8` floor inside it is a GLOBAL max over the
    // segment it is handed, which is why the front end is driven once per
    // segment and not once over the whole waveform.
    const AudioKwargs one = front_end_->ProcessWaveform(
        samples + c.start, c.length, sample_rate);
    if (one.n_frames != cfg_.chunk_mel_frames() || one.n_mels != cfg_.n_mels) {
      // Upstream's own assert (`audio.py:215`), kept rather than trusted.
      throw std::runtime_error(
          "dots3-note audio processor: the front end produced " +
          std::to_string(one.n_mels) + " x " + std::to_string(one.n_frames) +
          " mel for a chunk of " + std::to_string(cfg_.n_mels) + " x " +
          std::to_string(cfg_.chunk_mel_frames()) +
          " (nvidia/audio.py:215 @ 9035151d6)");
    }
    // `torch.stack(mel_features, dim=0)` (`:220`), flattened.
    out.input_features.insert(out.input_features.end(),
                              one.input_features.begin(),
                              one.input_features.end());
    // `audio_sample_lens` and `token_lens` (`:217-218`).
    out.chunk_num_samples.push_back(c.length);
    out.chunk_num_tokens.push_back(c.num_tokens);
  }
  return out;
}

std::string Dots3NoteAudioProcessor::HashAudio(const float* samples,
                                               int64_t num_samples) const {
  return front_end_->HashAudio(samples, num_samples);
}

std::string Dots3NoteAudioProcessor::HashAudio(
    const float* samples, int64_t num_samples, int sample_rate,
    const std::vector<float>* resampled) const {
  // W7c-2 (#2828) CREATED the defect this closes, and closes it in the same
  // change. While every served rate was `audio_config.sampling_rate` the raw
  // waveform was an unambiguous encoder-cache key. It stops being one the
  // moment two rates are served: a file carrying N PCM16 samples at 16000 Hz
  // and a file carrying THE IDENTICAL N SAMPLES at 44100 Hz decode to identical
  // float buffers, hash identically under the two-argument form, and must
  // produce different features. `mm_hash` is a CROSS-REQUEST key —
  // `EncoderCacheManager::cached_` is keyed on it and `scheduler.cpp:511-590`
  // reuses a hit — so the second request would be handed the first's
  // embeddings.
  //
  // BE EXACT ABOUT WHAT THAT COSTS, because the two also differ in RESAMPLED
  // LENGTH: a collision needs identical raw buffers, and identical buffers at
  // different rates cannot resample to the same row count. So the observable
  // failure is a wrong-length splice rather than a quiet substitution. Either
  // way the key is wrong, and a key that is only accidentally caught downstream
  // is not a key.
  //
  // The key is therefore the RESAMPLED waveform — the buffer the tower actually
  // consumes. That also makes two requests that resample to the same waveform
  // share a cache entry, which is correct, and it is why this hashes the output
  // rather than merely mixing the rate into the input.
  if (sample_rate == cfg_.sampling_rate) {
    return front_end_->HashAudio(samples, num_samples);
  }
  // RESAMPLE ONCE (PR #2842 F2). `RouteDots3NoteAudioWav` has just driven
  // `ProcessWaveform` over this same waveform and this same rate, so it already
  // holds the buffer this would otherwise rebuild; on the request measured in
  // spec §4.17.10 that rebuild was a second 1220.7 MB allocation, doubling the
  // cost of a 40 KB upload. The argument is the answer and not a hint: it is
  // hashed as-is. A caller that does not hold it passes nothing and this
  // resamples for itself, which is the only behaviour that ever existed.
  if (resampled != nullptr) {
    return front_end_->HashAudio(resampled->data(),
                                 static_cast<int64_t>(resampled->size()));
  }
  const std::vector<float> own =
      ResampleAudioScipy(samples, num_samples, sample_rate, cfg_.sampling_rate);
  return front_end_->HashAudio(own.data(), static_cast<int64_t>(own.size()));
}

}  // namespace vllm::multimodal
